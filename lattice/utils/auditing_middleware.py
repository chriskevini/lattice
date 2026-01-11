"""Auditing middleware for LLM calls.

This module provides AuditingLLMClient which wraps the basic LLM client
to add audit tracking, feedback linkage, and error mirroring.

Technical Debt (Phase 2 of #169): This module contains late-bound imports
from discord_client (error_notifier, dream) as a workaround. These cross-package
dependencies will be eliminated in Phase 3-4 (DI Infrastructure) via
proper dependency injection. See: https://github.com/chriskevini/lattice/issues/169
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import os
import structlog

from lattice.utils.llm_client import GenerationResult


logger = structlog.get_logger(__name__)


@dataclass
class AuditResult(GenerationResult):
    """Generation result with audit ID for traceability."""

    audit_id: UUID | None = None
    prompt_key: str | None = None


class AuditingLLMClient:
    """Wrapper around LLMClient that automatically audits every LLM call.

    Every LLM call is automatically traced to a prompt_audit entry, enabling:
    - Complete audit trail for analysis
    - Feedback linkage for Dreaming Cycle
    - Unified mirror/display generation
    """

    def __init__(self, llm_client: Any) -> None:
        """Initialize the auditing client with underlying LLM client.

        Args:
            llm_client: The underlying LLM client to wrap.
        """
        self._client = llm_client

    async def complete(
        self,
        prompt: str,
        db_pool: Any,
        prompt_key: str | None = None,
        template_version: int | None = None,
        main_discord_message_id: int | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        dream_channel_id: int | None = None,
        bot: Any | None = None,
        audit_view: bool = False,
        audit_view_params: dict[str, Any] | None = None,
    ) -> AuditResult:
        """Complete a prompt with automatic audit tracking and optional AuditView.

        Args:
            prompt: The prompt to complete
            db_pool: Database pool for dependency injection
            prompt_key: Optional identifier for the prompt template/purpose
            template_version: Optional version of the template used
            main_discord_message_id: Discord message ID for audit linkage
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            dream_channel_id: Optional dream channel ID for AuditView
            bot: Optional Discord bot instance for AuditView
            audit_view: If True, automatically send an AuditView to the dream channel
            audit_view_params: Additional parameters for the AuditView:
                - input_text: Custom input text (default: prompt preview or user_message)
                - output_text: Custom output text (default: result.content)
                - main_message_url: Link to the main channel message
                - metadata: Additional metadata strings

        Returns:
            AuditResult with content, metadata, and audit_id
        """
        try:
            result = await self._client.complete(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            audit_id: UUID | None = None
            if main_discord_message_id is not None:
                from lattice.memory import prompt_audits

                audit_id = await prompt_audits.store_prompt_audit(
                    db_pool=db_pool,
                    prompt_key=prompt_key or "UNKNOWN",
                    rendered_prompt=prompt,
                    response_content=result.content,
                    main_discord_message_id=main_discord_message_id,
                    template_version=template_version,
                    model=result.model,
                    provider=result.provider,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    cost_usd=result.cost_usd,
                    latency_ms=result.latency_ms,
                )
                logger.info(
                    "Audited LLM call",
                    audit_id=str(audit_id),
                    prompt_key=prompt_key,
                    model=result.model,
                )

            # Post to dream channel if requested or if it's a tracked message.
            # This is the unified entry point for all LLM auditing UI.
            bot = bot or get_discord_bot()
            dream_channel_id_str = os.getenv("DISCORD_DREAM_CHANNEL_ID")

            should_post = audit_view or (
                bot is not None
                and dream_channel_id_str is not None
                and main_discord_message_id is not None
            )

            if should_post and bot:
                effective_dream_channel_id: int | None = None
                try:
                    if dream_channel_id:
                        effective_dream_channel_id = dream_channel_id
                    elif dream_channel_id_str:
                        effective_dream_channel_id = int(dream_channel_id_str)
                    elif hasattr(bot, "dream_channel_id"):
                        effective_dream_channel_id = bot.dream_channel_id
                except (ValueError, TypeError):
                    logger.warning("Invalid dream channel ID configuration")

                if effective_dream_channel_id:
                    params = audit_view_params or {}
                    from lattice.discord_client.dream import AuditViewBuilder

                    # Standardize metadata
                    metadata = params.get(
                        "metadata",
                        [
                            f"Model: {result.model}",
                            f"Tokens: {result.total_tokens}",
                            f"Latency: {result.latency_ms}ms",
                        ],
                    )
                    if audit_id:
                        metadata.append(f"Audit ID: {audit_id}")

                    try:
                        embed, view = AuditViewBuilder.build_standard_audit(
                            prompt_key=prompt_key or "UNKNOWN",
                            version=template_version or 1,
                            input_text=params.get("input_text", prompt[:200] + "..."),
                            output_text=params.get("output_text", result.content),
                            metadata_parts=metadata,
                            audit_id=audit_id,
                            rendered_prompt=prompt,
                            result=result,
                            message_id=main_discord_message_id,
                        )

                        channel = bot.get_channel(effective_dream_channel_id)
                        if channel:
                            await channel.send(embed=embed, view=view)
                            bot.add_view(view)
                    except Exception as e:
                        logger.warning(
                            "Failed to post rich audit to dream channel",
                            error=str(e),
                            prompt_key=prompt_key,
                        )

            return AuditResult(
                content=result.content,
                model=result.model,
                provider=result.provider,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
                temperature=result.temperature,
                audit_id=audit_id,
                prompt_key=prompt_key,
            )

        except Exception as e:
            logger.error(
                "LLM call failed",
                error=str(e),
                prompt_key=prompt_key,
                error_type=type(e).__name__,
            )

            from lattice.discord_client.error_notifier import mirror_llm_error

            effective_bot = bot if bot is not None else _get_discord_bot()
            effective_dream_channel_id = (
                dream_channel_id
                if dream_channel_id
                else (effective_bot.dream_channel_id if effective_bot else None)
            )

            if effective_dream_channel_id and effective_bot:
                await mirror_llm_error(
                    bot=effective_bot,
                    dream_channel_id=effective_dream_channel_id,
                    prompt_key=prompt_key or "UNKNOWN",
                    error_type=type(e).__name__,
                    error_message=str(e)[:500],
                    prompt_preview=prompt[:200],
                )

            from lattice.memory import prompt_audits

            failed_audit_id = await prompt_audits.store_prompt_audit(
                db_pool=db_pool,
                prompt_key=prompt_key or "UNKNOWN",
                rendered_prompt=prompt,
                response_content=f"ERROR: {type(e).__name__}: {str(e)}",
                main_discord_message_id=main_discord_message_id or 0,
                template_version=template_version,
                model="FAILED",
                provider=None,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0,
                latency_ms=0,
            )

            return AuditResult(
                content="",
                model="FAILED",
                provider=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0,
                latency_ms=0,
                temperature=temperature,
                audit_id=failed_audit_id,
                prompt_key=prompt_key,
            )


_discord_bot: Any | None = None


def set_discord_bot(bot: Any) -> None:
    """Set the global Discord bot instance for error mirroring.

    Args:
        bot: Discord bot instance
    """
    global _discord_bot
    _discord_bot = bot


def get_discord_bot() -> Any | None:
    """Get the global Discord bot instance.

    Returns:
        Discord bot instance or None if not set
    """
    return _discord_bot


def _get_discord_bot() -> Any | None:
    """Internal getter for Discord bot instance.

    Returns:
        Discord bot instance or None if not set
    """
    return _discord_bot
