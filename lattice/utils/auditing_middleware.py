"""Auditing middleware for LLM calls.

This module provides AuditingLLMClient which wraps the basic LLM client
to add audit tracking, feedback linkage, and error mirroring.

Technical Debt (Phase 2 of #169): This module contains late-bound imports
from discord_client (error_notifier, dream) as a workaround. These cross-package
dependencies will be eliminated in Phase 3-4 (DI Infrastructure) via
proper dependency injection. See: https://github.com/chriskevini/lattice/issues/169
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

import structlog

from lattice.utils.config import config
from lattice.utils.llm_client import GenerationResult

if TYPE_CHECKING:
    from lattice.memory.repositories import (
        PromptAuditRepository,
        UserFeedbackRepository,
    )


logger = structlog.get_logger(__name__)

PROMPT_TOKENS_WARNING_THRESHOLD = 5000


def _collect_warnings(
    max_tokens: int | None,
    prompt_tokens: int,
    finish_reason: str | None,
) -> list[str]:
    """Collect warnings for suspicious patterns in LLM calls.

    Args:
        max_tokens: Maximum tokens requested (None = unlimited)
        prompt_tokens: Number of tokens in the prompt
        finish_reason: Reason the LLM stopped generating

    Returns:
        List of warning messages
    """
    warnings = []

    if max_tokens is None:
        warnings.append("max_tokens=None (unlimited generation)")

    if prompt_tokens >= PROMPT_TOKENS_WARNING_THRESHOLD:
        warnings.append(f"prompt_tokens={prompt_tokens} (large prompt)")

    if finish_reason == "length":
        warnings.append("Response truncated by token limit")

    return warnings


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

    def __init__(
        self,
        llm_client: Any,
        audit_repo: Any = None,
        feedback_repo: Any = None,
    ) -> None:
        """Initialize the auditing client with underlying LLM client.

        Args:
            llm_client: The underlying LLM client to wrap.
            audit_repo: Optional repository for auditing.
            feedback_repo: Optional repository for feedback.
        """
        self._client = llm_client
        self.audit_repo = audit_repo
        self.feedback_repo = feedback_repo

    async def complete(
        self,
        prompt: str,
        audit_repo: Optional["PromptAuditRepository"] = None,
        feedback_repo: Optional["UserFeedbackRepository"] = None,
        prompt_key: str | None = None,
        template_version: int | None = None,
        main_discord_message_id: int | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        dream_channel_id: int | None = None,
        bot: Any | None = None,
        audit_view: bool = False,
        audit_view_params: dict[str, Any] | None = None,
        execution_metadata: dict[str, Any] | None = None,
    ) -> AuditResult:
        """Complete a prompt with automatic audit tracking and optional AuditView.

        Args:
            prompt: The prompt to complete
            audit_repo: Audit repository for dependency injection
            feedback_repo: Feedback repository for dependency injection
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
            execution_metadata: Execution metadata (context config, batch metrics, etc.)

        Returns:
            AuditResult with content, metadata, and audit_id
        """
        effective_audit_repo = audit_repo or self.audit_repo
        if effective_audit_repo is None:
            msg = (
                "audit_repo must be provided either at initialization or in complete()"
            )
            raise ValueError(msg)

        try:
            result = await self._client.complete(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            from lattice.memory import prompt_audits

            audit_id = await prompt_audits.store_prompt_audit(
                repo=effective_audit_repo,
                prompt_key=prompt_key or "UNKNOWN",
                response_content=result.content,
                main_discord_message_id=main_discord_message_id,
                rendered_prompt=prompt,
                template_version=template_version,
                model=result.model,
                provider=result.provider,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
                finish_reason=result.finish_reason,
                cache_discount_usd=result.cache_discount_usd,
                native_tokens_cached=result.native_tokens_cached,
                native_tokens_reasoning=result.native_tokens_reasoning,
                upstream_id=result.upstream_id,
                cancelled=result.cancelled,
                moderation_latency_ms=result.moderation_latency_ms,
                execution_metadata=execution_metadata,
            )

            logger.info(
                "Audited LLM call",
                audit_id=str(audit_id),
                prompt_key=prompt_key,
                model=result.model,
            )

            # Post to dream channel if requested or if it's a tracked message.
            dream_channel_id_from_config = config.discord_dream_channel_id

            should_post = audit_view or (
                bot is not None and dream_channel_id_from_config is not None
            )

            if should_post and bot:
                effective_dream_channel_id: int | None = None
                try:
                    if dream_channel_id:
                        effective_dream_channel_id = dream_channel_id
                    elif dream_channel_id_from_config:
                        effective_dream_channel_id = dream_channel_id_from_config
                    elif hasattr(bot, "dream_channel_id"):
                        effective_dream_channel_id = bot.dream_channel_id
                except (ValueError, TypeError):
                    logger.warning("Invalid dream channel ID configuration")

                if effective_dream_channel_id:
                    from lattice.discord_client.dream import AuditViewBuilder

                    params = audit_view_params or {}
                    metadata = params.get("metadata", [])

                    if result.cost_usd is not None:
                        metadata.append(f"Cost: ${result.cost_usd:.4f}")

                    metadata.append(f"Model: {result.model}")
                    metadata.append(f"Tokens: {result.total_tokens}")
                    metadata.append(f"Latency: {result.latency_ms}ms")

                    if result.finish_reason:
                        if result.finish_reason == "stop":
                            metadata.append("✓ Complete")
                        elif result.finish_reason == "length":
                            metadata.append("⚠️ Truncated")
                        elif result.finish_reason == "content_filter":
                            metadata.append("🚫 Filtered")
                        else:
                            metadata.append(f"Finish: {result.finish_reason}")

                    if (
                        result.cache_discount_usd is not None
                        and result.cache_discount_usd > 0
                    ):
                        metadata.append(f"🔒 Saved: ${result.cache_discount_usd:.4f}")
                        if result.native_tokens_cached:
                            metadata.append(f"({result.native_tokens_cached} cached)")

                    if result.native_tokens_reasoning:
                        metadata.append(
                            f"🧠 Reasoning: {result.native_tokens_reasoning}"
                        )

                    if audit_id:
                        metadata.append(f"Audit ID: {audit_id}")

                    warnings = _collect_warnings(
                        max_tokens=max_tokens,
                        prompt_tokens=result.prompt_tokens or 0,
                        finish_reason=result.finish_reason,
                    )

                    try:
                        embed, view = AuditViewBuilder.build_standard_audit(
                            prompt_key=prompt_key or "UNKNOWN",
                            version=template_version or 1,
                            input_text=params.get("input_text", prompt[:200] + "..."),
                            output_text=params.get("output_text", result.content),
                            metadata_parts=metadata,
                            audit_id=audit_id,
                            rendered_prompt=prompt,
                            audit_repo=effective_audit_repo,
                            feedback_repo=feedback_repo or self.feedback_repo,
                            result=result,
                            message_id=main_discord_message_id,
                            warnings=warnings,
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
                finish_reason=result.finish_reason,
                cache_discount_usd=result.cache_discount_usd,
                native_tokens_cached=result.native_tokens_cached,
                native_tokens_reasoning=result.native_tokens_reasoning,
                upstream_id=result.upstream_id,
                cancelled=result.cancelled,
                moderation_latency_ms=result.moderation_latency_ms,
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

            effective_dream_channel_id = (
                dream_channel_id
                if dream_channel_id
                else (bot.dream_channel_id if bot else None)
            )

            if effective_dream_channel_id and bot:
                await mirror_llm_error(
                    bot=bot,
                    dream_channel_id=effective_dream_channel_id,
                    prompt_key=prompt_key or "UNKNOWN",
                    error_type=type(e).__name__,
                    error_message=str(e)[:500],
                    prompt_preview=prompt[:200],
                )

            from lattice.memory import prompt_audits

            failed_audit_id = await prompt_audits.store_prompt_audit(
                repo=effective_audit_repo,
                prompt_key=prompt_key or "UNKNOWN",
                rendered_prompt=prompt,
                response_content=f"ERROR: {type(e).__name__}: {str(e)}",
                main_discord_message_id=main_discord_message_id,
                template_version=template_version,
                model="FAILED",
                provider=None,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0,
                latency_ms=0,
                finish_reason=None,
                cache_discount_usd=None,
                native_tokens_cached=None,
                native_tokens_reasoning=None,
                upstream_id=None,
                cancelled=None,
                moderation_latency_ms=None,
                execution_metadata=execution_metadata,
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
