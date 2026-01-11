"""Prompt audit module - tracks prompts, responses, and feedback for analysis.

Stores complete audit trail for Dreaming Cycle to analyze prompt effectiveness.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool


logger = structlog.get_logger(__name__)


@dataclass
class PromptAudit:
    """Represents a prompt audit entry."""

    # Core identifiers
    audit_id: UUID
    prompt_key: str
    template_version: int | None
    message_id: UUID | None

    # Prompt & response
    rendered_prompt: str
    response_content: str

    # Performance metrics
    model: str | None
    provider: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: Decimal | None
    latency_ms: int | None
    finish_reason: str | None

    # OpenRouter telemetry
    cache_discount_usd: Decimal | None
    native_tokens_cached: int | None
    native_tokens_reasoning: int | None
    upstream_id: str | None
    cancelled: bool | None
    moderation_latency_ms: int | None

    # Context configuration
    context_config: dict[str, Any] | None
    archetype_matched: str | None
    archetype_confidence: float | None

    # AI reasoning
    reasoning: dict[str, Any] | None

    # Discord linkage
    main_discord_message_id: int | None
    dream_discord_message_id: int | None

    # Feedback linkage
    feedback_id: UUID | None

    # Timestamp
    created_at: datetime


async def store_prompt_audit(
    db_pool: "DatabasePool",
    prompt_key: str,
    response_content: str,
    main_discord_message_id: int | None,
    rendered_prompt: str | None = None,
    template_version: int | None = None,
    message_id: UUID | None = None,
    model: str | None = None,
    provider: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_usd: float | None = None,
    latency_ms: int | None = None,
    finish_reason: str | None = None,
    cache_discount_usd: float | None = None,
    native_tokens_cached: int | None = None,
    native_tokens_reasoning: int | None = None,
    upstream_id: str | None = None,
    cancelled: bool | None = None,
    moderation_latency_ms: int | None = None,
    context_config: dict[str, Any] | None = None,
    archetype_matched: str | None = None,
    archetype_confidence: float | None = None,
    reasoning: dict[str, Any] | None = None,
    dream_discord_message_id: int | None = None,
) -> UUID:
    """Store a prompt audit entry.

    Args:
        db_pool: Database pool for dependency injection
        prompt_key: The prompt template key used
        response_content: The bot's response
        main_discord_message_id: Discord message ID in main channel
        rendered_prompt: The full rendered prompt sent to LLM (optional for internal LLM calls)
        template_version: Version of the prompt template
        message_id: UUID of the raw_messages entry
        model: LLM model used
        provider: LLM provider
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        cost_usd: Cost in USD
        latency_ms: Response latency in milliseconds
        finish_reason: Why generation stopped (stop, length, content_filter)
        cache_discount_usd: Cost savings from prompt caching
        native_tokens_cached: Number of tokens served from cache
        native_tokens_reasoning: Reasoning tokens for o1-style models
        upstream_id: Traceability to underlying provider
        cancelled: Whether the request was cancelled
        moderation_latency_ms: Content moderation overhead
        context_config: Context configuration used
        archetype_matched: Archetype that was matched
        archetype_confidence: Confidence score for archetype match
        reasoning: AI reasoning and decision factors
        dream_discord_message_id: Discord message ID in dream channel

    Returns:
        UUID of the stored audit entry
    """
    context_config_json = json.dumps(context_config) if context_config else None
    reasoning_json = json.dumps(reasoning) if reasoning else None

    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO prompt_audits (
                prompt_key,
                template_version,
                message_id,
                rendered_prompt,
                response_content,
                model,
                provider,
                prompt_tokens,
                completion_tokens,
                cost_usd,
                latency_ms,
                finish_reason,
                cache_discount_usd,
                native_tokens_cached,
                native_tokens_reasoning,
                upstream_id,
                cancelled,
                moderation_latency_ms,
                context_config,
                archetype_matched,
                archetype_confidence,
                reasoning,
                main_discord_message_id,
                dream_discord_message_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
            RETURNING id
            """,
            prompt_key,
            template_version,
            message_id,
            rendered_prompt,
            response_content,
            model,
            provider,
            prompt_tokens,
            completion_tokens,
            cost_usd,
            latency_ms,
            finish_reason,
            cache_discount_usd,
            native_tokens_cached,
            native_tokens_reasoning,
            upstream_id,
            cancelled,
            moderation_latency_ms,
            context_config_json,
            archetype_matched,
            archetype_confidence,
            reasoning_json,
            main_discord_message_id,
            dream_discord_message_id,
        )

        if row is None:
            # This should not happen with RETURNING, but for robustness
            msg = "Failed to store prompt audit"
            raise RuntimeError(msg)

        audit_id = row["id"]

        logger.info(
            "Stored prompt audit",
            audit_id=str(audit_id),
            prompt_key=prompt_key,
            main_message_id=main_discord_message_id,
            dream_message_id=dream_discord_message_id,
        )

        return audit_id


async def update_audit_dream_message(
    db_pool: "DatabasePool", audit_id: UUID, dream_discord_message_id: int
) -> bool:
    """Update audit with dream channel message ID.

    Args:
        db_pool: Database pool for dependency injection
        audit_id: UUID of the audit entry
        dream_discord_message_id: Discord message ID in dream channel

    Returns:
        True if updated, False if not found
    """
    async with db_pool.pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE prompt_audits
            SET dream_discord_message_id = $1
            WHERE id = $2
            """,
            dream_discord_message_id,
            audit_id,
        )

        updated = result == "UPDATE 1"

        if updated:
            logger.info(
                "Updated audit with dream message",
                audit_id=str(audit_id),
                dream_message_id=dream_discord_message_id,
            )

        return updated


async def link_feedback_to_audit(
    db_pool: "DatabasePool", dream_discord_message_id: int, feedback_id: UUID
) -> bool:
    """Link feedback to prompt audit via dream channel message ID.

    Args:
        db_pool: Database pool for dependency injection
        dream_discord_message_id: Discord message ID in dream channel
        feedback_id: UUID of the feedback entry

    Returns:
        True if linked, False if audit not found
    """
    async with db_pool.pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE prompt_audits
            SET feedback_id = $1
            WHERE dream_discord_message_id = $2
            """,
            feedback_id,
            dream_discord_message_id,
        )

        updated = result == "UPDATE 1"

        if updated:
            logger.info(
                "Linked feedback to audit",
                feedback_id=str(feedback_id),
                dream_message_id=dream_discord_message_id,
            )

        return updated


async def link_feedback_to_audit_by_id(
    db_pool: "DatabasePool", audit_id: UUID, feedback_id: UUID
) -> bool:
    """Link feedback to prompt audit via audit UUID.

    Unlike link_feedback_to_audit(), this function looks up audits by their internal UUID
    rather than dream_discord_message_id. This is necessary because batch extraction audits
    (MEMORY_CONSOLIDATION) may not always get mirrored to the dream channel,
    so they may have NULL for dream_discord_message_id.

    Args:
        db_pool: Database pool for dependency injection
        audit_id: UUID of the prompt audit entry
        feedback_id: UUID of the feedback entry

    Returns:
        True if linked, False if audit not found
    """
    async with db_pool.pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE prompt_audits
            SET feedback_id = $1
            WHERE id = $2
            """,
            feedback_id,
            audit_id,
        )

        updated = result == "UPDATE 1"

        if updated:
            logger.info(
                "Linked feedback to audit",
                feedback_id=str(feedback_id),
                audit_id=str(audit_id),
            )

        return updated


async def get_audit_by_dream_message(
    db_pool: "DatabasePool", dream_discord_message_id: int
) -> PromptAudit | None:
    """Get audit by dream channel message ID.

    Args:
        db_pool: Database pool for dependency injection
        dream_discord_message_id: Discord message ID in dream channel

    Returns:
        PromptAudit if found, None otherwise
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id, prompt_key, template_version, message_id,
                rendered_prompt, response_content,
                model, provider, prompt_tokens, completion_tokens,
                cost_usd, latency_ms,
                finish_reason, cache_discount_usd, native_tokens_cached,
                native_tokens_reasoning, upstream_id, cancelled, moderation_latency_ms,
                context_config, archetype_matched, archetype_confidence,
                reasoning,
                main_discord_message_id, dream_discord_message_id,
                feedback_id, created_at
            FROM prompt_audits
            WHERE dream_discord_message_id = $1
            """,
            dream_discord_message_id,
        )

        if not row:
            return None

        return PromptAudit(
            audit_id=row["id"],
            prompt_key=row["prompt_key"],
            template_version=row["template_version"],
            message_id=row["message_id"],
            rendered_prompt=row["rendered_prompt"],
            response_content=row["response_content"],
            model=row["model"],
            provider=row["provider"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            cost_usd=row["cost_usd"],
            latency_ms=row["latency_ms"],
            finish_reason=row["finish_reason"],
            cache_discount_usd=row["cache_discount_usd"],
            native_tokens_cached=row["native_tokens_cached"],
            native_tokens_reasoning=row["native_tokens_reasoning"],
            upstream_id=row["upstream_id"],
            cancelled=row["cancelled"],
            moderation_latency_ms=row["moderation_latency_ms"],
            context_config=row["context_config"],
            archetype_matched=row["archetype_matched"],
            archetype_confidence=row["archetype_confidence"],
            reasoning=row["reasoning"],
            main_discord_message_id=row["main_discord_message_id"],
            dream_discord_message_id=row["dream_discord_message_id"],
            feedback_id=row["feedback_id"],
            created_at=row["created_at"],
        )


async def get_audits_with_feedback(
    db_pool: "DatabasePool", limit: int = 100, offset: int = 0
) -> list[PromptAudit]:
    """Get prompt audits that have feedback.

    Args:
        db_pool: Database pool for dependency injection
        limit: Maximum number of audits to return
        offset: Offset for pagination

    Returns:
        List of prompt audits with feedback
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id, prompt_key, template_version, message_id,
                rendered_prompt, response_content,
                model, provider, prompt_tokens, completion_tokens,
                cost_usd, latency_ms,
                finish_reason, cache_discount_usd, native_tokens_cached,
                native_tokens_reasoning, upstream_id, cancelled, moderation_latency_ms,
                context_config, archetype_matched, archetype_confidence,
                reasoning,
                main_discord_message_id, dream_discord_message_id,
                feedback_id, created_at
            FROM prompt_audits
            WHERE feedback_id IS NOT NULL
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )

        return [
            PromptAudit(
                audit_id=row["id"],
                prompt_key=row["prompt_key"],
                template_version=row["template_version"],
                message_id=row["message_id"],
                rendered_prompt=row["rendered_prompt"],
                response_content=row["response_content"],
                model=row["model"],
                provider=row["provider"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                cost_usd=row["cost_usd"],
                latency_ms=row["latency_ms"],
                finish_reason=row["finish_reason"],
                cache_discount_usd=row["cache_discount_usd"],
                native_tokens_cached=row["native_tokens_cached"],
                native_tokens_reasoning=row["native_tokens_reasoning"],
                upstream_id=row["upstream_id"],
                cancelled=row["cancelled"],
                moderation_latency_ms=row["moderation_latency_ms"],
                context_config=row["context_config"],
                archetype_matched=row["archetype_matched"],
                archetype_confidence=row["archetype_confidence"],
                reasoning=row["reasoning"],
                main_discord_message_id=row["main_discord_message_id"],
                dream_discord_message_id=row["dream_discord_message_id"],
                feedback_id=row["feedback_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
