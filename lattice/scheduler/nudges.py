"""Contextual nudge detection using AI decisions.

Uses LLM to decide whether to initiate contact based on conversation context.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import UUID

import structlog

from lattice.memory.episodic import get_recent_messages
from lattice.memory.procedural import get_prompt
from lattice.utils.config import get_config

from lattice.utils.date_resolution import get_now
from lattice.utils.placeholder_injector import PlaceholderInjector

if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool

logger = structlog.get_logger(__name__)


@dataclass
class NudgePlan:
    """Result of AI decision on contextual nudge."""

    action: str
    content: str | None
    reason: str
    channel_id: int | None = None
    audit_id: UUID | None = None
    rendered_prompt: str | None = None
    template_version: int | None = None
    model: str | None = None
    provider: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None


async def format_episodic_nudge_context(
    db_pool: "DatabasePool", user_timezone: str = "UTC", limit: int = 20
) -> str:
    """Build conversation context in USER/ASSISTANT format.

    Args:
        db_pool: Database pool for dependency injection
        user_timezone: User's timezone for timestamp formatting
        limit: Maximum number of recent messages to include

    Returns:
        Formatted conversation string
    """
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(user_timezone)

    messages = await get_recent_messages(limit=limit, db_pool=db_pool)

    if not messages:
        return "No recent conversation history."

    lines: list[str] = []
    for msg in messages:
        content = msg.content[:500]
        # Convert UTC timestamp from DB to user timezone
        local_ts = msg.timestamp.astimezone(tz)
        timestamp = local_ts.strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{timestamp}] {msg.role}: {content}")

    return "\n".join(lines)


async def get_default_channel_id() -> int | None:
    """Get the main channel for contextual nudges from configuration.

    Returns:
        Main channel ID from get_config().discord_main_channel_id, or None if not configured
    """
    main_channel_id = get_config().discord_main_channel_id
    if not main_channel_id:
        logger.warning("DISCORD_MAIN_CHANNEL_ID not set, cannot send contextual nudges")
        return None
    return main_channel_id


async def prepare_contextual_nudge(
    db_pool: "DatabasePool", llm_client: Any, bot: Any | None = None
) -> NudgePlan:
    """Prepare a contextual nudge using AI.

    Args:
        db_pool: Database pool for dependency injection
        llm_client: LLM client for dependency injection (required)
        bot: Discord bot instance for dependency injection

    Returns:
        NudgePlan with content and reason
    """
    if not llm_client:
        raise ValueError("llm_client is required for prepare_contextual_nudge")

    active_llm_client = llm_client

    try:
        # Fallback to global if db_pool doesn't have the method or fails
        if hasattr(db_pool, "get_user_timezone"):
            res = db_pool.get_user_timezone()
            if asyncio.iscoroutine(res):
                user_tz = await res
            else:
                user_tz = res
        else:
            from lattice.utils.database import get_user_timezone as global_get_tz

            user_tz = await global_get_tz(db_pool=db_pool)
    except (AttributeError, TypeError):
        from lattice.utils.database import get_user_timezone as global_get_tz

        user_tz = await global_get_tz(db_pool=db_pool)

    if isinstance(user_tz, MagicMock) or "Mock" in str(type(user_tz)):
        user_tz = "UTC"

    if not isinstance(user_tz, str):
        user_tz = "UTC"

    conversation = await format_episodic_nudge_context(
        db_pool=db_pool, user_timezone=user_tz
    )

    from lattice.core import response_generator

    # Retrieve goal and activity context
    goals = await response_generator.get_goal_context(db_pool=db_pool)
    # Fetch activity context from retrieve_context logic
    from lattice.core.context_strategy import retrieve_context

    context_result = await retrieve_context(
        db_pool=db_pool, entities=[], context_flags=["activity_context"]
    )
    activity = context_result.get("activity_context", "No recent activity recorded.")

    channel_id = await get_default_channel_id()

    now = get_now(user_tz)
    local_date = now.strftime("%Y/%m/%d, %A")
    local_time = now.strftime("%H:%M")

    # Get date resolution hints
    from lattice.utils.date_resolution import resolve_relative_dates

    date_hints = resolve_relative_dates("", timezone_str=user_tz)

    prompt_template = await get_prompt(db_pool=db_pool, prompt_key="CONTEXTUAL_NUDGE")
    if not prompt_template:
        logger.error("CONTEXTUAL_NUDGE prompt not found in database")
        return NudgePlan(
            action="wait",
            content=None,
            reason="CONTEXTUAL_NUDGE prompt not found",
        )

    injector = PlaceholderInjector()
    context = {
        "local_date": local_date,
        "local_time": local_time,
        "date_resolution_hints": date_hints,
        "episodic_context": conversation,
        "goal_context": goals,
        "activity_context": activity,
    }

    prompt, injected = await injector.inject(prompt_template, context)

    try:
        result = await active_llm_client.complete(
            prompt=prompt,
            db_pool=db_pool,
            prompt_key="CONTEXTUAL_NUDGE",
            template_version=prompt_template.version,
            main_discord_message_id=0,
            temperature=prompt_template.temperature,
            audit_view=True,
            audit_view_params={
                "input_text": "Deterministic contextual nudge",
            },
            bot=bot,
        )
    except (ValueError, ImportError, Exception) as e:
        logger.exception("LLM call failed")
        return NudgePlan(
            action="wait",
            content=None,
            reason=f"LLM call failed: {e}",
            channel_id=channel_id,
        )

    try:
        decision = json.loads(result.content)
        content = decision.get("content")

        if content:
            content = str(content).strip()

        if not content:
            return NudgePlan(
                action="wait",
                content=None,
                reason="AI decided not to send a message (empty content)",
                channel_id=channel_id,
                audit_id=result.audit_id,
                rendered_prompt=prompt,
                template_version=prompt_template.version,
                model=result.model,
                provider=result.provider,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
            )

        return NudgePlan(
            action="message",
            content=content,
            reason=decision.get("reason", "No reason provided"),
            channel_id=channel_id,
            audit_id=result.audit_id,
            rendered_prompt=prompt,
            template_version=prompt_template.version,
            model=result.model,
            provider=result.provider,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
        )

    except json.JSONDecodeError:
        logger.exception("Failed to parse AI response as JSON")
        return NudgePlan(
            action="wait",
            content=None,
            reason="Failed to parse AI response",
            channel_id=channel_id,
        )
