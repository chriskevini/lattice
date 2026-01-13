"""Contextual nudge generation for proactive check-ins.

Uses LLM to generate goal-aligned check-in messages.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from lattice.utils.config import get_config
from lattice.utils.date_resolution import get_now
from lattice.utils.placeholder_injector import PlaceholderInjector

if TYPE_CHECKING:
    pass


logger = structlog.get_logger(__name__)


@dataclass
class NudgePlan:
    """Result of contextual nudge generation."""

    content: str | None
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


async def get_default_channel_id() -> int | None:
    """Get the main channel for contextual nudges from configuration."""
    main_channel_id = get_config().discord_main_channel_id
    if not main_channel_id:
        logger.warning("DISCORD_MAIN_CHANNEL_ID not set, cannot send contextual nudges")
        return None
    return main_channel_id


async def prepare_contextual_nudge(
    llm_client: Any,
    user_context_cache: Any,
    user_id: str,
    prompt_template: Any,
    db_pool: Any = None,
    bot: Any | None = None,
    semantic_repo: Any = None,
) -> NudgePlan:
    """Prepare a contextual nudge using AI.

    Args:
        llm_client: LLM client for dependency injection (required)
        user_context_cache: User-level cache for goals/activities/timezone
        user_id: Discord user ID for the owner
        prompt_template: Pre-fetched prompt template from DB
        db_pool: Database pool for fallback fetches (optional)
        bot: Discord bot instance for dependency injection
        semantic_repo: Semantic memory repository for goal retrieval (optional)

    Returns:
        NudgePlan with nudge content
    """
    if not llm_client:
        raise ValueError("llm_client is required for prepare_contextual_nudge")

    user_tz = user_context_cache.get_timezone(user_id)
    if not user_tz:
        user_tz = "UTC"

    from lattice.core import response_generator

    goals = user_context_cache.get_goals(user_id)
    if goals is None and semantic_repo:
        goals = await response_generator.get_goal_context(semantic_repo=semantic_repo)
        await user_context_cache.set_goals(user_id, goals)

    from lattice.core.context_strategy import retrieve_context

    activity = user_context_cache.get_activities(user_id)
    if activity is None and semantic_repo:
        context_result = await retrieve_context(
            db_pool=db_pool,
            semantic_repo=semantic_repo,
            entities=[],
            context_flags=["activity_context"],
        )
        activity = context_result.get(
            "activity_context", "No recent activity recorded."
        )
        await user_context_cache.set_activities(user_id, activity)

    channel_id = await get_default_channel_id()

    now = get_now(user_tz)
    local_date = now.strftime("%Y/%m/%d, %A")
    local_time = now.strftime("%H:%M")

    if not prompt_template:
        logger.error("CONTEXTUAL_NUDGE prompt not found in database")
        return NudgePlan(content=None, channel_id=channel_id)

    injector = PlaceholderInjector()
    context = {
        "local_date": local_date,
        "local_time": local_time,
        "goal_context": goals,
        "activity_context": activity,
    }

    prompt, injected = await injector.inject(prompt_template, context)

    try:
        result = await llm_client.complete(
            prompt=prompt,
            db_pool=db_pool,
            prompt_key="CONTEXTUAL_NUDGE",
            template_version=prompt_template.version,
            main_discord_message_id=None,
            temperature=prompt_template.temperature,
            audit_view=True,
            audit_view_params={
                "input_text": "Contextual nudge",
            },
            bot=bot,
        )
    except (ValueError, ImportError, Exception):
        logger.exception("LLM call failed")
        return NudgePlan(content=None, channel_id=channel_id)

    content = result.content.strip() if result.content else ""

    if not content:
        return NudgePlan(
            content=None,
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
        content=content,
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
