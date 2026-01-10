"""Proactive trigger detection using AI decisions.

Uses LLM to decide whether to initiate contact based on conversation context.
Respects adaptive active hours to avoid disturbing users outside their active times.
"""

import json
import os
from dataclasses import dataclass
from uuid import UUID

import structlog

from lattice.memory.episodic import EpisodicMessage, get_recent_messages
from lattice.memory.procedural import get_prompt
from lattice.scheduler.adaptive import is_within_active_hours
from lattice.utils.database import (
    get_system_health,
    get_user_timezone,
    set_system_health,
)
from lattice.utils.llm import get_auditing_llm_client


logger = structlog.get_logger(__name__)


@dataclass
class ProactiveDecision:
    """Result of AI decision on proactive contact."""

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


async def get_current_interval() -> int:
    """Get the current wait interval in minutes from system_health."""
    base = int(await get_system_health("scheduler_base_interval") or 15)
    current = await get_system_health("scheduler_current_interval")
    return int(current) if current else base


async def set_current_interval(minutes: int) -> None:
    """Set the current wait interval in system_health."""
    await set_system_health("scheduler_current_interval", str(minutes))


async def get_conversation_context(user_timezone: str = "UTC", limit: int = 20) -> str:
    """Build conversation context in USER/ASSISTANT format.

    Args:
        user_timezone: User's timezone for timestamp formatting
        limit: Maximum number of recent messages to include

    Returns:
        Formatted conversation string
    """
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(user_timezone)

    messages = await get_recent_messages(limit=limit)

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
    """Get the main channel for proactive messages from environment variable.

    Proactive messages should always go to the main channel, never the dream channel.
    Dream channel is reserved for meta discussion (feedback, prompt refinement).

    Returns:
        Main channel ID from DISCORD_MAIN_CHANNEL_ID, or None if not configured
    """
    main_channel_id = os.getenv("DISCORD_MAIN_CHANNEL_ID")
    if not main_channel_id:
        logger.warning(
            "DISCORD_MAIN_CHANNEL_ID not set, cannot send proactive messages"
        )
        return None
    return int(main_channel_id)


async def decide_proactive() -> ProactiveDecision:
    """Ask AI whether to send a proactive message.

    First checks if current time is within user's active hours.
    If outside active hours, immediately returns "wait" decision.

    Returns:
        ProactiveDecision with action, content, and reason
    """
    # Check active hours first to avoid disturbing user
    if not await is_within_active_hours():
        logger.info("Outside active hours, skipping proactive check")
        return ProactiveDecision(
            action="wait",
            content=None,
            reason="Outside user's active hours",
        )

    from lattice.utils.database import get_user_timezone

    user_tz = await get_user_timezone()

    conversation = await get_conversation_context(user_timezone=user_tz)

    from lattice.core import response_generator
    from lattice.utils.placeholder_injector import PlaceholderInjector

    goals = await response_generator.get_goal_context()

    channel_id = await get_default_channel_id()
    current_interval = await get_current_interval()
    from lattice.utils.date_resolution import get_now

    from zoneinfo import ZoneInfo

    now = get_now(user_tz)
    local_date = now.strftime("%Y/%m/%d, %A")
    local_time = now.strftime("%H:%M")

    logger.debug(
        "Proactive check time debug",
        user_tz=user_tz,
        local_date=local_date,
        local_time=local_time,
    )

    prompt_template = await get_prompt("PROACTIVE_CHECKIN")
    if not prompt_template:
        logger.error("PROACTIVE_CHECKIN prompt not found in database")
        return ProactiveDecision(
            action="wait",
            content=None,
            reason="PROACTIVE_CHECKIN prompt not found, defaulting to wait",
        )

    injector = PlaceholderInjector()
    context = {
        "user_timezone": user_tz,
        "local_date": local_date,
        "local_time": local_time,
        "episodic_context": conversation,
        "goal_context": goals,
        "scheduler_current_interval": current_interval,
    }

    prompt, injected = await injector.inject(prompt_template, context)

    llm_client = get_auditing_llm_client()
    try:
        result = await llm_client.complete(
            prompt=prompt,
            prompt_key="PROACTIVE_CHECKIN",
            template_version=prompt_template.version,
            main_discord_message_id=0,  # Placeholder since no message exists yet
            temperature=prompt_template.temperature,
            audit_view=True,
            audit_view_params={
                "input_text": "Proactive check-in check",
            },
        )
    except (ValueError, ImportError) as e:
        logger.exception("LLM call failed")
        return ProactiveDecision(
            action="wait",
            content=None,
            reason=f"LLM call failed: {e}",
            channel_id=channel_id,
        )

    try:
        decision = json.loads(result.content)
        action = decision.get("action", "wait")
        if action not in ("message", "wait"):
            logger.warning("Invalid action from LLM: %s, defaulting to wait", action)
            action = "wait"

        content = decision.get("content")
        if action == "message":
            if content is None or content == "":
                logger.warning("Missing or empty content from LLM, skipping message")
                action = "wait"
                content = None
            else:
                content = str(content).strip()
                if len(content) == 0:
                    logger.warning("Empty content from LLM, skipping message")
                    action = "wait"
                    content = None

        return ProactiveDecision(
            action=action,
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
        return ProactiveDecision(
            action="wait",
            content=None,
            reason="Failed to parse AI response",
            channel_id=channel_id,
        )
