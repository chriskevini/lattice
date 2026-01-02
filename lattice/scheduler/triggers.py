"""Proactive trigger detection using AI decisions.

Uses LLM to decide whether to initiate contact based on conversation context.
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from lattice.memory.episodic import EpisodicMessage, get_recent_messages
from lattice.memory.procedural import get_prompt
from lattice.utils.database import db_pool, get_system_health, set_system_health
from lattice.utils.llm import get_llm_client


logger = logging.getLogger(__name__)


@dataclass
class ProactiveDecision:
    """Result of AI decision on proactive contact."""

    action: str
    content: str | None
    reason: str
    channel_id: int | None = None


async def get_current_interval() -> int:
    """Get the current wait interval in minutes from system_health."""
    base = int(await get_system_health("scheduler_base_interval") or 15)
    current = await get_system_health("scheduler_current_interval")
    return int(current) if current else base


async def set_current_interval(minutes: int) -> None:
    """Set the current wait interval in system_health."""
    await set_system_health("scheduler_current_interval", str(minutes))


def format_message(msg: EpisodicMessage) -> str:
    """Format a message for the conversation context."""
    role = "ASSISTANT" if msg.is_bot else "USER"
    content = msg.content[:500]
    return f"{role}: {content}"


async def get_conversation_context(limit: int = 20) -> str:
    """Build conversation context in USER/ASSISTANT format.

    Args:
        limit: Maximum number of recent messages to include

    Returns:
        Formatted conversation string
    """
    messages = await get_recent_messages(limit=limit)

    if not messages:
        return "No recent conversation history."

    lines: list[str] = [format_message(msg) for msg in messages]
    return "\n".join(lines)


async def get_objectives_context() -> str:
    """Get user's goals and objectives.

    Returns:
        Formatted objectives string
    """
    async with db_pool.pool.acquire() as conn:
        objectives = await conn.fetch(
            "SELECT description, status FROM objectives WHERE status = 'pending' ORDER BY saliency_score DESC LIMIT 5"
        )

    if not objectives:
        return "No active objectives."

    lines: list[str] = [f"- {obj['description']} ({obj['status']})" for obj in objectives]
    return "User goals:\n" + "\n".join(lines)


async def get_default_channel_id() -> int | None:
    """Get the default channel for proactive messages.

    For single-user setup, returns the first text channel.

    Returns:
        Channel ID or None if no suitable channel found
    """
    async with db_pool.pool.acquire() as conn:
        channel_id = await conn.fetchval(
            "SELECT channel_id FROM raw_messages ORDER BY timestamp DESC LIMIT 1"
        )
    return channel_id


async def decide_proactive() -> ProactiveDecision:
    """Ask AI whether to send a proactive message.

    Returns:
        ProactiveDecision with action, content, and reason
    """
    conversation = await get_conversation_context()
    objectives = await get_objectives_context()
    channel_id = await get_default_channel_id()
    current_interval = await get_current_interval()
    current_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    prompt_template = await get_prompt("PROACTIVE_DECISION")
    if not prompt_template:
        logger.error("PROACTIVE_DECISION prompt not found in database")
        return ProactiveDecision(
            action="wait",
            content=None,
            reason="PROACTIVE_DECISION prompt not found, defaulting to wait",
        )

    prompt = prompt_template.template.format(
        current_time=current_time,
        current_interval=current_interval,
        conversation_context=conversation,
        objectives_context=objectives,
    )

    llm_client = get_llm_client()
    try:
        result = await llm_client.complete(
            prompt,
            temperature=prompt_template.temperature,
        )
    except Exception as e:
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
        if action == "message" and content:
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
        )

    except json.JSONDecodeError:
        logger.exception("Failed to parse AI response as JSON")
        return ProactiveDecision(
            action="wait",
            content=None,
            reason="Failed to parse AI response",
            channel_id=channel_id,
        )
