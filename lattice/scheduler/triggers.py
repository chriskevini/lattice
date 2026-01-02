"""Proactive trigger detection using AI decisions.

Uses LLM to decide whether to initiate contact based on conversation context.
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from lattice.memory.episodic import EpisodicMessage, get_recent_messages
from lattice.memory.procedural import get_prompt
from lattice.utils.database import db_pool
from lattice.utils.llm import get_llm_client


logger = logging.getLogger(__name__)


@dataclass
class ProactiveDecision:
    """Result of AI decision on proactive contact."""

    action: str
    content: str | None
    next_check_at: str
    reason: str
    channel_id: int | None = None


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
        ProactiveDecision with action, content, next_check_at, and reason
    """
    conversation = await get_conversation_context()
    objectives = await get_objectives_context()
    channel_id = await get_default_channel_id()

    prompt_template = await get_prompt("PROACTIVE_DECISION")
    if not prompt_template:
        logger.error("PROACTIVE_DECISION prompt not found in database")
        return ProactiveDecision(
            action="wait",
            content=None,
            next_check_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            reason="PROACTIVE_DECISION prompt not found, defaulting to wait",
        )

    prompt = prompt_template.template.format(
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
            next_check_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
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

        next_check_raw = decision.get("next_check_at")
        next_check_at = None
        if next_check_raw:
            try:
                parsed = datetime.fromisoformat(str(next_check_raw))
                if parsed > datetime.now(UTC):
                    next_check_at = parsed.isoformat()
            except (ValueError, TypeError):
                logger.warning("Invalid next_check_at format: %s", next_check_raw)

        if not next_check_at:
            next_check_at = (datetime.now(UTC) + timedelta(hours=1)).isoformat()

        return ProactiveDecision(
            action=action,
            content=content,
            next_check_at=next_check_at,
            reason=decision.get("reason", "No reason provided"),
            channel_id=channel_id,
        )

    except json.JSONDecodeError:
        logger.exception("Failed to parse AI response as JSON")
        return ProactiveDecision(
            action="wait",
            content=None,
            next_check_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            reason="Failed to parse AI response",
            channel_id=channel_id,
        )
