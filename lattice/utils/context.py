"""Context formatting utilities for LLM prompts."""

from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from lattice.memory.episodic import EpisodicMessage


def format_episodic_messages(messages: list["EpisodicMessage"]) -> str:
    """Format episodic messages with localized timestamps.

    Args:
        messages: List of episodic messages to format

    Returns:
        Formatted string like "[2026-01-09 14:30] USER: hello"
    """
    formatted_lines = []
    for msg in messages:
        try:
            # Handle timezone conversion
            user_tz = ZoneInfo(msg.user_timezone or "UTC")
            local_ts = msg.timestamp.astimezone(user_tz)
            ts_str = local_ts.strftime("%Y-%m-%d %H:%M")
        except Exception:
            # Fallback to UTC format if timezone logic fails
            ts_str = msg.timestamp.strftime("%Y-%m-%d %H:%M UTC")

        role = "ASSISTANT" if msg.is_bot else "USER"
        formatted_lines.append(f"[{ts_str}] {role}: {msg.content}")

    return "\n".join(formatted_lines)
