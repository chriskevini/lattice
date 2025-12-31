"""Detection functions for short-circuit logic.

Identifies invisible feedback and North Star declarations.
"""

import re
from dataclasses import dataclass
from typing import Any

import discord
import structlog


logger = structlog.get_logger(__name__)

SALUTE_EMOJI = "🫡"
WASTEBASKET_EMOJI = "🗑️"

NORTH_STAR_PATTERNS = [
    re.compile(r"(?i)^north\s*star[:\s]+(.+)$"),
    re.compile(r"(?i)^ns[:\s]+(.+)$"),
]

MIN_GOAL_LENGTH = 10
MAX_GOAL_LENGTH = 500


@dataclass
class DetectionResult:
    """Result of a detection check."""

    detected: bool
    content: str | None = None
    confidence: float = 0.0


def is_quote_or_reply(message: Any) -> bool:  # noqa: ANN401
    """Check if a message is a quote or reply to another message.

    Args:
        message: The Discord message to check

    Returns:
        True if message is a quote or reply
    """
    return (
        getattr(message, "reference", None) is not None
        or getattr(message, "type", None) == discord.MessageType.reply
    )


def get_referenced_message_id(message: Any) -> int | None:  # noqa: ANN401
    """Get the ID of the message being quoted/replied to.

    Args:
        message: The Discord message to check

    Returns:
        The message ID being referenced, or None
    """
    reference = getattr(message, "reference", None)
    if reference is None:
        return None
    return getattr(reference, "message_id", None)


def is_invisible_feedback(message: Any) -> DetectionResult:  # noqa: ANN401
    """Check if a message is invisible feedback (quote/reply to bot).

    Args:
        message: The Discord message to check

    Returns:
        DetectionResult with detected status and content
    """
    if getattr(message, "author", None) and getattr(message.author, "bot", False):
        return DetectionResult(detected=False)

    if not is_quote_or_reply(message):
        return DetectionResult(detected=False)

    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = ""

    logger.info(
        "Detected invisible feedback",
        author=getattr(getattr(message, "author", None), "name", "unknown"),
        referenced_message=get_referenced_message_id(message),
    )

    return DetectionResult(detected=True, content=content, confidence=1.0)


def is_north_star(message: Any) -> DetectionResult:  # noqa: ANN401
    """Check if a message is a North Star declaration.

    Args:
        message: The Discord message to check

    Returns:
        DetectionResult with detected status and content
    """
    if getattr(message, "author", None) and getattr(message.author, "bot", False):
        return DetectionResult(detected=False)

    content = getattr(message, "content", "").strip()

    if len(content) < MIN_GOAL_LENGTH or len(content) > MAX_GOAL_LENGTH:
        return DetectionResult(detected=False)

    best_match = None
    for pattern in NORTH_STAR_PATTERNS:
        match = pattern.match(content)
        if match:
            best_match = match
            break

    if best_match:
        extracted_goal = best_match.group(1).strip()
        if extracted_goal:
            confidence = 0.7 + (len(extracted_goal) / 1000)
            confidence = min(confidence, 0.95)

            author_name = getattr(getattr(message, "author", None), "name", "unknown")
            logger.info(
                "Detected North Star declaration",
                author=author_name,
                goal_preview=extracted_goal[:50],
                confidence=confidence,
            )
            return DetectionResult(detected=True, content=extracted_goal, confidence=confidence)

    return DetectionResult(detected=False)
