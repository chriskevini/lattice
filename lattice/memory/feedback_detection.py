"""Detection functions for short-circuit logic.

Identifies invisible feedback and North Star declarations.
"""

import re
from dataclasses import dataclass
from typing import Any

import structlog


logger = structlog.get_logger(__name__)

SALUTE_EMOJI = "🫡"
WASTEBASKET_EMOJI = "🗑️"

NORTH_STAR_PATTERNS = [
    re.compile(r"(?i)^(?:my\s+)?(?:north\s*star|goal|objective|purpose|mission)[:\s]+(.+)$"),
    re.compile(r"(?i)^i(?:'m| am) (?:trying|aiming|working) (?:to |for )(.+)$"),
    re.compile(r"(?i)^i(?:'m| am) (?:focused|dedicated|committed) (?:on |to )(.+)$"),
    re.compile(r"(?i)^my (?:main |primary |)focus (?:is |on |)[:\s]+(.+)$"),
    re.compile(r"(?i)^my goal (?:is |)[:\s]*(.+)$"),
    re.compile(r"(?i)^i(?:'m| am) (?:building|making|creating) (.+)$"),
    re.compile(r"(?i)^i(?:'m| am) learning (.+)$"),
    re.compile(r"(?i)^i(?:'m| am) interested in (.+)$"),
    re.compile(r"(?i)^my passion (?:is |for |)[:\s]+(.+)$"),
    re.compile(r"(?i)^i want (?:to |)[:\s]*(.+)$"),
    re.compile(r"(?i)^i(?:'m| am) looking (?:to |for )(.+)$"),
]

FEEDBACK_INDICATOR_PATTERNS = [
    re.compile(r"(?i)^(?:feedback|note|actually)[:,\s]+(.+)$"),
    re.compile(r"(?i)^(?:correction|update)[:,\s]+(.+)$"),
    re.compile(r"(?i)^(?:just )?(?:so you know|FYI)[:,\s]+(.+)$"),
    re.compile(r"(?i)^(?:small |quick )?correction[:,\s]*(.+)$"),
]


@dataclass
class DetectionResult:
    """Result of a detection check."""

    detected: bool
    content: str | None = None
    confidence: float = 0.0


def is_invisible_feedback(message: Any) -> DetectionResult:
    """Check if a message is invisible feedback (quote/reply to bot).

    Args:
        message: The Discord message to check

    Returns:
        DetectionResult with detected status and content
    """
    if getattr(message, "author", None) and getattr(message.author, "bot", False):
        return DetectionResult(detected=False)

    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = ""

    indicator_matches = [pattern.search(content) for pattern in FEEDBACK_INDICATOR_PATTERNS]
    valid_matches = [m for m in indicator_matches if m]

    is_reply = getattr(message, "reference", None) is not None
    has_quote = content.startswith(">")

    confidence = 0.5
    if is_reply:
        confidence += 0.3
    if has_quote:
        confidence += 0.2
    if valid_matches:
        confidence += 0.2
        extracted_content = valid_matches[0].group(1)
    else:
        extracted_content = content

    if confidence > 0.5:
        author_name = getattr(getattr(message, "author", None), "name", "unknown")
        logger.info(
            "Detected invisible feedback",
            author=author_name,
            is_reply=is_reply,
            has_quote=has_quote,
            confidence=confidence,
        )
        return DetectionResult(detected=True, content=extracted_content, confidence=confidence)

    return DetectionResult(detected=False)


def is_north_star(message: Any) -> DetectionResult:
    """Check if a message is a North Star declaration.

    Args:
        message: The Discord message to check

    Returns:
        DetectionResult with detected status and content
    """
    if getattr(message, "author", None) and getattr(message.author, "bot", False):
        return DetectionResult(detected=False)

    content = getattr(message, "content", "").strip()

    if len(content) < 10 or len(content) > 500:
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


def extract_goal_from_content(content: str) -> str | None:
    """Extract goal content from a message.

    Args:
        content: The message content

    Returns:
        Extracted goal text or None if no goal found
    """
    content = content.strip()

    for pattern in NORTH_STAR_PATTERNS:
        match = pattern.match(content)
        if match:
            goal = match.group(1).strip()
            if goal:
                return goal

    return None
