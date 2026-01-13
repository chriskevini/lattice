"""Memory renderers for formatting semantic memories.

This module provides renderer functions for different memory types.
Each renderer produces a consistent string representation for LLM context.
"""

from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo


def render_activity(
    subject: str,
    predicate: str,
    obj: str,
    created_at: datetime | None = None,
    user_timezone: str | None = None,
) -> str:
    """Render activity as [timestamp] object.

    Args:
        subject: Subject entity
        predicate: Relationship predicate
        obj: Object entity or value
        created_at: Memory creation timestamp
        user_timezone: User's timezone for display (e.g., "America/New_York")

    Returns:
        Formatted activity string with timestamp prefix

    Examples:
        >>> from datetime import datetime, timezone
        >>> render_activity("User", "did activity", "coding", datetime(2026, 1, 12, 10, 30, tzinfo=timezone.utc))
        "[2026-01-12 10:30:00] coding"
    """
    if created_at:
        if user_timezone:
            tz = ZoneInfo(user_timezone)
            local_dt = created_at.astimezone(tz)
            timestamp_str = f"[{local_dt.strftime('%Y-%m-%d %H:%M:%S')}] "
        else:
            timestamp_str = f"[{created_at.strftime('%Y-%m-%d %H:%M:%S')}] "
    else:
        timestamp_str = ""
    return f"{timestamp_str}{obj}"


def render_goal(
    subject: str,
    predicate: str,
    obj: str,
    created_at: datetime | None = None,
    user_timezone: str | None = None,
) -> str:
    """Render goal as object only (for bullet point display).

    Args:
        subject: Subject entity
        predicate: Relationship predicate
        obj: Object entity or value
        created_at: Memory creation timestamp (unused for goals)
        user_timezone: User's timezone (unused for goals)

    Returns:
        Formatted goal string as object only

    Examples:
        >>> render_goal("User", "has goal", "run marathon", None)
        "run marathon"
    """
    return obj


def render_basic(
    subject: str,
    predicate: str,
    obj: str,
    created_at: datetime | None = None,
    user_timezone: str | None = None,
) -> str:
    """Render standard memory as subject predicate object.

    Args:
        subject: Subject entity
        predicate: Relationship predicate
        obj: Object entity or value
        created_at: Memory creation timestamp (unused for basic memories)
        user_timezone: User's timezone (unused for basic memories)

    Returns:
        Formatted memory string as full triple

    Examples:
        >>> render_basic("User", "lives in city", "Portland", None)
        "User lives in city Portland"
    """
    return f"{subject} {predicate} {obj}"


MemoryRenderer = Callable[
    [str, str, str, datetime | None, str | None],
    str,
]

MEMORY_RENDERERS: dict[str, MemoryRenderer] = {
    "activity_context": render_activity,
    "goal_context": render_goal,
    "semantic_context": render_basic,
}


def get_renderer(context_type: str) -> MemoryRenderer:
    """Get renderer for a context type.

    Args:
        context_type: Type of context (activity_context, goal_context, etc.)

    Returns:
        Renderer function for the context type, defaults to render_basic

    Examples:
        >>> from datetime import datetime, timezone
        >>> renderer = get_renderer("activity_context")
        >>> renderer("User", "did activity", "coding", datetime(2026, 1, 12, tzinfo=timezone.utc), None)
        "[2026-01-12 00:00:00] coding"
    """
    return MEMORY_RENDERERS.get(context_type, render_basic)
