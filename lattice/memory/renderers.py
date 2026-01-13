"""Memory renderers for formatting semantic memories.

This module provides renderer functions for different memory types.
Each renderer produces a consistent string representation for LLM context.
"""

from datetime import datetime
from typing import Callable


def render_activity(
    subject: str,
    predicate: str,
    object: str,
    created_at: datetime | None = None,
) -> str:
    """Render activity as [timestamp] object.

    Args:
        subject: Subject entity
        predicate: Relationship predicate
        object: Object entity or value
        created_at: Memory creation timestamp

    Returns:
        Formatted activity string with timestamp prefix

    Examples:
        >>> render_activity("User", "did activity", "coding", datetime(2026, 1, 12, 10, 30))
        "[2026-01-12T10:30:00Z] coding"
    """
    timestamp_str = f"[{created_at.isoformat()}] " if created_at else ""
    return f"{timestamp_str}{object}"


def render_goal(
    subject: str,
    predicate: str,
    object: str,
    created_at: datetime | None = None,
) -> str:
    """Render goal as object only (for bullet point display).

    Args:
        subject: Subject entity
        predicate: Relationship predicate
        object: Object entity or value
        created_at: Memory creation timestamp (unused for goals)

    Returns:
        Formatted goal string as object only

    Examples:
        >>> render_goal("User", "has goal", "run marathon", None)
        "run marathon"
    """
    return object


def render_basic(
    subject: str,
    predicate: str,
    object: str,
    created_at: datetime | None = None,
) -> str:
    """Render standard memory as subject predicate object.

    Args:
        subject: Subject entity
        predicate: Relationship predicate
        object: Object entity or value
        created_at: Memory creation timestamp (unused for basic memories)

    Returns:
        Formatted memory string as full triple

    Examples:
        >>> render_basic("User", "lives in city", "Portland", None)
        "User lives in city Portland"
    """
    return f"{subject} {predicate} {object}"


# Registry mapping context types to renderers
MEMORY_RENDERERS: dict[str, Callable] = {
    "activity_context": render_activity,
    "goal_context": render_goal,
    "semantic_context": render_basic,
}


def get_renderer(context_type: str) -> Callable:
    """Get renderer for a context type.

    Args:
        context_type: Type of context (activity_context, goal_context, etc.)

    Returns:
        Renderer function for the context type, defaults to render_basic

    Examples:
        >>> renderer = get_renderer("activity_context")
        >>> renderer("User", "did activity", "coding", datetime(2026, 1, 12))
        "[2026-01-12T00:00:00Z] coding"
    """
    return MEMORY_RENDERERS.get(context_type, render_basic)
