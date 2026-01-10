"""Central placeholder registry for prompt templates.

This module provides a centralized registry for all placeholder definitions
used in prompt templates. Each placeholder has metadata about its purpose,
resolution function, and category.

The registry enables:
- Single source of truth for all available placeholders
- Automatic validation of templates against known placeholders
- Discoverability for the dreaming cycle optimizer
- Extensibility via runtime registration
"""

from dataclasses import dataclass
from typing import Any, Callable

from lattice.utils.date_resolution import (
    format_current_date,
    format_current_time,
    resolve_relative_dates,
)


@dataclass
class PlaceholderDef:
    """Definition for a placeholder in prompt templates.

    Attributes:
        name: Unique identifier for the placeholder
        description: Human-readable purpose
        resolver: Async or sync function that returns the placeholder value
        category: Organization category (time, context, user, system)
        required: Whether this placeholder must be provided
        lazy: If True, only resolve if placeholder is actually in template
    """

    name: str
    description: str
    resolver: Callable[..., Any]
    category: str = "default"
    required: bool = False
    lazy: bool = True


class PlaceholderRegistry:
    """Central registry for all prompt template placeholders."""

    def __init__(self) -> None:
        self._placeholders: dict[str, PlaceholderDef] = {}
        self._register_core_placeholders()

    def register(self, defn: PlaceholderDef) -> None:
        """Register a placeholder definition.

        Args:
            defn: PlaceholderDef to register

        Raises:
            ValueError: If placeholder with same name already exists
        """
        if defn.name in self._placeholders:
            msg = f"Placeholder '{defn.name}' already registered"
            raise ValueError(msg)
        self._placeholders[defn.name] = defn

    def get(self, name: str) -> PlaceholderDef | None:
        """Get a placeholder definition by name.

        Args:
            name: Placeholder name to look up

        Returns:
            PlaceholderDef if found, None otherwise
        """
        return self._placeholders.get(name)

    def get_all(self) -> dict[str, PlaceholderDef]:
        """Get all registered placeholder definitions.

        Returns:
            Dictionary mapping placeholder names to definitions
        """
        return self._placeholders.copy()

    def get_by_category(self, category: str) -> dict[str, PlaceholderDef]:
        """Get all placeholders in a category.

        Args:
            category: Category to filter by

        Returns:
            Dictionary mapping placeholder names to definitions in category
        """
        return {k: v for k, v in self._placeholders.items() if v.category == category}

    def get_names(self) -> set[str]:
        """Get set of all registered placeholder names.

        Returns:
            Set of placeholder names
        """
        return set(self._placeholders.keys())

    def validate_template(self, template: str) -> tuple[bool, list[str]]:
        """Validate that all placeholders in template are registered.

        Args:
            template: Template string to validate

        Returns:
            Tuple of (is_valid, unknown_placeholders)
        """
        import re

        template_placeholders = set(re.findall(r"\{(\w+)\}", template))
        known_placeholders = self.get_names()
        unknown = list(template_placeholders - known_placeholders)
        return (len(unknown) == 0, unknown)

    def _register_core_placeholders(self) -> None:
        """Register core placeholders needed for prompt templates.

        This method is called during initialization to register
        all fundamental placeholders used across the system.
        """

        # Time/date placeholders
        self.register(
            PlaceholderDef(
                name="local_date",
                description="Current date with day of week (e.g., 2026/01/08, Thursday)",
                resolver=lambda ctx: format_current_date(
                    ctx.get("user_timezone", "UTC")
                ),
                category="time",
            )
        )

        self.register(
            PlaceholderDef(
                name="local_time",
                description="Current time for proactive decisions (e.g., 14:30)",
                resolver=lambda ctx: format_current_time(
                    ctx.get("user_timezone", "UTC")
                ),
                category="time",
            )
        )

        self.register(
            PlaceholderDef(
                name="date_resolution_hints",
                description="Resolved relative dates (e.g., Friday → 2026-01-10)",
                resolver=lambda ctx: resolve_relative_dates(
                    ctx.get("message_content", ""), ctx.get("user_timezone", "UTC")
                ),
                category="time",
            )
        )

        # Context placeholders
        self.register(
            PlaceholderDef(
                name="episodic_context",
                description="Recent conversation history with timestamps",
                resolver=lambda ctx: ctx.get(
                    "episodic_context", "No recent conversation."
                ),
                category="context",
            )
        )

        self.register(
            PlaceholderDef(
                name="semantic_context",
                description="Relevant facts and graph relationships",
                resolver=lambda ctx: ctx.get(
                    "semantic_context", "No relevant context found."
                ),
                category="context",
            )
        )

        self.register(
            PlaceholderDef(
                name="bigger_episodic_context",
                description="Extended conversation history for consolidation",
                resolver=lambda ctx: ctx.get("bigger_episodic_context", ""),
                category="context",
            )
        )

        self.register(
            PlaceholderDef(
                name="smaller_episodic_context",
                description="Window of recent messages for analysis",
                resolver=lambda ctx: ctx.get("smaller_episodic_context", ""),
                category="context",
            )
        )

        # User input placeholders
        self.register(
            PlaceholderDef(
                name="user_message",
                description="The user's current message",
                resolver=lambda ctx: ctx.get("user_message", ""),
                category="user",
                required=True,
            )
        )

        self.register(
            PlaceholderDef(
                name="unresolved_entities",
                description="Entities requiring clarification (e.g., 'bf', 'lkea')",
                resolver=lambda ctx: ctx.get("unresolved_entities", "(none)"),
                category="user",
            )
        )

        # Memory/consolidation placeholders
        self.register(
            PlaceholderDef(
                name="canonical_entities",
                description="Comma-separated list of known canonical entities",
                resolver=lambda ctx: ctx.get("canonical_entities", "(empty)"),
                category="memory",
            )
        )

        self.register(
            PlaceholderDef(
                name="canonical_predicates",
                description="Comma-separated list of known canonical predicates",
                resolver=lambda ctx: ctx.get("canonical_predicates", "(empty)"),
                category="memory",
            )
        )

        # Dreaming/optimizer placeholders
        self.register(
            PlaceholderDef(
                name="feedback_samples",
                description="Feedback samples for prompt optimization",
                resolver=lambda ctx: ctx.get("feedback_samples", "(none)"),
                category="dreaming",
            )
        )

        self.register(
            PlaceholderDef(
                name="metrics",
                description="Performance metrics for prompt optimization",
                resolver=lambda ctx: ctx.get("metrics", ""),
                category="dreaming",
            )
        )

        self.register(
            PlaceholderDef(
                name="current_template",
                description="Current template being optimized",
                resolver=lambda ctx: ctx.get("current_template", ""),
                category="dreaming",
            )
        )


_global_registry: PlaceholderRegistry | None = None


def get_registry() -> PlaceholderRegistry:
    """Get the global placeholder registry instance.

    Returns:
        PlaceholderRegistry singleton instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PlaceholderRegistry()
    return _global_registry
