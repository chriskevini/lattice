"""Context strategy and in-memory caching.

This module provides the data structures and caching for context planning.
Extracted entities, context flags, and unresolved entities are stored in RAM
and optionally pre-warmed on bot restart.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ContextStrategy:
    """Represents context strategy from conversation window analysis.

    Analyzes a conversation window (including current message) to extract:
    - entities: Canonical or known entity mentions for graph traversal
    - context_flags: Flags indicating what additional context is needed
    - unresolved_entities: Entities requiring clarification before canonicalization
    """

    entities: list[str] = field(default_factory=list)
    context_flags: list[str] = field(default_factory=list)
    unresolved_entities: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ChannelContext:
    """Cached context for a specific channel with per-item TTL tracking."""

    # Key: item (entity/flag), Value: global message index when last seen
    entities: dict[str, int] = field(default_factory=dict)
    context_flags: dict[str, int] = field(default_factory=dict)
    unresolved_entities: dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class ContextCache:
    """In-memory cache for extracted context with per-item message-based TTL.

    Stores entities, context_flags, and unresolved_entities. Each item's TTL
    is refreshed when it appears in a fresh strategy extraction.
    """

    def __init__(self, ttl: int = 10) -> None:
        """Initialize cache.

        Args:
            ttl: How many messages an item stays in cache since last seen
        """
        self.ttl = ttl
        self._cache: dict[int, ChannelContext] = {}
        self._message_counter: int = 0

    def advance(self) -> int:
        """Increment global message counter and return new value."""
        self._message_counter += 1
        return self._message_counter

    def update(
        self,
        channel_id: int,
        fresh: ContextStrategy,
    ) -> ContextStrategy:
        """Update cache with fresh strategy, refreshing TTL for seen items."""
        ctx = self._cache.setdefault(channel_id, ChannelContext())

        # Update per-item counters to current global index
        for entity in fresh.entities:
            ctx.entities[entity] = self._message_counter
        for flag in fresh.context_flags:
            ctx.context_flags[flag] = self._message_counter
        for unresolved in fresh.unresolved_entities:
            ctx.unresolved_entities[unresolved] = self._message_counter

        ctx.created_at = fresh.created_at
        return self.get_active(channel_id)

    def get_active(self, channel_id: int) -> ContextStrategy:
        """Get active context for a channel, filtering out expired items."""
        ctx = self._cache.get(channel_id)
        if not ctx:
            return ContextStrategy()

        # Filter items by message-based TTL
        active_entities = [
            e
            for e, idx in ctx.entities.items()
            if self._message_counter - idx <= self.ttl
        ]
        active_flags = [
            f
            for f, idx in ctx.context_flags.items()
            if self._message_counter - idx <= self.ttl
        ]
        active_unresolved = [
            u
            for u, idx in ctx.unresolved_entities.items()
            if self._message_counter - idx <= self.ttl
        ]

        # Cleanup internal dicts to prevent memory leaks
        ctx.entities = {
            e: idx for e, idx in ctx.entities.items() if e in active_entities
        }
        ctx.context_flags = {
            f: idx for f, idx in ctx.context_flags.items() if f in active_flags
        }
        ctx.unresolved_entities = {
            u: idx
            for u, idx in ctx.unresolved_entities.items()
            if u in active_unresolved
        }

        if not any([active_entities, active_flags, active_unresolved]):
            del self._cache[channel_id]
            return ContextStrategy()

        return ContextStrategy(
            entities=active_entities,
            context_flags=active_flags,
            unresolved_entities=active_unresolved,
            created_at=ctx.created_at,
        )

    def clear(self) -> None:
        """Clear all cached context."""
        self._cache.clear()
        self._message_counter = 0

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics for debugging/monitoring."""
        return {
            "cached_channels": len(self._cache),
            "total_entities": sum(len(ctx.entities) for ctx in self._cache.values()),
            "total_flags": sum(len(ctx.context_flags) for ctx in self._cache.values()),
            "message_counter": self._message_counter,
        }
