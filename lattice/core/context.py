"""Context strategy and in-memory caching.

This module provides the data structures and caching for context planning.
Extracted entities, context flags, and unresolved entities are stored in RAM
and optionally pre-warmed on bot restart.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from lattice.memory.episodic import EpisodicMessage


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
    """Cached context for a specific channel."""

    strategy: ContextStrategy = field(default_factory=ContextStrategy)
    last_message_idx: int = 0


class ContextCache:
    """In-memory cache for extracted context, keyed by channel.

    Stores entities, context_flags, and unresolved_entities extracted from recent
    messages. Uses a global message counter for TTL expiration.

    Thread safety: Not thread-safe. For single-instance bot use only.
    """

    def __init__(self, ttl: int = 10) -> None:
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
        """Update cache with fresh strategy, merging with existing entities/flags."""
        ctx = self._cache.setdefault(channel_id, ChannelContext())

        # Merge deduplicated
        merged_entities = self._merge_deduplicate(ctx.strategy.entities, fresh.entities)
        merged_flags = self._merge_deduplicate(
            ctx.strategy.context_flags, fresh.context_flags
        )
        merged_unresolved = self._merge_deduplicate(
            ctx.strategy.unresolved_entities, fresh.unresolved_entities
        )

        ctx.strategy = ContextStrategy(
            entities=merged_entities,
            context_flags=merged_flags,
            unresolved_entities=merged_unresolved,
            created_at=fresh.created_at,
        )
        ctx.last_message_idx = self._message_counter
        return ctx.strategy

    def get_active(self, channel_id: int) -> ContextStrategy:
        """Get active cached context for a channel, or empty if expired."""
        ctx = self._cache.get(channel_id)
        if not ctx:
            return ContextStrategy()

        if self._message_counter - ctx.last_message_idx > self.ttl:
            del self._cache[channel_id]
            return ContextStrategy()

        return ctx.strategy

    def clear(self) -> None:
        """Clear all cached context."""
        self._cache.clear()
        self._message_counter = 0

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics for debugging/monitoring."""
        return {
            "cached_channels": len(self._cache),
            "total_entities": sum(
                len(ctx.strategy.entities) for ctx in self._cache.values()
            ),
            "total_flags": sum(
                len(ctx.strategy.context_flags) for ctx in self._cache.values()
            ),
            "message_counter": self._message_counter,
        }

    @staticmethod
    def _merge_deduplicate(cached: list[str], fresh: list[str]) -> list[str]:
        """Merge two lists and deduplicate while preserving order."""
        seen: set[str] = set()
        result: list[str] = []

        for item in cached:
            if item not in seen:
                seen.add(item)
                result.append(item)

        for item in fresh:
            if item not in seen:
                seen.add(item)
                result.append(item)

        return result
