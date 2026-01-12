"""Context strategy and in-memory caching.

This module provides the data structures and caching for context planning.
Extracted entities, context flags, and unresolved entities are stored in RAM
and optionally pre-warmed on bot restart.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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

    # Key: item (entity/flag), Value: channel-local message index when last seen
    entities: dict[str, int] = field(default_factory=dict)
    context_flags: dict[str, int] = field(default_factory=dict)
    unresolved_entities: dict[str, int] = field(default_factory=dict)
    message_counter: int = 0
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

    def advance(self, channel_id: int) -> int:
        """Increment channel message counter and return new value."""
        ctx = self._cache.setdefault(channel_id, ChannelContext())
        ctx.message_counter += 1
        return ctx.message_counter

    def update(
        self,
        channel_id: int,
        fresh: ContextStrategy,
    ) -> ContextStrategy:
        """Update cache with fresh strategy, refreshing TTL for seen items."""
        ctx = self._cache.setdefault(channel_id, ChannelContext())

        # Update per-item counters to current channel index
        for entity in fresh.entities:
            ctx.entities[entity] = ctx.message_counter
        for flag in fresh.context_flags:
            ctx.context_flags[flag] = ctx.message_counter
        for unresolved in fresh.unresolved_entities:
            ctx.unresolved_entities[unresolved] = ctx.message_counter

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
            if ctx.message_counter - idx <= self.ttl
        ]
        active_flags = [
            f
            for f, idx in ctx.context_flags.items()
            if ctx.message_counter - idx <= self.ttl
        ]
        active_unresolved = [
            u
            for u, idx in ctx.unresolved_entities.items()
            if ctx.message_counter - idx <= self.ttl
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
            # Keep the message counter even if items are cleared
            # or delete the whole context if it's been long enough?
            # For now, let's keep it if message_counter is small or just delete
            # if everything is expired to avoid channel leak.
            # However, deleting it resets message_counter.
            # Let's only delete if it's truly empty AND we haven't seen a message in a while?
            # Actually, deleting it is fine, next message starts at counter 1.
            del self._cache[channel_id]
            return ContextStrategy()

        return ContextStrategy(
            entities=active_entities,
            context_flags=active_flags,
            unresolved_entities=active_unresolved,
            created_at=ctx.created_at,
        )

    async def save_to_db(self, db_pool: Any) -> None:
        """Persist cache to database."""
        async with db_pool.pool.acquire() as conn:
            for channel_id, ctx in self._cache.items():
                strategy_json = {
                    "entities": ctx.entities,
                    "context_flags": ctx.context_flags,
                    "unresolved_entities": ctx.unresolved_entities,
                }

                await conn.execute(
                    """
                    INSERT INTO context_cache_persistence (channel_id, strategy, message_counter, updated_at)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (channel_id) DO UPDATE SET
                        strategy = EXCLUDED.strategy,
                        message_counter = EXCLUDED.message_counter,
                        updated_at = EXCLUDED.updated_at
                    """,
                    channel_id,
                    json.dumps(strategy_json),
                    ctx.message_counter,
                    ctx.created_at,
                )

    async def load_from_db(self, db_pool: Any) -> None:
        """Load cache from database."""
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT channel_id, strategy, message_counter, updated_at FROM context_cache_persistence"
            )

            for row in rows:
                strategy_data = json.loads(row["strategy"])
                self._cache[row["channel_id"]] = ChannelContext(
                    entities=strategy_data["entities"],
                    context_flags=strategy_data["context_flags"],
                    unresolved_entities=strategy_data["unresolved_entities"],
                    message_counter=row["message_counter"],
                    created_at=row["updated_at"],
                )

    def clear(self) -> None:
        """Clear all cached context."""
        self._cache.clear()

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics for debugging/monitoring."""
        return {
            "cached_channels": len(self._cache),
            "total_entities": sum(len(ctx.entities) for ctx in self._cache.values()),
            "total_flags": sum(len(ctx.context_flags) for ctx in self._cache.values()),
            "total_messages_tracked": sum(
                ctx.message_counter for ctx in self._cache.values()
            ),
        }


@dataclass
class UserContextCache:
    """User-level cache for goals and activities with time-based TTL."""

    def __init__(self, ttl_minutes: int = 30) -> None:
        self.ttl = ttl_minutes
        self._goals: dict[str, tuple[str, datetime]] = {}
        self._activities: dict[str, tuple[str, datetime]] = {}

    def get_goals(self, user_id: str) -> str | None:
        """Get cached goals for user, or None if expired/missing."""
        if user_id not in self._goals:
            return None
        content, cached_at = self._goals[user_id]
        if self._is_expired(cached_at):
            del self._goals[user_id]
            return None
        return content

    def set_goals(self, user_id: str, content: str) -> None:
        """Cache goals for user."""
        self._goals[user_id] = (content, datetime.now())

    def get_activities(self, user_id: str) -> str | None:
        """Get cached activities for user, or None if expired/missing."""
        if user_id not in self._activities:
            return None
        content, cached_at = self._activities[user_id]
        if self._is_expired(cached_at):
            del self._activities[user_id]
            return None
        return content

    def set_activities(self, user_id: str, content: str) -> None:
        """Cache activities for user."""
        self._activities[user_id] = (content, datetime.now())

    def _is_expired(self, cached_at: datetime) -> bool:
        return (datetime.now() - cached_at).total_seconds() > self.ttl * 60

    def clear(self) -> None:
        """Clear all cached user context."""
        self._goals.clear()
        self._activities.clear()

    def get_stats(self) -> dict[str, int]:
        return {
            "cached_users": len(set(self._goals.keys()) | set(self._activities.keys())),
            "cached_goals": len(self._goals),
            "cached_activities": len(self._activities),
        }
