import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from lattice.memory.repositories import ContextRepository


logger = logging.getLogger(__name__)


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


class ContextCacheBase:
    """Base class for persistent context caches."""

    def __init__(self, context_type: str, repository: ContextRepository) -> None:
        self.context_type = context_type
        self._repository = repository

    async def _save(self, target_id: str, data: dict[str, Any]) -> None:
        """Upsert context data via repository."""
        await self._repository.save_context(self.context_type, target_id, data)

    async def _load_type(self) -> list[dict[str, Any]]:
        """Load all entries of this context type via repository."""
        return await self._repository.load_context_type(self.context_type)


class ChannelContextCache(ContextCacheBase):
    """In-memory cache for extracted channel context with persistence.

    Stores entities, context_flags, and unresolved_entities. Each item's TTL
    is refreshed when it appears in a fresh strategy extraction.
    """

    def __init__(self, repository: ContextRepository, ttl: int = 10) -> None:
        """Initialize cache.

        Args:
            repository: Context repository for persistence
            ttl: How many messages an item stays in cache since last seen
        """
        super().__init__(context_type="channel", repository=repository)
        self.ttl = ttl
        self._cache: dict[int, ChannelContext] = {}

    async def advance(self, channel_id: int) -> int:
        """Increment channel message counter and return new value."""
        ctx = self._cache.setdefault(channel_id, ChannelContext())
        ctx.message_counter += 1
        await self._persist(channel_id)
        return ctx.message_counter

    async def update(
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
        await self._persist(channel_id)
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
            # If everything is expired, we don't delete immediately to keep message_counter,
            # but we return empty strategy.
            return ContextStrategy(created_at=ctx.created_at)

        return ContextStrategy(
            entities=active_entities,
            context_flags=active_flags,
            unresolved_entities=active_unresolved,
            created_at=ctx.created_at,
        )

    async def _persist(self, channel_id: int) -> None:
        """Persist a single channel's context to DB."""
        ctx = self._cache.get(channel_id)
        if not ctx:
            return

        data = {
            "entities": ctx.entities,
            "context_flags": ctx.context_flags,
            "unresolved_entities": ctx.unresolved_entities,
            "message_counter": ctx.message_counter,
            "created_at": ctx.created_at.isoformat(),
        }
        ctx_after = self._cache.get(channel_id)
        if ctx_after is not ctx:
            return
        await self._save(str(channel_id), data)

    async def load_from_db(self) -> None:
        """Load all channel context from database."""
        rows = await self._load_type()
        for row in rows:
            try:
                channel_id = int(row["target_id"])
                data = json.loads(row["data"])
                self._cache[channel_id] = ChannelContext(
                    entities=data["entities"],
                    context_flags=data["context_flags"],
                    unresolved_entities=data["unresolved_entities"],
                    message_counter=data["message_counter"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                )
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                target_id = row.get("target_id", "unknown")
                logger.warning(
                    "Failed to load channel context from DB: target_id=%s, error=%s",
                    target_id,
                    e,
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


class UserContextCache(ContextCacheBase):
    """User-level cache for goals and activities with persistence.

    Stores goals, activities, and timezone with automatic write-through
    persistence to the database.
    """

    def __init__(self, repository: ContextRepository, ttl_minutes: int = 30) -> None:
        """Initialize cache.

        Args:
            repository: Context repository for persistence
            ttl_minutes: TTL for cached items in minutes
        """
        super().__init__(context_type="user", repository=repository)
        self.ttl = ttl_minutes
        self._goals: dict[str, tuple[str, datetime]] = {}
        self._activities: dict[str, tuple[str, datetime]] = {}
        self._timezone: dict[str, tuple[str, datetime]] = {}

    def get_goals(self, user_id: str) -> str | None:
        """Get cached goals for user, or None if expired/missing."""
        if user_id not in self._goals:
            return None
        content, cached_at = self._goals[user_id]
        if self._is_expired(cached_at):
            del self._goals[user_id]
            return None
        return content

    async def set_goals(self, user_id: str, content: str) -> None:
        """Cache goals for user and persist to DB."""
        self._goals[user_id] = (content, datetime.now())
        await self._persist(user_id)

    def get_activities(self, user_id: str) -> str | None:
        """Get cached activities for user, or None if expired/missing."""
        if user_id not in self._activities:
            return None
        content, cached_at = self._activities[user_id]
        if self._is_expired(cached_at):
            del self._activities[user_id]
            return None
        return content

    async def set_activities(self, user_id: str, content: str) -> None:
        """Cache activities for user and persist to DB."""
        self._activities[user_id] = (content, datetime.now())
        await self._persist(user_id)

    def get_timezone(self, user_id: str) -> str | None:
        """Get cached timezone for user, or None if expired/missing."""
        if user_id not in self._timezone:
            return None
        tz_value, cached_at = self._timezone[user_id]
        if self._is_expired(cached_at):
            del self._timezone[user_id]
            return None
        return tz_value

    async def set_timezone(self, user_id: str, timezone: str) -> None:
        """Cache timezone for user and persist to DB."""
        self._timezone[user_id] = (timezone, datetime.now())
        await self._persist(user_id)

    def _is_expired(self, cached_at: datetime) -> bool:
        return (datetime.now() - cached_at).total_seconds() > self.ttl * 60

    async def _persist(self, user_id: str) -> None:
        """Persist user context to DB via repository."""
        goals_data = self._goals.get(user_id)
        activities_data = self._activities.get(user_id)
        tz_data = self._timezone.get(user_id)

        data = {
            "goals": [goals_data[0], goals_data[1].isoformat()] if goals_data else None,
            "activities": [activities_data[0], activities_data[1].isoformat()]
            if activities_data
            else None,
            "timezone": [tz_data[0], tz_data[1].isoformat()] if tz_data else None,
        }
        await self._save(user_id, data)

    async def load_from_db(self) -> None:
        """Load all user context from database via repository."""
        rows = await self._load_type()
        for row in rows:
            try:
                user_id = row["target_id"]
                data = row["data"]
                if isinstance(data, str):
                    data = json.loads(data)
                if data.get("goals"):
                    self._goals[user_id] = (
                        data["goals"][0],
                        datetime.fromisoformat(data["goals"][1]),
                    )
                if data.get("activities"):
                    self._activities[user_id] = (
                        data["activities"][0],
                        datetime.fromisoformat(data["activities"][1]),
                    )
                if data.get("timezone"):
                    self._timezone[user_id] = (
                        data["timezone"][0],
                        datetime.fromisoformat(data["timezone"][1]),
                    )
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                target_id = row.get("target_id", "unknown")
                logger.warning(
                    "Failed to load user context from DB: target_id=%s, error=%s",
                    target_id,
                    e,
                )

    def clear(self) -> None:
        """Clear all cached user context."""
        self._goals.clear()
        self._activities.clear()
        self._timezone.clear()

    def get_stats(self) -> dict[str, int]:
        return {
            "cached_users": len(set(self._goals.keys()) | set(self._activities.keys())),
            "cached_goals": len(self._goals),
            "cached_activities": len(self._activities),
            "cached_timezone": len(self._timezone),
        }
