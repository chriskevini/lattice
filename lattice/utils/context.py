"""Context formatting utilities for LLM prompts and in-memory context caching."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from lattice.memory.episodic import EpisodicMessage


@dataclass
class ChannelContext:
    """Cached context for a specific channel."""

    entities: list[str] = field(default_factory=list)
    context_flags: list[str] = field(default_factory=list)
    unresolved_entities: list[str] = field(default_factory=list)
    last_message_idx: int = 0


class InMemoryContextCache:
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
        """Increment global message counter and return new value.

        Call this once per message processed.
        """
        self._message_counter += 1
        return self._message_counter

    def add(
        self,
        channel_id: int,
        entities: list[str],
        context_flags: list[str],
        unresolved_entities: list[str],
    ) -> None:
        """Add extracted context for a channel, deduplicating against existing and within batch."""
        ctx = self._cache.setdefault(channel_id, ChannelContext())
        seen_entities: set[str] = set(ctx.entities)
        for entity in entities:
            if entity not in seen_entities:
                seen_entities.add(entity)
                ctx.entities.append(entity)
        seen_flags: set[str] = set(ctx.context_flags)
        for flag in context_flags:
            if flag not in seen_flags:
                seen_flags.add(flag)
                ctx.context_flags.append(flag)
        seen_unresolved: set[str] = set(ctx.unresolved_entities)
        for entity in unresolved_entities:
            if entity not in seen_unresolved:
                seen_unresolved.add(entity)
                ctx.unresolved_entities.append(entity)
        ctx.last_message_idx = self._message_counter

    def get_active(self, channel_id: int) -> tuple[list[str], list[str], list[str]]:
        """Get active cached context for a channel, or empty if expired."""
        ctx = self._cache.get(channel_id)
        if not ctx:
            return [], [], []

        if self._message_counter - ctx.last_message_idx > self.ttl:
            del self._cache[channel_id]
            return [], [], []

        return ctx.entities, ctx.context_flags, ctx.unresolved_entities

    def get_entities(self, channel_id: int) -> list[str]:
        """Get only active entities for a channel (convenience method)."""
        entities, _, _ = self.get_active(channel_id)
        return entities

    def prune_expired(self) -> None:
        """Remove expired entries from cache. Call periodically or on access."""
        expired_channels = [
            ch_id
            for ch_id, ctx in self._cache.items()
            if self._message_counter - ctx.last_message_idx > self.ttl
        ]
        for ch_id in expired_channels:
            del self._cache[ch_id]

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
            user_tz = ZoneInfo(msg.user_timezone or "UTC")
            local_ts = msg.timestamp.astimezone(user_tz)
            ts_str = local_ts.strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts_str = msg.timestamp.strftime("%Y-%m-%d %H:%M UTC")

        role = "ASSISTANT" if msg.is_bot else "USER"
        formatted_lines.append(f"[{ts_str}] {role}: {msg.content}")

    return "\n".join(formatted_lines)
