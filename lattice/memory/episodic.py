"""Episodic memory module - handles raw_messages table.

Stores immutable conversation history with timestamp-based retrieval and timezone tracking.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from lattice.memory.repositories import MessageRepository
from lattice.utils.date_resolution import get_now

if TYPE_CHECKING:
    from lattice.memory.repositories import MessageRepository
else:
    MessageRepository = Any


logger = structlog.get_logger(__name__)


class EpisodicMessage:
    """Represents a message in episodic memory."""

    def __init__(
        self,
        content: str,
        discord_message_id: int,
        channel_id: int,
        is_bot: bool,
        is_proactive: bool = False,
        message_id: UUID | None = None,
        timestamp: datetime | None = None,
        generation_metadata: dict[str, Any] | None = None,
        user_timezone: str | None = None,
    ) -> None:
        """Initialize an episodic message.

        Args:
            content: Message text content
            discord_message_id: Discord's unique message ID
            channel_id: Discord channel ID
            is_bot: Whether the message was sent by the bot
            is_proactive: Whether the bot initiated this message (vs replying)
            message_id: Internal UUID (auto-generated if None)
            timestamp: Message timestamp (defaults to now)
            generation_metadata: LLM generation metadata (model, tokens, etc.)
            user_timezone: IANA timezone for this message (e.g., 'America/New_York')
        """
        self.content = content
        self.discord_message_id = discord_message_id
        self.channel_id = channel_id
        self.is_bot = is_bot
        self.is_proactive = is_proactive
        self.message_id = message_id
        self.timestamp = timestamp or get_now("UTC")
        self.generation_metadata = generation_metadata
        self.user_timezone = user_timezone or "UTC"

    @property
    def role(self) -> str:
        """Return the role name for this message."""
        return "ASSISTANT" if self.is_bot else "USER"


async def store_message(repo: MessageRepository, message: "EpisodicMessage") -> UUID:
    """Store a message in episodic memory.

    Args:
        repo: Message repository for episodic memory
        message: The message to store

    Returns:
        UUID of the stored message
    """
    message_id = await repo.store_message(
        content=message.content,
        discord_message_id=message.discord_message_id,
        channel_id=message.channel_id,
        is_bot=message.is_bot,
        is_proactive=message.is_proactive,
        generation_metadata=message.generation_metadata,
        user_timezone=message.user_timezone,
    )

    logger.info(
        "Stored episodic message",
        message_id=str(message_id),
        discord_id=message.discord_message_id,
        is_bot=message.is_bot,
    )

    return message_id


async def get_recent_messages(
    repo: MessageRepository, channel_id: int | None = None, limit: int = 10
) -> list[EpisodicMessage]:
    """Get recent messages from a channel or all channels.

    Args:
        repo: Message repository
        channel_id: Discord channel ID (None for all channels)
        limit: Maximum number of messages to return

    Returns:
        List of recent messages, ordered by timestamp (oldest first)
    """
    rows = await repo.get_recent_messages(channel_id=channel_id, limit=limit)

    return [
        EpisodicMessage(
            message_id=row["id"],
            discord_message_id=row["discord_message_id"],
            channel_id=row["channel_id"],
            content=row["content"],
            is_bot=row["is_bot"],
            is_proactive=row["is_proactive"],
            timestamp=row["timestamp"],
            user_timezone=row["user_timezone"] or "UTC",
        )
        for row in reversed(rows)
    ]


def format_messages(messages: list[EpisodicMessage]) -> str:
    """Format a list of episodic messages for LLM context.

    Args:
        messages: List of messages to format

    Returns:
        Formatted string of messages
    """
    formatted_lines = []
    for msg in messages:
        role = "Assistant" if msg.is_bot else "User"
        formatted_lines.append(f"{role}: {msg.content}")

    return "\n".join(formatted_lines)


async def store_semantic_memories(
    repo: MessageRepository,
    message_id: UUID,
    memories: list[dict[str, str]],
    source_batch_id: str | None = None,
) -> None:
    """Store extracted memories in semantic_memories table.

    Args:
        repo: Message repository
        message_id: UUID of origin message
        memories: List of {"subject": str, "predicate": str, "object": str}
        source_batch_id: Optional batch identifier for traceability

    Raises:
        Exception: If database operation fails
    """
    if not memories:
        return

    await repo.store_semantic_memories(
        message_id=message_id, memories=memories, source_batch_id=source_batch_id
    )
