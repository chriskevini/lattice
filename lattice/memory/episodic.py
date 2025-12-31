"""Episodic memory module - handles raw_messages table.

Stores immutable conversation history with temporal chaining.
"""

from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import structlog

from lattice.utils.database import db_pool


logger = structlog.get_logger(__name__)


class EpisodicMessage:
    """Represents a message in episodic memory."""

    def __init__(
        self,
        content: str,
        discord_message_id: int,
        channel_id: int,
        is_bot: bool,
        message_id: UUID | None = None,
        prev_turn_id: UUID | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize an episodic message.

        Args:
            content: Message text content
            discord_message_id: Discord's unique message ID
            channel_id: Discord channel ID
            is_bot: Whether the message was sent by the bot
            message_id: Internal UUID (auto-generated if None)
            prev_turn_id: UUID of the previous turn in conversation chain
            timestamp: Message timestamp (defaults to now)
        """
        self.content = content
        self.discord_message_id = discord_message_id
        self.channel_id = channel_id
        self.is_bot = is_bot
        self.message_id = message_id
        self.prev_turn_id = prev_turn_id
        self.timestamp = timestamp or datetime.now(UTC)


async def store_message(message: EpisodicMessage) -> UUID:
    """Store a message in episodic memory.

    Args:
        message: The message to store

    Returns:
        UUID of the stored message

    Raises:
        Exception: If database operation fails
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO raw_messages (
                discord_message_id, channel_id, content, is_bot, prev_turn_id
            )
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            message.discord_message_id,
            message.channel_id,
            message.content,
            message.is_bot,
            message.prev_turn_id,
        )

        message_id = cast("UUID", row["id"])

        logger.info(
            "Stored episodic message",
            message_id=str(message_id),
            discord_id=message.discord_message_id,
            is_bot=message.is_bot,
        )

        return message_id


async def get_recent_messages(
    channel_id: int,
    limit: int = 10,
) -> list[EpisodicMessage]:
    """Get recent messages from a channel.

    Args:
        channel_id: Discord channel ID
        limit: Maximum number of messages to return

    Returns:
        List of recent messages, ordered by timestamp (oldest first)
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, discord_message_id, channel_id, content, is_bot,
                   prev_turn_id, timestamp
            FROM raw_messages
            WHERE channel_id = $1
            ORDER BY timestamp DESC
            LIMIT $2
            """,
            channel_id,
            limit,
        )

        return [
            EpisodicMessage(
                message_id=row["id"],
                discord_message_id=row["discord_message_id"],
                channel_id=row["channel_id"],
                content=row["content"],
                is_bot=row["is_bot"],
                prev_turn_id=row["prev_turn_id"],
                timestamp=row["timestamp"],
            )
            for row in reversed(rows)
        ]


async def get_last_message_id(channel_id: int) -> UUID | None:
    """Get the ID of the last message in a channel.

    Args:
        channel_id: Discord channel ID

    Returns:
        UUID of last message, or None if no messages exist
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id
            FROM raw_messages
            WHERE channel_id = $1
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            channel_id,
        )

        return cast("UUID", row["id"]) if row else None
