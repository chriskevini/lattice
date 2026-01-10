"""Episodic memory module - handles raw_messages table.

Stores immutable conversation history with timestamp-based retrieval and timezone tracking.
"""

import asyncio
import json
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import asyncpg
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
        self.timestamp = timestamp or datetime.now(UTC)
        self.generation_metadata = generation_metadata
        self.user_timezone = user_timezone or "UTC"

    @property
    def role(self) -> str:
        """Return the role name for this message."""
        return "ASSISTANT" if self.is_bot else "USER"


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
                discord_message_id, channel_id, content, is_bot, is_proactive, generation_metadata, user_timezone
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            message.discord_message_id,
            message.channel_id,
            message.content,
            message.is_bot,
            message.is_proactive,
            json.dumps(message.generation_metadata)
            if message.generation_metadata
            else None,
            message.user_timezone,
        )

        message_id = cast("UUID", row["id"])

        logger.info(
            "Stored episodic message",
            message_id=str(message_id),
            discord_id=message.discord_message_id,
            is_bot=message.is_bot,
        )

    from lattice.memory import batch_consolidation

    asyncio.create_task(batch_consolidation.check_and_run_batch())

    return message_id


async def get_recent_messages(
    channel_id: int | None = None,
    limit: int = 10,
) -> list[EpisodicMessage]:
    """Get recent messages from a channel or all channels.

    Args:
        channel_id: Discord channel ID (None for all channels)
        limit: Maximum number of messages to return

    Returns:
        List of recent messages, ordered by timestamp (oldest first)
    """
    if channel_id:
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, discord_message_id, channel_id, content, is_bot, is_proactive, timestamp, user_timezone
                FROM raw_messages
                WHERE channel_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                channel_id,
                limit,
            )
    else:
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, discord_message_id, channel_id, content, is_bot, is_proactive, timestamp, user_timezone
                FROM raw_messages
                ORDER BY timestamp DESC
                LIMIT $1
                """,
                limit,
            )

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


async def store_semantic_memories(
    message_id: UUID,
    memories: list[dict[str, str]],
    source_batch_id: str | None = None,
) -> None:
    """Store extracted memories in semantic_memories table.

    Args:
        message_id: UUID of origin message
        memories: List of {"subject": str, "predicate": str, "object": str}
        source_batch_id: Optional batch identifier for traceability

    Raises:
        Exception: If database operation fails
    """
    if not memories:
        return

    async with db_pool.pool.acquire() as conn, conn.transaction():
        for memory in memories:
            subject = memory.get("subject", "").strip()
            predicate = memory.get("predicate", "").strip()
            obj = memory.get("object", "").strip()

            if not (subject and predicate and obj):
                logger.warning(
                    "Skipping invalid memory",
                    memory=memory,
                )
                continue

            try:
                await conn.execute(
                    """
                    INSERT INTO semantic_memories (
                        subject, predicate, object, source_batch_id
                    )
                    VALUES ($1, $2, $3, $4)
                    """,
                    subject,
                    predicate,
                    obj,
                    source_batch_id,
                )

                logger.debug(
                    "Stored semantic memory",
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    message_id=str(message_id),
                )

            except asyncpg.PostgresError:
                logger.exception(
                    "Failed to store memory",
                    memory=memory,
                    message_id=str(message_id),
                )
