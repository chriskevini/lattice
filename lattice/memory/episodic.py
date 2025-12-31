"""Episodic memory module - handles raw_messages table.

Stores immutable conversation history with temporal chaining.
"""

import json
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import asyncpg
import structlog

from lattice.memory.procedural import get_prompt
from lattice.utils.database import db_pool
from lattice.utils.embeddings import EmbeddingModel, embedding_model
from lattice.utils.llm import get_llm_client
from lattice.utils.triple_parsing import parse_triples


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
        generation_metadata: dict[str, Any] | None = None,
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
            generation_metadata: LLM generation metadata (model, tokens, etc.)
        """
        self.content = content
        self.discord_message_id = discord_message_id
        self.channel_id = channel_id
        self.is_bot = is_bot
        self.message_id = message_id
        self.prev_turn_id = prev_turn_id
        self.timestamp = timestamp or datetime.now(UTC)
        self.generation_metadata = generation_metadata


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
                discord_message_id, channel_id, content, is_bot, prev_turn_id, generation_metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            message.discord_message_id,
            message.channel_id,
            message.content,
            message.is_bot,
            message.prev_turn_id,
            json.dumps(message.generation_metadata) if message.generation_metadata else None,
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


async def consolidate_message(
    message_id: UUID,
    _content: str,
    context: list[str],
) -> None:
    """Extract semantic triples from a message asynchronously.

    Args:
        message_id: UUID of the stored message
        content: Message content
        context: Recent conversation context
    """
    triple_prompt = await get_prompt("TRIPLE_EXTRACTION")
    if not triple_prompt:
        logger.warning("TRIPLE_EXTRACTION prompt not found")
        return

    filled_prompt = triple_prompt.template.format(CONTEXT="\n".join(context))

    llm_client = get_llm_client()
    raw_triples = await llm_client.complete(
        filled_prompt,
        temperature=triple_prompt.temperature,
    )

    triples = parse_triples(raw_triples)
    if not triples:
        return

    async with db_pool.pool.acquire() as conn, conn.transaction():
        for triple in triples:
            subject_id = await _ensure_fact(triple["subject"], message_id, conn, embedding_model)
            object_id = await _ensure_fact(triple["object"], message_id, conn, embedding_model)

            await conn.execute(
                """
                    INSERT INTO semantic_triples (subject_id, predicate, object_id, origin_id)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT DO NOTHING
                    """,
                subject_id,
                triple["predicate"],
                object_id,
                message_id,
            )


async def _ensure_fact(
    content: str,
    origin_id: UUID,
    conn: asyncpg.Connection,
    embedding_model: EmbeddingModel,
) -> UUID:
    """Ensure fact exists, return its ID.

    Args:
        content: Fact content
        origin_id: Origin message ID
        conn: Database connection
        embedding_model: Embedding model for vector generation

    Returns:
        UUID of existing or new fact
    """
    normalized = content.lower().strip()

    existing = await conn.fetchval(
        """
        SELECT id FROM stable_facts WHERE content = $1
        """,
        normalized,
    )

    if existing:
        return cast("UUID", existing)

    embedding = embedding_model.encode_single(normalized)

    row = await conn.fetchrow(
        """
        INSERT INTO stable_facts (content, embedding, origin_id, entity_type)
        VALUES ($1, $2::vector, $3, 'inferred')
        RETURNING id
        """,
        normalized,
        embedding,
        origin_id,
    )

    if not row:
        msg = "Failed to insert fact"
        raise RuntimeError(msg)

    logger.info("Created new fact from triple", content_preview=normalized[:50])
    return cast("UUID", row["id"])
