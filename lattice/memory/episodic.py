"""Episodic memory module - handles raw_messages table.

Stores immutable conversation history with timestamp-based retrieval.
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
from lattice.utils.objective_parsing import MIN_SALIENCY_DELTA, parse_objectives
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
        is_proactive: bool = False,
        message_id: UUID | None = None,
        timestamp: datetime | None = None,
        generation_metadata: dict[str, Any] | None = None,
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
        """
        self.content = content
        self.discord_message_id = discord_message_id
        self.channel_id = channel_id
        self.is_bot = is_bot
        self.is_proactive = is_proactive
        self.message_id = message_id
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
                discord_message_id, channel_id, content, is_bot, is_proactive, generation_metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            message.discord_message_id,
            message.channel_id,
            message.content,
            message.is_bot,
            message.is_proactive,
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
                SELECT id, discord_message_id, channel_id, content, is_bot, is_proactive, timestamp
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
                SELECT id, discord_message_id, channel_id, content, is_bot, is_proactive, timestamp
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
        )
        for row in reversed(rows)
    ]


async def consolidate_message(
    message_id: UUID,
    content: str,
    context: list[str],
) -> None:
    """Extract semantic triples and objectives from a message asynchronously.

    Args:
        message_id: UUID of the stored message
        content: Message content to extract from
        context: Recent conversation context for extraction
    """
    logger.info("Starting consolidation", message_id=str(message_id))

    triple_prompt = await get_prompt("TRIPLE_EXTRACTION")
    if not triple_prompt:
        logger.warning("TRIPLE_EXTRACTION prompt not found")
        return

    filled_prompt = triple_prompt.template.format(CONTEXT=content)
    logger.info("Calling LLM for triple extraction", message_id=str(message_id))

    llm_client = get_llm_client()
    result = await llm_client.complete(
        filled_prompt,
        temperature=triple_prompt.temperature,
    )

    logger.info(
        "LLM response received", message_id=str(message_id), content_preview=result.content[:100]
    )

    triples = parse_triples(result.content)
    logger.info("Parsed triples", count=len(triples) if triples else 0)

    if triples:
        async with db_pool.pool.acquire() as conn, conn.transaction():
            for triple in triples:
                subject_id = await _ensure_fact(
                    triple["subject"], message_id, conn, embedding_model
                )
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

    objectives = await extract_objectives(message_id, content, context)
    if objectives:
        await store_objectives(message_id, objectives)


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


async def extract_objectives(
    message_id: UUID,
    content: str,
    context: list[str] | None = None,
) -> list[dict[str, str | float]]:
    """Extract objectives from a message asynchronously.

    Args:
        message_id: UUID of the stored message
        content: Message content to extract from
        context: Recent conversation context for additional context

    Returns:
        List of extracted objectives with description, saliency, and status
    """
    logger.info("Starting objective extraction", message_id=str(message_id))

    objective_prompt = await get_prompt("OBJECTIVE_EXTRACTION")
    if not objective_prompt:
        logger.warning("OBJECTIVE_EXTRACTION prompt not found")
        return []

    context_text = content
    if context:
        context_text = "\n".join(context[-5:]) + "\n\nCurrent message:\n" + content

    filled_prompt = objective_prompt.template.format(CONTEXT=context_text)
    logger.info("Calling LLM for objective extraction", message_id=str(message_id))

    llm_client = get_llm_client()
    result = await llm_client.complete(
        filled_prompt,
        temperature=objective_prompt.temperature,
    )

    logger.info(
        "LLM response received",
        message_id=str(message_id),
        content_preview=result.content[:100],
    )

    objectives = parse_objectives(result.content)
    logger.info("Parsed objectives", count=len(objectives) if objectives else 0)

    return objectives


async def store_objectives(
    message_id: UUID,
    objectives: list[dict[str, str | float]],
) -> None:
    """Store extracted objectives in the database.

    Handles upsert of existing objectives (status and saliency updates) and
    insertion of new objectives.

    Args:
        message_id: UUID of the origin message
        objectives: List of extracted objectives
    """
    if not objectives:
        return

    async with db_pool.pool.acquire() as conn, conn.transaction():
        for objective in objectives:
            description = cast("str", objective["description"])
            saliency_value = objective["saliency"]
            saliency = float(saliency_value) if isinstance(saliency_value, (int, float)) else 0.5
            status = cast("str", objective["status"])

            normalized = description.lower().strip()

            existing = await conn.fetchrow(
                """
                SELECT id, status, saliency_score FROM objectives WHERE LOWER(description) = $1
                """,
                normalized,
            )

            if existing:
                current_saliency = (
                    float(existing["saliency_score"]) if existing["saliency_score"] else 0.5
                )
                status_changed = existing["status"] != status
                saliency_changed = abs(current_saliency - saliency) > MIN_SALIENCY_DELTA

                if status_changed or saliency_changed:
                    await conn.execute(
                        """
                        UPDATE objectives
                        SET status = $1, saliency_score = $2, last_updated = now()
                        WHERE id = $3
                        """,
                        status,
                        saliency,
                        existing["id"],
                    )
                    logger.info(
                        "Updated objective",
                        description_preview=description[:50],
                        old_status=existing["status"],
                        new_status=status,
                        old_saliency=current_saliency,
                        new_saliency=saliency,
                    )
            else:
                await conn.execute(
                    """
                    INSERT INTO objectives (description, saliency_score, status, origin_id)
                    VALUES ($1, $2, $3, $4)
                    """,
                    description,
                    saliency,
                    status,
                    message_id,
                )
                logger.info(
                    "Created new objective",
                    description_preview=description[:50],
                    saliency=saliency,
                )
