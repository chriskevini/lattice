"""Semantic memory module - handles stable_facts and vector search.

Stores extracted facts with embeddings for semantic similarity search.
"""

from typing import cast
from uuid import UUID

import structlog

from lattice.utils.database import db_pool
from lattice.utils.embeddings import embedding_model


logger = structlog.get_logger(__name__)

MAX_SEARCH_LIMIT = 100
MIN_SIMILARITY_THRESHOLD = 0.0
MAX_SIMILARITY_THRESHOLD = 1.0


class StableFact:
    """Represents a fact in semantic memory."""

    def __init__(
        self,
        content: str,
        fact_id: UUID | None = None,
        embedding: list[float] | None = None,
        origin_id: UUID | None = None,
        entity_type: str | None = None,
    ) -> None:
        """Initialize a stable fact.

        Args:
            content: Textual representation of the fact
            fact_id: UUID of the fact (auto-generated if None)
            embedding: Vector embedding of the content
            origin_id: UUID of the raw_message this fact was extracted from
            entity_type: Type of entity (e.g., 'person', 'preference', 'event')
        """
        self.content = content
        self.fact_id = fact_id
        self.embedding = embedding
        self.origin_id = origin_id
        self.entity_type = entity_type


async def store_fact(fact: StableFact) -> UUID:
    """Store a fact in semantic memory.

    Args:
        fact: The fact to store

    Returns:
        UUID of the stored fact
    """
    if fact.embedding is None:
        fact.embedding = embedding_model.encode_single(fact.content)

    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO stable_facts (content, embedding, origin_id, entity_type)
            VALUES ($1, $2::vector, $3, $4)
            RETURNING id
            """,
            fact.content,
            fact.embedding,
            fact.origin_id,
            fact.entity_type,
        )

        fact_id = cast("UUID", row["id"])

        logger.info(
            "Stored semantic fact",
            fact_id=str(fact_id),
            entity_type=fact.entity_type,
            content_preview=fact.content[:50],
        )

        return fact_id


async def search_similar_facts(
    query: str,
    limit: int = 5,
    similarity_threshold: float = 0.7,
) -> list[StableFact]:
    """Search for facts similar to a query using vector similarity.

    Args:
        query: Query text to search for
        limit: Maximum number of results to return (1-100)
        similarity_threshold: Minimum cosine similarity (0.0-1.0)

    Returns:
        List of similar facts, ordered by similarity (most similar first)

    Raises:
        ValueError: If limit or similarity_threshold are out of valid ranges
    """
    if limit < 1 or limit > MAX_SEARCH_LIMIT:
        msg = f"limit must be between 1 and {MAX_SEARCH_LIMIT}, got {limit}"
        raise ValueError(msg)
    if not (
        MIN_SIMILARITY_THRESHOLD <= similarity_threshold <= MAX_SIMILARITY_THRESHOLD
    ):
        msg = f"similarity_threshold must be between 0.0 and 1.0, got {similarity_threshold}"
        raise ValueError(msg)

    if not query.strip():
        logger.warning("Empty query provided to semantic search")
        return []

    query_embedding = embedding_model.encode_single(query)

    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, embedding, origin_id, entity_type,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM stable_facts
            WHERE 1 - (embedding <=> $1::vector) >= $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            query_embedding,
            similarity_threshold,
            limit,
        )

        facts = [
            StableFact(
                fact_id=row["id"],
                content=row["content"],
                embedding=list(row["embedding"]),
                origin_id=row["origin_id"],
                entity_type=row["entity_type"],
            )
            for row in rows
        ]

        logger.info(
            "Semantic search completed",
            query_preview=query[:50],
            results_found=len(facts),
        )

        return facts
