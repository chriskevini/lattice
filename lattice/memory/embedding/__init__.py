"""Embedding-based memory repository.

Provides vector similarity search for semantic memory storage and retrieval.
Uses pgvector for efficient nearest-neighbor search.
"""

import json
import structlog
from typing import Any
from uuid import UUID

from lattice.memory.repositories import PostgresRepository
from lattice.utils.config import config

logger = structlog.get_logger(__name__)


def embedding_to_bytes(embedding: list[float]) -> bytes:
    """Convert embedding list to bytes for storage."""
    import struct

    expected_dimension = config.embedding_dimension
    actual_dimension = len(embedding)

    if actual_dimension != expected_dimension:
        logger.warning(
            f"Embedding dimension mismatch: expected {expected_dimension}, got {actual_dimension}. "
            "Data may cause issues with vector operations."
        )

    return struct.pack(f"{actual_dimension}f", *embedding)


def bytes_to_embedding(data: bytes, dimension: int) -> list[float]:
    """Convert stored bytes back to embedding list."""
    import struct

    expected_size = dimension * 4  # 4 bytes per float
    if len(data) != expected_size:
        logger.warning(
            f"Embedding size mismatch: expected {expected_size}, got {len(data)}"
        )
        # Pad or truncate as needed
        embedding = list(struct.unpack(f"{len(data) // 4}f", data))
        while len(embedding) < dimension:
            embedding.append(0.0)
        return embedding[:dimension]
    return list(struct.unpack(f"{dimension}f", data))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class PostgresEmbeddingMemoryRepository(PostgresRepository):
    """PostgreSQL implementation of embedding-based memory storage."""

    async def store_memory(
        self,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        source_batch_id: str | None = None,
    ) -> UUID:
        """Store a memory with its embedding.

        Args:
            content: The memory content (text)
            embedding: Vector representation of the content
            metadata: Optional metadata dict
            source_batch_id: Optional batch identifier

        Returns:
            UUID of the stored memory
        """
        embedding_bytes = embedding_to_bytes(embedding)
        metadata_json = json.dumps(metadata) if metadata else None

        async with self._db_pool.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memory_embeddings (content, embedding, metadata, source_batch_id)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                content,
                embedding_bytes,
                metadata_json,
                source_batch_id,
            )
            return row["id"]

    async def store_memories_batch(
        self,
        items: list[dict[str, Any]],
        source_batch_id: str | None = None,
    ) -> int:
        """Store multiple memories in a batch.

        Args:
            items: List of dicts with keys: content, embedding, metadata
            source_batch_id: Optional batch identifier for all items

        Returns:
            Number of memories stored
        """
        if not items:
            return 0

        async with self._db_pool.pool.acquire() as conn:
            async with conn.transaction():
                count = 0
                for item in items:
                    content = item.get("content", "").strip()
                    embedding = item.get("embedding", [])
                    metadata = item.get("metadata")

                    if not content or not embedding:
                        continue

                    embedding_bytes = embedding_to_bytes(embedding)
                    metadata_json = json.dumps(metadata) if metadata else None

                    await conn.execute(
                        """
                        INSERT INTO memory_embeddings (content, embedding, metadata, source_batch_id)
                        VALUES ($1, $2, $3, $4)
                        """,
                        content,
                        embedding_bytes,
                        metadata_json,
                        source_batch_id,
                    )
                    count += 1

        return count

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for memories similar to the query embedding.

        Args:
            query_embedding: The query vector
            limit: Maximum number of results
            threshold: Minimum similarity score (0-1)
            filter_metadata: Optional metadata filters

        Returns:
            List of memories with similarity scores, ordered by similarity
        """
        threshold = threshold or config.embedding_similarity_threshold

        async with self._db_pool.pool.acquire() as conn:
            # Check if pgvector is available
            has_vector = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )

            if has_vector:
                # Use pgvector for efficient search
                embedding_bytes = embedding_to_bytes(query_embedding)

                if filter_metadata:
                    # Filter by metadata
                    rows = await conn.fetch(
                        """
                        SELECT id, content, embedding, metadata, source_batch_id, created_at,
                               1 - (embedding <=> $1) as similarity
                        FROM memory_embeddings
                        WHERE (1 - (embedding <=> $1)) >= $2
                          AND metadata::text LIKE $3
                        ORDER BY embedding <=> $1
                        LIMIT $4
                        """,
                        embedding_bytes,
                        threshold,
                        f"%{json.dumps(filter_metadata)}%",
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT id, content, embedding, metadata, source_batch_id, created_at,
                               1 - (embedding <=> $1) as similarity
                        FROM memory_embeddings
                        WHERE (1 - (embedding <=> $1)) >= $2
                        ORDER BY embedding <=> $1
                        LIMIT $3
                        """,
                        embedding_bytes,
                        threshold,
                        limit,
                    )
            else:
                # Fallback to full-text search
                logger.warning(
                    "pgvector not available, falling back to full-text search"
                )

                if filter_metadata:
                    conditions = []
                    params = []
                    param_idx = 3
                    for k, v in filter_metadata.items():
                        conditions.append(
                            "metadata->>$%d = $%d" % (param_idx, param_idx)
                        )
                        params.append(v)
                        param_idx += 1

                    query = """
                        SELECT id, content, embedding, metadata, source_batch_id, created_at,
                               ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) as similarity
                        FROM memory_embeddings
                        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                          AND %s
                        ORDER BY similarity DESC
                        LIMIT $2
                    """ % " AND ".join(conditions)
                    rows = await conn.fetch(
                        query,
                        query_embedding
                        if isinstance(query_embedding, str)
                        else " ".join(str(x) for x in query_embedding[:10]),
                        limit,
                        *params,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT id, content, embedding, metadata, source_batch_id, created_at,
                               ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) as similarity
                        FROM memory_embeddings
                        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                        ORDER BY similarity DESC
                        LIMIT $2
                        """,
                        query_embedding
                        if isinstance(query_embedding, str)
                        else " ".join(str(x) for x in query_embedding[:10]),
                        limit,
                    )

        results = []
        for row in rows:
            result = dict(row)
            result["similarity"] = float(result.get("similarity", 0))
            if result.get("metadata"):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    pass
            results.append(result)

        return results

    async def search_by_text(
        self,
        query_text: str,
        limit: int = 10,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search for memories similar to query text (generates embedding internally).

        Args:
            query_text: The text to search for
            limit: Maximum number of results
            threshold: Minimum similarity score

        Returns:
            List of memories with similarity scores
        """
        from lattice.utils.llm_client import _LLMClient

        llm_client = _LLMClient()
        embedding = await llm_client.embed(query_text)

        return await self.search_similar(embedding, limit, threshold)

    async def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent memories regardless of similarity.

        Args:
            limit: Maximum number of results

        Returns:
            List of recent memories
        """
        async with self._db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, metadata, source_batch_id, created_at
                FROM memory_embeddings
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )

        results = []
        for row in rows:
            result = dict(row)
            if result.get("metadata"):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    pass
            results.append(result)

        return results

    async def get_by_id(self, memory_id: UUID) -> dict[str, Any] | None:
        """Get a memory by its ID.

        Args:
            memory_id: UUID of the memory

        Returns:
            Memory dict or None if not found
        """
        async with self._db_pool.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, content, embedding, metadata, source_batch_id, created_at
                FROM memory_embeddings
                WHERE id = $1
                """,
                memory_id,
            )

        if not row:
            return None

        result = dict(row)
        if result.get("metadata"):
            try:
                result["metadata"] = json.loads(result["metadata"])
            except json.JSONDecodeError:
                pass
        return result

    async def count(self) -> int:
        """Get total count of stored memories.

        Returns:
            Number of memories
        """
        async with self._db_pool.pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM memory_embeddings")
