"""Context analyzer for semantic archetype matching.

Determines optimal context configuration (turns, vectors, similarity, depth)
by matching incoming messages against pre-defined conversation archetypes.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np
import structlog

from lattice.utils.database import db_pool
from lattice.utils.embeddings import embedding_model


logger = structlog.get_logger(__name__)

# Default fallback values if no archetypes exist
DEFAULT_CONTEXT_TURNS = 5
DEFAULT_CONTEXT_VECTORS = 6
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_TRIPLE_DEPTH = 1


@dataclass
class ContextConfig:
    """Context configuration determined by archetype matching."""

    turns: int  # Number of recent conversation turns to include
    vectors: int  # Number of semantic facts to retrieve
    similarity: float  # Similarity threshold for vector search
    depth: int  # Depth for graph traversal
    archetype_name: str | None = None  # Matched archetype name
    match_similarity: float | None = None  # Similarity score of match


class ContextAnalyzer:
    """Determines context configuration via semantic archetype matching."""

    def __init__(self) -> None:
        """Initialize context analyzer."""
        self.archetypes: list[dict[str, Any]] = []
        self.cache_updated_at: datetime | None = None
        self._refresh_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start background archetype refresh loop."""
        await self._load_archetypes()
        self._refresh_task = asyncio.create_task(self._refresh_archetypes_loop())
        logger.info("ContextAnalyzer started")

    async def stop(self) -> None:
        """Stop background refresh loop."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("ContextAnalyzer stopped")

    async def _refresh_archetypes_loop(self) -> None:
        """Reload archetypes from database every 60 seconds."""
        while True:
            try:
                await asyncio.sleep(60)
                await self._load_archetypes()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error refreshing archetypes")

    async def _load_archetypes(self) -> None:
        """Load active archetypes and compute/cache centroids."""
        if not db_pool.is_initialized():
            logger.warning("Database pool not initialized, skipping archetype load")
            return

        async with db_pool.pool.acquire() as conn:
            archetypes_data = await conn.fetch(
                """
                SELECT * FROM context_archetypes WHERE active = true
                ORDER BY archetype_name
                """
            )

        new_archetypes = []

        for arch in archetypes_data:
            # Compute centroid if not cached
            if arch["centroid_embedding"] is None:
                logger.info(
                    "Computing centroid for archetype",
                    archetype=arch["archetype_name"],
                )
                examples = arch["example_messages"]
                embeddings = embedding_model.encode(examples)
                centroid = embeddings.mean(axis=0)

                # Cache in database
                async with db_pool.pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE context_archetypes
                        SET centroid_embedding = $1, updated_at = now()
                        WHERE id = $2
                        """,
                        centroid.tolist(),
                        arch["id"],
                    )
                centroid_array = centroid
            else:
                # Convert from database vector format
                centroid_array = np.array(arch["centroid_embedding"])

            new_archetypes.append(
                {
                    "id": arch["id"],
                    "name": arch["archetype_name"],
                    "centroid": centroid_array,
                    "config": {
                        "turns": arch["context_turns"],
                        "vectors": arch["context_vectors"],
                        "similarity": arch["similarity_threshold"],
                        "depth": arch["triple_depth"],
                    },
                }
            )

        self.archetypes = new_archetypes
        self.cache_updated_at = datetime.now()
        logger.info(
            "Loaded archetypes",
            count=len(self.archetypes),
            names=[a["name"] for a in self.archetypes],
        )

    async def analyze(self, message: str) -> ContextConfig:
        """Classify message and return optimal context configuration.

        Args:
            message: The incoming message to classify

        Returns:
            ContextConfig with turns, vectors, similarity, depth
        """
        # If no archetypes loaded, use defaults
        if not self.archetypes:
            logger.warning("No archetypes available, using defaults")
            return ContextConfig(
                turns=DEFAULT_CONTEXT_TURNS,
                vectors=DEFAULT_CONTEXT_VECTORS,
                similarity=DEFAULT_SIMILARITY_THRESHOLD,
                depth=DEFAULT_TRIPLE_DEPTH,
            )

        # Generate embedding for message
        msg_embedding = embedding_model.encode([message])
        msg_embedding = msg_embedding[0]  # Extract from batch

        # Find best matching archetype
        best_match: dict[str, Any] | None = None
        best_similarity = -1.0

        for archetype in self.archetypes:
            similarity = self._cosine_similarity(msg_embedding, archetype["centroid"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = archetype

        if best_match is None:
            # Should never happen if archetypes list is non-empty
            logger.warning("No archetype matched, using defaults")
            return ContextConfig(
                turns=DEFAULT_CONTEXT_TURNS,
                vectors=DEFAULT_CONTEXT_VECTORS,
                similarity=DEFAULT_SIMILARITY_THRESHOLD,
                depth=DEFAULT_TRIPLE_DEPTH,
            )

        # Update statistics asynchronously (don't block response)
        asyncio.create_task(
            self._update_archetype_stats(best_match["id"], best_similarity)
        )

        # Log classification
        logger.debug(
            "Archetype matched",
            archetype=best_match["name"],
            similarity=f"{best_similarity:.3f}",
            config=best_match["config"],
        )

        # Return configuration
        config = best_match["config"]
        return ContextConfig(
            turns=config["turns"],
            vectors=config["vectors"],
            similarity=config["similarity"],
            depth=config["depth"],
            archetype_name=best_match["name"],
            match_similarity=best_similarity,
        )

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def _update_archetype_stats(
        self, archetype_id: UUID, similarity: float
    ) -> None:
        """Update match statistics for archetype."""
        try:
            if not db_pool.is_initialized():
                return

            async with db_pool.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE context_archetypes
                    SET
                        match_count = match_count + 1,
                        avg_similarity = CASE
                            WHEN avg_similarity IS NULL THEN $2
                            ELSE (avg_similarity * match_count + $2) / (match_count + 1)
                        END,
                        updated_at = now()
                    WHERE id = $1
                    """,
                    archetype_id,
                    similarity,
                )
        except Exception:
            logger.exception(
                "Failed to update archetype stats", archetype_id=archetype_id
            )


# Global singleton instance
_context_analyzer: ContextAnalyzer | None = None


def get_context_analyzer() -> ContextAnalyzer:
    """Get or create the global context analyzer instance."""
    global _context_analyzer
    if _context_analyzer is None:
        _context_analyzer = ContextAnalyzer()
    return _context_analyzer
