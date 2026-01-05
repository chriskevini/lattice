"""Semantic memory module - STUB during Issue #61 refactor.

This module is being rewritten with a graph-first architecture using
query extraction instead of vector embeddings.

See: https://github.com/chriskevini/lattice/issues/61
"""

from uuid import UUID

import structlog


logger = structlog.get_logger(__name__)


class StableFact:
    """Stub class for backward compatibility during refactor."""

    def __init__(
        self,
        content: str,
        fact_id: UUID | None = None,
        embedding: list[float] | None = None,
        origin_id: UUID | None = None,
        entity_type: str | None = None,
    ) -> None:
        """Initialize a stable fact (stub implementation).

        Args:
            content: Textual representation of the fact
            fact_id: UUID of the fact (auto-generated if None)
            embedding: Vector embedding (deprecated, unused)
            origin_id: UUID of the raw_message this fact was extracted from
            entity_type: Type of entity (e.g., 'person', 'preference', 'event')
        """
        self.content = content
        self.fact_id = fact_id
        self.embedding = embedding
        self.origin_id = origin_id
        self.entity_type = entity_type


async def store_fact(fact: StableFact) -> UUID:
    """Stub for storing facts (disabled during Issue #61 refactor).

    Args:
        fact: The fact to store

    Returns:
        Dummy UUID

    Note:
        This function is stubbed out during the Issue #61 refactor.
        Semantic fact storage will be reimplemented using the new
        graph-first architecture with query extraction.
    """
    logger.debug(
        "store_fact called (stubbed during Issue #61 refactor)",
        content_preview=fact.content[:50],
        entity_type=fact.entity_type,
    )
    # Return a dummy UUID to maintain compatibility
    return UUID("00000000-0000-0000-0000-000000000000")


async def search_similar_facts(
    query: str,
    limit: int = 5,
    similarity_threshold: float = 0.7,
) -> list[StableFact]:
    """Stub for semantic search (disabled during Issue #61 refactor).

    Args:
        query: Query text to search for
        limit: Maximum number of results to return
        similarity_threshold: Minimum similarity threshold

    Returns:
        Empty list (semantic search disabled during refactor)

    Note:
        This function is stubbed out during the Issue #61 refactor.
        Semantic retrieval will be reimplemented using context-aware
        graph traversal based on extracted query structure.
    """
    logger.debug(
        "search_similar_facts called (stubbed during Issue #61 refactor)",
        query_preview=query[:50],
        limit=limit,
    )
    return []
