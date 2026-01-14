"""Graph traversal utilities for semantic memory relationships.

This module provides text-based semantic memory queries with BFS traversal
capability. The semantic_memories table uses text columns (subject, predicate,
object) rather than entity IDs, following the timestamp-based evolution design
from issue #131.

BFS Traversal Design:
- Multi-hop traversal is implemented iteratively using text matching
- Each hop expands to memories containing entities discovered in the previous hop
- Cycle detection prevents infinite loops by tracking visited entities
"""

from datetime import datetime

import structlog

from lattice.memory.repositories import SemanticMemoryRepository


logger = structlog.get_logger(__name__)


MAX_ENTITY_NAME_LENGTH = 255


class GraphTraversal:
    """Text-based graph traversal for semantic memories.

    Provides BFS traversal for multi-hop reasoning using text-based memory storage.
    Unlike entity-ID based traversal, this uses iterative text matching to discover
    related entities across multiple hops.
    """

    def __init__(
        self,
        repo: SemanticMemoryRepository,
        max_depth: int = 3,
    ) -> None:
        """Initialize graph traverser.

        Args:
            repo: Semantic memory repository
            max_depth: Default maximum traversal depth for BFS
        """
        self.repo = repo
        self.max_depth = max_depth

    async def traverse_from_entity(
        self,
        entity_name: str,
        predicate_filter: set[str] | None = None,
        max_hops: int | None = None,
    ) -> list[dict[str, str | datetime]]:
        """Perform iterative BFS traversal starting from an entity name.

        Args:
            entity_name: Starting entity to traverse from
            predicate_filter: Optional set of predicates to follow (default: all)
            max_hops: Maximum number of hops to traverse (default: self.max_depth)

        Returns:
            List of memories found during traversal, ordered by distance (hops) from start
        """
        max_hops = max_hops or self.max_depth
        return await self.repo.traverse_from_entity(
            entity_name=entity_name,
            predicate_filter=predicate_filter,
            max_hops=max_hops,
        )
