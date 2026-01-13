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
from typing import TYPE_CHECKING, Any

import structlog

from lattice.memory.repositories import SemanticMemoryRepository

if TYPE_CHECKING:
    from lattice.memory.repositories import SemanticMemoryRepository
    from lattice.utils.database import DatabasePool
else:
    SemanticMemoryRepository = Any
    DatabasePool = Any


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
        repo: Any = None,
        max_depth: int = 3,
        db_pool: DatabasePool | None = None,
    ) -> None:
        """Initialize graph traverser.

        Args:
            repo: Semantic memory repository or legacy DatabasePool
            max_depth: Default maximum traversal depth for BFS
            db_pool: Optional database pool (for legacy support)
        """
        if repo is not None:
            # Check if it's a repository (has find_memories) or a mock that looks like one
            if hasattr(repo, "find_memories"):
                self.repo = repo
            elif hasattr(repo, "pool"):
                # Treat as legacy db_pool
                from lattice.memory.context import PostgresSemanticMemoryRepository

                self.repo = PostgresSemanticMemoryRepository(repo)
            else:
                # Fallback for generic mocks or other objects
                self.repo = repo
        elif db_pool is not None:
            from lattice.memory.context import PostgresSemanticMemoryRepository

            self.repo = PostgresSemanticMemoryRepository(db_pool)
        else:
            raise ValueError("Either repo or db_pool must be provided")

        self.max_depth = max_depth

    async def traverse_from_entity(
        self,
        entity_name: str,
        predicate_filter: set[str] | None = None,
        max_hops: int | None = None,
    ) -> list[dict[str, Any]]:
        """BFS traversal starting from an entity name.

        Performs breadth-first search to find entities connected through chains
        of relationships. For example, given "Alice", it can discover:
        - Hop 1: Alice works_at Acme Corp
        - Hop 2: Acme Corp acquired_by TechCorp
        - Hop 3: TechCorp in Technology Industry

        Args:
            entity_name: Starting entity name (case-insensitive partial match)
            predicate_filter: Optional set of predicates to follow (None = all)
            max_hops: Maximum traversal depth (defaults to self.max_depth)

        Returns:
            List of discovered memories with metadata including depth
        """
        if max_hops is None:
            max_hops = self.max_depth

        return await self.repo.traverse_from_entity(
            entity_name=entity_name,
            predicate_filter=predicate_filter,
            max_hops=max_hops,
        )

    async def find_semantic_memories(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find memories matching any combination of criteria.

        Provides flexible querying for semantic memories. Any combination of
        subject, predicate, object, and date range can be specified - omitted
        fields are not filtered.

        Args:
            subject: Optional subject to filter by (exact match)
            predicate: Optional predicate to filter by (exact match)
            object: Optional object to filter by (exact match)
            start_date: Optional start of date range filter
            end_date: Optional end of date range filter
            limit: Maximum number of results (default 50)

        Returns:
            List of memories with keys: subject, predicate, object, created_at

        Examples:
            # Find all activities
            await find_semantic_memories(predicate="did activity")

            # Find User's activities from last week
            await find_semantic_memories(
                subject="User",
                predicate="did activity",
                start_date=get_now("UTC") - timedelta(days=7)
            )

            # Find all memories about a specific object
            await find_semantic_memories(object="ran 5k")
        """
        logger.info(
            "Finding memories by criteria",
            subject=subject,
            predicate=predicate,
            object=object,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
        results = await self.repo.find_memories(
            subject=subject,
            predicate=predicate,
            object=object,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        logger.info(
            "Memory query completed",
            subject=subject,
            predicate=predicate,
            object=object,
            result_count=len(results),
        )
        return results
