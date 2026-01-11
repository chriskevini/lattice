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


if TYPE_CHECKING:
    import asyncpg


logger = structlog.get_logger(__name__)


MAX_ENTITY_NAME_LENGTH = 255


class GraphTraversal:
    """Text-based graph traversal for semantic memories.

    Provides BFS traversal for multi-hop reasoning using text-based memory storage.
    Unlike entity-ID based traversal, this uses iterative text matching to discover
    related entities across multiple hops.
    """

    def __init__(self, db_pool: Any, max_depth: int = 3) -> None:
        """Initialize graph traverser.

        Args:
            db_pool: Database connection pool (asyncpg.Pool)
            max_depth: Default maximum traversal depth for BFS
        """
        self.db_pool = db_pool
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

        all_memories: list[dict[str, Any]] = []
        visited_entities: set[str] = set()
        frontier: set[str] = {entity_name.lower()}
        seen_memories: set[tuple[str, str, str]] = set()
        predicates = list(predicate_filter) if predicate_filter else []

        async with self.db_pool.acquire() as conn:
            for depth in range(1, max_hops + 1):
                if not frontier:
                    break

                next_frontier: set[str] = set()

                for entity in frontier:
                    entity_lower = entity.lower()
                    if entity_lower in visited_entities:
                        continue
                    visited_entities.add(entity_lower)

                    # Sanitize for ILIKE matching
                    sanitized_entity = entity.replace("%", "\\%").replace("_", "\\_")
                    if predicates:
                        query = """
                        SELECT
                            subject,
                            predicate,
                            object,
                            created_at
                        FROM semantic_memories
                        WHERE (subject ILIKE $1 OR object ILIKE $1)
                          AND predicate = ANY($2)
                        ORDER BY created_at DESC
                        LIMIT 50
                        """
                        params = [sanitized_entity, predicates]
                    else:
                        query = """
                        SELECT
                            subject,
                            predicate,
                            object,
                            created_at
                        FROM semantic_memories
                        WHERE (subject ILIKE $1 OR object ILIKE $1)
                        ORDER BY created_at DESC
                        LIMIT 50
                        """
                        params = [sanitized_entity]
                    memories = await conn.fetch(query, *params)

                    for row in memories:
                        memory = dict(row)
                        memory_key = (
                            memory["subject"].lower(),
                            memory["predicate"].lower(),
                            memory["object"].lower(),
                        )
                        if memory_key in seen_memories:
                            continue
                        seen_memories.add(memory_key)
                        memory["depth"] = depth
                        all_memories.append(memory)

                        discovered_entity = (
                            memory["object"]
                            if memory["subject"].lower() == entity_lower
                            else memory["subject"]
                        )
                        if discovered_entity.lower() not in visited_entities:
                            next_frontier.add(discovered_entity)

                frontier = next_frontier

        return all_memories

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
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT
                        subject,
                        predicate,
                        object,
                        created_at
                    FROM semantic_memories
                    WHERE 1=1
                """
                params: list[Any] = []

                if subject is not None:
                    query += " AND subject = $" + str(len(params) + 1)
                    params.append(subject)

                if predicate is not None:
                    query += " AND predicate = $" + str(len(params) + 1)
                    params.append(predicate)

                if object is not None:
                    query += " AND object = $" + str(len(params) + 1)
                    params.append(object)

                if start_date is not None:
                    query += " AND created_at >= $" + str(len(params) + 1)
                    params.append(start_date)

                if end_date is not None:
                    query += " AND created_at <= $" + str(len(params) + 1)
                    params.append(end_date)

                query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
                params.append(limit)

                rows = await conn.fetch(query, *params)
                results = [dict(row) for row in rows]

                logger.info(
                    "Memory query completed",
                    subject=subject,
                    predicate=predicate,
                    object=object,
                    result_count=len(results),
                )
                return results
        except Exception as e:
            logger.error(
                "Memory query failed",
                subject=subject,
                predicate=predicate,
                object=object,
                error=str(e),
            )
            return []
