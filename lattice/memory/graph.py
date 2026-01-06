"""Graph traversal utilities for semantic triple relationships."""

import logging
from typing import Any
from uuid import UUID

import asyncpg


logger = logging.getLogger(__name__)


class GraphTraversal:
    """Neuro-symbolic graph traversal for relational reasoning."""

    def __init__(self, db_pool: asyncpg.Pool, max_depth: int = 3) -> None:
        """Initialize graph traverser.

        Args:
            db_pool: Database connection pool
            max_depth: Default maximum traversal depth
        """
        self.db_pool = db_pool
        self.max_depth = max_depth

    async def traverse_from_fact(
        self,
        fact_id: UUID,
        predicate_filter: set[str] | None = None,
        max_hops: int | None = None,
    ) -> list[dict[str, Any]]:
        """BFS traversal starting from an entity.

        Args:
            fact_id: Starting entity UUID
            predicate_filter: Optional set of predicates to follow (None = all)
            max_hops: Maximum traversal depth (defaults to self.max_depth)

        Returns:
            List of discovered triples with metadata
        """
        if max_hops is None:
            max_hops = self.max_depth

        async with self.db_pool.acquire() as conn:
            result = await conn.fetch(
                """
                WITH RECURSIVE traversal AS (
                    SELECT
                        t.id,
                        t.subject_id,
                        s.name AS subject_content,
                        t.predicate,
                        t.object_id,
                        o.name AS object_content,
                        1 AS depth,
                        ARRAY[t.object_id] AS visited
                    FROM semantic_triples t
                    JOIN entities s ON t.subject_id = s.id
                    JOIN entities o ON t.object_id = o.id
                    WHERE t.subject_id = $1
                      AND ($2 IS NULL OR t.predicate = ANY($2))

                    UNION ALL

                    SELECT
                        t.id,
                        t.subject_id,
                        s.name AS subject_content,
                        t.predicate,
                        t.object_id,
                        o.name AS object_content,
                        traversal.depth + 1,
                        traversal.visited || t.object_id
                    FROM semantic_triples t
                    JOIN entities s ON t.subject_id = s.id
                    JOIN entities o ON t.object_id = o.id
                    JOIN traversal ON t.subject_id = traversal.object_id
                    WHERE traversal.depth < $3
                      AND NOT t.object_id = ANY(traversal.visited)
                      AND ($4 IS NULL OR t.predicate = ANY($4))
                )
                SELECT * FROM traversal
                """,
                fact_id,
                list(predicate_filter) if predicate_filter else None,
                max_hops,
                list(predicate_filter) if predicate_filter else None,
            )

            return [dict(row) for row in result]

    async def find_entity_relationships(
        self,
        entity_name: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find all relationships involving a specific entity by name.

        Args:
            entity_name: Entity name to search for
            limit: Maximum number of results

        Returns:
            List of triples involving the entity (includes origin_id for source attribution)
        """
        async with self.db_pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT
                    s.name AS subject,
                    t.predicate,
                    o.name AS object,
                    t.created_at,
                    t.origin_id
                FROM semantic_triples t
                JOIN entities s ON t.subject_id = s.id
                JOIN entities o ON t.object_id = o.id
                WHERE s.name ILIKE $1 OR o.name ILIKE $1
                ORDER BY t.created_at DESC
                LIMIT $2
                """,
                f"%{entity_name}%",
                limit,
            )
