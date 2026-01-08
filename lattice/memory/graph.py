"""Graph traversal utilities for semantic triple relationships.

This module provides text-based semantic triple queries. The semantic_triple
table uses text columns (subject, predicate, object) rather than entity IDs,
following the timestamp-based evolution design from issue #131.
"""

import logging
from typing import Any

import asyncpg


logger = logging.getLogger(__name__)


class GraphTraversal:
    """Text-based graph traversal for semantic triples.

    Note: This implementation uses simple text matching rather than
    entity ID-based graph traversal, following the design decision
    in issue #131 to use text-based triples for simplicity and flexibility.
    """

    def __init__(self, db_pool: asyncpg.Pool, max_depth: int = 3) -> None:
        """Initialize graph traverser.

        Args:
            db_pool: Database connection pool
            max_depth: Not used in text-based implementation (kept for API compatibility)
        """
        self.db_pool = db_pool
        self.max_depth = max_depth

    async def find_entity_relationships(
        self,
        entity_name: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find all relationships involving a specific entity by name.

        Uses text-based matching against subject and object columns.
        Returns most recent triples first (timestamp-based evolution).

        Args:
            entity_name: Entity name to search for (case-insensitive partial match)
            limit: Maximum number of results

        Returns:
            List of triples with keys: subject, predicate, object, created_at
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    subject,
                    predicate,
                    object,
                    created_at
                FROM semantic_triple
                WHERE subject ILIKE $1 OR object ILIKE $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                f"%{entity_name}%",
                limit,
            )
            return [dict(row) for row in rows]
