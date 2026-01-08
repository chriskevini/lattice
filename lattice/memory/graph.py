"""Graph traversal utilities for semantic triple relationships.

This module provides text-based semantic triple queries with BFS traversal
capability. The semantic_triple table uses text columns (subject, predicate,
object) rather than entity IDs, following the timestamp-based evolution design
from issue #131.

BFS Traversal Design:
- Multi-hop traversal is implemented iteratively using text matching
- Each hop expands to triples containing entities discovered in the previous hop
- Cycle detection prevents infinite loops by tracking visited entities
"""

import logging
from typing import Any

import asyncpg


logger = logging.getLogger(__name__)

MAX_ENTITY_NAME_LENGTH = 255


def _escape_like_pattern(entity: str) -> str:
    """Escape special characters for ILIKE pattern matching.

    Prevents injection of wildcards (%) and underscore (_) that could
    cause unexpected matching behavior.

    Args:
        entity: The entity name to escape

    Returns:
        Entity name with ILIKE wildcards escaped
    """
    return entity.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _sanitize_entity_name(entity: str) -> str:
    """Sanitize entity name for safe pattern matching.

    Truncates long names and escapes special characters.

    Args:
        entity: The entity name to sanitize

    Returns:
        Sanitized entity name safe for ILIKE queries
    """
    if len(entity) > MAX_ENTITY_NAME_LENGTH:
        entity = entity[:MAX_ENTITY_NAME_LENGTH]
    return _escape_like_pattern(entity)


class GraphTraversal:
    """Text-based graph traversal for semantic triples.

    Provides BFS traversal for multi-hop reasoning using text-based triple storage.
    Unlike entity-ID based traversal, this uses iterative text matching to discover
    related entities across multiple hops.
    """

    def __init__(self, db_pool: asyncpg.Pool, max_depth: int = 3) -> None:
        """Initialize graph traverser.

        Args:
            db_pool: Database connection pool
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
            List of discovered triples with metadata including depth
        """
        if max_hops is None:
            max_hops = self.max_depth

        all_triples: list[dict[str, Any]] = []
        visited_entities: set[str] = set()
        frontier: set[str] = {entity_name.lower()}
        seen_triples: set[tuple[str, str, str]] = set()

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

                    sanitized_entity = _sanitize_entity_name(entity)
                    triples = await conn.fetch(
                        """
                        SELECT
                            subject,
                            predicate,
                            object,
                            created_at
                        FROM semantic_triple
                        WHERE (subject ILIKE $1 OR object ILIKE $1)
                          AND ($2 IS NULL OR predicate = ANY($2))
                        ORDER BY created_at DESC
                        LIMIT 50
                        """,
                        f"%{sanitized_entity}%",
                        list(predicate_filter) if predicate_filter else None,
                    )

                    for row in triples:
                        triple = dict(row)
                        triple_key = (
                            triple["subject"].lower(),
                            triple["predicate"].lower(),
                            triple["object"].lower(),
                        )
                        if triple_key in seen_triples:
                            continue
                        seen_triples.add(triple_key)
                        triple["depth"] = depth
                        all_triples.append(triple)

                        discovered_entity = (
                            triple["object"]
                            if triple["subject"].lower() == entity_lower
                            else triple["subject"]
                        )
                        if discovered_entity.lower() not in visited_entities:
                            next_frontier.add(discovered_entity)

                frontier = next_frontier

        return all_triples

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
            sanitized_name = _sanitize_entity_name(entity_name)
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
                f"%{sanitized_name}%",
                limit,
            )
            return [dict(row) for row in rows]
