"""Canonical entity and predicate registry for normalization.

Stores canonical entity and predicate names extracted from conversation triples.
Used as reference context for LLM-based normalization.

Usage Guide:
- Use *_list() functions when: You need ordered list for display/context
- Use *_set() functions when: You need fast O(1) membership checks

Example:
    # For display/context (preserves order)
    entities = await get_canonical_entities_list()

    # For membership checks (O(1) lookup)
    if "Friday" in await get_canonical_entities_set():
        ...

Tables:
- entities: Stores canonical entity names (id, name, created_at)
- predicates: Stores canonical predicate names (id, name, created_at)
"""

import structlog

from lattice.utils.database import db_pool


logger = structlog.get_logger(__name__)


class CanonicalRegistryError(Exception):
    """Base exception for canonical registry errors."""

    pass


async def get_canonical_entities_list() -> list[str]:
    """Fetch all canonical entity names from database.

    Returns:
        List of entity names sorted by creation date (newest first)
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch("SELECT name FROM entities ORDER BY created_at DESC")
        return [row["name"] for row in rows]


async def get_canonical_predicates_list() -> list[str]:
    """Fetch all canonical predicate names from database.

    Returns:
        List of predicate names sorted by creation date (newest first)
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch("SELECT name FROM predicates ORDER BY created_at DESC")
        return [row["name"] for row in rows]


async def get_canonical_entities_set() -> set[str]:
    """Fetch all canonical entities as a set for O(1) lookup.

    Returns:
        Set of entity names for fast membership testing
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch("SELECT name FROM entities")
        return {row["name"] for row in rows}


async def get_canonical_predicates_set() -> set[str]:
    """Fetch all canonical predicates as a set for O(1) lookup.

    Returns:
        Set of predicate names for fast membership testing
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch("SELECT name FROM predicates")
        return {row["name"] for row in rows}


async def store_canonical_entities(names: list[str]) -> int:
    """Store new canonical entities, ignoring duplicates.

    Args:
        names: List of entity names to store

    Returns:
        Number of entities actually inserted
    """
    if not names:
        return 0

    async with db_pool.pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO entities (name) VALUES ($1) ON CONFLICT (name) DO NOTHING",
            [(name,) for name in names],
        )

    logger.debug("Stored canonical entities", count=len(names))
    return len(names)


async def store_canonical_predicates(names: list[str]) -> int:
    """Store new canonical predicates, ignoring duplicates.

    Args:
        names: List of predicate names to store

    Returns:
        Number of predicates actually inserted
    """
    if not names:
        return 0

    async with db_pool.pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO predicates (name) VALUES ($1) ON CONFLICT (name) DO NOTHING",
            [(name,) for name in names],
        )

    logger.debug("Stored canonical predicates", count=len(names))
    return len(names)


async def entity_exists(name: str) -> bool:
    """Check if an entity name already exists.

    Args:
        name: Entity name to check

    Returns:
        True if entity exists
    """
    async with db_pool.pool.acquire() as conn:
        result = await conn.fetchval(
            "SELECT 1 FROM entities WHERE name = $1 LIMIT 1", name
        )
        return result is not None


async def predicate_exists(name: str) -> bool:
    """Check if a predicate name already exists.

    Args:
        name: Predicate name to check

    Returns:
        True if predicate exists
    """
    async with db_pool.pool.acquire() as conn:
        result = await conn.fetchval(
            "SELECT 1 FROM predicates WHERE name = $1 LIMIT 1", name
        )
        return result is not None
