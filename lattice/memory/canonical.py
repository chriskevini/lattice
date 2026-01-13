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
from typing import TYPE_CHECKING, Any

from lattice.memory.repositories import CanonicalRepository

if TYPE_CHECKING:
    from lattice.memory.repositories import CanonicalRepository
    from lattice.utils.database import DatabasePool
else:
    CanonicalRepository = Any
    DatabasePool = Any


logger = structlog.get_logger(__name__)


class CanonicalRegistryError(Exception):
    """Base exception for canonical registry errors."""

    pass


async def get_canonical_entities_list(
    repo: CanonicalRepository | None = None, db_pool: DatabasePool | None = None
) -> list[str]:
    """Fetch all canonical entity names from database.

    Args:
        repo: Canonical repository
        db_pool: Optional database pool (for legacy support)

    Returns:
        List of entity names sorted by creation date (newest first)
    """
    if repo:
        return await repo.get_entities_list()
    if db_pool:
        # Fallback for tests not yet using repositories
        from lattice.memory.repositories import PostgresRepository

        class TempCanonicalRepo(PostgresRepository):
            async def get_entities_list(self):
                async with self._db_pool.pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT name FROM entities ORDER BY created_at DESC"
                    )
                    return [row["name"] for row in rows]

            async def get_predicates_list(self):
                return []

            async def get_entities_set(self):
                return set()

            async def get_predicates_set(self):
                return set()

            async def store_entities(self, names):
                return 0

            async def store_predicates(self, names):
                return 0

            async def entity_exists(self, name):
                return False

            async def predicate_exists(self, name):
                return False

        temp_repo = TempCanonicalRepo(db_pool)
        return await temp_repo.get_entities_list()
    raise ValueError("Either repo or db_pool must be provided")


async def get_canonical_predicates_list(
    repo: CanonicalRepository | None = None, db_pool: DatabasePool | None = None
) -> list[str]:
    """Fetch all canonical predicate names from database.

    Args:
        repo: Canonical repository
        db_pool: Optional database pool (for legacy support)

    Returns:
        List of predicate names sorted by creation date (newest first)
    """
    if repo:
        return await repo.get_predicates_list()
    if db_pool:
        # Fallback for tests not yet using repositories
        from lattice.memory.repositories import PostgresRepository

        class TempCanonicalRepo(PostgresRepository):
            async def get_entities_list(self):
                return []

            async def get_predicates_list(self):
                async with self._db_pool.pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT name FROM predicates ORDER BY created_at DESC"
                    )
                    return [row["name"] for row in rows]

            async def get_entities_set(self):
                return set()

            async def get_predicates_set(self):
                return set()

            async def store_entities(self, names):
                return 0

            async def store_predicates(self, names):
                return 0

            async def entity_exists(self, name):
                return False

            async def predicate_exists(self, name):
                return False

        temp_repo = TempCanonicalRepo(db_pool)
        return await temp_repo.get_predicates_list()
    raise ValueError("Either repo or db_pool must be provided")


async def get_canonical_entities_set(
    repo: CanonicalRepository | None = None, db_pool: DatabasePool | None = None
) -> set[str]:
    """Fetch all canonical entities as a set for O(1) lookup.

    Args:
        repo: Canonical repository
        db_pool: Optional database pool (for legacy support)

    Returns:
        Set of entity names for fast membership testing
    """
    if repo:
        return await repo.get_entities_set()
    if db_pool:
        # Fallback for tests not yet using repositories
        from lattice.memory.context import PostgresCanonicalRepository

        repo = PostgresCanonicalRepository(db_pool)
        return await repo.get_entities_set()
    raise ValueError("Either repo or db_pool must be provided")


async def get_canonical_predicates_set(
    repo: CanonicalRepository | None = None, db_pool: DatabasePool | None = None
) -> set[str]:
    """Fetch all canonical predicates as a set for O(1) lookup.

    Args:
        repo: Canonical repository
        db_pool: Optional database pool (for legacy support)

    Returns:
        Set of predicate names for fast membership testing
    """
    if repo:
        return await repo.get_predicates_set()
    if db_pool:
        # Fallback for tests not yet using repositories
        from lattice.memory.context import PostgresCanonicalRepository

        repo = PostgresCanonicalRepository(db_pool)
        return await repo.get_predicates_set()
    raise ValueError("Either repo or db_pool must be provided")


async def store_canonical_entities(
    repo: CanonicalRepository | None = None,
    names: list[str] = [],
    db_pool: DatabasePool | None = None,
) -> int:
    """Store new canonical entities, ignoring duplicates.

    Args:
        repo: Canonical repository
        names: List of entity names to store
        db_pool: Optional database pool (for legacy support)

    Returns:
        Number of entities actually inserted
    """
    if repo:
        return await repo.store_entities(names)
    if db_pool:
        from lattice.memory.context import PostgresCanonicalRepository

        repo = PostgresCanonicalRepository(db_pool)
        return await repo.store_entities(names)
    raise ValueError("Either repo or db_pool must be provided")


async def store_canonical_predicates(
    repo: CanonicalRepository | None = None,
    names: list[str] = [],
    db_pool: DatabasePool | None = None,
) -> int:
    """Store new canonical predicates, ignoring duplicates.

    Args:
        repo: Canonical repository
        names: List of predicate names to store
        db_pool: Optional database pool (for legacy support)

    Returns:
        Number of predicates actually inserted
    """
    if repo:
        return await repo.store_predicates(names)
    if db_pool:
        from lattice.memory.context import PostgresCanonicalRepository

        repo = PostgresCanonicalRepository(db_pool)
        return await repo.store_predicates(names)
    raise ValueError("Either repo or db_pool must be provided")


async def entity_exists(
    repo: CanonicalRepository | None = None,
    name: str = "",
    db_pool: DatabasePool | None = None,
) -> bool:
    """Check if an entity name already exists.

    Args:
        repo: Canonical repository
        name: Entity name to check
        db_pool: Optional database pool (for legacy support)

    Returns:
        True if entity exists
    """
    if repo:
        return await repo.entity_exists(name)
    if db_pool:
        from lattice.memory.context import PostgresCanonicalRepository

        repo = PostgresCanonicalRepository(db_pool)
        return await repo.entity_exists(name)
    raise ValueError("Either repo or db_pool must be provided")


async def predicate_exists(
    repo: CanonicalRepository | None = None,
    name: str = "",
    db_pool: DatabasePool | None = None,
) -> bool:
    """Check if a predicate name already exists.

    Args:
        repo: Canonical repository
        name: Predicate name to check
        db_pool: Optional database pool (for legacy support)

    Returns:
        True if predicate exists
    """
    if repo:
        return await repo.predicate_exists(name)
    if db_pool:
        from lattice.memory.context import PostgresCanonicalRepository

        repo = PostgresCanonicalRepository(db_pool)
        return await repo.predicate_exists(name)
    raise ValueError("Either repo or db_pool must be provided")


def _is_entity_like(text: str) -> bool:
    """Determine if text looks like an entity rather than a value.

    Returns False for:
    - ISO dates (2026-01-10)
    - Durations (3 hours, 30 minutes)
    - Status values (active, completed, high, low)

    Returns True for:
    - Proper nouns (Mother, IKEA, Seattle)
    - Concepts (coding, mobile app, marathon)
    """
    import re

    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return False

    if re.match(r"^\d+\s+(hour|minute|day|week|month|year)s?$", text.lower()):
        return False

    if text.lower() in {
        "active",
        "completed",
        "abandoned",
        "high",
        "medium",
        "low",
        "pending",
        "yes",
        "no",
        "true",
        "false",
        "alpha",
    }:
        return False

    return True


def extract_canonical_forms(
    triples: list[dict[str, str]],
    known_entities: set[str],
    known_predicates: set[str],
) -> tuple[list[str], list[str]]:
    """Extract new canonical entities and predicates from triples.

    This is deterministic logic (no LLM involved).

    Args:
        triples: List of extracted triples with keys: subject, predicate, object
        known_entities: Set of already-known entity names
        known_predicates: Set of already-known predicate names

    Returns:
        Tuple of (new_entities, new_predicates) lists, sorted alphabetically
    """
    new_entities: set[str] = set()
    new_predicates: set[str] = set()

    for triple in triples:
        subject = triple["subject"]
        predicate = triple["predicate"]
        obj = triple["object"]

        if subject not in known_entities:
            new_entities.add(subject)

        if predicate not in known_predicates:
            new_predicates.add(predicate)

        if _is_entity_like(obj) and obj not in known_entities:
            new_entities.add(obj)

    return sorted(new_entities), sorted(new_predicates)


async def store_canonical_forms(
    repo: CanonicalRepository | None = None,
    new_entities: list[str] = [],
    new_predicates: list[str] = [],
    db_pool: DatabasePool | None = None,
) -> dict[str, int]:
    """Store new canonical entities and predicates in database.

    Args:
        repo: Canonical repository
        new_entities: Sorted list of new entity names to store
        new_predicates: Sorted list of new predicate names to store
        db_pool: Optional database pool (for legacy support)

    Returns:
        Dictionary with 'entities' and 'predicates' counts of items stored
    """
    entity_count = await store_canonical_entities(
        repo=repo, names=new_entities, db_pool=db_pool
    )
    predicate_count = await store_canonical_predicates(
        repo=repo, names=new_predicates, db_pool=db_pool
    )

    if new_entities or new_predicates:
        logger.info(
            "Stored canonical forms",
            new_entities=len(new_entities),
            new_predicates=len(new_predicates),
        )

    return {"entities": entity_count, "predicates": predicate_count}
