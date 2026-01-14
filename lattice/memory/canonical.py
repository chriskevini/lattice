"""Canonical entity and predicate registry for normalization.

Stores canonical entity and predicate names extracted from conversation triples.
Used as reference context for LLM-based normalization.

Usage Guide:
- Use *_list() functions when: You need ordered list for display/context
- Use *_set() functions when: You need fast O(1) membership checks

Example:
    # For display/context (preserves order)
    entities = await get_canonical_entities_list(repo)

    # For membership checks (O(1) lookup)
    if "Friday" in await get_canonical_entities_set(repo):
        ...

Tables:
- entities: Stores canonical entity names (id, name, created_at)
- predicates: Stores canonical predicate names (id, name, created_at)
"""

import structlog

from lattice.memory.repositories import CanonicalRepository


logger = structlog.get_logger(__name__)


class CanonicalRegistryError(Exception):
    """Base exception for canonical registry errors."""

    pass


async def get_canonical_entities_list(repo: CanonicalRepository) -> list[str]:
    """Fetch all canonical entity names from database.

    Args:
        repo: Canonical repository

    Returns:
        List of entity names sorted by creation date (newest first)
    """
    return await repo.get_entities_list()


async def get_canonical_predicates_list(repo: CanonicalRepository) -> list[str]:
    """Fetch all canonical predicate names from database.

    Args:
        repo: Canonical repository

    Returns:
        List of predicate names sorted by creation date (newest first)
    """
    return await repo.get_predicates_list()


async def get_canonical_entities_set(repo: CanonicalRepository) -> set[str]:
    """Fetch all canonical entities as a set for O(1) lookup.

    Args:
        repo: Canonical repository

    Returns:
        Set of entity names for fast membership testing
    """
    return await repo.get_entities_set()


async def get_canonical_predicates_set(repo: CanonicalRepository) -> set[str]:
    """Fetch all canonical predicates as a set for O(1) lookup.

    Args:
        repo: Canonical repository

    Returns:
        Set of predicate names for fast membership testing
    """
    return await repo.get_predicates_set()


async def store_canonical_entities(repo: CanonicalRepository, names: list[str]) -> int:
    """Store new canonical entities, ignoring duplicates.

    Args:
        repo: Canonical repository
        names: List of entity names to store

    Returns:
        Number of entities actually inserted
    """
    return await repo.store_entities(names)


async def store_canonical_predicates(
    repo: CanonicalRepository, names: list[str]
) -> int:
    """Store new canonical predicates, ignoring duplicates.

    Args:
        repo: Canonical repository
        names: List of predicate names to store

    Returns:
        Number of predicates actually inserted
    """
    return await repo.store_predicates(names)


async def entity_exists(repo: CanonicalRepository, name: str) -> bool:
    """Check if an entity name already exists.

    Args:
        repo: Canonical repository
        name: Entity name to check

    Returns:
        True if entity exists
    """
    return await repo.entity_exists(name)


async def predicate_exists(repo: CanonicalRepository, name: str) -> bool:
    """Check if a predicate name already exists.

    Args:
        repo: Canonical repository
        name: Predicate name to check

    Returns:
        True if predicate exists
    """
    return await repo.predicate_exists(name)


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
    repo: CanonicalRepository,
    new_entities: list[str],
    new_predicates: list[str],
) -> dict[str, int]:
    """Store new canonical entities and predicates in database.

    Args:
        repo: Canonical repository
        new_entities: Sorted list of new entity names to store
        new_predicates: Sorted list of new predicate names to store

    Returns:
        Dictionary with 'entities' and 'predicates' counts of items stored
    """
    entity_count = await store_canonical_entities(repo=repo, names=new_entities)
    predicate_count = await store_canonical_predicates(repo=repo, names=new_predicates)

    if new_entities or new_predicates:
        logger.info(
            "Stored canonical forms",
            new_entities=len(new_entities),
            new_predicates=len(new_predicates),
        )

    return {"entities": entity_count, "predicates": predicate_count}
