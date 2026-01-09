"""Canonical entity and predicate registry for normalization.

Provides deterministic canonical form lookup for entities and predicates,
enabling consistent matching across different surface forms without embeddings.

Example:
    - Entity: "mom" → "Mother", "bf" → "boyfriend"
    - Predicate: "lives in" → "lives_in", "works at" → "works_as"
"""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import asyncpg
import structlog

from lattice.utils.database import db_pool


logger = structlog.get_logger(__name__)

_cache_lock = asyncio.Lock()
_entities_cache: dict[str, tuple[set[str], str | None]] | None = None
_entity_variants_cache: dict[str, str] | None = None
_predicates_cache: dict[str, str] | None = None
_cache_timestamp: datetime | None = None

CACHE_TTL_SECONDS = 300


class CanonicalRegistryError(Exception):
    """Base exception for canonical registry errors."""

    pass


class EntityNotFoundError(CanonicalRegistryError):
    """Raised when an entity variant is not found in the registry."""

    pass


class PredicateNotFoundError(CanonicalRegistryError):
    """Raised when a predicate variant is not found in the registry."""

    pass


async def _refresh_cache() -> None:
    """Refresh the in-memory cache of canonical forms."""
    global _entities_cache, _entity_variants_cache, _predicates_cache, _cache_timestamp

    async with _cache_lock:
        try:
            async with db_pool.pool.acquire() as conn:
                entities: dict[str, tuple[set[str], str | None]] = {}
                entity_variants: dict[str, str] = {}
                rows = await conn.fetch(
                    "SELECT canonical_form, variants, category FROM canonical_entities"
                )
                for row in rows:
                    canonical = row["canonical_form"]
                    variants = set(row["variants"])
                    variants.add(canonical.lower())
                    entities[canonical] = (variants, row["category"])
                    for variant in variants:
                        entity_variants[variant.lower()] = canonical

                predicates: dict[str, str] = {}
                pred_rows = await conn.fetch(
                    "SELECT canonical_form, variants FROM canonical_predicates"
                )
                for row in pred_rows:
                    canonical = row["canonical_form"]
                    variants = set(row["variants"])
                    variants.add(canonical)
                    for variant in variants:
                        predicates[variant.lower()] = canonical

                _entities_cache = entities
                _entity_variants_cache = entity_variants
                _predicates_cache = predicates
                _cache_timestamp = datetime.now(UTC)

                logger.debug(
                    "Refreshed canonical registry cache",
                    entity_count=len(entities),
                    predicate_count=len(predicates),
                )
        except asyncpg.PostgresError as e:
            logger.error("Failed to refresh canonical registry cache", error=str(e))
            raise CanonicalRegistryError(f"Failed to refresh cache: {e}") from e


async def _ensure_cache_valid() -> None:
    """Ensure cache is valid, refreshing if stale or empty."""
    global _cache_timestamp
    now = datetime.now(UTC)
    needs_refresh = (
        _entities_cache is None
        or _entity_variants_cache is None
        or _predicates_cache is None
        or _cache_timestamp is None
        or (now - _cache_timestamp).total_seconds() > CACHE_TTL_SECONDS
    )
    if needs_refresh:
        await _refresh_cache()


async def get_canonical_entity(
    variant: str,
    default_on_missing: bool = False,
) -> str:
    """Look up the canonical form for an entity variant.

    Args:
        variant: The entity variant to look up (e.g., "mom", "bf")
        default_on_missing: If True, return the variant itself when not found

    Returns:
        The canonical form of the entity

    Raises:
        EntityNotFoundError: If variant not found and default_on_missing is False
    """
    await _ensure_cache_valid()

    if _entity_variants_cache is None:
        raise CanonicalRegistryError("Entity cache not initialized")

    variant_lower = variant.lower()
    if variant_lower in _entity_variants_cache:
        return _entity_variants_cache[variant_lower]

    if default_on_missing:
        return variant

    raise EntityNotFoundError(f"Entity variant not found: {variant}")


async def get_canonical_entities(variant: list[str]) -> list[str]:
    """Look up canonical forms for multiple entity variants.

    Args:
        variant: List of entity variants to look up

    Returns:
        List of canonical forms (order preserved, missing items dropped)
    """
    results = []
    for v in variant:
        try:
            canonical = await get_canonical_entity(v)
            results.append(canonical)
        except EntityNotFoundError:
            logger.debug("Skipping unknown entity variant", variant=v)
    return results


async def get_canonical_entity_with_category(variant: str) -> tuple[str, str | None]:
    """Look up the canonical form and category for an entity variant.

    Args:
        variant: The entity variant to look up

    Returns:
        Tuple of (canonical_form, category) where category may be None

    Raises:
        EntityNotFoundError: If variant not found
    """
    await _ensure_cache_valid()

    if _entity_variants_cache is None or _entities_cache is None:
        raise CanonicalRegistryError("Entity cache not initialized")

    variant_lower = variant.lower()
    if variant_lower in _entity_variants_cache:
        canonical = _entity_variants_cache[variant_lower]
        _, category = _entities_cache[canonical]
        return canonical, category

    raise EntityNotFoundError(f"Entity variant not found: {variant}")


async def get_canonical_predicate(variant: str) -> str:
    """Look up the canonical form for a predicate variant.

    Args:
        variant: The predicate variant to look up (e.g., "lives in", "works at")

    Returns:
        The canonical form of the predicate

    Raises:
        PredicateNotFoundError: If variant not found
    """
    await _ensure_cache_valid()

    if _predicates_cache is None:
        raise CanonicalRegistryError("Predicate cache not initialized")

    variant_lower = variant.lower()
    if variant_lower in _predicates_cache:
        return _predicates_cache[variant_lower]

    raise PredicateNotFoundError(f"Predicate variant not found: {variant}")


async def get_all_canonical_entities() -> dict[str, tuple[set[str], str | None]]:
    """Get all canonical entities and their variants.

    Returns:
        Dict mapping canonical form to (variants set, category)
    """
    await _ensure_cache_valid()
    if _entities_cache is None:
        return {}
    return _entities_cache


async def get_all_canonical_predicates() -> set[str]:
    """Get all canonical predicate forms.

    Returns:
        Set of canonical predicate forms
    """
    await _ensure_cache_valid()
    if _predicates_cache is None:
        return set()
    return set(_predicates_cache.values())


async def list_entities_by_category(category: str) -> list[str]:
    """List all canonical entities in a specific category.

    Args:
        category: The category to filter by

    Returns:
        List of canonical entity forms in the category
    """
    await _ensure_cache_valid()

    if _entities_cache is None:
        return []

    return [
        canonical for canonical, (_, cat) in _entities_cache.items() if cat == category
    ]


async def seed_canonical_entities(
    entities: list[dict[str, str | list[str] | None]],
) -> None:
    """Seed the canonical_entities table with initial data.

    Args:
        entities: List of {"canonical": str, "variants": list[str], "category": str | None}
    """
    async with db_pool.pool.acquire() as conn, conn.transaction():
        for entity in entities:
            canonical = entity["canonical"]
            variants = entity.get("variants", [])
            category = entity.get("category")

            await conn.execute(
                """
                INSERT INTO canonical_entities (canonical_form, variants, category, updated_at)
                VALUES ($1, $2, $3, now())
                ON CONFLICT (canonical_form)
                DO UPDATE SET
                    variants = EXCLUDED.variants,
                    category = EXCLUDED.category,
                    updated_at = now()
                """,
                canonical,
                list(variants) if variants else [],
                category,
            )

        logger.info("Seeded canonical entities", count=len(entities))


async def seed_canonical_predicates(
    predicates: list[dict[str, str | list[str]]],
) -> None:
    """Seed the canonical_predicates table with initial data.

    Args:
        predicates: List of {"canonical": str, "variants": list[str]}
    """
    async with db_pool.pool.acquire() as conn, conn.transaction():
        for pred in predicates:
            canonical = pred["canonical"]
            variants = pred.get("variants", [])

            await conn.execute(
                """
                INSERT INTO canonical_predicates (canonical_form, variants, updated_at)
                VALUES ($1, $2, now())
                ON CONFLICT (canonical_form)
                DO UPDATE SET
                    variants = EXCLUDED.variants,
                    updated_at = now()
                """,
                canonical,
                list(variants) if variants else [],
            )

        logger.info("Seeded canonical predicates", count=len(predicates))


async def invalidate_cache() -> None:
    """Invalidate the in-memory cache, forcing refresh on next access."""
    global _entities_cache, _entity_variants_cache, _predicates_cache, _cache_timestamp
    _entities_cache = None
    _entity_variants_cache = None
    _predicates_cache = None
    _cache_timestamp = None
    logger.debug("Invalidated canonical registry cache")


async def get_canonical_entities_stream(
    batch_size: int = 100,
) -> AsyncGenerator[dict[str, tuple[set[str], str | None]], None]:
    """Stream all canonical entities in batches.

    Useful for bulk processing without loading everything into memory.

    Args:
        batch_size: Number of entities to fetch per batch

    Yields:
        Dict mapping canonical form to (variants set, category)
    """
    async with db_pool.pool.acquire() as conn:
        async with conn.cursor(
            "SELECT canonical_form, variants, category FROM canonical_entities ORDER BY canonical_form"
        ) as cursor:
            while True:
                rows = await cursor.fetch(batch_size)
                if not rows:
                    break
                for row in rows:
                    canonical = row["canonical_form"]
                    variants = set(row["variants"]) if row["variants"] else set()
                    yield {canonical: (variants, row["category"])}
