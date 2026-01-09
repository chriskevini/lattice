"""Database connection and pool management.

Provides async PostgreSQL connection pooling.
"""

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfoNotFoundError

import asyncpg
import structlog


logger = structlog.get_logger(__name__)

CANONICAL_CACHE_TTL_SECONDS = 300  # 5 minutes


class DatabasePool:
    """Manages asyncpg connection pool for the database."""

    def __init__(self) -> None:
        """Initialize the database pool manager."""
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Create the connection pool."""
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            msg = "DATABASE_URL environment variable not set"
            raise ValueError(msg)

        min_size = int(os.getenv("DB_POOL_MIN_SIZE", "2"))
        max_size = int(os.getenv("DB_POOL_MAX_SIZE", "5"))

        logger.info(
            "Initializing database pool",
            min_size=min_size,
            max_size=max_size,
        )

        self._pool = await asyncpg.create_pool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            command_timeout=30,
        )

        logger.info("Database pool initialized")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed")

    def is_initialized(self) -> bool:
        """Check if the pool is initialized.

        Returns:
            True if pool is initialized, False otherwise
        """
        return self._pool is not None

    @property
    def pool(self) -> asyncpg.Pool:
        """Get the connection pool.

        Returns:
            The asyncpg connection pool

        Raises:
            RuntimeError: If pool is not initialized
        """
        if not self._pool:
            msg = "Database pool not initialized. Call initialize() first."
            raise RuntimeError(msg)
        return self._pool


async def get_system_health(key: str) -> str | None:
    """Get a value from the system_health table.

    Args:
        key: The metric key to retrieve

    Returns:
        The metric value, or None if not found
    """
    async with db_pool.pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT metric_value FROM system_health WHERE metric_key = $1",
            key,
        )


async def set_system_health(key: str, value: str) -> None:
    """Set a value in the system_health table.

    Args:
        key: The metric key to set
        value: The value to store
    """
    async with db_pool.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO system_health (metric_key, metric_value, recorded_at)
            VALUES ($1, $2, now())
            ON CONFLICT (metric_key)
            DO UPDATE SET metric_value = EXCLUDED.metric_value, recorded_at = now()
            """,
            key,
            str(value),
        )


async def get_next_check_at() -> datetime | None:
    """Get the next proactive check timestamp.

    Returns:
        The next check datetime, or None if not set
    """
    value = await get_system_health("next_check_at")
    if value:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


async def set_next_check_at(dt: datetime) -> None:
    """Set the next proactive check timestamp.

    Args:
        dt: The datetime for next check
    """
    await set_system_health("next_check_at", dt.isoformat())


async def get_user_timezone() -> str:
    """Get system-wide user timezone.

    Returns:
        IANA timezone string (e.g., 'America/New_York'), defaults to 'UTC'
    """
    return await get_system_health("user_timezone") or "UTC"


async def set_user_timezone(timezone: str) -> None:
    """Set system-wide user timezone.

    Args:
        timezone: IANA timezone string (e.g., 'America/New_York')

    Raises:
        ValueError: If timezone is invalid
    """
    from zoneinfo import ZoneInfo

    # Validate timezone
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as e:
        msg = f"Invalid timezone: {timezone}"
        raise ValueError(msg) from e

    await set_system_health("user_timezone", timezone)
    logger.info("System timezone updated", timezone=timezone)


_entities_cache: list[str] | None = None
_entities_cache_timestamp: float = 0.0
_predicates_cache: list[str] | None = None
_predicates_cache_timestamp: float = 0.0


async def get_canonical_entities() -> list[str]:
    """Fetch all canonical entity names from entities table.

    Returns:
        List of entity names sorted by creation date (newest first)
    """
    global _entities_cache, _entities_cache_timestamp

    current_time = time.time()

    if (
        _entities_cache is not None
        and (current_time - _entities_cache_timestamp) < CANONICAL_CACHE_TTL_SECONDS
    ):
        return _entities_cache

    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch("SELECT name FROM entities ORDER BY created_at DESC")
        _entities_cache = [row["name"] for row in rows]
        _entities_cache_timestamp = current_time

    logger.debug("Fetched canonical entities from database", count=len(_entities_cache))
    return _entities_cache


async def get_canonical_predicates() -> list[str]:
    """Fetch all canonical predicate names from predicates table.

    Returns:
        List of predicate names sorted by creation date (newest first)
    """
    global _predicates_cache, _predicates_cache_timestamp

    current_time = time.time()

    if (
        _predicates_cache is not None
        and (current_time - _predicates_cache_timestamp) < CANONICAL_CACHE_TTL_SECONDS
    ):
        return _predicates_cache

    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch("SELECT name FROM predicates ORDER BY created_at DESC")
        _predicates_cache = [row["name"] for row in rows]
        _predicates_cache_timestamp = current_time

    logger.debug(
        "Fetched canonical predicates from database", count=len(_predicates_cache)
    )
    return _predicates_cache


def clear_canonical_cache() -> None:
    """Clear the canonical entities and predicates caches.

    Useful after batch operations that add new canonical forms.
    """
    global _entities_cache, _predicates_cache
    _entities_cache = None
    _predicates_cache = None
    logger.debug("Cleared canonical caches")


db_pool = DatabasePool()
