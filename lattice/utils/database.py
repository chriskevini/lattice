"""Database connection and pool management.

Provides async PostgreSQL connection pooling.
"""

from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfoNotFoundError

import asyncpg
import structlog

from lattice.utils.config import config


if TYPE_CHECKING:
    pass


logger = structlog.get_logger(__name__)

_user_timezone_cache: str | None = None


class DatabasePool:
    """Manages asyncpg connection pool for the database."""

    def __init__(self) -> None:
        """Initialize the database pool manager."""
        self._pool: "asyncpg.Pool | None" = None

    async def initialize(self) -> None:
        """Create the connection pool."""
        database_url = config.database_url
        if not database_url:
            msg = "DATABASE_URL environment variable not set"
            raise ValueError(msg)

        min_size = config.db_pool_min_size
        max_size = config.db_pool_max_size

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
    def pool(self) -> "asyncpg.Pool":
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

    async def get_system_health(self, key: str) -> str | None:
        """Get a value from the system_health table.

        Args:
            key: The metric key to retrieve

        Returns:
            The metric value, or None if not found
        """
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT metric_value FROM system_health WHERE metric_key = $1",
                key,
            )

    async def set_system_health(self, key: str, value: str) -> None:
        """Set a value in the system_health table.

        Args:
            key: The metric key to set
            value: The value to store
        """
        async with self.pool.acquire() as conn:
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

    async def get_next_check_at(self) -> datetime | None:
        """Get the next proactive check timestamp.

        Returns:
            The next check datetime, or None if not set
        """
        value = await self.get_system_health("next_check_at")
        if value:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return None

    async def set_next_check_at(self, dt: datetime) -> None:
        """Set the next proactive check timestamp.

        Args:
            dt: The datetime for next check
        """
        await self.set_system_health("next_check_at", dt.isoformat())


async def get_user_timezone(db_pool: DatabasePool) -> str:
    """Get user timezone from semantic memory or cache.

    Args:
        db_pool: Database pool for DI

    Returns:
        IANA timezone string (e.g., 'America/New_York'), defaults to 'UTC'
    """
    global _user_timezone_cache
    if _user_timezone_cache:
        return _user_timezone_cache

    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT object FROM semantic_memories WHERE subject = 'User' AND predicate = 'lives in timezone' ORDER BY created_at DESC LIMIT 1"
        )
        if row:
            tz = row["object"]
            try:
                from zoneinfo import ZoneInfo

                ZoneInfo(tz)
                _user_timezone_cache = tz
                return tz
            except ZoneInfoNotFoundError:
                pass

    _user_timezone_cache = "UTC"
    return "UTC"


async def get_system_health(key: str, db_pool: DatabasePool) -> str | None:
    """Get a value from the system_health table.

    Args:
        key: The metric key to retrieve
        db_pool: Database pool (required for DI)

    Returns:
        The metric value, or None if not found
    """
    return await db_pool.get_system_health(key)


async def set_system_health(key: str, value: str, db_pool: DatabasePool) -> None:
    """Set a value in the system_health table.

    Args:
        key: The metric key to set
        value: The value to store
        db_pool: Database pool (required for DI)
    """
    await db_pool.set_system_health(key, value)


async def get_next_check_at(db_pool: DatabasePool) -> datetime | None:
    """Get the next proactive check timestamp.

    Args:
        db_pool: Database pool (required for DI)

    Returns:
        The next check datetime, or None if not set
    """
    return await db_pool.get_next_check_at()


async def set_next_check_at(dt: datetime, db_pool: DatabasePool) -> None:
    """Set the next proactive check timestamp.

    Args:
        dt: The datetime for next check
        db_pool: Database pool (required for DI)
    """
    await db_pool.set_next_check_at(dt)
