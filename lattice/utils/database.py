"""Database utilities for Lattice.

Provides connection pooling and common database operations.
"""

import os
from datetime import datetime

import asyncpg
import pgvector.asyncpg
import structlog


logger = structlog.get_logger(__name__)


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
            setup=setup_connection,
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


async def setup_connection(conn: asyncpg.Connection) -> None:
    """Set up a database connection with pgvector type support.

    Args:
        conn: The asyncpg connection to set up
    """
    await pgvector.asyncpg.register_vector(conn)


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


db_pool = DatabasePool()
