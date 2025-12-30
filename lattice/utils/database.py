"""Database utilities for Lattice.

Provides connection pooling and common database operations.
"""

import os

import asyncpg
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
        )

        logger.info("Database pool initialized")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Database pool closed")

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


# Global database pool instance
db_pool = DatabasePool()
