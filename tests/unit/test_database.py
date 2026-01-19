"""Comprehensive unit tests for lattice/utils/database.py."""

from datetime import datetime, timezone
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.utils.config import get_config
from lattice.utils.database import (
    DatabasePool,
    get_next_check_at,
    get_system_metrics,
    get_user_timezone,
    set_next_check_at,
    set_system_metrics,
)


class TestDatabasePoolInitialization:
    """Tests for DatabasePool initialization and lifecycle."""

    @pytest.fixture(autouse=True)
    def mock_config(self) -> Generator:
        """Mock config for database tests."""
        config = get_config(reload=True)
        original_url = config.database_url
        original_min = config.db_pool_min_size
        original_max = config.db_pool_max_size
        config.database_url = "postgresql://test:test@localhost/test"
        config.db_pool_min_size = 2
        config.db_pool_max_size = 5
        yield config
        config.database_url = original_url
        config.db_pool_min_size = original_min
        config.db_pool_max_size = original_max

    def test_initial_state_uninitialized(self) -> None:
        """Test that DatabasePool starts in uninitialized state."""
        pool = DatabasePool()
        assert pool._pool is None
        assert pool.is_initialized() is False

    @pytest.mark.asyncio
    async def test_initialize_raises_on_missing_database_url(self) -> None:
        """Test that initialize raises ValueError when DATABASE_URL is not set."""
        pool = DatabasePool()
        config = get_config()
        original_url = config.database_url
        config.database_url = ""

        try:
            with pytest.raises(
                ValueError, match="DATABASE_URL environment variable not set"
            ):
                await pool.initialize()
        finally:
            config.database_url = original_url

    @pytest.mark.asyncio
    async def test_initialize_success(self) -> None:
        """Test successful pool initialization with default settings."""
        pool = DatabasePool()
        mock_pool = AsyncMock()

        with patch(
            "asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)
        ) as mock_create:
            await pool.initialize()

            mock_create.assert_called_once_with(
                "postgresql://test:test@localhost/test",
                min_size=2,
                max_size=5,
                command_timeout=30,
            )
            assert pool.is_initialized() is True
            assert pool._pool == mock_pool

    @pytest.mark.asyncio
    async def test_initialize_with_custom_pool_sizes(self) -> None:
        """Test initialization with custom pool size configuration."""
        pool = DatabasePool()
        mock_pool = AsyncMock()
        config = get_config()
        config.db_pool_min_size = 3
        config.db_pool_max_size = 10

        with patch(
            "asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)
        ) as mock_create:
            await pool.initialize()

            mock_create.assert_called_once_with(
                "postgresql://test:test@localhost/test",
                min_size=3,
                max_size=10,
                command_timeout=30,
            )
            assert pool.is_initialized() is True

    @pytest.mark.asyncio
    async def test_initialize_replaces_existing_pool(self) -> None:
        """Test that re-initializing replaces the existing pool."""
        pool = DatabasePool()
        first_mock_pool = AsyncMock()
        second_mock_pool = AsyncMock()

        with patch(
            "asyncpg.create_pool",
            new=AsyncMock(side_effect=[first_mock_pool, second_mock_pool]),
        ):
            await pool.initialize()
            assert pool._pool == first_mock_pool

            await pool.initialize()
            assert pool._pool == second_mock_pool

    @pytest.mark.asyncio
    async def test_initialize_handles_pool_creation_failure(self) -> None:
        """Test that pool initialization failure is properly propagated."""
        pool = DatabasePool()

        with patch(
            "asyncpg.create_pool",
            new=AsyncMock(side_effect=ConnectionError("Connection refused")),
        ):
            with pytest.raises(ConnectionError, match="Connection refused"):
                await pool.initialize()

        assert pool.is_initialized() is False


class TestDatabasePoolClose:
    """Tests for DatabasePool close method."""

    @pytest.fixture(autouse=True)
    def mock_config(self) -> Generator:
        """Mock config for database tests."""
        config = get_config(reload=True)
        original_url = config.database_url
        original_min = config.db_pool_min_size
        original_max = config.db_pool_max_size
        config.database_url = "postgresql://test:test@localhost/test"
        config.db_pool_min_size = 2
        config.db_pool_max_size = 5
        yield config
        config.database_url = original_url
        config.db_pool_min_size = original_min
        config.db_pool_max_size = original_max

    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self) -> None:
        """Test that close is safe when pool is not initialized."""
        pool = DatabasePool()
        await pool.close()
        assert pool._pool is None
        assert pool.is_initialized() is False

    @pytest.mark.asyncio
    async def test_close_when_initialized(self) -> None:
        """Test that close properly closes and resets the pool."""
        pool = DatabasePool()
        mock_pool = AsyncMock()
        pool._pool = mock_pool

        await pool.close()

        mock_pool.close.assert_called_once()
        assert pool._pool is None
        assert pool.is_initialized() is False

    @pytest.mark.asyncio
    async def test_close_handles_close_failure(self) -> None:
        """Test that close propagates pool close failure."""
        pool = DatabasePool()
        mock_pool = AsyncMock()
        mock_pool.close = AsyncMock(side_effect=OSError("Connection reset by peer"))
        pool._pool = mock_pool

        with pytest.raises(OSError, match="Connection reset by peer"):
            await pool.close()

        # Pool should still be set because the exception was raised before setting to None
        assert pool._pool == mock_pool
        assert pool.is_initialized() is True


class TestDatabasePoolProperty:
    """Tests for DatabasePool pool property."""

    @pytest.mark.asyncio
    async def test_pool_property_raises_when_not_initialized(self) -> None:
        """Test that pool property raises RuntimeError when not initialized."""
        pool = DatabasePool()

        with pytest.raises(RuntimeError, match="Database pool not initialized"):
            _ = pool.pool

    @pytest.mark.asyncio
    async def test_pool_property_returns_pool_when_initialized(self) -> None:
        """Test that pool property returns the pool when initialized."""
        pool = DatabasePool()
        mock_pool = AsyncMock()
        pool._pool = mock_pool

        result = pool.pool

        assert result == mock_pool


class TestDatabasePoolSystemMetrics:
    """Tests for DatabasePool system metrics methods."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock database pool with connection."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return mock_pool

    @pytest.mark.asyncio
    async def test_get_system_metrics_success(self, mock_pool: MagicMock) -> None:
        """Test successful retrieval of system metric."""
        db_pool = DatabasePool()
        db_pool._pool = mock_pool
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval = AsyncMock(return_value="test_value")

        result = await db_pool.get_system_metrics("test_key")

        assert result == "test_value"
        mock_conn.fetchval.assert_called_once_with(
            "SELECT metric_value FROM system_metrics WHERE metric_key = $1",
            "test_key",
        )

    @pytest.mark.asyncio
    async def test_get_system_metrics_not_found(self, mock_pool: MagicMock) -> None:
        """Test get_system_metrics returns None when key not found."""
        db_pool = DatabasePool()
        db_pool._pool = mock_pool
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval = AsyncMock(return_value=None)

        result = await db_pool.get_system_metrics("nonexistent_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_system_metrics_handles_numeric_value_as_string(self) -> None:
        """Test that set_system_metrics properly handles numeric values converted to string."""
        db_pool = DatabasePool()
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        db_pool._pool = mock_pool

        await db_pool.set_system_metrics("numeric_key", "42")

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert call_args[0][1] == "numeric_key"
        assert call_args[0][2] == "42"


class TestDatabasePoolNextCheckAt:
    """Tests for DatabasePool next_check_at methods."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock database pool with connection."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return mock_pool

    @pytest.mark.asyncio
    async def test_get_next_check_at_with_value(self, mock_pool: MagicMock) -> None:
        """Test get_next_check_at parses ISO datetime with Z suffix."""
        db_pool = DatabasePool()
        db_pool._pool = mock_pool
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval = AsyncMock(return_value="2026-01-15T10:30:00Z")

        result = await db_pool.get_next_check_at()

        expected = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert result == expected

    @pytest.mark.asyncio
    async def test_get_next_check_at_with_plus_suffix(
        self, mock_pool: MagicMock
    ) -> None:
        """Test get_next_check_at handles +00:00 suffix."""
        db_pool = DatabasePool()
        db_pool._pool = mock_pool
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval = AsyncMock(return_value="2026-01-15T10:30:00+00:00")

        result = await db_pool.get_next_check_at()

        expected = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert result == expected

    @pytest.mark.asyncio
    async def test_get_next_check_at_returns_none_when_not_set(
        self, mock_pool: MagicMock
    ) -> None:
        """Test get_next_check_at returns None when key not found."""
        db_pool = DatabasePool()
        db_pool._pool = mock_pool
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval = AsyncMock(return_value=None)

        result = await db_pool.get_next_check_at()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_next_check_at_handles_invalid_format(
        self, mock_pool: MagicMock
    ) -> None:
        """Test get_next_check_at raises ValueError for invalid datetime format."""
        db_pool = DatabasePool()
        db_pool._pool = mock_pool
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval = AsyncMock(return_value="invalid-datetime-format")

        with pytest.raises(ValueError):
            await db_pool.get_next_check_at()

    @pytest.mark.asyncio
    async def test_set_next_check_at_success(self, mock_pool: MagicMock) -> None:
        """Test set_next_check_at stores ISO formatted datetime."""
        db_pool = DatabasePool()
        db_pool._pool = mock_pool
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute = AsyncMock()
        test_dt = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

        await db_pool.set_next_check_at(test_dt)

        mock_conn.execute.assert_called_once()


class TestStandaloneFunctions:
    """Tests for standalone database utility functions."""

    @pytest.mark.asyncio
    async def test_get_system_metrics_delegates_to_db_pool(self) -> None:
        """Test get_system_metrics delegates to DatabasePool method."""
        mock_db_pool = AsyncMock()
        mock_db_pool.get_system_metrics = AsyncMock(return_value="test_value")

        result = await get_system_metrics("test_key", db_pool=mock_db_pool)

        assert result == "test_value"
        mock_db_pool.get_system_metrics.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_get_system_metrics_handles_none(self) -> None:
        """Test get_system_metrics handles None return value."""
        mock_db_pool = AsyncMock()
        mock_db_pool.get_system_metrics = AsyncMock(return_value=None)

        result = await get_system_metrics("nonexistent", db_pool=mock_db_pool)

        assert result is None

    @pytest.mark.asyncio
    async def test_set_system_metrics_delegates_to_db_pool(self) -> None:
        """Test set_system_metrics delegates to DatabasePool method."""
        mock_db_pool = AsyncMock()
        mock_db_pool.set_system_metrics = AsyncMock()

        await set_system_metrics("test_key", "test_value", db_pool=mock_db_pool)

        mock_db_pool.set_system_metrics.assert_called_once_with(
            "test_key", "test_value"
        )

    @pytest.mark.asyncio
    async def test_get_next_check_at_delegates_to_db_pool(self) -> None:
        """Test get_next_check_at delegates to DatabasePool method."""
        expected_dt = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        mock_db_pool = AsyncMock()
        mock_db_pool.get_next_check_at = AsyncMock(return_value=expected_dt)

        result = await get_next_check_at(db_pool=mock_db_pool)

        assert result == expected_dt

    @pytest.mark.asyncio
    async def test_get_next_check_at_handles_none(self) -> None:
        """Test get_next_check_at handles None return value."""
        mock_db_pool = AsyncMock()
        mock_db_pool.get_next_check_at = AsyncMock(return_value=None)

        result = await get_next_check_at(db_pool=mock_db_pool)

        assert result is None

    @pytest.mark.asyncio
    async def test_set_next_check_at_delegates_to_db_pool(self) -> None:
        """Test set_next_check_at delegates to DatabasePool method."""
        mock_db_pool = AsyncMock()
        mock_db_pool.set_next_check_at = AsyncMock()
        test_dt = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

        await set_next_check_at(test_dt, db_pool=mock_db_pool)

        mock_db_pool.set_next_check_at.assert_called_once_with(test_dt)


class TestGetUserTimezone:
    """Tests for get_user_timezone function."""

    @pytest.fixture(autouse=True)
    def reset_timezone_cache(self) -> Generator[None, None, None]:
        """Reset the timezone cache before each test."""
        from lattice.utils import database

        original = database._user_timezone_cache
        database._user_timezone_cache = None
        yield
        database._user_timezone_cache = original

    @pytest.fixture
    def mock_db_pool_with_conn(self) -> MagicMock:
        """Create a mock database pool with connection that returns proper values."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )
        mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return mock_pool

    @pytest.mark.asyncio
    async def test_get_user_timezone_with_cached_value(self) -> None:
        """Test get_user_timezone returns cached value when available."""
        from lattice.utils import database

        database._user_timezone_cache = "America/Los_Angeles"

        mock_db_pool = MagicMock()

        result = await get_user_timezone(db_pool=mock_db_pool)

        assert result == "America/Los_Angeles"
        # Verify pool was never accessed since cache was set
        mock_db_pool.pool.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_user_timezone_retrieves_from_database(
        self, mock_db_pool_with_conn: MagicMock
    ) -> None:
        """Test get_user_timezone queries database when cache is empty."""
        mock_conn = (
            mock_db_pool_with_conn.pool.acquire.return_value.__aenter__.return_value
        )
        mock_conn.fetchrow = AsyncMock(return_value={"object": "America/New_York"})

        result = await get_user_timezone(db_pool=mock_db_pool_with_conn)

        assert result == "America/New_York"
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_timezone_caches_valid_value(
        self, mock_db_pool_with_conn: MagicMock
    ) -> None:
        """Test that valid timezone is cached for subsequent calls."""
        from lattice.utils import database

        mock_conn = (
            mock_db_pool_with_conn.pool.acquire.return_value.__aenter__.return_value
        )
        mock_conn.fetchrow = AsyncMock(return_value={"object": "Europe/London"})

        result = await get_user_timezone(db_pool=mock_db_pool_with_conn)

        assert result == "Europe/London"
        assert database._user_timezone_cache == "Europe/London"

    @pytest.mark.asyncio
    async def test_get_user_timezone_defaults_to_utc_when_not_found(
        self, mock_db_pool_with_conn: MagicMock
    ) -> None:
        """Test get_user_timezone defaults to UTC when no timezone is stored."""
        mock_conn = (
            mock_db_pool_with_conn.pool.acquire.return_value.__aenter__.return_value
        )
        mock_conn.fetchrow = AsyncMock(return_value=None)

        result = await get_user_timezone(db_pool=mock_db_pool_with_conn)

        assert result == "UTC"

    @pytest.mark.asyncio
    async def test_get_user_timezone_defaults_to_utc_for_invalid_timezone(
        self, mock_db_pool_with_conn: MagicMock
    ) -> None:
        """Test get_user_timezone defaults to UTC for invalid timezone strings."""
        mock_conn = (
            mock_db_pool_with_conn.pool.acquire.return_value.__aenter__.return_value
        )
        mock_conn.fetchrow = AsyncMock(return_value={"object": "Invalid/Timezone_123"})

        result = await get_user_timezone(db_pool=mock_db_pool_with_conn)

        assert result == "UTC"

    @pytest.mark.asyncio
    async def test_get_user_timezone_uses_cache_on_subsequent_calls(
        self, mock_db_pool_with_conn: MagicMock
    ) -> None:
        """Test that cache is used on second call, avoiding database query."""
        from lattice.utils import database

        database._user_timezone_cache = "Asia/Tokyo"

        result = await get_user_timezone(db_pool=mock_db_pool_with_conn)

        assert result == "Asia/Tokyo"
        mock_db_pool_with_conn.pool.acquire.assert_not_called()


class TestConnectionHandling:
    """Tests for connection pool handling and edge cases."""

    @pytest.fixture(autouse=True)
    def mock_config(self) -> Generator:
        """Mock config for database tests."""
        config = get_config(reload=True)
        original_url = config.database_url
        original_min = config.db_pool_min_size
        original_max = config.db_pool_max_size
        config.database_url = "postgresql://test:test@localhost/test"
        config.db_pool_min_size = 2
        config.db_pool_max_size = 5
        yield config
        config.database_url = original_url
        config.db_pool_min_size = original_min
        config.db_pool_max_size = original_max

    @pytest.mark.asyncio
    async def test_get_system_metrics_handles_connection_error(self) -> None:
        """Test get_system_metrics handles connection errors gracefully."""
        db_pool = DatabasePool()
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=OSError("Connection lost"))
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        db_pool._pool = mock_pool

        with pytest.raises(OSError, match="Connection lost"):
            await db_pool.get_system_metrics("test_key")

    @pytest.mark.asyncio
    async def test_set_system_metrics_handles_connection_error(self) -> None:
        """Test set_system_metrics handles connection errors gracefully."""
        db_pool = DatabasePool()
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=OSError("Connection lost"))
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        db_pool._pool = mock_pool

        with pytest.raises(OSError, match="Connection lost"):
            await db_pool.set_system_metrics("test_key", "test_value")


class TestTransactionHandling:
    """Tests for transaction handling in database operations."""

    @pytest.fixture
    def mock_pool_with_transaction(self) -> MagicMock:
        """Create a mock database pool that simulates transaction behavior."""
        mock_pool = MagicMock()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetchval = AsyncMock()
        mock_conn.fetchrow = AsyncMock()

        mock_ctx_manager = MagicMock()
        mock_ctx_manager.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx_manager.__aexit__ = AsyncMock(return_value=None)

        mock_pool.acquire.return_value = mock_ctx_manager
        return mock_pool

    @pytest.mark.asyncio
    async def test_system_metrics_use_context_manager(
        self, mock_pool_with_transaction: MagicMock
    ) -> None:
        """Test that system metrics operations use context manager for connections."""
        db_pool = DatabasePool()
        db_pool._pool = mock_pool_with_transaction

        await db_pool.get_system_metrics("test_key")

        mock_pool_with_transaction.acquire.assert_called_once()
        mock_ctx = mock_pool_with_transaction.acquire.return_value
        mock_ctx.__aenter__.assert_called_once()
        mock_ctx.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_system_metrics_uses_upsert(
        self, mock_pool_with_transaction: MagicMock
    ) -> None:
        """Test that set_system_metrics uses INSERT ... ON CONFLICT DO UPDATE."""
        db_pool = DatabasePool()
        db_pool._pool = mock_pool_with_transaction

        await db_pool.set_system_metrics("test_key", "test_value")

        call_args = mock_pool_with_transaction.acquire.return_value.__aenter__.return_value.execute.call_args
        sql_query = call_args[0][0]
        assert "ON CONFLICT" in sql_query
        assert "DO UPDATE SET" in sql_query


class TestPoolExhaustionHandling:
    """Tests for connection pool exhaustion scenarios."""

    @pytest.fixture(autouse=True)
    def mock_config(self) -> Generator:
        """Mock config for database tests."""
        config = get_config(reload=True)
        original_url = config.database_url
        original_min = config.db_pool_min_size
        original_max = config.db_pool_max_size
        config.database_url = "postgresql://test:test@localhost/test"
        config.db_pool_min_size = 2
        config.db_pool_max_size = 5
        yield config
        config.database_url = original_url
        config.db_pool_min_size = original_min
        config.db_pool_max_size = original_max

    @pytest.mark.asyncio
    async def test_initialize_with_zero_min_size(self, mock_config) -> None:
        """Test initialization with min_size=0 is handled correctly."""
        pool = DatabasePool()
        mock_pool = AsyncMock()
        mock_config.db_pool_min_size = 0

        with patch(
            "asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)
        ) as mock_create:
            await pool.initialize()

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["min_size"] == 0

    @pytest.mark.asyncio
    async def test_pool_property_with_none_pool_after_close(self) -> None:
        """Test that pool property raises error after pool is closed."""
        pool = DatabasePool()
        mock_pool = AsyncMock()
        pool._pool = mock_pool

        await pool.close()

        with pytest.raises(RuntimeError, match="Database pool not initialized"):
            _ = pool.pool


class TestEdgeCases:
    """Edge case tests for database operations."""

    @pytest.fixture(autouse=True)
    def mock_config(self) -> Generator:
        """Mock config for database tests."""
        config = get_config(reload=True)
        original_url = config.database_url
        original_min = config.db_pool_min_size
        original_max = config.db_pool_max_size
        config.database_url = "postgresql://test:test@localhost/test"
        config.db_pool_min_size = 2
        config.db_pool_max_size = 5
        yield config
        config.database_url = original_url
        config.db_pool_min_size = original_min
        config.db_pool_max_size = original_max

    @pytest.mark.asyncio
    async def test_get_next_check_at_with_microseconds(self) -> None:
        """Test get_next_check_at handles microseconds in datetime."""
        db_pool = DatabasePool()
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.fetchval = AsyncMock(return_value="2026-01-15T10:30:00.123456Z")
        db_pool._pool = mock_pool

        result = await db_pool.get_next_check_at()

        assert result is not None
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 15

    @pytest.mark.asyncio
    async def test_get_system_metrics_with_special_characters(self) -> None:
        """Test get_system_metrics handles special characters in key/value."""
        db_pool = DatabasePool()
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.fetchval = AsyncMock(
            return_value="value with 'quotes' and \"double quotes\""
        )
        db_pool._pool = mock_pool

        result = await db_pool.get_system_metrics("key with spaces & special chars")

        assert result == "value with 'quotes' and \"double quotes\""

    @pytest.mark.asyncio
    async def test_get_user_timezone_with_unicode_timezone(self) -> None:
        """Test get_user_timezone handles timezone names correctly."""

        db_pool = MagicMock()
        mock_conn = AsyncMock()
        db_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db_pool.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.fetchrow = AsyncMock(return_value={"object": "UTC"})

        result = await get_user_timezone(db_pool=db_pool)

        assert result == "UTC"

    @pytest.mark.asyncio
    async def test_close_twice_is_idempotent(self) -> None:
        """Test that calling close twice is safe and idempotent."""
        pool = DatabasePool()
        mock_pool = AsyncMock()
        pool._pool = mock_pool

        await pool.close()
        await pool.close()

        assert mock_pool.close.call_count == 1
        assert pool._pool is None
