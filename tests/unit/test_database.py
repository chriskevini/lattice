"""Unit tests for database module."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from lattice.utils.database import (
    DatabasePool,
    get_next_check_at,
    get_system_metrics,
    get_user_timezone,
    set_next_check_at,
    set_system_metrics,
)


class TestDatabasePool:
    """Tests for DatabasePool class."""

    @pytest.mark.asyncio
    async def test_initialize_creates_pool_with_config(self) -> None:
        """Test that initialize creates a connection pool with configuration."""
        with patch("lattice.utils.database.asyncpg.create_pool") as mock_create_pool:
            mock_pool = MagicMock()
            mock_create_pool.return_value = AsyncMock(return_value=mock_pool)()

            with patch("lattice.utils.database.config") as mock_config:
                mock_config.database_url = "postgresql://test:test@localhost/test"
                mock_config.db_pool_min_size = 2
                mock_config.db_pool_max_size = 5

                db_pool = DatabasePool()
                await db_pool.initialize()

                mock_create_pool.assert_called_once_with(
                    "postgresql://test:test@localhost/test",
                    min_size=2,
                    max_size=5,
                    command_timeout=30,
                )
                assert db_pool._pool == mock_pool
                assert db_pool.is_initialized()

    @pytest.mark.asyncio
    async def test_initialize_raises_when_database_url_not_set(self) -> None:
        """Test that initialize raises ValueError when DATABASE_URL is not set."""
        with patch("lattice.utils.database.config") as mock_config:
            mock_config.database_url = None

            db_pool = DatabasePool()

            with pytest.raises(
                ValueError, match="DATABASE_URL environment variable not set"
            ):
                await db_pool.initialize()

    @pytest.mark.asyncio
    async def test_close_closes_pool(self) -> None:
        """Test that close method closes the connection pool."""
        mock_pool = AsyncMock()

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        await db_pool.close()

        mock_pool.close.assert_called_once()
        assert db_pool._pool is None

    @pytest.mark.asyncio
    async def test_close_handles_none_pool(self) -> None:
        """Test that close handles case when pool is None."""
        db_pool = DatabasePool()
        db_pool._pool = None

        # Should not raise an exception
        await db_pool.close()
        assert db_pool._pool is None

    def test_is_initialized_returns_false_when_pool_none(self) -> None:
        """Test is_initialized returns False when pool is None."""
        db_pool = DatabasePool()
        assert not db_pool.is_initialized()

    def test_is_initialized_returns_true_when_pool_exists(self) -> None:
        """Test is_initialized returns True when pool exists."""
        db_pool = DatabasePool()
        db_pool._pool = MagicMock()
        assert db_pool.is_initialized()

    def test_pool_property_raises_when_not_initialized(self) -> None:
        """Test that accessing pool property raises error when not initialized."""
        db_pool = DatabasePool()

        with pytest.raises(RuntimeError, match="Database pool not initialized"):
            _ = db_pool.pool

    def test_pool_property_returns_pool_when_initialized(self) -> None:
        """Test that pool property returns the pool when initialized."""
        mock_pool = MagicMock()
        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        assert db_pool.pool == mock_pool

    @pytest.mark.asyncio
    async def test_get_system_metrics_retrieves_value(self) -> None:
        """Test get_system_metrics retrieves a value from the database."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value="test_value")

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        result = await db_pool.get_system_metrics("test_key")

        assert result == "test_value"
        mock_conn.fetchval.assert_called_once_with(
            "SELECT metric_value FROM system_metrics WHERE metric_key = $1",
            "test_key",
        )

    @pytest.mark.asyncio
    async def test_get_system_metrics_returns_none_when_not_found(self) -> None:
        """Test get_system_metrics returns None when key not found."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        result = await db_pool.get_system_metrics("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_system_metrics_inserts_or_updates(self) -> None:
        """Test set_system_metrics inserts or updates a value."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        await db_pool.set_system_metrics("test_key", "test_value")

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO system_metrics" in call_args[0][0]
        assert "ON CONFLICT" in call_args[0][0]
        assert call_args[0][1] == "test_key"
        assert call_args[0][2] == "test_value"

    @pytest.mark.asyncio
    async def test_get_next_check_at_returns_datetime(self) -> None:
        """Test get_next_check_at returns a datetime object."""
        test_time = "2026-01-19T12:00:00+00:00"
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=test_time)

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        result = await db_pool.get_next_check_at()

        assert isinstance(result, datetime)
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 19

    @pytest.mark.asyncio
    async def test_get_next_check_at_returns_none_when_not_set(self) -> None:
        """Test get_next_check_at returns None when not set."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        result = await db_pool.get_next_check_at()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_next_check_at_handles_z_suffix(self) -> None:
        """Test get_next_check_at handles Z suffix in ISO format."""
        test_time = "2026-01-19T12:00:00Z"
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=test_time)

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        result = await db_pool.get_next_check_at()

        assert isinstance(result, datetime)
        # Z should be replaced with +00:00
        assert result.year == 2026

    @pytest.mark.asyncio
    async def test_set_next_check_at_stores_datetime(self) -> None:
        """Test set_next_check_at stores a datetime as ISO format."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        test_dt = datetime(2026, 1, 19, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        await db_pool.set_next_check_at(test_dt)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert call_args[0][1] == "next_check_at"
        # Should store as ISO format string
        assert "2026-01-19" in call_args[0][2]


class TestGetUserTimezone:
    """Tests for get_user_timezone function."""

    @pytest.mark.asyncio
    async def test_returns_cached_timezone(self) -> None:
        """Test that cached timezone is returned without DB query."""
        # Set up cache
        import lattice.utils.database as db_module

        db_module._user_timezone_cache = "America/New_York"

        mock_pool = MagicMock()
        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        result = await get_user_timezone(db_pool)

        assert result == "America/New_York"
        # Should not have made any DB calls
        mock_pool.acquire.assert_not_called()

        # Clean up cache
        db_module._user_timezone_cache = None

    @pytest.mark.asyncio
    async def test_queries_db_when_cache_empty(self) -> None:
        """Test that DB is queried when cache is empty."""
        import lattice.utils.database as db_module

        db_module._user_timezone_cache = None

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"object": "America/Los_Angeles"})

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        result = await get_user_timezone(db_pool)

        assert result == "America/Los_Angeles"
        # Cache should be set
        assert db_module._user_timezone_cache == "America/Los_Angeles"

        # Clean up cache
        db_module._user_timezone_cache = None

    @pytest.mark.asyncio
    async def test_returns_utc_when_no_timezone_in_db(self) -> None:
        """Test that UTC is returned when no timezone found in DB."""
        import lattice.utils.database as db_module

        db_module._user_timezone_cache = None

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        result = await get_user_timezone(db_pool)

        assert result == "UTC"
        assert db_module._user_timezone_cache == "UTC"

        # Clean up cache
        db_module._user_timezone_cache = None

    @pytest.mark.asyncio
    async def test_returns_utc_when_invalid_timezone(self) -> None:
        """Test that UTC is returned when timezone string is invalid."""
        import lattice.utils.database as db_module

        db_module._user_timezone_cache = None

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"object": "Invalid/Timezone"})

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        result = await get_user_timezone(db_pool)

        assert result == "UTC"
        assert db_module._user_timezone_cache == "UTC"

        # Clean up cache
        db_module._user_timezone_cache = None

    @pytest.mark.asyncio
    async def test_queries_correct_semantic_memory(self) -> None:
        """Test that correct semantic memory query is used."""
        import lattice.utils.database as db_module

        db_module._user_timezone_cache = None

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"object": "Europe/London"})

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        await get_user_timezone(db_pool)

        # Verify the query structure
        call_args = mock_conn.fetchrow.call_args
        query = call_args[0][0]
        assert "SELECT object FROM semantic_memories" in query
        assert "subject = 'User'" in query
        assert "predicate = 'lives in timezone'" in query
        assert "ORDER BY created_at DESC" in query
        assert "LIMIT 1" in query

        # Clean up cache
        db_module._user_timezone_cache = None


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_get_system_metrics_calls_pool_method(self) -> None:
        """Test that module-level get_system_metrics calls pool method."""
        mock_pool = DatabasePool()
        mock_pool.get_system_metrics = AsyncMock(return_value="value")

        result = await get_system_metrics("key", mock_pool)

        assert result == "value"
        mock_pool.get_system_metrics.assert_called_once_with("key")

    @pytest.mark.asyncio
    async def test_set_system_metrics_calls_pool_method(self) -> None:
        """Test that module-level set_system_metrics calls pool method."""
        mock_pool = DatabasePool()
        mock_pool.set_system_metrics = AsyncMock()

        await set_system_metrics("key", "value", mock_pool)

        mock_pool.set_system_metrics.assert_called_once_with("key", "value")

    @pytest.mark.asyncio
    async def test_get_next_check_at_calls_pool_method(self) -> None:
        """Test that module-level get_next_check_at calls pool method."""
        test_dt = datetime(2026, 1, 19, tzinfo=ZoneInfo("UTC"))
        mock_pool = DatabasePool()
        mock_pool.get_next_check_at = AsyncMock(return_value=test_dt)

        result = await get_next_check_at(mock_pool)

        assert result == test_dt
        mock_pool.get_next_check_at.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_next_check_at_calls_pool_method(self) -> None:
        """Test that module-level set_next_check_at calls pool method."""
        test_dt = datetime(2026, 1, 19, tzinfo=ZoneInfo("UTC"))
        mock_pool = DatabasePool()
        mock_pool.set_next_check_at = AsyncMock()

        await set_next_check_at(test_dt, mock_pool)

        mock_pool.set_next_check_at.assert_called_once_with(test_dt)


class TestDatabasePoolErrorHandling:
    """Tests for database pool error handling."""

    @pytest.mark.asyncio
    async def test_initialize_propagates_connection_error(self) -> None:
        """Test that connection errors during initialization propagate."""
        with patch("lattice.utils.database.asyncpg.create_pool") as mock_create_pool:
            mock_create_pool.side_effect = Exception("Connection failed")

            with patch("lattice.utils.database.config") as mock_config:
                mock_config.database_url = "postgresql://test:test@localhost/test"
                mock_config.db_pool_min_size = 2
                mock_config.db_pool_max_size = 5

                db_pool = DatabasePool()

                with pytest.raises(Exception, match="Connection failed"):
                    await db_pool.initialize()

    @pytest.mark.asyncio
    async def test_get_system_metrics_propagates_db_error(self) -> None:
        """Test that database errors propagate from get_system_metrics."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=Exception("DB error"))

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        with pytest.raises(Exception, match="DB error"):
            await db_pool.get_system_metrics("test_key")

    @pytest.mark.asyncio
    async def test_set_system_metrics_propagates_db_error(self) -> None:
        """Test that database errors propagate from set_system_metrics."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("DB error"))

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        db_pool = DatabasePool()
        db_pool._pool = mock_pool

        with pytest.raises(Exception, match="DB error"):
            await db_pool.set_system_metrics("test_key", "test_value")
