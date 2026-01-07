"""Unit tests for Lattice utility modules."""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.utils.database import (
    DatabasePool,
    get_next_check_at,
    get_system_health,
    get_user_timezone,
    set_next_check_at,
    set_system_health,
    set_user_timezone,
)
from lattice.utils.objective_parsing import parse_objectives


# NOTE: EmbeddingModel tests removed during Issue #61 refactor
# Embedding functionality is being replaced with query extraction


class TestDatabasePool:
    """Tests for the database pool module."""

    def test_database_pool_initial_state(self) -> None:
        """Test that DatabasePool starts uninitialized."""
        pool = DatabasePool()

        assert pool.is_initialized() is False
        assert pool._pool is None

    def test_database_pool_property_raises_when_not_initialized(self) -> None:
        """Test that pool property raises RuntimeError when not initialized."""
        pool = DatabasePool()

        with pytest.raises(RuntimeError, match="Database pool not initialized"):
            _ = pool.pool

    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self) -> None:
        """Test that close is safe when pool is not initialized."""
        pool = DatabasePool()

        await pool.close()

        assert pool._pool is None

    @pytest.mark.asyncio
    async def test_initialize_missing_database_url(self) -> None:
        """Test that initialize raises ValueError when DATABASE_URL is missing."""
        pool = DatabasePool()

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="DATABASE_URL environment variable not set"
            ):
                await pool.initialize()

    @pytest.mark.asyncio
    async def test_initialize_success_with_default_pool_sizes(self) -> None:
        """Test successful initialization with default pool sizes."""
        pool = DatabasePool()
        mock_pool = AsyncMock()

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}, clear=True):
            with patch(
                "lattice.utils.database.asyncpg.create_pool",
                new=AsyncMock(return_value=mock_pool),
            ) as mock_create:
                await pool.initialize()

                mock_create.assert_called_once_with(
                    "postgresql://test",
                    min_size=2,
                    max_size=5,
                    command_timeout=30,
                )
                assert pool.is_initialized() is True
                assert pool._pool == mock_pool

    @pytest.mark.asyncio
    async def test_initialize_success_with_custom_pool_sizes(self) -> None:
        """Test successful initialization with custom pool sizes."""
        pool = DatabasePool()
        mock_pool = AsyncMock()

        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://test",
                "DB_POOL_MIN_SIZE": "3",
                "DB_POOL_MAX_SIZE": "10",
            },
            clear=True,
        ):
            with patch(
                "lattice.utils.database.asyncpg.create_pool",
                new=AsyncMock(return_value=mock_pool),
            ) as mock_create:
                await pool.initialize()

                mock_create.assert_called_once_with(
                    "postgresql://test",
                    min_size=3,
                    max_size=10,
                    command_timeout=30,
                )
                assert pool.is_initialized() is True
                assert pool._pool == mock_pool

    @pytest.mark.asyncio
    async def test_initialize_invalid_pool_size_format(self) -> None:
        """Test that initialize raises ValueError for non-numeric pool sizes."""
        pool = DatabasePool()

        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://test",
                "DB_POOL_MIN_SIZE": "invalid",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="invalid literal for int()"):
                await pool.initialize()

    @pytest.mark.asyncio
    async def test_close_when_initialized(self) -> None:
        """Test that close properly closes the pool when initialized."""
        pool = DatabasePool()
        mock_pool = AsyncMock()
        pool._pool = mock_pool

        await pool.close()

        mock_pool.close.assert_called_once()
        assert pool._pool is None
        assert pool.is_initialized() is False

    @pytest.mark.asyncio
    async def test_pool_property_returns_pool_when_initialized(self) -> None:
        """Test that pool property returns the pool when initialized."""
        pool = DatabasePool()
        mock_pool = AsyncMock()
        pool._pool = mock_pool

        result = pool.pool

        assert result == mock_pool


class TestSystemHealthFunctions:
    """Tests for system health database functions."""

    @pytest.mark.asyncio
    async def test_get_system_health_success(self) -> None:
        """Test get_system_health retrieves value from database."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value="test_value")
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("lattice.utils.database.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await get_system_health("test_key")

            assert result == "test_value"
            mock_conn.fetchval.assert_called_once_with(
                "SELECT metric_value FROM system_health WHERE metric_key = $1",
                "test_key",
            )

    @pytest.mark.asyncio
    async def test_get_system_health_not_found(self) -> None:
        """Test get_system_health returns None when key not found."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("lattice.utils.database.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await get_system_health("nonexistent_key")

            assert result is None

    @pytest.mark.asyncio
    async def test_set_system_health_success(self) -> None:
        """Test set_system_health inserts or updates value in database."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("lattice.utils.database.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            await set_system_health("test_key", "test_value")

            mock_conn.execute.assert_called_once()
            call_args = mock_conn.execute.call_args
            assert "INSERT INTO system_health" in call_args[0][0]
            assert "ON CONFLICT (metric_key)" in call_args[0][0]
            assert call_args[0][1] == "test_key"
            assert call_args[0][2] == "test_value"


class TestNextCheckAtFunctions:
    """Tests for next check timestamp functions."""

    @pytest.mark.asyncio
    async def test_get_next_check_at_with_value(self) -> None:
        """Test get_next_check_at parses ISO datetime string."""
        test_dt = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        iso_string = test_dt.isoformat()

        with patch("lattice.utils.database.get_system_health", return_value=iso_string):
            result = await get_next_check_at()

            assert result == test_dt

    @pytest.mark.asyncio
    async def test_get_next_check_at_with_z_suffix(self) -> None:
        """Test get_next_check_at handles Z suffix for UTC."""
        with patch(
            "lattice.utils.database.get_system_health",
            return_value="2026-01-15T10:30:00Z",
        ):
            result = await get_next_check_at()

            assert result == datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    @pytest.mark.asyncio
    async def test_get_next_check_at_none(self) -> None:
        """Test get_next_check_at returns None when not set."""
        with patch("lattice.utils.database.get_system_health", return_value=None):
            result = await get_next_check_at()

            assert result is None

    @pytest.mark.asyncio
    async def test_set_next_check_at_success(self) -> None:
        """Test set_next_check_at stores ISO formatted datetime."""
        test_dt = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

        with patch("lattice.utils.database.set_system_health") as mock_set:
            await set_next_check_at(test_dt)

            mock_set.assert_called_once_with("next_check_at", test_dt.isoformat())

    @pytest.mark.asyncio
    async def test_get_next_check_at_invalid_format(self) -> None:
        """Test get_next_check_at raises ValueError for invalid datetime format."""
        with patch(
            "lattice.utils.database.get_system_health",
            return_value="not-a-valid-datetime",
        ):
            with pytest.raises(ValueError, match="Invalid isoformat string"):
                await get_next_check_at()


class TestUserTimezoneFunctions:
    """Tests for user timezone functions."""

    @pytest.mark.asyncio
    async def test_get_user_timezone_with_value(self) -> None:
        """Test get_user_timezone retrieves stored timezone."""
        with patch(
            "lattice.utils.database.get_system_health", return_value="America/New_York"
        ):
            result = await get_user_timezone()

            assert result == "America/New_York"

    @pytest.mark.asyncio
    async def test_get_user_timezone_defaults_to_utc(self) -> None:
        """Test get_user_timezone defaults to UTC when not set."""
        with patch("lattice.utils.database.get_system_health", return_value=None):
            result = await get_user_timezone()

            assert result == "UTC"

    @pytest.mark.asyncio
    async def test_set_user_timezone_valid_timezone(self) -> None:
        """Test set_user_timezone validates and stores valid timezone."""
        with patch("lattice.utils.database.set_system_health") as mock_set:
            await set_user_timezone("America/Los_Angeles")

            mock_set.assert_called_once_with("user_timezone", "America/Los_Angeles")

    @pytest.mark.asyncio
    async def test_set_user_timezone_invalid_timezone(self) -> None:
        """Test set_user_timezone raises ValueError for invalid timezone."""
        with pytest.raises(ValueError, match="Invalid timezone: Invalid/Timezone"):
            await set_user_timezone("Invalid/Timezone")


class TestObjectiveParsing:
    """Tests for objective parsing utilities."""

    def test_parse_objectives_valid_json(self) -> None:
        """Test parsing valid objective JSON array."""
        raw = (
            '[{"description": "Build a startup", "saliency": 0.9, "status": "pending"}]'
        )

        result = parse_objectives(raw)

        assert len(result) == 1
        assert result[0]["description"] == "Build a startup"
        assert result[0]["saliency"] == 0.9
        assert result[0]["status"] == "pending"

    def test_parse_objectives_multiple(self) -> None:
        """Test parsing multiple objectives."""
        raw = (
            '[{"description": "Learn Python", "saliency": 0.8, "status": "pending"}, '
            '{"description": "Build a project", "saliency": 0.7, "status": "completed"}]'
        )

        result = parse_objectives(raw)

        assert len(result) == 2

    def test_parse_objectives_empty_array(self) -> None:
        """Test parsing empty objective array."""
        raw = "[]"

        result = parse_objectives(raw)

        assert result == []

    def test_parse_objectives_with_code_block(self) -> None:
        """Test parsing objectives wrapped in code block."""
        raw = """```json
[{"description": "Test goal", "saliency": 0.5, "status": "pending"}]
```"""

        result = parse_objectives(raw)

        assert len(result) == 1
        assert result[0]["description"] == "Test goal"

    def test_parse_objectives_saliency_clamping(self) -> None:
        """Test that saliency is clamped to valid range."""
        raw = '[{"description": "Goal", "saliency": 1.5, "status": "pending"}]'

        result = parse_objectives(raw)

        assert result[0]["saliency"] == 1.0

    def test_parse_objectives_saliency_default(self) -> None:
        """Test that missing saliency uses default."""
        raw = '[{"description": "Goal", "status": "pending"}]'

        result = parse_objectives(raw)

        assert result[0]["saliency"] == 0.5

    def test_parse_objectives_invalid_status(self) -> None:
        """Test that invalid status defaults to pending."""
        raw = '[{"description": "Goal", "saliency": 0.5, "status": "invalid"}]'

        result = parse_objectives(raw)

        assert result[0]["status"] == "pending"

    def test_parse_objectives_invalid_json(self) -> None:
        """Test that invalid JSON returns empty list."""
        raw = "not valid json"

        result = parse_objectives(raw)

        assert result == []

    def test_parse_objectives_missing_description(self) -> None:
        """Test that objectives without description are filtered out."""
        raw = (
            '[{"description": "Valid", "saliency": 0.5, "status": "pending"}, '
            '{"saliency": 0.5, "status": "pending"}]'
        )

        result = parse_objectives(raw)

        assert len(result) == 1
        assert result[0]["description"] == "Valid"

    def test_parse_objectives_negative_saliency(self) -> None:
        """Test that negative saliency is clamped to 0.0."""
        raw = '[{"description": "Goal", "saliency": -0.5, "status": "pending"}]'

        result = parse_objectives(raw)

        assert result[0]["saliency"] == 0.0

    def test_parse_objectives_empty_description(self) -> None:
        """Test that empty description is filtered out."""
        raw = '[{"description": "", "saliency": 0.5, "status": "pending"}]'

        result = parse_objectives(raw)

        assert len(result) == 0

    def test_parse_objectives_whitespace_description(self) -> None:
        """Test that whitespace-only description is filtered out."""
        raw = '[{"description": "   ", "saliency": 0.5, "status": "pending"}]'

        result = parse_objectives(raw)

        assert len(result) == 0

    def test_parse_objectives_none_description(self) -> None:
        """Test that None description is filtered out."""
        raw = '[{"description": null, "saliency": 0.5, "status": "pending"}]'

        result = parse_objectives(raw)

        assert len(result) == 0

    def test_parse_objectives_status_case_insensitive(self) -> None:
        """Test that status is normalized to lowercase."""
        raw = '[{"description": "Goal", "saliency": 0.5, "status": "COMPLETED"}]'

        result = parse_objectives(raw)

        assert result[0]["status"] == "completed"
