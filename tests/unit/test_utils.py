"""Unit tests for Lattice utility modules."""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from zoneinfo import ZoneInfo

from lattice.utils.database import (
    DatabasePool,
    get_next_check_at,
    get_system_health,
    get_user_timezone,
    set_next_check_at,
    set_system_health,
    set_user_timezone,
)
from lattice.utils.date_resolution import (
    DateRange,
    InvalidTimezoneError,
    InvalidWeekdayError,
    UserDatetime,
    format_current_date,
    format_current_time,
    parse_relative_date_range,
    resolve_relative_dates,
)
from lattice.utils.objective_parsing import parse_goals


# NOTE: EmbeddingModel tests removed during Issue #61 refactor
# Embedding functionality is being replaced with entity extraction


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
    async def test_initialize_when_already_initialized(self) -> None:
        """Test that re-initializing pool replaces existing pool."""
        pool = DatabasePool()
        first_mock_pool = AsyncMock()
        second_mock_pool = AsyncMock()

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}, clear=True):
            with patch(
                "lattice.utils.database.asyncpg.create_pool",
                new=AsyncMock(side_effect=[first_mock_pool, second_mock_pool]),
            ):
                # First initialization
                await pool.initialize()
                assert pool._pool == first_mock_pool

                # Second initialization replaces the pool
                await pool.initialize()
                assert pool._pool == second_mock_pool

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

    def test_parse_goals_valid_json(self) -> None:
        """Test parsing valid objective JSON array."""
        raw = (
            '[{"description": "Build a startup", "saliency": 0.9, "status": "pending"}]'
        )

        result = parse_goals(raw)

        assert len(result) == 1
        assert result[0]["description"] == "Build a startup"
        assert result[0]["saliency"] == 0.9
        assert result[0]["status"] == "pending"

    def test_parse_goals_multiple(self) -> None:
        """Test parsing multiple objectives."""
        raw = (
            '[{"description": "Learn Python", "saliency": 0.8, "status": "pending"}, '
            '{"description": "Build a project", "saliency": 0.7, "status": "completed"}]'
        )

        result = parse_goals(raw)

        assert len(result) == 2

    def test_parse_goals_empty_array(self) -> None:
        """Test parsing empty objective array."""
        raw = "[]"

        result = parse_goals(raw)

        assert result == []

    def test_parse_goals_with_code_block(self) -> None:
        """Test parsing objectives wrapped in code block."""
        raw = """```json
[{"description": "Test goal", "saliency": 0.5, "status": "pending"}]
```"""

        result = parse_goals(raw)

        assert len(result) == 1
        assert result[0]["description"] == "Test goal"

    def test_parse_goals_saliency_clamping(self) -> None:
        """Test that saliency is clamped to valid range."""
        raw = '[{"description": "Goal", "saliency": 1.5, "status": "pending"}]'

        result = parse_goals(raw)

        assert result[0]["saliency"] == 1.0

    def test_parse_goals_saliency_default(self) -> None:
        """Test that missing saliency uses default."""
        raw = '[{"description": "Goal", "status": "pending"}]'

        result = parse_goals(raw)

        assert result[0]["saliency"] == 0.5

    def test_parse_goals_invalid_status(self) -> None:
        """Test that invalid status defaults to pending."""
        raw = '[{"description": "Goal", "saliency": 0.5, "status": "invalid"}]'

        result = parse_goals(raw)

        assert result[0]["status"] == "pending"

    def test_parse_goals_invalid_json(self) -> None:
        """Test that invalid JSON returns empty list."""
        raw = "not valid json"

        result = parse_goals(raw)

        assert result == []

    def test_parse_goals_missing_description(self) -> None:
        """Test that objectives without description are filtered out."""
        raw = (
            '[{"description": "Valid", "saliency": 0.5, "status": "pending"}, '
            '{"saliency": 0.5, "status": "pending"}]'
        )

        result = parse_goals(raw)

        assert len(result) == 1
        assert result[0]["description"] == "Valid"

    def test_parse_goals_negative_saliency(self) -> None:
        """Test that negative saliency is clamped to 0.0."""
        raw = '[{"description": "Goal", "saliency": -0.5, "status": "pending"}]'

        result = parse_goals(raw)

        assert result[0]["saliency"] == 0.0

    def test_parse_goals_empty_description(self) -> None:
        """Test that empty description is filtered out."""
        raw = '[{"description": "", "saliency": 0.5, "status": "pending"}]'

        result = parse_goals(raw)

        assert len(result) == 0

    def test_parse_goals_whitespace_description(self) -> None:
        """Test that whitespace-only description is filtered out."""
        raw = '[{"description": "   ", "saliency": 0.5, "status": "pending"}]'

        result = parse_goals(raw)

        assert len(result) == 0

    def test_parse_goals_none_description(self) -> None:
        """Test that None description is filtered out."""
        raw = '[{"description": null, "saliency": 0.5, "status": "pending"}]'

        result = parse_goals(raw)

        assert len(result) == 0

    def test_parse_goals_status_case_insensitive(self) -> None:
        """Test that status is normalized to lowercase."""
        raw = '[{"description": "Goal", "saliency": 0.5, "status": "COMPLETED"}]'

        result = parse_goals(raw)

        assert result[0]["status"] == "completed"


@pytest.fixture
def fixed_user_datetime() -> UserDatetime:
    """Fixture providing a UserDatetime with fixed test date."""
    return UserDatetime.for_testing(
        datetime(2026, 1, 8, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
    )


@pytest.fixture
def mock_user_datetime(fixed_user_datetime: UserDatetime) -> MagicMock:
    """Fixture providing a mocked UserDatetime for date resolution tests."""
    mock_instance = MagicMock()
    mock_instance.now = fixed_user_datetime.now
    mock_instance.format.side_effect = lambda fmt: fixed_user_datetime.format(fmt)
    # Use side_effect to return different values based on the argument
    mock_instance.add_days.side_effect = lambda days: fixed_user_datetime.add_days(days)
    mock_instance.add_weeks.side_effect = lambda weeks: fixed_user_datetime.add_weeks(
        weeks
    )
    mock_instance.add_months.side_effect = lambda: fixed_user_datetime.add_months()
    mock_instance.add_years.side_effect = lambda: fixed_user_datetime.add_years()
    mock_instance.next_weekday.side_effect = (
        lambda day: fixed_user_datetime.next_weekday(day)
    )
    return mock_instance


class TestUserDatetime:
    """Tests for the UserDatetime class."""

    def test_init_default_timezone(self) -> None:
        """Test that UserDatetime defaults to UTC."""
        user_dt = UserDatetime()
        assert user_dt._timezone_str == "UTC"

    def test_init_custom_timezone(self) -> None:
        """Test initialization with custom timezone."""
        user_dt = UserDatetime("America/New_York")
        assert user_dt._timezone_str == "America/New_York"

    def test_invalid_timezone_raises_error(self) -> None:
        """Test that invalid timezone raises InvalidTimezoneError when tz property is accessed."""
        user_dt = UserDatetime("Invalid/Timezone")
        with pytest.raises(InvalidTimezoneError):
            _ = user_dt.tz

    def test_next_weekday_valid(self, fixed_user_datetime: UserDatetime) -> None:
        """Test next_weekday returns correct date."""
        next_monday = fixed_user_datetime.next_weekday("monday")
        # Today is Thursday 2026-01-08, next Monday is 2026-01-12
        assert next_monday.strftime("%Y-%m-%d") == "2026-01-12"

    def test_next_weekday_today_is_target(
        self, fixed_user_datetime: UserDatetime
    ) -> None:
        """Test that next_weekday returns next week if today is target."""
        # Today is Thursday, next Thursday should be 7 days from now
        next_thursday = fixed_user_datetime.next_weekday("thursday")
        assert next_thursday.strftime("%Y-%m-%d") == "2026-01-15"

    def test_next_weekday_invalid_raises_error(
        self, fixed_user_datetime: UserDatetime
    ) -> None:
        """Test that invalid weekday raises InvalidWeekdayError."""
        with pytest.raises(InvalidWeekdayError):
            fixed_user_datetime.next_weekday("funday")

    def test_add_days(self, fixed_user_datetime: UserDatetime) -> None:
        """Test add_days method."""
        result = fixed_user_datetime.add_days(5)
        assert result.strftime("%Y-%m-%d") == "2026-01-13"

    def test_add_weeks(self, fixed_user_datetime: UserDatetime) -> None:
        """Test add_weeks method."""
        result = fixed_user_datetime.add_weeks(1)
        assert result.strftime("%Y-%m-%d") == "2026-01-15"

    def test_add_months(self, fixed_user_datetime: UserDatetime) -> None:
        """Test add_months method."""
        result = fixed_user_datetime.add_months()
        assert result.strftime("%Y-%m-%d") == "2026-02-08"

    def test_add_years(self, fixed_user_datetime: UserDatetime) -> None:
        """Test add_years method."""
        result = fixed_user_datetime.add_years()
        assert result.strftime("%Y-%m-%d") == "2027-01-08"


class TestFormatCurrentDateAndTime:
    """Tests for format_current_date and format_current_time functions."""

    def test_format_current_date_default_timezone(
        self, mock_user_datetime: MagicMock
    ) -> None:
        """Test formatting current date with default UTC timezone."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = format_current_date()
            assert result == "2026/01/08, Thursday"

    def test_format_current_date_custom_timezone(
        self, mock_user_datetime: MagicMock
    ) -> None:
        """Test formatting with custom timezone."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = format_current_date("America/New_York")
            assert "2026/01/08" in result
            assert "Thursday" in result

    def test_format_current_time_default_timezone(
        self, mock_user_datetime: MagicMock
    ) -> None:
        """Test formatting current time with default UTC timezone."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = format_current_time()
            assert result == "12:00"

    def test_format_current_time_custom_timezone(
        self, mock_user_datetime: MagicMock
    ) -> None:
        """Test formatting time with custom timezone."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = format_current_time("America/New_York")
            assert ":" in result  # Time format with hours and minutes


class TestResolveRelativeDates:
    """Tests for resolve_relative_dates function."""

    def test_resolve_today(self, mock_user_datetime: MagicMock) -> None:
        """Test that 'today' returns empty hints (redundant with {local_date})."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Let's meet today", "UTC")
            assert result == ""

    def test_resolve_tonight(self, mock_user_datetime: MagicMock) -> None:
        """Test that 'tonight' returns empty hints (redundant with {local_date})."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Watching a movie tonight", "UTC")
            assert result == ""

    def test_resolve_tomorrow(self, mock_user_datetime: MagicMock) -> None:
        """Test resolving 'tomorrow'."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Let's meet tomorrow", "UTC")
            assert "tomorrow → 2026-01-09" in result

    def test_resolve_yesterday(self, mock_user_datetime: MagicMock) -> None:
        """Test resolving 'yesterday'."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("That was yesterday", "UTC")
            assert "yesterday → 2026-01-07" in result

    def test_resolve_day_after_tomorrow(self, mock_user_datetime: MagicMock) -> None:
        """Test resolving 'day after tomorrow'."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("See you day after tomorrow", "UTC")
            assert "day after tomorrow → 2026-01-10" in result

    def test_resolve_day_before_yesterday(self, mock_user_datetime: MagicMock) -> None:
        """Test resolving 'day before yesterday'."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("That happened day before yesterday", "UTC")
            assert "day before yesterday → 2026-01-06" in result

    def test_resolve_next_week(self, mock_user_datetime: MagicMock) -> None:
        """Test resolving 'next week'."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Let's meet next week", "UTC")
            assert "next week → 2026-01-15" in result

    def test_resolve_next_month(self, mock_user_datetime: MagicMock) -> None:
        """Test resolving 'next month'."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Due next month", "UTC")
            assert "next month → 2026-02-08" in result

    def test_resolve_next_year(self, mock_user_datetime: MagicMock) -> None:
        """Test resolving 'next year'."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Coming next year", "UTC")
            assert "next year → 2027-01-08" in result

    def test_resolve_plain_weekday_this_week(
        self, mock_user_datetime: MagicMock
    ) -> None:
        """Test resolving plain weekday (this week's occurrence)."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Meeting on Friday", "UTC")
            # Today is Thursday 2026-01-08, Friday is tomorrow (1 day ahead)
            assert "friday → 2026-01-09" in result

    def test_resolve_next_weekday_plus_7_days(
        self, mock_user_datetime: MagicMock
    ) -> None:
        """Test that 'next wednesday' is 7 days after this wednesday."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Deadline is next wednesday", "UTC")
            # Today is Thursday 2026-01-08, next wednesday is 2026-01-14 + 7 = 2026-01-21
            assert "next wednesday → 2026-01-21" in result

    def test_resolve_weekday_next_week_format(
        self, mock_user_datetime: MagicMock
    ) -> None:
        """Test resolving 'wednesday next week' format."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Let's meet wednesday next week", "UTC")
            # "wednesday next week" is normalized to "next wednesday"
            assert "next wednesday → 2026-01-21" in result

    def test_resolve_multiple_patterns(self, mock_user_datetime: MagicMock) -> None:
        """Test resolving multiple date patterns in one message."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Meet tomorrow and finish by Friday", "UTC")
            assert "tomorrow → 2026-01-09" in result
            # Today is Thursday, Friday is tomorrow (1 day ahead)
            assert "friday → 2026-01-09" in result

    def test_resolve_next_weekday_with_the(self, mock_user_datetime: MagicMock) -> None:
        """Test resolving 'next the wednesday' pattern."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Due next the wednesday", "UTC")
            # "next the wednesday" → next wednesday
            assert "next wednesday → 2026-01-21" in result

    def test_resolve_case_insensitive(self, mock_user_datetime: MagicMock) -> None:
        """Test that pattern matching is case insensitive."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("TOMORROW and NEXT FRIDAY", "UTC")
            assert "tomorrow → 2026-01-09" in result
            # Today is Thursday, next Friday is 2026-01-09 + 7 = 2026-01-16
            assert "next friday → 2026-01-16" in result

    def test_resolve_mixed_patterns(self, mock_user_datetime: MagicMock) -> None:
        """Test resolving mixed patterns with weekdays and relative dates."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Next Friday and day after tomorrow", "UTC")
            # Next Friday: today is Thursday, this Friday is tomorrow 2026-01-09, next Friday +7 = 2026-01-16
            assert "next friday → 2026-01-16" in result
            # Day after tomorrow: today is Thursday, day after tomorrow is Saturday 2026-01-10
            assert "day after tomorrow → 2026-01-10" in result

    def test_resolve_plain_weekday_when_next_weekday_also_mentioned(
        self, mock_user_datetime: MagicMock
    ) -> None:
        """Test that plain weekday is not duplicated when next weekday is mentioned."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates("Next Friday and also Friday", "UTC")
            assert "next friday → 2026-01-16" in result
            # Today is Thursday, Friday is tomorrow 2026-01-09
            assert "friday → 2026-01-09" in result
            # Should not have duplicate entries
            assert result.count("friday") == 2  # Once in "next friday", once as plain

    def test_resolve_all_weekdays(self, mock_user_datetime: MagicMock) -> None:
        """Test resolving all weekday names."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = resolve_relative_dates(
                "monday tuesday wednesday thursday friday saturday sunday", "UTC"
            )
            # Today is Thursday 2026-01-08
            # Monday: 2026-01-12 (4 days ahead)
            assert "monday → 2026-01-12" in result
            # Tuesday: 2026-01-13 (5 days ahead)
            assert "tuesday → 2026-01-13" in result
            # Wednesday: 2026-01-14 (6 days ahead)
            assert "wednesday → 2026-01-14" in result
            # Thursday: 2026-01-15 (7 days ahead - next week's Thursday)
            assert "thursday → 2026-01-15" in result
            # Friday: 2026-01-09 (1 day ahead - this week's Friday/tomorrow)
            assert "friday → 2026-01-09" in result
            # Saturday: 2026-01-10 (2 days ahead)
            assert "saturday → 2026-01-10" in result
            # Sunday: 2026-01-11 (3 days ahead)
            assert "sunday → 2026-01-11" in result


class TestDateRange:
    """Tests for the DateRange class."""

    def test_daterange_init(self) -> None:
        """Test DateRange initialization."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 7, tzinfo=timezone.utc)
        date_range = DateRange(start=start, end=end)
        assert date_range.start == start
        assert date_range.end == end

    def test_daterange_repr(self) -> None:
        """Test DateRange string representation."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 7, tzinfo=timezone.utc)
        date_range = DateRange(start=start, end=end)
        repr_str = repr(date_range)
        assert "DateRange" in repr_str
        assert "2026-01-01" in repr_str
        assert "2026-01-07" in repr_str


class TestParseRelativeDateRange:
    """Tests for parse_relative_date_range function."""

    def test_parse_last_week(self, mock_user_datetime: MagicMock) -> None:
        """Test parsing 'last week' pattern."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = parse_relative_date_range("What did I do last week?", "UTC")
            assert result is not None
            # Today is Thursday 2026-01-08
            # Last week: Monday 2025-12-29 to Sunday 2026-01-04
            assert result.start.year == 2025
            assert result.start.month == 12
            assert result.start.day == 29
            assert result.end.year == 2026
            assert result.end.month == 1
            assert result.end.day == 4

    def test_parse_this_week(self, mock_user_datetime: MagicMock) -> None:
        """Test parsing 'this week' pattern."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = parse_relative_date_range("What did I do this week?", "UTC")
            assert result is not None
            # Today is Thursday 2026-01-08
            # This week: Monday 2026-01-05 to now
            assert result.start.year == 2026
            assert result.start.month == 1
            assert result.start.day == 5

    def test_parse_last_month(self, mock_user_datetime: MagicMock) -> None:
        """Test parsing 'last month' pattern."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = parse_relative_date_range("Summarize last month", "UTC")
            assert result is not None
            # Today is Thursday 2026-01-08
            # Last month: December 2025
            assert result.start.year == 2025
            assert result.start.month == 12
            assert result.start.day == 1
            assert result.end.year == 2025
            assert result.end.month == 12
            assert result.end.day == 31

    def test_parse_yesterday(self, mock_user_datetime: MagicMock) -> None:
        """Test parsing 'yesterday' pattern."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = parse_relative_date_range("What did I do yesterday?", "UTC")
            assert result is not None
            # Today is Thursday 2026-01-08
            # Yesterday: 2026-01-07
            assert result.start.year == 2026
            assert result.start.month == 1
            assert result.start.day == 7
            assert result.end.year == 2026
            assert result.end.month == 1
            assert result.end.day == 7

    def test_parse_today(self, mock_user_datetime: MagicMock) -> None:
        """Test parsing 'today' pattern."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = parse_relative_date_range("What did I do today?", "UTC")
            assert result is not None
            # Today: from midnight to now
            assert result.start.year == 2026
            assert result.start.month == 1
            assert result.start.day == 8
            assert result.start.hour == 0
            assert result.start.minute == 0

    def test_parse_last_7_days(self, mock_user_datetime: MagicMock) -> None:
        """Test parsing 'last 7 days' pattern."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = parse_relative_date_range("Show me last 7 days", "UTC")
            assert result is not None
            # Last 7 days: 2026-01-01 to now
            assert result.start.year == 2026
            assert result.start.month == 1
            assert result.start.day == 1  # 8 - 7 + 1 = 2 (inclusive)

    def test_parse_last_30_days(self, mock_user_datetime: MagicMock) -> None:
        """Test parsing 'last 30 days' pattern."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = parse_relative_date_range("What about last 30 days?", "UTC")
            assert result is not None
            # Last 30 days: 2025-12-09 to now
            assert result.start.year == 2025
            assert result.start.month == 12
            assert result.start.day == 9  # 8 - 30 + 1 = -21, wraps to Dec

    def test_parse_no_date_range(self, mock_user_datetime: MagicMock) -> None:
        """Test that messages without date patterns return None."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = parse_relative_date_range("I need to finish the project", "UTC")
            assert result is None

    def test_parse_case_insensitive(self, mock_user_datetime: MagicMock) -> None:
        """Test that pattern matching is case insensitive."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = parse_relative_date_range("LAST WEEK was busy", "UTC")
            assert result is not None

    def test_parse_with_custom_timezone(self, mock_user_datetime: MagicMock) -> None:
        """Test parsing with custom timezone."""
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = parse_relative_date_range(
                "What did I do last week?", "America/New_York"
            )
            assert result is not None

    def test_parse_last_month_when_january(self, mock_user_datetime: MagicMock) -> None:
        """Test parsing 'last month' when current month is January."""
        # Today is Thursday 2026-01-08, which is January
        with patch(
            "lattice.utils.date_resolution._get_user_datetime",
            return_value=mock_user_datetime,
        ):
            result = parse_relative_date_range("Summarize last month", "UTC")
            assert result is not None
            # Last month should be December 2025
            assert result.start.year == 2025
            assert result.start.month == 12
