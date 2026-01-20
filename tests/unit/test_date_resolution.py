"""Unit tests for date resolution utilities."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from lattice.utils.date_resolution import (
    DateRange,
    InvalidTimezoneError,
    InvalidWeekdayError,
    UserDatetime,
    format_current_date,
    format_current_time,
    get_next_weekday,
    get_now,
    get_user_datetime,
    parse_relative_date_range,
    resolve_relative_dates,
)


class TestUserDatetime:
    """Tests for UserDatetime class."""

    def test_init_with_default_timezone(self) -> None:
        """Test UserDatetime initializes with UTC by default."""
        udt = UserDatetime()
        assert udt.tz == ZoneInfo("UTC")

    def test_init_with_custom_timezone(self) -> None:
        """Test UserDatetime initializes with custom timezone."""
        udt = UserDatetime("America/New_York")
        assert udt.tz == ZoneInfo("America/New_York")

    def test_init_with_invalid_timezone(self) -> None:
        """Test UserDatetime raises error for invalid timezone."""
        with pytest.raises(InvalidTimezoneError, match="Invalid timezone"):
            udt = UserDatetime("Invalid/Timezone")
            _ = udt.tz  # Access tz to trigger validation

    def test_now_returns_current_time(self) -> None:
        """Test that now property returns current time."""
        udt = UserDatetime("UTC")
        now = udt.now
        assert isinstance(now, datetime)
        assert now.tzinfo == ZoneInfo("UTC")

    def test_now_with_fixed_time_for_testing(self) -> None:
        """Test UserDatetime with fixed time for testing."""
        fixed_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)
        assert udt.now == fixed_time

    def test_for_testing_classmethod(self) -> None:
        """Test for_testing classmethod creates instance with fixed time."""
        fixed_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime.for_testing(fixed_time)
        assert udt.now == fixed_time

    def test_next_weekday_same_week(self) -> None:
        """Test next_weekday returns correct date in same week."""
        # Thursday, Jan 15, 2026
        fixed_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)

        # Next Friday should be Jan 16, 2026 (tomorrow)
        friday = udt.next_weekday("friday")
        assert friday.day == 16
        assert friday.month == 1
        assert friday.year == 2026

    def test_next_weekday_next_week(self) -> None:
        """Test next_weekday returns date in next week when day has passed."""
        # Thursday, Jan 15, 2026
        fixed_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)

        # Next Monday should be Jan 19, 2026 (next week)
        monday = udt.next_weekday("monday")
        assert monday.day == 19
        assert monday.month == 1
        assert monday.year == 2026

    def test_next_weekday_case_insensitive(self) -> None:
        """Test next_weekday is case insensitive."""
        fixed_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)

        friday1 = udt.next_weekday("Friday")
        friday2 = udt.next_weekday("FRIDAY")
        friday3 = udt.next_weekday("friday")

        assert friday1 == friday2 == friday3

    def test_next_weekday_invalid_weekday(self) -> None:
        """Test next_weekday raises error for invalid weekday."""
        udt = UserDatetime()
        with pytest.raises(InvalidWeekdayError, match="Invalid weekday"):
            udt.next_weekday("notaday")

    def test_format_returns_formatted_string(self) -> None:
        """Test format method returns formatted datetime string."""
        fixed_time = datetime(2026, 1, 15, 12, 30, 45, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)

        assert udt.format("%Y-%m-%d") == "2026-01-15"
        assert udt.format("%H:%M:%S") == "12:30:45"
        assert udt.format("%A, %B %d, %Y") == "Thursday, January 15, 2026"

    def test_add_days(self) -> None:
        """Test add_days adds correct number of days."""
        fixed_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)

        future = udt.add_days(5)
        assert future.day == 20
        assert future.month == 1
        assert future.year == 2026

    def test_add_days_crosses_month_boundary(self) -> None:
        """Test add_days works across month boundaries."""
        fixed_time = datetime(2026, 1, 29, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)

        future = udt.add_days(5)
        assert future.day == 3
        assert future.month == 2
        assert future.year == 2026

    def test_add_weeks(self) -> None:
        """Test add_weeks adds correct number of weeks."""
        fixed_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)

        future = udt.add_weeks(2)
        assert future.day == 29
        assert future.month == 1
        assert future.year == 2026

    def test_add_months_within_year(self) -> None:
        """Test add_months adds one month correctly."""
        fixed_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)

        future = udt.add_months()
        assert future.day == 15
        assert future.month == 2
        assert future.year == 2026

    def test_add_months_crosses_year_boundary(self) -> None:
        """Test add_months works across year boundaries."""
        fixed_time = datetime(2025, 12, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)

        future = udt.add_months()
        assert future.day == 15
        assert future.month == 1
        assert future.year == 2026

    def test_add_months_handles_day_clamping(self) -> None:
        """Test add_months clamps day when target month has fewer days."""
        # Jan 31 + 1 month should become Feb 28 (or 29 in leap year)
        fixed_time = datetime(2026, 1, 31, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)

        future = udt.add_months()
        assert future.day == 28  # 2026 is not a leap year
        assert future.month == 2
        assert future.year == 2026

    def test_add_months_leap_year_clamping(self) -> None:
        """Test add_months handles leap year correctly."""
        # Jan 31, 2024 + 1 month should become Feb 29 (2024 is leap year)
        fixed_time = datetime(2024, 1, 31, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)

        future = udt.add_months()
        assert future.day == 29  # 2024 is a leap year
        assert future.month == 2
        assert future.year == 2024

    def test_add_years(self) -> None:
        """Test add_years adds one year."""
        fixed_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = UserDatetime(now=fixed_time)

        future = udt.add_years()
        # add_years uses timedelta(days=365), so exact year may vary
        assert future.day == 15
        assert future.month == 1
        assert future.year == 2027


class TestGetNow:
    """Tests for get_now function."""

    def test_get_now_returns_utc_by_default(self) -> None:
        """Test get_now returns UTC time by default."""
        now = get_now()
        assert isinstance(now, datetime)
        assert now.tzinfo == ZoneInfo("UTC")

    def test_get_now_with_custom_timezone(self) -> None:
        """Test get_now returns time in custom timezone."""
        now = get_now("America/New_York")
        assert now.tzinfo == ZoneInfo("America/New_York")

    def test_get_now_invalid_timezone(self) -> None:
        """Test get_now raises error for invalid timezone."""
        with pytest.raises(InvalidTimezoneError):
            get_now("Invalid/Timezone")


class TestGetUserDatetime:
    """Tests for get_user_datetime function."""

    def test_returns_user_datetime_instance(self) -> None:
        """Test get_user_datetime returns UserDatetime instance."""
        udt = get_user_datetime("UTC")
        assert isinstance(udt, UserDatetime)

    def test_with_fixed_time(self) -> None:
        """Test get_user_datetime with fixed time for testing."""
        fixed_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        udt = get_user_datetime("UTC", now=fixed_time)
        assert udt.now == fixed_time


class TestGetNextWeekday:
    """Tests for get_next_weekday function."""

    def test_get_next_weekday_returns_correct_date(self) -> None:
        """Test get_next_weekday returns correct next weekday."""
        fixed_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))  # Thursday
        udt = UserDatetime(now=fixed_time)

        saturday = get_next_weekday("saturday", udt)
        assert saturday.day == 17
        assert saturday.month == 1
        assert saturday.year == 2026


class TestResolveRelativeDates:
    """Tests for resolve_relative_dates function."""

    def test_resolve_today(self) -> None:
        """Test resolving 'today' - should be skipped (already shown in local_date)."""
        result = resolve_relative_dates("Let's meet today", "UTC")
        # "today" is intentionally skipped as it's already shown in {local_date} placeholder
        assert result == ""

    def test_resolve_tomorrow(self) -> None:
        """Test resolving 'tomorrow' to ISO date."""
        result = resolve_relative_dates("See you tomorrow", "UTC")
        assert "tomorrow → " in result

    def test_resolve_yesterday(self) -> None:
        """Test resolving 'yesterday' to ISO date."""
        result = resolve_relative_dates("I finished it yesterday", "UTC")
        assert "yesterday → " in result

    def test_resolve_day_after_tomorrow(self) -> None:
        """Test resolving 'day after tomorrow'."""
        result = resolve_relative_dates("Let's meet day after tomorrow", "UTC")
        assert "day after tomorrow → " in result

    def test_resolve_day_before_yesterday(self) -> None:
        """Test resolving 'day before yesterday'."""
        result = resolve_relative_dates("I saw it day before yesterday", "UTC")
        assert "day before yesterday → " in result

    def test_resolve_weekday_name(self) -> None:
        """Test resolving weekday name to next occurrence."""
        result = resolve_relative_dates("Let's meet on Friday", "UTC")
        assert "friday → " in result.lower()

    def test_resolve_next_weekday(self) -> None:
        """Test resolving 'next <weekday>'."""
        result = resolve_relative_dates("See you next Monday", "UTC")
        assert "next monday → " in result.lower()

    def test_resolve_next_week(self) -> None:
        """Test resolving 'next week'."""
        result = resolve_relative_dates("Let's talk next week", "UTC")
        assert "next week → " in result.lower()

    def test_resolve_next_month(self) -> None:
        """Test resolving 'next month'."""
        result = resolve_relative_dates("Starting next month", "UTC")
        assert "next month → " in result.lower()

    def test_resolve_next_year(self) -> None:
        """Test resolving 'next year'."""
        result = resolve_relative_dates("Goals for next year", "UTC")
        assert "next year → " in result.lower()

    def test_resolve_multiple_dates(self) -> None:
        """Test resolving multiple relative dates in one message."""
        result = resolve_relative_dates("Meet tomorrow and next Friday", "UTC")
        assert "tomorrow → " in result
        assert "next friday → " in result.lower()

    def test_resolve_no_dates(self) -> None:
        """Test message with no relative dates returns empty string."""
        result = resolve_relative_dates("Just a regular message", "UTC")
        assert result == ""

    def test_resolve_case_insensitive(self) -> None:
        """Test date resolution is case insensitive."""
        result1 = resolve_relative_dates("Meet on FRIDAY", "UTC")
        result2 = resolve_relative_dates("Meet on friday", "UTC")
        # Both should resolve friday
        assert "friday → " in result1.lower()
        assert "friday → " in result2.lower()

    def test_resolve_with_timezone(self) -> None:
        """Test date resolution respects timezone."""
        # Different timezones may result in different dates depending on current time
        result_utc = resolve_relative_dates("Meet tomorrow", "UTC")
        result_ny = resolve_relative_dates("Meet tomorrow", "America/New_York")

        # Both should resolve, but dates might differ near midnight
        assert "tomorrow → " in result_utc
        assert "tomorrow → " in result_ny


class TestFormatCurrentDate:
    """Tests for format_current_date function."""

    def test_format_current_date_returns_formatted_string(self) -> None:
        """Test format_current_date returns date with day name."""
        result = format_current_date("UTC")
        # Should match YYYY/MM/DD, Weekday pattern
        assert "/" in result
        assert "," in result
        # Should have at least year, month, day, and day name
        parts = result.split(",")
        assert len(parts) == 2
        date_part = parts[0].strip()
        day_name = parts[1].strip()
        # Date part should be YYYY/MM/DD
        assert len(date_part.split("/")) == 3
        # Day name should be a weekday
        assert day_name in [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

    def test_format_current_date_utc_by_default(self) -> None:
        """Test format_current_date uses UTC by default."""
        result = format_current_date()
        assert isinstance(result, str)
        # Should have the format YYYY/MM/DD, Weekday
        assert "/" in result
        assert "," in result


class TestFormatCurrentTime:
    """Tests for format_current_time function."""

    def test_format_current_time_returns_time_string(self) -> None:
        """Test format_current_time returns time in HH:MM format."""
        result = format_current_time("UTC")
        # Should match HH:MM pattern
        assert len(result) == 5
        assert result[2] == ":"

    def test_format_current_time_utc_by_default(self) -> None:
        """Test format_current_time uses UTC by default."""
        result = format_current_time()
        assert isinstance(result, str)
        assert len(result) == 5


class TestDateRange:
    """Tests for DateRange class."""

    def test_date_range_init(self) -> None:
        """Test DateRange initializes with start and end dates."""
        start = datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC"))
        end = datetime(2026, 1, 31, tzinfo=ZoneInfo("UTC"))
        dr = DateRange(start, end)

        assert dr.start == start
        assert dr.end == end

    def test_date_range_repr(self) -> None:
        """Test DateRange string representation."""
        start = datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC"))
        end = datetime(2026, 1, 31, tzinfo=ZoneInfo("UTC"))
        dr = DateRange(start, end)
        repr_str = repr(dr)

        assert "2026-01-01" in repr_str
        assert "2026-01-31" in repr_str


class TestParseRelativeDateRange:
    """Tests for parse_relative_date_range function."""

    def test_parse_last_week(self) -> None:
        """Test parsing 'last week' returns correct date range."""
        result = parse_relative_date_range("Show me last week's data", "UTC")

        assert result is not None
        assert isinstance(result, DateRange)
        assert isinstance(result.start, datetime)
        assert isinstance(result.end, datetime)
        assert result.start < result.end

    def test_parse_this_week(self) -> None:
        """Test parsing 'this week' returns correct date range."""
        result = parse_relative_date_range("What happened this week?", "UTC")

        assert result is not None
        assert isinstance(result, DateRange)
        assert isinstance(result.start, datetime)
        assert isinstance(result.end, datetime)

    def test_parse_last_month(self) -> None:
        """Test parsing 'last month' returns correct date range."""
        result = parse_relative_date_range("Review last month", "UTC")

        assert result is not None
        assert isinstance(result, DateRange)
        assert isinstance(result.start, datetime)
        assert isinstance(result.end, datetime)

    def test_parse_yesterday(self) -> None:
        """Test parsing 'yesterday' returns single-day range."""
        result = parse_relative_date_range("What did I do yesterday?", "UTC")

        assert result is not None
        assert isinstance(result, DateRange)
        # Start and end should be the same day (but potentially different times)
        assert result.start.date() == result.end.date()

    def test_parse_today(self) -> None:
        """Test parsing 'today' returns today's date range."""
        result = parse_relative_date_range("What happened today?", "UTC")

        assert result is not None
        assert isinstance(result, DateRange)
        assert result.start.date() == result.end.date()

    def test_parse_last_n_days(self) -> None:
        """Test parsing 'last N days' returns correct range."""
        result = parse_relative_date_range("Show last 7 days", "UTC")

        assert result is not None
        assert isinstance(result, DateRange)
        assert isinstance(result.start, datetime)
        assert isinstance(result.end, datetime)

    def test_parse_no_date_range(self) -> None:
        """Test message without date range returns None."""
        result = parse_relative_date_range("Just a regular message", "UTC")

        assert result is None

    def test_parse_with_timezone(self) -> None:
        """Test parse_relative_date_range respects timezone."""
        result_utc = parse_relative_date_range("last week", "UTC")
        result_ny = parse_relative_date_range("last week", "America/New_York")

        # Both should return valid date ranges
        assert result_utc is not None
        assert result_ny is not None
        assert isinstance(result_utc, DateRange)
        assert isinstance(result_ny, DateRange)
