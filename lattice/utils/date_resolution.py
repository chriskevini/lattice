"""Date resolution utilities for converting relative time expressions to ISO dates."""

import calendar
import re
from datetime import datetime, timedelta

from zoneinfo import ZoneInfo

import lattice.utils.database

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

RELATIVE_PATTERNS = [
    (r"\bday after tomorrow\b", lambda d: (d + timedelta(days=2)).strftime("%Y-%m-%d")),
    (
        r"\bday before yesterday\b",
        lambda d: (d - timedelta(days=2)).strftime("%Y-%m-%d"),
    ),
    (r"\btomorrow\b", lambda d: (d + timedelta(days=1)).strftime("%Y-%m-%d")),
    (r"\byesterday\b", lambda d: (d - timedelta(days=1)).strftime("%Y-%m-%d")),
    (r"\btonight\b", lambda d: d.strftime("%Y-%m-%d")),
    (r"\btoday\b", lambda d: d.strftime("%Y-%m-%d")),
    (r"\bnext week\b", lambda d: (d + timedelta(weeks=1)).strftime("%Y-%m-%d")),
    (
        r"\bnext month\b",
        lambda d: (d.replace(day=1) + timedelta(days=32))
        .replace(day=1)
        .strftime("%Y-%m-%d"),
    ),
    (r"\bnext year\b", lambda d: (d + timedelta(days=365)).strftime("%Y-%m-%d")),
]

DATE_RANGE_PATTERNS = {
    "last_week": re.compile(r"\blast\s+week\b", re.IGNORECASE),
    "this_week": re.compile(r"\bthis\s+week\b", re.IGNORECASE),
    "last_month": re.compile(r"\blast\s+month\b", re.IGNORECASE),
    "yesterday": re.compile(r"\byesterday\b", re.IGNORECASE),
    "today": re.compile(r"\btoday\b", re.IGNORECASE),
    "last_days": re.compile(r"\blast\s+(\d+)\s+days?\b", re.IGNORECASE),
}


class DateResolutionError(Exception):
    """Base exception for date resolution errors."""

    pass


class InvalidTimezoneError(DateResolutionError):
    """Raised when an invalid timezone string is provided."""

    pass


class InvalidWeekdayError(DateResolutionError):
    """Raised when an invalid weekday name is provided."""

    pass


class UserDatetime:
    """Wrapper for timezone-aware datetime operations.

    Ensures consistent timezone handling across the codebase.
    """

    def __init__(
        self, timezone_str: str | None = None, now: datetime | None = None
    ) -> None:
        """Initialize with a timezone.

        Args:
            timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.
            now: Optional fixed time for testing.

        Raises:
            InvalidTimezoneError: If the timezone string is invalid.
        """
        self._timezone_str = timezone_str or "UTC"
        self._tz: ZoneInfo | None = None
        self._now: datetime | None = now

    @property
    def tz(self) -> ZoneInfo:
        """Lazy-load the timezone ZoneInfo."""
        if self._tz is None:
            try:
                self._tz = ZoneInfo(self._timezone_str)
            except Exception as e:
                raise InvalidTimezoneError(
                    f"Invalid timezone: {self._timezone_str}"
                ) from e
        return self._tz

    @property
    def now(self) -> datetime:
        """Get current time in the user's timezone."""
        if self._now is None:
            self._now = datetime.now(self.tz)
        return self._now

    def next_weekday(self, target_weekday: str) -> datetime:
        """Get the next occurrence of a weekday.

        Args:
            target_weekday: Day name (e.g., 'Friday')

        Returns:
            The next occurrence of the target weekday

        Raises:
            InvalidWeekdayError: If the weekday name is invalid.
        """
        target = WEEKDAYS.get(target_weekday.lower())
        if target is None:
            raise InvalidWeekdayError(
                f"Invalid weekday: {target_weekday}. Expected one of: {list(WEEKDAYS.keys())}"
            )

        current_weekday = self.now.weekday()
        days_ahead = target - current_weekday
        if days_ahead <= 0:
            days_ahead += 7
        return self.now + timedelta(days=days_ahead)

    def format(self, fmt: str) -> str:
        """Format the current time.

        Args:
            fmt: strftime format string

        Returns:
            Formatted datetime string
        """
        return self.now.strftime(fmt)

    def add_days(self, days: int) -> datetime:
        """Add days to current time."""
        return self.now + timedelta(days=days)

    def add_weeks(self, weeks: int) -> datetime:
        """Add weeks to current time."""
        return self.now + timedelta(weeks=weeks)

    def add_months(self) -> datetime:
        """Add one month to current time.

        Handles months with fewer days by clamping the day (e.g., Jan 31 + 1 month
        becomes Feb 28/29 instead of an invalid Feb 31).

        Returns:
            datetime: A new datetime one month ahead.
        """
        current = self.now
        year = current.year
        month = current.month + 1
        if month > 12:
            month = 1
            year += 1
        # Handle months with fewer days (e.g., Feb 30 doesn't exist)
        max_day = calendar.monthrange(year, month)[1]
        new_day = min(current.day, max_day)
        return current.replace(year=year, month=month, day=new_day)

    def add_years(self) -> datetime:
        """Add one year to current time."""
        return self.now + timedelta(days=365)

    @classmethod
    def for_testing(cls, dt: datetime) -> "UserDatetime":
        """Create a UserDatetime instance with a fixed time for testing.

        Args:
            dt: The datetime to use (should include timezone info).

        Returns:
            UserDatetime instance with fixed time.
        """
        instance = cls()
        instance._now = dt
        return instance


def get_now(timezone_str: str | None = None) -> datetime:
    """Get current time in the specified timezone (defaulting to UTC).

    This is the recommended way to get current time across the project
    to ensure consistent timezone handling.

    Args:
        timezone_str: IANA timezone string. Defaults to 'UTC'.

    Returns:
        Timezone-aware datetime object.
    """
    return _get_user_datetime(timezone_str).now


def get_user_datetime(
    timezone_str: str | None = None, now: datetime | None = None
) -> UserDatetime:
    """Get a UserDatetime instance for the given timezone.

    Args:
        timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.
        now: Optional fixed time for testing.

    Returns:
        UserDatetime instance.
    """
    return _get_user_datetime(timezone_str, now=now)


def _get_user_datetime(
    timezone_str: str | None = None, now: datetime | None = None
) -> UserDatetime:
    """Get a UserDatetime instance for the given timezone.

    This is the single entry point for getting timezone-aware datetime.

    Raises:
        InvalidTimezoneError: If the timezone string is invalid.
    """
    return UserDatetime(timezone_str, now=now)


def get_next_weekday(target_weekday: str, user_dt: UserDatetime) -> datetime:
    """Get the next occurrence of a weekday after the reference date.

    Args:
        target_weekday: Day name (e.g., 'Friday')
        user_dt: UserDatetime instance

    Returns:
        The next occurrence of the target weekday

    Raises:
        InvalidWeekdayError: If the weekday name is invalid.
    """
    return user_dt.next_weekday(target_weekday)


def _normalize_message(message: str) -> str:
    """Normalize date-related phrases for consistent parsing.

    Converts variations like "wednesday next week" to "next wednesday".

    Args:
        message: Original user message

    Returns:
        Normalized message
    """
    normalized = re.sub(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+next\s+week\b",
        lambda m: f"next {m.group(1)}",
        message,
        flags=re.IGNORECASE,
    )
    return normalized


def resolve_relative_dates(message: str, timezone_str: str | None = None) -> str:
    """Extract relative date expressions and resolve them to ISO dates.

    Handles:
    - Relative days: today, tomorrow, yesterday, day after/before tomorrow/yesterday
    - Weekdays: monday, tuesday, etc. (next occurrence)
    - "next <weekday>": weekday + 7 days (next week)
    - Periods: next week, next month, next year

    Args:
        message: The user message containing potential relative dates
        timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to user's timezone from memory.

    Returns:
        A hint string like "Friday → 2026-01-10, tomorrow → 2026-01-09"
        or empty string if no relative dates found
    """
    if timezone_str is None:
        timezone_str = lattice.utils.database._user_timezone_cache or "UTC"
    user_dt = _get_user_datetime(timezone_str)
    hints: list[str] = []

    normalized_msg = _normalize_message(message)
    normalized_lower = normalized_msg.lower()

    # Match "next <weekday>" pattern (next wednesday, next friday)
    next_weekday_pattern = r"\bnext\s+(?:the\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    next_weekday_matches = list(re.finditer(next_weekday_pattern, normalized_lower))

    # Match all weekdays to exclude those already captured by "next <weekday>"
    all_weekday_pattern = (
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    )
    all_weekday_matches = list(re.finditer(all_weekday_pattern, normalized_lower))

    # Track which weekday occurrences were captured by "next <weekday>"
    # Use (start, end) ranges to track positions
    captured_weekday_ranges: list[tuple[int, int]] = []
    for match in next_weekday_matches:
        captured_weekday_ranges.append(match.span())

    # Process "next <weekday>" → +7 days
    for match in next_weekday_matches:
        day = match.group(1)
        try:
            next_date = get_next_weekday(day, user_dt) + timedelta(days=7)
            iso_date = next_date.strftime("%Y-%m-%d")
            hints.append(f"next {day} → {iso_date}")
        except InvalidWeekdayError:
            continue

    # Process plain weekdays (not part of "next <weekday>")
    for match in all_weekday_matches:
        day = match.group(1)
        match_start, match_end = match.span()

        # Skip if this weekday is contained within a "next <weekday>" match
        is_contained = any(
            r_start <= match_start and match_end <= r_end
            for r_start, r_end in captured_weekday_ranges
        )
        if is_contained:
            continue

        try:
            next_date = get_next_weekday(day, user_dt)
            iso_date = next_date.strftime("%Y-%m-%d")
            hints.append(f"{day} → {iso_date}")
        except InvalidWeekdayError:
            continue

    # Process relative date patterns
    # Sort patterns by pattern length (longer first) to avoid substring matching issues
    sorted_patterns = sorted(RELATIVE_PATTERNS, key=lambda x: len(x[0]), reverse=True)

    # Track which parts of the message have been consumed by longer patterns
    consumed_ranges: list[tuple[int, int]] = []

    for pattern, resolver in sorted_patterns:
        matches = list(re.finditer(pattern, normalized_lower))
        for match in matches:
            match_start, match_end = match.span()

            # Check if this match overlaps with already consumed ranges
            is_consumed = False
            for r_start, r_end in consumed_ranges:
                # Check for overlap: two ranges [a, b) and [c, d) overlap if a < d and c < b
                if match_start < r_end and r_start < match_end:
                    is_consumed = True
                    break
            if is_consumed:
                continue

            match_text = match.group(0)
            try:
                if "next month" in pattern.lower():
                    iso_date = user_dt.add_months().strftime("%Y-%m-%d")
                elif "next year" in pattern.lower():
                    iso_date = user_dt.add_years().strftime("%Y-%m-%d")
                elif "day after tomorrow" in pattern.lower():
                    iso_date = user_dt.add_days(2).strftime("%Y-%m-%d")
                elif "day before yesterday" in pattern.lower():
                    iso_date = user_dt.add_days(-2).strftime("%Y-%m-%d")
                elif "tomorrow" in pattern.lower():
                    iso_date = user_dt.add_days(1).strftime("%Y-%m-%d")
                elif "yesterday" in pattern.lower():
                    iso_date = user_dt.add_days(-1).strftime("%Y-%m-%d")
                elif "tonight" in pattern.lower():
                    iso_date = user_dt.now.strftime("%Y-%m-%d")
                elif "next week" in pattern.lower():
                    iso_date = user_dt.add_weeks(1).strftime("%Y-%m-%d")
                elif "today" in pattern.lower():
                    iso_date = user_dt.now.strftime("%Y-%m-%d")
                else:
                    iso_date = resolver(user_dt.now)
                # Skip "today" and "tonight" hints - already shown in {local_date}
                if match_text.lower() in ("today", "tonight"):
                    consumed_ranges.append((match_start, match_end))
                    continue
                hints.append(f"{match_text} → {iso_date}")
                consumed_ranges.append((match_start, match_end))
            except (InvalidTimezoneError, InvalidWeekdayError, ValueError):
                continue

    if not hints:
        return ""

    return ", ".join(hints)


def format_current_date(timezone_str: str | None = None) -> str:
    """Format current date with day of week for prompt context.

    Args:
        timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to user's timezone from memory.

    Returns:
        Formatted string like "2026/01/08, Thursday"
    """
    if timezone_str is None:
        timezone_str = lattice.utils.database._user_timezone_cache or "UTC"
    return _get_user_datetime(timezone_str).now.strftime("%Y/%m/%d, %A")


def format_current_time(timezone_str: str | None = None) -> str:
    """Format current time for time-sensitive proactive decisions.

    Args:
        timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to user's timezone from memory.

    Returns:
        Formatted string like "14:30"
    """
    if timezone_str is None:
        timezone_str = lattice.utils.database._user_timezone_cache or "UTC"
    return _get_user_datetime(timezone_str).now.strftime("%H:%M")


class DateRange:
    """Represents a date range with start and end datetimes."""

    def __init__(self, start: datetime, end: datetime) -> None:
        """Initialize a date range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
        """
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        """Return a string representation of the DateRange."""
        return f"DateRange(start={self.start.isoformat()}, end={self.end.isoformat()})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on start and end datetimes.

        Args:
            other: Object to compare against

        Returns:
            True if both start and end datetimes are equal
        """
        if not isinstance(other, DateRange):
            return NotImplemented
        return self.start == other.start and self.end == other.end


def parse_relative_date_range(
    message: str, timezone_str: str | None = None
) -> DateRange | None:
    """Parse relative date expressions to determine a date range for queries.

    Handles:
    - "last week": Monday of previous week to Sunday of previous week
    - "this week": Monday of current week to now
    - "last month": First to last day of previous month
    - "yesterday": Previous day
    - "today": Current day from midnight to now
    - "last X days": Past X days

    Args:
        message: The user message containing relative date expressions
        timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to user's timezone from memory.

    Returns:
        DateRange object with start and end datetimes, or None if no date range detected
    """
    if timezone_str is None:
        timezone_str = lattice.utils.database._user_timezone_cache or "UTC"
    user_dt = _get_user_datetime(timezone_str)
    now = user_dt.now
    message_lower = message.lower()

    current_weekday = now.weekday()
    days_since_monday = current_weekday if current_weekday != 0 else 7
    start_of_week = now - timedelta(days=days_since_monday)
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)

    if DATE_RANGE_PATTERNS["last_week"].search(message_lower):
        last_week_start = start_of_week - timedelta(weeks=1)
        last_week_end = last_week_start + timedelta(days=6)
        last_week_end = last_week_end.replace(hour=23, minute=59, second=59)
        return DateRange(start=last_week_start, end=last_week_end)

    if DATE_RANGE_PATTERNS["this_week"].search(message_lower):
        return DateRange(start=start_of_week, end=now)

    if DATE_RANGE_PATTERNS["last_month"].search(message_lower):
        if now.month == 1:
            last_month = now.replace(year=now.year - 1, month=12)
        else:
            last_month = now.replace(month=now.month - 1)
        _, last_day = calendar.monthrange(last_month.year, last_month.month)
        last_month_start = last_month.replace(day=1, hour=0, minute=0, second=0)
        last_month_end = last_month.replace(day=last_day, hour=23, minute=59, second=59)
        return DateRange(start=last_month_start, end=last_month_end)

    if DATE_RANGE_PATTERNS["yesterday"].search(message_lower):
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        yesterday_end = yesterday_start.replace(hour=23, minute=59, second=59)
        return DateRange(start=yesterday_start, end=yesterday_end)

    if DATE_RANGE_PATTERNS["today"].search(message_lower):
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return DateRange(start=today_start, end=now)

    last_days_match = DATE_RANGE_PATTERNS["last_days"].search(message_lower)
    if last_days_match:
        num_days = int(last_days_match.group(1))
        range_start = now - timedelta(days=num_days)
        range_start = range_start.replace(hour=0, minute=0, second=0)
        return DateRange(start=range_start, end=now)

    return None
