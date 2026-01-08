"""Date resolution utilities for converting relative time expressions to ISO dates."""

import re
from datetime import datetime, timedelta

from zoneinfo import ZoneInfo

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

    def __init__(self, timezone_str: str | None = None) -> None:
        """Initialize with a timezone.

        Args:
            timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.

        Raises:
            InvalidTimezoneError: If the timezone string is invalid.
        """
        self._timezone_str = timezone_str or "UTC"
        self._tz: ZoneInfo | None = None
        self._now: datetime | None = None

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
        """Add one month to current time."""
        current = self.now
        year = current.year
        month = current.month + 1
        if month > 12:
            month = 1
            year += 1
        # Handle months with fewer days (e.g., Feb 30 doesn't exist)
        import calendar

        max_day = calendar.monthrange(year, month)[1]
        new_day = min(current.day, max_day)
        return current.replace(year=year, month=month, day=new_day)

    def add_years(self) -> datetime:
        """Add one year to current time."""
        return self.now + timedelta(days=365)


def _get_user_datetime(timezone_str: str | None = None) -> UserDatetime:
    """Get a UserDatetime instance for the given timezone.

    This is the single entry point for getting timezone-aware datetime.

    Raises:
        InvalidTimezoneError: If the timezone string is invalid.
    """
    return UserDatetime(timezone_str)


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
        timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.

    Returns:
        A hint string like "Friday → 2026-01-10, tomorrow → 2026-01-09"
        or empty string if no relative dates found
    """
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
            except Exception:  # nosec[B112]
                continue

    if not hints:
        return ""

    return ", ".join(hints)


def format_current_date(timezone_str: str | None = None) -> str:
    """Format current date with day of week for prompt context.

    Args:
        timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.

    Returns:
        Formatted string like "2026/01/08, Thursday"
    """
    user_dt = _get_user_datetime(timezone_str)
    return user_dt.format("%Y/%m/%d, %A")


def format_current_time(timezone_str: str | None = None) -> str:
    """Format current time for time-sensitive proactive decisions.

    Args:
        timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.

    Returns:
        Formatted string like "14:30"
    """
    user_dt = _get_user_datetime(timezone_str)
    return user_dt.format("%H:%M")
