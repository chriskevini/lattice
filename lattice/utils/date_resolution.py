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
    (r"\btoday\b", lambda d: d.strftime("%Y-%m-%d")),
    (r"\btomorrow\b", lambda d: (d + timedelta(days=1)).strftime("%Y-%m-%d")),
    (r"\byesterday\b", lambda d: (d - timedelta(days=1)).strftime("%Y-%m-%d")),
    (r"\bday after tomorrow\b", lambda d: (d + timedelta(days=2)).strftime("%Y-%m-%d")),
    (
        r"\bday before yesterday\b",
        lambda d: (d - timedelta(days=2)).strftime("%Y-%m-%d"),
    ),
    (r"\bnext week\b", lambda d: (d + timedelta(weeks=1)).strftime("%Y-%m-%d")),
    (
        r"\bnext month\b",
        lambda d: (d.replace(day=1) + timedelta(days=32))
        .replace(day=1)
        .strftime("%Y-%m-%d"),
    ),
    (r"\bnext year\b", lambda d: (d + timedelta(days=365)).strftime("%Y-%m-%d")),
]


class UserDatetime:
    """Wrapper for timezone-aware datetime operations.

    Ensures consistent timezone handling across the codebase.
    """

    def __init__(self, timezone_str: str | None = None) -> None:
        """Initialize with a timezone.

        Args:
            timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.
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
            except Exception:
                self._tz = ZoneInfo("UTC")
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
        """
        target = WEEKDAYS.get(target_weekday.lower())
        if target is None:
            return self.now

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
        return (current.replace(day=1) + timedelta(days=32)).replace(day=1)

    def add_years(self) -> datetime:
        """Add one year to current time."""
        return self.now + timedelta(days=365)


def _get_user_datetime(timezone_str: str | None = None) -> UserDatetime:
    """Get a UserDatetime instance for the given timezone.

    This is the single entry point for getting timezone-aware datetime.
    """
    return UserDatetime(timezone_str)


def get_next_weekday(target_weekday: str, user_dt: UserDatetime) -> datetime:
    """Get the next occurrence of a weekday after the reference date."""
    return user_dt.next_weekday(target_weekday)


def resolve_relative_dates(message: str, timezone_str: str | None = None) -> str:
    """Extract relative date expressions and resolve them to ISO dates.

    Args:
        message: The user message containing potential relative dates
        timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.

    Returns:
        A hint string like "Friday → 2026-01-10, tomorrow → 2026-01-09"
        or empty string if no relative dates found
    """
    user_dt = _get_user_datetime(timezone_str)
    hints: list[str] = []

    for pattern, resolver in RELATIVE_PATTERNS:
        matches = re.findall(pattern, message, re.IGNORECASE)
        for match in matches:
            try:
                if "next month" in pattern.lower():
                    iso_date = user_dt.add_months().strftime("%Y-%m-%d")
                elif "next year" in pattern.lower():
                    iso_date = user_dt.add_years().strftime("%Y-%m-%d")
                elif "tomorrow" in pattern.lower():
                    iso_date = user_dt.add_days(1).strftime("%Y-%m-%d")
                elif "yesterday" in pattern.lower():
                    iso_date = user_dt.add_days(-1).strftime("%Y-%m-%d")
                elif "day after tomorrow" in pattern.lower():
                    iso_date = user_dt.add_days(2).strftime("%Y-%m-%d")
                elif "day before yesterday" in pattern.lower():
                    iso_date = user_dt.add_days(-2).strftime("%Y-%m-%d")
                elif "next week" in pattern.lower():
                    iso_date = user_dt.add_weeks(1).strftime("%Y-%m-%d")
                else:
                    iso_date = resolver(user_dt.now)
                hints.append(f"{match} → {iso_date}")
            except Exception:  # nosec
                continue

    weekday_matches = re.findall(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        message,
        re.IGNORECASE,
    )
    for match in weekday_matches:
        try:
            next_date = get_next_weekday(match, user_dt)
            iso_date = next_date.strftime("%Y-%m-%d")
            hints.append(f"{match} → {iso_date}")
        except Exception:  # nosec
            continue

    if not hints:
        return ""

    return ", ".join(hints)


def format_current_time(timezone_str: str | None = None) -> str:
    """Format current time with day of week for prompt context.

    Args:
        timezone_str: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.

    Returns:
        Formatted string like "2026/01/08, Thursday"
    """
    user_dt = _get_user_datetime(timezone_str)
    return user_dt.format("%Y/%m/%d, %A")
