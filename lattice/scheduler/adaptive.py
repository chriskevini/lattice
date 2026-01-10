"""Adaptive active hours calculation based on user message patterns.

Analyzes historical message data to determine when the user is most active,
enabling respectful proactive messaging that aligns with their natural schedule.
"""

from datetime import UTC, datetime, timedelta
from typing import TypedDict
from zoneinfo import ZoneInfo

import structlog

from lattice.utils.database import (
    db_pool,
    get_system_health,
    get_user_timezone,
    set_system_health,
)


logger = structlog.get_logger(__name__)

# Default active hours (9 AM - 9 PM) if insufficient data
DEFAULT_ACTIVE_START = 9
DEFAULT_ACTIVE_END = 21

# Minimum messages required for adaptive calculation
MIN_MESSAGES_REQUIRED = 20

# Rolling window for analysis (days)
ANALYSIS_WINDOW_DAYS = 30


class ActiveHoursResult(TypedDict):
    """Result of active hours analysis."""

    start_hour: int  # Hour in user's local timezone (0-23)
    end_hour: int  # Hour in user's local timezone (0-23)
    confidence: float  # 0.0 - 1.0
    sample_size: int  # Number of messages analyzed
    timezone: str  # IANA timezone used


async def calculate_active_hours() -> ActiveHoursResult:
    """Calculate user's active hours from message patterns.

    Analyzes messages from the last 30 days to determine peak activity hours.
    Falls back to default 9 AM - 9 PM if insufficient data.

    Returns:
        ActiveHoursResult with start_hour, end_hour, confidence, and sample_size

    Algorithm:
        1. Get all user messages from last 30 days
        2. Convert timestamps to user's local timezone
        3. Build histogram of messages by hour (0-23)
        4. Find continuous window with highest activity
        5. Ensure window is at least 6 hours wide
        6. Return start/end hours with confidence score
    """
    user_tz = await get_user_timezone()
    tz = ZoneInfo(user_tz)

    # Get messages from last 30 days
    cutoff = get_now(user_tz) - timedelta(days=ANALYSIS_WINDOW_DAYS)

    async with db_pool.pool.acquire() as conn:
        # Get all user messages (not bot messages) from the analysis window
        rows = await conn.fetch(
            """
            SELECT timestamp, user_timezone
            FROM raw_messages
            WHERE is_bot = FALSE
              AND timestamp >= $1
            ORDER BY timestamp ASC
            """,
            cutoff,
        )

    if len(rows) < MIN_MESSAGES_REQUIRED:
        logger.info(
            "Insufficient messages for adaptive hours",
            message_count=len(rows),
            min_required=MIN_MESSAGES_REQUIRED,
            using_defaults=True,
        )
        return ActiveHoursResult(
            start_hour=DEFAULT_ACTIVE_START,
            end_hour=DEFAULT_ACTIVE_END,
            confidence=0.0,
            sample_size=len(rows),
            timezone=user_tz,
        )

    # Build histogram of messages by hour (in user's local timezone)
    hour_counts = [0] * 24

    for row in rows:
        utc_time = row["timestamp"]
        # Convert to user's timezone
        local_time = utc_time.astimezone(tz)
        hour_counts[local_time.hour] += 1

    logger.debug(
        "Message hour distribution",
        hour_counts=hour_counts,
        total_messages=sum(hour_counts),
        timezone=user_tz,
    )

    # Find the continuous window with highest activity
    # Use sliding window to find best 12-hour period
    best_start = 0
    best_count = 0
    window_size = 12  # Look for 12-hour windows

    for start in range(24):
        # Calculate count for window wrapping around midnight
        window_count = sum(hour_counts[(start + i) % 24] for i in range(window_size))
        if window_count > best_count:
            best_count = window_count
            best_start = start

    # The window is 12 hours starting at best_start
    start_hour = best_start
    end_hour = (best_start + window_size) % 24

    # Calculate confidence based on how concentrated the activity is
    total_messages = sum(hour_counts)
    window_messages = best_count
    confidence = window_messages / total_messages if total_messages > 0 else 0.0

    # Adjust confidence based on sample size
    # Full confidence at 100+ messages, scale down for fewer
    size_multiplier = min(1.0, len(rows) / 100.0)
    confidence *= size_multiplier

    logger.info(
        "Calculated adaptive active hours",
        start_hour=start_hour,
        end_hour=end_hour,
        confidence=confidence,
        sample_size=len(rows),
        timezone=user_tz,
        window_messages=window_messages,
        total_messages=total_messages,
    )

    return ActiveHoursResult(
        start_hour=start_hour,
        end_hour=end_hour,
        confidence=confidence,
        sample_size=len(rows),
        timezone=user_tz,
    )


from lattice.utils.date_resolution import get_now


async def is_within_active_hours(check_time: datetime | None = None) -> bool:
    """Check if the given time (or now) is within the user's active hours.

    Args:
        check_time: Time to check. Defaults to current time in user's timezone.

    Returns:
        True if within active hours, False otherwise
    """
    from lattice.utils.database import get_system_health, get_user_timezone

    user_tz = await get_user_timezone()

    # Get stored active hours
    start_hour = int(await get_system_health("active_hours_start") or 9)
    end_hour = int(await get_system_health("active_hours_end") or 21)

    if check_time is None:
        check_time = get_now(user_tz)
    elif check_time.tzinfo is None or check_time.tzinfo == UTC:
        # Convert UTC or naive to user timezone
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(user_tz)
        check_time = check_time.astimezone(tz)

    current_hour = check_time.hour

    # Handle window that wraps around midnight
    if start_hour <= end_hour:
        # Normal case: 9 AM - 9 PM
        within_hours = start_hour <= current_hour < end_hour
    else:
        # Wrap case: 9 PM - 9 AM (night owl)
        within_hours = current_hour >= start_hour or current_hour < end_hour

    logger.debug(
        "Active hours check",
        current_hour=current_hour,
        start_hour=start_hour,
        end_hour=end_hour,
        within_hours=within_hours,
        timezone=user_tz,
    )

    return within_hours


async def update_active_hours() -> ActiveHoursResult:
    """Recalculate and store user's active hours.

    This should be called periodically (e.g., daily) to keep active hours current.

    Returns:
        ActiveHoursResult with updated hours
    """
    result = await calculate_active_hours()

    # Store in system_health
    await set_system_health("active_hours_start", str(result["start_hour"]))
    await set_system_health("active_hours_end", str(result["end_hour"]))
    await set_system_health("active_hours_confidence", str(result["confidence"]))
    # Use UTC for internal last_updated tracking
    await set_system_health("active_hours_last_updated", get_now("UTC").isoformat())

    logger.info(
        "Updated active hours",
        start_hour=result["start_hour"],
        end_hour=result["end_hour"],
        confidence=result["confidence"],
        sample_size=result["sample_size"],
    )

    return result
