"""Adaptive active hours calculation based on user message patterns.

Analyzes historical message data to determine when the user is most active,
enabling respectful proactive messaging that aligns with their natural schedule.
"""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, TypedDict
from zoneinfo import ZoneInfo

import structlog

from lattice.utils.date_resolution import get_now

if TYPE_CHECKING:
    from lattice.memory.repositories import SystemMetricsRepository, MessageRepository


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


async def calculate_active_hours(
    system_metrics_repo: "SystemMetricsRepository",
    message_repo: "MessageRepository",
) -> ActiveHoursResult:
    """Calculate user's active hours from message patterns.

    Analyzes messages from the last 30 days to determine peak activity hours.
    Falls back to default 9 AM - 9 PM if insufficient data.

    Args:
        system_metrics_repo: System metrics repository for timezone lookup
        message_repo: Message repository for fetching message history

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
    user_tz = await system_metrics_repo.get_user_timezone()
    tz = ZoneInfo(user_tz)

    cutoff = get_now(user_tz) - timedelta(days=ANALYSIS_WINDOW_DAYS)

    rows = await message_repo.get_message_timestamps_since(
        since=cutoff,
        is_bot=False,
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

    hour_counts = [0] * 24

    for row in rows:
        utc_time = row["timestamp"]
        local_time = utc_time.astimezone(tz)
        hour_counts[local_time.hour] += 1

    logger.debug(
        "Message hour distribution",
        hour_counts=hour_counts,
        total_messages=sum(hour_counts),
        timezone=user_tz,
    )

    best_start = 0
    best_count = 0
    window_size = 12

    for start in range(24):
        window_count = sum(hour_counts[(start + i) % 24] for i in range(window_size))
        if window_count > best_count:
            best_count = window_count
            best_start = start

    start_hour = best_start
    end_hour = (best_start + window_size) % 24

    total_messages = sum(hour_counts)
    window_messages = best_count
    confidence = window_messages / total_messages if total_messages > 0 else 0.0

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


async def is_within_active_hours(
    system_metrics_repo: "SystemMetricsRepository",
    check_time: datetime | None = None,
) -> bool:
    """Check if the given time (or now) is within the user's active hours.

    Args:
        system_metrics_repo: System metrics repository for timezone and hours lookup
        check_time: Time to check. Defaults to current time in user's timezone.

    Returns:
        True if within active hours, False otherwise
    """
    user_tz = await system_metrics_repo.get_user_timezone()

    start_hour_raw = await system_metrics_repo.get_metric("active_hours_start")
    end_hour_raw = await system_metrics_repo.get_metric("active_hours_end")

    start_hour = int(start_hour_raw) if start_hour_raw else DEFAULT_ACTIVE_START
    end_hour = int(end_hour_raw) if end_hour_raw else DEFAULT_ACTIVE_END

    if check_time is None:
        check_time = get_now(user_tz)
    elif check_time.tzinfo is None or check_time.tzinfo == UTC:
        tz = ZoneInfo(user_tz)
        check_time = check_time.astimezone(tz)

    current_hour = check_time.hour

    if start_hour <= end_hour:
        within_hours = start_hour <= current_hour < end_hour
    else:
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


async def update_active_hours(
    system_metrics_repo: "SystemMetricsRepository",
    message_repo: "MessageRepository",
) -> ActiveHoursResult:
    """Recalculate and store user's active hours.

    This should be called periodically (e.g., daily) to keep active hours current.

    Args:
        system_metrics_repo: System metrics repository for storing active hours
        message_repo: Message repository for fetching message history

    Returns:
        ActiveHoursResult with updated hours
    """
    result = await calculate_active_hours(
        system_metrics_repo=system_metrics_repo,
        message_repo=message_repo,
    )

    await system_metrics_repo.set_metric(
        "active_hours_start", str(result["start_hour"])
    )
    await system_metrics_repo.set_metric("active_hours_end", str(result["end_hour"]))
    await system_metrics_repo.set_metric(
        "active_hours_confidence", str(result["confidence"])
    )
    await system_metrics_repo.set_metric(
        "active_hours_last_updated", get_now("UTC").isoformat()
    )

    logger.info(
        "Updated active hours",
        start_hour=result["start_hour"],
        end_hour=result["end_hour"],
        confidence=result["confidence"],
        sample_size=result["sample_size"],
    )

    return result
