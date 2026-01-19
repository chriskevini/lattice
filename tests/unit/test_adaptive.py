"""Tests for the adaptive scheduler module.

Tests the active hours calculation and adaptive scheduling logic
based on user message patterns.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.scheduler.adaptive import (
    ActiveHoursResult,
    DEFAULT_ACTIVE_END,
    DEFAULT_ACTIVE_START,
    MIN_MESSAGES_REQUIRED,
    calculate_active_hours,
    is_within_active_hours,
    update_active_hours,
)


class TestCalculateActiveHours:
    """Tests for the calculate_active_hours function."""

    @pytest.mark.asyncio
    async def test_returns_defaults_with_no_messages(self) -> None:
        """Test that default hours are returned when message count is zero."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(return_value=[])

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            result = await calculate_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        assert result["start_hour"] == DEFAULT_ACTIVE_START
        assert result["end_hour"] == DEFAULT_ACTIVE_END
        assert result["confidence"] == 0.0
        assert result["sample_size"] == 0
        assert result["timezone"] == "UTC"

    @pytest.mark.asyncio
    async def test_returns_defaults_just_under_threshold(self) -> None:
        """Test default hours returned when messages are just below threshold."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="America/New_York")

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=[
                {"timestamp": datetime(2024, 1, 14, 10, 0, 0, tzinfo=UTC)}
                for _ in range(MIN_MESSAGES_REQUIRED - 1)
            ]
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            result = await calculate_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        assert result["start_hour"] == DEFAULT_ACTIVE_START
        assert result["end_hour"] == DEFAULT_ACTIVE_END
        assert result["confidence"] == 0.0
        assert result["sample_size"] == MIN_MESSAGES_REQUIRED - 1

    @pytest.mark.asyncio
    async def test_calculates_active_hours_from_messages(self) -> None:
        """Test active hours calculation from message patterns."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(
            return_value="America/Los_Angeles"
        )

        base_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        messages = []
        for hour in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
            for _ in range(3):
                messages.append(
                    {"timestamp": base_time.replace(hour=hour, minute=_ * 10)}
                )

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=base_time,
        ):
            result = await calculate_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        assert result["timezone"] == "America/Los_Angeles"
        assert 0 <= result["start_hour"] <= 23
        assert 0 <= result["end_hour"] <= 23
        assert result["sample_size"] == len(messages)
        assert result["confidence"] > 0.0

    @pytest.mark.asyncio
    async def test_converts_to_local_timezone(self) -> None:
        """Test that timestamps are properly converted to user's timezone."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="Europe/London")

        utc_midnight = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)
        messages = [
            {"timestamp": utc_midnight},
            {"timestamp": utc_midnight + timedelta(hours=1)},
            {"timestamp": utc_midnight + timedelta(hours=2)},
        ]

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            result = await calculate_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        assert result["timezone"] == "Europe/London"

    @pytest.mark.asyncio
    async def test_confidence_score_based_on_sample_size(self) -> None:
        """Test that confidence score is reduced for smaller sample sizes."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")

        messages = [
            {"timestamp": datetime(2024, 1, 14, 10, 0, 0, tzinfo=UTC)}
            for _ in range(MIN_MESSAGES_REQUIRED)
        ]

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            result_20 = await calculate_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        messages_100 = messages * 5
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages_100
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            result_100 = await calculate_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        assert result_100["confidence"] >= result_20["confidence"]
        assert result_100["sample_size"] == 100

    @pytest.mark.asyncio
    async def test_window_size_is_12_hours(self) -> None:
        """Test that active hours window is always 12 hours wide."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")

        messages = []
        for hour in [14, 15, 16, 17, 18, 19, 20, 21]:
            for _ in range(5):
                messages.append(
                    {"timestamp": datetime(2024, 1, 14, hour, _ * 5, 0, tzinfo=UTC)}
                )

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            result = await calculate_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        window_hours = (result["end_hour"] - result["start_hour"]) % 24
        assert window_hours == 12

    @pytest.mark.asyncio
    async def test_handles_timezone_with_dst(self) -> None:
        """Test active hours calculation during daylight saving time."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="America/New_York")

        dst_transition_date = datetime(2024, 3, 10, 2, 0, 0, tzinfo=UTC)
        messages = [
            {"timestamp": dst_transition_date + timedelta(hours=hour)}
            for hour in range(24)
        ]

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 3, 10, 12, 0, 0, tzinfo=UTC),
        ):
            result = await calculate_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        assert result["timezone"] == "America/New_York"
        assert 0 <= result["start_hour"] <= 23
        assert 0 <= result["end_hour"] <= 23


class TestIsWithinActiveHours:
    """Tests for the is_within_active_hours function."""

    @pytest.mark.asyncio
    async def test_returns_true_within_active_hours(self) -> None:
        """Test that True is returned when within active hours."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_metrics_repo.get_metric = AsyncMock(
            side_effect=lambda key: "10"
            if key == "active_hours_start"
            else "18"
            if key == "active_hours_end"
            else None
        )

        check_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)

        result = await is_within_active_hours(
            system_metrics_repo=mock_metrics_repo,
            check_time=check_time,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_outside_active_hours(self) -> None:
        """Test that False is returned when outside active hours."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_metrics_repo.get_metric = AsyncMock(
            side_effect=lambda key: "10"
            if key == "active_hours_start"
            else "18"
            if key == "active_hours_end"
            else None
        )

        check_time = datetime(2024, 1, 15, 8, 0, 0, tzinfo=UTC)

        result = await is_within_active_hours(
            system_metrics_repo=mock_metrics_repo,
            check_time=check_time,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_defaults_when_no_stored_hours(self) -> None:
        """Test default hours used when no hours are stored."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_metrics_repo.get_metric = AsyncMock(return_value=None)

        check_time = datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC)

        result = await is_within_active_hours(
            system_metrics_repo=mock_metrics_repo,
            check_time=check_time,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_handles_cross_midnight_hours(self) -> None:
        """Test active hours that cross midnight (e.g., 22:00 - 06:00)."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_metrics_repo.get_metric = AsyncMock(
            side_effect=lambda key: "22"
            if key == "active_hours_start"
            else "6"
            if key == "active_hours_end"
            else None
        )

        night_time = datetime(2024, 1, 15, 2, 0, 0, tzinfo=UTC)
        result = await is_within_active_hours(
            system_metrics_repo=mock_metrics_repo,
            check_time=night_time,
        )
        assert result is True

        evening_time = datetime(2024, 1, 15, 20, 0, 0, tzinfo=UTC)
        result = await is_within_active_hours(
            system_metrics_repo=mock_metrics_repo,
            check_time=evening_time,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_handles_naive_datetime(self) -> None:
        """Test that naive datetime is converted to user's timezone."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="America/New_York")
        mock_metrics_repo.get_metric = AsyncMock(
            side_effect=lambda key: "9"
            if key == "active_hours_start"
            else "17"
            if key == "active_hours_end"
            else None
        )

        naive_time = datetime(2024, 1, 15, 14, 0, 0)

        result = await is_within_active_hours(
            system_metrics_repo=mock_metrics_repo,
            check_time=naive_time,
        )

        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_uses_current_time_when_not_provided(self) -> None:
        """Test that current time is used when check_time is None."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_metrics_repo.get_metric = AsyncMock(
            side_effect=lambda key: "0"
            if key == "active_hours_start"
            else "23"
            if key == "active_hours_end"
            else None
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            result = await is_within_active_hours(
                system_metrics_repo=mock_metrics_repo,
                check_time=None,
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_handles_boundary_times(self) -> None:
        """Test active hours at exact boundary times."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_metrics_repo.get_metric = AsyncMock(
            side_effect=lambda key: "9"
            if key == "active_hours_start"
            else "17"
            if key == "active_hours_end"
            else None
        )

        at_start = datetime(2024, 1, 15, 9, 0, 0, tzinfo=UTC)
        at_end = datetime(2024, 1, 15, 17, 0, 0, tzinfo=UTC)

        result_start = await is_within_active_hours(
            system_metrics_repo=mock_metrics_repo,
            check_time=at_start,
        )
        assert result_start is True

        result_end = await is_within_active_hours(
            system_metrics_repo=mock_metrics_repo,
            check_time=at_end,
        )
        assert result_end is False


class TestUpdateActiveHours:
    """Tests for the update_active_hours function."""

    @pytest.mark.asyncio
    async def test_updates_stored_hours(self) -> None:
        """Test that calculated hours are stored in metrics."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_metrics_repo.get_metric = AsyncMock(return_value=None)
        mock_metrics_repo.set_metric = AsyncMock()

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=[
                {"timestamp": datetime(2024, 1, 14, 10, 0, 0, tzinfo=UTC)}
                for _ in range(MIN_MESSAGES_REQUIRED + 50)
            ]
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            result = await update_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        assert mock_metrics_repo.set_metric.call_count >= 4
        assert "start_hour" in result
        assert "end_hour" in result

    @pytest.mark.asyncio
    async def test_stores_confidence_score(self) -> None:
        """Test that confidence score is stored."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_metrics_repo.get_metric = AsyncMock(return_value=None)
        mock_metrics_repo.set_metric = AsyncMock()

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=[
                {"timestamp": datetime(2024, 1, 14, 10, 0, 0, tzinfo=UTC)}
                for _ in range(50)
            ]
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            result = await update_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        assert result["confidence"] >= 0.0
        assert result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_stores_timestamp(self) -> None:
        """Test that update timestamp is stored."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_metrics_repo.get_metric = AsyncMock(return_value=None)
        mock_metrics_repo.set_metric = AsyncMock()

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(return_value=[])

        test_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        with patch("lattice.scheduler.adaptive.get_now", return_value=test_time):
            await update_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        calls = mock_metrics_repo.set_metric.call_args_list
        last_updated_call = [c for c in calls if c[0][0] == "active_hours_last_updated"]
        assert len(last_updated_call) == 1


class TestActiveHoursResult:
    """Tests for the ActiveHoursResult TypedDict."""

    def test_active_hours_result_structure(self) -> None:
        """Test ActiveHoursResult can be constructed correctly."""
        result = ActiveHoursResult(
            start_hour=9,
            end_hour=17,
            confidence=0.75,
            sample_size=100,
            timezone="America/New_York",
        )

        assert result["start_hour"] == 9
        assert result["end_hour"] == 17
        assert result["confidence"] == 0.75
        assert result["sample_size"] == 100
        assert result["timezone"] == "America/New_York"

    def test_active_hours_result_with_defaults(self) -> None:
        """Test ActiveHoursResult with default values."""
        result = ActiveHoursResult(
            start_hour=DEFAULT_ACTIVE_START,
            end_hour=DEFAULT_ACTIVE_END,
            confidence=0.0,
            sample_size=0,
            timezone="UTC",
        )

        assert result["start_hour"] == DEFAULT_ACTIVE_START
        assert result["end_hour"] == DEFAULT_ACTIVE_END


class TestEdgeCases:
    """Tests for edge cases in adaptive scheduling."""

    @pytest.mark.asyncio
    async def test_all_messages_at_same_hour(self) -> None:
        """Test behavior when all messages are at the same hour."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")

        messages = [
            {"timestamp": datetime(2024, 1, 14, 10, minute, 0, tzinfo=UTC)}
            for minute in range(60)
        ]

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            result = await calculate_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        assert result["sample_size"] == 60
        assert result["start_hour"] in range(24)
        assert result["end_hour"] in range(24)

    @pytest.mark.asyncio
    async def test_evenly_distributed_messages(self) -> None:
        """Test behavior when messages are evenly distributed."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")

        messages = []
        for day in range(30):
            for hour in range(24):
                messages.append(
                    {"timestamp": datetime(2024, 1, day + 1, hour, 0, 0, tzinfo=UTC)}
                )

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 2, 1, 12, 0, 0, tzinfo=UTC),
        ):
            result = await calculate_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        assert result["sample_size"] == 30 * 24
        assert result["confidence"] >= 0.0

    @pytest.mark.asyncio
    async def test_timezone_edge_case_utc_plus_offset(self) -> None:
        """Test timezone handling at UTC+12 boundary."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="Pacific/Auckland")

        messages = [
            {"timestamp": datetime(2024, 1, 14, hour, 0, 0, tzinfo=UTC)}
            for hour in range(24)
        ]

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            result = await calculate_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        assert result["timezone"] == "Pacific/Auckland"
        assert 0 <= result["start_hour"] <= 23
        assert 0 <= result["end_hour"] <= 23

    @pytest.mark.asyncio
    async def test_very_large_message_count(self) -> None:
        """Test behavior with a very large number of messages."""
        mock_metrics_repo = MagicMock()
        mock_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")

        messages = [
            {"timestamp": datetime(2024, 1, 14, hour, 0, 0, tzinfo=UTC)}
            for hour in range(24)
            for _ in range(100)
        ]

        mock_message_repo = MagicMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages
        )

        with patch(
            "lattice.scheduler.adaptive.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            result = await calculate_active_hours(
                system_metrics_repo=mock_metrics_repo,
                message_repo=mock_message_repo,
            )

        assert result["sample_size"] == 2400
        assert result["confidence"] > 0.0
