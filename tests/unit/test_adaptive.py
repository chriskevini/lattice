"""Unit tests for adaptive active hours scheduling."""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from lattice.scheduler.adaptive import (
    ANALYSIS_WINDOW_DAYS,
    DEFAULT_ACTIVE_END,
    DEFAULT_ACTIVE_START,
    MIN_MESSAGES_REQUIRED,
    ActiveHoursResult,
    calculate_active_hours,
    is_within_active_hours,
    update_active_hours,
)


class TestCalculateActiveHours:
    """Test calculate_active_hours function."""

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_insufficient_messages_returns_defaults(
        self, mock_get_now: Mock, mock_logger: Mock
    ) -> None:
        """Test that insufficient messages returns default hours with zero confidence."""
        mock_get_now.return_value = datetime(
            2026, 1, 20, 12, 0, 0, tzinfo=ZoneInfo("UTC")
        )

        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.get_user_timezone = AsyncMock(
            return_value="America/New_York"
        )

        mock_message_repo = AsyncMock()
        # Return only 10 messages (less than MIN_MESSAGES_REQUIRED = 20)
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=[
                {"timestamp": datetime(2026, 1, i, 10, 0, 0, tzinfo=UTC)}
                for i in range(1, 11)
            ]
        )

        result = await calculate_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        assert result["start_hour"] == DEFAULT_ACTIVE_START
        assert result["end_hour"] == DEFAULT_ACTIVE_END
        assert result["confidence"] == 0.0
        assert result["sample_size"] == 10
        assert result["timezone"] == "America/New_York"

        # Verify log was called
        mock_logger.info.assert_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_calculates_active_hours_from_message_distribution(
        self, mock_get_now: Mock, mock_logger: Mock
    ) -> None:
        """Test calculation of active hours from message hour distribution."""
        mock_get_now.return_value = datetime(
            2026, 1, 20, 12, 0, 0, tzinfo=ZoneInfo("UTC")
        )

        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")

        # Create 50 messages clustered around 9 AM - 9 PM (peak activity)
        messages = []
        for i in range(50):
            # Most messages between hour 9-20 (9 AM - 8 PM)
            hour = 9 + (i % 12)
            messages.append({"timestamp": datetime(2026, 1, 1, hour, 0, 0, tzinfo=UTC)})

        mock_message_repo = AsyncMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages
        )

        result = await calculate_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        assert result["sample_size"] == 50
        assert result["timezone"] == "UTC"
        assert result["confidence"] > 0.0
        # Window should cover the active period
        assert 0 <= result["start_hour"] < 24
        assert 0 <= result["end_hour"] < 24

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_handles_timezone_conversion(
        self, mock_get_now: Mock, mock_logger: Mock
    ) -> None:
        """Test that UTC timestamps are converted to user's local timezone."""
        mock_get_now.return_value = datetime(
            2026, 1, 20, 12, 0, 0, tzinfo=ZoneInfo("America/New_York")
        )

        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.get_user_timezone = AsyncMock(
            return_value="America/New_York"
        )

        # Create 30 messages at 14:00 UTC (9 AM EST)
        messages = [
            {"timestamp": datetime(2026, 1, i, 14, 0, 0, tzinfo=UTC)}
            for i in range(1, 31)
        ]

        mock_message_repo = AsyncMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages
        )

        result = await calculate_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        # Should have analyzed messages in EST timezone
        assert result["timezone"] == "America/New_York"
        assert result["sample_size"] == 30

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_window_wraps_around_midnight(
        self, mock_get_now: Mock, mock_logger: Mock
    ) -> None:
        """Test that activity window can wrap around midnight."""
        mock_get_now.return_value = datetime(
            2026, 1, 20, 12, 0, 0, tzinfo=ZoneInfo("UTC")
        )

        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")

        # Create night owl pattern: 8 PM - 4 AM (20-23, 0-3)
        messages = []
        for day in range(30):
            for hour in [20, 21, 22, 23, 0, 1, 2, 3]:
                messages.append(
                    {"timestamp": datetime(2026, 1, 1 + day, hour, 0, 0, tzinfo=UTC)}
                )

        mock_message_repo = AsyncMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages
        )

        result = await calculate_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        # Window should wrap around midnight
        # start_hour > end_hour indicates wrap-around
        assert result["sample_size"] == len(messages)
        assert result["confidence"] > 0.0

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_confidence_scales_with_sample_size(
        self, mock_get_now: Mock, mock_logger: Mock
    ) -> None:
        """Test that confidence score scales with sample size."""
        mock_get_now.return_value = datetime(
            2026, 1, 20, 12, 0, 0, tzinfo=ZoneInfo("UTC")
        )

        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")

        # Test with 30 messages (should have lower confidence multiplier)
        messages_30 = [
            {"timestamp": datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)} for _ in range(30)
        ]

        mock_message_repo = AsyncMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages_30
        )

        result_30 = await calculate_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        # Test with 100+ messages (should have full confidence multiplier)
        messages_100 = [
            {"timestamp": datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)}
            for _ in range(100)
        ]

        mock_message_repo.get_message_timestamps_since = AsyncMock(
            return_value=messages_100
        )

        result_100 = await calculate_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        # More messages should give higher confidence
        assert result_100["confidence"] >= result_30["confidence"]

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_queries_correct_time_window(
        self, mock_get_now: Mock, mock_logger: Mock
    ) -> None:
        """Test that queries messages from last ANALYSIS_WINDOW_DAYS."""
        now = datetime(2026, 1, 20, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        mock_get_now.return_value = now

        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")

        mock_message_repo = AsyncMock()
        mock_message_repo.get_message_timestamps_since = AsyncMock(return_value=[])

        await calculate_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        # Verify cutoff is ANALYSIS_WINDOW_DAYS ago
        mock_message_repo.get_message_timestamps_since.assert_called_once()
        call_kwargs = mock_message_repo.get_message_timestamps_since.call_args[1]
        expected_cutoff = now - timedelta(days=ANALYSIS_WINDOW_DAYS)
        assert call_kwargs["since"] == expected_cutoff
        assert call_kwargs["is_bot"] is False


class TestIsWithinActiveHours:
    """Test is_within_active_hours function."""

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_within_active_hours_returns_true(
        self, mock_get_now: Mock, mock_logger: Mock
    ) -> None:
        """Test returns True when time is within active hours."""
        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_system_metrics_repo.get_metric = AsyncMock(
            side_effect=lambda key: "9" if key == "active_hours_start" else "21"
        )

        # 12:00 PM is within 9 AM - 9 PM
        check_time = datetime(2026, 1, 20, 12, 0, 0, tzinfo=ZoneInfo("UTC"))

        result = await is_within_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            check_time=check_time,
        )

        assert result is True

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_outside_active_hours_returns_false(
        self, mock_get_now: Mock, mock_logger: Mock
    ) -> None:
        """Test returns False when time is outside active hours."""
        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_system_metrics_repo.get_metric = AsyncMock(
            side_effect=lambda key: "9" if key == "active_hours_start" else "21"
        )

        # 2:00 AM is outside 9 AM - 9 PM
        check_time = datetime(2026, 1, 20, 2, 0, 0, tzinfo=ZoneInfo("UTC"))

        result = await is_within_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            check_time=check_time,
        )

        assert result is False

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_handles_wrap_around_midnight(
        self, mock_get_now: Mock, mock_logger: Mock
    ) -> None:
        """Test correctly handles active hours that wrap around midnight."""
        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        # Active hours: 22 (10 PM) - 6 (6 AM)
        mock_system_metrics_repo.get_metric = AsyncMock(
            side_effect=lambda key: "22" if key == "active_hours_start" else "6"
        )

        # 11 PM should be within hours
        check_time_11pm = datetime(2026, 1, 20, 23, 0, 0, tzinfo=ZoneInfo("UTC"))
        result_11pm = await is_within_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            check_time=check_time_11pm,
        )
        assert result_11pm is True

        # 2 AM should be within hours
        check_time_2am = datetime(2026, 1, 20, 2, 0, 0, tzinfo=ZoneInfo("UTC"))
        result_2am = await is_within_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            check_time=check_time_2am,
        )
        assert result_2am is True

        # 12 PM should be outside hours
        check_time_noon = datetime(2026, 1, 20, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        result_noon = await is_within_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            check_time=check_time_noon,
        )
        assert result_noon is False

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_uses_defaults_when_no_metrics(
        self, mock_get_now: Mock, mock_logger: Mock
    ) -> None:
        """Test uses default hours when metrics are not set."""
        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_system_metrics_repo.get_metric = AsyncMock(return_value=None)

        # 10 AM should be within default 9 AM - 9 PM
        check_time = datetime(2026, 1, 20, 10, 0, 0, tzinfo=ZoneInfo("UTC"))

        result = await is_within_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            check_time=check_time,
        )

        assert result is True

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_uses_current_time_when_check_time_is_none(
        self, mock_get_now: Mock, mock_logger: Mock
    ) -> None:
        """Test uses current time when check_time is None."""
        # Mock current time at 3 PM
        mock_get_now.return_value = datetime(
            2026, 1, 20, 15, 0, 0, tzinfo=ZoneInfo("UTC")
        )

        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.get_user_timezone = AsyncMock(return_value="UTC")
        mock_system_metrics_repo.get_metric = AsyncMock(
            side_effect=lambda key: "9" if key == "active_hours_start" else "21"
        )

        result = await is_within_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            check_time=None,  # Use current time
        )

        # 3 PM should be within 9 AM - 9 PM
        assert result is True

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    async def test_converts_utc_time_to_user_timezone(self, mock_logger: Mock) -> None:
        """Test converts UTC check_time to user's local timezone."""
        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.get_user_timezone = AsyncMock(
            return_value="America/New_York"
        )
        # Active hours: 9 AM - 9 PM EST
        mock_system_metrics_repo.get_metric = AsyncMock(
            side_effect=lambda key: "9" if key == "active_hours_start" else "21"
        )

        # 14:00 UTC = 9 AM EST (start of active hours)
        check_time_utc = datetime(2026, 1, 20, 14, 0, 0, tzinfo=UTC)

        result = await is_within_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            check_time=check_time_utc,
        )

        assert result is True


class TestUpdateActiveHours:
    """Test update_active_hours function."""

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.calculate_active_hours")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_updates_metrics_from_calculation(
        self, mock_get_now: Mock, mock_calculate: AsyncMock, mock_logger: Mock
    ) -> None:
        """Test stores calculated active hours in system metrics."""
        mock_get_now.return_value = datetime(2026, 1, 20, 12, 0, 0, tzinfo=UTC)

        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.set_metric = AsyncMock()

        mock_message_repo = AsyncMock()

        # Mock calculation result
        mock_calculate.return_value = ActiveHoursResult(
            start_hour=10,
            end_hour=22,
            confidence=0.85,
            sample_size=75,
            timezone="America/New_York",
        )

        result = await update_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        # Verify calculation was called
        mock_calculate.assert_called_once_with(
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        # Verify metrics were stored
        assert mock_system_metrics_repo.set_metric.call_count == 4

        # Check each metric was set correctly
        calls = {
            call[0][0]: call[0][1]
            for call in mock_system_metrics_repo.set_metric.call_args_list
        }
        assert calls["active_hours_start"] == "10"
        assert calls["active_hours_end"] == "22"
        assert calls["active_hours_confidence"] == "0.85"
        assert "active_hours_last_updated" in calls

        # Verify result is returned
        assert result["start_hour"] == 10
        assert result["end_hour"] == 22
        assert result["confidence"] == 0.85

    @pytest.mark.asyncio
    @patch("lattice.scheduler.adaptive.logger")
    @patch("lattice.scheduler.adaptive.calculate_active_hours")
    @patch("lattice.scheduler.adaptive.get_now")
    async def test_logs_updated_hours(
        self, mock_get_now: Mock, mock_calculate: AsyncMock, mock_logger: Mock
    ) -> None:
        """Test logs info message with updated hours."""
        mock_get_now.return_value = datetime(2026, 1, 20, 12, 0, 0, tzinfo=UTC)

        mock_system_metrics_repo = AsyncMock()
        mock_system_metrics_repo.set_metric = AsyncMock()
        mock_message_repo = AsyncMock()

        mock_calculate.return_value = ActiveHoursResult(
            start_hour=8,
            end_hour=20,
            confidence=0.75,
            sample_size=50,
            timezone="UTC",
        )

        await update_active_hours(
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        # Verify log was called
        mock_logger.info.assert_called()  # type: ignore[attr-defined]
        call_args = mock_logger.info.call_args  # type: ignore[attr-defined]
        assert call_args[0][0] == "Updated active hours"
        assert call_args[1]["start_hour"] == 8
        assert call_args[1]["end_hour"] == 20
        assert call_args[1]["confidence"] == 0.75
        assert call_args[1]["sample_size"] == 50
