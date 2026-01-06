"""Unit tests for scheduler components."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.memory.episodic import EpisodicMessage
from lattice.scheduler.adaptive import (
    calculate_active_hours,
    is_within_active_hours,
    update_active_hours,
)
from lattice.scheduler.triggers import (
    ProactiveDecision,
    decide_proactive,
    format_message,
)


class TestFormatMessage:
    """Tests for format_message helper."""

    def test_user_message(self) -> None:
        msg = EpisodicMessage(
            content="Hello, world!",
            discord_message_id=12345,
            channel_id=67890,
            is_bot=False,
            timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        result = format_message(msg)
        assert result == "[2024-01-01 00:00] USER: Hello, world!"

    def test_assistant_message(self) -> None:
        msg = EpisodicMessage(
            content="Hello there!",
            discord_message_id=12346,
            channel_id=67890,
            is_bot=True,
            timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        result = format_message(msg)
        assert result == "[2024-01-01 00:00] ASSISTANT: Hello there!"

    def test_long_content_truncated(self) -> None:
        msg = EpisodicMessage(
            content="x" * 1000,
            discord_message_id=12347,
            channel_id=67890,
            is_bot=False,
            timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        result = format_message(msg)
        assert len(result) < 600
        assert result.startswith("[2024-01-01 00:00] USER: ")


class TestProactiveDecision:
    """Tests for ProactiveDecision dataclass."""

    def test_message_decision(self) -> None:
        decision = ProactiveDecision(
            action="message",
            content="Hey! How's it going?",
            reason="User has been active",
            channel_id=12345,
        )
        assert decision.action == "message"
        assert decision.content == "Hey! How's it going?"
        assert decision.channel_id == 12345

    def test_wait_decision(self) -> None:
        decision = ProactiveDecision(
            action="wait",
            content=None,
            reason="User just responded",
        )
        assert decision.action == "wait"
        assert decision.content is None


class TestDecideProactive:
    """Tests for decide_proactive function."""

    @pytest.mark.asyncio
    async def test_decide_proactive_with_missing_prompt(self) -> None:
        """Test that missing PROACTIVE_DECISION prompt returns wait."""
        with (
            patch(
                "lattice.scheduler.triggers.is_within_active_hours", return_value=True
            ),
            patch("lattice.scheduler.triggers.get_prompt", return_value=None),
            patch(
                "lattice.scheduler.triggers.get_conversation_context",
                return_value="No recent conversation history.",
            ),
            patch(
                "lattice.scheduler.triggers.get_objectives_context",
                return_value="No active objectives.",
            ),
            patch(
                "lattice.scheduler.triggers.get_default_channel_id", return_value=None
            ),
            patch("lattice.scheduler.triggers.get_current_interval", return_value=15),
            patch("lattice.scheduler.triggers.get_system_health", return_value="15"),
        ):
            result = await decide_proactive()
            assert result.action == "wait"
            assert "prompt not found" in result.reason.lower()


class TestAdaptiveActiveHours:
    """Tests for adaptive active hours functionality."""

    @pytest.mark.asyncio
    async def test_calculate_active_hours_insufficient_data(self) -> None:
        """Test that default hours are used when insufficient messages."""
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[])  # No messages

        with (
            patch("lattice.scheduler.adaptive.db_pool") as mock_pool,
            patch("lattice.scheduler.adaptive.get_user_timezone", return_value="UTC"),
        ):
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            result = await calculate_active_hours()

            assert result["start_hour"] == 9  # Default
            assert result["end_hour"] == 21  # Default
            assert result["confidence"] == 0.0
            assert result["sample_size"] == 0

    @pytest.mark.asyncio
    async def test_calculate_active_hours_with_messages(self) -> None:
        """Test active hours calculation with sufficient messages."""
        # Create 100 messages concentrated in evening hours (18-22)
        now = datetime.now(UTC)
        messages = []
        for i in range(100):
            # Most messages in evening
            hour = 18 + (i % 5)  # 18-22
            timestamp = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            messages.append({"timestamp": timestamp, "user_timezone": "UTC"})

        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=messages)

        with (
            patch("lattice.scheduler.adaptive.db_pool") as mock_pool,
            patch("lattice.scheduler.adaptive.get_user_timezone", return_value="UTC"),
        ):
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            result = await calculate_active_hours()

            # Should detect evening activity window
            assert result["sample_size"] == 100
            assert result["confidence"] > 0.7  # High concentration
            # Window should capture evening hours (12-hour window starting around 11-12)
            assert 10 <= result["start_hour"] <= 20

    @pytest.mark.asyncio
    async def test_is_within_active_hours_normal_window(self) -> None:
        """Test active hours check for normal window (9 AM - 9 PM)."""
        with (
            patch(
                "lattice.scheduler.adaptive.get_system_health",
                side_effect=lambda key: {
                    "active_hours_start": "9",
                    "active_hours_end": "21",
                }.get(key),
            ),
            patch("lattice.scheduler.adaptive.get_user_timezone", return_value="UTC"),
        ):
            # Test time within active hours (noon)
            within_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
            assert await is_within_active_hours(within_time) is True

            # Test time outside active hours (2 AM)
            outside_time = datetime(2024, 1, 1, 2, 0, 0, tzinfo=UTC)
            assert await is_within_active_hours(outside_time) is False

    @pytest.mark.asyncio
    async def test_is_within_active_hours_wrap_around(self) -> None:
        """Test active hours check for window wrapping midnight (9 PM - 9 AM)."""
        with (
            patch(
                "lattice.scheduler.adaptive.get_system_health",
                side_effect=lambda key: {
                    "active_hours_start": "21",  # 9 PM
                    "active_hours_end": "9",  # 9 AM
                }.get(key),
            ),
            patch("lattice.scheduler.adaptive.get_user_timezone", return_value="UTC"),
        ):
            # Test time within active hours (midnight)
            within_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
            assert await is_within_active_hours(within_time) is True

            # Test time within active hours (6 AM)
            within_time2 = datetime(2024, 1, 1, 6, 0, 0, tzinfo=UTC)
            assert await is_within_active_hours(within_time2) is True

            # Test time outside active hours (noon)
            outside_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
            assert await is_within_active_hours(outside_time) is False

    @pytest.mark.asyncio
    async def test_is_within_active_hours_defaults_to_now(self) -> None:
        """Test active hours check defaults to current time when not specified."""
        with (
            patch(
                "lattice.scheduler.adaptive.get_system_health",
                side_effect=lambda key: {
                    "active_hours_start": "0",  # Midnight
                    "active_hours_end": "24",  # End of day (wraps to 0) - covers whole day
                }.get(key),
            ),
            patch("lattice.scheduler.adaptive.get_user_timezone", return_value="UTC"),
        ):
            # Call without specifying check_time (should use datetime.now())
            result = await is_within_active_hours()
            # With 0-24 range (wraps), should always be within hours
            assert result is True

    @pytest.mark.asyncio
    async def test_update_active_hours_stores_result(self) -> None:
        """Test that update_active_hours calculates and stores result."""
        mock_result = {
            "start_hour": 10,
            "end_hour": 22,
            "confidence": 0.85,
            "sample_size": 50,
            "timezone": "UTC",
        }

        with (
            patch(
                "lattice.scheduler.adaptive.calculate_active_hours",
                return_value=mock_result,
            ),
            patch("lattice.scheduler.adaptive.set_system_health") as mock_set,
        ):
            result = await update_active_hours()

            assert result == mock_result
            # Verify system_health was updated
            assert mock_set.call_count >= 4  # start, end, confidence, last_updated

    @pytest.mark.asyncio
    async def test_decide_proactive_respects_active_hours(self) -> None:
        """Test that decide_proactive checks active hours first."""
        with patch(
            "lattice.scheduler.triggers.is_within_active_hours", return_value=False
        ):
            result = await decide_proactive()

            assert result.action == "wait"
            assert "active hours" in result.reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
