"""Unit tests for scheduler components."""

from datetime import datetime, UTC
from unittest.mock import patch

import pytest

from lattice.scheduler.triggers import (
    ProactiveDecision,
    decide_proactive,
    format_message,
)
from lattice.memory.episodic import EpisodicMessage


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
