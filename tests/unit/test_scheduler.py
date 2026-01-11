"""Unit tests for scheduler components."""

from datetime import datetime
from zoneinfo import ZoneInfo
from lattice.utils.date_resolution import get_now
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.utils.config import get_config
from lattice.memory.episodic import EpisodicMessage
from lattice.scheduler.adaptive import (
    calculate_active_hours,
    is_within_active_hours,
    update_active_hours,
)
from lattice.scheduler.nudges import (
    NudgePlan,
    prepare_contextual_nudge,
    format_episodic_nudge_context,
    get_default_channel_id,
)
from lattice.core.response_generator import get_goal_context


class TestFormatEpisodicNudgeContext:
    """Tests for format_episodic_nudge_context helper."""

    @pytest.mark.asyncio
    @patch("lattice.scheduler.nudges.get_recent_messages")
    async def test_format_episodic_nudge_context(
        self, mock_get_recent: MagicMock
    ) -> None:
        mock_get_recent.return_value = [
            EpisodicMessage(
                content="Hello",
                discord_message_id=1,
                channel_id=1,
                is_bot=False,
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=ZoneInfo("UTC")),
            ),
            EpisodicMessage(
                content="Hi there",
                discord_message_id=2,
                channel_id=1,
                is_bot=True,
                timestamp=datetime(2024, 1, 1, 10, 1, tzinfo=ZoneInfo("UTC")),
            ),
        ]

        mock_pool = MagicMock()
        mock_pool.pool = mock_pool

        result = await format_episodic_nudge_context(
            db_pool=mock_pool, user_timezone="UTC"
        )
        assert "[2024-01-01 10:00] USER: Hello" in result
        assert "[2024-01-01 10:01] ASSISTANT: Hi there" in result

        result = await format_episodic_nudge_context(
            db_pool=mock_pool, user_timezone="America/Vancouver"
        )
        assert "[2024-01-01 02:00] USER: Hello" in result
        assert "[2024-01-01 02:01] ASSISTANT: Hi there" in result

    @pytest.mark.asyncio
    async def test_format_episodic_nudge_context_with_no_messages(self) -> None:
        """Test conversation context when no messages exist."""
        with patch(
            "lattice.scheduler.nudges.get_recent_messages",
            return_value=[],
        ):
            mock_pool = MagicMock()
            mock_pool.pool = mock_pool

            result = await format_episodic_nudge_context(db_pool=mock_pool)
            assert result == "No recent conversation history."

    @pytest.mark.asyncio
    async def test_format_episodic_nudge_context_with_messages(self) -> None:
        """Test conversation context with messages."""
        messages = [
            EpisodicMessage(
                content="Hello!",
                discord_message_id=1,
                channel_id=12345,
                is_bot=False,
                timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC")),
            ),
            EpisodicMessage(
                content="Hi there!",
                discord_message_id=2,
                channel_id=12345,
                is_bot=True,
                timestamp=datetime(2024, 1, 1, 10, 1, 0, tzinfo=ZoneInfo("UTC")),
            ),
        ]
        with patch(
            "lattice.scheduler.nudges.get_recent_messages",
            return_value=messages,
        ):
            mock_pool = MagicMock()
            mock_pool.pool = mock_pool

            result = await format_episodic_nudge_context(db_pool=mock_pool)
            assert "[2024-01-01 10:00] USER: Hello!" in result
            assert "[2024-01-01 10:01] ASSISTANT: Hi there!" in result


class TestNudgePlan:
    """Tests for NudgePlan dataclass."""

    def test_message_decision(self) -> None:
        decision = NudgePlan(
            action="message",
            content="Hey! How's it going?",
            reason="User has been active",
            channel_id=12345,
        )
        assert decision.action == "message"
        assert decision.content == "Hey! How's it going?"
        assert decision.channel_id == 12345

    def test_wait_decision(self) -> None:
        decision = NudgePlan(
            action="wait",
            content=None,
            reason="User just responded",
        )
        assert decision.action == "wait"
        assert decision.content is None


class TestPrepareContextualNudge:
    """Tests for prepare_contextual_nudge function."""

    @pytest.mark.asyncio
    async def test_prepare_contextual_nudge_with_missing_prompt(self) -> None:
        """Test that missing CONTEXTUAL_NUDGE prompt returns wait."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        mock_pool = MagicMock()
        mock_pool.get_user_timezone = AsyncMock(return_value="UTC")

        with (
            patch(
                "lattice.scheduler.nudges.format_episodic_nudge_context",
                return_value="No recent conversation history.",
            ),
            patch("lattice.scheduler.nudges.get_prompt", return_value=None),
            patch(
                "lattice.core.response_generator.get_goal_context",
                return_value="No active objectives.",
            ),
            patch(
                "lattice.core.context_strategy.retrieve_context",
                AsyncMock(
                    return_value={"activity_context": "No recent activity recorded."}
                ),
            ),
            patch(
                "lattice.scheduler.nudges.get_default_channel_id", return_value=12345
            ),
        ):
            result = await prepare_contextual_nudge(db_pool=mock_pool)
            assert result.action == "wait"
            assert "prompt not found" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_prepare_contextual_nudge_with_llm_exception(self) -> None:
        """Test that LLM exceptions are handled gracefully."""
        mock_prompt = MagicMock()
        mock_prompt.template = "Current date: {local_date}\nCurrent time: {local_time}\n{episodic_context}\n{goal_context}\n{activity_context}"
        mock_prompt.temperature = 0.7
        mock_prompt.version = 1

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=ValueError("LLM service unavailable"))

        # Injected pool and client
        mock_pool = MagicMock()
        mock_pool.get_user_timezone = AsyncMock(return_value="UTC")

        with (
            patch(
                "lattice.scheduler.nudges.format_episodic_nudge_context",
                return_value="No recent conversation history.",
            ),
            patch("lattice.scheduler.nudges.get_prompt", return_value=mock_prompt),
            patch(
                "lattice.core.response_generator.get_goal_context",
                return_value="No active objectives.",
            ),
            patch(
                "lattice.core.context_strategy.retrieve_context",
                AsyncMock(
                    return_value={"activity_context": "No recent activity recorded."}
                ),
            ),
            patch(
                "lattice.scheduler.nudges.get_default_channel_id", return_value=12345
            ),
            patch(
                "lattice.utils.database.get_user_timezone",
                new_callable=AsyncMock,
                return_value="UTC",
            ),
            patch(
                "lattice.scheduler.nudges.get_auditing_llm_client",
                return_value=mock_llm,
            ),
        ):
            result = await prepare_contextual_nudge(
                db_pool=mock_pool, llm_client=mock_llm
            )
            assert result.action == "wait"
            assert "LLM call failed" in result.reason
            assert result.channel_id == 12345

    @pytest.mark.asyncio
    async def test_prepare_contextual_nudge_with_json_parse_error(self) -> None:
        """Test that invalid JSON from LLM is handled gracefully."""
        mock_prompt = MagicMock()
        mock_prompt.template = "Current date: {local_date}\nCurrent time: {local_time}\n{episodic_context}\n{goal_context}\n{activity_context}"
        mock_prompt.temperature = 0.7
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = "This is not valid JSON"
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 100
        mock_result.completion_tokens = 50
        mock_result.cost_usd = 0.01
        mock_result.latency_ms = 500

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=mock_result)

        mock_pool = MagicMock()
        mock_pool.get_user_timezone = AsyncMock(return_value="UTC")

        with (
            patch(
                "lattice.scheduler.nudges.format_episodic_nudge_context",
                return_value="No recent conversation history.",
            ),
            patch("lattice.scheduler.nudges.get_prompt", return_value=mock_prompt),
            patch(
                "lattice.core.response_generator.get_goal_context",
                return_value="No active objectives.",
            ),
            patch(
                "lattice.core.context_strategy.retrieve_context",
                AsyncMock(
                    return_value={"activity_context": "No recent activity recorded."}
                ),
            ),
            patch(
                "lattice.scheduler.nudges.get_default_channel_id", return_value=12345
            ),
            patch(
                "lattice.scheduler.nudges.get_auditing_llm_client",
                return_value=mock_llm,
            ),
        ):
            result = await prepare_contextual_nudge(db_pool=mock_pool)
            assert result.action == "wait"
            assert "Failed to parse AI response" in result.reason
            assert result.channel_id == 12345

    @pytest.mark.asyncio
    async def test_prepare_contextual_nudge_with_empty_content(self) -> None:
        """Test that empty content in message action defaults to wait."""
        mock_prompt = MagicMock()
        mock_prompt.template = "Current date: {local_date}\nCurrent time: {local_time}\n{episodic_context}\n{goal_context}\n{activity_context}"
        mock_prompt.temperature = 0.7
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = (
            '{"action": "message", "content": "   ", "reason": "Testing empty content"}'
        )
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 100
        mock_result.completion_tokens = 50
        mock_result.cost_usd = 0.01
        mock_result.latency_ms = 500

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=mock_result)

        mock_pool = MagicMock()
        mock_pool.get_user_timezone = AsyncMock(return_value="UTC")

        with (
            patch(
                "lattice.scheduler.nudges.format_episodic_nudge_context",
                return_value="No recent conversation history.",
            ),
            patch("lattice.scheduler.nudges.get_prompt", return_value=mock_prompt),
            patch(
                "lattice.core.response_generator.get_goal_context",
                return_value="No active objectives.",
            ),
            patch(
                "lattice.core.context_strategy.retrieve_context",
                AsyncMock(
                    return_value={"activity_context": "No recent activity recorded."}
                ),
            ),
            patch(
                "lattice.scheduler.nudges.get_default_channel_id", return_value=12345
            ),
            patch(
                "lattice.scheduler.nudges.get_auditing_llm_client",
                return_value=mock_llm,
            ),
        ):
            result = await prepare_contextual_nudge(db_pool=mock_pool)
            assert result.action == "wait"
            assert result.content is None

    @pytest.mark.asyncio
    async def test_prepare_contextual_nudge_with_literal_empty_string(self) -> None:
        """Test that literal empty string content defaults to wait."""
        mock_prompt = MagicMock()
        mock_prompt.template = "Current date: {local_date}\nCurrent time: {local_time}\n{episodic_context}\n{goal_context}\n{activity_context}"
        mock_prompt.temperature = 0.7
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = '{"action": "message", "content": "", "reason": "Testing literal empty string"}'
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 100
        mock_result.completion_tokens = 50
        mock_result.cost_usd = 0.01
        mock_result.latency_ms = 500

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=mock_result)

        mock_pool = MagicMock()
        mock_pool.get_user_timezone = AsyncMock(return_value="UTC")

        with (
            patch(
                "lattice.scheduler.nudges.format_episodic_nudge_context",
                return_value="No recent conversation history.",
            ),
            patch("lattice.scheduler.nudges.get_prompt", return_value=mock_prompt),
            patch(
                "lattice.core.response_generator.get_goal_context",
                return_value="No active objectives.",
            ),
            patch(
                "lattice.core.context_strategy.retrieve_context",
                AsyncMock(
                    return_value={"activity_context": "No recent activity recorded."}
                ),
            ),
            patch(
                "lattice.scheduler.nudges.get_default_channel_id", return_value=12345
            ),
            patch(
                "lattice.scheduler.nudges.get_auditing_llm_client",
                return_value=mock_llm,
            ),
        ):
            result = await prepare_contextual_nudge(db_pool=mock_pool)
            assert result.action == "wait"
            assert result.content is None

    @pytest.mark.asyncio
    async def test_prepare_contextual_nudge_with_missing_content(self) -> None:
        """Test that missing content field defaults to wait."""
        mock_prompt = MagicMock()
        mock_prompt.template = "Current date: {local_date}\nCurrent time: {local_time}\n{episodic_context}\n{goal_context}\n{activity_context}"
        mock_prompt.temperature = 0.7
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = (
            '{"action": "message", "reason": "Testing missing content"}'
        )
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 100
        mock_result.completion_tokens = 50
        mock_result.cost_usd = 0.01
        mock_result.latency_ms = 500

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=mock_result)

        mock_pool = MagicMock()
        mock_pool.get_user_timezone = AsyncMock(return_value="UTC")

        with (
            patch(
                "lattice.scheduler.nudges.format_episodic_nudge_context",
                return_value="No recent conversation history.",
            ),
            patch("lattice.scheduler.nudges.get_prompt", return_value=mock_prompt),
            patch(
                "lattice.core.response_generator.get_goal_context",
                return_value="No active objectives.",
            ),
            patch(
                "lattice.core.context_strategy.retrieve_context",
                AsyncMock(
                    return_value={"activity_context": "No recent activity recorded."}
                ),
            ),
            patch(
                "lattice.scheduler.nudges.get_default_channel_id", return_value=12345
            ),
            patch(
                "lattice.scheduler.nudges.get_auditing_llm_client",
                return_value=mock_llm,
            ),
        ):
            result = await prepare_contextual_nudge(db_pool=mock_pool)
            assert result.action == "wait"
            assert result.content is None


class TestGetGoalContext:
    """Tests for get_goal_context function."""

    @pytest.mark.asyncio
    async def test_get_goal_context_with_no_goals(self) -> None:
        """Test goal context when no goals exist."""
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch("lattice.utils.database.db_pool") as mock_pool:
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            result = await get_goal_context()
            assert result == "No active goals."

    @pytest.mark.asyncio
    async def test_get_goal_context_with_goals(self) -> None:
        """Test goal context with goals and predicates from semantic_triple."""
        goals = [
            {"object": "Complete project by Friday"},
            {"object": "Review documentation"},
        ]
        predicates = [
            {
                "subject": "Complete project by Friday",
                "predicate": "due_by",
                "object": "2026-01-10",
            },
            {
                "subject": "Complete project by Friday",
                "predicate": "priority",
                "object": "high",
            },
            {
                "subject": "Complete project by Friday",
                "predicate": "status",
                "object": "active",
            },
        ]

        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(side_effect=[goals, predicates])

        with patch("lattice.utils.database.db_pool") as mock_pool:
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            result = await get_goal_context()
            assert "User goals:" in result
            assert "Complete project by Friday" in result
            assert "Review documentation" in result
            assert "due_by: 2026-01-10" in result
            assert "priority: high" in result


class TestGetDefaultChannelId:
    """Tests for get_default_channel_id function."""

    @pytest.mark.asyncio
    async def test_get_default_channel_id_with_env_var(self) -> None:
        """Test retrieving channel ID from environment variable."""
        config = get_config(reload=True)
        config.discord_main_channel_id = 123456789
        result = await get_default_channel_id()
        assert result == 123456789

    @pytest.mark.asyncio
    async def test_get_default_channel_id_without_env_var(self) -> None:
        """Test retrieving channel ID when env var is not set."""
        config = get_config(reload=True)
        config.discord_main_channel_id = 0  # 0 or None based on Config default
        result = await get_default_channel_id()
        assert result is None or result == 0


class TestAdaptiveActiveHours:
    """Tests for adaptive active hours functionality."""

    @pytest.mark.asyncio
    async def test_calculate_active_hours_insufficient_data(self) -> None:
        """Test that default hours are used when insufficient messages."""
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[])  # No messages

        with (
            patch("lattice.utils.database.db_pool") as mock_pool,
            patch("lattice.utils.database.get_user_timezone", return_value="UTC"),
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
        now = get_now(timezone_str="UTC")
        messages = []
        for i in range(100):
            # Most messages in evening
            hour = 18 + (i % 5)  # 18-22
            timestamp = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            messages.append({"timestamp": timestamp, "user_timezone": "UTC"})

        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=messages)

        with (
            patch("lattice.utils.database.db_pool") as mock_pool,
            patch("lattice.utils.database.get_user_timezone", return_value="UTC"),
        ):
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            result = await calculate_active_hours()

            # Should detect evening activity window
            assert result["sample_size"] == 100
            assert result["confidence"] >= 0.95  # High concentration for clear pattern
            # Window should capture evening hours (12-hour window starting around 11-12)
            assert 10 <= result["start_hour"] <= 20

    @pytest.mark.asyncio
    async def test_is_within_active_hours_normal_window(self) -> None:
        """Test active hours check for normal window (9 AM - 9 PM)."""
        mock_pool = MagicMock()
        mock_pool.get_system_health = AsyncMock(
            side_effect=lambda key: {
                "active_hours_start": "9",
                "active_hours_end": "21",
            }.get(key)
        )
        mock_pool.get_user_timezone = AsyncMock(return_value="UTC")

        with (
            patch(
                "lattice.utils.database.get_system_health",
                side_effect=lambda key: {
                    "active_hours_start": "9",
                    "active_hours_end": "21",
                }.get(key),
            ),
            patch("lattice.utils.database.get_user_timezone", return_value="UTC"),
        ):
            # Test time within active hours (noon)
            within_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
            assert await is_within_active_hours(within_time, db_pool=mock_pool) is True

            # Test time outside active hours (2 AM)
            outside_time = datetime(2024, 1, 1, 2, 0, 0, tzinfo=ZoneInfo("UTC"))
            assert (
                await is_within_active_hours(outside_time, db_pool=mock_pool) is False
            )

    @pytest.mark.asyncio
    async def test_is_within_active_hours_wrap_around(self) -> None:
        """Test active hours check for window wrapping midnight (9 PM - 9 AM)."""
        mock_pool = MagicMock()
        mock_pool.get_system_health = AsyncMock(
            side_effect=lambda key: {
                "active_hours_start": "21",
                "active_hours_end": "9",
            }.get(key)
        )
        mock_pool.get_user_timezone = AsyncMock(return_value="UTC")

        with (
            patch(
                "lattice.utils.database.get_system_health",
                side_effect=lambda key: {
                    "active_hours_start": "21",  # 9 PM
                    "active_hours_end": "9",  # 9 AM
                }.get(key),
            ),
            patch("lattice.utils.database.get_user_timezone", return_value="UTC"),
            patch("lattice.utils.database.get_user_timezone", return_value="UTC"),
            patch(
                "lattice.utils.database.get_system_health",
                side_effect=lambda key: {
                    "active_hours_start": "21",
                    "active_hours_end": "9",
                }.get(key),
            ),
        ):
            # Test time within active hours (midnight)
            within_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("UTC"))
            assert await is_within_active_hours(within_time, db_pool=mock_pool) is True

            # Test time within active hours (6 AM)
            within_time2 = datetime(2024, 1, 1, 6, 0, 0, tzinfo=ZoneInfo("UTC"))
            assert await is_within_active_hours(within_time2, db_pool=mock_pool) is True

            # Test time outside active hours (noon)
            outside_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
            assert (
                await is_within_active_hours(outside_time, db_pool=mock_pool) is False
            )

    @pytest.mark.asyncio
    async def test_is_within_active_hours_defaults_to_now(self) -> None:
        """Test active hours check defaults to current time when not specified."""
        mock_pool = MagicMock()
        mock_pool.get_system_health = AsyncMock(
            side_effect=lambda key: {
                "active_hours_start": "9",
                "active_hours_end": "21",
            }.get(key)
        )
        mock_pool.get_user_timezone = AsyncMock(return_value="UTC")

        with (
            patch(
                "lattice.utils.database.get_system_health",
                side_effect=lambda key: {
                    "active_hours_start": "9",  # Realistic production defaults
                    "active_hours_end": "21",  # 9 AM - 9 PM
                }.get(key),
            ),
            patch("lattice.utils.database.get_user_timezone", return_value="UTC"),
            patch("lattice.utils.database.get_user_timezone", return_value="UTC"),
            patch(
                "lattice.utils.database.get_system_health",
                side_effect=lambda key: {
                    "active_hours_start": "9",
                    "active_hours_end": "21",
                }.get(key),
            ),
            patch("lattice.scheduler.adaptive.get_now") as mock_get_now,
        ):
            # Mock get_now() to return a time within active hours (3 PM)
            mock_now = datetime(2024, 1, 1, 15, 0, 0, tzinfo=ZoneInfo("UTC"))
            mock_get_now.return_value = mock_now

            # Call without specifying check_time (should use mocked get_now())
            result = await is_within_active_hours(db_pool=mock_pool)
            assert result is True  # 3 PM is within 9 AM - 9 PM window

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

        mock_pool = MagicMock()
        mock_pool.set_system_health = AsyncMock()

        with (
            patch(
                "lattice.scheduler.adaptive.calculate_active_hours",
                return_value=mock_result,
            ),
            patch("lattice.utils.database.set_system_health"),
        ):
            result = await update_active_hours(db_pool=mock_pool)

            assert result == mock_result
            # Verify system_health was updated via injected pool
            assert (
                mock_pool.set_system_health.call_count >= 4
            )  # start, end, confidence, last_updated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
