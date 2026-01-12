"""Tests for the scheduler module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.scheduler.nudges import (
    NudgePlan,
    prepare_contextual_nudge,
)


class TestNudgePlan:
    """Tests for NudgePlan dataclass."""

    def test_message_nudge(self) -> None:
        nudge = NudgePlan(
            content="Hey! How's it going?",
            channel_id=12345,
        )
        assert nudge.content == "Hey! How's it going?"
        assert nudge.channel_id == 12345

    def test_wait_nudge(self) -> None:
        nudge = NudgePlan(
            content=None,
        )
        assert nudge.content is None


class TestPrepareContextualNudge:
    """Tests for prepare_contextual_nudge function."""

    @pytest.mark.asyncio
    async def test_prepare_contextual_nudge_with_missing_prompt(self) -> None:
        """Test that missing CONTEXTUAL_NUDGE prompt returns None content."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        mock_pool = MagicMock()
        mock_pool.get_user_timezone = AsyncMock(return_value="UTC")

        mock_cache = MagicMock()
        mock_cache.get_goals.return_value = None
        mock_cache.set_goals = AsyncMock()
        mock_cache.get_activities.return_value = None
        mock_cache.set_activities = AsyncMock()

        with (
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
            result = await prepare_contextual_nudge(
                db_pool=mock_pool,
                llm_client=mock_llm,
                user_context_cache=mock_cache,
                user_id="user",
            )
            assert result.content is None
            assert result.channel_id == 12345

    @pytest.mark.asyncio
    async def test_prepare_contextual_nudge_with_llm_exception(self) -> None:
        """Test that LLM exceptions are handled gracefully."""
        mock_prompt = MagicMock()
        mock_prompt.template = "Current date: {local_date}\nCurrent time: {local_time}\n{goal_context}\n{activity_context}"
        mock_prompt.temperature = 0.7
        mock_prompt.version = 1

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=ValueError("LLM service unavailable"))

        mock_pool = MagicMock()
        mock_pool.get_user_timezone = AsyncMock(return_value="UTC")

        mock_cache = MagicMock()
        mock_cache.get_goals.return_value = None
        mock_cache.set_goals = AsyncMock()
        mock_cache.get_activities.return_value = None
        mock_cache.set_activities = AsyncMock()

        with (
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
        ):
            result = await prepare_contextual_nudge(
                db_pool=mock_pool,
                llm_client=mock_llm,
                user_context_cache=mock_cache,
                user_id="user",
            )
            assert result.content is None
            assert result.channel_id == 12345

    @pytest.mark.asyncio
    async def test_prepare_contextual_nudge_with_empty_content(self) -> None:
        """Test that empty content from LLM returns None content."""
        mock_prompt = MagicMock()
        mock_prompt.template = "Current date: {local_date}\nCurrent time: {local_time}\n{goal_context}\n{activity_context}"
        mock_prompt.temperature = 0.7
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = ""
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

        mock_cache = MagicMock()
        mock_cache.get_goals.return_value = None
        mock_cache.set_goals = AsyncMock()
        mock_cache.get_activities.return_value = None
        mock_cache.set_activities = AsyncMock()

        with (
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
        ):
            result = await prepare_contextual_nudge(
                db_pool=mock_pool,
                llm_client=mock_llm,
                user_context_cache=mock_cache,
                user_id="user",
            )
            assert result.content is None

    @pytest.mark.asyncio
    async def test_prepare_contextual_nudge_with_valid_content(self) -> None:
        """Test that valid content from LLM is returned."""
        mock_prompt = MagicMock()
        mock_prompt.template = "Current date: {local_date}\nCurrent time: {local_time}\n{goal_context}\n{activity_context}"
        mock_prompt.temperature = 0.7
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = "Hey! How's your project going?"
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 100
        mock_result.completion_tokens = 50
        mock_result.cost_usd = 0.01
        mock_result.latency_ms = 500
        mock_result.audit_id = None

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=mock_result)

        mock_pool = MagicMock()
        mock_pool.get_user_timezone = AsyncMock(return_value="UTC")

        mock_cache = MagicMock()
        mock_cache.get_goals.return_value = None
        mock_cache.set_goals = AsyncMock()
        mock_cache.get_activities.return_value = None
        mock_cache.set_activities = AsyncMock()

        with (
            patch("lattice.scheduler.nudges.get_prompt", return_value=mock_prompt),
            patch(
                "lattice.core.response_generator.get_goal_context",
                return_value="User goals:\n├── Finish project",
            ),
            patch(
                "lattice.core.context_strategy.retrieve_context",
                AsyncMock(
                    return_value={
                        "activity_context": "Recent user activities:\n- Worked on project"
                    }
                ),
            ),
            patch(
                "lattice.scheduler.nudges.get_default_channel_id", return_value=12345
            ),
        ):
            result = await prepare_contextual_nudge(
                db_pool=mock_pool,
                llm_client=mock_llm,
                user_context_cache=mock_cache,
                user_id="user",
            )
            assert result.content == "Hey! How's your project going?"
            assert result.channel_id == 12345
            assert result.model == "gpt-4"


class TestGetGoalContext:
    """Tests for get_goal_context function."""
