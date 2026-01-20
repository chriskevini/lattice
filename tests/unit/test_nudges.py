"""Unit tests for contextual nudge generation."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from lattice.scheduler.nudges import (
    NudgePlan,
    get_default_channel_id,
    prepare_contextual_nudge,
)


class TestNudgePlan:
    """Test NudgePlan dataclass."""

    def test_nudge_plan_with_minimal_fields(self) -> None:
        """Test NudgePlan can be created with just content."""
        plan = NudgePlan(content="Check in with user")
        assert plan.content == "Check in with user"
        assert plan.channel_id is None
        assert plan.audit_id is None

    def test_nudge_plan_with_all_fields(self) -> None:
        """Test NudgePlan with all fields populated."""
        audit_id = uuid4()
        plan = NudgePlan(
            content="How's your project going?",
            channel_id=123456789,
            audit_id=audit_id,
            rendered_prompt="prompt text",
            template_version=2,
            model="gpt-4",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.01,
            latency_ms=100,
        )
        assert plan.content == "How's your project going?"
        assert plan.channel_id == 123456789
        assert plan.audit_id == audit_id
        assert plan.model == "gpt-4"
        assert plan.cost_usd == 0.01


class TestGetDefaultChannelId:
    """Test get_default_channel_id function."""

    @pytest.mark.asyncio
    @patch("lattice.scheduler.nudges.get_config")
    async def test_returns_main_channel_id_from_config(
        self, mock_get_config: Mock
    ) -> None:
        """Test returns main channel ID from configuration."""
        mock_config = Mock()
        mock_config.discord_main_channel_id = 987654321
        mock_get_config.return_value = mock_config

        channel_id = await get_default_channel_id()
        assert channel_id == 987654321

    @pytest.mark.asyncio
    @patch("lattice.scheduler.nudges.logger")
    @patch("lattice.scheduler.nudges.get_config")
    async def test_returns_none_and_logs_warning_when_not_configured(
        self, mock_get_config: Mock, mock_logger: Mock
    ) -> None:
        """Test returns None and logs warning when main channel not configured."""
        mock_config = Mock()
        mock_config.discord_main_channel_id = None
        mock_get_config.return_value = mock_config

        channel_id = await get_default_channel_id()

        assert channel_id is None
        mock_logger.warning.assert_called_once()  # type: ignore[attr-defined]
        assert "DISCORD_MAIN_CHANNEL_ID not set" in mock_logger.warning.call_args[0][0]  # type: ignore[attr-defined]


class TestPrepareContextualNudge:
    """Test prepare_contextual_nudge function."""

    @pytest.mark.asyncio
    async def test_raises_value_error_when_no_llm_client(self) -> None:
        """Test raises ValueError when llm_client is not provided."""
        with pytest.raises(ValueError, match="llm_client is required"):
            await prepare_contextual_nudge(
                llm_client=None,
                user_context_cache=Mock(),
                user_id="123",
                prompt_template=Mock(),
            )

    @pytest.mark.asyncio
    @patch("lattice.scheduler.nudges.logger")
    @patch("lattice.scheduler.nudges.get_default_channel_id")
    async def test_returns_none_content_when_no_prompt_template(
        self, mock_get_channel_id: AsyncMock, mock_logger: Mock
    ) -> None:
        """Test returns NudgePlan with None content when prompt template not found."""
        mock_get_channel_id.return_value = 123456789

        mock_llm_client = AsyncMock()
        mock_user_cache = Mock()
        mock_user_cache.get_timezone = Mock(return_value="UTC")
        mock_user_cache.get_goals = Mock(return_value="Test goals")
        mock_user_cache.get_activities = Mock(return_value="Test activity")

        result = await prepare_contextual_nudge(
            llm_client=mock_llm_client,
            user_context_cache=mock_user_cache,
            user_id="123",
            prompt_template=None,  # No template
        )

        assert result.content is None
        assert result.channel_id == 123456789

        # Verify error was logged
        mock_logger.error.assert_called()  # type: ignore[attr-defined]
        assert "CONTEXTUAL_NUDGE prompt not found" in mock_logger.error.call_args[0][0]  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    @patch("lattice.scheduler.nudges.PlaceholderInjector")
    @patch("lattice.scheduler.nudges.get_now")
    @patch("lattice.scheduler.nudges.get_default_channel_id")
    async def test_successfully_generates_nudge_content(
        self,
        mock_get_channel_id: AsyncMock,
        mock_get_now: Mock,
        mock_injector_class: Mock,
    ) -> None:
        """Test successfully generates nudge content with LLM."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        mock_get_channel_id.return_value = 123456789
        mock_get_now.return_value = datetime(
            2026, 1, 20, 15, 30, tzinfo=ZoneInfo("UTC")
        )

        # Mock LLM client
        mock_llm_client = AsyncMock()
        mock_audit_result = Mock()
        mock_audit_result.content = "How's your progress on the project?"
        mock_audit_result.audit_id = uuid4()
        mock_audit_result.model = "gpt-4"
        mock_audit_result.provider = "openai"
        mock_audit_result.prompt_tokens = 50
        mock_audit_result.completion_tokens = 10
        mock_audit_result.cost_usd = 0.005
        mock_audit_result.latency_ms = 200
        mock_llm_client.complete = AsyncMock(return_value=mock_audit_result)

        # Mock user context cache
        mock_user_cache = Mock()
        mock_user_cache.get_timezone = Mock(return_value="UTC")
        mock_user_cache.get_goals = Mock(return_value="Complete the project")
        mock_user_cache.get_activities = Mock(return_value="Working on tasks")

        # Mock prompt template
        mock_template = Mock()
        mock_template.version = 1
        mock_template.temperature = 0.7

        # Mock placeholder injector
        mock_injector = AsyncMock()
        mock_injector.inject = AsyncMock(return_value=("injected prompt", {}))
        mock_injector_class.return_value = mock_injector

        result = await prepare_contextual_nudge(
            llm_client=mock_llm_client,
            user_context_cache=mock_user_cache,
            user_id="123",
            prompt_template=mock_template,
        )

        assert result.content == "How's your progress on the project?"
        assert result.channel_id == 123456789
        assert result.model == "gpt-4"
        assert result.cost_usd == 0.005

        # Verify LLM was called with correct parameters
        mock_llm_client.complete.assert_called_once()
        call_kwargs = mock_llm_client.complete.call_args[1]
        assert call_kwargs["prompt_key"] == "CONTEXTUAL_NUDGE"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["audit_view"] is True

    @pytest.mark.asyncio
    @patch("lattice.scheduler.nudges.PlaceholderInjector")
    @patch("lattice.scheduler.nudges.get_now")
    @patch("lattice.scheduler.nudges.get_default_channel_id")
    async def test_returns_none_content_when_llm_returns_empty_string(
        self,
        mock_get_channel_id: AsyncMock,
        mock_get_now: Mock,
        mock_injector_class: Mock,
    ) -> None:
        """Test returns None content when LLM returns empty string."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        mock_get_channel_id.return_value = 123456789
        mock_get_now.return_value = datetime(
            2026, 1, 20, 15, 30, tzinfo=ZoneInfo("UTC")
        )

        # Mock LLM client returning empty content
        mock_llm_client = AsyncMock()
        mock_audit_result = Mock()
        mock_audit_result.content = "   "  # Whitespace only
        mock_audit_result.audit_id = uuid4()
        mock_audit_result.model = "gpt-4"
        mock_audit_result.provider = "openai"
        mock_audit_result.prompt_tokens = 50
        mock_audit_result.completion_tokens = 0
        mock_audit_result.cost_usd = 0.001
        mock_audit_result.latency_ms = 100
        mock_llm_client.complete = AsyncMock(return_value=mock_audit_result)

        mock_user_cache = Mock()
        mock_user_cache.get_timezone = Mock(return_value="UTC")
        mock_user_cache.get_goals = Mock(return_value="goals")
        mock_user_cache.get_activities = Mock(return_value="activity")

        mock_template = Mock()
        mock_template.version = 1
        mock_template.temperature = 0.7

        mock_injector = AsyncMock()
        mock_injector.inject = AsyncMock(return_value=("prompt", {}))
        mock_injector_class.return_value = mock_injector

        result = await prepare_contextual_nudge(
            llm_client=mock_llm_client,
            user_context_cache=mock_user_cache,
            user_id="123",
            prompt_template=mock_template,
        )

        # Should return None content when result is empty
        assert result.content is None
        assert result.channel_id == 123456789
        # But should still have audit metadata
        assert result.audit_id == mock_audit_result.audit_id
        assert result.model == "gpt-4"

    @pytest.mark.asyncio
    @patch("lattice.scheduler.nudges.logger")
    @patch("lattice.scheduler.nudges.PlaceholderInjector")
    @patch("lattice.scheduler.nudges.get_now")
    @patch("lattice.scheduler.nudges.get_default_channel_id")
    async def test_handles_llm_exception_gracefully(
        self,
        mock_get_channel_id: AsyncMock,
        mock_get_now: Mock,
        mock_injector_class: Mock,
        mock_logger: Mock,
    ) -> None:
        """Test handles LLM exception and returns None content."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        mock_get_channel_id.return_value = 123456789
        mock_get_now.return_value = datetime(
            2026, 1, 20, 15, 30, tzinfo=ZoneInfo("UTC")
        )

        # Mock LLM client that raises exception
        mock_llm_client = AsyncMock()
        mock_llm_client.complete = AsyncMock(side_effect=ValueError("LLM error"))

        mock_user_cache = Mock()
        mock_user_cache.get_timezone = Mock(return_value="UTC")
        mock_user_cache.get_goals = Mock(return_value="goals")
        mock_user_cache.get_activities = Mock(return_value="activity")

        mock_template = Mock()
        mock_template.version = 1
        mock_template.temperature = 0.7

        mock_injector = AsyncMock()
        mock_injector.inject = AsyncMock(return_value=("prompt", {}))
        mock_injector_class.return_value = mock_injector

        result = await prepare_contextual_nudge(
            llm_client=mock_llm_client,
            user_context_cache=mock_user_cache,
            user_id="123",
            prompt_template=mock_template,
        )

        assert result.content is None
        assert result.channel_id == 123456789

        # Verify exception was logged
        mock_logger.exception.assert_called_once_with("LLM call failed")  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    @patch("lattice.core.response_generator.get_goal_context")
    @patch("lattice.scheduler.nudges.PlaceholderInjector")
    @patch("lattice.scheduler.nudges.get_now")
    @patch("lattice.scheduler.nudges.get_default_channel_id")
    async def test_fetches_goals_from_semantic_repo_when_cache_miss(
        self,
        mock_get_channel_id: AsyncMock,
        mock_get_now: Mock,
        mock_injector_class: Mock,
        mock_get_goal_context: AsyncMock,
    ) -> None:
        """Test fetches goals from semantic repo when not in cache."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        mock_get_channel_id.return_value = 123456789
        mock_get_now.return_value = datetime(
            2026, 1, 20, 15, 30, tzinfo=ZoneInfo("UTC")
        )

        mock_get_goal_context.return_value = "Fetched goals from DB"

        mock_llm_client = AsyncMock()
        mock_result = Mock(content="nudge", audit_id=uuid4())
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        # Cache returns None for goals
        mock_user_cache = AsyncMock()
        mock_user_cache.get_timezone = Mock(return_value="UTC")
        mock_user_cache.get_goals = Mock(return_value=None)  # Cache miss
        mock_user_cache.get_activities = Mock(return_value="activity")
        mock_user_cache.set_goals = AsyncMock()

        mock_semantic_repo = Mock()

        mock_template = Mock(version=1, temperature=0.7)

        mock_injector = AsyncMock()
        mock_injector.inject = AsyncMock(return_value=("prompt", {}))
        mock_injector_class.return_value = mock_injector

        result = await prepare_contextual_nudge(
            llm_client=mock_llm_client,
            user_context_cache=mock_user_cache,
            user_id="123",
            prompt_template=mock_template,
            semantic_repo=mock_semantic_repo,
        )

        # Verify goals were fetched and cached
        mock_get_goal_context.assert_called_once_with(semantic_repo=mock_semantic_repo)
        mock_user_cache.set_goals.assert_called_once_with(
            "123", "Fetched goals from DB"
        )

    @pytest.mark.asyncio
    @patch("lattice.core.context_strategy.retrieve_context")
    @patch("lattice.scheduler.nudges.PlaceholderInjector")
    @patch("lattice.scheduler.nudges.get_now")
    @patch("lattice.scheduler.nudges.get_default_channel_id")
    async def test_fetches_activity_from_semantic_repo_when_cache_miss(
        self,
        mock_get_channel_id: AsyncMock,
        mock_get_now: Mock,
        mock_injector_class: Mock,
        mock_retrieve_context: AsyncMock,
    ) -> None:
        """Test fetches activity from semantic repo when not in cache."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        mock_get_channel_id.return_value = 123456789
        mock_get_now.return_value = datetime(
            2026, 1, 20, 15, 30, tzinfo=ZoneInfo("UTC")
        )

        mock_retrieve_context.return_value = {
            "activity_context": "Fetched activity from DB"
        }

        mock_llm_client = AsyncMock()
        mock_result = Mock(content="nudge", audit_id=uuid4())
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        # Cache returns None for activity
        mock_user_cache = AsyncMock()
        mock_user_cache.get_timezone = Mock(return_value="UTC")
        mock_user_cache.get_goals = Mock(return_value="goals")
        mock_user_cache.get_activities = Mock(return_value=None)  # Cache miss
        mock_user_cache.set_activities = AsyncMock()

        mock_semantic_repo = Mock()

        mock_template = Mock(version=1, temperature=0.7)

        mock_injector = AsyncMock()
        mock_injector.inject = AsyncMock(return_value=("prompt", {}))
        mock_injector_class.return_value = mock_injector

        result = await prepare_contextual_nudge(
            llm_client=mock_llm_client,
            user_context_cache=mock_user_cache,
            user_id="123",
            prompt_template=mock_template,
            semantic_repo=mock_semantic_repo,
        )

        # Verify activity was fetched and cached
        mock_retrieve_context.assert_called_once()
        call_kwargs = mock_retrieve_context.call_args[1]
        assert call_kwargs["context_flags"] == ["activity_context"]
        mock_user_cache.set_activities.assert_called_once_with(
            "123", "Fetched activity from DB"
        )

    @pytest.mark.asyncio
    @patch("lattice.scheduler.nudges.PlaceholderInjector")
    @patch("lattice.scheduler.nudges.get_now")
    @patch("lattice.scheduler.nudges.get_default_channel_id")
    async def test_uses_utc_when_no_timezone_in_cache(
        self,
        mock_get_channel_id: AsyncMock,
        mock_get_now: Mock,
        mock_injector_class: Mock,
    ) -> None:
        """Test uses UTC timezone when cache returns None."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        mock_get_channel_id.return_value = 123456789
        mock_now = datetime(2026, 1, 20, 15, 30, tzinfo=ZoneInfo("UTC"))
        mock_get_now.return_value = mock_now

        mock_llm_client = AsyncMock()
        mock_result = Mock(content="nudge", audit_id=uuid4())
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        # Cache returns None for timezone
        mock_user_cache = Mock()
        mock_user_cache.get_timezone = Mock(return_value=None)
        mock_user_cache.get_goals = Mock(return_value="goals")
        mock_user_cache.get_activities = Mock(return_value="activity")

        mock_template = Mock(version=1, temperature=0.7)

        mock_injector = AsyncMock()
        mock_injector.inject = AsyncMock(return_value=("prompt", {}))
        mock_injector_class.return_value = mock_injector

        result = await prepare_contextual_nudge(
            llm_client=mock_llm_client,
            user_context_cache=mock_user_cache,
            user_id="123",
            prompt_template=mock_template,
        )

        # Verify get_now was called with UTC
        mock_get_now.assert_called_with("UTC")
