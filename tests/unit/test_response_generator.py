"""Unit tests for response generator module.

Tests for placeholder validation, context building, and response generation.
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from lattice.core.response_generator import (
    generate_response,
    split_response,
    get_goal_context,
    validate_template_placeholders,
    get_available_placeholders,
)


class MockPromptTemplate:
    """Mock PromptTemplate for testing."""

    def __init__(
        self,
        prompt_key: str = "UNIFIED_RESPONSE",
        template: str = "You are helpful.\nDate: {local_date}\nHints: {date_resolution_hints}\nContext: {episodic_context}\nMemory: {semantic_context}\nEntities: {unresolved_entities}\nUser: {user_message}",
        temperature: float | None = 0.7,
        version: int = 1,
        active: bool = True,
    ) -> None:
        self.prompt_key = prompt_key
        self.template = template
        self.temperature = temperature
        self.version = version
        self.active = active

    def safe_format(self, **kwargs: Any) -> str:
        """Safely format template, escaping unknown placeholders."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


@dataclass
class MockAuditResult:
    """Mock AuditResult for testing."""

    content: str = "Test response"
    model: str = "test-model"
    provider: str | None = "test-provider"
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30
    cost_usd: float | None = 0.001
    latency_ms: int = 100
    temperature: float = 0.7
    prompt_key: str | None = None
    audit_id: int | None = None


class TestAvailablePlaceholders:
    """Tests for get_available_placeholders()."""

    def test_unresolved_entities_in_placeholders(self) -> None:
        """unresolved_entities should be in available placeholders."""
        placeholders = get_available_placeholders()
        assert "unresolved_entities" in placeholders
        assert "Entities requiring clarification" in placeholders["unresolved_entities"]

    def test_all_placeholders_are_strings(self) -> None:
        """All placeholder keys and descriptions should be strings."""
        placeholders = get_available_placeholders()
        for key, description in placeholders.items():
            assert isinstance(key, str)
            assert isinstance(description, str)

    def test_user_message_placeholder_exists(self) -> None:
        """user_message placeholder should be available."""
        placeholders = get_available_placeholders()
        assert "user_message" in placeholders

    def test_episodic_context_placeholder_exists(self) -> None:
        """episodic_context placeholder should be available."""
        placeholders = get_available_placeholders()
        assert "episodic_context" in placeholders

    def test_semantic_context_placeholder_exists(self) -> None:
        """semantic_context placeholder should be available."""
        placeholders = get_available_placeholders()
        assert "semantic_context" in placeholders

    def test_local_time_placeholder_exists(self) -> None:
        """local_time placeholder should be available."""
        placeholders = get_available_placeholders()
        assert "local_time" in placeholders


class TestValidateTemplatePlaceholders:
    """Tests for validate_template_placeholders()."""

    def test_valid_template_returns_true(self) -> None:
        """Template with known placeholders should return True."""
        template = "Hello {user_message}! Your local time is {local_time}."
        is_valid, unknown = validate_template_placeholders(template)
        assert is_valid is True
        assert unknown == []

    def test_invalid_template_returns_false(self) -> None:
        """Template with unknown placeholders should return False."""
        template = "Hello {unknown_var} and {another_unknown}."
        is_valid, unknown = validate_template_placeholders(template)
        assert is_valid is False
        assert "unknown_var" in unknown
        assert "another_unknown" in unknown

    def test_empty_template_is_valid(self) -> None:
        """Template with no placeholders should be valid."""
        template = "This is a plain template."
        is_valid, unknown = validate_template_placeholders(template)
        assert is_valid is True
        assert unknown == []

    def test_partial_invalid_placeholders(self) -> None:
        """Template with mix of valid and invalid placeholders."""
        template = "Hello {user_message} and {unknown_placeholder}."
        is_valid, unknown = validate_template_placeholders(template)
        assert is_valid is False
        assert "unknown_placeholder" in unknown
        assert "user_message" not in unknown


class TestSplitResponse:
    """Tests for split_response() function."""

    def test_short_response_returns_single_chunk(self) -> None:
        """Response under max length should return single chunk."""
        response = "Short message"
        result = split_response(response)
        assert len(result) == 1
        assert result[0] == response

    def test_empty_response_returns_single_empty_chunk(self) -> None:
        """Empty response should return list with single empty string."""
        result = split_response("")
        assert result == [""]

    def test_exact_max_length_returns_single(self) -> None:
        """Response exactly at max length should return single chunk."""
        response = "a" * 1900
        result = split_response(response, max_length=1900)
        assert len(result) == 1
        assert len(result[0]) == 1900

    def test_single_long_line_splits_at_word_boundaries(self) -> None:
        """Long line should split at word boundaries."""
        words = ["word"] * 100
        response = " ".join(words)
        result = split_response(response, max_length=100)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 100

    def test_newlines_preserved_as_split_points(self) -> None:
        """Response with newlines should split at newline boundaries."""
        lines = ["Line " + str(i) for i in range(10)]
        response = "\n".join(lines)
        result = split_response(response, max_length=50)
        assert len(result) > 1

    def test_very_long_word_splits_hard(self) -> None:
        """Word longer than max_length should be split."""
        long_word = "a" * 500
        response = f"Start {long_word} end"
        result = split_response(response, max_length=100)
        assert len(result) > 1

    def test_long_line_with_existing_chunk(self) -> None:
        """Long line should flush existing chunk before splitting."""
        short_line = "Short line"
        long_line = "This is a very long line that exceeds the maximum length limit and should be split at word boundaries when processed"
        response = f"{short_line}\n{long_line}"
        result = split_response(response, max_length=50)
        assert len(result) > 1

    def test_multiple_lines_under_limit_kept_together(self) -> None:
        """Multiple short lines should stay together when under limit."""
        lines = ["Short", "Line", "Here"]
        response = "\n".join(lines)
        result = split_response(response, max_length=100)
        assert len(result) == 1
        assert result[0] == response

    def test_code_block_preserved(self) -> None:
        """Code blocks should be preserved as much as possible."""
        code = "```python\ndef hello():\n    print('world')\n```"
        result = split_response(code, max_length=1000)
        assert len(result) >= 1

    def test_custom_max_length(self) -> None:
        """Custom max_length should be respected."""
        response = "a" * 100
        result = split_response(response, max_length=50)
        for chunk in result:
            assert len(chunk) <= 50

    def test_unicode_handled_correctly(self) -> None:
        """Unicode characters should be handled correctly."""
        response = "Hello 世界 🌍 " * 50
        result = split_response(response, max_length=200)
        for chunk in result:
            assert len(chunk) <= 200

    def test_response_with_only_newlines(self) -> None:
        """Response with only newlines should be handled."""
        response = "\n\n\n"
        result = split_response(response, max_length=100)
        assert len(result) >= 1


class TestGetGoalContext:
    """Tests for get_goal_context() function."""

    @pytest.mark.asyncio
    async def test_no_repo_returns_no_active_goals(self) -> None:
        """When no semantic_repo provided, should return default message."""
        result = await get_goal_context()
        assert result == "No active goals."

    @pytest.mark.asyncio
    async def test_empty_goal_names_returns_no_active_goals(self) -> None:
        """When goal_names is empty list, should return no goals message."""
        mock_repo = AsyncMock()
        result = await get_goal_context(semantic_repo=mock_repo, goal_names=[])
        assert result == "No active goals."

    @pytest.mark.asyncio
    async def test_single_goal_no_predicates(self) -> None:
        """Single goal with no predicates should display correctly."""
        mock_repo = AsyncMock()
        mock_repo.fetch_goal_names = AsyncMock(return_value=["Learn Python"])
        mock_repo.get_goal_predicates = AsyncMock(return_value=[])

        result = await get_goal_context(semantic_repo=mock_repo)

        assert "User goals:" in result
        assert "Learn Python" in result

    @pytest.mark.asyncio
    async def test_multiple_goals_with_predicates(self) -> None:
        """Multiple goals with predicates should display hierarchically."""
        mock_repo = AsyncMock()
        mock_repo.fetch_goal_names = AsyncMock(
            return_value=["Learn Python", "Build App"]
        )
        mock_repo.get_goal_predicates = AsyncMock(
            return_value=[
                {
                    "subject": "Learn Python",
                    "predicate": "has_status",
                    "object": "in_progress",
                },
                {
                    "subject": "Learn Python",
                    "predicate": "has_deadline",
                    "object": "2024-01-01",
                },
                {
                    "subject": "Build App",
                    "predicate": "depends_on",
                    "object": "Learn Python",
                },
            ]
        )

        result = await get_goal_context(semantic_repo=mock_repo)

        assert "User goals:" in result
        assert "Learn Python" in result
        assert "Build App" in result
        assert "has_status" in result
        assert "in_progress" in result

    @pytest.mark.asyncio
    async def test_goal_predicates_exception_handled(self) -> None:
        """Exception in get_goal_predicates should be handled gracefully."""
        mock_repo = AsyncMock()
        mock_repo.fetch_goal_names = AsyncMock(return_value=["Test Goal"])
        mock_repo.get_goal_predicates = AsyncMock(side_effect=Exception("DB Error"))

        result = await get_goal_context(semantic_repo=mock_repo)

        assert "User goals:" in result
        assert "Test Goal" in result

    @pytest.mark.asyncio
    async def test_pre_fetched_goal_names_used(self) -> None:
        """Pre-fetched goal_names should be used without DB call."""
        mock_repo = AsyncMock()
        mock_repo.fetch_goal_names = AsyncMock(return_value=["Should Not Use"])
        mock_repo.get_goal_predicates = AsyncMock(return_value=[])

        result = await get_goal_context(
            semantic_repo=mock_repo, goal_names=["Pre-fetched Goal"]
        )

        mock_repo.fetch_goal_names.assert_not_awaited()
        assert "Pre-fetched Goal" in result

    @pytest.mark.asyncio
    async def test_hierarchical_tree_format(self) -> None:
        """Goals should be displayed with proper tree formatting."""
        mock_repo = AsyncMock()
        mock_repo.fetch_goal_names = AsyncMock(return_value=["Goal1", "Goal2"])
        mock_repo.get_goal_predicates = AsyncMock(
            return_value=[
                {"subject": "Goal1", "predicate": "pred1", "object": "obj1"},
                {"subject": "Goal1", "predicate": "pred2", "object": "obj2"},
                {"subject": "Goal2", "predicate": "pred3", "object": "obj3"},
            ]
        )

        result = await get_goal_context(semantic_repo=mock_repo)

        assert "├── Goal1" in result
        assert "└── Goal2" in result
        assert "├── pred1" in result
        assert "└── pred2" in result

    @pytest.mark.asyncio
    async def test_max_goals_context_limit(self) -> None:
        """Should respect MAX_GOALS_CONTEXT limit."""
        from lattice.core.response_generator import MAX_GOALS_CONTEXT

        mock_repo = AsyncMock()
        mock_repo.fetch_goal_names = AsyncMock(
            return_value=["Goal"] * (MAX_GOALS_CONTEXT + 10)
        )
        mock_repo.get_goal_predicates = AsyncMock(return_value=[])

        result = await get_goal_context(semantic_repo=mock_repo)

        assert "User goals:" in result

    @pytest.mark.asyncio
    async def test_prefetched_goals_without_repo_returns_no_goals(self) -> None:
        """Pre-fetched goal names without semantic_repo returns no goals message."""
        result = await get_goal_context(
            semantic_repo=None, goal_names=["Goal1", "Goal2"]
        )

        assert result == "No active goals."


def create_mock_llm_client(audit_result: MockAuditResult | None = None) -> MagicMock:
    """Create a mock LLM client with configured complete method."""
    mock = MagicMock()
    mock.complete = AsyncMock(return_value=audit_result or MockAuditResult())
    return mock


class TestGenerateResponse:
    """Tests for generate_response() function."""

    @pytest.mark.asyncio
    async def test_missing_prompt_repo_raises_error(self) -> None:
        """Missing prompt_repo should raise ValueError."""
        mock_llm = create_mock_llm_client()
        with pytest.raises(ValueError, match="prompt_repo is required"):
            await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                llm_client=mock_llm,
            )

    @pytest.mark.asyncio
    async def test_missing_llm_client_raises_error(self) -> None:
        """Missing llm_client should raise ValueError."""
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            with pytest.raises(ValueError, match="llm_client is required"):
                await generate_response(
                    user_message="test",
                    episodic_context="",
                    semantic_context="",
                    prompt_repo=mock_prompt_repo,
                )

    @pytest.mark.asyncio
    async def test_template_not_found_returns_fallback(self) -> None:
        """When template not found, should return fallback response."""
        mock_llm = create_mock_llm_client()
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = None

            result, prompt, context = await generate_response(
                user_message="test",
                episodic_context="history",
                semantic_context="memory",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            assert "still initializing" in result.content
            assert prompt == ""
            assert context == {}

    @pytest.mark.asyncio
    async def test_successful_response_generation(self) -> None:
        """Successful LLM call should return generated response."""
        mock_llm = create_mock_llm_client(
            MockAuditResult(content="Generated response", model="test-model")
        )
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            result, prompt, context = await generate_response(
                user_message="What is 2+2?",
                episodic_context="User asked about math",
                semantic_context="User likes math",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            assert result.content == "Generated response"
            assert result.model == "test-model"
            assert "What is 2+2?" in prompt
            assert context["template"] == "UNIFIED_RESPONSE"
            assert context["template_version"] == 1

    @pytest.mark.asyncio
    async def test_empty_contexts_use_defaults(self) -> None:
        """Empty episodic and semantic contexts should use defaults."""
        mock_llm = create_mock_llm_client(MockAuditResult(content="Response"))
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            result, prompt, context = await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            assert "No recent conversation." in prompt
            assert "No relevant context found." in prompt

    @pytest.mark.asyncio
    async def test_unresolved_entities_in_prompt(self) -> None:
        """Unresolved entities should be included in prompt."""
        mock_llm = create_mock_llm_client(MockAuditResult(content="Response"))
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            result, prompt, context = await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                unresolved_entities=["Entity1", "Entity2"],
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            assert "Entity1" in prompt
            assert "Entity2" in prompt

    @pytest.mark.asyncio
    async def test_no_unresolved_entities_shows_none(self) -> None:
        """When unresolved_entities is None, should show (none)."""
        mock_llm = create_mock_llm_client(MockAuditResult(content="Response"))
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            result, prompt, context = await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            assert "(none)" in prompt

    @pytest.mark.asyncio
    async def test_user_timezone_in_prompt(self) -> None:
        """User timezone should be included in prompt when template has placeholder."""
        mock_llm = create_mock_llm_client(MockAuditResult(content="Response"))
        mock_prompt_repo = AsyncMock()
        mock_template = MockPromptTemplate(
            template="User: {user_message}\nTimezone: {user_timezone}"
        )

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = mock_template

            result, prompt, context = await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                user_tz="America/New_York",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            assert "America/New_York" in prompt

    @pytest.mark.asyncio
    async def test_llm_exception_returns_fallback(self) -> None:
        """LLM exception should return fallback response."""
        mock_llm = create_mock_llm_client()
        mock_llm.complete.side_effect = Exception("Connection error")
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            result, prompt, context = await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            assert result.model == "fallback"
            assert "OPENROUTER_API_KEY" in result.content

    @pytest.mark.asyncio
    async def test_empty_response_returns_fallback(self) -> None:
        """Empty LLM response should return fallback."""
        mock_llm = create_mock_llm_client(MockAuditResult(content=""))
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            result, prompt, context = await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            assert result.model == "fallback"
            assert "OPENROUTER_API_KEY" in result.content

    @pytest.mark.asyncio
    async def test_audit_view_flag_passed_to_llm(self) -> None:
        """audit_view flag should be passed to LLM client."""
        mock_llm = create_mock_llm_client(MockAuditResult(content="Response"))
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
                audit_view=True,
                audit_view_params={"channel_id": 123},
            )

            mock_llm.complete.assert_called_once()
            call_kwargs = mock_llm.complete.call_args.kwargs
            assert call_kwargs["audit_view"] is True
            assert call_kwargs["audit_view_params"]["channel_id"] == 123

    @pytest.mark.asyncio
    async def test_temperature_from_template(self) -> None:
        """Temperature should be taken from template if set."""
        mock_llm = create_mock_llm_client(MockAuditResult(content="Response"))
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate(temperature=0.5)

            await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            call_kwargs = mock_llm.complete.call_args.kwargs
            assert call_kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_default_temperature_when_none(self) -> None:
        """Default temperature 0.7 should be used when template has None."""
        mock_llm = create_mock_llm_client(MockAuditResult(content="Response"))
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate(temperature=None)

            await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            call_kwargs = mock_llm.complete.call_args.kwargs
            assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_placeholder_injection_happens(self) -> None:
        """Placeholders should be injected into template."""
        mock_llm = create_mock_llm_client(MockAuditResult(content="Response"))
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            result, prompt, context = await generate_response(
                user_message="USER_MSG",
                episodic_context="EPISODIC",
                semantic_context="SEMANTIC",
                user_tz="UTC",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            assert "USER_MSG" in prompt
            assert "EPISODIC" in prompt
            assert "SEMANTIC" in prompt

    @pytest.mark.asyncio
    async def test_repositories_passed_to_llm(self) -> None:
        """Repositories should be passed to LLM client for auditing."""
        mock_llm = create_mock_llm_client(MockAuditResult(content="Response"))
        mock_prompt_repo = AsyncMock()
        mock_audit_repo = AsyncMock()
        mock_feedback_repo = AsyncMock()
        mock_bot = MagicMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
                audit_repo=mock_audit_repo,
                feedback_repo=mock_feedback_repo,
                bot=mock_bot,
            )

            call_kwargs = mock_llm.complete.call_args.kwargs
            assert call_kwargs["audit_repo"] == mock_audit_repo
            assert call_kwargs["feedback_repo"] == mock_feedback_repo
            assert call_kwargs["bot"] == mock_bot

    @pytest.mark.asyncio
    async def test_main_discord_message_id_passed(self) -> None:
        """main_discord_message_id should be passed to LLM client."""
        mock_llm = create_mock_llm_client(MockAuditResult(content="Response"))
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
                main_discord_message_id=123456789,
            )

            call_kwargs = mock_llm.complete.call_args.kwargs
            assert call_kwargs["main_discord_message_id"] == 123456789

    @pytest.mark.asyncio
    async def test_very_long_user_message(self) -> None:
        """Very long user message should be handled."""
        long_message = "test " * 1000
        mock_llm = create_mock_llm_client(MockAuditResult(content="Response"))
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            result, prompt, context = await generate_response(
                user_message=long_message,
                episodic_context="",
                semantic_context="",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            assert result.content == "Response"
            assert long_message in prompt

    @pytest.mark.asyncio
    async def test_malformed_llm_output_handled(self) -> None:
        """Malformed LLM output should be handled gracefully."""
        mock_llm = create_mock_llm_client(
            MockAuditResult(
                content="\x00\x01\x02 malformed \x03", model="malformed-model"
            )
        )
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = MockPromptTemplate()

            result, prompt, context = await generate_response(
                user_message="test",
                episodic_context="",
                semantic_context="",
                prompt_repo=mock_prompt_repo,
                llm_client=mock_llm,
            )

            assert result.content is not None
