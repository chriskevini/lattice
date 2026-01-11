"""Unit tests for response template selection and generation."""

import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from lattice.utils.date_resolution import get_now
from unittest.mock import ANY, AsyncMock, patch

import pytest
from lattice.core.context_strategy import ContextStrategy
from lattice.core.response_generator import (
    generate_response,
    get_available_placeholders,
    validate_template_placeholders,
)
from lattice.memory.episodic import EpisodicMessage
from lattice.memory.procedural import PromptTemplate
from lattice.utils.llm import AuditResult


@pytest.fixture
def mock_extraction_declaration() -> ContextStrategy:
    """Create a mock ContextStrategy for goal message type."""
    return ContextStrategy(
        id=uuid.uuid4(),
        message_id=uuid.uuid4(),
        entities=["lattice project", "Friday"],
        context_flags=["goal_context"],
        unresolved_entities=[],
        rendered_prompt="test prompt",
        raw_response="test response",
        strategy_method="api",
        created_at=get_now(timezone_str="UTC"),
    )


@pytest.fixture
def mock_extraction_query() -> ContextStrategy:
    """Create a mock ContextStrategy for question message type."""
    return ContextStrategy(
        id=uuid.uuid4(),
        message_id=uuid.uuid4(),
        entities=["lattice", "deadline"],
        context_flags=[],
        unresolved_entities=[],
        rendered_prompt="test prompt",
        raw_response="test response",
        strategy_method="api",
        created_at=get_now(timezone_str="UTC"),
    )


@pytest.fixture
def mock_extraction_activity() -> ContextStrategy:
    """Create a mock ContextStrategy for activity_update message type."""
    return ContextStrategy(
        id=uuid.uuid4(),
        message_id=uuid.uuid4(),
        entities=["coding"],
        context_flags=["activity_context"],
        unresolved_entities=[],
        rendered_prompt="test prompt",
        raw_response="test response",
        strategy_method="api",
        created_at=get_now(timezone_str="UTC"),
    )


@pytest.fixture
def mock_extraction_conversation() -> ContextStrategy:
    """Create a mock ContextStrategy for conversation message type."""
    return ContextStrategy(
        id=uuid.uuid4(),
        message_id=uuid.uuid4(),
        entities=["tea"],
        context_flags=[],
        unresolved_entities=[],
        rendered_prompt="test prompt",
        raw_response="test response",
        strategy_method="api",
        created_at=get_now(timezone_str="UTC"),
    )


@pytest.fixture
def mock_recent_messages() -> list[EpisodicMessage]:
    """Create mock recent messages for context."""
    return [
        EpisodicMessage(
            content="I need to finish the lattice project by Friday",
            discord_message_id=1234567890,
            channel_id=9876543210,
            is_bot=False,
            message_id=uuid.uuid4(),
            timestamp=datetime(2026, 1, 4, 10, 0, tzinfo=ZoneInfo("UTC")),
        ),
        EpisodicMessage(
            content="Got it! Friday deadline for lattice.",
            discord_message_id=1234567891,
            channel_id=9876543210,
            is_bot=True,
            message_id=uuid.uuid4(),
            timestamp=datetime(2026, 1, 4, 10, 1, tzinfo=ZoneInfo("UTC")),
        ),
    ]


@pytest.fixture
def mock_prompt_template() -> PromptTemplate:
    """Create a mock UNIFIED_RESPONSE template."""
    return PromptTemplate(
        prompt_key="UNIFIED_RESPONSE",
        template=(
            "Context: {episodic_context}\n"
            "Semantic: {semantic_context}\n"
            "User: {user_message}"
        ),
        temperature=0.7,
        version=1,
        active=True,
    )


@pytest.fixture
def mock_generation_result() -> AuditResult:
    """Create a mock AuditResult."""
    return AuditResult(
        content="Got it! Friday deadline for lattice. That's coming up quick—how's it looking so far?",
        model="anthropic/claude-3.5-sonnet",
        provider="anthropic",
        prompt_tokens=150,
        completion_tokens=30,
        total_tokens=180,
        cost_usd=0.002,
        latency_ms=600,
        temperature=0.7,
        audit_id=None,
        prompt_key="UNIFIED_RESPONSE",
    )


class TestPlaceholderRegistry:
    """Tests for placeholder registry and validation."""

    def test_get_available_placeholders(self) -> None:
        """Test retrieving the canonical placeholder list."""
        placeholders = get_available_placeholders()

        # Check core placeholders are present
        assert "episodic_context" in placeholders
        assert "semantic_context" in placeholders
        assert "user_message" in placeholders

        # Check descriptions exist
        assert isinstance(placeholders["user_message"], str)
        assert len(placeholders["user_message"]) > 0

    def test_validate_template_with_known_placeholders(self) -> None:
        """Test validation succeeds for templates with known placeholders."""
        template = "Hello {user_message}! Context: {episodic_context}"
        is_valid, unknown = validate_template_placeholders(template)

        assert is_valid is True
        assert unknown == []

    def test_validate_template_with_unknown_placeholders(self) -> None:
        """Test validation fails for templates with unknown placeholders."""
        template = "Hello {user_message}! Unknown: {fake_var} and {another_fake}"
        is_valid, unknown = validate_template_placeholders(template)

        assert is_valid is False
        assert "fake_var" in unknown
        assert "another_fake" in unknown
        assert "user_message" not in unknown

    def test_validate_template_with_no_placeholders(self) -> None:
        """Test validation succeeds for static templates."""
        template = "This is a static template with no placeholders."
        is_valid, unknown = validate_template_placeholders(template)

        assert is_valid is True
        assert unknown == []

    def test_validate_template_with_extraction_placeholders(self) -> None:
        """Test validation succeeds for core placeholders only (Design D)."""
        template = "Context: {episodic_context}\nUser: {user_message}"
        is_valid, unknown = validate_template_placeholders(template)

        assert is_valid is True
        assert unknown == []

    def test_placeholder_registry_completeness(self) -> None:
        """Test that registry includes core placeholders only (Design D)."""
        # These are the placeholders that generate_response() actually populates
        expected_placeholders = {
            "episodic_context",
            "semantic_context",
            "user_message",
        }

        available = set(get_available_placeholders().keys())

        # All expected placeholders should be documented
        missing = expected_placeholders - available
        assert missing == set(), f"Missing placeholders in registry: {missing}"


class TestUnifiedResponseTemplate:
    """Tests verifying UNIFIED_RESPONSE is used for all message types.

    Since the refactor removed select_response_template() and always uses
    UNIFIED_RESPONSE, these tests verify the integration point where the
    template is selected in generate_response().
    """

    @pytest.mark.asyncio
    async def test_generate_response_uses_unified_template_for_goal(
        self,
        mock_extraction_declaration: ContextStrategy,
        mock_recent_messages: list[EpisodicMessage],
        mock_generation_result: AuditResult,
    ) -> None:
        """Test that UNIFIED_RESPONSE is selected for goal-type messages."""
        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template="Context: {episodic_context}\nUser: {user_message}",
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            await generate_response(
                user_message="I need to finish the lattice project by Friday",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
            )

            mock_get_prompt.assert_any_call(db_pool=ANY, prompt_key="UNIFIED_RESPONSE")


class TestGenerateResponseWithTemplates:
    """Tests for generate_response with template selection."""

    @pytest.fixture(autouse=True)
    def reset_auditing_client(self):
        """Reset global auditing client before and after each test."""
        import lattice.utils.llm

        lattice.utils.llm._auditing_client = None
        yield
        lattice.utils.llm._auditing_client = None

    @pytest.mark.asyncio
    async def test_generate_with_goal_extraction(
        self,
        mock_extraction_declaration: ContextStrategy,
        mock_recent_messages: list[EpisodicMessage],
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
    ) -> None:
        """Test response generation with goal extraction."""
        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="I need to finish the lattice project by Friday",
                episodic_context="Recent conversation history",
                semantic_context="I need to finish the lattice project by Friday",
            )

            # Verify template selection
            mock_get_prompt.assert_any_call(db_pool=ANY, prompt_key="UNIFIED_RESPONSE")

            # Verify extraction fields in rendered prompt
            assert "I need to finish the lattice project by Friday" in rendered_prompt

            # Verify context info includes extraction
            assert context_info["template"] == "UNIFIED_RESPONSE"

            # Verify result
            assert result == mock_generation_result

    @pytest.mark.asyncio
    async def test_generate_with_context_strategy(
        self,
        mock_extraction_query: ContextStrategy,
        mock_recent_messages: list[EpisodicMessage],
        mock_generation_result: AuditResult,
    ) -> None:
        """Test response generation with entity extraction."""
        mock_template = PromptTemplate(
            prompt_key="UNIFIED_RESPONSE",
            template=("Context: {episodic_context}\nUser: {user_message}"),
            temperature=0.5,
            version=1,
            active=True,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="When is the lattice deadline?",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
            )

            # Verify template selection
            mock_get_prompt.assert_any_call(db_pool=ANY, prompt_key="UNIFIED_RESPONSE")

            # User message should be in prompt
            assert "When is the lattice deadline?" in rendered_prompt

            # Verify temperature from template is used
            mock_client.complete.assert_called_once()
            call_kwargs = mock_client.complete.call_args
            assert call_kwargs[1]["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_with_activity_extraction(
        self,
        mock_extraction_activity: ContextStrategy,
        mock_recent_messages: list[EpisodicMessage],
        mock_generation_result: AuditResult,
    ) -> None:
        """Test response generation with activity update extraction."""
        mock_template = PromptTemplate(
            prompt_key="UNIFIED_RESPONSE",
            template=("Context: {episodic_context}\nUser: {user_message}"),
            temperature=0.7,
            version=1,
            active=True,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Spent 180 minutes coding today",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
            )

            # Verify template selection
            mock_get_prompt.assert_any_call(db_pool=ANY, prompt_key="UNIFIED_RESPONSE")

            # User message should be in prompt
            assert "Spent 180 minutes coding today" in rendered_prompt

    @pytest.mark.asyncio
    async def test_generate_with_none_extraction(
        self,
        mock_recent_messages: list[EpisodicMessage],
        mock_generation_result: AuditResult,
    ) -> None:
        """Test response generation without extraction."""
        mock_template = PromptTemplate(
            prompt_key="UNIFIED_RESPONSE",
            template=("Context: {episodic_context}\nUser: {user_message}"),
            temperature=0.7,
            version=1,
            active=True,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Hello",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
            )

            # Verify UNIFIED_RESPONSE is used
            mock_get_prompt.assert_any_call(db_pool=ANY, prompt_key="UNIFIED_RESPONSE")

            # Verify context info includes template info
            assert context_info["template"] == "UNIFIED_RESPONSE"

    @pytest.mark.asyncio
    async def test_generate_template_fallback(
        self,
        mock_extraction_declaration: ContextStrategy,
        mock_recent_messages: list[EpisodicMessage],
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
    ) -> None:
        """Test fallback to UNIFIED_RESPONSE when template doesn't exist."""
        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            # Template lookup returns None (simulating template not found)
            mock_get_prompt.return_value = None
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="I need to finish the lattice project by Friday",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
            )

            # Verify UNIFIED_RESPONSE was tried once
            mock_get_prompt.assert_any_call(db_pool=ANY, prompt_key="UNIFIED_RESPONSE")

    @pytest.mark.asyncio
    async def test_generate_with_graph_memories(
        self,
        mock_extraction_query: ContextStrategy,
        mock_recent_messages: list[EpisodicMessage],
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
    ) -> None:
        """Test response generation includes graph memories in context."""
        # Text-based memories use 'subject' and 'object' keys

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="When is the lattice deadline?",
                episodic_context="Recent conversation history",
                semantic_context="lattice project has deadline Friday\nuser working on lattice project",
            )

            # Verify graph memories are formatted in semantic context
            assert "lattice project has deadline Friday" in rendered_prompt
            assert "user working on lattice project" in rendered_prompt

            # Verify context info includes template
            assert context_info["template"] == "UNIFIED_RESPONSE"

    @pytest.mark.asyncio
    async def test_generate_empty_extraction_fields(
        self,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test handling of extraction with empty/None fields."""
        ContextStrategy(
            id=uuid.uuid4(),
            message_id=uuid.uuid4(),
            entities=[],  # Empty list
            context_flags=[],
            unresolved_entities=[],
            rendered_prompt="test",
            raw_response="test",
            strategy_method="api",
            created_at=get_now(timezone_str="UTC"),
        )

        mock_template = PromptTemplate(
            prompt_key="UNIFIED_RESPONSE",
            template=("Context: {episodic_context}\nUser: {user_message}"),
            temperature=0.7,
            version=1,
            active=True,
        )

        mock_result = AuditResult(
            content="Response",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
            audit_id=None,
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Hello",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
            )

            # Verify empty fields are handled gracefully (no extraction fields in Design D)
            assert "Hello" in rendered_prompt

    @pytest.mark.asyncio
    async def test_filter_current_message_from_episodic_context(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test that current user message is filtered from episodic context."""
        user_message_content = "What's the weather?"
        user_discord_id = 9999999999

        # Create conversation including the current message
        [
            EpisodicMessage(
                content="Hello!",
                discord_message_id=1111111111,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 0, tzinfo=ZoneInfo("UTC")),
            ),
            EpisodicMessage(
                content="Hi there!",
                discord_message_id=2222222222,
                channel_id=123,
                is_bot=True,
                timestamp=datetime(2026, 1, 5, 10, 1, tzinfo=ZoneInfo("UTC")),
            ),
            EpisodicMessage(
                content=user_message_content,  # Current message (should be filtered)
                discord_message_id=user_discord_id,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 2, tzinfo=ZoneInfo("UTC")),
            ),
        ]

        mock_result = AuditResult(
            content="It's sunny!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
            audit_id=None,
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Hello!",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
            )

            # Verify current message is NOT in episodic context
            # Count occurrences of the message content in rendered prompt
            message_occurrences = rendered_prompt.count("Hello!")

            # Should appear exactly once (in {user_message} placeholder)
            # NOT in episodic_context
            assert message_occurrences == 1, (
                f"Expected user message to appear exactly once, "
                f"but found {message_occurrences} occurrences"
            )

            # Verify previous messages ARE in episodic context (if we passed them)
            # In this test we passed "Recent conversation history" as episodic_context
            assert "Recent conversation history" in rendered_prompt

    @pytest.mark.asyncio
    async def test_handle_duplicate_message_content(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test that ID-based filtering handles duplicate message content correctly."""
        duplicate_content = "What's the weather?"

        # User sends same message twice
        [
            EpisodicMessage(
                content="Hello!",
                discord_message_id=1111111111,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 0, tzinfo=ZoneInfo("UTC")),
            ),
            EpisodicMessage(
                content=duplicate_content,  # First time asking
                discord_message_id=2222222222,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 1, tzinfo=ZoneInfo("UTC")),
            ),
            EpisodicMessage(
                content="It's sunny!",
                discord_message_id=3333333333,
                channel_id=123,
                is_bot=True,
                timestamp=datetime(2026, 1, 5, 10, 2, tzinfo=ZoneInfo("UTC")),
            ),
            EpisodicMessage(
                content=duplicate_content,  # Asking again (current message)
                discord_message_id=4444444444,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 3, tzinfo=ZoneInfo("UTC")),
            ),
        ]

        mock_result = AuditResult(
            content="Still sunny!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
            audit_id=None,
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message=duplicate_content,
                episodic_context=f"Hello!\n{duplicate_content}\nIt's sunny!",
                semantic_context="Relevant facts",
            )

            # Count occurrences of duplicate content in rendered prompt
            message_occurrences = rendered_prompt.count(duplicate_content)

            # Should appear exactly TWICE:
            # 1. Once in episodic_context (the first time at 10:01)
            # 2. Once in {user_message} placeholder (current message)
            # NOT three times (which would happen with buggy content-based filtering)
            assert message_occurrences == 2, (
                f"Expected duplicate message to appear twice "
                f"(once in history, once as current), "
                f"but found {message_occurrences} occurrences"
            )

            # Verify the first instance is still in context
            assert "Hello!" in rendered_prompt
            assert "It's sunny!" in rendered_prompt

    @pytest.mark.asyncio
    async def test_fallback_to_content_filtering_without_message_id(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test backward compatibility: content-based filtering when ID not provided."""
        user_message_content = "What's the weather?"

        [
            EpisodicMessage(
                content="Hello!",
                discord_message_id=1111111111,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 0, tzinfo=ZoneInfo("UTC")),
            ),
            EpisodicMessage(
                content=user_message_content,  # Should be filtered by content
                discord_message_id=2222222222,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 1, tzinfo=ZoneInfo("UTC")),
            ),
        ]

        mock_result = AuditResult(
            content="It's sunny!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
            audit_id=None,
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            # Don't pass user_discord_message_id (backward compatibility test)
            result, rendered_prompt, context_info = await generate_response(
                user_message=user_message_content,
                episodic_context="Hello!",
                semantic_context="Relevant facts",
            )

            # Should still filter by content (fallback behavior)
            message_occurrences = rendered_prompt.count(user_message_content)
            assert message_occurrences == 1

            # Previous messages should be preserved
            assert "Hello!" in rendered_prompt


class TestExtractionFieldsNotInPrompts:
    """Tests to verify extraction fields are not included in specialized templates.

    As of migration 018 (2026-01-05), extraction fields should be excluded from
    prompts sent to the LLM to avoid redundancy. Modern LLMs can extract this
    information naturally from the user message.
    """

    @pytest.mark.asyncio
    async def test_goal_response_no_extraction_fields(
        self,
        mock_extraction_declaration: ContextStrategy,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test UNIFIED_RESPONSE template does not include extraction fields in prompt."""
        # Use the actual template from migration 018 (no extraction section)
        mock_template = PromptTemplate(
            prompt_key="UNIFIED_RESPONSE",
            template=(
                "You are a warm, supportive AI companion.\n\n"
                "## Context\n"
                "**Recent conversation history:**\n{episodic_context}\n\n"
                "**Relevant facts from past conversations:**\n{semantic_context}\n\n"
                "**User message:** {user_message}\n\n"
                "## Your Task\n"
                "The user has declared a goal. Respond naturally."
            ),
            temperature=0.7,
            version=2,
            active=True,
        )

        mock_result = AuditResult(
            content="Got it!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
            audit_id=None,
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="When is the lattice deadline?",
                episodic_context="Recent conversation history",
                semantic_context="I need to finish the lattice project by Friday",
            )

            # Verify extraction fields are NOT in the rendered prompt
            assert "Extracted information:" not in rendered_prompt
            assert "Entities mentioned:" not in rendered_prompt
            assert "Time constraint:" not in rendered_prompt
            assert "Urgency:" not in rendered_prompt

            # Verify core fields ARE present
            assert "I need to finish the lattice project by Friday" in rendered_prompt
            assert "Your Task" in rendered_prompt

    @pytest.mark.asyncio
    async def test_query_response_no_extraction_fields(
        self,
        mock_extraction_query: ContextStrategy,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test QUERY_RESPONSE template does not include extraction fields in prompt."""
        mock_template = PromptTemplate(
            prompt_key="QUERY_RESPONSE",
            template=(
                "You are a helpful AI companion.\n\n"
                "## Context\n"
                "**Recent conversation history:**\n{episodic_context}\n\n"
                "**Relevant facts from past conversations:**\n{semantic_context}\n\n"
                "**User message:** {user_message}\n\n"
                "## Your Task\n"
                "Answer the query directly."
            ),
            temperature=0.5,
            version=2,
            active=True,
        )

        mock_result = AuditResult(
            content="The deadline is Friday.",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.5,
            audit_id=None,
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="When is the lattice deadline?",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
            )

            # Verify extraction fields are NOT in the rendered prompt
            assert "Extracted information:" not in rendered_prompt
            assert "Query reformulation:" not in rendered_prompt
            assert "Entities mentioned:" not in rendered_prompt

            # Verify core fields ARE present
            assert "When is the lattice deadline?" in rendered_prompt

    @pytest.mark.asyncio
    async def test_activity_response_no_extraction_fields(
        self,
        mock_extraction_activity: ContextStrategy,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test UNIFIED_RESPONSE template does not include extraction fields."""
        mock_template = PromptTemplate(
            prompt_key="UNIFIED_RESPONSE",
            template=(
                "You are a friendly AI companion.\n\n"
                "## Context\n"
                "**Recent conversation history:**\n{episodic_context}\n\n"
                "**Relevant facts from past conversations:**\n{semantic_context}\n\n"
                "**User message:** {user_message}\n\n"
                "## Your Task\n"
                "Respond naturally based on the user's message."
            ),
            temperature=0.7,
            version=2,
            active=True,
        )

        mock_result = AuditResult(
            content="Nice session!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
            audit_id=None,
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Spent 180 minutes coding today",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
            )

            # Verify extraction fields are NOT in the rendered prompt
            assert "Extracted information:" not in rendered_prompt
            assert "Activity:" not in rendered_prompt
            assert "Entities mentioned:" not in rendered_prompt

            # Verify core fields ARE present
            assert "Spent 180 minutes coding today" in rendered_prompt

    @pytest.mark.asyncio
    async def test_conversation_response_no_extraction_fields(
        self,
        mock_extraction_conversation: ContextStrategy,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test UNIFIED_RESPONSE template does not include extraction fields."""
        mock_template = PromptTemplate(
            prompt_key="UNIFIED_RESPONSE",
            template=(
                "You are a warm AI companion.\n\n"
                "## Context\n"
                "**Recent conversation history:**\n{episodic_context}\n\n"
                "**Relevant facts from past conversations:**\n{semantic_context}\n\n"
                "**User message:** {user_message}\n\n"
                "## Your Task\n"
                "Engage naturally in conversation."
            ),
            temperature=0.7,
            version=2,
            active=True,
        )

        mock_result = AuditResult(
            content="Nice! What kind?",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
            audit_id=None,
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Spent 180 minutes coding today",
                episodic_context="Just made some tea",
                semantic_context="Relevant facts",
            )

            # Verify extraction fields are NOT in the rendered prompt
            assert "Extracted information:" not in rendered_prompt
            assert "Is continuation:" not in rendered_prompt
            assert "Entities mentioned:" not in rendered_prompt

            # Verify core fields ARE present
            assert "Just made some tea" in rendered_prompt

    @pytest.mark.asyncio
    async def test_extraction_data_still_populated_internally(
        self,
        mock_extraction_declaration: ContextStrategy,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test that extraction data is still available for routing/analytics.

        Even though extraction fields aren't shown to the LLM, they should still
        be populated internally for template selection and analytics purposes.
        """
        mock_template = PromptTemplate(
            prompt_key="UNIFIED_RESPONSE",
            template="User: {user_message}",  # Minimal template, no extraction fields
            temperature=0.7,
            version=2,
            active=True,
        )

        mock_result = AuditResult(
            content="Got it!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
            audit_id=None,
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="When is the lattice deadline?",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
            )

            # Verify extraction was used for template selection
            mock_get_prompt.assert_any_call(db_pool=ANY, prompt_key="UNIFIED_RESPONSE")

            # Verify extraction metadata is in context_info for analytics
            assert context_info["template"] == "UNIFIED_RESPONSE"
            assert context_info["template_version"] == 2


class TestNoTemplateFound:
    """Tests for error handling when no templates are available."""

    @pytest.mark.asyncio
    async def test_generate_response_no_template_found(
        self,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test fallback response when no templates are available in database."""
        ContextStrategy(
            id=uuid.uuid4(),
            message_id=uuid.uuid4(),
            entities=["test"],
            context_flags=[],
            unresolved_entities=[],
            rendered_prompt="test",
            raw_response="test",
            strategy_method="api",
            created_at=get_now(timezone_str="UTC"),
        )

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            # Simulate template lookup returning None
            mock_get_prompt.return_value = None

            result, rendered_prompt, context_info = await generate_response(
                user_message="I need to finish the lattice project by Friday",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
            )

            # Verify fallback response is returned
            assert (
                result.content
                == "I'm still initializing. Please try again in a moment."
            )
            assert result.model == "unknown"
            assert result.provider is None
            assert result.prompt_tokens == 0
            assert result.completion_tokens == 0
            assert result.total_tokens == 0
            assert result.cost_usd is None
            assert result.latency_ms == 0
            assert result.temperature == 0.0

            # Verify empty rendered prompt and context info
            assert rendered_prompt == ""
            assert context_info == {}

            # Verify UNIFIED_RESPONSE was tried once
            mock_get_prompt.assert_any_call(db_pool=ANY, prompt_key="UNIFIED_RESPONSE")


class TestTimezoneConversionFailure:
    """Tests for timezone conversion error handling."""

    @pytest.mark.asyncio
    async def test_invalid_timezone_falls_back_to_utc(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test that invalid timezone falls back to UTC formatting.

        When an invalid timezone is provided, the function should:
        1. Log a warning (verified manually in test output)
        2. Fall back to UTC formatting
        3. Continue processing without raising an exception
        """
        # Create message with invalid timezone
        [
            EpisodicMessage(
                content="Hello!",
                discord_message_id=1234567890,
                channel_id=123,
                is_bot=False,
                user_timezone="Invalid/Timezone",  # Invalid timezone
                timestamp=datetime(2026, 1, 4, 10, 0, tzinfo=ZoneInfo("UTC")),
            ),
        ]

        mock_result = AuditResult(
            content="Hi!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
            audit_id=None,
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch(
                "lattice.core.response_generator.get_auditing_llm_client"
            ) as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            # Should not raise an exception
            result, rendered_prompt, context_info = await generate_response(
                user_message="Test message",
                episodic_context="2026-01-04 10:00 UTC\nHello!",
                semantic_context="Relevant facts",
            )

            # Verify fallback to UTC formatting (the key behavior to test)
            assert "2026-01-04 10:00 UTC" in rendered_prompt
            assert "Hello!" in rendered_prompt

            # Verify generation completed successfully
            assert result.content == "Hi!"


class TestSplitResponse:
    """Tests for split_response function."""

    def test_split_response_short_message(self) -> None:
        """Test that short messages are not split."""
        from lattice.core.response_generator import split_response

        short_message = "This is a short message."
        chunks = split_response(short_message, max_length=1900)

        assert len(chunks) == 1
        assert chunks[0] == short_message

    def test_split_response_exact_limit(self) -> None:
        """Test message exactly at the limit."""
        from lattice.core.response_generator import split_response

        # Create a message exactly 1900 chars
        message = "a" * 1900
        chunks = split_response(message, max_length=1900)

        assert len(chunks) == 1
        assert chunks[0] == message

    def test_split_response_long_message_with_newlines(self) -> None:
        """Test that long messages are split at newlines."""
        from lattice.core.response_generator import split_response

        # Create a message that exceeds the limit with newlines
        lines = [f"Line {i}: " + ("x" * 100) for i in range(30)]
        long_message = "\n".join(lines)

        chunks = split_response(long_message, max_length=500)

        # Verify multiple chunks created
        assert len(chunks) > 1, "Long message should be split into multiple chunks"

        # Verify each chunk is within limit
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 500, f"Chunk {i} length {len(chunk)} exceeds limit 500"

        # Verify all lines are preserved in order
        rejoined = "\n".join(chunks)
        # Each original line should appear exactly once in the rejoined content
        for line in lines:
            assert rejoined.count(line) == 1, (
                f"Line '{line[:50]}...' should appear exactly once"
            )

    def test_split_response_preserves_content(self) -> None:
        """Test that splitting preserves all content."""
        from lattice.core.response_generator import split_response

        lines = ["First line", "Second line", "Third line", "Fourth line"]
        message = "\n".join(lines)

        chunks = split_response(message, max_length=20)

        # Verify all lines are present in some chunk
        all_content = "\n".join(chunks)
        for line in lines:
            assert line in all_content

    def test_split_response_single_long_line(self) -> None:
        """Test splitting a single line that exceeds the limit."""
        from lattice.core.response_generator import split_response

        # Single line longer than limit (2000 chars, limit is 1900)
        long_line = "x" * 2000

        chunks = split_response(long_line, max_length=1900)

        # Should split the long line into multiple chunks
        assert len(chunks) > 1, "Long line should be split into multiple chunks"

        # Each chunk should be within the limit
        for chunk in chunks:
            assert len(chunk) <= 1900, f"Chunk length {len(chunk)} exceeds limit 1900"

        # All content should be preserved
        rejoined = "".join(chunks)
        assert rejoined == long_line, "Content should be preserved after splitting"

    def test_split_response_long_line_with_spaces(self) -> None:
        """Test splitting a long line with spaces at word boundaries."""
        from lattice.core.response_generator import split_response

        # Create a long line with spaces (should split at word boundaries)
        words = ["word" * 10 for _ in range(50)]  # 50 words of 40 chars each
        long_line = " ".join(words)  # ~2050 chars with spaces

        chunks = split_response(long_line, max_length=1900)

        # Should split into multiple chunks
        assert len(chunks) > 1

        # Each chunk should respect the limit
        for chunk in chunks:
            assert len(chunk) <= 1900, f"Chunk length {len(chunk)} exceeds limit"

        # Content should be mostly preserved (spaces may differ)
        rejoined = " ".join(chunks)
        # All words should be present
        for word in words:
            assert word in rejoined

    def test_split_response_long_line_after_normal_lines(self) -> None:
        """Test that accumulated normal lines are flushed before processing a long line.

        This tests lines 310-312: when current_chunk has accumulated normal lines,
        and then a long line is encountered, the current_chunk should be flushed
        before handling the long line.
        """
        from lattice.core.response_generator import split_response

        # Create message with normal lines followed by a long line
        lines = ["Normal line 1", "Normal line 2", "x" * 2000]
        text = "\n".join(lines)

        chunks = split_response(text, max_length=100)

        # Should create multiple chunks
        assert len(chunks) > 1

        # First chunk should contain both normal lines together
        assert "Normal line 1" in chunks[0]
        assert "Normal line 2" in chunks[0]

        # Remaining chunks should be the split long line
        # Each chunk should respect the limit
        for chunk in chunks:
            assert len(chunk) <= 100, f"Chunk length {len(chunk)} exceeds limit 100"

        # Verify long line was split and all content preserved
        rejoined_long_line = "".join(chunks[1:])
        assert rejoined_long_line == "x" * 2000

    def test_split_response_super_long_word_after_normal_words(self) -> None:
        """Test that temp_line is flushed before hard-splitting a super-long word.

        This tests lines 322-323: when temp_line has accumulated normal words,
        and then a super-long word (exceeding max_length) is encountered,
        temp_line should be flushed before hard-splitting the long word.
        """
        from lattice.core.response_generator import split_response

        # Create a single line with normal words followed by a super-long word
        words = ["hello", "world", "x" * 2000]  # Last word exceeds limit
        long_line = " ".join(words)

        chunks = split_response(long_line, max_length=100)

        # Should create multiple chunks
        assert len(chunks) > 1

        # First chunk should contain the normal words together
        assert chunks[0] == "hello world"

        # Remaining chunks should be the hard-split super-long word
        # Each should be exactly 100 chars (except possibly the last)
        for chunk in chunks[1:-1]:  # All but last
            assert len(chunk) == 100, f"Chunk length {len(chunk)} should be exactly 100"

        # Last chunk should be <= 100
        assert len(chunks[-1]) <= 100

        # Verify super-long word was split and all content preserved
        rejoined_word = "".join(chunks[1:])
        assert rejoined_word == "x" * 2000

    def test_split_response_custom_max_length(self) -> None:
        """Test split_response with custom max_length."""
        from lattice.core.response_generator import split_response

        lines = [f"Line {i}" for i in range(10)]
        message = "\n".join(lines)

        chunks = split_response(message, max_length=30)

        # Verify chunks respect custom limit
        for chunk in chunks:
            assert len(chunk) <= 30

        # Verify content is preserved
        assert all(f"Line {i}" in "\n".join(chunks) for i in range(10))

    def test_split_response_empty_message(self) -> None:
        """Test split_response with empty message."""
        from lattice.core.response_generator import split_response

        chunks = split_response("", max_length=1900)

        assert len(chunks) == 1
        assert chunks[0] == ""
