"""Unit tests for response template selection and generation."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from lattice.core.query_extraction import QueryExtraction
from lattice.core.response_generator import (
    generate_response,
    get_available_placeholders,
    select_response_template,
    validate_template_placeholders,
)
from lattice.memory.episodic import EpisodicMessage
from lattice.memory.procedural import PromptTemplate
from lattice.utils.llm import GenerationResult


@pytest.fixture
def mock_extraction_declaration() -> QueryExtraction:
    """Create a mock QueryExtraction for declaration message type."""
    return QueryExtraction(
        id=uuid.uuid4(),
        message_id=uuid.uuid4(),
        message_type="goal",
        entities=["lattice project", "Friday"],
        rendered_prompt="test prompt",
        raw_response="test response",
        extraction_method="api",
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_extraction_query() -> QueryExtraction:
    """Create a mock QueryExtraction for query message type."""
    return QueryExtraction(
        id=uuid.uuid4(),
        message_id=uuid.uuid4(),
        message_type="question",
        entities=["lattice project", "deadline"],
        rendered_prompt="test prompt",
        raw_response="test response",
        extraction_method="api",
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_extraction_activity() -> QueryExtraction:
    """Create a mock QueryExtraction for activity_update message type."""
    return QueryExtraction(
        id=uuid.uuid4(),
        message_id=uuid.uuid4(),
        message_type="activity_update",
        entities=["coding"],
        rendered_prompt="test prompt",
        raw_response="test response",
        extraction_method="api",
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_extraction_conversation() -> QueryExtraction:
    """Create a mock QueryExtraction for conversation message type."""
    return QueryExtraction(
        id=uuid.uuid4(),
        message_id=uuid.uuid4(),
        message_type="conversation",
        entities=["tea"],
        rendered_prompt="test prompt",
        raw_response="test response",
        extraction_method="api",
        created_at=datetime.now(UTC),
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
            timestamp=datetime(2026, 1, 4, 10, 0, tzinfo=UTC),
        ),
        EpisodicMessage(
            content="Got it! Friday deadline for lattice.",
            discord_message_id=1234567891,
            channel_id=9876543210,
            is_bot=True,
            message_id=uuid.uuid4(),
            timestamp=datetime(2026, 1, 4, 10, 1, tzinfo=UTC),
        ),
    ]


@pytest.fixture
def mock_prompt_template() -> PromptTemplate:
    """Create a mock prompt template."""
    return PromptTemplate(
        prompt_key="GOAL_RESPONSE",
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
def mock_basic_template() -> PromptTemplate:
    """Create a mock BASIC_RESPONSE template without extraction fields."""
    return PromptTemplate(
        prompt_key="BASIC_RESPONSE",
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
def mock_generation_result() -> GenerationResult:
    """Create a mock GenerationResult."""
    return GenerationResult(
        content="Got it! Friday deadline for lattice. That's coming up quick—how's it looking so far?",
        model="anthropic/claude-3.5-sonnet",
        provider="anthropic",
        prompt_tokens=150,
        completion_tokens=30,
        total_tokens=180,
        cost_usd=0.002,
        latency_ms=600,
        temperature=0.7,
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


class TestSelectResponseTemplate:
    """Tests for select_response_template function."""

    def test_select_goal_response(
        self, mock_extraction_declaration: QueryExtraction
    ) -> None:
        """Test selection of GOAL_RESPONSE for declaration message type."""
        template = select_response_template(mock_extraction_declaration)
        assert template == "GOAL_RESPONSE"

    def test_select_query_response(
        self, mock_extraction_query: QueryExtraction
    ) -> None:
        """Test selection of QUESTION_RESPONSE for question message type."""
        template = select_response_template(mock_extraction_query)
        assert template == "QUESTION_RESPONSE"

    def test_select_activity_response(
        self, mock_extraction_activity: QueryExtraction
    ) -> None:
        """Test selection of ACTIVITY_RESPONSE for activity_update message type."""
        template = select_response_template(mock_extraction_activity)
        assert template == "ACTIVITY_RESPONSE"

    def test_select_conversation_response(
        self, mock_extraction_conversation: QueryExtraction
    ) -> None:
        """Test selection of CONVERSATION_RESPONSE for conversation message type."""
        template = select_response_template(mock_extraction_conversation)
        assert template == "CONVERSATION_RESPONSE"

    def test_select_basic_response_none_extraction(self) -> None:
        """Test fallback to BASIC_RESPONSE when extraction is None."""
        template = select_response_template(None)
        assert template == "BASIC_RESPONSE"

    def test_select_default_for_unknown_type(self) -> None:
        """Test default to BASIC_RESPONSE for unknown message types."""
        extraction = QueryExtraction(
            id=uuid.uuid4(),
            message_id=uuid.uuid4(),
            message_type="unknown_type",  # Invalid type
            entities=[],
            rendered_prompt="test",
            raw_response="test",
            extraction_method="api",
            created_at=datetime.now(UTC),
        )
        template = select_response_template(extraction)
        assert template == "BASIC_RESPONSE"


class TestGenerateResponseWithTemplates:
    """Tests for generate_response with template selection."""

    @pytest.mark.asyncio
    async def test_generate_with_declaration_extraction(
        self,
        mock_extraction_declaration: QueryExtraction,
        mock_recent_messages: list[EpisodicMessage],
        mock_prompt_template: PromptTemplate,
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test response generation with declaration extraction."""
        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="I need to finish the lattice project by Friday",
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_declaration,
            )

            # Verify template selection
            mock_get_prompt.assert_called_once_with("GOAL_RESPONSE")

            # Verify extraction fields in rendered prompt
            assert "I need to finish the lattice project by Friday" in rendered_prompt

            # Verify context info includes extraction
            assert context_info["template"] == "GOAL_RESPONSE"
            assert context_info["extraction_id"] == str(mock_extraction_declaration.id)

            # Verify result
            assert result == mock_generation_result

    @pytest.mark.asyncio
    async def test_generate_with_query_extraction(
        self,
        mock_extraction_query: QueryExtraction,
        mock_recent_messages: list[EpisodicMessage],
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test response generation with query extraction."""
        mock_template = PromptTemplate(
            prompt_key="QUESTION_RESPONSE",
            template=("Context: {episodic_context}\nUser: {user_message}"),
            temperature=0.5,
            version=1,
            active=True,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="When is the lattice deadline?",
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_query,
            )

            # Verify template selection
            mock_get_prompt.assert_called_once_with("QUESTION_RESPONSE")

            # User message should be in prompt
            assert "When is the lattice deadline?" in rendered_prompt

            # Verify temperature from template is used
            mock_client.complete.assert_called_once()
            call_kwargs = mock_client.complete.call_args
            assert call_kwargs[1]["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_with_activity_extraction(
        self,
        mock_extraction_activity: QueryExtraction,
        mock_recent_messages: list[EpisodicMessage],
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test response generation with activity update extraction."""
        mock_template = PromptTemplate(
            prompt_key="ACTIVITY_RESPONSE",
            template=("Context: {episodic_context}\nUser: {user_message}"),
            temperature=0.7,
            version=1,
            active=True,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Spent 3 hours coding today",
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_activity,
            )

            # Verify template selection
            mock_get_prompt.assert_called_once_with("ACTIVITY_RESPONSE")

            # User message should be in prompt
            assert "Spent 3 hours coding today" in rendered_prompt

    @pytest.mark.asyncio
    async def test_generate_with_conversation_extraction(
        self,
        mock_extraction_conversation: QueryExtraction,
        mock_recent_messages: list[EpisodicMessage],
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test response generation with conversation extraction."""
        mock_template = PromptTemplate(
            prompt_key="CONVERSATION_RESPONSE",
            template=("Context: {episodic_context}\nUser: {user_message}"),
            temperature=0.7,
            version=1,
            active=True,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Just made some tea",
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_conversation,
            )

            # Verify template selection
            mock_get_prompt.assert_called_once_with("CONVERSATION_RESPONSE")

            # User message should be in prompt
            assert "Just made some tea" in rendered_prompt

    @pytest.mark.asyncio
    async def test_generate_without_extraction_legacy(
        self,
        mock_recent_messages: list[EpisodicMessage],
        mock_basic_template: PromptTemplate,
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test backward compatibility when extraction is None."""
        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_basic_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Hello",
                recent_messages=mock_recent_messages,
                extraction=None,
            )

            # Verify BASIC_RESPONSE is used for backward compatibility
            mock_get_prompt.assert_called_once_with("BASIC_RESPONSE")

            # Verify context info shows no extraction
            assert context_info["extraction_id"] is None

    @pytest.mark.asyncio
    async def test_generate_template_fallback(
        self,
        mock_extraction_declaration: QueryExtraction,
        mock_recent_messages: list[EpisodicMessage],
        mock_basic_template: PromptTemplate,
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test fallback to BASIC_RESPONSE when selected template doesn't exist."""
        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            # First call returns None (template not found), second call returns basic template
            mock_get_prompt.side_effect = [None, mock_basic_template]
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Test message",
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_declaration,
            )

            # Verify both templates were tried
            assert mock_get_prompt.call_count == 2
            mock_get_prompt.assert_any_call("GOAL_RESPONSE")
            mock_get_prompt.assert_any_call("BASIC_RESPONSE")

    @pytest.mark.asyncio
    async def test_generate_with_graph_triples(
        self,
        mock_extraction_query: QueryExtraction,
        mock_recent_messages: list[EpisodicMessage],
        mock_prompt_template: PromptTemplate,
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test response generation includes graph triples in context."""
        graph_triples = [
            {
                "subject_content": "lattice project",
                "predicate": "has deadline",
                "object_content": "Friday",
            },
            {
                "subject_content": "user",
                "predicate": "working on",
                "object_content": "lattice project",
            },
        ]

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_generation_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="When is the deadline?",
                recent_messages=mock_recent_messages,
                graph_triples=graph_triples,
                extraction=mock_extraction_query,
            )

            # Verify graph triples are formatted in semantic context
            assert "lattice project has deadline Friday" in rendered_prompt
            assert "user working on lattice project" in rendered_prompt

            # Verify context info includes template
            assert context_info["template"] == "QUESTION_RESPONSE"

    @pytest.mark.asyncio
    async def test_generate_empty_extraction_fields(
        self,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test handling of extraction with empty/None fields."""
        extraction = QueryExtraction(
            id=uuid.uuid4(),
            message_id=uuid.uuid4(),
            message_type="conversation",
            entities=[],  # Empty list
            rendered_prompt="test",
            raw_response="test",
            extraction_method="api",
            created_at=datetime.now(UTC),
        )

        mock_template = PromptTemplate(
            prompt_key="CONVERSATION_RESPONSE",
            template=("Context: {episodic_context}\nUser: {user_message}"),
            temperature=0.7,
            version=1,
            active=True,
        )

        mock_result = GenerationResult(
            content="Response",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Hello",
                recent_messages=mock_recent_messages,
                extraction=extraction,
            )

            # Verify empty fields are handled gracefully (no extraction fields in Design D)
            assert "Hello" in rendered_prompt

    @pytest.mark.asyncio
    async def test_filter_current_message_from_episodic_context(
        self,
        mock_basic_template: PromptTemplate,
    ) -> None:
        """Test that current user message is filtered from episodic context."""
        user_message_content = "What's the weather?"
        user_discord_id = 9999999999

        # Create conversation including the current message
        recent_messages = [
            EpisodicMessage(
                content="Hello!",
                discord_message_id=1111111111,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 0, tzinfo=UTC),
            ),
            EpisodicMessage(
                content="Hi there!",
                discord_message_id=2222222222,
                channel_id=123,
                is_bot=True,
                timestamp=datetime(2026, 1, 5, 10, 1, tzinfo=UTC),
            ),
            EpisodicMessage(
                content=user_message_content,  # Current message (should be filtered)
                discord_message_id=user_discord_id,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 2, tzinfo=UTC),
            ),
        ]

        mock_result = GenerationResult(
            content="It's sunny!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_basic_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message=user_message_content,
                recent_messages=recent_messages,
                user_discord_message_id=user_discord_id,
            )

            # Verify current message is NOT in episodic context
            # Count occurrences of the message content in rendered prompt
            message_occurrences = rendered_prompt.count(user_message_content)

            # Should appear exactly once (in {user_message} placeholder)
            # NOT in episodic_context
            assert message_occurrences == 1, (
                f"Expected user message to appear exactly once, "
                f"but found {message_occurrences} occurrences"
            )

            # Verify previous messages ARE in episodic context
            assert "Hello!" in rendered_prompt
            assert "Hi there!" in rendered_prompt

    @pytest.mark.asyncio
    async def test_handle_duplicate_message_content(
        self,
        mock_basic_template: PromptTemplate,
    ) -> None:
        """Test that ID-based filtering handles duplicate message content correctly."""
        duplicate_content = "What's the weather?"

        # User sends same message twice
        recent_messages = [
            EpisodicMessage(
                content="Hello!",
                discord_message_id=1111111111,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 0, tzinfo=UTC),
            ),
            EpisodicMessage(
                content=duplicate_content,  # First time asking
                discord_message_id=2222222222,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 1, tzinfo=UTC),
            ),
            EpisodicMessage(
                content="It's sunny!",
                discord_message_id=3333333333,
                channel_id=123,
                is_bot=True,
                timestamp=datetime(2026, 1, 5, 10, 2, tzinfo=UTC),
            ),
            EpisodicMessage(
                content=duplicate_content,  # Asking again (current message)
                discord_message_id=4444444444,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 3, tzinfo=UTC),
            ),
        ]

        mock_result = GenerationResult(
            content="Still sunny!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_basic_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message=duplicate_content,
                recent_messages=recent_messages,
                user_discord_message_id=4444444444,  # Current message ID
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
        mock_basic_template: PromptTemplate,
    ) -> None:
        """Test backward compatibility: content-based filtering when ID not provided."""
        user_message_content = "What's the weather?"

        recent_messages = [
            EpisodicMessage(
                content="Hello!",
                discord_message_id=1111111111,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 0, tzinfo=UTC),
            ),
            EpisodicMessage(
                content=user_message_content,  # Should be filtered by content
                discord_message_id=2222222222,
                channel_id=123,
                is_bot=False,
                timestamp=datetime(2026, 1, 5, 10, 1, tzinfo=UTC),
            ),
        ]

        mock_result = GenerationResult(
            content="It's sunny!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_basic_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            # Don't pass user_discord_message_id (backward compatibility test)
            result, rendered_prompt, context_info = await generate_response(
                user_message=user_message_content,
                recent_messages=recent_messages,
                # user_discord_message_id=None (implicit)
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
        mock_extraction_declaration: QueryExtraction,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test GOAL_RESPONSE template does not include extraction fields in prompt."""
        # Use the actual template from migration 018 (no extraction section)
        mock_template = PromptTemplate(
            prompt_key="GOAL_RESPONSE",
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

        mock_result = GenerationResult(
            content="Got it!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="I need to finish the lattice project by Friday",
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_declaration,
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
        mock_extraction_query: QueryExtraction,
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

        mock_result = GenerationResult(
            content="The deadline is Friday.",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.5,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="When is the lattice deadline?",
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_query,
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
        mock_extraction_activity: QueryExtraction,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test ACTIVITY_RESPONSE template does not include extraction fields."""
        mock_template = PromptTemplate(
            prompt_key="ACTIVITY_RESPONSE",
            template=(
                "You are a friendly AI companion.\n\n"
                "## Context\n"
                "**Recent conversation history:**\n{episodic_context}\n\n"
                "**Relevant facts from past conversations:**\n{semantic_context}\n\n"
                "**User message:** {user_message}\n\n"
                "## Your Task\n"
                "Acknowledge the activity update naturally."
            ),
            temperature=0.7,
            version=2,
            active=True,
        )

        mock_result = GenerationResult(
            content="Nice session!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Spent 3 hours coding today",
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_activity,
            )

            # Verify extraction fields are NOT in the rendered prompt
            assert "Extracted information:" not in rendered_prompt
            assert "Activity:" not in rendered_prompt
            assert "Entities mentioned:" not in rendered_prompt

            # Verify core fields ARE present
            assert "Spent 3 hours coding today" in rendered_prompt

    @pytest.mark.asyncio
    async def test_conversation_response_no_extraction_fields(
        self,
        mock_extraction_conversation: QueryExtraction,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test CONVERSATION_RESPONSE template does not include extraction fields."""
        mock_template = PromptTemplate(
            prompt_key="CONVERSATION_RESPONSE",
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

        mock_result = GenerationResult(
            content="Nice! What kind?",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Just made some tea",
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_conversation,
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
        mock_extraction_declaration: QueryExtraction,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test that extraction data is still available for routing/analytics.

        Even though extraction fields aren't shown to the LLM, they should still
        be populated internally for template selection and analytics purposes.
        """
        mock_template = PromptTemplate(
            prompt_key="GOAL_RESPONSE",
            template="User: {user_message}",  # Minimal template, no extraction fields
            temperature=0.7,
            version=2,
            active=True,
        )

        mock_result = GenerationResult(
            content="Got it!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="I need to finish the lattice project by Friday",
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_declaration,
            )

            # Verify extraction was used for template selection
            mock_get_prompt.assert_called_once_with("GOAL_RESPONSE")

            # Verify extraction metadata is in context_info for analytics
            assert context_info["template"] == "GOAL_RESPONSE"
            assert context_info["extraction_id"] == str(mock_extraction_declaration.id)


class TestNoTemplateFound:
    """Tests for error handling when no templates are available."""

    @pytest.mark.asyncio
    async def test_generate_response_no_template_found(
        self,
        mock_recent_messages: list[EpisodicMessage],
    ) -> None:
        """Test fallback response when no templates are available in database."""
        extraction = QueryExtraction(
            id=uuid.uuid4(),
            message_id=uuid.uuid4(),
            message_type="goal",
            entities=["test"],
            rendered_prompt="test",
            raw_response="test",
            extraction_method="api",
            created_at=datetime.now(UTC),
        )

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            # Simulate both template lookups returning None
            mock_get_prompt.return_value = None

            result, rendered_prompt, context_info = await generate_response(
                user_message="Test message",
                recent_messages=mock_recent_messages,
                extraction=extraction,
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

            # Verify both templates were tried
            assert mock_get_prompt.call_count == 2
            mock_get_prompt.assert_any_call("GOAL_RESPONSE")
            mock_get_prompt.assert_any_call("BASIC_RESPONSE")


class TestTimezoneConversionFailure:
    """Tests for timezone conversion error handling."""

    @pytest.mark.asyncio
    async def test_invalid_timezone_falls_back_to_utc(
        self,
        mock_basic_template: PromptTemplate,
    ) -> None:
        """Test that invalid timezone falls back to UTC formatting."""
        # Create message with invalid timezone
        recent_messages = [
            EpisodicMessage(
                content="Hello!",
                discord_message_id=1234567890,
                channel_id=123,
                is_bot=False,
                user_timezone="Invalid/Timezone",  # Invalid timezone
                timestamp=datetime(2026, 1, 4, 10, 0, tzinfo=UTC),
            ),
        ]

        mock_result = GenerationResult(
            content="Hi!",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        with (
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_get_prompt,
            patch("lattice.core.response_generator.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = mock_basic_template
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_result
            mock_get_client.return_value = mock_client

            result, rendered_prompt, context_info = await generate_response(
                user_message="Test",
                recent_messages=recent_messages,
            )

            # Verify fallback to UTC formatting
            assert "2026-01-04 10:00 UTC" in rendered_prompt
            assert "Hello!" in rendered_prompt


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
        assert len(chunks) > 1

        # Verify each chunk is within limit
        for chunk in chunks:
            assert len(chunk) <= 500

        # Verify chunks can be rejoined (with single newlines)
        rejoined = "\n".join(chunks)
        # Account for potential extra newlines between chunks
        assert rejoined.replace("\n\n", "\n") == long_message

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

        # Single line longer than limit
        long_line = "x" * 2000

        chunks = split_response(long_line, max_length=1900)

        # Should create a single chunk with the long line
        # (function doesn't split within lines, only at newlines)
        assert len(chunks) == 1
        assert chunks[0] == long_line

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
