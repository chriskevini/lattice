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
from lattice.memory.semantic import StableFact
from lattice.utils.llm import GenerationResult


@pytest.fixture
def mock_extraction_declaration() -> QueryExtraction:
    """Create a mock QueryExtraction for declaration message type."""
    return QueryExtraction(
        id=uuid.uuid4(),
        message_id=uuid.uuid4(),
        message_type="declaration",
        entities=["lattice project"],
        predicates=["need to finish"],
        time_constraint="2026-01-10T23:59:59Z",
        activity=None,
        query=None,
        urgency="high",
        continuation=False,
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
        message_type="query",
        entities=["lattice project"],
        predicates=["deadline"],
        time_constraint=None,
        activity=None,
        query="When is the lattice project deadline?",
        urgency=None,
        continuation=False,
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
        predicates=["spent time"],
        time_constraint=None,
        activity="coding",
        query=None,
        urgency=None,
        continuation=False,
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
        predicates=[],
        time_constraint=None,
        activity=None,
        query=None,
        urgency=None,
        continuation=True,
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
def mock_semantic_facts() -> list[StableFact]:
    """Create mock semantic facts for context."""
    return [
        StableFact(
            content="User is working on lattice project",
            fact_id=uuid.uuid4(),
        ),
    ]


@pytest.fixture
def mock_prompt_template() -> PromptTemplate:
    """Create a mock prompt template with extraction fields."""
    return PromptTemplate(
        prompt_key="GOAL_RESPONSE",
        template=(
            "Context: {episodic_context}\n"
            "Semantic: {semantic_context}\n"
            "User: {user_message}\n"
            "Entities: {entities}\n"
            "Time: {time_constraint}\n"
            "Urgency: {urgency}"
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

        # Check extraction placeholders are present
        assert "entities" in placeholders
        assert "query" in placeholders
        assert "activity" in placeholders

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
        """Test validation succeeds for extraction-specific placeholders."""
        template = (
            "Query: {query}\n"
            "Entities: {entities}\n"
            "Activity: {activity}\n"
            "Urgency: {urgency}"
        )
        is_valid, unknown = validate_template_placeholders(template)

        assert is_valid is True
        assert unknown == []

    def test_placeholder_registry_completeness(self) -> None:
        """Test that registry includes all placeholders used in generate_response."""
        # These are the placeholders that generate_response() actually populates
        expected_placeholders = {
            "episodic_context",
            "semantic_context",
            "user_message",
            "entities",
            "query",
            "activity",
            "time_constraint",
            "urgency",
            "continuation",
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
        """Test selection of QUERY_RESPONSE for query message type."""
        template = select_response_template(mock_extraction_query)
        assert template == "QUERY_RESPONSE"

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
            predicates=[],
            time_constraint=None,
            activity=None,
            query=None,
            urgency=None,
            continuation=False,
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
        mock_semantic_facts: list[StableFact],
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
                semantic_facts=mock_semantic_facts,
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_declaration,
            )

            # Verify template selection
            mock_get_prompt.assert_called_once_with("GOAL_RESPONSE")

            # Verify extraction fields in rendered prompt
            assert "lattice project" in rendered_prompt
            assert "2026-01-10T23:59:59Z" in rendered_prompt
            assert "high" in rendered_prompt

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
        mock_semantic_facts: list[StableFact],
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test response generation with query extraction."""
        mock_template = PromptTemplate(
            prompt_key="QUERY_RESPONSE",
            template=(
                "Context: {episodic_context}\n"
                "User: {user_message}\n"
                "Query: {query}\n"
                "Entities: {entities}"
            ),
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
                semantic_facts=mock_semantic_facts,
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_query,
            )

            # Verify template selection
            mock_get_prompt.assert_called_once_with("QUERY_RESPONSE")

            # Verify query field in rendered prompt
            assert "When is the lattice project deadline?" in rendered_prompt

            # Verify temperature from template is used
            mock_client.complete.assert_called_once()
            call_kwargs = mock_client.complete.call_args
            assert call_kwargs[1]["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_with_activity_extraction(
        self,
        mock_extraction_activity: QueryExtraction,
        mock_recent_messages: list[EpisodicMessage],
        mock_semantic_facts: list[StableFact],
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test response generation with activity update extraction."""
        mock_template = PromptTemplate(
            prompt_key="ACTIVITY_RESPONSE",
            template=(
                "Context: {episodic_context}\n"
                "User: {user_message}\n"
                "Activity: {activity}\n"
                "Entities: {entities}"
            ),
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
                semantic_facts=mock_semantic_facts,
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_activity,
            )

            # Verify template selection
            mock_get_prompt.assert_called_once_with("ACTIVITY_RESPONSE")

            # Verify activity field in rendered prompt
            assert "coding" in rendered_prompt

    @pytest.mark.asyncio
    async def test_generate_with_conversation_extraction(
        self,
        mock_extraction_conversation: QueryExtraction,
        mock_recent_messages: list[EpisodicMessage],
        mock_semantic_facts: list[StableFact],
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test response generation with conversation extraction."""
        mock_template = PromptTemplate(
            prompt_key="CONVERSATION_RESPONSE",
            template=(
                "Context: {episodic_context}\n"
                "User: {user_message}\n"
                "Continuation: {continuation}\n"
                "Entities: {entities}"
            ),
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
                semantic_facts=mock_semantic_facts,
                recent_messages=mock_recent_messages,
                extraction=mock_extraction_conversation,
            )

            # Verify template selection
            mock_get_prompt.assert_called_once_with("CONVERSATION_RESPONSE")

            # Verify continuation field in rendered prompt
            assert "yes" in rendered_prompt  # continuation=True -> "yes"

    @pytest.mark.asyncio
    async def test_generate_without_extraction_legacy(
        self,
        mock_recent_messages: list[EpisodicMessage],
        mock_semantic_facts: list[StableFact],
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
                semantic_facts=mock_semantic_facts,
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
        mock_semantic_facts: list[StableFact],
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
                semantic_facts=mock_semantic_facts,
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
        mock_semantic_facts: list[StableFact],
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
                semantic_facts=mock_semantic_facts,
                recent_messages=mock_recent_messages,
                graph_triples=graph_triples,
                extraction=mock_extraction_query,
            )

            # Verify graph triples are formatted in semantic context
            assert "lattice project has deadline Friday" in rendered_prompt
            assert "user working on lattice project" in rendered_prompt

            # Verify context info includes graph count
            assert context_info["graph"] == 2

    @pytest.mark.asyncio
    async def test_generate_empty_extraction_fields(
        self,
        mock_recent_messages: list[EpisodicMessage],
        mock_semantic_facts: list[StableFact],
    ) -> None:
        """Test handling of extraction with empty/None fields."""
        extraction = QueryExtraction(
            id=uuid.uuid4(),
            message_id=uuid.uuid4(),
            message_type="conversation",
            entities=[],  # Empty list
            predicates=[],
            time_constraint=None,
            activity=None,
            query=None,
            urgency=None,
            continuation=False,
            rendered_prompt="test",
            raw_response="test",
            extraction_method="api",
            created_at=datetime.now(UTC),
        )

        mock_template = PromptTemplate(
            prompt_key="CONVERSATION_RESPONSE",
            template=(
                "Context: {episodic_context}\n"
                "Entities: {entities}\n"
                "Activity: {activity}\n"
                "Query: {query}"
            ),
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
                semantic_facts=mock_semantic_facts,
                recent_messages=mock_recent_messages,
                extraction=extraction,
            )

            # Verify empty fields are handled gracefully
            assert "Entities: None" in rendered_prompt
            assert "Activity: N/A" in rendered_prompt
            assert "Query: N/A" in rendered_prompt
