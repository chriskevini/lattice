"""Unit tests for response template selection and generation."""

import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any
from lattice.utils.date_resolution import get_now
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lattice.core.context import ContextStrategy
from lattice.core.response_generator import (
    generate_response,
    get_available_placeholders,
    validate_template_placeholders,
)
from lattice.memory.episodic import EpisodicMessage
from lattice.memory.procedural import PromptTemplate
from lattice.memory.repositories import (
    PromptRegistryRepository,
)
from lattice.utils.llm import AuditResult


@pytest.fixture
def mock_extraction_declaration() -> ContextStrategy:
    """Create a mock ContextStrategy for goal message type."""
    return ContextStrategy(
        entities=["lattice project", "Friday"],
        context_flags=["goal_context"],
        unresolved_entities=[],
        created_at=get_now(timezone_str="UTC"),
    )


@pytest.fixture
def mock_extraction_query() -> ContextStrategy:
    """Create a mock ContextStrategy for question message type."""
    return ContextStrategy(
        entities=["lattice", "deadline"],
        context_flags=[],
        unresolved_entities=[],
        created_at=get_now(timezone_str="UTC"),
    )


@pytest.fixture
def mock_extraction_activity() -> ContextStrategy:
    """Create a mock ContextStrategy for activity_update message type."""
    return ContextStrategy(
        entities=["coding"],
        context_flags=["activity_context"],
        unresolved_entities=[],
        created_at=get_now(timezone_str="UTC"),
    )


@pytest.fixture
def mock_extraction_conversation() -> ContextStrategy:
    """Create a mock ContextStrategy for conversation message type."""
    return ContextStrategy(
        entities=["tea"],
        context_flags=[],
        unresolved_entities=[],
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


@pytest.fixture
def mock_pool() -> Any:
    """Create a mock database pool for dependency injection."""
    from unittest.mock import MagicMock

    pool = MagicMock()
    pool.pool = pool
    return pool


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
        mock_pool: Any,
    ) -> None:
        """Test that UNIFIED_RESPONSE is selected for goal-type messages."""
        mock_template = PromptTemplate(
            prompt_key="UNIFIED_RESPONSE",
            template="Context: {episodic_context}\nUser: {user_message}",
            temperature=0.7,
            version=1,
            active=True,
        )

        mock_client = AsyncMock()
        mock_client.complete.return_value = mock_generation_result

        with patch(
            "lattice.core.response_generator.procedural.get_prompt",
            return_value=mock_template,
        ):
            await generate_response(
                user_message="I need to finish the lattice project by Friday",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
                llm_client=mock_client,
                prompt_repo=MagicMock(spec=PromptRegistryRepository),
            )

            mock_client.complete.assert_called_once()


class TestGenerateResponseWithTemplates:
    """Tests for generate_response with template selection."""

    @pytest.mark.asyncio
    async def test_generate_with_goal_extraction(
        self,
        mock_extraction_declaration: ContextStrategy,
        mock_recent_messages: list[EpisodicMessage],
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
        mock_pool: Any,
    ) -> None:
        """Test response generation with goal extraction."""
        mock_client = AsyncMock()
        mock_client.complete.return_value = mock_generation_result

        with patch(
            "lattice.core.response_generator.procedural.get_prompt",
            return_value=mock_prompt_template,
        ):
            result, rendered_prompt, context_info = await generate_response(
                user_message="I need to finish the lattice project by Friday",
                episodic_context="Recent conversation history",
                semantic_context="I need to finish the lattice project by Friday",
                llm_client=mock_client,
                prompt_repo=MagicMock(spec=PromptRegistryRepository),
            )

            # Verify template selection
            mock_client.complete.assert_called_once()

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
        mock_pool: Any,
    ) -> None:
        """Test response generation with entity extraction."""
        mock_template = PromptTemplate(
            prompt_key="UNIFIED_RESPONSE",
            template=("Context: {episodic_context}\nUser: {user_message}"),
            temperature=0.5,
            version=1,
            active=True,
        )

        mock_client = AsyncMock()
        mock_client.complete.return_value = mock_generation_result

        with patch(
            "lattice.core.response_generator.procedural.get_prompt",
            return_value=mock_template,
        ):
            result, rendered_prompt, context_info = await generate_response(
                user_message="What's the deadline for the lattice project?",
                episodic_context="Recent conversation history",
                semantic_context="Relevant facts",
                llm_client=mock_client,
                prompt_repo=MagicMock(spec=PromptRegistryRepository),
            )

            # Verify extraction fields in rendered prompt
            assert "What's the deadline for the lattice project?" in rendered_prompt

            # Verify context info
            assert context_info["template"] == "UNIFIED_RESPONSE"
            assert result == mock_generation_result
