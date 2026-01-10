"""Unit tests for entity extraction module.

Renamed from test_query_extraction.py to match context_strategy.py module.
"""

import json
import uuid
from datetime import datetime, UTC
from unittest.mock import AsyncMock, patch

import pytest

from lattice.core.context_strategy import (
    build_smaller_episodic_context,
    context_strategy,
    get_context_strategy,
    get_message_strategy,
)
from lattice.memory.episodic import EpisodicMessage
from lattice.memory.procedural import PromptTemplate
from lattice.utils.llm import AuditResult


@pytest.fixture
def mock_prompt_template() -> PromptTemplate:
    """Create a mock CONTEXT_STRATEGY prompt template."""
    return PromptTemplate(
        prompt_key="CONTEXT_STRATEGY",
        template="Extract entities from: {message_content}\nContext: {context}",
        temperature=0.2,
        version=3,
        active=True,
    )


@pytest.fixture
def mock_llm_response() -> str:
    """Create a mock LLM response with valid extraction JSON."""
    return json.dumps(
        {
            "entities": ["lattice project", "Friday"],
            "context_flags": [],
            "unresolved_entities": [],
        }
    )


@pytest.fixture
def mock_generation_result(mock_llm_response: str) -> AuditResult:
    """Create a mock AuditResult."""
    return AuditResult(
        content=mock_llm_response,
        model="anthropic/claude-3.5-sonnet",
        provider="anthropic",
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        cost_usd=0.0005,
        latency_ms=300,
        temperature=0.2,
        audit_id=None,
        prompt_key=None,
    )


class TestContextStrategyFunction:
    """Tests for the context_strategy function."""

    @pytest.mark.asyncio
    async def test_context_strategy_success(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
    ) -> None:
        """Test successful context strategy."""
        message_id = uuid.uuid4()
        extraction_data = {
            "entities": ["lattice project", "Friday"],
            "context_flags": ["goal_context"],
            "unresolved_entities": [],
        }
        mock_generation_result.content = json.dumps(extraction_data)

        with (
            patch(
                "lattice.core.context_strategy.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.memory.canonical.get_canonical_entities_list",
                return_value=["Mother", "boyfriend"],
            ),
            patch(
                "lattice.core.context_strategy.get_auditing_llm_client",
            ) as mock_llm_client,
            patch("lattice.core.context_strategy.db_pool") as mock_db_pool,
            patch(
                "lattice.core.context_strategy.parse_llm_json_response",
                return_value=extraction_data,
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_generation_result)
            mock_llm_client.return_value = mock_client

            mock_conn = AsyncMock()
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            recent_messages = [
                EpisodicMessage(
                    content="Working on mobile app",
                    discord_message_id=1,
                    channel_id=123,
                    is_bot=False,
                ),
            ]

            strategy = await context_strategy(
                message_id=message_id,
                message_content="I need to finish the lattice project by Friday",
                recent_messages=recent_messages,
            )

            assert strategy.message_id == message_id
            assert strategy.entities == ["lattice project", "Friday"]
            assert strategy.context_flags == ["goal_context"]
            assert strategy.strategy_method == "api"

    @pytest.mark.asyncio
    async def test_context_strategy_missing_prompt_template(self) -> None:
        """Test context strategy with missing prompt template."""
        message_id = uuid.uuid4()

        with patch(
            "lattice.core.context_strategy.get_prompt",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="CONTEXT_STRATEGY prompt template"):
                await context_strategy(
                    message_id=message_id,
                    message_content="Test message",
                    recent_messages=[],
                )

    @pytest.mark.asyncio
    async def test_context_strategy_invalid_json(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
    ) -> None:
        """Test context strategy with invalid JSON response."""
        message_id = uuid.uuid4()

        from lattice.utils.json_parser import JSONParseError

        with (
            patch(
                "lattice.core.context_strategy.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.memory.canonical.get_canonical_entities_list",
                return_value=[],
            ),
            patch(
                "lattice.core.context_strategy.get_auditing_llm_client",
            ) as mock_llm_client,
            patch(
                "lattice.core.context_strategy.parse_llm_json_response",
            ) as mock_parse,
            patch(
                "lattice.core.context_strategy.notify_parse_error_to_dream",
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_generation_result)
            mock_llm_client.return_value = mock_client

            # Simulate parse error
            parse_error = JSONParseError(
                raw_content="not json",
                parse_error=json.JSONDecodeError("Expecting value", "", 0),
            )
            mock_parse.side_effect = parse_error

            with pytest.raises(JSONParseError):
                await context_strategy(
                    message_id=message_id,
                    message_content="Test message",
                    recent_messages=[],
                )

    @pytest.mark.asyncio
    async def test_context_strategy_missing_required_fields(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test context strategy with missing required fields."""
        message_id = uuid.uuid4()
        mock_result = AuditResult(
            content="{}",
            model="test",
            provider=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=None,
            latency_ms=0,
            temperature=0.0,
            audit_id=None,
            prompt_key="CONTEXT_STRATEGY",
        )

        with (
            patch(
                "lattice.core.context_strategy.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.memory.canonical.get_canonical_entities_list",
                return_value=[],
            ),
            patch(
                "lattice.core.context_strategy.get_auditing_llm_client",
            ) as mock_llm_client,
            patch(
                "lattice.core.context_strategy.parse_llm_json_response",
                return_value={
                    "entities": []
                },  # Missing context_flags and unresolved_entities
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_result)
            mock_llm_client.return_value = mock_client

            with pytest.raises(ValueError, match="Missing required field"):
                await context_strategy(
                    message_id=message_id,
                    message_content="Test message",
                    recent_messages=[],
                )


class TestGetContextStrategy:
    """Tests for the get_context_strategy function."""

    @pytest.mark.asyncio
    async def test_get_context_strategy_success(self) -> None:
        """Test retrieving a context strategy by ID."""
        strategy_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_row = {
            "id": strategy_id,
            "message_id": message_id,
            "strategy": {
                "entities": ["Alice", "yesterday"],
                "context_flags": ["goal_context"],
                "unresolved_entities": [],
            },
            "rendered_prompt": "test prompt",
            "raw_response": "test response",
            "created_at": datetime.now(UTC),
        }

        with patch("lattice.core.context_strategy.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=mock_row)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            strategy = await get_context_strategy(strategy_id)

            assert strategy is not None
            assert strategy.id == strategy_id
            assert strategy.entities == ["Alice", "yesterday"]
            assert strategy.context_flags == ["goal_context"]

    @pytest.mark.asyncio
    async def test_get_message_strategy_success(self) -> None:
        """Test retrieving a strategy by message ID."""
        strategy_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_row = {
            "id": strategy_id,
            "message_id": message_id,
            "strategy": {
                "entities": ["marathon"],
                "context_flags": [],
                "unresolved_entities": [],
            },
            "rendered_prompt": "test prompt",
            "raw_response": "test response",
            "created_at": datetime.now(UTC),
        }

        with patch("lattice.core.context_strategy.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=mock_row)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            strategy = await get_message_strategy(message_id)

            assert strategy is not None
            assert strategy.message_id == message_id
            assert strategy.entities == ["marathon"]


class TestBuildSmallerEpisodicContext:
    """Tests for the build_smaller_episodic_context function."""

    def test_build_smaller_episodic_context_basic(self) -> None:
        """Test building context with recent messages."""
        recent_messages = [
            EpisodicMessage(
                content="Working on mobile app",
                discord_message_id=1,
                channel_id=123,
                is_bot=False,
            ),
            EpisodicMessage(
                content="How's it coming?",
                discord_message_id=2,
                channel_id=123,
                is_bot=True,
            ),
        ]
        current_message = "Pretty good"

        context = build_smaller_episodic_context(
            recent_messages=recent_messages,
            current_message=current_message,
            window_size=3,
        )

        assert "USER: Working on mobile app" in context
        assert "ASSISTANT: How's it coming?" in context
        assert "USER: Pretty good" in context

    def test_build_smaller_episodic_context_empty_history(self) -> None:
        """Test building context with no recent messages."""
        current_message = "Hello there"

        context = build_smaller_episodic_context(
            recent_messages=[],
            current_message=current_message,
            window_size=5,
        )

        assert "USER: Hello there" in context
