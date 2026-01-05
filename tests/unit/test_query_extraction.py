"""Unit tests for query extraction module."""

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.core.query_extraction import (
    QueryExtraction,
    extract_query_structure,
    get_extraction,
    get_message_extraction,
)
from lattice.memory.procedural import PromptTemplate
from lattice.utils.llm import GenerationResult


@pytest.fixture
def mock_prompt_template() -> PromptTemplate:
    """Create a mock QUERY_EXTRACTION prompt template."""
    return PromptTemplate(
        prompt_key="QUERY_EXTRACTION",
        template="Extract structured data from: {message_content}\nContext: {context}",
        temperature=0.2,
        version=1,
        active=True,
    )


@pytest.fixture
def mock_llm_response() -> str:
    """Create a mock LLM response with valid extraction JSON."""
    return json.dumps(
        {
            "message_type": "declaration",
            "entities": ["lattice project"],
            "predicates": ["need to finish"],
            "time_constraint": "2026-01-10T23:59:59Z",
            "activity": None,
            "query": None,
            "urgency": "high",
            "continuation": False,
        }
    )


@pytest.fixture
def mock_generation_result(mock_llm_response: str) -> GenerationResult:
    """Create a mock GenerationResult."""
    return GenerationResult(
        content=mock_llm_response,
        model="anthropic/claude-3.5-sonnet",
        provider="anthropic",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost_usd=0.001,
        latency_ms=500,
        temperature=0.2,
    )


class TestQueryExtraction:
    """Tests for the QueryExtraction dataclass."""

    def test_query_extraction_init(self) -> None:
        """Test QueryExtraction initialization."""
        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()
        now = datetime.now()

        extraction = QueryExtraction(
            id=extraction_id,
            message_id=message_id,
            message_type="declaration",
            entities=["lattice"],
            predicates=["working on"],
            time_constraint="2026-01-10T23:59:59Z",
            activity=None,
            query=None,
            urgency="high",
            continuation=False,
            rendered_prompt="test prompt",
            raw_response="test response",
            created_at=now,
        )

        assert extraction.id == extraction_id
        assert extraction.message_id == message_id
        assert extraction.message_type == "declaration"
        assert extraction.entities == ["lattice"]
        assert extraction.predicates == ["working on"]
        assert extraction.time_constraint == "2026-01-10T23:59:59Z"
        assert extraction.activity is None
        assert extraction.query is None
        assert extraction.urgency == "high"
        assert extraction.continuation is False
        assert extraction.rendered_prompt == "test prompt"
        assert extraction.raw_response == "test response"
        assert extraction.created_at == now


class TestExtractQueryStructure:
    """Tests for extract_query_structure function."""

    @pytest.mark.asyncio
    async def test_extract_query_structure_success(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test successful query extraction."""
        message_id = uuid.uuid4()
        message_content = "I need to finish the lattice project by Friday"
        context = "User has been working on lattice"

        # Mock database pool
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        # Mock dependencies
        with (
            patch("lattice.core.query_extraction.get_prompt") as mock_get_prompt,
            patch("lattice.core.query_extraction.get_llm_client") as mock_get_llm,
            patch("lattice.core.query_extraction.db_pool") as mock_db_pool,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_llm_client = AsyncMock()
            mock_llm_client.complete = AsyncMock(return_value=mock_generation_result)
            mock_get_llm.return_value = mock_llm_client
            mock_db_pool.pool = mock_pool

            # Execute
            result = await extract_query_structure(
                message_id=message_id,
                message_content=message_content,
                context=context,
            )

            # Assertions
            assert result.message_id == message_id
            assert result.message_type == "declaration"
            assert result.entities == ["lattice project"]
            assert result.predicates == ["need to finish"]
            assert result.time_constraint == "2026-01-10T23:59:59Z"
            assert result.activity is None
            assert result.query is None
            assert result.urgency == "high"
            assert result.continuation is False

            # Verify database insert was called
            mock_conn.execute.assert_called_once()
            call_args = mock_conn.execute.call_args
            assert call_args[0][0].strip().startswith("INSERT INTO message_extractions")

            # Verify LLM was called with rendered prompt
            mock_llm_client.complete.assert_called_once()
            assert (
                "I need to finish the lattice project by Friday"
                in result.rendered_prompt
            )
            assert "User has been working on lattice" in result.rendered_prompt

    @pytest.mark.asyncio
    async def test_extract_query_structure_handles_markdown_json(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test extraction handles JSON wrapped in markdown code blocks."""
        message_id = uuid.uuid4()
        message_content = "What did I work on yesterday?"

        # Create response with markdown wrapper
        json_response = json.dumps(
            {
                "message_type": "query",
                "entities": [],
                "predicates": ["work on"],
                "time_constraint": None,
                "activity": None,
                "query": "What activities did the user work on yesterday?",
                "urgency": None,
                "continuation": False,
            }
        )
        markdown_response = f"```json\n{json_response}\n```"

        mock_generation_result = GenerationResult(
            content=markdown_response,
            model="test-model",
            provider=None,
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
            cost_usd=None,
            latency_ms=300,
            temperature=0.2,
        )

        # Mock database pool
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with (
            patch("lattice.core.query_extraction.get_prompt") as mock_get_prompt,
            patch("lattice.core.query_extraction.get_llm_client") as mock_get_llm,
            patch("lattice.core.query_extraction.db_pool") as mock_db_pool,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_llm_client = AsyncMock()
            mock_llm_client.complete = AsyncMock(return_value=mock_generation_result)
            mock_get_llm.return_value = mock_llm_client
            mock_db_pool.pool = mock_pool

            # Execute
            result = await extract_query_structure(
                message_id=message_id,
                message_content=message_content,
            )

            # Should successfully parse JSON despite markdown wrapper
            assert result.message_type == "query"
            assert result.query == "What activities did the user work on yesterday?"
            assert result.continuation is False

    @pytest.mark.asyncio
    async def test_extract_query_structure_missing_prompt_template(self) -> None:
        """Test extraction fails gracefully when prompt template not found."""
        message_id = uuid.uuid4()

        with patch("lattice.core.query_extraction.get_prompt") as mock_get_prompt:
            mock_get_prompt.return_value = None

            with pytest.raises(
                ValueError, match="QUERY_EXTRACTION prompt template not found"
            ):
                await extract_query_structure(
                    message_id=message_id,
                    message_content="test message",
                )

    @pytest.mark.asyncio
    async def test_extract_query_structure_invalid_json(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test extraction fails gracefully with invalid JSON response."""
        message_id = uuid.uuid4()

        # Create invalid JSON response
        mock_generation_result = GenerationResult(
            content="This is not valid JSON",
            model="test-model",
            provider=None,
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            cost_usd=None,
            latency_ms=200,
            temperature=0.2,
        )

        with (
            patch("lattice.core.query_extraction.get_prompt") as mock_get_prompt,
            patch("lattice.core.query_extraction.get_llm_client") as mock_get_llm,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_llm_client = AsyncMock()
            mock_llm_client.complete = AsyncMock(return_value=mock_generation_result)
            mock_get_llm.return_value = mock_llm_client

            with pytest.raises(json.JSONDecodeError):
                await extract_query_structure(
                    message_id=message_id,
                    message_content="test message",
                )

    @pytest.mark.asyncio
    async def test_extract_query_structure_missing_required_fields(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test extraction fails when required fields are missing."""
        message_id = uuid.uuid4()

        # Create JSON response missing required fields
        incomplete_json = json.dumps(
            {
                "message_type": "query",
                "entities": [],
                # Missing predicates, continuation
            }
        )

        mock_generation_result = GenerationResult(
            content=incomplete_json,
            model="test-model",
            provider=None,
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            cost_usd=None,
            latency_ms=200,
            temperature=0.2,
        )

        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with (
            patch("lattice.core.query_extraction.get_prompt") as mock_get_prompt,
            patch("lattice.core.query_extraction.get_llm_client") as mock_get_llm,
            patch("lattice.core.query_extraction.db_pool") as mock_db_pool,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_llm_client = AsyncMock()
            mock_llm_client.complete = AsyncMock(return_value=mock_generation_result)
            mock_get_llm.return_value = mock_llm_client
            mock_db_pool.pool = mock_pool

            with pytest.raises(
                ValueError, match="Missing required field in extraction"
            ):
                await extract_query_structure(
                    message_id=message_id,
                    message_content="test message",
                )

    @pytest.mark.asyncio
    async def test_extract_query_structure_invalid_message_type(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test extraction fails when message_type is invalid."""
        message_id = uuid.uuid4()

        # Create JSON response with invalid message_type
        invalid_json = json.dumps(
            {
                "message_type": "unknown",  # Invalid type
                "entities": [],
                "predicates": [],
                "continuation": False,
            }
        )

        mock_generation_result = GenerationResult(
            content=invalid_json,
            model="test-model",
            provider=None,
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            cost_usd=None,
            latency_ms=200,
            temperature=0.2,
        )

        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with (
            patch("lattice.core.query_extraction.get_prompt") as mock_get_prompt,
            patch("lattice.core.query_extraction.get_llm_client") as mock_get_llm,
            patch("lattice.core.query_extraction.db_pool") as mock_db_pool,
        ):
            mock_get_prompt.return_value = mock_prompt_template
            mock_llm_client = AsyncMock()
            mock_llm_client.complete = AsyncMock(return_value=mock_generation_result)
            mock_get_llm.return_value = mock_llm_client
            mock_db_pool.pool = mock_pool

            with pytest.raises(ValueError, match="Invalid message_type"):
                await extract_query_structure(
                    message_id=message_id,
                    message_content="test message",
                )


class TestGetExtraction:
    """Tests for get_extraction function."""

    @pytest.mark.asyncio
    async def test_get_extraction_success(self) -> None:
        """Test retrieving an extraction by ID."""
        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_row = {
            "id": extraction_id,
            "message_id": message_id,
            "extraction": {
                "message_type": "conversation",
                "entities": ["Alice"],
                "predicates": ["talked to"],
                "time_constraint": None,
                "activity": None,
                "query": None,
                "urgency": None,
                "continuation": True,
            },
            "rendered_prompt": "test prompt",
            "raw_response": "test response",
            "created_at": datetime.now(),
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with patch("lattice.core.query_extraction.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await get_extraction(extraction_id)

            assert result is not None
            assert result.id == extraction_id
            assert result.message_id == message_id
            assert result.message_type == "conversation"
            assert result.entities == ["Alice"]
            assert result.continuation is True

    @pytest.mark.asyncio
    async def test_get_extraction_not_found(self) -> None:
        """Test retrieving a non-existent extraction."""
        extraction_id = uuid.uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with patch("lattice.core.query_extraction.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await get_extraction(extraction_id)

            assert result is None


class TestGetMessageExtraction:
    """Tests for get_message_extraction function."""

    @pytest.mark.asyncio
    async def test_get_message_extraction_success(self) -> None:
        """Test retrieving extraction for a message."""
        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_row = {
            "id": extraction_id,
            "message_id": message_id,
            "extraction": {
                "message_type": "activity_update",
                "entities": [],
                "predicates": ["spent"],
                "time_constraint": None,
                "activity": "coding",
                "query": None,
                "urgency": None,
                "continuation": False,
            },
            "rendered_prompt": "test prompt",
            "raw_response": "test response",
            "created_at": datetime.now(),
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with patch("lattice.core.query_extraction.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await get_message_extraction(message_id)

            assert result is not None
            assert result.message_id == message_id
            assert result.message_type == "activity_update"
            assert result.activity == "coding"

    @pytest.mark.asyncio
    async def test_get_message_extraction_not_found(self) -> None:
        """Test retrieving extraction for non-existent message."""
        message_id = uuid.uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with patch("lattice.core.query_extraction.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await get_message_extraction(message_id)

            assert result is None
