"""Unit tests for query extraction module."""

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, patch

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
        version=2,  # Updated in Design D
        active=True,
    )


@pytest.fixture
def mock_llm_response() -> str:
    """Create a mock LLM response with valid extraction JSON (Design D: 2 fields)."""
    return json.dumps(
        {
            "message_type": "goal",
            "entities": ["lattice project", "Friday"],
        }
    )


@pytest.fixture
def mock_generation_result(mock_llm_response: str) -> GenerationResult:
    """Create a mock GenerationResult."""
    return GenerationResult(
        content=mock_llm_response,
        model="anthropic/claude-3.5-sonnet",
        provider="anthropic",
        prompt_tokens=50,  # Reduced (simpler prompt)
        completion_tokens=20,  # Reduced (simpler output)
        total_tokens=70,
        cost_usd=0.0005,
        latency_ms=300,
        temperature=0.2,
    )


class TestQueryExtraction:
    """Tests for the QueryExtraction dataclass."""

    def test_query_extraction_init(self) -> None:
        """Test QueryExtraction initialization (Design D: simplified to 2 fields)."""
        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()
        now = datetime.now()

        extraction = QueryExtraction(
            id=extraction_id,
            message_id=message_id,
            message_type="goal",
            entities=["lattice project", "Friday"],
            rendered_prompt="test prompt",
            raw_response="test response",
            extraction_method="api",
            created_at=now,
        )

        assert extraction.id == extraction_id
        assert extraction.message_id == message_id
        assert extraction.message_type == "goal"
        assert extraction.entities == ["lattice project", "Friday"]
        assert extraction.rendered_prompt == "test prompt"
        assert extraction.raw_response == "test response"
        assert extraction.extraction_method == "api"
        assert extraction.created_at == now


class TestExtractQueryStructure:
    """Tests for the extract_query_structure function."""

    @pytest.mark.asyncio
    async def test_extract_query_structure_success(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test successful extraction."""
        message_id = uuid.uuid4()

        with (
            patch(
                "lattice.core.query_extraction.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.core.query_extraction.get_llm_client",
            ) as mock_llm_client,
            patch("lattice.core.query_extraction.db_pool") as mock_db_pool,
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_generation_result)
            mock_llm_client.return_value = mock_client

            mock_conn = AsyncMock()
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            extraction = await extract_query_structure(
                message_id=message_id,
                message_content="I need to finish the lattice project by Friday",
                context="Previous context",
            )

            assert extraction.message_id == message_id
            assert extraction.message_type == "goal"
            assert extraction.entities == ["lattice project", "Friday"]
            assert extraction.extraction_method == "api"

    @pytest.mark.asyncio
    async def test_extract_query_structure_missing_prompt_template(self) -> None:
        """Test extraction with missing prompt template."""
        message_id = uuid.uuid4()

        with patch(
            "lattice.core.query_extraction.get_prompt",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="QUERY_EXTRACTION prompt template"):
                await extract_query_structure(
                    message_id=message_id,
                    message_content="Test message",
                )

    @pytest.mark.asyncio
    async def test_extract_query_structure_invalid_json(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test extraction with invalid JSON response."""
        message_id = uuid.uuid4()

        invalid_response = GenerationResult(
            content="This is not JSON",
            model="test",
            provider=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=None,
            latency_ms=0,
            temperature=0.0,
        )

        with (
            patch(
                "lattice.core.query_extraction.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.core.query_extraction.get_llm_client",
            ) as mock_llm_client,
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=invalid_response)
            mock_llm_client.return_value = mock_client

            with pytest.raises(json.JSONDecodeError):
                await extract_query_structure(
                    message_id=message_id,
                    message_content="Test message",
                )

    @pytest.mark.asyncio
    async def test_extract_query_structure_missing_required_fields(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test extraction with missing required fields."""
        message_id = uuid.uuid4()

        incomplete_response = GenerationResult(
            content=json.dumps({"message_type": "goal"}),  # Missing entities
            model="test",
            provider=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=None,
            latency_ms=0,
            temperature=0.0,
        )

        with (
            patch(
                "lattice.core.query_extraction.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.core.query_extraction.get_llm_client",
            ) as mock_llm_client,
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=incomplete_response)
            mock_llm_client.return_value = mock_client

            with pytest.raises(ValueError, match="Missing required field"):
                await extract_query_structure(
                    message_id=message_id,
                    message_content="Test message",
                )

    @pytest.mark.asyncio
    async def test_extract_query_structure_invalid_message_type(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test extraction with invalid message type."""
        message_id = uuid.uuid4()

        invalid_type_response = GenerationResult(
            content=json.dumps(
                {
                    "message_type": "invalid_type",
                    "entities": [],
                }
            ),
            model="test",
            provider=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=None,
            latency_ms=0,
            temperature=0.0,
        )

        with (
            patch(
                "lattice.core.query_extraction.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.core.query_extraction.get_llm_client",
            ) as mock_llm_client,
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=invalid_type_response)
            mock_llm_client.return_value = mock_client

            with pytest.raises(ValueError, match="Invalid message_type"):
                await extract_query_structure(
                    message_id=message_id,
                    message_content="Test message",
                )

    @pytest.mark.asyncio
    async def test_extract_query_structure_handles_markdown_json(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test extraction handles markdown-wrapped JSON."""
        message_id = uuid.uuid4()

        markdown_response = GenerationResult(
            content="```json\n"
            + json.dumps({"message_type": "goal", "entities": []})
            + "\n```",
            model="test",
            provider=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=None,
            latency_ms=0,
            temperature=0.0,
        )

        with (
            patch(
                "lattice.core.query_extraction.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.core.query_extraction.get_llm_client",
            ) as mock_llm_client,
            patch("lattice.core.query_extraction.db_pool") as mock_db_pool,
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=markdown_response)
            mock_llm_client.return_value = mock_client

            mock_conn = AsyncMock()
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            extraction = await extract_query_structure(
                message_id=message_id,
                message_content="Test message",
            )

            assert extraction.message_type == "goal"
            assert extraction.entities == []


class TestGetExtraction:
    """Tests for the get_extraction function."""

    @pytest.mark.asyncio
    async def test_get_extraction_success(self) -> None:
        """Test retrieving an extraction."""
        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_row = {
            "id": extraction_id,
            "message_id": message_id,
            "extraction": {
                "message_type": "question",
                "entities": ["Alice", "yesterday"],
            },
            "rendered_prompt": "test prompt",
            "raw_response": "test response",
            "created_at": datetime.now(),
        }

        with patch("lattice.core.query_extraction.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=mock_row)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            extraction = await get_extraction(extraction_id)

            assert extraction is not None
            assert extraction.id == extraction_id
            assert extraction.message_type == "question"
            assert extraction.entities == ["Alice", "yesterday"]

    @pytest.mark.asyncio
    async def test_get_extraction_not_found(self) -> None:
        """Test retrieving a non-existent extraction."""
        extraction_id = uuid.uuid4()

        with patch("lattice.core.query_extraction.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=None)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            extraction = await get_extraction(extraction_id)

            assert extraction is None


class TestGetMessageExtraction:
    """Tests for the get_message_extraction function."""

    @pytest.mark.asyncio
    async def test_get_message_extraction_success(self) -> None:
        """Test retrieving an extraction by message ID."""
        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_row = {
            "id": extraction_id,
            "message_id": message_id,
            "extraction": {
                "message_type": "activity_update",
                "entities": [],
            },
            "rendered_prompt": "test prompt",
            "raw_response": "test response",
            "created_at": datetime.now(),
        }

        with patch("lattice.core.query_extraction.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=mock_row)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            extraction = await get_message_extraction(message_id)

            assert extraction is not None
            assert extraction.message_id == message_id
            assert extraction.message_type == "activity_update"
            assert extraction.entities == []

    @pytest.mark.asyncio
    async def test_get_message_extraction_not_found(self) -> None:
        """Test retrieving a non-existent message extraction."""
        message_id = uuid.uuid4()

        with patch("lattice.core.query_extraction.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=None)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            extraction = await get_message_extraction(message_id)

            assert extraction is None
