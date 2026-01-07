"""Unit tests for entity extraction module."""

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from lattice.core.query_extraction import (
    EntityExtraction,
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
        template="Extract entities from: {message_content}\nContext: {context}",
        temperature=0.2,
        version=2,
        active=True,
    )


@pytest.fixture
def mock_llm_response() -> str:
    """Create a mock LLM response with valid extraction JSON."""
    return json.dumps(
        {
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
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        cost_usd=0.0005,
        latency_ms=300,
        temperature=0.2,
    )


class TestEntityExtraction:
    """Tests for the EntityExtraction dataclass."""

    def test_entity_extraction_init(self) -> None:
        """Test EntityExtraction initialization."""
        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()
        now = datetime.now()

        extraction = EntityExtraction(
            id=extraction_id,
            message_id=message_id,
            entities=["lattice project", "Friday"],
            rendered_prompt="test prompt",
            raw_response="test response",
            extraction_method="api",
            created_at=now,
        )

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
            content="{}",  # Missing entities
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
    async def test_extract_query_structure_handles_markdown_json(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test extraction handles markdown-wrapped JSON."""
        message_id = uuid.uuid4()

        markdown_response = GenerationResult(
            content="```json\n" + json.dumps({"entities": []}) + "\n```",
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
