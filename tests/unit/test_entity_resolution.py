"""Unit tests for entity resolution module."""

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.core.entity_resolution import (
    EntityResolution,
    get_entity_by_id,
    get_entity_by_name,
    resolve_entity,
)
from lattice.memory.procedural import PromptTemplate
from lattice.utils.llm import GenerationResult


@pytest.fixture
def mock_prompt_template() -> PromptTemplate:
    """Create a mock ENTITY_NORMALIZATION prompt template."""
    return PromptTemplate(
        prompt_key="ENTITY_NORMALIZATION",
        template="Normalize: {entity_mention}\nContext: {message_context}\nExisting: {existing_entities}",
        temperature=0.2,
        version=1,
        active=True,
    )


@pytest.fixture
def mock_normalization_response() -> str:
    """Create a mock LLM normalization response."""
    return json.dumps(
        {
            "canonical_name": "Alice Johnson",
            "reasoning": "Normalized from lowercase to proper capitalization",
        }
    )


@pytest.fixture
def mock_generation_result(mock_normalization_response: str) -> GenerationResult:
    """Create a mock GenerationResult."""
    return GenerationResult(
        content=mock_normalization_response,
        model="anthropic/claude-3.5-sonnet",
        provider="anthropic",
        prompt_tokens=50,
        completion_tokens=25,
        total_tokens=75,
        cost_usd=0.0005,
        latency_ms=300,
        temperature=0.2,
    )


class TestEntityResolution:
    """Tests for the EntityResolution dataclass."""

    def test_entity_resolution_init(self) -> None:
        """Test EntityResolution initialization."""
        entity_id = uuid.uuid4()

        resolution = EntityResolution(
            canonical_name="Alice",
            entity_id=entity_id,
            is_new=False,
            reasoning="Direct match",
        )

        assert resolution.canonical_name == "Alice"
        assert resolution.entity_id == entity_id
        assert resolution.is_new is False
        assert resolution.reasoning == "Direct match"


class TestResolveEntity:
    """Tests for resolve_entity function."""

    @pytest.mark.asyncio
    async def test_resolve_entity_tier1_direct_match(self) -> None:
        """Test Tier 1 resolution with direct case-insensitive match."""
        entity_id = uuid.uuid4()
        entity_mention = "alice"

        # Mock database response for existing entity
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={"id": entity_id, "name": "Alice Johnson"}
        )

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with patch("lattice.core.entity_resolution.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await resolve_entity(
                entity_mention=entity_mention,
                message_context="alice helped me today",
            )

            assert result.canonical_name == "Alice Johnson"
            assert result.entity_id == entity_id
            assert result.is_new is False
            assert "Direct case-insensitive match" in result.reasoning

    @pytest.mark.asyncio
    async def test_resolve_entity_tier3_llm_normalization_existing(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test Tier 3 resolution where LLM normalizes to existing entity."""
        entity_id = uuid.uuid4()
        entity_mention = "alice"

        # Mock database: no direct match, then find after LLM normalization
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                None,  # First call: no direct match
                {"id": entity_id, "name": "Alice Johnson"},  # After LLM: found
            ]
        )
        mock_conn.fetch = AsyncMock(return_value=[{"name": "Bob"}, {"name": "Charlie"}])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with (
            patch("lattice.core.entity_resolution.db_pool") as mock_db_pool,
            patch("lattice.core.entity_resolution.get_prompt") as mock_get_prompt,
            patch("lattice.core.entity_resolution.get_llm_client") as mock_get_llm,
        ):
            mock_db_pool.pool = mock_pool
            mock_get_prompt.return_value = mock_prompt_template

            mock_llm_client = AsyncMock()
            mock_llm_client.complete = AsyncMock(return_value=mock_generation_result)
            mock_get_llm.return_value = mock_llm_client

            result = await resolve_entity(
                entity_mention=entity_mention,
                message_context="talked to alice yesterday",
            )

            assert result.canonical_name == "Alice Johnson"
            assert result.entity_id == entity_id
            assert result.is_new is False
            assert "LLM normalization" in result.reasoning

    @pytest.mark.asyncio
    async def test_resolve_entity_tier3_creates_new_entity(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: GenerationResult,
    ) -> None:
        """Test Tier 3 resolution creates new entity when none exists."""
        entity_mention = "john"
        new_entity_id = uuid.uuid4()

        # Mock database: no matches, create new entity
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                None,  # First call: no direct match for "john"
                None,  # Second call: no match for "John" after LLM
                {"id": new_entity_id},  # Third call: entity creation returns ID
            ]
        )
        mock_conn.fetch = AsyncMock(return_value=[{"name": "Alice"}, {"name": "Bob"}])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        # Mock LLM response for new entity
        new_entity_response = json.dumps(
            {
                "canonical_name": "John",
                "reasoning": "New entity, proper name capitalization",
            }
        )
        mock_result = GenerationResult(
            content=new_entity_response,
            model="test-model",
            provider=None,
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            cost_usd=None,
            latency_ms=200,
            temperature=0.2,
        )

        with (
            patch("lattice.core.entity_resolution.db_pool") as mock_db_pool,
            patch("lattice.core.entity_resolution.get_prompt") as mock_get_prompt,
            patch("lattice.core.entity_resolution.get_llm_client") as mock_get_llm,
        ):
            mock_db_pool.pool = mock_pool
            mock_get_prompt.return_value = mock_prompt_template

            mock_llm_client = AsyncMock()
            mock_llm_client.complete = AsyncMock(return_value=mock_result)
            mock_get_llm.return_value = mock_llm_client

            result = await resolve_entity(
                entity_mention=entity_mention,
                message_context="met john at the conference",
            )

            assert result.canonical_name == "John"
            assert result.entity_id == new_entity_id
            assert result.is_new is True
            assert "New entity" in result.reasoning

    @pytest.mark.asyncio
    async def test_resolve_entity_handles_markdown_json(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test entity resolution handles JSON wrapped in markdown."""
        entity_mention = "python"
        new_entity_id = uuid.uuid4()

        # Mock database
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                None,  # No direct match
                None,  # No match after LLM
                {"id": new_entity_id},  # Entity creation
            ]
        )
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        # Mock LLM response with markdown wrapper
        markdown_response = f"""```json
{
            json.dumps(
                {
                    "canonical_name": "Python",
                    "reasoning": "Programming language, proper capitalization",
                }
            )
        }
```"""
        mock_result = GenerationResult(
            content=markdown_response,
            model="test-model",
            provider=None,
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            cost_usd=None,
            latency_ms=200,
            temperature=0.2,
        )

        with (
            patch("lattice.core.entity_resolution.db_pool") as mock_db_pool,
            patch("lattice.core.entity_resolution.get_prompt") as mock_get_prompt,
            patch("lattice.core.entity_resolution.get_llm_client") as mock_get_llm,
        ):
            mock_db_pool.pool = mock_pool
            mock_get_prompt.return_value = mock_prompt_template

            mock_llm_client = AsyncMock()
            mock_llm_client.complete = AsyncMock(return_value=mock_result)
            mock_get_llm.return_value = mock_llm_client

            result = await resolve_entity(
                entity_mention=entity_mention,
                message_context="learning python programming",
            )

            assert result.canonical_name == "Python"
            assert result.is_new is True

    @pytest.mark.asyncio
    async def test_resolve_entity_missing_prompt_template(self) -> None:
        """Test resolution fails when prompt template not found."""
        # Mock database: no direct match
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with (
            patch("lattice.core.entity_resolution.db_pool") as mock_db_pool,
            patch("lattice.core.entity_resolution.get_prompt") as mock_get_prompt,
        ):
            mock_db_pool.pool = mock_pool
            mock_get_prompt.return_value = None

            with pytest.raises(
                ValueError, match="ENTITY_NORMALIZATION prompt template not found"
            ):
                await resolve_entity(
                    entity_mention="test",
                    message_context="test context",
                )

    @pytest.mark.asyncio
    async def test_resolve_entity_invalid_json_response(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test resolution fails gracefully with invalid JSON."""
        # Mock database: no matches
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        # Mock invalid JSON response
        mock_result = GenerationResult(
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
            patch("lattice.core.entity_resolution.db_pool") as mock_db_pool,
            patch("lattice.core.entity_resolution.get_prompt") as mock_get_prompt,
            patch("lattice.core.entity_resolution.get_llm_client") as mock_get_llm,
        ):
            mock_db_pool.pool = mock_pool
            mock_get_prompt.return_value = mock_prompt_template

            mock_llm_client = AsyncMock()
            mock_llm_client.complete = AsyncMock(return_value=mock_result)
            mock_get_llm.return_value = mock_llm_client

            with pytest.raises(json.JSONDecodeError):
                await resolve_entity(
                    entity_mention="test",
                    message_context="test context",
                )

    @pytest.mark.asyncio
    async def test_resolve_entity_missing_canonical_name(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test resolution fails when canonical_name missing from response."""
        # Mock database: no matches
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        # Mock response missing canonical_name
        incomplete_response = json.dumps(
            {
                "reasoning": "Some reasoning",
                # Missing canonical_name
            }
        )
        mock_result = GenerationResult(
            content=incomplete_response,
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
            patch("lattice.core.entity_resolution.db_pool") as mock_db_pool,
            patch("lattice.core.entity_resolution.get_prompt") as mock_get_prompt,
            patch("lattice.core.entity_resolution.get_llm_client") as mock_get_llm,
        ):
            mock_db_pool.pool = mock_pool
            mock_get_prompt.return_value = mock_prompt_template

            mock_llm_client = AsyncMock()
            mock_llm_client.complete = AsyncMock(return_value=mock_result)
            mock_get_llm.return_value = mock_llm_client

            with pytest.raises(ValueError, match="Missing 'canonical_name'"):
                await resolve_entity(
                    entity_mention="test",
                    message_context="test context",
                )

    @pytest.mark.asyncio
    async def test_resolve_entity_empty_canonical_name(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test resolution fails when canonical_name is empty or whitespace."""
        # Mock database: no matches
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        # Mock response with empty canonical_name
        empty_response = json.dumps(
            {
                "canonical_name": "   ",  # Whitespace-only
                "reasoning": "Some reasoning",
            }
        )
        mock_result = GenerationResult(
            content=empty_response,
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
            patch("lattice.core.entity_resolution.db_pool") as mock_db_pool,
            patch("lattice.core.entity_resolution.get_prompt") as mock_get_prompt,
            patch("lattice.core.entity_resolution.get_llm_client") as mock_get_llm,
        ):
            mock_db_pool.pool = mock_pool
            mock_get_prompt.return_value = mock_prompt_template

            mock_llm_client = AsyncMock()
            mock_llm_client.complete = AsyncMock(return_value=mock_result)
            mock_get_llm.return_value = mock_llm_client

            with pytest.raises(
                ValueError, match="Empty or whitespace-only canonical_name"
            ):
                await resolve_entity(
                    entity_mention="test",
                    message_context="test context",
                )


class TestGetEntityById:
    """Tests for get_entity_by_id function."""

    @pytest.mark.asyncio
    async def test_get_entity_by_id_success(self) -> None:
        """Test retrieving an entity by ID."""
        entity_id = uuid.uuid4()
        now = datetime.now(UTC)

        mock_row = {
            "id": entity_id,
            "name": "Alice Johnson",
            "entity_type": "person",
            "metadata": {"role": "developer"},
            "first_mentioned": now,
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with patch("lattice.core.entity_resolution.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await get_entity_by_id(entity_id)

            assert result is not None
            assert result["id"] == entity_id
            assert result["name"] == "Alice Johnson"
            assert result["entity_type"] == "person"
            assert result["metadata"] == {"role": "developer"}
            assert result["first_mentioned"] == now

    @pytest.mark.asyncio
    async def test_get_entity_by_id_not_found(self) -> None:
        """Test retrieving a non-existent entity."""
        entity_id = uuid.uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with patch("lattice.core.entity_resolution.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await get_entity_by_id(entity_id)

            assert result is None


class TestGetEntityByName:
    """Tests for get_entity_by_name function."""

    @pytest.mark.asyncio
    async def test_get_entity_by_name_success(self) -> None:
        """Test retrieving an entity by name (case-insensitive)."""
        entity_id = uuid.uuid4()
        now = datetime.now(UTC)

        mock_row = {
            "id": entity_id,
            "name": "Alice Johnson",
            "entity_type": "person",
            "metadata": {},
            "first_mentioned": now,
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with patch("lattice.core.entity_resolution.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            # Test case-insensitive lookup
            result = await get_entity_by_name("alice johnson")

            assert result is not None
            assert result["name"] == "Alice Johnson"

    @pytest.mark.asyncio
    async def test_get_entity_by_name_not_found(self) -> None:
        """Test retrieving a non-existent entity by name."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        with patch("lattice.core.entity_resolution.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await get_entity_by_name("nonexistent")

            assert result is None
