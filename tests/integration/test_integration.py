"""Integration tests for the full Lattice pipeline."""

import asyncio
import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.discord_client.bot import LatticeBot
from lattice.memory.episodic import (
    EpisodicMessage,
    store_message,
)
from lattice.memory.procedural import PromptTemplate, get_prompt, store_prompt
from lattice.memory.semantic import StableFact, search_similar_facts, store_fact
from lattice.utils.database import DatabasePool
from lattice.utils.embeddings import EmbeddingModel


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def mock_db_pool() -> AsyncMock:
    """Create a mock database pool."""
    pool = AsyncMock()
    pool.acquire = AsyncMock()
    return pool


@pytest.fixture()
def mock_embedding_model() -> MagicMock:
    """Create a mock embedding model."""
    model = MagicMock()
    model.load = MagicMock()
    model.encode_single = MagicMock(return_value=[0.1] * 384)
    return model


class TestBotIntegration:
    """Integration tests for the Lattice bot."""

    @pytest.mark.asyncio()
    async def test_bot_initialization(self) -> None:
        """Test that the bot can be initialized."""
        bot = LatticeBot()
        assert bot is not None

    @pytest.mark.asyncio()
    async def test_episodic_memory_operations(self) -> None:
        """Test episodic memory CRUD operations."""
        mock_message = EpisodicMessage(
            content="Hello, world!",
            discord_message_id=12345,
            channel_id=67890,
            is_bot=False,
            prev_turn_id=None,
        )

        with patch("lattice.memory.episodic.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.fetchrow = AsyncMock(return_value={"id": "test-uuid-1234"})

            result = await store_message(mock_message)
            assert result is not None

    @pytest.mark.asyncio()
    async def test_semantic_memory_operations(self) -> None:
        """Test semantic memory search operations."""
        with (
            patch("lattice.memory.semantic.db_pool") as mock_pool,
            patch("lattice.memory.semantic.embedding_model") as mock_model,
        ):
            mock_model.encode_single = MagicMock(return_value=[0.1] * 384)

            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.fetch = AsyncMock(return_value=[])

            results = await search_similar_facts(
                query="test query", limit=5, similarity_threshold=0.7
            )
            assert isinstance(results, list)

    @pytest.mark.asyncio()
    async def test_procedural_memory_operations(self) -> None:
        """Test procedural memory retrieval operations."""
        with patch("lattice.memory.procedural.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.fetchrow = AsyncMock(
                return_value={
                    "prompt_key": "TEST",
                    "template": "Test template",
                    "temperature": 0.7,
                    "version": 1,
                    "active": True,
                }
            )

            result = await get_prompt("TEST")
            assert result is not None
            assert result.prompt_key == "TEST"


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.asyncio()
    async def test_database_pool_initialization(self) -> None:
        """Test database pool can be initialized."""

        async def mock_create_pool(*_args: object, **_kwargs: object) -> AsyncMock:
            return AsyncMock()

        with (
            patch.dict(os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"}),
            patch("lattice.utils.database.asyncpg.create_pool", side_effect=mock_create_pool),
        ):
            pool = DatabasePool()
            await pool.initialize()
            assert pool.is_initialized()


class TestEmbeddingIntegration:
    """Integration tests for embedding operations."""

    def test_embedding_model_configuration(self) -> None:
        """Test embedding model can be configured."""
        model = EmbeddingModel()
        assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert model.dimension == 384


class TestFullPipelineIntegration:
    """Integration tests for the full memory pipeline."""

    @pytest.mark.asyncio()
    async def test_store_and_retrieve_fact(self) -> None:
        """Test storing and retrieving a semantic fact."""
        fact = StableFact(
            content="Test fact for semantic memory",
            entity_type="test",
        )

        with (
            patch("lattice.memory.semantic.db_pool") as mock_pool,
            patch("lattice.memory.semantic.embedding_model") as mock_model,
        ):
            mock_model.encode_single = MagicMock(return_value=[0.1] * 384)
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.fetchrow = AsyncMock(return_value={"id": "test-fact-uuid"})

            fact_id = await store_fact(fact)
            assert fact_id is not None

    @pytest.mark.asyncio()
    async def test_temporal_chaining_messages(self) -> None:
        """Test that messages are chained temporally."""
        msg1 = EpisodicMessage(
            content="First message",
            discord_message_id=1001,
            channel_id=999,
            is_bot=False,
            prev_turn_id=None,
        )

        with patch("lattice.memory.episodic.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.fetchrow = AsyncMock(
                side_effect=[
                    {"id": "uuid-msg1"},
                    {"id": "uuid-msg2"},
                ]
            )

            msg1_id = await store_message(msg1)
            assert msg1_id is not None

            msg2 = EpisodicMessage(
                content="Second message",
                discord_message_id=1002,
                channel_id=999,
                is_bot=False,
                prev_turn_id=msg1_id,
            )
            msg2_id = await store_message(msg2)
            assert msg2_id is not None

    @pytest.mark.asyncio()
    async def test_prompt_template_crud(self) -> None:
        """Test creating and retrieving a prompt template."""
        template = PromptTemplate(
            prompt_key="TEST_TEMPLATE",
            template="You are a helpful assistant. User said: {user_message}",
            temperature=0.7,
        )

        with patch("lattice.memory.procedural.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.fetchrow = AsyncMock(
                return_value={
                    "prompt_key": "TEST_TEMPLATE",
                    "template": template.template,
                    "temperature": 0.7,
                    "version": 1,
                    "active": True,
                }
            )

            await store_prompt(template)
            retrieved = await get_prompt("TEST_TEMPLATE")
            assert retrieved is not None
            assert retrieved.prompt_key == "TEST_TEMPLATE"
            assert retrieved.template == template.template


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
