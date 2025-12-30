"""Integration tests for Lattice - requires test database."""

import os
from datetime import UTC
from uuid import UUID

import asyncpg
import pytest


# Set test database URL before importing modules
os.environ["DATABASE_URL"] = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://lattice:lattice_dev_password@localhost:5432/lattice_test",
)

from lattice.memory import episodic, procedural, semantic
from lattice.utils.database import db_pool
from lattice.utils.embeddings import embedding_model


@pytest.fixture(scope="session")
async def test_db():
    """Set up test database with schema."""
    database_url = os.environ["DATABASE_URL"]

    # Create test database if it doesn't exist
    try:
        # Connect to default postgres database
        conn = await asyncpg.connect(database_url.replace("/lattice_test", "/postgres"))
        try:
            await conn.execute("CREATE DATABASE lattice_test")
        except asyncpg.DuplicateDatabaseError:
            pass  # Database already exists
        finally:
            await conn.close()
    except Exception:
        pass  # Ignore if we can't create (might already exist)

    # Initialize database pool
    await db_pool.initialize()

    # Create schema
    async with db_pool.pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_registry (
                prompt_key TEXT PRIMARY KEY,
                template TEXT NOT NULL,
                version INT DEFAULT 1,
                temperature FLOAT DEFAULT 0.2,
                updated_at TIMESTAMPTZ DEFAULT now(),
                active BOOLEAN DEFAULT true,
                pending_approval BOOLEAN DEFAULT false,
                proposed_template TEXT
            )
        """
        )

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_messages (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                discord_message_id BIGINT UNIQUE NOT NULL,
                channel_id BIGINT NOT NULL,
                content TEXT NOT NULL,
                is_bot BOOLEAN DEFAULT false,
                prev_turn_id UUID REFERENCES raw_messages(id),
                timestamp TIMESTAMPTZ DEFAULT now()
            )
        """
        )

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stable_facts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                embedding VECTOR(384),
                origin_id UUID REFERENCES raw_messages(id),
                entity_type TEXT,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS stable_facts_embedding_idx
            ON stable_facts USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
        )

    # Load embedding model
    embedding_model.load()

    yield

    # Cleanup
    await db_pool.close()


@pytest.fixture()
async def clean_db(test_db):
    """Clean database before each test."""
    async with db_pool.pool.acquire() as conn:
        await conn.execute("TRUNCATE raw_messages CASCADE")
        await conn.execute("TRUNCATE stable_facts CASCADE")
        await conn.execute("TRUNCATE prompt_registry CASCADE")


@pytest.mark.asyncio()
class TestEpisodicMemoryIntegration:
    """Integration tests for episodic memory."""

    async def test_store_and_retrieve_message(self, clean_db):
        """Test storing and retrieving a message."""
        message = episodic.EpisodicMessage(
            content="Test message",
            discord_message_id=123456789,
            channel_id=987654321,
            is_bot=False,
        )

        # Store message
        message_id = await episodic.store_message(message)
        assert isinstance(message_id, UUID)

        # Retrieve recent messages
        messages = await episodic.get_recent_messages(channel_id=987654321, limit=10)
        assert len(messages) == 1
        assert messages[0].content == "Test message"
        assert messages[0].discord_message_id == 123456789

    async def test_temporal_chaining(self, clean_db):
        """Test that messages are chained with prev_turn_id."""
        # Store first message
        msg1 = episodic.EpisodicMessage(
            content="First message",
            discord_message_id=111,
            channel_id=999,
            is_bot=False,
        )
        msg1_id = await episodic.store_message(msg1)

        # Store second message with chain
        msg2 = episodic.EpisodicMessage(
            content="Second message",
            discord_message_id=222,
            channel_id=999,
            is_bot=True,
            prev_turn_id=msg1_id,
        )
        msg2_id = await episodic.store_message(msg2)

        # Verify chain
        last_id = await episodic.get_last_message_id(999)
        assert last_id == msg2_id

        messages = await episodic.get_recent_messages(channel_id=999, limit=10)
        assert len(messages) == 2
        assert messages[0].content == "First message"
        assert messages[1].content == "Second message"
        assert messages[1].prev_turn_id == msg1_id

    async def test_timezone_aware_timestamps(self, clean_db):
        """Test that timestamps are timezone-aware."""
        message = episodic.EpisodicMessage(
            content="Timestamp test",
            discord_message_id=999999,
            channel_id=111111,
            is_bot=False,
        )

        # Message should have UTC timestamp
        assert message.timestamp.tzinfo == UTC


@pytest.mark.asyncio()
class TestSemanticMemoryIntegration:
    """Integration tests for semantic memory."""

    async def test_store_and_search_facts(self, clean_db):
        """Test storing facts and searching with vector similarity."""
        # Store a fact
        fact = semantic.StableFact(
            content="User prefers dark mode for coding",
            entity_type="preference",
        )
        fact_id = await semantic.store_fact(fact)
        assert isinstance(fact_id, UUID)

        # Search for similar facts
        results = await semantic.search_similar_facts(
            query="What are the user's preferences for themes?",
            limit=5,
            similarity_threshold=0.3,  # Lower threshold for test
        )

        assert len(results) >= 1
        assert any("dark mode" in r.content.lower() for r in results)

    async def test_vector_similarity_ordering(self, clean_db):
        """Test that results are ordered by similarity."""
        # Store multiple facts
        facts = [
            semantic.StableFact(
                content="User loves Python programming",
                entity_type="preference",
            ),
            semantic.StableFact(
                content="User enjoys JavaScript development",
                entity_type="preference",
            ),
            semantic.StableFact(
                content="User's favorite color is blue",
                entity_type="preference",
            ),
        ]

        for fact in facts:
            await semantic.store_fact(fact)

        # Search for programming-related query
        results = await semantic.search_similar_facts(
            query="programming languages the user likes",
            limit=3,
            similarity_threshold=0.2,
        )

        # Should find programming-related facts
        assert len(results) >= 2
        programming_facts = [
            r for r in results if "python" in r.content.lower() or "javascript" in r.content.lower()
        ]
        assert len(programming_facts) >= 2

    async def test_search_input_validation(self, clean_db):
        """Test that search validates inputs correctly."""
        # Test invalid limit
        with pytest.raises(ValueError, match="limit must be between"):
            await semantic.search_similar_facts("query", limit=0)

        with pytest.raises(ValueError, match="limit must be between"):
            await semantic.search_similar_facts("query", limit=101)

        # Test invalid similarity threshold
        with pytest.raises(ValueError, match="similarity_threshold must be"):
            await semantic.search_similar_facts("query", similarity_threshold=-0.1)

        with pytest.raises(ValueError, match="similarity_threshold must be"):
            await semantic.search_similar_facts("query", similarity_threshold=1.5)

    async def test_empty_query_returns_empty_list(self, clean_db):
        """Test that empty query returns empty list."""
        results = await semantic.search_similar_facts("   ", limit=5)
        assert results == []


@pytest.mark.asyncio()
class TestProceduralMemoryIntegration:
    """Integration tests for procedural memory."""

    async def test_store_and_retrieve_prompt(self, clean_db):
        """Test storing and retrieving prompt templates."""
        template = procedural.PromptTemplate(
            prompt_key="TEST_PROMPT",
            template="Hello {name}, how are you?",
            temperature=0.8,
        )

        await procedural.store_prompt(template)

        # Retrieve the prompt
        retrieved = await procedural.get_prompt("TEST_PROMPT")
        assert retrieved is not None
        assert retrieved.prompt_key == "TEST_PROMPT"
        assert retrieved.template == "Hello {name}, how are you?"
        assert retrieved.temperature == 0.8

    async def test_prompt_versioning(self, clean_db):
        """Test that updating a prompt increments version."""
        # Store initial version
        template_v1 = procedural.PromptTemplate(
            prompt_key="VERSIONED",
            template="Version 1",
            temperature=0.7,
        )
        await procedural.store_prompt(template_v1)

        # Update the prompt
        template_v2 = procedural.PromptTemplate(
            prompt_key="VERSIONED",
            template="Version 2",
            temperature=0.8,
        )
        await procedural.store_prompt(template_v2)

        # Retrieve and check version incremented
        retrieved = await procedural.get_prompt("VERSIONED")
        assert retrieved is not None
        assert retrieved.template == "Version 2"
        assert retrieved.version >= 2

    async def test_nonexistent_prompt_returns_none(self, clean_db):
        """Test that getting non-existent prompt returns None."""
        result = await procedural.get_prompt("NONEXISTENT_PROMPT")
        assert result is None


@pytest.mark.asyncio()
class TestUtilModules:
    """Integration tests for utility modules."""

    def test_embedding_model_encoding(self):
        """Test that embedding model can encode text."""
        # Model should be loaded by test_db fixture
        text = "This is a test sentence"
        embedding = embedding_model.encode_single(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384  # Model dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_embedding_batch_encoding(self):
        """Test batch encoding of multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = embedding_model.encode(texts)

        assert embeddings.shape == (3, 384)

    async def test_database_pool_operations(self, test_db):
        """Test basic database pool operations."""
        # Pool should be initialized by test_db fixture
        assert db_pool.pool is not None

        # Test a simple query
        async with db_pool.pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1
