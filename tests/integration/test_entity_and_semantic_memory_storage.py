"""Integration tests for semantic memory storage.

These tests require a real database connection and are skipped in CI.
To run locally, set DATABASE_URL environment variable.

Note: After issue #131, semantic memories use text-based storage
(subject/predicate/object as TEXT) rather than entity IDs.
The entities table is no longer used for semantic memory storage.
"""

import os
import time
from typing import AsyncGenerator
from uuid import uuid4

import pytest

from lattice.memory import episodic
from lattice.memory.context import PostgresMessageRepository
from lattice.utils.database import DatabasePool


# Skip all tests if DATABASE_URL not set (CI environment)
pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set - requires real database for integration tests",
)


@pytest.fixture
async def db_pool() -> AsyncGenerator[DatabasePool, None]:
    """Fixture to provide a managed DatabasePool for each test."""
    pool = DatabasePool()
    await pool.initialize()
    yield pool
    await pool.close()


@pytest.fixture
def message_repo(db_pool: DatabasePool) -> PostgresMessageRepository:
    """Fixture to provide a PostgresMessageRepository."""
    return PostgresMessageRepository(db_pool)


@pytest.fixture(autouse=True)
async def cleanup_test_data(db_pool: DatabasePool) -> None:
    """Fixture to clean up test data before each test."""
    async with db_pool.pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM raw_messages WHERE discord_message_id IN (12345, 12346, 12348, 12349, 12350, 12351, 12352)"
        )
        # Also clean up semantic memories associated with these test batches
        await conn.execute(
            "DELETE FROM semantic_memories WHERE source_batch_id LIKE 'test_batch_%'"
        )


@pytest.mark.asyncio
class TestSemanticMemoryStorage:
    """Test semantic memory storage functionality."""

    async def test_store_simple_memory(
        self, db_pool: DatabasePool, message_repo: PostgresMessageRepository
    ) -> None:
        """Test storing a simple semantic memory."""
        # Create a test message
        message = episodic.EpisodicMessage(
            content="Alice works at OpenAI",
            discord_message_id=12345,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Store semantic memory
        memories = [{"subject": "user", "predicate": "works_at", "object": "OpenAI"}]
        await episodic.store_semantic_memories(message_repo, message_id, memories)

        # Verify memory was stored
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT subject, predicate, object
                FROM semantic_memories
                WHERE source_batch_id IS NULL
                ORDER BY created_at DESC
                LIMIT 1
                """
            )

            assert len(rows) >= 1
            found = False
            for row in rows:
                if (
                    row["subject"] == "user"
                    and row["predicate"] == "works_at"
                    and row["object"] == "OpenAI"
                ):
                    found = True
                    break
            assert found, "Memory not found in database"

    async def test_store_multiple_memories(
        self, db_pool: DatabasePool, message_repo: PostgresMessageRepository
    ) -> None:
        """Test storing multiple memories from same message."""
        # Create test message
        message = episodic.EpisodicMessage(
            content="Alice works at OpenAI and likes Python",
            discord_message_id=12346,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Store multiple memories with a batch ID for tracking
        batch_id = "test_batch_12346"
        memories = [
            {"subject": "user", "predicate": "works_at", "object": "OpenAI"},
            {"subject": "user", "predicate": "likes", "object": "Python"},
        ]
        await episodic.store_semantic_memories(
            message_repo, message_id, memories, source_batch_id=batch_id
        )

        # Verify both memories stored
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT COUNT(*) as count
                FROM semantic_memories
                WHERE source_batch_id = $1
                """,
                batch_id,
            )

            assert rows[0]["count"] == 2

    async def test_skip_invalid_memories(
        self, db_pool: DatabasePool, message_repo: PostgresMessageRepository
    ) -> None:
        """Test that invalid memories are skipped gracefully."""
        message = episodic.EpisodicMessage(
            content="Test message",
            discord_message_id=12348,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Mix of valid and invalid memories
        batch_id = "test_batch_12348"
        memories = [
            {"subject": "user", "predicate": "works_at", "object": "OpenAI"},  # Valid
            {
                "subject": "",
                "predicate": "likes",
                "object": "Python",
            },  # Invalid - empty subject
            {
                "subject": "user",
                "predicate": "",
                "object": "Coffee",
            },  # Invalid - empty predicate
            {
                "subject": "user",
                "predicate": "knows",
                "object": "",
            },  # Invalid - empty object
        ]

        # Should not raise exception
        await episodic.store_semantic_memories(
            message_repo, message_id, memories, source_batch_id=batch_id
        )

        # Verify only valid memory was stored
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT subject, predicate, object
                FROM semantic_memories
                WHERE source_batch_id = $1
                """,
                batch_id,
            )

            # Should have stored only the valid memory
            assert len(rows) == 1
            assert rows[0]["subject"] == "user"
            assert rows[0]["predicate"] == "works_at"
            assert rows[0]["object"] == "OpenAI"

    async def test_empty_memories_list(
        self, db_pool: DatabasePool, message_repo: PostgresMessageRepository
    ) -> None:
        """Test that empty memories list is handled gracefully."""
        message = episodic.EpisodicMessage(
            content="No memories here",
            discord_message_id=12349,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Should not raise exception
        batch_id = "test_batch_12349"
        await episodic.store_semantic_memories(
            message_repo, message_id, [], source_batch_id=batch_id
        )

        # Verify no memories stored
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT COUNT(*) as count
                FROM semantic_memories
                WHERE source_batch_id = $1
                """,
                batch_id,
            )
            assert rows[0]["count"] == 0

    async def test_text_based_memory_search(
        self, db_pool: DatabasePool, message_repo: PostgresMessageRepository
    ) -> None:
        """Test searching for memories by text matching."""
        # Use a unique batch ID to avoid collisions in parallel test execution
        unique_suffix = str(uuid4())[:8]
        batch_id = f"test_batch_12350_{unique_suffix}"

        message = episodic.EpisodicMessage(
            content="Test memory search",
            discord_message_id=12350,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Store a memory with unique identifiers
        memories = [
            {"subject": "user", "predicate": "lives_in", "object": "Richmond, BC"}
        ]
        await episodic.store_semantic_memories(
            message_repo, message_id, memories, source_batch_id=batch_id
        )

        # Small delay to ensure data is committed and visible across connections
        time.sleep(0.1)

        # Search by subject
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT subject, predicate, object
                FROM semantic_memories
                WHERE subject ILIKE $1 AND source_batch_id = $2
                """,
                "%user%",
                batch_id,
            )
            assert len(rows) == 1
            assert rows[0]["predicate"] == "lives_in"

        # Search by object
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT subject, predicate, object
                FROM semantic_memories
                WHERE object ILIKE $1 AND source_batch_id = $2
                """,
                "%Richmond%",
                batch_id,
            )
            assert len(rows) == 1
            assert rows[0]["subject"] == "user"

    async def test_bidirectional_alias_storage(
        self, db_pool: DatabasePool, message_repo: PostgresMessageRepository
    ) -> None:
        """Test that 'has alias' triples create bidirectional relationships."""
        # Use a unique batch ID
        unique_suffix = str(uuid4())[:8]
        batch_id = f"test_batch_alias_{unique_suffix}"

        message = episodic.EpisodicMessage(
            content="Test alias storage",
            discord_message_id=12351,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Store alias triple
        memories = [{"subject": "mom", "predicate": "has alias", "object": "Mother"}]
        await episodic.store_semantic_memories(
            message_repo, message_id, memories, source_batch_id=batch_id
        )

        # Verify both directions exist
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT subject, predicate, object
                FROM semantic_memories
                WHERE source_batch_id = $1
                ORDER BY subject, object
                """,
                batch_id,
            )

            assert len(rows) == 2
            # Check original direction
            assert rows[0]["subject"] == "mom"
            assert rows[0]["predicate"] == "has alias"
            assert rows[0]["object"] == "Mother"
            # Check reverse direction
            assert rows[1]["subject"] == "Mother"
            assert rows[1]["predicate"] == "has alias"
            assert rows[1]["object"] == "mom"

    async def test_duplicate_alias_handling(
        self, db_pool: DatabasePool, message_repo: PostgresMessageRepository
    ) -> None:
        """Test that duplicate alias triples are ignored."""
        unique_suffix = str(uuid4())[:8]
        batch_id = f"test_batch_dup_{unique_suffix}"

        message = episodic.EpisodicMessage(
            content="Test duplicate handling",
            discord_message_id=12352,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Store the same alias twice
        memories = [
            {"subject": "bf", "predicate": "has alias", "object": "boyfriend"},
            {"subject": "bf", "predicate": "has alias", "object": "boyfriend"},
        ]
        result_count = await episodic.store_semantic_memories(
            message_repo, message_id, memories, source_batch_id=batch_id
        )

        # Should return 2 (one for each original alias, reverse should be deduped)
        assert result_count == 2

        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT COUNT(*) as count
                FROM semantic_memories
                WHERE source_batch_id = $1
                """,
                batch_id,
            )
            # Only 2 unique triples should exist (bf->boyfriend and boyfriend->bf)
            assert rows[0]["count"] == 2


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
