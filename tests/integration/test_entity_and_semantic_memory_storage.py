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

from lattice.core.constants import ALIAS_PREDICATE
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
            "DELETE FROM raw_messages WHERE discord_message_id IN (12345, 12346, 12348, 12349, 12350, 12351, 12352, 12353, 12354)"
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

        # Store semantic memory with batch_id for cleanup
        batch_id = "test_batch_12345"
        memories = [{"subject": "user", "predicate": "works_at", "object": "OpenAI"}]
        await episodic.store_semantic_memories(
            message_repo, message_id, memories, source_batch_id=batch_id
        )

        # Verify memory was stored
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT subject, predicate, object
                FROM semantic_memories
                WHERE source_batch_id = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                batch_id,
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

            # With unique constraint, both distinct memories are stored
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
            {"subject": "user", "predicate": "likes", "object": "Coffee"},  # Valid
            {
                "subject": "",
                "predicate": "likes",
                "object": "Python",
            },  # Invalid - empty subject
            {
                "subject": "user",
                "predicate": "",
                "object": "Tea",
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
            assert rows[0]["predicate"] == "likes"
            assert rows[0]["object"] == "Coffee"

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
        memories = [
            {"subject": "mom", "predicate": ALIAS_PREDICATE, "object": "Mother"}
        ]
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
            # Check both directions exist regardless of order
            triples = [(r["subject"], r["predicate"], r["object"]) for r in rows]
            assert ("mom", ALIAS_PREDICATE, "Mother") in triples
            assert ("Mother", ALIAS_PREDICATE, "mom") in triples

    async def test_duplicate_alias_handling(
        self, db_pool: DatabasePool, message_repo: PostgresMessageRepository
    ) -> None:
        """Test that duplicate alias triples are ignored."""
        unique_suffix = str(uuid4())[:8]
        batch_id = f"test_batch_dup_{unique_suffix}"

        # First verify the batch doesn't exist
        async with db_pool.pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT COUNT(*) FROM semantic_memories WHERE source_batch_id = $1",
                batch_id,
            )
            assert existing == 0, f"Pre-existing data found: {existing}"

        message = episodic.EpisodicMessage(
            content="Test duplicate handling",
            discord_message_id=12352,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Store same alias twice
        memories = [
            {"subject": "bf", "predicate": ALIAS_PREDICATE, "object": "boyfriend"},
            {"subject": "bf", "predicate": ALIAS_PREDICATE, "object": "boyfriend"},
        ]
        result_count = await episodic.store_semantic_memories(
            message_repo, message_id, memories, source_batch_id=batch_id
        )

        # Should return 2:
        # - First forward (bf->boyfriend) inserts
        # - First reverse (boyfriend->bf) inserts
        # - Second forward is duplicate, returns 0
        # - Second reverse is duplicate (from first reverse), returns 0
        assert result_count == 2, f"Expected 2, got {result_count}"

        # Verify what was actually stored
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT subject, predicate, object FROM semantic_memories WHERE source_batch_id = $1 ORDER BY subject",
                batch_id,
            )
            assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
            assert rows[0]["subject"] == "bf"
            assert rows[0]["predicate"] == ALIAS_PREDICATE
            assert rows[0]["object"] == "boyfriend"
            assert rows[1]["subject"] == "boyfriend"
            assert rows[1]["predicate"] == ALIAS_PREDICATE
            assert rows[1]["object"] == "bf"

        # Verify the count of unique triples stored (before cleanup)
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

        # Clean up test data manually
        async with db_pool.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM semantic_memories WHERE source_batch_id = $1",
                batch_id,
            )

    async def test_self_alias_skipped(
        self, db_pool: DatabasePool, message_repo: PostgresMessageRepository
    ) -> None:
        """Test that self-aliases (entity aliased to itself) don't create duplicate entries."""
        unique_suffix = str(uuid4())[:8]
        batch_id = f"test_batch_self_{unique_suffix}"

        message = episodic.EpisodicMessage(
            content="Testing self-alias",
            discord_message_id=12353,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Try to create self-alias
        memories = [
            {"subject": "Alice", "predicate": ALIAS_PREDICATE, "object": "Alice"}
        ]
        result_count = await episodic.store_semantic_memories(
            message_repo, message_id, memories, source_batch_id=batch_id
        )

        # Should store 1 (forward) but reverse would be duplicate, so total = 1
        assert result_count == 1, f"Expected 1, got {result_count}"

        # Verify only one entry exists
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT subject, object FROM semantic_memories WHERE source_batch_id = $1",
                batch_id,
            )
            assert len(rows) == 1
            assert rows[0]["subject"] == "Alice"
            assert rows[0]["object"] == "Alice"

    async def test_circular_aliases(
        self, db_pool: DatabasePool, message_repo: PostgresMessageRepository
    ) -> None:
        """Test that circular aliases (A->B, B->C, C->A) are handled gracefully."""
        unique_suffix = str(uuid4())[:8]
        batch_id = f"test_batch_circular_{unique_suffix}"

        message = episodic.EpisodicMessage(
            content="Testing circular aliases",
            discord_message_id=12354,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Create circular chain: mom -> Mother -> mama -> mom
        memories = [
            {"subject": "mom", "predicate": ALIAS_PREDICATE, "object": "Mother"},
            {"subject": "Mother", "predicate": ALIAS_PREDICATE, "object": "mama"},
            {"subject": "mama", "predicate": ALIAS_PREDICATE, "object": "mom"},
        ]
        result_count = await episodic.store_semantic_memories(
            message_repo, message_id, memories, source_batch_id=batch_id
        )

        # Each forward alias creates a reverse, so 3 forward + 3 reverse = 6 total
        # Unless some are duplicates
        assert result_count == 6, f"Expected 6, got {result_count}"

        # Verify all bidirectional relationships exist
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT subject, object FROM semantic_memories 
                WHERE source_batch_id = $1
                ORDER BY subject, object
                """,
                batch_id,
            )
            triples = {(r["subject"], r["object"]) for r in rows}

            # Check all forward and reverse exist
            assert ("mom", "Mother") in triples
            assert ("Mother", "mom") in triples
            assert ("Mother", "mama") in triples
            assert ("mama", "Mother") in triples
            assert ("mama", "mom") in triples
            assert ("mom", "mama") in triples


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
