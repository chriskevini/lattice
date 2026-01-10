"""Integration tests for semantic memory storage.

These tests require a real database connection and are skipped in CI.
To run locally, set DATABASE_URL environment variable.

Note: After issue #131, semantic memories use text-based storage
(subject/predicate/object as TEXT) rather than entity IDs.
The entities table is no longer used for semantic memory storage.
"""

import os

import pytest

from lattice.memory import episodic
from lattice.utils.database import db_pool


# Skip all tests if DATABASE_URL not set (CI environment)
pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set - requires real database for integration tests",
)


@pytest.mark.integration
class TestSemanticMemoryStorage:
    """Test semantic memory storage functionality."""

    @pytest.mark.asyncio
    async def test_store_simple_memory(self) -> None:
        """Test storing a simple semantic memory."""
        if not db_pool.is_initialized():
            await db_pool.initialize()

        # Create a test message
        message = episodic.EpisodicMessage(
            content="Alice works at OpenAI",
            discord_message_id=12345,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message)

        # Store semantic memory
        memories = [{"subject": "user", "predicate": "works_at", "object": "OpenAI"}]
        await episodic.store_semantic_memories(message_id, memories)

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

    @pytest.mark.asyncio
    async def test_store_multiple_memories(self) -> None:
        """Test storing multiple memories from same message."""
        if not db_pool.is_initialized():
            await db_pool.initialize()

        # Create test message
        message = episodic.EpisodicMessage(
            content="Alice works at OpenAI and likes Python",
            discord_message_id=12346,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message)

        # Store multiple memories with a batch ID for tracking
        batch_id = "test_batch_12346"
        memories = [
            {"subject": "user", "predicate": "works_at", "object": "OpenAI"},
            {"subject": "user", "predicate": "likes", "object": "Python"},
        ]
        await episodic.store_semantic_memories(
            message_id, memories, source_batch_id=batch_id
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

    @pytest.mark.asyncio
    async def test_skip_invalid_memories(self) -> None:
        """Test that invalid memories are skipped gracefully."""
        if not db_pool.is_initialized():
            await db_pool.initialize()

        message = episodic.EpisodicMessage(
            content="Test message",
            discord_message_id=12348,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message)

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
            message_id, memories, source_batch_id=batch_id
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

    @pytest.mark.asyncio
    async def test_empty_memories_list(self) -> None:
        """Test that empty memories list is handled gracefully."""
        if not db_pool.is_initialized():
            await db_pool.initialize()

        message = episodic.EpisodicMessage(
            content="No memories here",
            discord_message_id=12349,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message)

        # Should not raise exception
        batch_id = "test_batch_12349"
        await episodic.store_semantic_memories(message_id, [], source_batch_id=batch_id)

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

    @pytest.mark.asyncio
    async def test_text_based_memory_search(self) -> None:
        """Test searching for memories by text matching."""
        if not db_pool.is_initialized():
            await db_pool.initialize()

        message = episodic.EpisodicMessage(
            content="Test memory search",
            discord_message_id=12350,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message)

        # Store a memory with unique identifiers
        batch_id = "test_batch_12350"
        memories = [
            {"subject": "user", "predicate": "lives_in", "object": "Richmond, BC"}
        ]
        await episodic.store_semantic_memories(
            message_id, memories, source_batch_id=batch_id
        )

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
