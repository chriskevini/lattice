"""Integration tests for semantic memory storage.

These tests require a real database connection and are skipped in CI.
To run locally, set DATABASE_URL environment variable.

Note: After issue #131, semantic memories use text-based storage
(subject/predicate/object as TEXT) rather than entity IDs.
The entities table is no longer used for semantic memory storage.
"""

from typing import Protocol

import pytest

from lattice.core.constants import ALIAS_PREDICATE
from lattice.memory import episodic
from lattice.memory.context import PostgresMessageRepository
from lattice.utils.database import DatabasePool


class UniqueIdGenerator(Protocol):
    """Protocol for unique ID generators from conftest."""

    def next_id(self) -> int:
        """Generate the next unique ID."""
        ...


@pytest.fixture
def message_repo(db_pool: DatabasePool) -> PostgresMessageRepository:
    """Fixture to provide a PostgresMessageRepository."""
    return PostgresMessageRepository(db_pool)


@pytest.mark.asyncio
class TestSemanticMemoryStorage:
    """Test semantic memory storage functionality."""

    async def test_store_simple_memory(
        self,
        db_pool: DatabasePool,
        message_repo: PostgresMessageRepository,
        unique_discord_id: UniqueIdGenerator,
        unique_batch_id: str,
    ) -> None:
        """Test storing a simple semantic memory."""
        # Create a test message with unique ID
        msg_id = unique_discord_id.next_id()
        message = episodic.EpisodicMessage(
            content="Alice works at OpenAI",
            discord_message_id=msg_id,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Store semantic memory with unique batch_id and unique values to avoid conflicts
        batch_id = f"{unique_batch_id}_simple"
        # Use unique object to avoid conflict with existing data
        unique_object = f"TestCompany_{unique_batch_id[:8]}"
        memories = [
            {"subject": "TestUser", "predicate": "works_at", "object": unique_object}
        ]
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
                    row["subject"] == "TestUser"
                    and row["predicate"] == "works_at"
                    and row["object"] == unique_object
                ):
                    found = True
                    break
            assert found, "Memory not found in database"

    async def test_store_multiple_memories(
        self,
        db_pool: DatabasePool,
        message_repo: PostgresMessageRepository,
        unique_discord_id: UniqueIdGenerator,
        unique_batch_id: str,
    ) -> None:
        """Test storing multiple memories from same message."""
        # Create test message with unique ID
        msg_id = unique_discord_id.next_id()
        message = episodic.EpisodicMessage(
            content="Alice works at OpenAI and likes Python",
            discord_message_id=msg_id,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Store multiple memories with unique batch ID and unique values
        batch_id = f"{unique_batch_id}_multiple"
        unique_suffix = unique_batch_id[:8]
        memories = [
            {
                "subject": f"TestUser_{unique_suffix}",
                "predicate": "works_at",
                "object": f"TestCompany_{unique_suffix}",
            },
            {
                "subject": f"TestUser_{unique_suffix}",
                "predicate": "likes",
                "object": f"TestLang_{unique_suffix}",
            },
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
        self,
        db_pool: DatabasePool,
        message_repo: PostgresMessageRepository,
        unique_discord_id: UniqueIdGenerator,
        unique_batch_id: str,
    ) -> None:
        """Test that invalid memories are skipped gracefully."""
        msg_id = unique_discord_id.next_id()
        message = episodic.EpisodicMessage(
            content="Test message",
            discord_message_id=msg_id,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Mix of valid and invalid memories - use unique values
        batch_id = f"{unique_batch_id}_invalid"
        unique_suffix = unique_batch_id[:8]
        memories = [
            {
                "subject": f"TestUser_{unique_suffix}",
                "predicate": "likes",
                "object": f"Coffee_{unique_suffix}",
            },  # Valid
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
            assert rows[0]["subject"] == f"TestUser_{unique_suffix}"
            assert rows[0]["predicate"] == "likes"
            assert rows[0]["object"] == f"Coffee_{unique_suffix}"

    async def test_empty_memories_list(
        self,
        db_pool: DatabasePool,
        message_repo: PostgresMessageRepository,
        unique_discord_id: UniqueIdGenerator,
        unique_batch_id: str,
    ) -> None:
        """Test that empty memories list is handled gracefully."""
        msg_id = unique_discord_id.next_id()
        message = episodic.EpisodicMessage(
            content="No memories here",
            discord_message_id=msg_id,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Should not raise exception
        batch_id = f"{unique_batch_id}_empty"
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
        self,
        db_pool: DatabasePool,
        message_repo: PostgresMessageRepository,
        unique_discord_id: UniqueIdGenerator,
        unique_batch_id: str,
    ) -> None:
        """Test searching for memories by text matching."""
        batch_id = f"{unique_batch_id}_search"
        msg_id = unique_discord_id.next_id()

        message = episodic.EpisodicMessage(
            content="Test memory search",
            discord_message_id=msg_id,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Store a memory with unique identifiers
        unique_suffix = unique_batch_id[:8]
        unique_subject = f"SearchUser_{unique_suffix}"
        unique_location = f"Richmond_{unique_suffix}"
        memories = [
            {
                "subject": unique_subject,
                "predicate": "lives_in",
                "object": unique_location,
            }
        ]
        await episodic.store_semantic_memories(
            message_repo, message_id, memories, source_batch_id=batch_id
        )

        # Search by subject
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT subject, predicate, object
                FROM semantic_memories
                WHERE subject ILIKE $1 AND source_batch_id = $2
                """,
                f"%SearchUser_{unique_suffix}%",
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
                f"%Richmond_{unique_suffix}%",
                batch_id,
            )
            assert len(rows) == 1
            assert rows[0]["subject"] == unique_subject

    async def test_bidirectional_alias_storage(
        self,
        db_pool: DatabasePool,
        message_repo: PostgresMessageRepository,
        unique_discord_id: UniqueIdGenerator,
        unique_batch_id: str,
    ) -> None:
        """Test that 'has alias' triples create bidirectional relationships."""
        batch_id = f"{unique_batch_id}_alias"
        msg_id = unique_discord_id.next_id()
        unique_suffix = unique_batch_id[:8]

        message = episodic.EpisodicMessage(
            content="Test alias storage",
            discord_message_id=msg_id,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Store alias triple with unique values
        alias_from = f"mom_{unique_suffix}"
        alias_to = f"Mother_{unique_suffix}"
        memories = [
            {"subject": alias_from, "predicate": ALIAS_PREDICATE, "object": alias_to}
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
            assert (alias_from, ALIAS_PREDICATE, alias_to) in triples
            assert (alias_to, ALIAS_PREDICATE, alias_from) in triples

    async def test_duplicate_alias_handling(
        self,
        db_pool: DatabasePool,
        message_repo: PostgresMessageRepository,
        unique_discord_id: UniqueIdGenerator,
        unique_batch_id: str,
    ) -> None:
        """Test that duplicate alias triples are ignored."""
        batch_id = f"{unique_batch_id}_dup"
        msg_id = unique_discord_id.next_id()
        unique_suffix = unique_batch_id[:8]

        # First verify the batch doesn't exist
        async with db_pool.pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT COUNT(*) FROM semantic_memories WHERE source_batch_id = $1",
                batch_id,
            )
            assert existing == 0, f"Pre-existing data found: {existing}"

        message = episodic.EpisodicMessage(
            content="Test duplicate handling",
            discord_message_id=msg_id,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Store same alias twice with unique values
        alias_from = f"bf_{unique_suffix}"
        alias_to = f"boyfriend_{unique_suffix}"
        memories = [
            {"subject": alias_from, "predicate": ALIAS_PREDICATE, "object": alias_to},
            {"subject": alias_from, "predicate": ALIAS_PREDICATE, "object": alias_to},
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
            assert rows[0]["subject"] == alias_from
            assert rows[0]["predicate"] == ALIAS_PREDICATE
            assert rows[0]["object"] == alias_to
            assert rows[1]["subject"] == alias_to
            assert rows[1]["predicate"] == ALIAS_PREDICATE
            assert rows[1]["object"] == alias_from

        # Verify the count of unique triples stored
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

    async def test_self_alias_skipped(
        self,
        db_pool: DatabasePool,
        message_repo: PostgresMessageRepository,
        unique_discord_id: UniqueIdGenerator,
        unique_batch_id: str,
    ) -> None:
        """Test that self-aliases (entity aliased to itself) don't create duplicate entries."""
        batch_id = f"{unique_batch_id}_self"
        msg_id = unique_discord_id.next_id()
        unique_suffix = unique_batch_id[:8]

        message = episodic.EpisodicMessage(
            content="Testing self-alias",
            discord_message_id=msg_id,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Try to create self-alias with unique value
        self_alias = f"Alice_{unique_suffix}"
        memories = [
            {"subject": self_alias, "predicate": ALIAS_PREDICATE, "object": self_alias}
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
            assert rows[0]["subject"] == self_alias
            assert rows[0]["object"] == self_alias

    async def test_circular_aliases(
        self,
        db_pool: DatabasePool,
        message_repo: PostgresMessageRepository,
        unique_discord_id: UniqueIdGenerator,
        unique_batch_id: str,
    ) -> None:
        """Test that circular aliases (A->B, B->C, C->A) are handled gracefully."""
        batch_id = f"{unique_batch_id}_circular"
        msg_id = unique_discord_id.next_id()
        unique_suffix = unique_batch_id[:8]

        message = episodic.EpisodicMessage(
            content="Testing circular aliases",
            discord_message_id=msg_id,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message_repo, message)

        # Create circular chain with unique values: mom -> Mother -> mama -> mom
        mom = f"mom_{unique_suffix}"
        mother = f"Mother_{unique_suffix}"
        mama = f"mama_{unique_suffix}"
        memories = [
            {"subject": mom, "predicate": ALIAS_PREDICATE, "object": mother},
            {"subject": mother, "predicate": ALIAS_PREDICATE, "object": mama},
            {"subject": mama, "predicate": ALIAS_PREDICATE, "object": mom},
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
            assert (mom, mother) in triples
            assert (mother, mom) in triples
            assert (mother, mama) in triples
            assert (mama, mother) in triples
            assert (mama, mom) in triples
            assert (mom, mama) in triples


if __name__ == "__main__":
    import pytest as pytest_main

    pytest_main.main([__file__, "-v"])
