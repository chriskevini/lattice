"""Integration tests for entity and semantic triple storage."""

import pytest

from lattice.memory import episodic
from lattice.utils.database import db_pool


@pytest.mark.integration
class TestEntityStorage:
    """Test entity upsert functionality."""

    @pytest.mark.asyncio
    async def test_create_new_entity(self) -> None:
        """Test creating a new entity."""
        if not db_pool.is_initialized():
            await db_pool.initialize()

        entity_id = await episodic.upsert_entity(name="TestEntity", entity_type="test")

        assert entity_id is not None

        # Verify entity exists in database
        async with db_pool.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, name, entity_type FROM entities WHERE id = $1", entity_id
            )
            assert row is not None
            assert row["name"] == "TestEntity"
            assert row["entity_type"] == "test"

    @pytest.mark.asyncio
    async def test_upsert_existing_entity(self) -> None:
        """Test that upserting existing entity returns same ID."""
        if not db_pool.is_initialized():
            await db_pool.initialize()

        # Create entity first time
        entity_id_1 = await episodic.upsert_entity(name="DuplicateTest")

        # Create same entity again
        entity_id_2 = await episodic.upsert_entity(name="DuplicateTest")

        # Should return same ID
        assert entity_id_1 == entity_id_2

    @pytest.mark.asyncio
    async def test_entity_case_insensitive_matching(self) -> None:
        """Test that entity matching is case-insensitive."""
        if not db_pool.is_initialized():
            await db_pool.initialize()

        # Create with lowercase
        entity_id_1 = await episodic.upsert_entity(name="casetest")

        # Try with different casing
        entity_id_2 = await episodic.upsert_entity(name="CaseTest")
        entity_id_3 = await episodic.upsert_entity(name="CASETEST")

        # All should match
        assert entity_id_1 == entity_id_2 == entity_id_3


@pytest.mark.integration
class TestTripleStorage:
    """Test semantic triple storage functionality."""

    @pytest.mark.asyncio
    async def test_store_simple_triple(self) -> None:
        """Test storing a simple semantic triple."""
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

        # Store triple
        triples = [{"subject": "Alice", "predicate": "works_at", "object": "OpenAI"}]
        await episodic.store_semantic_triples(message_id, triples)

        # Verify triple was stored
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT s.name as subject, t.predicate, o.name as object, t.origin_id
                FROM semantic_triples t
                JOIN entities s ON t.subject_id = s.id
                JOIN entities o ON t.object_id = o.id
                WHERE t.origin_id = $1
                """,
                message_id,
            )

            assert len(rows) == 1
            assert rows[0]["subject"] == "Alice"
            assert rows[0]["predicate"] == "works_at"
            assert rows[0]["object"] == "OpenAI"
            assert rows[0]["origin_id"] == message_id

    @pytest.mark.asyncio
    async def test_store_multiple_triples(self) -> None:
        """Test storing multiple triples from same message."""
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

        # Store multiple triples
        triples = [
            {"subject": "Alice", "predicate": "works_at", "object": "OpenAI"},
            {"subject": "Alice", "predicate": "likes", "object": "Python"},
        ]
        await episodic.store_semantic_triples(message_id, triples)

        # Verify both triples stored
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT COUNT(*) as count
                FROM semantic_triples
                WHERE origin_id = $1
                """,
                message_id,
            )

            assert rows[0]["count"] >= 2  # May have more from previous tests

    @pytest.mark.asyncio
    async def test_store_triple_with_normalized_predicate(self) -> None:
        """Test that predicates are normalized correctly."""
        if not db_pool.is_initialized():
            await db_pool.initialize()

        message = episodic.EpisodicMessage(
            content="Bob loves pizza",
            discord_message_id=12347,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message)

        # Use synonym predicate
        triples = [{"subject": "Bob", "predicate": "loves", "object": "pizza"}]
        await episodic.store_semantic_triples(message_id, triples)

        # Verify predicate was normalized to "likes"
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT predicate
                FROM semantic_triples
                WHERE origin_id = $1
                """,
                message_id,
            )

            assert len(rows) == 1
            # Should be normalized to canonical form
            assert rows[0]["predicate"] == "likes"

    @pytest.mark.asyncio
    async def test_skip_invalid_triples(self) -> None:
        """Test that invalid triples are skipped gracefully."""
        if not db_pool.is_initialized():
            await db_pool.initialize()

        message = episodic.EpisodicMessage(
            content="Test message",
            discord_message_id=12348,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message)

        # Mix of valid and invalid triples
        triples = [
            {"subject": "Alice", "predicate": "works_at", "object": "OpenAI"},  # Valid
            {
                "subject": "",
                "predicate": "likes",
                "object": "Python",
            },  # Invalid - empty subject
            {
                "subject": "Bob",
                "predicate": "",
                "object": "Coffee",
            },  # Invalid - empty predicate
            {
                "subject": "Carol",
                "predicate": "knows",
                "object": "",
            },  # Invalid - empty object
        ]

        # Should not raise exception
        await episodic.store_semantic_triples(message_id, triples)

        # Verify only valid triple was stored
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT s.name as subject
                FROM semantic_triples t
                JOIN entities s ON t.subject_id = s.id
                WHERE t.origin_id = $1
                """,
                message_id,
            )

            # Should have stored only the valid triple
            assert len(rows) == 1
            assert rows[0]["subject"] == "Alice"

    @pytest.mark.asyncio
    async def test_empty_triples_list(self) -> None:
        """Test that empty triples list is handled gracefully."""
        if not db_pool.is_initialized():
            await db_pool.initialize()

        message = episodic.EpisodicMessage(
            content="No triples here",
            discord_message_id=12349,
            channel_id=67890,
            is_bot=False,
        )
        message_id = await episodic.store_message(message)

        # Should not raise exception
        await episodic.store_semantic_triples(message_id, [])

        # Verify no triples stored
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT COUNT(*) as count FROM semantic_triples WHERE origin_id = $1",
                message_id,
            )
            assert rows[0]["count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
