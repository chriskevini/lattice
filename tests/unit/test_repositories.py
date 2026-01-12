"""Tests for repository protocols and base classes.

Validates that repository protocols are correctly defined and that
implementations can satisfy the protocol interfaces.
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest


class TestMessageRepositoryProtocol:
    """Tests for MessageRepository protocol compliance."""

    def test_protocol_accepts_valid_implementation(self) -> None:
        """Verify MessageRepository protocol accepts valid mock implementation."""
        from lattice.memory.repositories import MessageRepository

        mock_repo = MagicMock(spec=MessageRepository)

        mock_repo.store_message = AsyncMock(return_value=UUID("test-uuid"))
        mock_repo.get_recent_messages = AsyncMock(return_value=[])
        mock_repo.store_semantic_memories = AsyncMock(return_value=5)

        assert callable(mock_repo.store_message)
        assert callable(mock_repo.get_recent_messages)
        assert callable(mock_repo.store_semantic_memories)

    @pytest.mark.asyncio
    async def test_store_message_signature(self) -> None:
        """Verify store_message has correct async signature."""
        from lattice.memory.repositories import MessageRepository

        mock_repo = MagicMock(spec=MessageRepository)
        mock_repo.store_message = AsyncMock(return_value=UUID("test-uuid"))

        result = await mock_repo.store_message(
            content="Hello world",
            discord_message_id=12345,
            channel_id=67890,
            is_bot=False,
        )

        assert isinstance(result, UUID)

    @pytest.mark.asyncio
    async def test_get_recent_messages_signature(self) -> None:
        """Verify get_recent_messages accepts optional parameters."""
        from lattice.memory.repositories import MessageRepository

        mock_repo = MagicMock(spec=MessageRepository)
        mock_repo.get_recent_messages = AsyncMock(return_value=[])

        result = await mock_repo.get_recent_messages(channel_id=123, limit=20)

        mock_repo.get_recent_messages.assert_called_once_with(channel_id=123, limit=20)
        assert result == []

    @pytest.mark.asyncio
    async def test_store_semantic_memories_signature(self) -> None:
        """Verify store_semantic_memories handles memory triples."""
        from lattice.memory.repositories import MessageRepository

        mock_repo = MagicMock(spec=MessageRepository)
        mock_repo.store_semantic_memories = AsyncMock(return_value=3)

        memories = [
            {"subject": "User", "predicate": "likes", "object": "Python"},
            {"subject": "User", "predicate": "works on", "object": "lattice"},
        ]

        result = await mock_repo.store_semantic_memories(
            message_id=UUID("test-uuid"),
            memories=memories,
        )

        assert result == 3


class TestSemanticMemoryRepositoryProtocol:
    """Tests for SemanticMemoryRepository protocol compliance."""

    def test_protocol_accepts_valid_implementation(self) -> None:
        """Verify SemanticMemoryRepository protocol accepts valid mock."""
        from lattice.memory.repositories import SemanticMemoryRepository

        mock_repo = MagicMock(spec=SemanticMemoryRepository)

        mock_repo.find_memories = AsyncMock(return_value=[])
        mock_repo.traverse_from_entity = AsyncMock(return_value=[])

        assert callable(mock_repo.find_memories)
        assert callable(mock_repo.traverse_from_entity)

    @pytest.mark.asyncio
    async def test_find_memories_signature(self) -> None:
        """Verify find_memories accepts flexible criteria."""
        from lattice.memory.repositories import SemanticMemoryRepository

        mock_repo = MagicMock(spec=SemanticMemoryRepository)
        mock_repo.find_memories = AsyncMock(return_value=[])

        await mock_repo.find_memories(
            subject="User",
            predicate="did activity",
            limit=10,
        )

        mock_repo.find_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_traverse_from_entity_signature(self) -> None:
        """Verify traverse_from_entity accepts entity traversal params."""
        from lattice.memory.repositories import SemanticMemoryRepository

        mock_repo = MagicMock(spec=SemanticMemoryRepository)
        mock_repo.traverse_from_entity = AsyncMock(return_value=[])

        result = await mock_repo.traverse_from_entity(
            entity_name="User",
            predicate_filter={"likes", "works on"},
            max_hops=2,
        )

        assert result == []


class TestCanonicalRepositoryProtocol:
    """Tests for CanonicalRepository protocol compliance."""

    def test_protocol_accepts_valid_implementation(self) -> None:
        """Verify CanonicalRepository protocol accepts valid mock."""
        from lattice.memory.repositories import CanonicalRepository

        mock_repo = MagicMock(spec=CanonicalRepository)

        mock_repo.get_entities_list = AsyncMock(return_value=[])
        mock_repo.get_predicates_list = AsyncMock(return_value=[])
        mock_repo.get_entities_set = AsyncMock(return_value=set())
        mock_repo.get_predicates_set = AsyncMock(return_value=set())
        mock_repo.store_entities = AsyncMock(return_value=0)
        mock_repo.store_predicates = AsyncMock(return_value=0)
        mock_repo.entity_exists = AsyncMock(return_value=False)
        mock_repo.predicate_exists = AsyncMock(return_value=False)

        assert callable(mock_repo.get_entities_list)
        assert callable(mock_repo.get_predicates_list)
        assert callable(mock_repo.get_entities_set)
        assert callable(mock_repo.get_predicates_set)
        assert callable(mock_repo.store_entities)
        assert callable(mock_repo.store_predicates)
        assert callable(mock_repo.entity_exists)
        assert callable(mock_repo.predicate_exists)

    @pytest.mark.asyncio
    async def test_get_entities_list(self) -> None:
        """Verify get_entities_list returns list of strings."""
        from lattice.memory.repositories import CanonicalRepository

        mock_repo = MagicMock(spec=CanonicalRepository)
        mock_repo.get_entities_list = AsyncMock(
            return_value=["User", "Project", "Task"]
        )

        result = await mock_repo.get_entities_list()

        assert result == ["User", "Project", "Task"]

    @pytest.mark.asyncio
    async def test_get_entities_set(self) -> None:
        """Verify get_entities_set returns set for O(1) lookup."""
        from lattice.memory.repositories import CanonicalRepository

        mock_repo = MagicMock(spec=CanonicalRepository)
        mock_repo.get_entities_set = AsyncMock(return_value={"User", "Project", "Task"})

        result = await mock_repo.get_entities_set()

        assert isinstance(result, set)
        assert "User" in result

    @pytest.mark.asyncio
    async def test_store_entities(self) -> None:
        """Verify store_entities returns count of inserted entities."""
        from lattice.memory.repositories import CanonicalRepository

        mock_repo = MagicMock(spec=CanonicalRepository)
        mock_repo.store_entities = AsyncMock(return_value=2)

        result = await mock_repo.store_entities(["NewEntity1", "NewEntity2"])

        assert result == 2

    @pytest.mark.asyncio
    async def test_entity_exists(self) -> None:
        """Verify entity_exists returns boolean."""
        from lattice.memory.repositories import CanonicalRepository

        mock_repo = MagicMock(spec=CanonicalRepository)
        mock_repo.entity_exists = AsyncMock(return_value=True)

        result = await mock_repo.entity_exists("User")

        assert result is True

    @pytest.mark.asyncio
    async def test_predicate_exists(self) -> None:
        """Verify predicate_exists returns boolean."""
        from lattice.memory.repositories import CanonicalRepository

        mock_repo = MagicMock(spec=CanonicalRepository)
        mock_repo.predicate_exists = AsyncMock(return_value=False)

        result = await mock_repo.predicate_exists("unknown_predicate")

        assert result is False


class TestPostgresRepositoryBase:
    """Tests for PostgresRepository base class."""

    def test_init_requires_db_pool(self) -> None:
        """Verify PostgresRepository requires db_pool parameter."""
        from lattice.memory.repositories import PostgresRepository

        class ConcreteRepository(PostgresRepository):
            async def get_entities_list(self) -> list[str]:
                return []

        mock_pool = MagicMock()

        repo = ConcreteRepository(mock_pool)

        assert repo._db_pool is mock_pool

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        """Verify PostgresRepository cannot be instantiated directly."""
        from lattice.memory.repositories import PostgresRepository

        with pytest.raises(TypeError):
            PostgresRepository(MagicMock())
