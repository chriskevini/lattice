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

    def test_protocol_importable(self) -> None:
        """Verify MessageRepository protocol can be imported."""
        from lattice.memory.repositories import MessageRepository

        assert MessageRepository is not None

    def test_protocol_has_store_message_method(self) -> None:
        """Verify MessageRepository has store_message in protocol."""
        from lattice.memory.repositories import MessageRepository

        assert hasattr(MessageRepository, "__protocol_attrs__")
        attrs = MessageRepository.__protocol_attrs__
        assert "store_message" in attrs

    def test_protocol_has_get_recent_messages_method(self) -> None:
        """Verify MessageRepository has get_recent_messages in protocol."""
        from lattice.memory.repositories import MessageRepository

        attrs = MessageRepository.__protocol_attrs__
        assert "get_recent_messages" in attrs

    def test_protocol_has_store_semantic_memories_method(self) -> None:
        """Verify MessageRepository has store_semantic_memories in protocol."""
        from lattice.memory.repositories import MessageRepository

        attrs = MessageRepository.__protocol_attrs__
        assert "store_semantic_memories" in attrs


class TestSemanticMemoryRepositoryProtocol:
    """Tests for SemanticMemoryRepository protocol compliance."""

    def test_protocol_importable(self) -> None:
        """Verify SemanticMemoryRepository protocol can be imported."""
        from lattice.memory.repositories import SemanticMemoryRepository

        assert SemanticMemoryRepository is not None

    def test_protocol_has_find_memories_method(self) -> None:
        """Verify SemanticMemoryRepository has find_memories in protocol."""
        from lattice.memory.repositories import SemanticMemoryRepository

        attrs = SemanticMemoryRepository.__protocol_attrs__
        assert "find_memories" in attrs

    def test_protocol_has_traverse_from_entity_method(self) -> None:
        """Verify SemanticMemoryRepository has traverse_from_entity in protocol."""
        from lattice.memory.repositories import SemanticMemoryRepository

        attrs = SemanticMemoryRepository.__protocol_attrs__
        assert "traverse_from_entity" in attrs


class TestCanonicalRepositoryProtocol:
    """Tests for CanonicalRepository protocol compliance."""

    def test_protocol_importable(self) -> None:
        """Verify CanonicalRepository protocol can be imported."""
        from lattice.memory.repositories import CanonicalRepository

        assert CanonicalRepository is not None

    def test_protocol_has_required_methods(self) -> None:
        """Verify CanonicalRepository has all required methods."""
        from lattice.memory.repositories import CanonicalRepository

        required_methods = {
            "get_entities_list",
            "get_predicates_list",
            "get_entities_set",
            "get_predicates_set",
            "store_entities",
            "store_predicates",
            "entity_exists",
            "predicate_exists",
        }
        attrs = CanonicalRepository.__protocol_attrs__
        for method in required_methods:
            assert method in attrs, f"Missing method: {method}"


class TestRepositoryImplementations:
    """Tests for concrete repository implementations."""

    @pytest.mark.asyncio
    async def test_message_repository_implementation(self) -> None:
        """Verify MessageRepository can be implemented."""
        from lattice.memory.repositories import MessageRepository

        class ConcreteMessageRepository:
            def __init__(self, db_pool: Any) -> None:
                self._db_pool = db_pool

            async def store_message(
                self,
                content: str,
                discord_message_id: int,
                channel_id: int,
                is_bot: bool,
                is_proactive: bool = False,
                generation_metadata: dict[str, Any] | None = None,
                user_timezone: str | None = None,
            ) -> UUID:
                return UUID("test-uuid")

            async def get_recent_messages(
                self,
                channel_id: int | None = None,
                limit: int = 10,
            ) -> list[dict[str, Any]]:
                return []

            async def store_semantic_memories(
                self,
                message_id: UUID,
                memories: list[dict[str, str]],
                source_batch_id: str | None = None,
            ) -> int:
                return len(memories)

        repo = ConcreteMessageRepository(MagicMock())

        assert isinstance(repo, MessageRepository)

        result = await repo.store_message(
            content="Hello",
            discord_message_id=123,
            channel_id=456,
            is_bot=False,
        )
        assert isinstance(result, UUID)

    @pytest.mark.asyncio
    async def test_canonical_repository_implementation(self) -> None:
        """Verify CanonicalRepository can be implemented."""
        from lattice.memory.repositories import CanonicalRepository

        class ConcreteCanonicalRepository:
            def __init__(self, db_pool: Any) -> None:
                self._db_pool = db_pool

            async def get_entities_list(self) -> list[str]:
                return ["User", "Project"]

            async def get_predicates_list(self) -> list[str]:
                return ["likes", "works on"]

            async def get_entities_set(self) -> set[str]:
                return {"User", "Project"}

            async def get_predicates_set(self) -> set[str]:
                return {"likes", "works on"}

            async def store_entities(self, names: list[str]) -> int:
                return len(names)

            async def store_predicates(self, names: list[str]) -> int:
                return len(names)

            async def entity_exists(self, name: str) -> bool:
                return name in {"User", "Project"}

            async def predicate_exists(self, name: str) -> bool:
                return name in {"likes", "works on"}

        repo = ConcreteCanonicalRepository(MagicMock())

        assert isinstance(repo, CanonicalRepository)

        entities = await repo.get_entities_list()
        assert entities == ["User", "Project"]

        exists = await repo.entity_exists("User")
        assert exists is True


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

    def test_subclass_must_implement_abstract_methods(self) -> None:
        """Verify PostgresRepository subclasses must implement abstract methods."""
        from lattice.memory.repositories import PostgresRepository

        with pytest.raises(TypeError):

            class IncompleteRepository(PostgresRepository):
                pass

    def test_subclass_with_all_abstract_methods_works(self) -> None:
        """Verify PostgresRepository with all abstract methods can be instantiated."""
        from lattice.memory.repositories import PostgresRepository

        class CompleteRepository(PostgresRepository):
            async def get_entities_list(self) -> list[str]:
                return []

            async def get_predicates_list(self) -> list[str]:
                return []

            async def get_entities_set(self) -> set[str]:
                return set()

            async def get_predicates_set(self) -> set[str]:
                return set()

            async def store_entities(self, names: list[str]) -> int:
                return 0

            async def store_predicates(self, names: list[str]) -> int:
                return 0

            async def entity_exists(self, name: str) -> bool:
                return False

            async def predicate_exists(self, name: str) -> bool:
                return False

        repo = CompleteRepository(MagicMock())
        assert repo._db_pool is not None
