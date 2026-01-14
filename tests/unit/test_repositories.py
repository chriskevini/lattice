"""Tests for repository protocols and base classes.

Validates that repository protocols are correctly defined and that
implementations can satisfy the protocol interfaces.
"""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest


class TestMessageRepositoryProtocol:
    """Tests for MessageRepository protocol compliance."""

    def test_protocol_importable(self) -> None:
        """Verify MessageRepository protocol can be imported."""
        from lattice.memory.repositories import MessageRepository

        assert MessageRepository is not None

    def test_protocol_has_required_methods(self) -> None:
        """Verify MessageRepository has all required methods."""
        from lattice.memory.repositories import MessageRepository

        required = {"store_message", "get_recent_messages", "store_semantic_memories"}
        for method in required:
            assert hasattr(MessageRepository, method), f"Missing method: {method}"


class TestSemanticMemoryRepositoryProtocol:
    """Tests for SemanticMemoryRepository protocol compliance."""

    def test_protocol_importable(self) -> None:
        """Verify SemanticMemoryRepository protocol can be imported."""
        from lattice.memory.repositories import SemanticMemoryRepository

        assert SemanticMemoryRepository is not None

    def test_protocol_has_required_methods(self) -> None:
        """Verify SemanticMemoryRepository has all required methods."""
        from lattice.memory.repositories import SemanticMemoryRepository

        required = {"find_memories", "traverse_from_entity"}
        for method in required:
            assert hasattr(SemanticMemoryRepository, method), (
                f"Missing method: {method}"
            )


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
        for method in required_methods:
            assert hasattr(CanonicalRepository, method), f"Missing method: {method}"


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
                return UUID("12345678-1234-1234-1234-123456789abc")

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

            async def get_messages_since_cursor(
                self,
                cursor_message_id: int,
                limit: int = 18,
            ) -> list[dict[str, Any]]:
                return []

            async def get_message_timestamps_since(
                self,
                since: datetime,
                is_bot: bool = False,
            ) -> list[dict[str, Any]]:
                return []

        repo = ConcreteMessageRepository(MagicMock())

        assert isinstance(repo, MessageRepository)

        result = await repo.store_message(
            content="Hello",
            discord_message_id=123,
            channel_id=456,
            is_bot=False,
        )
        assert isinstance(result, UUID)
        assert str(result) == "12345678-1234-1234-1234-123456789abc"

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

        mock_pool = MagicMock()

        repo = ConcreteRepository(mock_pool)

        assert repo._db_pool is mock_pool

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


class TestPromptAuditRepositoryProtocol:
    """Tests for PromptAuditRepository protocol compliance."""

    def test_protocol_importable(self) -> None:
        """Verify PromptAuditRepository protocol can be imported."""
        from lattice.memory.repositories import PromptAuditRepository

        assert PromptAuditRepository is not None

    def test_protocol_has_required_methods(self) -> None:
        """Verify PromptAuditRepository has all required methods."""
        from lattice.memory.repositories import PromptAuditRepository

        required_methods = {
            "store_audit",
            "update_dream_message",
            "link_feedback",
            "link_feedback_by_id",
            "get_by_dream_message",
            "get_with_feedback",
            "analyze_prompt_effectiveness",
            "get_feedback_samples",
            "get_feedback_with_context",
        }
        for method in required_methods:
            assert hasattr(PromptAuditRepository, method), f"Missing method: {method}"

    @pytest.mark.asyncio
    async def test_repository_implementation_with_new_methods(self) -> None:
        """Verify PromptAuditRepository can be implemented with new analytical methods."""
        from lattice.memory.repositories import PromptAuditRepository

        class ConcretePromptAuditRepository:
            def __init__(self, db_pool: Any) -> None:
                self._db_pool = db_pool

            async def store_audit(
                self,
                prompt_key: str,
                response_content: str,
                main_discord_message_id: int | None,
                rendered_prompt: str | None = None,
                template_version: int | None = None,
                message_id: UUID | None = None,
                model: str | None = None,
                provider: str | None = None,
                prompt_tokens: int | None = None,
                completion_tokens: int | None = None,
                cost_usd: float | None = None,
                latency_ms: int | None = None,
                finish_reason: str | None = None,
                cache_discount_usd: float | None = None,
                native_tokens_cached: int | None = None,
                native_tokens_reasoning: int | None = None,
                upstream_id: str | None = None,
                cancelled: bool | None = None,
                moderation_latency_ms: int | None = None,
                execution_metadata: dict[str, Any] | None = None,
                archetype_matched: str | None = None,
                archetype_confidence: float | None = None,
                reasoning: dict[str, Any] | None = None,
                dream_discord_message_id: int | None = None,
            ) -> UUID:
                return UUID("12345678-1234-1234-1234-123456789abc")

            async def update_dream_message(
                self, audit_id: UUID, dream_discord_message_id: int
            ) -> bool:
                return True

            async def link_feedback(
                self, dream_discord_message_id: int, feedback_id: UUID
            ) -> bool:
                return True

            async def link_feedback_by_id(
                self, audit_id: UUID, feedback_id: UUID
            ) -> bool:
                return True

            async def get_by_dream_message(
                self, dream_discord_message_id: int
            ) -> dict[str, Any] | None:
                return None

            async def get_with_feedback(
                self, limit: int = 100, offset: int = 0
            ) -> list[dict[str, Any]]:
                return []

            async def analyze_prompt_effectiveness(
                self,
                min_uses: int = 10,
                lookback_days: int = 30,
                min_feedback: int = 10,
            ) -> list[dict[str, Any]]:
                return []

            async def get_feedback_samples(
                self,
                prompt_key: str,
                limit: int = 10,
                sentiment_filter: str | None = None,
            ) -> list[str]:
                return []

            async def get_feedback_with_context(
                self,
                prompt_key: str,
                limit: int = 10,
                include_rendered_prompt: bool = True,
                max_prompt_chars: int = 5000,
            ) -> list[dict[str, Any]]:
                return []

        repo = ConcretePromptAuditRepository(MagicMock())

        assert isinstance(repo, PromptAuditRepository)

        result = await repo.analyze_prompt_effectiveness()
        assert isinstance(result, list)

        samples = await repo.get_feedback_samples("test_prompt")
        assert isinstance(samples, list)

        context = await repo.get_feedback_with_context("test_prompt")
        assert isinstance(context, list)
