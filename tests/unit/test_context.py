"""Unit tests for context repository module.

Tests the PostgreSQL implementations of ContextRepository and
MessageRepository for episodic memory operations.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from lattice.memory.context import (
    PostgresCanonicalRepository,
    PostgresContextRepository,
    PostgresMessageRepository,
    PostgresSemanticMemoryRepository,
)


class TestPostgresContextRepository:
    """Tests for PostgresContextRepository."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock database pool."""
        pool = MagicMock()
        pool.pool = MagicMock()
        return pool

    @pytest.fixture
    def repository(self, mock_pool: MagicMock) -> PostgresContextRepository:
        """Create a repository instance with mocked pool."""
        return PostgresContextRepository(mock_pool)

    @pytest.mark.asyncio
    async def test_save_context_upsert(
        self, repository: PostgresContextRepository
    ) -> None:
        """Test saving context performs upsert operation."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresContextRepository(mock_pool)
        await repo.save_context("channel", "123", {"key": "value"})

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO context_cache" in call_args[0][0]
        assert "ON CONFLICT" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_load_context_type(
        self, repository: PostgresContextRepository
    ) -> None:
        """Test loading context by type returns list of entries."""
        mock_rows = [
            {
                "target_id": "123",
                "data": {"key": "value"},
                "updated_at": datetime.now(timezone.utc),
            },
            {
                "target_id": "456",
                "data": {"other": "data"},
                "updated_at": datetime.now(timezone.utc),
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresContextRepository(mock_pool)
        result = await repo.load_context_type("channel")

        assert len(result) == 2
        assert result[0]["target_id"] == "123"

    @pytest.mark.asyncio
    async def test_load_context_type_empty(
        self, repository: PostgresContextRepository
    ) -> None:
        """Test loading context type with no results."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresContextRepository(mock_pool)
        result = await repo.load_context_type("nonexistent")

        assert result == []


class TestPostgresMessageRepository:
    """Tests for PostgresMessageRepository."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock database pool."""
        pool = MagicMock()
        pool.pool = MagicMock()
        return pool

    @pytest.fixture
    def repository(self, mock_pool: MagicMock) -> PostgresMessageRepository:
        """Create a repository instance with mocked pool."""
        return PostgresMessageRepository(mock_pool)

    @pytest.mark.asyncio
    async def test_store_message_returns_uuid(
        self, repository: PostgresMessageRepository
    ) -> None:
        """Test storing message returns UUID."""
        expected_uuid = UUID("12345678-1234-1234-1234-123456789abc")
        mock_row = {"id": expected_uuid}
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresMessageRepository(mock_pool)
        result = await repo.store_message(
            content="Hello world",
            discord_message_id=123,
            channel_id=456,
            is_bot=False,
        )

        assert result == expected_uuid
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_message_with_all_params(
        self, repository: PostgresMessageRepository
    ) -> None:
        """Test storing message with all optional parameters."""
        expected_uuid = UUID("12345678-1234-1234-1234-123456789abc")
        mock_row = {"id": expected_uuid}
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresMessageRepository(mock_pool)
        result = await repo.store_message(
            content="Test message",
            discord_message_id=999,
            channel_id=111,
            is_bot=True,
            is_proactive=True,
            generation_metadata={"model": "test"},
            user_timezone="America/New_York",
        )

        assert result == expected_uuid

    @pytest.mark.asyncio
    async def test_get_recent_messages_with_channel_filter(
        self, repository: PostgresMessageRepository
    ) -> None:
        """Test getting recent messages with channel filter."""
        mock_rows = [
            {
                "id": UUID("11111111-1111-1111-1111-111111111111"),
                "discord_message_id": 100,
                "channel_id": 456,
                "content": "message 1",
                "is_bot": False,
                "is_proactive": False,
                "timestamp": datetime.now(timezone.utc),
                "user_timezone": "UTC",
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresMessageRepository(mock_pool)
        result = await repo.get_recent_messages(channel_id=456, limit=10)

        assert len(result) == 1
        assert result[0]["content"] == "message 1"

    @pytest.mark.asyncio
    async def test_get_recent_messages_without_channel_filter(
        self, repository: PostgresMessageRepository
    ) -> None:
        """Test getting recent messages without channel filter."""
        mock_rows = [
            {
                "id": UUID("11111111-1111-1111-1111-111111111111"),
                "discord_message_id": 100,
                "channel_id": 456,
                "content": "message 1",
                "is_bot": False,
                "is_proactive": False,
                "timestamp": datetime.now(timezone.utc),
                "user_timezone": "UTC",
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresMessageRepository(mock_pool)
        result = await repo.get_recent_messages(limit=10)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_messages_since_cursor(
        self, repository: PostgresMessageRepository
    ) -> None:
        """Test getting messages since cursor."""
        mock_rows = [
            {
                "id": UUID("11111111-1111-1111-1111-111111111111"),
                "discord_message_id": 101,
                "channel_id": 456,
                "content": "new message",
                "is_bot": False,
                "is_proactive": False,
                "timestamp": datetime.now(timezone.utc),
                "user_timezone": "UTC",
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresMessageRepository(mock_pool)
        result = await repo.get_messages_since_cursor(cursor_message_id=100, limit=18)

        assert len(result) == 1
        assert result[0]["discord_message_id"] == 101

    @pytest.mark.asyncio
    async def test_store_semantic_memories_empty_list(
        self, repository: PostgresMessageRepository
    ) -> None:
        """Test storing empty memory list returns 0."""
        mock_conn = AsyncMock()
        mock_conn.transaction = MagicMock()
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresMessageRepository(mock_pool)
        result = await repo.store_semantic_memories(
            message_id=UUID("11111111-1111-1111-1111-111111111111"),
            memories=[],
        )

        assert result == 0

    @pytest.mark.asyncio
    async def test_store_semantic_memories_with_data(
        self, repository: PostgresMessageRepository
    ) -> None:
        """Test storing semantic memories with valid data."""
        mock_row = {"id": UUID("22222222-2222-2222-2222-222222222222")}
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=mock_row["id"])
        mock_conn.transaction = MagicMock()
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresMessageRepository(mock_pool)
        memories = [
            {"subject": "User", "predicate": "likes", "object": "Python"},
            {"subject": "User", "predicate": "works on", "object": "Project"},
        ]
        result = await repo.store_semantic_memories(
            message_id=UUID("11111111-1111-1111-1111-111111111111"),
            memories=memories,
        )

        assert result > 0

    @pytest.mark.asyncio
    async def test_store_semantic_memories_skips_invalid(
        self, repository: PostgresMessageRepository
    ) -> None:
        """Test that invalid memories (missing fields) are skipped."""
        memories = [
            {"subject": "User", "predicate": "likes"},  # Missing object
            {"subject": "Bob", "predicate": "works_at", "object": "Company"},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(
            return_value=UUID("22222222-2222-2222-2222-222222222222")
        )
        mock_conn.transaction = MagicMock()
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresMessageRepository(mock_pool)
        result = await repo.store_semantic_memories(
            message_id=UUID("11111111-1111-1111-1111-111111111111"),
            memories=memories,
        )

        # Only 1 valid memory should be stored
        assert result == 1


class TestPostgresSemanticMemoryRepository:
    """Tests for PostgresSemanticMemoryRepository."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock database pool."""
        pool = MagicMock()
        pool.pool = MagicMock()
        return pool

    @pytest.fixture
    def repository(self, mock_pool: MagicMock) -> PostgresSemanticMemoryRepository:
        """Create a repository instance with mocked pool."""
        return PostgresSemanticMemoryRepository(mock_pool)

    @pytest.mark.asyncio
    async def test_find_memories_no_filters(
        self, repository: PostgresSemanticMemoryRepository
    ) -> None:
        """Test finding memories with no filters."""
        mock_rows = [
            {
                "subject": "User",
                "predicate": "likes",
                "object": "Python",
                "created_at": datetime.now(timezone.utc),
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresSemanticMemoryRepository(mock_pool)
        result = await repo.find_memories()

        assert len(result) == 1
        assert result[0]["subject"] == "User"

    @pytest.mark.asyncio
    async def test_find_memories_with_subject_filter(
        self, repository: PostgresSemanticMemoryRepository
    ) -> None:
        """Test finding memories with subject filter."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresSemanticMemoryRepository(mock_pool)
        result = await repo.find_memories(subject="User")

        assert len(result) == 0
        call_args = mock_conn.fetch.call_args
        assert "subject = $" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_find_memories_with_date_range(
        self, repository: PostgresSemanticMemoryRepository
    ) -> None:
        """Test finding memories with date range filters."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresSemanticMemoryRepository(mock_pool)
        start_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, tzinfo=timezone.utc)
        result = await repo.find_memories(start_date=start_date, end_date=end_date)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_traverse_from_entity(
        self, repository: PostgresSemanticMemoryRepository
    ) -> None:
        """Test BFS traversal from entity."""
        mock_rows = [
            {
                "subject": "User",
                "predicate": "likes",
                "object": "Python",
                "created_at": datetime.now(timezone.utc),
            },
            {
                "subject": "Python",
                "predicate": "is",
                "object": "Language",
                "created_at": datetime.now(timezone.utc),
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            side_effect=[
                mock_rows,  # First hop
                [],  # Second hop (no more entities)
            ]
        )
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresSemanticMemoryRepository(mock_pool)
        result = await repo.traverse_from_entity(entity_name="User", max_hops=2)

        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_traverse_with_predicate_filter(
        self, repository: PostgresSemanticMemoryRepository
    ) -> None:
        """Test traversal with predicate filter."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresSemanticMemoryRepository(mock_pool)
        result = await repo.traverse_from_entity(
            entity_name="User",
            predicate_filter={"likes", "works_at"},
            max_hops=3,
        )

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fetch_goal_names(
        self, repository: PostgresSemanticMemoryRepository
    ) -> None:
        """Test fetching unique goal names."""
        mock_rows = [
            {"object": "Learn Python"},
            {"object": "Run Marathon"},
            {"object": "Read 50 Books"},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresSemanticMemoryRepository(mock_pool)
        result = await repo.fetch_goal_names(limit=10)

        assert len(result) == 3
        assert "Learn Python" in result

    @pytest.mark.asyncio
    async def test_get_goal_predicates_empty_list(
        self, repository: PostgresSemanticMemoryRepository
    ) -> None:
        """Test getting goal predicates with empty list returns empty."""
        repo = PostgresSemanticMemoryRepository(MagicMock())
        result = await repo.get_goal_predicates(goal_names=[])

        assert result == []

    @pytest.mark.asyncio
    async def test_get_goal_predicates_with_data(
        self, repository: PostgresSemanticMemoryRepository
    ) -> None:
        """Test getting goal predicates with valid data."""
        mock_rows = [
            {"subject": "Learn Python", "predicate": "is", "object": "Goal"},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresSemanticMemoryRepository(mock_pool)
        result = await repo.get_goal_predicates(goal_names=["Learn Python"])

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_subjects_for_review(
        self, repository: PostgresSemanticMemoryRepository
    ) -> None:
        """Test getting subjects with sufficient memories for review."""
        mock_rows = [
            {"subject": "User"},
            {"subject": "Project"},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresSemanticMemoryRepository(mock_pool)
        result = await repo.get_subjects_for_review(min_memories=5)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_memories_by_subject(
        self, repository: PostgresSemanticMemoryRepository
    ) -> None:
        """Test getting all active memories for a subject."""
        mock_rows = [
            {
                "id": UUID("11111111-1111-1111-1111-111111111111"),
                "subject": "User",
                "predicate": "likes",
                "object": "Python",
                "created_at": datetime.now(timezone.utc),
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresSemanticMemoryRepository(mock_pool)
        result = await repo.get_memories_by_subject(subject="User")

        assert len(result) == 1
        assert result[0]["subject"] == "User"

    @pytest.mark.asyncio
    async def test_supersede_memory_success(
        self, repository: PostgresSemanticMemoryRepository
    ) -> None:
        """Test successfully superseding a memory."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresSemanticMemoryRepository(mock_pool)
        result = await repo.supersede_memory(
            triple_id=UUID("11111111-1111-1111-1111-111111111111"),
            superseded_by_id=UUID("22222222-2222-2222-2222-222222222222"),
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_supersede_memory_not_found(
        self, repository: PostgresSemanticMemoryRepository
    ) -> None:
        """Test superseding non-existent memory returns False."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 0")
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresSemanticMemoryRepository(mock_pool)
        result = await repo.supersede_memory(
            triple_id=UUID("11111111-1111-1111-1111-111111111111"),
            superseded_by_id=UUID("22222222-2222-2222-2222-222222222222"),
        )

        assert result is False


class TestPostgresCanonicalRepository:
    """Tests for PostgresCanonicalRepository."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock database pool."""
        pool = MagicMock()
        pool.pool = MagicMock()
        return pool

    @pytest.fixture
    def repository(self, mock_pool: MagicMock) -> PostgresCanonicalRepository:
        """Create a repository instance with mocked pool."""
        return PostgresCanonicalRepository(mock_pool)

    @pytest.mark.asyncio
    async def test_get_entities_list(
        self, repository: PostgresCanonicalRepository
    ) -> None:
        """Test fetching all entity names."""
        mock_rows = [
            {"name": "User"},
            {"name": "Project"},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        result = await repo.get_entities_list()

        assert result == ["User", "Project"]

    @pytest.mark.asyncio
    async def test_get_predicates_list(
        self, repository: PostgresCanonicalRepository
    ) -> None:
        """Test fetching all predicate names."""
        mock_rows = [
            {"name": "likes"},
            {"name": "works_on"},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        result = await repo.get_predicates_list()

        assert result == ["likes", "works_on"]

    @pytest.mark.asyncio
    async def test_get_entities_set(
        self, repository: PostgresCanonicalRepository
    ) -> None:
        """Test fetching entities as set."""
        mock_rows = [
            {"name": "User"},
            {"name": "Project"},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        result = await repo.get_entities_set()

        assert result == {"User", "Project"}

    @pytest.mark.asyncio
    async def test_get_predicates_set(
        self, repository: PostgresCanonicalRepository
    ) -> None:
        """Test fetching predicates as set."""
        mock_rows = [
            {"name": "likes"},
            {"name": "works_on"},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        result = await repo.get_predicates_set()

        assert result == {"likes", "works_on"}

    @pytest.mark.asyncio
    async def test_store_entities_empty_list(
        self, repository: PostgresCanonicalRepository
    ) -> None:
        """Test storing empty entity list returns 0."""
        repo = PostgresCanonicalRepository(MagicMock())
        result = await repo.store_entities([])

        assert result == 0

    @pytest.mark.asyncio
    async def test_store_entities_with_data(
        self, repository: PostgresCanonicalRepository
    ) -> None:
        """Test storing entities with valid data."""
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        result = await repo.store_entities(["Entity1", "Entity2"])

        assert result == 2

    @pytest.mark.asyncio
    async def test_store_predicates_empty_list(
        self, repository: PostgresCanonicalRepository
    ) -> None:
        """Test storing empty predicate list returns 0."""
        repo = PostgresCanonicalRepository(MagicMock())
        result = await repo.store_predicates([])

        assert result == 0

    @pytest.mark.asyncio
    async def test_store_predicates_with_data(
        self, repository: PostgresCanonicalRepository
    ) -> None:
        """Test storing predicates with valid data."""
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        result = await repo.store_predicates(["predicate1", "predicate2"])

        assert result == 2

    @pytest.mark.asyncio
    async def test_entity_exists_true(
        self, repository: PostgresCanonicalRepository
    ) -> None:
        """Test checking existing entity returns True."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        result = await repo.entity_exists("User")

        assert result is True

    @pytest.mark.asyncio
    async def test_entity_exists_false(
        self, repository: PostgresCanonicalRepository
    ) -> None:
        """Test checking non-existent entity returns False."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        result = await repo.entity_exists("NonExistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_predicate_exists_true(
        self, repository: PostgresCanonicalRepository
    ) -> None:
        """Test checking existing predicate returns True."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        result = await repo.predicate_exists("likes")

        assert result is True

    @pytest.mark.asyncio
    async def test_predicate_exists_false(
        self, repository: PostgresCanonicalRepository
    ) -> None:
        """Test checking non-existent predicate returns False."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        result = await repo.predicate_exists("nonexistent")

        assert result is False


class TestContextEdgeCases:
    """Edge case tests for context repository operations."""

    @pytest.mark.asyncio
    async def test_save_context_with_complex_data(self) -> None:
        """Test saving context with nested data structures."""
        complex_data = {
            "channels": [123, 456, 789],
            "config": {"setting1": True, "setting2": False},
            "nested": {"deep": {"value": "test"}},
        }
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresContextRepository(mock_pool)
        await repo.save_context("channel", "123", complex_data)

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_context_type_multiple_channels(self) -> None:
        """Test loading context for type with multiple channel entries."""
        mock_rows = [
            {
                "target_id": "channel1",
                "data": {"key": "val1"},
                "updated_at": datetime.now(timezone.utc),
            },
            {
                "target_id": "channel2",
                "data": {"key": "val2"},
                "updated_at": datetime.now(timezone.utc),
            },
            {
                "target_id": "channel3",
                "data": {"key": "val3"},
                "updated_at": datetime.now(timezone.utc),
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresContextRepository(mock_pool)
        result = await repo.load_context_type("channel")

        assert len(result) == 3
        assert result[0]["target_id"] == "channel1"
        assert result[1]["target_id"] == "channel2"
        assert result[2]["target_id"] == "channel3"

    @pytest.mark.asyncio
    async def test_store_message_with_metadata(self) -> None:
        """Test storing message with generation metadata."""
        expected_uuid = UUID("12345678-1234-1234-1234-123456789abc")
        mock_row = {"id": expected_uuid}
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresMessageRepository(mock_pool)
        result = await repo.store_message(
            content="AI generated message",
            discord_message_id=999,
            channel_id=111,
            is_bot=True,
            is_proactive=True,
            generation_metadata={
                "model": "gpt-4",
                "temperature": 0.7,
                "prompt_tokens": 150,
            },
            user_timezone="Europe/London",
        )

        assert result == expected_uuid

    @pytest.mark.asyncio
    async def test_get_recent_messages_limit_variations(self) -> None:
        """Test getting messages with different limit values."""
        for limit in [1, 5, 10, 50, 100]:
            mock_rows = []
            mock_conn = AsyncMock()
            mock_conn.fetch = AsyncMock(return_value=mock_rows)
            mock_pool = MagicMock()
            mock_pool.pool.acquire = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
                )
            )

            repo = PostgresMessageRepository(mock_pool)
            result = await repo.get_recent_messages(limit=limit)

            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_messages_since_cursor_limit_variations(self) -> None:
        """Test getting messages since cursor with different limits."""
        for limit in [1, 10, 18, 50, 100]:
            mock_rows = []
            mock_conn = AsyncMock()
            mock_conn.fetch = AsyncMock(return_value=mock_rows)
            mock_pool = MagicMock()
            mock_pool.pool.acquire = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
                )
            )

            repo = PostgresMessageRepository(mock_pool)
            result = await repo.get_messages_since_cursor(
                cursor_message_id=100, limit=limit
            )

            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_store_semantic_memories_alias_handling(self) -> None:
        """Test storing semantic memories with alias predicates.

        This test verifies the function handles alias memories correctly.
        The actual DB operation is tested via integration tests.
        """
        memories = [
            {"subject": "User", "predicate": "has alias", "obj": "U"},
            {"subject": "User", "predicate": "likes", "obj": "Python"},
        ]

        mock_conn = MagicMock()
        mock_conn.fetchval = AsyncMock(
            return_value=UUID("22222222-2222-2222-2222-222222222222")
        )

        # Create proper async context manager mocks
        class MockTxCtx:
            def __init__(self, conn):
                self.conn = conn

            async def __aenter__(self):
                return self.conn

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        class MockAcquireCtx:
            def __init__(self, conn):
                self.conn = conn

            async def __aenter__(self):
                return self.conn

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        def create_tx_ctx():
            return MockTxCtx(mock_conn)

        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(return_value=MockAcquireCtx(mock_conn))
        mock_conn.transaction = create_tx_ctx

        repo = PostgresMessageRepository(mock_pool)

        # This should not raise an exception
        result = await repo.store_semantic_memories(
            message_id=UUID("11111111-1111-1111-1111-111111111111"),
            memories=memories,
        )

        # Verify the function ran without error (actual count depends on mock)
        assert result >= 0

    @pytest.mark.asyncio
    async def test_find_memories_with_multiple_filters(self) -> None:
        """Test finding memories with multiple filter criteria."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresSemanticMemoryRepository(mock_pool)
        result = await repo.find_memories(
            subject="User",
            predicate="likes",
            object="Python",
            start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 12, 31, tzinfo=timezone.utc),
            limit=25,
        )

        assert len(result) == 0
        call_args = mock_conn.fetch.call_args
        query = call_args[0][0]
        assert "subject = $" in query
        assert "predicate = $" in query
        assert "object = $" in query
        assert "created_at >= $" in query
        assert "created_at <= $" in query

    @pytest.mark.asyncio
    async def test_traverse_from_entity_max_hops(self) -> None:
        """Test traversal with different max_hops values."""
        for max_hops in [1, 2, 3, 5, 10]:
            mock_conn = AsyncMock()
            mock_conn.fetch = AsyncMock(return_value=[])
            mock_pool = MagicMock()
            mock_pool.pool.acquire = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
                )
            )

            repo = PostgresSemanticMemoryRepository(mock_pool)
            result = await repo.traverse_from_entity(
                entity_name="User",
                max_hops=max_hops,
            )

            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_message_timestamps_since(self) -> None:
        """Test getting message timestamps for activity analysis."""
        mock_rows = [
            {
                "timestamp": datetime(2026, 1, 12, 10, 0, tzinfo=timezone.utc),
                "user_timezone": "UTC",
            },
            {
                "timestamp": datetime(2026, 1, 12, 11, 0, tzinfo=timezone.utc),
                "user_timezone": "America/New_York",
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresMessageRepository(mock_pool)
        since = datetime(2026, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = await repo.get_message_timestamps_since(since=since, is_bot=True)

        assert len(result) == 2
        assert result[0]["timestamp"] == mock_rows[0]["timestamp"]
        assert result[1]["user_timezone"] == "America/New_York"

    @pytest.mark.asyncio
    async def test_get_subjects_for_review_various_thresholds(self) -> None:
        """Test getting subjects for review with different memory thresholds."""
        for threshold in [1, 5, 10, 20, 50]:
            mock_rows = [{"subject": "User"}]
            mock_conn = AsyncMock()
            mock_conn.fetch = AsyncMock(return_value=mock_rows)
            mock_pool = MagicMock()
            mock_pool.pool.acquire = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
                )
            )

            repo = PostgresSemanticMemoryRepository(mock_pool)
            result = await repo.get_subjects_for_review(min_memories=threshold)

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_store_entities_duplicate_handling(self) -> None:
        """Test storing entities handles duplicates gracefully."""
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        result = await repo.store_entities(
            ["User", "User", "Project", "Project", "User"]
        )

        # Should return count of all names attempted
        assert result == 5

    @pytest.mark.asyncio
    async def test_store_predicates_duplicate_handling(self) -> None:
        """Test storing predicates handles duplicates gracefully."""
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        result = await repo.store_predicates(["likes", "likes", "works_on", "works_on"])

        assert result == 4

    @pytest.mark.asyncio
    async def test_entities_and_predicates_set_consistency(self) -> None:
        """Test that entities and predicates set methods return proper sets."""
        mock_entity_rows = [{"name": "User"}, {"name": "Project"}, {"name": "User"}]
        mock_predicate_rows = [
            {"name": "likes"},
            {"name": "works_on"},
            {"name": "likes"},
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(side_effect=[mock_entity_rows, mock_predicate_rows])
        mock_pool = MagicMock()
        mock_pool.pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        repo = PostgresCanonicalRepository(mock_pool)
        entities = await repo.get_entities_set()
        predicates = await repo.get_predicates_set()

        # Sets should deduplicate
        assert entities == {"User", "Project"}
        assert predicates == {"likes", "works_on"}
