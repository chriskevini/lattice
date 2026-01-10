"""Unit tests for episodic memory module."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import asyncpg
import pytest

from lattice.memory.episodic import (
    EpisodicMessage,
    get_recent_messages,
    store_message,
    store_semantic_memories,
)


def create_mock_pool_with_transaction() -> tuple[MagicMock, AsyncMock]:
    """Create a properly mocked database pool with transaction support.

    Returns:
        Tuple of (mock_pool, mock_conn) for use in tests.
    """
    mock_conn = AsyncMock()

    # Create async context manager for conn.transaction()
    mock_transaction_cm = MagicMock()
    mock_transaction_cm.__aenter__ = AsyncMock(return_value=None)
    mock_transaction_cm.__aexit__ = AsyncMock(return_value=None)

    # transaction() is a regular method that returns an async context manager
    mock_conn.transaction = MagicMock(return_value=mock_transaction_cm)

    # Create async context manager for pool.acquire()
    mock_acquire_cm = MagicMock()
    mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

    mock_pool = MagicMock()
    # acquire() is a regular method that returns an async context manager
    mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

    return mock_pool, mock_conn


class TestEpisodicMessage:
    """Tests for EpisodicMessage dataclass."""

    def test_init_with_all_fields(self) -> None:
        """Test EpisodicMessage initialization with all fields provided."""
        message_id = uuid4()
        timestamp = datetime.now(UTC)
        metadata = {"model": "gpt-4", "tokens": 100}

        msg = EpisodicMessage(
            content="Test message",
            discord_message_id=123456,
            channel_id=789,
            is_bot=True,
            is_proactive=True,
            message_id=message_id,
            timestamp=timestamp,
            generation_metadata=metadata,
            user_timezone="America/New_York",
        )

        assert msg.content == "Test message"
        assert msg.discord_message_id == 123456
        assert msg.channel_id == 789
        assert msg.is_bot is True
        assert msg.is_proactive is True
        assert msg.message_id == message_id
        assert msg.timestamp == timestamp
        assert msg.generation_metadata == metadata
        assert msg.user_timezone == "America/New_York"

    def test_init_with_minimal_fields(self) -> None:
        """Test EpisodicMessage initialization with only required fields."""
        msg = EpisodicMessage(
            content="Test",
            discord_message_id=123,
            channel_id=456,
            is_bot=False,
        )

        assert msg.content == "Test"
        assert msg.discord_message_id == 123
        assert msg.channel_id == 456
        assert msg.is_bot is False
        assert msg.is_proactive is False
        assert msg.message_id is None
        assert msg.generation_metadata is None

    def test_timestamp_defaults_to_now(self) -> None:
        """Test that timestamp defaults to current UTC time."""
        before = datetime.now(UTC)
        msg = EpisodicMessage(
            content="Test",
            discord_message_id=123,
            channel_id=456,
            is_bot=False,
        )
        after = datetime.now(UTC)

        assert before <= msg.timestamp <= after
        assert msg.timestamp.tzinfo == UTC

    def test_user_timezone_defaults_to_utc_when_none(self) -> None:
        """Test that user_timezone defaults to 'UTC' when None."""
        msg = EpisodicMessage(
            content="Test",
            discord_message_id=123,
            channel_id=456,
            is_bot=False,
            user_timezone=None,
        )

        assert msg.user_timezone == "UTC"

    def test_role_property_user_message(self) -> None:
        """Test role property returns USER for non-bot messages."""
        msg = EpisodicMessage(
            content="Test",
            discord_message_id=123,
            channel_id=456,
            is_bot=False,
        )

        assert msg.role == "USER"

    def test_role_property_bot_message(self) -> None:
        """Test role property returns ASSISTANT for bot messages."""
        msg = EpisodicMessage(
            content="Test",
            discord_message_id=123,
            channel_id=456,
            is_bot=True,
        )

        assert msg.role == "ASSISTANT"


class TestStoreMessage:
    """Tests for store_message function."""

    @pytest.mark.asyncio
    async def test_store_message_without_metadata(self) -> None:
        """Test storing a message without generation metadata."""
        message_id = uuid4()
        msg = EpisodicMessage(
            content="Hello",
            discord_message_id=12345,
            channel_id=67890,
            is_bot=False,
        )

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        mock_conn.fetchrow = AsyncMock(return_value={"id": message_id})

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await store_message(msg)

            assert result == message_id
            mock_conn.fetchrow.assert_called_once()
            call_args = mock_conn.fetchrow.call_args
            assert "INSERT INTO raw_messages" in call_args[0][0]
            assert call_args[0][1] == 12345  # discord_message_id
            assert call_args[0][2] == 67890  # channel_id
            assert call_args[0][3] == "Hello"  # content
            assert call_args[0][4] is False  # is_bot
            assert call_args[0][5] is False  # is_proactive
            assert call_args[0][6] is None  # generation_metadata (None)

    @pytest.mark.asyncio
    async def test_store_message_with_full_metadata(self) -> None:
        """Test storing a bot message with generation metadata."""
        message_id = uuid4()
        metadata = {"model": "gpt-4", "tokens": 150, "finish_reason": "stop"}
        msg = EpisodicMessage(
            content="Bot response",
            discord_message_id=11111,
            channel_id=22222,
            is_bot=True,
            is_proactive=False,
            generation_metadata=metadata,
        )

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        mock_conn.fetchrow = AsyncMock(return_value={"id": message_id})

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await store_message(msg)

            assert result == message_id
            call_args = mock_conn.fetchrow.call_args
            # Verify metadata was JSON serialized
            assert call_args[0][6] == json.dumps(metadata)

    @pytest.mark.asyncio
    async def test_store_message_proactive_flag(self) -> None:
        """Test storing a proactive bot message."""
        message_id = uuid4()
        msg = EpisodicMessage(
            content="Proactive message",
            discord_message_id=99999,
            channel_id=88888,
            is_bot=True,
            is_proactive=True,
        )

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        mock_conn.fetchrow = AsyncMock(return_value={"id": message_id})

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await store_message(msg)

            assert result == message_id
            call_args = mock_conn.fetchrow.call_args
            assert call_args[0][5] is True  # is_proactive

    @pytest.mark.asyncio
    async def test_store_message_database_error(self) -> None:
        """Test that database errors propagate from store_message."""
        msg = EpisodicMessage(
            content="Test",
            discord_message_id=123,
            channel_id=456,
            is_bot=False,
        )

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        mock_conn.fetchrow = AsyncMock(side_effect=Exception("DB connection failed"))

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            with pytest.raises(Exception, match="DB connection failed"):
                await store_message(msg)


class TestGetRecentMessages:
    """Tests for get_recent_messages function."""

    @pytest.mark.asyncio
    async def test_get_recent_messages_specific_channel(self) -> None:
        """Test retrieving recent messages from a specific channel."""
        # Mock DB returns newest-first (DESC order)
        mock_rows = [
            {
                "id": uuid4(),
                "content": "Message 2",  # Newer message first
                "discord_message_id": 2,
                "channel_id": 123,
                "is_bot": True,
                "is_proactive": False,
                "timestamp": datetime.now(UTC),
                "generation_metadata": json.dumps({"model": "gpt-4"}),
                "user_timezone": "UTC",
            },
            {
                "id": uuid4(),
                "content": "Message 1",  # Older message second
                "discord_message_id": 1,
                "channel_id": 123,
                "is_bot": False,
                "is_proactive": False,
                "timestamp": datetime.now(UTC),
                "generation_metadata": None,
                "user_timezone": "UTC",
            },
        ]

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            messages = await get_recent_messages(channel_id=123, limit=10)

            # Function reverses, so oldest comes first after reversal
            assert len(messages) == 2
            assert messages[0].content == "Message 1"  # Oldest first
            assert messages[1].content == "Message 2"  # Newest second
            mock_conn.fetch.assert_called_once()
            call_args = mock_conn.fetch.call_args
            assert "WHERE channel_id = $1" in call_args[0][0]
            assert call_args[0][1] == 123

    @pytest.mark.asyncio
    async def test_get_recent_messages_all_channels(self) -> None:
        """Test retrieving recent messages from all channels."""
        mock_rows = [
            {
                "id": uuid4(),
                "content": "Message from channel 1",
                "discord_message_id": 1,
                "channel_id": 100,
                "is_bot": False,
                "is_proactive": False,
                "timestamp": datetime.now(UTC),
                "generation_metadata": None,
                "user_timezone": "UTC",
            },
        ]

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            messages = await get_recent_messages(channel_id=None, limit=5)

            assert len(messages) == 1
            call_args = mock_conn.fetch.call_args
            # Should not have WHERE clause for channel_id when None
            assert "WHERE channel_id" not in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_recent_messages_empty_result(self) -> None:
        """Test get_recent_messages returns empty list when no messages exist."""
        mock_pool, mock_conn = create_mock_pool_with_transaction()
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            messages = await get_recent_messages(channel_id=999, limit=10)

            assert messages == []

    @pytest.mark.asyncio
    async def test_get_recent_messages_ordering(self) -> None:
        """Test that messages are ordered oldest-first (reversed from DESC query)."""
        # Simulate DB returning newest-first
        mock_rows = [
            {
                "id": uuid4(),
                "content": "Newest",
                "discord_message_id": 3,
                "channel_id": 123,
                "is_bot": False,
                "is_proactive": False,
                "timestamp": datetime(2024, 1, 3, tzinfo=UTC),
                "generation_metadata": None,
                "user_timezone": "UTC",
            },
            {
                "id": uuid4(),
                "content": "Oldest",
                "discord_message_id": 1,
                "channel_id": 123,
                "is_bot": False,
                "is_proactive": False,
                "timestamp": datetime(2024, 1, 1, tzinfo=UTC),
                "generation_metadata": None,
                "user_timezone": "UTC",
            },
        ]

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            messages = await get_recent_messages(channel_id=123, limit=10)

            # Function should reverse the list, so oldest comes first
            assert messages[0].content == "Oldest"
            assert messages[1].content == "Newest"

    @pytest.mark.asyncio
    async def test_get_recent_messages_null_timezone_defaults_to_utc(self) -> None:
        """Test that NULL timezone in DB defaults to 'UTC' in Python object."""
        mock_rows = [
            {
                "id": uuid4(),
                "content": "Message",
                "discord_message_id": 1,
                "channel_id": 123,
                "is_bot": False,
                "is_proactive": False,
                "timestamp": datetime.now(UTC),
                "generation_metadata": None,
                "user_timezone": None,  # NULL from DB
            },
        ]

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            messages = await get_recent_messages(channel_id=123, limit=10)

            assert messages[0].user_timezone == "UTC"


class TestStoreSemanticMemories:
    """Tests for store_semantic_memories function."""

    @pytest.mark.asyncio
    async def test_store_semantic_memories_empty_list(self) -> None:
        """Test that empty memories list returns early without DB access."""
        # result is None as it returns early
        await store_semantic_memories(uuid4(), [])  # type: ignore[func-returns-value]

    @pytest.mark.asyncio
    async def test_store_semantic_memories_valid_memory(self) -> None:
        """Test storing a valid semantic memory."""
        message_id = uuid4()

        memories = [{"subject": "Alice", "predicate": "likes", "object": "Python"}]

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        mock_conn.execute = AsyncMock()

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            await store_semantic_memories(message_id, memories)

            # Verify memory was inserted
            mock_conn.execute.assert_called_once()
            call_args = mock_conn.execute.call_args
            assert "INSERT INTO semantic_memories" in call_args[0][0]
            assert call_args[0][1] == "Alice"
            assert call_args[0][2] == "likes"
            assert call_args[0][3] == "Python"
            assert call_args[0][4] is None  # source_batch_id

    @pytest.mark.asyncio
    async def test_store_semantic_memories_skips_invalid(self) -> None:
        """Test that invalid memories (missing fields) are skipped with warning."""
        message_id = uuid4()

        invalid_memories = [
            {"subject": "", "predicate": "likes", "object": "Python"},  # Empty subject
            {
                "subject": "Alice",
                "predicate": "",
                "object": "Python",
            },  # Empty predicate
            {"subject": "Alice", "predicate": "likes", "object": ""},  # Empty object
            {"subject": "Alice", "predicate": "likes"},  # Missing object key
        ]

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        mock_conn.execute = AsyncMock()

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            await store_semantic_memories(message_id, invalid_memories)

            # Should not insert any memories (all invalid)
            mock_conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_semantic_memories_continues_on_error(self) -> None:
        """Test that errors executing INSERT don't stop other memories."""
        message_id = uuid4()

        memories = [
            {"subject": "Alice", "predicate": "likes", "object": "Python"},
            {"subject": "Bob", "predicate": "uses", "object": "Java"},
        ]

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        # First execute succeeds, second fails
        mock_conn.execute = AsyncMock(
            side_effect=[None, asyncpg.PostgresError("DB error")]
        )

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            # Should not raise exception, continues to process all memories
            await store_semantic_memories(message_id, memories)

            # Both memories attempted
            assert mock_conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_store_semantic_memories_continues_on_execute_error(self) -> None:
        """Test that errors executing INSERT don't stop other memories."""
        message_id = uuid4()

        memories = [
            {"subject": "Alice", "predicate": "likes", "object": "Python"},
            {"subject": "Bob", "predicate": "knows", "object": "Carol"},
        ]

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        # First execute succeeds, second fails
        mock_conn.execute = AsyncMock(
            side_effect=[None, asyncpg.PostgresError("DB error")]
        )

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            # Should not raise exception, continues to process all memories
            await store_semantic_memories(message_id, memories)

            # Both memories attempted
            assert mock_conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_store_semantic_memories_uses_transaction(self) -> None:
        """Test that store_semantic_memories uses a transaction."""
        message_id = uuid4()
        memories = [{"subject": "A", "predicate": "rel", "object": "B"}]

        mock_pool, mock_conn = create_mock_pool_with_transaction()
        mock_conn.execute = AsyncMock()

        with patch("lattice.memory.episodic.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            await store_semantic_memories(message_id, memories)

            # Verify transaction was used
            mock_conn.transaction.assert_called_once()
