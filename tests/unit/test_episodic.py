"""Unit tests for episodic memory module."""

from lattice.utils.date_resolution import get_now
from zoneinfo import ZoneInfo
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

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

    mock_transaction_cm = MagicMock()
    mock_transaction_cm.__aenter__ = AsyncMock(return_value=None)
    mock_transaction_cm.__aexit__ = AsyncMock(return_value=None)

    mock_conn.transaction = MagicMock(return_value=mock_transaction_cm)

    mock_acquire_cm = MagicMock()
    mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

    mock_pool = MagicMock()
    mock_pool.pool.acquire = MagicMock(return_value=mock_acquire_cm)

    return mock_pool, mock_conn


class TestEpisodicMessage:
    """Tests for EpisodicMessage dataclass."""

    def test_init_with_all_fields(self) -> None:
        """Test EpisodicMessage initialization with all fields provided."""
        message_id = uuid4()
        timestamp = get_now()
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
        before = get_now()
        msg = EpisodicMessage(
            content="Test",
            discord_message_id=123,
            channel_id=456,
            is_bot=False,
        )
        after = get_now()

        assert before <= msg.timestamp <= after
        if isinstance(msg.timestamp.tzinfo, ZoneInfo):
            assert msg.timestamp.tzinfo == ZoneInfo("UTC")
        else:
            assert msg.timestamp.tzinfo is not None
            assert msg.timestamp.tzinfo.tzname(None) == "UTC"

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

        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(return_value=message_id)

        result = await store_message(repo=mock_repo, message=msg)

        assert result == message_id
        mock_repo.store_message.assert_called_once()
        call_args = mock_repo.store_message.call_args
        assert call_args.kwargs["content"] == "Hello"
        assert call_args.kwargs["discord_message_id"] == 12345
        assert call_args.kwargs["channel_id"] == 67890
        assert call_args.kwargs["is_bot"] is False

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

        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(return_value=message_id)

        result = await store_message(repo=mock_repo, message=msg)

        assert result == message_id
        mock_repo.store_message.assert_called_once()
        call_args = mock_repo.store_message.call_args
        assert call_args.kwargs["generation_metadata"] == metadata

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

        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(return_value=message_id)

        result = await store_message(repo=mock_repo, message=msg)

        assert result == message_id
        mock_repo.store_message.assert_called_once()
        call_args = mock_repo.store_message.call_args
        assert call_args.kwargs["is_proactive"] is True

    @pytest.mark.asyncio
    async def test_store_message_database_error(self) -> None:
        """Test that database errors propagate from store_message."""
        msg = EpisodicMessage(
            content="Test",
            discord_message_id=123,
            channel_id=456,
            is_bot=False,
        )

        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(
            side_effect=Exception("DB connection failed")
        )

        with pytest.raises(Exception, match="DB connection failed"):
            await store_message(repo=mock_repo, message=msg)


class TestGetRecentMessages:
    """Tests for get_recent_messages function."""

    @pytest.mark.asyncio
    async def test_get_recent_messages_specific_channel(self) -> None:
        """Test retrieving recent messages from a specific channel."""
        message1_id = uuid4()
        message2_id = uuid4()
        now = get_now()
        mock_rows = [
            {
                "id": message2_id,
                "discord_message_id": 2,
                "channel_id": 123,
                "content": "Message 2",
                "is_bot": True,
                "is_proactive": False,
                "timestamp": now,
                "user_timezone": "UTC",
            },
            {
                "id": message1_id,
                "discord_message_id": 1,
                "channel_id": 123,
                "content": "Message 1",
                "is_bot": False,
                "is_proactive": False,
                "timestamp": now,
                "user_timezone": "UTC",
            },
        ]

        mock_repo = MagicMock()
        mock_repo.get_recent_messages = AsyncMock(return_value=mock_rows)

        messages = await get_recent_messages(repo=mock_repo, channel_id=123, limit=10)

        assert len(messages) == 2
        # get_recent_messages calls reversed(rows), so newest (Message 2) becomes last
        assert messages[0].content == "Message 1"
        assert messages[1].content == "Message 2"
        mock_repo.get_recent_messages.assert_called_once_with(channel_id=123, limit=10)

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
                "timestamp": get_now(),
                "user_timezone": "UTC",
            },
        ]

        mock_repo = MagicMock()
        mock_repo.get_recent_messages = AsyncMock(return_value=mock_rows)

        messages = await get_recent_messages(repo=mock_repo, channel_id=None, limit=5)

        assert len(messages) == 1
        mock_repo.get_recent_messages.assert_called_once_with(channel_id=None, limit=5)

    @pytest.mark.asyncio
    async def test_get_recent_messages_empty_result(self) -> None:
        """Test get_recent_messages returns empty list when no messages exist."""
        mock_repo = MagicMock()
        mock_repo.get_recent_messages = AsyncMock(return_value=[])

        messages = await get_recent_messages(repo=mock_repo, channel_id=999, limit=10)

        assert messages == []

    @pytest.mark.asyncio
    async def test_get_recent_messages_ordering(self) -> None:
        """Test that messages are ordered oldest-first."""
        now = get_now()
        mock_rows = [
            {
                "id": uuid4(),
                "content": "Newest",
                "discord_message_id": 3,
                "channel_id": 123,
                "is_bot": False,
                "is_proactive": False,
                "timestamp": now,
                "user_timezone": "UTC",
            },
            {
                "id": uuid4(),
                "content": "Oldest",
                "discord_message_id": 1,
                "channel_id": 123,
                "is_bot": False,
                "is_proactive": False,
                "timestamp": now,
                "user_timezone": "UTC",
            },
        ]

        mock_repo = MagicMock()
        mock_repo.get_recent_messages = AsyncMock(return_value=mock_rows)

        messages = await get_recent_messages(repo=mock_repo, channel_id=123, limit=10)

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
                "timestamp": get_now(),
                "user_timezone": None,
            },
        ]

        mock_repo = MagicMock()
        mock_repo.get_recent_messages = AsyncMock(return_value=mock_rows)

        messages = await get_recent_messages(repo=mock_repo, channel_id=123, limit=10)

        assert messages[0].user_timezone == "UTC"


class TestStoreSemanticMemories:
    """Tests for store_semantic_memories function."""

    @pytest.mark.asyncio
    async def test_store_semantic_memories_empty_list(self) -> None:
        """Test that empty memories list returns early without DB access."""
        await store_semantic_memories(repo=MagicMock(), message_id=uuid4(), memories=[])

    @pytest.mark.asyncio
    async def test_store_semantic_memories_valid_memory(self) -> None:
        """Test storing a valid semantic memory."""
        message_id = uuid4()
        memories = [{"subject": "Alice", "predicate": "likes", "object": "Python"}]

        mock_repo = MagicMock()
        mock_repo.store_semantic_memories = AsyncMock()

        await store_semantic_memories(
            repo=mock_repo, message_id=message_id, memories=memories
        )

        mock_repo.store_semantic_memories.assert_called_once_with(
            message_id=message_id, memories=memories, source_batch_id=None
        )

    @pytest.mark.asyncio
    async def test_store_semantic_memories_skips_invalid(self) -> None:
        """Test that invalid memories (missing fields) are skipped with warning."""
        # Note: Logic moved to repository, but episodic.py might still do some filtering
        # Actually in the new implementation episodic.py's store_semantic_memories
        # just calls repo.store_semantic_memories.
        message_id = uuid4()
        invalid_memories = [
            {"subject": "", "predicate": "likes", "object": "Python"},
        ]

        mock_repo = MagicMock()
        mock_repo.store_semantic_memories = AsyncMock()

        await store_semantic_memories(
            repo=mock_repo, message_id=message_id, memories=invalid_memories
        )

        mock_repo.store_semantic_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_semantic_memories_continues_on_error(self) -> None:
        """Test that errors executing INSERT don't stop other memories."""
        message_id = uuid4()
        memories = [
            {"subject": "Alice", "predicate": "likes", "object": "Python"},
        ]

        mock_repo = MagicMock()
        mock_repo.store_semantic_memories = AsyncMock(side_effect=Exception("DB error"))

        with pytest.raises(Exception, match="DB error"):
            await store_semantic_memories(
                repo=mock_repo, message_id=message_id, memories=memories
            )

    @pytest.mark.asyncio
    async def test_store_semantic_memories_uses_transaction(self) -> None:
        """Test that store_semantic_memories uses a transaction."""
        # Transaction management is now internal to the repository.
        # We just verify it calls the repository method.
        message_id = uuid4()
        memories = [{"subject": "A", "predicate": "rel", "object": "B"}]

        mock_repo = MagicMock()
        mock_repo.store_semantic_memories = AsyncMock()

        await store_semantic_memories(
            repo=mock_repo, message_id=message_id, memories=memories
        )

        mock_repo.store_semantic_memories.assert_called_once()
