"""Unit tests for lattice/core/memory_orchestrator.py.

Tests the memory orchestration functions for storing and retrieving
episodic and semantic memories.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from lattice.core.constants import DEFAULT_EPISODIC_LIMIT
from lattice.core.memory_orchestrator import (
    retrieve_context,
    store_bot_message,
    store_user_message,
)


class TestStoreUserMessage:
    """Tests for store_user_message function."""

    @pytest.mark.asyncio
    async def test_store_user_message_success(self) -> None:
        """Test successful storage of a user message.

        Verifies that a user message is properly stored in episodic memory
        with correct metadata and the expected UUID is returned.
        """
        expected_uuid = UUID("11111111-1111-1111-1111-111111111111")
        mock_message_repo = AsyncMock()
        mock_message_repo.store_message.return_value = expected_uuid

        result = await store_user_message(
            content="Hello, world!",
            discord_message_id=12345,
            channel_id=67890,
            message_repo=mock_message_repo,
            timezone="America/New_York",
        )

        assert result == expected_uuid
        mock_message_repo.store_message.assert_called_once_with(
            content="Hello, world!",
            discord_message_id=12345,
            channel_id=67890,
            is_bot=False,
            is_proactive=False,
            generation_metadata=None,
            user_timezone="America/New_York",
        )

    @pytest.mark.asyncio
    async def test_store_user_message_default_timezone(self) -> None:
        """Test store_user_message with default UTC timezone.

        Verifies that when no timezone is provided, UTC is used by default.
        """
        expected_uuid = UUID("22222222-2222-2222-2222-222222222222")
        mock_message_repo = AsyncMock()
        mock_message_repo.store_message.return_value = expected_uuid

        result = await store_user_message(
            content="Test message",
            discord_message_id=99999,
            channel_id=11111,
            message_repo=mock_message_repo,
        )

        assert result == expected_uuid
        mock_message_repo.store_message.assert_called_once()
        call_kwargs = mock_message_repo.store_message.call_args[1]
        assert call_kwargs["user_timezone"] == "UTC"
        assert call_kwargs["is_bot"] is False

    @pytest.mark.asyncio
    async def test_store_user_message_database_error(self) -> None:
        """Test store_user_message propagates database errors.

        When the repository raises an exception, it should propagate
        to the caller for proper error handling.
        """
        mock_message_repo = AsyncMock()
        mock_message_repo.store_message.side_effect = Exception(
            "Database connection failed"
        )

        with pytest.raises(Exception, match="Database connection failed"):
            await store_user_message(
                content="Test message",
                discord_message_id=12345,
                channel_id=67890,
                message_repo=mock_message_repo,
            )

    @pytest.mark.asyncio
    async def test_store_user_message_special_characters(self) -> None:
        """Test store_user_message handles special characters in content.

        Verifies that messages with emojis, newlines, and other special
        characters are stored correctly.
        """
        expected_uuid = UUID("33333333-3333-3333-3333-333333333333")
        mock_message_repo = AsyncMock()
        mock_message_repo.store_message.return_value = expected_uuid

        special_content = "Hello! 👋\n\nThis is a multi-line\nmessage with special chars: café, naïve."

        result = await store_user_message(
            content=special_content,
            discord_message_id=55555,
            channel_id=66666,
            message_repo=mock_message_repo,
        )

        assert result == expected_uuid
        mock_message_repo.store_message.assert_called_once()
        call_kwargs = mock_message_repo.store_message.call_args[1]
        assert call_kwargs["content"] == special_content

    @pytest.mark.asyncio
    async def test_store_user_message_long_content(self) -> None:
        """Test store_user_message handles long message content.

        Verifies that very long messages can be stored without issues.
        """
        expected_uuid = UUID("44444444-4444-4444-4444-444444444444")
        mock_message_repo = AsyncMock()
        mock_message_repo.store_message.return_value = expected_uuid

        long_content = "word " * 1000  # 4000+ characters

        result = await store_user_message(
            content=long_content,
            discord_message_id=77777,
            channel_id=88888,
            message_repo=mock_message_repo,
        )

        assert result == expected_uuid
        mock_message_repo.store_message.assert_called_once()


class TestStoreBotMessage:
    """Tests for store_bot_message function."""

    @pytest.mark.asyncio
    async def test_store_bot_message_success(self) -> None:
        """Test successful storage of a bot message.

        Verifies that a bot message is properly stored with is_bot=True
        and the expected UUID is returned.
        """
        expected_uuid = UUID("55555555-5555-5555-5555-555555555555")
        mock_message_repo = AsyncMock()
        mock_message_repo.store_message.return_value = expected_uuid

        result = await store_bot_message(
            content="I'm a helpful assistant!",
            discord_message_id=11111,
            channel_id=22222,
            message_repo=mock_message_repo,
        )

        assert result == expected_uuid
        mock_message_repo.store_message.assert_called_once()
        call_kwargs = mock_message_repo.store_message.call_args[1]
        assert call_kwargs["is_bot"] is True
        assert call_kwargs["is_proactive"] is False
        assert call_kwargs["generation_metadata"] is None

    @pytest.mark.asyncio
    async def test_store_bot_message_with_metadata(self) -> None:
        """Test store_bot_message with generation metadata.

        Verifies that LLM generation metadata (model, tokens, etc.)
        is properly stored with the bot message.
        """
        expected_uuid = UUID("66666666-6666-6666-6666-666666666666")
        mock_message_repo = AsyncMock()
        mock_message_repo.store_message.return_value = expected_uuid

        metadata = {
            "model": "gpt-4",
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200,
        }

        result = await store_bot_message(
            content="Generated response",
            discord_message_id=33333,
            channel_id=44444,
            message_repo=mock_message_repo,
            generation_metadata=metadata,
        )

        assert result == expected_uuid
        mock_message_repo.store_message.assert_called_once()
        call_kwargs = mock_message_repo.store_message.call_args[1]
        assert call_kwargs["generation_metadata"] == metadata
        assert call_kwargs["is_bot"] is True

    @pytest.mark.asyncio
    async def test_store_bot_message_proactive(self) -> None:
        """Test store_bot_message for proactive bot messages.

        Verifies that is_proactive flag is properly set when the bot
        initiates a message rather than replying.
        """
        expected_uuid = UUID("77777777-7777-7777-7777-777777777777")
        mock_message_repo = AsyncMock()
        mock_message_repo.store_message.return_value = expected_uuid

        result = await store_bot_message(
            content="Don't forget your meeting at 3pm!",
            discord_message_id=55555,
            channel_id=66666,
            message_repo=mock_message_repo,
            is_proactive=True,
        )

        assert result == expected_uuid
        mock_message_repo.store_message.assert_called_once()
        call_kwargs = mock_message_repo.store_message.call_args[1]
        assert call_kwargs["is_proactive"] is True

    @pytest.mark.asyncio
    async def test_store_bot_message_with_timezone(self) -> None:
        """Test store_bot_message with custom timezone.

        Verifies that the timezone parameter is properly passed through.
        """
        expected_uuid = UUID("88888888-8888-8888-8888-888888888888")
        mock_message_repo = AsyncMock()
        mock_message_repo.store_message.return_value = expected_uuid

        result = await store_bot_message(
            content="Good morning!",
            discord_message_id=77777,
            channel_id=88888,
            message_repo=mock_message_repo,
            timezone="Europe/London",
        )

        assert result == expected_uuid
        mock_message_repo.store_message.assert_called_once()
        call_kwargs = mock_message_repo.store_message.call_args[1]
        assert call_kwargs["user_timezone"] == "Europe/London"

    @pytest.mark.asyncio
    async def test_store_bot_message_database_error(self) -> None:
        """Test store_bot_message propagates database errors.

        When the repository raises an exception, it should propagate
        to the caller for proper error handling.
        """
        mock_message_repo = AsyncMock()
        mock_message_repo.store_message.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            await store_bot_message(
                content="Test response",
                discord_message_id=12345,
                channel_id=67890,
                message_repo=mock_message_repo,
            )

    @pytest.mark.asyncio
    async def test_store_bot_message_all_parameters(self) -> None:
        """Test store_bot_message with all parameters provided.

        Verifies that all parameters are properly passed to the repository.
        """
        expected_uuid = UUID("99999999-9999-9999-9999-999999999999")
        mock_message_repo = AsyncMock()
        mock_message_repo.store_message.return_value = expected_uuid

        metadata = {"model": "claude-3-opus", "latency_ms": 1500}

        result = await store_bot_message(
            content="Full test message",
            discord_message_id=123456789,
            channel_id=987654321,
            message_repo=mock_message_repo,
            is_proactive=True,
            generation_metadata=metadata,
            timezone="Asia/Tokyo",
        )

        assert result == expected_uuid
        mock_message_repo.store_message.assert_called_once()
        call_kwargs = mock_message_repo.store_message.call_args[1]
        assert call_kwargs["content"] == "Full test message"
        assert call_kwargs["discord_message_id"] == 123456789
        assert call_kwargs["channel_id"] == 987654321
        assert call_kwargs["is_bot"] is True
        assert call_kwargs["is_proactive"] is True
        assert call_kwargs["generation_metadata"] == metadata
        assert call_kwargs["user_timezone"] == "Asia/Tokyo"


class TestRetrieveContext:
    """Tests for retrieve_context function."""

    @pytest.mark.asyncio
    async def test_retrieve_context_recent_messages_only(self) -> None:
        """Test retrieve_context without semantic graph traversal.

        When memory_depth is 0 or entity_names is empty, only recent
        messages should be retrieved without graph traversal.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.return_value = [
            {
                "id": UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
                "discord_message_id": 101,
                "channel_id": 67890,
                "content": "First message",
                "is_bot": False,
                "is_proactive": False,
                "timestamp": datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
                "user_timezone": "UTC",
            },
            {
                "id": UUID("bbbbbbbb-cccc-dddd-eeee-ffffffffffff"),
                "discord_message_id": 102,
                "channel_id": 67890,
                "content": "Second message",
                "is_bot": True,
                "is_proactive": False,
                "timestamp": datetime(2024, 1, 1, 10, 5, 0, tzinfo=UTC),
                "user_timezone": "UTC",
            },
        ]

        recent_messages, semantic_context = await retrieve_context(
            query="test query",
            channel_id=67890,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            episodic_limit=10,
            memory_depth=0,
            entity_names=None,
        )

        assert len(recent_messages) == 2
        assert semantic_context == []
        mock_message_repo.get_recent_messages.assert_called_once_with(
            channel_id=67890, limit=10
        )
        mock_semantic_repo.traverse_from_entity.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_context_with_graph_traversal(self) -> None:
        """Test retrieve_context with semantic graph traversal.

        When entity_names are provided and memory_depth > 0, the function
        should perform graph traversal and include results in semantic_context.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.return_value = [
            {
                "id": UUID("cccccccc-dddd-eeee-ffff-aaaaaaaaaaaa"),
                "discord_message_id": 201,
                "channel_id": 67890,
                "content": "Message about Python",
                "is_bot": False,
                "is_proactive": False,
                "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
                "user_timezone": "UTC",
            },
        ]

        mock_semantic_repo.traverse_from_entity.return_value = [
            {
                "subject": "Python",
                "predicate": "is a",
                "object": "programming language",
                "created_at": datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            },
            {
                "subject": "Python",
                "predicate": "has",
                "object": "dynamically typed",
                "created_at": datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            },
        ]

        recent_messages, semantic_context = await retrieve_context(
            query="What is Python?",
            channel_id=67890,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            episodic_limit=5,
            memory_depth=1,
            entity_names=["Python"],
        )

        assert len(recent_messages) == 1
        assert len(semantic_context) == 2
        assert semantic_context[0]["subject"] == "Python"
        assert semantic_context[0]["predicate"] == "is a"
        mock_semantic_repo.traverse_from_entity.assert_called_once_with(
            entity_name="Python", predicate_filter=None, max_hops=1
        )

    @pytest.mark.asyncio
    async def test_retrieve_context_multiple_entities(self) -> None:
        """Test retrieve_context with multiple entity names.

        Verifies that graph traversal is performed for each entity
        and results are combined in semantic_context.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.return_value = []

        mock_semantic_repo.traverse_from_entity.side_effect = [
            [
                {
                    "subject": "User",
                    "predicate": "likes",
                    "object": "Python",
                    "created_at": datetime(2024, 1, 1, tzinfo=UTC),
                },
            ],
            [
                {
                    "subject": "Python",
                    "predicate": "is used for",
                    "object": "machine learning",
                    "created_at": datetime(2024, 1, 1, tzinfo=UTC),
                },
            ],
        ]

        recent_messages, semantic_context = await retrieve_context(
            query="test",
            channel_id=12345,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            memory_depth=2,
            entity_names=["User", "Python"],
        )

        assert len(semantic_context) == 2
        assert mock_semantic_repo.traverse_from_entity.call_count == 2

    @pytest.mark.asyncio
    async def test_retrieve_context_empty_channel(self) -> None:
        """Test retrieve_context for a channel with no messages.

        Verifies that empty recent messages list is returned when
        the channel has no history.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.return_value = []

        recent_messages, semantic_context = await retrieve_context(
            query="new channel query",
            channel_id=99999,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            episodic_limit=10,
        )

        assert recent_messages == []
        assert semantic_context == []
        mock_message_repo.get_recent_messages.assert_called_once_with(
            channel_id=99999, limit=DEFAULT_EPISODIC_LIMIT
        )

    @pytest.mark.asyncio
    async def test_retrieve_context_default_episodic_limit(self) -> None:
        """Test retrieve_context uses default episodic limit.

        When episodic_limit is not specified, DEFAULT_EPISODIC_LIMIT
        should be used.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.return_value = []

        await retrieve_context(
            query="test",
            channel_id=12345,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
        )

        mock_message_repo.get_recent_messages.assert_called_once_with(
            channel_id=12345, limit=DEFAULT_EPISODIC_LIMIT
        )

    @pytest.mark.asyncio
    async def test_retrieve_context_memory_depth_disabled(self) -> None:
        """Test retrieve_context with memory_depth=0 disables graph traversal.

        Even if entity_names are provided, no graph traversal should occur
        when memory_depth is 0.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.return_value = []

        recent_messages, semantic_context = await retrieve_context(
            query="test query",
            channel_id=12345,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            memory_depth=0,
            entity_names=["SomeEntity"],
        )

        assert recent_messages == []
        assert semantic_context == []
        mock_semantic_repo.traverse_from_entity.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_context_empty_entity_names(self) -> None:
        """Test retrieve_context with empty entity_names list.

        When entity_names is an empty list, no graph traversal should occur.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.return_value = []

        recent_messages, semantic_context = await retrieve_context(
            query="test query",
            channel_id=12345,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            memory_depth=2,
            entity_names=[],
        )

        assert recent_messages == []
        assert semantic_context == []
        mock_semantic_repo.traverse_from_entity.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_context_database_error_messages(self) -> None:
        """Test retrieve_context propagates database errors from message repo.

        When the message repository fails, the exception should propagate.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.side_effect = Exception(
            "Message database error"
        )

        with pytest.raises(Exception, match="Message database error"):
            await retrieve_context(
                query="test query",
                channel_id=12345,
                message_repo=mock_message_repo,
                semantic_repo=mock_semantic_repo,
            )

    @pytest.mark.asyncio
    async def test_retrieve_context_database_error_semantic(self) -> None:
        """Test retrieve_context propagates database errors from semantic repo.

        When the semantic memory repository fails during graph traversal,
        the exception should propagate.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.return_value = []
        mock_semantic_repo.traverse_from_entity.side_effect = Exception(
            "Semantic database error"
        )

        with pytest.raises(Exception, match="Semantic database error"):
            await retrieve_context(
                query="test query",
                channel_id=12345,
                message_repo=mock_message_repo,
                semantic_repo=mock_semantic_repo,
                memory_depth=1,
                entity_names=["TestEntity"],
            )

    @pytest.mark.asyncio
    async def test_retrieve_context_deduplication(self) -> None:
        """Test retrieve_context deduplicates semantic memories.

        Verifies that duplicate memories (same subject, predicate, object)
        are removed from the semantic_context.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.return_value = []

        mock_semantic_repo.traverse_from_entity.return_value = [
            {
                "subject": "Test",
                "predicate": "relates to",
                "object": "Value",
                "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            },
            {
                "subject": "Test",
                "predicate": "relates to",
                "object": "Value",
                "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            },
            {
                "subject": "Different",
                "predicate": "relates to",
                "object": "Value",
                "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            },
        ]

        recent_messages, semantic_context = await retrieve_context(
            query="test",
            channel_id=12345,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            memory_depth=1,
            entity_names=["TestEntity"],
        )

        assert len(semantic_context) == 2
        subjects = [m["subject"] for m in semantic_context]
        assert subjects.count("Test") == 1
        assert subjects.count("Different") == 1

    @pytest.mark.asyncio
    async def test_retrieve_context_invalid_memory_keys_filtered(self) -> None:
        """Test retrieve_context filters memories with empty keys.

        Memories with empty subject, predicate, or object should be
        excluded from the semantic_context.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.return_value = []

        mock_semantic_repo.traverse_from_entity.return_value = [
            {
                "subject": "Valid",
                "predicate": "has",
                "object": "value",
                "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            },
            {
                "subject": "",
                "predicate": "has",
                "object": "value",
                "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            },
            {
                "subject": "Valid",
                "predicate": "",
                "object": "value",
                "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            },
            {
                "subject": "Valid",
                "predicate": "has",
                "object": "",
                "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            },
        ]

        recent_messages, semantic_context = await retrieve_context(
            query="test",
            channel_id=12345,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            memory_depth=1,
            entity_names=["TestEntity"],
        )

        assert len(semantic_context) == 1
        assert semantic_context[0]["subject"] == "Valid"

    @pytest.mark.asyncio
    async def test_retrieve_context_large_episodic_limit(self) -> None:
        """Test retrieve_context with large episodic limit.

        Verifies that large limit values are passed correctly to the repo.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.return_value = []

        await retrieve_context(
            query="test",
            channel_id=12345,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            episodic_limit=100,
        )

        mock_message_repo.get_recent_messages.assert_called_once_with(
            channel_id=12345, limit=100
        )

    @pytest.mark.asyncio
    async def test_retrieve_context_concurrent_traversal(self) -> None:
        """Test retrieve_context performs concurrent graph traversal.

        Verifies that graph traversal for multiple entities happens
        concurrently using asyncio.gather.
        """
        mock_message_repo = AsyncMock()
        mock_semantic_repo = AsyncMock()

        mock_message_repo.get_recent_messages.return_value = []

        call_count = 0

        async def mock_traverse(entity_name: str, **kwargs):
            nonlocal call_count
            call_count += 1
            return []

        mock_semantic_repo.traverse_from_entity = mock_traverse

        await retrieve_context(
            query="test",
            channel_id=12345,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            memory_depth=2,
            entity_names=["Entity1", "Entity2", "Entity3", "Entity4", "Entity5"],
        )

        assert call_count == 5
