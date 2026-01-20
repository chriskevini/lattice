"""Unit tests for memory orchestrator module."""

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call
from uuid import UUID, uuid4

import pytest

from lattice.core.memory_orchestrator import (
    retrieve_context,
    store_bot_message,
    store_user_message,
)
from lattice.memory.episodic import EpisodicMessage


class TestStoreUserMessage:
    """Tests for store_user_message function."""

    @pytest.mark.asyncio
    async def test_stores_user_message_with_defaults(self) -> None:
        """Test storing a user message with default timezone."""
        message_id = uuid4()
        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(return_value=message_id)

        result = await store_user_message(
            content="Hello, bot!",
            discord_message_id=123456,
            channel_id=789,
            message_repo=mock_repo,
        )

        assert result == message_id
        mock_repo.store_message.assert_called_once()
        call_args = mock_repo.store_message.call_args
        assert call_args.kwargs["content"] == "Hello, bot!"
        assert call_args.kwargs["discord_message_id"] == 123456
        assert call_args.kwargs["channel_id"] == 789
        assert call_args.kwargs["is_bot"] is False
        assert call_args.kwargs["user_timezone"] == "UTC"

    @pytest.mark.asyncio
    async def test_stores_user_message_with_custom_timezone(self) -> None:
        """Test storing a user message with custom timezone."""
        message_id = uuid4()
        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(return_value=message_id)

        result = await store_user_message(
            content="Good morning!",
            discord_message_id=999,
            channel_id=111,
            message_repo=mock_repo,
            timezone="America/New_York",
        )

        assert result == message_id
        call_args = mock_repo.store_message.call_args
        assert call_args.kwargs["user_timezone"] == "America/New_York"

    @pytest.mark.asyncio
    async def test_stores_user_message_with_empty_content(self) -> None:
        """Test storing a user message with empty content."""
        message_id = uuid4()
        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(return_value=message_id)

        result = await store_user_message(
            content="",
            discord_message_id=123,
            channel_id=456,
            message_repo=mock_repo,
        )

        assert result == message_id
        call_args = mock_repo.store_message.call_args
        assert call_args.kwargs["content"] == ""

    @pytest.mark.asyncio
    async def test_store_user_message_propagates_db_errors(self) -> None:
        """Test that database errors propagate correctly."""
        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        with pytest.raises(Exception, match="Database connection failed"):
            await store_user_message(
                content="Test",
                discord_message_id=123,
                channel_id=456,
                message_repo=mock_repo,
            )


class TestStoreBotMessage:
    """Tests for store_bot_message function."""

    @pytest.mark.asyncio
    async def test_stores_bot_message_with_defaults(self) -> None:
        """Test storing a bot message with default values."""
        message_id = uuid4()
        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(return_value=message_id)

        result = await store_bot_message(
            content="Hello, human!",
            discord_message_id=987654,
            channel_id=321,
            message_repo=mock_repo,
        )

        assert result == message_id
        mock_repo.store_message.assert_called_once()
        call_args = mock_repo.store_message.call_args
        assert call_args.kwargs["content"] == "Hello, human!"
        assert call_args.kwargs["discord_message_id"] == 987654
        assert call_args.kwargs["channel_id"] == 321
        assert call_args.kwargs["is_bot"] is True
        assert call_args.kwargs["is_proactive"] is False
        assert call_args.kwargs["generation_metadata"] is None
        assert call_args.kwargs["user_timezone"] == "UTC"

    @pytest.mark.asyncio
    async def test_stores_bot_message_with_metadata(self) -> None:
        """Test storing a bot message with generation metadata."""
        message_id = uuid4()
        metadata = {
            "model": "anthropic/claude-3.5-sonnet",
            "tokens": 150,
            "finish_reason": "stop",
        }
        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(return_value=message_id)

        result = await store_bot_message(
            content="Response with metadata",
            discord_message_id=111,
            channel_id=222,
            message_repo=mock_repo,
            generation_metadata=metadata,
        )

        assert result == message_id
        call_args = mock_repo.store_message.call_args
        assert call_args.kwargs["generation_metadata"] == metadata

    @pytest.mark.asyncio
    async def test_stores_proactive_bot_message(self) -> None:
        """Test storing a proactive bot message."""
        message_id = uuid4()
        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(return_value=message_id)

        result = await store_bot_message(
            content="Proactive message",
            discord_message_id=333,
            channel_id=444,
            message_repo=mock_repo,
            is_proactive=True,
        )

        assert result == message_id
        call_args = mock_repo.store_message.call_args
        assert call_args.kwargs["is_proactive"] is True

    @pytest.mark.asyncio
    async def test_stores_bot_message_with_custom_timezone(self) -> None:
        """Test storing a bot message with custom timezone."""
        message_id = uuid4()
        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(return_value=message_id)

        result = await store_bot_message(
            content="Response",
            discord_message_id=555,
            channel_id=666,
            message_repo=mock_repo,
            timezone="Europe/London",
        )

        assert result == message_id
        call_args = mock_repo.store_message.call_args
        assert call_args.kwargs["user_timezone"] == "Europe/London"

    @pytest.mark.asyncio
    async def test_store_bot_message_propagates_db_errors(self) -> None:
        """Test that database errors propagate correctly."""
        mock_repo = MagicMock()
        mock_repo.store_message = AsyncMock(side_effect=Exception("DB error"))

        with pytest.raises(Exception, match="DB error"):
            await store_bot_message(
                content="Test",
                discord_message_id=777,
                channel_id=888,
                message_repo=mock_repo,
            )


class TestRetrieveContext:
    """Tests for retrieve_context function."""

    @pytest.mark.asyncio
    async def test_retrieves_episodic_messages_only_when_depth_zero(self) -> None:
        """Test retrieving only episodic messages when memory_depth is 0."""
        message_id = uuid4()
        mock_messages = [
            EpisodicMessage(
                message_id=message_id,
                content="Recent message",
                discord_message_id=123,
                channel_id=456,
                is_bot=False,
                timestamp=datetime.now(),
            )
        ]

        mock_message_repo = MagicMock()
        mock_message_repo.get_recent_messages = AsyncMock(
            return_value=[
                {
                    "id": message_id,
                    "content": "Recent message",
                    "discord_message_id": 123,
                    "channel_id": 456,
                    "is_bot": False,
                    "is_proactive": False,
                    "timestamp": datetime.now(),
                    "user_timezone": "UTC",
                }
            ]
        )
        mock_semantic_repo = MagicMock()

        recent, semantic = await retrieve_context(
            query="test query",
            channel_id=456,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            episodic_limit=10,
            memory_depth=0,
            entity_names=["Alice", "Bob"],
        )

        assert len(recent) == 1
        assert recent[0].content == "Recent message"
        assert len(semantic) == 0
        mock_message_repo.get_recent_messages.assert_called_once_with(
            channel_id=456, limit=10
        )

    @pytest.mark.asyncio
    async def test_retrieves_episodic_and_semantic_with_entities(self) -> None:
        """Test retrieving both episodic and semantic context with entities."""
        message_id = uuid4()
        mock_message_repo = MagicMock()
        mock_message_repo.get_recent_messages = AsyncMock(
            return_value=[
                {
                    "id": message_id,
                    "content": "Recent",
                    "discord_message_id": 1,
                    "channel_id": 123,
                    "is_bot": False,
                    "is_proactive": False,
                    "timestamp": datetime.now(),
                    "user_timezone": "UTC",
                }
            ]
        )

        mock_semantic_repo = MagicMock()
        mock_semantic_repo.traverse_from_entity = AsyncMock(
            return_value=[
                {
                    "subject": "Alice",
                    "predicate": "likes",
                    "object": "Python",
                    "created_at": datetime.now(),
                }
            ]
        )

        recent, semantic = await retrieve_context(
            query="Tell me about Alice",
            channel_id=123,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            episodic_limit=10,
            memory_depth=1,
            entity_names=["Alice"],
        )

        assert len(recent) == 1
        assert len(semantic) == 1
        assert semantic[0]["subject"] == "Alice"
        assert semantic[0]["predicate"] == "likes"
        assert semantic[0]["object"] == "Python"

    @pytest.mark.asyncio
    async def test_deduplicates_semantic_memories(self) -> None:
        """Test that duplicate semantic memories are filtered out."""
        mock_message_repo = MagicMock()
        mock_message_repo.get_recent_messages = AsyncMock(return_value=[])

        mock_semantic_repo = MagicMock()
        # Two entities return the same memory
        duplicate_memory = {
            "subject": "Alice",
            "predicate": "likes",
            "object": "Python",
            "created_at": datetime.now(),
        }
        # Both entities return the same memory, simulating duplicate
        mock_semantic_repo.traverse_from_entity = AsyncMock(
            return_value=[duplicate_memory]
        )

        recent, semantic = await retrieve_context(
            query="test",
            channel_id=123,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            episodic_limit=10,
            memory_depth=1,
            entity_names=["Alice", "AliceAlias"],
        )

        # Should only have one copy of the duplicate memory
        assert len(semantic) == 1
        assert semantic[0]["subject"] == "Alice"

    @pytest.mark.asyncio
    async def test_skips_memories_with_empty_fields(self) -> None:
        """Test that memories with empty subject/predicate/object are skipped."""
        mock_message_repo = MagicMock()
        mock_message_repo.get_recent_messages = AsyncMock(return_value=[])

        mock_semantic_repo = MagicMock()
        mock_semantic_repo.traverse_from_entity = AsyncMock(
            return_value=[
                {"subject": "", "predicate": "likes", "object": "Python"},
                {"subject": "Alice", "predicate": "", "object": "Python"},
                {"subject": "Alice", "predicate": "likes", "object": ""},
                {"subject": "Bob", "predicate": "knows", "object": "Carol"},
            ]
        )

        recent, semantic = await retrieve_context(
            query="test",
            channel_id=123,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            episodic_limit=10,
            memory_depth=1,
            entity_names=["Alice"],
        )

        # Only the last memory with all fields should be included
        assert len(semantic) == 1
        assert semantic[0]["subject"] == "Bob"

    @pytest.mark.asyncio
    async def test_handles_empty_entity_list(self) -> None:
        """Test that empty entity list returns no semantic context."""
        mock_message_repo = MagicMock()
        mock_message_repo.get_recent_messages = AsyncMock(return_value=[])
        mock_semantic_repo = MagicMock()

        recent, semantic = await retrieve_context(
            query="test",
            channel_id=123,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            episodic_limit=10,
            memory_depth=1,
            entity_names=[],
        )

        assert len(recent) == 0
        assert len(semantic) == 0

    @pytest.mark.asyncio
    async def test_handles_none_entity_list(self) -> None:
        """Test that None entity list returns no semantic context."""
        mock_message_repo = MagicMock()
        mock_message_repo.get_recent_messages = AsyncMock(return_value=[])
        mock_semantic_repo = MagicMock()

        recent, semantic = await retrieve_context(
            query="test",
            channel_id=123,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            episodic_limit=10,
            memory_depth=1,
            entity_names=None,
        )

        assert len(recent) == 0
        assert len(semantic) == 0

    @pytest.mark.asyncio
    async def test_respects_episodic_limit(self) -> None:
        """Test that episodic_limit is passed correctly to get_recent_messages."""
        mock_message_repo = MagicMock()
        mock_message_repo.get_recent_messages = AsyncMock(return_value=[])
        mock_semantic_repo = MagicMock()

        await retrieve_context(
            query="test",
            channel_id=123,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            episodic_limit=25,
            memory_depth=0,
        )

        mock_message_repo.get_recent_messages.assert_called_once_with(
            channel_id=123, limit=25
        )

    @pytest.mark.asyncio
    async def test_concurrent_graph_traversal(self) -> None:
        """Test that graph traversal happens concurrently for multiple entities."""
        mock_message_repo = MagicMock()
        mock_message_repo.get_recent_messages = AsyncMock(return_value=[])

        mock_semantic_repo = MagicMock()

        # Create a traverser that tracks calls
        traversal_calls = []

        async def mock_traverse(
            entity_name: str, max_hops: int
        ) -> list[dict[str, Any]]:
            traversal_calls.append(entity_name)
            await asyncio.sleep(0.01)  # Simulate async work
            return [
                {
                    "subject": entity_name,
                    "predicate": "test",
                    "object": "value",
                }
            ]

        # Patch GraphTraversal
        from unittest.mock import patch

        with patch("lattice.core.memory_orchestrator.GraphTraversal") as mock_graph:
            mock_instance = MagicMock()
            mock_instance.traverse_from_entity = mock_traverse
            mock_graph.return_value = mock_instance

            recent, semantic = await retrieve_context(
                query="test",
                channel_id=123,
                message_repo=mock_message_repo,
                semantic_repo=mock_semantic_repo,
                episodic_limit=10,
                memory_depth=2,
                entity_names=["Alice", "Bob", "Carol"],
            )

        # All entities should have been traversed
        assert len(traversal_calls) == 3
        assert set(traversal_calls) == {"Alice", "Bob", "Carol"}
        # Should get results from all three traversals
        assert len(semantic) == 3

    @pytest.mark.asyncio
    async def test_propagates_message_repo_errors(self) -> None:
        """Test that errors from message repository propagate correctly."""
        mock_message_repo = MagicMock()
        mock_message_repo.get_recent_messages = AsyncMock(
            side_effect=Exception("DB connection failed")
        )
        mock_semantic_repo = MagicMock()

        with pytest.raises(Exception, match="DB connection failed"):
            await retrieve_context(
                query="test",
                channel_id=123,
                message_repo=mock_message_repo,
                semantic_repo=mock_semantic_repo,
            )

    @pytest.mark.asyncio
    async def test_uses_default_episodic_limit(self) -> None:
        """Test that default episodic limit is used when not specified."""
        from lattice.core.constants import DEFAULT_EPISODIC_LIMIT

        mock_message_repo = MagicMock()
        mock_message_repo.get_recent_messages = AsyncMock(return_value=[])
        mock_semantic_repo = MagicMock()

        await retrieve_context(
            query="test",
            channel_id=123,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
        )

        mock_message_repo.get_recent_messages.assert_called_once_with(
            channel_id=123, limit=DEFAULT_EPISODIC_LIMIT
        )
