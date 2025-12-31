"""Unit tests for Lattice memory modules."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from lattice.memory.episodic import (
    EpisodicMessage,
    get_last_message_id,
    get_recent_messages,
    store_message,
)
from lattice.memory.procedural import PromptTemplate, get_prompt, store_prompt
from lattice.memory.semantic import StableFact, search_similar_facts, store_fact
from lattice.memory import feedback_detection


class TestEpisodicMessage:
    """Tests for the EpisodicMessage dataclass."""

    def test_episodic_message_init_defaults(self) -> None:
        """Test EpisodicMessage initialization with default values."""
        message = EpisodicMessage(
            content="Hello world",
            discord_message_id=12345,
            channel_id=67890,
            is_bot=False,
        )

        assert message.content == "Hello world"
        assert message.discord_message_id == 12345
        assert message.channel_id == 67890
        assert message.is_bot is False
        assert message.message_id is None
        assert message.prev_turn_id is None
        assert message.timestamp is not None

    def test_episodic_message_init_with_values(self) -> None:
        """Test EpisodicMessage initialization with all values."""
        custom_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        message_id = UUID("12345678-1234-5678-1234-567812345678")
        prev_id = UUID("87654321-4321-8765-4321-876543218765")

        message = EpisodicMessage(
            content="Test message",
            discord_message_id=11111,
            channel_id=22222,
            is_bot=True,
            message_id=message_id,
            prev_turn_id=prev_id,
            timestamp=custom_time,
        )

        assert message.message_id == message_id
        assert message.prev_turn_id == prev_id
        assert message.timestamp == custom_time


class TestEpisodicMemoryFunctions:
    """Tests for episodic memory functions."""

    @pytest.mark.asyncio
    async def test_store_message(self) -> None:
        """Test storing a message in episodic memory."""
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={"id": UUID("12345678-1234-5678-1234-567812345678")}
        )

        with patch("lattice.memory.episodic.db_pool") as mock_pool:
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            message = EpisodicMessage(
                content="Test message",
                discord_message_id=12345,
                channel_id=67890,
                is_bot=False,
            )

            result = await store_message(message)

            assert result == UUID("12345678-1234-5678-1234-567812345678")
            mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_recent_messages(self) -> None:
        """Test retrieving recent messages from a channel."""
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": UUID("22222222-2222-2222-2222-222222222222"),
                    "discord_message_id": 101,
                    "channel_id": 67890,
                    "content": "Second message",
                    "is_bot": True,
                    "prev_turn_id": UUID("11111111-1111-1111-1111-111111111111"),
                    "timestamp": datetime(2024, 1, 1, 10, 5, 0, tzinfo=UTC),
                },
                {
                    "id": UUID("11111111-1111-1111-1111-111111111111"),
                    "discord_message_id": 100,
                    "channel_id": 67890,
                    "content": "First message",
                    "is_bot": False,
                    "prev_turn_id": None,
                    "timestamp": datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
                },
            ]
        )

        with patch("lattice.memory.episodic.db_pool") as mock_pool:
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            messages = await get_recent_messages(channel_id=67890, limit=10)

            assert len(messages) == 2
            assert messages[0].content == "First message"
            assert messages[0].timestamp < messages[1].timestamp

    @pytest.mark.asyncio
    async def test_get_last_message_id_exists(self) -> None:
        """Test getting the last message ID when messages exist."""
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={"id": UUID("12345678-1234-5678-1234-567812345678")}
        )

        with patch("lattice.memory.episodic.db_pool") as mock_pool:
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            result = await get_last_message_id(channel_id=67890)

            assert result == UUID("12345678-1234-5678-1234-567812345678")

    @pytest.mark.asyncio
    async def test_get_last_message_id_empty(self) -> None:
        """Test getting the last message ID when no messages exist."""
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        with patch("lattice.memory.episodic.db_pool") as mock_pool:
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            result = await get_last_message_id(channel_id=67890)

            assert result is None


class TestStableFact:
    """Tests for the StableFact dataclass."""

    def test_stable_fact_init_defaults(self) -> None:
        """Test StableFact initialization with default values."""
        fact = StableFact(content="Test fact content")

        assert fact.content == "Test fact content"
        assert fact.fact_id is None
        assert fact.embedding is None
        assert fact.origin_id is None
        assert fact.entity_type is None

    def test_stable_fact_init_with_values(self) -> None:
        """Test StableFact initialization with all values."""
        fact_id = UUID("12345678-1234-5678-1234-567812345678")
        origin_id = UUID("87654321-4321-8765-4321-876543218765")
        embedding = [0.1, 0.2, 0.3, 0.4]

        fact = StableFact(
            content="User prefers dark mode",
            fact_id=fact_id,
            embedding=embedding,
            origin_id=origin_id,
            entity_type="preference",
        )

        assert fact.fact_id == fact_id
        assert fact.embedding == embedding
        assert fact.origin_id == origin_id
        assert fact.entity_type == "preference"


class TestSemanticMemoryFunctions:
    """Tests for semantic memory functions."""

    @pytest.mark.asyncio
    async def test_store_fact_generates_embedding(self) -> None:
        """Test storing a fact generates embedding if not provided."""
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={"id": UUID("12345678-1234-5678-1234-567812345678")}
        )

        with (
            patch("lattice.memory.semantic.db_pool") as mock_pool,
            patch("lattice.memory.semantic.embedding_model") as mock_embed,
        ):
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_embed.encode_single.return_value = [0.1, 0.2, 0.3]

            fact = StableFact(content="Test fact")
            result = await store_fact(fact)

            assert result == UUID("12345678-1234-5678-1234-567812345678")
            mock_embed.encode_single.assert_called_once_with("Test fact")

    @pytest.mark.asyncio
    async def test_store_fact_with_existing_embedding(self) -> None:
        """Test storing a fact uses existing embedding."""
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={"id": UUID("12345678-1234-5678-1234-567812345678")}
        )

        with (
            patch("lattice.memory.semantic.db_pool") as mock_pool,
            patch("lattice.memory.semantic.embedding_model") as mock_embed,
        ):
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            fact = StableFact(content="Test fact", embedding=[0.5, 0.6, 0.7])
            result = await store_fact(fact)

            assert result == UUID("12345678-1234-5678-1234-567812345678")
            mock_embed.encode_single.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_similar_facts_empty_query(self) -> None:
        """Test that empty query returns empty list."""
        with patch("lattice.memory.semantic.embedding_model") as mock_embed:
            result = await search_similar_facts(query="", limit=5)

            assert result == []
            mock_embed.encode_single.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_similar_facts_whitespace_query(self) -> None:
        """Test that whitespace-only query returns empty list."""
        with patch("lattice.memory.semantic.embedding_model") as mock_embed:
            result = await search_similar_facts(query="   ", limit=5)

            assert result == []
            mock_embed.encode_single.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_similar_facts_invalid_limit(self) -> None:
        """Test that invalid limit raises ValueError."""
        with pytest.raises(ValueError, match="limit must be between 1 and 100"):
            await search_similar_facts(query="test", limit=0)

    @pytest.mark.asyncio
    async def test_search_similar_facts_invalid_threshold(self) -> None:
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match=r"similarity_threshold must be between 0\.0 and 1\.0"):
            await search_similar_facts(query="test", limit=5, similarity_threshold=1.5)


class TestPromptTemplate:
    """Tests for the PromptTemplate dataclass."""

    def test_prompt_template_init_defaults(self) -> None:
        """Test PromptTemplate initialization with default values."""
        prompt = PromptTemplate(prompt_key="test_key", template="You are helpful.")

        assert prompt.prompt_key == "test_key"
        assert prompt.template == "You are helpful."
        assert prompt.temperature == 0.7
        assert prompt.version == 1
        assert prompt.active is True

    def test_prompt_template_init_with_values(self) -> None:
        """Test PromptTemplate initialization with all values."""
        prompt = PromptTemplate(
            prompt_key="custom_key",
            template="Respond with: {response}",
            temperature=0.5,
            version=3,
            active=False,
        )

        assert prompt.prompt_key == "custom_key"
        assert prompt.template == "Respond with: {response}"
        assert prompt.temperature == 0.5
        assert prompt.version == 3
        assert prompt.active is False


class TestProceduralMemoryFunctions:
    """Tests for procedural memory functions."""

    @pytest.mark.asyncio
    async def test_get_prompt_found(self) -> None:
        """Test retrieving an existing prompt template."""
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "prompt_key": "test_key",
                "template": "You are helpful.",
                "temperature": 0.7,
                "version": 1,
                "active": True,
            }
        )

        with patch("lattice.memory.procedural.db_pool") as mock_pool:
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            result = await get_prompt("test_key")

            assert result is not None
            assert result.prompt_key == "test_key"
            assert result.template == "You are helpful."

    @pytest.mark.asyncio
    async def test_get_prompt_not_found(self) -> None:
        """Test retrieving a non-existent prompt template."""
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        with patch("lattice.memory.procedural.db_pool") as mock_pool:
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            result = await get_prompt("nonexistent")

            assert result is None

    @pytest.mark.asyncio
    async def test_store_prompt(self) -> None:
        """Test storing a prompt template."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()

        with patch("lattice.memory.procedural.db_pool") as mock_pool:
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            prompt = PromptTemplate(prompt_key="new_key", template="New template")
            await store_prompt(prompt)

            mock_conn.execute.assert_called_once()


class MockMessage:
    """Mock Discord message for testing detection functions."""

    def __init__(
        self,
        content: str = "",
        author_bot: bool = False,
        author_name: str = "testuser",
        is_reply: bool = False,
    ) -> None:
        self.content = content
        self.author = MockUser(bot=author_bot, name=author_name)
        self.reference = MockReference() if is_reply else None


class MockUser:
    """Mock Discord user."""

    def __init__(self, bot: bool = False, name: str = "testuser") -> None:
        self.bot = bot
        self.name = name


class MockReference:  # noqa: N801
    """Mock Discord message reference."""

    def __init__(self) -> None:
        self.message_id: int | None = None


class TestFeedbackDetectionFunctions:
    """Tests for feedback detection functions."""

    def test_is_north_star_with_north_star_pattern(self) -> None:
        """Test detection of North Star with 'north star' pattern."""
        message = MockMessage(content="North Star: I want to learn Rust")
        result = feedback_detection.is_north_star(message)

        assert result.detected is True
        assert result.content == "I want to learn Rust"
        assert result.confidence > 0.7

    def test_is_north_star_with_goal_pattern(self) -> None:
        """Test detection of North Star with 'goal' pattern."""
        message = MockMessage(content="My goal is to build a web app")
        result = feedback_detection.is_north_star(message)

        assert result.detected is True
        assert result.content == "to build a web app"
        assert result.confidence > 0.7

    def test_is_north_star_with_trying_pattern(self) -> None:
        """Test detection of North Star with 'trying' pattern."""
        message = MockMessage(content="I'm trying to lose weight")
        result = feedback_detection.is_north_star(message)

        assert result.detected is True
        assert result.content == "lose weight"
        assert result.confidence > 0.7

    def test_is_north_star_with_focused_pattern(self) -> None:
        """Test detection of North Star with 'focused' pattern."""
        message = MockMessage(content="I'm focused on improving my coding skills")
        result = feedback_detection.is_north_star(message)

        assert result.detected is True
        assert result.content == "improving my coding skills"
        assert result.confidence > 0.7

    def test_is_north_star_with_building_pattern(self) -> None:
        """Test detection of North Star with 'building' pattern."""
        message = MockMessage(content="I'm building a startup")
        result = feedback_detection.is_north_star(message)

        assert result.detected is True
        assert result.content == "a startup"
        assert result.confidence > 0.7

    def test_is_north_star_with_learning_pattern(self) -> None:
        """Test detection of North Star with 'learning' pattern."""
        message = MockMessage(content="I'm learning Japanese")
        result = feedback_detection.is_north_star(message)

        assert result.detected is True
        assert result.content == "Japanese"
        assert result.confidence > 0.7

    def test_is_north_star_with_want_pattern(self) -> None:
        """Test detection of North Star with 'want' pattern."""
        message = MockMessage(content="I want to travel the world")
        result = feedback_detection.is_north_star(message)

        assert result.detected is True
        assert result.content == "travel the world"
        assert result.confidence > 0.7

    def test_is_north_star_bot_message(self) -> None:
        """Test that bot messages are not detected as North Star."""
        message = MockMessage(content="North Star: test", author_bot=True)
        result = feedback_detection.is_north_star(message)

        assert result.detected is False
        assert result.content is None

    def test_is_north_star_regular_message(self) -> None:
        """Test that regular messages are not detected as North Star."""
        message = MockMessage(content="Hello, how are you?")
        result = feedback_detection.is_north_star(message)

        assert result.detected is False

    def test_is_north_star_too_short(self) -> None:
        """Test that too short messages are not detected as North Star."""
        message = MockMessage(content="Hi")
        result = feedback_detection.is_north_star(message)

        assert result.detected is False

    def test_is_north_star_too_long(self) -> None:
        """Test that too long messages are not detected as North Star."""
        long_content = "a" * 600
        message = MockMessage(content=long_content)
        result = feedback_detection.is_north_star(message)

        assert result.detected is False

    def test_is_quote_or_reply_with_reference(self) -> None:
        """Test that message with reference is detected as quote/reply."""
        message = MockMessage(content="Your response was helpful", is_reply=True)
        result = feedback_detection.is_quote_or_reply(message)

        assert result is True

    def test_is_quote_or_reply_without_reference(self) -> None:
        """Test that message without reference is not detected as quote/reply."""
        message = MockMessage(content="Hello world")
        result = feedback_detection.is_quote_or_reply(message)

        assert result is False

    def test_get_referenced_message_id_with_reference(self) -> None:
        """Test getting referenced message ID."""
        message = MockMessage(content="Feedback", is_reply=True)
        message.reference = MockReference()
        message.reference.message_id = 12345

        result = feedback_detection.get_referenced_message_id(message)

        assert result == 12345

    def test_get_referenced_message_id_without_reference(self) -> None:
        """Test getting referenced message ID when no reference exists."""
        message = MockMessage(content="Hello")

        result = feedback_detection.get_referenced_message_id(message)

        assert result is None

    def test_is_invisible_feedback_with_reply(self) -> None:
        """Test detection of invisible feedback via reply."""
        message = MockMessage(content="Actually, that's wrong", is_reply=True)
        result = feedback_detection.is_invisible_feedback(message)

        assert result.detected is True
        assert result.content == "Actually, that's wrong"

    def test_is_invisible_feedback_bot_message(self) -> None:
        """Test that bot messages are not detected as invisible feedback."""
        message = MockMessage(content="Feedback: test", author_bot=True)
        result = feedback_detection.is_invisible_feedback(message)

        assert result.detected is False

    def test_is_invisible_feedback_regular_message(self) -> None:
        """Test that regular messages without quote/reply are not detected."""
        message = MockMessage(content="Hello, how are you?")
        result = feedback_detection.is_invisible_feedback(message)

        assert result.detected is False

    def test_extract_goal_from_content_north_star(self) -> None:
        """Test extracting goal from North Star pattern."""
        content = "North Star: I want to learn Python"
        result = feedback_detection.extract_goal_from_content(content)

        assert result == "I want to learn Python"

    def test_extract_goal_from_content_goal(self) -> None:
        """Test extracting goal from goal pattern."""
        content = "My goal is to finish this project"
        result = feedback_detection.extract_goal_from_content(content)

        assert result == "to finish this project"

    def test_extract_goal_from_content_no_match(self) -> None:
        """Test that non-goal content returns None."""
        content = "Hello, how are you?"
        result = feedback_detection.extract_goal_from_content(content)

        assert result is None
