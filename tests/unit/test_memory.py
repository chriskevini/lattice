"""Unit tests for Lattice memory modules."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from lattice.memory.episodic import (
    EpisodicMessage,
    get_recent_messages,
    store_message,
)
from lattice.memory.procedural import PromptTemplate, get_prompt, store_prompt
from lattice.memory.semantic import StableFact, search_similar_facts


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
        assert message.is_proactive is False
        assert message.timestamp is not None

    def test_episodic_message_init_with_values(self) -> None:
        """Test EpisodicMessage initialization with all values."""
        custom_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        message_id = UUID("12345678-1234-5678-1234-567812345678")

        message = EpisodicMessage(
            content="Test message",
            discord_message_id=11111,
            channel_id=22222,
            is_bot=True,
            message_id=message_id,
            is_proactive=True,
            timestamp=custom_time,
        )

        assert message.message_id == message_id
        assert message.is_proactive is True
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
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
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
                    "is_proactive": False,
                    "timestamp": datetime(2024, 1, 1, 10, 5, 0, tzinfo=UTC),
                },
                {
                    "id": UUID("11111111-1111-1111-1111-111111111111"),
                    "discord_message_id": 100,
                    "channel_id": 67890,
                    "content": "First message",
                    "is_bot": False,
                    "is_proactive": False,
                    "timestamp": datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
                },
            ]
        )

        with patch("lattice.memory.episodic.db_pool") as mock_pool:
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            messages = await get_recent_messages(channel_id=67890, limit=10)

            assert len(messages) == 2
            assert messages[0].content == "First message"
            assert messages[0].timestamp < messages[1].timestamp


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

    @pytest.mark.asyncio
    async def test_search_similar_facts_empty_query(self) -> None:
        """Test that empty query returns empty list."""
        # Stub implementation always returns empty list
        result = await search_similar_facts(query="", limit=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_search_similar_facts_whitespace_query(self) -> None:
        """Test that whitespace query returns empty list."""
        # Stub implementation always returns empty list
        result = await search_similar_facts(query="   ", limit=5)
        assert result == []


# NOTE: Additional semantic memory function tests removed during Issue #61 refactor
# Semantic memory is being rewritten with graph-first architecture
# New tests will be added in Phase 1 of Issue #61


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
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
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
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            result = await get_prompt("nonexistent")

            assert result is None

    @pytest.mark.asyncio
    async def test_store_prompt(self) -> None:
        """Test storing a prompt template."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()

        with patch("lattice.memory.procedural.db_pool") as mock_pool:
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
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
        channel_id: int = 12345,
    ) -> None:
        self.content = content
        self.author = MockUser(bot=author_bot, name=author_name)
        self.reference = MockReference() if is_reply else None
        self.channel = MockChannel(channel_id=channel_id)


class MockUser:
    """Mock Discord user."""

    def __init__(self, bot: bool = False, name: str = "testuser") -> None:
        self.bot = bot
        self.name = name


class MockReference:
    """Mock Discord message reference."""

    def __init__(self) -> None:
        self.message_id: int | None = None


class MockChannel:
    """Mock Discord channel."""

    def __init__(self, channel_id: int = 12345) -> None:
        self.id = channel_id


class TestMemoryFunctions:
    """Tests for memory functions."""
