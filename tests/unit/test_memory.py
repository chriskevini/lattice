"""Unit tests for Lattice memory modules."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from lattice.memory.episodic import (
    EpisodicMessage,
    get_recent_messages,
    store_message,
)
from lattice.memory.procedural import PromptTemplate, get_prompt


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
        mock_repo = AsyncMock()
        mock_repo.store_message.return_value = UUID(
            "12345678-1234-5678-1234-567812345678"
        )

        message = EpisodicMessage(
            content="Test message",
            discord_message_id=12345,
            channel_id=67890,
            is_bot=False,
        )

        result = await store_message(repo=mock_repo, message=message)

        assert result == UUID("12345678-1234-5678-1234-567812345678")
        mock_repo.store_message.assert_called_once_with(
            content="Test message",
            discord_message_id=12345,
            channel_id=67890,
            is_bot=False,
            is_proactive=False,
            generation_metadata=None,
            user_timezone="UTC",
        )

    @pytest.mark.asyncio
    async def test_get_recent_messages(self) -> None:
        """Test retrieving recent messages from a channel."""
        mock_repo = AsyncMock()
        mock_repo.get_recent_messages.return_value = [
            {
                "id": UUID("22222222-2222-2222-2222-222222222222"),
                "discord_message_id": 101,
                "channel_id": 67890,
                "content": "Second message",
                "is_bot": True,
                "is_proactive": False,
                "timestamp": datetime(2024, 1, 1, 10, 5, 0, tzinfo=UTC),
                "user_timezone": "UTC",
            },
            {
                "id": UUID("11111111-1111-1111-1111-111111111111"),
                "discord_message_id": 100,
                "channel_id": 67890,
                "content": "First message",
                "is_bot": False,
                "is_proactive": False,
                "timestamp": datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
                "user_timezone": "UTC",
            },
        ]

        messages = await get_recent_messages(repo=mock_repo, channel_id=67890, limit=10)

        assert len(messages) == 2
        # get_recent_messages reverses the order from repo (which is newest first)
        # to oldest first for the LLM context.
        assert messages[0].content == "First message"
        assert messages[1].content == "Second message"
        assert messages[0].timestamp < messages[1].timestamp


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

    def test_safe_format_basic(self) -> None:
        """Test safe_format with simple substitution."""
        prompt = PromptTemplate(prompt_key="test", template="Hello {name}!")
        result = prompt.safe_format(name="World")
        assert result == "Hello World!"

    def test_safe_format_preserves_unknown_placeholders(self) -> None:
        """Test safe_format preserves placeholders that aren't provided."""
        prompt = PromptTemplate(
            prompt_key="test",
            template="Hello {name}! Example: {example_placeholder}",
        )
        result = prompt.safe_format(name="World")
        assert result == "Hello World! Example: {example_placeholder}"

    def test_safe_format_multiple_variables(self) -> None:
        """Test safe_format with multiple known and unknown placeholders."""
        prompt = PromptTemplate(
            prompt_key="test",
            template="Message: {content}\nContext: {context}\nExample: {yesterday_date}",
        )
        result = prompt.safe_format(content="Hello", context="Test")
        assert result == "Message: Hello\nContext: Test\nExample: {yesterday_date}"

    def test_safe_format_no_placeholders(self) -> None:
        """Test safe_format with template that has no placeholders."""
        prompt = PromptTemplate(
            prompt_key="test", template="This is a static template."
        )
        result = prompt.safe_format()
        assert result == "This is a static template."

    def test_safe_format_nested_braces(self) -> None:
        """Test safe_format with JSON examples containing nested braces."""
        prompt = PromptTemplate(
            prompt_key="test",
            template='Respond: {instruction}\nFormat: {{"key": "{value}"}}',
        )
        result = prompt.safe_format(instruction="Be helpful")
        # The JSON example braces should be preserved
        assert result == 'Respond: Be helpful\nFormat: {{"key": "{value}"}}'

    def test_safe_format_real_context_strategy_example(self) -> None:
        """Test safe_format with real-world QUERY_EXTRACTION template snippet."""
        template_snippet = (
            "User Message: {message_content}\n"
            "Context: {context}\n"
            "Example: 'what did I do yesterday?' → "
            "'What activities did the user complete on {yesterday_date}?'"
        )
        prompt = PromptTemplate(prompt_key="test", template=template_snippet)
        result = prompt.safe_format(
            message_content="test message", context="test context"
        )
        expected = (
            "User Message: test message\n"
            "Context: test context\n"
            "Example: 'what did I do yesterday?' → "
            "'What activities did the user complete on {yesterday_date}?'"
        )
        assert result == expected


class TestProceduralMemoryFunctions:
    """Tests for procedural memory functions."""

    @pytest.mark.asyncio
    async def test_get_prompt_found(self) -> None:
        """Test retrieving an existing prompt template."""
        mock_repo = AsyncMock()
        mock_repo.get_template.return_value = {
            "prompt_key": "test_key",
            "template": "You are helpful.",
            "temperature": 0.7,
            "version": 1,
            "active": True,
        }

        result = await get_prompt(repo=mock_repo, prompt_key="test_key")

        assert result is not None
        assert result.prompt_key == "test_key"
        assert result.template == "You are helpful."

    @pytest.mark.asyncio
    async def test_get_prompt_not_found(self) -> None:
        """Test retrieving a non-existent prompt template."""
        mock_repo = AsyncMock()
        mock_repo.get_template.return_value = None

        result = await get_prompt(repo=mock_repo, prompt_key="nonexistent")

        assert result is None


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
