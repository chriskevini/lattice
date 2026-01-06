"""Unit tests for UnifiedPipeline message sending."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice.core.pipeline import UnifiedPipeline


class TestUnifiedPipeline:
    """Tests for UnifiedPipeline class."""

    def test_pipeline_initialization(self) -> None:
        """Test UnifiedPipeline initialization."""
        mock_db_pool = MagicMock()
        mock_bot = MagicMock()

        pipeline = UnifiedPipeline(db_pool=mock_db_pool, bot=mock_bot)

        assert pipeline.db_pool is mock_db_pool
        assert pipeline.bot is mock_bot

    @pytest.mark.asyncio
    async def test_send_response_success(self) -> None:
        """Test successful message sending."""
        mock_db_pool = MagicMock()
        mock_bot = MagicMock()
        mock_channel = MagicMock()
        mock_message = MagicMock()

        # Setup bot to return channel
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock(return_value=mock_message)

        pipeline = UnifiedPipeline(db_pool=mock_db_pool, bot=mock_bot)

        result = await pipeline.send_response(
            channel_id=123456789,
            content="Hello, world!",
        )

        # Verify channel was fetched
        mock_bot.get_channel.assert_called_once_with(123456789)

        # Verify message was sent
        mock_channel.send.assert_called_once_with("Hello, world!")

        # Verify result is the sent message
        assert result is mock_message

    @pytest.mark.asyncio
    async def test_send_response_channel_not_found(self) -> None:
        """Test send_response when channel doesn't exist."""
        mock_db_pool = MagicMock()
        mock_bot = MagicMock()

        # Setup bot to return None (channel not found)
        mock_bot.get_channel.return_value = None

        pipeline = UnifiedPipeline(db_pool=mock_db_pool, bot=mock_bot)

        result = await pipeline.send_response(
            channel_id=999999999,
            content="Test message",
        )

        # Verify channel was attempted
        mock_bot.get_channel.assert_called_once_with(999999999)

        # Verify None returned
        assert result is None

    @pytest.mark.asyncio
    async def test_send_proactive_message_success(self) -> None:
        """Test successful proactive message sending."""
        mock_db_pool = MagicMock()
        mock_bot = MagicMock()
        mock_channel = MagicMock()
        mock_message = MagicMock()

        # Setup bot to return channel
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock(return_value=mock_message)

        pipeline = UnifiedPipeline(db_pool=mock_db_pool, bot=mock_bot)

        result = await pipeline.send_proactive_message(
            content="Proactive reminder: Don't forget your meeting!",
            channel_id=123456789,
        )

        # Verify channel was fetched
        mock_bot.get_channel.assert_called_once_with(123456789)

        # Verify message was sent
        mock_channel.send.assert_called_once_with(
            "Proactive reminder: Don't forget your meeting!"
        )

        # Verify result is the sent message
        assert result is mock_message

    @pytest.mark.asyncio
    async def test_send_proactive_message_channel_not_found(self) -> None:
        """Test send_proactive_message when channel doesn't exist."""
        mock_db_pool = MagicMock()
        mock_bot = MagicMock()

        # Setup bot to return None (channel not found)
        mock_bot.get_channel.return_value = None

        pipeline = UnifiedPipeline(db_pool=mock_db_pool, bot=mock_bot)

        result = await pipeline.send_proactive_message(
            content="Proactive message",
            channel_id=999999999,
        )

        # Verify channel was attempted
        mock_bot.get_channel.assert_called_once_with(999999999)

        # Verify None returned
        assert result is None

    @pytest.mark.asyncio
    async def test_send_response_empty_content(self) -> None:
        """Test sending empty content."""
        mock_db_pool = MagicMock()
        mock_bot = MagicMock()
        mock_channel = MagicMock()
        mock_message = MagicMock()

        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock(return_value=mock_message)

        pipeline = UnifiedPipeline(db_pool=mock_db_pool, bot=mock_bot)

        result = await pipeline.send_response(
            channel_id=123456789,
            content="",
        )

        # Verify empty string was sent
        mock_channel.send.assert_called_once_with("")

        # Verify result is the sent message
        assert result is mock_message

    @pytest.mark.asyncio
    async def test_send_response_long_content(self) -> None:
        """Test sending long content (no splitting in this utility)."""
        mock_db_pool = MagicMock()
        mock_bot = MagicMock()
        mock_channel = MagicMock()
        mock_message = MagicMock()

        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock(return_value=mock_message)

        pipeline = UnifiedPipeline(db_pool=mock_db_pool, bot=mock_bot)

        long_content = "A" * 3000  # Longer than Discord's 2000 char limit

        result = await pipeline.send_response(
            channel_id=123456789,
            content=long_content,
        )

        # Verify content was passed as-is (no splitting here)
        mock_channel.send.assert_called_once_with(long_content)

        # Verify result is the sent message
        assert result is mock_message
