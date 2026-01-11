"""Unit tests for UnifiedPipeline message sending."""

from unittest.mock import AsyncMock, MagicMock

import discord
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
        """Test send_response when channel doesn't exist.

        Verifies that None is returned when channel not found.
        """
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
    async def test_dispatch_autonomous_nudge_success(self) -> None:
        """Test successful autonomous nudge sending."""
        mock_db_pool = MagicMock()
        mock_bot = MagicMock()
        mock_channel = MagicMock()
        mock_message = MagicMock()

        # Setup bot to return channel
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock(return_value=mock_message)

        pipeline = UnifiedPipeline(db_pool=mock_db_pool, bot=mock_bot)

        result = await pipeline.dispatch_autonomous_nudge(
            content="Nudge reminder: Don't forget your meeting!",
            channel_id=123456789,
        )

        # Verify channel was fetched
        mock_bot.get_channel.assert_called_once_with(123456789)

        # Verify message was sent
        mock_channel.send.assert_called_once_with(
            "Nudge reminder: Don't forget your meeting!"
        )

        # Verify result is the sent message
        assert result is mock_message

    @pytest.mark.asyncio
    async def test_dispatch_autonomous_nudge_channel_not_found(self) -> None:
        """Test dispatch_autonomous_nudge when channel doesn't exist."""
        mock_db_pool = MagicMock()
        mock_bot = MagicMock()

        # Setup bot to return None (channel not found)
        mock_bot.get_channel.return_value = None

        pipeline = UnifiedPipeline(db_pool=mock_db_pool, bot=mock_bot)

        result = await pipeline.dispatch_autonomous_nudge(
            content="Nudge message",
            channel_id=999999999,
        )

        # Verify channel was attempted
        mock_bot.get_channel.assert_called_once_with(999999999)

        # Verify None returned
        assert result is None

    @pytest.mark.asyncio
    async def test_send_response_http_exception(self) -> None:
        """Test that HTTPException from Discord API is propagated.

        When Discord API returns an HTTP error (e.g., rate limit, server error),
        the exception should propagate to the caller for proper error handling.
        """
        mock_db_pool = MagicMock()
        mock_bot = MagicMock()
        mock_channel = MagicMock()

        mock_bot.get_channel.return_value = mock_channel
        # Simulate Discord API error
        mock_channel.send = AsyncMock(
            side_effect=discord.HTTPException(
                response=MagicMock(status=500), message="Internal Server Error"
            )
        )

        pipeline = UnifiedPipeline(db_pool=mock_db_pool, bot=mock_bot)

        # Verify exception propagates
        with pytest.raises(discord.HTTPException):
            await pipeline.send_response(
                channel_id=123456789,
                content="Test message",
            )

    @pytest.mark.asyncio
    async def test_send_response_forbidden(self) -> None:
        """Test that Forbidden exception from Discord API is propagated.

        When the bot lacks permissions to send messages in a channel,
        Discord raises Forbidden. This should propagate to the caller.
        """
        mock_db_pool = MagicMock()
        mock_bot = MagicMock()
        mock_channel = MagicMock()

        mock_bot.get_channel.return_value = mock_channel
        # Simulate permission error
        mock_channel.send = AsyncMock(
            side_effect=discord.Forbidden(
                response=MagicMock(status=403), message="Missing Permissions"
            )
        )

        pipeline = UnifiedPipeline(db_pool=mock_db_pool, bot=mock_bot)

        # Verify exception propagates
        with pytest.raises(discord.Forbidden):
            await pipeline.send_response(
                channel_id=123456789,
                content="Test message",
            )
