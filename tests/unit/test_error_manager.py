"""Unit tests for Discord error manager module."""

from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from lattice.discord_client.error_manager import ErrorManager


class TestErrorManager:
    """Tests for ErrorManager class."""

    def test_initialization(self) -> None:
        """Test ErrorManager initialization."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=123)

        assert error_manager.bot == mock_bot
        assert error_manager.dream_channel_id == 123

    @pytest.mark.asyncio
    async def test_on_error_no_dream_channel(self) -> None:
        """Test error handling when dream channel is not configured."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=0)

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("Test error")
            except ValueError:
                await error_manager.on_error("on_message")

        mock_bot.get_channel.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_error_get_channel_returns_none(self) -> None:
        """Test error handling when get_channel returns None."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)
        mock_bot.get_channel.return_value = None

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("Test error")
            except ValueError:
                await error_manager.on_error("on_message")

        mock_bot.get_channel.assert_called_once_with(456)

    @pytest.mark.asyncio
    async def test_on_error_channel_not_text_channel(self) -> None:
        """Test error handling when channel is not a TextChannel."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_channel = MagicMock()
        mock_channel.__class__ = MagicMock
        mock_bot.get_channel.return_value = mock_channel

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("Test error")
            except ValueError:
                await error_manager.on_error("on_message")

        mock_bot.get_channel.assert_called_once_with(456)

    @pytest.mark.asyncio
    async def test_on_error_successfully_reports_to_dream_channel(self) -> None:
        """Test successful error reporting to dream channel."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("This is a test error")
            except ValueError:
                await error_manager.on_error("on_message", "arg1", kwarg1="value1")

        mock_dream_channel.send.assert_called_once()
        call_args = mock_dream_channel.send.call_args
        assert call_args[0][0] == "🚨🚨🚨 **SYSTEM ERROR** 🚨🚨🚨"
        embed = call_args[1]["embed"]
        assert "CRASH DETECTED" in embed.title
        assert "on_message" in embed.description

    @pytest.mark.asyncio
    async def test_on_error_embed_contains_error_type(self) -> None:
        """Test that embed contains the error type."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("This is a test error")
            except ValueError:
                await error_manager.on_error("test_event")

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        error_type_field = embed.fields[0]
        assert "ValueError" in error_type_field.value

    @pytest.mark.asyncio
    async def test_on_error_embed_contains_error_message(self) -> None:
        """Test that embed contains the error message."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("Test error message")
            except ValueError:
                await error_manager.on_error("test_event")

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        error_message_field = embed.fields[1]
        assert "Test error message" in error_message_field.value

    @pytest.mark.asyncio
    async def test_on_error_embed_contains_traceback(self) -> None:
        """Test that embed contains traceback information."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("Test error")
            except ValueError:
                await error_manager.on_error("test_event")

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        traceback_field = embed.fields[2]
        assert "ValueError" in traceback_field.value
        assert "Test error" in traceback_field.value

    @pytest.mark.asyncio
    async def test_on_error_embed_uses_event_name(self) -> None:
        """Test that embed description includes the event method name."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise RuntimeError("Something went wrong")
            except RuntimeError:
                await error_manager.on_error("on_interaction", 123, "arg2")

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        assert "on_interaction" in embed.description

    @pytest.mark.asyncio
    async def test_on_error_embed_footer_contains_event(self) -> None:
        """Test that embed footer contains the event name."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise Exception("Test")
            except BaseException:
                await error_manager.on_error("on_ready")

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        assert "Event: on_ready" in embed.footer.text

    @pytest.mark.asyncio
    async def test_on_error_handles_discord_exception(self) -> None:
        """Test that Discord exceptions during send are handled gracefully."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock(
            side_effect=discord.DiscordException("Failed to send")
        )

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("Test error")
            except ValueError:
                await error_manager.on_error("test_event")

        mock_dream_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_error_truncates_long_error_message(self) -> None:
        """Test that long error messages are truncated for Discord limits."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        long_message = "x" * 600
        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError(long_message)
            except ValueError:
                await error_manager.on_error("test_event")

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        error_message_field = embed.fields[1]
        # 500 chars + 6 for code block markers (```) = 506 max
        assert len(error_message_field.value) <= 506

    @pytest.mark.asyncio
    async def test_on_error_truncates_long_traceback(self) -> None:
        """Test that long tracebacks are truncated for Discord limits."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("Short error")
            except ValueError:
                await error_manager.on_error("test_event")

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        traceback_field = embed.fields[2]
        assert len(traceback_field.value) <= 1803  # 1800 + "```" code block

    @pytest.mark.asyncio
    async def test_on_error_with_different_exception_types(self) -> None:
        """Test error handling with different exception types."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)

        exception_types = [
            (ValueError, "Value error"),
            (TypeError, "Type error"),
            (KeyError, "Key error"),
            (RuntimeError, "Runtime error"),
            (AttributeError, "Attribute error"),
        ]

        for exc_type, exc_msg in exception_types:
            mock_bot.get_channel.return_value = mock_dream_channel
            mock_dream_channel.send.reset_mock()
            with patch("lattice.discord_client.error_manager.logger"):
                try:
                    raise exc_type(exc_msg)
                except Exception:
                    await error_manager.on_error("test_event")

            call_args = mock_dream_channel.send.call_args
            embed = call_args[1]["embed"]
            error_type_field = embed.fields[0]
            assert exc_type.__name__ in error_type_field.value

    @pytest.mark.asyncio
    async def test_on_error_embed_color_is_red(self) -> None:
        """Test that error embed uses red color."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("Test")
            except ValueError:
                await error_manager.on_error("test_event")

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        assert embed.color == discord.Color.red()

    @pytest.mark.asyncio
    async def test_on_error_with_event_args(self) -> None:
        """Test that on_error accepts event args and kwargs."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("Test")
            except ValueError:
                await error_manager.on_error("on_message", "arg1", "arg2", key="value")

        mock_dream_channel.send.assert_called_once()


class TestErrorManagerEdgeCases:
    """Tests for edge cases in ErrorManager."""

    def test_initialization(self) -> None:
        """Test ErrorManager initialization."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=123)

        assert error_manager.bot == mock_bot
        assert error_manager.dream_channel_id == 123

    @pytest.mark.asyncio
    async def test_on_error_empty_event_method(self) -> None:
        """Test on_error with empty event method name."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("Test")
            except ValueError:
                await error_manager.on_error("")

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        assert "``" in embed.description  # Empty string in backticks

    @pytest.mark.asyncio
    async def test_on_error_very_long_event_name(self) -> None:
        """Test on_error with very long event method name."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(bot=mock_bot, dream_channel_id=456)

        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        long_event_name = "very_long_event_method_name_that_exceeds_normal_length"

        with patch("lattice.discord_client.error_manager.logger"):
            try:
                raise ValueError("Test")
            except ValueError:
                await error_manager.on_error(long_event_name)

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        assert long_event_name in embed.description
        assert long_event_name in embed.footer.text
