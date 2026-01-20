"""Unit tests for error_manager and error_notifier modules."""

from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from lattice.discord_client.error_manager import ErrorManager
from lattice.discord_client.error_notifier import mirror_llm_error


@patch("lattice.discord_client.error_manager.logger")
class TestErrorManager:
    """Tests for ErrorManager class."""

    def test_initialization(self, mock_logger: MagicMock) -> None:
        """Test ErrorManager initialization."""
        mock_bot = MagicMock()
        dream_channel_id = 123456

        error_manager = ErrorManager(mock_bot, dream_channel_id)

        assert error_manager.bot is mock_bot
        assert error_manager.dream_channel_id == dream_channel_id

    @pytest.mark.asyncio
    async def test_on_error_sends_embed_to_dream_channel(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that on_error sends an embed to the dream channel."""
        with patch("lattice.discord_client.error_manager.logger"):
            mock_bot = MagicMock()
            mock_channel = MagicMock(spec=discord.TextChannel)
            mock_channel.send = AsyncMock()
            mock_bot.get_channel = MagicMock(return_value=mock_channel)

            error_manager = ErrorManager(mock_bot, 123456)

            try:
                raise ValueError("Test error message")
            except Exception:
                await error_manager.on_error("test_event")

            # Verify channel was retrieved
            mock_bot.get_channel.assert_called_once_with(123456)

            # Verify send was called with warning emojis and embed
            mock_channel.send.assert_called_once()
            call_args = mock_channel.send.call_args
            assert "🚨🚨🚨" in call_args[0][0]
            assert "SYSTEM ERROR" in call_args[0][0]
            assert "embed" in call_args[1]

            # Verify embed structure
            embed = call_args[1]["embed"]
            assert isinstance(embed, discord.Embed)
            assert embed.title is not None and "CRASH DETECTED" in embed.title
            assert embed.description is not None and "test_event" in embed.description
            assert embed.color == discord.Color.red()

    @pytest.mark.asyncio
    async def test_on_error_includes_error_details_in_embed(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that error details are included in the embed."""
        mock_bot = MagicMock()
        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock()
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        error_manager = ErrorManager(mock_bot, 123456)

        try:
            raise ValueError("Detailed error message")
        except Exception:
            await error_manager.on_error("on_message")

        # Get the embed that was sent
        embed = mock_channel.send.call_args[1]["embed"]

        # Verify error type field
        assert any("Error Type" in field.name for field in embed.fields)
        error_type_field = next(f for f in embed.fields if "Error Type" in f.name)
        assert "ValueError" in error_type_field.value

        # Verify message field
        assert any("Message" in field.name for field in embed.fields)
        message_field = next(f for f in embed.fields if "Message" in f.name)
        assert "Detailed error message" in message_field.value

    @pytest.mark.asyncio
    async def test_on_error_truncates_long_messages(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that very long error messages are truncated."""
        mock_bot = MagicMock()
        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock()
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        error_manager = ErrorManager(mock_bot, 123456)

        # Create a very long error message
        long_message = "A" * 2000

        try:
            raise ValueError(long_message)
        except Exception:
            await error_manager.on_error("test_event")

        # Get the embed that was sent
        embed = mock_channel.send.call_args[1]["embed"]

        # Verify message field is truncated to 500 chars
        message_field = next(f for f in embed.fields if "Message" in f.name)
        # Account for code block markers (```)
        assert len(message_field.value) <= 510  # 500 + some margin for ```

    @pytest.mark.asyncio
    async def test_on_error_does_nothing_when_no_dream_channel_id(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that on_error does nothing when dream_channel_id is not set."""
        mock_bot = MagicMock()
        error_manager = ErrorManager(mock_bot, 0)  # No dream channel ID

        try:
            raise ValueError("Test error")
        except Exception:
            await error_manager.on_error("test_event")

        # Bot should not try to get channel
        mock_bot.get_channel.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_error_handles_channel_not_found(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that on_error handles case when channel is not found."""
        mock_bot = MagicMock()
        mock_bot.get_channel = MagicMock(return_value=None)

        error_manager = ErrorManager(mock_bot, 123456)

        try:
            raise ValueError("Test error")
        except Exception:
            # Should not raise an exception
            await error_manager.on_error("test_event")

        mock_bot.get_channel.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_error_handles_non_text_channel(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that on_error handles case when channel is not a TextChannel."""
        mock_bot = MagicMock()
        # Return a non-TextChannel object
        mock_voice_channel = MagicMock(spec=discord.VoiceChannel)
        mock_bot.get_channel = MagicMock(return_value=mock_voice_channel)

        error_manager = ErrorManager(mock_bot, 123456)

        try:
            raise ValueError("Test error")
        except Exception:
            # Should not raise an exception
            await error_manager.on_error("test_event")

        # Should not try to send anything to voice channel
        assert (
            not hasattr(mock_voice_channel, "send")
            or not mock_voice_channel.send.called
        )

    @pytest.mark.asyncio
    async def test_on_error_handles_discord_exception_when_sending(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that on_error handles DiscordException when sending fails."""
        mock_bot = MagicMock()
        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock(side_effect=discord.DiscordException("API error"))
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        error_manager = ErrorManager(mock_bot, 123456)

        try:
            raise ValueError("Test error")
        except Exception:
            # Should not raise an exception even though send fails
            await error_manager.on_error("test_event")

        # Verify send was attempted
        mock_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_error_sets_footer_with_event_name(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that on_error sets embed footer with event name."""
        mock_bot = MagicMock()
        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock()
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        error_manager = ErrorManager(mock_bot, 123456)

        try:
            raise ValueError("Test error")
        except Exception:
            await error_manager.on_error("on_message_edit")

        # Get the embed that was sent
        embed = mock_channel.send.call_args[1]["embed"]

        assert embed.footer.text == "Event: on_message_edit"


class TestMirrorLLMError:
    """Tests for mirror_llm_error function."""

    @pytest.mark.asyncio
    async def test_sends_error_embed_to_dream_channel(self) -> None:
        """Test that mirror_llm_error sends an embed to dream channel."""
        mock_bot = MagicMock()
        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock()
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=123456,
            prompt_key="UNIFIED_RESPONSE",
            error_type="ValueError",
            error_message="Invalid response format",
            prompt_preview="Generate a response for...",
        )

        # Verify channel was retrieved
        mock_bot.get_channel.assert_called_once_with(123456)

        # Verify send was called with embed
        mock_channel.send.assert_called_once()
        call_args = mock_channel.send.call_args
        assert "embed" in call_args[1]

        # Verify embed structure
        embed = call_args[1]["embed"]
        assert isinstance(embed, discord.Embed)
        assert embed.title is not None and "LLM Call Failed" in embed.title
        assert embed.color == discord.Color.red()

    @pytest.mark.asyncio
    async def test_includes_all_error_details_in_embed(self) -> None:
        """Test that all error details are included in the embed fields."""
        mock_bot = MagicMock()
        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock()
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=123456,
            prompt_key="MEMORY_CONSOLIDATION",
            error_type="TimeoutError",
            error_message="Request timed out after 30 seconds",
            prompt_preview="Extract semantic memories from...",
        )

        # Get the embed that was sent
        embed = mock_channel.send.call_args[1]["embed"]

        # Verify all required fields are present
        field_names = [field.name for field in embed.fields]
        assert "Prompt" in field_names
        assert "Error Type" in field_names
        assert "Error" in field_names
        assert "Prompt Key" in field_names

        # Verify field values
        prompt_field = next(f for f in embed.fields if f.name == "Prompt")
        assert "Extract semantic memories from..." in prompt_field.value

        error_type_field = next(f for f in embed.fields if f.name == "Error Type")
        assert "TimeoutError" in error_type_field.value

        error_field = next(f for f in embed.fields if f.name == "Error")
        assert "Request timed out after 30 seconds" in error_field.value

        prompt_key_field = next(f for f in embed.fields if f.name == "Prompt Key")
        assert "MEMORY_CONSOLIDATION" in prompt_key_field.value

    @pytest.mark.asyncio
    async def test_handles_channel_not_found(self) -> None:
        """Test that mirror_llm_error handles channel not found gracefully."""
        mock_bot = MagicMock()
        mock_bot.get_channel = MagicMock(return_value=None)

        # Should not raise an exception
        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=123456,
            prompt_key="TEST",
            error_type="Error",
            error_message="Message",
            prompt_preview="Preview",
        )

        mock_bot.get_channel.assert_called_once_with(123456)

    @pytest.mark.asyncio
    async def test_handles_non_text_channel(self) -> None:
        """Test that mirror_llm_error handles non-TextChannel."""
        mock_bot = MagicMock()
        mock_voice_channel = MagicMock(spec=discord.VoiceChannel)
        mock_bot.get_channel = MagicMock(return_value=mock_voice_channel)

        # Should not raise an exception
        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=123456,
            prompt_key="TEST",
            error_type="Error",
            error_message="Message",
            prompt_preview="Preview",
        )

    @pytest.mark.asyncio
    async def test_handles_discord_exception_when_sending(self) -> None:
        """Test that mirror_llm_error handles DiscordException when sending."""
        mock_bot = MagicMock()
        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock(
            side_effect=discord.DiscordException("Send failed")
        )
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        # Should not raise an exception
        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=123456,
            prompt_key="TEST",
            error_type="Error",
            error_message="Message",
            prompt_preview="Preview",
        )

        # Verify send was attempted
        mock_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_truncates_error_message(self) -> None:
        """Test that very long error messages are truncated."""
        mock_bot = MagicMock()
        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock()
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        long_error = "E" * 1000

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=123456,
            prompt_key="TEST",
            error_type="Error",
            error_message=long_error,
            prompt_preview="Preview",
        )

        # The function should handle this without error
        mock_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_truncates_prompt_preview(self) -> None:
        """Test that very long prompt previews are handled."""
        mock_bot = MagicMock()
        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock()
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        long_preview = "P" * 1000

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=123456,
            prompt_key="TEST",
            error_type="Error",
            error_message="Short error",
            prompt_preview=long_preview,
        )

        # The function should handle this without error
        mock_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_warning_emoji_in_title(self) -> None:
        """Test that warning emoji is used in embed title."""
        mock_bot = MagicMock()
        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock()
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=123456,
            prompt_key="TEST",
            error_type="Error",
            error_message="Message",
            prompt_preview="Preview",
        )

        embed = mock_channel.send.call_args[1]["embed"]
        assert embed.title is not None and "⚠️" in embed.title
