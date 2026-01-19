"""Unit tests for Discord error notifier module."""

from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from lattice.discord_client.error_notifier import mirror_llm_error


class TestMirrorLlmError:
    """Tests for mirror_llm_error function."""

    @pytest.fixture
    def mock_bot(self) -> MagicMock:
        """Create a mock Discord bot."""
        bot = MagicMock()
        return bot

    @pytest.mark.asyncio
    async def test_mirror_llm_error_skips_when_channel_not_found(
        self, mock_bot: MagicMock
    ) -> None:
        """Test error mirroring is skipped when channel is not found."""
        mock_bot.get_channel.return_value = None

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="TEST_PROMPT",
            error_type="ValueError",
            error_message="Test error message",
            prompt_preview="Test prompt preview",
        )

        mock_bot.get_channel.assert_called_once_with(456)

    @pytest.mark.asyncio
    async def test_mirror_llm_error_skips_when_channel_not_text_channel(
        self, mock_bot: MagicMock
    ) -> None:
        """Test error mirroring is skipped when channel is not a TextChannel."""
        mock_channel = MagicMock()
        mock_channel.__class__ = MagicMock
        mock_bot.get_channel.return_value = mock_channel

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="TEST_PROMPT",
            error_type="ValueError",
            error_message="Test error message",
            prompt_preview="Test prompt preview",
        )

        mock_bot.get_channel.assert_called_once_with(456)

    @pytest.mark.asyncio
    async def test_mirror_llm_error_success(self, mock_bot: MagicMock) -> None:
        """Test successful LLM error mirroring."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="UNIFIED_RESPONSE",
            error_type="OpenAIError",
            error_message="Rate limit exceeded",
            prompt_preview="User: What's the weather?",
        )

        mock_dream_channel.send.assert_called_once()
        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        assert embed.title == "⚠️ LLM Call Failed"
        assert embed.color == discord.Color.red()

    @pytest.mark.asyncio
    async def test_mirror_llm_error_embed_contains_prompt(
        self, mock_bot: MagicMock
    ) -> None:
        """Test that embed contains the prompt preview."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="TEST_PROMPT",
            error_type="ValueError",
            error_message="Error message",
            prompt_preview="System: You are helpful...",
        )

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        prompt_field = embed.fields[0]
        assert "System: You are helpful..." in prompt_field.value

    @pytest.mark.asyncio
    async def test_mirror_llm_error_embed_contains_error_type(
        self, mock_bot: MagicMock
    ) -> None:
        """Test that embed contains the error type."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="TEST_PROMPT",
            error_type="TimeoutError",
            error_message="Request timed out",
            prompt_preview="Test prompt",
        )

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        error_type_field = embed.fields[1]
        assert error_type_field.name == "Error Type"
        assert "TimeoutError" in error_type_field.value

    @pytest.mark.asyncio
    async def test_mirror_llm_error_embed_contains_error_message(
        self, mock_bot: MagicMock
    ) -> None:
        """Test that embed contains the error message."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="TEST_PROMPT",
            error_type="ValueError",
            error_message="Invalid response format from API",
            prompt_preview="Test prompt",
        )

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        error_field = embed.fields[2]
        assert error_field.name == "Error"
        assert "Invalid response format from API" in error_field.value

    @pytest.mark.asyncio
    async def test_mirror_llm_error_embed_contains_prompt_key(
        self, mock_bot: MagicMock
    ) -> None:
        """Test that embed contains the prompt key."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="MEMORY_CONSOLIDATION",
            error_type="ValueError",
            error_message="Error message",
            prompt_preview="Test prompt",
        )

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        prompt_key_field = embed.fields[3]
        assert prompt_key_field.name == "Prompt Key"
        assert "MEMORY_CONSOLIDATION" in prompt_key_field.value

    @pytest.mark.asyncio
    async def test_mirror_llm_error_handles_discord_exception(
        self, mock_bot: MagicMock
    ) -> None:
        """Test that Discord exceptions during send are handled gracefully."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock(
            side_effect=discord.DiscordException("Failed to send")
        )

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="TEST_PROMPT",
            error_type="ValueError",
            error_message="Test error",
            prompt_preview="Test prompt",
        )

        mock_dream_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_mirror_llm_error_with_different_error_types(
        self, mock_bot: MagicMock
    ) -> None:
        """Test mirroring with different error types."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        error_types = [
            "ValueError",
            "TypeError",
            "KeyError",
            "RuntimeError",
            "OpenAIError",
            "AnthropicError",
            "NetworkError",
        ]

        for error_type in error_types:
            mock_dream_channel.send.reset_mock()
            await mirror_llm_error(
                bot=mock_bot,
                dream_channel_id=456,
                prompt_key="TEST_PROMPT",
                error_type=error_type,
                error_message="Error message",
                prompt_preview="Test prompt",
            )

            call_args = mock_dream_channel.send.call_args
            embed = call_args[1]["embed"]
            error_type_field = embed.fields[1]
            assert error_type in error_type_field.value

    @pytest.mark.asyncio
    async def test_mirror_llm_error_embed_field_inline_settings(
        self, mock_bot: MagicMock
    ) -> None:
        """Test that embed fields have correct inline settings."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="TEST_PROMPT",
            error_type="ValueError",
            error_message="Error message",
            prompt_preview="Test prompt",
        )

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]

        assert len(embed.fields) == 4

        assert embed.fields[0].name == "Prompt"
        assert embed.fields[0].inline is False

        assert embed.fields[1].name == "Error Type"
        assert embed.fields[1].inline is True

        assert embed.fields[2].name == "Error"
        assert embed.fields[2].inline is False

        assert embed.fields[3].name == "Prompt Key"
        assert embed.fields[3].inline is True


class TestMirrorLlmErrorEdgeCases:
    """Tests for edge cases in mirror_llm_error."""

    @pytest.mark.asyncio
    async def test_mirror_llm_error_empty_prompt_key(self) -> None:
        """Test mirroring with empty prompt key."""
        mock_bot = MagicMock()
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="",
            error_type="ValueError",
            error_message="Error message",
            prompt_preview="Test prompt",
        )

        mock_dream_channel.send.assert_called_once()
        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        prompt_key_field = embed.fields[3]
        assert "" in prompt_key_field.value

    @pytest.mark.asyncio
    async def test_mirror_llm_error_long_prompt_preview(self) -> None:
        """Test mirroring with long prompt preview - shows full prompt."""
        mock_bot = MagicMock()
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        long_prompt = "x" * 250
        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="TEST_PROMPT",
            error_type="ValueError",
            error_message="Error message",
            prompt_preview=long_prompt,
        )

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        prompt_field = embed.fields[0]
        # Prompt is wrapped in backticks, so 250 + 6 = 256
        assert len(prompt_field.value) == 256

    @pytest.mark.asyncio
    async def test_mirror_llm_error_long_error_message(self) -> None:
        """Test mirroring with long error message - shows full error."""
        mock_bot = MagicMock()
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        long_error = "x" * 600
        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="TEST_PROMPT",
            error_type="ValueError",
            error_message=long_error,
            prompt_preview="Test prompt",
        )

        call_args = mock_dream_channel.send.call_args
        embed = call_args[1]["embed"]
        error_field = embed.fields[2]
        # Error is not truncated, shows full 600 chars
        assert len(error_field.value) == 600

    @pytest.mark.asyncio
    async def test_mirror_llm_error_with_special_characters(self) -> None:
        """Test mirroring with special characters in error message."""
        mock_bot = MagicMock()
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        special_error = "Error with 'quotes' and \"double quotes\" and `backticks`"
        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="TEST_PROMPT",
            error_type="ValueError",
            error_message=special_error,
            prompt_preview="Test prompt",
        )

        mock_dream_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_mirror_llm_error_multiline_error_message(self) -> None:
        """Test mirroring with multiline error message."""
        mock_bot = MagicMock()
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        multiline_error = "Line 1\nLine 2\nLine 3"
        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="TEST_PROMPT",
            error_type="ValueError",
            error_message=multiline_error,
            prompt_preview="Test prompt",
        )

        mock_dream_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_mirror_llm_error_with_unicode_characters(self) -> None:
        """Test mirroring with unicode characters in error message."""
        mock_bot = MagicMock()
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel
        mock_dream_channel.send = AsyncMock()

        unicode_error = "Error with émojis 🎉 and ünicode"
        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=456,
            prompt_key="TEST_PROMPT",
            error_type="ValueError",
            error_message=unicode_error,
            prompt_preview="Test prompt",
        )

        mock_dream_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_mirror_llm_error_zero_dream_channel_id(self) -> None:
        """Test mirroring with zero dream channel ID."""
        mock_bot = MagicMock()
        mock_bot.get_channel.return_value = None

        await mirror_llm_error(
            bot=mock_bot,
            dream_channel_id=0,
            prompt_key="TEST_PROMPT",
            error_type="ValueError",
            error_message="Error message",
            prompt_preview="Test prompt",
        )

        mock_bot.get_channel.assert_called_once_with(0)
