"""Unit tests for Discord error handlers module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import discord
import pytest

from lattice.discord_client.error_handlers import notify_parse_error_to_dream
from lattice.utils.json_parser import JSONParseError
from lattice.utils.llm import AuditResult


class TestNotifyParseErrorToDream:
    """Tests for notify_parse_error_to_dream function."""

    @pytest.fixture
    def mock_bot(self) -> MagicMock:
        """Create a mock Discord bot."""
        bot = MagicMock()
        bot.dream_channel_id = 456
        bot.audit_repo = AsyncMock()
        bot.feedback_repo = AsyncMock()
        return bot

    @pytest.fixture
    def mock_audit_result(self) -> AuditResult:
        """Create a mock audit result."""
        return AuditResult(
            content="test response",
            model="gpt-4",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
            latency_ms=500,
            temperature=0.7,
            audit_id=uuid4(),
            prompt_key="TEST_PROMPT",
        )

    @pytest.mark.asyncio
    async def test_notify_parse_error_skips_without_audit_id(
        self, mock_bot: MagicMock
    ) -> None:
        """Test notification is skipped when audit_result has no audit_id."""
        error = JSONParseError(
            raw_content='{"invalid": "json"}',
            parse_error=json.JSONDecodeError("Invalid JSON", "", 0),
            audit_result=None,
            prompt_key="TEST_PROMPT",
        )
        context = {"parser_type": "json"}

        await notify_parse_error_to_dream(mock_bot, error, context)

        mock_bot.get_channel.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_parse_error_skips_without_dream_channel(
        self, mock_bot: MagicMock
    ) -> None:
        """Test notification is skipped when dream_channel_id is not set."""
        mock_bot.dream_channel_id = 0

        error = JSONParseError(
            raw_content='{"invalid": "json"}',
            parse_error=json.JSONDecodeError("Invalid JSON", "", 0),
            audit_result=AuditResult(
                content="test response",
                model="gpt-4",
                provider="openai",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.01,
                latency_ms=500,
                temperature=0.7,
                audit_id=uuid4(),
            ),
            prompt_key="TEST_PROMPT",
        )
        context = {"parser_type": "json"}

        await notify_parse_error_to_dream(mock_bot, error, context)

        mock_bot.get_channel.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_parse_error_skips_when_channel_not_text_channel(
        self, mock_bot: MagicMock
    ) -> None:
        """Test notification is skipped when dream channel is not a TextChannel."""
        mock_channel = MagicMock()
        mock_channel.__class__ = MagicMock
        mock_bot.get_channel.return_value = mock_channel

        error = JSONParseError(
            raw_content='{"invalid": "json"}',
            parse_error=json.JSONDecodeError("Invalid JSON", "", 0),
            audit_result=AuditResult(
                content="test response",
                model="gpt-4",
                provider="openai",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.01,
                latency_ms=500,
                temperature=0.7,
                audit_id=uuid4(),
            ),
            prompt_key="TEST_PROMPT",
        )
        context = {"parser_type": "json"}

        await notify_parse_error_to_dream(mock_bot, error, context)

        mock_bot.get_channel.assert_called_once_with(456)

    @pytest.mark.asyncio
    async def test_notify_parse_error_success(
        self, mock_bot: MagicMock, mock_audit_result: AuditResult
    ) -> None:
        """Test successful parse error notification."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel

        error = JSONParseError(
            raw_content='{"invalid": "json"}',
            parse_error=json.JSONDecodeError("Invalid JSON", "", 0),
            audit_result=mock_audit_result,
            prompt_key="TEST_PROMPT",
        )
        context = {
            "parser_type": "json",
            "rendered_prompt": "System: Parse JSON\nUser: Extract data",
        }

        mock_embed = MagicMock()
        mock_view = MagicMock()
        mock_view.message = MagicMock()
        with patch(
            "lattice.discord_client.error_handlers.AuditViewBuilder.build_standard_audit",
            new_callable=AsyncMock,
        ) as mock_build:
            mock_build.return_value = (mock_embed, mock_view, MagicMock())
            mock_dream_channel.send = AsyncMock()

            await notify_parse_error_to_dream(mock_bot, error, context)

            mock_build.assert_called_once()
            mock_embed.add_field.assert_called()
            mock_dream_channel.send.assert_called_once_with(
                embed=mock_embed, view=mock_view
            )

    @pytest.mark.asyncio
    async def test_notify_parse_error_truncates_long_response(
        self, mock_bot: MagicMock, mock_audit_result: AuditResult
    ) -> None:
        """Test that long responses are truncated for Discord embed limits."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel

        long_response = "x" * 1000
        error = JSONParseError(
            raw_content=long_response,
            parse_error=json.JSONDecodeError("Invalid JSON", "", 0),
            audit_result=mock_audit_result,
            prompt_key="TEST_PROMPT",
        )
        context = {"parser_type": "json", "rendered_prompt": ""}

        mock_embed = MagicMock()
        mock_view = MagicMock()
        mock_view.message = MagicMock()
        with patch(
            "lattice.discord_client.error_handlers.AuditViewBuilder.build_standard_audit",
            new_callable=AsyncMock,
        ) as mock_build:
            mock_build.return_value = (mock_embed, mock_view, MagicMock())
            mock_dream_channel.send = AsyncMock()

            await notify_parse_error_to_dream(mock_bot, error, context)

            call_args = mock_build.call_args
            assert len(call_args.kwargs["output_text"]) <= 503  # 500 + "..."

    @pytest.mark.asyncio
    async def test_notify_parse_error_handles_discord_exception(
        self, mock_bot: MagicMock, mock_audit_result: AuditResult
    ) -> None:
        """Test that Discord exceptions during send are handled gracefully."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel

        error = JSONParseError(
            raw_content='{"invalid": "json"}',
            parse_error=json.JSONDecodeError("Invalid JSON", "", 0),
            audit_result=mock_audit_result,
            prompt_key="TEST_PROMPT",
        )
        context = {"parser_type": "json", "rendered_prompt": ""}

        mock_embed = MagicMock()
        mock_view = MagicMock()
        mock_view.message = MagicMock()
        with patch(
            "lattice.discord_client.error_handlers.AuditViewBuilder.build_standard_audit",
            new_callable=AsyncMock,
        ) as mock_build:
            mock_build.return_value = (mock_embed, mock_view, MagicMock())
            mock_dream_channel.send = AsyncMock(
                side_effect=discord.DiscordException("Network error")
            )

            await notify_parse_error_to_dream(mock_bot, error, context)

            mock_dream_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_parse_error_unknown_parser_type(
        self, mock_bot: MagicMock, mock_audit_result: AuditResult
    ) -> None:
        """Test notification with unknown parser type uses default."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel

        error = JSONParseError(
            raw_content='{"invalid": "json"}',
            parse_error=json.JSONDecodeError("Invalid JSON", "", 0),
            audit_result=mock_audit_result,
            prompt_key="TEST_PROMPT",
        )
        context = {"parser_type": "unknown", "rendered_prompt": ""}

        mock_embed = MagicMock()
        mock_view = MagicMock()
        mock_view.message = MagicMock()
        with patch(
            "lattice.discord_client.error_handlers.AuditViewBuilder.build_standard_audit",
            new_callable=AsyncMock,
        ) as mock_build:
            mock_build.return_value = (mock_embed, mock_view, MagicMock())
            mock_dream_channel.send = AsyncMock()

            await notify_parse_error_to_dream(mock_bot, error, context)

            call_args = mock_build.call_args
            assert "[UNKNOWN]" in call_args.kwargs["input_text"]

    @pytest.mark.asyncio
    async def test_notify_parse_error_with_empty_prompt_key(
        self, mock_bot: MagicMock, mock_audit_result: AuditResult
    ) -> None:
        """Test notification handles empty prompt_key."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel

        error = JSONParseError(
            raw_content='{"invalid": "json"}',
            parse_error=json.JSONDecodeError("Invalid JSON", "", 0),
            audit_result=mock_audit_result,
            prompt_key="",
        )
        context = {"parser_type": "json", "rendered_prompt": ""}

        mock_embed = MagicMock()
        mock_view = MagicMock()
        mock_view.message = MagicMock()
        with patch(
            "lattice.discord_client.error_handlers.AuditViewBuilder.build_standard_audit",
            new_callable=AsyncMock,
        ) as mock_build:
            mock_build.return_value = (mock_embed, mock_view, MagicMock())
            mock_dream_channel.send = AsyncMock()

            await notify_parse_error_to_dream(mock_bot, error, context)

            call_args = mock_build.call_args
            assert call_args.kwargs["prompt_key"] == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_notify_parse_error_embed_description(
        self, mock_bot: MagicMock, mock_audit_result: AuditResult
    ) -> None:
        """Test that embed description is set with parse error message."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel

        error = JSONParseError(
            raw_content='{"invalid": "json"}',
            parse_error=json.JSONDecodeError("Invalid JSON: Expecting value", "", 0),
            audit_result=mock_audit_result,
            prompt_key="TEST_PROMPT",
        )
        context = {"parser_type": "json", "rendered_prompt": ""}

        mock_embed = MagicMock()
        mock_view = MagicMock()
        mock_view.message = MagicMock()
        with patch(
            "lattice.discord_client.error_handlers.AuditViewBuilder.build_standard_audit",
            new_callable=AsyncMock,
        ) as mock_build:
            mock_build.return_value = (mock_embed, mock_view, MagicMock())
            mock_dream_channel.send = AsyncMock()

            await notify_parse_error_to_dream(mock_bot, error, context)

            mock_embed.description = f"❌ **Parse Error**: {error.parse_error}"

    @pytest.mark.asyncio
    async def test_notify_parse_error_audit_result_attributes(
        self, mock_bot: MagicMock, mock_audit_result: AuditResult
    ) -> None:
        """Test that audit result attributes are used correctly in notification."""
        mock_dream_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel.return_value = mock_dream_channel

        error = JSONParseError(
            raw_content='{"invalid": "json"}',
            parse_error=json.JSONDecodeError("Invalid JSON", "", 0),
            audit_result=mock_audit_result,
            prompt_key="TEST_PROMPT",
        )
        context = {"parser_type": "yaml", "rendered_prompt": "YAML prompt"}

        mock_embed = MagicMock()
        mock_view = MagicMock()
        mock_view.message = MagicMock()
        with patch(
            "lattice.discord_client.error_handlers.AuditViewBuilder.build_standard_audit",
            new_callable=AsyncMock,
        ) as mock_build:
            mock_build.return_value = (mock_embed, mock_view, MagicMock())
            mock_dream_channel.send = AsyncMock()

            await notify_parse_error_to_dream(mock_bot, error, context)

            call_args = mock_build.call_args
            assert call_args.kwargs["audit_id"] == mock_audit_result.audit_id
            assert call_args.kwargs["rendered_prompt"] == "YAML prompt"

    @pytest.mark.asyncio
    async def test_notify_parse_error_get_channel_returns_none(
        self, mock_bot: MagicMock, mock_audit_result: AuditResult
    ) -> None:
        """Test notification handles when get_channel returns None."""
        mock_bot.get_channel.return_value = None

        error = JSONParseError(
            raw_content='{"invalid": "json"}',
            parse_error=json.JSONDecodeError("Invalid JSON", "", 0),
            audit_result=mock_audit_result,
            prompt_key="TEST_PROMPT",
        )
        context = {"parser_type": "json", "rendered_prompt": ""}

        await notify_parse_error_to_dream(mock_bot, error, context)

        mock_bot.get_channel.assert_called_once_with(456)
