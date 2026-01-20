"""Unit tests for audit mirror."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import discord
import pytest

from lattice.discord_client.audit_mirror import AuditMirror


class TestAuditMirror:
    """Test AuditMirror class."""

    def test_init_stores_dependencies(self) -> None:
        """Test AuditMirror initializes with dependencies."""
        mock_bot = Mock(spec=discord.Bot)
        mock_audit_repo = Mock()
        mock_feedback_repo = Mock()
        dream_channel_id = 123456789

        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=dream_channel_id,
            audit_repo=mock_audit_repo,
            feedback_repo=mock_feedback_repo,
        )

        assert mirror.bot is mock_bot
        assert mirror.dream_channel_id == dream_channel_id
        assert mirror.audit_repo is mock_audit_repo
        assert mirror.feedback_repo is mock_feedback_repo
        assert mirror._active_views == {}

    @pytest.mark.asyncio
    async def test_get_active_view_count_empty(self) -> None:
        """Test get_active_view_count returns 0 when no views are tracked."""
        mock_bot = Mock(spec=discord.Bot)
        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_repo=Mock(),
            feedback_repo=Mock(),
        )

        count = await mirror.get_active_view_count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_active_view_count_with_views(self) -> None:
        """Test get_active_view_count returns correct count."""
        mock_bot = Mock(spec=discord.Bot)
        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_repo=Mock(),
            feedback_repo=Mock(),
        )

        # Manually add some views
        mirror._active_views[111] = (Mock(), 1234567890.0)
        mirror._active_views[222] = (Mock(), 1234567891.0)
        mirror._active_views[333] = (Mock(), 1234567892.0)

        count = await mirror.get_active_view_count()
        assert count == 3

    @pytest.mark.asyncio
    @patch("lattice.discord_client.audit_mirror.logger")
    async def test_log_view_stats_with_no_views(self, mock_logger: Mock) -> None:
        """Test log_view_stats does not log when no views are active."""
        mock_bot = Mock(spec=discord.Bot)
        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_repo=Mock(),
            feedback_repo=Mock(),
        )

        await mirror.log_view_stats()

        # Should not log anything
        mock_logger.info.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    @patch("lattice.discord_client.audit_mirror.logger")
    async def test_log_view_stats_with_views(self, mock_logger: Mock) -> None:
        """Test log_view_stats logs when views are active."""
        mock_bot = Mock(spec=discord.Bot)
        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_repo=Mock(),
            feedback_repo=Mock(),
        )

        # Add some views
        mirror._active_views[111] = (Mock(), 1234567890.0)
        mirror._active_views[222] = (Mock(), 1234567891.0)

        await mirror.log_view_stats()

        # Should log statistics
        mock_logger.info.assert_called_once()  # type: ignore[attr-defined]
        call_args = mock_logger.info.call_args  # type: ignore[attr-defined]
        assert call_args[0][0] == "Active views statistics"
        assert call_args[1]["count"] == 2

    @pytest.mark.asyncio
    async def test_mirror_audit_returns_early_if_no_dream_channel_id(self) -> None:
        """Test mirror_audit returns early when dream_channel_id is 0 or None."""
        mock_bot = Mock(spec=discord.Bot)
        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=0,  # No channel configured
            audit_repo=Mock(),
            feedback_repo=Mock(),
        )

        # Should return early without calling get_channel
        await mirror.mirror_audit(
            audit_id=uuid4(),
            prompt_key="TEST",
            template_version=1,
            rendered_prompt="test prompt",
            result=Mock(latency_ms=100, cost_usd=0.01, content="test"),
            params={},
        )

        mock_bot.get_channel.assert_not_called()

    @pytest.mark.asyncio
    async def test_mirror_audit_returns_early_if_channel_not_text_channel(
        self,
    ) -> None:
        """Test mirror_audit returns early when channel is not TextChannel."""
        mock_bot = Mock(spec=discord.Bot)
        # Return a VoiceChannel instead of TextChannel
        mock_voice_channel = Mock(spec=discord.VoiceChannel)
        mock_bot.get_channel = Mock(return_value=mock_voice_channel)

        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_repo=Mock(),
            feedback_repo=Mock(),
        )

        # Should return early without building audit view
        await mirror.mirror_audit(
            audit_id=uuid4(),
            prompt_key="TEST",
            template_version=1,
            rendered_prompt="test prompt",
            result=Mock(latency_ms=100, cost_usd=0.01, content="test"),
            params={},
        )

        # Should have gotten channel but not proceeded further
        mock_bot.get_channel.assert_called_once_with(123456789)

    @pytest.mark.asyncio
    @patch("lattice.discord_client.audit_mirror.AuditViewBuilder.build_standard_audit")
    async def test_mirror_audit_builds_and_sends_audit_view(
        self, mock_build_audit: AsyncMock
    ) -> None:
        """Test mirror_audit builds and sends AuditView successfully."""
        mock_bot = Mock(spec=discord.Bot)
        mock_channel = Mock(spec=discord.TextChannel)
        mock_message = Mock()
        mock_message.id = 999888777
        mock_channel.send = AsyncMock(return_value=mock_message)
        mock_bot.get_channel = Mock(return_value=mock_channel)
        mock_bot.add_view = Mock()

        mock_embed = Mock()
        mock_view = Mock()
        mock_build_audit.return_value = (mock_embed, mock_view, None)

        mock_audit_repo = Mock()
        mock_feedback_repo = Mock()

        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_repo=mock_audit_repo,
            feedback_repo=mock_feedback_repo,
        )

        test_audit_id = uuid4()
        mock_result = Mock()
        mock_result.latency_ms = 150
        mock_result.cost_usd = 0.0234
        mock_result.content = "Test response content"

        await mirror.mirror_audit(
            audit_id=test_audit_id,
            prompt_key="TEST_PROMPT",
            template_version=2,
            rendered_prompt="This is the full prompt",
            result=mock_result,
            params={"metadata": ["extra", "info"], "input_text": "User input"},
        )

        # Verify build_standard_audit was called
        mock_build_audit.assert_called_once()
        call_kwargs = mock_build_audit.call_args[1]
        assert call_kwargs["prompt_key"] == "TEST_PROMPT"
        assert call_kwargs["version"] == 2
        assert call_kwargs["input_text"] == "User input"
        assert call_kwargs["output_text"] == "Test response content"
        assert call_kwargs["audit_id"] == test_audit_id
        assert call_kwargs["audit_repo"] is mock_audit_repo
        assert call_kwargs["feedback_repo"] is mock_feedback_repo

        # Check metadata includes latency and cost
        metadata = call_kwargs["metadata_parts"]
        assert "150ms" in metadata
        assert "$0.0234" in metadata
        assert "extra" in metadata
        assert "info" in metadata

        # Verify message was sent
        mock_channel.send.assert_called_once_with(embed=mock_embed, view=mock_view)

        # Verify view was added to bot
        mock_bot.add_view.assert_called_once_with(mock_view)

        # Verify active view was tracked
        assert mock_message.id in mirror._active_views
        tracked_view, timestamp = mirror._active_views[mock_message.id]
        assert tracked_view is mock_view
        assert isinstance(timestamp, float)

    @pytest.mark.asyncio
    @patch("lattice.discord_client.audit_mirror.AuditViewBuilder.build_standard_audit")
    @patch("lattice.memory.prompt_audits.update_audit_dream_message")
    async def test_mirror_audit_updates_audit_with_dream_message_id(
        self, mock_update_audit: AsyncMock, mock_build_audit: AsyncMock
    ) -> None:
        """Test mirror_audit updates audit record with dream message ID."""
        mock_bot = Mock(spec=discord.Bot)
        mock_channel = Mock(spec=discord.TextChannel)
        mock_message = Mock()
        mock_message.id = 999888777
        mock_channel.send = AsyncMock(return_value=mock_message)
        mock_bot.get_channel = Mock(return_value=mock_channel)
        mock_bot.add_view = Mock()

        mock_build_audit.return_value = (Mock(), Mock(), None)

        mock_audit_repo = Mock()

        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_repo=mock_audit_repo,
            feedback_repo=Mock(),
        )

        test_audit_id = uuid4()
        mock_result = Mock(latency_ms=100, cost_usd=0.01, content="test")

        await mirror.mirror_audit(
            audit_id=test_audit_id,
            prompt_key="TEST",
            template_version=1,
            rendered_prompt="prompt",
            result=mock_result,
            params={},
        )

        # Verify update_audit_dream_message was called
        mock_update_audit.assert_called_once_with(
            repo=mock_audit_repo,
            audit_id=test_audit_id,
            dream_discord_message_id=999888777,
        )

    @pytest.mark.asyncio
    @patch("lattice.discord_client.audit_mirror.logger")
    @patch("lattice.discord_client.audit_mirror.AuditViewBuilder.build_standard_audit")
    async def test_mirror_audit_handles_send_failure_gracefully(
        self, mock_build_audit: AsyncMock, mock_logger: Mock
    ) -> None:
        """Test mirror_audit logs exception if send fails."""
        mock_bot = Mock(spec=discord.Bot)
        mock_channel = Mock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock(side_effect=Exception("Discord API error"))
        mock_bot.get_channel = Mock(return_value=mock_channel)

        mock_build_audit.return_value = (Mock(), Mock(), None)

        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_repo=Mock(),
            feedback_repo=Mock(),
        )

        mock_result = Mock(latency_ms=100, cost_usd=0.01, content="test")

        # Should not raise
        await mirror.mirror_audit(
            audit_id=uuid4(),
            prompt_key="TEST",
            template_version=1,
            rendered_prompt="prompt",
            result=mock_result,
            params={},
        )

        # Verify exception was logged
        mock_logger.exception.assert_called_once_with(  # type: ignore[attr-defined]
            "Failed to send AuditView to dream channel"
        )

    @pytest.mark.asyncio
    @patch("lattice.discord_client.audit_mirror.AuditViewBuilder.build_standard_audit")
    async def test_mirror_audit_includes_main_message_url_in_metadata(
        self, mock_build_audit: AsyncMock
    ) -> None:
        """Test mirror_audit includes main message URL in metadata."""
        mock_bot = Mock(spec=discord.Bot)
        mock_channel = Mock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock(return_value=Mock(id=123))
        mock_bot.get_channel = Mock(return_value=mock_channel)
        mock_bot.add_view = Mock()

        mock_build_audit.return_value = (Mock(), Mock(), None)

        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_repo=Mock(),
            feedback_repo=Mock(),
        )

        mock_result = Mock(latency_ms=100, cost_usd=0.01, content="test")

        await mirror.mirror_audit(
            audit_id=uuid4(),
            prompt_key="TEST",
            template_version=1,
            rendered_prompt="prompt",
            result=mock_result,
            params={"main_message_url": "https://discord.com/channels/123/456/789"},
        )

        # Verify URL was added to metadata
        mock_build_audit.assert_called_once()
        metadata = mock_build_audit.call_args[1]["metadata_parts"]
        assert any(
            "[LINK](https://discord.com/channels/123/456/789)" in m for m in metadata
        )

    @pytest.mark.asyncio
    @patch("lattice.discord_client.audit_mirror.AuditViewBuilder.build_standard_audit")
    async def test_mirror_audit_uses_default_input_text_if_not_provided(
        self, mock_build_audit: AsyncMock
    ) -> None:
        """Test mirror_audit uses rendered_prompt preview as default input_text."""
        mock_bot = Mock(spec=discord.Bot)
        mock_channel = Mock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock(return_value=Mock(id=123))
        mock_bot.get_channel = Mock(return_value=mock_channel)
        mock_bot.add_view = Mock()

        mock_build_audit.return_value = (Mock(), Mock(), None)

        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_repo=Mock(),
            feedback_repo=Mock(),
        )

        long_prompt = "A" * 300  # 300 characters
        mock_result = Mock(latency_ms=100, cost_usd=0.01, content="test")

        await mirror.mirror_audit(
            audit_id=uuid4(),
            prompt_key="TEST",
            template_version=1,
            rendered_prompt=long_prompt,
            result=mock_result,
            params={},  # No input_text provided
        )

        # Verify default input_text is first 200 chars + "..."
        mock_build_audit.assert_called_once()
        input_text = mock_build_audit.call_args[1]["input_text"]
        assert input_text == "A" * 200 + "..."

    @pytest.mark.asyncio
    @patch("lattice.discord_client.audit_mirror.AuditViewBuilder.build_standard_audit")
    async def test_mirror_audit_omits_cost_if_none(
        self, mock_build_audit: AsyncMock
    ) -> None:
        """Test mirror_audit omits cost from metadata if result.cost_usd is None."""
        mock_bot = Mock(spec=discord.Bot)
        mock_channel = Mock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock(return_value=Mock(id=123))
        mock_bot.get_channel = Mock(return_value=mock_channel)
        mock_bot.add_view = Mock()

        mock_build_audit.return_value = (Mock(), Mock(), None)

        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_repo=Mock(),
            feedback_repo=Mock(),
        )

        mock_result = Mock(latency_ms=100, cost_usd=None, content="test")

        await mirror.mirror_audit(
            audit_id=uuid4(),
            prompt_key="TEST",
            template_version=1,
            rendered_prompt="prompt",
            result=mock_result,
            params={},
        )

        # Verify cost was NOT added to metadata
        mock_build_audit.assert_called_once()
        metadata = mock_build_audit.call_args[1]["metadata_parts"]
        assert not any("$" in m for m in metadata)
        assert "100ms" in metadata  # Latency should still be there

    @pytest.mark.asyncio
    @patch("lattice.discord_client.audit_mirror.AuditViewBuilder.build_standard_audit")
    @patch("lattice.memory.prompt_audits.update_audit_dream_message")
    async def test_mirror_audit_skips_audit_update_if_no_audit_id(
        self, mock_update_audit: AsyncMock, mock_build_audit: AsyncMock
    ) -> None:
        """Test mirror_audit skips audit update when audit_id is None."""
        mock_bot = Mock(spec=discord.Bot)
        mock_channel = Mock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock(return_value=Mock(id=123))
        mock_bot.get_channel = Mock(return_value=mock_channel)
        mock_bot.add_view = Mock()

        mock_build_audit.return_value = (Mock(), Mock(), None)

        mirror = AuditMirror(
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_repo=Mock(),
            feedback_repo=Mock(),
        )

        mock_result = Mock(latency_ms=100, cost_usd=0.01, content="test")

        await mirror.mirror_audit(
            audit_id=None,  # No audit ID
            prompt_key="TEST",
            template_version=1,
            rendered_prompt="prompt",
            result=mock_result,
            params={},
        )

        # Verify update was NOT called
        mock_update_audit.assert_not_called()
