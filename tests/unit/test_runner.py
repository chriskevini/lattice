"""Unit tests for scheduler runner (ProactiveScheduler)."""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import discord
import pytest

from lattice.scheduler.runner import ProactiveScheduler
from lattice.scheduler.triggers import ProactiveDecision


class TestProactiveScheduler:
    """Tests for ProactiveScheduler class."""

    @pytest.mark.asyncio
    async def test_init(self) -> None:
        """Test scheduler initialization."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(
            bot=mock_bot, check_interval=30, dream_channel_id=12345
        )

        assert scheduler.bot == mock_bot
        assert scheduler.check_interval == 30
        assert scheduler.dream_channel_id == 12345
        assert scheduler._running is False
        assert scheduler._scheduler_task is None
        assert scheduler._active_hours_task is None

    @pytest.mark.asyncio
    async def test_start_with_existing_next_check(self) -> None:
        """Test starting scheduler with existing next_check_at."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot)

        next_check = datetime.now(UTC) + timedelta(minutes=30)

        with (
            patch(
                "lattice.scheduler.runner.get_next_check_at",
                return_value=next_check,
            ),
            patch("lattice.scheduler.runner.set_next_check_at", AsyncMock()),
            patch.object(scheduler, "_scheduler_loop", AsyncMock()),
            patch.object(scheduler, "_active_hours_loop", AsyncMock()),
        ):
            await scheduler.start()

            assert scheduler._running is True
            assert scheduler._scheduler_task is not None
            assert scheduler._active_hours_task is not None

            # Cleanup
            scheduler._running = False
            if scheduler._scheduler_task:
                scheduler._scheduler_task.cancel()
            if scheduler._active_hours_task:
                scheduler._active_hours_task.cancel()

    @pytest.mark.asyncio
    async def test_start_without_existing_next_check(self) -> None:
        """Test starting scheduler without existing next_check_at."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot, check_interval=20)

        with (
            patch("lattice.scheduler.runner.get_next_check_at", return_value=None),
            patch(
                "lattice.scheduler.runner.set_next_check_at", AsyncMock()
            ) as mock_set,
            patch.object(scheduler, "_scheduler_loop", AsyncMock()),
            patch.object(scheduler, "_active_hours_loop", AsyncMock()),
        ):
            await scheduler.start()

            assert scheduler._running is True
            mock_set.assert_called_once()
            call_time = mock_set.call_args[0][0]
            assert isinstance(call_time, datetime)

            # Cleanup
            scheduler._running = False
            if scheduler._scheduler_task:
                scheduler._scheduler_task.cancel()
            if scheduler._active_hours_task:
                scheduler._active_hours_task.cancel()

    @pytest.mark.asyncio
    async def test_stop(self) -> None:
        """Test stopping scheduler."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot)

        # Mock running tasks
        scheduler._running = True
        scheduler._scheduler_task = asyncio.create_task(asyncio.sleep(0.1))
        scheduler._active_hours_task = asyncio.create_task(asyncio.sleep(0.1))

        await scheduler.stop()

        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_scheduler_loop_waits_for_next_check(self) -> None:
        """Test scheduler loop waits when next check is in future."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot)
        scheduler._running = True

        next_check = datetime.now(UTC) + timedelta(seconds=0.2)

        call_count = 0
        sleep_call_count = 0

        async def get_next_check_side_effect() -> datetime:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return next_check
            # Stop after first iteration
            scheduler._running = False
            return datetime.now(UTC) + timedelta(hours=1)

        async def sleep_side_effect(seconds: float) -> None:
            nonlocal sleep_call_count
            sleep_call_count += 1
            # Don't actually sleep, just stop the loop after first sleep call
            scheduler._running = False

        with (
            patch(
                "lattice.scheduler.runner.get_next_check_at",
                side_effect=get_next_check_side_effect,
            ),
            patch("asyncio.sleep", side_effect=sleep_side_effect),
            patch.object(scheduler, "_run_proactive_check", AsyncMock()) as mock_check,
        ):
            await scheduler._scheduler_loop()

            # Should have called sleep since next_check was in future
            assert sleep_call_count >= 1
            # Should not have run check since we stopped loop during sleep
            mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_scheduler_loop_runs_check_when_time(self) -> None:
        """Test scheduler loop runs check when time has come."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot)
        scheduler._running = True

        # Set next_check to past time
        past_check = datetime.now(UTC) - timedelta(seconds=1)

        call_count = 0

        async def get_next_check_side_effect() -> datetime:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return past_check
            # Stop after first iteration
            scheduler._running = False
            return datetime.now(UTC) + timedelta(hours=1)

        with (
            patch(
                "lattice.scheduler.runner.get_next_check_at",
                side_effect=get_next_check_side_effect,
            ),
            patch("asyncio.sleep", AsyncMock()),
            patch.object(scheduler, "_run_proactive_check", AsyncMock()) as mock_check,
        ):
            await scheduler._scheduler_loop()

            mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_scheduler_loop_handles_exceptions(self) -> None:
        """Test scheduler loop handles exceptions gracefully."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot, check_interval=1)
        scheduler._running = True

        call_count = 0

        async def get_next_check_side_effect() -> datetime:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test exception")
            # Stop after handling exception
            scheduler._running = False
            return datetime.now(UTC) + timedelta(hours=1)

        with (
            patch(
                "lattice.scheduler.runner.get_next_check_at",
                side_effect=get_next_check_side_effect,
            ),
            patch("asyncio.sleep", AsyncMock()),
        ):
            await scheduler._scheduler_loop()

            # Should have handled exception and continued
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_active_hours_loop_updates_periodically(self) -> None:
        """Test active hours loop updates periodically."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot)
        scheduler._running = True

        mock_result = {
            "start_hour": 9,
            "end_hour": 21,
            "confidence": 0.85,
            "sample_size": 100,
        }

        call_count = 0

        async def sleep_side_effect(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 1:  # After initial 60s sleep and one update cycle
                scheduler._running = False

        with (
            patch(
                "lattice.scheduler.runner.update_active_hours",
                return_value=mock_result,
            ) as mock_update,
            patch("asyncio.sleep", side_effect=sleep_side_effect),
        ):
            await scheduler._active_hours_loop()

            mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_active_hours_loop_handles_exceptions(self) -> None:
        """Test active hours loop handles exceptions and retries."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot)
        scheduler._running = True

        call_count = 0

        async def update_side_effect() -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test exception")
            scheduler._running = False
            return {
                "start_hour": 9,
                "end_hour": 21,
                "confidence": 0.85,
                "sample_size": 100,
            }

        with (
            patch(
                "lattice.scheduler.runner.update_active_hours",
                side_effect=update_side_effect,
            ),
            patch("asyncio.sleep", AsyncMock()),
        ):
            await scheduler._active_hours_loop()

            # Should have handled exception and retried
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_run_proactive_check_wait_action(self) -> None:
        """Test run_proactive_check with wait action (exponential backoff)."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot)

        decision = ProactiveDecision(
            action="wait", content=None, reason="Not the right time"
        )

        with (
            patch("lattice.scheduler.runner.decide_proactive", return_value=decision),
            patch("lattice.scheduler.runner.get_current_interval", return_value=15),
            patch(
                "lattice.scheduler.runner.get_system_health",
                return_value="1440",
            ),
            patch(
                "lattice.scheduler.runner.set_current_interval", AsyncMock()
            ) as mock_set,
            patch("lattice.scheduler.runner.set_next_check_at", AsyncMock()),
        ):
            await scheduler._run_proactive_check()

            # Should double interval: 15 * 2 = 30
            mock_set.assert_called_once_with(30)

    @pytest.mark.asyncio
    async def test_run_proactive_check_no_channel(self) -> None:
        """Test run_proactive_check with message action but no channel."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot)

        decision = ProactiveDecision(
            action="message",
            content="Hey there!",
            reason="Time to check in",
            channel_id=None,
        )

        with (
            patch("lattice.scheduler.runner.decide_proactive", return_value=decision),
            patch("lattice.scheduler.runner.get_current_interval", return_value=15),
            patch(
                "lattice.scheduler.runner.get_system_health",
                return_value="1440",
            ),
            patch(
                "lattice.scheduler.runner.set_current_interval", AsyncMock()
            ) as mock_set,
            patch("lattice.scheduler.runner.set_next_check_at", AsyncMock()),
        ):
            await scheduler._run_proactive_check()

            # Should apply exponential backoff: 15 * 2 = 30
            mock_set.assert_called_once_with(30)

    @pytest.mark.asyncio
    async def test_run_proactive_check_message_success(self) -> None:
        """Test run_proactive_check with successful message send."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot, dream_channel_id=67890)

        decision = ProactiveDecision(
            action="message",
            content="Hey! How's the project going?",
            reason="Deadline approaching",
            channel_id=12345,
            rendered_prompt="[prompt content]",
            template_version=1,
            model="gpt-4",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.01,
            latency_ms=500,
        )

        mock_message = MagicMock(spec=discord.Message)
        mock_message.content = "Hey! How's the project going?"
        mock_message.id = 999
        mock_message.jump_url = "https://discord.com/channels/123/456/999"
        mock_message.channel = MagicMock()
        mock_message.channel.id = 12345

        mock_pipeline = MagicMock()
        mock_pipeline.send_proactive_message = AsyncMock(return_value=mock_message)

        with (
            patch("lattice.scheduler.runner.decide_proactive", return_value=decision),
            patch(
                "lattice.scheduler.runner.UnifiedPipeline", return_value=mock_pipeline
            ),
            patch(
                "lattice.scheduler.runner.get_user_timezone",
                return_value="America/New_York",
            ),
            patch("lattice.scheduler.runner.episodic.store_message", return_value=1),
            patch(
                "lattice.scheduler.runner.prompt_audits.store_prompt_audit",
                return_value=uuid4(),
            ),
            patch.object(
                scheduler, "_mirror_proactive_to_dream", AsyncMock()
            ) as mock_mirror,
            patch(
                "lattice.scheduler.runner.get_system_health",
                return_value="15",
            ),
            patch(
                "lattice.scheduler.runner.set_current_interval", AsyncMock()
            ) as mock_set,
            patch("lattice.scheduler.runner.set_next_check_at", AsyncMock()),
        ):
            await scheduler._run_proactive_check()

            # Should reset to base interval
            mock_set.assert_called_once_with(15)
            # Should mirror to dream channel
            mock_mirror.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_proactive_check_send_failure(self) -> None:
        """Test run_proactive_check when message send fails."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot)

        decision = ProactiveDecision(
            action="message",
            content="Hey there!",
            reason="Check in time",
            channel_id=12345,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.send_proactive_message = AsyncMock(return_value=None)

        with (
            patch("lattice.scheduler.runner.decide_proactive", return_value=decision),
            patch(
                "lattice.scheduler.runner.UnifiedPipeline", return_value=mock_pipeline
            ),
            patch("lattice.scheduler.runner.get_current_interval", return_value=15),
            patch(
                "lattice.scheduler.runner.get_system_health",
                return_value="1440",
            ),
            patch(
                "lattice.scheduler.runner.set_current_interval", AsyncMock()
            ) as mock_set,
            patch("lattice.scheduler.runner.set_next_check_at", AsyncMock()),
        ):
            await scheduler._run_proactive_check()

            # Should apply exponential backoff: 15 * 2 = 30
            mock_set.assert_called_once_with(30)

    @pytest.mark.asyncio
    async def test_run_proactive_check_respects_max_interval(self) -> None:
        """Test run_proactive_check respects max_interval cap."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot)

        decision = ProactiveDecision(action="wait", content=None, reason="Not yet")

        with (
            patch("lattice.scheduler.runner.decide_proactive", return_value=decision),
            patch("lattice.scheduler.runner.get_current_interval", return_value=1000),
            patch(
                "lattice.scheduler.runner.get_system_health",
                return_value="1440",  # max = 1440 minutes (24 hours)
            ),
            patch(
                "lattice.scheduler.runner.set_current_interval", AsyncMock()
            ) as mock_set,
            patch("lattice.scheduler.runner.set_next_check_at", AsyncMock()),
        ):
            await scheduler._run_proactive_check()

            # Should cap at max: min(1000 * 2, 1440) = 1440
            mock_set.assert_called_once_with(1440)

    @pytest.mark.asyncio
    async def test_mirror_proactive_to_dream_no_channel(self) -> None:
        """Test mirror skips when dream channel not configured."""
        mock_bot = MagicMock()
        scheduler = ProactiveScheduler(bot=mock_bot, dream_channel_id=None)

        mock_message = MagicMock(spec=discord.Message)
        mock_message.content = "Test"
        mock_message.id = 999
        mock_message.jump_url = "https://discord.com/channels/123/456/999"

        # Should return early without errors
        await scheduler._mirror_proactive_to_dream(
            bot_message=mock_message,
            reasoning="Test reason",
        )

    @pytest.mark.asyncio
    async def test_mirror_proactive_to_dream_channel_not_found(self) -> None:
        """Test mirror handles missing dream channel gracefully."""
        mock_bot = MagicMock()
        mock_bot.get_channel.return_value = None
        scheduler = ProactiveScheduler(bot=mock_bot, dream_channel_id=12345)

        mock_message = MagicMock(spec=discord.Message)
        mock_message.content = "Test"
        mock_message.id = 999
        mock_message.jump_url = "https://discord.com/channels/123/456/999"

        # Should handle missing channel gracefully
        await scheduler._mirror_proactive_to_dream(
            bot_message=mock_message,
            reasoning="Test reason",
        )

    @pytest.mark.asyncio
    async def test_mirror_proactive_to_dream_success(self) -> None:
        """Test successful mirroring to dream channel."""
        mock_dream_channel = AsyncMock()
        mock_bot = MagicMock()
        mock_bot.get_channel.return_value = mock_dream_channel
        scheduler = ProactiveScheduler(bot=mock_bot, dream_channel_id=12345)

        mock_message = MagicMock(spec=discord.Message)
        mock_message.content = "Hey! How's it going?"
        mock_message.id = 999
        mock_message.jump_url = "https://discord.com/channels/123/456/999"

        mock_embed = MagicMock()
        mock_view = MagicMock()

        with patch(
            "lattice.scheduler.runner.DreamMirrorBuilder.build_proactive_mirror",
            return_value=(mock_embed, mock_view),
        ):
            await scheduler._mirror_proactive_to_dream(
                bot_message=mock_message,
                reasoning="Deadline approaching",
                audit_id=uuid4(),
                prompt_key="PROACTIVE_DECISION",
                template_version=1,
                rendered_prompt="[prompt]",
            )

            mock_dream_channel.send.assert_called_once_with(
                embed=mock_embed, view=mock_view
            )

    @pytest.mark.asyncio
    async def test_mirror_proactive_to_dream_handles_exceptions(self) -> None:
        """Test mirror handles exceptions during send."""
        mock_dream_channel = AsyncMock()
        mock_dream_channel.send.side_effect = Exception("Discord API error")
        mock_bot = MagicMock()
        mock_bot.get_channel.return_value = mock_dream_channel
        scheduler = ProactiveScheduler(bot=mock_bot, dream_channel_id=12345)

        mock_message = MagicMock(spec=discord.Message)
        mock_message.content = "Test"
        mock_message.id = 999
        mock_message.jump_url = "https://discord.com/channels/123/456/999"

        mock_embed = MagicMock()
        mock_view = MagicMock()

        with patch(
            "lattice.scheduler.runner.DreamMirrorBuilder.build_proactive_mirror",
            return_value=(mock_embed, mock_view),
        ):
            # Should not raise exception
            await scheduler._mirror_proactive_to_dream(
                bot_message=mock_message,
                reasoning="Test reason",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
