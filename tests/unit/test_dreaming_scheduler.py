"""Tests for the dreaming scheduler module.

Tests the autonomous prompt optimization cycle scheduler.
"""

import asyncio
from datetime import UTC, datetime, time, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from lattice.scheduler.dreaming import (
    DEFAULT_DREAM_TIME,
    DreamingConfig,
    DreamingScheduler,
    trigger_dreaming_cycle_manually,
)


class TestDreamingConfig:
    """Tests for the DreamingConfig dataclass."""

    def test_dreaming_config_creation(self) -> None:
        """Test that DreamingConfig can be created correctly."""
        config = DreamingConfig(
            min_uses=10,
            lookback_days=30,
            enabled=True,
        )

        assert config.min_uses == 10
        assert config.lookback_days == 30
        assert config.enabled is True

    def test_dreaming_config_defaults(self) -> None:
        """Test DreamingConfig with minimal arguments."""
        config = DreamingConfig(
            min_uses=5,
            lookback_days=14,
            enabled=False,
        )

        assert config.min_uses == 5
        assert config.lookback_days == 14
        assert config.enabled is False


class TestDreamingSchedulerInit:
    """Tests for DreamingScheduler initialization."""

    def test_requires_llm_client(self) -> None:
        """Test that initializing without llm_client raises TypeError."""
        with pytest.raises(TypeError, match="llm_client is required"):
            DreamingScheduler(
                bot=MagicMock(),
                prompt_repo=MagicMock(),
                proposal_repo=MagicMock(),
            )

    def test_requires_prompt_repo(self) -> None:
        """Test that initializing without prompt_repo raises TypeError."""
        with pytest.raises(TypeError, match="prompt_repo is required"):
            DreamingScheduler(
                bot=MagicMock(),
                llm_client=MagicMock(),
                proposal_repo=MagicMock(),
            )

    def test_requires_proposal_repo(self) -> None:
        """Test that initializing without proposal_repo raises TypeError."""
        with pytest.raises(TypeError, match="proposal_repo is required"):
            DreamingScheduler(
                bot=MagicMock(),
                llm_client=MagicMock(),
                prompt_repo=MagicMock(),
            )

    def test_initializes_with_all_repos(self) -> None:
        """Test successful initialization with all required repos."""
        mock_bot = MagicMock()
        mock_llm = MagicMock()
        mock_prompt_repo = MagicMock()
        mock_proposal_repo = MagicMock()

        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=mock_llm,
            prompt_repo=mock_prompt_repo,
            proposal_repo=mock_proposal_repo,
        )

        assert scheduler.bot is mock_bot
        assert scheduler.llm_client is mock_llm
        assert scheduler.prompt_repo is mock_prompt_repo
        assert scheduler.proposal_repo is mock_proposal_repo

    def test_initializes_with_optional_params(self) -> None:
        """Test initialization with optional parameters."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            dream_channel_id=12345,
            dream_time=time(4, 0),
            semantic_repo=MagicMock(),
            llm_client=MagicMock(),
            prompt_audit_repo=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        assert scheduler.dream_channel_id == 12345
        assert scheduler.dream_time == time(4, 0)
        assert scheduler.semantic_repo is not None
        assert scheduler.prompt_audit_repo is not None

    def test_default_dream_time(self) -> None:
        """Test that default dream time is 3:00 AM UTC."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        assert scheduler.dream_time == DEFAULT_DREAM_TIME


class TestDreamingSchedulerStartStop:
    """Tests for DreamingScheduler start and stop methods."""

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self) -> None:
        """Test that start sets the _running flag."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        await scheduler.start()

        assert scheduler._running is True
        assert scheduler._scheduler_task is not None

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running_flag(self) -> None:
        """Test that stop clears the _running flag."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        await scheduler.start()
        await scheduler.stop()

        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_stop_handles_cancelled_error(self) -> None:
        """Test that stop handles CancelledError gracefully."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        scheduler._running = True
        scheduler._scheduler_task = asyncio.create_task(asyncio.sleep(100))

        await scheduler.stop()

        assert scheduler._running is False
        assert scheduler._scheduler_task.cancelled()


class TestCalculateNextRun:
    """Tests for the _calculate_next_run method."""

    def test_returns_today_run_if_not_passed(self) -> None:
        """Test that next run is today if dream time hasn't passed."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            dream_time=time(12, 0),
        )

        with patch(
            "lattice.scheduler.dreaming.get_now",
            return_value=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        ):
            next_run = scheduler._calculate_next_run()

        assert next_run.date() == datetime(2024, 1, 15, tzinfo=UTC).date()
        assert next_run.hour == 12
        assert next_run.minute == 0

    def test_returns_tomorrow_run_if_passed(self) -> None:
        """Test that next run is tomorrow if dream time has passed."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            dream_time=time(2, 0),
        )

        with patch(
            "lattice.scheduler.dreaming.get_now",
            return_value=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        ):
            next_run = scheduler._calculate_next_run()

        expected = datetime(2024, 1, 16, 2, 0, 0, tzinfo=UTC)
        assert next_run == expected

    def test_handles_midnight_dream_time(self) -> None:
        """Test with midnight dream time."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            dream_time=time(0, 0),
        )

        with patch(
            "lattice.scheduler.dreaming.get_now",
            return_value=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        ):
            next_run = scheduler._calculate_next_run()

        expected = datetime(2024, 1, 16, 0, 0, 0, tzinfo=UTC)
        assert next_run == expected

    def test_handles_late_night_dream_time(self) -> None:
        """Test with late night dream time (23:59)."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            dream_time=time(23, 59),
        )

        with patch(
            "lattice.scheduler.dreaming.get_now",
            return_value=datetime(2024, 1, 15, 20, 0, 0, tzinfo=UTC),
        ):
            next_run = scheduler._calculate_next_run()

        assert next_run.date() == datetime(2024, 1, 15, tzinfo=UTC).date()
        assert next_run.hour == 23
        assert next_run.minute == 59


class TestIsEnabled:
    """Tests for the _is_enabled method."""

    @pytest.mark.asyncio
    async def test_returns_true_by_default(self) -> None:
        """Test that dreaming is enabled by default."""
        mock_bot = MagicMock()
        mock_bot.db_pool.get_system_metrics = AsyncMock(return_value=None)

        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        result = await scheduler._is_enabled()

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_when_enabled(self) -> None:
        """Test that returns True when enabled in database."""
        mock_bot = MagicMock()
        mock_bot.db_pool.get_system_metrics = AsyncMock(return_value="true")

        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        result = await scheduler._is_enabled()

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_disabled(self) -> None:
        """Test that returns False when disabled in database."""
        mock_bot = MagicMock()
        mock_bot.db_pool.get_system_metrics = AsyncMock(return_value="false")

        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        result = await scheduler._is_enabled()

        assert result is False


class TestGetDreamingConfig:
    """Tests for the _get_dreaming_config method."""

    @pytest.mark.asyncio
    async def test_returns_default_config(self) -> None:
        """Test that default config is returned when no database values."""
        mock_bot = MagicMock()
        mock_bot.db_pool.get_system_metrics = AsyncMock(return_value=None)

        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        config = await scheduler._get_dreaming_config()

        assert config.min_uses == 10
        assert config.lookback_days == 30
        assert config.enabled is True

    @pytest.mark.asyncio
    async def test_loads_config_from_database(self) -> None:
        """Test that config is loaded from database."""
        mock_bot = MagicMock()
        mock_bot.db_pool.get_system_metrics = AsyncMock(
            side_effect=lambda key: {
                "dreaming_min_uses": "5",
                "dreaming_lookback_days": "14",
                "dreaming_enabled": "false",
            }.get(key)
        )

        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        config = await scheduler._get_dreaming_config()

        assert config.min_uses == 5
        assert config.lookback_days == 14
        assert config.enabled is False


class TestRunDreamingCycle:
    """Tests for the _run_dreaming_cycle method."""

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self) -> None:
        """Test that cycle skips when dreaming is disabled."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            prompt_audit_repo=MagicMock(),
        )

        scheduler._get_dreaming_config = AsyncMock(
            return_value=DreamingConfig(min_uses=10, lookback_days=30, enabled=False)
        )

        with patch(
            "lattice.scheduler.dreaming.analyze_prompt_effectiveness",
            AsyncMock(return_value=[]),
        ):
            result = await scheduler._run_dreaming_cycle()

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_skips_when_no_prompt_audit_repo(self) -> None:
        """Test that cycle skips when prompt_audit_repo is not initialized."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        scheduler._get_dreaming_config = AsyncMock(
            return_value=DreamingConfig(min_uses=10, lookback_days=30, enabled=True)
        )

        result = await scheduler._run_dreaming_cycle()

        assert result["status"] == "skipped"
        assert "not initialized" in result["message"]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_metrics(self) -> None:
        """Test that empty result is returned when no prompts meet threshold."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            prompt_audit_repo=MagicMock(),
        )

        scheduler._get_dreaming_config = AsyncMock(
            return_value=DreamingConfig(min_uses=10, lookback_days=30, enabled=True)
        )

        with patch(
            "lattice.scheduler.dreaming.analyze_prompt_effectiveness",
            AsyncMock(return_value=[]),
        ):
            result = await scheduler._run_dreaming_cycle()

        assert result["status"] == "success"
        assert result["prompts_analyzed"] == 0
        assert result["proposals_created"] == 0

    @pytest.mark.asyncio
    async def test_creates_proposals_for_metrics(self) -> None:
        """Test that proposals are created for underperforming prompts."""
        mock_bot = MagicMock()
        mock_proposal = MagicMock()
        mock_proposal.proposal_id = uuid4()
        mock_proposal.prompt_key = "TEST_PROMPT"
        mock_proposal.current_version = 1
        mock_proposal.proposed_version = 2
        mock_proposal.proposal_metadata = {"pain_point": "Test issue"}

        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            prompt_audit_repo=MagicMock(),
        )

        scheduler._get_dreaming_config = AsyncMock(
            return_value=DreamingConfig(min_uses=10, lookback_days=30, enabled=True)
        )

        mock_metrics = MagicMock()
        mock_metrics.prompt_key = "TEST_PROMPT"

        with patch(
            "lattice.scheduler.dreaming.analyze_prompt_effectiveness",
            AsyncMock(return_value=[mock_metrics]),
        ):
            with patch(
                "lattice.scheduler.dreaming.reject_stale_proposals",
                AsyncMock(return_value=0),
            ):
                with patch(
                    "lattice.scheduler.dreaming.propose_optimization",
                    AsyncMock(return_value=mock_proposal),
                ):
                    with patch(
                        "lattice.scheduler.dreaming.store_proposal",
                        AsyncMock(return_value=mock_proposal.proposal_id),
                    ):
                        result = await scheduler._run_dreaming_cycle()

        assert result["status"] == "success"
        assert result["prompts_analyzed"] == 1
        assert result["proposals_created"] == 1

    @pytest.mark.asyncio
    async def test_force_bypasses_thresholds(self) -> None:
        """Test that force=True bypasses statistical thresholds."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            prompt_audit_repo=MagicMock(),
        )

        scheduler._get_dreaming_config = AsyncMock(
            return_value=DreamingConfig(min_uses=100, lookback_days=30, enabled=True)
        )

        with patch(
            "lattice.scheduler.dreaming.analyze_prompt_effectiveness",
            AsyncMock(return_value=[]),
        ):
            result = await scheduler._run_dreaming_cycle(force=True)

        assert result["status"] == "success"


class TestGetDreamChannel:
    """Tests for the _get_dream_channel method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_channel_id(self) -> None:
        """Test that None is returned when dream_channel_id is not set."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        result = await scheduler._get_dream_channel()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_channel_not_found(self) -> None:
        """Test that None is returned when channel doesn't exist."""
        mock_bot = MagicMock()
        mock_bot.get_channel = MagicMock(return_value=None)

        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            dream_channel_id=12345,
        )

        result = await scheduler._get_dream_channel()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_channel_when_found(self) -> None:
        """Test that channel is returned when found."""
        import discord

        mock_channel = MagicMock(spec=discord.TextChannel)

        mock_bot = MagicMock()
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            dream_channel_id=12345,
        )

        result = await scheduler._get_dream_channel()

        assert result is mock_channel


class TestFormatProposalSummary:
    """Tests for the _format_proposal_summary method."""

    def test_formats_proposal_correctly(self) -> None:
        """Test that proposal summary is formatted correctly."""
        mock_proposal = MagicMock()
        mock_proposal.prompt_key = "TEST_KEY"
        mock_proposal.current_version = 1
        mock_proposal.proposed_version = 2
        mock_proposal.proposal_id = uuid4()
        mock_proposal.proposal_metadata = {"pain_point": "Low success rate"}

        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        result = scheduler._format_proposal_summary(mock_proposal)

        assert "TEST_KEY" in result
        assert "v1" in result
        assert "v2" in result
        assert "Low success rate" in result


class TestTriggerDreamingCycleManually:
    """Tests for the trigger_dreaming_cycle_manually function."""

    @pytest.mark.asyncio
    async def test_creates_scheduler_and_runs_cycle(self) -> None:
        """Test that manual trigger creates scheduler and runs cycle."""
        mock_bot = MagicMock()
        mock_bot.prompt_repo = MagicMock()
        mock_bot.prompt_audit_repo = MagicMock()
        mock_bot.audit_repo = MagicMock()
        mock_bot.feedback_repo = MagicMock()
        mock_bot.proposal_repo = MagicMock()

        with patch("lattice.scheduler.dreaming.DreamingScheduler") as MockScheduler:
            mock_scheduler = MagicMock()
            mock_scheduler._run_dreaming_cycle = AsyncMock(
                return_value={"status": "success"}
            )
            MockScheduler.return_value = mock_scheduler

            await trigger_dreaming_cycle_manually(
                bot=mock_bot,
                dream_channel_id=12345,
                force=True,
            )

            MockScheduler.assert_called_once()
            mock_scheduler._run_dreaming_cycle.assert_called_once_with(force=True)


class TestSchedulerLoop:
    """Tests for the scheduler loop behavior."""

    @pytest.mark.asyncio
    async def test_loop_calculates_sleep_time(self) -> None:
        """Test that loop calculates correct sleep time."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        with patch(
            "lattice.scheduler.dreaming.get_now",
            return_value=datetime(2024, 1, 15, 2, 0, 0, tzinfo=UTC),
        ):
            scheduler._calculate_next_run = MagicMock(
                return_value=datetime(2024, 1, 15, 3, 0, 0, tzinfo=UTC)
            )
            scheduler._is_enabled = AsyncMock(return_value=True)
            scheduler._run_dreaming_cycle = AsyncMock(
                return_value={"status": "success"}
            )

            scheduler._running = True
            scheduler._scheduler_task = asyncio.create_task(_stop_scheduler(scheduler))

            await asyncio.sleep(0.1)


async def _stop_scheduler(scheduler: DreamingScheduler) -> None:
    """Helper to stop scheduler after a short delay."""
    await asyncio.sleep(0.05)
    scheduler._running = False


class TestMemoryReviewCycle:
    """Tests for memory review cycle integration."""

    @pytest.mark.asyncio
    async def test_runs_memory_review_after_proposals(self) -> None:
        """Test that memory review runs after proposal creation."""
        import discord

        mock_bot = MagicMock()
        mock_review_id = uuid4()
        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            prompt_audit_repo=MagicMock(),
            dream_channel_id=12345,
        )

        scheduler._get_dreaming_config = AsyncMock(
            return_value=DreamingConfig(min_uses=1, lookback_days=30, enabled=True)
        )
        scheduler._run_memory_review_cycle = AsyncMock(return_value=str(mock_review_id))

        mock_proposal = MagicMock()
        mock_proposal.proposal_id = uuid4()
        mock_proposal.prompt_key = "TEST"
        mock_proposal.current_version = 1
        mock_proposal.proposed_version = 2
        mock_proposal.proposal_metadata = {"pain_point": "test"}

        mock_metrics = MagicMock()
        mock_metrics.prompt_key = "TEST"

        with patch(
            "lattice.scheduler.dreaming.analyze_prompt_effectiveness",
            AsyncMock(return_value=[mock_metrics]),
        ):
            with patch(
                "lattice.scheduler.dreaming.reject_stale_proposals",
                AsyncMock(return_value=0),
            ):
                with patch(
                    "lattice.scheduler.dreaming.propose_optimization",
                    AsyncMock(return_value=mock_proposal),
                ):
                    with patch(
                        "lattice.scheduler.dreaming.store_proposal",
                        AsyncMock(return_value=mock_proposal.proposal_id),
                    ):
                        result = await scheduler._run_dreaming_cycle()

        assert result["status"] == "success"
        scheduler._run_memory_review_cycle.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases in dreaming scheduler."""

    @pytest.mark.asyncio
    async def test_handles_empty_proposal_list(self) -> None:
        """Test handling when no proposals are generated."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            prompt_audit_repo=MagicMock(),
        )

        scheduler._get_dreaming_config = AsyncMock(
            return_value=DreamingConfig(min_uses=10, lookback_days=30, enabled=True)
        )

        mock_metrics = MagicMock()
        mock_metrics.prompt_key = "TEST"

        with patch(
            "lattice.scheduler.dreaming.analyze_prompt_effectiveness",
            AsyncMock(return_value=[mock_metrics]),
        ):
            with patch(
                "lattice.scheduler.dreaming.reject_stale_proposals",
                AsyncMock(return_value=0),
            ):
                with patch(
                    "lattice.scheduler.dreaming.propose_optimization",
                    AsyncMock(return_value=None),
                ):
                    result = await scheduler._run_dreaming_cycle()

        assert result["proposals_created"] == 0

    @pytest.mark.asyncio
    async def test_limits_proposals_per_cycle(self) -> None:
        """Test that MAX_PROPOSALS_PER_CYCLE limit is respected."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            prompt_audit_repo=MagicMock(),
        )

        scheduler._get_dreaming_config = AsyncMock(
            return_value=DreamingConfig(min_uses=1, lookback_days=30, enabled=True)
        )

        mock_metrics = [MagicMock() for _ in range(10)]
        for i, m in enumerate(mock_metrics):
            m.prompt_key = f"PROMPT_{i}"

        with patch(
            "lattice.scheduler.dreaming.analyze_prompt_effectiveness",
            AsyncMock(return_value=mock_metrics),
        ):
            with patch(
                "lattice.scheduler.dreaming.reject_stale_proposals",
                AsyncMock(return_value=0),
            ):
                with patch(
                    "lattice.scheduler.dreaming.propose_optimization",
                    AsyncMock(return_value=None),
                ):
                    result = await scheduler._run_dreaming_cycle()

        assert result["prompts_analyzed"] == 10

    @pytest.mark.asyncio
    async def test_memory_review_failure_handled(self) -> None:
        """Test that memory review failure doesn't fail entire cycle."""
        mock_bot = MagicMock()
        scheduler = DreamingScheduler(
            bot=mock_bot,
            llm_client=MagicMock(),
            prompt_repo=MagicMock(),
            proposal_repo=MagicMock(),
            prompt_audit_repo=MagicMock(),
        )

        scheduler._get_dreaming_config = AsyncMock(
            return_value=DreamingConfig(min_uses=10, lookback_days=30, enabled=True)
        )
        scheduler._run_memory_review_cycle = AsyncMock(
            side_effect=Exception("Memory review failed")
        )

        with patch(
            "lattice.scheduler.dreaming.analyze_prompt_effectiveness",
            AsyncMock(return_value=[]),
        ):
            result = await scheduler._run_dreaming_cycle()

        assert result["status"] == "success"
