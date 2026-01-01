"""Unit tests for scheduler components."""

import pytest
from unittest.mock import MagicMock

from lattice.core.types import GhostContext, GhostTriggerReason, PipelineSourceType
from lattice.scheduler.adaptive import AdaptiveScheduler, ScheduleConfig
from lattice.scheduler.engagement import EngagementScorer
from lattice.scheduler.recursion_guard import RecursionGuard
from lattice.scheduler.safety_filter import GhostContentSafetyFilter
from lattice.scheduler.triggers import TriggerDetector


class TestRecursionGuard:
    """Tests for RecursionGuard safety component."""

    def test_initial_depth_is_zero(self) -> None:
        guard = RecursionGuard()
        assert guard._depth == 0

    def test_max_depth_default(self) -> None:
        assert RecursionGuard.MAX_GHOST_DEPTH == 2

    @pytest.mark.asyncio
    async def test_enter_ghost_increments_depth(self) -> None:
        guard = RecursionGuard()
        result = await guard.enter_ghost()
        assert result is True
        assert guard._depth == 1

    @pytest.mark.asyncio
    async def test_exit_ghost_decrements_depth(self) -> None:
        guard = RecursionGuard()
        await guard.enter_ghost()
        guard.exit_ghost()
        assert guard._depth == 0

    @pytest.mark.asyncio
    async def test_enter_ghost_blocks_at_max_depth(self) -> None:
        guard = RecursionGuard()
        guard._depth = guard.MAX_GHOST_DEPTH
        result = await guard.enter_ghost()
        assert result is False

    @pytest.mark.asyncio
    async def test_multiple_ghosts_allowable(self) -> None:
        guard = RecursionGuard()
        results = [await guard.enter_ghost() for _ in range(guard.MAX_GHOST_DEPTH)]
        assert all(results)
        assert guard._depth == guard.MAX_GHOST_DEPTH

    @pytest.mark.asyncio
    async def test_depth_never_negative(self) -> None:
        guard = RecursionGuard()
        guard.exit_ghost()
        guard.exit_ghost()
        assert guard._depth == 0


class TestGhostContentSafetyFilter:
    """Tests for GhostContentSafetyFilter."""

    def test_max_content_length(self) -> None:
        assert GhostContentSafetyFilter.MAX_CONTENT_LENGTH == 1900

    def test_self_reference_patterns(self) -> None:
        filter_ = GhostContentSafetyFilter(db_pool=MagicMock())
        assert "I am a bot" in filter_.SELF_REFERENCE_PATTERNS
        assert "I was programmed" in filter_.SELF_REFERENCE_PATTERNS

    @pytest.mark.asyncio
    async def test_empty_content_is_safe(self) -> None:
        filter_ = GhostContentSafetyFilter(db_pool=MagicMock())
        is_safe, _ = await filter_.filter("")
        assert is_safe is True

    @pytest.mark.asyncio
    async def test_content_too_long_is_unsafe(self) -> None:
        filter_ = GhostContentSafetyFilter(db_pool=MagicMock())
        long_content = "x" * 2000
        is_safe, reason = await filter_.filter(long_content)
        assert is_safe is False
        assert "too long" in reason

    @pytest.mark.asyncio
    async def test_self_reference_is_unsafe(self) -> None:
        filter_ = GhostContentSafetyFilter(db_pool=MagicMock())
        is_safe, reason = await filter_.filter("I am a bot and I help you")
        assert is_safe is False
        assert "self-reference" in reason

    def test_content_hash_computation(self) -> None:
        filter_ = GhostContentSafetyFilter(db_pool=MagicMock())
        hash1 = filter_._compute_content_hash("hello world")
        hash2 = filter_._compute_content_hash("hello world")
        assert hash1 == hash2
        assert len(hash1) == 64

        different_hash = filter_._compute_content_hash("different content")
        assert hash1 != different_hash


class TestEngagementScorer:
    """Tests for EngagementScorer."""

    def test_default_weights(self) -> None:
        scorer = EngagementScorer(db_pool=MagicMock())
        assert scorer.WEIGHTS["response_rate"] == 0.35
        assert scorer.WEIGHTS["message_frequency"] == 0.25
        assert scorer.WEIGHTS["feedback_positive"] == 0.25
        assert scorer.WEIGHTS["conversation_depth"] == 0.15

    def test_weights_sum_to_one(self) -> None:
        scorer = EngagementScorer(db_pool=MagicMock())
        total = sum(scorer.WEIGHTS.values())
        assert abs(total - 1.0) < 0.01


class TestAdaptiveScheduler:
    """Tests for AdaptiveScheduler."""

    def test_min_interval_minutes(self) -> None:
        assert AdaptiveScheduler.MIN_INTERVAL_MINUTES == 15

    def test_max_interval_minutes(self) -> None:
        assert AdaptiveScheduler.MAX_INTERVAL_MINUTES == 10080

    def test_adjustment_step(self) -> None:
        assert AdaptiveScheduler.ADJUSTMENT_STEP == 0.1

    def test_engagement_thresholds(self) -> None:
        assert AdaptiveScheduler.HIGH_ENGAGEMENT_THRESHOLD == 0.7
        assert AdaptiveScheduler.LOW_ENGAGEMENT_THRESHOLD == 0.3

    @pytest.mark.asyncio
    async def test_calculate_next_interval_high_engagement_decreases(self) -> None:
        scheduler = AdaptiveScheduler(db_pool=MagicMock())
        config = ScheduleConfig(
            user_id=1,
            channel_id=2,
            base_interval_minutes=60,
            current_interval_minutes=60,
            min_interval_minutes=15,
            max_interval_minutes=10080,
            last_engagement_score=0.5,
            engagement_count=10,
            is_paused=False,
            opt_out=False,
            user_timezone="UTC",
            last_ghost_at=None,
            next_scheduled_at=None,
        )
        new_interval = await scheduler.calculate_next_interval(config, engagement_score=0.8)
        assert new_interval < 60

    @pytest.mark.asyncio
    async def test_calculate_next_interval_low_engagement_increases(self) -> None:
        scheduler = AdaptiveScheduler(db_pool=MagicMock())
        config = ScheduleConfig(
            user_id=1,
            channel_id=2,
            base_interval_minutes=60,
            current_interval_minutes=60,
            min_interval_minutes=15,
            max_interval_minutes=10080,
            last_engagement_score=0.5,
            engagement_count=10,
            is_paused=False,
            opt_out=False,
            user_timezone="UTC",
            last_ghost_at=None,
            next_scheduled_at=None,
        )
        new_interval = await scheduler.calculate_next_interval(config, engagement_score=0.2)
        assert new_interval > 60

    @pytest.mark.asyncio
    async def test_calculate_next_interval_stable_engagement_no_change(self) -> None:
        scheduler = AdaptiveScheduler(db_pool=MagicMock())
        config = ScheduleConfig(
            user_id=1,
            channel_id=2,
            base_interval_minutes=60,
            current_interval_minutes=60,
            min_interval_minutes=15,
            max_interval_minutes=10080,
            last_engagement_score=0.5,
            engagement_count=10,
            is_paused=False,
            opt_out=False,
            user_timezone="UTC",
            last_ghost_at=None,
            next_scheduled_at=None,
        )
        new_interval = await scheduler.calculate_next_interval(config, engagement_score=0.5)
        assert new_interval == 60

    @pytest.mark.asyncio
    async def test_respects_min_interval(self) -> None:
        scheduler = AdaptiveScheduler(db_pool=MagicMock())
        config = ScheduleConfig(
            user_id=1,
            channel_id=2,
            base_interval_minutes=60,
            current_interval_minutes=20,
            min_interval_minutes=15,
            max_interval_minutes=10080,
            last_engagement_score=0.5,
            engagement_count=10,
            is_paused=False,
            opt_out=False,
            user_timezone="UTC",
            last_ghost_at=None,
            next_scheduled_at=None,
        )
        new_interval = await scheduler.calculate_next_interval(config, engagement_score=0.9)
        assert new_interval >= config.min_interval_minutes

    @pytest.mark.asyncio
    async def test_respects_max_interval(self) -> None:
        scheduler = AdaptiveScheduler(db_pool=MagicMock())
        config = ScheduleConfig(
            user_id=1,
            channel_id=2,
            base_interval_minutes=60,
            current_interval_minutes=5000,
            min_interval_minutes=15,
            max_interval_minutes=10080,
            last_engagement_score=0.5,
            engagement_count=10,
            is_paused=False,
            opt_out=False,
            user_timezone="UTC",
            last_ghost_at=None,
            next_scheduled_at=None,
        )
        new_interval = await scheduler.calculate_next_interval(config, engagement_score=0.2)
        assert new_interval <= config.max_interval_minutes


class TestTriggerDetector:
    """Tests for TriggerDetector."""

    def test_engagement_lapse_hours(self) -> None:
        assert TriggerDetector.ENGAGEMENT_LAPSE_HOURS == 72

    def test_milestone_keywords(self) -> None:
        detector = TriggerDetector(db_pool=MagicMock())
        assert "completed" in detector.MILESTONE_KEYWORDS
        assert "achieved" in detector.MILESTONE_KEYWORDS
        assert "finished" in detector.MILESTONE_KEYWORDS

    @pytest.mark.asyncio
    async def test_check_milestone_true(self) -> None:
        detector = TriggerDetector(db_pool=MagicMock())
        result = await detector.check_milestone("I just completed my goal!")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_milestone_false(self) -> None:
        detector = TriggerDetector(db_pool=MagicMock())
        result = await detector.check_milestone("Hello, how are you?")
        assert result is False


class TestGhostContext:
    """Tests for GhostContext dataclass."""

    def test_ghost_context_creation(self) -> None:
        ctx = GhostContext(
            user_id=12345,
            channel_id=67890,
            trigger_reason=GhostTriggerReason.SCHEDULED,
            engagement_score=0.75,
            last_ghost_at=None,
            recent_message_count=5,
            content_hash=None,
            metadata={},
        )
        assert ctx.user_id == 12345
        assert ctx.channel_id == 67890
        assert ctx.trigger_reason == GhostTriggerReason.SCHEDULED
        assert ctx.engagement_score == 0.75

    def test_ghost_trigger_reason_values(self) -> None:
        assert GhostTriggerReason.SCHEDULED.value == "scheduled"
        assert GhostTriggerReason.MILESTONE.value == "milestone"
        assert GhostTriggerReason.ENGAGEMENT_LAPSE.value == "engagement_lapse"
        assert GhostTriggerReason.CONTEXT_SWITCH.value == "context_switch"
        assert GhostTriggerReason.MANUAL.value == "manual"

    def test_pipeline_source_type_values(self) -> None:
        assert PipelineSourceType.USER.value == "user"
        assert PipelineSourceType.GHOST.value == "ghost"
        assert PipelineSourceType.SYSTEM.value == "system"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
