"""Tests for dreaming analyzer module.

Tests the PromptMetrics dataclass and analyzer functions for calculating
prompt effectiveness metrics from audit data and feedback.
"""

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from lattice.dreaming.analyzer import (
    PromptMetrics,
    _safe_float,
    _safe_float_optional,
    analyze_prompt_effectiveness,
    get_feedback_samples,
    get_feedback_with_context,
)


class TestSafeFloat:
    """Tests for _safe_float helper function."""

    def test_safe_float_with_none_returns_default(self) -> None:
        """Test that None returns the default value."""
        result = _safe_float(None)
        assert result == 0.0

    def test_safe_float_with_none_custom_default(self) -> None:
        """Test that None returns custom default value."""
        result = _safe_float(None, default=5.0)
        assert result == 5.0

    def test_safe_float_with_valid_float(self) -> None:
        """Test conversion of valid float."""
        result = _safe_float(3.14159)
        assert result == 3.14159

    def test_safe_float_with_valid_int(self) -> None:
        """Test conversion of valid integer."""
        result = _safe_float(42)
        assert result == 42.0

    def test_safe_float_with_valid_string(self) -> None:
        """Test conversion of valid string number."""
        result = _safe_float("2.718")
        assert result == 2.718

    def test_safe_float_with_invalid_string(self) -> None:
        """Test that invalid string returns default."""
        result = _safe_float("not a number")
        assert result == 0.0

    def test_safe_float_with_empty_string(self) -> None:
        """Test that empty string returns default."""
        result = _safe_float("")
        assert result == 0.0

    def test_safe_float_with_list(self) -> None:
        """Test that list returns default."""
        result = _safe_float([1, 2, 3])
        assert result == 0.0


class TestSafeFloatOptional:
    """Tests for _safe_float_optional helper function."""

    def test_safe_float_optional_with_none(self) -> None:
        """Test that None returns None."""
        result = _safe_float_optional(None)
        assert result is None

    def test_safe_float_optional_with_valid_value(self) -> None:
        """Test conversion of valid value returns float."""
        result = _safe_float_optional(1.5)
        assert result == 1.5

    def test_safe_float_optional_with_invalid_value(self) -> None:
        """Test that invalid value returns None."""
        result = _safe_float_optional("invalid")
        assert result is None


class TestPromptMetrics:
    """Tests for PromptMetrics dataclass."""

    def test_prompt_metrics_creation(self) -> None:
        """Test creating a PromptMetrics instance."""
        metrics = PromptMetrics(
            prompt_key="TEST_PROMPT",
            version=1,
            total_uses=100,
            uses_with_feedback=20,
            feedback_rate=0.2,
            positive_feedback=15,
            negative_feedback=5,
            success_rate=0.95,
            avg_latency_ms=150.0,
            avg_tokens=200.0,
            avg_cost_usd=Decimal("0.002"),
            priority_score=30.0,
        )

        assert metrics.prompt_key == "TEST_PROMPT"
        assert metrics.version == 1
        assert metrics.total_uses == 100
        assert metrics.positive_feedback == 15
        assert metrics.negative_feedback == 5
        assert metrics.priority_score == 30.0

    def test_prompt_metrics_optional_fields(self) -> None:
        """Test creating PromptMetrics with optional None fields."""
        metrics = PromptMetrics(
            prompt_key="TEST_PROMPT",
            version=1,
            total_uses=50,
            uses_with_feedback=0,
            feedback_rate=0.0,
            positive_feedback=0,
            negative_feedback=0,
            success_rate=1.0,
            avg_latency_ms=None,
            avg_tokens=None,
            avg_cost_usd=None,
            priority_score=0.0,
        )

        assert metrics.avg_latency_ms is None
        assert metrics.avg_tokens is None
        assert metrics.avg_cost_usd is None


class TestAnalyzePromptEffectiveness:
    """Tests for analyze_prompt_effectiveness function."""

    @pytest.mark.asyncio
    async def test_analyze_prompt_effectiveness_returns_metrics(self) -> None:
        """Test that analyze_prompt_effectiveness returns list of metrics."""
        mock_rows = [
            {
                "prompt_key": "BASIC_RESPONSE",
                "template_version": 1,
                "total_uses": 100,
                "uses_with_feedback": 10,
                "feedback_rate": 0.1,
                "positive_feedback": 5,
                "negative_feedback": 5,
                "success_rate": 0.95,
                "avg_latency_ms": 120.0,
                "avg_tokens": 150.0,
                "avg_cost_usd": Decimal("0.001"),
                "priority_score": 25.0,
            }
        ]

        mock_prompt_audit_repo = AsyncMock()
        mock_prompt_audit_repo.analyze_prompt_effectiveness = AsyncMock(
            return_value=mock_rows
        )

        metrics = await analyze_prompt_effectiveness(
            prompt_audit_repo=mock_prompt_audit_repo,
            min_uses=10,
            lookback_days=30,
            min_feedback=0,
        )

        assert len(metrics) == 1
        assert metrics[0].prompt_key == "BASIC_RESPONSE"
        assert metrics[0].total_uses == 100
        assert metrics[0].success_rate == 0.95
        assert metrics[0].priority_score == 25.0

    @pytest.mark.asyncio
    async def test_analyze_prompt_effectiveness_empty_results(self) -> None:
        """Test that empty results return empty list."""
        mock_prompt_audit_repo = AsyncMock()
        mock_prompt_audit_repo.analyze_prompt_effectiveness = AsyncMock(return_value=[])

        metrics = await analyze_prompt_effectiveness(
            prompt_audit_repo=mock_prompt_audit_repo
        )

        assert len(metrics) == 0

    @pytest.mark.asyncio
    async def test_analyze_prompt_effectiveness_multiple_prompts(self) -> None:
        """Test analysis with multiple prompts returns sorted metrics."""
        mock_rows = [
            {
                "prompt_key": "PROMPT_A",
                "template_version": 1,
                "total_uses": 100,
                "uses_with_feedback": 10,
                "feedback_rate": 0.1,
                "positive_feedback": 3,
                "negative_feedback": 7,
                "success_rate": 0.93,
                "avg_latency_ms": 120.0,
                "avg_tokens": 150.0,
                "avg_cost_usd": Decimal("0.001"),
                "priority_score": 50.0,
            },
            {
                "prompt_key": "PROMPT_B",
                "template_version": 2,
                "total_uses": 200,
                "uses_with_feedback": 30,
                "feedback_rate": 0.15,
                "positive_feedback": 28,
                "negative_feedback": 2,
                "success_rate": 0.99,
                "avg_latency_ms": 80.0,
                "avg_tokens": 100.0,
                "avg_cost_usd": Decimal("0.0005"),
                "priority_score": 10.0,
            },
        ]

        mock_prompt_audit_repo = AsyncMock()
        mock_prompt_audit_repo.analyze_prompt_effectiveness = AsyncMock(
            return_value=mock_rows
        )

        metrics = await analyze_prompt_effectiveness(
            prompt_audit_repo=mock_prompt_audit_repo
        )

        assert len(metrics) == 2
        # Should be sorted by priority_score descending
        assert metrics[0].prompt_key == "PROMPT_A"
        assert metrics[1].prompt_key == "PROMPT_B"

    @pytest.mark.asyncio
    async def test_analyze_prompt_effectiveness_handles_null_version(self) -> None:
        """Test that null template_version defaults to 1."""
        mock_rows = [
            {
                "prompt_key": "OLD_PROMPT",
                "template_version": None,
                "total_uses": 50,
                "uses_with_feedback": 5,
                "feedback_rate": 0.1,
                "positive_feedback": 4,
                "negative_feedback": 1,
                "success_rate": 0.98,
                "avg_latency_ms": 100.0,
                "avg_tokens": 120.0,
                "avg_cost_usd": Decimal("0.0008"),
                "priority_score": 15.0,
            }
        ]

        mock_prompt_audit_repo = AsyncMock()
        mock_prompt_audit_repo.analyze_prompt_effectiveness = AsyncMock(
            return_value=mock_rows
        )

        metrics = await analyze_prompt_effectiveness(
            prompt_audit_repo=mock_prompt_audit_repo
        )

        assert len(metrics) == 1
        assert metrics[0].version == 1

    @pytest.mark.asyncio
    async def test_analyze_prompt_effectiveness_with_custom_params(self) -> None:
        """Test that custom parameters are passed to repository."""
        mock_prompt_audit_repo = AsyncMock()
        mock_prompt_audit_repo.analyze_prompt_effectiveness = AsyncMock(return_value=[])

        await analyze_prompt_effectiveness(
            prompt_audit_repo=mock_prompt_audit_repo,
            min_uses=50,
            lookback_days=90,
            min_feedback=20,
        )

        mock_prompt_audit_repo.analyze_prompt_effectiveness.assert_called_once_with(
            min_uses=50,
            lookback_days=90,
            min_feedback=20,
        )

    @pytest.mark.asyncio
    async def test_analyze_prompt_effectiveness_handles_bad_data(self) -> None:
        """Test that malformed data is handled gracefully."""
        mock_rows = [
            {
                "prompt_key": "BAD_PROMPT",
                "template_version": 1,
                "total_uses": "not a number",  # Invalid type
                "uses_with_feedback": 10,
                "feedback_rate": "also invalid",
                "positive_feedback": 5,
                "negative_feedback": 5,
                "success_rate": None,
                "avg_latency_ms": "bad",
                "avg_tokens": None,
                "avg_cost_usd": None,
                "priority_score": "invalid",
            }
        ]

        mock_prompt_audit_repo = AsyncMock()
        mock_prompt_audit_repo.analyze_prompt_effectiveness = AsyncMock(
            return_value=mock_rows
        )

        metrics = await analyze_prompt_effectiveness(
            prompt_audit_repo=mock_prompt_audit_repo
        )

        assert len(metrics) == 1
        assert metrics[0].total_uses == "not a number"  # Kept as-is
        assert metrics[0].feedback_rate == 0.0  # Converted to default
        assert metrics[0].success_rate == 0.0  # Converted to default
        assert metrics[0].avg_latency_ms is None  # Optional returns None


class TestGetFeedbackSamples:
    """Tests for get_feedback_samples function."""

    @pytest.mark.asyncio
    async def test_get_feedback_samples_returns_list(self) -> None:
        """Test that get_feedback_samples returns a list of feedback strings."""
        mock_samples = ["Great response!", "Very helpful", "Thanks!"]
        mock_prompt_audit_repo = AsyncMock()
        mock_prompt_audit_repo.get_feedback_samples = AsyncMock(
            return_value=mock_samples
        )

        result = await get_feedback_samples(
            prompt_key="BASIC_RESPONSE",
            prompt_audit_repo=mock_prompt_audit_repo,
            limit=10,
        )

        assert result == mock_samples
        mock_prompt_audit_repo.get_feedback_samples.assert_called_once_with(
            prompt_key="BASIC_RESPONSE",
            limit=10,
            sentiment_filter=None,
        )

    @pytest.mark.asyncio
    async def test_get_feedback_samples_with_sentiment_filter(self) -> None:
        """Test that sentiment filter is passed correctly."""
        mock_prompt_audit_repo = AsyncMock()
        mock_prompt_audit_repo.get_feedback_samples = AsyncMock(return_value=[])

        await get_feedback_samples(
            prompt_key="BASIC_RESPONSE",
            prompt_audit_repo=mock_prompt_audit_repo,
            limit=5,
            sentiment_filter="negative",
        )

        mock_prompt_audit_repo.get_feedback_samples.assert_called_once_with(
            prompt_key="BASIC_RESPONSE",
            limit=5,
            sentiment_filter="negative",
        )

    @pytest.mark.asyncio
    async def test_get_feedback_samples_empty_result(self) -> None:
        """Test empty feedback list."""
        mock_prompt_audit_repo = AsyncMock()
        mock_prompt_audit_repo.get_feedback_samples = AsyncMock(return_value=[])

        result = await get_feedback_samples(
            prompt_key="UNKNOWN_PROMPT",
            prompt_audit_repo=mock_prompt_audit_repo,
        )

        assert result == []


class TestGetFeedbackWithContext:
    """Tests for get_feedback_with_context function."""

    @pytest.mark.asyncio
    async def test_get_feedback_with_context_returns_list(self) -> None:
        """Test that get_feedback_with_context returns list of dicts."""
        mock_data = [
            {
                "user_message": "Hello",
                "rendered_prompt": "You are helpful...",
                "response_content": "Hi there!",
                "feedback_content": "Good",
                "sentiment": "positive",
            }
        ]
        mock_prompt_audit_repo = AsyncMock()
        mock_prompt_audit_repo.get_feedback_with_context = AsyncMock(
            return_value=mock_data
        )

        result = await get_feedback_with_context(
            prompt_key="BASIC_RESPONSE",
            prompt_audit_repo=mock_prompt_audit_repo,
            limit=10,
            include_rendered_prompt=True,
            max_prompt_chars=5000,
        )

        assert result == mock_data
        mock_prompt_audit_repo.get_feedback_with_context.assert_called_once_with(
            prompt_key="BASIC_RESPONSE",
            limit=10,
            include_rendered_prompt=True,
            max_prompt_chars=5000,
        )

    @pytest.mark.asyncio
    async def test_get_feedback_with_context_excludes_prompt(self) -> None:
        """Test that rendered prompt can be excluded."""
        mock_prompt_audit_repo = AsyncMock()
        mock_prompt_audit_repo.get_feedback_with_context = AsyncMock(return_value=[])

        await get_feedback_with_context(
            prompt_key="BASIC_RESPONSE",
            prompt_audit_repo=mock_prompt_audit_repo,
            include_rendered_prompt=False,
        )

        mock_prompt_audit_repo.get_feedback_with_context.assert_called_once_with(
            prompt_key="BASIC_RESPONSE",
            limit=10,
            include_rendered_prompt=False,
            max_prompt_chars=5000,
        )

    @pytest.mark.asyncio
    async def test_get_feedback_with_context_empty_result(self) -> None:
        """Test empty context list."""
        mock_prompt_audit_repo = AsyncMock()
        mock_prompt_audit_repo.get_feedback_with_context = AsyncMock(return_value=[])

        result = await get_feedback_with_context(
            prompt_key="NEW_PROMPT",
            prompt_audit_repo=mock_prompt_audit_repo,
        )

        assert result == []
