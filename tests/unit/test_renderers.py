"""Tests for memory renderers."""

from datetime import datetime, timezone

import pytest

from lattice.memory.renderers import (
    get_renderer,
    render_activity,
    render_basic,
    render_goal,
)


class TestRenderActivity:
    """Tests for render_activity function."""

    def test_with_timestamp_utc(self) -> None:
        """Test activity rendering with UTC timestamp."""
        dt = datetime(2026, 1, 12, 10, 30, tzinfo=timezone.utc)
        result = render_activity("User", "did activity", "coding", dt)
        assert result == "[2026-01-12 10:30:00] coding"

    def test_with_timestamp_and_timezone(self) -> None:
        """Test activity rendering with user timezone conversion."""
        dt = datetime(2026, 1, 12, 10, 30, tzinfo=timezone.utc)
        result = render_activity(
            "User", "did activity", "coding", dt, "America/New_York"
        )
        assert result == "[2026-01-12 05:30:00] coding"

    def test_without_timestamp(self) -> None:
        """Test activity rendering without timestamp."""
        result = render_activity("User", "did activity", "coding", None)
        assert result == "coding"

    def test_with_timezone_but_no_timestamp(self) -> None:
        """Test that timezone is ignored when no timestamp."""
        result = render_activity(
            "User", "did activity", "coding", None, "America/New_York"
        )
        assert result == "coding"


class TestRenderGoal:
    """Tests for render_goal function."""

    def test_returns_object_only(self) -> None:
        """Test goal rendering returns object only."""
        result = render_goal("User", "has goal", "run marathon", None)
        assert result == "run marathon"

    def test_ignores_timestamp(self) -> None:
        """Test goal rendering ignores timestamp."""
        dt = datetime(2026, 1, 12, 10, 30, tzinfo=timezone.utc)
        result = render_goal("User", "has goal", "run marathon", dt)
        assert result == "run marathon"

    def test_ignores_timezone(self) -> None:
        """Test goal rendering ignores timezone."""
        dt = datetime(2026, 1, 12, 10, 30, tzinfo=timezone.utc)
        result = render_goal("User", "has goal", "run marathon", dt, "America/New_York")
        assert result == "run marathon"


class TestRenderBasic:
    """Tests for render_basic function."""

    def test_returns_full_triple(self) -> None:
        """Test basic rendering returns full triple."""
        result = render_basic("User", "lives in city", "Portland", None)
        assert result == "User lives in city Portland"

    def test_ignores_timestamp(self) -> None:
        """Test basic rendering ignores timestamp."""
        dt = datetime(2026, 1, 12, 10, 30, tzinfo=timezone.utc)
        result = render_basic("User", "lives in city", "Portland", dt)
        assert result == "User lives in city Portland"

    def test_ignores_timezone(self) -> None:
        """Test basic rendering ignores timezone."""
        dt = datetime(2026, 1, 12, 10, 30, tzinfo=timezone.utc)
        result = render_basic(
            "User", "lives in city", "Portland", dt, "America/New_York"
        )
        assert result == "User lives in city Portland"


class TestGetRenderer:
    """Tests for get_renderer registry lookup."""

    def test_activity_context(self) -> None:
        """Test getting activity renderer."""
        renderer = get_renderer("activity_context")
        assert renderer is render_activity

    def test_goal_context(self) -> None:
        """Test getting goal renderer."""
        renderer = get_renderer("goal_context")
        assert renderer is render_goal

    def test_semantic_context(self) -> None:
        """Test getting semantic/basic renderer."""
        renderer = get_renderer("semantic_context")
        assert renderer is render_basic

    def test_unknown_context_defaults_to_basic(self) -> None:
        """Test unknown context type defaults to basic renderer."""
        renderer = get_renderer("unknown_context")
        assert renderer is render_basic

    def test_empty_string_defaults_to_basic(self) -> None:
        """Test empty string defaults to basic renderer."""
        renderer = get_renderer("")
        assert renderer is render_basic


class TestTimezoneEdgeCases:
    """Tests for timezone edge cases."""

    def test_activity_pacific_time(self) -> None:
        """Test activity with Pacific timezone."""
        dt = datetime(2026, 1, 12, 20, 0, tzinfo=timezone.utc)
        result = render_activity(
            "User", "did activity", "dinner", dt, "America/Los_Angeles"
        )
        assert result == "[2026-01-12 12:00:00] dinner"

    def test_activity_tokyo_time(self) -> None:
        """Test activity with Tokyo timezone."""
        dt = datetime(2026, 1, 12, 10, 0, tzinfo=timezone.utc)
        result = render_activity("User", "did activity", "meeting", dt, "Asia/Tokyo")
        assert result == "[2026-01-12 19:00:00] meeting"

    def test_activity_naive_datetime(self) -> None:
        """Test activity with naive datetime (no timezone info)."""
        dt = datetime(2026, 1, 12, 10, 30)
        result = render_activity("User", "did activity", "coding", dt)
        assert result == "[2026-01-12 10:30:00] coding"
