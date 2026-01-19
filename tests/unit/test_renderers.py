"""Tests for memory renderers."""

from datetime import datetime, timezone

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

    def test_naive_datetime_treated_as_utc_with_timezone(self) -> None:
        """Test naive datetime is treated as UTC when converting to user timezone."""
        dt = datetime(2026, 1, 12, 10, 30)
        result = render_activity(
            "User", "did activity", "coding", dt, "America/New_York"
        )
        assert result == "[2026-01-12 05:30:00] coding"


class TestGetRendererEdgeCases:
    """Additional tests for get_renderer function edge cases."""

    def test_get_renderer_none_type(self) -> None:
        """Test that None type returns render_basic as default."""
        renderer = get_renderer(None)  # type: ignore
        assert renderer == render_basic

    def test_get_renderer_case_sensitive(self) -> None:
        """Test that context type lookup is case-sensitive."""
        renderer = get_renderer("ACTIVITY_CONTEXT")
        assert renderer == render_basic  # Not found, defaults to basic

    def test_get_renderer_whitespace(self) -> None:
        """Test that whitespace-only string returns default."""
        renderer = get_renderer("   ")
        assert renderer == render_basic

    def test_get_renderer_with_underscores(self) -> None:
        """Test renderer lookup with underscores."""
        renderer = get_renderer("activity_context")
        assert renderer == render_activity

    def test_get_renderer_custom_types_return_basic(self) -> None:
        """Test that custom/unknown types return basic renderer."""
        custom_types = ["custom", "unknown", "random", "test_context"]
        for ctx_type in custom_types:
            renderer = get_renderer(ctx_type)
            assert renderer == render_basic


class TestRendererExamples:
    """Tests for renderer examples from docstrings."""

    def test_render_activity_docstring_example(self) -> None:
        """Test the example from render_activity docstring."""
        from datetime import datetime, timezone

        dt = datetime(2026, 1, 12, 10, 30, tzinfo=timezone.utc)
        result = render_activity("User", "did activity", "coding", dt)
        assert result == "[2026-01-12 10:30:00] coding"

    def test_render_goal_docstring_example(self) -> None:
        """Test the example from render_goal docstring."""
        result = render_goal("User", "has goal", "run marathon", None)
        assert result == "run marathon"

    def test_render_basic_docstring_example(self) -> None:
        """Test the example from render_basic docstring."""
        result = render_basic("User", "lives in city", "Portland", None)
        assert result == "User lives in city Portland"

    def test_get_renderer_docstring_example(self) -> None:
        """Test the example from get_renderer docstring."""
        from datetime import datetime, timezone

        renderer = get_renderer("activity_context")
        dt = datetime(2026, 1, 12, tzinfo=timezone.utc)
        result = renderer("User", "did activity", "coding", dt, None)
        assert result == "[2026-01-12 00:00:00] coding"
