"""Unit tests for response generator module.

Tests for relevance detection, context building, and response generation.
"""

from lattice.core.response_generator import (
    _has_entity_goal_overlap,
    _has_planning_intent,
    PLANNING_KEYWORDS,
)


class TestPlanningIntentDetection:
    """Tests for _has_planning_intent function."""

    def test_goal_keywords(self) -> None:
        """Messages explicitly about goals should trigger."""
        assert _has_planning_intent("what are my goals")
        assert _has_planning_intent("what are my objectives")
        assert _has_planning_intent("what's my goal for this week")

    def test_deadline_keywords(self) -> None:
        """Messages about deadlines should trigger."""
        assert _has_planning_intent("what's my deadline")
        assert _has_planning_intent("when is my due date")
        assert _has_planning_intent("milestone coming up")

    def test_possessive_keywords(self) -> None:
        """Messages with 'my' possessive should trigger."""
        assert _has_planning_intent("what's my plan for today")
        assert _has_planning_intent("what are my priorities")
        assert _has_planning_intent("my tasks for the week")

    def test_non_planning_messages(self) -> None:
        """Casual messages should not trigger."""
        assert not _has_planning_intent("hello how are you")
        assert not _has_planning_intent("what should i watch tonight")
        assert not _has_planning_intent("focus on the bug report")
        assert not _has_planning_intent("what's the next step in the process")
        assert not _has_planning_intent("remind me to call mom")

    def test_case_insensitive(self) -> None:
        """Keyword detection should be case insensitive."""
        assert _has_planning_intent("WHAT ARE MY GOALS")
        assert _has_planning_intent("What are my objectives")
        assert _has_planning_intent("My Plan for today")


class TestEntityGoalOverlap:
    """Tests for _has_entity_goal_overlap function."""

    def test_exact_match(self) -> None:
        """Exact entity-goal matches should trigger."""
        entities = ["run a marathon"]
        goals = ["run a marathon"]
        assert _has_entity_goal_overlap(entities, goals)

    def test_partial_match(self) -> None:
        """Partial entity-goal matches should trigger."""
        entities = ["marathon"]
        goals = ["run a marathon"]
        assert _has_entity_goal_overlap(entities, goals)

    def test_goal_in_entity(self) -> None:
        """Goal name found within entity should trigger."""
        entities = ["run a marathon training plan"]
        goals = ["run a marathon"]
        assert _has_entity_goal_overlap(entities, goals)

    def test_no_match(self) -> None:
        """Unrelated entities and goals should not trigger."""
        entities = ["pizza", "movie"]
        goals = ["run a marathon", "finish project"]
        assert not _has_entity_goal_overlap(entities, goals)

    def test_empty_entities(self) -> None:
        """Empty entities list should return False."""
        assert not _has_entity_goal_overlap([], ["run a marathon"])

    def test_empty_goals(self) -> None:
        """Empty goals list should return False."""
        assert not _has_entity_goal_overlap(["marathon"], [])

    def test_mixed_match(self) -> None:
        """Mixed entities with some matches should trigger."""
        entities = ["pizza", "run a marathon"]
        goals = ["run a marathon", "finish project"]
        assert _has_entity_goal_overlap(entities, goals)


class TestPlanningKeywords:
    """Tests for PLANNING_KEYWORDS constant."""

    def test_keywords_are_strings(self) -> None:
        """All keywords should be strings."""
        for kw in PLANNING_KEYWORDS:
            assert isinstance(kw, str)

    def test_no_empty_keywords(self) -> None:
        """Keywords should not be empty strings."""
        for kw in PLANNING_KEYWORDS:
            assert len(kw) > 0

    def test_keywords_are_lowercase(self) -> None:
        """Keywords should be lowercase for consistent matching."""
        for kw in PLANNING_KEYWORDS:
            assert kw == kw.lower()
