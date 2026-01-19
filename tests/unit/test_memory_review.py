"""Tests for dreaming memory review module.

Tests semantic memory conflict detection and resolution workflow:
- Conflict detection and parsing
- Memory review cycle
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from lattice.dreaming.memory_review import (
    ConflictResolution,
    MIN_MEMORIES_PER_SUBJECT,
    _parse_memory_review_response,
    analyze_subject_memories,
    get_memories_by_subject,
    get_subjects_for_review,
    run_memory_review,
)


class TestParseMemoryReviewResponse:
    """Tests for _parse_memory_review_response function."""

    def test_parse_valid_json_response(self) -> None:
        """Test parsing valid JSON response."""
        response = json.dumps(
            [
                {
                    "type": "contradiction",
                    "subject": "User preference",
                    "canonical_memory": {
                        "subject": "User preference",
                        "predicate": "likes",
                        "object": "Python",
                        "created_at": "2024-01-15T10:00:00Z",
                    },
                    "superseded_memories": [
                        {
                            "subject": "User preference",
                            "predicate": "likes",
                            "object": "JavaScript",
                            "created_at": "2024-01-01T10:00:00Z",
                        }
                    ],
                    "reason": "User switched preference",
                }
            ]
        )

        conflicts, error = _parse_memory_review_response(response, "test_subject")

        assert error is None
        assert len(conflicts) == 1
        assert conflicts[0]["type"] == "contradiction"
        assert conflicts[0]["canonical_memory"]["object"] == "Python"

    def test_parse_invalid_json(self) -> None:
        """Test handling of invalid JSON."""
        response = "not valid json {"

        conflicts, error = _parse_memory_review_response(response, "test_subject")

        assert conflicts == []
        assert error is not None
        assert error.startswith("JSON parse error")

    def test_parse_empty_response(self) -> None:
        """Test handling of empty response."""
        response = ""

        conflicts, error = _parse_memory_review_response(response, "test_subject")

        assert conflicts == []
        assert error is not None

    def test_parse_non_array_response(self) -> None:
        """Test handling of non-array JSON response."""
        response = json.dumps({"error": "No conflicts found"})

        conflicts, error = _parse_memory_review_response(response, "test_subject")

        assert conflicts == []
        assert error == "Expected array of conflict resolutions"

    def test_parse_array_with_malformed_items(self) -> None:
        """Test parsing array with malformed items still succeeds."""
        response = json.dumps(
            [
                {
                    "type": "contradiction",
                    "canonical_memory": {},
                    "superseded_memories": [],
                    "reason": "Test",
                }
            ]
        )

        conflicts, error = _parse_memory_review_response(response, "test_subject")

        assert error is None
        assert len(conflicts) == 1

    def test_parse_multiple_conflicts(self) -> None:
        """Test parsing multiple conflicts."""
        response = json.dumps(
            [
                {
                    "type": "contradiction",
                    "canonical_memory": {
                        "subject": "Subject1",
                        "predicate": "rel",
                        "object": "Obj1",
                        "created_at": "2024-01-15T10:00:00Z",
                    },
                    "superseded_memories": [],
                    "reason": "Conflict 1",
                },
                {
                    "type": "redundancy",
                    "canonical_memory": {
                        "subject": "Subject2",
                        "predicate": "rel",
                        "object": "Obj2",
                        "created_at": "2024-01-15T10:00:00Z",
                    },
                    "superseded_memories": [],
                    "reason": "Conflict 2",
                },
            ]
        )

        conflicts, error = _parse_memory_review_response(response, "test_subject")

        assert error is None
        assert len(conflicts) == 2
        assert conflicts[0]["type"] == "contradiction"
        assert conflicts[1]["type"] == "redundancy"


class TestConflictResolution:
    """Tests for ConflictResolution dataclass."""

    def test_conflict_resolution_creation(self) -> None:
        """Test creating a ConflictResolution instance."""
        resolution = ConflictResolution(
            conflict_type="contradiction",
            superseded_memories=[
                {
                    "subject": "User",
                    "predicate": "likes",
                    "object": "Old thing",
                    "created_at": "2024-01-01T10:00:00Z",
                }
            ],
            canonical_memory={
                "subject": "User",
                "predicate": "likes",
                "object": "New thing",
                "created_at": "2024-01-15T10:00:00Z",
            },
            reason="User updated preference",
        )

        assert resolution.conflict_type == "contradiction"
        assert len(resolution.superseded_memories) == 1
        assert resolution.canonical_memory["object"] == "New thing"

    def test_conflict_resolution_empty_superseded(self) -> None:
        """Test ConflictResolution with no superseded memories."""
        resolution = ConflictResolution(
            conflict_type="stale",
            superseded_memories=[],
            canonical_memory={
                "subject": "Fact",
                "predicate": "is",
                "object": "True",
                "created_at": "2024-01-15T10:00:00Z",
            },
            reason="Updated information available",
        )

        assert resolution.conflict_type == "stale"
        assert len(resolution.superseded_memories) == 0


class TestGetSubjectsForReview:
    """Tests for get_subjects_for_review function."""

    @pytest.mark.asyncio
    async def test_get_subjects_for_review_returns_list(self) -> None:
        """Test that get_subjects_for_review returns list of subjects."""
        mock_subjects = ["User preferences", "Project goals", "Meeting notes"]
        mock_semantic_repo = AsyncMock()
        mock_semantic_repo.get_subjects_for_review = AsyncMock(
            return_value=mock_subjects
        )

        result = await get_subjects_for_review(
            semantic_repo=mock_semantic_repo,
            min_memories=10,
        )

        assert result == mock_subjects
        mock_semantic_repo.get_subjects_for_review.assert_called_once_with(10)

    @pytest.mark.asyncio
    async def test_get_subjects_for_review_empty_list(self) -> None:
        """Test empty subjects list."""
        mock_semantic_repo = AsyncMock()
        mock_semantic_repo.get_subjects_for_review = AsyncMock(return_value=[])

        result = await get_subjects_for_review(semantic_repo=mock_semantic_repo)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_subjects_for_review_custom_min(self) -> None:
        """Test custom minimum memories parameter."""
        mock_semantic_repo = AsyncMock()
        mock_semantic_repo.get_subjects_for_review = AsyncMock(return_value=[])

        await get_subjects_for_review(
            semantic_repo=mock_semantic_repo,
            min_memories=20,
        )

        mock_semantic_repo.get_subjects_for_review.assert_called_once_with(20)


class TestGetMemoriesBySubject:
    """Tests for get_memories_by_subject function."""

    @pytest.mark.asyncio
    async def test_get_memories_by_subject_returns_list(self) -> None:
        """Test that get_memories_by_subject returns list of memories."""
        mock_memories = [
            {
                "id": str(uuid4()),
                "subject": "User",
                "predicate": "likes",
                "object": "Python",
                "created_at": "2024-01-15T10:00:00Z",
            }
        ]
        mock_semantic_repo = AsyncMock()
        mock_semantic_repo.get_memories_by_subject = AsyncMock(
            return_value=mock_memories
        )

        result = await get_memories_by_subject(
            semantic_repo=mock_semantic_repo,
            subject="User",
        )

        assert result == mock_memories
        mock_semantic_repo.get_memories_by_subject.assert_called_once_with("User")

    @pytest.mark.asyncio
    async def test_get_memories_by_subject_empty(self) -> None:
        """Test empty memories list."""
        mock_semantic_repo = AsyncMock()
        mock_semantic_repo.get_memories_by_subject = AsyncMock(return_value=[])

        result = await get_memories_by_subject(
            semantic_repo=mock_semantic_repo,
            subject="Unknown",
        )

        assert result == []


class TestAnalyzeSubjectMemories:
    """Tests for analyze_subject_memories function."""

    @pytest.mark.asyncio
    async def test_analyze_subject_memories_requires_llm_client(self) -> None:
        """Test that missing llm_client raises ValueError."""
        mock_semantic_repo = AsyncMock()
        mock_prompt_repo = AsyncMock()

        with pytest.raises(ValueError, match="llm_client is required"):
            await analyze_subject_memories(
                semantic_repo=mock_semantic_repo,
                llm_client=None,
                subject="Test",
                prompt_repo=mock_prompt_repo,
            )

    @pytest.mark.asyncio
    async def test_analyze_subject_memories_insufficient_memories(self) -> None:
        """Test with insufficient memories returns empty list."""
        mock_semantic_repo = AsyncMock()
        mock_semantic_repo.get_memories_by_subject = AsyncMock(
            return_value=[
                {
                    "id": str(uuid4()),
                    "subject": "Test",
                    "predicate": "rel",
                    "object": "Obj",
                    "created_at": "2024-01-15T10:00:00Z",
                }
            ]
        )
        mock_llm_client = AsyncMock()
        mock_prompt_repo = AsyncMock()

        result = await analyze_subject_memories(
            semantic_repo=mock_semantic_repo,
            llm_client=mock_llm_client,
            subject="Test",
            prompt_repo=mock_prompt_repo,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_subject_memories_missing_prompt(self) -> None:
        """Test missing MEMORY_REVIEW prompt returns empty list."""
        mock_memories = [
            {
                "id": str(uuid4()),
                "subject": "Test",
                "predicate": "rel",
                "object": "Obj1",
                "created_at": "2024-01-15T10:00:00Z",
            },
            {
                "id": str(uuid4()),
                "subject": "Test",
                "predicate": "rel",
                "object": "Obj2",
                "created_at": "2024-01-14T10:00:00Z",
            },
        ]
        mock_semantic_repo = AsyncMock()
        mock_semantic_repo.get_memories_by_subject = AsyncMock(
            return_value=mock_memories
        )
        mock_llm_client = AsyncMock()
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.dreaming.memory_review.get_prompt",
            return_value=None,
        ):
            result = await analyze_subject_memories(
                semantic_repo=mock_semantic_repo,
                llm_client=mock_llm_client,
                subject="Test",
                prompt_repo=mock_prompt_repo,
            )

        assert result == []


class TestRunMemoryReview:
    """Tests for run_memory_review function."""

    @pytest.mark.asyncio
    async def test_run_memory_review_requires_llm_client(self) -> None:
        """Test that missing llm_client raises ValueError."""
        mock_semantic_repo = AsyncMock()
        mock_prompt_repo = AsyncMock()

        with pytest.raises(ValueError, match="llm_client is required"):
            await run_memory_review(
                semantic_repo=mock_semantic_repo,
                llm_client=None,
                prompt_repo=mock_prompt_repo,
            )

    @pytest.mark.asyncio
    async def test_run_memory_review_no_subjects(self) -> None:
        """Test with no subjects returns None."""
        mock_semantic_repo = AsyncMock()
        mock_semantic_repo.get_subjects_for_review = AsyncMock(return_value=[])
        mock_llm_client = MagicMock()
        mock_prompt_repo = AsyncMock()

        result = await run_memory_review(
            semantic_repo=mock_semantic_repo,
            llm_client=mock_llm_client,
            prompt_repo=mock_prompt_repo,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_run_memory_review_no_conflicts(self) -> None:
        """Test with no conflicts returns None."""
        mock_semantic_repo = AsyncMock()
        mock_semantic_repo.get_subjects_for_review = AsyncMock(
            return_value=["User preference"]
        )
        mock_semantic_repo.get_memories_by_subject = AsyncMock(return_value=[])

        mock_llm_client = MagicMock()

        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.dreaming.memory_review.analyze_subject_memories",
            return_value=[],
        ):
            result = await run_memory_review(
                semantic_repo=mock_semantic_repo,
                llm_client=mock_llm_client,
                prompt_repo=mock_prompt_repo,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_run_memory_review_multiple_subjects_no_conflicts(self) -> None:
        """Test memory review with multiple subjects but no conflicts."""
        mock_semantic_repo = AsyncMock()
        mock_semantic_repo.get_subjects_for_review = AsyncMock(
            return_value=["Subject1", "Subject2", "Subject3"]
        )
        mock_semantic_repo.get_memories_by_subject = AsyncMock(return_value=[])

        mock_llm_client = MagicMock()
        mock_prompt_repo = AsyncMock()

        with patch(
            "lattice.dreaming.memory_review.analyze_subject_memories",
            return_value=[],
        ):
            result = await run_memory_review(
                semantic_repo=mock_semantic_repo,
                llm_client=mock_llm_client,
                prompt_repo=mock_prompt_repo,
            )

        assert result is None


class TestMemoryReviewConstants:
    """Tests for memory review module constants."""

    def test_min_memories_per_subject_constant(self) -> None:
        """Test that MIN_MEMORIES_PER_SUBJECT constant exists."""
        assert MIN_MEMORIES_PER_SUBJECT == 10
