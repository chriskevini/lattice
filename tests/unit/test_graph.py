"""Unit tests for graph traversal and semantic memory parsing utilities."""

from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from zoneinfo import ZoneInfo
from lattice.utils.date_resolution import get_now

import pytest

from lattice.memory.repositories import SemanticMemoryRepository
from lattice.memory.graph import GraphTraversal
from lattice.utils.memory_parsing import parse_semantic_memories


class TestParseSemanticMemories:
    """Tests for parse_semantic_memories function."""

    def test_valid_json_array(self) -> None:
        """Test parsing valid JSON array of memories."""
        result = parse_semantic_memories(
            '[{"subject": "Alice", "predicate": "works_at", "object": "Company X"}]'
        )
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        assert result[0]["predicate"] == "works_at"
        assert result[0]["object"] == "Company X"

    def test_multiple_memories(self) -> None:
        """Test parsing multiple memories."""
        result = parse_semantic_memories(
            '[{"subject": "Alice", "predicate": "works_at", "object": "Acme"},'
            '{"subject": "Bob", "predicate": "likes", "object": "Pizza"}]'
        )
        assert len(result) == 2
        assert result[0]["subject"] == "Alice"
        assert result[1]["subject"] == "Bob"

    def test_empty_array(self) -> None:
        """Test parsing empty array."""
        assert parse_semantic_memories("[]") == []

    def test_malformed_json(self) -> None:
        """Test parsing malformed JSON returns empty list."""
        assert parse_semantic_memories("not json") == []
        assert parse_semantic_memories("{invalid}") == []

    def test_missing_fields(self) -> None:
        """Test parsing JSON with missing fields."""
        assert parse_semantic_memories('[{"subject": "Alice"}]') == []
        assert parse_semantic_memories('[{"predicate": "works_at"}]') == []
        assert parse_semantic_memories('[{"object": "Acme"}]') == []

    def test_invalid_field_types(self) -> None:
        """Test parsing JSON with invalid field types."""
        assert (
            parse_semantic_memories(
                '[{"subject": "Alice", "predicate": 123, "object": "Acme"}]'
            )
            == []
        )

    def test_with_markdown_code_block(self) -> None:
        """Test parsing JSON with markdown code block."""
        result = parse_semantic_memories(
            '```json\n[{"subject": "Bob", "predicate": "likes", "object": "Pizza"}]\n```'
        )
        assert len(result) == 1
        assert result[0]["subject"] == "Bob"

    def test_with_json_code_block(self) -> None:
        """Test parsing JSON with json language marker."""
        result = parse_semantic_memories(
            '```\n[{"subject": "Bob", "predicate": "likes", "object": "Pizza"}]\n```'
        )
        assert len(result) == 1

    def test_predicate_normalization(self) -> None:
        """Test that predicates are normalized to lowercase."""
        result = parse_semantic_memories(
            '[{"subject": "Alice", "predicate": "WORKS_AT", "object": "Acme"}]'
        )
        assert len(result) == 1
        assert result[0]["predicate"] == "works_at"

    def test_whitespace_stripping(self) -> None:
        """Test that whitespace is stripped from values."""
        result = parse_semantic_memories(
            '[  {"subject": "  Alice  ", "predicate": "likes", "object": "  Pizza  "}  ]'
        )
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        assert result[0]["object"] == "Pizza"

    def test_case_insensitive_content(self) -> None:
        """Test that subject/object content is not normalized."""
        result = parse_semantic_memories(
            '[{"subject": "Alice", "predicate": "likes", "object": "TECHNOLOGY"}]'
        )
        assert len(result) == 1
        assert result[0]["object"] == "TECHNOLOGY"


class TestGraphTraversal:
    """Tests for GraphTraversal class - BFS traversal delegation."""

    @pytest.mark.asyncio
    async def test_traverse_from_entity_delegation(self) -> None:
        """Test that traverse_from_entity delegates to repository."""
        mock_repo = MagicMock(spec=SemanticMemoryRepository)
        mock_repo.traverse_from_entity = AsyncMock(
            return_value=[
                {
                    "subject": "A",
                    "predicate": "relates_to",
                    "object": "B",
                    "created_at": get_now(),
                }
            ]
        )

        traverser = GraphTraversal(mock_repo, max_depth=3)
        result = await traverser.traverse_from_entity("A")

        assert len(result) == 1
        mock_repo.traverse_from_entity.assert_called_once_with(
            entity_name="A",
            predicate_filter=None,
            max_hops=3,
        )

    @pytest.mark.asyncio
    async def test_traverse_with_predicate_filter(self) -> None:
        """Test traversal with predicate filter."""
        mock_repo = MagicMock(spec=SemanticMemoryRepository)
        mock_repo.traverse_from_entity = AsyncMock(return_value=[])

        traverser = GraphTraversal(mock_repo, max_depth=2)
        await traverser.traverse_from_entity("User", predicate_filter={"did activity"})

        mock_repo.traverse_from_entity.assert_called_once_with(
            entity_name="User",
            predicate_filter={"did activity"},
            max_hops=2,
        )

    @pytest.mark.asyncio
    async def test_traverse_custom_max_hops(self) -> None:
        """Test that custom max_hops overrides default."""
        mock_repo = MagicMock(spec=SemanticMemoryRepository)
        mock_repo.traverse_from_entity = AsyncMock(return_value=[])

        traverser = GraphTraversal(mock_repo, max_depth=3)
        await traverser.traverse_from_entity("test", max_hops=5)

        mock_repo.traverse_from_entity.assert_called_once_with(
            entity_name="test",
            predicate_filter=None,
            max_hops=5,
        )

    @pytest.mark.asyncio
    async def test_bfs_sanitization(self) -> None:
        """Test that BFS traversal handles special characters in entity names."""
        mock_repo = MagicMock(spec=SemanticMemoryRepository)
        mock_repo.traverse_from_entity = AsyncMock(return_value=[])

        traverser = GraphTraversal(mock_repo, max_depth=1)
        await traverser.traverse_from_entity("test%_entity")

        mock_repo.traverse_from_entity.assert_called_once_with(
            entity_name="test%_entity",
            predicate_filter=None,
            max_hops=1,
        )

    @pytest.mark.asyncio
    async def test_bfs_cycle_detection(self) -> None:
        """Test that cycle detection is handled by repository.

        GraphTraversal delegates to the repository, which is responsible for
        cycle detection logic.
        """
        mock_repo = MagicMock(spec=SemanticMemoryRepository)
        cycle_memories = [
            {
                "subject": "A",
                "predicate": "connects_to",
                "object": "B",
                "created_at": get_now(),
                "depth": 1,
            },
            {
                "subject": "B",
                "predicate": "connects_to",
                "object": "C",
                "created_at": get_now(),
                "depth": 2,
            },
            {
                "subject": "C",
                "predicate": "connects_to",
                "object": "A",
                "created_at": get_now(),
                "depth": 3,
            },
        ]
        mock_repo.traverse_from_entity = AsyncMock(return_value=cycle_memories)

        traverser = GraphTraversal(mock_repo, max_depth=3)
        result = await traverser.traverse_from_entity("A", max_hops=5)

        assert len(result) == 3
        mock_repo.traverse_from_entity.assert_called_once_with(
            entity_name="A",
            predicate_filter=None,
            max_hops=5,
        )

    @pytest.mark.asyncio
    async def test_bfs_max_hops_limit(self) -> None:
        """Test that max_hops parameter is passed to repository."""
        mock_repo = MagicMock(spec=SemanticMemoryRepository)
        mock_repo.traverse_from_entity = AsyncMock(return_value=[])

        traverser = GraphTraversal(mock_repo, max_depth=1)
        result = await traverser.traverse_from_entity("test", max_hops=2)

        assert len(result) == 0
        mock_repo.traverse_from_entity.assert_called_once_with(
            entity_name="test",
            predicate_filter=None,
            max_hops=2,
        )
