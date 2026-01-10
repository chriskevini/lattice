"""Unit tests for graph traversal and semantic memory parsing utilities."""

from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from zoneinfo import ZoneInfo
from lattice.utils.date_resolution import get_now

import pytest

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
    """Tests for GraphTraversal class with text-based memories."""

    @pytest.mark.asyncio
    async def test_find_semantic_memories_by_predicate(self) -> None:
        """Test finding memories by predicate only."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "did activity",
                    "object": "coding",
                    "created_at": get_now(),
                }
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(predicate="did activity")

        assert len(result) == 1
        assert result[0]["predicate"] == "did activity"
        assert result[0]["object"] == "coding"

    @pytest.mark.asyncio
    async def test_find_semantic_memories_by_subject_and_predicate(self) -> None:
        """Test finding memories by subject and predicate."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "User",
                    "predicate": "did activity",
                    "object": "running",
                    "created_at": get_now(),
                }
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(
            subject="User", predicate="did activity"
        )

        assert len(result) == 1
        assert result[0]["subject"] == "User"
        assert result[0]["predicate"] == "did activity"

    @pytest.mark.asyncio
    async def test_find_semantic_memories_no_filters(self) -> None:
        """Test finding memories with no filters returns all."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "lives_in",
                    "object": "Vancouver",
                    "created_at": get_now(),
                }
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories()

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_find_semantic_memories_empty_result(self) -> None:
        """Test finding memories returns empty list when no matches."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(subject="nonexistent")

        assert result == []

    @pytest.mark.asyncio
    async def test_find_semantic_memories_with_date_range_v2(self) -> None:
        """Test finding memories by predicate within date range."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        start_date = datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC"))
        end_date = datetime(2026, 1, 8, tzinfo=ZoneInfo("UTC"))
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "did activity",
                    "object": "coding",
                    "created_at": datetime(2026, 1, 5, tzinfo=ZoneInfo("UTC")),
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(
            predicate="did activity",
            start_date=start_date,
            end_date=end_date,
            limit=50,
        )

        assert len(result) == 1
        assert result[0]["object"] == "coding"

    @pytest.mark.asyncio
    async def test_find_semantic_memories_with_start_date_only_v2(self) -> None:
        """Test finding memories with start date only."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        start_date = datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC"))
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "did activity",
                    "object": "coding",
                    "created_at": datetime(2026, 1, 10, tzinfo=ZoneInfo("UTC")),
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(
            predicate="did activity",
            start_date=start_date,
            limit=50,
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_find_semantic_memories_with_end_date_only_v2(self) -> None:
        """Test finding memories with end date only."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        end_date = datetime(2026, 1, 8, tzinfo=ZoneInfo("UTC"))
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "did activity",
                    "object": "coding",
                    "created_at": datetime(2026, 1, 5, tzinfo=ZoneInfo("UTC")),
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(
            predicate="did activity",
            end_date=end_date,
            limit=50,
        )

        assert len(result) == 1
        assert result[0]["object"] == "coding"

    @pytest.mark.asyncio
    async def test_find_semantic_memories_with_start_date_only(self) -> None:
        """Test finding memories with start date only."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        start_date = datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC"))
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "did activity",
                    "object": "coding",
                    "created_at": datetime(2026, 1, 10, tzinfo=ZoneInfo("UTC")),
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(
            predicate="did activity",
            start_date=start_date,
            limit=50,
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_find_semantic_memories_with_end_date_only(self) -> None:
        """Test finding memories with end date only."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        end_date = datetime(2026, 1, 8, tzinfo=ZoneInfo("UTC"))
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "did activity",
                    "object": "coding",
                    "created_at": datetime(2026, 1, 5, tzinfo=ZoneInfo("UTC")),
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(
            predicate="did activity",
            end_date=end_date,
            limit=50,
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_find_semantic_memories_respects_limit(self) -> None:
        """Test that limit parameter is respected."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        memories = [
            {
                "subject": "user",
                "predicate": f"predicate_{i}",
                "object": f"object_{i}",
                "created_at": get_now(),
            }
            for i in range(5)
        ]
        mock_conn.fetch = AsyncMock(return_value=memories[:2])
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(limit=2)

        assert len(result) == 2


class TestSanitizationFunctions:
    """Tests for ILIKE pattern sanitization functions."""

    def test_escape_wildcard_percent(self) -> None:
        """Test that % wildcard is escaped."""
        from lattice.memory.graph import _escape_like_pattern

        result = _escape_like_pattern("test%entity")
        assert "\\%test\\%entity" in result or "test\\%entity" in result

    def test_escape_wildcard_underscore(self) -> None:
        """Test that _ wildcard is escaped."""
        from lattice.memory.graph import _escape_like_pattern

        result = _escape_like_pattern("test_entity")
        assert "\\_" in result or "_" in result

    def test_escape_backslash(self) -> None:
        """Test that backslash is escaped."""
        from lattice.memory.graph import _escape_like_pattern

        result = _escape_like_pattern("test\\entity")
        assert "\\\\" in result or "\\" in result

    def test_sanitize_long_entity_name(self) -> None:
        """Test that long entity names are truncated."""
        from lattice.memory.graph import _sanitize_entity_name, MAX_ENTITY_NAME_LENGTH

        long_name = "x" * (MAX_ENTITY_NAME_LENGTH + 100)
        result = _sanitize_entity_name(long_name)
        assert len(result) == MAX_ENTITY_NAME_LENGTH

    def test_sanitize_preserves_content(self) -> None:
        """Test that sanitization preserves normal entity names."""
        from lattice.memory.graph import _sanitize_entity_name

        normal_name = "Alice works at Acme Corp"
        result = _sanitize_entity_name(normal_name)
        assert result == normal_name


class TestBFSTraversal:
    """Tests for BFS multi-hop traversal functionality."""

    @pytest.mark.asyncio
    async def test_bfs_cycle_detection(self) -> None:
        """Test that BFS prevents infinite loops in cycles.

        Creates A -> B -> C -> A cycle and verifies:
        - No infinite loop (algorithm terminates)
        - Only unique memories returned (no duplicates)
        """
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        cycle_memories = [
            {
                "subject": "A",
                "predicate": "connects_to",
                "object": "B",
                "created_at": get_now(),
            },
            {
                "subject": "B",
                "predicate": "connects_to",
                "object": "C",
                "created_at": get_now(),
            },
            {
                "subject": "C",
                "predicate": "connects_to",
                "object": "A",
                "created_at": get_now(),
            },
        ]

        mock_conn.fetch = AsyncMock(return_value=cycle_memories)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.traverse_from_entity("A", max_hops=5)

        subject_objects = {(t["subject"], t["object"]) for t in result}
        assert len(subject_objects) == 3, (
            f"Expected 3 unique memories, got {len(subject_objects)}: {subject_objects}"
        )

        seen_subjects = {t["subject"] for t in result}
        seen_objects = {t["object"] for t in result}
        assert seen_subjects == {"A", "B", "C"}, (
            f"Expected subjects A, B, C, got {seen_subjects}"
        )
        assert seen_objects == {"A", "B", "C"}, (
            f"Expected objects A, B, C, got {seen_objects}"
        )

    @pytest.mark.asyncio
    async def test_bfs_max_hops_limit(self) -> None:
        """Test that max_hops parameter limits traversal depth."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=1)
        result = await traverser.traverse_from_entity("test", max_hops=2)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_find_semantic_memories_with_end_date_only(self) -> None:
        """Test finding memories with end date only."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        end_date = datetime(2026, 1, 8, tzinfo=ZoneInfo("UTC"))
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "did activity",
                    "object": "coding",
                    "created_at": datetime(2026, 1, 5, tzinfo=ZoneInfo("UTC")),
                },
            ]
        )

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.traverse_from_entity("Alice", max_hops=1)

        assert len(result) == 1
        assert "depth" in result[0]
        assert result[0]["depth"] == 1

    @pytest.mark.asyncio
    async def test_find_semantic_memories_with_date_range_v3(self) -> None:
        """Test finding memories by predicate within date range."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        start_date = datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC"))
        end_date = datetime(2026, 1, 8, tzinfo=ZoneInfo("UTC"))
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "did activity",
                    "object": "coding",
                    "created_at": datetime(2026, 1, 5, tzinfo=ZoneInfo("UTC")),
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(
            predicate="did activity",
            start_date=start_date,
            end_date=end_date,
            limit=50,
        )

        assert len(result) == 1
        assert result[0]["object"] == "coding"

    @pytest.mark.asyncio
    async def test_find_semantic_memories_with_start_date_only_v3(self) -> None:
        """Test finding memories with start date only."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        start_date = datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC"))
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "did activity",
                    "object": "coding",
                    "created_at": datetime(2026, 1, 10, tzinfo=ZoneInfo("UTC")),
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(
            predicate="did activity",
            start_date=start_date,
            limit=50,
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_find_semantic_memories_with_end_date_only_v3(self) -> None:
        """Test finding memories with end date only."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        end_date = datetime(2026, 1, 8, tzinfo=ZoneInfo("UTC"))
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "did activity",
                    "object": "coding",
                    "created_at": datetime(2026, 1, 5, tzinfo=ZoneInfo("UTC")),
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(
            predicate="did activity",
            end_date=end_date,
            limit=50,
        )

        assert len(result) == 1
        assert result[0]["object"] == "coding"

    @pytest.mark.asyncio
    async def test_find_semantic_memories_with_dates(self) -> None:
        """Test finding memories with date filtering."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        start_date = datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC"))
        end_date = datetime(2026, 1, 8, tzinfo=ZoneInfo("UTC"))

        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "did activity",
                    "object": "coding",
                    "created_at": datetime(2026, 1, 5, tzinfo=ZoneInfo("UTC")),
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(
            predicate="did activity",
            start_date=start_date,
            end_date=end_date,
        )

        assert len(result) == 1
        assert result[0]["object"] == "coding"
