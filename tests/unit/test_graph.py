"""Unit tests for graph traversal and triple parsing utilities."""

from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, UTC

import pytest

from lattice.memory.graph import GraphTraversal
from lattice.utils.triple_parsing import parse_triples


class TestParseTriples:
    """Tests for parse_triples function."""

    def test_valid_json_array(self) -> None:
        """Test parsing valid JSON array of triples."""
        result = parse_triples(
            '[{"subject": "Alice", "predicate": "works_at", "object": "Company X"}]'
        )
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        assert result[0]["predicate"] == "works_at"
        assert result[0]["object"] == "Company X"

    def test_multiple_triples(self) -> None:
        """Test parsing multiple triples."""
        result = parse_triples(
            '[{"subject": "Alice", "predicate": "works_at", "object": "Acme"},'
            '{"subject": "Bob", "predicate": "likes", "object": "Pizza"}]'
        )
        assert len(result) == 2
        assert result[0]["subject"] == "Alice"
        assert result[1]["subject"] == "Bob"

    def test_empty_array(self) -> None:
        """Test parsing empty array."""
        assert parse_triples("[]") == []

    def test_malformed_json(self) -> None:
        """Test parsing malformed JSON returns empty list."""
        assert parse_triples("not json") == []
        assert parse_triples("{invalid}") == []

    def test_missing_fields(self) -> None:
        """Test parsing JSON with missing fields."""
        assert parse_triples('[{"subject": "Alice"}]') == []
        assert parse_triples('[{"predicate": "works_at"}]') == []
        assert parse_triples('[{"object": "Acme"}]') == []

    def test_invalid_field_types(self) -> None:
        """Test parsing JSON with invalid field types."""
        assert (
            parse_triples('[{"subject": "Alice", "predicate": 123, "object": "Acme"}]')
            == []
        )

    def test_with_markdown_code_block(self) -> None:
        """Test parsing JSON with markdown code block."""
        result = parse_triples(
            '```json\n[{"subject": "Bob", "predicate": "likes", "object": "Pizza"}]\n```'
        )
        assert len(result) == 1
        assert result[0]["subject"] == "Bob"

    def test_with_json_code_block(self) -> None:
        """Test parsing JSON with json language marker."""
        result = parse_triples(
            '```\n[{"subject": "Bob", "predicate": "likes", "object": "Pizza"}]\n```'
        )
        assert len(result) == 1

    def test_predicate_normalization(self) -> None:
        """Test that predicates are normalized to lowercase."""
        result = parse_triples(
            '[{"subject": "Alice", "predicate": "WORKS_AT", "object": "Acme"}]'
        )
        assert len(result) == 1
        assert result[0]["predicate"] == "works_at"

    def test_whitespace_stripping(self) -> None:
        """Test that whitespace is stripped from values."""
        result = parse_triples(
            '[  {"subject": "  Alice  ", "predicate": "likes", "object": "  Pizza  "}  ]'
        )
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        assert result[0]["object"] == "Pizza"

    def test_case_insensitive_content(self) -> None:
        """Test that subject/object content is not normalized."""
        result = parse_triples(
            '[{"subject": "Alice", "predicate": "likes", "object": "TECHNOLOGY"}]'
        )
        assert len(result) == 1
        assert result[0]["object"] == "TECHNOLOGY"


class TestGraphTraversal:
    """Tests for GraphTraversal class with text-based triples."""

    @pytest.mark.asyncio
    async def test_find_entity_relationships(self) -> None:
        """Test finding all relationships involving an entity."""
        # Create mock pool and connection
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "works_at",
                    "object": "Company X",
                    "created_at": datetime.now(UTC),
                }
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        # Create traverser with mock pool
        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_entity_relationships("Company X", limit=10)

        assert len(result) == 1
        assert result[0]["subject"] == "user"
        assert result[0]["predicate"] == "works_at"
        assert result[0]["object"] == "Company X"

    @pytest.mark.asyncio
    async def test_find_entity_relationships_case_insensitive(self) -> None:
        """Test that entity search is case-insensitive."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "lives_in",
                    "object": "Richmond, BC",
                    "created_at": datetime.now(UTC),
                }
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_entity_relationships("richmond", limit=10)

        # Verify ILIKE pattern was used (case-insensitive)
        assert len(result) == 1
        assert "Richmond" in result[0]["object"]

    @pytest.mark.asyncio
    async def test_find_entity_partial_match(self) -> None:
        """Test that entity search uses partial matching."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "user",
                    "predicate": "has_goal",
                    "object": "run a marathon",
                    "created_at": datetime.now(UTC),
                }
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_entity_relationships("marathon", limit=10)

        assert len(result) == 1
        assert "marathon" in result[0]["object"]

    @pytest.mark.asyncio
    async def test_empty_result(self) -> None:
        """Test traversal returns empty list when no relationships exist."""
        # Create mock pool and connection
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        # Create traverser with mock pool
        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_entity_relationships("nonexistent")

        assert result == []

    @pytest.mark.asyncio
    async def test_respects_limit(self) -> None:
        """Test that limit parameter is respected."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        # Create multiple triples
        triples = [
            {
                "subject": "user",
                "predicate": f"predicate_{i}",
                "object": f"object_{i}",
                "created_at": datetime.now(UTC),
            }
            for i in range(5)
        ]
        mock_conn.fetch = AsyncMock(return_value=triples[:3])  # Simulates LIMIT 3
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_entity_relationships("user", limit=3)

        assert len(result) == 3


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
        - Only unique triples returned (no duplicates)
        """
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        cycle_triples = [
            {
                "subject": "A",
                "predicate": "connects_to",
                "object": "B",
                "created_at": datetime.now(UTC),
            },
            {
                "subject": "B",
                "predicate": "connects_to",
                "object": "C",
                "created_at": datetime.now(UTC),
            },
            {
                "subject": "C",
                "predicate": "connects_to",
                "object": "A",
                "created_at": datetime.now(UTC),
            },
        ]

        mock_conn.fetch = AsyncMock(return_value=cycle_triples)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.traverse_from_entity("A", max_hops=5)

        subject_objects = {(t["subject"], t["object"]) for t in result}
        assert len(subject_objects) == 3, (
            f"Expected 3 unique triples, got {len(subject_objects)}: {subject_objects}"
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
    async def test_bfs_returns_depth_info(self) -> None:
        """Test that BFS returns triples with depth metadata."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "Alice",
                    "predicate": "works_at",
                    "object": "Acme Corp",
                    "created_at": datetime.now(UTC),
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
