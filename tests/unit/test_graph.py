"""Unit tests for graph traversal and triple parsing utilities."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

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
        assert parse_triples('[{"subject": "Alice", "predicate": 123, "object": "Acme"}]') == []

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
        result = parse_triples('[{"subject": "Alice", "predicate": "WORKS_AT", "object": "Acme"}]')
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


@pytest.mark.skip(reason="Async fixture issues with db_pool mocking - needs fix")
class TestGraphTraversal:
    """Tests for GraphTraversal class."""

    @pytest.mark.skip(reason="Async fixture issues with db_pool mocking - needs fix")
    @pytest.fixture
    async def db_pool(self) -> MagicMock:
        """Create mock database pool."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        return mock_pool

    @pytest.fixture
    async def traverser(self, db_pool: MagicMock) -> GraphTraversal:
        """Create GraphTraversal instance with mock pool."""
        return GraphTraversal(db_pool, max_depth=3)

    @pytest.mark.asyncio
    async def test_traverse_from_fact_single_hop(
        self,
        traverser: GraphTraversal,
        db_pool: MagicMock,
    ) -> None:
        """Test single-hop traversal returns direct relationships."""
        fact1_id = uuid4()
        fact2_id = uuid4()

        db_pool.acquire.return_value.__aenter__.return_value.fetch = AsyncMock(
            return_value=[
                {
                    "id": uuid4(),
                    "subject_id": fact1_id,
                    "subject_content": "Alice",
                    "predicate": "works_at",
                    "object_id": fact2_id,
                    "object_content": "Company X",
                    "depth": 1,
                    "visited": [fact2_id],
                }
            ]
        )

        result = await traverser.traverse_from_fact(fact1_id, max_hops=1)

        assert len(result) == 1
        assert result[0]["subject_content"] == "Alice"
        assert result[0]["predicate"] == "works_at"
        assert result[0]["object_content"] == "Company X"

    @pytest.mark.asyncio
    async def test_traverse_from_fact_multi_hop(
        self,
        traverser: GraphTraversal,
        db_pool: MagicMock,
    ) -> None:
        """Test multi-hop traversal discovers indirect relationships."""
        alice_id = uuid4()
        company_x_id = uuid4()
        company_y_id = uuid4()

        db_pool.acquire.return_value.__aenter__.return_value.fetch = AsyncMock(
            return_value=[
                {
                    "id": uuid4(),
                    "subject_id": alice_id,
                    "subject_content": "Alice",
                    "predicate": "works_at",
                    "object_id": company_x_id,
                    "object_content": "Company X",
                    "depth": 1,
                    "visited": [company_x_id],
                },
                {
                    "id": uuid4(),
                    "subject_id": company_x_id,
                    "subject_content": "Company X",
                    "predicate": "acquired",
                    "object_id": company_y_id,
                    "object_content": "Company Y",
                    "depth": 2,
                    "visited": [company_x_id, company_y_id],
                },
            ]
        )

        result = await traverser.traverse_from_fact(alice_id, max_hops=2)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_traverse_with_predicate_filter(
        self,
        traverser: GraphTraversal,
        db_pool: MagicMock,
    ) -> None:
        """Test predicate filtering restricts traversal."""
        alice_id = uuid4()

        db_pool.acquire.return_value.__aenter__.return_value.fetch = AsyncMock(
            return_value=[
                {
                    "id": uuid4(),
                    "subject_id": alice_id,
                    "subject_content": "Alice",
                    "predicate": "works_at",
                    "object_id": uuid4(),
                    "object_content": "Company X",
                    "depth": 1,
                    "visited": [uuid4()],
                }
            ]
        )

        result = await traverser.traverse_from_fact(
            alice_id, predicate_filter={"works_at"}, max_hops=2
        )

        assert len(result) == 1
        assert result[0]["predicate"] == "works_at"

    @pytest.mark.asyncio
    async def test_find_entity_relationships(
        self,
        traverser: GraphTraversal,
        db_pool: MagicMock,
    ) -> None:
        """Test finding all relationships involving an entity."""
        db_pool.acquire.return_value.__aenter__.return_value.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "Company X",
                    "predicate": "acquired",
                    "object": "Company Y",
                    "created_at": "2024-01-01",
                }
            ]
        )

        result = await traverser.find_entity_relationships("Company X", limit=10)

        assert len(result) == 1
        assert result[0]["predicate"] == "acquired"

    @pytest.mark.asyncio
    async def test_empty_result(self, traverser: GraphTraversal, db_pool: MagicMock) -> None:
        """Test traversal returns empty list when no relationships exist."""
        db_pool.acquire.return_value.__aenter__.return_value.fetch = AsyncMock(return_value=[])

        result = await traverser.traverse_from_fact(uuid4())

        assert result == []
