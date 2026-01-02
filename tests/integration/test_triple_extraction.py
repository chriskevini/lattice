"""Integration tests for semantic triple extraction and graph traversal."""

import asyncio
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from lattice.memory.graph import GraphTraversal
from lattice.utils.triple_parsing import parse_triples


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.mark.skip(reason="Async fixture issues with db_pool mocking - needs fix")
class TestMultiHopReasoningIntegration:
    """Integration tests for multi-hop graph traversal."""

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
    async def test_three_hop_reasoning_chain(
        self,
        traverser: GraphTraversal,
        db_pool: MagicMock,
    ) -> None:
        """Test three-hop reasoning through graph traversal."""
        alice_id = uuid4()
        acme_id = uuid4()
        techcorp_id = uuid4()
        industry_id = uuid4()

        db_pool.acquire.return_value.__aenter__.return_value.fetch = AsyncMock(
            return_value=[
                {
                    "id": uuid4(),
                    "subject_id": alice_id,
                    "subject_content": "Alice",
                    "predicate": "works_at",
                    "object_id": acme_id,
                    "object_content": "Acme Corp",
                    "depth": 1,
                    "visited": [acme_id],
                },
                {
                    "id": uuid4(),
                    "subject_id": acme_id,
                    "subject_content": "Acme Corp",
                    "predicate": "acquired_by",
                    "object_id": techcorp_id,
                    "object_content": "TechCorp",
                    "depth": 2,
                    "visited": [acme_id, techcorp_id],
                },
                {
                    "id": uuid4(),
                    "subject_id": techcorp_id,
                    "subject_content": "TechCorp",
                    "predicate": "in",
                    "object_id": industry_id,
                    "object_content": "Technology Industry",
                    "depth": 3,
                    "visited": [acme_id, techcorp_id, industry_id],
                },
            ]
        )

        result = await traverser.traverse_from_fact(alice_id, max_hops=3)

        assert len(result) == 3
        assert result[0]["subject_content"] == "Alice"
        assert result[0]["predicate"] == "works_at"
        assert result[1]["subject_content"] == "Acme Corp"
        assert result[1]["predicate"] == "acquired_by"
        assert result[2]["subject_content"] == "TechCorp"
        assert result[2]["predicate"] == "in"

    @pytest.mark.asyncio
    async def test_filtered_multi_hop_traversal(
        self,
        traverser: GraphTraversal,
        db_pool: MagicMock,
    ) -> None:
        """Test multi-hop traversal with predicate filter."""
        alice_id = uuid4()

        db_pool.acquire.return_value.__aenter__.return_value.fetch = AsyncMock(
            return_value=[
                {
                    "id": uuid4(),
                    "subject_id": alice_id,
                    "subject_content": "Alice",
                    "predicate": "works_at",
                    "object_id": uuid4(),
                    "object_content": "Acme Corp",
                    "depth": 1,
                    "visited": [uuid4()],
                }
            ]
        )

        result = await traverser.traverse_from_fact(
            alice_id, predicate_filter={"works_at", "likes"}, max_hops=2
        )

        assert len(result) == 1
        assert result[0]["predicate"] == "works_at"


class TestTextFormatParsingIntegration:
    """Integration tests for text format triple parsing."""

    def test_parse_triples_arrow_format(self) -> None:
        """Test parsing triples with -> separator."""
        result = parse_triples("alice -> works_at -> Acme Corp\nbob -> likes -> pizza")
        assert len(result) == 2
        assert result[0]["subject"] == "alice"
        assert result[0]["predicate"] == "works_at"
        assert result[0]["object"] == "Acme Corp"
        assert result[1]["subject"] == "bob"
        assert result[1]["predicate"] == "likes"

    def test_parse_triples_unicode_arrow(self) -> None:
        """Test parsing triples with unicode arrow separator."""
        result = parse_triples("alice → works_at → Acme Corp")
        assert len(result) == 1
        assert result[0]["subject"] == "alice"
        assert result[0]["predicate"] == "works_at"

    def test_parse_triples_with_header(self) -> None:
        """Test parsing triples with Triples: header."""
        result = parse_triples("Triples:\nalice -> works_at -> Acme Corp")
        assert len(result) == 1

    def test_parse_triples_mixed_format(self) -> None:
        """Test parsing mixed JSON and text format."""
        json_result = parse_triples(
            '[{"subject": "Alice", "predicate": "works_at", "object": "Acme"}]'
        )
        text_result = parse_triples("Alice -> works_at -> Acme")

        assert len(json_result) == 1
        assert len(text_result) == 1
        assert json_result[0]["predicate"] == text_result[0]["predicate"] == "works_at"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
