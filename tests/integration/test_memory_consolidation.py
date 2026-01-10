"""Integration tests for semantic memory extraction and graph traversal."""

import asyncio
from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice.memory.graph import GraphTraversal
from lattice.utils.memory_parsing import parse_semantic_memories


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestMultiHopReasoningIntegration:
    """Integration tests for multi-hop graph traversal using text-based storage.

    Note: Uses mocked database connections to test traversal logic.
    Real database integration is covered by test_entity_and_triple_storage.py.
    """

    @pytest.mark.asyncio
    async def test_three_hop_reasoning_chain(self) -> None:
        """Test three-hop reasoning through graph traversal.

        Verifies that BFS can discover entities across multiple hops:
        Alice -> works_at -> Acme Corp -> acquired_by -> TechCorp -> in -> Technology Industry
        """
        # Create mock pool and connection
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
                {
                    "subject": "Acme Corp",
                    "predicate": "acquired_by",
                    "object": "TechCorp",
                    "created_at": datetime.now(UTC),
                },
                {
                    "subject": "TechCorp",
                    "predicate": "in",
                    "object": "Technology Industry",
                    "created_at": datetime.now(UTC),
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        # Create traverser with mock pool
        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.find_semantic_memories(subject="Alice")

        assert len(result) == 3
        assert result[0]["subject"] == "Alice"
        assert result[0]["predicate"] == "works_at"
        assert result[1]["subject"] == "Acme Corp"
        assert result[1]["predicate"] == "acquired_by"
        assert result[2]["subject"] == "TechCorp"
        assert result[2]["predicate"] == "in"

    @pytest.mark.asyncio
    async def test_filtered_multi_hop_traversal(self) -> None:
        """Test multi-hop traversal with predicate filter."""
        # Create mock pool and connection
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "Alice",
                    "predicate": "works_at",
                    "object": "Acme Corp",
                    "created_at": datetime.now(UTC),
                }
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        # Create traverser with mock pool
        traverser = GraphTraversal(mock_pool, max_depth=3)
        result = await traverser.traverse_from_entity(
            "Alice", predicate_filter={"works_at", "likes"}, max_hops=2
        )

        assert len(result) == 1
        assert result[0]["predicate"] == "works_at"


class TestTextFormatParsingIntegration:
    """Integration tests for text format memory parsing."""

    def test_parse_memories_arrow_format(self) -> None:
        """Test parsing memories with -> separator."""
        result = parse_semantic_memories(
            "alice -> works_at -> Acme Corp\nbob -> likes -> pizza"
        )
        assert len(result) == 2
        assert result[0]["subject"] == "alice"
        assert result[0]["predicate"] == "works_at"
        assert result[0]["object"] == "Acme Corp"
        assert result[1]["subject"] == "bob"
        assert result[1]["predicate"] == "likes"

    def test_parse_memories_unicode_arrow(self) -> None:
        """Test parsing memories with unicode arrow separator."""
        result = parse_semantic_memories("alice → works_at → Acme Corp")
        assert len(result) == 1
        assert result[0]["subject"] == "alice"
        assert result[0]["predicate"] == "works_at"

    def test_parse_memories_dash_arrow(self) -> None:
        """Test parsing memories with dash-arrow separator."""
        result = parse_semantic_memories("alice --> works_at --> Acme Corp")
        assert len(result) == 1
        assert result[0]["subject"] == "alice"
        assert result[0]["predicate"] == "works_at"
        assert result[0]["object"] == "Acme Corp"

    def test_parse_memories_with_header(self) -> None:
        """Test parsing memories with Memories: header."""
        result = parse_semantic_memories("Memories:\nalice -> works_at -> Acme Corp")
        assert len(result) == 1

    def test_parse_memories_mixed_format(self) -> None:
        """Test parsing mixed JSON and text format."""
        json_result = parse_semantic_memories(
            '[{"subject": "Alice", "predicate": "works_at", "object": "Acme"}]'
        )
        text_result = parse_semantic_memories("Alice -> works_at -> Acme")

        assert len(json_result) == 1
        assert len(text_result) == 1
        assert json_result[0]["predicate"] == text_result[0]["predicate"] == "works_at"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
