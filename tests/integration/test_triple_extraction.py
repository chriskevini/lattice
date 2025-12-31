"""Integration tests for semantic triple extraction and graph traversal."""

import asyncio
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from lattice.memory.episodic import _ensure_fact, consolidate_message
from lattice.memory.graph import GraphTraversal
from lattice.utils.triple_parsing import parse_triples


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestTripleExtractionIntegration:
    """Integration tests for the triple extraction pipeline."""

    @pytest.mark.asyncio
    async def test_full_triple_extraction(self) -> None:
        """Test complete triple extraction from message to database."""
        message_id = uuid4()
        content = "Alice works at Acme Corp and lives in Seattle"
        context = ["Previous message about work"]

        with (
            patch("lattice.memory.episodic.db_pool") as mock_pool,
            patch("lattice.memory.procedural.get_prompt") as mock_get_prompt,
            patch("lattice.memory.episodic.embedding_model") as mock_model,
            patch("lattice.utils.llm.get_llm_client") as mock_get_client,
        ):
            mock_model.encode_single = MagicMock(return_value=[0.1] * 384)

            mock_get_prompt.return_value = MagicMock(
                template="Extract triples from: {CONTEXT}",
                temperature=0.1,
            )

            mock_client = MagicMock()
            mock_client.complete = AsyncMock(
                return_value="["
                '{"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"}, '
                '{"subject": "Alice", "predicate": "located_in", "object": "Seattle"}'
                "]"
            )
            mock_get_client.return_value = mock_client

            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.transaction = MagicMock(
                return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock())
            )
            mock_conn.fetchval = AsyncMock(return_value=None)
            mock_conn.fetchrow = MagicMock(return_value={"id": uuid4()})
            mock_conn.execute = AsyncMock()

            await consolidate_message(message_id, content, context)

            mock_get_prompt.assert_called_once_with("TRIPLE_EXTRACTION")
            mock_client.complete.assert_called_once()
            mock_conn.fetchval.assert_called()

    @pytest.mark.asyncio
    async def test_extraction_no_results(self) -> None:
        """Test extraction when LLM returns no triples."""
        message_id = uuid4()
        content = "Hello, how are you?"
        context = []

        with (
            patch("lattice.memory.episodic.db_pool") as mock_pool,
            patch("lattice.memory.procedural.get_prompt") as mock_get_prompt,
            patch("lattice.utils.llm.get_llm_client") as mock_get_client,
        ):
            mock_get_prompt.return_value = MagicMock(
                template="Extract triples from: {CONTEXT}",
                temperature=0.1,
            )

            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value="[]")
            mock_get_client.return_value = mock_client

            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            await consolidate_message(message_id, content, context)

            mock_client.complete.assert_called_once()
            mock_conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_extraction_llm_unavailable(self) -> None:
        """Test that extraction uses placeholder when llm_client unavailable."""
        message_id = uuid4()
        content = "Test message"
        context = []

        with (
            patch("lattice.memory.episodic.db_pool") as mock_pool,
            patch("lattice.memory.procedural.get_prompt") as mock_get_prompt,
        ):
            mock_get_prompt.return_value = MagicMock(
                template="Extract triples from: {CONTEXT}",
                temperature=0.1,
            )

            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            await consolidate_message(message_id, content, context)

            mock_conn.execute.assert_not_called()


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


class TestRaceConditionIntegration:
    """Integration tests for concurrent access to shared resources."""

    @pytest.mark.asyncio
    async def test_concurrent_consolidation_requests(self) -> None:
        """Test that concurrent consolidations don't cause conflicts."""
        message_id = uuid4()
        content = "Alice works at Company"
        context = []

        with (
            patch("lattice.memory.episodic.db_pool") as mock_pool,
            patch("lattice.memory.procedural.get_prompt") as mock_get_prompt,
            patch("lattice.memory.episodic.embedding_model") as mock_model,
            patch("lattice.utils.llm.get_llm_client") as mock_get_client,
        ):
            mock_model.encode_single = MagicMock(return_value=[0.1] * 384)

            mock_get_prompt.return_value = MagicMock(
                template="Extract triples from: {CONTEXT}",
                temperature=0.1,
            )

            mock_client = MagicMock()
            mock_client.complete = AsyncMock(
                return_value='[{"subject": "Alice", "predicate": "works_at", "object": "Company"}]'
            )
            mock_get_client.return_value = mock_client

            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.transaction = MagicMock(
                return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock())
            )
            mock_conn.fetchval = AsyncMock(return_value=None)
            mock_conn.fetchrow = MagicMock(return_value={"id": uuid4()})
            mock_conn.execute = AsyncMock()

            tasks = [
                consolidate_message(message_id, content, context),
                consolidate_message(message_id, content, context),
                consolidate_message(message_id, content, context),
            ]

            await asyncio.gather(*tasks)

            mock_conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_idempotent_fact_retrieval(self) -> None:
        """Test that duplicate facts are not inserted."""
        alice_uuid = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=alice_uuid)

        result = await _ensure_fact("Alice", uuid4(), mock_conn, MagicMock())

        assert result == alice_uuid
        mock_conn.fetchval.assert_called()


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
