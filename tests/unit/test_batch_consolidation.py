"""Unit tests for batch consolidation module."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from lattice.memory.batch_consolidation import (
    BATCH_SIZE,
    check_and_run_batch,
    run_batch_consolidation,
)


def create_mock_pool_with_conn() -> tuple[MagicMock, AsyncMock]:
    """Create a properly mocked database pool with connection.

    Returns:
        Tuple of (mock_pool, mock_conn) for use in tests.
    """
    mock_conn = AsyncMock()

    mock_acquire_cm = MagicMock()
    mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

    return mock_pool, mock_conn


class TestBatchSize:
    """Tests for BATCH_SIZE constant."""

    def test_batch_size_is_18(self) -> None:
        """Test that BATCH_SIZE is 18 as specified in design."""
        assert BATCH_SIZE == 18


class TestCheckAndRunBatch:
    """Tests for check_and_run_batch function."""

    @pytest.mark.asyncio
    async def test_batch_not_ready_below_threshold(self) -> None:
        """Test that batch doesn't run when below 18 messages."""
        mock_pool, mock_conn = create_mock_pool_with_conn()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},  # last_batch_message_id
                {"cnt": 5},  # Only 5 new messages
            ]
        )

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch(
                "lattice.memory.batch_consolidation.run_batch_consolidation"
            ) as mock_run:
                mock_run.return_value = None
                await check_and_run_batch()
                mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_ready_at_threshold(self) -> None:
        """Test that batch runs when exactly at 18 messages."""
        mock_pool, mock_conn = create_mock_pool_with_conn()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},  # last_batch_message_id
                {"cnt": 18},  # Exactly at threshold
            ]
        )

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch(
                "lattice.memory.batch_consolidation.run_batch_consolidation"
            ) as mock_run:
                mock_run.return_value = None
                await check_and_run_batch()
                mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_ready_above_threshold(self) -> None:
        """Test that batch runs when above 18 messages."""
        mock_pool, mock_conn = create_mock_pool_with_conn()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},  # last_batch_message_id
                {"cnt": 25},  # Above threshold
            ]
        )

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch(
                "lattice.memory.batch_consolidation.run_batch_consolidation"
            ) as mock_run:
                mock_run.return_value = None
                await check_and_run_batch()
                mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_first_batch_initializes_from_zero(self) -> None:
        """Test that first batch initializes correctly when last_batch_id is '0'."""
        mock_pool, mock_conn = create_mock_pool_with_conn()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "0"},  # last_batch_message_id is "0"
                {"cnt": 20},  # Above threshold
            ]
        )

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch(
                "lattice.memory.batch_consolidation.run_batch_consolidation"
            ) as mock_run:
                mock_run.return_value = None
                await check_and_run_batch()
                # Verify query used int("0") = 0
                call_args = mock_conn.fetchrow.call_args_list[1]
                assert "discord_message_id > $1" in call_args[0][0]
                assert call_args[0][1] == 0
                mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_numeric_last_batch_id_handled(self) -> None:
        """Test that non-numeric last_batch_id defaults to 0."""
        mock_pool, mock_conn = create_mock_pool_with_conn()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "invalid"},  # Non-numeric
                {"cnt": 20},  # Above threshold
            ]
        )

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch(
                "lattice.memory.batch_consolidation.run_batch_consolidation"
            ) as mock_run:
                mock_run.return_value = None
                await check_and_run_batch()
                # Should use 0 when parsing fails
                call_args = mock_conn.fetchrow.call_args_list[1]
                assert call_args[0][1] == 0


class TestRunBatchConsolidation:
    """Tests for run_batch_consolidation function."""

    @pytest.mark.asyncio
    async def test_no_messages_returns_early(self) -> None:
        """Test that batch returns early when no new messages."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"metric_value": "100"})
        mock_conn.fetch = AsyncMock(return_value=[])  # No messages

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch("lattice.memory.batch_consolidation.get_prompt") as mock_prompt:
                await run_batch_consolidation()
                mock_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prompt_template_aborts(self) -> None:
        """Test that batch aborts if BATCH_MEMORY_EXTRACTION template missing."""
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "Test message",
                "is_bot": False,
                "timestamp": datetime.now(UTC),
            }
        ]

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"metric_value": "100"})
        mock_conn.fetch = AsyncMock(
            side_effect=[messages, []]
        )  # messages + empty memories

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch("lattice.memory.batch_consolidation.get_prompt") as mock_prompt:
                mock_prompt.return_value = None  # No template
                with patch(
                    "lattice.memory.batch_consolidation.get_llm_client"
                ) as mock_llm:
                    await run_batch_consolidation()
                    mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_extracts_and_stores_triples(self) -> None:
        """Test that batch extracts triples and stores them."""
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "I live in Vancouver",
                "is_bot": False,
                "timestamp": datetime.now(UTC),
            }
        ]
        memories = [
            {
                "subject": "user",
                "predicate": "lives_in",
                "object": "Richmond",
                "created_at": datetime.now(UTC),
            }
        ]

        mock_conn1 = AsyncMock()
        mock_conn1.fetchrow = AsyncMock(return_value={"metric_value": "100"})
        mock_conn1.fetch = AsyncMock(side_effect=[messages, memories])

        mock_conn2 = AsyncMock()
        mock_conn2.execute = AsyncMock()

        mock_acquire_cm1 = MagicMock()
        mock_acquire_cm1.__aenter__ = AsyncMock(return_value=mock_conn1)
        mock_acquire_cm1.__aexit__ = AsyncMock(return_value=None)

        mock_acquire_cm2 = MagicMock()
        mock_acquire_cm2.__aenter__ = AsyncMock(return_value=mock_conn2)
        mock_acquire_cm2.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(side_effect=[mock_acquire_cm1, mock_acquire_cm2])

        mock_prompt = MagicMock()
        mock_prompt.template = "Template with {MEMORY_CONTEXT} and {MESSAGE_HISTORY}"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="Formatted prompt")

        mock_result = MagicMock()
        mock_result.content = (
            '[{"subject": "user", "predicate": "lives_in", "object": "Vancouver"}]'
        )
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 100
        mock_result.completion_tokens = 50
        mock_result.total_tokens = 150
        mock_result.cost_usd = 0.002
        mock_result.latency_ms = 500

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch(
                "lattice.memory.batch_consolidation.get_prompt",
                return_value=mock_prompt,
            ):
                with patch(
                    "lattice.memory.batch_consolidation.get_llm_client",
                    return_value=mock_llm_client,
                ):
                    with patch(
                        "lattice.memory.batch_consolidation.store_semantic_triples"
                    ) as mock_store:
                        await run_batch_consolidation()
                        mock_store.assert_called_once()
                        call_args = mock_store.call_args
                        assert len(call_args[0][1]) == 1
                        assert call_args[0][1][0]["subject"] == "user"
                        assert call_args[0][1][0]["predicate"] == "lives_in"
                        assert call_args[0][1][0]["object"] == "Vancouver"
                        assert call_args.kwargs["source_batch_id"] == "101"

    @pytest.mark.asyncio
    async def test_updates_last_batch_message_id(self) -> None:
        """Test that last_batch_message_id is updated after successful batch."""
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 118,
                "content": "Test",
                "is_bot": False,
                "timestamp": datetime.now(UTC),
            }
        ]

        mock_conn1 = AsyncMock()
        mock_conn1.fetchrow = AsyncMock(return_value={"metric_value": "100"})
        mock_conn1.fetch = AsyncMock(side_effect=[messages, []])  # No memories

        mock_conn2 = AsyncMock()
        mock_conn2.execute = AsyncMock()

        mock_acquire_cm1 = MagicMock()
        mock_acquire_cm1.__aenter__ = AsyncMock(return_value=mock_conn1)
        mock_acquire_cm1.__aexit__ = AsyncMock(return_value=None)

        mock_acquire_cm2 = MagicMock()
        mock_acquire_cm2.__aenter__ = AsyncMock(return_value=mock_conn2)
        mock_acquire_cm2.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(side_effect=[mock_acquire_cm1, mock_acquire_cm2])

        mock_prompt = MagicMock()
        mock_prompt.template = "Template"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="Formatted")

        mock_result = MagicMock()
        mock_result.content = "[]"  # No triples
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 50
        mock_result.completion_tokens = 10
        mock_result.total_tokens = 60
        mock_result.cost_usd = 0.001
        mock_result.latency_ms = 200

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch(
                "lattice.memory.batch_consolidation.get_prompt",
                return_value=mock_prompt,
            ):
                with patch(
                    "lattice.memory.batch_consolidation.get_llm_client",
                    return_value=mock_llm_client,
                ):
                    await run_batch_consolidation()
                    # Verify last_batch_message_id was updated
                    update_call = mock_conn2.execute.call_args
                    assert "UPDATE system_health" in update_call[0][0]
                    assert update_call[0][1] == "118"  # Updated to last message id

    @pytest.mark.asyncio
    async def test_handles_json_code_blocks(self) -> None:
        """Test that JSON code blocks are stripped before parsing."""
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "I have a dog",
                "is_bot": False,
                "timestamp": datetime.now(UTC),
            }
        ]

        mock_conn1 = AsyncMock()
        mock_conn1.fetchrow = AsyncMock(return_value={"metric_value": "100"})
        mock_conn1.fetch = AsyncMock(side_effect=[messages, []])

        mock_conn2 = AsyncMock()
        mock_conn2.execute = AsyncMock()

        mock_acquire_cm1 = MagicMock()
        mock_acquire_cm1.__aenter__ = AsyncMock(return_value=mock_conn1)
        mock_acquire_cm1.__aexit__ = AsyncMock(return_value=None)

        mock_acquire_cm2 = MagicMock()
        mock_acquire_cm2.__aenter__ = AsyncMock(return_value=mock_conn2)
        mock_acquire_cm2.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(side_effect=[mock_acquire_cm1, mock_acquire_cm2])

        mock_prompt = MagicMock()
        mock_prompt.template = "Template"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="Formatted")

        mock_result = MagicMock()
        mock_result.content = """```json
[{"subject": "user", "predicate": "has_pet", "object": "dog"}]
```"""
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 100
        mock_result.completion_tokens = 50
        mock_result.total_tokens = 150
        mock_result.cost_usd = 0.002
        mock_result.latency_ms = 500

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch(
                "lattice.memory.batch_consolidation.get_prompt",
                return_value=mock_prompt,
            ):
                with patch(
                    "lattice.memory.batch_consolidation.get_llm_client",
                    return_value=mock_llm_client,
                ):
                    with patch(
                        "lattice.memory.batch_consolidation.store_semantic_triples"
                    ) as mock_store:
                        await run_batch_consolidation()
                        mock_store.assert_called_once()
                        triples = mock_store.call_args[0][1]
                        assert len(triples) == 1
                        assert triples[0]["predicate"] == "has_pet"

    @pytest.mark.asyncio
    async def test_invalid_json_stores_no_triples(self) -> None:
        """Test that invalid JSON response results in no triples stored."""
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "Test",
                "is_bot": False,
                "timestamp": datetime.now(UTC),
            }
        ]

        mock_conn1 = AsyncMock()
        mock_conn1.fetchrow = AsyncMock(return_value={"metric_value": "100"})
        mock_conn1.fetch = AsyncMock(side_effect=[messages, []])

        mock_conn2 = AsyncMock()
        mock_conn2.execute = AsyncMock()

        mock_acquire_cm1 = MagicMock()
        mock_acquire_cm1.__aenter__ = AsyncMock(return_value=mock_conn1)
        mock_acquire_cm1.__aexit__ = AsyncMock(return_value=None)

        mock_acquire_cm2 = MagicMock()
        mock_acquire_cm2.__aenter__ = AsyncMock(return_value=mock_conn2)
        mock_acquire_cm2.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(side_effect=[mock_acquire_cm1, mock_acquire_cm2])

        mock_prompt = MagicMock()
        mock_prompt.template = "Template"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="Formatted")

        mock_result = MagicMock()
        mock_result.content = "not valid json at all"
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 50
        mock_result.completion_tokens = 5
        mock_result.total_tokens = 55
        mock_result.cost_usd = 0.001
        mock_result.latency_ms = 150

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch(
                "lattice.memory.batch_consolidation.get_prompt",
                return_value=mock_prompt,
            ):
                with patch(
                    "lattice.memory.batch_consolidation.get_llm_client",
                    return_value=mock_llm_client,
                ):
                    with patch(
                        "lattice.memory.batch_consolidation.store_semantic_triples"
                    ) as mock_store:
                        await run_batch_consolidation()
                        mock_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_message_history_format(self) -> None:
        """Test that message history is formatted correctly with User/Bot prefix."""
        message_id = uuid4()
        now = datetime.now(UTC)
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "Hello",
                "is_bot": False,  # User
                "timestamp": now,
            },
            {
                "id": uuid4(),
                "discord_message_id": 102,
                "content": "Hi there!",
                "is_bot": True,  # Bot
                "timestamp": now,
            },
        ]

        mock_conn1 = AsyncMock()
        mock_conn1.fetchrow = AsyncMock(return_value={"metric_value": "100"})
        mock_conn1.fetch = AsyncMock(side_effect=[messages, []])

        mock_conn2 = AsyncMock()
        mock_conn2.execute = AsyncMock()

        mock_acquire_cm1 = MagicMock()
        mock_acquire_cm1.__aenter__ = AsyncMock(return_value=mock_conn1)
        mock_acquire_cm1.__aexit__ = AsyncMock(return_value=None)

        mock_acquire_cm2 = MagicMock()
        mock_acquire_cm2.__aenter__ = AsyncMock(return_value=mock_conn2)
        mock_acquire_cm2.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(side_effect=[mock_acquire_cm1, mock_acquire_cm2])

        mock_prompt = MagicMock()
        mock_prompt.template = "{MESSAGE_HISTORY}"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="User: Hello\nBot: Hi there!")

        mock_result = MagicMock()
        mock_result.content = "[]"
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 50
        mock_result.completion_tokens = 10
        mock_result.total_tokens = 60
        mock_result.cost_usd = 0.001
        mock_result.latency_ms = 200

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch(
                "lattice.memory.batch_consolidation.get_prompt",
                return_value=mock_prompt,
            ):
                with patch(
                    "lattice.memory.batch_consolidation.get_llm_client",
                    return_value=mock_llm_client,
                ):
                    await run_batch_consolidation()
                    # Verify safe_format was called with correct message history
                    call_args = mock_prompt.safe_format.call_args
                    assert "MESSAGE_HISTORY" in call_args[1]
                    history = call_args[1]["MESSAGE_HISTORY"]
                    assert "User: Hello" in history
                    assert "Bot: Hi there!" in history

    @pytest.mark.asyncio
    async def test_memory_context_format(self) -> None:
        """Test that memory context is formatted as bullet points."""
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "Test message",
                "is_bot": False,
                "timestamp": datetime.now(UTC),
            }
        ]
        memories = [
            {
                "subject": "user",
                "predicate": "lives_in",
                "object": "Richmond",
                "created_at": datetime.now(UTC),
            },
            {
                "subject": "user",
                "predicate": "has_goal",
                "object": "run a marathon",
                "created_at": datetime.now(UTC),
            },
        ]

        mock_conn1 = AsyncMock()
        mock_conn1.fetchrow = AsyncMock(return_value={"metric_value": "100"})
        mock_conn1.fetch = AsyncMock(side_effect=[messages, memories])

        mock_conn2 = AsyncMock()
        mock_conn2.execute = AsyncMock()

        mock_acquire_cm1 = MagicMock()
        mock_acquire_cm1.__aenter__ = AsyncMock(return_value=mock_conn1)
        mock_acquire_cm1.__aexit__ = AsyncMock(return_value=None)

        mock_acquire_cm2 = MagicMock()
        mock_acquire_cm2.__aenter__ = AsyncMock(return_value=mock_conn2)
        mock_acquire_cm2.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(side_effect=[mock_acquire_cm1, mock_acquire_cm2])

        mock_prompt = MagicMock()
        mock_prompt.template = "{MEMORY_CONTEXT}"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="Memories")

        mock_result = MagicMock()
        mock_result.content = "[]"
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 50
        mock_result.completion_tokens = 10
        mock_result.total_tokens = 60
        mock_result.cost_usd = 0.001
        mock_result.latency_ms = 200

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch(
                "lattice.memory.batch_consolidation.get_prompt",
                return_value=mock_prompt,
            ):
                with patch(
                    "lattice.memory.batch_consolidation.get_llm_client",
                    return_value=mock_llm_client,
                ):
                    await run_batch_consolidation()
                    call_args = mock_prompt.safe_format.call_args
                    assert "MEMORY_CONTEXT" in call_args[1]
                    context = call_args[1]["MEMORY_CONTEXT"]
                    assert "user --lives_in--> Richmond" in context
                    assert "user --has_goal--> run a marathon" in context

    @pytest.mark.asyncio
    async def test_empty_previous_memories_shows_placeholder(self) -> None:
        """Test that empty previous memories shows placeholder text."""
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "Test",
                "is_bot": False,
                "timestamp": datetime.now(UTC),
            }
        ]

        mock_conn1 = AsyncMock()
        mock_conn1.fetchrow = AsyncMock(return_value={"metric_value": "100"})
        mock_conn1.fetch = AsyncMock(side_effect=[messages, []])  # No memories

        mock_conn2 = AsyncMock()
        mock_conn2.execute = AsyncMock()

        mock_acquire_cm1 = MagicMock()
        mock_acquire_cm1.__aenter__ = AsyncMock(return_value=mock_conn1)
        mock_acquire_cm1.__aexit__ = AsyncMock(return_value=None)

        mock_acquire_cm2 = MagicMock()
        mock_acquire_cm2.__aenter__ = AsyncMock(return_value=mock_conn2)
        mock_acquire_cm2.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(side_effect=[mock_acquire_cm1, mock_acquire_cm2])

        mock_prompt = MagicMock()
        mock_prompt.template = "{MEMORY_CONTEXT}"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="(No previous memories)")

        mock_result = MagicMock()
        mock_result.content = "[]"
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 50
        mock_result.completion_tokens = 10
        mock_result.total_tokens = 60
        mock_result.cost_usd = 0.001
        mock_result.latency_ms = 200

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        with patch("lattice.memory.batch_consolidation.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool
            with patch(
                "lattice.memory.batch_consolidation.get_prompt",
                return_value=mock_prompt,
            ):
                with patch(
                    "lattice.memory.batch_consolidation.get_llm_client",
                    return_value=mock_llm_client,
                ):
                    await run_batch_consolidation()
                    call_args = mock_prompt.safe_format.call_args
                    assert call_args[1]["MEMORY_CONTEXT"] == "(No previous memories)"
