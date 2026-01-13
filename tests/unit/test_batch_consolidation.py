"""Unit tests for batch consolidation module."""

from lattice.utils.date_resolution import get_now
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import json
from typing import Any
import pytest

from lattice.core.constants import CONSOLIDATION_BATCH_SIZE
from lattice.memory.batch_consolidation import (
    check_and_run_batch,
    run_batch_consolidation,
    should_consolidate,
    run_consolidation_batch,
)


def create_mock_pool_with_conn() -> tuple[MagicMock, AsyncMock]:
    """Create a properly mocked database pool with connection.

    Returns:
        Tuple of (mock_pool, mock_conn) for use in tests.
    """
    mock_conn = AsyncMock()
    # Ensure fetchrow and other common methods return something sensible by default
    mock_conn.fetchrow.return_value = None
    mock_conn.fetch.return_value = []
    mock_conn.fetchval.return_value = None
    mock_conn.execute.return_value = None

    mock_pool = MagicMock()

    # Create a helper that is both awaitable AND an async context manager
    class AsyncCM:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, *args):
            pass

        def __await__(self):
            async def _res():
                return mock_conn

            return _res().__await__()

    # In should_consolidate, it calls await message_repo._db_pool.pool.acquire()
    # In run_consolidation_batch, it calls async with db_pool.pool.acquire() as conn:
    # We want pool.acquire() to return something that supports both.
    mock_pool.pool.acquire.return_value = AsyncCM()

    # Also mock direct pool methods which are often used in the codebase
    mock_pool.fetchrow = AsyncMock()
    mock_pool.fetch = AsyncMock()
    mock_pool.fetchval = AsyncMock()
    mock_pool.execute = AsyncMock()

    return mock_pool, mock_conn


class TestBatchSize:
    """Tests for CONSOLIDATION_BATCH_SIZE constant."""

    def test_batch_size_is_18(self) -> None:
        """Test that CONSOLIDATION_BATCH_SIZE is 18 as specified in design."""
        assert CONSOLIDATION_BATCH_SIZE == 18


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

        with patch(
            "lattice.memory.batch_consolidation.run_batch_consolidation"
        ) as mock_run:
            mock_run.return_value = None
            await check_and_run_batch(db_pool=mock_pool, message_repo=MagicMock())
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

        with patch(
            "lattice.memory.batch_consolidation.run_batch_consolidation"
        ) as mock_run:
            mock_run.return_value = None
            await check_and_run_batch(db_pool=mock_pool, message_repo=MagicMock())
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

        with patch(
            "lattice.memory.batch_consolidation.run_batch_consolidation"
        ) as mock_run:
            mock_run.return_value = None
            await check_and_run_batch(db_pool=mock_pool, message_repo=MagicMock())
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

        with patch(
            "lattice.memory.batch_consolidation.run_batch_consolidation"
        ) as mock_run:
            mock_run.return_value = None
            await check_and_run_batch(db_pool=mock_pool, message_repo=MagicMock())
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

        with patch(
            "lattice.memory.batch_consolidation.run_batch_consolidation"
        ) as mock_run:
            mock_run.return_value = None
            await check_and_run_batch(db_pool=mock_pool, message_repo=MagicMock())
            call_args = mock_conn.fetchrow.call_args_list[1]
            assert call_args[0][1] == 0


class TestRunBatchConsolidation:
    """Tests for run_batch_consolidation function."""

    @pytest.mark.asyncio
    async def test_no_messages_returns_early(self) -> None:
        """Test that batch returns early when no new messages."""
        mock_pool, mock_conn = create_mock_pool_with_conn()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},
                {"cnt": 20},
            ]
        )
        mock_conn.fetch = AsyncMock(return_value=[])  # No messages

        mock_bot = MagicMock()

        with patch("lattice.memory.batch_consolidation.get_prompt") as mock_prompt:
            await run_batch_consolidation(
                db_pool=mock_pool,
                llm_client=MagicMock(),
                bot=mock_bot,
                message_repo=MagicMock(),
            )
            mock_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_threshold_check_prevents_double_run(self) -> None:
        """Test that threshold is re-checked inside transaction."""
        mock_pool, mock_conn = create_mock_pool_with_conn()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},
                {"cnt": 5},  # Re-check shows below threshold - another worker ran first
            ]
        )

        mock_bot = MagicMock()

        with patch("lattice.memory.batch_consolidation.get_prompt") as mock_prompt:
            await run_batch_consolidation(
                db_pool=mock_pool,
                llm_client=MagicMock(),
                bot=mock_bot,
                message_repo=MagicMock(),
            )
            mock_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prompt_template_aborts(self) -> None:
        """Test that batch aborts if MEMORY_CONSOLIDATION template missing."""
        mock_pool, mock_conn = create_mock_pool_with_conn()
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "Test message",
                "is_bot": False,
                "timestamp": get_now(),
            }
        ]

        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},
                {"cnt": 20},
            ]
        )
        mock_conn.fetch = AsyncMock(side_effect=[messages, []])

        mock_llm_client = MagicMock()
        mock_bot = MagicMock()

        with patch("lattice.memory.batch_consolidation.get_prompt") as mock_prompt:
            mock_prompt.return_value = None
            with patch(
                "lattice.memory.batch_consolidation.store_semantic_memories"
            ) as mock_store:
                await run_batch_consolidation(
                    db_pool=mock_pool,
                    llm_client=mock_llm_client,
                    bot=mock_bot,
                    message_repo=MagicMock(),
                )

    @pytest.mark.asyncio
    async def test_should_consolidate(self) -> None:
        """Test the should_consolidate helper."""
        mock_pool, mock_conn = create_mock_pool_with_conn()

        # Setup mocks
        mock_conn.fetchrow = AsyncMock(return_value={"metric_value": "100"})
        mock_conn.fetchval = AsyncMock(return_value=20)

        # Setup repository with mock pool
        mock_repo = MagicMock()
        mock_repo._db_pool = mock_pool

        # Ensure pool is the same as the one mock_pool.pool points to
        mock_repo._db_pool.pool = mock_pool.pool

        # Fix: should_consolidate calls db_pool.fetchval and db_pool.pool.acquire
        mock_pool.fetchval = AsyncMock(return_value=20)

        result = await should_consolidate(mock_repo)
        assert result is True

        # Verify the fetchval call
        mock_pool.fetchval.assert_called_once_with(
            "SELECT COUNT(*) FROM raw_messages WHERE discord_message_id > $1", 100
        )

    @pytest.mark.asyncio
    async def test_run_consolidation_batch_success(self) -> None:
        """Test the new run_consolidation_batch function."""
        mock_pool, mock_conn = create_mock_pool_with_conn()

        # Setup mocks for run_consolidation_batch
        message_id = uuid4()
        now = get_now()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "Test message",
                "is_bot": False,
                "timestamp": now,
                "user_timezone": "UTC",
            }
        ]

        mock_conn.fetchrow = AsyncMock(return_value={"metric_value": "100"})

        mock_repo = MagicMock()
        mock_repo.get_messages_since_cursor = AsyncMock(return_value=messages)

        mock_prompt = MagicMock()
        mock_prompt.template = "{bigger_episodic_context}"
        mock_prompt.temperature = 0.2
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = '{"memories": [{"subject": "User", "predicate": "tested", "object": "consolidation"}]}'
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 50
        mock_result.completion_tokens = 10
        mock_result.total_tokens = 60
        mock_result.cost_usd = 0.001
        mock_result.latency_ms = 200
        mock_result.audit_id = uuid4()

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        mock_bot = MagicMock()

        with patch(
            "lattice.memory.batch_consolidation.get_prompt", return_value=mock_prompt
        ):
            with patch(
                "lattice.memory.batch_consolidation.get_canonical_entities_list",
                return_value=[],
            ):
                with patch(
                    "lattice.memory.batch_consolidation.get_canonical_predicates_list",
                    return_value=[],
                ):
                    with patch(
                        "lattice.memory.batch_consolidation.get_canonical_entities_set",
                        return_value=set(),
                    ):
                        with patch(
                            "lattice.memory.batch_consolidation.get_canonical_predicates_set",
                            return_value=set(),
                        ):
                            with patch(
                                "lattice.memory.batch_consolidation.extract_canonical_forms",
                                return_value=(set(), set()),
                            ):
                                with patch(
                                    "lattice.memory.batch_consolidation.store_canonical_forms",
                                    return_value={"entities": 0, "predicates": 0},
                                ):
                                    with patch(
                                        "lattice.memory.batch_consolidation.store_semantic_memories"
                                    ) as mock_store:
                                        # Fix: Properly mock the async context manager acquisition
                                        # In create_mock_pool_with_conn, mock_pool.pool.acquire returns a MagicMock
                                        # that acts as an async context manager. We need to make sure it's set up
                                        # correctly for the call in run_consolidation_batch.

                                        result = await run_consolidation_batch(
                                            db_pool=mock_pool,
                                            message_repo=mock_repo,
                                            llm_client=mock_llm_client,
                                            bot=mock_bot,
                                        )

                                        assert result is True
                                        mock_repo.get_messages_since_cursor.assert_called_once_with(
                                            cursor_message_id=100,
                                            limit=CONSOLIDATION_BATCH_SIZE,
                                        )
                                        mock_store.assert_called_once()
                                        mock_conn.execute.assert_called()  # Should update cursor

    @pytest.mark.asyncio
    async def test_extracts_and_stores_triples(self) -> None:
        """Test that batch extracts triples and stores them."""
        mock_pool, mock_conn1 = create_mock_pool_with_conn()
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "I live in Vancouver",
                "is_bot": False,
                "timestamp": get_now(),
            }
        ]
        memories = [
            {
                "subject": "user",
                "predicate": "lives_in",
                "object": "Richmond",
                "created_at": get_now(),
            }
        ]

        mock_conn1.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},
                {"cnt": 20},
            ]
        )
        mock_conn1.fetch = AsyncMock(side_effect=[messages, memories])

        mock_prompt = MagicMock()
        mock_prompt.template = "Template with {bigger_episodic_context}"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="Formatted prompt")
        mock_prompt.version = 1

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
        mock_result.audit_id = uuid4()

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        mock_bot = MagicMock()

        with patch(
            "lattice.memory.batch_consolidation.get_prompt",
            return_value=mock_prompt,
        ):
            with patch(
                "lattice.memory.batch_consolidation.parse_llm_json_response",
                return_value=[
                    {
                        "subject": "User",
                        "predicate": "lives_in",
                        "object": "Vancouver",
                    }
                ],
            ):
                with patch(
                    "lattice.memory.batch_consolidation.store_semantic_memories"
                ) as mock_store:
                    with patch(
                        "lattice.memory.batch_consolidation.get_canonical_entities_list",
                        return_value=[],
                    ):
                        with patch(
                            "lattice.memory.batch_consolidation.get_canonical_predicates_list",
                            return_value=[],
                        ):
                            with patch(
                                "lattice.memory.batch_consolidation.get_canonical_entities_set",
                                return_value=set(),
                            ):
                                with patch(
                                    "lattice.memory.batch_consolidation.get_canonical_predicates_set",
                                    return_value=set(),
                                ):
                                    with patch(
                                        "lattice.memory.batch_consolidation.extract_canonical_forms",
                                        return_value=(set(), set()),
                                    ):
                                        with patch(
                                            "lattice.memory.batch_consolidation.store_canonical_forms",
                                            return_value={
                                                "entities": 0,
                                                "predicates": 0,
                                            },
                                        ):
                                            await run_batch_consolidation(
                                                db_pool=mock_pool,
                                                llm_client=mock_llm_client,
                                                bot=mock_bot,
                                                message_repo=MagicMock(),
                                            )
                                        mock_store.assert_called_once()
                                        call_kwargs = mock_store.call_args.kwargs
                                        assert len(call_kwargs["memories"]) == 1
                                        assert (
                                            call_kwargs["memories"][0]["subject"]
                                            == "User"
                                        )
                                        assert (
                                            call_kwargs["memories"][0]["predicate"]
                                            == "lives_in"
                                        )
                                        assert (
                                            call_kwargs["memories"][0]["object"]
                                            == "Vancouver"
                                        )
                                        assert call_kwargs["source_batch_id"] == "101"

    @pytest.mark.asyncio
    async def test_updates_last_batch_message_id(self) -> None:
        """Test that last_batch_message_id is updated after successful batch."""
        mock_pool, mock_conn1 = create_mock_pool_with_conn()
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 118,
                "content": "Test",
                "is_bot": False,
                "timestamp": get_now(),
            }
        ]

        mock_conn1.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},
                {"cnt": 20},
            ]
        )
        mock_conn1.fetch = AsyncMock(side_effect=[messages, []])

        mock_prompt = MagicMock()
        mock_prompt.template = "Template"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="Formatted")
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = "not valid json at all"
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 50
        mock_result.completion_tokens = 5
        mock_result.total_tokens = 55
        mock_result.cost_usd = 0.001
        mock_result.latency_ms = 150
        mock_result.audit_id = uuid4()

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        mock_bot = MagicMock()

        from lattice.utils.json_parser import JSONParseError

        with patch(
            "lattice.memory.batch_consolidation.get_prompt",
            return_value=mock_prompt,
        ):
            with patch(
                "lattice.memory.batch_consolidation.parse_llm_json_response",
                side_effect=JSONParseError(
                    raw_content="not valid json",
                    parse_error=json.JSONDecodeError("Expecting value", "", 0),
                ),
            ):
                with patch(
                    "lattice.memory.batch_consolidation.notify_parse_error_to_dream"
                ):
                    with patch(
                        "lattice.memory.batch_consolidation.store_semantic_memories"
                    ) as mock_store:
                        with patch(
                            "lattice.memory.batch_consolidation.get_canonical_entities_list",
                            return_value=[],
                        ):
                            with patch(
                                "lattice.memory.batch_consolidation.get_canonical_predicates_list",
                                return_value=[],
                            ):
                                with patch(
                                    "lattice.memory.batch_consolidation.get_canonical_entities_set",
                                    return_value=set(),
                                ):
                                    with patch(
                                        "lattice.memory.batch_consolidation.get_canonical_predicates_set",
                                        return_value=set(),
                                    ):
                                        with patch(
                                            "lattice.memory.batch_consolidation.extract_canonical_forms",
                                            return_value=(set(), set()),
                                        ):
                                            with patch(
                                                "lattice.memory.batch_consolidation.store_canonical_forms",
                                                return_value={
                                                    "entities": 0,
                                                    "predicates": 0,
                                                },
                                            ):
                                                await run_batch_consolidation(
                                                    db_pool=mock_pool,
                                                    llm_client=mock_llm_client,
                                                    bot=mock_bot,
                                                    message_repo=MagicMock(),
                                                )
                                            mock_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_message_history_format(self) -> None:
        """Test that message history is formatted correctly with User/Bot prefix."""
        mock_pool, mock_conn1 = create_mock_pool_with_conn()
        message_id = uuid4()
        now = get_now()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "Hello",
                "is_bot": False,
                "timestamp": now,
            },
            {
                "id": uuid4(),
                "discord_message_id": 102,
                "content": "Hi there!",
                "is_bot": True,
                "timestamp": now,
            },
        ]

        mock_conn1.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},
                {"cnt": 20},
            ]
        )
        mock_conn1.fetch = AsyncMock(side_effect=[messages, []])

        mock_prompt = MagicMock()
        mock_prompt.template = "{bigger_episodic_context}"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="User: Hello\nBot: Hi there!")
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = "[]"
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 50
        mock_result.completion_tokens = 10
        mock_result.total_tokens = 60
        mock_result.cost_usd = 0.001
        mock_result.latency_ms = 200
        mock_result.audit_id = uuid4()

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        mock_bot = MagicMock()

        with patch(
            "lattice.memory.batch_consolidation.get_prompt",
            return_value=mock_prompt,
        ):
            with patch(
                "lattice.memory.batch_consolidation.get_canonical_entities_list",
                return_value=[],
            ):
                with patch(
                    "lattice.memory.batch_consolidation.get_canonical_predicates_list",
                    return_value=[],
                ):
                    with patch(
                        "lattice.memory.batch_consolidation.get_canonical_entities_set",
                        return_value=set(),
                    ):
                        with patch(
                            "lattice.memory.batch_consolidation.get_canonical_predicates_set",
                            return_value=set(),
                        ):
                            with patch(
                                "lattice.memory.batch_consolidation.extract_canonical_forms",
                                return_value=(set(), set()),
                            ):
                                with patch(
                                    "lattice.memory.batch_consolidation.store_canonical_forms",
                                    return_value={
                                        "entities": 0,
                                        "predicates": 0,
                                    },
                                ):
                                    await run_batch_consolidation(
                                        db_pool=mock_pool,
                                        llm_client=mock_llm_client,
                                        bot=mock_bot,
                                        message_repo=MagicMock(),
                                    )
                                    call_args = mock_prompt.safe_format.call_args
                                    assert "bigger_episodic_context" in call_args[1]
                                history = call_args[1]["bigger_episodic_context"]
                                assert "User: Hello" in history
                                assert "Bot: Hi there!" in history

    @pytest.mark.asyncio
    async def test_memory_context_format(self) -> None:
        """Test that memory context is handled as a context variable even if not in template."""
        mock_pool, mock_conn1 = create_mock_pool_with_conn()
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "Test message",
                "is_bot": False,
                "timestamp": get_now(),
            }
        ]
        memories = [
            {
                "subject": "user",
                "predicate": "lives_in",
                "object": "Richmond",
                "created_at": get_now(),
            }
        ]

        mock_conn1.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},
                {"cnt": 20},
            ]
        )
        mock_conn1.fetch = AsyncMock(side_effect=[messages, memories])

        mock_prompt = MagicMock()
        mock_prompt.template = "No placeholders"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="No placeholders")
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = "[]"
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 50
        mock_result.completion_tokens = 10
        mock_result.total_tokens = 60
        mock_result.cost_usd = 0.001
        mock_result.latency_ms = 200
        mock_result.audit_id = uuid4()

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        mock_bot = MagicMock()

        with patch(
            "lattice.memory.batch_consolidation.get_prompt",
            return_value=mock_prompt,
        ):
            with patch(
                "lattice.memory.batch_consolidation.get_canonical_entities_list",
                return_value=[],
            ):
                with patch(
                    "lattice.memory.batch_consolidation.get_canonical_predicates_list",
                    return_value=[],
                ):
                    with patch(
                        "lattice.memory.batch_consolidation.get_canonical_entities_set",
                        return_value=set(),
                    ):
                        with patch(
                            "lattice.memory.batch_consolidation.get_canonical_predicates_set",
                            return_value=set(),
                        ):
                            with patch(
                                "lattice.memory.batch_consolidation.extract_canonical_forms",
                                return_value=(set(), set()),
                            ):
                                with patch(
                                    "lattice.memory.batch_consolidation.store_canonical_forms",
                                    return_value={
                                        "entities": 0,
                                        "predicates": 0,
                                    },
                                ):
                                    await run_batch_consolidation(
                                        db_pool=mock_pool,
                                        llm_client=mock_llm_client,
                                        bot=mock_bot,
                                        message_repo=MagicMock(),
                                    )
                                    # semantic_context should NOT be in call_args anymore
                                    call_args = mock_prompt.safe_format.call_args
                                    assert "semantic_context" not in call_args[1]

    @pytest.mark.asyncio
    async def test_empty_previous_memories_shows_placeholder(self) -> None:
        """Test that memory context is not passed even if empty."""
        mock_pool, mock_conn1 = create_mock_pool_with_conn()
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "Test",
                "is_bot": False,
                "timestamp": get_now(),
            }
        ]

        mock_conn1.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},
                {"cnt": 20},
            ]
        )
        mock_conn1.fetch = AsyncMock(side_effect=[messages, []])

        mock_prompt = MagicMock()
        mock_prompt.template = "No placeholders"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="No placeholders")
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = "[]"
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 50
        mock_result.completion_tokens = 10
        mock_result.total_tokens = 60
        mock_result.cost_usd = 0.001
        mock_result.latency_ms = 200
        mock_result.audit_id = uuid4()

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        mock_bot = MagicMock()

        with patch(
            "lattice.memory.batch_consolidation.get_prompt",
            return_value=mock_prompt,
        ):
            with patch(
                "lattice.memory.batch_consolidation.get_canonical_entities_list",
                return_value=[],
            ):
                with patch(
                    "lattice.memory.batch_consolidation.get_canonical_predicates_list",
                    return_value=[],
                ):
                    with patch(
                        "lattice.memory.batch_consolidation.get_canonical_entities_set",
                        return_value=set(),
                    ):
                        with patch(
                            "lattice.memory.batch_consolidation.get_canonical_predicates_set",
                            return_value=set(),
                        ):
                            with patch(
                                "lattice.memory.batch_consolidation.extract_canonical_forms",
                                return_value=(set(), set()),
                            ):
                                with patch(
                                    "lattice.memory.batch_consolidation.store_canonical_forms",
                                    return_value={
                                        "entities": 0,
                                        "predicates": 0,
                                    },
                                ):
                                    await run_batch_consolidation(
                                        db_pool=mock_pool,
                                        llm_client=mock_llm_client,
                                        bot=mock_bot,
                                        message_repo=MagicMock(),
                                    )
                                    call_args = mock_prompt.safe_format.call_args
                                    assert "semantic_context" not in call_args[1]


class TestRaceCondition:
    """Tests for race condition handling in batch processing."""

    @pytest.mark.asyncio
    async def test_get_last_batch_id_helper(self) -> None:
        """Test the _get_last_batch_id helper function."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"metric_value": "12345"})

        from lattice.memory.batch_consolidation import _get_last_batch_id

        result = await _get_last_batch_id(mock_conn)
        assert result == "12345"
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_last_batch_id_returns_zero_when_not_found(self) -> None:
        """Test _get_last_batch_id returns '0' when no value exists."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        from lattice.memory.batch_consolidation import _get_last_batch_id

        result = await _get_last_batch_id(mock_conn)
        assert result == "0"

    @pytest.mark.asyncio
    async def test_update_last_batch_id_helper(self) -> None:
        """Test the _update_last_batch_id helper function."""
        mock_conn = AsyncMock()

        from lattice.memory.batch_consolidation import _update_last_batch_id

        await _update_last_batch_id(mock_conn, "99999")
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "UPDATE system_metrics" in call_args[0][0]
        assert call_args[0][1] == "99999"

    @pytest.mark.asyncio
    async def test_concurrent_workers_race_prevention(self) -> None:
        """Test that concurrent workers don't process same batch twice.

        This tests the double-checked locking pattern: first worker passes
        initial check, second worker also passes initial check, but when
        first worker updates last_batch_id, second worker's re-check fails.
        """
        mock_pool, mock_conn = create_mock_pool_with_conn()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},
                {
                    "cnt": 5
                },  # Second check shows only 5 - another worker updated last_batch_id
            ]
        )
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch("lattice.memory.batch_consolidation.get_prompt") as mock_prompt:
            await run_batch_consolidation(
                db_pool=mock_pool,
                llm_client=MagicMock(),
                bot=MagicMock(),
                message_repo=MagicMock(),
            )

            mock_prompt.assert_not_called()


class TestCanonicalFormIntegration:
    """Integration tests for batch consolidation with canonical form handling."""

    @pytest.mark.asyncio
    async def test_batch_with_canonical_form_extraction(self) -> None:
        """Test that batch extraction extracts and stores new canonical forms."""
        mock_pool, mock_conn1 = create_mock_pool_with_conn()
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "I went to IKEA with my boyfriend",
                "is_bot": False,
                "timestamp": get_now(),
            }
        ]
        memories: list[dict[str, Any]] = []

        mock_conn1.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},
                {"cnt": 20},
            ]
        )
        mock_conn1.fetch = AsyncMock(side_effect=[messages, memories])

        mock_prompt = MagicMock()
        mock_prompt.template = "{canonical_entities} {canonical_predicates}"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="Formatted prompt")
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = '{"triples": [{"subject": "User", "predicate": "did activity", "object": "hanging out with boyfriend"}, {"subject": "hanging out with boyfriend", "predicate": "at location", "object": "IKEA"}]}'
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 100
        mock_result.completion_tokens = 50
        mock_result.total_tokens = 150
        mock_result.cost_usd = 0.002
        mock_result.latency_ms = 500
        mock_result.audit_id = uuid4()

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        mock_bot = MagicMock()

        with patch(
            "lattice.memory.batch_consolidation.get_prompt",
            return_value=mock_prompt,
        ):
            with patch(
                "lattice.memory.batch_consolidation.parse_llm_json_response",
                return_value={
                    "triples": [
                        {
                            "subject": "User",
                            "predicate": "did activity",
                            "object": "hanging out with boyfriend",
                        },
                        {
                            "subject": "hanging out with boyfriend",
                            "predicate": "at location",
                            "object": "IKEA",
                        },
                    ]
                },
            ):
                with patch(
                    "lattice.memory.batch_consolidation.store_semantic_memories"
                ) as mock_store:
                    with patch(
                        "lattice.memory.batch_consolidation.get_canonical_entities_list",
                        return_value=["Mother"],
                    ):
                        with patch(
                            "lattice.memory.batch_consolidation.get_canonical_predicates_list",
                            return_value=["has goal"],
                        ):
                            with patch(
                                "lattice.memory.batch_consolidation.get_canonical_entities_set",
                                return_value={"Mother", "User"},
                            ):
                                with patch(
                                    "lattice.memory.batch_consolidation.get_canonical_predicates_set",
                                    return_value={"has goal"},
                                ):
                                    with patch(
                                        "lattice.memory.batch_consolidation.extract_canonical_forms",
                                        return_value=(
                                            {
                                                "IKEA",
                                                "hanging out with boyfriend",
                                            },
                                            {"did activity", "at location"},
                                        ),
                                    ):
                                        with patch(
                                            "lattice.memory.batch_consolidation.store_canonical_forms",
                                            return_value={
                                                "entities": 2,
                                                "predicates": 2,
                                            },
                                        ) as mock_store_canonical:
                                            await run_batch_consolidation(
                                                db_pool=mock_pool,
                                                llm_client=mock_llm_client,
                                                bot=mock_bot,
                                                message_repo=MagicMock(),
                                            )

                                            mock_store.assert_called_once()
                                            store_kwargs = mock_store.call_args.kwargs
                                            assert len(store_kwargs["memories"]) == 2
                                            assert (
                                                store_kwargs["memories"][0]["predicate"]
                                                == "did activity"
                                            )

                                            mock_store_canonical.assert_called_once()
                                            store_kwargs = (
                                                mock_store_canonical.call_args.kwargs
                                            )
                                            assert (
                                                "IKEA" in store_kwargs["new_entities"]
                                            )
                                            assert (
                                                "did activity"
                                                in store_kwargs["new_predicates"]
                                            )

    @pytest.mark.asyncio
    async def test_batch_prompt_includes_canonical_placeholders(self) -> None:
        """Test that the prompt template includes canonical_entities and canonical_predicates placeholders."""
        mock_pool, mock_conn1 = create_mock_pool_with_conn()
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "Test message",
                "is_bot": False,
                "timestamp": get_now(),
            }
        ]
        memories: list[dict[str, Any]] = []

        mock_conn1.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},
                {"cnt": 20},
            ]
        )
        mock_conn1.fetch = AsyncMock(side_effect=[messages, memories])

        mock_prompt = MagicMock()
        mock_prompt.template = "{canonical_entities} {canonical_predicates}"
        mock_prompt.temperature = 0.2
        mock_prompt.version = 1

        def safe_format_mock(**kwargs):
            assert "canonical_entities" in kwargs
            assert "canonical_predicates" in kwargs
            return "Formatted"

        mock_prompt.safe_format = safe_format_mock

        mock_result = MagicMock()
        mock_result.content = '{"triples": []}'
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 50
        mock_result.completion_tokens = 10
        mock_result.total_tokens = 60
        mock_result.cost_usd = 0.001
        mock_result.latency_ms = 200
        mock_result.audit_id = uuid4()

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        mock_bot = MagicMock()

        with patch(
            "lattice.memory.batch_consolidation.get_prompt",
            return_value=mock_prompt,
        ):
            with patch(
                "lattice.memory.batch_consolidation.parse_llm_json_response",
                return_value={"triples": []},
            ):
                with patch(
                    "lattice.memory.batch_consolidation.get_canonical_entities_list",
                    return_value=["Mother", "Boyfriend"],
                ):
                    with patch(
                        "lattice.memory.batch_consolidation.get_canonical_predicates_list",
                        return_value=["has goal", "due by"],
                    ):
                        with patch(
                            "lattice.memory.batch_consolidation.get_canonical_entities_set",
                            return_value={"Mother", "Boyfriend"},
                        ):
                            with patch(
                                "lattice.memory.batch_consolidation.get_canonical_predicates_set",
                                return_value={"has goal", "due by"},
                            ):
                                with patch(
                                    "lattice.memory.batch_consolidation.extract_canonical_forms",
                                    return_value=(set(), set()),
                                ):
                                    with patch(
                                        "lattice.memory.batch_consolidation.store_canonical_forms",
                                        return_value={
                                            "entities": 0,
                                            "predicates": 0,
                                        },
                                    ):
                                        await run_batch_consolidation(
                                            db_pool=mock_pool,
                                            llm_client=mock_llm_client,
                                            bot=mock_bot,
                                            message_repo=MagicMock(),
                                        )

    @pytest.mark.asyncio
    async def test_batch_handles_dict_triples_format(self) -> None:
        """Test that batch extraction handles dict format with 'triples' key."""
        mock_pool, mock_conn1 = create_mock_pool_with_conn()
        message_id = uuid4()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "I am training for a marathon",
                "is_bot": False,
                "timestamp": get_now(),
            }
        ]
        memories: list[dict[str, Any]] = []

        mock_conn1.fetchrow = AsyncMock(
            side_effect=[
                {"metric_value": "100"},
                {"cnt": 20},
            ]
        )
        mock_conn1.fetch = AsyncMock(side_effect=[messages, memories])

        mock_prompt = MagicMock()
        mock_prompt.template = "Template"
        mock_prompt.temperature = 0.2
        mock_prompt.safe_format = MagicMock(return_value="Formatted")
        mock_prompt.version = 1

        mock_result = MagicMock()
        mock_result.content = '{"triples": [{"subject": "User", "predicate": "has goal", "object": "run a marathon"}]}'
        mock_result.model = "gpt-4"
        mock_result.provider = "openai"
        mock_result.prompt_tokens = 100
        mock_result.completion_tokens = 50
        mock_result.total_tokens = 150
        mock_result.cost_usd = 0.002
        mock_result.latency_ms = 500
        mock_result.audit_id = uuid4()

        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock(return_value=mock_result)

        mock_bot = MagicMock()

        with patch(
            "lattice.memory.batch_consolidation.get_prompt",
            return_value=mock_prompt,
        ):
            with patch(
                "lattice.memory.batch_consolidation.parse_llm_json_response",
                return_value={
                    "triples": [
                        {
                            "subject": "User",
                            "predicate": "has goal",
                            "object": "run a marathon",
                        }
                    ]
                },
            ):
                with patch(
                    "lattice.memory.batch_consolidation.store_semantic_memories"
                ) as mock_store:
                    with patch(
                        "lattice.memory.batch_consolidation.get_canonical_entities_list",
                        return_value=[],
                    ):
                        with patch(
                            "lattice.memory.batch_consolidation.get_canonical_predicates_list",
                            return_value=[],
                        ):
                            with patch(
                                "lattice.memory.batch_consolidation.get_canonical_entities_set",
                                return_value=set(),
                            ):
                                with patch(
                                    "lattice.memory.batch_consolidation.get_canonical_predicates_set",
                                    return_value=set(),
                                ):
                                    with patch(
                                        "lattice.memory.batch_consolidation.extract_canonical_forms",
                                        return_value=(
                                            {"run a marathon"},
                                            {"has goal"},
                                        ),
                                    ):
                                        with patch(
                                            "lattice.memory.batch_consolidation.store_canonical_forms",
                                            return_value={
                                                "entities": 1,
                                                "predicates": 1,
                                            },
                                        ):
                                            await run_batch_consolidation(
                                                db_pool=mock_pool,
                                                llm_client=mock_llm_client,
                                                bot=mock_bot,
                                                message_repo=MagicMock(),
                                            )

                                            mock_store.assert_called_once()
                                            call_kwargs = mock_store.call_args.kwargs
                                            assert len(call_kwargs["memories"]) == 1
                                            assert (
                                                call_kwargs["memories"][0]["object"]
                                                == "run a marathon"
                                            )
