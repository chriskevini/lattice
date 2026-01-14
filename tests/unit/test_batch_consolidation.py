"""Unit tests for batch consolidation module."""

from lattice.utils.date_resolution import get_now
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from lattice.core.constants import CONSOLIDATION_BATCH_SIZE
from lattice.memory.batch_consolidation import (
    should_consolidate,
    run_consolidation_batch,
)


def create_mock_system_metrics_repo() -> MagicMock:
    """Create a mock system metrics repository."""
    mock_repo = MagicMock()
    mock_repo.get_metric = AsyncMock(return_value=None)
    mock_repo.set_metric = AsyncMock(return_value=None)
    return mock_repo


def create_mock_message_repo() -> MagicMock:
    """Create a mock message repository."""
    mock_repo = MagicMock()
    mock_repo.get_messages_since_cursor = AsyncMock(return_value=[])
    return mock_repo


class TestBatchSize:
    """Tests for CONSOLIDATION_BATCH_SIZE constant."""

    def test_batch_size_is_18(self) -> None:
        """Test that CONSOLIDATION_BATCH_SIZE is 18 as specified in design."""
        assert CONSOLIDATION_BATCH_SIZE == 18


class TestShouldConsolidate:
    """Tests for should_consolidate function."""

    @pytest.mark.asyncio
    async def test_should_consolidate_returns_true_above_threshold(self) -> None:
        """Test should_consolidate returns True when messages exceed threshold."""
        system_metrics_repo = create_mock_system_metrics_repo()
        system_metrics_repo.get_metric = AsyncMock(return_value="100")

        message_repo = create_mock_message_repo()
        message_repo.get_messages_since_cursor = AsyncMock(
            return_value=[
                {"id": uuid4(), "discord_message_id": i} for i in range(101, 121)
            ]
        )

        result = await should_consolidate(system_metrics_repo, message_repo)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_consolidate_returns_false_below_threshold(self) -> None:
        """Test should_consolidate returns False when messages below threshold."""
        system_metrics_repo = create_mock_system_metrics_repo()
        system_metrics_repo.get_metric = AsyncMock(return_value="100")

        message_repo = create_mock_message_repo()
        message_repo.get_messages_since_cursor = AsyncMock(
            return_value=[
                {"id": uuid4(), "discord_message_id": i} for i in range(101, 111)
            ]
        )

        result = await should_consolidate(system_metrics_repo, message_repo)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_consolidate_returns_true_at_threshold(self) -> None:
        """Test should_consolidate returns True when exactly at threshold."""
        system_metrics_repo = create_mock_system_metrics_repo()
        system_metrics_repo.get_metric = AsyncMock(return_value="100")

        message_repo = create_mock_message_repo()
        message_repo.get_messages_since_cursor = AsyncMock(
            return_value=[
                {"id": uuid4(), "discord_message_id": i} for i in range(101, 119)
            ]
        )

        result = await should_consolidate(system_metrics_repo, message_repo)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_consolidate_handles_zero_cursor(self) -> None:
        """Test should_consolidate works with initial zero cursor."""
        system_metrics_repo = create_mock_system_metrics_repo()
        system_metrics_repo.get_metric = AsyncMock(return_value="0")

        message_repo = create_mock_message_repo()
        message_repo.get_messages_since_cursor = AsyncMock(
            return_value=[
                {"id": uuid4(), "discord_message_id": i} for i in range(1, 26)
            ]
        )

        result = await should_consolidate(system_metrics_repo, message_repo)
        assert result is True

        message_repo.get_messages_since_cursor.assert_called_once_with(
            cursor_message_id=0,
            limit=CONSOLIDATION_BATCH_SIZE,
        )


class TestRunConsolidationBatch:
    """Tests for run_consolidation_batch function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_no_messages(self) -> None:
        """Test that batch returns False when no messages to process."""
        system_metrics_repo = create_mock_system_metrics_repo()
        system_metrics_repo.get_metric = AsyncMock(return_value="100")

        message_repo = create_mock_message_repo()
        message_repo.get_messages_since_cursor = AsyncMock(return_value=[])

        result = await run_consolidation_batch(
            system_metrics_repo=system_metrics_repo,
            message_repo=message_repo,
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            feedback_repo=MagicMock(),
            llm_client=MagicMock(),
            bot=MagicMock(),
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_no_prompt_template(self) -> None:
        """Test that batch returns True (skip) when MEMORY_CONSOLIDATION template missing.

        When no prompt template is found, the batch is considered "processed"
        to avoid repeatedly retrying the same messages. The system logs a warning.
        """
        system_metrics_repo = create_mock_system_metrics_repo()
        system_metrics_repo.get_metric = AsyncMock(return_value="100")

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

        message_repo = create_mock_message_repo()
        message_repo.get_messages_since_cursor = AsyncMock(return_value=messages)

        with patch("lattice.memory.batch_consolidation.get_prompt", return_value=None):
            result = await run_consolidation_batch(
                system_metrics_repo=system_metrics_repo,
                message_repo=message_repo,
                prompt_repo=MagicMock(),
                audit_repo=MagicMock(),
                feedback_repo=MagicMock(),
                llm_client=MagicMock(),
                bot=MagicMock(),
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_success_returns_true_and_updates_cursor(self) -> None:
        """Test successful batch processing returns True and updates cursor."""
        system_metrics_repo = create_mock_system_metrics_repo()
        system_metrics_repo.get_metric = AsyncMock(return_value="100")

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

        message_repo = create_mock_message_repo()
        message_repo.get_messages_since_cursor = AsyncMock(return_value=messages)

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
        mock_canonical_repo = MagicMock()
        mock_prompt_repo = MagicMock()
        mock_audit_repo = MagicMock()
        mock_feedback_repo = MagicMock()

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
                                    ):
                                        result = await run_consolidation_batch(
                                            system_metrics_repo=system_metrics_repo,
                                            message_repo=message_repo,
                                            canonical_repo=mock_canonical_repo,
                                            prompt_repo=mock_prompt_repo,
                                            audit_repo=mock_audit_repo,
                                            feedback_repo=mock_feedback_repo,
                                            llm_client=mock_llm_client,
                                            bot=mock_bot,
                                        )

                                        assert result is True
                                        message_repo.get_messages_since_cursor.assert_called_once_with(
                                            cursor_message_id=100,
                                            limit=CONSOLIDATION_BATCH_SIZE,
                                        )
                                        system_metrics_repo.set_metric.assert_called()

    @pytest.mark.asyncio
    async def test_extracts_and_stores_triples(self) -> None:
        """Test that batch extracts triples and stores them correctly."""
        system_metrics_repo = create_mock_system_metrics_repo()
        system_metrics_repo.get_metric = AsyncMock(return_value="100")

        message_id = uuid4()
        now = get_now()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "I live in Vancouver",
                "is_bot": False,
                "timestamp": now,
                "user_timezone": "UTC",
            }
        ]

        message_repo = create_mock_message_repo()
        message_repo.get_messages_since_cursor = AsyncMock(return_value=messages)

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
        mock_canonical_repo = MagicMock()
        mock_prompt_repo = MagicMock()
        mock_audit_repo = MagicMock()
        mock_feedback_repo = MagicMock()

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
                                            await run_consolidation_batch(
                                                system_metrics_repo=system_metrics_repo,
                                                message_repo=message_repo,
                                                canonical_repo=mock_canonical_repo,
                                                prompt_repo=mock_prompt_repo,
                                                audit_repo=mock_audit_repo,
                                                feedback_repo=mock_feedback_repo,
                                                llm_client=mock_llm_client,
                                                bot=mock_bot,
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
                                            assert (
                                                call_kwargs["source_batch_id"] == "101"
                                            )


class TestCanonicalFormIntegration:
    """Integration tests for batch consolidation with canonical form handling."""

    @pytest.mark.asyncio
    async def test_batch_with_canonical_form_extraction(self) -> None:
        """Test that batch extraction extracts and stores new canonical forms."""
        system_metrics_repo = create_mock_system_metrics_repo()
        system_metrics_repo.get_metric = AsyncMock(return_value="100")

        message_id = uuid4()
        now = get_now()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "I went to IKEA with my boyfriend",
                "is_bot": False,
                "timestamp": now,
                "user_timezone": "UTC",
            }
        ]

        message_repo = create_mock_message_repo()
        message_repo.get_messages_since_cursor = AsyncMock(return_value=messages)

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
        mock_canonical_repo = MagicMock()
        mock_prompt_repo = MagicMock()
        mock_audit_repo = MagicMock()
        mock_feedback_repo = MagicMock()

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
                                            await run_consolidation_batch(
                                                system_metrics_repo=system_metrics_repo,
                                                message_repo=message_repo,
                                                canonical_repo=mock_canonical_repo,
                                                prompt_repo=mock_prompt_repo,
                                                audit_repo=mock_audit_repo,
                                                feedback_repo=mock_feedback_repo,
                                                llm_client=mock_llm_client,
                                                bot=mock_bot,
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
    async def test_batch_handles_dict_triples_format(self) -> None:
        """Test that batch extraction handles dict format with 'triples' key."""
        system_metrics_repo = create_mock_system_metrics_repo()
        system_metrics_repo.get_metric = AsyncMock(return_value="100")

        message_id = uuid4()
        now = get_now()
        messages = [
            {
                "id": message_id,
                "discord_message_id": 101,
                "content": "I am training for a marathon",
                "is_bot": False,
                "timestamp": now,
                "user_timezone": "UTC",
            }
        ]

        message_repo = create_mock_message_repo()
        message_repo.get_messages_since_cursor = AsyncMock(return_value=messages)

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
        mock_canonical_repo = MagicMock()
        mock_prompt_repo = MagicMock()
        mock_audit_repo = MagicMock()
        mock_feedback_repo = MagicMock()

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
                                            await run_consolidation_batch(
                                                system_metrics_repo=system_metrics_repo,
                                                message_repo=message_repo,
                                                canonical_repo=mock_canonical_repo,
                                                prompt_repo=mock_prompt_repo,
                                                audit_repo=mock_audit_repo,
                                                feedback_repo=mock_feedback_repo,
                                                llm_client=mock_llm_client,
                                                bot=mock_bot,
                                            )

                                            mock_store.assert_called_once()
                                            call_kwargs = mock_store.call_args.kwargs
                                            assert len(call_kwargs["memories"]) == 1
                                            assert (
                                                call_kwargs["memories"][0]["object"]
                                                == "run a marathon"
                                            )
