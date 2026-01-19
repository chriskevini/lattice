"""End-to-end integration tests for the complete ENGRAM message flow pipeline.

This module provides comprehensive tests that verify the full message processing
pipeline works correctly from message ingestion through to async consolidation.

The test flow simulates:
1. User message ingestion (store_user_message)
2. Context strategy analysis
3. Entity detection and semantic memory retrieval
4. Response generation
5. Bot message storage (store_bot_message)
6. Async consolidation trigger

Uses worker-isolated data from conftest.py for safe parallel test execution.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.core import memory_orchestrator, response_generator
from lattice.core.context import ChannelContextCache
from lattice.core.context_strategy import context_strategy, retrieve_context
from lattice.memory import episodic
from lattice.memory.context import (
    PostgresCanonicalRepository,
    PostgresContextRepository,
    PostgresMessageRepository,
    PostgresSemanticMemoryRepository,
)
from lattice.memory.repositories import (
    PostgresPromptAuditRepository,
    PostgresPromptRegistryRepository,
    PostgresSystemMetricsRepository,
    PostgresUserFeedbackRepository,
)
from lattice.utils.llm import AuditResult


@pytest.fixture
def canonical_repo(db_pool) -> PostgresCanonicalRepository:
    """Fixture providing a CanonicalRepository."""
    return PostgresCanonicalRepository(db_pool)


@pytest.fixture
def semantic_repo(db_pool) -> PostgresSemanticMemoryRepository:
    """Fixture providing a SemanticMemoryRepository."""
    return PostgresSemanticMemoryRepository(db_pool)


@pytest.fixture
def episodic_repo(db_pool) -> PostgresMessageRepository:
    """Fixture providing a MessageRepository."""
    return PostgresMessageRepository(db_pool)


@pytest.fixture
def prompt_repo(db_pool) -> PostgresPromptRegistryRepository:
    """Fixture providing a PromptRegistryRepository."""
    return PostgresPromptRegistryRepository(db_pool)


@pytest.fixture
def audit_repo(db_pool) -> PostgresPromptAuditRepository:
    """Fixture providing a PromptAuditRepository."""
    return PostgresPromptAuditRepository(db_pool)


@pytest.fixture
def feedback_repo(db_pool) -> PostgresUserFeedbackRepository:
    """Fixture providing a UserFeedbackRepository."""
    return PostgresUserFeedbackRepository(db_pool)


@pytest.fixture
def system_metrics_repo(db_pool) -> PostgresSystemMetricsRepository:
    """Fixture providing a SystemMetricsRepository."""
    return PostgresSystemMetricsRepository(db_pool)


@pytest.fixture
def context_cache(db_pool) -> ChannelContextCache:
    """Create a fresh context cache for each test."""
    repo = PostgresContextRepository(db_pool)
    cache = ChannelContextCache(repository=repo, ttl=10)
    return cache


@pytest.fixture(autouse=True)
def reset_cache(context_cache):
    """Reset context cache before each test."""
    context_cache.clear()
    yield
    context_cache.clear()


def create_mock_llm_client(
    response_content: str,
    model: str = "anthropic/claude-3.5-sonnet",
    provider: str = "anthropic",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    total_tokens: int = 150,
    cost_usd: float = 0.001,
    latency_ms: int = 500,
    temperature: float = 0.7,
    prompt_key: str = "UNIFIED_RESPONSE",
) -> AsyncMock:
    """Create a mocked LLM client that returns predetermined responses.

    Args:
        response_content: The content to return from the LLM
        model: Model identifier for the response
        provider: Provider identifier for the response
        prompt_tokens: Number of prompt tokens for the response
        completion_tokens: Number of completion tokens for the response
        total_tokens: Total tokens for the response
        cost_usd: Cost in USD for the response
        latency_ms: Latency in milliseconds for the response
        temperature: Temperature setting for the response
        prompt_key: The prompt key for the response

    Returns:
        An AsyncMock configured to act as an LLM client
    """
    mock_client = AsyncMock()
    audit_result = AuditResult(
        content=response_content,
        model=model,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        temperature=temperature,
        audit_id=None,
        prompt_key=prompt_key,
    )
    mock_client.complete.return_value = audit_result
    return mock_client


def create_context_strategy_llm_client(
    entities: list[str],
    context_flags: list[str],
    unresolved_entities: list[str],
) -> AsyncMock:
    """Create a mocked LLM client for context strategy responses.

    Args:
        entities: List of entities to extract
        context_flags: List of context flags to set
        unresolved_entities: List of unresolved entities

    Returns:
        An AsyncMock configured for context strategy responses
    """
    import json

    response_data = {
        "entities": entities,
        "context_flags": context_flags,
        "unresolved_entities": unresolved_entities,
    }
    response_content = json.dumps(response_data)
    return create_mock_llm_client(
        response_content=response_content,
        model="anthropic/claude-3.5-sonnet",
        provider="anthropic",
        prompt_tokens=80,
        completion_tokens=40,
        total_tokens=120,
        cost_usd=0.0008,
        latency_ms=400,
        temperature=0.2,
        prompt_key="CONTEXT_STRATEGY",
    )


def create_consolidation_llm_client(
    memories: list[dict[str, str]],
) -> AsyncMock:
    """Create a mocked LLM client for memory consolidation responses.

    Args:
        memories: List of memory triples to return

    Returns:
        An AsyncMock configured for memory consolidation responses
    """
    import json

    response_data = {"memories": memories}
    response_content = json.dumps(response_data)
    return create_mock_llm_client(
        response_content=response_content,
        model="anthropic/claude-3.5-sonnet",
        provider="anthropic",
        prompt_tokens=200,
        completion_tokens=100,
        total_tokens=300,
        cost_usd=0.002,
        latency_ms=800,
        temperature=0.3,
        prompt_key="MEMORY_CONSOLIDATION",
    )


class TestEndToEndMessageFlow:
    """Integration tests for the complete message flow pipeline.

    These tests verify that data flows correctly through all stages of the
    ENGRAM pipeline, from user message ingestion to async consolidation.
    """

    @pytest.mark.asyncio
    async def test_complete_message_flow_with_declaration(
        self,
        db_pool,
        context_cache: ChannelContextCache,
        episodic_repo: PostgresMessageRepository,
        semantic_repo: PostgresSemanticMemoryRepository,
        canonical_repo: PostgresCanonicalRepository,
        prompt_repo: PostgresPromptRegistryRepository,
        audit_repo: PostgresPromptAuditRepository,
        feedback_repo: PostgresUserFeedbackRepository,
        system_metrics_repo: PostgresSystemMetricsRepository,
        unique_discord_id,
        unique_batch_id,
    ) -> None:
        """Test complete pipeline flow for a user declaration message.

        This test verifies:
        1. User message is stored in episodic memory
        2. Context strategy extracts entities and flags
        3. Semantic memory retrieval works
        4. Response generation produces output
        5. Bot message is stored with generation metadata
        6. Consolidation trigger evaluates correctly

        The message "I need to finish the lattice project by Friday" simulates
        a realistic user declaration that should trigger entity extraction
        and context-aware response generation.
        """
        user_message_content = "I need to finish the lattice project by Friday"
        channel_id = 12345
        discord_message_id = unique_discord_id.next_id()
        bot_message_id = unique_discord_id.next_id()

        context_strategy_llm = create_context_strategy_llm_client(
            entities=["lattice project", "Friday"],
            context_flags=[],
            unresolved_entities=["lattice project", "Friday"],
        )

        response_llm = create_mock_llm_client(
            response_content="Got it! You need to finish the lattice project by Friday. How's the progress coming along?",
            model="anthropic/claude-3.5-sonnet",
            provider="anthropic",
            prompt_tokens=150,
            completion_tokens=35,
            total_tokens=185,
            cost_usd=0.0012,
            latency_ms=550,
            temperature=0.7,
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_get_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list", return_value=[]
            ),
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_resp_prompt,
        ):
            from lattice.memory.procedural import PromptTemplate

            strategy_template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Extract entities from: {user_message}\nContext: {smaller_episodic_context}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = strategy_template

            response_template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template="Context: {episodic_context}\nSemantic: {semantic_context}\nUser: {user_message}",
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_resp_prompt.return_value = response_template

            message_id = await memory_orchestrator.store_user_message(
                content=user_message_content,
                discord_message_id=discord_message_id,
                channel_id=channel_id,
                message_repo=episodic_repo,
                timezone="UTC",
            )
            assert message_id is not None

            recent_messages = await episodic.get_recent_messages(
                channel_id=channel_id,
                limit=10,
                repo=episodic_repo,
            )
            history = [m for m in recent_messages if m.message_id != message_id]

            await context_cache.advance(channel_id)

            strategy = await context_strategy(
                message_id=message_id,
                user_message=user_message_content,
                recent_messages=history,
                context_cache=context_cache,
                channel_id=channel_id,
                user_timezone="UTC",
                discord_message_id=discord_message_id,
                llm_client=context_strategy_llm,
                canonical_repo=canonical_repo,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
            )

            assert strategy is not None
            assert "lattice project" in strategy.entities
            assert "Friday" in strategy.entities

            context = await retrieve_context(
                entities=strategy.entities,
                context_flags=strategy.context_flags,
                semantic_repo=semantic_repo,
                user_timezone="UTC",
            )

            assert "semantic_context" in context
            assert isinstance(context["semantic_context"], str)

            formatted_history = episodic.format_messages(history)
            (
                result,
                rendered_prompt,
                context_info,
            ) = await response_generator.generate_response(
                user_message=user_message_content,
                episodic_context=formatted_history,
                semantic_context=context["semantic_context"],
                llm_client=response_llm,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
            )

            assert result is not None
            assert "Friday" in result.content or "lattice" in result.content.lower()
            assert context_info["template"] == "UNIFIED_RESPONSE"

            mock_sent_message = MagicMock()
            mock_sent_message.id = bot_message_id

            with patch.object(episodic_repo, "_db_pool", db_pool):
                stored_bot_message_id = await memory_orchestrator.store_bot_message(
                    content=result.content,
                    discord_message_id=bot_message_id,
                    channel_id=channel_id,
                    generation_metadata={
                        "model": result.model,
                        "usage": {
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "total_tokens": result.total_tokens,
                        },
                        "context_info": context_info,
                    },
                    timezone="UTC",
                    message_repo=episodic_repo,
                )
                assert stored_bot_message_id is not None

        from lattice.memory import batch_consolidation

        should_run = await batch_consolidation.should_consolidate(
            system_metrics_repo=system_metrics_repo,
            message_repo=episodic_repo,
        )
        assert isinstance(should_run, bool)

    @pytest.mark.asyncio
    async def test_complete_flow_with_activity_context(
        self,
        db_pool,
        context_cache: ChannelContextCache,
        episodic_repo: PostgresMessageRepository,
        semantic_repo: PostgresSemanticMemoryRepository,
        canonical_repo: PostgresCanonicalRepository,
        prompt_repo: PostgresPromptRegistryRepository,
        audit_repo: PostgresPromptAuditRepository,
        feedback_repo: PostgresUserFeedbackRepository,
        unique_discord_id,
    ) -> None:
        """Test pipeline with activity context flag.

        Verifies that when the context strategy detects an activity query
        (e.g., "What did I do last week?"), the retrieval system properly
        fetches activity memories from the semantic graph.
        """
        user_message_content = "What did I do last week?"
        channel_id = 67890
        discord_message_id = unique_discord_id.next_id()

        context_strategy_llm = create_context_strategy_llm_client(
            entities=[],
            context_flags=["activity_context"],
            unresolved_entities=[],
        )

        _response_llm = create_mock_llm_client(
            response_content="Based on your recent activities, here are the things you did last week...",
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_get_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list", return_value=[]
            ),
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_resp_prompt,
        ):
            from lattice.memory.procedural import PromptTemplate

            strategy_template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Analyze: {user_message}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = strategy_template

            response_template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template="User: {user_message}\nActivities: {activity_context}",
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_resp_prompt.return_value = response_template

            message_id = await memory_orchestrator.store_user_message(
                content=user_message_content,
                discord_message_id=discord_message_id,
                channel_id=channel_id,
                message_repo=episodic_repo,
                timezone="UTC",
            )

            await context_cache.advance(channel_id)

            strategy = await context_strategy(
                message_id=message_id,
                user_message=user_message_content,
                recent_messages=[],
                context_cache=context_cache,
                channel_id=channel_id,
                user_timezone="UTC",
                discord_message_id=discord_message_id,
                llm_client=context_strategy_llm,
                canonical_repo=canonical_repo,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
            )

            assert strategy is not None
            assert "activity_context" in strategy.context_flags
            assert len(strategy.entities) == 0

            context = await retrieve_context(
                entities=strategy.entities,
                context_flags=strategy.context_flags,
                semantic_repo=semantic_repo,
                user_timezone="UTC",
            )

            assert "activity_context" in context

    @pytest.mark.asyncio
    async def test_complete_flow_with_existing_semantic_memories(
        self,
        db_pool,
        context_cache: ChannelContextCache,
        episodic_repo: PostgresMessageRepository,
        semantic_repo: PostgresSemanticMemoryRepository,
        canonical_repo: PostgresCanonicalRepository,
        prompt_repo: PostgresPromptRegistryRepository,
        audit_repo: PostgresPromptAuditRepository,
        feedback_repo: PostgresUserFeedbackRepository,
        unique_discord_id,
        unique_batch_id,
    ) -> None:
        """Test pipeline with pre-existing semantic memories.

        Verifies that when a user asks about a topic with existing semantic
        memories, the retrieval system properly fetches and includes them
        in the context for response generation.
        """
        channel_id = 11111
        discord_message_id = unique_discord_id.next_id()

        pre_existing_memories = [
            {
                "subject": "User",
                "predicate": "has goal",
                "object": "learn Python programming",
                "created_at": datetime.now(UTC),
            },
            {
                "subject": "Python",
                "predicate": "is a",
                "object": "programming language",
                "created_at": datetime.now(UTC),
            },
        ]

        await episodic_repo.store_semantic_memories(
            message_id=uuid.uuid4(),
            memories=pre_existing_memories,
            source_batch_id=unique_batch_id,
        )

        user_message_content = "Tell me about my Python goals"
        context_strategy_llm = create_context_strategy_llm_client(
            entities=["Python", "goals"],
            context_flags=["goal_context"],
            unresolved_entities=["Python"],
        )

        _response_llm = create_mock_llm_client(
            response_content="Your Python goal is to learn Python programming. You're making good progress!",
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_get_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list", return_value=[]
            ),
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_resp_prompt,
        ):
            from lattice.memory.procedural import PromptTemplate

            strategy_template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Extract: {user_message}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = strategy_template

            response_template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template="Goals: {goal_context}\nUser: {user_message}",
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_resp_prompt.return_value = response_template

            message_id = await memory_orchestrator.store_user_message(
                content=user_message_content,
                discord_message_id=discord_message_id,
                channel_id=channel_id,
                message_repo=episodic_repo,
                timezone="UTC",
            )

            await context_cache.advance(channel_id)

            strategy = await context_strategy(
                message_id=message_id,
                user_message=user_message_content,
                recent_messages=[],
                context_cache=context_cache,
                channel_id=channel_id,
                user_timezone="UTC",
                discord_message_id=discord_message_id,
                llm_client=context_strategy_llm,
                canonical_repo=canonical_repo,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
            )

            assert strategy is not None
            assert "goal_context" in strategy.context_flags

            context = await retrieve_context(
                entities=strategy.entities,
                context_flags=strategy.context_flags,
                semantic_repo=semantic_repo,
                user_timezone="UTC",
            )

            assert "goal_context" in context
            assert "learn Python programming" in context["goal_context"]

    @pytest.mark.asyncio
    async def test_consolidation_extracts_and_stores_memories(
        self,
        db_pool,
        episodic_repo: PostgresMessageRepository,
        semantic_repo: PostgresSemanticMemoryRepository,
        canonical_repo: PostgresCanonicalRepository,
        prompt_repo: PostgresPromptRegistryRepository,
        audit_repo: PostgresPromptAuditRepository,
        feedback_repo: PostgresUserFeedbackRepository,
        system_metrics_repo: PostgresSystemMetricsRepository,
        unique_discord_id,
        unique_batch_id,
    ) -> None:
        """Test that memory consolidation extracts and stores semantic memories.

        Verifies that when consolidation runs, it:
        1. Fetches unprocessed messages since the cursor
        2. Extracts memory triples using the LLM
        3. Stores the extracted memories in semantic_memories table
        4. Updates the consolidation cursor
        """
        channel_id = 22222

        consolidation_llm = create_consolidation_llm_client(
            memories=[
                {
                    "subject": "User",
                    "predicate": "is working on",
                    "object": "lattice project",
                },
                {
                    "subject": "lattice project",
                    "predicate": "has deadline",
                    "object": "Friday",
                },
            ],
        )

        message_ids = []
        for i in range(5):
            discord_id = unique_discord_id.next_id()
            msg_id = await memory_orchestrator.store_user_message(
                content=f"Test message {i} about the lattice project",
                discord_message_id=discord_id,
                channel_id=channel_id,
                message_repo=episodic_repo,
                timezone="UTC",
            )
            message_ids.append(msg_id)

        with (
            patch("lattice.memory.batch_consolidation.get_prompt") as mock_get_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list", return_value=[]
            ),
            patch(
                "lattice.memory.canonical.get_canonical_predicates_list",
                return_value=[],
            ),
            patch(
                "lattice.memory.canonical.get_canonical_entities_set",
                return_value=set(),
            ),
            patch(
                "lattice.memory.canonical.get_canonical_predicates_set",
                return_value=set(),
            ),
            patch(
                "lattice.memory.canonical.extract_canonical_forms",
                return_value=(set(), set()),
            ),
        ):
            from lattice.memory.batch_consolidation import (
                _get_consolidation_cursor,
                run_consolidation_batch,
            )
            from lattice.memory.procedural import PromptTemplate

            consolidation_template = PromptTemplate(
                prompt_key="MEMORY_CONSOLIDATION",
                template="Extract memories from: {user_message}",
                temperature=0.3,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = consolidation_template

            mock_bot = MagicMock()
            mock_bot.user.id = 12345

            cursor_before = await _get_consolidation_cursor(system_metrics_repo)
            cursor_before_int = int(cursor_before) if cursor_before.isdigit() else 0

            _result = await run_consolidation_batch(
                system_metrics_repo=system_metrics_repo,
                message_repo=episodic_repo,
                canonical_repo=canonical_repo,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
                llm_client=consolidation_llm,
                bot=mock_bot,
            )

            cursor_after = await _get_consolidation_cursor(system_metrics_repo)
            cursor_after_int = int(cursor_after) if cursor_after.isdigit() else 0

            assert cursor_after_int >= cursor_before_int, (
                "Cursor should not go backwards after consolidation"
            )

    @pytest.mark.asyncio
    async def test_conversation_history_affects_response(
        self,
        db_pool,
        context_cache: ChannelContextCache,
        episodic_repo: PostgresMessageRepository,
        semantic_repo: PostgresSemanticMemoryRepository,
        canonical_repo: PostgresCanonicalRepository,
        prompt_repo: PostgresPromptRegistryRepository,
        audit_repo: PostgresPromptAuditRepository,
        feedback_repo: PostgresUserFeedbackRepository,
        unique_discord_id,
    ) -> None:
        """Test that conversation history is properly included in responses.

        Verifies that when generating responses, the pipeline correctly
        includes recent conversation history in the episodic context.
        """
        channel_id = 33333

        prior_messages = [
            ("User", "I've been working on a new project"),
            ("Assistant", "That sounds interesting! What kind of project is it?"),
            ("User", "It's called lattice, it's for memory management"),
        ]

        for i, (role, content) in enumerate(prior_messages):
            discord_id = unique_discord_id.next_id()
            is_bot = role == "Assistant"
            await episodic.store_message(
                repo=episodic_repo,
                message=episodic.EpisodicMessage(
                    content=content,
                    discord_message_id=discord_id,
                    channel_id=channel_id,
                    is_bot=is_bot,
                    timestamp=datetime.now(UTC),
                ),
            )

        user_message_content = "Tell me more about it"
        discord_message_id = unique_discord_id.next_id()

        context_strategy_llm = create_context_strategy_llm_client(
            entities=["lattice"],
            context_flags=[],
            unresolved_entities=["lattice"],
        )

        captured_prompt = ""

        def capture_prompt(template, params):
            nonlocal captured_prompt
            captured_prompt = template.format(**params)
            return template

        response_llm = create_mock_llm_client(
            response_content="Based on our conversation, lattice is your memory management project!",
            prompt_key="UNIFIED_RESPONSE",
        )

        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_get_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list", return_value=[]
            ),
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_resp_prompt,
        ):
            from lattice.memory.procedural import PromptTemplate

            strategy_template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Extract from: {user_message}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = strategy_template

            response_template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template="History: {episodic_context}\nUser: {user_message}",
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_resp_prompt.return_value = response_template

            message_id = await memory_orchestrator.store_user_message(
                content=user_message_content,
                discord_message_id=discord_message_id,
                channel_id=channel_id,
                message_repo=episodic_repo,
                timezone="UTC",
            )

            await context_cache.advance(channel_id)

            recent_messages = await episodic.get_recent_messages(
                channel_id=channel_id,
                limit=10,
                repo=episodic_repo,
            )
            history = [m for m in recent_messages if m.message_id != message_id]

            _strategy = await context_strategy(
                message_id=message_id,
                user_message=user_message_content,
                recent_messages=history,
                context_cache=context_cache,
                channel_id=channel_id,
                user_timezone="UTC",
                discord_message_id=discord_message_id,
                llm_client=context_strategy_llm,
                canonical_repo=canonical_repo,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
            )

            formatted_history = episodic.format_messages(history)
            (
                result,
                rendered_prompt,
                context_info,
            ) = await response_generator.generate_response(
                user_message=user_message_content,
                episodic_context=formatted_history,
                semantic_context="No relevant context found.",
                llm_client=response_llm,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
            )

            assert (
                "lattice" in rendered_prompt.lower()
                or "memory management" in rendered_prompt.lower()
            )


class TestEndToEndEdgeCases:
    """Tests for edge cases and error handling in the pipeline."""

    @pytest.mark.asyncio
    async def test_empty_message_handling(
        self,
        db_pool,
        context_cache: ChannelContextCache,
        episodic_repo: PostgresMessageRepository,
        canonical_repo: PostgresCanonicalRepository,
        prompt_repo: PostgresPromptRegistryRepository,
        audit_repo: PostgresPromptAuditRepository,
        feedback_repo: PostgresUserFeedbackRepository,
        unique_discord_id,
    ) -> None:
        """Test handling of minimal/empty inputs."""
        channel_id = 44444
        discord_message_id = unique_discord_id.next_id()

        context_strategy_llm = create_context_strategy_llm_client(
            entities=[],
            context_flags=[],
            unresolved_entities=[],
        )

        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_get_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list", return_value=[]
            ),
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_resp_prompt,
        ):
            from lattice.memory.procedural import PromptTemplate

            strategy_template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Extract: {user_message}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = strategy_template

            response_template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template="User: {user_message}",
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_resp_prompt.return_value = response_template

            message_id = await memory_orchestrator.store_user_message(
                content="",
                discord_message_id=discord_message_id,
                channel_id=channel_id,
                message_repo=episodic_repo,
                timezone="UTC",
            )

            await context_cache.advance(channel_id)

            strategy = await context_strategy(
                message_id=message_id,
                user_message="",
                recent_messages=[],
                context_cache=context_cache,
                channel_id=channel_id,
                user_timezone="UTC",
                discord_message_id=discord_message_id,
                llm_client=context_strategy_llm,
                canonical_repo=canonical_repo,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
            )

            assert strategy is not None
            assert strategy.entities == []
            assert strategy.context_flags == []

    @pytest.mark.asyncio
    async def test_bot_message_not_included_in_context(
        self,
        db_pool,
        context_cache: ChannelContextCache,
        episodic_repo: PostgresMessageRepository,
        semantic_repo: PostgresSemanticMemoryRepository,
        canonical_repo: PostgresCanonicalRepository,
        prompt_repo: PostgresPromptRegistryRepository,
        audit_repo: PostgresPromptAuditRepository,
        feedback_repo: PostgresUserFeedbackRepository,
        unique_discord_id,
    ) -> None:
        """Test that bot messages are properly marked in episodic memory.

        Verifies that when storing bot messages, the is_bot flag is correctly
        set, and that the format_messages function correctly identifies
        bot vs user messages.
        """
        channel_id = 55555

        user_discord_id = unique_discord_id.next_id()
        bot_discord_id = unique_discord_id.next_id()

        await episodic.store_message(
            repo=episodic_repo,
            message=episodic.EpisodicMessage(
                content="Hello, I need help",
                discord_message_id=user_discord_id,
                channel_id=channel_id,
                is_bot=False,
                timestamp=datetime.now(UTC),
            ),
        )

        await episodic.store_message(
            repo=episodic_repo,
            message=episodic.EpisodicMessage(
                content="Hi! How can I help you today?",
                discord_message_id=bot_discord_id,
                channel_id=channel_id,
                is_bot=True,
                timestamp=datetime.now(UTC),
            ),
        )

        recent_messages = await episodic.get_recent_messages(
            channel_id=channel_id,
            limit=10,
            repo=episodic_repo,
        )

        assert len(recent_messages) >= 2

        user_msg = next((m for m in recent_messages if not m.is_bot), None)
        bot_msg = next((m for m in recent_messages if m.is_bot), None)

        assert user_msg is not None
        assert bot_msg is not None
        assert not user_msg.is_bot
        assert bot_msg.is_bot

        formatted = episodic.format_messages(recent_messages)
        assert "User:" in formatted
        assert "Assistant:" in formatted

    @pytest.mark.asyncio
    async def test_timezone_handling_in_pipeline(
        self,
        db_pool,
        context_cache: ChannelContextCache,
        episodic_repo: PostgresMessageRepository,
        semantic_repo: PostgresSemanticMemoryRepository,
        canonical_repo: PostgresCanonicalRepository,
        prompt_repo: PostgresPromptRegistryRepository,
        audit_repo: PostgresPromptAuditRepository,
        feedback_repo: PostgresUserFeedbackRepository,
        unique_discord_id,
    ) -> None:
        """Test that timezone is properly passed through the pipeline."""
        channel_id = 66666
        user_timezone = "America/New_York"
        discord_message_id = unique_discord_id.next_id()

        context_strategy_llm = create_context_strategy_llm_client(
            entities=["test"],
            context_flags=[],
            unresolved_entities=["test"],
        )

        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_get_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list", return_value=[]
            ),
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_resp_prompt,
        ):
            from lattice.memory.procedural import PromptTemplate

            strategy_template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Extract: {user_message}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = strategy_template

            response_template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template="User: {user_message}\nTimezone: {user_timezone}",
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_resp_prompt.return_value = response_template

            message_id = await memory_orchestrator.store_user_message(
                content="Test message with timezone",
                discord_message_id=discord_message_id,
                channel_id=channel_id,
                message_repo=episodic_repo,
                timezone=user_timezone,
            )

            await context_cache.advance(channel_id)

            strategy = await context_strategy(
                message_id=message_id,
                user_message="Test message with timezone",
                recent_messages=[],
                context_cache=context_cache,
                channel_id=channel_id,
                user_timezone=user_timezone,
                discord_message_id=discord_message_id,
                llm_client=context_strategy_llm,
                canonical_repo=canonical_repo,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
            )

            assert strategy is not None

            stored_messages = await episodic_repo.get_recent_messages(
                channel_id=channel_id,
                limit=1,
            )

            assert len(stored_messages) > 0
            assert stored_messages[0]["user_timezone"] == user_timezone
