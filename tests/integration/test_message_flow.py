"""Integration tests for end-to-end message flow.

Tests the complete pipeline from user message receipt through:
1. Ingestion (store in episodic memory)
2. Analysis (context strategy)
3. Retrieval (semantic context)
4. Generation (response)
5. Storage (bot message)

Uses real database fixtures from conftest.py.
"""

import uuid
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.core import memory_orchestrator, response_generator
from lattice.core.context import ChannelContextCache
from lattice.core.context_strategy import context_strategy, retrieve_context
from lattice.core.pipeline import UnifiedPipeline
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
    PostgresUserFeedbackRepository,
)
from lattice.utils.database import DatabasePool
from lattice.utils.llm import AuditResult


@pytest.fixture
def canonical_repo(db_pool: DatabasePool) -> PostgresCanonicalRepository:
    """Fixture providing a CanonicalRepository."""
    return PostgresCanonicalRepository(db_pool)


@pytest.fixture
def semantic_repo(db_pool: DatabasePool) -> PostgresSemanticMemoryRepository:
    """Fixture providing a SemanticMemoryRepository."""
    return PostgresSemanticMemoryRepository(db_pool)


@pytest.fixture
def message_repo(db_pool: DatabasePool) -> PostgresMessageRepository:
    """Fixture providing a MessageRepository."""
    return PostgresMessageRepository(db_pool)


@pytest.fixture
def prompt_repo(db_pool: DatabasePool) -> PostgresPromptRegistryRepository:
    """Fixture providing a PromptRegistryRepository."""
    return PostgresPromptRegistryRepository(db_pool)


@pytest.fixture
def audit_repo(db_pool: DatabasePool) -> PostgresPromptAuditRepository:
    """Fixture providing a PromptAuditRepository."""
    return PostgresPromptAuditRepository(db_pool)


@pytest.fixture
def feedback_repo(db_pool: DatabasePool) -> PostgresUserFeedbackRepository:
    """Fixture providing a UserFeedbackRepository."""
    return PostgresUserFeedbackRepository(db_pool)


@pytest.fixture
def context_cache(db_pool: DatabasePool) -> ChannelContextCache:
    """Create a fresh context cache for each test."""
    repo = PostgresContextRepository(db_pool)
    cache = ChannelContextCache(repository=repo, ttl=10)
    return cache


@pytest.fixture(autouse=True)
def reset_cache(context_cache: ChannelContextCache) -> Generator[None, None, None]:
    """Reset context cache before each test."""
    context_cache.clear()
    yield
    context_cache.clear()


class TestMessageFlowIntegration:
    """Integration tests for complete message flow."""

    @pytest.mark.asyncio
    async def test_store_user_message_creates_record(
        self,
        message_repo: PostgresMessageRepository,
        unique_discord_id,
    ) -> None:
        """Test that storing a user message creates a database record."""
        content = "Hello, this is a test message"
        discord_id = unique_discord_id.next_id()
        channel_id = unique_discord_id.next_id()  # Use unique channel ID

        # Store message
        message_id = await memory_orchestrator.store_user_message(
            content=content,
            discord_message_id=discord_id,
            channel_id=channel_id,
            timezone="America/New_York",
            message_repo=message_repo,
        )

        # Verify it was stored
        assert message_id is not None

        # Retrieve and verify content
        messages = await memory_orchestrator.episodic.get_recent_messages(
            channel_id=channel_id,
            limit=1,
            repo=message_repo,
        )

        assert len(messages) == 1
        assert messages[0].content == content
        assert messages[0].discord_message_id == discord_id
        assert messages[0].channel_id == channel_id
        assert messages[0].is_bot is False

    @pytest.mark.asyncio
    async def test_store_bot_message_creates_record(
        self,
        message_repo: PostgresMessageRepository,
        unique_discord_id,
    ) -> None:
        """Test that storing a bot message creates a database record."""
        content = "Bot response to user"
        discord_id = unique_discord_id.next_id()
        channel_id = unique_discord_id.next_id()  # Use unique channel ID
        metadata = {
            "model": "anthropic/claude-3.5-sonnet",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

        # Store message
        message_id = await memory_orchestrator.store_bot_message(
            content=content,
            discord_message_id=discord_id,
            channel_id=channel_id,
            generation_metadata=metadata,
            timezone="UTC",
            message_repo=message_repo,
        )

        # Verify it was stored
        assert message_id is not None

        # Retrieve and verify content
        messages = await memory_orchestrator.episodic.get_recent_messages(
            channel_id=channel_id,
            limit=1,
            repo=message_repo,
        )

        assert len(messages) == 1
        assert messages[0].content == content
        assert messages[0].is_bot is True
        # Note: generation_metadata is not returned by get_recent_messages()

    @pytest.mark.asyncio
    async def test_context_strategy_extracts_entities(
        self,
        context_cache: ChannelContextCache,
        canonical_repo: PostgresCanonicalRepository,
        prompt_repo: PostgresPromptRegistryRepository,
        audit_repo: PostgresPromptAuditRepository,
        feedback_repo: PostgresUserFeedbackRepository,
        unique_discord_id,
    ) -> None:
        """Test context strategy extracts entities from user message."""
        message_id = uuid.uuid4()
        message_content = "I'm working on the lattice project"
        channel_id = 789012
        discord_id = unique_discord_id.next_id()

        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_get_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list", return_value=[]
            ),
        ):
            from lattice.memory.procedural import PromptTemplate

            # Mock prompt template
            template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Extract: {user_message}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = template

            # Mock LLM client
            mock_llm = AsyncMock()
            mock_llm.complete.return_value = AuditResult(
                content='{"entities": ["lattice project"], "context_flags": [], "unresolved_entities": ["lattice project"]}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                cost_usd=0.001,
                latency_ms=300,
                temperature=0.2,
                audit_id=None,
                prompt_key="CONTEXT_STRATEGY",
            )

            # Run context strategy
            result = await context_strategy(
                message_id=message_id,
                user_message=message_content,
                recent_messages=[],
                context_cache=context_cache,
                channel_id=channel_id,
                discord_message_id=discord_id,
                llm_client=mock_llm,
                canonical_repo=canonical_repo,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
            )

            # Verify extraction
            assert result is not None
            assert "lattice project" in result.entities
            assert len(result.context_flags) == 0

    @pytest.mark.asyncio
    async def test_retrieve_context_returns_semantic_data(
        self,
        semantic_repo: PostgresSemanticMemoryRepository,
    ) -> None:
        """Test context retrieval returns properly formatted semantic data."""
        # Retrieve context with no entities (should return empty context)
        context = await retrieve_context(
            entities=[],
            context_flags=[],
            semantic_repo=semantic_repo,
        )

        # Verify structure
        assert isinstance(context, dict)
        assert "semantic_context" in context
        assert "goal_context" in context
        assert context["semantic_context"] == "No relevant context found."

    @pytest.mark.asyncio
    async def test_generate_response_returns_content(
        self,
        prompt_repo: PostgresPromptRegistryRepository,
        audit_repo: PostgresPromptAuditRepository,
        feedback_repo: PostgresUserFeedbackRepository,
    ) -> None:
        """Test response generation returns formatted content."""
        user_message = "What's the status?"
        episodic_context = "User: Working on project\nBot: Great progress!"
        semantic_context = "lattice project: in progress"

        with patch(
            "lattice.core.response_generator.procedural.get_prompt"
        ) as mock_get_prompt:
            from lattice.memory.procedural import PromptTemplate

            # Mock prompt template
            template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template="Context: {episodic_context}\nUser: {user_message}",
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = template

            # Mock LLM client
            mock_llm = AsyncMock()
            mock_llm.complete.return_value = AuditResult(
                content="Looking good! Keep up the momentum.",
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=100,
                completion_tokens=30,
                total_tokens=130,
                cost_usd=0.002,
                latency_ms=400,
                temperature=0.7,
                audit_id=None,
                prompt_key="UNIFIED_RESPONSE",
            )

            # Generate response
            (
                result,
                rendered_prompt,
                context_info,
            ) = await response_generator.generate_response(
                user_message=user_message,
                episodic_context=episodic_context,
                semantic_context=semantic_context,
                llm_client=mock_llm,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
            )

            # Verify response
            assert result is not None
            assert result.content == "Looking good! Keep up the momentum."
            assert result.model == "anthropic/claude-3.5-sonnet"
            assert context_info["template"] == "UNIFIED_RESPONSE"

    @pytest.mark.asyncio
    async def test_full_pipeline_end_to_end(
        self,
        message_repo: PostgresMessageRepository,
        semantic_repo: PostgresSemanticMemoryRepository,
        canonical_repo: PostgresCanonicalRepository,
        prompt_repo: PostgresPromptRegistryRepository,
        audit_repo: PostgresPromptAuditRepository,
        feedback_repo: PostgresUserFeedbackRepository,
        context_cache: ChannelContextCache,
        unique_discord_id,
    ) -> None:
        """Test complete end-to-end message flow through all pipeline stages.

        This test covers:
        1. User message ingestion
        2. Context strategy (entity extraction)
        3. Context retrieval (semantic memory)
        4. Response generation
        5. Bot message storage
        """
        user_content = "I need to finish the report by Friday"
        channel_id = 456789
        discord_id = unique_discord_id.next_id()

        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_strategy_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list", return_value=[]
            ),
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_response_prompt,
        ):
            from lattice.memory.procedural import PromptTemplate

            # Mock strategy prompt
            strategy_template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Extract: {user_message}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_strategy_prompt.return_value = strategy_template

            # Mock response prompt
            response_template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template="User: {user_message}\nContext: {episodic_context}",
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_response_prompt.return_value = response_template

            # Mock LLM clients
            strategy_llm = AsyncMock()
            strategy_llm.complete.return_value = AuditResult(
                content='{"entities": ["report", "Friday"], "context_flags": [], "unresolved_entities": ["report", "Friday"]}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=60,
                completion_tokens=25,
                total_tokens=85,
                cost_usd=0.001,
                latency_ms=350,
                temperature=0.2,
                audit_id=None,
                prompt_key="CONTEXT_STRATEGY",
            )

            response_llm = AsyncMock()
            response_llm.complete.return_value = AuditResult(
                content="Got it! Friday deadline for the report. I'll help you stay on track.",
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=120,
                completion_tokens=35,
                total_tokens=155,
                cost_usd=0.002,
                latency_ms=450,
                temperature=0.7,
                audit_id=None,
                prompt_key="UNIFIED_RESPONSE",
            )

            # 1. Ingest user message
            message_id = await memory_orchestrator.store_user_message(
                content=user_content,
                discord_message_id=discord_id,
                channel_id=channel_id,
                timezone="UTC",
                message_repo=message_repo,
            )
            assert message_id is not None

            # 2. Analyze - get recent messages and run context strategy
            recent_messages = await memory_orchestrator.episodic.get_recent_messages(
                channel_id=channel_id,
                limit=10,
                repo=message_repo,
            )
            history = [m for m in recent_messages if m.message_id != message_id]

            await context_cache.advance(channel_id)

            strategy = await context_strategy(
                message_id=message_id,
                user_message=user_content,
                recent_messages=history,
                context_cache=context_cache,
                channel_id=channel_id,
                discord_message_id=discord_id,
                llm_client=strategy_llm,
                canonical_repo=canonical_repo,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
            )

            # Verify entity extraction
            assert strategy is not None
            assert "report" in strategy.entities
            assert "Friday" in strategy.entities

            # 3. Retrieve context
            context = await retrieve_context(
                entities=strategy.entities,
                context_flags=strategy.context_flags,
                semantic_repo=semantic_repo,
            )

            # Verify context structure
            assert "semantic_context" in context
            assert "goal_context" in context

            # 4. Generate response
            formatted_history = memory_orchestrator.episodic.format_messages(history)
            result, _, context_info = await response_generator.generate_response(
                user_message=user_content,
                episodic_context=formatted_history,
                semantic_context=context["semantic_context"],
                llm_client=response_llm,
                prompt_repo=prompt_repo,
                audit_repo=audit_repo,
                feedback_repo=feedback_repo,
            )

            # Verify response generation
            assert result is not None
            assert "Friday deadline" in result.content
            assert result.model == "anthropic/claude-3.5-sonnet"

            # 5. Store bot message
            bot_discord_id = unique_discord_id.next_id()
            bot_message_id = await memory_orchestrator.store_bot_message(
                content=result.content,
                discord_message_id=bot_discord_id,
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
                message_repo=message_repo,
            )

            assert bot_message_id is not None

            # Verify both messages are in episodic memory
            all_messages = await memory_orchestrator.episodic.get_recent_messages(
                channel_id=channel_id,
                limit=10,
                repo=message_repo,
            )

            assert len(all_messages) >= 2
            user_msg = next(
                (m for m in all_messages if m.message_id == message_id), None
            )
            bot_msg = next(
                (m for m in all_messages if m.message_id == bot_message_id), None
            )

            assert user_msg is not None
            assert user_msg.content == user_content
            assert user_msg.is_bot is False

            assert bot_msg is not None
            assert bot_msg.content == result.content
            assert bot_msg.is_bot is True

    @pytest.mark.asyncio
    async def test_pipeline_process_message_integration(
        self,
        message_repo: PostgresMessageRepository,
        semantic_repo: PostgresSemanticMemoryRepository,
        canonical_repo: PostgresCanonicalRepository,
        prompt_repo: PostgresPromptRegistryRepository,
        audit_repo: PostgresPromptAuditRepository,
        feedback_repo: PostgresUserFeedbackRepository,
        context_cache: ChannelContextCache,
        unique_discord_id,
    ) -> None:
        """Test UnifiedPipeline.process_message() orchestrates full flow.

        This test verifies that the UnifiedPipeline class correctly
        orchestrates all pipeline stages in the correct order.
        """
        user_content = "What's my schedule for tomorrow?"
        channel_id = 999888
        discord_id = unique_discord_id.next_id()

        # Mock Discord bot
        mock_bot = MagicMock()
        mock_channel = MagicMock()
        mock_sent_message = MagicMock()
        mock_sent_message.id = unique_discord_id.next_id()
        mock_channel.send = AsyncMock(return_value=mock_sent_message)
        mock_bot.get_channel = MagicMock(return_value=mock_channel)

        # Create pipeline
        pipeline = UnifiedPipeline(
            bot=mock_bot,
            context_cache=context_cache,
            message_repo=message_repo,
            semantic_repo=semantic_repo,
            canonical_repo=canonical_repo,
            prompt_repo=prompt_repo,
            audit_repo=audit_repo,
            feedback_repo=feedback_repo,
        )

        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_strategy_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list", return_value=[]
            ),
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_response_prompt,
        ):
            from lattice.memory.procedural import PromptTemplate

            # Mock prompts
            strategy_template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Extract: {user_message}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_strategy_prompt.return_value = strategy_template

            response_template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template="User: {user_message}",
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_response_prompt.return_value = response_template

            # Mock LLM client
            mock_llm = AsyncMock()
            mock_llm.complete = AsyncMock(
                side_effect=[
                    # Context strategy response
                    AuditResult(
                        content='{"entities": ["schedule", "tomorrow"], "context_flags": [], "unresolved_entities": ["schedule"]}',
                        model="anthropic/claude-3.5-sonnet",
                        provider="anthropic",
                        prompt_tokens=50,
                        completion_tokens=20,
                        total_tokens=70,
                        cost_usd=0.001,
                        latency_ms=300,
                        temperature=0.2,
                        audit_id=None,
                        prompt_key="CONTEXT_STRATEGY",
                    ),
                    # Response generation response
                    AuditResult(
                        content="Let me check your schedule for tomorrow.",
                        model="anthropic/claude-3.5-sonnet",
                        provider="anthropic",
                        prompt_tokens=100,
                        completion_tokens=25,
                        total_tokens=125,
                        cost_usd=0.002,
                        latency_ms=400,
                        temperature=0.7,
                        audit_id=None,
                        prompt_key="UNIFIED_RESPONSE",
                    ),
                ]
            )
            pipeline.llm_client = mock_llm

            # Process message through pipeline
            result = await pipeline.process_message(
                content=user_content,
                discord_message_id=discord_id,
                channel_id=channel_id,
                timezone="America/Los_Angeles",
            )

            # Verify pipeline sent response
            assert result is not None
            assert result == mock_sent_message
            mock_channel.send.assert_called_once()

            # Verify messages were stored
            messages = await memory_orchestrator.episodic.get_recent_messages(
                channel_id=channel_id,
                limit=10,
                repo=message_repo,
            )

            # Should have both user and bot messages
            assert len(messages) >= 2
            user_msgs = [m for m in messages if not m.is_bot]
            bot_msgs = [m for m in messages if m.is_bot]

            assert len(user_msgs) >= 1
            assert user_msgs[0].content == user_content

            assert len(bot_msgs) >= 1
            assert bot_msgs[0].content == "Let me check your schedule for tomorrow."
