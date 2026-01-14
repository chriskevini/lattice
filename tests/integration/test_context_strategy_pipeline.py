"""Integration tests for entity extraction pipeline.

Tests the full flow from message receipt to response generation with extraction.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.core import response_generator
from lattice.core.context import ChannelContextCache
from lattice.core.context_strategy import context_strategy, retrieve_context
from lattice.utils.date_resolution import get_now
from lattice.memory import episodic
from lattice.memory.context import (
    PostgresCanonicalRepository,
    PostgresMessageRepository,
    PostgresSemanticMemoryRepository,
)


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
def prompt_repo(db_pool):
    """Fixture providing a PromptRegistryRepository."""
    from lattice.memory.repositories import PostgresPromptRegistryRepository

    return PostgresPromptRegistryRepository(db_pool)


@pytest.fixture
def audit_repo(db_pool):
    """Fixture providing a PromptAuditRepository."""
    from lattice.memory.repositories import PostgresPromptAuditRepository

    return PostgresPromptAuditRepository(db_pool)


@pytest.fixture
def feedback_repo(db_pool):
    """Fixture providing a UserFeedbackRepository."""
    from lattice.memory.repositories import PostgresUserFeedbackRepository

    return PostgresUserFeedbackRepository(db_pool)


@pytest.fixture
def context_cache(db_pool) -> ChannelContextCache:
    """Create a fresh context cache for each test."""
    from lattice.memory.context import PostgresContextRepository

    repo = PostgresContextRepository(db_pool)
    cache = ChannelContextCache(repository=repo, ttl=10)
    return cache


@pytest.fixture(autouse=True)
def reset_cache(context_cache):
    """Reset context cache before each test."""
    context_cache.clear()
    yield
    context_cache.clear()


@pytest.fixture
async def db_pool():
    """Create a database pool for tests."""
    from lattice.utils.database import DatabasePool
    from lattice.utils.config import config
    from unittest import mock
    import os

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set")

    with mock.patch.object(config, "database_url", database_url):
        pool = DatabasePool()
        await pool.initialize()
        yield pool
        await pool.close()


class TestContextStrategyPipeline:
    """Integration tests for the entity extraction pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_declaration(
        self,
        db_pool,
        context_cache: ChannelContextCache,
        canonical_repo,
        prompt_repo,
        audit_repo,
        feedback_repo,
        unique_discord_id,
    ) -> None:
        """Test complete pipeline flow for a declaration message."""
        message_id = uuid.uuid4()
        message_content = "I need to finish the lattice project by Friday"
        channel_id = 67890
        discord_id = unique_discord_id.next_id()

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
            from lattice.utils.llm import AuditResult

            extraction_template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Extract: {user_message}\nContext: {context}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = extraction_template

            extraction_llm = AsyncMock()
            extraction_result = AuditResult(
                content='{"entities":["lattice project","Friday"], "context_flags":[], "unresolved_entities":[]}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.001,
                latency_ms=500,
                temperature=0.2,
                audit_id=None,
                prompt_key="CONTEXT_STRATEGY",
            )
            extraction_llm.complete.return_value = extraction_result

            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {
                "id": uuid.uuid4(),
                "message_id": message_id,
                "extraction": {
                    "entities": ["lattice project", "Friday"],
                },
                "prompt_key": "CONTEXT_STRATEGY",
                "prompt_version": 1,
                "created_at": get_now("UTC"),
            }
            mock_pool = MagicMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            with patch.object(db_pool, "_pool", mock_pool.pool):
                async with db_pool.pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO raw_messages (id, content, discord_message_id, channel_id, is_bot) VALUES ($1, $2, $3, $4, $5)",
                        message_id,
                        message_content,
                        discord_id,
                        channel_id,
                        False,
                    )
                response_template = PromptTemplate(
                    prompt_key="UNIFIED_RESPONSE",
                    template=("Context: {episodic_context}\nUser: {user_message}"),
                    temperature=0.7,
                    version=1,
                    active=True,
                )
                mock_resp_prompt.return_value = response_template

                response_llm = AsyncMock()
                response_llm.complete.return_value = AuditResult(
                    content="Got it! Friday deadline for lattice. That's coming up quick—how's it looking so far?",
                    model="anthropic/claude-3.5-sonnet",
                    provider="anthropic",
                    prompt_tokens=120,
                    completion_tokens=30,
                    total_tokens=150,
                    cost_usd=0.001,
                    latency_ms=450,
                    temperature=0.7,
                    audit_id=None,
                    prompt_key="UNIFIED_RESPONSE",
                )

                extraction = await context_strategy(
                    message_id=message_id,
                    user_message=message_content,
                    recent_messages=[],
                    context_cache=context_cache,
                    channel_id=channel_id,
                    discord_message_id=discord_id,
                    llm_client=extraction_llm,
                    canonical_repo=canonical_repo,
                    prompt_repo=prompt_repo,
                    audit_repo=audit_repo,
                    feedback_repo=feedback_repo,
                )

                assert extraction is not None
                assert "lattice project" in extraction.entities

                (
                    result,
                    rendered_prompt,
                    context_info,
                ) = await response_generator.generate_response(
                    user_message=message_content,
                    episodic_context="Recent conversation history",
                    semantic_context="Relevant facts",
                    llm_client=response_llm,
                    prompt_repo=prompt_repo,
                    audit_repo=audit_repo,
                    feedback_repo=feedback_repo,
                )

                assert result is not None
                assert "Friday deadline" in result.content
                assert context_info["template"] == "UNIFIED_RESPONSE"

                assert (
                    "lattice project" in rendered_prompt or "Friday" in rendered_prompt
                )

    @pytest.mark.asyncio
    async def test_context_strategy_activity_context(
        self,
        db_pool,
        context_cache: ChannelContextCache,
        canonical_repo,
        prompt_repo,
        audit_repo,
        feedback_repo,
        unique_discord_id,
    ) -> None:
        """Test context strategy detects activity queries."""
        message_id = uuid.uuid4()
        message_content = "What did I do last week?"
        channel_id = 67890
        discord_id = unique_discord_id.next_id()

        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_get_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list", return_value=[]
            ),
            patch(
                "lattice.memory.canonical.get_canonical_entities_list"
            ) as mock_canonical,
        ):
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import AuditResult

            planning_template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Test template",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = planning_template
            mock_canonical.return_value = []

            planning_llm = AsyncMock()
            planning_result = AuditResult(
                content='{"entities": [], "context_flags": ["activity_context"], "unresolved_entities": []}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=60,
                completion_tokens=30,
                total_tokens=90,
                cost_usd=0.0006,
                latency_ms=300,
                temperature=0.2,
                audit_id=None,
                prompt_key="CONTEXT_STRATEGY",
            )
            planning_llm.complete.return_value = planning_result

            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {
                "id": uuid.uuid4(),
                "message_id": message_id,
                "extraction": {
                    "entities": [],
                    "context_flags": ["activity_context"],
                    "unresolved_entities": [],
                    "_strategy_method": "api",
                },
                "rendered_prompt": "test prompt",
                "raw_response": '{"entities": [], "context_flags": ["activity_context"], "unresolved_entities": []}',
                "created_at": get_now("UTC"),
            }
            mock_pool = MagicMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            with patch.object(db_pool, "_pool", mock_pool.pool):
                async with db_pool.pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO raw_messages (id, content, discord_message_id, channel_id, is_bot) VALUES ($1, $2, $3, $4, $5)",
                        message_id,
                        message_content,
                        discord_id,
                        channel_id,
                        False,
                    )
                recent_messages: list[episodic.EpisodicMessage] = []
                planning = await context_strategy(
                    message_id=message_id,
                    user_message=message_content,
                    recent_messages=recent_messages,
                    context_cache=context_cache,
                    channel_id=channel_id,
                    discord_message_id=discord_id,
                    llm_client=planning_llm,
                    canonical_repo=canonical_repo,
                    prompt_repo=prompt_repo,
                    audit_repo=audit_repo,
                    feedback_repo=feedback_repo,
                )

            assert planning is not None
            assert len(planning.entities) == 0
            assert "activity_context" in planning.context_flags

    @pytest.mark.asyncio
    async def test_context_strategy_topic_switch(
        self,
        db_pool,
        context_cache: ChannelContextCache,
        canonical_repo,
        prompt_repo,
        audit_repo,
        feedback_repo,
        unique_discord_id,
    ) -> None:
        """Test context strategy returns empty when topic switches."""
        message_id = uuid.uuid4()
        message_content = "Actually, what's the weather like?"
        discord_id = unique_discord_id.next_id()

        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_get_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list", return_value=[]
            ),
            patch(
                "lattice.memory.canonical.get_canonical_entities_list"
            ) as mock_canonical,
        ):
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import AuditResult

            planning_template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Test template",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = planning_template
            mock_canonical.return_value = ["mobile app", "marathon"]

            planning_llm = AsyncMock()
            planning_result = AuditResult(
                content='{"entities": [], "context_flags": [], "unresolved_entities": []}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                cost_usd=0.0005,
                latency_ms=250,
                temperature=0.2,
                audit_id=None,
                prompt_key="CONTEXT_STRATEGY",
            )
            planning_llm.complete.return_value = planning_result

            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {
                "id": uuid.uuid4(),
                "message_id": message_id,
                "extraction": {
                    "entities": [],
                    "context_flags": [],
                    "unresolved_entities": [],
                    "_strategy_method": "api",
                },
                "rendered_prompt": "test prompt",
                "raw_response": '{"entities": [], "context_flags": [], "unresolved_entities": []}',
                "created_at": get_now("UTC"),
            }
            mock_pool = MagicMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()

            with patch.object(db_pool, "_pool", mock_pool.pool):
                async with db_pool.pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO raw_messages (id, content, discord_message_id, channel_id, is_bot) VALUES ($1, $2, $3, $4, $5)",
                        message_id,
                        message_content,
                        discord_id,
                        67890,
                        False,
                    )
                recent_messages = [
                    episodic.EpisodicMessage(
                        content="Working on mobile app",
                        discord_message_id=unique_discord_id.next_id(),
                        channel_id=456,
                        is_bot=False,
                        message_id=uuid.uuid4(),
                        timestamp=datetime(2026, 1, 8, 10, 0, tzinfo=UTC),
                    ),
                    episodic.EpisodicMessage(
                        content="Any blockers?",
                        discord_message_id=unique_discord_id.next_id(),
                        channel_id=456,
                        is_bot=True,
                        message_id=uuid.uuid4(),
                        timestamp=datetime(2026, 1, 8, 10, 1, tzinfo=UTC),
                    ),
                ]

                planning = await context_strategy(
                    message_id=message_id,
                    user_message=message_content,
                    recent_messages=recent_messages,
                    context_cache=context_cache,
                    channel_id=67890,
                    llm_client=planning_llm,
                    canonical_repo=canonical_repo,
                    prompt_repo=prompt_repo,
                    audit_repo=audit_repo,
                    feedback_repo=feedback_repo,
                )

            assert planning is not None
            assert len(planning.entities) == 0
            assert len(planning.context_flags) == 0
            assert len(planning.unresolved_entities) == 0

    @pytest.mark.asyncio
    async def test_context_strategy_missing_template(
        self,
        context_cache: ChannelContextCache,
        canonical_repo,
        prompt_repo,
    ) -> None:
        """Test context strategy fails gracefully when template missing."""
        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_get_prompt,
        ):
            mock_get_prompt.return_value = None

            with pytest.raises(ValueError, match="CONTEXT_STRATEGY prompt template"):
                await context_strategy(
                    message_id=uuid.uuid4(),
                    user_message="Test message",
                    recent_messages=[],
                    context_cache=context_cache,
                    channel_id=123,
                    discord_message_id=123,
                    canonical_repo=canonical_repo,
                    prompt_repo=prompt_repo,
                )

    @pytest.mark.asyncio
    async def test_context_strategy_missing_fields(
        self,
        context_cache: ChannelContextCache,
        canonical_repo,
        prompt_repo,
    ) -> None:
        """Test context strategy handles missing fields gracefully."""
        message_id = uuid.uuid4()
        channel_id = 67890

        with (
            patch("lattice.core.context_strategy.get_prompt") as mock_get_prompt,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list"
            ) as mock_canonical,
        ):
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import AuditResult

            planning_template = PromptTemplate(
                prompt_key="CONTEXT_STRATEGY",
                template="Test template",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = planning_template
            mock_canonical.return_value = []

            planning_llm = AsyncMock()
            # Only entities provided, flags and unresolved missing
            planning_result = AuditResult(
                content='{"entities": ["test"]}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                cost_usd=0.0005,
                latency_ms=250,
                temperature=0.2,
                audit_id=None,
                prompt_key="CONTEXT_STRATEGY",
            )
            planning_llm.complete.return_value = planning_result

            planning = await context_strategy(
                message_id=message_id,
                user_message="Test message",
                recent_messages=[],
                context_cache=context_cache,
                channel_id=channel_id,
                discord_message_id=channel_id,
                llm_client=planning_llm,
                canonical_repo=canonical_repo,
                prompt_repo=prompt_repo,
            )

            assert planning is not None
            assert planning.entities == ["test"]
            assert planning.context_flags == []
            assert planning.unresolved_entities == ["test"]


class TestRetrieveContext:
    """Integration tests for the retrieve_context() function.

    Tests Phase 4 context retrieval using flags from CONTEXT_STRATEGY.
    """

    @pytest.mark.asyncio
    async def test_retrieve_context_returns_structure(self, semantic_repo) -> None:
        """Test retrieve_context returns expected dict structure."""
        context = await retrieve_context(
            entities=[],
            context_flags=[],
            memory_depth=2,
            semantic_repo=semantic_repo,
        )

        assert isinstance(context, dict)
        assert "semantic_context" in context
        assert "goal_context" in context

    @pytest.mark.asyncio
    async def test_retrieve_context_empty_inputs(self, semantic_repo) -> None:
        """Test retrieve_context with no entities and no flags."""
        context = await retrieve_context(
            entities=[],
            context_flags=[],
            memory_depth=2,
            semantic_repo=semantic_repo,
        )

        assert context["semantic_context"] == "No relevant context found."
        assert context["goal_context"] in ("", "No active goals.")

    @pytest.mark.asyncio
    async def test_retrieve_context_activity_flag(self, semantic_repo) -> None:
        """Test retrieve_context with activity_context flag."""
        context = await retrieve_context(
            entities=[],
            context_flags=["activity_context"],
            memory_depth=2,
            semantic_repo=semantic_repo,
        )

        assert "semantic_context" in context

    @pytest.mark.asyncio
    async def test_retrieve_context_goal_flag(self, semantic_repo) -> None:
        """Test retrieve_context with goal_context flag."""
        context = await retrieve_context(
            entities=[],
            context_flags=["goal_context"],
            memory_depth=0,
            semantic_repo=semantic_repo,
        )

        assert "goal_context" in context

    @pytest.mark.asyncio
    async def test_retrieve_context_multiple_flags(self, semantic_repo) -> None:
        """Test retrieve_context with multiple context flags."""
        context = await retrieve_context(
            entities=["test"],
            context_flags=["goal_context", "activity_context"],
            memory_depth=2,
            semantic_repo=semantic_repo,
        )

        assert "semantic_context" in context
        assert "goal_context" in context
