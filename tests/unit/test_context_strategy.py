"""Unit tests for entity extraction module.

Renamed from test_query_extraction.py to match context_strategy.py module.
"""

import json
import uuid
from lattice.utils.date_resolution import get_now
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.core.context import ContextCache, ContextStrategy, UserContextCache
from lattice.core.context_strategy import (
    build_smaller_episodic_context,
    context_strategy,
)
from lattice.memory.episodic import EpisodicMessage
from lattice.memory.procedural import PromptTemplate
from lattice.utils.llm import AuditResult


@pytest.fixture
def mock_prompt_template() -> PromptTemplate:
    """Create a mock CONTEXT_STRATEGY prompt template."""
    return PromptTemplate(
        prompt_key="CONTEXT_STRATEGY",
        template="Extract entities from: {user_message}\nContext: {context}",
        temperature=0.2,
        version=3,
        active=True,
    )


@pytest.fixture
def context_cache() -> ContextCache:
    """Create a fresh context cache for each test."""
    cache = ContextCache(ttl=10)
    return cache


@pytest.fixture
def mock_llm_response() -> str:
    """Create a mock LLM response with valid extraction JSON."""
    return json.dumps(
        {
            "entities": ["lattice project", "Friday"],
            "context_flags": [],
        }
    )


@pytest.fixture
def mock_generation_result(mock_llm_response: str) -> AuditResult:
    """Create a mock AuditResult."""
    return AuditResult(
        content=mock_llm_response,
        model="anthropic/claude-3.5-sonnet",
        provider="anthropic",
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        cost_usd=0.0005,
        latency_ms=300,
        temperature=0.2,
        audit_id=None,
        prompt_key=None,
    )


def _create_mock_pool(mock_conn: AsyncMock) -> MagicMock:
    """Create a mock database pool with the given connection."""
    mock_pool = MagicMock()
    mock_pool.pool = mock_pool
    mock_acquire_cm = MagicMock()
    mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire_cm.__aexit__ = AsyncMock()
    mock_pool.pool.acquire = MagicMock(return_value=mock_acquire_cm)
    return mock_pool


class TestContextStrategyFunction:
    """Tests for the context_strategy function."""

    @pytest.mark.asyncio
    async def test_context_strategy_success(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
        context_cache: ContextCache,
    ) -> None:
        """Test successful context strategy."""
        message_id = uuid.uuid4()
        extraction_data = {
            "entities": ["lattice project", "Friday"],
            "context_flags": ["goal_context"],
        }
        mock_generation_result.content = json.dumps(extraction_data)

        mock_conn = AsyncMock()
        mock_pool = _create_mock_pool(mock_conn)

        with (
            patch(
                "lattice.core.context_strategy.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.memory.canonical.get_canonical_entities_list",
                return_value=["Mother", "boyfriend"],
            ),
            patch(
                "lattice.core.context_strategy.parse_llm_json_response",
                return_value=extraction_data,
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_generation_result)

            recent_messages = [
                EpisodicMessage(
                    content="Working on mobile app",
                    discord_message_id=1,
                    channel_id=123,
                    is_bot=False,
                ),
            ]

            strategy = await context_strategy(
                db_pool=mock_pool,
                message_id=message_id,
                user_message="I need to finish the lattice project by Friday",
                recent_messages=recent_messages,
                llm_client=mock_client,
                context_cache=context_cache,
                channel_id=123,
            )

            assert strategy.entities == ["lattice project", "Friday"]
            assert strategy.context_flags == ["goal_context"]

    @pytest.mark.asyncio
    async def test_context_strategy_missing_prompt_template(
        self, context_cache: ContextCache
    ) -> None:
        """Test context strategy with missing prompt template."""
        message_id = uuid.uuid4()

        mock_conn = AsyncMock()
        mock_pool = _create_mock_pool(mock_conn)

        with patch(
            "lattice.core.context_strategy.get_prompt",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="CONTEXT_STRATEGY prompt template"):
                await context_strategy(
                    db_pool=mock_pool,
                    message_id=message_id,
                    user_message="Test message",
                    recent_messages=[],
                    discord_message_id=12345,
                    context_cache=context_cache,
                    llm_client=AsyncMock(),
                    channel_id=123,
                )

    @pytest.mark.asyncio
    async def test_context_strategy_invalid_json(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
        context_cache: ContextCache,
    ) -> None:
        """Test context strategy with invalid JSON response."""
        message_id = uuid.uuid4()

        from lattice.utils.json_parser import JSONParseError

        mock_conn = AsyncMock()
        mock_pool = _create_mock_pool(mock_conn)

        with (
            patch(
                "lattice.core.context_strategy.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.memory.canonical.get_canonical_entities_list",
                return_value=[],
            ),
            patch(
                "lattice.core.context_strategy.parse_llm_json_response",
            ) as mock_parse,
            patch(
                "lattice.core.context_strategy.notify_parse_error_to_dream",
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_generation_result)

            parse_error = JSONParseError(
                raw_content="not json",
                parse_error=json.JSONDecodeError("Expecting value", "", 0),
            )
            mock_parse.side_effect = parse_error

            with pytest.raises(JSONParseError):
                await context_strategy(
                    db_pool=mock_pool,
                    message_id=message_id,
                    user_message="Test message",
                    recent_messages=[],
                    discord_message_id=12345,
                    llm_client=mock_client,
                    context_cache=context_cache,
                    channel_id=123,
                )


class TestBuildSmallerEpisodicContext:
    """Tests for the build_smaller_episodic_context function."""

    def test_build_smaller_episodic_context_basic(self) -> None:
        """Test building context with recent messages."""
        recent_messages = [
            EpisodicMessage(
                content="Working on mobile app",
                discord_message_id=1,
                channel_id=123,
                is_bot=False,
            ),
            EpisodicMessage(
                content="How's it coming?",
                discord_message_id=2,
                channel_id=123,
                is_bot=True,
            ),
        ]
        current_message = "Pretty good"

        context = build_smaller_episodic_context(
            recent_messages=recent_messages,
            current_message=current_message,
            window_size=3,
        )

        assert "USER: Working on mobile app" in context
        assert "ASSISTANT: How's it coming?" in context
        assert "USER: Pretty good" in context

    def test_build_smaller_episodic_context_empty_history(self) -> None:
        """Test building context with no recent messages."""
        current_message = "Hello there"

        context = build_smaller_episodic_context(
            recent_messages=[],
            current_message=current_message,
            window_size=5,
        )

        assert "USER: Hello there" in context


class TestContextCacheIntegration:
    """Tests for in-memory context cache integration via context_strategy."""

    @pytest.mark.asyncio
    async def test_context_strategy_with_cache(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
        context_cache: ContextCache,
    ) -> None:
        """Test that context strategy merges with cached context."""
        message_id = uuid.uuid4()
        extraction_data = {
            "entities": ["project", "Friday"],
            "context_flags": ["goal_context"],
        }
        mock_generation_result.content = json.dumps(extraction_data)

        mock_conn = AsyncMock()
        mock_pool = _create_mock_pool(mock_conn)

        with (
            patch(
                "lattice.core.context_strategy.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.memory.canonical.get_canonical_entities_list",
                return_value=[],
            ),
            patch(
                "lattice.core.context_strategy.parse_llm_json_response",
                return_value=extraction_data,
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_generation_result)

            recent_messages = [
                EpisodicMessage(
                    content="Working on something",
                    discord_message_id=1,
                    channel_id=123,
                    is_bot=False,
                    timestamp=get_now("UTC"),
                ),
            ]

            strategy = await context_strategy(
                db_pool=mock_pool,
                message_id=message_id,
                user_message="I need to finish by Friday",
                recent_messages=recent_messages,
                llm_client=mock_client,
                context_cache=context_cache,
                channel_id=123,
            )

            assert strategy.entities == ["project", "Friday"]
            assert strategy.context_flags == ["goal_context"]

            second_extraction = {
                "entities": ["weekend"],
                "context_flags": [],
            }
            mock_generation_result.content = json.dumps(second_extraction)

            with patch(
                "lattice.core.context_strategy.parse_llm_json_response",
                return_value=second_extraction,
            ):
                message_id_2 = uuid.uuid4()
                strategy_2 = await context_strategy(
                    db_pool=mock_pool,
                    message_id=message_id_2,
                    user_message="Planning for weekend",
                    recent_messages=recent_messages,
                    llm_client=mock_client,
                    context_cache=context_cache,
                    channel_id=123,
                )

                assert "project" in strategy_2.entities
                assert "Friday" in strategy_2.entities
                assert "weekend" in strategy_2.entities


class TestContextCacheDirect:
    """Unit tests for ContextCache TTL and basic operations."""

    def test_cache_ttl_expiration(self) -> None:
        """Test that entries expire after TTL advances."""
        cache = ContextCache(ttl=2)

        cache.advance(1)
        cache.update(1, ContextStrategy(entities=["entity1"], context_flags=["flag1"]))

        strategy = cache.get_active(1)
        assert strategy.entities == ["entity1"]
        assert strategy.context_flags == ["flag1"]

        cache.advance(1)
        strategy = cache.get_active(1)
        assert strategy.entities == ["entity1"]
        assert strategy.context_flags == ["flag1"]

        cache.advance(1)
        cache.advance(1)
        strategy = cache.get_active(1)
        assert strategy.entities == []
        assert strategy.context_flags == []

    def test_cache_clear(self) -> None:
        """Test that clear resets cache and counter."""
        cache = ContextCache(ttl=10)

        cache.advance(1)
        cache.advance(1)
        cache.update(1, ContextStrategy(entities=["entity"]))

        cache.clear()

        assert cache.get_active(1).entities == []
        assert cache.get_stats()["cached_channels"] == 0


class TestUserContextCache:
    """Unit tests for UserContextCache TTL and basic operations."""

    def test_get_set_goals(self) -> None:
        """Test basic goals get/set operations."""
        cache = UserContextCache(ttl_minutes=30)

        assert cache.get_goals("user") is None

        cache.set_goals("user", "Finish project by Friday")
        assert cache.get_goals("user") == "Finish project by Friday"

    def test_get_set_activities(self) -> None:
        """Test basic activities get/set operations."""
        cache = UserContextCache(ttl_minutes=30)

        assert cache.get_activities("user") is None

        cache.set_activities("user", "Working on mobile app")
        assert cache.get_activities("user") == "Working on mobile app"

    def test_missing_user_returns_none(self) -> None:
        """Test that missing users return None for both goals and activities."""
        cache = UserContextCache(ttl_minutes=30)

        assert cache.get_goals("unknown_user") is None
        assert cache.get_activities("unknown_user") is None

    def test_goals_ttl_expiration(self) -> None:
        """Test that goals expire after TTL."""
        cache = UserContextCache(ttl_minutes=1)

        cache.set_goals("user", "Test goals")
        assert cache.get_goals("user") == "Test goals"

        import time

        time.sleep(0.1)

        assert cache.get_goals("user") == "Test goals"

    def test_activities_ttl_expiration(self) -> None:
        """Test that activities expire after TTL."""
        cache = UserContextCache(ttl_minutes=1)

        cache.set_activities("user", "Test activities")
        assert cache.get_activities("user") == "Test activities"

        import time

        time.sleep(0.1)

        assert cache.get_activities("user") == "Test activities"

    def test_cache_clear(self) -> None:
        """Test that clear removes all cached data."""
        cache = UserContextCache(ttl_minutes=30)

        cache.set_goals("user1", "Goals 1")
        cache.set_goals("user2", "Goals 2")
        cache.set_activities("user1", "Activities 1")

        assert cache.get_goals("user1") == "Goals 1"
        assert cache.get_goals("user2") == "Goals 2"
        assert cache.get_activities("user1") == "Activities 1"

        cache.clear()

        assert cache.get_goals("user1") is None
        assert cache.get_goals("user2") is None
        assert cache.get_activities("user1") is None
        assert cache.get_stats()["cached_users"] == 0
        assert cache.get_stats()["cached_goals"] == 0
        assert cache.get_stats()["cached_activities"] == 0

    def test_get_stats(self) -> None:
        """Test cache statistics."""
        cache = UserContextCache(ttl_minutes=30)

        assert cache.get_stats() == {
            "cached_users": 0,
            "cached_goals": 0,
            "cached_activities": 0,
        }

        cache.set_goals("user1", "Goals 1")
        cache.set_activities("user1", "Activities 1")

        assert cache.get_stats() == {
            "cached_users": 1,
            "cached_goals": 1,
            "cached_activities": 1,
        }

        cache.set_goals("user2", "Goals 2")

        assert cache.get_stats() == {
            "cached_users": 2,
            "cached_goals": 2,
            "cached_activities": 1,
        }

    def test_concurrent_users(self) -> None:
        """Test cache handles multiple users independently."""
        cache = UserContextCache(ttl_minutes=30)

        cache.set_goals("user1", "User1 goals")
        cache.set_goals("user2", "User2 goals")
        cache.set_activities("user1", "User1 activities")

        assert cache.get_goals("user1") == "User1 goals"
        assert cache.get_goals("user2") == "User2 goals"
        assert cache.get_activities("user1") == "User1 activities"
        assert cache.get_activities("user2") is None

    def test_update_goals_replaces_existing(self) -> None:
        """Test that setting goals replaces previous value."""
        cache = UserContextCache(ttl_minutes=30)

        cache.set_goals("user", "Original goals")
        assert cache.get_goals("user") == "Original goals"

        cache.set_goals("user", "Updated goals")
        assert cache.get_goals("user") == "Updated goals"

        assert cache.get_stats()["cached_goals"] == 1
