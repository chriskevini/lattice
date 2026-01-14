import json
import uuid
from datetime import datetime
from lattice.utils.date_resolution import get_now
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.core.context import (
    ChannelContextCache,
    ContextStrategy,
    UserContextCache,
)
from lattice.core.context_strategy import context_strategy
from lattice.memory.episodic import EpisodicMessage
from lattice.utils.llm import AuditResult
from lattice.memory.procedural import PromptTemplate
from lattice.memory.repositories import ContextRepository


@pytest.fixture
def mock_repo() -> MagicMock:
    """Create a mock context repository."""
    repo = MagicMock(spec=ContextRepository)
    repo.save_context = AsyncMock()
    repo.load_context_type = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_canonical_repo() -> AsyncMock:
    """Create a mock canonical repository."""
    repo = AsyncMock()
    repo.get_entities_list = AsyncMock(return_value=["Mother", "boyfriend"])
    repo.get_predicates_list = AsyncMock(return_value=[])
    repo.get_entities_set = AsyncMock(return_value=set())
    repo.get_predicates_set = AsyncMock(return_value=set())
    return repo


@pytest.fixture
def context_cache(mock_repo) -> ChannelContextCache:
    """Create a fresh context cache for each test."""
    cache = ChannelContextCache(repository=mock_repo, ttl=10)
    return cache


@pytest.fixture
def user_context_cache(mock_repo) -> UserContextCache:
    """Create a fresh user context cache for each test."""
    cache = UserContextCache(repository=mock_repo, ttl_minutes=30)
    return cache


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock database pool."""
    mock = MagicMock()
    mock.pool.acquire = MagicMock()
    return mock


@pytest.fixture
def mock_prompt_template() -> PromptTemplate:
    """Create a mock CONTEXT_STRATEGY template."""
    return PromptTemplate(
        prompt_key="CONTEXT_STRATEGY",
        template="Analyze: {user_message}",
        temperature=0.0,
        version=1,
        active=True,
    )


@pytest.fixture
def mock_generation_result() -> AuditResult:
    """Create a mock AuditResult."""
    return AuditResult(
        content=json.dumps({"entities": ["test"], "context_flags": []}),
        model="test-model",
        provider="test-provider",
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0,
        latency_ms=100,
        temperature=0.0,
        prompt_key="CONTEXT_STRATEGY",
    )


@pytest.fixture
def mock_llm_response() -> str:
    """Create a mock LLM response with valid extraction JSON."""
    return json.dumps(
        {
            "entities": ["lattice project", "Friday"],
            "context_flags": [],
        }
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
        context_cache: ChannelContextCache,
        mock_canonical_repo: AsyncMock,
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
                canonical_repo=mock_canonical_repo,
            )

            assert strategy.entities == ["lattice project", "Friday"]
            assert strategy.context_flags == ["goal_context"]

    @pytest.mark.asyncio
    async def test_context_strategy_missing_prompt_template(
        self, context_cache: ChannelContextCache
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
        context_cache: ChannelContextCache,
        mock_canonical_repo: AsyncMock,
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
                    canonical_repo=mock_canonical_repo,
                )


class TestChannelContextCacheIntegration:
    """Tests for in-memory context cache integration via context_strategy."""

    @pytest.mark.asyncio
    async def test_context_strategy_with_cache(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
        context_cache: ChannelContextCache,
        mock_canonical_repo: AsyncMock,
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
                canonical_repo=mock_canonical_repo,
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
                    canonical_repo=mock_canonical_repo,
                )

                assert "project" in strategy_2.entities
                assert "Friday" in strategy_2.entities
                assert "weekend" in strategy_2.entities


class TestChannelContextCacheDirect:
    """Unit tests for ChannelContextCache TTL and basic operations."""

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, context_cache) -> None:
        """Test that entries expire after TTL advances."""
        context_cache.ttl = 2

        with patch.object(ChannelContextCache, "_persist", new_callable=AsyncMock):
            await context_cache.advance(1)
            await context_cache.update(
                1,
                ContextStrategy(entities=["entity1"], context_flags=["flag1"]),
            )

            strategy = context_cache.get_active(1)
            assert strategy.entities == ["entity1"]
            assert strategy.context_flags == ["flag1"]

            await context_cache.advance(1)
            strategy = context_cache.get_active(1)
            assert strategy.entities == ["entity1"]
            assert strategy.context_flags == ["flag1"]

            await context_cache.advance(1)
            await context_cache.advance(1)
            strategy = context_cache.get_active(1)
            assert strategy.entities == []
            assert strategy.context_flags == []

    def test_cache_clear(self, context_cache) -> None:
        """Test that clear resets cache and counter."""
        context_cache.clear()
        assert context_cache.get_active(1).entities == []
        assert context_cache.get_stats()["cached_channels"] == 0


class TestChannelContextCachePersistence:
    """Tests for ChannelContextCache persistence roundtrip."""

    @pytest.mark.asyncio
    async def test_persistence_roundtrip(self, context_cache, mock_repo) -> None:
        """Test that cache data survives clear and reload."""
        context_cache.ttl = 10
        mock_repo.load_context_type.return_value = [
            {
                "target_id": "123",
                "data": json.dumps(
                    {
                        "entities": {"test_entity": 1},
                        "context_flags": {"test_flag": 1},
                        "unresolved_entities": {},
                        "message_counter": 1,
                        "created_at": "2026-01-12T00:00:00",
                    }
                ),
                "updated_at": datetime.now(),
            }
        ]

        stats_before = context_cache.get_stats()
        assert stats_before["cached_channels"] == 0

        with patch.object(ChannelContextCache, "_persist", new_callable=AsyncMock):
            await context_cache.update(
                123,
                ContextStrategy(entities=["test_entity"], context_flags=["test_flag"]),
            )
            await context_cache.advance(123)

        stats_after_update = context_cache.get_stats()
        assert stats_after_update["cached_channels"] == 1

        context_cache.clear()
        assert context_cache.get_stats()["cached_channels"] == 0

        await context_cache.load_from_db()
        stats_reloaded = context_cache.get_stats()
        assert stats_reloaded["cached_channels"] == 1

        strategy = context_cache.get_active(123)
        assert "test_entity" in strategy.entities
        assert "test_flag" in strategy.context_flags


class TestUserContextCache:
    """Unit tests for UserContextCache TTL and basic operations."""

    @pytest.mark.asyncio
    async def test_get_set_goals(self, user_context_cache) -> None:
        """Test basic goals get/set operations."""
        assert user_context_cache.get_goals("user") is None

        with patch.object(UserContextCache, "_persist", new_callable=AsyncMock):
            await user_context_cache.set_goals("user", "Finish project by Friday")
            assert user_context_cache.get_goals("user") == "Finish project by Friday"

    @pytest.mark.asyncio
    async def test_get_set_activities(self, user_context_cache) -> None:
        """Test basic activities get/set operations."""
        assert user_context_cache.get_activities("user") is None

        with patch.object(UserContextCache, "_persist", new_callable=AsyncMock):
            await user_context_cache.set_activities("user", "Working on mobile app")
            assert user_context_cache.get_activities("user") == "Working on mobile app"

    def test_missing_user_returns_none(self, user_context_cache) -> None:
        """Test that missing users return None for both goals and activities."""
        assert user_context_cache.get_goals("unknown_user") is None
        assert user_context_cache.get_activities("unknown_user") is None

    @pytest.mark.asyncio
    async def test_goals_ttl_expiration(self, user_context_cache) -> None:
        """Test that goals expire after TTL."""
        user_context_cache.ttl = 1.0 / 60.0  # 1 second (TTL is in minutes)

        with patch.object(UserContextCache, "_persist", new_callable=AsyncMock):
            await user_context_cache.set_goals("user", "Test goals")
            assert user_context_cache.get_goals("user") == "Test goals"

            import asyncio

            await asyncio.sleep(1.1)

            assert user_context_cache.get_goals("user") is None

    @pytest.mark.asyncio
    async def test_activities_ttl_expiration(self, user_context_cache) -> None:
        """Test that activities expire after TTL."""
        user_context_cache.ttl = 1.0 / 60.0  # 1 second (TTL is in minutes)

        with patch.object(UserContextCache, "_persist", new_callable=AsyncMock):
            await user_context_cache.set_activities("user", "Test activities")
            assert user_context_cache.get_activities("user") == "Test activities"

            import asyncio

            await asyncio.sleep(1.1)

            assert user_context_cache.get_activities("user") is None

    def test_cache_clear(self, user_context_cache) -> None:
        """Test that clear removes all cached data."""
        user_context_cache.clear()
        assert user_context_cache.get_stats()["cached_users"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_users(self, user_context_cache) -> None:
        """Test cache handles multiple users independently."""
        with patch.object(UserContextCache, "_persist", new_callable=AsyncMock):
            await user_context_cache.set_goals("user1", "User1 goals")
            await user_context_cache.set_goals("user2", "User2 goals")
            await user_context_cache.set_activities("user1", "User1 activities")

            assert user_context_cache.get_goals("user1") == "User1 goals"
            assert user_context_cache.get_goals("user2") == "User2 goals"
            assert user_context_cache.get_activities("user1") == "User1 activities"
            assert user_context_cache.get_activities("user2") is None

    @pytest.mark.asyncio
    async def test_get_set_timezone(self, user_context_cache) -> None:
        """Test basic timezone get/set operations."""
        assert user_context_cache.get_timezone("user") is None

        with patch.object(UserContextCache, "_persist", new_callable=AsyncMock):
            await user_context_cache.set_timezone("user", "America/New_York")
            assert user_context_cache.get_timezone("user") == "America/New_York"

    @pytest.mark.asyncio
    async def test_timezone_ttl_expiration(self, user_context_cache) -> None:
        """Test that timezone expires after TTL."""
        user_context_cache.ttl = 1.0 / 60.0  # 1 second (TTL is in minutes)

        with patch.object(UserContextCache, "_persist", new_callable=AsyncMock):
            await user_context_cache.set_timezone("user", "Europe/London")
            assert user_context_cache.get_timezone("user") == "Europe/London"

            import asyncio

            await asyncio.sleep(1.1)

            assert user_context_cache.get_timezone("user") is None

    @pytest.mark.asyncio
    async def test_timezone_concurrent_users(self, user_context_cache) -> None:
        """Test timezone cache handles multiple users independently."""
        with patch.object(UserContextCache, "_persist", new_callable=AsyncMock):
            await user_context_cache.set_timezone("user1", "UTC")
            await user_context_cache.set_timezone("user2", "America/Los_Angeles")

            assert user_context_cache.get_timezone("user1") == "UTC"
            assert user_context_cache.get_timezone("user2") == "America/Los_Angeles"
            assert user_context_cache.get_timezone("user3") is None

    @pytest.mark.asyncio
    async def test_load_from_db_with_timezone(self, user_context_cache) -> None:
        """Test that load_from_db correctly reconstructs timezone data."""
        now = datetime.now()
        user_context_cache._repository.load_context_type = AsyncMock(
            return_value=[
                {
                    "target_id": "user1",
                    "data": {
                        "timezone": ["America/New_York", now.isoformat()],
                    },
                    "updated_at": now.isoformat(),
                },
                {
                    "target_id": "user2",
                    "data": {
                        "timezone": ["Europe/London", now.isoformat()],
                    },
                    "updated_at": now.isoformat(),
                },
            ]
        )

        await user_context_cache.load_from_db()

        assert user_context_cache.get_timezone("user1") == "America/New_York"
        assert user_context_cache.get_timezone("user2") == "Europe/London"
        assert user_context_cache.get_timezone("user3") is None
