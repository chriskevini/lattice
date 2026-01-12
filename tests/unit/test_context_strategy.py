"""Unit tests for entity extraction module.

Renamed from test_query_extraction.py to match context_strategy.py module.
"""

import json
import uuid
from lattice.utils.date_resolution import get_now
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.core.context_strategy import (
    _merge_deduplicate,
    build_smaller_episodic_context,
    context_strategy,
    get_context_strategy,
    get_message_strategy,
)
from lattice.memory.episodic import EpisodicMessage
from lattice.memory.procedural import PromptTemplate
from lattice.utils.context import InMemoryContextCache
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
def context_cache() -> InMemoryContextCache:
    """Create a fresh context cache for each test."""
    cache = InMemoryContextCache(ttl=10)
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
        context_cache: InMemoryContextCache,
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
            )

            assert strategy.message_id == message_id
            assert strategy.entities == ["lattice project", "Friday"]
            assert strategy.context_flags == ["goal_context"]
            assert strategy.strategy_method == "api"

    @pytest.mark.asyncio
    async def test_context_strategy_missing_prompt_template(
        self, context_cache: InMemoryContextCache
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
                )

    @pytest.mark.asyncio
    async def test_context_strategy_invalid_json(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
        context_cache: InMemoryContextCache,
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
                )

    @pytest.mark.asyncio
    async def test_context_strategy_missing_required_fields(
        self,
        mock_prompt_template: PromptTemplate,
        context_cache: InMemoryContextCache,
    ) -> None:
        """Test context strategy with missing required fields."""
        message_id = uuid.uuid4()
        mock_result = AuditResult(
            content="{}",
            model="test",
            provider=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=None,
            latency_ms=0,
            temperature=0.0,
            audit_id=None,
            prompt_key="CONTEXT_STRATEGY",
        )

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
                return_value={"entities": []},
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_result)

            with pytest.raises(ValueError, match="Missing required field"):
                await context_strategy(
                    db_pool=mock_pool,
                    message_id=message_id,
                    user_message="Test message",
                    recent_messages=[],
                    discord_message_id=12345,
                    llm_client=mock_client,
                    context_cache=context_cache,
                )


class TestGetContextStrategy:
    """Tests for the get_context_strategy function."""

    @pytest.mark.asyncio
    async def test_get_context_strategy_success(self) -> None:
        """Test retrieving a context strategy by ID."""
        strategy_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_row = {
            "id": strategy_id,
            "message_id": message_id,
            "strategy": {
                "entities": ["Alice", "yesterday"],
                "context_flags": ["goal_context"],
                "unresolved_entities": [],
            },
            "rendered_prompt": "test prompt",
            "raw_response": "test response",
            "created_at": get_now(timezone_str="UTC"),
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool = _create_mock_pool(mock_conn)

        strategy = await get_context_strategy(
            db_pool=mock_pool, strategy_id=strategy_id
        )

        assert strategy is not None
        assert strategy.id == strategy_id
        assert strategy.entities == ["Alice", "yesterday"]
        assert strategy.context_flags == ["goal_context"]

    @pytest.mark.asyncio
    async def test_get_message_strategy_success(self) -> None:
        """Test retrieving a strategy by message ID."""
        strategy_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_row = {
            "id": strategy_id,
            "message_id": message_id,
            "strategy": {
                "entities": ["marathon"],
                "context_flags": [],
                "unresolved_entities": [],
            },
            "rendered_prompt": "test prompt",
            "raw_response": "test response",
            "created_at": get_now(timezone_str="UTC"),
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool = _create_mock_pool(mock_conn)

        strategy = await get_message_strategy(db_pool=mock_pool, message_id=message_id)

        assert strategy is not None
        assert strategy.message_id == message_id
        assert strategy.entities == ["marathon"]


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


class TestMergeDeduplicate:
    """Tests for the _merge_deduplicate function."""

    def test_merge_deduplicate_empty(self) -> None:
        """Test merging empty lists."""
        result = _merge_deduplicate([], [])
        assert result == []

    def test_merge_deduplicate_cached_only(self) -> None:
        """Test with only cached items."""
        result = _merge_deduplicate(["a", "b", "c"], [])
        assert result == ["a", "b", "c"]

    def test_merge_deduplicate_fresh_only(self) -> None:
        """Test with only fresh items."""
        result = _merge_deduplicate([], ["x", "y"])
        assert result == ["x", "y"]

    def test_merge_deduplicate_no_overlap(self) -> None:
        """Test merging with no overlap."""
        result = _merge_deduplicate(["a", "b"], ["x", "y"])
        assert result == ["a", "b", "x", "y"]

    def test_merge_deduplicate_with_overlap(self) -> None:
        """Test merging with overlapping items."""
        result = _merge_deduplicate(["a", "b", "c"], ["b", "d"])
        assert result == ["a", "b", "c", "d"]

    def test_merge_deduplicate_order_preserved(self) -> None:
        """Test that order is preserved with cached first."""
        result = _merge_deduplicate(["z", "a", "m"], ["a", "z", "b"])
        assert result == ["z", "a", "m", "b"]


class TestInMemoryContextCache:
    """Tests for in-memory context cache integration via context_strategy."""

    @pytest.mark.asyncio
    async def test_context_strategy_with_cache(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
        context_cache: InMemoryContextCache,
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
                )

                assert "project" in strategy_2.entities
                assert "Friday" in strategy_2.entities
                assert "weekend" in strategy_2.entities


class TestInMemoryContextCacheDirect:
    """Unit tests for InMemoryContextCache TTL and basic operations."""

    def test_cache_ttl_expiration(self) -> None:
        """Test that entries expire after TTL advances."""
        cache = InMemoryContextCache(ttl=2)

        cache.advance()
        cache.add(1, ["entity1"], ["flag1"], [])

        entities, flags, unresolved = cache.get_active(1)
        assert entities == ["entity1"]
        assert flags == ["flag1"]

        cache.advance()
        entities, flags, unresolved = cache.get_active(1)
        assert entities == ["entity1"]
        assert flags == ["flag1"]

        cache.advance()
        cache.advance()
        entities, flags, unresolved = cache.get_active(1)
        assert entities == []
        assert flags == []

    def test_cache_prune_expired(self) -> None:
        """Test that prune_expired removes expired entries."""
        cache = InMemoryContextCache(ttl=2)

        cache.advance()
        cache.add(1, ["entity1"], [], [])
        cache.advance()
        cache.add(2, ["entity2"], [], [])

        assert cache.get_active(1)[0] == ["entity1"]
        assert cache.get_active(2)[0] == ["entity2"]

        cache.advance()
        cache.advance()
        cache.advance()
        cache.prune_expired()

        assert cache.get_active(1) == ([], [], [])
        assert cache.get_active(2) == ([], [], [])

    def test_cache_clear(self) -> None:
        """Test that clear resets cache and counter."""
        cache = InMemoryContextCache(ttl=10)

        cache.advance()
        cache.advance()
        cache.add(1, ["entity"], [], [])

        cache.clear()

        assert cache.get_active(1) == ([], [], [])
        assert cache.get_stats()["message_counter"] == 0

    def test_cache_get_entities_convenience(self) -> None:
        """Test the get_entities convenience method."""
        cache = InMemoryContextCache(ttl=10)

        cache.advance()
        cache.add(1, ["entity1", "entity2"], [], [])

        entities = cache.get_entities(1)
        assert entities == ["entity1", "entity2"]
