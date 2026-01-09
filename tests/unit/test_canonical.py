"""Unit tests for canonical entity and predicate registry."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.memory import canonical as canonical_module


class TestCacheManagement:
    """Tests for cache management functionality."""

    def setup_method(self) -> None:
        """Save original cache state."""
        self._original_entities = canonical_module._entities_cache
        self._original_predicates = canonical_module._predicates_cache
        self._original_timestamp = canonical_module._cache_timestamp

    def teardown_method(self) -> None:
        """Restore original cache state."""
        canonical_module._entities_cache = self._original_entities
        canonical_module._predicates_cache = self._original_predicates
        canonical_module._cache_timestamp = self._original_timestamp

    @pytest.mark.asyncio
    async def test_invalidate_cache_clears_state(self) -> None:
        """Test that invalidate_cache clears all cache state."""
        canonical_module._entities_cache = {"Test": ({"test"}, None)}
        canonical_module._predicates_cache = {"test_pred"}
        canonical_module._cache_timestamp = datetime.now(UTC)

        await canonical_module.invalidate_cache()

        assert canonical_module._entities_cache is None
        assert canonical_module._predicates_cache is None
        assert canonical_module._cache_timestamp is None


class TestGetCanonicalEntity:
    """Tests for get_canonical_entity function."""

    def setup_method(self) -> None:
        """Save original cache state."""
        self._original_entities = canonical_module._entities_cache
        self._original_predicates = canonical_module._predicates_cache
        self._original_timestamp = canonical_module._cache_timestamp

    def teardown_method(self) -> None:
        """Restore original cache state."""
        canonical_module._entities_cache = self._original_entities
        canonical_module._predicates_cache = self._original_predicates
        canonical_module._cache_timestamp = self._original_timestamp

    @pytest.mark.asyncio
    async def test_lookup_family_member_mom(self) -> None:
        """Test looking up 'mom' returns 'Mother'."""
        canonical_module._entities_cache = {
            "Mother": ({"mom", "mum", "mama", "ma", "mother"}, "family"),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_canonical_entity("mom")
        assert result == "Mother"

    @pytest.mark.asyncio
    async def test_lookup_family_member_bf(self) -> None:
        """Test looking up 'bf' returns 'Spouse'."""
        canonical_module._entities_cache = {
            "Spouse": (
                {"wife", "husband", "partner", "bf", "gf", "boyfriend", "girlfriend"},
                "family",
            ),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_canonical_entity("bf")
        assert result == "Spouse"

    @pytest.mark.asyncio
    async def test_lookup_case_insensitive(self) -> None:
        """Test that entity lookup is case-insensitive."""
        canonical_module._entities_cache = {
            "Mother": ({"mom", "mum", "mama", "ma", "mother"}, "family"),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_canonical_entity("MOM")
        assert result == "Mother"

        result = await canonical_module.get_canonical_entity("Mom")
        assert result == "Mother"

    @pytest.mark.asyncio
    async def test_lookup_unknown_entity_raises_error(self) -> None:
        """Test that unknown entity raises EntityNotFoundError."""
        canonical_module._entities_cache = {
            "Mother": ({"mom", "mum"}, "family"),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        with pytest.raises(canonical_module.EntityNotFoundError) as exc_info:
            await canonical_module.get_canonical_entity("unknown_entity")

        assert "unknown_entity" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_lookup_unknown_entity_with_default(self) -> None:
        """Test that unknown entity returns itself when default_on_missing=True."""
        canonical_module._entities_cache = {
            "Mother": ({"mom", "mum"}, "family"),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_canonical_entity(
            "unknown_entity", default_on_missing=True
        )
        assert result == "unknown_entity"

    @pytest.mark.asyncio
    async def test_lookup_activity_coding(self) -> None:
        """Test looking up coding-related variants."""
        canonical_module._entities_cache = {
            "Coding": (
                {"coding", "programming", "dev", "software development"},
                "activity",
            ),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        assert await canonical_module.get_canonical_entity("coding") == "Coding"
        assert await canonical_module.get_canonical_entity("programming") == "Coding"
        assert await canonical_module.get_canonical_entity("dev") == "Coding"


class TestGetCanonicalEntities:
    """Tests for get_canonical_entities function."""

    def setup_method(self) -> None:
        """Save original cache state."""
        self._original_entities = canonical_module._entities_cache
        self._original_predicates = canonical_module._predicates_cache
        self._original_timestamp = canonical_module._cache_timestamp

    def teardown_method(self) -> None:
        """Restore original cache state."""
        canonical_module._entities_cache = self._original_entities
        canonical_module._predicates_cache = self._original_predicates
        canonical_module._cache_timestamp = self._original_timestamp

    @pytest.mark.asyncio
    async def test_multiple_entities_preserves_order(self) -> None:
        """Test that multiple entity lookup preserves order."""
        canonical_module._entities_cache = {
            "Mother": ({"mom"}, "family"),
            "Father": ({"dad"}, "family"),
            "Coding": ({"coding"}, "activity"),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_canonical_entities(["mom", "dad", "coding"])
        assert result == ["Mother", "Father", "Coding"]

    @pytest.mark.asyncio
    async def test_missing_entities_dropped(self) -> None:
        """Test that unknown entities are dropped from results."""
        canonical_module._entities_cache = {
            "Mother": ({"mom"}, "family"),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_canonical_entities(
            ["mom", "unknown", "also_unknown"]
        )
        assert result == ["Mother"]

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self) -> None:
        """Test that empty input returns empty list."""
        canonical_module._entities_cache = {}
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_canonical_entities([])
        assert result == []


class TestGetCanonicalEntityWithCategory:
    """Tests for get_canonical_entity_with_category function."""

    def setup_method(self) -> None:
        """Save original cache state."""
        self._original_entities = canonical_module._entities_cache
        self._original_predicates = canonical_module._predicates_cache
        self._original_timestamp = canonical_module._cache_timestamp

    def teardown_method(self) -> None:
        """Restore original cache state."""
        canonical_module._entities_cache = self._original_entities
        canonical_module._predicates_cache = self._original_predicates
        canonical_module._cache_timestamp = self._original_timestamp

    @pytest.mark.asyncio
    async def test_lookup_returns_canonical_and_category(self) -> None:
        """Test that lookup returns both canonical form and category."""
        canonical_module._entities_cache = {
            "Mother": ({"mom", "mum"}, "family"),
            "Coding": ({"coding"}, "activity"),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        canonical, category = await canonical_module.get_canonical_entity_with_category(
            "mom"
        )
        assert canonical == "Mother"
        assert category == "family"

    @pytest.mark.asyncio
    async def test_lookup_returns_none_for_category(self) -> None:
        """Test that category can be None."""
        canonical_module._entities_cache = {
            "TestEntity": ({"test"}, None),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        canonical, category = await canonical_module.get_canonical_entity_with_category(
            "test"
        )
        assert canonical == "TestEntity"
        assert category is None


class TestGetCanonicalPredicate:
    """Tests for get_canonical_predicate function."""

    def setup_method(self) -> None:
        """Save original cache state."""
        self._original_entities = canonical_module._entities_cache
        self._original_predicates = canonical_module._predicates_cache
        self._original_timestamp = canonical_module._cache_timestamp

    def teardown_method(self) -> None:
        """Restore original cache state."""
        canonical_module._entities_cache = self._original_entities
        canonical_module._predicates_cache = self._original_predicates
        canonical_module._cache_timestamp = self._original_timestamp

    @pytest.mark.asyncio
    async def test_lookup_lives_in(self) -> None:
        """Test looking up 'lives_in' returns 'lives_in'."""
        canonical_module._entities_cache = {}
        canonical_module._predicates_cache = {"lives_in"}
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_canonical_predicate("lives_in")
        assert result == "lives_in"

    @pytest.mark.asyncio
    async def test_lookup_works_at(self) -> None:
        """Test looking up 'works_as' returns 'works_as'."""
        canonical_module._entities_cache = {}
        canonical_module._predicates_cache = {"works_as"}
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_canonical_predicate("works_as")
        assert result == "works_as"

    @pytest.mark.asyncio
    async def test_lookup_case_insensitive(self) -> None:
        """Test that predicate lookup is case-insensitive."""
        canonical_module._entities_cache = {}
        canonical_module._predicates_cache = {"lives_in"}
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_canonical_predicate("LIVES_IN")
        assert result == "lives_in"

        result = await canonical_module.get_canonical_predicate("Lives_In")
        assert result == "lives_in"

    @pytest.mark.asyncio
    async def test_lookup_unknown_predicate_raises_error(self) -> None:
        """Test that unknown predicate raises PredicateNotFoundError."""
        canonical_module._entities_cache = {}
        canonical_module._predicates_cache = {"lives_in"}
        canonical_module._cache_timestamp = datetime.now(UTC)

        with pytest.raises(canonical_module.PredicateNotFoundError) as exc_info:
            await canonical_module.get_canonical_predicate("unknown_predicate")

        assert "unknown_predicate" in str(exc_info.value)


class TestGetAllCanonicalEntities:
    """Tests for get_all_canonical_entities function."""

    def setup_method(self) -> None:
        """Save original cache state."""
        self._original_entities = canonical_module._entities_cache
        self._original_predicates = canonical_module._predicates_cache
        self._original_timestamp = canonical_module._cache_timestamp

    def teardown_method(self) -> None:
        """Restore original cache state."""
        canonical_module._entities_cache = self._original_entities
        canonical_module._predicates_cache = self._original_predicates
        canonical_module._cache_timestamp = self._original_timestamp

    @pytest.mark.asyncio
    async def test_returns_all_entities(self) -> None:
        """Test that function returns all cached entities."""
        canonical_module._entities_cache = {
            "Mother": ({"mom", "mum"}, "family"),
            "Father": ({"dad"}, "family"),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_all_canonical_entities()

        assert len(result) == 2
        assert "Mother" in result
        assert "Father" in result

    @pytest.mark.asyncio
    async def test_returns_empty_when_cache_empty(self) -> None:
        """Test that empty cache returns empty dict."""
        canonical_module._entities_cache = {}
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_all_canonical_entities()
        assert result == {}


class TestGetAllCanonicalPredicates:
    """Tests for get_all_canonical_predicates function."""

    def setup_method(self) -> None:
        """Save original cache state."""
        self._original_entities = canonical_module._entities_cache
        self._original_predicates = canonical_module._predicates_cache
        self._original_timestamp = canonical_module._cache_timestamp

    def teardown_method(self) -> None:
        """Restore original cache state."""
        canonical_module._entities_cache = self._original_entities
        canonical_module._predicates_cache = self._original_predicates
        canonical_module._cache_timestamp = self._original_timestamp

    @pytest.mark.asyncio
    async def test_returns_all_predicates(self) -> None:
        """Test that function returns all cached predicates."""
        canonical_module._entities_cache = {}
        canonical_module._predicates_cache = {"lives_in", "works_as", "has_goal"}
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_all_canonical_predicates()

        assert len(result) == 3
        assert "lives_in" in result
        assert "works_as" in result
        assert "has_goal" in result

    @pytest.mark.asyncio
    async def test_returns_empty_when_cache_empty(self) -> None:
        """Test that empty cache returns empty set."""
        canonical_module._entities_cache = {}
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.get_all_canonical_predicates()
        assert result == set()


class TestListEntitiesByCategory:
    """Tests for list_entities_by_category function."""

    def setup_method(self) -> None:
        """Save original cache state."""
        self._original_entities = canonical_module._entities_cache
        self._original_predicates = canonical_module._predicates_cache
        self._original_timestamp = canonical_module._cache_timestamp

    def teardown_method(self) -> None:
        """Restore original cache state."""
        canonical_module._entities_cache = self._original_entities
        canonical_module._predicates_cache = self._original_predicates
        canonical_module._cache_timestamp = self._original_timestamp

    @pytest.mark.asyncio
    async def test_filter_by_family_category(self) -> None:
        """Test filtering entities by family category."""
        canonical_module._entities_cache = {
            "Mother": ({"mom"}, "family"),
            "Father": ({"dad"}, "family"),
            "Coding": ({"coding"}, "activity"),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.list_entities_by_category("family")

        assert len(result) == 2
        assert "Mother" in result
        assert "Father" in result

    @pytest.mark.asyncio
    async def test_filter_by_activity_category(self) -> None:
        """Test filtering entities by activity category."""
        canonical_module._entities_cache = {
            "Coding": ({"coding"}, "activity"),
            "Running": ({"running"}, "activity"),
            "Mother": ({"mom"}, "family"),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.list_entities_by_category("activity")

        assert len(result) == 2
        assert "Coding" in result
        assert "Running" in result

    @pytest.mark.asyncio
    async def test_empty_result_for_unknown_category(self) -> None:
        """Test that unknown category returns empty list."""
        canonical_module._entities_cache = {
            "Mother": ({"mom"}, "family"),
        }
        canonical_module._predicates_cache = set()
        canonical_module._cache_timestamp = datetime.now(UTC)

        result = await canonical_module.list_entities_by_category("nonexistent")
        assert result == []


class TestSeedCanonicalEntities:
    """Tests for seed_canonical_entities function."""

    @pytest.mark.asyncio
    async def test_seed_entities_calls_db(self) -> None:
        """Test that seed_entities calls database."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_tx = MagicMock()
        mock_tx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_tx.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction.return_value = mock_tx
        mock_pool = MagicMock()
        mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )
        mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        entities: list[dict[str, str | list[str] | None]] = [
            {"canonical": "TestEntity", "variants": ["test", "t"], "category": "test"},
        ]

        with patch.object(canonical_module, "db_pool", mock_pool):
            await canonical_module.seed_canonical_entities(entities)

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_seed_entities_handles_multiple(self) -> None:
        """Test seeding multiple entities."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_tx = MagicMock()
        mock_tx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_tx.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction.return_value = mock_tx
        mock_pool = MagicMock()
        mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )
        mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        entities: list[dict[str, str | list[str] | None]] = [
            {"canonical": "Entity1", "variants": ["e1"], "category": "cat1"},
            {"canonical": "Entity2", "variants": ["e2"], "category": "cat2"},
            {"canonical": "Entity3", "variants": ["e3"], "category": None},
        ]

        with patch.object(canonical_module, "db_pool", mock_pool):
            await canonical_module.seed_canonical_entities(entities)

        assert mock_conn.execute.call_count == 3


class TestSeedCanonicalPredicates:
    """Tests for seed_canonical_predicates function."""

    @pytest.mark.asyncio
    async def test_seed_predicates_calls_db(self) -> None:
        """Test that seed_predicates calls database."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_tx = MagicMock()
        mock_tx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_tx.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction.return_value = mock_tx
        mock_pool = MagicMock()
        mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )
        mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        predicates: list[dict[str, str | list[str]]] = [
            {"canonical": "test_pred", "variants": ["test predicate"]},
        ]

        with patch.object(canonical_module, "db_pool", mock_pool):
            await canonical_module.seed_canonical_predicates(predicates)

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_seed_predicates_handles_multiple(self) -> None:
        """Test seeding multiple predicates."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_tx = MagicMock()
        mock_tx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_tx.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction.return_value = mock_tx
        mock_pool = MagicMock()
        mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )
        mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        predicates: list[dict[str, str | list[str]]] = [
            {"canonical": "pred1", "variants": ["predicate 1"]},
            {"canonical": "pred2", "variants": ["predicate 2"]},
        ]

        with patch.object(canonical_module, "db_pool", mock_pool):
            await canonical_module.seed_canonical_predicates(predicates)

        assert mock_conn.execute.call_count == 2


class TestExceptionClasses:
    """Tests for exception classes."""

    def test_entity_not_found_error_message(self) -> None:
        """Test EntityNotFoundError has correct message."""
        error = canonical_module.EntityNotFoundError("test_entity")
        assert "test_entity" in str(error)

    def test_predicate_not_found_error_message(self) -> None:
        """Test PredicateNotFoundError has correct message."""
        error = canonical_module.PredicateNotFoundError("test_predicate")
        assert "test_predicate" in str(error)

    def test_canonical_registry_error_is_base(self) -> None:
        """Test that CanonicalRegistryError is base class."""
        error = canonical_module.CanonicalRegistryError("test")
        assert isinstance(error, canonical_module.CanonicalRegistryError)
        assert not isinstance(error, canonical_module.EntityNotFoundError)
        assert not isinstance(error, canonical_module.PredicateNotFoundError)
