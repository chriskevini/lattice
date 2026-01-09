"""Unit tests for canonical entity and predicate registry."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.memory import canonical as canonical_module


def create_mock_pool_with_conn(mock_conn: AsyncMock) -> MagicMock:
    """Create a mock database pool with the given connection."""
    mock_inner_pool = MagicMock()

    # Set up as an async context manager for acquire
    mock_acquire = AsyncMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock(return_value=None)
    mock_inner_pool.acquire.return_value = mock_acquire

    # DatabasePool has a .pool property that returns _pool
    mock_pool = MagicMock()
    mock_pool._pool = mock_inner_pool
    mock_pool.pool = mock_inner_pool
    return mock_pool


class TestGetCanonicalEntitiesList:
    """Tests for get_canonical_entities_list function."""

    @pytest.mark.asyncio
    async def test_returns_list_of_entities(self) -> None:
        """Test that it returns a list of entity names."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {"name": "Mother"},
                {"name": "boyfriend"},
                {"name": "marathon"},
            ]
        )
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            result = await canonical_module.get_canonical_entities_list()

            assert isinstance(result, list)
            assert len(result) == 3
            assert "Mother" in result
            assert "boyfriend" in result
            assert "marathon" in result
            mock_conn.fetch.assert_called_once_with(
                "SELECT name FROM entities ORDER BY created_at DESC"
            )

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_empty(self) -> None:
        """Test that it returns empty list when no entities."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            result = await canonical_module.get_canonical_entities_list()

            assert result == []


class TestGetCanonicalPredicatesList:
    """Tests for get_canonical_predicates_list function."""

    @pytest.mark.asyncio
    async def test_returns_list_of_predicates(self) -> None:
        """Test that it returns a list of predicate names."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {"name": "has goal"},
                {"name": "due by"},
                {"name": "did activity"},
            ]
        )
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            result = await canonical_module.get_canonical_predicates_list()

            assert isinstance(result, list)
            assert len(result) == 3
            assert "has goal" in result
            assert "due by" in result
            assert "did activity" in result
            mock_conn.fetch.assert_called_once_with(
                "SELECT name FROM predicates ORDER BY created_at DESC"
            )

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_empty(self) -> None:
        """Test that it returns empty list when no predicates."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            result = await canonical_module.get_canonical_predicates_list()

            assert result == []


class TestGetCanonicalEntitiesSet:
    """Tests for get_canonical_entities_set function."""

    @pytest.mark.asyncio
    async def test_returns_set_of_entities(self) -> None:
        """Test that it returns a set for O(1) lookup."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {"name": "Mother"},
                {"name": "boyfriend"},
            ]
        )
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            result = await canonical_module.get_canonical_entities_set()

            assert isinstance(result, set)
            assert "Mother" in result
            assert "boyfriend" in result


class TestGetCanonicalPredicatesSet:
    """Tests for get_canonical_predicates_set function."""

    @pytest.mark.asyncio
    async def test_returns_set_of_predicates(self) -> None:
        """Test that it returns a set for O(1) lookup."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {"name": "has goal"},
                {"name": "due by"},
            ]
        )
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            result = await canonical_module.get_canonical_predicates_set()

            assert isinstance(result, set)
            assert "has goal" in result
            assert "due by" in result


class TestStoreCanonicalEntities:
    """Tests for store_canonical_entities function."""

    @pytest.mark.asyncio
    async def test_stores_entities(self) -> None:
        """Test that it stores entity names."""
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            count = await canonical_module.store_canonical_entities(
                ["entity1", "entity2"]
            )

            assert count == 2
            mock_conn.executemany.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_list(self) -> None:
        """Test that it returns 0 for empty input."""
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            count = await canonical_module.store_canonical_entities([])

            assert count == 0
            mock_conn.executemany.assert_not_called()


class TestStoreCanonicalPredicates:
    """Tests for store_canonical_predicates function."""

    @pytest.mark.asyncio
    async def test_stores_predicates(self) -> None:
        """Test that it stores predicate names."""
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            count = await canonical_module.store_canonical_predicates(
                ["has goal", "due by"]
            )

            assert count == 2
            mock_conn.executemany.assert_called_once()


class TestEntityExists:
    """Tests for entity_exists function."""

    @pytest.mark.asyncio
    async def test_returns_true_when_exists(self) -> None:
        """Test that it returns True when entity exists."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            result = await canonical_module.entity_exists("Mother")

            assert result is True
            mock_conn.fetchval.assert_called_once_with(
                "SELECT 1 FROM entities WHERE name = $1 LIMIT 1", "Mother"
            )

    @pytest.mark.asyncio
    async def test_returns_false_when_not_exists(self) -> None:
        """Test that it returns False when entity doesn't exist."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            result = await canonical_module.entity_exists("Unknown")

            assert result is False


class TestPredicateExists:
    """Tests for predicate_exists function."""

    @pytest.mark.asyncio
    async def test_returns_true_when_exists(self) -> None:
        """Test that it returns True when predicate exists."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            result = await canonical_module.predicate_exists("has goal")

            assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_exists(self) -> None:
        """Test that it returns False when predicate doesn't exist."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)
        mock_pool = create_mock_pool_with_conn(mock_conn)

        with patch.object(canonical_module, "db_pool", mock_pool):
            result = await canonical_module.predicate_exists("unknown predicate")

            assert result is False


class TestExceptionClass:
    """Tests for CanonicalRegistryError exception class."""

    def test_can_raise_and_catch(self) -> None:
        """Test that the exception can be raised and caught."""
        with pytest.raises(canonical_module.CanonicalRegistryError):
            raise canonical_module.CanonicalRegistryError("test error")
