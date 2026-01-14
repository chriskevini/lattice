"""Unit tests for canonical entity and predicate registry."""

from unittest.mock import AsyncMock

import pytest

from lattice.memory import canonical as canonical_module


def create_mock_canonical_repo() -> AsyncMock:
    """Create a mock CanonicalRepository."""
    mock_repo = AsyncMock()
    return mock_repo


class TestGetCanonicalEntitiesList:
    """Tests for get_canonical_entities_list function."""

    @pytest.mark.asyncio
    async def test_returns_list_of_entities(self) -> None:
        """Test that it returns a list of entity names."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.get_entities_list = AsyncMock(
            return_value=["Mother", "boyfriend", "marathon"]
        )

        result = await canonical_module.get_canonical_entities_list(repo=mock_repo)

        assert isinstance(result, list)
        assert len(result) == 3
        assert "Mother" in result
        assert "boyfriend" in result
        assert "marathon" in result
        mock_repo.get_entities_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_empty(self) -> None:
        """Test that it returns empty list when no entities."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.get_entities_list = AsyncMock(return_value=[])

        result = await canonical_module.get_canonical_entities_list(repo=mock_repo)

        assert result == []


class TestGetCanonicalPredicatesList:
    """Tests for get_canonical_predicates_list function."""

    @pytest.mark.asyncio
    async def test_returns_list_of_predicates(self) -> None:
        """Test that it returns a list of predicate names."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.get_predicates_list = AsyncMock(
            return_value=["has goal", "due by", "did activity"]
        )

        result = await canonical_module.get_canonical_predicates_list(repo=mock_repo)

        assert isinstance(result, list)
        assert len(result) == 3
        assert "has goal" in result
        assert "due by" in result
        assert "did activity" in result
        mock_repo.get_predicates_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_empty(self) -> None:
        """Test that it returns empty list when no predicates."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.get_predicates_list = AsyncMock(return_value=[])

        result = await canonical_module.get_canonical_predicates_list(repo=mock_repo)

        assert result == []


class TestGetCanonicalEntitiesSet:
    """Tests for get_canonical_entities_set function."""

    @pytest.mark.asyncio
    async def test_returns_set_of_entities(self) -> None:
        """Test that it returns a set for O(1) lookup."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.get_entities_set = AsyncMock(return_value={"Mother", "boyfriend"})

        result = await canonical_module.get_canonical_entities_set(repo=mock_repo)

        assert isinstance(result, set)
        assert "Mother" in result
        assert "boyfriend" in result


class TestGetCanonicalPredicatesSet:
    """Tests for get_canonical_predicates_set function."""

    @pytest.mark.asyncio
    async def test_returns_set_of_predicates(self) -> None:
        """Test that it returns a set for O(1) lookup."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.get_predicates_set = AsyncMock(return_value={"has goal", "due by"})

        result = await canonical_module.get_canonical_predicates_set(repo=mock_repo)

        assert isinstance(result, set)
        assert "has goal" in result
        assert "due by" in result


class TestStoreCanonicalEntities:
    """Tests for store_canonical_entities function."""

    @pytest.mark.asyncio
    async def test_stores_entities(self) -> None:
        """Test that it stores entity names."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.store_entities = AsyncMock(return_value=2)

        count = await canonical_module.store_canonical_entities(
            repo=mock_repo, names=["entity1", "entity2"]
        )

        assert count == 2
        mock_repo.store_entities.assert_called_once_with(["entity1", "entity2"])

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_list(self) -> None:
        """Test that it returns 0 for empty input."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.store_entities = AsyncMock(return_value=0)

        count = await canonical_module.store_canonical_entities(
            repo=mock_repo, names=[]
        )

        assert count == 0
        mock_repo.store_entities.assert_called_once_with([])


class TestStoreCanonicalPredicates:
    """Tests for store_canonical_predicates function."""

    @pytest.mark.asyncio
    async def test_stores_predicates(self) -> None:
        """Test that it stores predicate names."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.store_predicates = AsyncMock(return_value=2)

        count = await canonical_module.store_canonical_predicates(
            repo=mock_repo, names=["has goal", "due by"]
        )

        assert count == 2
        mock_repo.store_predicates.assert_called_once_with(["has goal", "due by"])


class TestEntityExists:
    """Tests for entity_exists function."""

    @pytest.mark.asyncio
    async def test_returns_true_when_exists(self) -> None:
        """Test that it returns True when entity exists."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.entity_exists = AsyncMock(return_value=True)

        result = await canonical_module.entity_exists(repo=mock_repo, name="Mother")

        assert result is True
        mock_repo.entity_exists.assert_called_once_with("Mother")

    @pytest.mark.asyncio
    async def test_returns_false_when_not_exists(self) -> None:
        """Test that it returns False when entity doesn't exist."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.entity_exists = AsyncMock(return_value=False)

        result = await canonical_module.entity_exists(repo=mock_repo, name="Unknown")

        assert result is False


class TestPredicateExists:
    """Tests for predicate_exists function."""

    @pytest.mark.asyncio
    async def test_returns_true_when_exists(self) -> None:
        """Test that it returns True when predicate exists."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.predicate_exists = AsyncMock(return_value=True)

        result = await canonical_module.predicate_exists(
            repo=mock_repo, name="has goal"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_exists(self) -> None:
        """Test that it returns False when predicate doesn't exist."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.predicate_exists = AsyncMock(return_value=False)

        result = await canonical_module.predicate_exists(repo=mock_repo, name="unknown")

        assert result is False


class TestIsEntityLike:
    """Tests for _is_entity_like helper."""

    def test_rejects_iso_date(self) -> None:
        """Test that ISO dates are not considered entities."""
        assert canonical_module._is_entity_like("2026-01-10") is False

    def test_rejects_duration(self) -> None:
        """Test that durations are not considered entities."""
        assert canonical_module._is_entity_like("3 hours") is False
        assert canonical_module._is_entity_like("30 minutes") is False
        assert canonical_module._is_entity_like("2 days") is False

    def test_rejects_status_values(self) -> None:
        """Test that status values are not considered entities."""
        assert canonical_module._is_entity_like("active") is False
        assert canonical_module._is_entity_like("completed") is False
        assert canonical_module._is_entity_like("high") is False

    def test_accepts_proper_nouns(self) -> None:
        """Test that proper nouns are considered entities."""
        assert canonical_module._is_entity_like("Mother") is True
        assert canonical_module._is_entity_like("IKEA") is True
        assert canonical_module._is_entity_like("Seattle") is True

    def test_accepts_concepts(self) -> None:
        """Test that concepts are considered entities."""
        assert canonical_module._is_entity_like("coding") is True
        assert canonical_module._is_entity_like("mobile app") is True
        assert canonical_module._is_entity_like("marathon") is True


class TestExtractCanonicalForms:
    """Tests for extract_canonical_forms function."""

    def test_extracts_new_entities_from_subjects(self) -> None:
        """Test that new entities are extracted from triple subjects."""
        triples = [
            {"subject": "John", "predicate": "has goal", "object": "run marathon"}
        ]
        known_entities = set()
        known_predicates = set()

        new_entities, new_predicates = canonical_module.extract_canonical_forms(
            triples, known_entities, known_predicates
        )

        assert "John" in new_entities
        assert "run marathon" in new_entities
        assert "has goal" in new_predicates

    def test_ignores_known_entities(self) -> None:
        """Test that already-known entities are not extracted."""
        triples = [
            {"subject": "John", "predicate": "has goal", "object": "run marathon"}
        ]
        known_entities = {"John"}
        known_predicates = set()

        new_entities, new_predicates = canonical_module.extract_canonical_forms(
            triples, known_entities, known_predicates
        )

        assert "John" not in new_entities
        assert "run marathon" in new_entities

    def test_ignores_non_entity_objects(self) -> None:
        """Test that dates and durations in object position are not extracted."""
        triples = [
            {"subject": "task", "predicate": "due by", "object": "2026-01-15"},
            {"subject": "workout", "predicate": "duration", "object": "45 minutes"},
        ]
        known_entities = set()
        known_predicates = set()

        new_entities, new_predicates = canonical_module.extract_canonical_forms(
            triples, known_entities, known_predicates
        )

        assert "2026-01-15" not in new_entities
        assert "45 minutes" not in new_entities
        assert "task" in new_entities
        assert "workout" in new_entities

    def test_returns_sorted_lists(self) -> None:
        """Test that results are sorted alphabetically."""
        triples = [
            {"subject": "Zebra", "predicate": "is", "object": "animal"},
            {"subject": "Apple", "predicate": "is", "object": "fruit"},
        ]
        known_entities = set()
        known_predicates = set()

        new_entities, new_predicates = canonical_module.extract_canonical_forms(
            triples, known_entities, known_predicates
        )

        # Should be sorted
        assert new_entities == sorted(new_entities)
        assert new_predicates == sorted(new_predicates)


class TestStoreCanonicalForms:
    """Tests for store_canonical_forms function."""

    @pytest.mark.asyncio
    async def test_stores_both_entities_and_predicates(self) -> None:
        """Test that it stores both entities and predicates."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.store_entities = AsyncMock(return_value=2)
        mock_repo.store_predicates = AsyncMock(return_value=1)

        result = await canonical_module.store_canonical_forms(
            repo=mock_repo,
            new_entities=["entity1", "entity2"],
            new_predicates=["has goal"],
        )

        assert result["entities"] == 2
        assert result["predicates"] == 1
        mock_repo.store_entities.assert_called_once_with(["entity1", "entity2"])
        mock_repo.store_predicates.assert_called_once_with(["has goal"])

    @pytest.mark.asyncio
    async def test_handles_empty_lists(self) -> None:
        """Test that it handles empty input lists."""
        mock_repo = create_mock_canonical_repo()
        mock_repo.store_entities = AsyncMock(return_value=0)
        mock_repo.store_predicates = AsyncMock(return_value=0)

        result = await canonical_module.store_canonical_forms(
            repo=mock_repo, new_entities=[], new_predicates=[]
        )

        assert result["entities"] == 0
        assert result["predicates"] == 0
