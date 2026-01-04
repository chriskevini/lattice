"""Unit tests for Lattice utility modules."""

import os
from unittest.mock import patch

import numpy as np
import pytest

from lattice.utils.database import DatabasePool
from lattice.utils.embeddings import EmbeddingModel
from lattice.utils.objective_parsing import parse_objectives


class TestEmbeddingModel:
    """Tests for the embedding model module."""

    def test_embedding_model_initialization(self) -> None:
        """Test that EmbeddingModel initializes with correct defaults."""
        model = EmbeddingModel()

        assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert model.dimension == 384
        assert model.cache_dir.name == "models"
        assert model._model is None

    def test_embedding_model_custom_config(self) -> None:
        """Test that EmbeddingModel uses custom environment config."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_MODEL": "test-model",
                "EMBEDDING_DIMENSION": "256",
                "EMBEDDING_CACHE_DIR": "/tmp/test-cache",
            },
        ):
            model = EmbeddingModel()

            assert model.model_name == "test-model"
            assert model.dimension == 256
            assert str(model.cache_dir) == "/tmp/test-cache"

    def test_embedding_model_load_idempotent(self) -> None:
        """Test that loading the model multiple times is idempotent."""
        model = EmbeddingModel()

        model._model = None
        model.load()

        assert model._model is not None

        original_model = model._model
        model.load()

        assert model._model is original_model

    def test_encode_single_returns_list_of_floats(self) -> None:
        """Test that encode_single returns a list of floats."""
        model = EmbeddingModel()

        with patch.object(model, "encode") as mock_encode:
            mock_encode.return_value = np.array([[0.1, 0.2, 0.3]])

            result = model.encode_single("test text")

            assert isinstance(result, list)
            assert len(result) == 3
            assert all(isinstance(x, float) for x in result)

    def test_encode_raises_when_not_loaded(self) -> None:
        """Test that encode raises RuntimeError when model not loaded."""
        model = EmbeddingModel()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.encode("test text")

    def test_encode_single_raises_when_not_loaded(self) -> None:
        """Test that encode_single raises RuntimeError when model not loaded."""
        model = EmbeddingModel()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.encode_single("test text")

    def test_model_property_raises_when_not_loaded(self) -> None:
        """Test that model property raises RuntimeError when not loaded."""
        model = EmbeddingModel()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = model.model


class TestDatabasePool:
    """Tests for the database pool module."""

    def test_database_pool_initial_state(self) -> None:
        """Test that DatabasePool starts uninitialized."""
        pool = DatabasePool()

        assert pool.is_initialized() is False
        assert pool._pool is None

    def test_database_pool_property_raises_when_not_initialized(self) -> None:
        """Test that pool property raises RuntimeError when not initialized."""
        pool = DatabasePool()

        with pytest.raises(RuntimeError, match="Database pool not initialized"):
            _ = pool.pool

    def test_close_when_not_initialized(self) -> None:
        """Test that close is safe when pool is not initialized."""
        pool = DatabasePool()

        pool.close()  # type: ignore[unused-coroutine]

        assert pool._pool is None


class TestObjectiveParsing:
    """Tests for objective parsing utilities."""

    def test_parse_objectives_valid_json(self) -> None:
        """Test parsing valid objective JSON array."""
        raw = (
            '[{"description": "Build a startup", "saliency": 0.9, "status": "pending"}]'
        )

        result = parse_objectives(raw)

        assert len(result) == 1
        assert result[0]["description"] == "Build a startup"
        assert result[0]["saliency"] == 0.9
        assert result[0]["status"] == "pending"

    def test_parse_objectives_multiple(self) -> None:
        """Test parsing multiple objectives."""
        raw = (
            '[{"description": "Learn Python", "saliency": 0.8, "status": "pending"}, '
            '{"description": "Build a project", "saliency": 0.7, "status": "completed"}]'
        )

        result = parse_objectives(raw)

        assert len(result) == 2

    def test_parse_objectives_empty_array(self) -> None:
        """Test parsing empty objective array."""
        raw = "[]"

        result = parse_objectives(raw)

        assert result == []

    def test_parse_objectives_with_code_block(self) -> None:
        """Test parsing objectives wrapped in code block."""
        raw = """```json
[{"description": "Test goal", "saliency": 0.5, "status": "pending"}]
```"""

        result = parse_objectives(raw)

        assert len(result) == 1
        assert result[0]["description"] == "Test goal"

    def test_parse_objectives_saliency_clamping(self) -> None:
        """Test that saliency is clamped to valid range."""
        raw = '[{"description": "Goal", "saliency": 1.5, "status": "pending"}]'

        result = parse_objectives(raw)

        assert result[0]["saliency"] == 1.0

    def test_parse_objectives_saliency_default(self) -> None:
        """Test that missing saliency uses default."""
        raw = '[{"description": "Goal", "status": "pending"}]'

        result = parse_objectives(raw)

        assert result[0]["saliency"] == 0.5

    def test_parse_objectives_invalid_status(self) -> None:
        """Test that invalid status defaults to pending."""
        raw = '[{"description": "Goal", "saliency": 0.5, "status": "invalid"}]'

        result = parse_objectives(raw)

        assert result[0]["status"] == "pending"

    def test_parse_objectives_invalid_json(self) -> None:
        """Test that invalid JSON returns empty list."""
        raw = "not valid json"

        result = parse_objectives(raw)

        assert result == []

    def test_parse_objectives_missing_description(self) -> None:
        """Test that objectives without description are filtered out."""
        raw = (
            '[{"description": "Valid", "saliency": 0.5, "status": "pending"}, '
            '{"saliency": 0.5, "status": "pending"}]'
        )

        result = parse_objectives(raw)

        assert len(result) == 1
        assert result[0]["description"] == "Valid"

    def test_parse_objectives_negative_saliency(self) -> None:
        """Test that negative saliency is clamped to 0.0."""
        raw = '[{"description": "Goal", "saliency": -0.5, "status": "pending"}]'

        result = parse_objectives(raw)

        assert result[0]["saliency"] == 0.0

    def test_parse_objectives_empty_description(self) -> None:
        """Test that empty description is filtered out."""
        raw = '[{"description": "", "saliency": 0.5, "status": "pending"}]'

        result = parse_objectives(raw)

        assert len(result) == 0

    def test_parse_objectives_whitespace_description(self) -> None:
        """Test that whitespace-only description is filtered out."""
        raw = '[{"description": "   ", "saliency": 0.5, "status": "pending"}]'

        result = parse_objectives(raw)

        assert len(result) == 0

    def test_parse_objectives_none_description(self) -> None:
        """Test that None description is filtered out."""
        raw = '[{"description": null, "saliency": 0.5, "status": "pending"}]'

        result = parse_objectives(raw)

        assert len(result) == 0

    def test_parse_objectives_status_case_insensitive(self) -> None:
        """Test that status is normalized to lowercase."""
        raw = '[{"description": "Goal", "saliency": 0.5, "status": "COMPLETED"}]'

        result = parse_objectives(raw)

        assert result[0]["status"] == "completed"
