"""Unit tests for Lattice utility modules."""

import os
from unittest.mock import patch

import numpy as np
import pytest

from lattice.utils.database import DatabasePool
from lattice.utils.embeddings import EmbeddingModel


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
