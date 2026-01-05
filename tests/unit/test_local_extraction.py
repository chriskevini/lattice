"""Unit tests for local extraction model."""

import asyncio
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from lattice.core.local_extraction import (
    LocalExtractionModel,
    LocalModelConfig,
    extract_with_local_model,
    get_local_model,
)


@pytest.fixture
def model_config():
    """Create test model configuration."""
    return LocalModelConfig(
        model_path="/tmp/test_model.gguf",
        max_context=512,
        idle_timeout_seconds=1,  # Short timeout for tests
        memory_limit_mb=100,
        gpu_layers=0,
    )


@pytest.fixture
def mock_llama():
    """Mock llama_cpp.Llama model."""
    mock = MagicMock()
    mock.return_value = {
        "choices": [
            {
                "text": '{"message_type": "query", "entities": [], "predicates": [], "continuation": false}'
            }
        ]
    }
    return mock


@pytest.mark.asyncio
async def test_local_model_extract_success(model_config, mock_llama):
    """Test successful extraction with local model."""
    # Mock llama_cpp module
    mock_llama_module = MagicMock()
    mock_llama_module.Llama = mock_llama

    with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
        with patch("os.path.exists", return_value=True):
            model = LocalExtractionModel(model_config)
            result = await model.extract("Test prompt")

            assert result is not None
            data = json.loads(result)
            assert data["message_type"] == "query"


@pytest.mark.asyncio
async def test_local_model_load_on_demand(model_config, mock_llama):
    """Test model is loaded on first extraction."""
    # Mock llama_cpp module
    mock_llama_module = MagicMock()
    mock_llama_module.Llama = mock_llama

    with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
        with patch("os.path.exists", return_value=True):
            model = LocalExtractionModel(model_config)

            # Model should not be loaded initially
            assert model._model is None

            # First extraction should load model
            await model.extract("Test prompt")
            assert model._model is not None
            assert mock_llama.called


@pytest.mark.asyncio
async def test_local_model_unload_after_idle(model_config, mock_llama):
    """Test model is unloaded after idle timeout."""
    mock_llama_module = MagicMock()
    mock_llama_module.Llama = mock_llama
    with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
        with patch("os.path.exists", return_value=True):
            model = LocalExtractionModel(model_config)

            # Extract and verify model is loaded
            await model.extract("Test prompt")
            assert model._model is not None

            # Wait for idle timeout
            await asyncio.sleep(model_config.idle_timeout_seconds + 0.2)

            # Model should be unloaded
            assert model._model is None


@pytest.mark.asyncio
async def test_local_model_keeps_loaded_with_frequent_use(model_config, mock_llama):
    """Test model stays loaded when used frequently."""
    mock_llama_module = MagicMock()
    mock_llama_module.Llama = mock_llama
    with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
        with patch("os.path.exists", return_value=True):
            model = LocalExtractionModel(model_config)

            # Extract multiple times within timeout
            for _ in range(3):
                await model.extract("Test prompt")
                await asyncio.sleep(0.2)  # Less than idle timeout

            # Model should still be loaded
            assert model._model is not None


@pytest.mark.asyncio
async def test_local_model_memory_check(model_config, mock_llama):
    """Test memory check prevents loading when insufficient."""
    mock_llama_module = MagicMock()
    mock_llama_module.Llama = mock_llama
    with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
        with patch("os.path.exists", return_value=True):
            # Mock psutil to report high memory usage
            mock_process = MagicMock()
            mock_process.memory_info.return_value = MagicMock(rss=1800 * 1024 * 1024)

            with patch(
                "lattice.core.local_extraction.psutil.Process",
                return_value=mock_process,
            ):
                model = LocalExtractionModel(model_config)

                # Should raise MemoryError
                with pytest.raises(MemoryError, match="Insufficient memory"):
                    await model.extract("Test prompt")


@pytest.mark.asyncio
async def test_local_model_file_not_found(model_config):
    """Test error when model file doesn't exist."""
    with patch("os.path.exists", return_value=False):
        model = LocalExtractionModel(model_config)

        with pytest.raises(RuntimeError, match="Model file not found"):
            await model.extract("Test prompt")


@pytest.mark.asyncio
async def test_is_available_success(model_config, mock_llama):
    """Test is_available returns True when model exists and llama-cpp available."""
    with patch("os.path.exists", return_value=True):
        mock_llama_module = MagicMock()
        with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
            model = LocalExtractionModel(model_config)
            assert await model.is_available()


@pytest.mark.asyncio
async def test_is_available_no_model_file(model_config):
    """Test is_available returns False when model file doesn't exist."""
    with patch("os.path.exists", return_value=False):
        model = LocalExtractionModel(model_config)
        assert not await model.is_available()


@pytest.mark.asyncio
async def test_is_available_no_llama_cpp(model_config):
    """Test is_available returns False when llama-cpp-python not installed."""
    with patch("os.path.exists", return_value=True):
        with patch("builtins.__import__", side_effect=ImportError):
            model = LocalExtractionModel(model_config)
            assert not await model.is_available()


@pytest.mark.asyncio
async def test_extract_with_local_model_success(mock_llama):
    """Test extract_with_local_model convenience function."""
    config = LocalModelConfig(
        model_path="/tmp/test.gguf",
        idle_timeout_seconds=1,
    )

    mock_llama_module = MagicMock()
    mock_llama_module.Llama = mock_llama

    with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
        with patch("lattice.core.local_extraction.get_local_model") as mock_get:
            mock_model = LocalExtractionModel(config)
            mock_model._model = mock_llama()
            mock_get.return_value = mock_model

            with patch("os.path.exists", return_value=True):
                result = await extract_with_local_model("Test prompt")
                assert result is not None


@pytest.mark.asyncio
async def test_extract_with_local_model_not_configured():
    """Test extract_with_local_model returns None when not configured."""
    with patch("os.getenv", return_value=""):
        result = await extract_with_local_model("Test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_extract_with_local_model_invalid_json(mock_llama):
    """Test extract_with_local_model returns None for invalid JSON."""
    # Mock to return invalid JSON
    mock_llama.return_value = {"choices": [{"text": "not valid json"}]}

    config = LocalModelConfig(model_path="/tmp/test.gguf", idle_timeout_seconds=1)

    with patch("lattice.core.local_extraction.get_local_model") as mock_get:
        mock_model = LocalExtractionModel(config)
        mock_model._model = mock_llama()
        mock_get.return_value = mock_model

        with patch("os.path.exists", return_value=True):
            mock_llama_module = MagicMock()
    mock_llama_module.Llama = mock_llama
    with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
        result = await extract_with_local_model("Test prompt")
        assert result is None


def test_get_local_model_configured():
    """Test get_local_model returns instance when configured."""
    env_vars = {
        "LOCAL_EXTRACTION_MODEL_PATH": "/tmp/test.gguf",
        "LOCAL_EXTRACTION_MAX_CONTEXT": "2048",
        "LOCAL_EXTRACTION_IDLE_TIMEOUT": "30",
        "LOCAL_EXTRACTION_MEMORY_LIMIT": "300",
        "LOCAL_EXTRACTION_GPU_LAYERS": "0",
    }
    with patch.dict("os.environ", env_vars, clear=False):
        # Clear global instance
        import lattice.core.local_extraction

        lattice.core.local_extraction._local_model = None

        model = get_local_model()
        assert model is not None
        assert isinstance(model, LocalExtractionModel)


def test_get_local_model_not_configured():
    """Test get_local_model returns None when not configured."""
    with patch("os.getenv", return_value=""):
        model = get_local_model()
        assert model is None


def test_get_local_model_singleton():
    """Test get_local_model returns same instance."""
    env_vars = {
        "LOCAL_EXTRACTION_MODEL_PATH": "/tmp/test.gguf",
        "LOCAL_EXTRACTION_MAX_CONTEXT": "2048",
        "LOCAL_EXTRACTION_IDLE_TIMEOUT": "30",
        "LOCAL_EXTRACTION_MEMORY_LIMIT": "300",
        "LOCAL_EXTRACTION_GPU_LAYERS": "0",
    }
    with patch.dict("os.environ", env_vars, clear=False):
        # Clear global instance
        import lattice.core.local_extraction

        lattice.core.local_extraction._local_model = None

        model1 = get_local_model()
        model2 = get_local_model()
        assert model1 is model2
