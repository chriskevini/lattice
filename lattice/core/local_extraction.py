"""Local extraction model using FunctionGemma-270M for low-latency query extraction.

This module provides on-demand loading of a small local model for extraction tasks,
with automatic memory management to stay within 2GB constraint.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import structlog


logger = structlog.get_logger(__name__)


@dataclass
class LocalModelConfig:
    """Configuration for local extraction model."""

    model_path: str
    max_context: int = 2048
    idle_timeout_seconds: int = 30
    memory_limit_mb: int = 300  # Max model memory footprint
    gpu_layers: int = 0  # CPU only for 2GB constraint


class LocalExtractionModel:
    """Manages local FunctionGemma-270M model with memory-aware loading/unloading.

    The model is loaded on-demand when extraction is requested, and automatically
    unloaded after idle timeout to conserve memory. This allows operating within
    the 2GB RAM constraint while providing low-latency extraction.

    Memory Management:
        - Load: On first extraction request or after unload
        - Keep Loaded: If requests arrive within idle_timeout_seconds
        - Unload: After idle_timeout_seconds with no requests
        - Fallback: If load fails or memory pressure detected
    """

    def __init__(self, config: LocalModelConfig) -> None:
        """Initialize local extraction model manager.

        Args:
            config: Model configuration including path and memory limits
        """
        self.config = config
        self._model: Any = None
        self._last_used: float = 0.0
        self._unload_task: asyncio.Task[None] | None = None
        self._load_lock = asyncio.Lock()

    async def extract(self, prompt: str, temperature: float = 0.1) -> str:
        """Extract structured data using local model.

        Args:
            prompt: The extraction prompt
            temperature: Sampling temperature (default 0.1 for extraction)

        Returns:
            JSON string with extraction results

        Raises:
            RuntimeError: If model loading fails
            MemoryError: If insufficient memory for model
        """
        # Load model if needed
        if self._model is None:
            await self._load_model()

        # Update last used timestamp
        self._last_used = time.monotonic()

        # Cancel existing unload task
        if self._unload_task is not None and not self._unload_task.done():
            self._unload_task.cancel()

        # Schedule new unload task
        self._unload_task = asyncio.create_task(self._schedule_unload())

        logger.info("Running local extraction", prompt_length=len(prompt))

        try:
            # Run extraction in thread pool to avoid blocking
            start_time = time.monotonic()
            result: dict[str, Any] = await asyncio.to_thread(
                self._model,
                prompt,
                temperature=temperature,
                max_tokens=512,
                stop=["```", "\n\n\n"],
            )
            latency_ms = int((time.monotonic() - start_time) * 1000)

            content: str = result["choices"][0]["text"]
            logger.info(
                "Local extraction completed",
                latency_ms=latency_ms,
                response_length=len(content),
            )

            return content

        except Exception as e:
            logger.error(
                "Local extraction failed", error=str(e), error_type=type(e).__name__
            )
            raise

    async def _load_model(self) -> None:
        """Load the local model with memory checks.

        Raises:
            RuntimeError: If model file not found or loading fails
            MemoryError: If insufficient memory available
        """
        async with self._load_lock:
            # Check if already loaded by another coroutine
            if self._model is not None:
                return

            # Check if model file exists
            if not os.path.exists(self.config.model_path):
                msg = f"Model file not found: {self.config.model_path}"
                raise RuntimeError(msg)

            # Check available memory if psutil available
            try:
                import psutil

                process = psutil.Process()
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024

                # Estimate if we have room (current + model < 1.8GB to leave headroom)
                if mem_mb + self.config.memory_limit_mb > 1800:
                    msg = f"Insufficient memory: using {mem_mb:.0f}MB, need {self.config.memory_limit_mb}MB more"
                    raise MemoryError(msg)

                logger.info(
                    "Loading local extraction model",
                    current_memory_mb=int(mem_mb),
                    model_path=self.config.model_path,
                )
            except ImportError:
                logger.warning("psutil not available, skipping memory check")

            # Load model
            try:
                from llama_cpp import Llama

                self._model = await asyncio.to_thread(
                    Llama,
                    model_path=self.config.model_path,
                    n_ctx=self.config.max_context,
                    n_gpu_layers=self.config.gpu_layers,
                    verbose=False,
                )

                logger.info("Local extraction model loaded successfully")

            except ImportError as e:
                msg = "llama-cpp-python not installed. Run: uv pip install lattice[local-extraction]"
                raise RuntimeError(msg) from e
            except Exception as e:
                logger.error("Failed to load local model", error=str(e))
                raise

    async def _schedule_unload(self) -> None:
        """Schedule model unload after idle timeout."""
        try:
            await asyncio.sleep(self.config.idle_timeout_seconds)

            # Check if model was used during wait
            idle_time = time.monotonic() - self._last_used
            if idle_time >= self.config.idle_timeout_seconds:
                await self._unload_model()
        except asyncio.CancelledError:
            # Unload was cancelled because model was used again
            pass

    async def _unload_model(self) -> None:
        """Unload the model to free memory."""
        async with self._load_lock:
            if self._model is not None:
                logger.info("Unloading local extraction model after idle timeout")
                self._model = None
                # Force garbage collection
                import gc

                gc.collect()

    async def is_available(self) -> bool:
        """Check if local extraction is available (model file exists).

        Returns:
            True if model file exists and can be loaded
        """
        if not os.path.exists(self.config.model_path):
            return False

        # Check if llama-cpp-python is installed
        try:
            import llama_cpp  # noqa: F401

            return True
        except ImportError:
            return False


# Global instance (lazy initialized)
_local_model: LocalExtractionModel | None = None


def get_local_model() -> LocalExtractionModel | None:
    """Get the global local model instance if configured.

    Returns:
        LocalExtractionModel instance or None if not configured
    """
    global _local_model

    if _local_model is None:
        model_path = os.getenv("LOCAL_EXTRACTION_MODEL_PATH", "")
        if not model_path:
            return None

        config = LocalModelConfig(
            model_path=model_path,
            max_context=int(os.getenv("LOCAL_EXTRACTION_MAX_CONTEXT", "2048")),
            idle_timeout_seconds=int(os.getenv("LOCAL_EXTRACTION_IDLE_TIMEOUT", "30")),
            memory_limit_mb=int(os.getenv("LOCAL_EXTRACTION_MEMORY_LIMIT", "300")),
            gpu_layers=int(os.getenv("LOCAL_EXTRACTION_GPU_LAYERS", "0")),
        )

        _local_model = LocalExtractionModel(config)

    return _local_model


async def extract_with_local_model(prompt: str, temperature: float = 0.1) -> str | None:
    """Attempt extraction with local model, return None if unavailable or fails.

    This is a convenience function that handles all the availability checks
    and error handling, making it easy to use local extraction with fallback.

    Args:
        prompt: The extraction prompt
        temperature: Sampling temperature

    Returns:
        JSON string with extraction or None if local extraction unavailable/failed
    """
    local_model = get_local_model()
    if local_model is None:
        return None

    # Check availability
    if not await local_model.is_available():
        logger.info("Local extraction model not available")
        return None

    # Try extraction
    try:
        result = await local_model.extract(prompt, temperature)

        # Validate it's valid JSON
        try:
            json.loads(result)
            return result
        except json.JSONDecodeError:
            logger.warning("Local model returned invalid JSON", response=result[:200])
            return None

    except (RuntimeError, MemoryError) as e:
        logger.warning("Local extraction failed, will fallback to API", error=str(e))
        return None
