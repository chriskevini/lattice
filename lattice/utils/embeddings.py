"""Embedding utilities for Lattice.

Provides text embedding functionality using lightweight sentence-transformers.
"""

import os
from pathlib import Path
from typing import cast

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer


logger = structlog.get_logger(__name__)


class EmbeddingModel:
    """Manages the embedding model for semantic similarity."""

    def __init__(self) -> None:
        """Initialize the embedding model manager."""
        self._model: SentenceTransformer | None = None
        self.model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        cache_dir = os.getenv("EMBEDDING_CACHE_DIR", "./models")
        self.cache_dir = Path(cache_dir)

    def load(self) -> None:
        """Load the embedding model into memory."""
        if self._model is not None:
            return

        logger.info(
            "Loading embedding model",
            model=self.model_name,
            cache_dir=str(self.cache_dir),
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = SentenceTransformer(
            self.model_name,
            cache_folder=str(self.cache_dir),
            device="cpu",
        )

        logger.info("Embedding model loaded", dimension=self.dimension)

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode text(s) into embeddings.

        Args:
            texts: Single text string or list of text strings

        Returns:
            Numpy array of embeddings (shape: [N, dimension])

        Raises:
            RuntimeError: If model is not loaded
        """
        if self._model is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        if isinstance(texts, str):
            texts = [texts]

        batch_size = int(os.getenv("MAX_EMBEDDING_BATCH_SIZE", "16"))

        result = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return cast("np.ndarray", result)

    def encode_single(self, text: str) -> list[float]:
        """Encode a single text into an embedding.

        Args:
            text: Text string to encode

        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.encode(text)
        return embedding[0].tolist()  # type: ignore[no-any-return]

    @property
    def model(self) -> SentenceTransformer:
        """Get the loaded model.

        Returns:
            The SentenceTransformer model

        Raises:
            RuntimeError: If model is not loaded
        """
        if self._model is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)
        return self._model


embedding_model = EmbeddingModel()
