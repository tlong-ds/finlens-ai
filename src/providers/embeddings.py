"""Shared Granite dense embedding contract for indexing and retrieval."""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from src.config import Settings


logger = logging.getLogger(__name__)

# Transformers temporarily changes process-global torch dtype while constructing
# some model architectures. Serializing model construction prevents concurrent
# loads from leaving a ModernBERT instance with mixed Float/BFloat16 parameters.
_MODEL_LOAD_LOCK = threading.Lock()

EMBEDDING_MODEL_DEFAULT = "ibm-granite/granite-embedding-97m-multilingual-r2"
EMBEDDING_REVISION_DEFAULT = "835ad14087e140460703cf0fae09f97d469d65c2"
EMBEDDING_VECTOR_SIZE = 384
EMBEDDING_MAX_LENGTH = 512
EMBEDDING_MODEL_MAX_LENGTH = 32_768
DENSE_VECTOR_NAME = "dense"


class EmbeddingError(RuntimeError):
    """Raised when the configured dense embedding model cannot be used."""


class DenseEmbeddingModel:
    """Pinned Granite multilingual encoder with query/passage-aware methods."""

    def __init__(
        self,
        *,
        model_id: str = EMBEDDING_MODEL_DEFAULT,
        revision: str = EMBEDDING_REVISION_DEFAULT,
        model_path: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = EMBEDDING_MAX_LENGTH,
    ) -> None:
        if batch_size < 1:
            raise ValueError("embedding batch_size must be at least 1")
        if not 1 <= max_length <= EMBEDDING_MODEL_MAX_LENGTH:
            raise ValueError(
                "embedding max_length must be between 1 and "
                f"{EMBEDDING_MODEL_MAX_LENGTH}"
            )
        self.model_id = model_id
        self.revision = revision
        self.model_path = model_path
        self.device = device.strip().lower() or "auto"
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = None
        self._encode_lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "DenseEmbeddingModel":
        """Build an encoder from the validated application settings."""
        return cls(
            model_id=settings.embedding_model,
            revision=settings.embedding_revision,
            model_path=settings.embedding_model_path or None,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            max_length=settings.embedding_max_length,
        )

    def load(self) -> None:
        if self._model is not None:
            return
        with _MODEL_LOAD_LOCK:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise EmbeddingError(
                    "Missing sentence-transformers. Run: uv sync --frozen"
                ) from exc

            source = self.model_id
            kwargs: dict[str, Any] = {"trust_remote_code": False}
            if self.model_path:
                path = Path(self.model_path).resolve()
                if not path.is_dir():
                    raise EmbeddingError(f"EMBEDDING_MODEL_PATH does not exist: {path}")
                source = str(path)
            else:
                kwargs["revision"] = self.revision
            if self.device != "auto":
                kwargs["device"] = self.device

            try:
                model = SentenceTransformer(source, **kwargs)
            except Exception as exc:
                raise EmbeddingError(
                    f"Cannot load embedding model {self.model_id}@{self.revision}"
                ) from exc
            model.max_seq_length = self.max_length
            get_dimension = getattr(model, "get_embedding_dimension", None)
            if not callable(get_dimension):
                get_dimension = model.get_sentence_embedding_dimension
            dimension = get_dimension()
            if int(dimension or 0) != EMBEDDING_VECTOR_SIZE:
                raise EmbeddingError(
                    f"Embedding dimension is {dimension}; expected {EMBEDDING_VECTOR_SIZE}"
                )
            self._model = model
        logger.info(
            "Loaded %s@%s (device=%s, max_length=%d)",
            self.model_id,
            self.revision,
            self.device,
            self.max_length,
        )

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared: list[str] = []
        for text in texts:
            value = str(text).strip()
            if not value:
                raise EmbeddingError("Cannot embed empty text")
            prepared.append(value)

        self.load()
        try:
            with self._encode_lock:
                result = self._model.encode(
                    prepared,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except Exception as exc:
            raise EmbeddingError("Embedding inference failed") from exc

        raw_vectors = result.tolist() if hasattr(result, "tolist") else list(result)
        vectors: list[list[float]] = []
        for raw in raw_vectors:
            vector = [float(value) for value in raw]
            if len(vector) != EMBEDDING_VECTOR_SIZE:
                raise EmbeddingError(
                    f"Embedding dimension is {len(vector)}; "
                    f"expected {EMBEDDING_VECTOR_SIZE}"
                )
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingError("Embedding contains NaN or Inf")
            norm = math.sqrt(sum(value * value for value in vector))
            if norm <= 0:
                raise EmbeddingError("Embedding has zero norm")
            vectors.append([value / norm for value in vector])
        if len(vectors) != len(texts):
            raise EmbeddingError("Embedding count does not match input count")
        return vectors

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode indexed metadata without model-specific text prefixes."""
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode user queries without model-specific text prefixes."""
        return self._encode(texts)
