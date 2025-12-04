"""Embedding utilities using sentence-transformers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings, load_settings


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    """Lazy-load and cache the sentence-transformers model."""

    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def load_embedder() -> "EmbeddingModel":
    """Return a cached :class:`EmbeddingModel` instance."""

    return EmbeddingModel()


@dataclass
class EmbeddingModel:
    """Wrapper around a sentence-transformers embedding model with lazy init."""

    settings: Settings
    _cached_model: Optional[SentenceTransformer] = None

    def __init__(self, settings: Optional[Settings] = None) -> None:
        if settings is None:
            settings = load_settings()
        self.settings = settings
        self._cached_model = None

    @property
    def model_name(self) -> str:
        """Determine the configured model name with backward compatibility."""

        name = getattr(self.settings, "embedding_model_name", None)
        if not name:
            name = getattr(self.settings, "embedding_model", None)
        if not name:
            name = "thenlper/gte-small"
        return name

    @property
    def model(self) -> SentenceTransformer:
        """Load the underlying embedding model on first access."""

        if self._cached_model is None:
            self._cached_model = _load_model(self.model_name)
        return self._cached_model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, respecting explicit overrides."""

        dim = getattr(self.settings, "embedding_dimension", None)
        if dim is not None:
            return int(dim)

        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception:
            vec = self.encode(["_probe_"])[0]
            return int(vec.shape[-1])

    def encode(self, texts: Iterable[str]) -> List[np.ndarray]:
        """Encode text to L2-normalized numpy arrays ready for cosine kNN."""

        items: List[str] = list(texts)
        if len(items) == 0:
            return []

        arr = self.model.encode(
            items,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return [arr[i] for i in range(arr.shape[0])]
        if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(items) == 1:
            return [arr]
        return [np.asarray(v, dtype=float) for v in arr]


def embed_question(question: str) -> List[float]:
    """Embed a single question into a plain Python list."""

    embedder = load_embedder()
    vecs = embedder.encode([question])
    return to_list(vecs[0])


def normalize_vector_hits(
    res_vector: Dict[str, Any], *, label: str = "VECTOR"
) -> List[Dict[str, Any]]:
    """Normalize vector hits to look like BM25 hits for downstream reuse."""

    hits = res_vector.get("hits", {}).get("hits", []) or []
    out: List[Dict[str, Any]] = []

    for hit in hits:
        if not isinstance(hit, dict):
            continue
        hit["_store_label"] = res_vector.get("_store_label", label)
        hit["_index_used"] = res_vector.get("_index_used", "?")
        source = hit.get("_source") or {}
        if not isinstance(source, dict):
            source = {}
        if "filepath" not in source:
            source["filepath"] = source.get("path") or source.get("rel_path") or "<unknown>"
        if "content" not in source:
            source["content"] = source.get("text") or ""
        hit["_source"] = source
        out.append(hit)

    return out


def to_list(vec: Union[np.ndarray, Sequence[float], List[float]]) -> List[float]:
    """Convert a vector-like object to a Python list of floats."""

    if isinstance(vec, np.ndarray):
        return vec.astype(float).tolist()
    return [float(x) for x in vec]


__all__ = [
    "load_embedder",
    "EmbeddingModel",
    "embed_question",
    "normalize_vector_hits",
    "to_list",
]
