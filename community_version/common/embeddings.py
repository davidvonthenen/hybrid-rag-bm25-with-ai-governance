"""Embedding utilities using sentence-transformers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Union, Optional, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings, load_settings


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    """Lazy-load and cache the sentence-transformers model."""
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def load_embedder() -> EmbeddingModel:
    # SentenceTransformer-ish models are not cheap to load. Cache it.
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
        """
        Prefer `embedding_model_name` if present for backward-compat,
        otherwise use `embedding_model` (the current field name).
        """
        name = getattr(self.settings, "embedding_model_name", None)
        if not name:
            name = getattr(self.settings, "embedding_model", None)
        if not name:
            # Sensible default for this project
            name = "thenlper/gte-small"
        return name

    @property
    def model(self) -> SentenceTransformer:
        if self._cached_model is None:
            self._cached_model = _load_model(self.model_name)
        return self._cached_model

    @property
    def dimension(self) -> int:
        """
        Returns the embedding dimension. If `Settings` declares an explicit
        `embedding_dimension` override, use it; otherwise ask the model.
        """
        dim = getattr(self.settings, "embedding_dimension", None)
        if dim is not None:
            return int(dim)

        # Most SentenceTransformer models expose this method.
        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception:
            # Fallback: run a tiny encode to infer dimensionality.
            vec = self.encode(["_probe_"])[0]
            return int(vec.shape[-1])

    def encode(self, texts: Iterable[str]) -> List[np.ndarray]:
        """
        Encode a list/iterable of texts to a list of 1-D numpy arrays
        (L2-normalized) for direct use with cosine-similarity kNN.
        """
        items: List[str] = list(texts)
        if len(items) == 0:
            return []

        arr = self.model.encode(
            items,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Ensure we always return List[np.ndarray] of shape (D,)
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return [arr[i] for i in range(arr.shape[0])]
        if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(items) == 1:
            return [arr]
        return [np.asarray(v, dtype=float) for v in arr]


def embed_question(question: str) -> List[float]:
    embedder = load_embedder()
    vecs = embedder.encode([question])
    # vecs can be list/np array/torch tensor – to_list handles the element.
    return to_list(vecs[0])


def normalize_vector_hits(res_vector: Dict[str, Any], *, label: str = "VECTOR") -> List[Dict[str, Any]]:
    """
    Normalize vector hits to look like BM25 hits so downstream code can reuse
    render_matches() and _build_context() without special cases.

    Vector docs typically have:
      - path (instead of filepath)
      - text (instead of content)

    We copy/alias them into `filepath` and `content`.
    """
    hits = res_vector.get("hits", {}).get("hits", []) or []
    out: List[Dict[str, Any]] = []

    for h in hits:
        if not isinstance(h, dict):
            continue
        h["_store_label"] = res_vector.get("_store_label", label)
        h["_index_used"] = res_vector.get("_index_used", "?")
        src = h.get("_source") or {}
        if not isinstance(src, dict):
            src = {}
        # Alias fields so the rest of the pipeline can treat vector hits like BM25 hits.
        if "filepath" not in src:
            src["filepath"] = src.get("path") or src.get("rel_path") or "<unknown>"
        if "content" not in src:
            src["content"] = src.get("text") or ""
        h["_source"] = src
        out.append(h)

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
    "to_list"
]
