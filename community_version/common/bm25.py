"""BM25-based re-ranking utilities for OpenSearch hits."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import bm25s

try:
    import Stemmer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Stemmer = None  # type: ignore


_BM25_STOPWORDS = "en"
_STEMMER = Stemmer.Stemmer("english") if Stemmer else None  # type: ignore[attr-defined]


def rerank_hits_with_bm25(
    question: str,
    res_long: Dict[str, Any],
    res_hot: Dict[str, Any],
    top_k: int = 10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Re-rank hits from LONG and HOT stores using BM25.

    Parameters
    ----------
    question:
        Natural-language query posed by the user.
    res_long:
        Raw OpenSearch response for the LONG index.
    res_hot:
        Raw OpenSearch response for the HOT index.
    top_k:
        Maximum number of combined hits to return.

    Returns
    -------
    (keep_long, keep_hot, combined)
        keep_long:
            Re-ranked hits belonging to the LONG store.
        keep_hot:
            Re-ranked hits belonging to the HOT store.
        combined:
            Cross-store ranking limited to ``top_k`` documents.
    """
    if top_k <= 0:
        return [], [], []

    hits: List[Dict[str, Any]] = []
    corpus: List[str] = []

    # Flatten hits from both stores into a single list, tracking origin.
    for res in (res_long, res_hot):
        store_label = res.get("_store_label", "?")
        index_used = res.get("_index_used", "?")
        for hit in res.get("hits", {}).get("hits", []) or []:
            if not isinstance(hit, dict):
                continue
            hit["_store_label"] = store_label
            hit["_index_used"] = index_used
            hits.append(hit)
            text = (hit.get("_source", {}).get("content") or "").strip()
            corpus.append(text)

    if not hits:
        return [], [], []

    corpus_tokens = bm25s.tokenize(corpus, stopwords=_BM25_STOPWORDS, stemmer=_STEMMER)
    has_tokens = any(len(doc_tokens) > 0 for doc_tokens in corpus_tokens)
    query_tokens = bm25s.tokenize(question, stemmer=_STEMMER)

    if not has_tokens or not query_tokens:
        # Fallback: respect the original OpenSearch scores.
        sorted_hits = sorted(
            hits,
            key=lambda h: h.get("_score", float("-inf")),
            reverse=True,
        )
        top_hits = sorted_hits[: min(top_k, len(sorted_hits))]
        for hit in top_hits:
            hit.setdefault("_bm25_score", hit.get("_score"))
    else:
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        k = min(top_k, len(hits))
        results, scores = retriever.retrieve(query_tokens, k=k)

        # bm25s returns arrays shaped (n_queries, k). We only issue one query.
        doc_ids = list(results[0])
        doc_scores = list(scores[0])

        top_hits: List[Dict[str, Any]] = []
        for doc_id, score in zip(doc_ids, doc_scores):
            doc_index = int(doc_id)
            if doc_index < 0 or doc_index >= len(hits):
                continue
            hit = hits[doc_index]
            if "_original_score" not in hit and "_score" in hit:
                hit["_original_score"] = hit["_score"]
            hit["_score"] = float(score)
            hit["_bm25_score"] = float(score)
            top_hits.append(hit)

        if not top_hits:
            sorted_hits = sorted(
                hits,
                key=lambda h: h.get("_score", float("-inf")),
                reverse=True,
            )
            top_hits = sorted_hits[: min(top_k, len(sorted_hits))]

    combined = top_hits[: min(top_k, len(top_hits))]
    for hit in combined:
        hit.setdefault("_bm25_score", hit.get("_score"))

    long_label = res_long.get("_store_label", "LONG")
    hot_label = res_hot.get("_store_label", "HOT")

    keep_long = [h for h in combined if h.get("_store_label") == long_label]
    keep_hot = [h for h in combined if h.get("_store_label") == hot_label]

    return keep_long, keep_hot, combined


__all__ = ["rerank_hits_with_bm25"]
