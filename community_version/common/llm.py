"""LLM loading and answering utilities built on llama.cpp."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import json

from llama_cpp import Llama

from .config import Settings, load_settings
from .logging import get_logger

from .embeddings import embed_question, normalize_vector_hits
from .named_entity import post_ner, normalize_entities
from .opensearch_client import (
    create_long_client, create_hot_client, create_vector_client, render_observability_summary, render_matches,
    search_one, knn_search_one, rank_hits, combine_hits, build_query_external_ranking, build_query_opensearch_ranking,
)
from .bm25 import rerank_hits_with_bm25

LOGGER = get_logger(__name__)


@lru_cache(maxsize=1)
def load_llm(settings: Optional[Settings] = None) -> Llama:
    """
    Construct and cache a llama.cpp model instance based on configuration.

    The cache key is derived from the Settings object identity, so callers
    should typically pass the same Settings instance.
    """
    if settings is None:
        settings = load_settings()

    model_path = Path(settings.llama_model_path).expanduser()
    LOGGER.info("Loading LLaMA model from %s", model_path)

    kwargs: Dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": settings.llama_ctx,
        "n_threads": settings.llama_n_threads,
        "n_gpu_layers": settings.llama_n_gpu_layers,
        "n_batch": settings.llama_n_batch,
        "chat_format": "chatml",
        "verbose": False,
    }

    if settings.llama_n_ubatch is not None:
        kwargs["n_ubatch"] = settings.llama_n_ubatch
    if settings.llama_low_vram:
        kwargs["low_vram"] = True

    return Llama(**kwargs)


def _save_results(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_context(hits: List[Dict[str, Any]], max_chars_per_doc: int = 0) -> str:
    """Construct a text block to feed into the LLM from OpenSearch hits."""
    if not hits:
        return ""

    out: List[str] = []
    for h in hits:
        src = h.get("_source", {})
        fp = src.get("filepath", "<unknown>")
        store = h.get("_store_label", "?")
        content = (src.get("content") or "").strip()
        if max_chars_per_doc > 0:
            content = content[:max_chars_per_doc]
        out.append(f"---\nStore: {store}\nDoc: {fp}\n{content}\n")
    return "\n".join(out)


def _hit_key(hit: Dict[str, Any]) -> str:
    src = hit.get("_source", {}) or {}
    path = src.get("filepath") or src.get("path") or ""
    chunk = src.get("chunk_index")
    if chunk is None:
        return str(path)
    try:
        return f"{path}::chunk-{int(chunk):03d}"
    except Exception:
        return f"{path}::{chunk}"


def _merge_hybrid_ranked(
    bm25_hits: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
    *,
    top_k: int,
    vector_fraction: float,
) -> List[Dict[str, Any]]:
    """
    Merge BM25 + vector lists while:
      - guaranteeing representation from both modalities (when available),
      - deduping same doc/chunk across modalities,
      - preserving each list's internal ranking (no fancy fusion).
    """
    if top_k <= 0:
        return []

    vf = max(0.0, min(1.0, float(vector_fraction)))

    # Budgets ensure "hybrid" actually means hybrid.
    vec_budget = int(round(top_k * vf))
    vec_budget = max(0, min(top_k, vec_budget))
    bm_budget = top_k - vec_budget

    initial = bm25_hits[:bm_budget] + vector_hits[:vec_budget]

    seen: set[str] = set()
    merged: List[Dict[str, Any]] = []
    for h in initial:
        k = _hit_key(h)
        if k in seen:
            continue
        seen.add(k)
        merged.append(h)

    # Backfill from leftovers if we removed dups or were short.
    if len(merged) < top_k:
        leftovers = bm25_hits[bm_budget:] + vector_hits[vec_budget:]
        for h in leftovers:
            if len(merged) >= top_k:
                break
            k = _hit_key(h)
            if k in seen:
                continue
            seen.add(k)
            merged.append(h)

    return merged


def generate_answer(
    llm: Llama,
    question: str,
    context: str,
    *,
    observability: bool = False,
    max_tokens: int = 32768,
    temperature: float = 0.2,
    top_p: float = 0.8,
) -> str:
    """
    Run a chat completion against the LLM using the provided context.

    If the context is empty or whitespace-only, a fixed informative string
    is returned instead of querying the model.
    """
    if not context.strip():
        return "No supporting documents found."

    system_msg = "Answer using ONLY the provided context below."
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
    )

    if observability:
        LOGGER.info("LLM prompt context length=%d chars", len(context))

    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return resp["choices"][0]["message"]["content"].strip()

def ask(
    llm: Llama,
    question: str,
    *,
    observability: bool = True,
    external_ranker: bool = True,
    top_k: int = 10,
    save_path: str | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    # 1) NER
    ner = post_ner(question)
    entities = normalize_entities(ner)

    # 2) Build BM25 query
    if external_ranker:
        if observability:
            print("\n\nUsing EXTERNAL ranking with BM25 re-ranking after retrieval.\n\n")
        query = build_query_external_ranking(question, entities)
    else:
        if observability:
            print("\n\nUsing INTERNAL OpenSearch ranking only.\n\n")
        query = build_query_opensearch_ranking(question, entities)

    if observability:
        if entities:
            print(f"[NER] entities: {entities}")
            print("\n[QUERY] dis_max (entity path):")
        else:
            print("[NER] No entities detected; using full-question match only.")
            print("\n[QUERY] dis_max (no-entity path):")
        print(json.dumps(query, indent=2))

    # 3) Embed the question for vector search
    query_vec = embed_question(question)

    # 4) Connect stores
    long_client, long_index = create_long_client()
    hot_client, hot_index = create_hot_client()
    vector_client, vector_index = create_vector_client()

    # 5) Execute in parallel (LONG, HOT, VECTOR)
    vec_k = max(1, min(int(top_k), 200))  # guardrail
    vec_candidates = max(vector_client.settings.search_size, vec_k)

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_long = ex.submit(search_one, "LONG", long_client, long_index, query, long_client.settings)
        fut_hot = ex.submit(search_one, "HOT", hot_client, hot_index, query, hot_client.settings)
        fut_vec = ex.submit(
            knn_search_one,
            "VECTOR",
            vector_client,
            vector_index,
            query_vec,
            k=vec_k,
            num_candidates=vec_candidates,
        )
        res_long = fut_long.result()
        res_hot = fut_hot.result()
        res_vec = fut_vec.result()

    # 6) Rank BM25 per store (plus optional external BM25 rerank)
    if external_ranker:
        keep_long, keep_hot, bm25_combined = rerank_hits_with_bm25(
            question, res_long, res_hot, top_k=top_k
        )
    else:
        keep_long = rank_hits(res_long)
        keep_hot = rank_hits(res_hot)
        bm25_combined = combine_hits(keep_long, keep_hot, top_k=top_k)

    # Vector hits are already ranked by similarity.
    vector_hits = normalize_vector_hits(res_vec)

    # Merge BM25 + VECTOR without `combine_hits` (as requested)
    hybrid_hits = _merge_hybrid_ranked(
        bm25_combined,
        vector_hits,
        top_k=top_k,
        vector_fraction=vector_client.settings.ranking_alpha,
    )

    # 7) Observability prints
    if observability:
        print(render_observability_summary(res_long))
        print(render_observability_summary(res_hot))
        print(render_observability_summary(res_vec))

        print(f"\n[RESULTS] LONG kept={len(keep_long)} of {len(res_long.get('hits',{}).get('hits',[]))}")
        print(f"[RESULTS] HOT  kept={len(keep_hot)} of {len(res_hot.get('hits',{}).get('hits',[]))}")
        print(f"[RESULTS] VEC  kept={len(vector_hits)} of {len(res_vec.get('hits',{}).get('hits',[]))}")

        print("\n[HYBRID MATCHES]")
        print(render_matches(hybrid_hits))

    # 8) Optional save (compact JSONL)
    if save_path:
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "entities": entities,
            "alpha": vector_client.settings.ranking_alpha,
            "size": vector_client.settings.search_size,
            "preference": vector_client.settings.search_preference,
            "vector": {
                "index": res_vec.get("_index_used"),
                "total": res_vec.get("hits", {}).get("total", {}).get("value", 0),
                "error": res_vec.get("_error"),
                "kept_filepaths": [h.get("_source", {}).get("filepath") for h in vector_hits[:top_k]],
            },
            "long": {
                "index": res_long.get("_index_used"),
                "total": res_long.get("hits", {}).get("total", {}).get("value", 0),
                "error": res_long.get("_error"),
                "kept_filepaths": [h.get("_source", {}).get("filepath") for h in keep_long],
            },
            "hot": {
                "index": res_hot.get("_index_used"),
                "total": res_hot.get("hits", {}).get("total", {}).get("value", 0),
                "error": res_hot.get("_error"),
                "kept_filepaths": [h.get("_source", {}).get("filepath") for h in keep_hot],
            },
            "hybrid_filepaths": [h.get("_source", {}).get("filepath") for h in hybrid_hits],
        }
        _save_results(save_path, payload)

    # 9) Build context and ask the LLM
    context_block = _build_context(hybrid_hits)
    answer = generate_answer(llm, question, context_block, observability=observability)

    return answer, hybrid_hits


__all__ = ["load_llm", "generate_answer", "ask"]
