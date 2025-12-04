#!/usr/bin/env python3
"""
Hybrid ingestion for BBC dataset:
- Full-document BM25 into `bbc` (or configurable).
- Paragraph-level BM25 into `bbc-bm25-chunks` (or configurable).
- Paragraph-level dense vectors into `bbc-vec-chunks` (or configurable).
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Sequence

from tqdm import tqdm
from opensearchpy import OpenSearch

from common.opensearch_client import create_long_client, create_vector_client
from common.named_entity import post_ner, normalize_entities
from common.embeddings import EmbeddingModel, to_list
from common.logging import get_logger

LOGGER = get_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid BM25 + vector ingestion for BBC dataset")
    p.add_argument("--data-dir", type=str, default="bbc", help="Path to BBC dataset root (category subdirs)")
    p.add_argument("--full-index", type=str, default="bbc-bm25", help="Full-document BM25 index")
    p.add_argument("--bm25-chunk-index", type=str, default="bbc-bm25-chunks", help="Paragraph-level BM25 index")
    p.add_argument("--vec-chunk-index", type=str, default="bbc-vec-chunks", help="Paragraph-level vector index")
    p.add_argument("--batch-size", type=int, default=32, help="Vector batch size (paragraphs)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into non-empty paragraphs separated by blank lines."""
    paragraphs: List[str] = []
    current: List[str] = []

    for line in text.splitlines():
        if line.strip():
            current.append(line)
            continue

        if current:
            paragraphs.append("\n".join(current).strip())
            current = []

    if current:
        paragraphs.append("\n".join(current).strip())

    return paragraphs or ([text.strip()] if text.strip() else [])


def extract_entities(text: str) -> List[str]:
    ner_result = post_ner(text)
    return normalize_entities(ner_result)


# ---------------------------------------------------------------------------
# BM25 index creation (from your bm25 ingest script)
# ---------------------------------------------------------------------------

def ensure_bm25_index(
    client: OpenSearch,
    index_name: str,
    extra_properties: Dict[str, dict] | None = None,
) -> None:
    if client.indices.exists(index=index_name):
        return
    body: Dict[str, object] = {
        "settings": {
            "analysis": {
                "normalizer": {
                    "lowercase_normalizer": {
                        "type": "custom",
                        "char_filter": [],
                        "filter": ["lowercase"],
                    }
                }
            },
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "category": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "filepath": {"type": "keyword"},
                "explicit_terms": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "explicit_terms_text": {"type": "text"},
                "ingested_at_ms": {"type": "date", "format": "epoch_millis"},
                "doc_version": {"type": "long"},
            }
        },
    }
    if extra_properties:
        body["mappings"]["properties"].update(extra_properties)
    client.indices.create(index=index_name, body=body)


# ---------------------------------------------------------------------------
# Index management for vector store
# ---------------------------------------------------------------------------


def ensure_vector_index(client: OpenSearch, index_name: str, dim: int) -> None:
    """Ensure that the target index exists with the correct mapping."""
    if client.indices.exists(index=index_name):
        LOGGER.info("OpenSearch index '%s' already exists", index_name)
        return

    body = {
        "settings": {
            "index": {
                # Enable k-NN index structures
                "knn": True,
                # Lucene engine ignores ef_search and derives it from k,
                # but keeping this setting is harmless if you switch engines later.
                "knn.algo_param.ef_search": 256,
            }
        },
        "mappings": {
            "properties": {
                "path": {"type": "keyword"},
                "title": {"type": "keyword"},
                "category": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        # You can add "parameters": {"m": 16, "ef_construction": 128} here if desired.
                    },
                },
            }
        },
    }
    LOGGER.info("Creating OpenSearch index '%s'", index_name)
    client.indices.create(index=index_name, body=body)


# ---------------------------------------------------------------------------
# File iteration
# ---------------------------------------------------------------------------

def iter_bbc_files(data_dir: Path):
    """Yield (category, file_path, text) for each BBC article."""
    for category_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        category = category_dir.name
        for fp in sorted(category_dir.glob("*.txt")):
            text = fp.read_text(encoding="utf-8", errors="ignore")
            yield category, fp, text


# ---------------------------------------------------------------------------
# Ingestion logic
# ---------------------------------------------------------------------------

def doc_sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def ingest_hybrid(
    data_dir: Path,
    batch_size: int,
) -> None:
    # 1) Connect BM25 cluster (LONG) and create indices
    bm25_client, _ = create_long_client()
    ensure_bm25_index(bm25_client, bm25_client.settings.opensearch_hot_index)
    ensure_bm25_index(
        bm25_client,
        bm25_client.settings.opensearch_long_index,
        extra_properties={
            "chunk_index": {"type": "integer"},
            "chunk_count": {"type": "integer"},
            "parent_filepath": {"type": "keyword"},
        },
    )

    # 2) Connect vector store and ensure index exists with right dimension
    vec_client, _ = create_vector_client()
    embedder = EmbeddingModel()
    ensure_vector_index(vec_client, vec_client.settings.opensearch_vector_index, embedder.dimension)

    now_ms = int(time.time() * 1000)

    total_docs = 0
    total_chunks_bm25 = 0
    total_chunks_vec = 0

    progress = tqdm(desc="Hybrid ingest (docs)", unit="doc")
    try:
        for category, fp, text in iter_bbc_files(data_dir):
            rel_path = fp.relative_to(data_dir).as_posix()

            # ---------------- BM25 full-document ----------------
            explicit_terms_doc = extract_entities(text)
            full_doc = {
                "content": text,
                "category": category,
                "filepath": rel_path,
                "explicit_terms": explicit_terms_doc,
                "explicit_terms_text": " ".join(explicit_terms_doc) if explicit_terms_doc else "",
                "ingested_at_ms": now_ms,
                "doc_version": now_ms,
            }
            bm25_client.index(index=bm25_client.settings.opensearch_full_index, id=rel_path, body=full_doc, refresh=False)

            # ---------------- Paragraph-level chunking ----------------
            paragraphs = split_into_paragraphs(text)
            chunk_count = len(paragraphs)

            # BM25 chunks
            for idx, paragraph in enumerate(paragraphs):
                para = paragraph.strip()
                if not para:
                    continue
                chunk_terms = extract_entities(para) if para else []
                chunk_doc = {
                    "content": para,
                    "category": category,
                    "filepath": rel_path,
                    "parent_filepath": rel_path,
                    "chunk_index": idx,
                    "chunk_count": chunk_count,
                    "explicit_terms": chunk_terms,
                    "explicit_terms_text": " ".join(chunk_terms) if chunk_terms else "",
                    "ingested_at_ms": now_ms,
                    "doc_version": now_ms,
                }
                chunk_id = f"{rel_path}::chunk-{idx:03d}"
                bm25_client.index(index=bm25_client.settings.opensearch_long_index, id=chunk_id, body=chunk_doc, refresh=False)
                total_chunks_bm25 += 1

            # Vector chunks: reuse the same paragraphs so RAG can align hits
            vec_chunks: List[Dict[str, object]] = []
            for idx, paragraph in enumerate(paragraphs):
                para = paragraph.strip()
                if not para:
                    continue
                vec_chunks.append(
                    {
                        "category": category,
                        "rel_path": rel_path,
                        "chunk_index": idx,
                        "chunk_count": chunk_count,
                        "text": para,
                    }
                )

            # batch embedding per document
            for i in range(0, len(vec_chunks), batch_size):
                batch = vec_chunks[i : i + batch_size]
                if not batch:
                    continue
                texts = [c["text"] for c in batch]
                embeddings = embedder.encode(texts)
                for c, emb in zip(batch, embeddings):
                    emb_vec = to_list(emb)
                    chunk_id = f"{c['rel_path']}::chunk-{c['chunk_index']:03d}"
                    vec_id = doc_sha1(chunk_id)
                    body = {
                        "path": c["rel_path"],
                        "category": c["category"],
                        "chunk_index": c["chunk_index"],
                        "chunk_count": c["chunk_count"],
                        "text": c["text"],
                        "embedding": emb_vec,
                    }
                    vec_client.index(index=vec_client.settings.opensearch_vector_index, id=vec_id, body=body, refresh=False)
                    total_chunks_vec += 1

            total_docs += 1
            progress.update(1)
    finally:
        progress.close()

    # Final refresh so queries see everything
    bm25_client.indices.refresh(index=bm25_client.settings.opensearch_full_index)
    bm25_client.indices.refresh(index=bm25_client.settings.opensearch_long_index)
    vec_client.indices.refresh(index=vec_client.settings.opensearch_vector_index)

    LOGGER.info(
        "Hybrid ingest complete: %d docs, %d BM25 chunks, %d vector chunks "
        "into indices full='%s', bm25_chunks='%s', vec_chunks='%s'",
        total_docs,
        total_chunks_bm25,
        total_chunks_vec,
        bm25_client.settings.opensearch_full_index,
        bm25_client.settings.opensearch_long_index,
        vec_client.settings.opensearch_vector_index,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    ingest_hybrid(
        data_dir=data_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
