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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tqdm import tqdm
from opensearchpy import OpenSearch

import spacy
from common.opensearch_client import create_long_client, create_vector_client
from common.named_entity import post_ner, normalize_entities
from common.embeddings import EmbeddingModel, to_list
from common.logging import get_logger

LOGGER = get_logger(__name__)

# Load spacy model for chunking
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    LOGGER.info("Downloading spacy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the ingestion job."""

    parser = argparse.ArgumentParser(
        description="Hybrid BM25 + vector ingestion for BBC dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="bbc",
        help="Path to BBC dataset root (category subdirs)",
    )
    parser.add_argument(
        "--full-index",
        type=str,
        default="bbc-bm25",
        help="Full-document BM25 index",
    )
    parser.add_argument(
        "--bm25-chunk-index",
        type=str,
        default="bbc-bm25-chunks",
        help="Paragraph-level BM25 index",
    )
    parser.add_argument(
        "--vec-chunk-index",
        type=str,
        default="bbc-vec-chunks",
        help="Paragraph-level vector index",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Vector batch size (paragraphs)",
    )
    return parser.parse_args(argv)


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


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks of approximately `chunk_size` characters with `overlap`.
    Respects sentence boundaries using SpaCy.
    """
    if not text.strip():
        return []

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sents = []
        chunk_len = 0
        j = i

        while j < len(sentences):
            sent = sentences[j]
            sent_len = len(sent)

            # If adding this sentence exceeds chunk size...
            if chunk_len + sent_len > chunk_size and chunk_sents:
                break

            chunk_sents.append(sent)
            chunk_len += sent_len + 1  # +1 for space
            j += 1

        if not chunk_sents and j < len(sentences):
            # Single sentence is larger than chunk_size, take it anyway
            chunk_sents.append(sentences[j])
            j += 1

        chunks.append(" ".join(chunk_sents))

        if j >= len(sentences):
            break

        # Calculate start of next chunk (overlap)
        overlap_len = 0
        next_i = j
        while next_i > i:
            sent_len = len(sentences[next_i-1])
            if overlap_len + sent_len > overlap:
                break
            overlap_len += sent_len + 1
            next_i -= 1

        # Ensure we always advance
        if next_i == i:
            next_i += 1

        i = next_i

    return chunks


def extract_entities(text: str) -> List[str]:
    """Run NER and return normalized entity strings."""

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


@dataclass
class IngestStats:
    """Counters for the ingestion run."""

    docs: int = 0
    bm25_chunks: int = 0
    vector_chunks: int = 0


def _build_chunk_documents(
    paragraphs: Iterable[str],
    *,
    category: str,
    rel_path: str,
    chunk_count: int,
    now_ms: int,
) -> List[Dict[str, object]]:
    """Create chunk documents for BM25 indexing."""

    chunk_docs: List[Dict[str, object]] = []
    for idx, paragraph in enumerate(paragraphs):
        para = paragraph.strip()
        if not para:
            continue
        chunk_terms = extract_entities(para) if para else []
        chunk_docs.append(
            {
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
        )
    return chunk_docs


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
    stats = IngestStats()

    progress = tqdm(desc="Hybrid ingest (docs)", unit="doc")
    try:
        for category, fp, text in iter_bbc_files(data_dir):
            rel_path = fp.relative_to(data_dir).as_posix()

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
            bm25_client.index(
                index=bm25_client.settings.opensearch_full_index,
                id=rel_path,
                body=full_doc,
                refresh=False,
            )

            chunks = split_into_chunks(text, chunk_size=1000, overlap=200)
            chunk_docs = _build_chunk_documents(
                chunks,
                category=category,
                rel_path=rel_path,
                chunk_count=len(chunks),
                now_ms=now_ms,
            )

            for chunk_doc in chunk_docs:
                chunk_id = f"{rel_path}::chunk-{int(chunk_doc['chunk_index']):03d}"
                bm25_client.index(
                    index=bm25_client.settings.opensearch_long_index,
                    id=chunk_id,
                    body=chunk_doc,
                    refresh=False,
                )
                stats.bm25_chunks += 1

            vec_chunks: List[Dict[str, object]] = [
                {
                    "category": category,
                    "rel_path": rel_path,
                    "chunk_index": chunk_doc["chunk_index"],
                    "chunk_count": chunk_doc["chunk_count"],
                    "text": chunk_doc["content"],
                }
                for chunk_doc in chunk_docs
            ]

            for i in range(0, len(vec_chunks), batch_size):
                batch = vec_chunks[i : i + batch_size]
                if not batch:
                    continue
                texts = [c["text"] for c in batch]
                embeddings = embedder.encode(texts)
                for chunk_meta, embedding in zip(batch, embeddings):
                    emb_vec = to_list(embedding)
                    chunk_id = f"{chunk_meta['rel_path']}::chunk-{int(chunk_meta['chunk_index']):03d}"
                    vec_id = doc_sha1(chunk_id)
                    body = {
                        "path": chunk_meta["rel_path"],
                        "category": chunk_meta["category"],
                        "chunk_index": chunk_meta["chunk_index"],
                        "chunk_count": chunk_meta["chunk_count"],
                        "text": chunk_meta["text"],
                        "embedding": emb_vec,
                    }
                    vec_client.index(
                        index=vec_client.settings.opensearch_vector_index,
                        id=vec_id,
                        body=body,
                        refresh=False,
                    )
                    stats.vector_chunks += 1

            stats.docs += 1
            progress.update(1)
    finally:
        progress.close()

    bm25_client.indices.refresh(index=bm25_client.settings.opensearch_full_index)
    bm25_client.indices.refresh(index=bm25_client.settings.opensearch_long_index)
    vec_client.indices.refresh(index=vec_client.settings.opensearch_vector_index)

    LOGGER.info(
        "Hybrid ingest complete: %d docs, %d BM25 chunks, %d vector chunks "
        "into indices full='%s', bm25_chunks='%s', vec_chunks='%s'",
        stats.docs,
        stats.bm25_chunks,
        stats.vector_chunks,
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
