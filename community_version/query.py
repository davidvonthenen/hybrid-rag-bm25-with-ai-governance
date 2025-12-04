#!/usr/bin/env python3
"""CLI entrypoint for querying the hybrid RAG pipeline."""

from __future__ import annotations

import argparse
import time
from typing import List

from common.llm import ask, load_llm


def parse_args() -> argparse.Namespace:
    """Build the CLI parser for querying documents."""

    parser = argparse.ArgumentParser(
        description="Dual-store BM25 retrieval with parallel LONG/HOT searches (governance-first)."
    )
    parser.add_argument("--question", help="User question to answer.")
    parser.add_argument(
        "--observability",
        action="store_true",
        default=False,
        help="Print query JSON and match summaries (default OFF).",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Append compact JSONL result records to this path (default OFF).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top documents to retrieve and provide to the LLM (default 10).",
    )
    return parser.parse_args()


def run_queries(
    questions: List[str], *, observability: bool, save_path: str | None, top_k: int
) -> None:
    """Execute one or more queries and print answers."""

    llm = load_llm()

    for question in questions:
        print("\n" + "=" * 88)
        print(f"QUESTION: {question}")
        print("=" * 88)
        start = time.time()
        answer, hits = ask(
            llm,
            question,
            observability=observability,
            save_path=save_path,
            top_k=top_k,
        )
        duration = time.time() - start

        print("\n" + "=" * 88)
        print(f"ANSWER: {answer}")
        print("\n" + "=" * 88)
        print(f"\nQuery time: {duration:.2f}s   (stores queried in parallel)")
        print(f"Docs provided to LLM: {len(hits)}\n")


def main() -> None:
    args = parse_args()
    questions: List[str]
    if args.question:
        questions = [args.question]
    else:
        questions = [
            "How much did Google purchase Windsurf for?",
            "How much did OpenAI purchase Windsurf for?",
        ]

    run_queries(
        questions,
        observability=args.observability,
        save_path=args.save_results,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
