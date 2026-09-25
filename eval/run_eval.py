"""
The evaluation harness. This is the single most important file in the
project -- it is what turns "I built a RAG app" into "I measured a RAG
app and can show you the numbers."

Usage:
    python -m eval.run_eval                 # retrieval only (BM25 / dense / hybrid)
    python -m eval.run_eval --generate       # also run the LLM and score refusals/citations
                                              # (needs ANTHROPIC_API_KEY in .env, or
                                              # LLM_PROVIDER=none to smoke-test the stub)

Writes a full per-question trace to eval/results.jsonl and prints a
summary table to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RetrievalConfig, GenerationConfig  # noqa: E402
from src.retrieve import Retriever  # noqa: E402
from src.generate import generate_answer  # noqa: E402
from eval.metrics import hit_at_k, mrr, refusal_correct, citation_present  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent / "eval_questions.jsonl"
RESULTS_PATH = Path(__file__).resolve().parent / "results.jsonl"


def load_questions() -> list[dict]:
    with open(EVAL_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_retriever_mode(retriever: Retriever, questions: list[dict], mode: str, k: int = 4):
    """
    mode in {"bm25", "dense", "hybrid"} -- lets us report the improvement
    hybrid search gives over either signal alone, instead of asserting it.
    """
    hits, mrrs = [], []
    for q in questions:
        if mode == "bm25":
            ranked = [cid for cid, _ in retriever.bm25_search(q["question"], top_k=k)]
        elif mode == "dense":
            ranked = [cid for cid, _ in retriever.dense_search(q["question"], top_k=k)]
        else:
            ranked = [r.chunk_id for r in retriever.retrieve(q["question"], top_k=k)]
        doc_ids = [retriever.chunks[cid]["doc_id"] for cid in ranked]
        hits.append(hit_at_k(doc_ids, q["expected_doc_id"], k))
        mrrs.append(mrr(doc_ids, q["expected_doc_id"]))
    return {
        "mode": mode,
        f"hit_rate@{k}": sum(hits) / len(hits),
        "mrr": sum(mrrs) / len(mrrs),
        "n": len(questions),
    }


def run_generation(retriever: Retriever, questions: list[dict], gen_cfg: GenerationConfig):
    rows = []
    for q in questions:
        chunks = retriever.retrieve(q["question"])
        ans = generate_answer(q["question"], chunks, gen_cfg)
        row = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_doc_id": q["expected_doc_id"],
            "retrieved_doc_ids": [c.doc_id for c in chunks],
            "answer": ans.text,
            "refused": ans.refused,
            "citations": ans.citations,
            "refusal_correct": refusal_correct(ans.text, ans.refused, q["expected_doc_id"]),
            "citation_correct": citation_present(ans.citations, q["expected_doc_id"]),
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--generate", action="store_true", help="also run the LLM step")
    args = parser.parse_args()

    questions = load_questions()
    retriever = Retriever(RetrievalConfig())

    print(f"Loaded {len(questions)} eval questions from {EVAL_PATH.name}\n")
    print("=== Retrieval: BM25 vs Dense vs Hybrid ===")
    print(f"{'mode':<8} {'hit_rate@' + str(args.k):<12} {'mrr':<8} n")
    for mode in ("bm25", "dense", "hybrid"):
        res = evaluate_retriever_mode(retriever, questions, mode, k=args.k)
        print(f"{res['mode']:<8} {res[f'hit_rate@{args.k}']:<12.3f} {res['mrr']:<8.3f} {res['n']}")

    results_payload = {"retrieval": {}}
    for mode in ("bm25", "dense", "hybrid"):
        results_payload["retrieval"][mode] = evaluate_retriever_mode(retriever, questions, mode, k=args.k)

    if args.generate:
        gen_cfg = GenerationConfig()
        print(f"\n=== Generation (provider={gen_cfg.provider}, model={gen_cfg.model}) ===")
        rows = run_generation(retriever, questions, gen_cfg)
        n = len(rows)
        refusal_acc = sum(r["refusal_correct"] for r in rows) / n
        citation_acc = sum(r["citation_correct"] for r in rows) / n
        print(f"refusal_correct: {refusal_acc:.3f}   citation_correct: {citation_acc:.3f}")
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nPer-question trace -> {RESULTS_PATH}")
        results_payload["generation"] = {"refusal_correct": refusal_acc, "citation_correct": citation_acc}

    print()


if __name__ == "__main__":
    main()
