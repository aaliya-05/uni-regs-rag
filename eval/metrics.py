"""Retrieval and generation metrics used by run_eval.py."""
from __future__ import annotations


def hit_at_k(retrieved_doc_ids: list[str], expected_doc_id: str | None, k: int) -> bool:
    """True if expected_doc_id appears among the top-k retrieved doc_ids.
    An unanswerable question (expected_doc_id is None) always counts as a
    "hit" here -- there is nothing to find, so retrieval isn't being tested."""
    if expected_doc_id is None:
        return True
    return expected_doc_id in retrieved_doc_ids[:k]


def mrr(retrieved_doc_ids: list[str], expected_doc_id: str | None) -> float:
    """Mean-reciprocal-rank contribution of a single query."""
    if expected_doc_id is None:
        return 1.0
    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == expected_doc_id:
            return 1.0 / i
    return 0.0


def refusal_correct(answer_text: str, refused: bool, expected_doc_id: str | None) -> bool:
    """
    For an unanswerable question, correctness means the system refused.
    For an answerable question, correctness means it did NOT refuse.
    This is the cheapest possible faithfulness proxy and is intentionally
    separate from "is the answer's content right", which needs either a
    human or an LLM-as-judge pass (see run_eval.py --judge).
    """
    if expected_doc_id is None:
        return refused
    return not refused


def citation_present(citations: list, expected_doc_id: str | None) -> bool:
    if expected_doc_id is None:
        return True
    return any(doc_id == expected_doc_id for doc_id, _page in citations)
