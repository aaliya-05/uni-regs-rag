"""
Hybrid retrieval: BM25 (lexical) + dense (semantic) with score-fusion,
plus an optional reranking pass.

Why hybrid at all: BM25 wins on exact terms a student will actually type
("IDC 1202", "80% attendance", "Circular No. 946"); dense search wins on
paraphrase ("What happens if I miss my final exam because I was sick?"
when the source says "absent ... without proper consent ... an E grade
will be issued"). The eval harness (eval/run_eval.py) measures each
retriever alone and the hybrid, so the gain from combining them is a
number, not an assertion.
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    CHUNKS_PATH,
    BM25_INDEX_PATH,
    DENSE_INDEX_PATH,
    RetrievalConfig,
)
from src.embeddings import get_embedder, TfidfSvdEmbedder  # noqa: E402


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    doc_type: str
    effective_date: str
    supersedes: list
    page_number: int
    text: str
    faculty: str | None = None
    approx_pages: bool = False
    bm25_score: float = 0.0
    dense_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None


def _minmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class Retriever:
    def __init__(self, cfg: RetrievalConfig | None = None):
        self.cfg = cfg or RetrievalConfig()
        self.chunks: dict[str, dict] = {}
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                self.chunks[c["chunk_id"]] = c

        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_chunk_ids = bm25_data["chunk_ids"]

        dense_data = np.load(DENSE_INDEX_PATH, allow_pickle=True)
        self.dense_vectors = dense_data["vectors"]
        self.dense_chunk_ids = list(dense_data["chunk_ids"])

        if self.cfg.embedding_backend == "tfidf_svd":
            self.embedder = TfidfSvdEmbedder.load(
                DENSE_INDEX_PATH.with_suffix(".embedder.pkl")
            )
        else:
            self.embedder = get_embedder(
                self.cfg.embedding_backend, self.cfg.embedding_model_name, self.cfg.svd_dims
            )

        self._reranker = None
        if self.cfg.use_reranker:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.cfg.reranker_model_name)

    # -- individual retrievers -------------------------------------------------
    def bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.bm25_chunk_ids[i], float(scores[i])) for i in order]

    def dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        qvec = self.embedder.encode([query])[0]
        sims = self.dense_vectors @ qvec  # vectors are pre-normalized
        order = np.argsort(sims)[::-1][:top_k]
        return [(self.dense_chunk_ids[i], float(sims[i])) for i in order]

    # -- lexical-overlap reranker fallback (no network / no cross-encoder) ----
    @staticmethod
    def _lexical_overlap_score(query: str, text: str) -> float:
        q_terms = set(query.lower().split())
        t_terms = set(text.lower().split())
        if not q_terms:
            return 0.0
        return len(q_terms & t_terms) / len(q_terms)

    # -- public API -------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        cfg = self.cfg
        top_k = top_k or cfg.top_k_final

        bm25_hits = dict(self.bm25_search(query, cfg.top_k_bm25))
        dense_hits = dict(self.dense_search(query, cfg.top_k_dense))

        candidate_ids = set(bm25_hits) | set(dense_hits)
        bm25_arr = np.array([bm25_hits.get(cid, 0.0) for cid in candidate_ids])
        dense_arr = np.array([dense_hits.get(cid, 0.0) for cid in candidate_ids])
        bm25_norm = _minmax(bm25_arr)
        dense_norm = _minmax(dense_arr)

        fused = {}
        for cid, b, d in zip(candidate_ids, bm25_norm, dense_norm):
            fused[cid] = cfg.bm25_weight * b + cfg.dense_weight * d

        ranked_ids = sorted(fused, key=lambda c: fused[c], reverse=True)[: cfg.top_k_fused]

        results = []
        for cid in ranked_ids:
            c = self.chunks[cid]
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=c["doc_id"],
                    title=c["title"],
                    doc_type=c["doc_type"],
                    effective_date=c["effective_date"],
                    supersedes=c["supersedes"],
                    page_number=c["page_number"],
                    text=c["text"],
                    faculty=c.get("faculty"),
                    approx_pages=c.get("approx_pages", False),
                    bm25_score=bm25_hits.get(cid, 0.0),
                    dense_score=dense_hits.get(cid, 0.0),
                    fused_score=fused[cid],
                )
            )

        if self._reranker is not None:
            pairs = [(query, r.text) for r in results]
            scores = self._reranker.predict(pairs)
            for r, s in zip(results, scores):
                r.rerank_score = float(s)
            results.sort(key=lambda r: r.rerank_score, reverse=True)
        else:
            # Cheap fallback re-ordering: break fused-score near-ties using
            # raw lexical overlap, which tends to demote generic
            # boilerplate paragraphs that only matched on stopwords.
            for r in results:
                r.rerank_score = self._lexical_overlap_score(query, r.text)
            results.sort(key=lambda r: (round(r.fused_score, 3), r.rerank_score), reverse=True)

        # Prefer the superseding document when two results cover the same
        # ground and one explicitly supersedes the other (see config.py).
        results = self._prefer_superseding(results)
        return results[:top_k]

    @staticmethod
    def _prefer_superseding(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        doc_ids_present = {r.doc_id for r in results}
        superseded = set()
        for r in results:
            superseded.update(d for d in r.supersedes if d in doc_ids_present)
        if not superseded:
            return results
        # Keep superseded-doc chunks but push them down, rather than
        # dropping them -- a good answer often says "the 2014 rule was X,
        # but the 2025 by-law now says Y."
        return sorted(results, key=lambda r: r.doc_id in superseded)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top_k", type=int, default=4)
    args = parser.parse_args()

    retriever = Retriever()
    hits = retriever.retrieve(args.query, top_k=args.top_k)
    for h in hits:
        print(f"\n[{h.doc_id} p.{h.page_number}] fused={h.fused_score:.3f} "
              f"bm25={h.bm25_score:.2f} dense={h.dense_score:.2f}")
        print(h.text[:300].replace("\n", " ") + ("..." if len(h.text) > 300 else ""))
