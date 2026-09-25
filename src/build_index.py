"""
Build the BM25 and dense indexes from data/processed/chunks.jsonl.

    python -m src.build_index
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    CHUNKS_PATH,
    BM25_INDEX_PATH,
    DENSE_INDEX_PATH,
    RetrievalConfig,
)
from src.embeddings import get_embedder  # noqa: E402


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise SystemExit("Run `python -m src.chunk` first.")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    cfg = RetrievalConfig()
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"Building indexes for {len(texts)} chunks "
          f"(embedding_backend={cfg.embedding_backend!r})")

    # --- BM25 (lexical) ---
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": [c["chunk_id"] for c in chunks]}, f)
    print(f"  BM25 index      -> {BM25_INDEX_PATH}")

    # --- Dense (semantic-ish) ---
    embedder = get_embedder(cfg.embedding_backend, cfg.embedding_model_name, cfg.svd_dims)
    vectors = embedder.fit(texts)
    np.savez(
        DENSE_INDEX_PATH,
        vectors=vectors,
        chunk_ids=np.array([c["chunk_id"] for c in chunks]),
    )
    embedder_state_path = DENSE_INDEX_PATH.with_suffix(".embedder.pkl")
    embedder.save(embedder_state_path)
    print(f"  Dense index     -> {DENSE_INDEX_PATH}  (dim={vectors.shape[1]})")
    print(f"  Embedder state  -> {embedder_state_path}")


if __name__ == "__main__":
    main()
