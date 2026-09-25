"""
Pluggable embedding backends.

- "sentence_transformers": real sentence embeddings. Best choice, but the
  first run downloads a model from Hugging Face, which needs open internet
  (this project was built inside a sandboxed dev environment that blocks
  that download -- see README "Environment notes"). Use this on your own
  machine or in Colab.
- "tfidf_svd": a fully offline fallback. TF-IDF vectors compressed with
  truncated SVD (i.e. classic LSA) give a cheap, dependency-light
  approximation of a dense embedding, good enough to develop and test the
  whole retrieval pipeline without any network access. It is measurably
  weaker than a real sentence embedding -- the eval harness quantifies
  exactly how much weaker, which is the point: you should be able to show
  a reviewer the before/after numbers when you swap it out.

Both backends expose the same `.fit(corpus)` / `.encode(texts)` interface
so retrieve.py never needs to know which one is active.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class TfidfSvdEmbedder:
    def __init__(self, n_components: int = 256, random_state: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(
            max_features=50_000, ngram_range=(1, 2), sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._fitted = False

    def fit(self, corpus: list[str]):
        tfidf = self.vectorizer.fit_transform(corpus)
        # SVD components can exceed the corpus/feature rank for tiny corpora.
        self.svd.n_components = min(self.svd.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self.svd.n_components = max(self.svd.n_components, 2)
        vecs = self.svd.fit_transform(tfidf)
        self._fitted = True
        return self._normalize(vecs)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit(corpus) before .encode() for TfidfSvdEmbedder.")
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        return self._normalize(vecs)

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd}, f)

    @classmethod
    def load(cls, path: Path) -> "TfidfSvdEmbedder":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.vectorizer = data["vectorizer"]
        obj.svd = data["svd"]
        obj._fitted = True
        return obj


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._fitted = True  # nothing to fit -- pretrained

    def fit(self, corpus: list[str]) -> np.ndarray:
        return self.encode(corpus)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))

    def save(self, path: Path):
        pass  # stateless wrapper around a pretrained model; nothing to persist

    @classmethod
    def load(cls, path: Path, model_name: str) -> "SentenceTransformerEmbedder":
        return cls(model_name)


def get_embedder(backend: str, model_name: str, svd_dims: int):
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name)
    if backend == "tfidf_svd":
        return TfidfSvdEmbedder(n_components=svd_dims)
    raise ValueError(f"Unknown embedding backend: {backend}")
