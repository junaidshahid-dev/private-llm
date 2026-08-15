"""store.py — the vector index for RAG. Local embeddings, cosine search, abstention support.

Everything here runs on CPU and nothing leaves the machine: embeddings are computed by a local
sentence-transformer, the index is a numpy array on disk. That matters for two reasons — it
works without a GPU or paid API, and (the project's own constraint) no document content is sent
to a third-party service.

For a personal corpus (thousands of chunks) plain numpy cosine is milliseconds per query, so
there is no need for a heavier index. Swap in faiss here later if the corpus grows past ~100k
chunks; the interface would not change.
"""
from __future__ import annotations

import json
import os

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 80MB, fast on CPU, well-tested


class Store:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self.emb: np.ndarray | None = None          # [n, d] float32, L2-normalised
        self.chunks: list[dict] = []                 # parallel to emb: {text, source, ...}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        # normalize so cosine similarity == dot product
        return np.asarray(self.model.encode(texts, normalize_embeddings=True,
                                             show_progress_bar=len(texts) > 200),
                          dtype=np.float32)

    def build(self, chunks: list[dict]) -> None:
        if not chunks:
            raise ValueError("no chunks to index")
        self.chunks = chunks
        self.emb = self.encode([c["text"] for c in chunks])

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "embeddings.npy"), self.emb)
        with open(os.path.join(path, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c) + "\n")
        with open(os.path.join(path, "index_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name, "n_chunks": len(self.chunks),
                       "dim": int(self.emb.shape[1])}, f, indent=2)

    def load(self, path: str) -> "Store":
        meta = json.load(open(os.path.join(path, "index_meta.json"), encoding="utf-8"))
        if meta["model_name"] != self.model_name:
            # embeddings from a different model are not comparable to this model's query vectors
            raise SystemExit(f"index was built with {meta['model_name']!r} but this Store uses "
                             f"{self.model_name!r}. Rebuild the index or match the model.")
        self.emb = np.load(os.path.join(path, "embeddings.npy"))
        self.chunks = [json.loads(l) for l in
                       open(os.path.join(path, "chunks.jsonl"), encoding="utf-8")]
        return self

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the top-k chunks with a cosine score in [-1, 1], highest first."""
        if self.emb is None:
            raise SystemExit("index not loaded")
        qv = self.encode([query])[0]                 # [d]
        scores = self.emb @ qv                        # [n] cosine (both normalised)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]
