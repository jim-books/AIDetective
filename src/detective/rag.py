"""RAG over interview transcripts.

Each interview row becomes a single chunk (transcripts are short — max ~250
chars). Embeddings are persisted to `.cache/embeddings.npz` keyed by a
content hash so re-runs do not re-pay the embedding API. Retrieval is cosine
top-k against L2-normalised vectors.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .data import Evidence

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
CACHE_PATH = CACHE_DIR / "embeddings.npz"
EMBED_MODEL = "text-embedding-3-small"  # default / standard OpenAI name
EMBED_BATCH = 500


def _embed_model() -> str:
    """Deployment/model name for embeddings. Reads env so Azure overrides work."""
    return os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", EMBED_MODEL)


@dataclass
class Chunk:
    chunk_id: str
    person_id: int
    text: str


def build_chunks(ev: Evidence) -> list[Chunk]:
    """One chunk per non-empty interview, in deterministic order."""
    rows = sorted(
        (i for i in ev.interviews if i.get("transcript")),
        key=lambda i: i["person_id"],
    )
    return [Chunk(chunk_id=f"interview:{r['person_id']}", person_id=r["person_id"], text=r["transcript"]) for r in rows]


def _content_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    h.update(EMBED_MODEL.encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(b"\x00")
        h.update(c.text.encode())
        h.update(b"\x01")
    return h.hexdigest()


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# Embedder callable: takes a list[str], returns np.ndarray of shape (n, dim).
EmbedFn = Callable[[list[str]], np.ndarray]


def openai_embedder(client: Any, model: str | None = None) -> EmbedFn:
    """Return an EmbedFn backed by the OpenAI (or Azure OpenAI) embeddings API.

    `model` is the model or deployment name. Defaults to the
    AZURE_OPENAI_EMBED_DEPLOYMENT env var, falling back to EMBED_MODEL.
    """
    deploy = model or _embed_model()

    def _embed(texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i : i + EMBED_BATCH]
            resp = client.embeddings.create(model=deploy, input=batch)
            out.extend(d.embedding for d in resp.data)
        return np.asarray(out, dtype=np.float32)

    return _embed


@dataclass
class VectorIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray  # shape (n, dim), L2-normalised
    person_ids: np.ndarray  # shape (n,)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Cosine top-k. `query_vec` may be 1D or 2D; will be normalised."""
        q = query_vec.reshape(-1).astype(np.float32)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:top_k]
        return [
            {
                "chunk_id": self.chunks[i].chunk_id,
                "person_id": int(self.chunks[i].person_id),
                "text": self.chunks[i].text,
                "score": float(scores[i]),
            }
            for i in top
        ]


def build_index(
    ev: Evidence,
    embed_fn: EmbedFn,
    *,
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> VectorIndex:
    """Build (or load from cache) the interview vector index."""
    chunks = build_chunks(ev)
    cache_path = cache_path if cache_path is not None else CACHE_PATH
    digest = _content_hash(chunks)

    if use_cache and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if str(cached["content_hash"]) == digest and cached["embeddings"].shape[0] == len(chunks):
                return VectorIndex(
                    chunks=chunks,
                    embeddings=cached["embeddings"],
                    person_ids=cached["person_ids"],
                )
        except (KeyError, ValueError, OSError):
            pass  # cache corrupt — recompute

    raw = embed_fn([c.text for c in chunks])
    embeddings = _normalise(raw)
    person_ids = np.array([c.person_id for c in chunks], dtype=np.int64)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            embeddings=embeddings,
            person_ids=person_ids,
            content_hash=np.array(digest),
        )

    return VectorIndex(chunks=chunks, embeddings=embeddings, person_ids=person_ids)


def retrieve(index: VectorIndex, query: str, embed_fn: EmbedFn, top_k: int = 5) -> list[dict[str, Any]]:
    """Embed `query` and return top-k matching interview chunks."""
    qv = embed_fn([query])[0]
    return index.search(qv, top_k=top_k)
