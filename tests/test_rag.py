"""RAG layer tests using a fake deterministic embedder."""

from pathlib import Path

import numpy as np
import pytest

from detective.data import Evidence, load_evidence
from detective.rag import VectorIndex, build_index, retrieve, build_chunks


def _fake_embedder(dim: int = 32):
    """A deterministic toy embedder: hashes each text into a `dim`-vector."""
    def fn(texts):
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t):
                out[i, j % dim] += (ord(ch) % 17) - 8
            if not np.any(out[i]):
                out[i, 0] = 1.0
        return out
    return fn


def _tiny_evidence() -> Evidence:
    """Hand-rolled mini Evidence with three interviews. Only `interviews` is used by RAG."""
    ev = Evidence(
        persons=[],
        licenses=[],
        incomes=[],
        members=[],
        checkins=[],
        events=[],
        interviews=[
            {"person_id": 1, "transcript": "I saw the killer enter the gym."},
            {"person_id": 2, "transcript": "The car had a license plate starting with H42."},
            {"person_id": 3, "transcript": "Nothing unusual that day, just rain."},
        ],
    )
    return ev


def test_build_chunks_skips_empty():
    ev = _tiny_evidence()
    ev.interviews.append({"person_id": 4, "transcript": ""})
    ev.interviews.append({"person_id": 5, "transcript": None})
    chunks = build_chunks(ev)
    assert [c.person_id for c in chunks] == [1, 2, 3]


def test_index_build_and_search(tmp_path: Path):
    ev = _tiny_evidence()
    embed = _fake_embedder()
    idx = build_index(ev, embed, cache_path=tmp_path / "emb.npz")
    # All vectors L2-normalised
    norms = np.linalg.norm(idx.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    hits = retrieve(idx, "license plate H42", embed, top_k=1)
    assert hits[0]["person_id"] == 2


def test_cache_round_trip(tmp_path: Path, mocker):
    ev = _tiny_evidence()
    cache = tmp_path / "emb.npz"
    embed = _fake_embedder()
    spy = mocker.Mock(side_effect=embed)
    idx1 = build_index(ev, spy, cache_path=cache)
    assert spy.call_count == 1
    assert cache.exists()

    # Second build should hit the cache and not call embedder again
    spy2 = mocker.Mock(side_effect=embed)
    idx2 = build_index(ev, spy2, cache_path=cache)
    assert spy2.call_count == 0
    np.testing.assert_allclose(idx1.embeddings, idx2.embeddings)


def test_cache_invalidated_on_corpus_change(tmp_path: Path, mocker):
    ev = _tiny_evidence()
    cache = tmp_path / "emb.npz"
    embed = _fake_embedder()
    build_index(ev, embed, cache_path=cache)

    # Change the corpus → content hash changes → cache must be rebuilt
    ev.interviews[0]["transcript"] = "Different testimony entirely."
    spy = mocker.Mock(side_effect=embed)
    build_index(ev, spy, cache_path=cache)
    assert spy.call_count == 1


def test_real_corpus_chunk_count_matches_data():
    ev = load_evidence()
    chunks = build_chunks(ev)
    # 4991 interview rows, all non-empty (verified during exploration)
    assert len(chunks) == 4991
    assert all(c.text for c in chunks)
