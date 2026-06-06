"""Integration tests: every FAISS backend recovers high recall@10 at small scale."""

from __future__ import annotations

import numpy as np
import pytest

from vdb.index.backends import make_backend
from vdb.metrics.score import recall_at_k
from vdb.types import IndexBackend, IndexConfig


def _data(n: int = 2_000, dim: int = 64, n_q: int = 200, seed: int = 11):
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(20, dim)).astype("float32")
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    cluster = rng.integers(0, 20, size=n)
    docs = centroids[cluster] + 0.05 * rng.normal(size=(n, dim)).astype("float32")
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    qrng = np.random.default_rng(seed + 2)
    q_cluster = qrng.integers(0, 20, size=n_q)
    queries = centroids[q_cluster] + 0.02 * qrng.normal(size=(n_q, dim)).astype("float32")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    scores = queries @ docs.T
    gold = np.argsort(-scores, axis=1)[:, :10].astype("int32")
    return docs.astype("float32"), queries.astype("float32"), gold


@pytest.mark.integration
def test_faiss_flat_matches_brute_force() -> None:
    pytest.importorskip("faiss")
    docs, q, gold = _data()
    bk = make_backend(IndexBackend.FAISS_FLAT, IndexConfig())
    bk.build(docs)
    ids, _ = bk.search(q, 10)
    assert recall_at_k(ids, gold, 10) > 0.99


@pytest.mark.integration
def test_faiss_hnsw_high_recall() -> None:
    pytest.importorskip("faiss")
    docs, q, gold = _data()
    bk = make_backend(
        IndexBackend.FAISS_HNSW, IndexConfig(hnsw_M=32, hnsw_efConstruction=200, hnsw_efSearch=128)
    )
    bk.build(docs)
    ids, _ = bk.search(q, 10)
    assert recall_at_k(ids, gold, 10) > 0.85


@pytest.mark.integration
def test_faiss_ivf_pq_runs() -> None:
    pytest.importorskip("faiss")
    docs, q, _gold = _data()
    bk = make_backend(IndexBackend.FAISS_IVF_PQ, IndexConfig(ivf_nlist=64, pq_m=8, pq_nbits=8))
    bk.build(docs)
    ids, _ = bk.search(q, 10)
    assert ids.shape == (200, 10)
    # PQ trades recall for memory, so we don't gate on a tight recall floor.
