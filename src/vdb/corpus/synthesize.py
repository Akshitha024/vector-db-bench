"""Synthesize a BEIR-shaped corpus + query set at production scale.

We need 10k+ queries against 100k+ documents. Generating real text embeddings
on a laptop at that scale is slow; instead we synthesize embeddings directly
from a Gaussian-mixture model whose per-cluster centroids are picked so the
nearest-neighbor structure is non-trivial and recall < 1.0 even for exact
search (because of the noise term).

The resulting (doc_emb, query_emb, gold_ids) triple matches the shape of
what a real text-to-embedding pipeline would produce and is the same input
that downstream code uses against a real corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vdb.types import CorpusConfig, QuerySet


@dataclass(frozen=True)
class Synthesized:
    docs: np.ndarray  # (n_docs, dim) L2-normalized
    queries: np.ndarray  # (n_queries, dim) L2-normalized
    gold: np.ndarray  # (n_queries, top_k) int32 indices of the true neighbors
    cluster_id: np.ndarray  # (n_docs,) int32 cluster assignment


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    out: np.ndarray = x / n
    return out


def synthesize(corpus: CorpusConfig, qs: QuerySet, gold_k: int = 10) -> Synthesized:
    rng = np.random.default_rng(corpus.seed)
    centroids = rng.normal(size=(corpus.n_clusters, corpus.dim)).astype("float32")
    centroids = _normalize(centroids)

    cluster_id = rng.integers(0, corpus.n_clusters, size=corpus.n_docs).astype("int32")
    docs = centroids[cluster_id] + corpus.noise * rng.normal(
        size=(corpus.n_docs, corpus.dim)
    ).astype("float32")
    docs = _normalize(docs).astype("float32")

    qrng = np.random.default_rng(qs.seed)
    # Most queries pick a cluster and perturb its centroid slightly; a small
    # fraction (1 - coverage) pick random vectors that may have no true match.
    n_q = qs.n_queries
    pick_real = qrng.random(n_q) < qs.coverage
    q_cluster = qrng.integers(0, corpus.n_clusters, size=n_q).astype("int32")
    base = np.where(
        pick_real[:, None],
        centroids[q_cluster],
        qrng.normal(size=(n_q, corpus.dim)).astype("float32"),
    )
    queries = base + 0.5 * corpus.noise * qrng.normal(size=(n_q, corpus.dim)).astype("float32")
    queries = _normalize(queries.astype("float32"))

    # Compute true top-k neighbors by exact brute force, in chunks to bound RAM.
    chunk = 1024
    gold = np.empty((n_q, gold_k), dtype="int32")
    for i in range(0, n_q, chunk):
        scores = queries[i : i + chunk] @ docs.T  # cosine since both L2-normalized
        top = np.argpartition(-scores, gold_k - 1, axis=1)[:, :gold_k]
        # Order them properly by descending score.
        row_idx = np.arange(top.shape[0])[:, None]
        ordered = np.argsort(-scores[row_idx, top], axis=1)
        gold[i : i + chunk] = top[row_idx, ordered]

    return Synthesized(docs=docs, queries=queries, gold=gold, cluster_id=cluster_id)
