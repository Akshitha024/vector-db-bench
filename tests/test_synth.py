"""Tests on the synthesized corpus + query set."""

from __future__ import annotations

import numpy as np
import pytest

from vdb.corpus.synthesize import synthesize
from vdb.types import CorpusConfig, QuerySet


@pytest.mark.parametrize("seed", [11, 13, 17, 23, 29])
def test_synthesize_deterministic(seed: int) -> None:
    cfg = CorpusConfig(n_docs=500, dim=32, seed=seed)
    qs = QuerySet(n_queries=100, seed=seed + 1)
    a = synthesize(cfg, qs, gold_k=10)
    b = synthesize(cfg, qs, gold_k=10)
    assert np.allclose(a.docs, b.docs)
    assert np.allclose(a.queries, b.queries)
    assert np.array_equal(a.gold, b.gold)


def test_synthesize_norms() -> None:
    cfg = CorpusConfig(n_docs=1024, dim=64)
    qs = QuerySet(n_queries=256)
    syn = synthesize(cfg, qs, gold_k=10)
    doc_norms = np.linalg.norm(syn.docs, axis=1)
    qry_norms = np.linalg.norm(syn.queries, axis=1)
    assert np.allclose(doc_norms, 1.0, atol=1e-5)
    assert np.allclose(qry_norms, 1.0, atol=1e-5)


def test_synthesize_gold_shape() -> None:
    cfg = CorpusConfig(n_docs=1024, dim=32)
    qs = QuerySet(n_queries=200)
    syn = synthesize(cfg, qs, gold_k=20)
    assert syn.gold.shape == (200, 20)
    # Gold ids must be in range.
    assert syn.gold.min() >= 0
    assert syn.gold.max() < 1024
    # Each row should be unique.
    for row in syn.gold:
        assert len(set(row.tolist())) == row.shape[0]


def test_synthesize_at_10k_runs_fast() -> None:
    """Sanity: 10k corpus + 1k queries should synthesize in a few seconds."""
    import time as _t

    cfg = CorpusConfig(n_docs=10_000, dim=64)
    qs = QuerySet(n_queries=1_000)
    t0 = _t.perf_counter()
    syn = synthesize(cfg, qs, gold_k=10)
    elapsed = _t.perf_counter() - t0
    assert syn.docs.shape == (10_000, 64)
    assert syn.queries.shape == (1_000, 64)
    assert elapsed < 30, f"synthesize too slow: {elapsed:.1f}s"


@pytest.mark.parametrize("coverage", [0.5, 0.8, 0.95, 1.0])
def test_query_coverage_parameter(coverage: float) -> None:
    """Higher coverage means more queries near a cluster centroid."""
    cfg = CorpusConfig(n_docs=2_000, dim=32)
    qs = QuerySet(n_queries=400, coverage=coverage)
    syn = synthesize(cfg, qs, gold_k=10)
    assert syn.queries.shape[0] == 400
