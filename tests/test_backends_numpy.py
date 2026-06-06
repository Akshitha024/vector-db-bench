"""Tests on the flat-numpy backend (the ground truth)."""

from __future__ import annotations

import numpy as np

from vdb.index.backends import FlatNumpy


def test_flat_numpy_search_returns_sorted_scores() -> None:
    docs = np.random.default_rng(0).normal(size=(200, 32)).astype("float32")
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    q = np.random.default_rng(1).normal(size=(5, 32)).astype("float32")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    bk = FlatNumpy()
    bk.build(docs)
    ids, scores = bk.search(q, k=10)
    assert ids.shape == (5, 10)
    assert scores.shape == (5, 10)
    for row in scores:
        diffs = np.diff(row)
        assert np.all(diffs <= 1e-5), "scores should be non-increasing"


def test_flat_numpy_matches_argpartition_ground_truth() -> None:
    """The flat backend must return the same neighbors as a direct numpy
    top-k call. This is the test that locks the ground truth."""
    docs = np.random.default_rng(2).normal(size=(500, 16)).astype("float32")
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    q = np.random.default_rng(3).normal(size=(8, 16)).astype("float32")
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    bk = FlatNumpy()
    bk.build(docs)
    pred_ids, _ = bk.search(q, k=10)

    scores = q @ docs.T
    expected = np.argsort(-scores, axis=1)[:, :10]
    # Compare as sets per row since ties in scores may reorder.
    for p, e in zip(pred_ids, expected, strict=True):
        assert set(p.tolist()) == set(e.tolist())
