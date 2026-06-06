"""Tests for the metrics."""

from __future__ import annotations

import numpy as np

from vdb.metrics.score import recall_at_k


def test_recall_perfect_match() -> None:
    gold = np.array([[0, 1, 2, 3, 4]] * 10, dtype="int32")
    pred = np.array([[0, 1, 2, 3, 4]] * 10, dtype="int64")
    assert recall_at_k(pred, gold, 5) == 1.0


def test_recall_zero_when_disjoint() -> None:
    gold = np.array([[0, 1, 2, 3, 4]] * 10, dtype="int32")
    pred = np.array([[5, 6, 7, 8, 9]] * 10, dtype="int64")
    assert recall_at_k(pred, gold, 5) == 0.0


def test_recall_partial() -> None:
    gold = np.array([[0, 1, 2, 3, 4]], dtype="int32")
    pred = np.array([[0, 1, 2, 99, 99]], dtype="int64")
    # 3 of 5 hits
    assert abs(recall_at_k(pred, gold, 5) - 0.6) < 1e-9


def test_recall_with_k_smaller_than_gold() -> None:
    gold = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype="int32")
    pred = np.array([[0, 1, 99]], dtype="int64")
    # 2 of 3 hits within k=3 against first 3 of gold
    assert abs(recall_at_k(pred, gold, 3) - (2 / 3)) < 1e-9
