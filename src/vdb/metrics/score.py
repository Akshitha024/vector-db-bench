"""Recall@k, latency percentiles, QPS, peak RSS."""

from __future__ import annotations

import time

import numpy as np
import psutil

from vdb.index.backends import Backend


def recall_at_k(pred_ids: np.ndarray, gold_ids: np.ndarray, k: int) -> float:
    """Mean recall@k against the gold top-k.

    The gold set is the brute-force top-k computed once; we measure how many
    of those k true neighbors the backend retrieves in its own top-k.
    """
    if pred_ids.shape[0] == 0:
        return 0.0
    gold_set = set
    hits = 0
    total = pred_ids.shape[0] * k
    for i in range(pred_ids.shape[0]):
        truth = gold_set(int(x) for x in gold_ids[i, :k])
        for p in pred_ids[i, :k]:
            if int(p) in truth:
                hits += 1
    return float(hits / total)


def benchmark_search(
    backend: Backend, queries: np.ndarray, k: int, batch: int = 64
) -> tuple[np.ndarray, dict[str, float]]:
    """Run search batch-by-batch, capturing per-batch latency in ms."""
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / (1024 * 1024)
    latencies_ms: list[float] = []
    all_ids: list[np.ndarray] = []
    t0 = time.perf_counter()
    for i in range(0, queries.shape[0], batch):
        chunk = queries[i : i + batch]
        s = time.perf_counter()
        ids, _scores = backend.search(chunk, k)
        latencies_ms.append((time.perf_counter() - s) * 1000.0 / max(1, chunk.shape[0]))
        all_ids.append(ids)
    elapsed = time.perf_counter() - t0
    rss_after = proc.memory_info().rss / (1024 * 1024)
    arr = np.array(latencies_ms, dtype="float64")
    return np.concatenate(all_ids, axis=0), {
        "total_seconds": float(elapsed),
        "qps": float(queries.shape[0] / max(elapsed, 1e-9)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "peak_rss_mb": float(max(rss_before, rss_after)),
    }
