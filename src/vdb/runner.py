"""End-to-end runner."""

from __future__ import annotations

import json
from pathlib import Path

from vdb.bench.sweep import run_sweep
from vdb.types import CorpusConfig, IndexBackend, IndexConfig, QuerySet, RunResult
from vdb.viz.charts import (
    build_vs_recall,
    latency_bars,
    latency_distribution_box,
    memory_bar,
    recall_curve_at_ks,
    recall_vs_qps,
)


def run(
    out_dir: Path,
    n_docs: int = 100_000,
    n_queries: int = 10_000,
    dim: int = 128,
    seed: int = 17,
    ks: tuple[int, ...] = (10, 50, 100),
    backends: list[IndexBackend] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = Path("results/figures")
    corpus_cfg = CorpusConfig(n_docs=n_docs, dim=dim, seed=seed)
    qs = QuerySet(n_queries=n_queries, seed=seed + 2)
    if backends is None:
        backends = [
            IndexBackend.FAISS_FLAT,
            IndexBackend.FAISS_HNSW,
            IndexBackend.FAISS_IVF_PQ,
            IndexBackend.CHROMA,
        ]
    results: list[RunResult] = run_sweep(
        corpus_cfg, qs, backends=backends, ks=ks, index_cfg=IndexConfig()
    )
    if results:
        recall_vs_qps(results, figs / "recall_vs_qps.png")
        latency_bars(results, figs / "latency_bars.png")
        build_vs_recall(results, figs / "build_vs_recall.png")
        memory_bar(results, figs / "memory.png")
        recall_curve_at_ks(results, figs / "recall_curve.png")
        latency_distribution_box(results, figs / "latency_box.png")

    summary: dict[str, object] = {
        "corpus": corpus_cfg.model_dump(),
        "queries": qs.model_dump(),
        "ks": list(ks),
        "n_runs": len(results),
        "results": [r.model_dump() for r in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
