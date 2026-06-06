"""Cross-backend benchmark sweep.

Each (backend, k) cell builds the index against the synthesized corpus,
runs the query set, computes recall@k against the gold top-k, captures
latency percentiles, and returns a `RunResult`.
"""

from __future__ import annotations

import logging

from rich.console import Console

from vdb.corpus.synthesize import Synthesized, synthesize
from vdb.index.backends import make_backend
from vdb.metrics.score import benchmark_search, recall_at_k
from vdb.types import CorpusConfig, IndexBackend, IndexConfig, QuerySet, RunResult

log = logging.getLogger("vdb.bench")


def run_sweep(
    corpus_cfg: CorpusConfig,
    qs: QuerySet,
    backends: list[IndexBackend],
    ks: tuple[int, ...] = (10,),
    index_cfg: IndexConfig | None = None,
    console: Console | None = None,
    skip_on_error: bool = True,
) -> list[RunResult]:
    cfg = index_cfg or IndexConfig()
    console = console or Console()
    syn: Synthesized = synthesize(corpus_cfg, qs, gold_k=max(ks))
    results: list[RunResult] = []
    for bkind in backends:
        for k in ks:
            label = f"{bkind.value} @ k={k}"
            console.log(f"-> {label} - build")
            try:
                bk = make_backend(bkind, cfg)
                build_s = bk.build(syn.docs)
            except Exception as e:
                if not skip_on_error:
                    raise
                console.log(f"   {label} BUILD FAILED: {e}")
                continue
            console.log(f"   {label} - search")
            try:
                ids, m = benchmark_search(bk, syn.queries, k)
            except Exception as e:
                if not skip_on_error:
                    raise
                console.log(f"   {label} SEARCH FAILED: {e}")
                continue
            rk = recall_at_k(ids, syn.gold, k)
            results.append(
                RunResult(
                    backend=bkind,
                    n_docs=corpus_cfg.n_docs,
                    n_queries=qs.n_queries,
                    k=k,
                    build_seconds=build_s,
                    search_seconds_total=m["total_seconds"],
                    qps=m["qps"],
                    latency_ms_p50=m["p50_ms"],
                    latency_ms_p95=m["p95_ms"],
                    latency_ms_p99=m["p99_ms"],
                    recall_at_k=rk,
                    peak_rss_mb=m["peak_rss_mb"],
                )
            )
            console.log(
                f"   {label} recall={rk:.3f} qps={m['qps']:.1f} "
                f"p99={m['p99_ms']:.2f}ms build={build_s:.2f}s"
            )
    return results
