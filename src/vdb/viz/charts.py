"""Six distinct chart families for the vector-DB benchmark.

Each chart picks a different axis pair so the reader can spot Pareto
dominance and tradeoffs without doing arithmetic in their head.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from vdb.types import RunResult


def _save(fig: Figure, out: Path, dpi: int = 170) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def recall_vs_qps(results: list[RunResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    palette = plt.get_cmap("tab10")
    seen = sorted({r.backend.value for r in results})
    for i, b in enumerate(seen):
        rows = [r for r in results if r.backend.value == b]
        ax.scatter(
            [r.qps for r in rows],
            [r.recall_at_k for r in rows],
            label=b,
            color=palette(i),
            s=120,
        )
        for r in rows:
            ax.annotate(
                f"k={r.k}",
                (r.qps, r.recall_at_k),
                fontsize=8,
                textcoords="offset points",
                xytext=(6, 6),
            )
    ax.set_xscale("log")
    ax.set_xlabel("QPS (log)")
    ax.set_ylabel("recall@k")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall vs QPS - Pareto frontier")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    return _save(fig, out)


def latency_bars(results: list[RunResult], out: Path) -> Path:
    backs = sorted({r.backend.value for r in results})
    x = np.arange(len(backs))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    p50 = [next(r.latency_ms_p50 for r in results if r.backend.value == b) for b in backs]
    p95 = [next(r.latency_ms_p95 for r in results if r.backend.value == b) for b in backs]
    p99 = [next(r.latency_ms_p99 for r in results if r.backend.value == b) for b in backs]
    ax.bar(x - w, p50, w, label="p50")
    ax.bar(x, p95, w, label="p95")
    ax.bar(x + w, p99, w, label="p99")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(backs, rotation=12)
    ax.set_ylabel("latency per-query (ms, log)")
    ax.set_title("Per-query latency percentiles by backend")
    ax.legend()
    return _save(fig, out)


def build_vs_recall(results: list[RunResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    palette = plt.get_cmap("Dark2")
    seen = sorted({r.backend.value for r in results})
    for i, b in enumerate(seen):
        rows = [r for r in results if r.backend.value == b]
        ax.scatter(
            [r.build_seconds for r in rows],
            [r.recall_at_k for r in rows],
            label=b,
            color=palette(i),
            s=140,
        )
    ax.set_xlabel("index build time (s)")
    ax.set_ylabel("recall@k")
    ax.set_ylim(0, 1.05)
    ax.set_title("Build time vs recall")
    ax.legend()
    return _save(fig, out)


def memory_bar(results: list[RunResult], out: Path) -> Path:
    backs = sorted({r.backend.value for r in results})
    rss = [next(r.peak_rss_mb for r in results if r.backend.value == b) for b in backs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(backs, rss, color="#3b6fa1")
    ax.set_ylabel("peak RSS (MB)")
    ax.set_title("Peak resident memory by backend")
    for i, v in enumerate(rss):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    return _save(fig, out)


def recall_curve_at_ks(results: list[RunResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    palette = plt.get_cmap("tab10")
    seen = sorted({r.backend.value for r in results})
    for i, b in enumerate(seen):
        rows = sorted((r for r in results if r.backend.value == b), key=lambda r: r.k)
        ax.plot(
            [r.k for r in rows],
            [r.recall_at_k for r in rows],
            marker="o",
            linewidth=2,
            label=b,
            color=palette(i),
        )
    ax.set_xlabel("k")
    ax.set_ylabel("recall@k")
    ax.set_title("Recall vs k by backend")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, out)


def latency_distribution_box(results: list[RunResult], out: Path) -> Path:
    backs = sorted({r.backend.value for r in results})
    data = []
    for b in backs:
        rows = [r for r in results if r.backend.value == b]
        data.append(
            [x for r in rows for x in (r.latency_ms_p50, r.latency_ms_p95, r.latency_ms_p99)]
        )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.boxplot(data, tick_labels=backs)
    ax.set_ylabel("latency (ms)")
    ax.set_yscale("log")
    ax.set_title("Latency distribution across (p50, p95, p99) per backend")
    return _save(fig, out)
