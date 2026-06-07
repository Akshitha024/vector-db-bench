"""Typer CLI for the vector-DB benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vdb.runner import run
from vdb.types import IndexBackend

app = typer.Typer(no_args_is_help=True, help="Production vector-DB benchmark harness.")
console = Console()


@app.command()
def info() -> None:
    console.print("vector-db-bench: see `vdb bench --help` and `vdb report --help`.")


@app.command()
def bench(
    out_dir: Path = typer.Option(Path("runs/latest")),
    n_docs: int = typer.Option(100_000, help="Corpus size"),
    n_queries: int = typer.Option(9_847, help="Number of queries"),
    dim: int = typer.Option(128),
    backends: str = typer.Option(
        "faiss_flat,faiss_hnsw,faiss_ivf_pq",
        help="Comma-separated subset of: flat_numpy,faiss_flat,faiss_hnsw,faiss_ivf_pq,chroma",
    ),
    ks: str = typer.Option("10,50,100"),
    seed: int = typer.Option(17),
) -> None:
    """Run the bench: sweep backends x k over a {n_docs} x {n_queries} workload."""
    backend_list = [IndexBackend(b.strip()) for b in backends.split(",") if b.strip()]
    k_list = tuple(int(x) for x in ks.split(","))
    res = run(
        out_dir,
        n_docs=n_docs,
        n_queries=n_queries,
        dim=dim,
        seed=seed,
        ks=k_list,
        backends=backend_list,
    )
    console.print_json(json.dumps({"n_runs": res["n_runs"]}, default=str))


@app.command()
def report(out_dir: Path = typer.Option(Path("runs/latest"))) -> None:
    """Pretty-print the run table."""
    data = json.loads((out_dir / "summary.json").read_text())
    table = Table(title=f"Bench results ({data['n_runs']} runs)")
    for col in ("backend", "k", "recall@k", "QPS", "p50 ms", "p99 ms", "build s", "RSS MB"):
        table.add_column(col)
    for r in data["results"]:
        table.add_row(
            r["backend"],
            str(r["k"]),
            f"{r['recall_at_k']:.3f}",
            f"{r['qps']:.1f}",
            f"{r['latency_ms_p50']:.2f}",
            f"{r['latency_ms_p99']:.2f}",
            f"{r['build_seconds']:.2f}",
            f"{r['peak_rss_mb']:.1f}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
