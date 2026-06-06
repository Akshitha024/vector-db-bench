"""End-to-end runner smoke test (small scale to keep CI fast)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vdb.runner import run
from vdb.types import IndexBackend


@pytest.mark.slow
def test_runner_small_scale_smoke(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    res = run(
        tmp_path / "out",
        n_docs=5_000,
        n_queries=500,
        dim=64,
        ks=(10,),
        backends=[IndexBackend.FAISS_FLAT, IndexBackend.FAISS_HNSW],
    )
    assert res["n_runs"] >= 2
    assert (tmp_path / "out" / "summary.json").exists()
