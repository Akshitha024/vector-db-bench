"""Type definitions for the vector-DB benchmark."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class IndexBackend(StrEnum):
    """Vector-DB backend under test."""

    FLAT_NUMPY = "flat_numpy"
    FAISS_FLAT = "faiss_flat"
    FAISS_HNSW = "faiss_hnsw"
    FAISS_IVF_PQ = "faiss_ivf_pq"
    CHROMA = "chroma"


class CorpusConfig(BaseModel):
    """Synthetic corpus parameters - calibrated to match real BEIR distributions."""

    n_docs: int = Field(..., ge=1)
    dim: int = Field(..., ge=8)
    n_clusters: int = Field(default=128, ge=2)
    noise: float = Field(default=0.05, ge=0, le=1)
    seed: int = 17


class QuerySet(BaseModel):
    """Query set parameters."""

    n_queries: int = Field(..., ge=1)
    seed: int = 19
    # Fraction of queries with at least one true positive in the corpus.
    coverage: float = Field(default=0.95, ge=0, le=1)


class IndexConfig(BaseModel):
    """Hyperparameters for the index backends that take them."""

    model_config = ConfigDict(extra="forbid")
    hnsw_M: int = 32
    hnsw_efConstruction: int = 200
    hnsw_efSearch: int = 64
    ivf_nlist: int = 256
    ivf_nprobe: int = 16
    pq_m: int = 8
    pq_nbits: int = 8


class RunResult(BaseModel):
    """One (backend, k) measurement."""

    backend: IndexBackend
    n_docs: int
    n_queries: int
    k: int
    build_seconds: float
    search_seconds_total: float
    qps: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    recall_at_k: float
    peak_rss_mb: float
    notes: str = ""
