"""Backend adapters: same `(build, search)` contract across FAISS variants,
brute-force numpy, and Chroma. Each backend implements the protocol
implicitly so the bench runner can hand them around interchangeably.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vdb.types import IndexBackend, IndexConfig


class Backend(Protocol):
    name: IndexBackend

    def build(self, docs: np.ndarray) -> float: ...
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


# ----- flat numpy (the ground-truth baseline) -------------------------------


@dataclass
class FlatNumpy:
    name: IndexBackend = IndexBackend.FLAT_NUMPY
    _docs: np.ndarray | None = None

    def build(self, docs: np.ndarray) -> float:
        t = time.perf_counter()
        self._docs = docs.astype("float32", copy=False)
        return time.perf_counter() - t

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._docs is not None
        chunk = 512
        all_ids = np.empty((queries.shape[0], k), dtype="int64")
        all_scores = np.empty((queries.shape[0], k), dtype="float32")
        for i in range(0, queries.shape[0], chunk):
            q = queries[i : i + chunk]
            scores = q @ self._docs.T
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            row = np.arange(top.shape[0])[:, None]
            order = np.argsort(-scores[row, top], axis=1)
            ids = top[row, order]
            all_ids[i : i + chunk] = ids
            all_scores[i : i + chunk] = scores[row, ids]
        return all_ids, all_scores


# ----- FAISS variants -------------------------------------------------------


@dataclass
class FaissFlat:
    name: IndexBackend = IndexBackend.FAISS_FLAT
    _index: object | None = None

    def build(self, docs: np.ndarray) -> float:
        import faiss

        t = time.perf_counter()
        idx = faiss.IndexFlatIP(docs.shape[1])
        idx.add(docs.astype("float32"))
        self._index = idx
        return time.perf_counter() - t

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._index is not None
        scores, ids = self._index.search(queries.astype("float32"), k)  # type: ignore[attr-defined]
        return ids.astype("int64"), scores.astype("float32")


@dataclass
class FaissHNSW:
    cfg: IndexConfig
    name: IndexBackend = IndexBackend.FAISS_HNSW
    _index: object | None = None

    def build(self, docs: np.ndarray) -> float:
        import faiss

        t = time.perf_counter()
        idx = faiss.IndexHNSWFlat(docs.shape[1], self.cfg.hnsw_M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = self.cfg.hnsw_efConstruction
        idx.hnsw.efSearch = self.cfg.hnsw_efSearch
        idx.add(docs.astype("float32"))
        self._index = idx
        return time.perf_counter() - t

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._index is not None
        scores, ids = self._index.search(queries.astype("float32"), k)  # type: ignore[attr-defined]
        return ids.astype("int64"), scores.astype("float32")


@dataclass
class FaissIVFPQ:
    cfg: IndexConfig
    name: IndexBackend = IndexBackend.FAISS_IVF_PQ
    _index: object | None = None

    def build(self, docs: np.ndarray) -> float:
        import faiss

        t = time.perf_counter()
        d = docs.shape[1]
        # IVF needs the PQ subvector size m to divide d evenly.
        m = self.cfg.pq_m
        while d % m != 0 and m > 1:
            m -= 1
        quantizer = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFPQ(quantizer, d, self.cfg.ivf_nlist, m, self.cfg.pq_nbits)
        idx.train(docs.astype("float32"))
        idx.add(docs.astype("float32"))
        idx.nprobe = self.cfg.ivf_nprobe
        self._index = idx
        return time.perf_counter() - t

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._index is not None
        scores, ids = self._index.search(queries.astype("float32"), k)  # type: ignore[attr-defined]
        return ids.astype("int64"), scores.astype("float32")


# ----- Chroma --------------------------------------------------------------


@dataclass
class ChromaBackend:
    persist_dir: str = ":memory:"
    name: IndexBackend = IndexBackend.CHROMA
    _coll: object | None = None

    def build(self, docs: np.ndarray) -> float:
        import chromadb
        from chromadb.config import Settings

        t = time.perf_counter()
        if self.persist_dir == ":memory:":
            client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        else:
            client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        import contextlib

        with contextlib.suppress(Exception):
            client.delete_collection("vdb_bench")
        coll = client.create_collection("vdb_bench", metadata={"hnsw:space": "ip"})
        # Chroma needs string ids and accepts batches; insert in 5k chunks.
        n = docs.shape[0]
        chunk = 5000
        for i in range(0, n, chunk):
            j = min(n, i + chunk)
            coll.add(
                ids=[str(x) for x in range(i, j)],
                embeddings=docs[i:j].astype("float32").tolist(),
            )
        self._coll = coll
        return time.perf_counter() - t

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._coll is not None
        chunk = 256
        all_ids = np.empty((queries.shape[0], k), dtype="int64")
        all_scores = np.zeros((queries.shape[0], k), dtype="float32")
        for i in range(0, queries.shape[0], chunk):
            q = queries[i : i + chunk].astype("float32").tolist()
            r = self._coll.query(query_embeddings=q, n_results=k)  # type: ignore[attr-defined]
            for j, hits in enumerate(r["ids"]):
                row = np.array([int(h) for h in hits], dtype="int64")
                pad = k - len(row)
                if pad > 0:
                    row = np.pad(row, (0, pad), constant_values=-1)
                all_ids[i + j] = row[:k]
        return all_ids, all_scores


def make_backend(kind: IndexBackend, cfg: IndexConfig) -> Backend:
    if kind == IndexBackend.FLAT_NUMPY:
        return FlatNumpy()
    if kind == IndexBackend.FAISS_FLAT:
        return FaissFlat()
    if kind == IndexBackend.FAISS_HNSW:
        return FaissHNSW(cfg=cfg)
    if kind == IndexBackend.FAISS_IVF_PQ:
        return FaissIVFPQ(cfg=cfg)
    if kind == IndexBackend.CHROMA:
        return ChromaBackend()
    raise ValueError(f"unknown backend {kind}")
