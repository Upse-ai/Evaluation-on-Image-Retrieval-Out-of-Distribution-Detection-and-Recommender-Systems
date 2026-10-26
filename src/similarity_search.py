"""
Similarity Search using FAISS

Implements:
- Exact kNN: FAISS.IndexFlatL2 (Euclidean) / FAISS.IndexFlatIP (Cosine)
- LSH:       FAISS.IndexLSH  (cosine ONLY - inner product is hardwired)
- HNSW:      FAISS.IndexHNSWFlat (both Euclidean and Cosine)

IMPORTANT: LSH in FAISS uses inner-product projection internally.
This means LSH can ONLY be used with cosine distance (via L2 normalization),
NOT with Euclidean distance.

Mandatory hyperparameter ranges from project spec:
  Images:
    LSH  nbits ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192}
    HNSW M ∈ {16, 32, 256}, efConstruction ∈ {16, 64, 256}, efSearch ∈ {32, 256, 512, 1024}
  Movies:
    LSH  nbits ∈ {256, 512, 1024, 2048, 3000, 4096, 6144}
    HNSW M ∈ {8, 16, 32}, efConstruction ∈ {8, 16, 32}, efSearch ∈ {4, 8, 16, 32, 64}
"""

import numpy as np
import faiss
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Any


# ─── Hyperparameter ranges from the project specification ─────────────────────

HYPERPARAM_RANGES = {
    'images': {
        'lsh':  {'nbits': [64, 128, 256, 512, 1024, 2048, 4096, 8192]},
        'hnsw': {
            'M':              [16, 32, 256],
            'ef_construction': [16, 64, 256],
            'ef_search':      [32, 256, 512, 1024]
        }
    },
    'movies': {
        'lsh':  {'nbits': [256, 512, 1024, 2048, 3000, 4096, 6144]},
        'hnsw': {
            'M':              [8, 16, 32],
            'ef_construction': [8, 16, 32],
            'ef_search':      [4, 8, 16, 32, 64]
        }
    }
}

K_VALUES = [1, 5, 10, 30]


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """All outputs from a search run."""
    indices: np.ndarray           # shape: (n_queries, k)
    distances: np.ndarray         # shape: (n_queries, k)
    query_time_total: float       # wall-clock seconds for all queries
    avg_query_time: float         # per-query seconds
    index_build_time: float       # seconds to build the index


# ─── Helper ───────────────────────────────────────────────────────────────────

def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row (required for cosine via inner product)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _to_f32(vectors: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(vectors, dtype=np.float32)


# ─── Exact kNN ────────────────────────────────────────────────────────────────

def build_exact_index(database: np.ndarray, metric: str) -> Tuple[faiss.Index, float]:
    """
    Build exact brute-force index.

    metric='euclidean' → IndexFlatL2
    metric='cosine'    → IndexFlatIP (after L2 normalization)

    Returns (index, build_time_seconds)
    """
    db = _to_f32(database)

    t0 = time.perf_counter()

    if metric == 'cosine':
        db = _normalize(db)
        index = faiss.IndexFlatIP(db.shape[1])
    elif metric == 'euclidean':
        index = faiss.IndexFlatL2(db.shape[1])
    else:
        raise ValueError(f"metric must be 'euclidean' or 'cosine', got '{metric}'")

    index.add(db)
    build_time = time.perf_counter() - t0
    return index, build_time


# ─── LSH ──────────────────────────────────────────────────────────────────────

def build_lsh_index(database: np.ndarray, nbits: int) -> Tuple[faiss.Index, float]:
    """
    Build LSH index (IndexLSH).

    NOTE: LSH in FAISS uses inner-product projections internally.
    Always normalize the database before calling this function.
    Only cosine distance is supported (not Euclidean).

    nbits for images: {64, 128, 256, 512, 1024, 2048, 4096, 8192}
    nbits for movies: {256, 512, 1024, 2048, 3000, 4096, 6144}
    """
    db = _to_f32(database)
    db = _normalize(db)  # cosine via normalization

    t0 = time.perf_counter()
    index = faiss.IndexLSH(db.shape[1], nbits)
    index.add(db)
    build_time = time.perf_counter() - t0
    return index, build_time


# ─── HNSW ─────────────────────────────────────────────────────────────────────

def build_hnsw_index(database: np.ndarray,
                     metric: str,
                     M: int,
                     ef_construction: int,
                     ef_search: int) -> Tuple[faiss.Index, float]:
    """
    Build HNSW index (IndexHNSWFlat).

    metric='euclidean' → faiss.METRIC_L2
    metric='cosine'    → faiss.METRIC_INNER_PRODUCT (after normalization)

    M:              max connections per node in graph
    ef_construction: candidate list size during index build
    ef_search:       candidate list size during query (set after build)

    Image ranges:  M ∈ {16, 32, 256}, efC ∈ {16, 64, 256}, efS ∈ {32, 256, 512, 1024}
    Movie ranges:  M ∈ {8, 16, 32},   efC ∈ {8, 16, 32},   efS ∈ {4, 8, 16, 32, 64}
    """
    db = _to_f32(database)

    t0 = time.perf_counter()

    if metric == 'cosine':
        db = _normalize(db)
        index = faiss.IndexHNSWFlat(db.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    elif metric == 'euclidean':
        index = faiss.IndexHNSWFlat(db.shape[1], M)  # default: METRIC_L2
    else:
        raise ValueError(f"metric must be 'euclidean' or 'cosine', got '{metric}'")

    index.hnsw.efConstruction = ef_construction
    index.add(db)
    build_time = time.perf_counter() - t0

    # efSearch can be changed after build to trade speed vs accuracy
    index.hnsw.efSearch = ef_search

    return index, build_time


# ─── Generic search ───────────────────────────────────────────────────────────

def search(index: faiss.Index,
           queries: np.ndarray,
           k: int,
           metric: str,
           index_build_time: float) -> SearchResult:
    """
    Run kNN queries against a pre-built FAISS index.

    Args:
        index: FAISS index (already built)
        queries: Query vectors [n_queries, dim]
        k: Number of neighbors
        metric: 'euclidean' or 'cosine'
        index_build_time: Recorded build time (seconds)

    Returns:
        SearchResult with indices, distances, and timing
    """
    q = _to_f32(queries)

    if metric == 'cosine':
        q = _normalize(q)

    t0 = time.perf_counter()
    distances, indices = index.search(q, k)
    total_time = time.perf_counter() - t0

    return SearchResult(
        indices=indices,
        distances=distances,
        query_time_total=total_time,
        avg_query_time=total_time / len(queries),
        index_build_time=index_build_time
    )


# ─── Amortized time ───────────────────────────────────────────────────────────

def amortized_time(build_time: float, total_query_time: float, n_queries: int) -> float:
    """
    Compute amortized time per query as required by the project spec:

        amortized = (index build time + total search time) / total number of queries
    """
    return (build_time + total_query_time) / n_queries


# ─── Convenience: run one full experiment configuration ───────────────────────

def run_single_config(database: np.ndarray,
                      queries: np.ndarray,
                      algorithm: str,
                      metric: str,
                      k: int,
                      **algo_kwargs) -> Dict[str, Any]:
    """
    Run one (algorithm, metric, k, hyperparams) configuration.

    Returns a dict with all raw results required by the project spec:
      algorithm, metric, k, hyperparams,
      index_build_time, avg_recall (computed separately), avg_query_time, amortized_time
    """
    algo = algorithm.lower()

    if algo == 'exact':
        if metric == 'euclidean':
            index, build_t = build_exact_index(database, 'euclidean')
        else:
            index, build_t = build_exact_index(database, 'cosine')

    elif algo == 'lsh':
        if metric != 'cosine':
            raise ValueError("LSH only supports cosine distance (not Euclidean)")
        nbits = algo_kwargs.get('nbits', 512)
        index, build_t = build_lsh_index(database, nbits)

    elif algo == 'hnsw':
        M = algo_kwargs.get('M', 32)
        ef_c = algo_kwargs.get('ef_construction', 64)
        ef_s = algo_kwargs.get('ef_search', 128)
        index, build_t = build_hnsw_index(database, metric, M, ef_c, ef_s)

    else:
        raise ValueError(f"Unknown algorithm: '{algorithm}'. Use 'exact', 'lsh', or 'hnsw'.")

    result = search(index, queries, k, metric, build_t)

    return {
        'algorithm': algorithm,
        'metric': metric,
        'k': k,
        'hyperparams': algo_kwargs,
        'index_build_time': build_t,
        'indices': result.indices,
        'distances': result.distances,
        'avg_query_time': result.avg_query_time,
        'query_time_total': result.query_time_total,
        'amortized_time': amortized_time(build_t, result.query_time_total, len(queries))
    }


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("FAISS Similarity Search Module")
    print("=" * 50)

    np.random.seed(42)
    dim, n_db, n_q = 64, 5000, 50

    db = np.random.randn(n_db, dim).astype(np.float32)
    q  = np.random.randn(n_q,  dim).astype(np.float32)

    for algo in ['exact', 'lsh', 'hnsw']:
        for metric in (['euclidean', 'cosine'] if algo != 'lsh' else ['cosine']):
            res = run_single_config(db, q, algo, metric, k=10)
            print(f"{algo:8s} {metric:10s} | build={res['index_build_time']:.4f}s "
                  f"| query={res['avg_query_time']*1000:.2f}ms/q")
