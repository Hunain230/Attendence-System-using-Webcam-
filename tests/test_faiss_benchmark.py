"""
Phase 6 Benchmark — FAISS IndexFlatIP 512-D Vector Search

Evaluates FAISS search latency and throughput under realistic enrolled vector scales
(100, 1,000, and 5,000 512-D ArcFace L2-normalized embeddings) on CPU.

Measures:
  - Vector add latency
  - Search latency (Average, Median, P95)
  - Throughput (searches/second)
"""

import sys
import time
import statistics
import numpy as np
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.recognition.matcher import FaceMatcher


def benchmark_faiss(enrolled_count: int = 5000, search_queries: int = 1000):
    print(f"\n--- FAISS Benchmark: {enrolled_count} Enrolled 512-D Embeddings ---", flush=True)

    # Initialize in-memory FaceMatcher (auto_save=False for speed benchmark)
    matcher = FaceMatcher(auto_save=False)

    rng = np.random.default_rng(42)

    # Generate synthetic 512-D L2-normalized embeddings
    print(f"Generating {enrolled_count} synthetic embeddings...", flush=True)
    raw_vecs = rng.normal(size=(enrolled_count, 512)).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    normalized_vecs = raw_vecs / norms

    # Benchmark Add Latency
    t0_add = time.perf_counter()
    for i in range(enrolled_count):
        matcher.add(normalized_vecs[i], employee_id=i + 1)
    add_elapsed = time.perf_counter() - t0_add
    avg_add_ms = (add_elapsed / enrolled_count) * 1000

    print(f"Index built: {matcher.total_embeddings} vectors added in {add_elapsed * 1000:.2f} ms ({avg_add_ms:.4f} ms/add)", flush=True)

    # Generate 1,000 query vectors
    query_raw = rng.normal(size=(search_queries, 512)).astype(np.float32)
    query_norms = np.linalg.norm(query_raw, axis=1, keepdims=True)
    query_vecs = query_raw / query_norms

    # Warmup queries
    for i in range(20):
        matcher.search(query_vecs[i % search_queries], top_k=5, threshold=0.0)

    # Benchmark Search Latency
    search_latencies = []
    t0_search = time.perf_counter()

    for i in range(search_queries):
        query = query_vecs[i]
        t0 = time.perf_counter()
        matcher.search(query, top_k=5, threshold=0.0)
        search_latencies.append(time.perf_counter() - t0)

    search_total_elapsed = time.perf_counter() - t0_search

    avg_sec = statistics.mean(search_latencies)
    median_sec = statistics.median(search_latencies)
    p95_sec = statistics.quantiles(search_latencies, n=20)[18]
    throughput = search_queries / search_total_elapsed

    print(f"\nRESULTS ({enrolled_count} enrolled vectors, {search_queries} queries):", flush=True)
    print(f"Average Search Latency : {avg_sec * 1000:.4f} ms ({avg_sec * 1e6:.2f} µs)", flush=True)
    print(f"Median Search Latency  : {median_sec * 1000:.4f} ms ({median_sec * 1e6:.2f} µs)", flush=True)
    print(f"P95 Search Latency     : {p95_sec * 1000:.4f} ms ({p95_sec * 1e6:.2f} µs)", flush=True)
    print(f"Search Throughput      : {throughput:.2f} queries/sec", flush=True)

    return {
        "enrolled": enrolled_count,
        "avg_ms": avg_sec * 1000,
        "median_ms": median_sec * 1000,
        "p95_ms": p95_sec * 1000,
        "throughput": throughput,
    }


def main():
    print("=" * 60, flush=True)
    print("PHASE 6: FAISS IndexFlatIP BENCHMARK", flush=True)
    print("=" * 60, flush=True)

    for scale in [100, 1000, 5000]:
        benchmark_faiss(enrolled_count=scale, search_queries=1000)

    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
