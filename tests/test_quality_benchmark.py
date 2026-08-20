"""
Phase 3 Benchmark — Face Quality Gate Latency & Throughput

Measures:
  - Average latency
  - Median latency
  - P95 latency
  - Throughput (checks/second)
"""

import time
import statistics
import numpy as np
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.recognition.quality import QualityGate


def run_quality_benchmark(runs: int = 1000):
    gate = QualityGate()

    # Synthetic sharp face crop
    rng = np.random.default_rng(42)
    crop = np.full((120, 120, 3), 128, dtype=np.uint8)
    noise = rng.integers(-40, 40, size=(120, 120, 3), dtype=np.int16)
    crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    bbox = np.array([10.0, 10.0, 130.0, 130.0], dtype=np.float32)
    landmarks = np.array(
        [
            [40.0, 45.0],
            [80.0, 45.0],
            [60.0, 69.0],
            [45.0, 95.0],
            [75.0, 95.0],
        ],
        dtype=np.float32,
    )

    # Warmup
    for _ in range(50):
        gate.check(crop, bbox, landmarks)

    latencies = []
    start_total = time.perf_counter()

    for _ in range(runs):
        t0 = time.perf_counter()
        gate.check(crop, bbox, landmarks)
        latencies.append(time.perf_counter() - t0)

    total_time = time.perf_counter() - start_total

    avg_sec = statistics.mean(latencies)
    median_sec = statistics.median(latencies)
    p95_sec = statistics.quantiles(latencies, n=20)[18]
    throughput = runs / total_time

    print("\n" + "=" * 50, flush=True)
    print("PHASE 3: FACE QUALITY GATE BENCHMARK", flush=True)
    print("=" * 50, flush=True)
    print(f"Evaluations          : {runs}", flush=True)
    print(f"Average latency      : {avg_sec * 1000:.4f} ms ({avg_sec * 1e6:.2f} µs)", flush=True)
    print(f"Median latency       : {median_sec * 1000:.4f} ms ({median_sec * 1e6:.2f} µs)", flush=True)
    print(f"P95 latency          : {p95_sec * 1000:.4f} ms ({p95_sec * 1e6:.2f} µs)", flush=True)
    print(f"Throughput           : {throughput:.2f} checks/sec", flush=True)
    print("=" * 50, flush=True)

    return {
        "runs": runs,
        "avg_ms": avg_sec * 1000,
        "median_ms": median_sec * 1000,
        "p95_ms": p95_sec * 1000,
        "throughput": throughput,
    }


if __name__ == "__main__":
    run_quality_benchmark()
