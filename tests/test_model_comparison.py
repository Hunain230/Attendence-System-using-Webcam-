"""
Model Comparison Benchmark — buffalo_s vs buffalo_l

Compares SCRFD detector and ArcFace recognizer ONNX models under CPUExecutionProvider.
Measures:
  - Average latency (ms)
  - Median latency (ms)
  - P95 latency (ms)
  - Inference FPS / Throughput
"""

import time
import statistics
import numpy as np
import insightface
from insightface.model_zoo import model_zoo
from pathlib import Path

# Paths to models downloaded by InsightFace
MODELS_DIR = Path.home() / ".insightface" / "models"
BUFFALO_S_DIR = MODELS_DIR / "buffalo_s"
BUFFALO_L_DIR = MODELS_DIR / "buffalo_l"


def benchmark_scrfd(model_path: Path, label: str, runs: int = 40):
    print(f"\n--- Benchmarking SCRFD Detector: {label} ---", flush=True)
    print(f"Model path: {model_path}", flush=True)
    if not model_path.exists():
        print(f"Model path does not exist: {model_path}", flush=True)
        return None

    detector = model_zoo.get_model(str(model_path), providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=0, input_size=(640, 640))

    # Synthetic 720p frame
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        detector.detect(frame, max_num=0, metric="default")

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        detector.detect(frame, max_num=0, metric="default")
        latencies.append(time.perf_counter() - t0)

    avg = statistics.mean(latencies) * 1000
    median = statistics.median(latencies) * 1000
    p95 = statistics.quantiles(latencies, n=20)[18] * 1000
    fps = 1000.0 / avg

    res = {"label": label, "avg_ms": avg, "median_ms": median, "p95_ms": p95, "fps": fps}
    print(f"Avg: {avg:.2f} ms | Median: {median:.2f} ms | P95: {p95:.2f} ms | FPS: {fps:.2f}", flush=True)
    return res


def benchmark_arcface(model_path: Path, label: str, runs: int = 100):
    print(f"\n--- Benchmarking ArcFace Recognizer: {label} ---", flush=True)
    print(f"Model path: {model_path}", flush=True)
    if not model_path.exists():
        print(f"Model path does not exist: {model_path}", flush=True)
        return None

    recognizer = model_zoo.get_model(str(model_path), providers=["CPUExecutionProvider"])
    recognizer.prepare(ctx_id=0)

    # Synthetic 112x112 face crop
    face_crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        recognizer.get_feat(face_crop)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        feat = recognizer.get_feat(face_crop)
        latencies.append(time.perf_counter() - t0)

    avg = statistics.mean(latencies) * 1000
    median = statistics.median(latencies) * 1000
    p95 = statistics.quantiles(latencies, n=20)[18] * 1000
    fps = 1000.0 / avg

    res = {
        "label": label,
        "avg_ms": avg,
        "median_ms": median,
        "p95_ms": p95,
        "fps": fps,
        "dim": feat.shape,
    }
    print(
        f"Avg: {avg:.2f} ms | Median: {median:.2f} ms | P95: {p95:.2f} ms | FPS: {fps:.2f} | Shape: {feat.shape}",
        flush=True
    )
    return res


def main():
    print("=" * 60, flush=True)
    print("MODEL COMPARISON: buffalo_s vs buffalo_l", flush=True)
    print("=" * 60, flush=True)

    # Detectors
    scrfd_s = BUFFALO_S_DIR / "det_500m.onnx"
    scrfd_l = BUFFALO_L_DIR / "det_10g.onnx"

    res_scrfd_s = benchmark_scrfd(scrfd_s, "buffalo_s (det_500m)")
    res_scrfd_l = benchmark_scrfd(scrfd_l, "buffalo_l (det_10g)")

    # Recognizers
    arcface_s = BUFFALO_S_DIR / "w600k_mbf.onnx"
    arcface_l = BUFFALO_L_DIR / "w600k_r50.onnx"
    if not arcface_l.exists():
        # Check alternative model names in buffalo_l
        for p in BUFFALO_L_DIR.glob("*.onnx"):
            if "det" not in p.name and "2d106" not in p.name and "gender" not in p.name:
                arcface_l = p
                break

    res_arcface_s = benchmark_arcface(arcface_s, "buffalo_s (MobileFaceNet w600k_mbf)")
    res_arcface_l = benchmark_arcface(arcface_l, "buffalo_l (ResNet-50 w600k_r50)")

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY COMPARISON TABLE", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Component/Model':<35} | {'Avg (ms)':<9} | {'Median (ms)':<11} | {'P95 (ms)':<9} | {'FPS':<8}", flush=True)
    print("-" * 80, flush=True)
    for r in [res_scrfd_s, res_scrfd_l, res_arcface_s, res_arcface_l]:
        if r:
            print(
                f"{r['label']:<35} | {r['avg_ms']:<9.2f} | {r['median_ms']:<11.2f} | {r['p95_ms']:<9.2f} | {r['fps']:<8.2f}",
                flush=True
            )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
