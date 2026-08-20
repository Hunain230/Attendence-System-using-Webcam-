import cv2
import time
import statistics
import numpy as np
from insightface.model_zoo import model_zoo


def run_arcface_benchmark():
    MODEL = r"C:\Users\surface\.insightface\models\buffalo_s\w600k_mbf.onnx"

    print("Loading ArcFace...")

    recognizer = model_zoo.get_model(
        MODEL,
        providers=["CPUExecutionProvider"]
    )

    recognizer.prepare(ctx_id=0)

    print("ArcFace loaded.")

    # Create a realistic 112x112 face image
    face = np.random.randint(
        0, 255, (112, 112, 3), dtype=np.uint8
    )

    latencies = []

    # Warmup
    for _ in range(20):
        recognizer.get_feat(face)

    print("Running ArcFace benchmark for 30 seconds...")

    start = time.perf_counter()

    while time.perf_counter() - start < 30:

        t0 = time.perf_counter()

        embedding = recognizer.get_feat(face)

        latency = time.perf_counter() - t0

        latencies.append(latency)

    avg = statistics.mean(latencies)
    median = statistics.median(latencies)

    p95 = statistics.quantiles(
        latencies,
        n=20
    )[18]

    fps = 1 / avg

    print("\n" + "=" * 50)
    print("ARCFACE BENCHMARK")
    print("=" * 50)

    print(f"Embeddings generated : {len(latencies)}")
    print(f"Average latency      : {avg * 1000:.2f} ms")
    print(f"Median latency       : {median * 1000:.2f} ms")
    print(f"P95 latency          : {p95 * 1000:.2f} ms")
    print(f"Inference FPS        : {fps:.2f}")

    norm = np.linalg.norm(embedding)

    print(f"Embedding shape      : {embedding.shape}")
    print(f"Embedding norm       : {norm:.4f}")

    print("=" * 50)


if __name__ == "__main__":
    run_arcface_benchmark()