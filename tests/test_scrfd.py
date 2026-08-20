import cv2
import time
import statistics
from insightface.model_zoo import model_zoo

def run_scrfd_benchmark():
    MODEL = r"C:\Users\surface\.insightface\models\buffalo_s\det_500m.onnx"

    print("Loading SCRFD...")

    detector = model_zoo.get_model(
        MODEL,
        providers=["CPUExecutionProvider"]
    )

    detector.prepare(ctx_id=0, input_size=(640, 640))

    print("SCRFD loaded.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    time.sleep(2)

    latencies = []
    frames = 0
    faces_total = 0

    # Warmup
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            detector.detect(frame)

    print("Running for 30 seconds...")

    start = time.perf_counter()

    while time.perf_counter() - start < 30:

        ret, frame = cap.read()

        if not ret:
            continue

        t0 = time.perf_counter()

        bboxes, kpss = detector.detect(
            frame,
            max_num=0,
            metric="default"
        )

        latency = time.perf_counter() - t0

        latencies.append(latency)
        frames += 1
        faces_total += len(bboxes)

    cap.release()

    avg = statistics.mean(latencies)
    median = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else avg
    p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else avg

    print("\n" + "=" * 50)
    print("SCRFD BENCHMARK")
    print("=" * 50)
    print(f"Frames             : {frames}")
    print(f"Average latency    : {avg * 1000:.2f} ms")
    print(f"Median latency     : {median * 1000:.2f} ms")
    print(f"P95 latency        : {p95 * 1000:.2f} ms")
    print(f"P99 latency        : {p99 * 1000:.2f} ms")
    print(f"Inference FPS      : {1 / avg:.2f}")
    print(f"Average faces/frame: {faces_total / frames if frames else 0:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    run_scrfd_benchmark()