import cv2
import time
import statistics
from insightface.app import FaceAnalysis


CAMERA_INDEX = 0
WIDTH = 1280
HEIGHT = 720
DURATION = 30


print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded.")


cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

time.sleep(2)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera: {actual_w}x{actual_h}")
print(f"Running benchmark for {DURATION} seconds...\n")

# Warmup
for _ in range(10):
    ret, frame = cap.read()
    if ret:
        app.get(frame)

latencies = []
frames = 0
failed = 0
faces_total = 0

start = time.perf_counter()

while time.perf_counter() - start < DURATION:

    ret, frame = cap.read()

    if not ret:
        failed += 1
        continue

    t0 = time.perf_counter()

    faces = app.get(frame)

    inference_time = time.perf_counter() - t0

    latencies.append(inference_time)
    frames += 1
    faces_total += len(faces)


elapsed = time.perf_counter() - start

cap.release()

camera_fps = frames / elapsed

avg_latency = statistics.mean(latencies) * 1000
median_latency = statistics.median(latencies) * 1000
p95_latency = statistics.quantiles(latencies, n=20)[18] * 1000

inference_fps = 1 / statistics.mean(latencies)

avg_faces = faces_total / frames if frames else 0


print("\n" + "=" * 50)
print("INSIGHTFACE BENCHMARK")
print("=" * 50)

print(f"Resolution          : {actual_w}x{actual_h}")
print(f"Camera FPS          : {camera_fps:.2f}")
print(f"Failed frames       : {failed}")

print()
print(f"Average latency     : {avg_latency:.2f} ms")
print(f"Median latency      : {median_latency:.2f} ms")
print(f"P95 latency         : {p95_latency:.2f} ms")
print(f"Inference FPS       : {inference_fps:.2f}")

print()
print(f"Average faces/frame : {avg_faces:.2f}")

print("=" * 50)