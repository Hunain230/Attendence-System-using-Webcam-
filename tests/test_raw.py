
import cv2
import time
import statistics


CAMERA_INDEX = 0
DURATION = 30


def benchmark(width, height):
    print("\n" + "=" * 60)
    print(f"RAW CAMERA TEST: {width}x{height}")
    print("=" * 60)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Camera could not be opened")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Warmup
    for _ in range(30):
        cap.read()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timestamps = []
    failed = 0
    frames = 0

    start = time.perf_counter()

    while time.perf_counter() - start < DURATION:

        ret, frame = cap.read()

        if not ret:
            failed += 1
            continue

        frames += 1
        timestamps.append(time.perf_counter())

    elapsed = time.perf_counter() - start

    cap.release()

    fps = frames / elapsed

    intervals = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
    ]

    if intervals:
        avg_interval = statistics.mean(intervals)
        jitter = statistics.stdev(intervals)
    else:
        avg_interval = 0
        jitter = 0

    print(f"Actual resolution : {actual_w}x{actual_h}")
    print(f"Measured FPS      : {fps:.2f}")
    print(f"Frames            : {frames}")
    print(f"Failed frames     : {failed}")
    print(f"Average interval  : {avg_interval * 1000:.2f} ms")
    print(f"Jitter            : {jitter * 1000:.2f} ms")


if __name__ == "__main__":
    benchmark(1280, 720)
    benchmark(1920, 1080)
