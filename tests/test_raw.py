import cv2
import time
import statistics

def test(width, height, duration=20):
    print(f"\nTesting {width}x{height}...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("FAILED: camera did not open")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Give camera time to apply settings
    time.sleep(2)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    times = []
    frames = 0
    failed = 0

    start = time.perf_counter()

    while time.perf_counter() - start < duration:
        ret, frame = cap.read()

        if not ret:
            failed += 1
            continue

        frames += 1
        times.append(time.perf_counter())

    elapsed = time.perf_counter() - start
    cap.release()

    fps = frames / elapsed

    intervals = [
        times[i] - times[i - 1]
        for i in range(1, len(times))
    ]

    jitter = statistics.stdev(intervals) * 1000 if len(intervals) > 1 else 0

    print(f"Requested        : {width}x{height}")
    print(f"Actual           : {actual_w}x{actual_h}")
    print(f"FPS              : {fps:.2f}")
    print(f"Frames           : {frames}")
    print(f"Failed           : {failed}")
    print(f"Jitter           : {jitter:.2f} ms")


test(1280, 720)
test(1920, 1080)