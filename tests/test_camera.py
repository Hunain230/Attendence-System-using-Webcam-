import cv2
import time
import statistics


CAMERA_INDEX = 0
TEST_DURATION = 30


def run_camera_benchmark(width, height):
    print("\n" + "=" * 60)
    print(f"Testing requested resolution: {width}x{height}")
    print("=" * 60)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    # Request resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Warm up camera
    print("Warming up camera...")
    for _ in range(30):
        cap.read()

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Actual resolution : {actual_width}x{actual_height}")
    print(f"Camera reported FPS: {reported_fps}")

    print(f"\nMeasuring for {TEST_DURATION} seconds...")
    print("Press Q to stop early.\n")

    frame_count = 0
    failed_frames = 0
    timestamps = []

    start = time.perf_counter()

    while time.perf_counter() - start < TEST_DURATION:

        frame_start = time.perf_counter()

        ret, frame = cap.read()

        if not ret:
            failed_frames += 1
            continue

        frame_count += 1
        timestamps.append(time.perf_counter())

        # Display
        cv2.putText(
            frame,
            f"Frames: {frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Camera Benchmark", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = time.perf_counter() - start

    cap.release()
    cv2.destroyAllWindows()

    # Results
    actual_fps = frame_count / elapsed

    intervals = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
    ]

    if intervals:
        avg_interval = statistics.mean(intervals)
        jitter = statistics.stdev(intervals) if len(intervals) > 1 else 0
    else:
        avg_interval = 0
        jitter = 0

    print("\nRESULTS")
    print("-" * 40)
    print(f"Resolution       : {actual_width}x{actual_height}")
    print(f"Reported FPS     : {reported_fps:.2f}")
    print(f"Measured FPS     : {actual_fps:.2f}")
    print(f"Frames captured  : {frame_count}")
    print(f"Failed frames    : {failed_frames}")
    print(f"Average interval : {avg_interval * 1000:.2f} ms")
    print(f"Frame jitter     : {jitter * 1000:.2f} ms")


if __name__ == "__main__":

    # Test 720p
    run_camera_benchmark(1280, 720)

    # Test 1080p
    run_camera_benchmark(1920, 1080)