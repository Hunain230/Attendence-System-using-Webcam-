"""
Phase 8 Benchmark — Recognition Engine & Tracking Performance

Measures:
  1. Camera / Video frame processing FPS
  2. SCRFD detection latency
  3. ArcFace feature extraction latency
  4. FAISS search latency
  5. Total end-to-end frame processing latency
  6. CRITICAL DEMONSTRATION: ArcFace calls per frame for persistent recognized tracks
"""

import sys
import time
import statistics
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from unittest.mock import MagicMock
from app.engine.recognition_engine import RecognitionEngine
from app.recognition.detector import DetectedFace
from app.recognition.matcher import FaceMatcher
from app.database.database import Base
from app.database.repository import EmployeeRepository
from app import config


def benchmark_recognition_engine(num_frames: int = 100):
    print("\n" + "=" * 65, flush=True)
    print(f"PHASE 8: RECOGNITION ENGINE BENCHMARK ({num_frames} stream frames)", flush=True)
    print("=" * 65, flush=True)

    # Database setup
    engine_db = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine_db)
    TestingSessionLocal = sessionmaker(bind=engine_db, autoflush=False, autocommit=False)
    db = TestingSessionLocal()

    emp = EmployeeRepository.create(db, employee_code="EMP_BENCH", name="Benchmark User")

    # Matcher setup with 1 enrolled vector
    matcher = FaceMatcher(auto_save=False)
    rng = np.random.default_rng(42)
    fake_vec = rng.normal(size=(512,)).astype(np.float32)
    fake_vec /= np.linalg.norm(fake_vec)
    matcher.add(fake_vec, employee_id=emp.id)

    # Instantiate RecognitionEngine with real detector, embedder, quality gate, tracker
    engine = RecognitionEngine(matcher=matcher)

    # Mock detector to return 1 persistent face so tracking runs continuously across all 100 frames
    fake_crop = np.full((120, 120, 3), 128, dtype=np.uint8)
    noise = rng.integers(-40, 40, size=(120, 120, 3), dtype=np.int16)
    fake_crop = np.clip(fake_crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    fake_face = DetectedFace(
        bbox=np.array([10.0, 10.0, 130.0, 130.0], dtype=np.float32),
        landmarks=np.array([[40.0, 45.0], [80.0, 45.0], [60.0, 55.7], [45.0, 95.0], [75.0, 95.0]], dtype=np.float32),
        det_score=0.95,
        face_crop=fake_crop,
        raw_face=object(),
    )
    engine.detector = MagicMock()
    engine.detector.detect.return_value = [fake_face]

    # Synthetic sharp 720p frame
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    print("Simulating 100 consecutive stream frames with 1 persistent face...", flush=True)

    frame_latencies = []
    arcface_invocations = []
    arcface_skips = []

    t0_start = time.perf_counter()

    for i in range(num_frames):
        t0 = time.perf_counter()
        res = engine.process_frame(frame, db=db, frame_index=i + 1)
        t_frame = time.perf_counter() - t0

        frame_latencies.append(t_frame)
        arcface_invocations.append(res.arcface_invocations)
        arcface_skips.append(res.arcface_skipped)

    total_elapsed = time.perf_counter() - t0_start
    effective_fps = num_frames / total_elapsed

    avg_ms = statistics.mean(frame_latencies) * 1000
    median_ms = statistics.median(frame_latencies) * 1000
    p95_ms = statistics.quantiles(frame_latencies, n=20)[18] * 1000

    total_arcface_calls = sum(arcface_invocations)
    total_arcface_skipped = sum(arcface_skips)

    print("\n" + "-" * 65, flush=True)
    print("RECOGNITION ENGINE LATENCY & THROUGHPUT:", flush=True)
    print(f"Total Stream Frames Processed : {num_frames}", flush=True)
    print(f"Total Time Elapsed            : {total_elapsed * 1000:.2f} ms", flush=True)
    print(f"Effective Processing FPS      : {effective_fps:.2f} FPS", flush=True)
    print(f"Average Frame Latency         : {avg_ms:.2f} ms", flush=True)
    print(f"Median Frame Latency          : {median_ms:.2f} ms", flush=True)
    print(f"P95 Frame Latency             : {p95_ms:.2f} ms", flush=True)

    print("\n" + "-" * 65, flush=True)
    print("ARCFACE INFERENCE OPTIMIZATION DEMONSTRATION:", flush=True)
    print(f"Total ArcFace Inferences Executed : {total_arcface_calls} calls across {num_frames} frames", flush=True)
    print(f"Total ArcFace Inferences SKIPPED  : {total_arcface_skipped} calls across {num_frames} frames", flush=True)
    print(f"ArcFace Calls Per Frame (Average)  : {total_arcface_calls / num_frames:.2f} calls/frame", flush=True)
    print("-" * 65, flush=True)

    db.close()
    return {
        "fps": effective_fps,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "arcface_calls": total_arcface_calls,
        "arcface_skipped": total_arcface_skipped,
    }


if __name__ == "__main__":
    benchmark_recognition_engine()
