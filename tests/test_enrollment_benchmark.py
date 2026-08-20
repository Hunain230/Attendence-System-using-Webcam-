"""
Phase 7 Benchmark — Employee Enrollment Pipeline

Measures latency per stage in the enrollment pipeline:
  1. SCRFD Detection latency
  2. QualityGate evaluation latency
  3. ArcFace embedding generation latency
  4. FAISS insertion latency
  5. SQLite transaction persistence latency
  6. Total end-to-end enrollment sample latency
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

from app.recognition.enrollment import EnrollmentService
from app.recognition.quality import QualityGate
from app.recognition.matcher import FaceMatcher
from app.database.database import Base
from app.database.repository import EmployeeRepository
from app import config


def benchmark_enrollment_pipeline(runs: int = 50):
    print("\n" + "=" * 60, flush=True)
    print(f"PHASE 7: ENROLLMENT PIPELINE BENCHMARK ({runs} runs)", flush=True)
    print("=" * 60, flush=True)

    # Database setup
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = TestingSessionLocal()

    # Create test employee
    emp = EmployeeRepository.create(db, employee_code="EMP_BENCH", name="Benchmark User")

    # Matcher setup
    matcher = FaceMatcher(auto_save=False)

    # Complete EnrollmentService with real models & QualityGate
    service = EnrollmentService(matcher=matcher)

    # Synthetic sharp, well-lit face crop (120x120) & landmarks
    rng = np.random.default_rng(42)
    fake_crop = np.full((120, 120, 3), 128, dtype=np.uint8)
    noise = rng.integers(-40, 40, size=(120, 120, 3), dtype=np.int16)
    fake_crop = np.clip(fake_crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    fake_bbox = np.array([10.0, 10.0, 130.0, 130.0], dtype=np.float32)
    fake_kps = np.array([[40.0, 45.0], [80.0, 45.0], [60.0, 55.7], [45.0, 95.0], [75.0, 95.0]], dtype=np.float32)

    class DummyFace:
        pass
    dummy_face = DummyFace()
    dummy_face.face_crop = fake_crop

    detection_times = []
    quality_times = []
    embedding_times = []
    faiss_times = []
    sqlite_times = []
    total_times = []

    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    for i in range(runs):
        t0_total = time.perf_counter()

        # Stage 1: SCRFD Detection
        t0 = time.perf_counter()
        _ = service.detector.detect(frame)
        t_det = time.perf_counter() - t0
        detection_times.append(t_det)

        # Stage 2: QualityGate
        t0 = time.perf_counter()
        qres = service.quality_gate.check(fake_crop, fake_bbox, fake_kps)
        t_qual = time.perf_counter() - t0
        quality_times.append(t_qual)

        # Stage 3: ArcFace Embedding
        t0 = time.perf_counter()
        embedding = service.embedder.embed(frame, dummy_face)
        t_emb = time.perf_counter() - t0
        embedding_times.append(t_emb)

        # Stage 4: FAISS Insertion
        t0 = time.perf_counter()
        faiss_id = service.matcher.add(embedding, employee_id=emp.id)
        t_faiss = time.perf_counter() - t0
        faiss_times.append(t_faiss)

        # Stage 5: SQLite Persistence
        t0 = time.perf_counter()
        from app.database.repository import EmbeddingRepository
        EmbeddingRepository.add(db, employee_id=emp.id, faiss_id=faiss_id, quality_score=0.9)
        t_sqlite = time.perf_counter() - t0
        sqlite_times.append(t_sqlite)

        t_total = time.perf_counter() - t0_total
        total_times.append(t_total)

    db.close()

    def stats(latencies_sec):
        if not latencies_sec:
            return 0.0, 0.0, 0.0
        avg = statistics.mean(latencies_sec) * 1000
        med = statistics.median(latencies_sec) * 1000
        p95 = statistics.quantiles(latencies_sec, n=20)[18] * 1000 if len(latencies_sec) >= 20 else avg
        return avg, med, p95

    det_avg, det_med, det_p95 = stats(detection_times)
    qual_avg, qual_med, qual_p95 = stats(quality_times)
    emb_avg, emb_med, emb_p95 = stats(embedding_times)
    faiss_avg, faiss_med, faiss_p95 = stats(faiss_times)
    sql_avg, sql_med, sql_p95 = stats(sqlite_times)
    tot_avg, tot_med, tot_p95 = stats(total_times)

    print(f"{'Stage':<25} | {'Avg (ms)':<10} | {'Median (ms)':<11} | {'P95 (ms)':<10}", flush=True)
    print("-" * 65, flush=True)
    print(f"{'1. SCRFD Detection':<25} | {det_avg:<10.2f} | {det_med:<11.2f} | {det_p95:<10.2f}", flush=True)
    print(f"{'2. QualityGate':<25} | {qual_avg:<10.2f} | {qual_med:<11.2f} | {qual_p95:<10.2f}", flush=True)
    print(f"{'3. ArcFace Embedding':<25} | {emb_avg:<10.2f} | {emb_med:<11.2f} | {emb_p95:<10.2f}", flush=True)
    print(f"{'4. FAISS Vector Insert':<25} | {faiss_avg:<10.2f} | {faiss_med:<11.2f} | {faiss_p95:<10.2f}", flush=True)
    print(f"{'5. SQLite Metadata DB':<25} | {sql_avg:<10.2f} | {sql_med:<11.2f} | {sql_p95:<10.2f}", flush=True)
    print("-" * 65, flush=True)
    print(f"{'TOTAL ENROLLMENT SAMPLE':<25} | {tot_avg:<10.2f} | {tot_med:<11.2f} | {tot_p95:<10.2f}", flush=True)
    print("=" * 60, flush=True)

    return {
        "det_med": det_med,
        "qual_med": qual_med,
        "emb_med": emb_med,
        "faiss_med": faiss_med,
        "sql_med": sql_med,
        "tot_med": tot_med,
    }


if __name__ == "__main__":
    benchmark_enrollment_pipeline()
