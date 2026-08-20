"""
Phase 9 Benchmark — Attendance Service Latency & Throughput

Measures:
  1. Attendance Check-In creation latency (SQLite insertion)
  2. Debounced/cached lookup latency for continuous stream recognitions
  3. Explicit Check-Out latency
  4. Query throughput (get_today_attendance & history)
"""

import sys
import time
import statistics
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.attendance.service import AttendanceService
from app.database.database import Base
from app.database.repository import EmployeeRepository


def benchmark_attendance_service(num_employees: int = 100, recognitions_per_emp: int = 100):
    print("\n" + "=" * 65, flush=True)
    print(f"PHASE 9: ATTENDANCE SERVICE BENCHMARK", flush=True)
    print("=" * 65, flush=True)

    # Database setup
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = TestingSessionLocal()

    # Pre-create employees
    emp_ids = []
    for i in range(num_employees):
        emp = EmployeeRepository.create(db, employee_code=f"EMP_B_{i}", name=f"Bench User {i}")
        emp_ids.append(emp.id)

    service = AttendanceService()

    # 1. First Recognition Check-In Latency (SQLite Insert)
    checkin_latencies = []
    for emp_id in emp_ids:
        t0 = time.perf_counter()
        service.mark_if_needed(emp_id, confidence=0.88, db=db)
        checkin_latencies.append(time.perf_counter() - t0)

    avg_checkin = statistics.mean(checkin_latencies) * 1000
    med_checkin = statistics.median(checkin_latencies) * 1000
    p95_checkin = statistics.quantiles(checkin_latencies, n=20)[18] * 1000

    print(f"First Check-In Latency (SQLite Insert):", flush=True)
    print(f"  Average : {avg_checkin:.4f} ms ({avg_checkin * 1000:.2f} µs)", flush=True)
    print(f"  Median  : {med_checkin:.4f} ms ({med_checkin * 1000:.2f} µs)", flush=True)
    print(f"  P95     : {p95_checkin:.4f} ms ({p95_checkin * 1000:.2f} µs)", flush=True)

    # 2. Cached / Debounced Same-Day Recognition Latency
    cached_latencies = []
    total_cached_calls = num_employees * recognitions_per_emp

    t0_cache = time.perf_counter()
    for _ in range(recognitions_per_emp):
        for emp_id in emp_ids:
            t0 = time.perf_counter()
            service.mark_if_needed(emp_id, confidence=0.88, db=db)
            cached_latencies.append(time.perf_counter() - t0)
    cache_elapsed = time.perf_counter() - t0_cache

    avg_cache = statistics.mean(cached_latencies) * 1000
    med_cache = statistics.median(cached_latencies) * 1000
    p95_cache = statistics.quantiles(cached_latencies, n=20)[18] * 1000
    cache_throughput = total_cached_calls / cache_elapsed

    print(f"\nDebounced / Cached Stream Recognition Latency ({total_cached_calls} calls):", flush=True)
    print(f"  Average : {avg_cache:.6f} ms ({avg_cache * 1000:.2f} µs)", flush=True)
    print(f"  Median  : {med_cache:.6f} ms ({med_cache * 1000:.2f} µs)", flush=True)
    print(f"  P95     : {p95_cache:.6f} ms ({p95_cache * 1000:.2f} µs)", flush=True)
    print(f"  Throughput : {cache_throughput:.2f} checks/sec", flush=True)

    # 3. Explicit Check-Out Latency
    checkout_latencies = []
    for emp_id in emp_ids:
        t0 = time.perf_counter()
        service.explicit_checkout(emp_id, db=db)
        checkout_latencies.append(time.perf_counter() - t0)

    avg_checkout = statistics.mean(checkout_latencies) * 1000
    med_checkout = statistics.median(checkout_latencies) * 1000

    print(f"\nExplicit Check-Out Latency:", flush=True)
    print(f"  Average : {avg_checkout:.4f} ms ({avg_checkout * 1000:.2f} µs)", flush=True)
    print(f"  Median  : {med_checkout:.4f} ms ({med_checkout * 1000:.2f} µs)", flush=True)
    print("=" * 65, flush=True)

    db.close()
    return {
        "checkin_med_ms": med_checkin,
        "cache_med_us": med_cache * 1000,
        "checkout_med_ms": med_checkout,
    }


if __name__ == "__main__":
    benchmark_attendance_service()
