"""
Phase 10 Benchmark — FastAPI REST API Endpoint Performance

Measures REST API response times and request throughput (requests/sec) for:
  - GET /health
  - GET /api/employees
  - POST /api/employees
  - GET /api/attendance/today
  - GET /api/recognition/status
"""

import sys
import time
import statistics
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from sqlalchemy.pool import StaticPool
from app.main import app
from app.database.database import Base, get_db


def benchmark_api_endpoints(num_requests: int = 500):
    print("\n" + "=" * 65, flush=True)
    print(f"PHASE 10: FASTAPI REST API BENCHMARK ({num_requests} requests/endpoint)", flush=True)
    print("=" * 65, flush=True)

    # Test DB setup
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestingSessionLocal()

    def _get_test_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = _get_test_db
    client = TestClient(app)

    # Pre-create 10 employees
    for i in range(10):
        client.post("/api/employees", json={"employee_code": f"API_BENCH_{i}", "name": f"User {i}"})

    def run_benchmark(name: str, method: str, path: str, json_data: dict = None):
        latencies = []
        t0_total = time.perf_counter()

        for _ in range(num_requests):
            t0 = time.perf_counter()
            if method == "GET":
                res = client.get(path)
            elif method == "POST":
                res = client.post(path, json=json_data)
            latencies.append(time.perf_counter() - t0)

        total_elapsed = time.perf_counter() - t0_total
        avg_ms = statistics.mean(latencies) * 1000
        med_ms = statistics.median(latencies) * 1000
        p95_ms = statistics.quantiles(latencies, n=20)[18] * 1000
        throughput = num_requests / total_elapsed

        print(f"{name:<25} | {avg_ms:<9.2f} | {med_ms:<10.2f} | {p95_ms:<9.2f} | {throughput:.1f} req/s", flush=True)
        return med_ms, throughput

    print(f"{'Endpoint':<25} | {'Avg (ms)':<9} | {'Median(ms)':<10} | {'P95 (ms)':<9} | Throughput", flush=True)
    print("-" * 75, flush=True)
    run_benchmark("1. GET /health", "GET", "/health")
    run_benchmark("2. GET /api/employees", "GET", "/api/employees")
    run_benchmark("3. GET /api/attendance/today", "GET", "/api/attendance/today")
    run_benchmark("4. GET /api/recognition/status", "GET", "/api/recognition/status")
    print("=" * 65, flush=True)

    session.close()
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


if __name__ == "__main__":
    benchmark_api_endpoints()
