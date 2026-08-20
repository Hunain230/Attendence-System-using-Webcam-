"""
Comprehensive API Unit and Integration Tests for Phase 10 (main.py & routers)

Verifies:
  1. Health check endpoint (/health)
  2. Employee CRUD endpoints (/api/employees)
  3. Employee enrollment endpoints (/api/enrollment)
  4. Attendance query & explicit checkout endpoints (/api/attendance)
  5. Recognition Engine control endpoints (/api/recognition)
  6. MJPEG streaming video endpoint (/api/stream)
  7. HTTP status codes, error handling, and JSON schema validation
"""

import pytest
import base64
import numpy as np
import cv2
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.pool import StaticPool
from app.main import app
from app.database.database import Base, get_db
from app.database.repository import EmployeeRepository, AttendanceRepository
from app.attendance.service import AttendanceService


@pytest.fixture
def test_db():
    """Provides an isolated in-memory SQLite database session for FastAPI dependencies."""
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
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        app.dependency_overrides.clear()


@pytest.fixture
def client(test_db):
    """Provides FastAPI TestClient."""
    return TestClient(app)


# ── 1. Health Check ───────────────────────────────────────────


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}


# ── 2. Employee CRUD Endpoints ─────────────────────────────────


def test_create_employee_api(client):
    payload = {
        "employee_code": "EMP_API_01",
        "name": "Jane Doe",
        "department": "Engineering",
    }
    response = client.post("/api/employees", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["id"] is not None
    assert data["employee_code"] == "EMP_API_01"
    assert data["name"] == "Jane Doe"
    assert data["department"] == "Engineering"
    assert data["active"] is True


def test_create_duplicate_employee_api_fails(client):
    payload = {"employee_code": "EMP_DUP", "name": "User 1"}
    client.post("/api/employees", json=payload)

    dup_response = client.post("/api/employees", json=payload)
    assert dup_response.status_code == 400
    assert "already exists" in dup_response.json()["detail"]


def test_list_and_get_employees_api(client):
    client.post("/api/employees", json={"employee_code": "E1", "name": "Person 1"})
    client.post("/api/employees", json={"employee_code": "E2", "name": "Person 2"})

    # List endpoint
    list_res = client.get("/api/employees")
    assert list_res.status_code == 200
    assert len(list_res.json()) == 2

    # Get single endpoint
    emp_id = list_res.json()[0]["id"]
    get_res = client.get(f"/api/employees/{emp_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == emp_id

    # 404 for missing
    missing_res = client.get("/api/employees/99999")
    assert missing_res.status_code == 404


def test_deactivate_employee_api(client):
    res = client.post("/api/employees", json={"employee_code": "E_DEL", "name": "Delete Me"})
    emp_id = res.json()["id"]

    del_res = client.delete(f"/api/employees/{emp_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Listing active only returns 0
    active_list = client.get("/api/employees?active_only=true")
    assert len(active_list.json()) == 0


# ── 3. Enrollment Endpoints ───────────────────────────────────


def test_start_enrollment_api(client):
    payload = {
        "employee_code": "ENROLL_01",
        "name": "Enroll User",
        "department": "Security",
    }
    response = client.post("/api/enrollment/start", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["employee_id"] is not None
    assert data["current_pose"] == "straight"
    assert "Look straight" in data["instructions"]


def test_process_enrollment_sample_api(client):
    # Start session first
    start_res = client.post(
        "/api/enrollment/start",
        json={"employee_code": "ENROLL_02", "name": "Sample Person"},
    )
    emp_id = start_res.json()["employee_id"]

    # Create synthetic base64 image
    img = np.full((120, 120, 3), 128, dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    sample_payload = {
        "employee_code": "ENROLL_02",
        "name": "Sample Person",
        "employee_id": emp_id,
        "image_base64": img_b64,
    }

    sample_res = client.post("/api/enrollment/sample", json=sample_payload)
    assert sample_res.status_code == 200
    assert "success" in sample_res.json()


# ── 4. Attendance Endpoints ───────────────────────────────────


def test_attendance_endpoints(client, test_db):
    emp = EmployeeRepository.create(test_db, employee_code="ATT_API", name="Attendance Person")
    # Mark check-in
    AttendanceRepository.check_in(test_db, employee_id=emp.id, confidence=0.91)

    # GET /api/attendance/today
    today_res = client.get("/api/attendance/today")
    assert today_res.status_code == 200
    assert len(today_res.json()) == 1
    assert today_res.json()[0]["employee_code"] == "ATT_API"

    # GET /api/attendance?date=...
    date_str = str(today_res.json()[0]["date"])
    date_res = client.get(f"/api/attendance?date={date_str}")
    assert date_res.status_code == 200
    assert len(date_res.json()) == 1

    # POST /api/attendance/{employee_id}/checkout
    checkout_res = client.post(f"/api/attendance/{emp.id}/checkout")
    assert checkout_res.status_code == 200
    assert checkout_res.json()["success"] is True


# ── 5. Recognition Engine Control & Stream Endpoints ──────────


def test_recognition_engine_api_controls(client):
    # GET status
    status_res = client.get("/api/recognition/status")
    assert status_res.status_code == 200
    assert "running" in status_res.json()

    # POST start
    start_res = client.post("/api/recognition/start")
    assert start_res.status_code == 200

    # POST stop
    stop_res = client.post("/api/recognition/stop")
    assert stop_res.status_code == 200


def test_mjpeg_stream_endpoint(client):
    response = client.get("/api/stream?single_frame=true")
    assert response.status_code == 200
    assert "multipart/x-mixed-replace" in response.headers["content-type"]
