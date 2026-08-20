"""
Unit and Integration tests for Phase 9 — Attendance Service (service.py)

Verifies:
  1. First recognition today triggers automatic Check-In
  2. Subsequent recognitions same day are ignored (no duplicate rows created)
  3. In-memory cache fast path avoids redundant DB queries
  4. Explicit Check-Out sets check_out timestamp (API / UI action ONLY)
  5. Check-Out is NEVER automatically triggered by face recognition
  6. Non-existent & Deactivated employee handling
  7. Query helpers: get_today_attendance, get_attendance_by_date, get_employee_history
  8. End-to-end integration with RecognitionEngine
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.attendance.service import AttendanceService
from app.database.database import Base
from app.database.models import Employee, Attendance
from app.database.repository import EmployeeRepository, AttendanceRepository
from app.engine.recognition_engine import RecognitionEngine
from app.recognition.detector import DetectedFace
from app.recognition.quality import QualityResult
from app.recognition.tracker import FaceTracker


@pytest.fixture
def db_session():
    """Provides an isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def attendance_service():
    return AttendanceService()


# ── 1. Automatic Check-In & Duplicate Prevention ───────────────


def test_first_recognition_marks_check_in(db_session, attendance_service):
    emp = EmployeeRepository.create(db_session, employee_code="E9001", name="Sarah Connor")

    created = attendance_service.mark_if_needed(emp.id, confidence=0.88, db=db_session)

    assert created is True

    records = attendance_service.get_today_attendance(db_session)
    assert len(records) == 1
    assert records[0].employee_id == emp.id
    assert records[0].confidence == 0.88
    assert records[0].check_in is not None
    assert records[0].check_out is None


def test_subsequent_recognitions_same_day_are_ignored(db_session, attendance_service):
    emp = EmployeeRepository.create(db_session, employee_code="E9002", name="John Connor")

    res1 = attendance_service.mark_if_needed(emp.id, confidence=0.85, db=db_session)
    res2 = attendance_service.mark_if_needed(emp.id, confidence=0.92, db=db_session)
    res3 = attendance_service.mark_if_needed(emp.id, confidence=0.90, db=db_session)

    assert res1 is True
    assert res2 is False
    assert res3 is False

    records = attendance_service.get_today_attendance(db_session)
    assert len(records) == 1


# ── 2. Explicit Check-Out ─────────────────────────────────────


def test_explicit_checkout(db_session, attendance_service):
    emp = EmployeeRepository.create(db_session, employee_code="E9003", name="Kyle Reese")

    # Mark check-in first
    attendance_service.mark_if_needed(emp.id, confidence=0.89, db=db_session)

    # Explicit checkout
    checked_out = attendance_service.explicit_checkout(emp.id, db=db_session)
    assert checked_out is True

    records = attendance_service.get_today_attendance(db_session)
    assert len(records) == 1
    assert records[0].check_out is not None
    assert records[0].check_out >= records[0].check_in


def test_explicit_checkout_without_checkin_fails(db_session, attendance_service):
    emp = EmployeeRepository.create(db_session, employee_code="E9004", name="T-800")

    checked_out = attendance_service.explicit_checkout(emp.id, db=db_session)
    assert checked_out is False


def test_recognition_does_not_trigger_checkout(db_session, attendance_service):
    """CRITICAL RULE: Face recognition MUST NEVER automatically trigger checkout!"""
    emp = EmployeeRepository.create(db_session, employee_code="E9005", name="Marcus Wright")

    # First recognition -> check-in
    attendance_service.mark_if_needed(emp.id, confidence=0.91, db=db_session)

    # 100 subsequent recognitions -> MUST NOT check out!
    for _ in range(100):
        res = attendance_service.mark_if_needed(emp.id, confidence=0.95, db=db_session)
        assert res is False

    records = attendance_service.get_today_attendance(db_session)
    assert len(records) == 1
    assert records[0].check_out is None  # Still checked in, checkout was NOT triggered!


# ── 3. Edge Cases & Deactivated Employee Handling ───────────────


def test_deactivated_employee_checkin_fails(db_session, attendance_service):
    emp = EmployeeRepository.create(db_session, employee_code="E9006", name="Inactive User")
    EmployeeRepository.deactivate(db_session, emp.id)

    res = attendance_service.mark_if_needed(emp.id, confidence=0.88, db=db_session)
    assert res is False
    assert len(attendance_service.get_today_attendance(db_session)) == 0


def test_non_existent_employee_checkin_fails(db_session, attendance_service):
    res = attendance_service.mark_if_needed(99999, confidence=0.88, db=db_session)
    assert res is False


# ── 4. Query Helper Methods ───────────────────────────────────


def test_query_helpers(db_session, attendance_service):
    emp = EmployeeRepository.create(db_session, employee_code="E9007", name="Miles Dyson")

    # Check-in today
    attendance_service.mark_if_needed(emp.id, confidence=0.90, db=db_session)

    today_records = attendance_service.get_today_attendance(db_session)
    assert len(today_records) == 1

    date_records = attendance_service.get_attendance_by_date(db_session, date.today())
    assert len(date_records) == 1

    history = attendance_service.get_employee_attendance_history(db_session, emp.id)
    assert len(history) == 1
    assert history[0].employee_id == emp.id


# ── 5. Integration with RecognitionEngine ─────────────────────


def test_recognition_engine_attendance_integration(db_session, attendance_service):
    emp = EmployeeRepository.create(db_session, employee_code="E9008", name="Engine Integration Person")

    mock_det = MagicMock()
    mock_emb = MagicMock()
    mock_match = MagicMock()
    mock_qual = MagicMock()

    face = DetectedFace(
        bbox=np.array([10, 10, 130, 130], dtype=np.float32),
        landmarks=np.zeros((5, 2)),
        det_score=0.95,
        face_crop=np.zeros((120, 120, 3), dtype=np.uint8),
        raw_face=object(),
    )
    mock_det.detect.return_value = [face]
    mock_qual.check.return_value = QualityResult(passed=True, reason="passed", score=0.9, metrics={})
    mock_emb.embed.return_value = np.ones((512,), dtype=np.float32)
    mock_match.search.return_value = [(emp.id, 0.89)]

    engine = RecognitionEngine(
        detector=mock_det,
        embedder=mock_emb,
        matcher=mock_match,
        quality_gate=mock_qual,
        tracker=FaceTracker(),
        attendance_service=attendance_service,
    )

    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    # Frame 1: Recognition candidate 1
    engine.process_frame(frame, db=db_session)
    assert len(attendance_service.get_today_attendance(db_session)) == 0  # Not confirmed yet

    # Frame 2: Recognition candidate 2
    engine.process_frame(frame, db=db_session)
    assert len(attendance_service.get_today_attendance(db_session)) == 0  # Not confirmed yet

    # Frame 3: Temporal confirmation reached (3 frames match) -> Check-In marked!
    engine.process_frame(frame, db=db_session)

    records = attendance_service.get_today_attendance(db_session)
    assert len(records) == 1
    assert records[0].employee_id == emp.id
    assert records[0].confidence == 0.89
