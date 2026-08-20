"""
Unit tests for Phase 5 — Database Layer (SQLite + SQLAlchemy ORM)

Verifies:
  - Database initialization and table schema creation
  - Employee CRUD & unique constraint enforcement
  - FaceEmbedding mapping and cascade deletions
  - Attendance check-in idempotency & explicit check-out functionality
"""

import pytest
from datetime import date, datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from app.database.database import Base
from app.database.models import Employee, FaceEmbedding, Attendance
from app.database.repository import (
    EmployeeRepository,
    EmbeddingRepository,
    AttendanceRepository,
)


@pytest.fixture(scope="function")
def db_session():
    """Provides an isolated in-memory SQLite database session for each test function."""
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


# ── Employee Repository Tests ─────────────────────────────────


def test_create_and_get_employee(db_session):
    emp = EmployeeRepository.create(
        db_session, employee_code="EMP001", name="Hunain Shahid", department="Engineering"
    )
    assert emp.id is not None
    assert emp.employee_code == "EMP001"
    assert emp.name == "Hunain Shahid"
    assert emp.active is True

    fetched = EmployeeRepository.get_by_code(db_session, "EMP001")
    assert fetched is not None
    assert fetched.id == emp.id


def test_duplicate_employee_code_fails(db_session):
    EmployeeRepository.create(
        db_session, employee_code="EMP001", name="Hunain Shahid"
    )
    with pytest.raises(IntegrityError):
        EmployeeRepository.create(
            db_session, employee_code="EMP001", name="Duplicate Person"
        )


def test_deactivate_employee(db_session):
    emp = EmployeeRepository.create(
        db_session, employee_code="EMP002", name="Ali Ahmed"
    )
    assert EmployeeRepository.deactivate(db_session, emp.id) is True

    fetched = EmployeeRepository.get_by_id(db_session, emp.id)
    assert fetched.active is False

    active_list = EmployeeRepository.list_all(db_session, active_only=True)
    assert len(active_list) == 0

    all_list = EmployeeRepository.list_all(db_session, active_only=False)
    assert len(all_list) == 1


# ── Embedding Repository Tests ────────────────────────────────


def test_add_and_get_embedding(db_session):
    emp = EmployeeRepository.create(
        db_session, employee_code="EMP003", name="Sara Khan"
    )
    emb = EmbeddingRepository.add(
        db_session, employee_id=emp.id, faiss_id=0, quality_score=0.92
    )

    assert emb.id is not None
    assert emb.faiss_id == 0
    assert emb.quality_score == 0.92

    embeddings = EmbeddingRepository.get_by_employee(db_session, emp.id)
    assert len(embeddings) == 1
    assert embeddings[0].faiss_id == 0

    mapping = EmbeddingRepository.get_by_faiss_id(db_session, 0)
    assert mapping is not None
    assert mapping.employee_id == emp.id


def test_cascade_delete_embeddings(db_session):
    emp = EmployeeRepository.create(
        db_session, employee_code="EMP004", name="Test Cascade"
    )
    EmbeddingRepository.add(db_session, employee_id=emp.id, faiss_id=10, quality_score=0.85)
    EmbeddingRepository.add(db_session, employee_id=emp.id, faiss_id=11, quality_score=0.88)

    db_session.delete(emp)
    db_session.commit()

    assert len(EmbeddingRepository.get_by_employee(db_session, emp.id)) == 0


# ── Attendance Repository Tests ───────────────────────────────


def test_first_recognition_marks_check_in(db_session):
    emp = EmployeeRepository.create(
        db_session, employee_code="EMP005", name="Checkin Test"
    )
    today = date.today()

    record = AttendanceRepository.check_in(
        db_session, employee_id=emp.id, confidence=0.85, attendance_date=today
    )
    assert record is not None
    assert record.employee_id == emp.id
    assert record.date == today
    assert record.confidence == 0.85
    assert record.check_out is None


def test_subsequent_recognition_same_day_returns_existing(db_session):
    emp = EmployeeRepository.create(
        db_session, employee_code="EMP006", name="Idempotency Test"
    )
    today = date.today()

    first = AttendanceRepository.check_in(
        db_session, employee_id=emp.id, confidence=0.82, attendance_date=today
    )
    second = AttendanceRepository.check_in(
        db_session, employee_id=emp.id, confidence=0.95, attendance_date=today
    )

    assert first.id == second.id
    assert second.confidence == 0.82  # Original check-in confidence retained


def test_explicit_checkout(db_session):
    emp = EmployeeRepository.create(
        db_session, employee_code="EMP007", name="Checkout Test"
    )
    today = date.today()

    AttendanceRepository.check_in(
        db_session, employee_id=emp.id, confidence=0.88, attendance_date=today
    )

    checked_out = AttendanceRepository.check_out(
        db_session, employee_id=emp.id, attendance_date=today
    )

    assert checked_out is not None
    assert checked_out.check_out is not None

    today_records = AttendanceRepository.get_by_date(db_session, today)
    assert len(today_records) == 1
    assert today_records[0].check_out is not None
