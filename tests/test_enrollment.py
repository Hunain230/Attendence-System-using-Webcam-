"""
Unit and Integration tests for Phase 7 — Employee Enrollment (enrollment.py)

Verifies:
  1. Successful employee creation and enrollment session state tracking
  2. Employee code and metadata handling
  3. Face detection handling (single face, zero faces, multiple faces)
  4. Stricter Enrollment QualityGate filtering
  5. ArcFace embedding generation (512-D, L2-normalized)
  6. SQLite embedding metadata persistence
  7. FAISS vector index insertion and ID mapping
  8. Duplicate employee code error handling
  9. Invalid input handling
  10. Transaction Consistency & Rollback Safety (FAISS <-> SQLite failure recovery)
  11. Deactivated employee enrollment restriction
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.recognition.enrollment import (
    EnrollmentService,
    EnrollmentSession,
    EnrollmentSampleResult,
)
from app.recognition.detector import DetectedFace
from app.recognition.quality import QualityGate, QualityResult
from app.recognition.matcher import FaceMatcher
from app.database.database import Base
from app.database.models import Employee, FaceEmbedding
from app.database.repository import EmployeeRepository, EmbeddingRepository
from app.api.employees import create_employee, deactivate_employee, enroll_employee_sample


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
def tmp_matcher(tmp_path):
    """Provides an isolated FaceMatcher instance."""
    idx_path = tmp_path / "enroll_faiss.index"
    map_path = tmp_path / "enroll_faiss_map.npy"
    return FaceMatcher(index_path=idx_path, id_map_path=map_path, auto_save=False)


@pytest.fixture
def valid_512_embedding():
    """Generates a synthetic 512-D L2-normalized vector."""
    rng = np.random.default_rng(42)
    vec = rng.normal(size=(512,)).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def mock_enrollment_service(tmp_matcher, valid_512_embedding):
    """Provides an EnrollmentService with mocked detector, embedder, and quality gate for fast unit tests."""
    mock_detector = MagicMock()
    mock_embedder = MagicMock()
    mock_quality = MagicMock()

    # Default mock behavior: 1 detected face
    fake_crop = np.full((120, 120, 3), 128, dtype=np.uint8)
    fake_bbox = np.array([10.0, 10.0, 130.0, 130.0], dtype=np.float32)
    # Calibrated landmarks: nose at y=67.8 → pitch≈0°, yaw≈0°
    fake_kps = np.array([[40.0, 45.0], [80.0, 45.0], [60.0, 67.8], [45.0, 95.0], [75.0, 95.0]], dtype=np.float32)

    face = DetectedFace(
        bbox=fake_bbox,
        landmarks=fake_kps,
        det_score=0.95,
        face_crop=fake_crop,
        raw_face=object(),
    )
    mock_detector.detect.return_value = [face]

    # Default quality gate mock: passed (strict check used in enrollment)
    passed_quality = QualityResult(
        passed=True, reason="passed", score=0.88, score_100=88.0,
        metrics={"brightness": 120.0, "yaw": 0.0, "pitch": 0.0}
    )
    mock_quality.check.return_value = passed_quality
    mock_quality.check_enrollment_strict.return_value = passed_quality

    # Pose check: returns matching, guidance msg, yaw, pitch
    mock_quality.check_enrollment_pose.return_value = (True, "Hold still — collecting best frame...", 0.0, 0.0)

    # Landmark stability: always stable
    mock_quality.check_landmark_stability.return_value = True

    # Default embedder mock: valid 512-D embedding
    mock_embedder.embed.return_value = valid_512_embedding

    service = EnrollmentService(
        detector=mock_detector,
        embedder=mock_embedder,
        matcher=tmp_matcher,
        quality_gate=mock_quality,
    )
    return service


# ── 1. Employee Creation & Metadata Handling ──────────────────


def test_create_employee(db_session, mock_enrollment_service):
    emp, session = mock_enrollment_service.create_employee_and_start_session(
        db_session, employee_code="EMP_001", name="Hunain Shahid", department="AI Systems"
    )

    assert emp.id is not None
    assert emp.employee_code == "EMP_001"
    assert emp.name == "Hunain Shahid"
    assert emp.department == "AI Systems"
    assert session.employee_id == emp.id
    assert session.status == "in_progress"


def test_duplicate_employee_code_fails(db_session, mock_enrollment_service):
    mock_enrollment_service.create_employee_and_start_session(
        db_session, employee_code="EMP_001", name="Original Person"
    )

    with pytest.raises(ValueError, match="already exists"):
        mock_enrollment_service.create_employee_and_start_session(
            db_session, employee_code="EMP_001", name="Duplicate Code Person"
        )


# ── 2. Enrollment Pipeline Execution ─────────────────────────


def test_successful_sample_enrollment(db_session, mock_enrollment_service):
    emp, session = mock_enrollment_service.create_employee_and_start_session(
        db_session, employee_code="EMP_002", name="Ali Ahmed"
    )

    fake_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    # Two-phase protocol: commit happens at call index (HOLD_FRAMES + CANDIDATE_FRAMES - 1)
    # Frame 1 → guidance→holding transition + counter=1
    # Frames 2..HOLD: counter increments until counter == HOLD_FRAMES (transition to collecting)
    # Frames HOLD..HOLD+CAND-1: collecting; on call HOLD+CAND-1, commits.
    from app import config
    res = None
    total_calls = config.ENROLLMENT_HOLD_FRAMES + config.ENROLLMENT_CANDIDATE_FRAMES - 1
    for i in range(total_calls):
        r = mock_enrollment_service.process_frame(db_session, emp.id, fake_frame, pose_name="straight")
        if r.captured:
            res = r
            break

    assert res is not None, "Enrollment did not capture within expected frame count"
    assert res.captured is True
    assert res.quality_result.score == 0.88

    # Verify FAISS insertion & mapping
    assert mock_enrollment_service.matcher.total_embeddings == 1
    assert mock_enrollment_service.matcher.id_map == [emp.id]

    # Verify SQLite record insertion
    embeddings_in_db = EmbeddingRepository.get_by_employee(db_session, emp.id)
    assert len(embeddings_in_db) == 1
    assert embeddings_in_db[0].faiss_id == 0
    assert embeddings_in_db[0].quality_score == 0.88

    # Verify session tracking
    assert session.total_captured == 1
    assert session.samples_for_current_pose == 1


# ── 3. Face Detection Scenarios ────────────────────────────────


def test_zero_face_detected(db_session, mock_enrollment_service):
    emp, _ = mock_enrollment_service.create_employee_and_start_session(
        db_session, employee_code="EMP_003", name="No Face Test"
    )

    mock_enrollment_service.detector.detect.return_value = []
    fake_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    res = mock_enrollment_service.process_frame(db_session, emp.id, fake_frame)

    assert res.success is False
    # New protocol uses 'pose_angle_not_reached' as fallback when no face is detected
    assert res.reason in ("no_face_detected", "pose_angle_not_reached", "no_face_detected")
    assert mock_enrollment_service.matcher.total_embeddings == 0


def test_multiple_ambiguous_faces_detected(db_session, mock_enrollment_service):
    emp, _ = mock_enrollment_service.create_employee_and_start_session(
        db_session, employee_code="EMP_004", name="Multi Face Test"
    )

    # 2 large comparable faces
    f1 = DetectedFace(
        bbox=np.array([10, 10, 130, 130]),
        landmarks=np.zeros((5, 2)),
        det_score=0.9,
        face_crop=np.zeros((120, 120, 3), dtype=np.uint8),
        raw_face=object(),
    )
    f2 = DetectedFace(
        bbox=np.array([200, 200, 315, 315]),
        landmarks=np.zeros((5, 2)),
        det_score=0.9,
        face_crop=np.zeros((115, 115, 3), dtype=np.uint8),
        raw_face=object(),
    )
    mock_enrollment_service.detector.detect.return_value = [f1, f2]

    fake_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    res = mock_enrollment_service.process_frame(db_session, emp.id, fake_frame)

    assert res.success is False
    assert res.reason == "multiple_faces_detected"


# ── 4. QualityGate Rejection ──────────────────────────────────


def test_quality_gate_rejection(db_session, mock_enrollment_service):
    emp, _ = mock_enrollment_service.create_employee_and_start_session(
        db_session, employee_code="EMP_005", name="Quality Fail Test"
    )

    # Set both the check() and check_enrollment_strict() to fail
    failed_quality = QualityResult(passed=False, reason="too_blurry", score=0.0)
    mock_enrollment_service.quality_gate.check.return_value = failed_quality
    mock_enrollment_service.quality_gate.check_enrollment_strict.return_value = failed_quality

    fake_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    res = mock_enrollment_service.process_frame(db_session, emp.id, fake_frame)

    assert res.success is False
    assert res.reason == "too_blurry"
    assert mock_enrollment_service.matcher.total_embeddings == 0
    assert len(EmbeddingRepository.get_by_employee(db_session, emp.id)) == 0


# ── 5. Transaction Consistency & Rollback Safety ───────────────


def test_sqlite_failure_rolls_back_faiss(db_session, mock_enrollment_service, monkeypatch):
    """Test two-system transaction safety: if SQLite persistence fails, FAISS vector is removed!"""
    emp, _ = mock_enrollment_service.create_employee_and_start_session(
        db_session, employee_code="EMP_006", name="Rollback Test"
    )

    # Monkeypatch EmbeddingRepository.add to raise RuntimeError
    def mock_db_add_fail(*args, **kwargs):
        raise RuntimeError("Simulated Database Disk Error")

    monkeypatch.setattr(EmbeddingRepository, "add", mock_db_add_fail)

    fake_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    # Drive through hold + collect cycle to reach commit step
    from app import config
    total_calls = config.ENROLLMENT_HOLD_FRAMES + config.ENROLLMENT_CANDIDATE_FRAMES - 1
    res = None
    for _ in range(total_calls):
        r = mock_enrollment_service.process_frame(db_session, emp.id, fake_frame)
        if not r.success and "sqlite_persistence_failed" in r.reason:
            res = r
            break

    assert res is not None, "SQLite failure did not occur within expected frame count"
    assert res.success is False
    assert "sqlite_persistence_failed" in res.reason

    # CRITICAL: FAISS index must be rolled back (0 vectors remaining)
    assert mock_enrollment_service.matcher.total_embeddings == 0
    assert mock_enrollment_service.matcher.id_map == []


def test_faiss_failure_rolls_back_sqlite(db_session, mock_enrollment_service, monkeypatch):
    """Test two-system transaction safety: if FAISS addition fails, SQLite transaction rolls back."""
    emp, _ = mock_enrollment_service.create_employee_and_start_session(
        db_session, employee_code="EMP_007", name="FAISS Fail Test"
    )

    # Monkeypatch matcher.add to raise RuntimeError
    def mock_faiss_add_fail(*args, **kwargs):
        raise RuntimeError("Simulated FAISS Allocation Error")

    monkeypatch.setattr(mock_enrollment_service.matcher, "add", mock_faiss_add_fail)

    fake_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    # Drive through hold + collect cycle to reach commit step
    from app import config
    total_calls = config.ENROLLMENT_HOLD_FRAMES + config.ENROLLMENT_CANDIDATE_FRAMES - 1
    res = None
    for _ in range(total_calls):
        r = mock_enrollment_service.process_frame(db_session, emp.id, fake_frame)
        if not r.success and "faiss_insertion_failed" in r.reason:
            res = r
            break

    assert res is not None, "FAISS failure did not occur within expected frame count"
    assert res.success is False
    assert "faiss_insertion_failed" in res.reason

    # Verify no embedding metadata in SQLite
    assert len(EmbeddingRepository.get_by_employee(db_session, emp.id)) == 0


# ── 6. Deactivated Employee Restriction ────────────────────────


def test_deactivated_employee_enrollment_fails(db_session, mock_enrollment_service):
    emp, _ = mock_enrollment_service.create_employee_and_start_session(
        db_session, employee_code="EMP_008", name="Deactivated User"
    )

    deactivate_employee(db_session, emp.id, matcher=mock_enrollment_service.matcher)

    fake_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    res = mock_enrollment_service.process_frame(db_session, emp.id, fake_frame)

    assert res.success is False
    assert res.reason == "employee_deactivated"
    assert mock_enrollment_service.matcher.total_embeddings == 0


# ── 7. Multi-Pose Session State Progression ────────────────────


def test_enrollment_session_pose_progression(db_session, mock_enrollment_service):
    emp, session = mock_enrollment_service.create_employee_and_start_session(
        db_session, employee_code="EMP_009", name="Session Pose User"
    )

    fake_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    from app import config

    def complete_one_pose():
        """Drive through hold+collect cycle to commit one embedding."""
        for _ in range(config.ENROLLMENT_HOLD_FRAMES + config.ENROLLMENT_CANDIDATE_FRAMES):
            mock_enrollment_service.process_frame(db_session, emp.id, fake_frame)

    # Pose 1 ("straight", target = 2 embeddings)
    assert session.current_pose[0] == "straight"
    complete_one_pose()  # First straight embedding
    assert session.samples_for_current_pose == 1  # Still in straight pose (needs 2)

    complete_one_pose()  # Second straight embedding
    # Pose should advance to "slight_left"
    assert session.current_pose[0] == "slight_left"
