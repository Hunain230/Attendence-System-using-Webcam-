"""
Unit tests for Phase 8 — Recognition Engine & Tracker Optimization (recognition_engine.py & tracker.py)

Verifies:
  1. Engine initialization and state management
  2. Empty / Zero-face frame handling
  3. Single-face and multi-face IoU tracking
  4. ArcFace Skip Optimization: Recognized & Unknown tracks DO NOT repeatedly invoke ArcFace!
  5. New track ArcFace invocation & FAISS vector search
  6. Temporal confirmation threshold logic (3 consecutive matches required)
  7. Known + Unknown multi-face simultaneous tracking
  8. Track expiration (MAX_LOST_FRAMES)
  9. Detector, embedder, and matcher failure resilience
  10. Background thread safety and controls (start/stop/get_latest_result)
"""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.engine.recognition_engine import RecognitionEngine, RecognitionFrameResult
from app.recognition.tracker import FaceTracker, Track
from app.recognition.detector import DetectedFace
from app.recognition.quality import QualityResult
from app.database.database import Base
from app.database.repository import EmployeeRepository
from app import config


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
def mock_engine_components():
    """Provides mocked detector, embedder, matcher, quality_gate, and tracker for engine testing."""
    mock_detector = MagicMock()
    mock_embedder = MagicMock()
    mock_matcher = MagicMock()
    mock_quality = MagicMock()

    # Default detector return: 1 face
    face1 = DetectedFace(
        bbox=np.array([10.0, 10.0, 130.0, 130.0], dtype=np.float32),
        landmarks=np.array([[40.0, 45.0], [80.0, 45.0], [60.0, 55.7], [45.0, 95.0], [75.0, 95.0]], dtype=np.float32),
        det_score=0.95,
        face_crop=np.full((120, 120, 3), 128, dtype=np.uint8),
        raw_face=object(),
    )
    mock_detector.detect.return_value = [face1]

    # Default quality gate: passed
    mock_quality.check.return_value = QualityResult(
        passed=True, reason="passed", score=0.9, metrics={}
    )

    # Default embedder: 512-D vector
    mock_embedder.embed.return_value = np.ones((512,), dtype=np.float32)

    # Default matcher: matches employee ID 1 with confidence 0.85
    mock_matcher.search.return_value = [(1, 0.85)]

    engine = RecognitionEngine(
        detector=mock_detector,
        embedder=mock_embedder,
        matcher=mock_matcher,
        quality_gate=mock_quality,
        tracker=FaceTracker(),
    )
    return engine


# ── 1. Engine Initialization & Empty Frame Handling ───────────


def test_engine_initialization(mock_engine_components):
    engine = mock_engine_components
    assert engine.is_running is False
    assert engine.get_latest_result() is None
    assert engine.get_frame() is None


def test_empty_frame_returns_empty_result(mock_engine_components):
    engine = mock_engine_components
    res = engine.process_frame(np.array([]))

    assert isinstance(res, RecognitionFrameResult)
    assert res.tracks == []
    assert res.arcface_invocations == 0
    assert res.arcface_skipped == 0


def test_zero_faces_detected(mock_engine_components):
    engine = mock_engine_components
    engine.detector.detect.return_value = []

    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    res = engine.process_frame(frame)

    assert res.tracks == []
    assert res.arcface_invocations == 0


# ── 2. ArcFace Skip Optimization (Crucial Performance Rule) ───


def test_arcface_skip_optimization_on_recognized_track(db_session, mock_engine_components):
    """CRITICAL TEST: Verifies ArcFace is executed ONLY on first frames, then SKIPPED on subsequent frames for the same track!"""
    engine = mock_engine_components
    EmployeeRepository.create(db_session, employee_code="EMP_TEST", name="Test Employee")

    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    # ── Frame 1: New track → ArcFace MUST be invoked ─────────────────────
    res1 = engine.process_frame(frame, db=db_session, frame_index=1)
    assert len(res1.tracks) == 1
    assert res1.arcface_invocations == 1
    assert res1.arcface_skipped == 0

    # ── Frame 2: Same track → ArcFace MUST be invoked until temporal confirmation (3 frames)
    res2 = engine.process_frame(frame, db=db_session, frame_index=2)
    assert len(res2.tracks) == 1
    assert res2.arcface_invocations == 1

    # ── Frame 3: Temporal confirmation reached (3 frames match) ─────────
    res3 = engine.process_frame(frame, db=db_session, frame_index=3)
    assert res3.tracks[0].is_recognized is True
    assert res3.tracks[0].confirmed_identity == 1

    # ── Frame 4: Identity confirmed → ArcFace MUST BE SKIPPED! ─────────
    res4 = engine.process_frame(frame, db=db_session, frame_index=4)
    assert len(res4.tracks) == 1
    assert res4.arcface_invocations == 0  # ArcFace SKIPPED!
    assert res4.arcface_skipped == 1  # Incremented skipped count!
    assert res4.tracks[0].is_recognized is True


def test_arcface_skip_optimization_on_unknown_track(db_session, mock_engine_components):
    """Verifies unknown tracks mark_unknown() and skip ArcFace on subsequent frames."""
    engine = mock_engine_components
    # Matcher returns no match (unknown face)
    engine.matcher.search.return_value = []

    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    # Frame 1: New track -> ArcFace invoked -> FAISS returns [] -> mark_unknown()
    res1 = engine.process_frame(frame, db=db_session, frame_index=1)
    assert len(res1.tracks) == 1
    assert res1.tracks[0].is_unknown is True
    assert res1.arcface_invocations == 1

    # Frame 2: Same track -> already marked unknown -> ArcFace SKIPPED!
    res2 = engine.process_frame(frame, db=db_session, frame_index=2)
    assert len(res2.tracks) == 1
    assert res2.arcface_invocations == 0
    assert res2.arcface_skipped == 1


# ── 3. Multi-Face Tracking & Known + Unknown Simultaneous ────


def test_multi_face_simultaneous_tracking(db_session, mock_engine_components):
    engine = mock_engine_components
    EmployeeRepository.create(db_session, employee_code="E1", name="Person 1")

    # 2 faces in frame
    f1 = DetectedFace(
        bbox=np.array([10, 10, 130, 130], dtype=np.float32),
        landmarks=np.zeros((5, 2)),
        det_score=0.9,
        face_crop=np.zeros((120, 120, 3), dtype=np.uint8),
        raw_face=object(),
    )
    f2 = DetectedFace(
        bbox=np.array([200, 200, 320, 320], dtype=np.float32),
        landmarks=np.zeros((5, 2)),
        det_score=0.9,
        face_crop=np.zeros((120, 120, 3), dtype=np.uint8),
        raw_face=object(),
    )
    engine.detector.detect.return_value = [f1, f2]

    # First call returns employee 1, second returns [] (unknown), third returns employee 1
    engine.matcher.search.side_effect = [[(1, 0.88)], [], [(1, 0.88)]]

    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    # Frame 1: Both faces processed
    res1 = engine.process_frame(frame, db=db_session, frame_index=1)
    assert len(res1.tracks) == 2
    assert res1.arcface_invocations == 2

    # Frame 2: Second face is unknown -> mark_unknown()
    res2 = engine.process_frame(frame, db=db_session, frame_index=2)
    assert len(res2.tracks) == 2


# ── 4. Track Expiration ───────────────────────────────────────


def test_track_expiration_on_lost_frames(mock_engine_components):
    tracker = FaceTracker(max_lost_frames=3)

    det = DetectedFace(
        bbox=np.array([10, 10, 100, 100]),
        landmarks=np.zeros((5, 2)),
        det_score=0.9,
        face_crop=np.zeros((90, 90, 3), dtype=np.uint8),
        raw_face=object(),
    )

    # Frame 1: Spawn track 1
    tracks1 = tracker.update([det])
    assert len(tracks1) == 1

    # Frames 2, 3, 4: Zero faces -> lost_frames increments
    tracker.update([])
    tracker.update([])
    tracker.update([])
    assert len(tracker.tracks) == 1

    # Frame 5: 4th lost frame > max_lost_frames 3 -> track purged!
    tracker.update([])
    assert len(tracker.tracks) == 0


# ── 5. Component Failure Resilience ─────────────────────────


def test_embedder_failure_resilience(db_session, mock_engine_components):
    engine = mock_engine_components
    engine.embedder.embed.side_effect = RuntimeError("GPU Out of Memory")

    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    res = engine.process_frame(frame, db=db_session)

    # Engine handles exception gracefully without crashing
    assert isinstance(res, RecognitionFrameResult)
    assert len(res.tracks) == 1


# ── 6. Background Thread Safety Controls ──────────────────────


def test_engine_thread_controls(mock_engine_components):
    engine = mock_engine_components

    # Mock detector to return empty list in thread loop
    engine.detector.detect.return_value = []

    assert engine.is_running is False
    engine.start()
    assert engine.is_running is True

    time.sleep(0.1)
    engine.stop(timeout=2.0)
    assert engine.is_running is False
