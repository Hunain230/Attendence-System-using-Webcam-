"""
Recognition Engine — Background Pipeline Controller & Optimization Engine

Orchestrates the OpenCV camera frame -> SCRFD Detection -> IoU Face Tracking
-> Quality Gate -> ArcFace Feature Embedding -> FAISS Vector Matching pipeline.

CRITICAL PERFORMANCE OPTIMIZATION:
ArcFace feature extraction & FAISS matching are invoked ONLY for new or unconfirmed tracks.
Subsequent frames for already recognized or evaluated unknown tracks SKIP ArcFace completely,
amortizing heavy neural network inference costs across video stream frames.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import time
import threading
import numpy as np
import cv2
from sqlalchemy.orm import Session

from app import config
from app.recognition.detector import FaceDetector, DetectedFace
from app.recognition.quality import QualityGate, QualityResult
from app.recognition.embedder import FaceEmbedder
from app.recognition.matcher import FaceMatcher
from app.recognition.tracker import FaceTracker, Track
from app.database.database import SessionLocal
from app.database.repository import EmployeeRepository


@dataclass
class RecognitionFrameResult:
    """Detailed metadata result for a processed video frame."""

    frame_index: int
    timestamp: float
    tracks: List[Track]
    arcface_invocations: int  # Number of ArcFace inferences run in this frame
    arcface_skipped: int  # Number of ArcFace inferences skipped (already recognized/unknown)
    fps: float


class RecognitionEngine:
    """Background engine thread running camera capture, detection, tracking, and recognition."""

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        matcher: Optional[FaceMatcher] = None,
        quality_gate: Optional[QualityGate] = None,
        tracker: Optional[FaceTracker] = None,
    ):
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()
        self.matcher = matcher or FaceMatcher()
        self.quality_gate = quality_gate or QualityGate()
        self.tracker = tracker or FaceTracker()

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_result: Optional[RecognitionFrameResult] = None
        self._frame_count: int = 0
        self._last_fps_time: float = time.time()
        self._current_fps: float = 0.0

    @property
    def is_running(self) -> bool:
        """Returns True if the background engine thread is active."""
        return self._running

    def start(self):
        """Starts the background recognition thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        """Stops the background recognition thread."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            self._thread = None

    def get_latest_result(self) -> Optional[RecognitionFrameResult]:
        """Thread-safe getter for the latest frame recognition result."""
        with self._lock:
            return self._latest_result

    def get_frame(self) -> Optional[np.ndarray]:
        """Thread-safe getter for the latest annotated video frame (for MJPEG stream)."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def process_frame(
        self,
        frame: np.ndarray,
        db: Optional[Session] = None,
        frame_index: int = 0,
    ) -> RecognitionFrameResult:
        """Processes a single BGR image frame through the complete recognition engine pipeline.

        Pipeline Steps:
          1. SCRFD Detection: Detect faces in frame.
          2. IoU Tracking: Match detections to existing active tracks.
          3. For each track:
             a. Check ArcFace Skip Rule: If track is already recognized or marked unknown,
                SKIP ArcFace & FAISS matching!
             b. Quality Gate: Evaluate face size, blur, brightness, head pose.
             c. ArcFace Feature Embedding: Generate 512-D L2-normalized vector.
             d. FAISS Vector Search: Retrieve top nearest neighbor candidates.
             e. Identity Confirmation: Record candidate recognition and check temporal confirmation.
        """
        start_time = time.perf_counter()

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return RecognitionFrameResult(
                frame_index=frame_index,
                timestamp=time.time(),
                tracks=[],
                arcface_invocations=0,
                arcface_skipped=0,
                fps=self._current_fps,
            )

        # ── Step 1: SCRFD Detection ──────────────────────────────────
        detections = self.detector.detect(frame)

        # ── Step 2: IoU Face Tracking ────────────────────────────────
        tracks = self.tracker.update(detections)

        arcface_invocations = 0
        arcface_skipped = 0

        # DB session setup for employee name lookups
        created_session = False
        if db is None:
            db = SessionLocal()
            created_session = True

        try:
            # ── Step 3: Process Tracks ───────────────────────────────
            for track in tracks:
                # ── CRITICAL OPTIMIZATION: ArcFace Skip Rule ─────────
                if track.should_skip_recognition:
                    arcface_skipped += 1
                    continue

                # Step 3a: Quality Gate check
                quality_res = self.quality_gate.check(
                    track.face_crop, track.bbox, track.landmarks
                )
                if not quality_res.passed:
                    continue

                # Step 3b: ArcFace Feature Embedding
                try:
                    embedding = self.embedder.embed(frame, track.raw_face)
                    arcface_invocations += 1
                except Exception:
                    continue

                # Step 3c: FAISS Vector Matching
                matches = self.matcher.search(embedding)

                if not matches:
                    track.mark_unknown()
                    continue

                employee_id, confidence = matches[0]

                # Step 3d: Database lookup for employee name
                employee = EmployeeRepository.get_by_id(db, employee_id)
                emp_name = employee.name if employee else f"Employee #{employee_id}"

                # Step 3e: Temporal Confirmation
                track.add_recognition(employee_id, confidence, name=emp_name)

        finally:
            if created_session and db is not None:
                db.close()

        elapsed = time.perf_counter() - start_time
        instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0

        return RecognitionFrameResult(
            frame_index=frame_index,
            timestamp=time.time(),
            tracks=tracks,
            arcface_invocations=arcface_invocations,
            arcface_skipped=arcface_skipped,
            fps=instant_fps,
        )

    def _loop(self):
        """Main background thread capture & processing loop."""
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)

        self._frame_count = 0
        self._last_fps_time = time.time()

        db_session = SessionLocal()

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                self._frame_count += 1

                # Calculate rolling FPS
                now = time.time()
                if now - self._last_fps_time >= 1.0:
                    self._current_fps = self._frame_count / (now - self._last_fps_time)
                    self._frame_count = 0
                    self._last_fps_time = now

                # Process frame through recognition pipeline
                result = self.process_frame(frame, db=db_session, frame_index=self._frame_count)

                # Annotate frame for live display stream
                annotated = self.annotate_frame(frame, result.tracks, result.fps)

                with self._lock:
                    self._latest_frame = annotated
                    self._latest_result = result

        finally:
            db_session.close()
            cap.release()

    def annotate_frame(
        self, frame: np.ndarray, tracks: List[Track], fps: float = 0.0
    ) -> np.ndarray:
        """Draws bounding boxes, track IDs, recognized names, and performance stats on frame."""
        annotated = frame.copy()

        # FPS overlay
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        for track in tracks:
            bbox = track.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            if track.is_recognized:
                color = (0, 255, 0)  # Green for recognized
                label = f"{track.confirmed_name or f'ID #{track.confirmed_identity}'} (Track #{track.track_id})"
            elif track.is_unknown:
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown (Track #{track.track_id})"
            else:
                color = (255, 255, 0)  # Cyan for evaluating
                label = f"Evaluating... (Track #{track.track_id})"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated
