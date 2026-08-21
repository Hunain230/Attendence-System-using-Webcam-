"""
Recognition Engine — Asynchronous Pipeline Controller

Three-thread architecture:
  1. Capture thread  : reads webcam at full speed, puts frames into frame_queue.
  2. Recognition thread: pulls from frame_queue, runs detection → tracking →
                         quality gate → liveness → ArcFace → FAISS → attendance.
  3. MJPEG stream   : reads annotated_frame (shared buffer) independently.

This decoupling ensures:
  - Camera capture never blocks on ArcFace inference.
  - MJPEG stream stays smooth even if recognition takes longer.
  - Recognition thread never starves on slow capture.

Adaptive Recognition (via Track state machine):
  - new / evaluating : ArcFace every frame
  - confirmed        : ArcFace every RECOGNIZED_RECHECK_INTERVAL frames (catches identity switches)
  - uncertain        : ArcFace every 10 frames
  - unknown          : ArcFace every UNKNOWN_RETRY_INTERVAL frames (fixes permanent-Unknown bug)

Performance Metrics:
  All latency and resource stats are tracked per-loop and exposed via
  /api/recognition/metrics for the frontend diagnostic panel.
"""

from __future__ import annotations

import os
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app import config
from app.recognition.detector import FaceDetector, DetectedFace
from app.recognition.quality import QualityGate, QualityResult
from app.recognition.embedder import FaceEmbedder
from app.recognition.matcher import FaceMatcher, MatchResult
from app.recognition.tracker import FaceTracker, Track
from app.recognition.liveness import LivenessChecker
from app.attendance.service import AttendanceService
from app.database.database import SessionLocal
from app.database.repository import EmployeeRepository


@dataclass
class PerformanceMetrics:
    """Real-time performance statistics for the recognition pipeline."""
    fps: float = 0.0
    detection_latency_ms: float = 0.0
    arcface_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    arcface_invocations: int = 0
    arcface_skipped: int = 0
    active_tracks: int = 0
    cpu_percent: float = 0.0
    ram_mb: float = 0.0


@dataclass
class RecognitionFrameResult:
    """Detailed metadata result for a processed video frame."""

    frame_index: int
    timestamp: float
    tracks: List[Track]
    arcface_invocations: int
    arcface_skipped: int
    fps: float
    detection_latency_ms: float = 0.0
    arcface_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class RecognitionEngine:
    """Background engine running camera capture, detection, tracking, and recognition.

    Two-thread architecture separates capture from recognition so the MJPEG
    stream remains smooth regardless of ArcFace inference latency.
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        matcher: Optional[FaceMatcher] = None,
        quality_gate: Optional[QualityGate] = None,
        tracker: Optional[FaceTracker] = None,
        attendance_service: Optional[AttendanceService] = None,
        liveness_checker: Optional[LivenessChecker] = None,
    ):
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()
        self.matcher = matcher or FaceMatcher()
        self.quality_gate = quality_gate or QualityGate()
        self.tracker = tracker or FaceTracker()
        self.attendance_service = attendance_service or AttendanceService()
        self.liveness_checker = liveness_checker or LivenessChecker()

        self._running: bool = False
        self._capture_thread: Optional[threading.Thread] = None
        self._recognition_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Thread communication queues
        # maxsize=2: drops old frame if recognition is slow (keeps latency low)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # Shared results (written by recognition thread, read by MJPEG stream)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_result: Optional[RecognitionFrameResult] = None
        self._metrics = PerformanceMetrics()

        # FPS tracking (capture thread)
        self._capture_frame_count: int = 0
        self._capture_fps_time: float = time.time()
        self._capture_fps: float = 0.0

        # Debug overlay toggle (can be changed at runtime)
        self._debug_overlay: bool = config.DEBUG_OVERLAY

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Starts the background capture and recognition threads."""
        if self._running:
            return
        self.matcher.reload()
        self._running = True

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CaptureThread"
        )
        self._recognition_thread = threading.Thread(
            target=self._recognition_loop, daemon=True, name="RecognitionThread"
        )
        self._capture_thread.start()
        self._recognition_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stops the background threads cleanly."""
        self._running = False
        # Unblock recognition thread if waiting on queue
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        for t in (self._capture_thread, self._recognition_thread):
            if t is not None and t.is_alive():
                t.join(timeout=timeout)
        self._capture_thread = None
        self._recognition_thread = None

    def get_latest_result(self) -> Optional[RecognitionFrameResult]:
        """Thread-safe getter for the latest recognition result."""
        with self._lock:
            return self._latest_result

    def get_frame(self) -> Optional[np.ndarray]:
        """Thread-safe getter for the latest annotated frame (for MJPEG stream)."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_metrics(self) -> PerformanceMetrics:
        """Thread-safe getter for current performance metrics."""
        with self._lock:
            return self._metrics

    def set_debug_overlay(self, enabled: bool) -> None:
        """Toggles the diagnostic overlay at runtime."""
        self._debug_overlay = enabled

    # ── Capture thread ─────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Reads frames from camera and puts them into the frame queue."""
        # Support both USB index and IP/RTSP URL
        if config.CAMERA_URL:
            cap = cv2.VideoCapture(config.CAMERA_URL)
        else:
            cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize capture buffer lag

        self._capture_frame_count = 0
        self._capture_fps_time = time.time()

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                self._capture_frame_count += 1
                now = time.time()
                elapsed = now - self._capture_fps_time
                if elapsed >= 0.5:
                    self._capture_fps = self._capture_frame_count / elapsed
                    self._capture_frame_count = 0
                    self._capture_fps_time = now

                # Drop old frame if recognition is busy — keeps latency low
                try:
                    self._frame_queue.put_nowait((frame, self._capture_fps))
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._frame_queue.put_nowait((frame, self._capture_fps))
        finally:
            cap.release()

    # ── Recognition thread ─────────────────────────────────────────────────────

    def _recognition_loop(self) -> None:
        """Pulls frames from queue and runs the full recognition pipeline."""
        frame_index = 0
        db_session = SessionLocal()

        try:
            while self._running:
                try:
                    item = self._frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if item is None:
                    break  # Stop signal

                frame, capture_fps = item
                frame_index += 1

                result = self._process_frame(frame, db=db_session, frame_index=frame_index)
                annotated = self._annotate_frame(frame, result.tracks, capture_fps)

                # Update CPU/RAM metrics
                cpu_pct, ram_mb = self._get_resource_usage()

                with self._lock:
                    self._latest_frame = annotated
                    self._latest_result = result
                    self._metrics = PerformanceMetrics(
                        fps=capture_fps,
                        detection_latency_ms=result.detection_latency_ms,
                        arcface_latency_ms=result.arcface_latency_ms,
                        total_latency_ms=result.total_latency_ms,
                        arcface_invocations=result.arcface_invocations,
                        arcface_skipped=result.arcface_skipped,
                        active_tracks=len(result.tracks),
                        cpu_percent=cpu_pct,
                        ram_mb=ram_mb,
                    )
        finally:
            db_session.close()

    def _process_frame(
        self,
        frame: np.ndarray,
        db: Optional[Session] = None,
        frame_index: int = 0,
    ) -> RecognitionFrameResult:
        """Runs the complete recognition pipeline on a single frame."""
        loop_start = time.perf_counter()
        arcface_invocations = 0
        arcface_skipped = 0

        # ── Step 1: Detection (periodic) ──────────────────────────────────────
        det_start = time.perf_counter()
        run_detection = (
            frame_index % config.DETECTION_INTERVAL == 0
            or len(self.tracker.tracks) == 0
        )

        if run_detection:
            h, w = frame.shape[:2]
            target_w = config.PROCESS_WIDTH
            if w > target_w:
                scale = target_w / float(w)
                small = cv2.resize(frame, (target_w, int(h * scale)))
                detections = self.detector.detect(small)
                for det in detections:
                    det.bbox = det.bbox / scale
                    det.landmarks = det.landmarks / scale
                    x1 = max(0, int(det.bbox[0]))
                    y1 = max(0, int(det.bbox[1]))
                    x2 = min(w, int(det.bbox[2]))
                    y2 = min(h, int(det.bbox[3]))
                    det.face_crop = frame[y1:y2, x1:x2]
                    if hasattr(det.raw_face, "kps"):
                        det.raw_face.kps = det.landmarks
                    if hasattr(det.raw_face, "bbox"):
                        det.raw_face.bbox = det.bbox
            else:
                detections = self.detector.detect(frame)

            tracks = self.tracker.update(detections)
        else:
            tracks = self.tracker.update([])

        det_latency_ms = (time.perf_counter() - det_start) * 1000.0

        # ── Step 2: Liveness check (per track) ────────────────────────────────
        if config.LIVENESS_ENABLED:
            for track in tracks:
                self.liveness_checker.update(track)

        # ── Step 3: Per-track recognition ─────────────────────────────────────
        arcface_total_ms = 0.0
        created_session = False
        if db is None:
            db = SessionLocal()
            created_session = True

        try:
            for track in tracks:
                # Adaptive skip rule (state machine)
                if track.should_skip_recognition:
                    arcface_skipped += 1
                    continue

                # Quality gate
                quality_res = self.quality_gate.check(
                    track.face_crop, track.bbox, track.landmarks
                )
                track.last_quality_score = quality_res.score
                if quality_res.metrics:
                    track.last_yaw = quality_res.metrics.get("yaw", 0.0)
                    track.last_pitch = quality_res.metrics.get("pitch", 0.0)

                if not quality_res.passed:
                    continue

                # Liveness gate — block attendance but still run recognition for display
                if config.LIVENESS_ENABLED and track.liveness_failed:
                    arcface_skipped += 1
                    continue

                # ArcFace embedding
                arc_start = time.perf_counter()
                try:
                    embedding = self.embedder.embed(
                        frame, face_obj=track.raw_face, landmarks=track.landmarks
                    )
                    arcface_invocations += 1
                except Exception:
                    continue
                arcface_total_ms += (time.perf_counter() - arc_start) * 1000.0

                # FAISS matching with detailed result
                match: MatchResult = self.matcher.search_detailed(embedding)

                if not match.accepted:
                    track.record_unmatched()
                    continue

                # Employee name lookup
                employee = EmployeeRepository.get_by_id(db, match.best_employee_id)
                emp_name = employee.name if employee else f"Employee #{match.best_employee_id}"

                # Update track with full match details
                track.add_recognition(
                    employee_id=match.best_employee_id,
                    similarity=match.best_score,
                    name=emp_name,
                    second_id=match.second_employee_id,
                    second_similarity=match.second_score,
                    margin=match.margin,
                )

                # Attendance check-in (only when confirmed and liveness OK)
                if track.is_confirmed and not track.liveness_failed:
                    self.attendance_service.mark_if_needed(
                        employee_id=match.best_employee_id,
                        confidence=match.best_score,
                        db=db,
                    )

        finally:
            if created_session and db is not None:
                db.close()

        total_latency_ms = (time.perf_counter() - loop_start) * 1000.0
        avg_arcface_ms = arcface_total_ms / max(arcface_invocations, 1)

        return RecognitionFrameResult(
            frame_index=frame_index,
            timestamp=time.time(),
            tracks=tracks,
            arcface_invocations=arcface_invocations,
            arcface_skipped=arcface_skipped,
            fps=0.0,  # Filled in by _recognition_loop from capture_fps
            detection_latency_ms=det_latency_ms,
            arcface_latency_ms=avg_arcface_ms,
            total_latency_ms=total_latency_ms,
        )

    # ── Frame annotation ──────────────────────────────────────────────────────

    def _annotate_frame(
        self, frame: np.ndarray, tracks: List[Track], fps: float = 0.0
    ) -> np.ndarray:
        """Draws bounding boxes, labels, and optional debug overlay on a frame copy."""
        annotated = frame.copy()

        # FPS overlay (always shown)
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

            # Color coding by state
            state = track.display_state
            if state == "confirmed":
                color = (0, 220, 0)   # Green
                label = f"{track.confirmed_name or f'ID#{track.confirmed_identity}'} [{track.best_similarity:.2f}]"
            elif state == "unknown":
                color = (0, 0, 220)   # Red
                label = f"Unknown (T#{track.track_id})"
            elif state == "liveness_failed":
                color = (0, 165, 255) # Orange
                label = f"Liveness Fail (T#{track.track_id})"
            elif state == "uncertain":
                color = (0, 180, 255) # Yellow-orange
                label = f"Uncertain (T#{track.track_id})"
            else:
                color = (220, 220, 0) # Cyan for new/evaluating
                label = f"Checking... (T#{track.track_id})"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            # ── Debug overlay (only when enabled) ────────────────────────────
            if self._debug_overlay:
                self._draw_debug_overlay(annotated, track, x1, y1, x2, y2)

        return annotated

    def _draw_debug_overlay(
        self,
        frame: np.ndarray,
        track: Track,
        x1: int, y1: int, x2: int, y2: int,
    ) -> None:
        """Draws verbose diagnostic information below the bounding box."""
        debug_lines = [
            f"T#{track.track_id} | State: {track.state}",
            f"Q: {track.last_quality_score * 100:.0f} | Yaw: {track.last_yaw:+.1f} | Pitch: {track.last_pitch:+.1f}",
        ]

        # Show top recognition candidates from last history entry
        if track.recognition_history:
            last = track.recognition_history[-1]
            if last.employee_id is not None:
                debug_lines.append(f"Best: ID#{last.employee_id} {last.similarity:.3f}")
                if last.second_id is not None:
                    debug_lines.append(f"2nd:  ID#{last.second_id} {last.second_similarity:.3f}")
                debug_lines.append(f"Margin: {last.margin:.3f}")

        y_offset = y2 + 18
        for line in debug_lines:
            cv2.putText(
                frame,
                line,
                (x1, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 0),
                1,
                cv2.LINE_AA,
            )
            y_offset += 16

    # ── Resource monitoring ───────────────────────────────────────────────────

    @staticmethod
    def _get_resource_usage() -> tuple[float, float]:
        """Returns (cpu_percent, ram_mb). Lightweight — uses psutil if available."""
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            cpu = proc.cpu_percent(interval=None)
            ram = proc.memory_info().rss / (1024 * 1024)
            return cpu, ram
        except ImportError:
            return 0.0, 0.0

    # ── Legacy: synchronous single-frame processing (for tests) ───────────────

    def process_frame(
        self,
        frame: np.ndarray,
        db: Optional[Session] = None,
        frame_index: int = 0,
    ) -> RecognitionFrameResult:
        """Processes a single frame synchronously. Used by unit tests."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return RecognitionFrameResult(
                frame_index=frame_index,
                timestamp=time.time(),
                tracks=[],
                arcface_invocations=0,
                arcface_skipped=0,
                fps=0.0,
            )
        return self._process_frame(frame, db=db, frame_index=frame_index)

    def annotate_frame(
        self, frame: np.ndarray, tracks: List[Track], fps: float = 0.0
    ) -> np.ndarray:
        """Public wrapper for annotation (for tests)."""
        return self._annotate_frame(frame, tracks, fps)
