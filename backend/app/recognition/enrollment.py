"""
Employee Enrollment Service

Implements a Windows-Hello-inspired two-phase quality-gated enrollment protocol.

Protocol (per pose):
  Phase 1 — HOLD DETECTION
    The face must satisfy all quality checks AND the target pose for
    ENROLLMENT_HOLD_FRAMES consecutive frames. Any failing frame resets
    the hold counter. This prevents capturing transitional frames.

  Phase 2 — CANDIDATE COLLECTION
    Once the hold is achieved, up to ENROLLMENT_CANDIDATE_FRAMES quality-checked
    frames are collected as candidates. Each frame is quality-scored.

  Phase 3 — BEST FRAME SELECTION
    The single frame with the highest composite quality score is selected.
    Its ArcFace embedding is computed using 5-point landmark alignment.
    The embedding is L2-normalized and validated before storage.

Pose State Machine (per sample):
  guidance  → holding → collecting → captured → [next pose / complete]
       ↑____________|
       (any fail during hold resets to guidance)

Key Improvements vs Previous Version:
  - No longer captures the FIRST frame that clears the angle.
  - Never claims success until the backend confirms the capture.
  - Pose ranges are tighter (from quality.py check_enrollment_pose).
  - Landmark stability is checked during candidate collection.
  - Embedding is validated (norm check) before storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import config
from app.recognition.detector import FaceDetector, DetectedFace
from app.recognition.embedder import FaceEmbedder
from app.recognition.matcher import FaceMatcher
from app.recognition.quality import QualityGate, QualityResult
from app.database.models import Employee
from app.database.repository import EmployeeRepository, EmbeddingRepository


EnrollmentPhase = Literal["guidance", "holding", "collecting", "captured", "complete"]


@dataclass
class EnrollmentSampleResult:
    """Result of processing a single frame for an enrollment sample."""

    success: bool
    reason: str
    captured: bool = False
    quality_result: Optional[QualityResult] = None
    faiss_id: Optional[int] = None
    pose_name: Optional[str] = None
    next_pose: Optional[str] = None
    next_instructions: Optional[str] = None
    guidance: Optional[str] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    samples_count: int = 0
    total_target: int = 7
    is_complete: bool = False

    # Phase-tracking fields for frontend state display
    phase: EnrollmentPhase = "guidance"
    hold_progress: int = 0           # frames held so far (Phase 1)
    hold_required: int = 0           # frames required to complete hold
    collect_progress: int = 0        # candidates collected so far (Phase 2)
    collect_required: int = 0        # candidates required


@dataclass
class _Candidate:
    """A single quality-checked candidate frame during enrollment."""
    score: float
    frame: np.ndarray
    landmarks: np.ndarray
    raw_face: object
    quality_result: QualityResult


@dataclass
class _PoseSession:
    """Per-pose state for the two-phase enrollment protocol."""
    phase: EnrollmentPhase = "guidance"
    hold_counter: int = 0
    candidates: List[_Candidate] = field(default_factory=list)
    prev_landmarks: Optional[np.ndarray] = None


class EnrollmentSession:
    """Manages multi-pose enrollment state for a single employee."""

    POSES: List[Tuple[str, str, int]] = [
        ("straight",    "Look straight directly into the camera lens", 2),
        ("slight_left",  "Turn head slightly to the left (← ~20°)",    1),
        ("slight_right", "Turn head slightly to the right (→ ~20°)",   1),
        ("slight_up",    "Tilt chin slightly upward (↑ ~12°)",          1),
        ("slight_down",  "Tilt chin slightly downward (↓ ~12°)",        1),
        ("smile",        "Smile naturally towards the camera lens",     1),
    ]

    def __init__(self, employee_id: int, employee_code: str, name: str):
        self.employee_id = employee_id
        self.employee_code = employee_code
        self.name = name
        self.current_pose_idx = 0
        self.samples_for_current_pose = 0
        self.captured_faiss_ids: List[int] = []
        self.quality_scores: List[float] = []
        self.status = "in_progress"

        # One PoseSession per pose index
        self._pose_sessions: Dict[int, _PoseSession] = {}

    @property
    def current_pose(self) -> Tuple[str, str, int]:
        if self.current_pose_idx < len(self.POSES):
            return self.POSES[self.current_pose_idx]
        return ("complete", "All angles successfully captured and indexed", 0)

    @property
    def total_captured(self) -> int:
        return len(self.captured_faiss_ids)

    @property
    def is_complete(self) -> bool:
        return self.current_pose_idx >= len(self.POSES) or self.total_captured >= 7

    @property
    def _current_pose_session(self) -> _PoseSession:
        idx = self.current_pose_idx
        if idx not in self._pose_sessions:
            self._pose_sessions[idx] = _PoseSession()
        return self._pose_sessions[idx]

    def advance_pose_if_needed(self) -> None:
        """Advances to the next pose when the current pose target count is reached."""
        current_target = self.current_pose[2]
        if self.samples_for_current_pose >= current_target:
            self.current_pose_idx += 1
            self.samples_for_current_pose = 0
            if self.is_complete:
                self.status = "completed"


class EnrollmentService:
    """Service encapsulating the complete two-phase enrollment pipeline."""

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        matcher: Optional[FaceMatcher] = None,
        quality_gate: Optional[QualityGate] = None,
    ):
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()
        self.matcher = matcher or FaceMatcher()
        self.quality_gate = quality_gate or QualityGate()

        self.active_sessions: Dict[int, EnrollmentSession] = {}

    def create_employee_and_start_session(
        self,
        db: Session,
        employee_code: str,
        name: str,
        department: Optional[str] = None,
    ) -> Tuple[Employee, EnrollmentSession]:
        """Creates a new employee in SQLite and starts an EnrollmentSession."""
        existing = EmployeeRepository.get_by_code(db, employee_code)
        if existing:
            raise ValueError(f"Employee code '{employee_code}' already exists")

        try:
            employee = EmployeeRepository.create(
                db, employee_code=employee_code, name=name, department=department
            )
        except IntegrityError as err:
            db.rollback()
            raise ValueError(f"Employee creation failed: {err}") from err

        session = EnrollmentSession(
            employee_id=employee.id,
            employee_code=employee.employee_code,
            name=employee.name,
        )
        self.active_sessions[employee.id] = session
        return employee, session

    def reset_employee_embeddings(self, db: Session, employee_id: int) -> bool:
        """Removes all existing embeddings for the employee and resets their session.

        Used by POST /api/enrollment/reset/{employee_id}.
        Enables re-enrollment without deleting the employee record.
        """
        from app.database.models import FaceEmbedding

        # Remove from FAISS
        self.matcher.remove_employee(employee_id)

        # Remove from SQLite embedding table
        try:
            db.query(FaceEmbedding).filter(
                FaceEmbedding.employee_id == employee_id
            ).delete()
            db.commit()
        except Exception:
            db.rollback()
            return False

        # Reset the in-memory session
        employee = EmployeeRepository.get_by_id(db, employee_id)
        if employee:
            session = EnrollmentSession(
                employee_id=employee.id,
                employee_code=employee.employee_code,
                name=employee.name,
            )
            self.active_sessions[employee_id] = session

        return True

    def process_frame(
        self,
        db: Session,
        employee_id: int,
        frame: np.ndarray,
        pose_name: Optional[str] = None,
        force_capture: bool = False,
    ) -> EnrollmentSampleResult:
        """Processes a camera frame through the two-phase enrollment protocol.

        Phase 1 (HOLD DETECTION):
          The pose must be valid for ENROLLMENT_HOLD_FRAMES consecutive frames.

        Phase 2 (CANDIDATE COLLECTION):
          Up to ENROLLMENT_CANDIDATE_FRAMES quality frames are collected.
          The best-scoring frame is selected, embedded, and stored.

        Phase states returned in result.phase:
          "guidance"   — pose not yet matching; show directional guidance
          "holding"    — pose matches; counting hold frames
          "collecting" — hold complete; collecting candidate frames
          "captured"   — best frame selected and committed (result.captured = True)
          "complete"   — all poses done
        """
        # ── Validate employee ──────────────────────────────────────────────────
        employee = EmployeeRepository.get_by_id(db, employee_id)
        if not employee:
            return EnrollmentSampleResult(success=False, reason="employee_not_found")
        if not employee.active:
            return EnrollmentSampleResult(success=False, reason="employee_deactivated")

        # ── Get or create active session ──────────────────────────────────────
        sess = self.active_sessions.get(employee_id)
        if sess is None:
            sess = EnrollmentSession(
                employee_id=employee.id,
                employee_code=employee.employee_code,
                name=employee.name,
            )
            self.active_sessions[employee_id] = sess

        if sess.is_complete:
            return EnrollmentSampleResult(
                success=True,
                reason="already_complete",
                is_complete=True,
                samples_count=sess.total_captured,
                total_target=7,
                phase="complete",
            )

        target_pose = pose_name or sess.current_pose[0]
        pose_session = sess._current_pose_session

        # ── Step 1: Face Detection ────────────────────────────────────────────
        faces = self.detector.detect(frame)

        if not faces:
            pose_session.hold_counter = 0  # Reset hold on no detection
            pose_session.prev_landmarks = None
            return self._guidance_result(
                sess, target_pose, pose_session,
                guidance="No face detected — position yourself in the camera frame",
            )

        # Step 2: Face Selection — reject if multiple comparable faces
        selected_face = self._select_face(faces)
        if selected_face is None:
            pose_session.hold_counter = 0
            return self._guidance_result(
                sess, target_pose, pose_session,
                guidance="Multiple faces detected — only one person should be in frame",
                reason="multiple_faces_detected",
            )

        # ── Step 3: Basic quality check ───────────────────────────────────────
        quality_res = self.quality_gate.check_enrollment_strict(
            selected_face.face_crop, selected_face.bbox, selected_face.landmarks
        )

        if not quality_res.passed:
            pose_session.hold_counter = 0  # Reset hold on quality failure
            pose_session.prev_landmarks = None
            guidance_map = {
                "face_too_small": "Move closer to the camera",
                "too_dark": "Improve lighting — face is too dark",
                "too_blurry": "Stay still — image is blurry",
                "occluded_or_extreme_angle": "Remove obstacles and face the camera",
                "quality_too_low": "Adjust position for better quality",
            }
            return self._guidance_result(
                sess, target_pose, pose_session,
                guidance=guidance_map.get(quality_res.reason, quality_res.reason.replace("_", " ")),
                reason=quality_res.reason,
                quality_result=quality_res,
                yaw=quality_res.metrics.get("yaw"),
                pitch=quality_res.metrics.get("pitch"),
            )

        # ── Step 4: Pose angle check ──────────────────────────────────────────
        matches_angle, pose_guidance, yaw, pitch = self.quality_gate.check_enrollment_pose(
            selected_face.landmarks, target_pose
        )

        if not matches_angle and not force_capture:
            pose_session.hold_counter = 0  # Reset hold if pose drifts
            pose_session.prev_landmarks = None
            return self._guidance_result(
                sess, target_pose, pose_session,
                guidance=pose_guidance,
                reason="pose_angle_not_reached",
                quality_result=quality_res,
                yaw=yaw,
                pitch=pitch,
            )

        # ── Phase 1: HOLD DETECTION ───────────────────────────────────────────
        if pose_session.phase == "guidance":
            pose_session.phase = "holding"
            pose_session.hold_counter = 0

        if pose_session.phase == "holding":
            # Check landmark stability between frames
            is_stable = self.quality_gate.check_landmark_stability(
                pose_session.prev_landmarks, selected_face.landmarks
            )
            pose_session.prev_landmarks = selected_face.landmarks.copy()

            if is_stable:
                pose_session.hold_counter += 1
            else:
                # Face is moving — reset hold counter but keep phase
                pose_session.hold_counter = max(0, pose_session.hold_counter - 1)

            hold_required = config.ENROLLMENT_HOLD_FRAMES

            if pose_session.hold_counter < hold_required:
                return EnrollmentSampleResult(
                    success=True,
                    reason="holding",
                    captured=False,
                    guidance=f"Hold still... ({pose_session.hold_counter}/{hold_required})",
                    yaw=yaw,
                    pitch=pitch,
                    quality_result=quality_res,
                    pose_name=target_pose,
                    samples_count=sess.total_captured,
                    total_target=7,
                    is_complete=sess.is_complete,
                    phase="holding",
                    hold_progress=pose_session.hold_counter,
                    hold_required=hold_required,
                )

            # Hold complete → transition to candidate collection
            pose_session.phase = "collecting"
            pose_session.candidates = []
            pose_session.prev_landmarks = None

        # ── Phase 2: CANDIDATE COLLECTION ────────────────────────────────────
        if pose_session.phase == "collecting":
            collect_required = config.ENROLLMENT_CANDIDATE_FRAMES

            # Check landmark stability for candidate frame
            is_stable = self.quality_gate.check_landmark_stability(
                pose_session.prev_landmarks, selected_face.landmarks
            )
            pose_session.prev_landmarks = selected_face.landmarks.copy()

            if is_stable and quality_res.passed:
                candidate = _Candidate(
                    score=quality_res.score,
                    frame=frame.copy(),
                    landmarks=selected_face.landmarks.copy(),
                    raw_face=selected_face.raw_face,
                    quality_result=quality_res,
                )
                pose_session.candidates.append(candidate)

            collected = len(pose_session.candidates)

            if collected < collect_required:
                return EnrollmentSampleResult(
                    success=True,
                    reason="collecting",
                    captured=False,
                    guidance=f"Capturing best frame... ({collected}/{collect_required})",
                    yaw=yaw,
                    pitch=pitch,
                    quality_result=quality_res,
                    pose_name=target_pose,
                    samples_count=sess.total_captured,
                    total_target=7,
                    is_complete=sess.is_complete,
                    phase="collecting",
                    hold_progress=config.ENROLLMENT_HOLD_FRAMES,
                    hold_required=config.ENROLLMENT_HOLD_FRAMES,
                    collect_progress=collected,
                    collect_required=collect_required,
                )

            # ── Phase 3: BEST FRAME SELECTION ────────────────────────────────
            return self._commit_best_candidate(db, sess, pose_session, target_pose, yaw, pitch)

        # Fallback (shouldn't reach here)
        return EnrollmentSampleResult(success=False, reason="invalid_phase_state")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _commit_best_candidate(
        self,
        db: Session,
        sess: EnrollmentSession,
        pose_session: _PoseSession,
        target_pose: str,
        yaw: float,
        pitch: float,
    ) -> EnrollmentSampleResult:
        """Selects the best candidate frame, generates its embedding, and commits to storage."""
        candidates = pose_session.candidates
        if not candidates:
            pose_session.phase = "guidance"
            pose_session.hold_counter = 0
            return EnrollmentSampleResult(
                success=False,
                reason="no_valid_candidates",
                guidance="Could not collect quality frames. Please try again.",
                pose_name=target_pose,
                samples_count=sess.total_captured,
                total_target=7,
                is_complete=sess.is_complete,
                phase="guidance",
            )

        # Select the frame with the highest quality score
        best = max(candidates, key=lambda c: c.score)

        # Generate ArcFace embedding
        try:
            embedding = self.embedder.embed(best.frame, face_obj=best.raw_face, landmarks=best.landmarks)
        except Exception as err:
            pose_session.phase = "guidance"
            pose_session.hold_counter = 0
            return EnrollmentSampleResult(
                success=False,
                reason=f"embedding_generation_failed: {err}",
                guidance="Failed to generate face embedding — try again",
                pose_name=target_pose,
                samples_count=sess.total_captured,
                total_target=7,
                is_complete=sess.is_complete,
                phase="guidance",
            )

        # Validate embedding norm (should be ~1.0 for L2-normalized vectors)
        norm = float(np.linalg.norm(embedding))
        if abs(norm - 1.0) > 0.05 or norm == 0.0:
            pose_session.phase = "guidance"
            pose_session.hold_counter = 0
            return EnrollmentSampleResult(
                success=False,
                reason="invalid_embedding_norm",
                guidance="Embedding validation failed — try again",
                pose_name=target_pose,
                samples_count=sess.total_captured,
                total_target=7,
                is_complete=sess.is_complete,
                phase="guidance",
            )

        # ── FAISS insertion ───────────────────────────────────────────────────
        faiss_id = None
        try:
            faiss_id = self.matcher.add(embedding, employee_id=sess.employee_id)
        except Exception as err:
            pose_session.phase = "guidance"
            pose_session.hold_counter = 0
            return EnrollmentSampleResult(
                success=False,
                reason=f"faiss_insertion_failed: {err}",
                guidance="FAISS indexing failed",
                pose_name=target_pose,
                samples_count=sess.total_captured,
                total_target=7,
                is_complete=sess.is_complete,
                phase="guidance",
            )

        # ── SQLite metadata insertion ─────────────────────────────────────────
        try:
            EmbeddingRepository.add(
                db,
                employee_id=sess.employee_id,
                faiss_id=faiss_id,
                quality_score=best.score,
            )
        except Exception as err:
            db.rollback()
            if faiss_id is not None:
                self.matcher.remove_employee(sess.employee_id)
            pose_session.phase = "guidance"
            pose_session.hold_counter = 0
            return EnrollmentSampleResult(
                success=False,
                reason=f"sqlite_persistence_failed: {err}",
                guidance="Database recording failed",
                pose_name=target_pose,
                samples_count=sess.total_captured,
                total_target=7,
                is_complete=sess.is_complete,
                phase="guidance",
            )

        # ── Session state update ──────────────────────────────────────────────
        sess.captured_faiss_ids.append(faiss_id)
        sess.quality_scores.append(best.score)
        sess.samples_for_current_pose += 1
        sess.advance_pose_if_needed()

        # Reset pose session for next pose
        pose_session.phase = "guidance" if not sess.is_complete else "complete"
        pose_session.hold_counter = 0
        pose_session.candidates = []
        pose_session.prev_landmarks = None

        next_pose_name, next_pose_desc, _ = sess.current_pose

        return EnrollmentSampleResult(
            success=True,
            captured=True,
            reason="passed",
            guidance="Captured ✓",
            yaw=yaw,
            pitch=pitch,
            quality_result=best.quality_result,
            faiss_id=faiss_id,
            pose_name=target_pose,
            next_pose=next_pose_name,
            next_instructions=next_pose_desc,
            samples_count=sess.total_captured,
            total_target=7,
            is_complete=sess.is_complete,
            phase="captured",
            hold_progress=config.ENROLLMENT_HOLD_FRAMES,
            hold_required=config.ENROLLMENT_HOLD_FRAMES,
            collect_progress=config.ENROLLMENT_CANDIDATE_FRAMES,
            collect_required=config.ENROLLMENT_CANDIDATE_FRAMES,
        )

    def _select_face(self, faces: List[DetectedFace]) -> Optional[DetectedFace]:
        """Selects the single subject face from a list of detections.

        Returns None if multiple comparable-sized faces are present (ambiguous).
        """
        if len(faces) == 1:
            return faces[0]

        areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        sorted_idx = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
        largest = areas[sorted_idx[0]]
        second = areas[sorted_idx[1]]

        # If two faces are comparable size (> 70%), reject as ambiguous
        if second > 0.7 * largest:
            return None

        return faces[sorted_idx[0]]

    def _guidance_result(
        self,
        sess: EnrollmentSession,
        target_pose: str,
        pose_session: _PoseSession,
        guidance: str,
        reason: str = "pose_angle_not_reached",
        quality_result: Optional[QualityResult] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
    ) -> EnrollmentSampleResult:
        """Returns a guidance-phase result (no capture)."""
        pose_session.phase = "guidance"
        return EnrollmentSampleResult(
            success=False,
            reason=reason,
            captured=False,
            guidance=guidance,
            yaw=yaw,
            pitch=pitch,
            quality_result=quality_result,
            pose_name=target_pose,
            samples_count=sess.total_captured,
            total_target=7,
            is_complete=sess.is_complete,
            phase="guidance",
            hold_required=config.ENROLLMENT_HOLD_FRAMES,
            collect_required=config.ENROLLMENT_CANDIDATE_FRAMES,
        )
