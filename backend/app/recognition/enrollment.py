"""
Employee Enrollment Service

Implements the multi-sample guided enrollment protocol as specified by Implementation Plan v3.0.
Pipeline sequence per sample frame:
  Frame -> SCRFD Detection -> Face Selection -> Quality Gate (Stricter)
  -> Face Alignment & ArcFace Embedding -> L2 Normalization
  -> FAISS Insertion -> SQLite Metadata Persistence -> Transaction Consistency Rollback
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app import config
from app.recognition.detector import FaceDetector, DetectedFace
from app.recognition.quality import QualityGate, QualityResult
from app.recognition.embedder import FaceEmbedder
from app.recognition.matcher import FaceMatcher
from app.database.repository import EmployeeRepository, EmbeddingRepository
from app.database.models import Employee


@dataclass
class EnrollmentSampleResult:
    """Result of processing a single frame for an enrollment sample."""

    success: bool
    reason: str
    quality_result: Optional[QualityResult] = None
    faiss_id: Optional[int] = None
    pose_name: Optional[str] = None


class EnrollmentSession:
    """Manages multi-pose enrollment state for an employee."""

    POSES: List[Tuple[str, str, int]] = [
        ("straight", "Look straight at the camera", 2),
        ("slight_left", "Turn head slightly to the left", 1),
        ("slight_right", "Turn head slightly to the right", 1),
        ("slight_up", "Tilt head slightly upward", 1),
        ("slight_down", "Tilt head slightly downward", 1),
        ("smile", "Smile naturally", 1),
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

    @property
    def current_pose(self) -> Tuple[str, str, int]:
        if self.current_pose_idx < len(self.POSES):
            return self.POSES[self.current_pose_idx]
        return ("complete", "Enrollment complete", 0)

    @property
    def total_captured(self) -> int:
        return len(self.captured_faiss_ids)

    @property
    def is_complete(self) -> bool:
        return self.current_pose_idx >= len(self.POSES) or self.total_captured >= 7

    def advance_pose_if_needed(self):
        """Advances to the next pose when current pose target count is reached."""
        current_target = self.current_pose[2]
        if self.samples_for_current_pose >= current_target:
            self.current_pose_idx += 1
            self.samples_for_current_pose = 0
            if self.is_complete:
                self.status = "completed"


class EnrollmentService:
    """Service encapsulating the complete enrollment pipeline and transaction consistency."""

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

        # Stricter QualityGate thresholds for enrollment (Plan Section 10.6)
        self.quality_gate = quality_gate or QualityGate(
            min_face_size=(120, 120),
            blur_threshold=80.0,
            min_brightness=60.0,
            max_yaw=35.0,
            max_pitch=35.0,
        )

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

    def process_frame(
        self,
        db: Session,
        employee_id: int,
        frame: np.ndarray,
        pose_name: Optional[str] = None,
    ) -> EnrollmentSampleResult:
        """Processes a camera frame for enrollment.

        Follows strict pipeline order:
          1. Check employee active status
          2. SCRFD Detection
          3. Face Selection (Zero faces / Multiple faces handling)
          4. Quality Gate validation (Stricter enrollment thresholds)
          5. ArcFace Embedding & L2 Normalization
          6. FAISS Insertion & SQLite Metadata Mapping (Transaction Consistency)
        """
        # Step 0: Validate employee status
        employee = EmployeeRepository.get_by_id(db, employee_id)
        if not employee:
            return EnrollmentSampleResult(success=False, reason="employee_not_found")
        if not employee.active:
            return EnrollmentSampleResult(success=False, reason="employee_deactivated")

        # Step 1: SCRFD Detection
        faces = self.detector.detect(frame)

        # Step 2: Face Selection
        if not faces:
            return EnrollmentSampleResult(success=False, reason="no_face_detected")

        if len(faces) > 1:
            # Sort by face crop area (width * height) descending
            areas = [
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces
            ]
            sorted_indices = np.argsort(areas)[::-1]
            largest_area = areas[sorted_indices[0]]
            second_largest_area = areas[sorted_indices[1]]

            # If two faces are of comparable size (e.g. second is >70% of largest), reject as ambiguous
            if second_largest_area > 0.7 * largest_area:
                return EnrollmentSampleResult(
                    success=False, reason="multiple_faces_detected"
                )

            selected_face = faces[sorted_indices[0]]
        else:
            selected_face = faces[0]

        # Step 3: Quality Gate
        quality_res = self.quality_gate.check(
            selected_face.face_crop, selected_face.bbox, selected_face.landmarks
        )
        if not quality_res.passed:
            return EnrollmentSampleResult(
                success=False,
                reason=quality_res.reason,
                quality_result=quality_res,
            )

        # Step 4: ArcFace Embedding Generation & L2 Normalization
        try:
            embedding = self.embedder.embed(frame, selected_face.raw_face)
        except Exception as err:
            return EnrollmentSampleResult(
                success=False, reason=f"embedding_generation_failed: {err}"
            )

        # Step 5: Persistence & 2-System Transaction Consistency
        # A) Insert vector into FAISS
        faiss_id = None
        try:
            faiss_id = self.matcher.add(embedding, employee_id=employee_id)
        except Exception as err:
            return EnrollmentSampleResult(
                success=False, reason=f"faiss_insertion_failed: {err}"
            )

        # B) Insert metadata into SQLite
        try:
            EmbeddingRepository.add(
                db,
                employee_id=employee_id,
                faiss_id=faiss_id,
                quality_score=quality_res.score,
            )
        except Exception as err:
            # ── ROLLBACK CONSISTENCY SAFETY ─────────────────────────
            # SQLite insertion failed → remove the vector from FAISS so FAISS & SQLite stay synced
            db.rollback()
            if faiss_id is not None:
                self.matcher.remove_employee(employee_id)
            return EnrollmentSampleResult(
                success=False, reason=f"sqlite_persistence_failed: {err}"
            )

        # Session tracking update if active session exists
        if employee_id in self.active_sessions:
            sess = self.active_sessions[employee_id]
            sess.captured_faiss_ids.append(faiss_id)
            sess.quality_scores.append(quality_res.score)
            sess.samples_for_current_pose += 1
            sess.advance_pose_if_needed()

        return EnrollmentSampleResult(
            success=True,
            reason="passed",
            quality_result=quality_res,
            faiss_id=faiss_id,
            pose_name=pose_name,
        )
