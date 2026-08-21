"""
Employee Enrollment API Router

Exposes REST endpoints for guided face enrollment.

Endpoints:
  POST /api/enrollment/start                    — Create employee + start session
  POST /api/enrollment/start_session/{emp_id}   — Start/reset session for existing employee
  POST /api/enrollment/sample                   — Process a single enrollment frame
  POST /api/enrollment/reset/{employee_id}      — Delete embeddings + restart enrollment
"""

from typing import Optional
import base64

import numpy as np
import cv2
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.recognition.enrollment import EnrollmentService, EnrollmentSampleResult
from app.api.recognition import global_engine
from app.schemas.enrollment import (
    EnrollmentStartRequest,
    EnrollmentStartResponse,
    EnrollmentSampleResponse,
)


router = APIRouter(prefix="/enrollment", tags=["enrollment"])

# Shared enrollment service instance — uses the global engine's matcher
# so FAISS stays synchronized between enrollment and recognition.
enrollment_service = EnrollmentService(matcher=global_engine.matcher)


@router.post("/start", response_model=EnrollmentStartResponse, status_code=status.HTTP_201_CREATED)
def start_enrollment_route(data: EnrollmentStartRequest, db: Session = Depends(get_db)):
    """Creates a new employee record and starts a guided enrollment session."""
    try:
        emp, session = enrollment_service.create_employee_and_start_session(
            db,
            employee_code=data.employee_code,
            name=data.name,
            department=data.department,
        )
        pose_name, pose_desc, _ = session.current_pose
        return EnrollmentStartResponse(
            employee_id=emp.id,
            employee_code=emp.employee_code,
            name=emp.name,
            status=session.status,
            current_pose=pose_name,
            instructions=pose_desc,
        )
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)
        ) from err


@router.post("/start_session/{employee_id}", response_model=EnrollmentStartResponse)
def start_existing_employee_session_route(employee_id: int, db: Session = Depends(get_db)):
    """Starts or resets an enrollment session for an existing employee.

    Does NOT delete existing embeddings — use /reset/{employee_id} for that.
    """
    from app.database.repository import EmployeeRepository
    from app.recognition.enrollment import EnrollmentSession

    emp = EmployeeRepository.get_by_id(db, employee_id)
    if not emp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Employee #{employee_id} not found",
        )

    session = EnrollmentSession(
        employee_id=emp.id,
        employee_code=emp.employee_code,
        name=emp.name,
    )
    enrollment_service.active_sessions[emp.id] = session
    pose_name, pose_desc, _ = session.current_pose

    return EnrollmentStartResponse(
        employee_id=emp.id,
        employee_code=emp.employee_code,
        name=emp.name,
        status=session.status,
        current_pose=pose_name,
        instructions=pose_desc,
    )


@router.post("/reset/{employee_id}", response_model=EnrollmentStartResponse)
def reset_employee_embeddings_route(employee_id: int, db: Session = Depends(get_db)):
    """Deletes all existing FAISS embeddings + SQLite metadata for an employee
    and starts a fresh enrollment session.

    Use this when recognition quality is poor and re-enrollment is needed.
    The employee record itself is preserved.
    """
    from app.database.repository import EmployeeRepository

    emp = EmployeeRepository.get_by_id(db, employee_id)
    if not emp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Employee #{employee_id} not found",
        )

    success = enrollment_service.reset_employee_embeddings(db, employee_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset embeddings. Check server logs.",
        )

    # Reload matcher in the global engine so recognition also sees the reset
    global_engine.matcher.reload()

    session = enrollment_service.active_sessions.get(employee_id)
    pose_name, pose_desc, _ = session.current_pose if session else ("straight", "Look straight", 2)

    return EnrollmentStartResponse(
        employee_id=emp.id,
        employee_code=emp.employee_code,
        name=emp.name,
        status="in_progress",
        current_pose=pose_name,
        instructions=pose_desc,
    )


class SampleFramePayload(BaseModel):
    employee_id: int
    employee_code: Optional[str] = None
    name: Optional[str] = None
    image_base64: str  # Base64-encoded JPEG frame
    pose_name: Optional[str] = None
    force_capture: bool = False


@router.post("/sample", response_model=EnrollmentSampleResponse)
def process_enrollment_sample_route(
    payload: SampleFramePayload, db: Session = Depends(get_db)
):
    """Processes a single camera frame through the two-phase enrollment protocol.

    Response includes `phase` field:
      "guidance"   — pose not matching; show directional hint
      "holding"    — counting hold frames; show progress bar
      "collecting" — collecting candidate frames; show "Capturing best frame..."
      "captured"   — best frame committed; show "Captured ✓"
      "complete"   — all poses done

    The frontend MUST wait for captured=True before claiming success to the user.
    """
    try:
        img_bytes = base64.b64decode(payload.image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            raise ValueError("Decoded image frame is empty or invalid")
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {err}",
        ) from err

    res: EnrollmentSampleResult = enrollment_service.process_frame(
        db,
        employee_id=payload.employee_id,
        frame=frame,
        pose_name=payload.pose_name,
        force_capture=payload.force_capture,
    )

    return EnrollmentSampleResponse(
        success=res.success,
        reason=res.reason,
        captured=res.captured,
        score=res.quality_result.score if res.quality_result else None,
        score_100=res.quality_result.score_100 if res.quality_result else None,
        faiss_id=res.faiss_id,
        pose_name=res.pose_name,
        next_pose=res.next_pose,
        next_instructions=res.next_instructions,
        guidance=res.guidance,
        yaw=res.yaw,
        pitch=res.pitch,
        samples_count=res.samples_count,
        total_target=res.total_target,
        is_complete=res.is_complete,
        phase=res.phase,
        hold_progress=res.hold_progress,
        hold_required=res.hold_required,
        collect_progress=res.collect_progress,
        collect_required=res.collect_required,
    )
