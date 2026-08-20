"""
Employee Enrollment API Router

Exposes REST endpoints for initiating and processing guided employee face sample enrollment.
"""

from typing import Optional
import base64
import numpy as np
import cv2
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

# Shared enrollment service instance using the global engine matcher
enrollment_service = EnrollmentService(matcher=global_engine.matcher)


@router.post("/start", response_model=EnrollmentStartResponse, status_code=status.HTTP_201_CREATED)
def start_enrollment_route(data: EnrollmentStartRequest, db: Session = Depends(get_db)):
    """Starts a guided face enrollment session for a new employee."""
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


class SampleFramePayload(EnrollmentStartRequest):
    employee_id: int
    image_base64: str  # Base64-encoded BGR/JPEG frame image


@router.post("/sample", response_model=EnrollmentSampleResponse)
def process_enrollment_sample_route(
    payload: SampleFramePayload, db: Session = Depends(get_db)
):
    """Processes a single frame for an enrollment sample."""
    try:
        # Decode base64 image
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
        db, employee_id=payload.employee_id, frame=frame
    )

    return EnrollmentSampleResponse(
        success=res.success,
        reason=res.reason,
        score=res.quality_result.score if res.quality_result else None,
        faiss_id=res.faiss_id,
        pose_name=res.pose_name,
    )
