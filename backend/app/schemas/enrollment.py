"""
Enrollment API Pydantic Schemas
"""

from typing import Literal, Optional
from pydantic import BaseModel


class EnrollmentStartRequest(BaseModel):
    employee_code: str
    name: str
    department: Optional[str] = None


class EnrollmentStartResponse(BaseModel):
    employee_id: int
    employee_code: str
    name: str
    status: str
    current_pose: str
    instructions: str


class EnrollmentSampleResponse(BaseModel):
    success: bool
    reason: str
    captured: bool = False
    score: Optional[float] = None
    score_100: Optional[float] = None  # 0–100 display score
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
    phase: str = "guidance"        # guidance | holding | collecting | captured | complete
    hold_progress: int = 0         # frames held so far (Phase 1)
    hold_required: int = 0         # frames required to complete hold
    collect_progress: int = 0      # candidates collected so far (Phase 2)
    collect_required: int = 0      # candidates required
