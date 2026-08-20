"""
Enrollment API Pydantic Schemas
"""

from typing import Optional
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
    score: Optional[float] = None
    faiss_id: Optional[int] = None
    pose_name: Optional[str] = None
