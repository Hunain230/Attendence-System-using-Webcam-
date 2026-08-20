"""
Attendance API Pydantic Schemas
"""

from typing import Optional
from datetime import datetime, date
from pydantic import BaseModel, ConfigDict


class AttendanceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    employee_id: int
    employee_code: Optional[str] = None
    employee_name: Optional[str] = None
    date: date
    check_in: datetime
    check_out: Optional[datetime] = None
    confidence: float


class CheckoutResponse(BaseModel):
    success: bool
    message: str
    check_out: Optional[datetime] = None
