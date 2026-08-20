"""
Employee API Pydantic Schemas
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class EmployeeCreate(BaseModel):
    employee_code: str
    name: str
    department: Optional[str] = None


class EmployeeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    employee_code: str
    name: str
    department: Optional[str] = None
    created_at: datetime
    active: bool
