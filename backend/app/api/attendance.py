"""
Attendance API Router

Exposes REST endpoints for querying attendance logs and explicitly checking out employees.
"""

from typing import List, Optional
from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.attendance.service import AttendanceService
from app.schemas.attendance import AttendanceResponse, CheckoutResponse
from app.database.models import Employee


router = APIRouter(prefix="/attendance", tags=["attendance"])
attendance_service = AttendanceService()


def _format_record(rec) -> AttendanceResponse:
    return AttendanceResponse(
        id=rec.id,
        employee_id=rec.employee_id,
        employee_code=rec.employee.employee_code if rec.employee else None,
        employee_name=rec.employee.name if rec.employee else None,
        date=rec.date,
        check_in=rec.check_in,
        check_out=rec.check_out,
        confidence=rec.confidence,
    )


@router.get("", response_model=List[AttendanceResponse])
def get_attendance_route(
    target_date: Optional[date] = Query(None, alias="date"),
    db: Session = Depends(get_db),
):
    """Gets attendance records for a target date (defaults to today)."""
    d = target_date or date.today()
    records = attendance_service.get_attendance_by_date(db, d)
    return [_format_record(r) for r in records]


@router.get("/today", response_model=List[AttendanceResponse])
def get_today_attendance_route(db: Session = Depends(get_db)):
    """Gets today's attendance records."""
    records = attendance_service.get_today_attendance(db)
    return [_format_record(r) for r in records]


@router.get("/employee/{employee_id}", response_model=List[AttendanceResponse])
def get_employee_attendance_history_route(
    employee_id: int, db: Session = Depends(get_db)
):
    """Gets full attendance history for an employee."""
    records = attendance_service.get_employee_attendance_history(db, employee_id)
    return [_format_record(r) for r in records]


@router.post("/{employee_id}/checkout", response_model=CheckoutResponse)
def explicit_checkout_route(employee_id: int, db: Session = Depends(get_db)):
    """Explicitly checks out an employee today."""
    success = attendance_service.explicit_checkout(employee_id, db=db)
    if success:
        return CheckoutResponse(
            success=True,
            message=f"Employee #{employee_id} checked out successfully",
            check_out=datetime.now(),
        )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"No active check-in found today for employee #{employee_id}",
    )
