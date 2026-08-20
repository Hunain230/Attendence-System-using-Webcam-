"""
Attendance Service — First-Recognition Check-In & Explicit Checkout

Rules:
  1. First recognition of the day -> Automatically creates Check-In record.
  2. Subsequent recognitions same day -> Ignored (no duplicate rows created).
  3. Check-Out -> EXPLICIT ONLY via API/UI action (never automatic).
  4. In-memory cooldown cache avoids redundant SQLite DB queries for continuous video stream recognitions.
"""

from typing import List, Optional, Dict, Tuple
from datetime import datetime, date
from sqlalchemy.orm import Session

from app.database.database import SessionLocal
from app.database.models import Attendance
from app.database.repository import AttendanceRepository, EmployeeRepository


class AttendanceService:
    """Service handling attendance check-in, explicit checkout, and queries."""

    def __init__(self):
        # Cache of (employee_id, date) -> checked_in_flag to avoid DB queries on every video frame
        self._checked_in_cache: Dict[Tuple[int, date], bool] = {}

    def mark_if_needed(
        self, employee_id: int, confidence: float, db: Optional[Session] = None
    ) -> bool:
        """Attempts to mark check-in for an employee upon confirmed recognition.

        Rules:
          - First recognition today -> Creates check-in record (returns True).
          - Already checked in today -> Ignored (returns False).
        """
        today = date.today()
        cache_key = (employee_id, today)

        # Fast path: check in-memory cache first
        if self._checked_in_cache.get(cache_key, False):
            return False

        created_session = False
        if db is None:
            db = SessionLocal()
            created_session = True

        try:
            # Check if employee exists and is active
            emp = EmployeeRepository.get_by_id(db, employee_id)
            if not emp or not emp.active:
                return False

            # Check repository / DB if record exists today
            already_checked_in = AttendanceRepository.has_checked_in_today(
                db, employee_id
            )
            if already_checked_in:
                self._checked_in_cache[cache_key] = True
                return False

            # First recognition today -> perform check-in
            _ = AttendanceRepository.check_in(
                db, employee_id=employee_id, confidence=confidence
            )
            self._checked_in_cache[cache_key] = True
            return True

        finally:
            if created_session and db is not None:
                db.close()

    def explicit_checkout(
        self, employee_id: int, db: Optional[Session] = None
    ) -> bool:
        """Performs explicit checkout for an employee today. Called via API or UI button ONLY."""
        created_session = False
        if db is None:
            db = SessionLocal()
            created_session = True

        try:
            record = AttendanceRepository.check_out(db, employee_id=employee_id)
            return record is not None
        finally:
            if created_session and db is not None:
                db.close()

    def get_today_attendance(self, db: Session) -> List[Attendance]:
        """Returns all attendance records for today."""
        return AttendanceRepository.get_today(db)

    def get_attendance_by_date(
        self, db: Session, target_date: date
    ) -> List[Attendance]:
        """Returns all attendance records for a specific date."""
        return AttendanceRepository.get_by_date(db, target_date)

    def get_employee_attendance_history(
        self, db: Session, employee_id: int
    ) -> List[Attendance]:
        """Returns full attendance history for a given employee."""
        return (
            db.query(Attendance)
            .filter(Attendance.employee_id == employee_id)
            .order_by(Attendance.date.desc())
            .all()
        )

    def clear_cache(self):
        """Clears the in-memory check-in cache."""
        self._checked_in_cache.clear()
