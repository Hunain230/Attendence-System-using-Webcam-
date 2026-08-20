"""
Database Repositories — CRUD Helpers

Encapsulates database access operations for Employees, FaceEmbeddings, and Attendance records.
"""

from datetime import date, datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.database.models import Employee, FaceEmbedding, Attendance


class EmployeeRepository:
    """Repository helper for Employee entity operations."""

    @staticmethod
    def create(
        db: Session, employee_code: str, name: str, department: Optional[str] = None
    ) -> Employee:
        """Creates and persists a new employee."""
        employee = Employee(
            employee_code=employee_code,
            name=name,
            department=department,
        )
        db.add(employee)
        db.commit()
        db.refresh(employee)
        return employee

    @staticmethod
    def get_by_id(db: Session, employee_id: int) -> Optional[Employee]:
        """Retrieves an employee by internal primary key ID."""
        return db.query(Employee).filter(Employee.id == employee_id).first()

    @staticmethod
    def get_by_code(db: Session, employee_code: str) -> Optional[Employee]:
        """Retrieves an employee by unique business code."""
        return db.query(Employee).filter(Employee.employee_code == employee_code).first()

    @staticmethod
    def list_all(db: Session, active_only: bool = True) -> List[Employee]:
        """Lists all employees in the database."""
        query = db.query(Employee)
        if active_only:
            query = query.filter(Employee.active.is_(True))
        return query.all()

    @staticmethod
    def update(
        db: Session,
        employee_id: int,
        name: Optional[str] = None,
        department: Optional[str] = None,
        active: Optional[bool] = None,
    ) -> Optional[Employee]:
        """Updates employee fields."""
        employee = EmployeeRepository.get_by_id(db, employee_id)
        if not employee:
            return None

        if name is not None:
            employee.name = name
        if department is not None:
            employee.department = department
        if active is not None:
            employee.active = active

        db.commit()
        db.refresh(employee)
        return employee

    @staticmethod
    def deactivate(db: Session, employee_id: int) -> bool:
        """Soft-deletes an employee by setting active to False."""
        employee = EmployeeRepository.get_by_id(db, employee_id)
        if not employee:
            return False
        employee.active = False
        db.commit()
        return True


class EmbeddingRepository:
    """Repository helper for FaceEmbedding metadata storage."""

    @staticmethod
    def add(
        db: Session, employee_id: int, faiss_id: int, quality_score: float
    ) -> FaceEmbedding:
        """Stores a new face embedding mapping."""
        embedding = FaceEmbedding(
            employee_id=employee_id,
            faiss_id=faiss_id,
            quality_score=quality_score,
        )
        db.add(embedding)
        db.commit()
        db.refresh(embedding)
        return embedding

    @staticmethod
    def get_by_employee(db: Session, employee_id: int) -> List[FaceEmbedding]:
        """Gets all embedding records associated with an employee."""
        return (
            db.query(FaceEmbedding)
            .filter(FaceEmbedding.employee_id == employee_id)
            .all()
        )

    @staticmethod
    def get_by_faiss_id(db: Session, faiss_id: int) -> Optional[FaceEmbedding]:
        """Maps a FAISS vector index ID to its DB metadata record."""
        return (
            db.query(FaceEmbedding)
            .filter(FaceEmbedding.faiss_id == faiss_id)
            .first()
        )

    @staticmethod
    def delete_by_employee(db: Session, employee_id: int) -> int:
        """Removes all embedding records for an employee."""
        count = (
            db.query(FaceEmbedding)
            .filter(FaceEmbedding.employee_id == employee_id)
            .delete()
        )
        db.commit()
        return count


class AttendanceRepository:
    """Repository helper for daily attendance check-in and check-out tracking."""

    @staticmethod
    def check_in(
        db: Session,
        employee_id: int,
        confidence: float,
        attendance_date: Optional[date] = None,
        check_in_time: Optional[datetime] = None,
    ) -> Optional[Attendance]:
        """Creates a check-in attendance record for today (or specified date).

        If an attendance record already exists for (employee_id, date), returns the existing record.
        """
        target_date = attendance_date or date.today()
        now = check_in_time or datetime.now(timezone.utc)

        existing = AttendanceRepository.get_by_employee_and_date(
            db, employee_id, target_date
        )
        if existing:
            return existing

        attendance = Attendance(
            employee_id=employee_id,
            date=target_date,
            check_in=now,
            confidence=confidence,
        )
        try:
            db.add(attendance)
            db.commit()
            db.refresh(attendance)
            return attendance
        except IntegrityError:
            db.rollback()
            return AttendanceRepository.get_by_employee_and_date(
                db, employee_id, target_date
            )

    @staticmethod
    def check_out(
        db: Session,
        employee_id: int,
        attendance_date: Optional[date] = None,
        check_out_time: Optional[datetime] = None,
    ) -> Optional[Attendance]:
        """Performs explicit check-out for an employee's attendance record on the given date."""
        target_date = attendance_date or date.today()
        now = check_out_time or datetime.now(timezone.utc)

        attendance = AttendanceRepository.get_by_employee_and_date(
            db, employee_id, target_date
        )
        if not attendance:
            return None

        attendance.check_out = now
        db.commit()
        db.refresh(attendance)
        return attendance

    @staticmethod
    def get_by_employee_and_date(
        db: Session, employee_id: int, attendance_date: date
    ) -> Optional[Attendance]:
        """Retrieves attendance record for a specific employee and date."""
        return (
            db.query(Attendance)
            .filter(
                Attendance.employee_id == employee_id,
                Attendance.date == attendance_date,
            )
            .first()
        )

    @staticmethod
    def has_checked_in_today(db: Session, employee_id: int) -> bool:
        """Returns True if the employee has a check-in record today."""
        record = AttendanceRepository.get_by_employee_and_date(
            db, employee_id, date.today()
        )
        return record is not None

    @staticmethod
    def get_today(db: Session) -> List[Attendance]:
        """Lists all attendance records for today."""
        return AttendanceRepository.get_by_date(db, date.today())

    @staticmethod
    def get_by_date(db: Session, target_date: date) -> List[Attendance]:
        """Lists all attendance records for a specific date."""
        return (
            db.query(Attendance)
            .filter(Attendance.date == target_date)
            .all()
        )
