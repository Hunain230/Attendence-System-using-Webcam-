"""
Employee Management & Enrollment API Helper Functions

Provides CRUD operations and enrollment entry points for employee entities.
"""

from typing import List, Optional, Tuple
from sqlalchemy.orm import Session

from app.database.models import Employee
from app.database.repository import EmployeeRepository, EmbeddingRepository
from app.recognition.enrollment import EnrollmentService, EnrollmentSampleResult, EnrollmentSession
from app.recognition.matcher import FaceMatcher


def create_employee(
    db: Session,
    employee_code: str,
    name: str,
    department: Optional[str] = None,
) -> Employee:
    """Creates a new employee record in SQLite."""
    existing = EmployeeRepository.get_by_code(db, employee_code)
    if existing:
        raise ValueError(f"Employee code '{employee_code}' already exists")

    return EmployeeRepository.create(
        db, employee_code=employee_code, name=name, department=department
    )


def get_employee(db: Session, employee_id: int) -> Optional[Employee]:
    """Gets an employee by ID."""
    return EmployeeRepository.get_by_id(db, employee_id)


def list_employees(db: Session, active_only: bool = True) -> List[Employee]:
    """Lists employees."""
    return EmployeeRepository.list_all(db, active_only=active_only)


def deactivate_employee(
    db: Session, employee_id: int, matcher: Optional[FaceMatcher] = None
) -> bool:
    """Deactivates an employee and removes their face embeddings from FAISS & DB."""
    success = EmployeeRepository.deactivate(db, employee_id)
    if success:
        # Delete embedding metadata records
        EmbeddingRepository.delete_by_employee(db, employee_id)
        if matcher is not None:
            matcher.remove_employee(employee_id)
    return success


def enroll_employee_sample(
    db: Session,
    enrollment_service: EnrollmentService,
    employee_id: int,
    frame: object,
    pose_name: Optional[str] = None,
) -> EnrollmentSampleResult:
    """Processes a single frame for an employee's enrollment sample."""
    return enrollment_service.process_frame(
        db, employee_id=employee_id, frame=frame, pose_name=pose_name
    )
