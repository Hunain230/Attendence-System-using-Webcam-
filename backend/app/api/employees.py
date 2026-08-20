"""
Employee Management API Router

Exposes REST endpoints for employee creation, listing, detail retrieval, and deactivation.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.repository import EmployeeRepository, EmbeddingRepository
from app.schemas.employee import EmployeeCreate, EmployeeResponse
from app.recognition.matcher import FaceMatcher


router = APIRouter(prefix="/employees", tags=["employees"])


@router.post("", response_model=EmployeeResponse, status_code=status.HTTP_201_CREATED)
def create_employee_route(data: EmployeeCreate, db: Session = Depends(get_db)):
    """Creates a new employee."""
    existing = EmployeeRepository.get_by_code(db, data.employee_code)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Employee code '{data.employee_code}' already exists",
        )

    try:
        employee = EmployeeRepository.create(
            db,
            employee_code=data.employee_code,
            name=data.name,
            department=data.department,
        )
        return employee
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create employee: {err}",
        ) from err


@router.get("", response_model=List[EmployeeResponse])
def list_employees_route(active_only: bool = True, db: Session = Depends(get_db)):
    """Lists employees."""
    return EmployeeRepository.list_all(db, active_only=active_only)


@router.get("/{employee_id}", response_model=EmployeeResponse)
def get_employee_route(employee_id: int, db: Session = Depends(get_db)):
    """Gets an employee by ID."""
    employee = EmployeeRepository.get_by_id(db, employee_id)
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Employee #{employee_id} not found",
        )
    return employee


@router.delete("/{employee_id}")
def deactivate_employee_route(employee_id: int, db: Session = Depends(get_db)):
    """Deactivates an employee and removes their face embeddings."""
    employee = EmployeeRepository.get_by_id(db, employee_id)
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Employee #{employee_id} not found",
        )

    success = EmployeeRepository.deactivate(db, employee_id)
    if success:
        EmbeddingRepository.delete_by_employee(db, employee_id)
        # Matcher vector removal from global engine
        from app.api.recognition import global_engine
        global_engine.matcher.remove_employee(employee_id)
        return {"success": True, "message": f"Employee #{employee_id} deactivated"}

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Failed to deactivate employee #{employee_id}",
    )


# ── Helper Functions for Direct Python Invocation & Unit Tests ──────


def create_employee(
    db: Session,
    employee_code: str,
    name: str,
    department: Optional[str] = None,
):
    """Creates a new employee record in SQLite."""
    existing = EmployeeRepository.get_by_code(db, employee_code)
    if existing:
        raise ValueError(f"Employee code '{employee_code}' already exists")

    return EmployeeRepository.create(
        db, employee_code=employee_code, name=name, department=department
    )


def deactivate_employee(
    db: Session, employee_id: int, matcher: Optional[FaceMatcher] = None
) -> bool:
    """Deactivates an employee and removes their face embeddings from FAISS & DB."""
    success = EmployeeRepository.deactivate(db, employee_id)
    if success:
        EmbeddingRepository.delete_by_employee(db, employee_id)
        if matcher is not None:
            matcher.remove_employee(employee_id)
    return success


def enroll_employee_sample(
    db: Session,
    enrollment_service: object,
    employee_id: int,
    frame: object,
    pose_name: Optional[str] = None,
):
    """Processes a single frame for an employee's enrollment sample."""
    return enrollment_service.process_frame(
        db, employee_id=employee_id, frame=frame, pose_name=pose_name
    )
