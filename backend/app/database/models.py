"""
SQLAlchemy ORM Models

Defines database entities for:
  - Employee: Master record for registered employees
  - FaceEmbedding: FAISS vector mapping & sample quality scores
  - Attendance: Check-in & explicit check-out records per employee per day
"""

from datetime import datetime, date, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Float,
    DateTime,
    Date,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from app.database.database import Base


def utc_now() -> datetime:
    """Returns current UTC timestamp."""
    return datetime.now(timezone.utc)


class Employee(Base):
    """Employee master entity."""

    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    department = Column(String, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    active = Column(Boolean, default=True, nullable=False)

    # Relationships
    embeddings = relationship(
        "FaceEmbedding",
        back_populates="employee",
        cascade="all, delete-orphan",
    )
    attendance_records = relationship(
        "Attendance",
        back_populates="employee",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Employee(id={self.id}, code='{self.employee_code}', name='{self.name}')>"


class FaceEmbedding(Base):
    """Stores FAISS index key mapping and sample quality metadata."""

    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(
        Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    faiss_id = Column(Integer, nullable=False, index=True)
    quality_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships
    employee = relationship("Employee", back_populates="embeddings")

    def __repr__(self) -> str:
        return f"<FaceEmbedding(id={self.id}, employee_id={self.employee_id}, faiss_id={self.faiss_id})>"


class Attendance(Base):
    """Daily attendance check-in and check-out logs per employee."""

    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(
        Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date = Column(Date, nullable=False, default=date.today, index=True)
    check_in = Column(DateTime, default=utc_now, nullable=False)
    check_out = Column(DateTime, nullable=True)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Constraints: One attendance record per employee per day
    __table_args__ = (
        UniqueConstraint("employee_id", "date", name="uq_employee_date"),
    )

    # Relationships
    employee = relationship("Employee", back_populates="attendance_records")

    def __repr__(self) -> str:
        return (
            f"<Attendance(id={self.id}, employee_id={self.employee_id}, "
            f"date={self.date}, check_in='{self.check_in}', check_out='{self.check_out}')>"
        )
