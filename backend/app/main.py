"""
FastAPI Application Entry Point & Lifespan Controller

Registers API routers, initializes CORS middleware, configures database lifecycle,
and controls background recognition engine shutdown.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.database import init_db
from app.api.employees import router as employees_router
from app.api.enrollment import router as enrollment_router
from app.api.attendance import router as attendance_router
from app.api.recognition import router as recognition_router, global_engine
from app.api.stream import router as stream_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup & shutdown tasks."""
    # Startup: Ensure SQLite database tables exist
    init_db()
    yield
    # Shutdown: Cleanly stop background engine thread
    global_engine.stop(timeout=5.0)


app = FastAPI(
    title="Attendance System API",
    description="Offline-first webcam attendance system API powered by OpenCV, InsightFace, FAISS, and SQLite.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for React frontend dev server (Vite default port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Route
@app.get("/health", tags=["system"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


# Register API Routers under /api
app.include_router(employees_router, prefix="/api")
app.include_router(enrollment_router, prefix="/api")
app.include_router(attendance_router, prefix="/api")
app.include_router(recognition_router, prefix="/api")
app.include_router(stream_router, prefix="/api")
