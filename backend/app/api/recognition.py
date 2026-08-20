"""
Recognition Engine API Router

Exposes REST endpoints for controlling the background RecognitionEngine (start/stop/status).
"""

from fastapi import APIRouter
from app.schemas.recognition import EngineStatusResponse, EngineActionResponse
from app.engine.recognition_engine import RecognitionEngine


router = APIRouter(prefix="/recognition", tags=["recognition"])

# Global background RecognitionEngine instance used across API
global_engine = RecognitionEngine()


@router.post("/start", response_model=EngineActionResponse)
def start_engine_route():
    """Starts the background recognition engine thread."""
    if global_engine.is_running:
        return EngineActionResponse(status="running", message="Engine is already running")

    global_engine.start()
    return EngineActionResponse(status="started", message="Engine started successfully")


@router.post("/stop", response_model=EngineActionResponse)
def stop_engine_route():
    """Stops the background recognition engine thread."""
    if not global_engine.is_running:
        return EngineActionResponse(status="stopped", message="Engine is not running")

    global_engine.stop(timeout=5.0)
    return EngineActionResponse(status="stopped", message="Engine stopped successfully")


@router.get("/status", response_model=EngineStatusResponse)
def get_engine_status_route():
    """Gets current background engine status and active tracks."""
    latest = global_engine.get_latest_result()
    active_tracks = len(latest.tracks) if latest else 0

    return EngineStatusResponse(
        running=global_engine.is_running,
        current_fps=global_engine._current_fps,
        active_tracks_count=active_tracks,
        latest_result={
            "frame_index": latest.frame_index if latest else 0,
            "arcface_invocations": latest.arcface_invocations if latest else 0,
            "arcface_skipped": latest.arcface_skipped if latest else 0,
        }
        if latest
        else None,
    )
