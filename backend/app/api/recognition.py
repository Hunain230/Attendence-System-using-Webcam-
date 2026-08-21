"""
Recognition Engine API Router

Exposes REST endpoints for controlling the background RecognitionEngine.

Endpoints:
  POST /api/recognition/start        — Start the background engine
  POST /api/recognition/stop         — Stop the background engine
  GET  /api/recognition/status       — Engine status + active track count
  GET  /api/recognition/metrics      — Detailed performance metrics (FPS, latency, CPU)
  POST /api/recognition/debug/{enabled} — Toggle diagnostic overlay at runtime
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.engine.recognition_engine import RecognitionEngine
from app.schemas.recognition import EngineStatusResponse, EngineActionResponse


router = APIRouter(prefix="/recognition", tags=["recognition"])

# Global background RecognitionEngine instance used across API
global_engine = RecognitionEngine()


class MetricsResponse(BaseModel):
    fps: float
    detection_latency_ms: float
    arcface_latency_ms: float
    total_latency_ms: float
    arcface_invocations: int
    arcface_skipped: int
    active_tracks: int
    cpu_percent: float
    ram_mb: float


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
    """Gets current background engine status and active track count."""
    latest = global_engine.get_latest_result()
    active_tracks = len(latest.tracks) if latest else 0
    metrics = global_engine.get_metrics()

    return EngineStatusResponse(
        running=global_engine.is_running,
        current_fps=metrics.fps,
        active_tracks_count=active_tracks,
        latest_result={
            "frame_index": latest.frame_index if latest else 0,
            "arcface_invocations": latest.arcface_invocations if latest else 0,
            "arcface_skipped": latest.arcface_skipped if latest else 0,
        }
        if latest
        else None,
    )


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics_route():
    """Returns detailed real-time performance metrics for the recognition pipeline.

    Includes FPS, per-stage latency, ArcFace invocation/skip counts,
    active track count, CPU%, and RAM usage.
    """
    m = global_engine.get_metrics()
    return MetricsResponse(
        fps=m.fps,
        detection_latency_ms=m.detection_latency_ms,
        arcface_latency_ms=m.arcface_latency_ms,
        total_latency_ms=m.total_latency_ms,
        arcface_invocations=m.arcface_invocations,
        arcface_skipped=m.arcface_skipped,
        active_tracks=m.active_tracks,
        cpu_percent=m.cpu_percent,
        ram_mb=m.ram_mb,
    )


@router.post("/debug/{enabled}", response_model=EngineActionResponse)
def toggle_debug_overlay_route(enabled: bool):
    """Toggles the diagnostic debug overlay on the MJPEG stream at runtime.

    When enabled, each tracked face shows: track ID, state, quality score,
    yaw/pitch, best/second similarity scores, and margin.

    WARNING: Disable in production — adds text rendering overhead per frame.
    """
    global_engine.set_debug_overlay(enabled)
    status = "enabled" if enabled else "disabled"
    return EngineActionResponse(
        status=status,
        message=f"Debug overlay {status}",
    )
