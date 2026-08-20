"""
Recognition Engine API Pydantic Schemas
"""

from typing import Optional, List, Any
from pydantic import BaseModel


class EngineStatusResponse(BaseModel):
    running: bool
    current_fps: float
    active_tracks_count: int
    latest_result: Optional[dict] = None


class EngineActionResponse(BaseModel):
    status: str
    message: str
