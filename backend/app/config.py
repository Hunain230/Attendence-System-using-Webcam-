"""
Attendance System — Configuration

All tunable values live here. Nothing is hardcoded in pipeline code.
Paths are project-relative so the project remains portable.
"""

import os
from pathlib import Path

# ── Project root (resolved at import time) ────────────
# This file lives at backend/app/config.py
# Project root is two levels up: backend/app/config.py → backend/ → root/
_THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = _THIS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# ── Paths (project-relative) ─────────────────────────
DATA_DIR = BACKEND_DIR / "data"
DB_PATH = DATA_DIR / "attendance.db"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
FAISS_ID_MAP_PATH = DATA_DIR / "faiss_id_map.npy"

# Ensure data directory exists at import time
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Database ──────────────────────────────────────────
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ── InsightFace ───────────────────────────────────────
INSIGHTFACE_MODEL: str = os.getenv("INSIGHTFACE_MODEL", "buffalo_s")
# Swap to "buffalo_l" to benchmark. Change via env var or here.

EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "512"))
# ArcFace output dimension. Configurable — never hardcoded elsewhere.

# ── Detection ─────────────────────────────────────────
DETECTION_INTERVAL: int = 3
# Run SCRFD every N frames. Tracking runs every frame.

DETECTION_SIZE: tuple[int, int] = (640, 640)
# SCRFD input resolution.

DETECTION_SCORE_THRESHOLD: float = 0.5
# Minimum detection confidence.

# ── Quality Gate ──────────────────────────────────────
MIN_FACE_SIZE: tuple[int, int] = (80, 80)
# Minimum face bounding box in pixels.

BLUR_THRESHOLD: float = 50.0
# Laplacian variance. Below this → blurry → reject.

MAX_YAW: float = 35.0
# Maximum horizontal head rotation in degrees.

MAX_PITCH: float = 35.0
# Maximum vertical head rotation in degrees.

MIN_BRIGHTNESS: float = 40.0
# Minimum mean pixel brightness of the face crop.

# ── Recognition ───────────────────────────────────────
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
# INITIAL EXPERIMENT ONLY.
# Must be tuned via FAR/FRR analysis on your own evaluation dataset.
# Do NOT ship to production without empirical validation.

TEMPORAL_FRAMES: int = 3
# Number of consecutive frames with the same identity required
# before confirming recognition.

FAISS_TOP_K: int = 5
# Number of nearest neighbors to retrieve from FAISS.

# ── Tracker (IoU-based) ──────────────────────────────
IOU_THRESHOLD: float = 0.3
# Minimum IoU overlap to consider a detection as the same track.

MAX_LOST_FRAMES: int = 15
# Number of frames a track can be lost before removal.

# ── Enrollment ────────────────────────────────────────
ENROLLMENT_POSES: list[str] = [
    "straight",
    "slight_left",
    "slight_right",
    "slight_up",
    "slight_down",
    "smile",
]
# Guided enrollment sequence. Each pose captures 1–2 quality-checked frames.

MIN_ENROLLMENT_SAMPLES: int = 5
# Minimum number of quality-passed samples to complete enrollment.

MAX_ENROLLMENT_SAMPLES: int = 12
# Maximum samples stored per employee.

# ── Camera ────────────────────────────────────────────
CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
# OpenCV VideoCapture device index. 0 = default webcam.

CAPTURE_WIDTH: int = 1280
CAPTURE_HEIGHT: int = 720
# Capture resolution from the webcam.

PROCESS_WIDTH: int = 720
# Frames are resized to this width for processing (aspect ratio preserved).

MJPEG_QUALITY: int = 80
# JPEG quality for MJPEG stream (0–100).

# ── Server ────────────────────────────────────────────
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8000"))
CORS_ORIGINS: list[str] = [
    "http://localhost:5173",   # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]
