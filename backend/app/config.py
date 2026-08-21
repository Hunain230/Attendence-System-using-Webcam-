"""
Attendance System — Configuration

All tunable values live here. Nothing is hardcoded in pipeline code.
Paths are project-relative so the project remains portable.

THRESHOLD GUIDE
───────────────
• SIMILARITY_THRESHOLD   : Minimum cosine similarity for a match to be accepted.
                           Run `python calibrate.py` to get a data-driven value.
• SIMILARITY_MARGIN      : Minimum gap between best and second-best employee.
                           Prevents confusing Hunain with Sobia when scores are close.
• TEMPORAL_FRAMES        : Number of recent recognitions that must agree before
                           an identity is confirmed. Higher = more stable, slower.
• UNKNOWN_RETRY_INTERVAL : Frames between recognition attempts for unknown tracks.
• RECOGNIZED_RECHECK_INTERVAL : Frames between re-verification for confirmed tracks.
                                Prevents ghost identity when a person leaves and
                                another walks into the same tracker slot.
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
# buffalo_s → SCRFD det_500m (~30 ms) + ArcFace w600k_mbf (~14 ms) ≈ 32 FPS on CPU
# buffalo_l → SCRFD det_10g  (~250 ms) + ArcFace w600k_r50 (~230 ms) ≈ 2 FPS on CPU
# Switch only after running calibrate.py and confirming genuine similarity improvement.

EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "512"))
# ArcFace output dimension. Configurable — never hardcoded elsewhere.

# ── Detection ─────────────────────────────────────────
DETECTION_INTERVAL: int = int(os.getenv("DETECTION_INTERVAL", "2"))
# Run SCRFD every N frames. Tracking runs every frame for max FPS.
# Lower = more accurate tracking, higher CPU. Range: 1–5.

DETECTION_SIZE: tuple[int, int] = (640, 640)
# SCRFD input resolution. Reduce to (320, 320) for lower latency at slight accuracy cost.

DETECTION_SCORE_THRESHOLD: float = float(os.getenv("DETECTION_SCORE_THRESHOLD", "0.5"))
# Minimum detection confidence. Raise to 0.6+ if you get phantom detections.

# ── Quality Gate ──────────────────────────────────────
MIN_FACE_SIZE: tuple[int, int] = (70, 70)
# Minimum face bounding box in pixels for live recognition.

ENROLLMENT_MIN_FACE_SIZE: tuple[int, int] = (90, 90)
# Minimum face size for enrollment (ensures sufficient detail without forcing user to stick face into camera).

BLUR_THRESHOLD: float = float(os.getenv("BLUR_THRESHOLD", "15.0"))
# Laplacian variance. Below this → severe motion blur/out-of-focus → reject.

ENROLLMENT_BLUR_THRESHOLD: float = float(os.getenv("ENROLLMENT_BLUR_THRESHOLD", "20.0"))
# Realistic blur threshold for clean stationary webcam enrollment frames.

MAX_YAW: float = float(os.getenv("MAX_YAW", "35.0"))
# Maximum horizontal head rotation for live recognition quality gate.

MAX_PITCH: float = float(os.getenv("MAX_PITCH", "35.0"))
# Maximum vertical head rotation for live recognition quality gate.

MIN_BRIGHTNESS: float = float(os.getenv("MIN_BRIGHTNESS", "35.0"))
# Minimum mean pixel brightness of the face crop (0-255).

ENROLLMENT_MIN_BRIGHTNESS: float = float(os.getenv("ENROLLMENT_MIN_BRIGHTNESS", "40.0"))
# Stricter brightness threshold for enrollment.

MIN_QUALITY_SCORE: float = float(os.getenv("MIN_QUALITY_SCORE", "0.35"))
# Minimum composite quality score (0–1) for recognition. Below this → skip ArcFace.

ENROLLMENT_MIN_QUALITY_SCORE: float = float(os.getenv("ENROLLMENT_MIN_QUALITY_SCORE", "0.45"))
# Minimum composite quality score required for a frame to be accepted as an enrollment sample.

# ── Enrollment Pose Ranges (degrees) ─────────────────
# These ranges are tighter than recognition-time limits.
# They ensure each enrolled angle is genuinely distinct.
POSE_STRAIGHT_MAX_YAW: float = 10.0    # |yaw| ≤ 10°
POSE_STRAIGHT_MAX_PITCH: float = 10.0  # |pitch| ≤ 10°

POSE_LEFT_YAW_MIN: float = 15.0        # yaw ∈ [-28°, -15°]
POSE_LEFT_YAW_MAX: float = 28.0
POSE_LEFT_MAX_PITCH: float = 15.0

POSE_RIGHT_YAW_MIN: float = 15.0       # yaw ∈ [+15°, +28°]
POSE_RIGHT_YAW_MAX: float = 28.0
POSE_RIGHT_MAX_PITCH: float = 15.0

POSE_UP_PITCH_MIN: float = 8.0         # pitch ∈ [-20°, -8°] (pitch negative = looking up)
POSE_UP_PITCH_MAX: float = 20.0
POSE_UP_MAX_YAW: float = 15.0

POSE_DOWN_PITCH_MIN: float = 8.0       # pitch ∈ [+8°, +20°] (pitch positive = looking down)
POSE_DOWN_PITCH_MAX: float = 20.0
POSE_DOWN_MAX_YAW: float = 15.0

POSE_SMILE_MAX_YAW: float = 10.0
POSE_SMILE_MAX_PITCH: float = 10.0

# ── Quality-Gated Enrollment Protocol ─────────────────
ENROLLMENT_HOLD_FRAMES: int = int(os.getenv("ENROLLMENT_HOLD_FRAMES", "4"))
# Number of consecutive frames the pose must remain valid before candidate collection begins.
# Prevents capturing transitional frames ("angle reached → immediately snapshot" bug).

ENROLLMENT_CANDIDATE_FRAMES: int = int(os.getenv("ENROLLMENT_CANDIDATE_FRAMES", "8"))
# Number of candidate frames collected per pose during the selection phase.
# The single frame with the highest quality score is selected and embedded.

ENROLLMENT_LANDMARK_STABILITY_THRESHOLD: float = float(
    os.getenv("ENROLLMENT_LANDMARK_STABILITY_THRESHOLD", "3.5")
)
# Maximum mean pixel displacement between landmark positions across consecutive frames.
# Above this → face is not sufficiently stable → don't collect candidate.

# ── Recognition ───────────────────────────────────────
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.52"))
# Calibrated cosine similarity threshold for ArcFace with 5-point landmark normalization.
# Run `python calibrate.py` to get the data-driven optimal value for your enrolled set.
# 0.52 is a conservative default — prefer false-reject over false-accept.

SIMILARITY_MARGIN: float = float(os.getenv("SIMILARITY_MARGIN", "0.12"))
# Minimum score gap between best and second-best employee.
# BUG FIX: was 0.05 with a `top_score < 0.65` bypass that disabled margin checking
# for high-confidence matches. Margin now applies unconditionally.
# Raise if Hunain/Sobia/Aurmish are confused. Lower if false-rejects increase.

TEMPORAL_FRAMES: int = int(os.getenv("TEMPORAL_FRAMES", "5"))
# Number of recent recognitions that must agree (via voting) before confirming identity.
# Was 2 — too easy to trigger with a single recognition flash.
# Range: 3–8. Higher = more stable identity, slightly slower first-confirm.

FAISS_TOP_K: int = int(os.getenv("FAISS_TOP_K", "10"))
# Number of nearest neighbors to retrieve from FAISS for employee candidate voting.

# ── Adaptive Recognition Frequency ────────────────────
# Controls how often ArcFace runs per track based on recognition state.
# These prevent running ArcFace every frame while also preventing the "permanent
# ghost identity" bug where a departed person's identity sticks to a new arrival.

RECOGNITION_INTERVAL_NEW: int = 1
# "new" state: recognize on every frame immediately.

RECOGNITION_INTERVAL_EVALUATING: int = 1
# "evaluating" state: recognize every frame until confirmed or declared unknown.

RECOGNITION_INTERVAL_UNCERTAIN: int = 10
# "uncertain" state: 3–7 failed recognitions. Try every 10 frames to save CPU.

UNKNOWN_RETRY_INTERVAL: int = int(os.getenv("UNKNOWN_RETRY_INTERVAL", "30"))
# "unknown" state: retry ArcFace every N frames. Fixes permanent-Unknown bug.
# At 30 FPS this means ~1 retry per second.

RECOGNIZED_RECHECK_INTERVAL: int = int(os.getenv("RECOGNIZED_RECHECK_INTERVAL", "90"))
# "confirmed" state: re-verify identity every N frames.
# Fixes ghost identity: if the person leaves and someone else enters the same
# tracker bounding box, the re-check will catch the different identity.
# At 30 FPS this is approximately every 3 seconds.

# Number of consecutive unmatched frames before downgrading states
UNMATCHED_TO_UNCERTAIN: int = 3   # confirmed/evaluating → uncertain after 3 misses
UNMATCHED_TO_UNKNOWN: int = 8     # uncertain → unknown after 8 total misses

# ── Tracker (IoU-based) ──────────────────────────────
IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.3"))
# Minimum IoU overlap to consider a detection as the same track.

MAX_LOST_FRAMES: int = int(os.getenv("MAX_LOST_FRAMES", "15"))
# Number of frames a track can be lost before removal.

# ── Liveness (anti-spoofing heuristics) ───────────────
LIVENESS_ENABLED: bool = os.getenv("LIVENESS_ENABLED", "true").lower() == "true"
# Enable lightweight liveness checks (optical flow + texture analysis).

LIVENESS_FLOW_THRESHOLD: float = float(os.getenv("LIVENESS_FLOW_THRESHOLD", "0.15"))
# Minimum optical flow magnitude across 10 frames.
# Too-still faces suggest a photo/screen. Calibrate against your lighting.

LIVENESS_LBP_THRESHOLD: float = float(os.getenv("LIVENESS_LBP_THRESHOLD", "0.80"))
# Maximum LBP histogram uniformity score.
# Screen captures have very uniform texture; real faces are less uniform.

LIVENESS_HISTORY_FRAMES: int = int(os.getenv("LIVENESS_HISTORY_FRAMES", "10"))
# Number of frames of history maintained per track for liveness analysis.

LIVENESS_MIN_FRAMES_FOR_CHECK: int = int(os.getenv("LIVENESS_MIN_FRAMES_FOR_CHECK", "6"))
# Minimum frames collected before a liveness verdict is made.

# ── Enrollment ────────────────────────────────────────
ENROLLMENT_POSES: list[str] = [
    "straight",
    "slight_left",
    "slight_right",
    "slight_up",
    "slight_down",
    "smile",
]
# Guided enrollment sequence. Each pose runs the two-phase quality-gated protocol.

MIN_ENROLLMENT_SAMPLES: int = 5
# Minimum number of quality-passed samples to complete enrollment.

MAX_ENROLLMENT_SAMPLES: int = 12
# Maximum samples stored per employee.

# ── Debug / Diagnostics ───────────────────────────────
DEBUG_OVERLAY: bool = os.getenv("DEBUG_OVERLAY", "false").lower() == "true"
# Draw verbose diagnostic info on the MJPEG stream. NEVER enable in production.
# Can also be toggled at runtime via POST /api/recognition/debug/{enabled}.

# ── Camera ────────────────────────────────────────────
CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
# OpenCV VideoCapture device index. 0 = default webcam.

CAMERA_URL: str = os.getenv("CAMERA_URL", "")
# Optional RTSP/HTTP URL for IP or phone cameras.
# If set, this overrides CAMERA_INDEX.
# Example: "rtsp://192.168.1.10:8080/video" or "http://192.168.1.10:8080/video"

CAPTURE_WIDTH: int = int(os.getenv("CAPTURE_WIDTH", "1280"))
CAPTURE_HEIGHT: int = int(os.getenv("CAPTURE_HEIGHT", "720"))
# Capture resolution from the webcam.

PROCESS_WIDTH: int = int(os.getenv("PROCESS_WIDTH", "720"))
# Frames are resized to this width for processing (aspect ratio preserved).

MJPEG_QUALITY: int = int(os.getenv("MJPEG_QUALITY", "80"))
# JPEG quality for MJPEG stream (0–100).

# ── Server ────────────────────────────────────────────
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8000"))
CORS_ORIGINS: list[str] = [
    "http://localhost:5173",   # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]
