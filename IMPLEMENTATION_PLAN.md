# 📋 Implementation Plan — Attendance System using Webcam

> **Version**: 3.0  
> **Target Hardware**: Surface Laptop Studio (i5-11300H / 16 GB / Iris Xe)  
> **Camera**: Built-in 1080p webcam  
> **Stack**: Python 3.12 + OpenCV + InsightFace + FAISS + SQLite + FastAPI + React  

---

## Table of Contents

- [1. System Architecture](#1-system-architecture)
- [2. Design Principles](#2-design-principles)
- [3. Configuration System](#3-configuration-system)
- [4. Phase 1 — Camera Validation](#4-phase-1--camera-validation)
- [5. Phase 2 — Face Detection (SCRFD)](#5-phase-2--face-detection-scrfd)
- [6. Phase 3 — Face Quality Gate](#6-phase-3--face-quality-gate)
- [7. Phase 4 — Face Embedding (ArcFace)](#7-phase-4--face-embedding-arcface)
- [8. Phase 5 — Database Layer (SQLite + SQLAlchemy)](#8-phase-5--database-layer-sqlite--sqlalchemy)
- [9. Phase 6 — Vector Index (FAISS)](#9-phase-6--vector-index-faiss)
- [10. Phase 7 — Employee Enrollment](#10-phase-7--employee-enrollment)
- [11. Phase 8 — Recognition Engine](#11-phase-8--recognition-engine)
- [12. Phase 9 — Attendance Service](#12-phase-9--attendance-service)
- [13. Phase 10 — FastAPI REST API](#13-phase-10--fastapi-rest-api)
- [14. Phase 11 — React Dashboard](#14-phase-11--react-dashboard)
- [15. Testing Strategy](#15-testing-strategy)
- [16. Performance Benchmarks](#16-performance-benchmarks)
- [17. Security Considerations](#17-security-considerations)
- [18. Future Enhancements](#18-future-enhancements)

---

## 1. System Architecture

### 1.1 Optimized Pipeline (with timing budgets)

```text
       Camera 720p / 30 FPS
               ↓
        ┌──────────────┐
        │    SCRFD     │  ~22 ms / frame
        │  Detection   │  → ~25–30 detection FPS
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ IoU Tracker  │  every frame (< 1 ms)
        └──────┬───────┘
               ↓
        ┌──────┴────────┐
        │               │
   Known face      New face
        │               │
        │          Quality Gate
        │               ↓
        │          Align face
        │               ↓
        │           ArcFace        ~8 ms
        │          (ONE embedding)
        │               ↓
        └───────┬───────┘
                ↓
          512-D embedding
                ↓
         ┌──────────────┐
         │    FAISS     │  ~5000 × 512 stored embeddings
         │  IndexFlatIP │  vector search (< 1 ms)
         └──────┬───────┘
                ↓
          Nearest person
                ↓
        Similarity threshold
                ↓
        Temporal confirmation
                ↓
         Attendance Service
                ↓
             SQLite
                ↓
     ┌──────────┴──────────┐
     ↓                     ↓
  FastAPI               MJPEG
     ↓                     ↓
  React UI ←───────────────┘
```

### 1.2 Critical Architecture Rule

> **ArcFace produces ONE 512-D embedding for the detected face.**  
> We do NOT run ArcFace against 5000 people.  
> FAISS searches ~5000 stored embeddings via vector similarity in < 1 ms.  
> Recognition runs ONLY for new/uncertain tracks, not every frame.

### 1.3 Recognition Engine Thread Model

```text
┌──────────────────────────────────────────────────────────────┐
│              Recognition Engine (Thread)                      │
│                                                              │
│  Camera Capture ─────────────── every frame                  │
│       ↓                                                      │
│  SCRFD Detection ────────────── ~22 ms (periodic, every N)   │
│       ↓                                                      │
│  IoU Tracker ────────────────── every frame (< 1 ms)         │
│       ↓                                                      │
│  ┌─ Known face ──→ track only (skip recognition)             │
│  └─ New face ────→ Quality Gate                              │
│                         ↓                                    │
│                    Face Alignment                            │
│                         ↓                                    │
│                    ArcFace ──── ~8 ms → ONE 512-D embedding  │
│                         ↓                                    │
│                    FAISS Search ── < 1 ms (5000 embeddings)  │
│                         ↓                                    │
│                    Temporal Confirmation                      │
│                         ↓                                    │
│                    Attendance Service                         │
│       ↓                                                      │
│  MJPEG Frame Buffer (annotated frame for UI)                 │
└──────────────────────────────────────────────────────────────┘
         ↕ control                      ↕ read
┌──────────────────────────────────────────────────────────────┐
│              FastAPI (HTTP Thread)                            │
│  /api/recognition/start|stop|status                          │
│  /api/employees, /api/attendance                             │
│  /api/stream (MJPEG endpoint)                                │
└──────────────────────────────────────────────────────────────┘
```

### 1.4 Performance Targets

| Metric | Target |
|:-------|:-------|
| Camera capture | 720p @ 30 FPS |
| SCRFD detection | ~22 ms/frame → ~25–30 FPS |
| Face tracking (IoU) | Every frame, < 1 ms |
| ArcFace embedding | ~8 ms, **only for new/uncertain faces** |
| FAISS search (5000 embeddings) | < 1 ms |
| End-to-end recognition | Person recognized within ~0.2–1 s |

### 1.5 Naive vs. Optimized Comparison

```text
Naive approach (FaceAnalysis.get() every frame):
  SCRFD + ArcFace on every frame → ~30 ms → ~5.8 FPS  ❌

Optimized approach:
  SCRFD every frame              → ~22 ms → ~25–30 FPS ✅
  Tracking every frame           → < 1 ms (IoU only)
  ArcFace ONLY for new faces     → ~8 ms (amortized: rare)
  FAISS search (5000 embeddings) → < 1 ms
```

---

## 2. Design Principles

### 2.1 Separation of Concerns

| Principle | Implementation |
|:----------|:--------------|
| **SCRFD ≠ ArcFace** | Detection and recognition are separate operations. SCRFD detects faces. ArcFace extracts embeddings. Never use `FaceAnalysis.get()` which runs both on every frame. |
| **Engine ≠ API** | The Recognition Engine runs in a background thread. FastAPI controls it but never runs CV inside HTTP handlers. |
| **Tracking ≠ Recognition** | IoU tracker runs every frame (< 1 ms). ArcFace runs only when tracker encounters a new/uncertain face. |
| **Embedding ≠ Matching** | ArcFace produces ONE embedding. FAISS handles the search against stored embeddings. |

### 2.2 Configuration-First

All tunable values live in `config.py`. Nothing is hardcoded in pipeline code:
- Model name (`buffalo_s` / `buffalo_l`)
- Embedding dimension (512)
- All quality thresholds
- Detection interval
- Camera parameters
- Similarity threshold
- All file paths (project-relative)

### 2.3 Data Flow

```text
ENROLLMENT:
  Employee → Camera → Quality Gate → ArcFace → L2-normalize
  → FAISS.add() + SQLite.insert(faiss_id ↔ employee_id)

RECOGNITION:
  Camera → SCRFD → Tracker → [new face?] → Quality Gate → ArcFace
  → L2-normalize → FAISS.search() → employee_id → Temporal Confirm
  → Attendance Service → SQLite

QUERY:
  React UI → FastAPI → SQLite → JSON response
```

---

## 3. Configuration System

### 3.1 File: `backend/app/config.py`

All settings are centralized here. Key categories:

```python
# ── Paths (project-relative via pathlib) ─────────────
DATA_DIR            # backend/data/
DB_PATH             # backend/data/attendance.db
FAISS_INDEX_PATH    # backend/data/faiss.index
FAISS_ID_MAP_PATH   # backend/data/faiss_id_map.npy

# ── InsightFace ──────────────────────────────────────
INSIGHTFACE_MODEL   = "buffalo_s"    # start small, benchmark buffalo_l
EMBEDDING_DIM       = 512            # ArcFace output, NEVER hardcoded elsewhere

# ── Detection ────────────────────────────────────────
DETECTION_INTERVAL  = 3              # run SCRFD every N frames
DETECTION_SIZE      = (640, 640)     # SCRFD input resolution
DETECTION_SCORE_THRESHOLD = 0.5      # minimum detection confidence

# ── Quality Gate ─────────────────────────────────────
MIN_FACE_SIZE       = (80, 80)       # minimum bbox in pixels
BLUR_THRESHOLD      = 50.0           # Laplacian variance
MAX_YAW / MAX_PITCH = 35.0           # degrees
MIN_BRIGHTNESS      = 40.0           # mean pixel value

# ── Recognition ──────────────────────────────────────
SIMILARITY_THRESHOLD = 0.45          # INITIAL EXPERIMENT — tune via FAR/FRR
TEMPORAL_FRAMES      = 3             # consecutive confirmations
FAISS_TOP_K          = 5             # nearest neighbors

# ── Tracker ──────────────────────────────────────────
IOU_THRESHOLD        = 0.3           # match threshold
MAX_LOST_FRAMES      = 15            # frames before track removal

# ── Camera ───────────────────────────────────────────
CAMERA_INDEX         = 0
CAPTURE_WIDTH        = 1280
CAPTURE_HEIGHT       = 720
PROCESS_WIDTH        = 720           # resize for processing

# ── Server ───────────────────────────────────────────
HOST / PORT / CORS_ORIGINS
```

### 3.2 Environment Variable Overrides

Key settings can be overridden via environment variables:

```powershell
$env:INSIGHTFACE_MODEL = "buffalo_l"
$env:SIMILARITY_THRESHOLD = "0.50"
$env:CAMERA_INDEX = "1"
$env:EMBEDDING_DIM = "512"
```

---

## 4. Phase 1 — Camera Validation

### 4.1 Goal

Confirm the Surface webcam works reliably before introducing any AI.

### 4.2 File: `tests/test_camera.py`

### 4.3 What to Implement

```python
# Pseudocode for camera validation
def test_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    dropped = 0
    start = time.time()

    for _ in range(1800):  # 60 seconds at 30 FPS
        ret, frame = cap.read()
        if not ret:
            dropped += 1
            continue
        frame_count += 1

    elapsed = time.time() - start
    actual_fps = frame_count / elapsed

    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Frames: {frame_count}, Dropped: {dropped}")
    print(f"FPS: {actual_fps:.1f}")
    print(f"Elapsed: {elapsed:.1f}s")

    assert actual_fps >= 25, f"FPS too low: {actual_fps}"
    assert dropped == 0, f"Dropped frames: {dropped}"
```

### 4.4 Metrics to Capture

| Metric | Target |
|:-------|:-------|
| Resolution | 1280 × 720 (or actual) |
| FPS | ≥ 25 stable |
| Dropped frames | 0 over 60 seconds |
| Latency | < 50 ms frame-to-display |
| CPU usage | < 10% for capture only |
| RAM usage | Baseline measurement |

### 4.5 Exit Criteria

- Stable 720p @ 30 FPS for 60 seconds
- Zero dropped frames
- Consistent frame timing (no spikes > 100 ms)

### 4.6 Troubleshooting

| Issue | Fix |
|:------|:----|
| Camera not found | Try `CAMERA_INDEX = 1` or `2` |
| Low FPS | Reduce to 640×480, check background apps |
| Inconsistent FPS | Disable power saving mode, ensure plugged in |
| Black frames | Check camera privacy toggle on Surface |

---

## 5. Phase 2 — Face Detection (SCRFD)

### 5.1 Goal

Load InsightFace SCRFD detection model separately from ArcFace. Draw bounding boxes on live video at ≥ 25 FPS.

### 5.2 File: `backend/app/recognition/detector.py`

### 5.3 Critical Implementation Detail

> **Do NOT use `FaceAnalysis.get()` for detection.**  
> `FaceAnalysis.get()` runs BOTH detection AND recognition on every frame → ~5.8 FPS.  
> Instead, use SCRFD detection only → ~25–30 FPS.

### 5.4 Implementation

```python
class FaceDetector:
    """SCRFD face detection wrapper.
    
    Runs detection only — no recognition.
    Uses InsightFace's model zoo to load SCRFD separately.
    """
    
    def __init__(self):
        # Load the full model pack for model access
        self.app = insightface.app.FaceAnalysis(
            name=config.INSIGHTFACE_MODEL,
            allowed_modules=['detection'],  # KEY: detection only!
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(
            ctx_id=0, 
            det_size=config.DETECTION_SIZE
        )
    
    def detect(self, frame: np.ndarray) -> list[Face]:
        """Run SCRFD detection on a frame.
        
        Returns list of Face objects with:
          - bbox: [x1, y1, x2, y2]
          - kps: 5 facial landmarks (for alignment)
          - det_score: detection confidence
        
        Does NOT run ArcFace recognition.
        """
        faces = self.app.get(frame)
        # Filter by detection score
        return [
            f for f in faces 
            if f.det_score >= config.DETECTION_SCORE_THRESHOLD
        ]
```

### 5.5 Output per Face

```python
@dataclass
class DetectedFace:
    bbox: np.ndarray          # [x1, y1, x2, y2] float32
    landmarks: np.ndarray     # (5, 2) — eyes, nose, mouth corners
    det_score: float          # 0.0–1.0
    face_crop: np.ndarray     # cropped face region from frame
```

### 5.6 Visualization

```python
def annotate_frame(frame, faces):
    """Draw bounding boxes + confidence on frame."""
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        score = face.det_score
        
        # Green box with confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame
```

### 5.7 Benchmark: `buffalo_s` vs `buffalo_l`

Run both and record:

| Model | Detection Time | FPS | CPU % | RAM |
|:------|:--------------|:----|:------|:----|
| `buffalo_s` | ~22 ms (expected) | ~25–30 | TBD | TBD |
| `buffalo_l` | TBD | TBD | TBD | TBD |

### 5.8 Exit Criteria

- SCRFD bounding boxes visible on live feed
- **≥ 25 FPS** with `buffalo_s` on 720p frames (SCRFD detection only)
- Detection confidence displayed per face
- Face landmarks (5-point) extracted correctly

---

## 6. Phase 3 — Face Quality Gate

### 6.1 Goal

Reject faces that will produce unreliable embeddings. This runs BEFORE ArcFace, saving ~8 ms per rejected face.

### 6.2 File: `backend/app/recognition/quality.py`

### 6.3 Quality Checks (in order)

```python
class QualityGate:
    """Reject faces unsuitable for embedding.
    
    Checks run in order of computational cost (cheapest first).
    If any check fails, the face is rejected immediately.
    """
    
    def check(self, face_crop: np.ndarray, bbox: np.ndarray, 
              landmarks: np.ndarray) -> tuple[bool, str, float]:
        """Returns (passed, reason, quality_score)."""
        
        # 1. Size check (< 0.01 ms)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w < config.MIN_FACE_SIZE[0] or h < config.MIN_FACE_SIZE[1]:
            return False, "face_too_small", 0.0
        
        # 2. Brightness check (< 0.1 ms)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < config.MIN_BRIGHTNESS:
            return False, "too_dark", brightness
        
        # 3. Blur check via Laplacian variance (< 0.5 ms)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < config.BLUR_THRESHOLD:
            return False, "too_blurry", laplacian_var
        
        # 4. Head pose estimation from landmarks (< 0.1 ms)
        yaw, pitch = self._estimate_pose(landmarks)
        if abs(yaw) > config.MAX_YAW:
            return False, "excessive_yaw", abs(yaw)
        if abs(pitch) > config.MAX_PITCH:
            return False, "excessive_pitch", abs(pitch)
        
        # All checks passed
        quality_score = self._compute_score(
            brightness, laplacian_var, yaw, pitch, w, h
        )
        return True, "passed", quality_score
```

### 6.4 Pose Estimation from 5-Point Landmarks

```text
Landmarks:
  0: left eye
  1: right eye
  2: nose tip
  3: left mouth corner
  4: right mouth corner

Yaw estimation:
  Compare nose-to-left-eye distance vs nose-to-right-eye distance.
  If significantly different → head is turned.

Pitch estimation:
  Compare vertical position of nose relative to eye midpoint.
  If nose is too high/low → head is tilted up/down.
```

```python
def _estimate_pose(self, landmarks: np.ndarray) -> tuple[float, float]:
    """Estimate yaw and pitch from 5-point landmarks.
    
    Returns approximate angles in degrees.
    Not perfect — but sufficient to reject extreme poses.
    """
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    
    eye_center = (left_eye + right_eye) / 2
    eye_width = np.linalg.norm(right_eye - left_eye)
    
    # Yaw: horizontal offset of nose from eye center
    nose_offset_x = (nose[0] - eye_center[0]) / (eye_width + 1e-6)
    yaw = np.degrees(np.arctan(nose_offset_x))
    
    # Pitch: vertical offset of nose below eye center
    nose_offset_y = (nose[1] - eye_center[1]) / (eye_width + 1e-6)
    pitch = np.degrees(np.arctan(nose_offset_y)) - 15  # nose is normally below eyes
    
    return yaw, pitch
```

### 6.5 Quality Score Calculation

```python
def _compute_score(self, brightness, sharpness, yaw, pitch, w, h):
    """Composite quality score 0.0–1.0 for enrollment ranking."""
    size_score = min(1.0, (w * h) / (200 * 200))
    blur_score = min(1.0, sharpness / 200.0)
    pose_score = 1.0 - (abs(yaw) + abs(pitch)) / 70.0
    bright_score = min(1.0, brightness / 150.0)
    
    return (size_score * 0.2 + blur_score * 0.3 + 
            pose_score * 0.3 + bright_score * 0.2)
```

### 6.6 Exit Criteria

- Blurry faces rejected (Laplacian variance < 50)
- Tiny faces rejected (< 80×80 px)
- Side-angle faces rejected (yaw/pitch > 35°)
- Dark faces rejected (brightness < 40)
- Quality gate adds < 1 ms total overhead per face

---

## 7. Phase 4 — Face Embedding (ArcFace)

### 7.1 Goal

Generate L2-normalized 512-D embeddings from quality-passed, aligned face crops.

### 7.2 File: `backend/app/recognition/embedder.py`

### 7.3 Critical: Separate from Detection

```text
WRONG:
  FaceAnalysis.get() → runs SCRFD + ArcFace every frame → 5.8 FPS

RIGHT:
  SCRFD.detect() → faces (every frame, ~22 ms)
  ArcFace.get_feat() → embedding (new faces only, ~8 ms)
```

### 7.4 Implementation

```python
class FaceEmbedder:
    """ArcFace embedding extraction.
    
    Produces ONE L2-normalized embedding per face.
    Uses InsightFace's recognition model loaded separately.
    """
    
    def __init__(self):
        # Load recognition model from the model pack
        model_pack = insightface.app.FaceAnalysis(
            name=config.INSIGHTFACE_MODEL,
            allowed_modules=['recognition'],  # recognition only
            providers=['CPUExecutionProvider']
        )
        model_pack.prepare(ctx_id=0)
        self.rec_model = model_pack.models.get('recognition', None)
        # Alternative: load ArcFace directly from model zoo
    
    def embed(self, frame: np.ndarray, face) -> np.ndarray:
        """Generate embedding for a single face.
        
        Args:
            frame: original frame (not cropped)
            face: InsightFace Face object with landmarks
            
        Returns:
            L2-normalized numpy array of shape (EMBEDDING_DIM,)
        """
        # InsightFace handles alignment internally using landmarks
        embedding = self.rec_model.get(frame, face)
        
        # L2-normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        assert embedding.shape == (config.EMBEDDING_DIM,), \
            f"Expected ({config.EMBEDDING_DIM},), got {embedding.shape}"
        
        return embedding
```

### 7.5 Why L2-Normalize?

```text
Inner product of L2-normalized vectors = cosine similarity

  cos(a, b) = (a · b) / (‖a‖ · ‖b‖)

If ‖a‖ = 1 and ‖b‖ = 1:
  cos(a, b) = a · b

So FAISS IndexFlatIP on normalized vectors gives cosine similarity directly.
```

### 7.6 Verification

| Test | Expected |
|:-----|:---------|
| Same person, same session | cosine similarity ≥ 0.5 |
| Same person, different conditions | cosine similarity ≥ 0.3 |
| Different people | cosine similarity ≤ 0.3 |
| Embedding shape | `(512,)` |
| Embedding norm | 1.0 (± 1e-6) |

### 7.7 Exit Criteria

- ArcFace produces (512,) embeddings in ~8 ms
- Embeddings are L2-normalized (norm = 1.0)
- Same person similarity ≥ 0.5
- Different people similarity ≤ 0.3
- `EMBEDDING_DIM` is read from config, never hardcoded

---

## 8. Phase 5 — Database Layer (SQLite + SQLAlchemy)

### 8.1 Goal

Create the SQLite database schema, ORM models, and CRUD helpers.

### 8.2 Files

| File | Purpose |
|:-----|:--------|
| `backend/app/database/models.py` | SQLAlchemy ORM models |
| `backend/app/database/database.py` | Engine + session factory |
| `backend/app/database/repository.py` | CRUD helpers |

### 8.3 Database Engine

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite needs this
    echo=False
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

def get_db():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
```

### 8.4 ORM Models

#### `employees` table

```python
class Employee(Base):
    __tablename__ = "employees"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    department = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)
    
    # Relationships
    embeddings = relationship("FaceEmbedding", back_populates="employee")
    attendance_records = relationship("Attendance", back_populates="employee")
```

#### `face_embeddings` table

```python
class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    faiss_id = Column(Integer, nullable=False)       # maps to FAISS vector index
    quality_score = Column(Float, nullable=False)     # from quality gate
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    employee = relationship("Employee", back_populates="embeddings")
```

#### `attendance` table

```python
class Attendance(Base):
    __tablename__ = "attendance"
    
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    date = Column(Date, nullable=False)
    check_in = Column(DateTime, nullable=False)       # first recognition
    check_out = Column(DateTime, nullable=True)        # EXPLICIT only
    confidence = Column(Float, nullable=False)         # mean recognition confidence
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('employee_id', 'date', name='uq_employee_date'),
    )
    
    # Relationships
    employee = relationship("Employee", back_populates="attendance_records")
```

### 8.5 Repository (CRUD Helpers)

```python
class EmployeeRepository:
    def create(self, db, code, name, department) -> Employee
    def get_by_id(self, db, employee_id) -> Employee | None
    def get_by_code(self, db, code) -> Employee | None
    def list_all(self, db, active_only=True) -> list[Employee]
    def deactivate(self, db, employee_id) -> bool

class EmbeddingRepository:
    def add(self, db, employee_id, faiss_id, quality_score) -> FaceEmbedding
    def get_by_employee(self, db, employee_id) -> list[FaceEmbedding]
    def get_by_faiss_id(self, db, faiss_id) -> FaceEmbedding | None
    def delete_by_employee(self, db, employee_id) -> int

class AttendanceRepository:
    def check_in(self, db, employee_id, confidence) -> Attendance
    def check_out(self, db, employee_id) -> Attendance | None
    def get_today(self, db) -> list[Attendance]
    def get_by_date(self, db, date) -> list[Attendance]
    def has_checked_in_today(self, db, employee_id) -> bool
```

### 8.6 FAISS ID ↔ Employee ID Mapping

```text
FAISS returns:  FAISS ID = 438
SQLite maps:    438 → employee_id = 129
SQLite looks up: employee_id 129 → "Hunain Shahid"

Mapping chain:
  FAISS vector index → faiss_id → face_embeddings.employee_id → employees
```

### 8.7 Exit Criteria

- All three tables created in SQLite
- CRUD operations work for all tables
- Unique constraint on (employee_id, date) in attendance
- Session management works with FastAPI dependency injection
- `init_db()` creates tables idempotently

---

## 9. Phase 6 — Vector Index (FAISS)

### 9.1 Goal

Build and persist a FAISS `IndexFlatIP` for cosine similarity search over stored embeddings.

### 9.2 File: `backend/app/recognition/matcher.py`

### 9.3 Why IndexFlatIP?

```text
IndexFlatIP = Inner Product search (exact, brute-force)

On L2-normalized vectors:
  Inner product = cosine similarity

5000 embeddings × 512 dimensions = ~10 MB
Search time: < 1 ms (brute-force is fine at this scale)

No need for HNSW or IVF until > 100K embeddings.
```

### 9.4 Implementation

```python
class FaceMatcher:
    """FAISS-based face matching.
    
    Manages the vector index and maps FAISS IDs to employee IDs.
    All embeddings must be L2-normalized before add/search.
    """
    
    def __init__(self):
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        self.id_map: list[int] = []  # faiss_position → employee_id
        self._load_if_exists()
    
    def add(self, embedding: np.ndarray, employee_id: int) -> int:
        """Add an embedding to the index.
        
        Returns the FAISS ID (position in the index).
        """
        assert embedding.shape == (config.EMBEDDING_DIM,)
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5, "Must be L2-normalized"
        
        faiss_id = self.index.ntotal
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        self.id_map.append(employee_id)
        
        self._save()
        return faiss_id
    
    def search(self, embedding: np.ndarray, top_k: int = None
              ) -> list[tuple[int, float]]:
        """Search for nearest matches.
        
        Returns list of (employee_id, similarity_score) tuples,
        sorted by descending similarity.
        """
        if self.index.ntotal == 0:
            return []
        
        top_k = top_k or config.FAISS_TOP_K
        top_k = min(top_k, self.index.ntotal)
        
        query = embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score >= config.SIMILARITY_THRESHOLD:
                results.append((self.id_map[idx], float(score)))
        
        return results
    
    def _save(self):
        """Persist index and ID map to disk."""
        faiss.write_index(self.index, str(config.FAISS_INDEX_PATH))
        np.save(str(config.FAISS_ID_MAP_PATH), np.array(self.id_map))
    
    def _load_if_exists(self):
        """Load index and ID map from disk if they exist."""
        if config.FAISS_INDEX_PATH.exists() and config.FAISS_ID_MAP_PATH.exists():
            self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
            self.id_map = np.load(str(config.FAISS_ID_MAP_PATH)).tolist()
    
    def remove_employee(self, employee_id: int):
        """Remove all embeddings for an employee.
        
        FAISS IndexFlatIP doesn't support deletion, so we rebuild.
        """
        # Rebuild index without this employee's embeddings
        # This is fine for < 50K embeddings
        ...
```

### 9.5 Vector Search Flow (5000 employees)

```text
5000 people
    ↓
~5000 × 512 stored embeddings (in FAISS IndexFlatIP)
    ↓
New face detected → ArcFace → ONE 512-D query embedding
    ↓
FAISS inner product search (< 1 ms)
    ↓
Top-K nearest persons
    ↓
Similarity threshold (0.45 initial)
    ↓
Employee ID (or "unknown")
```

### 9.6 Index Size

```text
5000 embeddings × 512 dimensions × 4 bytes (float32) = 10.24 MB
10,000 embeddings = 20.48 MB
50,000 embeddings = 102.4 MB

All trivially small for RAM and disk.
```

### 9.7 Exit Criteria

- FAISS index persists to disk and reloads correctly
- Add and search operations work
- Top-1 search over 100 embeddings returns correct employee ≥ 98%
- Search time < 1 ms for 5000 embeddings
- ID map correctly maps FAISS positions to employee IDs

---

## 10. Phase 7 — Employee Enrollment

### 10.1 Goal

Capture face samples with **guided, controlled variation** per employee. Not just 5–10 consecutive frames.

### 10.2 Files

| File | Purpose |
|:-----|:--------|
| `backend/app/api/employees.py` | Enrollment API endpoint |
| `backend/app/recognition/embedder.py` | Embedding generation |
| `backend/app/recognition/quality.py` | Sample quality validation |

### 10.3 Enrollment Protocol

```text
Employee enters: employee_code + name + department
                    ↓
Guided capture sequence:
  1. "Look straight at camera"       → capture 2 quality frames
  2. "Slight turn left (~15°)"       → capture 1 quality frame
  3. "Slight turn right (~15°)"      → capture 1 quality frame
  4. "Slight tilt up (~10°)"         → capture 1 quality frame
  5. "Slight tilt down (~10°)"       → capture 1 quality frame
  6. "Smile / different expression"  → capture 1 quality frame
                    ↓
Quality filter (reject samples below threshold)
                    ↓
Minimum 5 quality-passed samples required
                    ↓
ArcFace → 5–8 embeddings per employee
                    ↓
L2-normalize each embedding
                    ↓
FAISS.add() for each embedding
                    ↓
SQLite: insert face_embeddings rows (faiss_id ↔ employee_id)
```

### 10.4 Why Multiple Samples?

```text
Single photo enrollment:
  Employee with glasses → no match without glasses ❌
  Employee smiling → low match with neutral face ❌

Multi-variation enrollment:
  5–8 samples covering angles + expressions + accessories
  → robust matching under real-world conditions ✅
```

### 10.5 Enrollment State Machine

```python
class EnrollmentSession:
    POSES = [
        ("straight", "Look straight at the camera", 2),
        ("slight_left", "Turn head slightly to the left", 1),
        ("slight_right", "Turn head slightly to the right", 1),
        ("slight_up", "Tilt head slightly upward", 1),
        ("slight_down", "Tilt head slightly downward", 1),
        ("smile", "Smile naturally", 1),
    ]
    
    def __init__(self, employee_id: int):
        self.employee_id = employee_id
        self.current_pose_idx = 0
        self.captured_embeddings: list[np.ndarray] = []
        self.quality_scores: list[float] = []
        self.status = "in_progress"
    
    @property
    def current_pose(self):
        return self.POSES[self.current_pose_idx]
    
    @property
    def is_complete(self):
        return len(self.captured_embeddings) >= config.MIN_ENROLLMENT_SAMPLES
```

### 10.6 Sample Rejection During Enrollment

During enrollment, use **stricter** quality thresholds than real-time recognition:

| Check | Runtime Threshold | Enrollment Threshold |
|:------|:------------------|:---------------------|
| Min face size | 80 × 80 | 120 × 120 |
| Blur | 50.0 | 80.0 |
| Brightness | 40 | 60 |

Better enrollment samples → better recognition accuracy.

### 10.7 Exit Criteria

- Guided enrollment captures 5–8 quality samples per employee
- Each sample passes quality gate (stricter thresholds for enrollment)
- All embeddings stored in FAISS + metadata in SQLite
- Employee can be recognized after enrollment
- Poor samples are rejected and pose is re-requested

---

## 11. Phase 8 — Recognition Engine

### 11.1 Goal

Build the background thread that owns the entire camera → detection → tracking → recognition → attendance pipeline.

### 11.2 Files

| File | Purpose |
|:-----|:--------|
| `backend/app/engine/recognition_engine.py` | Main engine loop |
| `backend/app/recognition/tracker.py` | IoU-based face tracker |

### 11.3 Engine Thread Architecture

```python
class RecognitionEngine:
    """Background thread running the full recognition pipeline.
    
    FastAPI controls this via start()/stop()/status().
    Never run CV operations inside HTTP request handlers.
    """
    
    def __init__(self):
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.matcher = FaceMatcher()
        self.tracker = FaceTracker()
        self.quality_gate = QualityGate()
        self.attendance_service = AttendanceService()
        
        self._running = False
        self._thread: Thread | None = None
        self._current_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._stats = EngineStats()
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def get_frame(self) -> np.ndarray | None:
        """Get the latest annotated frame for MJPEG stream."""
        with self._frame_lock:
            return self._current_frame.copy() if self._current_frame is not None else None
    
    def _loop(self):
        """Main engine loop."""
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
        
        frame_count = 0
        
        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # ── Step 1: SCRFD Detection (~22 ms) ─────────────
            faces = self.detector.detect(frame)
            
            # ── Step 2: IoU Tracking (< 1 ms) ────────────────
            tracks = self.tracker.update(faces)
            
            # ── Step 3: Process each track ────────────────────
            for track in tracks:
                if track.is_recognized:
                    # Known face → just track, skip recognition
                    continue
                
                # New/uncertain face → full recognition pipeline
                
                # Step 3a: Quality Gate (< 1 ms)
                passed, reason, score = self.quality_gate.check(
                    track.face_crop, track.bbox, track.landmarks
                )
                if not passed:
                    continue
                
                # Step 3b: ArcFace Embedding (~8 ms)
                embedding = self.embedder.embed(frame, track.face)
                
                # Step 3c: FAISS Search (< 1 ms)
                matches = self.matcher.search(embedding)
                if not matches:
                    track.mark_unknown()
                    continue
                
                employee_id, confidence = matches[0]
                
                # Step 3d: Temporal Confirmation
                track.add_recognition(employee_id, confidence)
                
                if track.is_confirmed:
                    # Step 3e: Attendance
                    self.attendance_service.mark_if_needed(
                        employee_id, confidence
                    )
            
            # ── Step 4: Annotate frame for MJPEG ─────────────
            annotated = self._annotate(frame, tracks)
            with self._frame_lock:
                self._current_frame = annotated
        
        cap.release()
```

### 11.4 IoU Tracker

```python
class FaceTracker:
    """Simple IoU-based face tracker.
    
    Matches detected faces to existing tracks via bounding box overlap.
    Only new/unmatched detections trigger recognition.
    """
    
    @dataclass
    class Track:
        track_id: int
        bbox: np.ndarray
        landmarks: np.ndarray
        face: object                    # InsightFace Face object
        face_crop: np.ndarray
        
        # Recognition state
        recognition_history: list       # (employee_id, confidence) per frame
        confirmed_identity: int | None  # employee_id once confirmed
        frames_since_recognition: int
        lost_frames: int
        
        @property
        def is_recognized(self) -> bool:
            return self.confirmed_identity is not None
        
        @property
        def is_confirmed(self) -> bool:
            """Check if temporal confirmation threshold is met."""
            if len(self.recognition_history) < config.TEMPORAL_FRAMES:
                return False
            
            recent = self.recognition_history[-config.TEMPORAL_FRAMES:]
            ids = [r[0] for r in recent]
            
            # All recent recognitions must agree
            return len(set(ids)) == 1 and ids[0] is not None
        
        def add_recognition(self, employee_id: int, confidence: float):
            self.recognition_history.append((employee_id, confidence))
            if self.is_confirmed:
                self.confirmed_identity = employee_id
    
    def update(self, detections: list) -> list[Track]:
        """Update tracks with new detections.
        
        1. Compute IoU between all current tracks and new detections
        2. Match tracks to detections (Hungarian algorithm or greedy)
        3. Update matched tracks
        4. Create new tracks for unmatched detections
        5. Mark unmatched tracks as lost
        6. Remove tracks lost for > MAX_LOST_FRAMES
        """
        ...
    
    @staticmethod
    def _compute_iou(box1, box2) -> float:
        """Compute Intersection over Union between two bboxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
```

### 11.5 Temporal Confirmation

```text
Frame 1 → Hunain, confidence 0.81
Frame 2 → Hunain, confidence 0.84
Frame 3 → Hunain, confidence 0.86  ← 3 consecutive = CONFIRMED

Result: Hunain Shahid → attendance check-in
```

```text
Frame 1 → Hunain, confidence 0.51
Frame 2 → Ali, confidence 0.46       ← disagreement
Frame 3 → Hunain, confidence 0.53

Result: NOT confirmed (identities disagree). Continue tracking.
```

### 11.6 Pipeline Timing Summary

| Stage | Frequency | Cost | Total per Frame |
|:------|:----------|:-----|:----------------|
| Capture | Every frame | ~1 ms | ~1 ms |
| SCRFD Detection | Every frame | ~22 ms | ~22 ms |
| IoU Tracking | Every frame | < 1 ms | < 1 ms |
| Quality Gate | New faces only | < 1 ms | ~0 ms (amortized) |
| ArcFace | New faces only | ~8 ms | ~0 ms (amortized) |
| FAISS Search | New faces only | < 1 ms | ~0 ms (amortized) |
| **Total** | | | **~24 ms → ~25–30 FPS** |

### 11.7 Exit Criteria

- Engine runs as background thread, controllable via start/stop
- SCRFD detection at ≥ 25 FPS
- ArcFace runs only for new/uncertain tracks
- Temporal confirmation requires 3 consecutive agreeing frames
- Known faces are tracked without re-running recognition
- MJPEG frame buffer updated every frame
- Graceful shutdown on stop()

---

## 12. Phase 9 — Attendance Service

### 12.1 Goal

Mark attendance with first-recognition check-in and explicit-only checkout.

### 12.2 File: `backend/app/attendance/service.py`

### 12.3 Attendance Logic

```text
Temporal confirmation → Employee ID + Confidence
                ↓
         Has attendance row for today?
           ↙                    ↘
         YES                    NO
          ↓                      ↓
       Ignore               INSERT check_in
  (no duplicate rows)       (first recognition of the day)

Check-out:
┌──────────────────────────────────┐
│ EXPLICIT ONLY                    │
│ • API: POST /attendance/checkout │
│ • UI: "Check Out" button         │
│ • Configured departure event     │
└──────────────────────────────────┘
```

> **CRITICAL**: No automatic re-detection checkout. Someone returning from lunch must NOT accidentally become "checked out."

### 12.4 Implementation

```python
class AttendanceService:
    """Marks attendance based on confirmed recognition.
    
    Rules:
    1. First recognition today → check-in
    2. Subsequent recognitions → ignore (no duplicates)
    3. Check-out → explicit only (API/UI)
    """
    
    def mark_if_needed(self, employee_id: int, confidence: float) -> bool:
        """Attempt to mark check-in for the employee.
        
        Returns True if a new attendance record was created.
        """
        db = SessionLocal()
        try:
            today = date.today()
            
            # Check if already checked in today
            existing = db.query(Attendance).filter(
                Attendance.employee_id == employee_id,
                Attendance.date == today
            ).first()
            
            if existing:
                return False  # Already checked in, ignore
            
            # First recognition today → create check-in
            record = Attendance(
                employee_id=employee_id,
                date=today,
                check_in=datetime.utcnow(),
                confidence=confidence
            )
            db.add(record)
            db.commit()
            return True
        finally:
            db.close()
    
    def explicit_checkout(self, employee_id: int) -> bool:
        """Explicit checkout — called from API or UI only."""
        db = SessionLocal()
        try:
            today = date.today()
            record = db.query(Attendance).filter(
                Attendance.employee_id == employee_id,
                Attendance.date == today,
                Attendance.check_out.is_(None)
            ).first()
            
            if not record:
                return False
            
            record.check_out = datetime.utcnow()
            db.commit()
            return True
        finally:
            db.close()
```

### 12.5 Exit Criteria

- First recognition = check-in (one row per employee per day)
- Subsequent recognitions = no duplicate rows
- Check-out = explicit only (API endpoint)
- Confidence score stored with attendance record
- Thread-safe database access from engine thread

---

## 13. Phase 10 — FastAPI REST API

### 13.1 Goal

Expose all backend functionality over HTTP. FastAPI controls the Recognition Engine, never runs CV in request handlers.

### 13.2 Files

| File | Purpose |
|:-----|:--------|
| `backend/app/main.py` | App factory, router registration, engine lifecycle |
| `backend/app/api/employees.py` | Employee CRUD + enrollment |
| `backend/app/api/attendance.py` | Attendance queries + explicit checkout |
| `backend/app/api/recognition.py` | Engine control |
| `backend/app/api/stream.py` | MJPEG video stream |

### 13.3 Application Entry Point

```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import CORS_ORIGINS
from app.database.database import init_db
from app.engine.recognition_engine import RecognitionEngine

# Global engine instance
engine = RecognitionEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown
    engine.stop()

app = FastAPI(
    title="Attendance System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(employees_router, prefix="/api")
app.include_router(attendance_router, prefix="/api")
app.include_router(recognition_router, prefix="/api")
app.include_router(stream_router, prefix="/api")
```

### 13.4 API Endpoints

#### Employees

| Method | Path | Request Body | Response |
|:-------|:-----|:-------------|:---------|
| `POST` | `/api/employees` | `{code, name, department}` | `{id, code, name, ...}` |
| `GET` | `/api/employees` | — | `[{id, code, name, ...}]` |
| `GET` | `/api/employees/{id}` | — | `{id, code, name, ...}` |
| `DELETE` | `/api/employees/{id}` | — | `{success: true}` |
| `POST` | `/api/employees/{id}/enroll` | — (uses webcam) | `{status, samples, ...}` |

#### Recognition Engine

| Method | Path | Request Body | Response |
|:-------|:-----|:-------------|:---------|
| `POST` | `/api/recognition/start` | — | `{status: "running"}` |
| `POST` | `/api/recognition/stop` | — | `{status: "stopped"}` |
| `GET` | `/api/recognition/status` | — | `{running, fps, tracks, ...}` |

#### Attendance

| Method | Path | Query Params | Response |
|:-------|:-----|:-------------|:---------|
| `GET` | `/api/attendance` | `?date=YYYY-MM-DD` | `[{id, employee, check_in, ...}]` |
| `GET` | `/api/attendance/today` | — | `[{id, employee, check_in, ...}]` |
| `POST` | `/api/attendance/{id}/checkout` | — | `{success, check_out}` |

#### MJPEG Stream

| Method | Path | Response |
|:-------|:-----|:---------|
| `GET` | `/api/stream` | `multipart/x-mixed-replace` MJPEG stream |

### 13.5 MJPEG Stream Implementation

```python
# stream.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

def generate_mjpeg():
    """Yield JPEG frames as multipart stream."""
    while True:
        frame = engine.get_frame()
        if frame is None:
            continue
        
        _, buffer = cv2.imencode('.jpg', frame, 
                                  [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY])
        
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + 
            buffer.tobytes() + 
            b'\r\n'
        )

@router.get("/stream")
async def video_stream():
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

### 13.6 Exit Criteria

- All endpoints return correct responses
- Engine starts/stops via API
- MJPEG stream works in browser (`<img src="/api/stream" />`)
- CORS configured for React dev server (localhost:5173)
- Database initialized on startup
- Engine stopped on shutdown

---

## 14. Phase 11 — React Dashboard

### 14.1 Goal

Build a premium, modern dashboard with live camera view, employee management, and attendance tracking.

### 14.2 Framework

React 19 + Vite (already scaffolded in `frontend/`)

### 14.3 Pages

#### Dashboard (Home)

```text
┌──────────────────────────────────────────────────────┐
│  🎯 ATTENDANCE SYSTEM                    [Engine: ON] │
├──────────────────────────┬───────────────────────────┤
│                          │                           │
│     LIVE CAMERA FEED     │   RECENT DETECTIONS       │
│                          │                           │
│     ┌────────────────┐   │   ✅ Hunain Shahid        │
│     │                │   │      08:42 — 86% conf     │
│     │   [MJPEG]      │   │                           │
│     │                │   │   ✅ Ali Ahmed             │
│     └────────────────┘   │      08:44 — 91% conf     │
│                          │                           │
│                          │   ❓ Unknown face          │
│                          │      08:45                 │
├──────────────────────────┴───────────────────────────┤
│  TODAY'S SUMMARY                                      │
│                                                      │
│  Total: 12/50 checked in    |  Recognition FPS: 28   │
│  Last check-in: 08:44       |  Active tracks: 3      │
└──────────────────────────────────────────────────────┘
```

#### Employees

```text
┌──────────────────────────────────────────────────────┐
│  EMPLOYEES                          [+ Add Employee]  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  🔍 Search: [________________]                       │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Code   │ Name           │ Dept    │ Samples │ ⚙ │ │
│  ├────────┼────────────────┼─────────┼─────────┼───┤ │
│  │ E001   │ Hunain Shahid  │ Eng     │ 7/8     │ ✏ │ │
│  │ E002   │ Ali Ahmed      │ Design  │ 6/8     │ ✏ │ │
│  │ E003   │ Sara Khan      │ HR      │ 5/8     │ ✏ │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

#### Enrollment (Guided Capture)

```text
┌──────────────────────────────────────────────────────┐
│  ENROLL: Hunain Shahid (E001)            Step 2/6    │
├──────────────────────────────────────────────────────┤
│                                                      │
│     ┌────────────────┐     📌 Slight turn LEFT       │
│     │                │                               │
│     │   [CAMERA]     │     Turn your head ~15°       │
│     │                │     to the left               │
│     └────────────────┘                               │
│                                                      │
│     Quality: ████████░░ 82%                          │
│                                                      │
│     Progress: ██░░░░░░ 2/7 samples captured          │
│                                                      │
│     [◀ Previous]              [Capture ▶]            │
└──────────────────────────────────────────────────────┘
```

#### Attendance

```text
┌──────────────────────────────────────────────────────┐
│  ATTENDANCE                  📅 [2026-08-21 ▼]       │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Employee       │ Check-in │ Check-out │ Status  │ │
│  ├────────────────┼──────────┼───────────┼─────────┤ │
│  │ Hunain Shahid  │ 08:42    │ —         │ ✅ In   │ │
│  │ Ali Ahmed      │ 08:44    │ 17:05     │ ⬜ Out  │ │
│  │ Sara Khan      │ 09:01    │ —         │ ✅ In   │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  [Export CSV]                                        │
└──────────────────────────────────────────────────────┘
```

### 14.4 Key Components

```text
src/
├── components/
│   ├── Layout.jsx           # App shell, navigation
│   ├── CameraFeed.jsx       # <img src="/api/stream" /> MJPEG
│   ├── EmployeeCard.jsx     # Employee list item
│   ├── AttendanceTable.jsx  # Sortable/filterable table
│   ├── EnrollmentWizard.jsx # Guided capture flow
│   ├── EngineStatus.jsx     # Start/stop + stats
│   └── StatsPanel.jsx       # Today's summary
├── pages/
│   ├── Dashboard.jsx
│   ├── Employees.jsx
│   ├── Enrollment.jsx
│   └── Attendance.jsx
├── api/
│   └── client.js            # Fetch wrapper for FastAPI
├── App.jsx
├── App.css
├── index.css
└── main.jsx
```

### 14.5 Camera Feed Component

```jsx
// The simplest possible live camera feed.
// MJPEG streams work natively in <img> tags.
function CameraFeed() {
    return (
        <img 
            src="http://127.0.0.1:8000/api/stream"
            alt="Live camera feed"
            style={{ width: '100%', borderRadius: '12px' }}
        />
    );
}
```

### 14.6 Exit Criteria

- Dashboard shows live MJPEG feed
- Employee list with add/delete
- Enrollment wizard with pose instructions
- Attendance table with date filter + explicit checkout button
- Engine start/stop controls
- Responsive layout
- Premium, modern design (dark mode, gradients, smooth transitions)

---

## 15. Testing Strategy

### 15.1 Automated Tests

```bash
# Camera sanity
python -m pytest tests/test_camera.py -v

# Face detection (SCRFD only, no ArcFace)
python -m pytest tests/test_detection.py -v

# End-to-end recognition
python -m pytest tests/test_recognition.py -v

# Attendance logic
python -m pytest tests/test_attendance.py -v

# All tests
python -m pytest tests/ -v
```

### 15.2 Manual Verification Matrix

| Check | Method | Pass Criteria |
|:------|:-------|:-------------|
| Camera | Run test_camera.py | 720p @ 30 FPS, 0 drops |
| Detection FPS | Benchmark detector | ≥ 25 FPS (SCRFD only) |
| Model swap | `buffalo_s` → `buffalo_l` | Compare FPS, accuracy |
| Quality gate | Show bad faces | Rejects blurry/dark/side faces |
| Enrollment | Register 3 people | Guided poses, 5+ samples each |
| Recognition | Walk into view | Recognized within 1 second |
| Attendance | Check DB | One row per person per day |
| Checkout | Click UI button | check_out column updated |
| MJPEG stream | Open in browser | Smooth annotated video |

### 15.3 Threshold Tuning Dataset

Build before going to production:

| Category | Min Samples |
|:---------|:-----------|
| Genuine pairs (same person, different conditions) | 50 |
| Impostor pairs (different people) | 50 |
| Glasses on/off | 10 |
| Low light | 10 |
| Side angle (15°–30°) | 10 |
| Different distances (1m, 2m, 3m) | 10 |

Plot FAR vs FRR. Select threshold at operating point (e.g., FAR < 0.1%, FRR < 5%).

---

## 16. Performance Benchmarks

### 16.1 Benchmarks to Run

| Benchmark | Command | Target |
|:----------|:--------|:-------|
| Camera FPS | `test_camera.py` | ≥ 25 FPS |
| SCRFD detection | `test_detection.py` | ~22 ms, ≥ 25 FPS |
| ArcFace embedding | Timed in embedder | ~8 ms |
| FAISS search (100 emb) | Timed in matcher | < 1 ms |
| FAISS search (5000 emb) | Timed in matcher | < 1 ms |
| Full engine loop | Engine stats | ≥ 20 FPS |
| CPU usage (engine running) | Task Manager | < 50% |
| RAM usage (engine running) | Task Manager | < 2 GB |

### 16.2 Surface Laptop Studio Specs

| Component | Spec |
|:----------|:-----|
| CPU | Intel i5-11300H (4C/8T, 3.1–4.4 GHz) |
| RAM | 16 GB LPDDR4x |
| GPU | Intel Iris Xe (integrated) |
| Camera | 1080p front-facing |
| OS | Windows 11 |

### 16.3 Expected Results

```text
buffalo_s on i5-11300H:
  SCRFD detection:    ~22 ms → ~25–30 FPS
  ArcFace embedding:  ~8 ms
  FAISS search:       < 1 ms
  Full pipeline FPS:  ~25 FPS (detection only)
  Recognition latency: ~0.2–1 s (new face to identity)
```

---

## 17. Security Considerations

### 17.1 For v1 (Local Prototype)

| Area | Implementation |
|:-----|:--------------|
| Face embeddings | Biometric identifiers — never log or expose via API |
| SQLite file | Store in `data/` directory, gitignored |
| FAISS index | Store in `data/` directory, gitignored |
| API access | Localhost only (127.0.0.1) |
| CORS | Restricted to localhost origins |
| No auth | Acceptable for local prototype |

### 17.2 For Production (Future)

- API authentication (JWT / session-based)
- Admin roles for employee management
- Encrypted SQLite or migrate to PostgreSQL
- Audit logging for all attendance changes
- Data retention policies
- Employee consent workflow
- Liveness detection (anti-spoofing)
- Rate limiting on API endpoints
- HTTPS

---

## 18. Future Enhancements

### 18.1 Short-term

| Enhancement | Purpose |
|:------------|:--------|
| WebSocket events | Real-time recognition notifications to UI |
| ByteTrack | Better face tracking than IoU |
| Liveness detection | Anti-photo/video spoofing |
| CSV export | Download attendance reports |

### 18.2 Medium-term

| Enhancement | Purpose |
|:------------|:--------|
| PostgreSQL migration | Replace SQLite for concurrent access |
| pgvector | Replace FAISS for database-integrated vector search |
| Multiple cameras | Support USB cameras alongside built-in |
| GPU acceleration | ONNX Runtime with DirectML for Iris Xe |

### 18.3 Long-term

| Enhancement | Purpose |
|:------------|:--------|
| Mobile enrollment | Use S23 Ultra for high-quality photos |
| Cloud deployment | Supabase / PostgreSQL cloud backend |
| Multi-office | Federated attendance across locations |
| Analytics dashboard | Trends, late arrivals, patterns |

---

<p align="center">
  <sub>Implementation Plan v3.0 — Last updated: 2026-08-21</sub>
</p>
