# 🎯 Attendance System using Webcam

An **offline-first facial recognition attendance system** built with OpenCV, InsightFace, FAISS, SQLite, FastAPI, and React. Designed to run entirely locally on a Surface Laptop Studio (i5-11300H / 16 GB / Iris Xe) with the built-in 1080p webcam.

---

## Architecture

```text
                    ┌──────────────┐
                    │ Surface Cam  │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │   OpenCV     │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │    SCRFD     │
                    │  Detection   │
                    │  (periodic)  │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │ IoU Tracker  │
                    │(every frame) │
                    └──────┬───────┘
                           ↓
                     New / uncertain?
                       ↙         ↘
                     NO          YES
                     ↓            ↓
                  Track      Quality Gate
                  only            ↓
                             ArcFace
                                  ↓
                            512-D Vector
                                  ↓
                              FAISS
                                  ↓
                           Employee ID
                                  ↓
                         Temporal Confirm
                                  ↓
                         Attendance Service
                                  ↓
                              SQLite
                                  ↓
                    ┌─────────────┴─────────────┐
                    ↓                           ↓
                 FastAPI                    MJPEG
                    ↓                           ↓
                 React UI  ←────────────────────┘
```

### Key Design Decisions

| Decision | Detail |
|----------|--------|
| **Model** | `buffalo_s` (configurable to `buffalo_l`) |
| **Pipeline** | Detection (periodic) → Tracking (every frame) → Recognition (new tracks only) |
| **Embeddings** | 512-D ArcFace, L2-normalized, configurable dimension |
| **Vector search** | FAISS `IndexFlatIP` (cosine similarity via inner product) |
| **Attendance** | First recognition = check-in; check-out = **explicit only** |
| **Camera feed** | MJPEG stream (no WebSocket in v1) |
| **Paths** | All project-relative via `config.py` — fully portable |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Capture** | OpenCV | Webcam capture, frame processing |
| **Detection** | InsightFace (SCRFD) | Face detection with bounding boxes + landmarks |
| **Recognition** | InsightFace (ArcFace) | 512-D face embedding extraction |
| **Inference** | ONNX Runtime | Run InsightFace models efficiently on CPU |
| **Vector Search** | FAISS (CPU) | Local similarity search over embeddings |
| **Database** | SQLite + SQLAlchemy | Employees, attendance, embedding metadata |
| **Backend** | FastAPI + Uvicorn | REST API, engine control, MJPEG stream |
| **Frontend** | React + Vite | Dashboard, employee management, attendance view |

---

## Project Structure

```text
attendance-system/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI entry point
│   │   ├── config.py                   # All settings (model, thresholds, paths)
│   │   ├── api/
│   │   │   ├── employees.py            # Employee CRUD + enrollment
│   │   │   ├── attendance.py           # Attendance queries + explicit checkout
│   │   │   ├── recognition.py          # Engine start/stop/status
│   │   │   └── stream.py              # MJPEG video stream
│   │   ├── engine/
│   │   │   └── recognition_engine.py   # Background recognition thread
│   │   ├── recognition/
│   │   │   ├── detector.py             # SCRFD face detection
│   │   │   ├── embedder.py             # ArcFace embedding
│   │   │   ├── matcher.py              # FAISS search + threshold
│   │   │   ├── tracker.py              # IoU-based face tracking
│   │   │   └── quality.py              # Face quality checks
│   │   ├── database/
│   │   │   ├── models.py               # SQLAlchemy ORM models
│   │   │   ├── database.py             # Engine + session factory
│   │   │   └── repository.py           # CRUD helpers
│   │   └── attendance/
│   │       └── service.py              # Check-in / explicit checkout logic
│   ├── data/                           # Runtime data (gitignored)
│   └── requirements.txt
├── frontend/                           # React + Vite app
├── tests/
├── .gitignore
└── README.md
```

---

## Prerequisites

- **Python** 3.10 or 3.11
- **Node.js** 18+ and npm
- **Webcam** (Surface built-in or USB)
- **OS**: Windows 10/11

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Hunain230/Attendence-System-using-Webcam-.git
cd Attendence-System-using-Webcam-
```

### 2. Backend — Python environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r backend/requirements.txt
```

> **Note**: InsightFace will automatically download the `buffalo_s` model (~100 MB) on first run.

### 3. Frontend — React app

```bash
cd frontend
npm install
cd ..
```

### 4. Run the system

```bash
# Terminal 1: Backend
.\venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** for the dashboard.

---

## Configuration

All settings live in [`backend/app/config.py`](backend/app/config.py). Key values:

| Setting | Default | Description |
|---------|---------|-------------|
| `INSIGHTFACE_MODEL` | `buffalo_s` | InsightFace model pack |
| `EMBEDDING_DIM` | `512` | ArcFace embedding dimension |
| `DETECTION_INTERVAL` | `3` | Run SCRFD every N frames |
| `SIMILARITY_THRESHOLD` | `0.45` | Initial threshold (tune via FAR/FRR) |
| `TEMPORAL_FRAMES` | `3` | Consecutive confirmations needed |
| `CAMERA_INDEX` | `0` | OpenCV webcam device index |
| `CAPTURE_WIDTH` | `1280` | Webcam capture width |
| `CAPTURE_HEIGHT` | `720` | Webcam capture height |

Override via environment variables:

```bash
$env:INSIGHTFACE_MODEL = "buffalo_l"
$env:SIMILARITY_THRESHOLD = "0.50"
$env:CAMERA_INDEX = "1"
```

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/employees` | Create employee |
| `GET` | `/api/employees` | List employees |
| `GET` | `/api/employees/{id}` | Get single employee |
| `DELETE` | `/api/employees/{id}` | Deactivate (soft delete) |
| `POST` | `/api/employees/{id}/enroll` | Start guided enrollment |
| `POST` | `/api/recognition/start` | Start recognition engine |
| `POST` | `/api/recognition/stop` | Stop recognition engine |
| `GET` | `/api/recognition/status` | Engine state + stats |
| `GET` | `/api/attendance` | All attendance (filter by `?date=`) |
| `GET` | `/api/attendance/today` | Today's attendance |
| `POST` | `/api/attendance/{id}/checkout` | Explicit checkout |
| `GET` | `/api/stream` | MJPEG live video stream |

---

## Database Schema

### employees

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | Auto-increment |
| employee_code | TEXT UNIQUE | Business identifier |
| name | TEXT | Full name |
| department | TEXT | |
| created_at | DATETIME | UTC |
| active | BOOLEAN | Soft delete flag |

### face_embeddings

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| employee_id | FK → employees | |
| faiss_id | INTEGER | Maps to FAISS vector index |
| quality_score | REAL | Quality gate score at enrollment |
| created_at | DATETIME | UTC |

### attendance

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| employee_id | FK → employees | |
| date | DATE | One row per employee per day |
| check_in | DATETIME | First recognition of the day |
| check_out | DATETIME | Explicit only (nullable) |
| confidence | REAL | Mean confidence at recognition |
| created_at | DATETIME | UTC |

---

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Camera validation (OpenCV, 720p @ 30 FPS) | ⬜ |
| 2 | Face detection (SCRFD via `buffalo_s`) | ⬜ |
| 3 | Face quality gate (blur, size, angle, brightness) | ⬜ |
| 4 | Face embedding (ArcFace, 512-D, L2-normalized) | ⬜ |
| 5 | SQLite schema + ORM (SQLAlchemy) | ⬜ |
| 6 | FAISS index + ID mapping | ⬜ |
| 7 | Employee enrollment (guided variation capture) | ⬜ |
| 8 | Recognition engine + tracking + temporal confirmation | ⬜ |
| 9 | Attendance service (check-in + explicit checkout) | ⬜ |
| 10 | FastAPI REST API | ⬜ |
| 11 | React + Vite dashboard | ⬜ |

---

## Testing

```bash
# Activate venv
.\venv\Scripts\Activate.ps1

# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_camera.py -v
```

---

## License

This project is for educational and prototyping purposes.

---

## Author

**Hunain Shahid**
