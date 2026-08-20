<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.141-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/OpenCV-4.11-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/SQLite-Local-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<h1 align="center">🎯 Attendance System using Webcam</h1>

<p align="center">
  <strong>An offline-first facial recognition attendance system</strong><br>
  Built with InsightFace · FAISS · FastAPI · React<br>
  Runs entirely on local hardware — no cloud, no API keys, no internet required.
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-api-reference">API Reference</a> •
  <a href="#-database-schema">Database Schema</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## ✨ Features

- 🔍 **Real-time face detection** — SCRFD ~22 ms/frame → ~25–30 detection FPS
- 🧠 **Face recognition** — ArcFace produces ONE 512-D embedding per face (~8 ms), FAISS searches ~5000 stored embeddings (< 1 ms)
- 📹 **Live MJPEG stream** — annotated camera feed directly in the browser
- 👤 **Guided enrollment** — captures multiple poses (straight, left, right, up, down, smile)
- ✅ **Smart attendance** — auto check-in on first recognition; check-out is explicit only
- 🔒 **Fully offline** — no cloud dependencies, all data stays on your machine
- ⚡ **Optimized pipeline** — SCRFD detection → IoU tracking (every frame) → ArcFace only for new faces
- 🎛️ **Fully configurable** — model, thresholds, camera, quality gates — all in one `config.py`

---

## 🏗 Architecture

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

> **Recognition Engine** runs as a dedicated background thread. FastAPI controls it (start/stop/status) but never runs CV inside HTTP handlers. ArcFace produces ONE embedding per face — we do NOT run ArcFace against 5000 people. FAISS handles the vector search in < 1 ms.

### ⚡ Performance Targets

| Stage | Time | Frequency |
|:------|:-----|:----------|
| SCRFD detection | ~22 ms | Every frame → ~25–30 FPS |
| IoU tracking | < 1 ms | Every frame |
| ArcFace embedding | ~8 ms | New/uncertain faces only |
| FAISS search (5000 embeddings) | < 1 ms | Per new embedding |
| End-to-end recognition | ~0.2–1 s | Per new person |

---

## 🛠 Tech Stack

| Layer | Technology | Role |
|:------|:-----------|:-----|
| **Capture** | OpenCV 4.11 | Webcam capture, frame resize |
| **Detection** | InsightFace (SCRFD) | Face bounding boxes + 5-point landmarks |
| **Recognition** | InsightFace (ArcFace) | 512-D face embedding extraction |
| **Inference** | ONNX Runtime | Efficient CPU inference for InsightFace models |
| **Vector Search** | FAISS (CPU) | Cosine similarity search via `IndexFlatIP` |
| **Database** | SQLite + SQLAlchemy 2.0 | Employees, attendance, embedding metadata |
| **Backend** | FastAPI + Uvicorn | REST API, engine control, MJPEG stream |
| **Frontend** | React 19 + Vite | Dashboard, enrollment, attendance management |

---

## 📁 Project Structure

<details>
<summary><strong>Click to expand</strong></summary>

```text
attendance-system/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI entry point + engine lifecycle
│   │   ├── config.py                   # All settings (model, thresholds, paths)
│   │   │
│   │   ├── api/
│   │   │   ├── employees.py            # Employee CRUD + enrollment endpoints
│   │   │   ├── attendance.py           # Attendance queries + explicit checkout
│   │   │   ├── recognition.py          # Engine start / stop / status
│   │   │   └── stream.py              # MJPEG video stream
│   │   │
│   │   ├── engine/
│   │   │   └── recognition_engine.py   # Background thread: full CV pipeline
│   │   │
│   │   ├── recognition/
│   │   │   ├── detector.py             # SCRFD face detection wrapper
│   │   │   ├── embedder.py             # ArcFace embedding extraction
│   │   │   ├── matcher.py              # FAISS search + threshold logic
│   │   │   ├── tracker.py              # IoU-based face tracking
│   │   │   └── quality.py              # Face quality gate (blur, size, angle)
│   │   │
│   │   ├── database/
│   │   │   ├── models.py               # SQLAlchemy ORM models
│   │   │   ├── database.py             # Engine + session factory
│   │   │   └── repository.py           # CRUD helpers
│   │   │
│   │   └── attendance/
│   │       └── service.py              # Check-in / explicit checkout logic
│   │
│   ├── data/                           # Runtime data (gitignored)
│   │   ├── attendance.db
│   │   └── faiss.index
│   │
│   └── requirements.txt
│
├── frontend/                           # React + Vite app
│   ├── src/
│   ├── public/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── tests/
│   ├── test_camera.py
│   ├── test_detection.py
│   ├── test_recognition.py
│   └── test_attendance.py
│
├── .gitignore
└── README.md
```

</details>

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version |
|:------------|:--------|
| Python | 3.10 or 3.11 |
| Node.js | 18+ |
| npm | 9+ |
| Webcam | Any USB or built-in |
| OS | Windows 10/11 |

### 1. Clone the repository

```bash
git clone https://github.com/Hunain230/Attendence-System-using-Webcam-.git
cd Attendence-System-using-Webcam-
```

### 2. Set up the backend

```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1          # PowerShell
# or
.\venv\Scripts\activate.bat          # CMD

# Install Python dependencies
pip install -r backend/requirements.txt
```

> [!NOTE]
> InsightFace will automatically download the `buffalo_s` model pack (~100 MB) on first run.

### 3. Set up the frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Run the system

```bash
# Terminal 1 — Backend
.\venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## ⚙️ Configuration

All settings live in [`backend/app/config.py`](backend/app/config.py). Key values can be overridden via environment variables:

<details>
<summary><strong>Full configuration reference</strong></summary>

| Setting | Default | Env Var | Description |
|:--------|:--------|:--------|:------------|
| `INSIGHTFACE_MODEL` | `buffalo_s` | `INSIGHTFACE_MODEL` | Model pack (`buffalo_s` or `buffalo_l`) |
| `EMBEDDING_DIM` | `512` | `EMBEDDING_DIM` | ArcFace output dimension |
| `DETECTION_INTERVAL` | `3` | — | Run SCRFD every N frames |
| `SIMILARITY_THRESHOLD` | `0.45` | `SIMILARITY_THRESHOLD` | Initial threshold (tune via FAR/FRR) |
| `TEMPORAL_FRAMES` | `3` | — | Consecutive confirmations required |
| `FAISS_TOP_K` | `5` | — | Nearest neighbors to retrieve |
| `CAMERA_INDEX` | `0` | `CAMERA_INDEX` | OpenCV device index |
| `CAPTURE_WIDTH` | `1280` | — | Webcam capture width |
| `CAPTURE_HEIGHT` | `720` | — | Webcam capture height |
| `MIN_FACE_SIZE` | `(80, 80)` | — | Minimum face bounding box (px) |
| `BLUR_THRESHOLD` | `50.0` | — | Laplacian variance minimum |
| `MAX_YAW` / `MAX_PITCH` | `35.0°` | — | Maximum head rotation |
| `HOST` | `127.0.0.1` | `HOST` | Server bind address |
| `PORT` | `8000` | `PORT` | Server port |

</details>

**Quick override examples (PowerShell):**

```powershell
$env:INSIGHTFACE_MODEL = "buffalo_l"
$env:SIMILARITY_THRESHOLD = "0.50"
$env:CAMERA_INDEX = "1"
```

---

## 📡 API Reference

### Employees

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/api/employees` | Create a new employee |
| `GET` | `/api/employees` | List all employees |
| `GET` | `/api/employees/{id}` | Get employee details |
| `DELETE` | `/api/employees/{id}` | Deactivate employee (soft delete) |
| `POST` | `/api/employees/{id}/enroll` | Start guided face enrollment |

### Recognition Engine

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/api/recognition/start` | Start recognition engine |
| `POST` | `/api/recognition/stop` | Stop recognition engine |
| `GET` | `/api/recognition/status` | Engine state + live stats |

### Attendance

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/api/attendance` | All records (filter: `?date=YYYY-MM-DD`) |
| `GET` | `/api/attendance/today` | Today's attendance |
| `POST` | `/api/attendance/{id}/checkout` | Explicit checkout |

### Stream

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/api/stream` | MJPEG live video stream |

---

## 🗄 Database Schema

<details>
<summary><strong>employees</strong></summary>

| Column | Type | Constraint |
|:-------|:-----|:-----------|
| `id` | INTEGER | PRIMARY KEY, AUTO INCREMENT |
| `employee_code` | TEXT | UNIQUE, NOT NULL |
| `name` | TEXT | NOT NULL |
| `department` | TEXT | |
| `created_at` | DATETIME | UTC default |
| `active` | BOOLEAN | Default `true` |

</details>

<details>
<summary><strong>face_embeddings</strong></summary>

| Column | Type | Constraint |
|:-------|:-----|:-----------|
| `id` | INTEGER | PRIMARY KEY |
| `employee_id` | INTEGER | FK → `employees.id` |
| `faiss_id` | INTEGER | Maps to FAISS vector index |
| `quality_score` | REAL | Quality gate score at enrollment |
| `created_at` | DATETIME | UTC default |

</details>

<details>
<summary><strong>attendance</strong></summary>

| Column | Type | Constraint |
|:-------|:-----|:-----------|
| `id` | INTEGER | PRIMARY KEY |
| `employee_id` | INTEGER | FK → `employees.id` |
| `date` | DATE | One row per employee per day |
| `check_in` | DATETIME | First recognition of the day |
| `check_out` | DATETIME | Explicit only (nullable) |
| `confidence` | REAL | Mean confidence at recognition |
| `created_at` | DATETIME | UTC default |

</details>

---

## 🗺 Roadmap

| Phase | Description | Status |
|:-----:|:------------|:------:|
| 1 | Camera validation (720p @ 30 FPS) | ✅ |
| 2 | Face detection (SCRFD via `buffalo_s`) | ✅ |
| 3 | Face quality gate (Size, Brightness, Blur, Pose) | ✅ |
| 4 | Face embedding (ArcFace, 512-D) | ✅ |
| 5 | SQLite schema + ORM (SQLAlchemy) | ✅ |
| 6 | FAISS vector index + ID mapping (`IndexFlatIP`) | ✅ |
| 7 | Employee enrollment (guided multi-pose variation) | ✅ |
| 8 | Recognition engine + IoU tracker + skip rule optimization | ✅ |
| 9 | Attendance service (check-in + explicit checkout) | ⬜ |
| 10 | FastAPI REST API | ⬜ |
| 11 | React dashboard | ⬜ |

---

## 🧪 Testing

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run all tests
python -m pytest tests/ -v

# Run a specific test
python -m pytest tests/test_camera.py -v
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Hunain Shahid** — [@Hunain230](https://github.com/Hunain230)

---

<p align="center">
  <sub>Built with ❤️ using Python, OpenCV, InsightFace, FAISS, FastAPI & React</sub>
</p>
