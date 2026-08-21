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
- 📹 **Live MJPEG stream** — annotated camera feed directly in the browser with bounding boxes & employee name badges
- 👤 **Guided multi-pose enrollment** — interactive capture of multiple poses (frontal, left, right, up, down, smile)
- ✅ **Smart attendance** — auto check-in on first recognition of the day; explicit checkout support
- 🔒 **Fully offline** — no cloud dependencies, all vector indices and databases stay on your local machine
- ⚡ **Optimized pipeline** — SCRFD detection → IoU tracking (every frame) → ArcFace only for new/uncertain faces
- 🎛️ **Fully configurable** — model, thresholds, camera index, quality gates — all in one `config.py`

---

## 🏗 Architecture

The system operates on an offline-first, dual-pipeline architecture powered by a dedicated background recognition engine and an asynchronous REST API:

```text
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │                            LIVE RECOGNITION PIPELINE                              │
 └───────────────────────────────────────────────────────────────────────────────────┘
                                Webcam 720p / 30 FPS
                                         ↓
                               ┌───────────────────┐
                               │       SCRFD       │  ~22 ms / frame
                               │   Face Detection  │  → ~25–30 detection FPS
                               └─────────┬─────────┘
                                         ↓
                               ┌───────────────────┐
                               │    IoU Tracker    │  Track across frames (< 1 ms)
                               └─────────┬─────────┘
                                         ↓
                               ┌─────────┴─────────┐
                               │                   │
                          Known Face            New Face
                               │                   │
                               │             Quality Gate
                               │             (Size, Blur, Yaw, Pitch)
                               │                   ↓
                               │              Face Align
                               │                   ↓
                               │              ArcFace          ~8 ms
                               │          (ONE embedding)
                               │                   ↓
                               └─────────┬─────────┘
                                         ↓
                                  512-D Embedding
                                         ↓
                               ┌───────────────────┐
                               │   FAISS Index     │  ~5000 × 512 stored embeddings
                               │   IndexFlatIP     │  Vector similarity (< 1 ms)
                               └─────────┬─────────┘
                                         ↓
                                   Nearest Match
                                         ↓
                                  Cosine Distance
                                         ↓
                               Temporal Confirmation  (3 consecutive matches)
                                         ↓
                                Attendance Service     (Debounced Check-in / Checkout)
                                         ↓
                               ┌─────────┴─────────┐
                               ↓                   ↓
                         SQLite Database     MJPEG Stream Buffer
                               ↓                   ↓
                            FastAPI             FastAPI
                        (/api/attendance)    (/api/stream)
                               ↓                   ↓
                               └─────────┬─────────┘
                                         ↓
                              React 19 Frontend Dashboard

 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │                          GUIDED ENROLLMENT PIPELINE                               │
 └───────────────────────────────────────────────────────────────────────────────────┘
   React Enrollment UI  ──(Guided Poses: Frontal, Left, Right, Up, Down, Smile)──►
          │
          ▼
   API: /api/enrollment/sample  ──► Quality Gate Validation ──► ArcFace Feature Extraction
          │
          ▼
   FAISS Index (add_with_ids) + SQLite (employees & face_embeddings tables)
```

> **Thread-Isolated Recognition Engine**: The OpenCV capture and AI inference pipeline runs continuously inside an isolated background daemon thread (`recognition_engine.py`). FastAPI handles client requests and controls engine state without blocking CV execution. ArcFace extracts ONE embedding per newly detected face rather than comparing against each employee, allowing FAISS to perform vector lookup in sub-millisecond time.

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
| **Capture** | OpenCV 4.11 | Webcam capture, frame preprocessing, MJPEG encoding |
| **Detection** | InsightFace (SCRFD) | Face bounding boxes + 5-point facial landmarks |
| **Recognition** | InsightFace (ArcFace) | 512-D normalized face embedding extraction |
| **Inference** | ONNX Runtime | CPU-optimized deep learning inference |
| **Vector Search** | FAISS (CPU) | Sub-millisecond cosine similarity search via `IndexFlatIP` |
| **Database** | SQLite + SQLAlchemy 2.0 | Persistent employee, attendance, and vector metadata |
| **Backend** | FastAPI + Uvicorn | High-performance async REST API and stream server |
| **Frontend** | React 19 + Vite | Real-time monitoring, guided enrollment & attendance portal |

---

## 📁 Project Structure

<details>
<summary><strong>Click to expand full repository structure</strong></summary>

```text
attendance-system/
│
├── run.bat                             # One-click Windows startup script
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI entry point & lifespan controller
│   │   ├── config.py                   # Central configuration & hyperparameters
│   │   │
│   │   ├── api/                        # REST API endpoints
│   │   │   ├── employees.py            # Employee CRUD & deactivation
│   │   │   ├── enrollment.py           # Multi-pose guided enrollment endpoints
│   │   │   ├── attendance.py           # Daily & historical attendance logs, checkout
│   │   │   ├── recognition.py          # Engine start / stop / status
│   │   │   └── stream.py              # Real-time annotated MJPEG video stream
│   │   │
│   │   ├── engine/
│   │   │   └── recognition_engine.py   # Dedicated worker thread for camera & CV loop
│   │   │
│   │   ├── recognition/                # Computer Vision core modules
│   │   │   ├── detector.py             # SCRFD face detector wrapper
│   │   │   ├── embedder.py             # ArcFace embedding extractor
│   │   │   ├── matcher.py              # FAISS index wrapper & cosine search
│   │   │   ├── tracker.py              # IoU multi-face tracker
│   │   │   ├── quality.py              # Blur, illumination, size & pose quality gates
│   │   │   └── enrollment.py           # Multi-pose enrollment state machine
│   │   │
│   │   ├── database/                   # Storage layer
│   │   │   ├── models.py               # SQLAlchemy ORM models (Employees, Embeddings, Attendance)
│   │   │   ├── database.py             # SQLite engine & session factory
│   │   │   └── repository.py           # Data access objects & query helpers
│   │   │
│   │   ├── schemas/                    # Pydantic request/response schemas
│   │   │   ├── employee.py             # Employee schemas
│   │   │   ├── enrollment.py           # Enrollment schemas
│   │   │   ├── attendance.py           # Attendance schemas
│   │   │   └── recognition.py          # Engine status schemas
│   │   │
│   │   └── attendance/
│   │       └── service.py              # First-seen check-in & explicit checkout business logic
│   │
│   ├── data/                           # Runtime persistent data (gitignored)
│   │   ├── attendance.db               # SQLite database file
│   │   └── faiss.index                 # FAISS vector index binary
│   │
│   └── requirements.txt                # Python backend dependencies
│
├── frontend/                           # React 19 + Vite web application
│   ├── src/
│   │   ├── api/
│   │   │   └── client.js               # Backend API client
│   │   ├── pages/
│   │   │   ├── DashboardPage.jsx       # Overview statistics & quick actions
│   │   │   ├── LiveRecognitionPage.jsx # Real-time camera feed & live recognition events
│   │   │   ├── EnrollmentPage.jsx      # Interactive guided multi-pose face enrollment
│   │   │   ├── EmployeesPage.jsx       # Employee directory & profile management
│   │   │   └── AttendancePage.jsx      # Daily records, date filters, explicit checkout
│   │   ├── components/                 # Reusable UI components
│   │   ├── App.jsx                     # Root application layout & routing
│   │   ├── App.css                     # Component & layout styling
│   │   ├── index.css                   # Global styles & theme definitions
│   │   └── main.jsx                    # React DOM entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── tests/                              # Pytest test suite
│   ├── conftest.py
│   ├── test_api.py                     # FastAPI endpoint tests
│   ├── test_attendance_service.py      # Attendance logic & debouncing tests
│   ├── test_database.py                # Database CRUD tests
│   ├── test_enrollment.py              # Enrollment pipeline tests
│   ├── test_matcher.py                 # FAISS similarity & threshold tests
│   ├── test_quality.py                 # Quality gate & pose angle tests
│   └── test_recognition_engine.py      # Background engine lifecycle tests
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
| Python | 3.10, 3.11, or 3.12 |
| Node.js | 18+ |
| npm | 9+ |
| Webcam | Any USB or built-in webcam |
| OS | Windows 10/11 |

---

### ⚡ Option A: One-Click Launch (Windows)

Simply double-click [`run.bat`](run.bat) (or run in Command Prompt):

```cmd
run.bat
```

This automated launcher will:
1. Verify the Python virtual environment and install missing dependencies if needed.
2. Check frontend packages (`npm install` if needed).
3. Start the FastAPI backend server on `http://127.0.0.1:8000`.
4. Start the Vite React frontend server on `http://localhost:5173`.
5. Automatically open the web app in your default browser.

---

### 🛠 Option B: Manual Setup

#### 1. Clone the repository

```bash
git clone https://github.com/Hunain230/Attendence-System-using-Webcam-.git
cd Attendence-System-using-Webcam-
```

#### 2. Set up the backend

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

#### 3. Set up the frontend

```bash
cd frontend
npm install
cd ..
```

#### 4. Run the system

```bash
# Terminal 1 — Backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload

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
| `POST` | `/api/employees` | Create a new employee record |
| `GET` | `/api/employees` | List all employees (`?active_only=true`) |
| `GET` | `/api/employees/{id}` | Get employee details |
| `DELETE` | `/api/employees/{id}` | Deactivate employee and remove face embeddings |

### Guided Face Enrollment

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/api/enrollment/start` | Create employee and initialize multi-pose session |
| `POST` | `/api/enrollment/sample` | Process base64 frame for current required pose |

### Recognition Engine

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/api/recognition/start` | Start camera capture & background recognition engine |
| `POST` | `/api/recognition/stop` | Stop recognition engine |
| `GET` | `/api/recognition/status` | Current engine state & live processing statistics |

### Attendance

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/api/attendance` | Attendance records (filter: `?date=YYYY-MM-DD`) |
| `GET` | `/api/attendance/today` | Today's attendance list |
| `POST` | `/api/attendance/{id}/checkout` | Mark explicit checkout timestamp |

### Video Stream

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/api/stream` | MJPEG live annotated video stream with bounding boxes |

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
| 9 | Attendance service (first check-in + explicit checkout + debounced cache) | ✅ |
| 10 | FastAPI REST API (CRUD routers + Pydantic + MJPEG stream) | ✅ |
| 11 | React dashboard | ✅ |

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
