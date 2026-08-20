"""
Recognition Engine — Background Thread

Owns the full pipeline:
  Camera Capture → Detection (periodic) → Tracking (every frame)
  → Quality Gate (new/uncertain) → ArcFace Embedding → FAISS Search
  → Temporal Confirmation → Attendance Service → MJPEG Frame Buffer

FastAPI controls this engine (start/stop/status) but never runs
the computational loop inside HTTP request handlers.
"""
