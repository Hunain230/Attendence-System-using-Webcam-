"""
Face Quality Gate

Checks whether a detected face is suitable for embedding:
  - Minimum face size (MIN_FACE_SIZE)
  - Sharpness via Laplacian variance (BLUR_THRESHOLD)
  - Head pose within limits (MAX_YAW, MAX_PITCH)
  - Brightness above minimum (MIN_BRIGHTNESS)

All thresholds are configurable via config.py.
"""
