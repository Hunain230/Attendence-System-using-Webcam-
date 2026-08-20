"""
Face Quality Gate

Evaluates detected face crops before ArcFace feature extraction.
Checks run in order of computational cost (cheapest first):
  1. Face bounding box size check (MIN_FACE_SIZE)
  2. Mean pixel brightness check (MIN_BRIGHTNESS)
  3. Sharpness / blur check via Laplacian variance (BLUR_THRESHOLD)
  4. Head pose estimation (yaw/pitch) from 5 2D facial landmarks (MAX_YAW, MAX_PITCH)

Does NOT run ArcFace inference, FAISS vector matching, or database operations.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

from app import config


@dataclass
class QualityResult:
    """Explicit pass/fail result from QualityGate."""

    passed: bool
    reason: str
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)


class QualityGate:
    """Lightweight, deterministic face quality filter for CPU target."""

    def __init__(
        self,
        min_face_size: Optional[Tuple[int, int]] = None,
        blur_threshold: Optional[float] = None,
        max_yaw: Optional[float] = None,
        max_pitch: Optional[float] = None,
        min_brightness: Optional[float] = None,
    ):
        self.min_face_size = min_face_size or config.MIN_FACE_SIZE
        self.blur_threshold = blur_threshold or config.BLUR_THRESHOLD
        self.max_yaw = max_yaw or config.MAX_YAW
        self.max_pitch = max_pitch or config.MAX_PITCH
        self.min_brightness = min_brightness or config.MIN_BRIGHTNESS

    def check(
        self,
        face_crop: np.ndarray,
        bbox: np.ndarray,
        landmarks: np.ndarray,
    ) -> QualityResult:
        """Evaluates face crop quality criteria.

        Args:
            face_crop: BGR or Grayscale image crop of the face.
            bbox: Bounding box numpy array [x1, y1, x2, y2].
            landmarks: 5x2 2D facial landmarks array (eyes, nose, mouth).

        Returns:
            QualityResult(passed, reason, score, metrics)
        """
        # Malformed input validation
        if (
            face_crop is None
            or not isinstance(face_crop, np.ndarray)
            or face_crop.size == 0
        ):
            return QualityResult(
                passed=False,
                reason="invalid_image",
                score=0.0,
                metrics={},
            )

        if bbox is None or not isinstance(bbox, np.ndarray) or len(bbox) < 4:
            return QualityResult(
                passed=False,
                reason="invalid_bbox",
                score=0.0,
                metrics={},
            )

        if (
            landmarks is None
            or not isinstance(landmarks, np.ndarray)
            or landmarks.shape != (5, 2)
        ):
            return QualityResult(
                passed=False,
                reason="invalid_landmarks",
                score=0.0,
                metrics={},
            )

        # 1. Size check (< 0.01 ms)
        width = float(bbox[2] - bbox[0])
        height = float(bbox[3] - bbox[1])

        if width < self.min_face_size[0] or height < self.min_face_size[1]:
            return QualityResult(
                passed=False,
                reason="face_too_small",
                score=0.0,
                metrics={"width": width, "height": height},
            )

        # Prepare grayscale image for brightness & blur checks
        if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        elif len(face_crop.shape) == 2:
            gray = face_crop
        else:
            return QualityResult(
                passed=False,
                reason="invalid_image_format",
                score=0.0,
                metrics={},
            )

        # 2. Brightness check (< 0.1 ms)
        brightness = float(np.mean(gray))
        if brightness < self.min_brightness:
            return QualityResult(
                passed=False,
                reason="too_dark",
                score=0.0,
                metrics={"brightness": brightness},
            )

        # 3. Blur check via Laplacian variance (< 0.5 ms)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.blur_threshold:
            return QualityResult(
                passed=False,
                reason="too_blurry",
                score=0.0,
                metrics={"laplacian_var": laplacian_var, "brightness": brightness},
            )

        # 4. Head pose estimation from 5 landmarks (< 0.1 ms)
        yaw, pitch = self._estimate_pose(landmarks)

        metrics = {
            "width": width,
            "height": height,
            "brightness": brightness,
            "laplacian_var": laplacian_var,
            "yaw": yaw,
            "pitch": pitch,
        }

        if abs(yaw) > self.max_yaw:
            return QualityResult(
                passed=False,
                reason="excessive_yaw",
                score=0.0,
                metrics=metrics,
            )

        if abs(pitch) > self.max_pitch:
            return QualityResult(
                passed=False,
                reason="excessive_pitch",
                score=0.0,
                metrics=metrics,
            )

        # All quality checks passed → calculate composite score (0.0 - 1.0)
        quality_score = self._compute_score(
            brightness, laplacian_var, yaw, pitch, width, height
        )

        return QualityResult(
            passed=True,
            reason="passed",
            score=quality_score,
            metrics=metrics,
        )

    def _estimate_pose(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """Estimates approximate yaw and pitch angles in degrees from 5-point facial landmarks.

        Landmarks indexing:
          0: left eye
          1: right eye
          2: nose tip
          3: left mouth corner
          4: right mouth corner
        """
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]

        eye_center = (left_eye + right_eye) / 2.0
        eye_width = float(np.linalg.norm(right_eye - left_eye))

        if eye_width < 1e-6:
            return 0.0, 0.0

        # Yaw: horizontal offset of nose relative to eye midpoint
        nose_offset_x = (nose[0] - eye_center[0]) / eye_width
        yaw = float(np.degrees(np.arctan(nose_offset_x * 2.0)))

        # Pitch: vertical offset of nose relative to eye midpoint
        nose_offset_y = (nose[1] - eye_center[1]) / eye_width
        pitch = float(np.degrees(np.arctan((nose_offset_y - 0.6) * 2.0)))

        return yaw, pitch

    def _compute_score(
        self,
        brightness: float,
        sharpness: float,
        yaw: float,
        pitch: float,
        width: float,
        height: float,
    ) -> float:
        """Computes a normalized composite quality score 0.0–1.0."""
        size_score = min(1.0, float(width * height) / (200.0 * 200.0))
        blur_score = min(1.0, sharpness / 200.0)
        pose_score = max(0.0, 1.0 - (abs(yaw) + abs(pitch)) / 70.0)
        bright_score = min(1.0, brightness / 150.0)

        composite = (
            size_score * 0.2 + blur_score * 0.3 + pose_score * 0.3 + bright_score * 0.2
        )
        return float(np.clip(composite, 0.0, 1.0))
