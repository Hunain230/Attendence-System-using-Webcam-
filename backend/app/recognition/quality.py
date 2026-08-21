"""
Face Quality Gate

Evaluates detected face crops before ArcFace feature extraction.
Checks run in order of computational cost (cheapest first):
  1. Face bounding box size check
  2. Mean pixel brightness check
  3. Sharpness / blur check (Laplacian variance)
  4. Aspect ratio / occlusion heuristic
  5. Head pose estimation (yaw/pitch) from 5 2D facial landmarks
  6. Composite quality score (0–100)

Quality Score Formula (0–100):
  score = (
      size_score     * 0.20  +  # face resolution
      sharpness_score * 0.30  +  # Laplacian variance
      pose_score     * 0.25  +  # yaw + pitch distance from frontal
      brightness_score * 0.15 +  # mean gray pixel value
      aspect_score   * 0.10     # width/height ratio
  ) × 100

Does NOT run ArcFace inference, FAISS vector matching, or database operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

import numpy as np
import cv2

from app import config


@dataclass
class QualityResult:
    """Explicit pass/fail result from QualityGate."""

    passed: bool
    reason: str
    score: float           # 0.0 – 1.0 composite score (normalized)
    score_100: float = 0.0  # 0 – 100 score for display
    metrics: Dict[str, float] = field(default_factory=dict)


class QualityGate:
    """Lightweight, deterministic face quality filter for CPU target.

    Two modes:
      - Recognition mode: lenient thresholds, faster pipeline.
      - Enrollment mode: strict thresholds, rejects mediocre frames.
    """

    def __init__(
        self,
        min_face_size: Optional[Tuple[int, int]] = None,
        blur_threshold: Optional[float] = None,
        max_yaw: Optional[float] = None,
        max_pitch: Optional[float] = None,
        min_brightness: Optional[float] = None,
        min_quality_score: Optional[float] = None,
    ):
        self.min_face_size = min_face_size or config.MIN_FACE_SIZE
        self.blur_threshold = blur_threshold if blur_threshold is not None else config.BLUR_THRESHOLD
        self.max_yaw = max_yaw if max_yaw is not None else config.MAX_YAW
        self.max_pitch = max_pitch if max_pitch is not None else config.MAX_PITCH
        self.min_brightness = min_brightness if min_brightness is not None else config.MIN_BRIGHTNESS
        self.min_quality_score = min_quality_score if min_quality_score is not None else config.MIN_QUALITY_SCORE

    def check(
        self,
        face_crop: np.ndarray,
        bbox: np.ndarray,
        landmarks: np.ndarray,
    ) -> QualityResult:
        """Evaluates face crop quality criteria for recognition.

        Args:
            face_crop: BGR or Grayscale image crop of the face.
            bbox: Bounding box numpy array [x1, y1, x2, y2].
            landmarks: 5×2 2D facial landmarks array (eyes, nose, mouth).

        Returns:
            QualityResult(passed, reason, score, score_100, metrics)
        """
        # Malformed input validation
        if (
            face_crop is None
            or not isinstance(face_crop, np.ndarray)
            or face_crop.size == 0
        ):
            return QualityResult(passed=False, reason="invalid_image", score=0.0, score_100=0.0)

        if bbox is None or not isinstance(bbox, np.ndarray) or len(bbox) < 4:
            return QualityResult(passed=False, reason="invalid_bbox", score=0.0, score_100=0.0)

        if (
            landmarks is None
            or not isinstance(landmarks, np.ndarray)
            or landmarks.shape != (5, 2)
        ):
            return QualityResult(passed=False, reason="invalid_landmarks", score=0.0, score_100=0.0)

        # 1. Size check (< 0.01 ms)
        width = float(bbox[2] - bbox[0])
        height = float(bbox[3] - bbox[1])

        if width < self.min_face_size[0] or height < self.min_face_size[1]:
            return QualityResult(
                passed=False,
                reason="face_too_small",
                score=0.0,
                score_100=0.0,
                metrics={"width": width, "height": height},
            )

        # Prepare grayscale image for brightness & blur checks
        gray = self._to_gray(face_crop)
        if gray is None:
            return QualityResult(passed=False, reason="invalid_image_format", score=0.0, score_100=0.0)

        # 2. Brightness check (< 0.1 ms)
        brightness = float(np.mean(gray))
        if brightness < self.min_brightness:
            return QualityResult(
                passed=False,
                reason="too_dark",
                score=0.0,
                score_100=0.0,
                metrics={"brightness": brightness},
            )

        # 3. Blur check via Laplacian variance (< 0.5 ms)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.blur_threshold:
            return QualityResult(
                passed=False,
                reason="too_blurry",
                score=0.0,
                score_100=0.0,
                metrics={"laplacian_var": laplacian_var, "brightness": brightness},
            )

        # 4. Aspect ratio / occlusion heuristic (< 0.01 ms)
        # Extreme aspect ratios indicate occlusion, extreme angle, or partial detection
        aspect = width / (height + 1e-6)
        if aspect < 0.5 or aspect > 1.6:
            return QualityResult(
                passed=False,
                reason="occluded_or_extreme_angle",
                score=0.0,
                score_100=0.0,
                metrics={"aspect_ratio": aspect, "width": width, "height": height},
            )

        # 5. Head pose estimation from 5 landmarks (< 0.1 ms)
        yaw, pitch = self._estimate_pose(landmarks)

        metrics = {
            "width": width,
            "height": height,
            "brightness": brightness,
            "laplacian_var": laplacian_var,
            "yaw": yaw,
            "pitch": pitch,
            "aspect_ratio": aspect,
        }

        if abs(yaw) > self.max_yaw:
            return QualityResult(passed=False, reason="excessive_yaw", score=0.0, score_100=0.0, metrics=metrics)

        if abs(pitch) > self.max_pitch:
            return QualityResult(passed=False, reason="excessive_pitch", score=0.0, score_100=0.0, metrics=metrics)

        # 6. Compute composite quality score (0–1, then 0–100)
        raw_score = self._compute_score(brightness, laplacian_var, yaw, pitch, width, height, aspect)

        if raw_score < self.min_quality_score:
            return QualityResult(
                passed=False,
                reason="quality_too_low",
                score=raw_score,
                score_100=raw_score * 100.0,
                metrics=metrics,
            )

        return QualityResult(
            passed=True,
            reason="passed",
            score=raw_score,
            score_100=raw_score * 100.0,
            metrics=metrics,
        )

    def check_enrollment_strict(
        self,
        face_crop: np.ndarray,
        bbox: np.ndarray,
        landmarks: np.ndarray,
    ) -> QualityResult:
        """Stricter quality check for enrollment candidate frames.

        Uses ENROLLMENT_* thresholds from config. Otherwise identical pipeline.
        """
        strict_gate = QualityGate(
            min_face_size=config.ENROLLMENT_MIN_FACE_SIZE,
            blur_threshold=config.ENROLLMENT_BLUR_THRESHOLD,
            max_yaw=config.MAX_YAW,  # Overall max, pose-specific checked separately
            max_pitch=config.MAX_PITCH,
            min_brightness=config.ENROLLMENT_MIN_BRIGHTNESS,
            min_quality_score=config.ENROLLMENT_MIN_QUALITY_SCORE,
        )
        return strict_gate.check(face_crop, bbox, landmarks)

    def _estimate_pose(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """Estimates yaw and pitch from 5-point landmarks.

        Landmark indexing (InsightFace standard):
          0: left eye
          1: right eye
          2: nose tip
          3: left mouth corner
          4: right mouth corner

        Yaw  > 0 → head turned right (nose to the right of eye center)
        Yaw  < 0 → head turned left
        Pitch > 0 → chin down (nose below neutral)
        Pitch < 0 → chin up
        """
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]

        eye_center = (left_eye + right_eye) / 2.0
        eye_width = float(np.linalg.norm(right_eye - left_eye)) + 1e-6

        # Yaw: horizontal nose offset from eye midpoint, scaled by inter-eye distance
        nose_offset_x = (nose[0] - eye_center[0]) / eye_width
        yaw = float(np.degrees(np.arctan(nose_offset_x))) * 2.2

        # Pitch: vertical nose offset relative to neutral baseline (~0.57× inter-eye width)
        nose_offset_y = (nose[1] - eye_center[1]) / eye_width
        pitch = float(np.degrees(np.arctan(nose_offset_y - 0.57))) * 2.2

        return yaw, pitch

    def check_enrollment_pose(
        self, landmarks: np.ndarray, target_pose: str
    ) -> Tuple[bool, str, float, float]:
        """Checks if current head orientation satisfies a target enrollment pose.

        Uses tighter pose ranges than recognition-time checks to ensure
        each captured angle genuinely represents the target pose.

        Returns:
            (matches_pose, guidance_message, yaw, pitch)
        """
        yaw, pitch = self._estimate_pose(landmarks)

        # Tighter ranges as specified in config (replaces old ad-hoc per-pose checks)
        if target_pose == "straight":
            if abs(yaw) <= config.POSE_STRAIGHT_MAX_YAW and abs(pitch) <= config.POSE_STRAIGHT_MAX_PITCH:
                return True, "Hold still — collecting best frame...", yaw, pitch
            if abs(yaw) > config.POSE_STRAIGHT_MAX_YAW:
                direction = "left" if yaw < 0 else "right"
                return False, f"Look directly at the camera (turning too far {direction})", yaw, pitch
            return False, "Keep chin level and look straight at the camera", yaw, pitch

        elif target_pose == "slight_left":
            # Negative yaw = face turned left in image coordinates
            if (
                config.POSE_LEFT_YAW_MIN <= abs(yaw) <= config.POSE_LEFT_YAW_MAX
                and yaw < 0
                and abs(pitch) <= config.POSE_LEFT_MAX_PITCH
            ):
                return True, "Hold still — collecting best frame...", yaw, pitch
            if abs(yaw) < config.POSE_LEFT_YAW_MIN:
                return False, "Turn head slightly more to the left (←)", yaw, pitch
            if abs(yaw) > config.POSE_LEFT_YAW_MAX:
                return False, "Too far left — turn back slightly towards center", yaw, pitch
            if abs(pitch) > config.POSE_LEFT_MAX_PITCH:
                return False, "Keep chin level while turning left", yaw, pitch
            return False, "Adjust position", yaw, pitch

        elif target_pose == "slight_right":
            # Positive yaw = face turned right
            if (
                config.POSE_RIGHT_YAW_MIN <= abs(yaw) <= config.POSE_RIGHT_YAW_MAX
                and yaw > 0
                and abs(pitch) <= config.POSE_RIGHT_MAX_PITCH
            ):
                return True, "Hold still — collecting best frame...", yaw, pitch
            if abs(yaw) < config.POSE_RIGHT_YAW_MIN:
                return False, "Turn head slightly more to the right (→)", yaw, pitch
            if abs(yaw) > config.POSE_RIGHT_YAW_MAX:
                return False, "Too far right — turn back slightly towards center", yaw, pitch
            if abs(pitch) > config.POSE_RIGHT_MAX_PITCH:
                return False, "Keep chin level while turning right", yaw, pitch
            return False, "Adjust position", yaw, pitch

        elif target_pose == "slight_up":
            # Negative pitch = chin tilted up (nose above neutral)
            if (
                config.POSE_UP_PITCH_MIN <= abs(pitch) <= config.POSE_UP_PITCH_MAX
                and pitch < 0
                and abs(yaw) <= config.POSE_UP_MAX_YAW
            ):
                return True, "Hold still — collecting best frame...", yaw, pitch
            if abs(pitch) < config.POSE_UP_PITCH_MIN:
                return False, "Tilt chin slightly upward (↑)", yaw, pitch
            if abs(pitch) > config.POSE_UP_PITCH_MAX:
                return False, "Too far up — lower chin slightly", yaw, pitch
            if abs(yaw) > config.POSE_UP_MAX_YAW:
                return False, "Face the camera while tilting up", yaw, pitch
            return False, "Adjust position", yaw, pitch

        elif target_pose == "slight_down":
            # Positive pitch = chin tilted down (nose below neutral)
            if (
                config.POSE_DOWN_PITCH_MIN <= abs(pitch) <= config.POSE_DOWN_PITCH_MAX
                and pitch > 0
                and abs(yaw) <= config.POSE_DOWN_MAX_YAW
            ):
                return True, "Hold still — collecting best frame...", yaw, pitch
            if abs(pitch) < config.POSE_DOWN_PITCH_MIN:
                return False, "Tilt chin slightly downward (↓)", yaw, pitch
            if abs(pitch) > config.POSE_DOWN_PITCH_MAX:
                return False, "Too far down — raise chin slightly", yaw, pitch
            if abs(yaw) > config.POSE_DOWN_MAX_YAW:
                return False, "Face the camera while tilting down", yaw, pitch
            return False, "Adjust position", yaw, pitch

        elif target_pose == "smile":
            if abs(yaw) <= config.POSE_SMILE_MAX_YAW and abs(pitch) <= config.POSE_SMILE_MAX_PITCH:
                return True, "Hold still — collecting best frame...", yaw, pitch
            return False, "Smile naturally while looking straight at the camera", yaw, pitch

        # Unknown pose — accept
        return True, "Hold still — collecting best frame...", yaw, pitch

    def check_landmark_stability(
        self, prev_landmarks: np.ndarray, curr_landmarks: np.ndarray
    ) -> bool:
        """Checks whether landmark positions are stable between two consecutive frames.

        Used during enrollment candidate collection to avoid capturing motion-blurred
        or transitional frames.

        Returns:
            True if landmarks are sufficiently stable (mean displacement ≤ threshold).
        """
        if prev_landmarks is None or curr_landmarks is None:
            return True  # No history yet — accept

        displacement = np.mean(np.linalg.norm(curr_landmarks - prev_landmarks, axis=1))
        return float(displacement) <= config.ENROLLMENT_LANDMARK_STABILITY_THRESHOLD

    def _compute_score(
        self,
        brightness: float,
        sharpness: float,
        yaw: float,
        pitch: float,
        w: float,
        h: float,
        aspect: float,
    ) -> float:
        """Composite quality score 0.0–1.0.

        Components:
          size_score     : rewards larger face regions (more detail for ArcFace)
          sharpness_score: rewards Laplacian variance (sharper = higher score)
          pose_score     : rewards near-frontal pose (drops off with |yaw|+|pitch|)
          brightness_score: rewards well-lit faces
          aspect_score   : rewards normal face aspect ratios (~0.7–1.1)
        """
        # Size: normalize to a 200×200 reference face
        size_score = min(1.0, float(w * h) / (200.0 * 200.0))

        # Sharpness: normalize to Laplacian var of 80 (standard sharp webcam image)
        sharpness_score = min(1.0, sharpness / 80.0)

        # Pose: penalty grows linearly with angular deviation
        pose_score = max(0.0, 1.0 - (abs(yaw) + abs(pitch)) / 70.0)

        # Brightness: normalize to 160 (well-lit face)
        brightness_score = min(1.0, brightness / 160.0)

        # Aspect: ideal face aspect is ~0.75–1.05; penalize deviations
        ideal_aspect = 0.9
        aspect_score = max(0.0, 1.0 - abs(aspect - ideal_aspect) / 0.5)

        composite = (
            size_score * 0.20
            + sharpness_score * 0.30
            + pose_score * 0.25
            + brightness_score * 0.15
            + aspect_score * 0.10
        )
        return float(np.clip(composite, 0.0, 1.0))

    @staticmethod
    def _to_gray(img: np.ndarray) -> Optional[np.ndarray]:
        """Converts BGR or grayscale image to grayscale."""
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            return img
        return None
