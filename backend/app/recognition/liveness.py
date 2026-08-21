"""
Liveness Checker — Lightweight Anti-Spoofing Heuristics

Provides lightweight, CPU-safe liveness checks without requiring a dedicated
neural network inference model. This is NOT a certified anti-spoofing system,
but provides meaningful protection against casual photo/phone/screen attacks.

Three complementary checks:
  1. Optical Flow Stillness
     Tracks the inter-frame motion of detected landmarks using Lucas-Kanade
     optical flow estimation. A real face has small, natural micro-motion from
     breathing and subtle head sway. A printed photo or screen has near-zero motion.

  2. LBP Texture Uniformity
     Computes a Local Binary Pattern (LBP) histogram of the face crop.
     Real faces have rich, varied texture. Screen captures and glossy photos have
     highly uniform texture. High uniformity → suspicious.

  3. Impossible Pose Transition
     Monitors yaw/pitch changes across frames. A real face cannot jump more than
     ~40° in a single frame. If it does, a different photo is likely being held up.

All three checks use a sliding window of recent frames per track.
A verdict is only made after LIVENESS_MIN_FRAMES_FOR_CHECK frames are collected,
preventing false-positives on track initialization.

Configuration (all in config.py):
  LIVENESS_ENABLED              : Master on/off switch
  LIVENESS_FLOW_THRESHOLD       : Min optical flow magnitude (real face threshold)
  LIVENESS_LBP_THRESHOLD        : Max LBP uniformity score (screen/photo threshold)
  LIVENESS_HISTORY_FRAMES       : Frame history window size per track
  LIVENESS_MIN_FRAMES_FOR_CHECK : Min frames before first verdict

Liveness does NOT block recognition. It blocks ATTENDANCE. A liveness-failed track
is still displayed in the MJPEG stream with an "⚠ Liveness" warning.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from app import config


@dataclass
class _TrackLivenessState:
    """Per-track liveness analysis state."""
    landmark_history: Deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=config.LIVENESS_HISTORY_FRAMES)
    )
    flow_magnitudes: Deque[float] = field(
        default_factory=lambda: deque(maxlen=config.LIVENESS_HISTORY_FRAMES)
    )
    lbp_scores: Deque[float] = field(
        default_factory=lambda: deque(maxlen=config.LIVENESS_HISTORY_FRAMES)
    )
    prev_gray: Optional[np.ndarray] = None
    prev_yaw: Optional[float] = None
    prev_pitch: Optional[float] = None
    impossible_transition_count: int = 0


class LivenessChecker:
    """Manages per-track liveness analysis using lightweight heuristics."""

    def __init__(self) -> None:
        # Maps track_id → _TrackLivenessState
        self._states: Dict[int, _TrackLivenessState] = {}

    def update(self, track: "Track") -> None:  # type: ignore[name-defined]
        """Updates liveness analysis for a track and sets track.liveness_failed.

        Args:
            track: A Track object (from tracker.py) with face_crop and landmarks.
        """
        if not config.LIVENESS_ENABLED:
            track.liveness_failed = False
            return

        # Clean up state for tracks that have been removed
        tid = track.track_id
        if tid not in self._states:
            self._states[tid] = _TrackLivenessState()

        state = self._states[tid]

        face_crop = track.face_crop
        landmarks = track.landmarks

        if face_crop is None or face_crop.size == 0 or landmarks is None:
            return

        # Convert face to grayscale for analysis
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop

        # ── Check 1: Optical flow stillness ──────────────────────────────────
        flow_mag = 0.0
        if state.prev_gray is not None:
            try:
                # Resize both to same size for flow comparison
                h, w = gray.shape[:2]
                prev_resized = cv2.resize(state.prev_gray, (w, h))
                # Lucas-Kanade sparse optical flow on landmark points as features
                lk_points = landmarks.astype(np.float32).reshape(-1, 1, 2)
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_resized, gray, lk_points, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                )
                if next_pts is not None and status is not None:
                    good_mask = status.ravel() == 1
                    if good_mask.sum() > 0:
                        displacement = np.linalg.norm(
                            next_pts[good_mask] - lk_points[good_mask], axis=2
                        )
                        flow_mag = float(np.mean(displacement))
            except Exception:
                pass

        state.flow_magnitudes.append(flow_mag)
        state.prev_gray = gray.copy()

        # ── Check 2: LBP texture uniformity ──────────────────────────────────
        lbp_score = self._compute_lbp_uniformity(gray)
        state.lbp_scores.append(lbp_score)

        # ── Check 3: Impossible pose transition ───────────────────────────────
        if hasattr(track, "last_yaw") and hasattr(track, "last_pitch"):
            yaw = track.last_yaw
            pitch = track.last_pitch
            if state.prev_yaw is not None and state.prev_pitch is not None:
                yaw_jump = abs(yaw - state.prev_yaw)
                pitch_jump = abs(pitch - state.prev_pitch)
                if yaw_jump > 40.0 or pitch_jump > 35.0:
                    state.impossible_transition_count += 1
                else:
                    state.impossible_transition_count = max(0, state.impossible_transition_count - 1)
            state.prev_yaw = yaw
            state.prev_pitch = pitch

        state.landmark_history.append(landmarks.copy())

        # ── Verdict (only after enough frames) ────────────────────────────────
        min_frames = config.LIVENESS_MIN_FRAMES_FOR_CHECK
        if len(state.flow_magnitudes) < min_frames:
            track.liveness_failed = False  # Not enough data yet
            return

        # Evaluate each check
        avg_flow = float(np.mean(list(state.flow_magnitudes)))
        avg_lbp = float(np.mean(list(state.lbp_scores)))

        flow_suspicious = avg_flow < config.LIVENESS_FLOW_THRESHOLD
        texture_suspicious = avg_lbp > config.LIVENESS_LBP_THRESHOLD
        transition_suspicious = state.impossible_transition_count >= 2

        # Liveness fails if 2 or more checks are suspicious
        # (Single check alone can be a false positive — require corroboration)
        suspicious_count = sum([flow_suspicious, texture_suspicious, transition_suspicious])
        track.liveness_failed = suspicious_count >= 2

    def remove_track(self, track_id: int) -> None:
        """Cleans up liveness state for a removed track."""
        self._states.pop(track_id, None)

    def reset(self) -> None:
        """Resets all liveness state (e.g., when engine restarts)."""
        self._states.clear()

    @staticmethod
    def _compute_lbp_uniformity(gray: np.ndarray) -> float:
        """Computes LBP histogram uniformity score.

        A score close to 1.0 indicates very uniform texture (photo/screen).
        A score around 0.3–0.6 is typical for real faces.

        Returns:
            float in [0.0, 1.0] where 1.0 = perfectly uniform.
        """
        if gray.size == 0:
            return 0.5  # Unknown

        # Resize to fixed size for consistent LBP
        face_resized = cv2.resize(gray, (64, 64))

        # Simple 3×3 LBP
        h, w = face_resized.shape
        lbp = np.zeros_like(face_resized, dtype=np.uint8)

        # Compare center pixel with 8 neighbors
        center = face_resized[1:-1, 1:-1].astype(np.int16)
        shifts = [
            face_resized[:-2, :-2],   # top-left
            face_resized[:-2, 1:-1],  # top
            face_resized[:-2, 2:],    # top-right
            face_resized[1:-1, 2:],   # right
            face_resized[2:, 2:],     # bottom-right
            face_resized[2:, 1:-1],   # bottom
            face_resized[2:, :-2],    # bottom-left
            face_resized[1:-1, :-2],  # left
        ]

        lbp_val = np.zeros_like(center, dtype=np.uint8)
        for bit, neighbor in enumerate(shifts):
            lbp_val += ((neighbor.astype(np.int16) >= center).astype(np.uint8) << bit)

        # Compute histogram
        hist, _ = np.histogram(lbp_val.ravel(), bins=32, range=(0, 256))
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total == 0:
            return 0.5

        hist /= total

        # Uniformity = sum of squared bin probabilities (high = uniform texture)
        uniformity = float(np.sum(hist ** 2))

        # Scale to [0, 1]: max uniformity = 1.0 (all pixels same value),
        # min uniformity ≈ 1/32 = 0.03 (perfectly uniform histogram)
        min_uniform = 1.0 / 32.0
        normalized = (uniformity - min_uniform) / (1.0 - min_uniform + 1e-6)
        return float(np.clip(normalized, 0.0, 1.0))
