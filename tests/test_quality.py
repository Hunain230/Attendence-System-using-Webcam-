"""
Unit tests for Phase 3 — Face Quality Gate (quality.py)

Verifies:
  1. Valid/high-quality face passes quality checks
  2. Individual criteria failures (size, brightness, blur, yaw, pitch)
  3. Boundary condition handling
  4. Invalid / malformed inputs
  5. Multi-face evaluation
"""

import pytest
import numpy as np
import cv2

from app.recognition.quality import QualityGate, QualityResult
from app import config


@pytest.fixture
def quality_gate():
    return QualityGate()


@pytest.fixture
def valid_face_sample():
    """Generates a high-quality synthetic face crop, bbox, and landmarks."""
    # 120x120 sharp, well-lit image crop
    crop = np.full((120, 120, 3), 128, dtype=np.uint8)
    # Add random high-frequency texture so Laplacian variance is high (~200+)
    rng = np.random.default_rng(42)
    noise = rng.integers(-40, 40, size=(120, 120, 3), dtype=np.int16)
    crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    bbox = np.array([10.0, 10.0, 130.0, 130.0], dtype=np.float32)  # 120x120 size

    # Symmetrical, centered 5 facial landmarks
    # [left_eye, right_eye, nose, left_mouth, right_mouth]
    landmarks = np.array(
        [
            [40.0, 45.0],  # left eye
            [80.0, 45.0],  # right eye
            [60.0, 55.7],  # nose tip (eye_width=40, vertical dist=10.7 -> offset=0.2675 -> ~15° -> pitch ~0°)
            [45.0, 95.0],  # left mouth
            [75.0, 95.0],  # right mouth
        ],
        dtype=np.float32,
    )

    return crop, bbox, landmarks


# ── 1. Valid High-Quality Face ────────────────────────────────


def test_valid_face_passes(quality_gate, valid_face_sample):
    crop, bbox, landmarks = valid_face_sample
    res = quality_gate.check(crop, bbox, landmarks)

    assert isinstance(res, QualityResult)
    assert res.passed is True
    assert res.reason == "passed"
    assert res.score > 0.0
    assert "laplacian_var" in res.metrics
    assert "brightness" in res.metrics
    assert "yaw" in res.metrics
    assert "pitch" in res.metrics


# ── 2. Individual Failure Criteria ────────────────────────────


def test_face_too_small(quality_gate, valid_face_sample):
    crop, _, landmarks = valid_face_sample
    small_bbox = np.array([0.0, 0.0, 60.0, 60.0], dtype=np.float32)  # 60x60 < 80x80

    res = quality_gate.check(crop, small_bbox, landmarks)

    assert res.passed is False
    assert res.reason == "face_too_small"
    assert res.score == 0.0


def test_face_too_dark(quality_gate, valid_face_sample):
    _, bbox, landmarks = valid_face_sample
    dark_crop = np.full((120, 120, 3), 10, dtype=np.uint8)  # Brightness ~10 < 40

    res = quality_gate.check(dark_crop, bbox, landmarks)

    assert res.passed is False
    assert res.reason == "too_dark"
    assert res.metrics["brightness"] < config.MIN_BRIGHTNESS


def test_face_too_blurry(quality_gate, valid_face_sample):
    _, bbox, landmarks = valid_face_sample
    # Smooth solid crop (zero contrast variance)
    smooth_crop = np.full((120, 120, 3), 120, dtype=np.uint8)

    res = quality_gate.check(smooth_crop, bbox, landmarks)

    assert res.passed is False
    assert res.reason == "too_blurry"
    assert res.metrics["laplacian_var"] < config.BLUR_THRESHOLD


def test_excessive_yaw(quality_gate, valid_face_sample):
    crop, bbox, landmarks = valid_face_sample
    # Shift nose significantly to the right to simulate head turned left/right
    shifted_landmarks = landmarks.copy()
    shifted_landmarks[2] = [100.0, 69.0]  # nose far right relative to eye center 60

    res = quality_gate.check(crop, bbox, shifted_landmarks)

    assert res.passed is False
    assert res.reason == "excessive_yaw"
    assert abs(res.metrics["yaw"]) > config.MAX_YAW


def test_excessive_pitch(quality_gate, valid_face_sample):
    crop, bbox, landmarks = valid_face_sample
    # Shift nose significantly down to simulate head tilted upward/downward
    shifted_landmarks = landmarks.copy()
    shifted_landmarks[2] = [60.0, 110.0]  # nose far down

    res = quality_gate.check(crop, bbox, shifted_landmarks)

    assert res.passed is False
    assert res.reason == "excessive_pitch"
    assert abs(res.metrics["pitch"]) > config.MAX_PITCH


# ── 3. Boundary Condition Cases ───────────────────────────────


def test_size_boundary_conditions(valid_face_sample):
    crop, _, landmarks = valid_face_sample
    gate = QualityGate(min_face_size=(80, 80))

    # Exactly 80x80 -> should pass size check
    exact_bbox = np.array([0.0, 0.0, 80.0, 80.0], dtype=np.float32)
    res_exact = gate.check(crop, exact_bbox, landmarks)
    assert res_exact.reason != "face_too_small"

    # 79x79 -> should fail size check
    under_bbox = np.array([0.0, 0.0, 79.0, 79.0], dtype=np.float32)
    res_under = gate.check(crop, under_bbox, landmarks)
    assert res_under.reason == "face_too_small"


def test_brightness_boundary_conditions(valid_face_sample):
    _, bbox, landmarks = valid_face_sample
    gate = QualityGate(min_brightness=40.0)

    # Crop with mean 40
    crop_40 = np.full((120, 120, 3), 40, dtype=np.uint8)
    # Add small noise for blur check
    crop_40[:10, :10] = 200
    res_40 = gate.check(crop_40, bbox, landmarks)
    assert res_40.reason != "too_dark"

    # Crop with mean 39
    crop_39 = np.full((120, 120, 3), 39, dtype=np.uint8)
    res_39 = gate.check(crop_39, bbox, landmarks)
    assert res_39.reason == "too_dark"


# ── 4. Invalid & Malformed Inputs ─────────────────────────────


def test_invalid_inputs(quality_gate, valid_face_sample):
    crop, bbox, landmarks = valid_face_sample

    # None / Empty image
    assert quality_gate.check(None, bbox, landmarks).reason == "invalid_image"
    assert (
        quality_gate.check(np.array([]), bbox, landmarks).reason == "invalid_image"
    )

    # Invalid bbox
    assert quality_gate.check(crop, None, landmarks).reason == "invalid_bbox"
    assert (
        quality_gate.check(crop, np.array([10.0, 10.0]), landmarks).reason
        == "invalid_bbox"
    )

    # Invalid landmarks
    assert quality_gate.check(crop, bbox, None).reason == "invalid_landmarks"
    assert (
        quality_gate.check(crop, bbox, np.zeros((4, 2))).reason
        == "invalid_landmarks"
    )


# ── 5. Multi-Face Evaluation ──────────────────────────────────


def test_multi_face_evaluation(quality_gate, valid_face_sample):
    crop, bbox, landmarks = valid_face_sample

    # Good face + dark face + small face
    dark_crop = np.full((120, 120, 3), 10, dtype=np.uint8)
    small_bbox = np.array([0.0, 0.0, 40.0, 40.0], dtype=np.float32)

    faces_input = [
        (crop, bbox, landmarks),
        (dark_crop, bbox, landmarks),
        (crop, small_bbox, landmarks),
    ]

    results = [quality_gate.check(c, b, l) for c, b, l in faces_input]

    assert len(results) == 3
    assert results[0].passed is True
    assert results[0].reason == "passed"
    assert results[1].passed is False
    assert results[1].reason == "too_dark"
    assert results[2].passed is False
    assert results[2].reason == "face_too_small"
