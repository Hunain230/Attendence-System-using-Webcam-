"""
Face Tracker — IoU-Based Track State Manager

Tracks face bounding boxes across frames using Intersection over Union (IoU) overlap.
Maintains identity recognition state, temporal confirmation history, and expiration.
Crucial performance feature: Prevents repeated ArcFace feature extractions on already
recognized or evaluated faces.
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
import numpy as np

from app import config
from app.recognition.detector import DetectedFace


@dataclass
class Track:
    """Represents a tracked face across consecutive video frames."""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    landmarks: np.ndarray  # (5, 2)
    face_crop: np.ndarray  # BGR crop
    raw_face: object  # Raw InsightFace face object or wrapper

    # Recognition state
    recognition_history: List[Tuple[Optional[int], float]] = field(default_factory=list)
    confirmed_identity: Optional[int] = None  # employee_id once confirmed
    confirmed_name: Optional[str] = None  # employee name once confirmed
    is_unknown: bool = False
    frames_since_recognition: int = 0
    lost_frames: int = 0

    @property
    def is_recognized(self) -> bool:
        """Returns True if employee identity has been confirmed."""
        return self.confirmed_identity is not None

    @property
    def should_skip_recognition(self) -> bool:
        """ArcFace Skip Rule: Skip ArcFace if identity is confirmed OR marked unknown."""
        return self.is_recognized or self.is_unknown

    @property
    def is_confirmed(self) -> bool:
        """Returns True if the last TEMPORAL_FRAMES recognitions consistently agree."""
        if len(self.recognition_history) < config.TEMPORAL_FRAMES:
            return False

        recent = self.recognition_history[-config.TEMPORAL_FRAMES:]
        recent_ids = [r[0] for r in recent]

        # All recent recognitions must match the same non-None employee_id
        return len(set(recent_ids)) == 1 and recent_ids[0] is not None

    def add_recognition(
        self, employee_id: Optional[int], confidence: float, name: Optional[str] = None
    ):
        """Appends a recognition candidate result and updates confirmed identity."""
        self.recognition_history.append((employee_id, confidence))
        self.frames_since_recognition = 0

        if self.is_confirmed and self.confirmed_identity is None:
            self.confirmed_identity = employee_id
            self.confirmed_name = name

    def mark_unknown(self):
        """Marks this track as an unknown face so ArcFace is not repeatedly called."""
        self.is_unknown = True
        self.recognition_history.append((None, 0.0))


class FaceTracker:
    """IoU-based multi-face tracker."""

    def __init__(self, iou_threshold: Optional[float] = None, max_lost_frames: Optional[int] = None):
        self.iou_threshold = (
            iou_threshold if iou_threshold is not None else config.IOU_THRESHOLD
        )
        self.max_lost_frames = (
            max_lost_frames if max_lost_frames is not None else config.MAX_LOST_FRAMES
        )
        self.tracks: List[Track] = []
        self._next_track_id: int = 1

    def update(self, detections: List[DetectedFace]) -> List[Track]:
        """Updates active tracks with new frame detections.

        Algorithm:
          1. Calculate pairwise IoU matrix between existing active tracks and new detections.
          2. Greedy match tracks to detections based on maximum IoU >= threshold.
          3. Update matched tracks with new bbox, crop, and landmarks.
          4. Spawn new tracks for unmatched detections.
          5. Mark unmatched existing tracks as lost (+1 lost_frames).
          6. Purge tracks lost for longer than max_lost_frames.
        """
        if not self.tracks:
            # All detections become new tracks
            for det in detections:
                self._spawn_track(det)
            return self.tracks

        if not detections:
            # Increment lost_frames for all tracks
            for track in self.tracks:
                track.lost_frames += 1
            self._purge_expired_tracks()
            return self.tracks

        # Compute IoU matrix (N_tracks x M_detections)
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.compute_iou(track.bbox, det.bbox)

        matched_track_indices: Set[int] = set()
        matched_det_indices: Set[int] = set()

        # Flatten IoU matrix and sort matches by descending overlap
        flat_indices = np.argsort(iou_matrix.ravel())[::-1]

        for idx in flat_indices:
            track_idx = int(idx // num_dets)
            det_idx = int(idx % num_dets)

            if iou_matrix[track_idx, det_idx] < self.iou_threshold:
                break

            if track_idx in matched_track_indices or det_idx in matched_det_indices:
                continue

            # Match found
            matched_track_indices.add(track_idx)
            matched_det_indices.add(det_idx)

            # Update track
            track = self.tracks[track_idx]
            det = detections[det_idx]
            track.bbox = det.bbox
            track.landmarks = det.landmarks
            track.face_crop = det.face_crop
            track.raw_face = det.raw_face
            track.lost_frames = 0
            track.frames_since_recognition += 1

        # Unmatched detections -> spawn new tracks
        for j in range(num_dets):
            if j not in matched_det_indices:
                self._spawn_track(detections[j])

        # Unmatched tracks -> increment lost_frames
        for i in range(num_tracks):
            if i not in matched_track_indices:
                self.tracks[i].lost_frames += 1

        self._purge_expired_tracks()
        return self.tracks

    def reset(self):
        """Resets all tracks."""
        self.tracks = []
        self._next_track_id = 1

    def _spawn_track(self, det: DetectedFace):
        track = Track(
            track_id=self._next_track_id,
            bbox=det.bbox,
            landmarks=det.landmarks,
            face_crop=det.face_crop,
            raw_face=det.raw_face,
        )
        self.tracks.append(track)
        self._next_track_id += 1

    def _purge_expired_tracks(self):
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost_frames]

    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
        area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
        union = area1 + area2 - intersection

        return intersection / union if union > 0.0 else 0.0
