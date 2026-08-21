"""
Face Tracker — IoU-Based Track State Manager

Tracks face bounding boxes across frames using Intersection over Union (IoU) overlap.
Maintains a state machine per track for identity recognition lifecycle.

Track State Machine
───────────────────
  new → evaluating → confirmed
                   → uncertain → unknown
                              → confirmed (recovery)

State Transition Rules:
  new        : Track just created. ArcFace runs every frame (no skip).
  evaluating : Recognition attempts are ongoing. ArcFace every frame.
  confirmed  : Identity confirmed by temporal voting. ArcFace every
               RECOGNIZED_RECHECK_INTERVAL frames to catch identity switches.
  uncertain  : 3+ consecutive unmatched frames. ArcFace every 10 frames.
  unknown    : 8+ cumulative unmatched frames. ArcFace every UNKNOWN_RETRY_INTERVAL
               frames. FIXES: permanent-Unknown-lock bug (was never retried).

Key Bug Fixes vs Previous Version:
  - confirmed tracks are re-checked periodically (was: never re-checked → ghost identity)
  - unknown tracks retry at configurable interval (was: every 6 *unmatched* calls,
    but unmatched_count reset on any recognition, creating escape loop)
  - temporal voting requires TEMPORAL_FRAMES majority, not just consecutive agreement
  - identity switching: if re-check returns a different person with high confidence
    AND margin, the identity is updated (was: confirmed identity was permanent)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Literal, Optional, Tuple
import time

import numpy as np

from app import config
from app.recognition.detector import DetectedFace


TrackState = Literal["new", "evaluating", "confirmed", "uncertain", "unknown"]


@dataclass
class RecognitionEvent:
    """A single recognition attempt result stored in track history."""
    employee_id: Optional[int]
    similarity: float
    second_id: Optional[int]
    second_similarity: float
    margin: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Track:
    """Represents a tracked face across consecutive video frames.

    Recognition lifecycle is governed by a state machine (see module docstring).
    Temporal voting across TEMPORAL_FRAMES recent events determines confirmation.
    """

    # ── Core track fields ─────────────────────────────────────────────────────
    track_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    landmarks: np.ndarray     # (5, 2)
    face_crop: np.ndarray     # BGR crop
    raw_face: object          # Raw InsightFace face object or DummyFace wrapper

    # ── Recognition state ─────────────────────────────────────────────────────
    state: TrackState = "new"

    recognition_history: Deque[RecognitionEvent] = field(
        default_factory=lambda: deque(maxlen=20)
    )

    confirmed_identity: Optional[int] = None   # employee_id once confirmed
    confirmed_name: Optional[str] = None       # employee name once confirmed
    best_similarity: float = 0.0              # highest similarity seen when confirmed

    # Counters
    frames_since_recognition: int = 0   # frames elapsed since last ArcFace call
    consecutive_unmatched: int = 0      # unmatched frames in a row
    total_unmatched: int = 0            # cumulative unmatched events
    lost_frames: int = 0                # frames without any detection match

    # Liveness
    liveness_failed: bool = False       # set by LivenessChecker

    # Quality metrics (from last quality gate run — used for debug overlay)
    last_quality_score: float = 0.0
    last_yaw: float = 0.0
    last_pitch: float = 0.0

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_recognized(self) -> bool:
        """Returns True if employee identity has been confirmed by temporal voting."""
        return self.state == "confirmed" and self.confirmed_identity is not None

    @property
    def is_unknown(self) -> bool:
        """Returns True if the face has been conclusively marked as unknown."""
        return self.state == "unknown"

    @property
    def display_state(self) -> str:
        """Human-readable state string for UI display."""
        if self.liveness_failed:
            return "liveness_failed"
        return self.state

    @property
    def should_skip_recognition(self) -> bool:
        """Determines whether to skip ArcFace for this track this frame.

        Returns True (skip) when the adaptive schedule says it's not time yet.
        Each state has a different retry interval to balance accuracy vs CPU load.
        """
        interval = self._recognition_interval()
        if interval <= 1:
            return False  # always run
        return (self.frames_since_recognition % interval) != 0

    def _recognition_interval(self) -> int:
        """Returns the ArcFace run interval (in frames) for the current state."""
        if self.state == "new":
            return config.RECOGNITION_INTERVAL_NEW
        if self.state == "evaluating":
            return config.RECOGNITION_INTERVAL_EVALUATING
        if self.state == "uncertain":
            return config.RECOGNITION_INTERVAL_UNCERTAIN
        if self.state == "unknown":
            return config.UNKNOWN_RETRY_INTERVAL
        if self.state == "confirmed":
            return config.RECOGNIZED_RECHECK_INTERVAL
        return 1  # fallback

    # ── State transitions ─────────────────────────────────────────────────────

    def add_recognition(
        self,
        employee_id: Optional[int],
        similarity: float,
        name: Optional[str] = None,
        second_id: Optional[int] = None,
        second_similarity: float = 0.0,
        margin: float = 0.0,
    ) -> None:
        """Records a successful recognition result and updates state machine.

        Args:
            employee_id:       The matched employee id (None if sub-threshold).
            similarity:        Best cosine similarity score.
            name:              Human-readable employee name.
            second_id:         Second-best employee id (for margin tracking).
            second_similarity: Second-best score.
            margin:            Score gap between best and second-best.
        """
        event = RecognitionEvent(
            employee_id=employee_id,
            similarity=similarity,
            second_id=second_id,
            second_similarity=second_similarity,
            margin=margin,
        )
        self.recognition_history.append(event)
        self.frames_since_recognition = 0
        self.consecutive_unmatched = 0

        if employee_id is not None:
            self.total_unmatched = 0

        # Determine temporal vote majority
        majority_id, vote_count = self._majority_vote()

        if majority_id is not None and vote_count >= config.TEMPORAL_FRAMES:
            if self.state != "confirmed":
                # First confirmation
                self.state = "confirmed"
                self.confirmed_identity = majority_id
                # Name comes from the most recent event with this employee_id
                for ev in reversed(self.recognition_history):
                    if ev.employee_id == majority_id:
                        if name and ev.employee_id == employee_id:
                            self.confirmed_name = name
                        break
                if name and employee_id == majority_id:
                    self.confirmed_name = name
                self.best_similarity = similarity
            else:
                # Already confirmed — check for identity switch
                if (
                    majority_id != self.confirmed_identity
                    and vote_count >= config.TEMPORAL_FRAMES
                    and similarity >= config.SIMILARITY_THRESHOLD
                    and margin >= config.SIMILARITY_MARGIN
                ):
                    # High-confidence switch: different person has taken over the track
                    self.confirmed_identity = majority_id
                    if name and employee_id == majority_id:
                        self.confirmed_name = name
                    self.best_similarity = similarity
                else:
                    # Update best similarity for confirmed identity
                    if employee_id == self.confirmed_identity and similarity > self.best_similarity:
                        self.best_similarity = similarity
        elif self.state == "new":
            self.state = "evaluating"

    def record_unmatched(self) -> None:
        """Records a frame where no match exceeded the similarity threshold.

        Progressive degradation:
          evaluating/confirmed + 3 misses → uncertain
          uncertain              + 5 more  → unknown (total 8)
        """
        self.consecutive_unmatched += 1
        self.total_unmatched += 1
        self.frames_since_recognition = 0

        event = RecognitionEvent(
            employee_id=None,
            similarity=0.0,
            second_id=None,
            second_similarity=0.0,
            margin=0.0,
        )
        self.recognition_history.append(event)

        if self.state == "confirmed":
            if self.consecutive_unmatched >= config.UNMATCHED_TO_UNCERTAIN:
                self.state = "uncertain"
                # Do NOT clear confirmed_identity here — display continues
                # until track is explicitly unknown or re-confirmed differently
        elif self.state in ("new", "evaluating", "uncertain"):
            if self.total_unmatched >= config.UNMATCHED_TO_UNKNOWN:
                self.state = "unknown"
                self.confirmed_identity = None
                self.confirmed_name = None
            elif self.consecutive_unmatched >= config.UNMATCHED_TO_UNCERTAIN:
                if self.state != "unknown":
                    self.state = "uncertain"
        # "unknown" state: stay unknown until a future recognition succeeds

    def _majority_vote(self) -> Tuple[Optional[int], int]:
        """Counts votes from the last TEMPORAL_FRAMES recognition events.

        Returns (winning_employee_id, vote_count).
        Only counts events with a non-None employee_id.
        Returns (None, 0) if no valid majority.
        """
        window = list(self.recognition_history)[-config.TEMPORAL_FRAMES:]
        votes: dict[int, int] = {}
        for ev in window:
            if ev.employee_id is not None:
                votes[ev.employee_id] = votes.get(ev.employee_id, 0) + 1

        if not votes:
            return None, 0

        winner = max(votes, key=lambda k: votes[k])
        return winner, votes[winner]

    def mark_unknown(self) -> None:
        """Explicitly marks this track as unknown (used by liveness gate)."""
        self.state = "unknown"
        self.confirmed_identity = None
        self.confirmed_name = None

    @property
    def is_confirmed(self) -> bool:
        """Legacy compatibility property — True when state is confirmed."""
        return self.state == "confirmed" and self.confirmed_identity is not None


class FaceTracker:
    """IoU-based multi-face tracker with per-track state machines."""

    def __init__(
        self,
        iou_threshold: Optional[float] = None,
        max_lost_frames: Optional[int] = None,
    ):
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
          1. Compute pairwise IoU matrix: existing tracks × new detections.
          2. Greedy match tracks to detections (highest IoU ≥ threshold first).
          3. Update matched tracks with new bbox, crop, and landmarks.
          4. Spawn new tracks for unmatched detections.
          5. Increment lost_frames for unmatched existing tracks.
          6. Purge tracks exceeding max_lost_frames.
          7. Increment frames_since_recognition for all surviving tracks.
        """
        if not self.tracks:
            for det in detections:
                self._spawn_track(det)
            self._tick_all_tracks()
            return self.tracks

        if not detections:
            for track in self.tracks:
                track.lost_frames += 1
                track.frames_since_recognition += 1
            self._purge_expired_tracks()
            return self.tracks

        num_tracks = len(self.tracks)
        num_dets = len(detections)
        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.compute_iou(track.bbox, det.bbox)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        flat_indices = np.argsort(iou_matrix.ravel())[::-1]
        for idx in flat_indices:
            ti = int(idx // num_dets)
            di = int(idx % num_dets)

            if iou_matrix[ti, di] < self.iou_threshold:
                break
            if ti in matched_tracks or di in matched_dets:
                continue

            matched_tracks.add(ti)
            matched_dets.add(di)

            # Update matched track
            track = self.tracks[ti]
            det = detections[di]
            track.bbox = det.bbox
            track.landmarks = det.landmarks
            track.face_crop = det.face_crop
            track.raw_face = det.raw_face
            track.lost_frames = 0

        # Spawn new tracks for unmatched detections
        for j in range(num_dets):
            if j not in matched_dets:
                self._spawn_track(detections[j])

        # Increment lost frames for unmatched tracks
        for i in range(num_tracks):
            if i not in matched_tracks:
                self.tracks[i].lost_frames += 1

        self._purge_expired_tracks()
        self._tick_all_tracks()
        return self.tracks

    def reset(self) -> None:
        """Resets all tracks."""
        self.tracks = []
        self._next_track_id = 1

    def _spawn_track(self, det: DetectedFace) -> None:
        track = Track(
            track_id=self._next_track_id,
            bbox=det.bbox,
            landmarks=det.landmarks,
            face_crop=det.face_crop,
            raw_face=det.raw_face,
            state="new",
        )
        self.tracks.append(track)
        self._next_track_id += 1

    def _purge_expired_tracks(self) -> None:
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost_frames]

    def _tick_all_tracks(self) -> None:
        """Increments the recognition frame counter for all alive tracks."""
        for t in self.tracks:
            t.frames_since_recognition += 1

    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Computes Intersection over Union between two [x1,y1,x2,y2] bboxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        a1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
        a2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
        union = a1 + a2 - inter
        return inter / union if union > 0.0 else 0.0
