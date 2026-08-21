"""
Face Detector — SCRFD Wrapper

Runs face detection via InsightFace det_500m.onnx (buffalo_s).
Returns detected faces with bounding box, score, landmarks, and face crop.
"""

from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
import insightface
from insightface.model_zoo import model_zoo

from app import config


@dataclass
class DetectedFace:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    landmarks: np.ndarray  # (5, 2)
    det_score: float  # 0.0 - 1.0
    face_crop: np.ndarray  # Image crop
    raw_face: object  # InsightFace Face object for alignment/embedding


class FaceDetector:
    """SCRFD face detector."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.INSIGHTFACE_MODEL
        model_dir = Path.home() / ".insightface" / "models" / self.model_name
        det_files = list(model_dir.glob("det_*.onnx")) if model_dir.exists() else []

        if det_files:
            self.detector = model_zoo.get_model(
                str(det_files[0]), providers=["CPUExecutionProvider"]
            )
            self.detector.prepare(ctx_id=0, input_size=config.DETECTION_SIZE)
            self._use_zoo = True
        else:
            self.app = insightface.app.FaceAnalysis(
                name=self.model_name,
                allowed_modules=["detection"],
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=config.DETECTION_SIZE)
            self._use_zoo = False

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """Detect faces in a 3-channel BGR frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        results: List[DetectedFace] = []

        if self._use_zoo:
            bboxes, kpss = self.detector.detect(
                frame, max_num=0, metric="default"
            )
            if bboxes is None or len(bboxes) == 0:
                return []

            for i in range(len(bboxes)):
                box = bboxes[i][:4]
                score = float(bboxes[i][4])
                if score < config.DETECTION_SCORE_THRESHOLD:
                    continue
                kps = kpss[i]
                x1, y1, x2, y2 = (
                    max(0, int(box[0])),
                    max(0, int(box[1])),
                    min(frame.shape[1], int(box[2])),
                    min(frame.shape[0], int(box[3])),
                )
                crop = frame[y1:y2, x1:x2]

                class DummyFace:
                    pass

                face_obj = DummyFace()
                face_obj.bbox = box
                face_obj.kps = kps
                face_obj.det_score = score

                results.append(
                    DetectedFace(
                        bbox=box,
                        landmarks=kps,
                        det_score=score,
                        face_crop=crop,
                        raw_face=face_obj,
                    )
                )
        else:
            faces = self.app.get(frame)
            for f in faces:
                score = float(f.det_score)
                if score < config.DETECTION_SCORE_THRESHOLD:
                    continue
                box = f.bbox
                x1, y1, x2, y2 = (
                    max(0, int(box[0])),
                    max(0, int(box[1])),
                    min(frame.shape[1], int(box[2])),
                    min(frame.shape[0], int(box[3])),
                )
                crop = frame[y1:y2, x1:x2]
                results.append(
                    DetectedFace(
                        bbox=box,
                        landmarks=f.kps,
                        det_score=score,
                        face_crop=crop,
                        raw_face=f,
                    )
                )

        return results
