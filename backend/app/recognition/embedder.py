"""
Face Embedder — ArcFace Wrapper

Extracts L2-normalized 512-D embeddings from 5-point landmark aligned face crops.
Uses w600k_r50.onnx (buffalo_l) or w600k_mbf.onnx (buffalo_s).
"""

from typing import Optional
from pathlib import Path
import numpy as np
import cv2
import insightface
from insightface.model_zoo import model_zoo
from insightface.utils import face_align

from app import config


class FaceEmbedder:
    """ArcFace feature embedding extractor with 5-point facial landmark alignment."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.INSIGHTFACE_MODEL
        model_dir = Path.home() / ".insightface" / "models" / self.model_name
        rec_files = (
            list(model_dir.glob("w600k_*.onnx"))
            or list(model_dir.glob("glintr*.onnx"))
            or list(model_dir.glob("*.onnx"))
            if model_dir.exists()
            else []
        )

        if rec_files:
            self.recognizer = model_zoo.get_model(
                str(rec_files[0]), providers=["CPUExecutionProvider"]
            )
            self.recognizer.prepare(ctx_id=0)
            self._use_zoo = True
        else:
            self.app = insightface.app.FaceAnalysis(
                name=self.model_name,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=config.DETECTION_SIZE)
            self.recognizer = self.app.models.get("recognition", None)
            self._use_zoo = False

    def embed(
        self,
        frame: np.ndarray,
        face_obj: Optional[object] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extracts 512-D L2-normalized embedding vector using landmark affine alignment.

        Args:
            frame: Full BGR frame or 112x112 aligned image.
            face_obj: Optional DetectedFace or InsightFace Face object containing landmarks.
            landmarks: Optional 5x2 facial landmarks array.

        Returns:
            512-D L2-normalized float32 numpy array.
        """
        kps = None
        if landmarks is not None and isinstance(landmarks, np.ndarray) and landmarks.shape == (5, 2):
            kps = landmarks
        elif face_obj is not None:
            if hasattr(face_obj, "landmarks") and isinstance(face_obj.landmarks, np.ndarray) and face_obj.landmarks.shape == (5, 2):
                kps = face_obj.landmarks
            elif hasattr(face_obj, "kps") and isinstance(face_obj.kps, np.ndarray) and face_obj.kps.shape == (5, 2):
                kps = face_obj.kps
            elif hasattr(face_obj, "raw_face") and face_obj.raw_face is not None:
                raw = face_obj.raw_face
                if hasattr(raw, "kps") and isinstance(raw.kps, np.ndarray) and raw.kps.shape == (5, 2):
                    kps = raw.kps

        if kps is not None and frame is not None and isinstance(frame, np.ndarray) and frame.ndim == 3:
            # High-precision 5-point landmark similarity transform alignment
            aligned = face_align.norm_crop(frame, landmark=kps, image_size=112)
        elif isinstance(frame, np.ndarray) and frame.shape == (112, 112, 3):
            aligned = frame
        else:
            if (
                face_obj is not None
                and hasattr(face_obj, "face_crop")
                and face_obj.face_crop is not None
                and face_obj.face_crop.size > 0
            ):
                aligned = cv2.resize(face_obj.face_crop, (112, 112))
            elif frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                aligned = cv2.resize(frame, (112, 112))
            else:
                raise ValueError("Cannot extract embedding: invalid frame or face object")

        if self._use_zoo:
            feat = self.recognizer.get_feat(aligned)
        else:
            feat = self.recognizer.get(aligned)

        embedding = np.array(feat, dtype=np.float32).flatten()

        # L2-normalization
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm

        if embedding.shape[0] != config.EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding dimension {config.EMBEDDING_DIM}, got {embedding.shape[0]}"
            )

        return embedding
