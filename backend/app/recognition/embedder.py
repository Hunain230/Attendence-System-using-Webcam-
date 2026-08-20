"""
Face Embedder — ArcFace Wrapper

Extracts L2-normalized 512-D embeddings from aligned face crops using w600k_mbf.onnx (buffalo_s).
"""

from typing import Optional
from pathlib import Path
import numpy as np
import cv2
import insightface
from insightface.model_zoo import model_zoo

from app import config


class FaceEmbedder:
    """ArcFace feature embedding extractor."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.INSIGHTFACE_MODEL
        model_path = (
            Path.home()
            / ".insightface"
            / "models"
            / self.model_name
            / "w600k_mbf.onnx"
        )
        if model_path.exists():
            self.recognizer = model_zoo.get_model(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self.recognizer.prepare(ctx_id=0)
            self._use_zoo = True
        else:
            app = insightface.app.FaceAnalysis(
                name=self.model_name,
                allowed_modules=["recognition"],
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0)
            self.recognizer = app.models.get("recognition", None)
            self._use_zoo = False

    def embed(
        self, frame: np.ndarray, face_obj: Optional[object] = None
    ) -> np.ndarray:
        """Extracts 512-D L2-normalized embedding vector.

        Supports passing either a cropped 112x112 image, a DetectedFace, or an InsightFace Face object.
        """
        if isinstance(frame, np.ndarray) and frame.shape == (112, 112, 3):
            # Direct feature extraction on 112x112 crop
            feat = (
                self.recognizer.get_feat(frame)
                if self._use_zoo
                else self.recognizer.get(frame)
            )
        elif not self._use_zoo and hasattr(self.recognizer, "get") and face_obj is not None:
            feat = self.recognizer.get(frame, face_obj)
        else:
            if (
                face_obj is not None
                and hasattr(face_obj, "face_crop")
                and face_obj.face_crop is not None
                and face_obj.face_crop.size > 0
            ):
                resized = cv2.resize(face_obj.face_crop, (112, 112))
            else:
                resized = cv2.resize(frame, (112, 112))
            feat = self.recognizer.get_feat(resized)

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
