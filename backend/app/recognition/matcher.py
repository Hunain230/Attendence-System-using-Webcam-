"""
Face Matcher — FAISS Search & ID Mapping Layer

Manages FAISS IndexFlatIP vector index for exact cosine similarity search over
ArcFace 512-D L2-normalized face embeddings.
Maintains mapping between FAISS vector IDs (index positions) and Employee IDs.
Does NOT import model inference tools (SCRFD/ArcFace) or access SQLite directly.
"""

from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import faiss

from app import config


class FaceMatcher:
    """FAISS-based vector index manager for face embeddings."""

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        index_path: Optional[Path] = None,
        id_map_path: Optional[Path] = None,
        auto_save: bool = True,
    ):
        self.dim = embedding_dim or config.EMBEDDING_DIM
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else config.SIMILARITY_THRESHOLD
        )
        self.index_path = Path(index_path or config.FAISS_INDEX_PATH)
        self.id_map_path = Path(id_map_path or config.FAISS_ID_MAP_PATH)
        self.auto_save = auto_save

        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map: List[int] = []  # FAISS index position -> employee_id

        self._load_if_exists()

    @property
    def total_embeddings(self) -> int:
        """Returns total number of embeddings in the index."""
        return self.index.ntotal

    def add(self, embedding: np.ndarray, employee_id: int) -> int:
        """Adds an L2-normalized embedding vector to the FAISS index.

        Args:
            embedding: 1D (512,) or 2D (1, 512) numpy float32 array.
            employee_id: Master employee ID from database.

        Returns:
            faiss_id: The integer index position in FAISS.
        """
        query = self._validate_and_format_embedding(embedding)

        faiss_id = self.index.ntotal
        self.index.add(query)
        self.id_map.append(employee_id)

        if self.auto_save:
            self.save()

        return faiss_id

    def search(
        self,
        embedding: np.ndarray,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, float]]:
        """Searches for nearest neighbor candidates for the query embedding.

        Args:
            embedding: 1D (512,) or 2D (1, 512) numpy float32 array.
            top_k: Number of nearest neighbors to retrieve.
            threshold: Minimum similarity score threshold (overrides config if provided).

        Returns:
            List of (employee_id, similarity_score) tuples sorted by similarity.
        """
        query = self._validate_and_format_embedding(embedding)

        if self.index.ntotal == 0:
            return []

        k = min(top_k or config.FAISS_TOP_K, self.index.ntotal)
        min_score = (
            threshold if threshold is not None else self.similarity_threshold
        )

        scores, indices = self.index.search(query, k)

        results: List[Tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            sim_score = float(score)
            if sim_score >= min_score:
                results.append((self.id_map[idx], sim_score))

        return results

    def remove_employee(self, employee_id: int) -> int:
        """Removes all embeddings for a specific employee by rebuilding the index.

        Args:
            employee_id: Employee ID to remove.

        Returns:
            Count of removed embeddings.
        """
        if self.index.ntotal == 0 or employee_id not in self.id_map:
            return 0

        # Extract all existing vectors
        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)

        # Filter out vectors matching employee_id
        keep_mask = [eid != employee_id for eid in self.id_map]
        new_vectors = all_vectors[keep_mask]
        new_id_map = [eid for eid in self.id_map if eid != employee_id]

        removed_count = len(self.id_map) - len(new_id_map)

        # Reset FAISS index
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map = []

        if len(new_vectors) > 0:
            self.index.add(new_vectors.astype(np.float32))
            self.id_map = new_id_map

        if self.auto_save:
            self.save()

        return removed_count

    def clear(self) -> None:
        """Resets and clears the index and ID mapping."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map = []
        if self.auto_save:
            self.save()

    def save(self) -> None:
        """Persists the FAISS index and ID map to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.id_map_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))
        np.save(str(self.id_map_path), np.array(self.id_map, dtype=np.int64))

    def _load_if_exists(self) -> bool:
        """Loads index and ID map from disk if both files exist."""
        if self.index_path.exists() and self.id_map_path.exists():
            try:
                loaded_index = faiss.read_index(str(self.index_path))
                loaded_map = np.load(str(self.id_map_path)).tolist()

                if (
                    loaded_index.d == self.dim
                    and loaded_index.ntotal == len(loaded_map)
                ):
                    self.index = loaded_index
                    self.id_map = loaded_map
                    return True
            except Exception:
                pass
        return False

    def _validate_and_format_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Validates input embedding shape and L2-normalization."""
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be a numpy.ndarray")

        arr = embedding.squeeze()
        if arr.ndim != 1 or arr.shape[0] != self.dim:
            raise ValueError(
                f"Expected embedding dimension {self.dim}, got shape {embedding.shape}"
            )

        norm = float(np.linalg.norm(arr))
        if abs(norm - 1.0) > 1e-3 and norm > 0:
            # Normalize if slightly unnormalized
            arr = arr / norm

        return arr.reshape(1, self.dim).astype(np.float32)
