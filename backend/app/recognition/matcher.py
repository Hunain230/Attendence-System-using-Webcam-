"""
Face Matcher — FAISS Search & ID Mapping Layer

Manages FAISS IndexFlatIP vector index for cosine similarity search over
ArcFace 512-D L2-normalized face embeddings.
Maintains mapping between FAISS vector IDs (index positions) and Employee IDs.

Key Bug Fix vs Previous Version
────────────────────────────────
  Previous: margin check was guarded by `top_score < 0.65`, meaning any match
  with similarity >= 0.65 bypassed the margin check entirely. This caused
  Hunain/Sobia confusion when both had high but close scores.

  Fixed: margin is checked unconditionally. A high-confidence but ambiguous
  match is still rejected — the system prefers UNKNOWN over WRONG PERSON.

Employee-Level Scoring
──────────────────────
  FAISS returns individual vector similarities. Each employee may have 5–12
  enrolled vectors. We group by employee and compute:
    - max similarity (best matching vector)
    - mean of top-2 vectors (more robust than max alone)
  The mean-of-top-2 is used for final ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss

from app import config


@dataclass
class MatchResult:
    """Detailed result from a FAISS search with margin analysis.

    Used by the diagnostic overlay and recognition engine.
    """
    best_employee_id: Optional[int]
    best_score: float
    second_employee_id: Optional[int]
    second_score: float
    margin: float
    accepted: bool                        # True if accepted (above threshold AND margin)
    all_employee_scores: Dict[int, float] = None  # Full per-employee scores for overlay


class FaceMatcher:
    """FAISS-based vector index manager for face embeddings."""

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        similarity_margin: Optional[float] = None,
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
        self.similarity_margin = (
            similarity_margin
            if similarity_margin is not None
            else config.SIMILARITY_MARGIN
        )
        self.index_path = Path(index_path or config.FAISS_INDEX_PATH)
        self.id_map_path = Path(id_map_path or config.FAISS_ID_MAP_PATH)
        self.auto_save = auto_save

        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map: List[int] = []  # FAISS index position → employee_id

        self._load_if_exists()

    @property
    def total_embeddings(self) -> int:
        """Returns total number of embeddings in the index."""
        return self.index.ntotal

    @property
    def employee_ids(self) -> List[int]:
        """Returns the list of unique employee IDs in the index."""
        return list(set(self.id_map))

    def add(self, embedding: np.ndarray, employee_id: int) -> int:
        """Adds an L2-normalized embedding vector to the FAISS index.

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
        margin: Optional[float] = None,
    ) -> List[Tuple[int, float]]:
        """Searches for nearest neighbor candidates with threshold + margin check.

        Returns:
            List of (employee_id, similarity_score) tuples sorted by score.
            Returns [] if no match exceeds threshold, or if margin check fails.
        """
        result = self.search_detailed(embedding, top_k=top_k, threshold=threshold, margin=margin)
        if not result.accepted or result.best_employee_id is None:
            return []
        return [(result.best_employee_id, result.best_score)]

    def search_detailed(
        self,
        embedding: np.ndarray,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> MatchResult:
        """Full detailed search returning per-employee scores, margin, and accept decision.

        This is used by the recognition engine for both recognition logic and
        the debug diagnostic overlay.

        Scoring method:
          - Retrieve top_k vectors from FAISS.
          - Group by employee.
          - For each employee: compute score = mean of their top-2 vector similarities.
            (More robust than max alone — prevents a single good outlier vector dominating.)
          - Rank employees by this score.
          - Check: best_score >= threshold AND (best_score - second_score) >= margin.
          - The margin check is UNCONDITIONAL. No bypass for high scores.

        Returns:
            MatchResult with full details.
        """
        query = self._validate_and_format_embedding(embedding)

        empty_result = MatchResult(
            best_employee_id=None,
            best_score=0.0,
            second_employee_id=None,
            second_score=0.0,
            margin=0.0,
            accepted=False,
            all_employee_scores={},
        )

        if self.index.ntotal == 0:
            return empty_result

        k = min(top_k or config.FAISS_TOP_K, self.index.ntotal)
        min_score = threshold if threshold is not None else self.similarity_threshold
        min_margin = margin if margin is not None else self.similarity_margin

        scores, indices = self.index.search(query, k)

        # Group raw vector similarities by employee
        emp_vectors: Dict[int, List[float]] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            emp_id = self.id_map[idx]
            emp_vectors.setdefault(emp_id, []).append(float(score))

        if not emp_vectors:
            return empty_result

        # Compute per-employee score = mean of top-2 vectors
        emp_scores: Dict[int, float] = {}
        for emp_id, vec_scores in emp_vectors.items():
            top2 = sorted(vec_scores, reverse=True)[:2]
            emp_scores[emp_id] = float(np.mean(top2))

        # Rank employees
        ranked = sorted(emp_scores.items(), key=lambda x: x[1], reverse=True)

        best_id, best_score = ranked[0]
        second_id = ranked[1][0] if len(ranked) >= 2 else None
        second_score = ranked[1][1] if len(ranked) >= 2 else 0.0
        computed_margin = best_score - second_score if second_id is not None else 1.0

        # Threshold check
        if best_score < min_score:
            return MatchResult(
                best_employee_id=best_id,
                best_score=best_score,
                second_employee_id=second_id,
                second_score=second_score,
                margin=computed_margin,
                accepted=False,
                all_employee_scores=emp_scores,
            )

        # Margin check — UNCONDITIONAL (bug fix: was bypassed when top_score >= 0.65)
        if second_id is not None and computed_margin < min_margin:
            return MatchResult(
                best_employee_id=best_id,
                best_score=best_score,
                second_employee_id=second_id,
                second_score=second_score,
                margin=computed_margin,
                accepted=False,   # Ambiguous — prefer UNKNOWN over WRONG PERSON
                all_employee_scores=emp_scores,
            )

        return MatchResult(
            best_employee_id=best_id,
            best_score=best_score,
            second_employee_id=second_id,
            second_score=second_score,
            margin=computed_margin,
            accepted=True,
            all_employee_scores=emp_scores,
        )

    def get_all_vectors_for_employee(self, employee_id: int) -> np.ndarray:
        """Retrieves all stored embedding vectors for a specific employee.

        Used by the calibration tool to compute genuine similarity distributions.

        Returns:
            Array of shape (N, embedding_dim) or empty array if none found.
        """
        if self.index.ntotal == 0:
            return np.empty((0, self.dim), dtype=np.float32)

        indices = [i for i, eid in enumerate(self.id_map) if eid == employee_id]
        if not indices:
            return np.empty((0, self.dim), dtype=np.float32)

        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        return all_vectors[indices].astype(np.float32)

    def remove_employee(self, employee_id: int) -> int:
        """Removes all embeddings for a specific employee by rebuilding the index.

        Returns:
            Count of removed embeddings.
        """
        if self.index.ntotal == 0 or employee_id not in self.id_map:
            return 0

        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        keep_mask = [eid != employee_id for eid in self.id_map]
        new_vectors = all_vectors[[i for i, k in enumerate(keep_mask) if k]]
        new_id_map = [eid for eid in self.id_map if eid != employee_id]

        removed_count = len(self.id_map) - len(new_id_map)

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

    def reload(self) -> bool:
        """Reloads the FAISS index and ID map from disk."""
        return self._load_if_exists()

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
                    self.id_map = [int(x) for x in loaded_map]
                    return True
            except Exception:
                pass
        return False

    def _validate_and_format_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Validates input embedding shape and ensures L2-normalization."""
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be a numpy.ndarray")

        arr = embedding.squeeze()
        if arr.ndim != 1 or arr.shape[0] != self.dim:
            raise ValueError(
                f"Expected embedding dimension {self.dim}, got shape {embedding.shape}"
            )

        norm = float(np.linalg.norm(arr))
        if abs(norm - 1.0) > 1e-3 and norm > 0:
            arr = arr / norm

        return arr.reshape(1, self.dim).astype(np.float32)
