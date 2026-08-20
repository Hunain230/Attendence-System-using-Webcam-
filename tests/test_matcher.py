"""
Unit tests for Phase 6 — FAISS Vector Index & ID Mapping (matcher.py)

Verifies:
  1. Index initialization and embedding dimension checks
  2. Adding single & multiple 512-D L2-normalized embeddings
  3. Nearest-neighbor search & cosine similarity correctness
  4. FAISS ID -> Employee ID mapping
  5. Similarity threshold acceptance & rejection logic
  6. Empty index safety
  7. Invalid embedding dimensions & malformed input handling
  8. Employee vector removal (rebuilding index)
  9. Index & ID map persistence and reload
  10. Integration with Phase 5 Database Repositories
"""

import pytest
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.recognition.matcher import FaceMatcher
from app.database.database import Base
from app.database.repository import EmployeeRepository, EmbeddingRepository
from app import config


@pytest.fixture
def tmp_matcher(tmp_path):
    """Provides an isolated FaceMatcher with temporary file storage."""
    idx_path = tmp_path / "test_faiss.index"
    map_path = tmp_path / "test_faiss_map.npy"
    return FaceMatcher(index_path=idx_path, id_map_path=map_path, auto_save=True)


@pytest.fixture
def sample_embedding():
    """Generates an L2-normalized 512-D vector."""
    rng = np.random.default_rng(42)
    vec = rng.normal(size=(512,)).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def db_session():
    """Provides an in-memory SQLite database session for integration tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


# ── 1. Index Initialization & Empty Index ─────────────────────


def test_index_initialization(tmp_matcher):
    assert tmp_matcher.dim == 512
    assert tmp_matcher.total_embeddings == 0
    assert tmp_matcher.id_map == []


def test_empty_index_search_returns_empty(tmp_matcher, sample_embedding):
    results = tmp_matcher.search(sample_embedding)
    assert results == []


# ── 2. Adding Embeddings & ID Mapping ─────────────────────────


def test_add_single_embedding(tmp_matcher, sample_embedding):
    faiss_id = tmp_matcher.add(sample_embedding, employee_id=101)

    assert faiss_id == 0
    assert tmp_matcher.total_embeddings == 1
    assert tmp_matcher.id_map[0] == 101


def test_add_multiple_embeddings(tmp_matcher, sample_embedding):
    rng = np.random.default_rng(100)
    vec2 = rng.normal(size=(512,)).astype(np.float32)
    vec2 = vec2 / np.linalg.norm(vec2)

    id0 = tmp_matcher.add(sample_embedding, employee_id=101)
    id1 = tmp_matcher.add(vec2, employee_id=102)

    assert id0 == 0
    assert id1 == 1
    assert tmp_matcher.total_embeddings == 2
    assert tmp_matcher.id_map == [101, 102]


def test_accepts_2d_shape(tmp_matcher, sample_embedding):
    # Shape (1, 512)
    vec_2d = sample_embedding.reshape(1, 512)
    faiss_id = tmp_matcher.add(vec_2d, employee_id=103)
    assert faiss_id == 0
    assert tmp_matcher.total_embeddings == 1


# ── 3. Cosine Similarity & Search Correctness ─────────────────


def test_nearest_neighbor_exact_match(tmp_matcher, sample_embedding):
    tmp_matcher.add(sample_embedding, employee_id=101)

    results = tmp_matcher.search(sample_embedding, top_k=1, threshold=0.0)

    assert len(results) == 1
    emp_id, score = results[0]
    assert emp_id == 101
    assert abs(score - 1.0) < 1e-4  # Perfect self-similarity = 1.0


def test_cosine_similarity_correctness(tmp_matcher):
    # Create two orthogonal vectors -> similarity ~0.0
    v1 = np.zeros((512,), dtype=np.float32)
    v1[0] = 1.0

    v2 = np.zeros((512,), dtype=np.float32)
    v2[1] = 1.0

    tmp_matcher.add(v1, employee_id=1)
    results = tmp_matcher.search(v2, threshold=-1.0)

    assert len(results) == 1
    assert abs(results[0][1] - 0.0) < 1e-5


# ── 4. Threshold Acceptance & Rejection ───────────────────────


def test_similarity_threshold_filtering(tmp_matcher):
    rng = np.random.default_rng(200)

    # Base vector
    base_v = rng.normal(size=(512,)).astype(np.float32)
    base_v = base_v / np.linalg.norm(base_v)

    # Dissimilar vector
    diff_v = rng.normal(size=(512,)).astype(np.float32)
    diff_v = diff_v / np.linalg.norm(diff_v)

    tmp_matcher.add(base_v, employee_id=201)
    tmp_matcher.add(diff_v, employee_id=202)

    # With high threshold (0.9), only base_v matches itself
    high_res = tmp_matcher.search(base_v, threshold=0.9)
    assert len(high_res) == 1
    assert high_res[0][0] == 201

    # Search with diff_v against base_v threshold=0.9 -> should be rejected
    rejected_res = tmp_matcher.search(diff_v, threshold=0.9)
    # diff_v self-matches 202, but does not match 201 above 0.9
    assert all(emp_id != 201 for emp_id, _ in rejected_res)


# ── 5. Invalid Input & Dimension Validation ───────────────────


def test_invalid_dimension_raises_error(tmp_matcher):
    wrong_dim = np.zeros((128,), dtype=np.float32)

    with pytest.raises(ValueError, match="Expected embedding dimension 512"):
        tmp_matcher.add(wrong_dim, employee_id=1)

    with pytest.raises(ValueError, match="Expected embedding dimension 512"):
        tmp_matcher.search(wrong_dim)


def test_non_numpy_raises_error(tmp_matcher):
    with pytest.raises(ValueError, match="Embedding must be a numpy.ndarray"):
        tmp_matcher.add([1.0] * 512, employee_id=1)


# ── 6. Employee Removal (Index Rebuilding) ────────────────────


def test_remove_employee_embeddings(tmp_matcher, sample_embedding):
    rng = np.random.default_rng(300)
    v2 = rng.normal(size=(512,)).astype(np.float32)
    v2 /= np.linalg.norm(v2)

    v3 = rng.normal(size=(512,)).astype(np.float32)
    v3 /= np.linalg.norm(v3)

    tmp_matcher.add(sample_embedding, employee_id=10)
    tmp_matcher.add(v2, employee_id=10)  # Second sample for employee 10
    tmp_matcher.add(v3, employee_id=20)  # Employee 20

    assert tmp_matcher.total_embeddings == 3

    removed = tmp_matcher.remove_employee(employee_id=10)
    assert removed == 2
    assert tmp_matcher.total_embeddings == 1
    assert tmp_matcher.id_map == [20]


# ── 7. Persistence & Reloading ────────────────────────────────


def test_persistence_and_reload(tmp_path, sample_embedding):
    idx_file = tmp_path / "persisted.index"
    map_file = tmp_path / "persisted_map.npy"

    # Create & populate matcher 1
    m1 = FaceMatcher(index_path=idx_file, id_map_path=map_file, auto_save=True)
    m1.add(sample_embedding, employee_id=301)
    assert m1.total_embeddings == 1

    # Instantiate matcher 2 pointing to same files
    m2 = FaceMatcher(index_path=idx_file, id_map_path=map_file, auto_save=True)
    assert m2.total_embeddings == 1
    assert m2.id_map == [301]

    # Verify search on reloaded index
    results = m2.search(sample_embedding, threshold=0.5)
    assert len(results) == 1
    assert results[0][0] == 301


# ── 8. Integration with Phase 5 Database Layer ────────────────


def test_database_mapping_integration(tmp_matcher, db_session, sample_embedding):
    # 1. Create employee in SQLite DB
    emp = EmployeeRepository.create(
        db_session, employee_code="E500", name="FAISS Integration User"
    )

    # 2. Add vector to FAISS
    faiss_id = tmp_matcher.add(sample_embedding, employee_id=emp.id)

    # 3. Store FAISS ID mapping record in SQLite
    EmbeddingRepository.add(
        db_session, employee_id=emp.id, faiss_id=faiss_id, quality_score=0.95
    )

    # 4. Search FAISS index
    search_res = tmp_matcher.search(sample_embedding, top_k=1, threshold=0.5)
    assert len(search_res) == 1
    matched_emp_id, sim_score = search_res[0]

    # 5. Verify database lookup via mapped employee_id
    db_emp = EmployeeRepository.get_by_id(db_session, matched_emp_id)
    assert db_emp is not None
    assert db_emp.employee_code == "E500"
    assert db_emp.name == "FAISS Integration User"

    # 6. Verify SQLite embedding record lookup by faiss_id
    db_emb_record = EmbeddingRepository.get_by_faiss_id(db_session, faiss_id)
    assert db_emb_record is not None
    assert db_emb_record.employee_id == emp.id
