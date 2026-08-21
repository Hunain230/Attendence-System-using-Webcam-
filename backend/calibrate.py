#!/usr/bin/env python
"""
Similarity Calibration Tool

Analyzes the enrolled FAISS database to compute:
  - Per-employee genuine similarity distribution (same-person vectors)
  - Per-employee impostor similarity distribution (cross-person vectors)
  - Complete N×N similarity matrix
  - Recommended SIMILARITY_THRESHOLD and SIMILARITY_MARGIN

Usage:
    # From the project root with virtualenv activated:
    python backend/calibrate.py

    # With a specific model or threshold to test:
    INSIGHTFACE_MODEL=buffalo_l python backend/calibrate.py

Output:
    - Console report with per-employee statistics
    - Similarity matrix
    - Threshold and margin recommendations
    - Optionally writes backend/data/calibration_report.txt

Why This Matters:
    The default SIMILARITY_THRESHOLD=0.52 and SIMILARITY_MARGIN=0.12 are
    conservative starting points. The right values depend on your enrolled faces,
    lighting conditions, and camera quality.

    The calibration tool computes:
        recommended_threshold = min_genuine_mean - safety_gap
        recommended_margin    = (min_genuine_mean - max_impostor_mean) * 0.4

    If the recommended threshold differs significantly from config defaults,
    set it via environment variable:
        SIMILARITY_THRESHOLD=0.XX python -m uvicorn app.main:app ...
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Script lives at backend/calibrate.py → backend/ is one level up
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))  # backend/ → allows `from app import config`

from app import config
from app.database.database import SessionLocal, init_db
from app.database.repository import EmployeeRepository, EmbeddingRepository
from app.recognition.matcher import FaceMatcher

import numpy as np

# ── ANSI colors for terminal output ──────────────────────────────────────────
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

def _h(text: str) -> str:
    return f"{_BOLD}{_CYAN}{text}{_RESET}"

def _ok(text: str) -> str:
    return f"{_GREEN}{text}{_RESET}"

def _warn(text: str) -> str:
    return f"{_YELLOW}{text}{_RESET}"

def _err(text: str) -> str:
    return f"{_RED}{text}{_RESET}"


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    v1 = v1.flatten().astype(np.float64)
    v2 = v2.flatten().astype(np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1 / n1, v2 / n2))


def run_calibration() -> None:
    print()
    print(_h("=" * 60))
    print(_h("  FACE RECOGNITION SIMILARITY CALIBRATION TOOL"))
    print(_h("=" * 60))
    print(f"  Model:      {config.INSIGHTFACE_MODEL}")
    print(f"  FAISS:      {config.FAISS_INDEX_PATH}")
    print(f"  Config threshold: {config.SIMILARITY_THRESHOLD}")
    print(f"  Config margin:    {config.SIMILARITY_MARGIN}")
    print()

    # ── Load FAISS and database ───────────────────────────────────────────────
    init_db()
    matcher = FaceMatcher(auto_save=False)

    if matcher.total_embeddings == 0:
        print(_err("  ERROR: FAISS index is empty. Enroll at least 2 employees first."))
        sys.exit(1)

    db = SessionLocal()
    try:
        employees = EmployeeRepository.get_all(db)
    finally:
        db.close()

    if len(employees) < 2:
        print(_warn("  WARNING: At least 2 employees are required to compute impostor similarity."))
        if len(employees) == 0:
            print(_err("  No employees enrolled. Exiting."))
            sys.exit(1)

    # ── Collect vectors per employee ──────────────────────────────────────────
    emp_vectors: dict[int, np.ndarray] = {}
    emp_names: dict[int, str] = {}

    for emp in employees:
        vecs = matcher.get_all_vectors_for_employee(emp.id)
        if len(vecs) == 0:
            print(_warn(f"  WARNING: No vectors found in FAISS for {emp.name} (id={emp.id}). Skipping."))
            continue
        emp_vectors[emp.id] = vecs
        emp_names[emp.id] = emp.name
        print(f"  {emp.name:<20} : {len(vecs)} enrolled vectors (id={emp.id})")

    enrolled_ids = list(emp_vectors.keys())

    if len(enrolled_ids) < 2:
        print(_err("  Cannot compute impostor scores with fewer than 2 enrolled employees."))
        sys.exit(1)

    print()
    print(_h("─" * 60))
    print(_h("  PER-EMPLOYEE ANALYSIS"))
    print(_h("─" * 60))

    per_emp_stats: dict[int, dict] = {}
    report_lines: list[str] = []

    for eid in enrolled_ids:
        name = emp_names[eid]
        own_vecs = emp_vectors[eid]

        # ── Genuine similarity (same-person pairs) ────────────────────────────
        genuine_scores: list[float] = []
        for i in range(len(own_vecs)):
            for j in range(i + 1, len(own_vecs)):
                s = compute_cosine_similarity(own_vecs[i], own_vecs[j])
                genuine_scores.append(s)

        # ── Impostor similarity (this employee vs all others) ─────────────────
        impostor_scores: list[float] = []
        for other_eid in enrolled_ids:
            if other_eid == eid:
                continue
            other_vecs = emp_vectors[other_eid]
            for ov in other_vecs:
                for sv in own_vecs:
                    s = compute_cosine_similarity(sv, ov)
                    impostor_scores.append(s)

        gen_mean = float(np.mean(genuine_scores)) if genuine_scores else float("nan")
        gen_min  = float(np.min(genuine_scores))  if genuine_scores else float("nan")
        gen_max  = float(np.max(genuine_scores))  if genuine_scores else float("nan")

        imp_mean = float(np.mean(impostor_scores)) if impostor_scores else float("nan")
        imp_max  = float(np.max(impostor_scores))  if impostor_scores else float("nan")

        # Recommended threshold = halfway between min genuine and max impostor
        # (Conservative: we bias toward rejecting ambiguous matches)
        if not np.isnan(gen_min) and not np.isnan(imp_max) and gen_min > imp_max:
            rec_thresh = (gen_min + imp_max) / 2.0
            rec_margin = (gen_min - imp_max) * 0.4
            verdict = _ok("SEPARABLE")
        elif not np.isnan(gen_min) and not np.isnan(imp_max):
            rec_thresh = gen_mean - 0.05
            rec_margin = 0.08
            verdict = _warn("OVERLAPPING (tighten or re-enroll)")
        else:
            rec_thresh = config.SIMILARITY_THRESHOLD
            rec_margin = config.SIMILARITY_MARGIN
            verdict = _warn("INSUFFICIENT DATA")

        per_emp_stats[eid] = {
            "name": name,
            "gen_mean": gen_mean, "gen_min": gen_min, "gen_max": gen_max,
            "imp_mean": imp_mean, "imp_max": imp_max,
            "rec_thresh": rec_thresh, "rec_margin": rec_margin,
            "verdict": verdict,
        }

        lines = [
            f"\n  Employee: {_bold(name)} (id={eid})",
            f"    Genuine pairs:  {len(genuine_scores)}",
            f"    Genuine mean:   {_fmt(gen_mean)}",
            f"    Genuine min:    {_fmt(gen_min)}",
            f"    Genuine max:    {_fmt(gen_max)}",
            f"    Impostor pairs: {len(impostor_scores)}",
            f"    Impostor mean:  {_fmt(imp_mean)}",
            f"    Impostor max:   {_fmt(imp_max, red_above=0.45)}",
            f"    Verdict:        {verdict}",
            f"    Rec. threshold: {rec_thresh:.4f}",
            f"    Rec. margin:    {rec_margin:.4f}",
        ]
        for line in lines:
            print(line)
            report_lines.append(line)

    # ── Full N×N Similarity Matrix ────────────────────────────────────────────
    print()
    print(_h("─" * 60))
    print(_h("  SIMILARITY MATRIX (max similarity between any pair of vectors)"))
    print(_h("─" * 60))
    report_lines.append("\nSIMILARITY MATRIX")

    # Header
    col_w = 12
    header = " " * 22 + "".join(f"{emp_names[eid][:col_w]:^{col_w}}" for eid in enrolled_ids)
    print(f"  {header}")
    report_lines.append(header)

    for row_eid in enrolled_ids:
        row = f"  {emp_names[row_eid][:20]:<20} "
        for col_eid in enrolled_ids:
            if row_eid == col_eid:
                # Diagonal: mean genuine similarity
                vecs = emp_vectors[row_eid]
                if len(vecs) >= 2:
                    sims = [
                        compute_cosine_similarity(vecs[i], vecs[j])
                        for i in range(len(vecs))
                        for j in range(i + 1, len(vecs))
                    ]
                    val = float(np.mean(sims))
                else:
                    val = 1.0
                cell = _ok(f"{val:.3f}")
            else:
                rv = emp_vectors[row_eid]
                cv = emp_vectors[col_eid]
                max_sim = max(
                    compute_cosine_similarity(r, c)
                    for r in rv for c in cv
                )
                cell_str = f"{max_sim:.3f}"
                cell = _warn(cell_str) if max_sim > 0.40 else cell_str
            row += f"{_strip_ansi(cell):^{col_w}}"
        print(row)
        report_lines.append(row)

    # ── Global Recommendations ────────────────────────────────────────────────
    all_rec_thresh = [s["rec_thresh"] for s in per_emp_stats.values() if not np.isnan(s["rec_thresh"])]
    all_rec_margin = [s["rec_margin"] for s in per_emp_stats.values() if not np.isnan(s["rec_margin"])]

    global_thresh = min(all_rec_thresh) if all_rec_thresh else config.SIMILARITY_THRESHOLD
    global_margin = min(all_rec_margin) if all_rec_margin else config.SIMILARITY_MARGIN

    print()
    print(_h("─" * 60))
    print(_h("  GLOBAL RECOMMENDATIONS"))
    print(_h("─" * 60))
    print(f"\n  Recommended SIMILARITY_THRESHOLD = {_ok(f'{global_thresh:.4f}')}")
    print(f"  Recommended SIMILARITY_MARGIN    = {_ok(f'{global_margin:.4f}')}")
    print(f"\n  Current config values:")
    print(f"    SIMILARITY_THRESHOLD = {config.SIMILARITY_THRESHOLD}")
    print(f"    SIMILARITY_MARGIN    = {config.SIMILARITY_MARGIN}")

    if abs(global_thresh - config.SIMILARITY_THRESHOLD) > 0.03:
        print()
        print(_warn("  ⚠  Threshold differs significantly from recommended value."))
        print(f"     Set:  SIMILARITY_THRESHOLD={global_thresh:.4f}  in your environment or config.py")

    if any(np.isnan(s["gen_min"]) for s in per_emp_stats.values()):
        print()
        print(_warn("  ⚠  Some employees have only 1 enrolled vector — cannot compute genuine distribution."))
        print("     Re-enroll for more robust calibration.")

    # ── Write report to file ──────────────────────────────────────────────────
    report_path = config.DATA_DIR / "calibration_report.txt"
    try:
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"\n  Report saved → {report_path}")
    except Exception as e:
        print(f"\n  Could not write report: {e}")

    print()
    print(_h("=" * 60))
    print()


def _fmt(val: float, red_above: float = 1.0) -> str:
    """Formats a float similarity score with color coding."""
    if np.isnan(val):
        return "—"
    s = f"{val:.4f}"
    if val >= 0.5:
        return _ok(s)
    if val >= red_above:
        return _warn(s)
    return s


def _bold(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}"


def _strip_ansi(text: str) -> str:
    """Strips ANSI codes for plain-text alignment."""
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)


if __name__ == "__main__":
    run_calibration()
