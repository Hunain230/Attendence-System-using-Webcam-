"""
SQLAlchemy ORM Models

Tables:
  - employees        (id, employee_code, name, department, created_at, active)
  - face_embeddings  (id, employee_id, faiss_id, quality_score, created_at)
  - attendance       (id, employee_id, date, check_in, check_out, confidence, created_at)
"""
