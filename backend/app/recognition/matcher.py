"""
Face Matcher — FAISS Search + Threshold Logic

Manages the FAISS IndexFlatIP index.
Searches L2-normalized embeddings via inner product (= cosine similarity).
Applies SIMILARITY_THRESHOLD to filter candidates.
"""
