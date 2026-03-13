import numpy as np
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "researchers.json")) as f:
    RESEARCHERS = json.load(f)["researchers"]


def get_query_embedding(query: str) -> np.ndarray:
    q = query.lower()
    emb = np.array([
        1.0 if any(k in q for k in ["federated", "privacy", "distributed", "secure"]) else 0.1,
        1.0 if any(k in q for k in ["genomic", "omics", "genome", "rare disease", "exome", "rna"]) else 0.1,
        1.0 if any(k in q for k in ["quantum", "qaoa", "qubo", "annealing"]) else 0.1,
        1.0 if any(k in q for k in ["optimization", "drug", "molecular", "protein"]) else 0.1,
        1.0 if any(k in q for k in ["cancer", "tumor", "detection", "medical", "imaging"]) else 0.1,
        1.0 if any(k in q for k in ["nlp", "clinical", "text", "language", "notes", "ehr"]) else 0.1,
        1.0 if any(k in q for k in ["feature", "selection", "high-dimensional", "biomedical"]) else 0.1,
        1.0 if any(k in q for k in ["mri", "ct", "scan", "brain", "neuro", "fmri"]) else 0.1,
    ], dtype=float)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def rag_search(query: str, top_k: int = 6) -> list:
    query_emb = get_query_embedding(query)
    scored = []
    for r in RESEARCHERS:
        r_emb = np.array(r["embedding"], dtype=float)
        r_emb = r_emb / (np.linalg.norm(r_emb) or 1)
        sim = cosine_similarity(query_emb, r_emb)
        scored.append((r, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:top_k]]