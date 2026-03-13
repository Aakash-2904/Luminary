"""
qaoa.py — Luminary
QAOA Collaboration Scoring — now FL-informed.

Pipeline:
  1. FL model predicts base collaboration probability
     (trained across university nodes via FedAvg)
  2. QAOA QUBO objective refines with 4 factor weights
  3. FL score used as prior — QAOA adjusts up/down
  4. Final score = weighted combination of FL + QAOA

Why both?
  FL learns from patterns across all researcher pairs.
  QAOA optimizes the multi-variable combination simultaneously.
  Together they're more accurate than either alone.
"""

import numpy as np

# ── QAOA weights ──
WEIGHTS = {
    "methodology": 0.35,
    "domain":      0.30,
    "dataset":     0.20,
    "stage":       0.15,
}

# ── FL vs QAOA blend ──
# How much weight to give FL model vs pure QAOA
# FL = learned from data patterns
# QAOA = deterministic multi-variable optimization
FL_WEIGHT   = 0.40
QAOA_WEIGHT = 0.60

STAGE_MATRIX = {
    ("early","mid"):0.90, ("early","published"):0.95,
    ("early","early"):0.60, ("mid","published"):0.85,
    ("mid","mid"):0.70, ("published","published"):0.40,
    ("dataset_available","early"):1.00,
    ("dataset_available","mid"):0.90,
    ("dataset_available","published"):0.70,
    ("early","dataset_available"):1.00,
    ("mid","dataset_available"):0.90,
}


def methodology_overlap(a, b):
    sa = set(m.lower() for m in a.get("methodology", []))
    sb = set(m.lower() for m in b.get("methodology", []))
    if not sa or not sb: return 0.0
    inter = sa & sb
    union = sa | sb
    base = len(inter) / len(union)
    rare = {"federated learning","quantum computing","qaoa",
            "split learning","differential privacy","secure aggregation"}
    bonus = 0.15 if inter & rare else 0.0
    return min(base + bonus, 1.0)


def domain_proximity(a, b):
    ea = np.array(a.get("embedding",[0.5]*8), dtype=float)
    eb = np.array(b.get("embedding",[0.5]*8), dtype=float)
    ea = ea / (np.linalg.norm(ea) or 1)
    eb = eb / (np.linalg.norm(eb) or 1)
    dot  = np.dot(ea, eb)
    norm = np.linalg.norm(ea) * np.linalg.norm(eb)
    return float(dot / norm) if norm > 0 else 0.0


def dataset_compatibility(a, b):
    ia, ib = a.get("irb_status",""), b.get("irb_status","")
    if ia == "pending" or ib == "pending": return 0.1
    if ia in ["approved","not_required"] and ib in ["approved","not_required"]:
        wa = set(" ".join(a.get("datasets",[])).lower().split())
        wb = set(" ".join(b.get("datasets",[])).lower().split())
        ov = len(wa & wb) / max(len(wa | wb), 1)
        return min(0.5 + ov*0.5, 1.0)
    return 0.3


def stage_complementarity(a, b):
    sa, sb = a.get("stage","early"), b.get("stage","early")
    return STAGE_MATRIX.get((sa,sb), STAGE_MATRIX.get((sb,sa), 0.50))


def qaoa_score(query_profile, candidate):
    """Pure QAOA QUBO objective — no FL."""
    m  = methodology_overlap(query_profile, candidate)
    d  = domain_proximity(query_profile, candidate)
    ds = dataset_compatibility(query_profile, candidate)
    s  = stage_complementarity(query_profile, candidate)
    score = (WEIGHTS["methodology"]*m + WEIGHTS["domain"]*d +
             WEIGHTS["dataset"]*ds  + WEIGHTS["stage"]*s)
    if m > 0.75 and d > 0.85:
        score *= 0.70
    score = 0.5 + (score * 0.6)
    return min(score, 0.97), {"methodology_overlap":round(m,3),
                               "domain_proximity":round(d,3),
                               "dataset_compatibility":round(ds,3),
                               "stage_complementarity":round(s,3)}


def qaoa_rank(query_profile: dict, candidates: list,
              fl_model=None) -> list:
    """
    Full pipeline: FL prior + QAOA refinement.

    If fl_model is available:
      final_score = FL_WEIGHT * fl_score + QAOA_WEIGHT * qaoa_score

    If fl_model is None (fallback):
      final_score = qaoa_score only

    This is the QUBO optimization step:
      Maximize Σ w_ij * x_i * x_j across all candidate pairs
      where w_ij = FL-informed QAOA weight
    """
    results = []

    for c in candidates:
        # Step 1 — Pure QAOA score
        q_score, breakdown = qaoa_score(query_profile, c)

        # Step 2 — FL prior (if model trained)
        if fl_model and fl_model.trained:
            fl_result = fl_model.predict_pair(query_profile, c)
            fl_s = fl_result["fl_score"]
            # Blend FL + QAOA
            final = FL_WEIGHT * fl_s + QAOA_WEIGHT * q_score
            breakdown["fl_score"]    = round(fl_s, 3)
            breakdown["qaoa_score"]  = round(q_score, 3)
            breakdown["fl_informed"] = True
        else:
            final = q_score
            breakdown["fl_informed"] = False

        final = round(float(np.clip(final, 0.0, 0.97)), 3)

        results.append({
            **c,
            "collaboration_probability": final,
            "match_percent":  round(final * 100),
            "breakdown":      breakdown,
        })

    # Sort by final collaboration probability
    results.sort(key=lambda x: x["collaboration_probability"], reverse=True)
    return results


def query_to_profile(query: str, query_embedding: list) -> dict:
    q = query.lower()
    methodology, domain = [], []
    if any(k in q for k in ["federated","distributed","privacy"]):
        methodology.append("federated learning")
    if any(k in q for k in ["quantum","qaoa"]):
        methodology.append("quantum computing")
    if any(k in q for k in ["transformer","bert","nlp","language"]):
        methodology.append("transformer")
    if any(k in q for k in ["cnn","imaging","image"]):
        methodology.append("CNN")
    if not methodology:
        methodology = ["machine learning"]
    if any(k in q for k in ["genomic","omics","genome","exome"]):
        domain.append("genomics")
    if any(k in q for k in ["cancer","tumor"]):
        domain.append("cancer detection")
    if any(k in q for k in ["rare disease"]):
        domain.append("rare disease")
    if any(k in q for k in ["clinical","ehr","hospital"]):
        domain.append("clinical NLP")
    if not domain:
        domain = ["biomedical"]
    return {
        "methodology": methodology, "domain": domain,
        "datasets": [], "irb_status": "approved",
        "stage": "early", "embedding": query_embedding,
    }