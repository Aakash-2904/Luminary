"""
fl_data.py — Luminary
Converts researchers.json into FL training data.
Generates all researcher pairs with QAOA-derived collaboration labels.
Each university becomes one federated client node.
"""

import numpy as np
import pandas as pd
import json, os, itertools

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "researchers.json")) as f:
    RESEARCHERS = json.load(f)["researchers"]


# ── Stage complementarity matrix ──
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
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    base = len(inter) / len(union)
    rare = {"federated learning","quantum computing","qaoa",
            "split learning","differential privacy","secure aggregation"}
    bonus = 0.15 if inter & rare else 0.0
    return min(base + bonus, 1.0)


def domain_proximity(a, b):
    ea = np.array(a.get("embedding", [0.5]*8), dtype=float)
    eb = np.array(b.get("embedding", [0.5]*8), dtype=float)
    ea = ea / (np.linalg.norm(ea) or 1)
    eb = eb / (np.linalg.norm(eb) or 1)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) or 1))


def dataset_compat(a, b):
    ia, ib = a.get("irb_status",""), b.get("irb_status","")
    if ia == "pending" or ib == "pending":
        return 0.1
    if ia in ["approved","not_required"] and ib in ["approved","not_required"]:
        wa = set(" ".join(a.get("datasets",[])).lower().split())
        wb = set(" ".join(b.get("datasets",[])).lower().split())
        ov = len(wa & wb) / max(len(wa | wb), 1)
        return min(0.5 + ov * 0.5, 1.0)
    return 0.3


def stage_compat(a, b):
    sa, sb = a.get("stage","early"), b.get("stage","early")
    return STAGE_MATRIX.get((sa,sb), STAGE_MATRIX.get((sb,sa), 0.5))


def compute_qaoa_label(a, b):
    """
    Compute ground truth collaboration score
    using QAOA weights. This becomes our FL training target.
    """
    m  = methodology_overlap(a, b)
    d  = domain_proximity(a, b)
    ds = dataset_compat(a, b)
    s  = stage_compat(a, b)
    score = 0.35*m + 0.30*d + 0.20*ds + 0.15*s
    # Duplication penalty
    if m > 0.75 and d > 0.85:
        score *= 0.70
    # Normalize to 0.5–0.97 range
    score = 0.5 + (score * 0.6)
    return round(min(score, 0.97), 4)


def build_feature_row(a, b):
    """
    Build feature vector for a researcher pair (a, b).
    These are the FL model inputs — same 4 QAOA factors
    plus individual researcher embeddings.
    """
    m  = methodology_overlap(a, b)
    d  = domain_proximity(a, b)
    ds = dataset_compat(a, b)
    s  = stage_compat(a, b)

    # Individual embeddings normalized
    ea = np.array(a.get("embedding",[0.5]*8), dtype=float)
    eb = np.array(b.get("embedding",[0.5]*8), dtype=float)
    ea = ea / (np.linalg.norm(ea) or 1)
    eb = eb / (np.linalg.norm(eb) or 1)

    # IRB flags
    irb_a = 1.0 if a.get("irb_status") == "approved" else 0.0
    irb_b = 1.0 if b.get("irb_status") == "approved" else 0.0

    # Stage encoded
    stage_map = {"early":0,"mid":1,"published":2,"dataset_available":3}
    sa = stage_map.get(a.get("stage","early"), 0) / 3.0
    sb = stage_map.get(b.get("stage","early"), 0) / 3.0

    return {
        # QAOA factors
        "methodology_overlap":   m,
        "domain_proximity":      d,
        "dataset_compatibility": ds,
        "stage_complementarity": s,
        # Researcher A embeddings
        "a_emb_0": ea[0], "a_emb_1": ea[1], "a_emb_2": ea[2], "a_emb_3": ea[3],
        "a_emb_4": ea[4], "a_emb_5": ea[5], "a_emb_6": ea[6], "a_emb_7": ea[7],
        # Researcher B embeddings
        "b_emb_0": eb[0], "b_emb_1": eb[1], "b_emb_2": eb[2], "b_emb_3": eb[3],
        "b_emb_4": eb[4], "b_emb_5": eb[5], "b_emb_6": eb[6], "b_emb_7": eb[7],
        # Metadata features
        "irb_a": irb_a,
        "irb_b": irb_b,
        "stage_a": sa,
        "stage_b": sb,
        # University node (for federated splitting)
        "university_a": a.get("university",""),
        "university_b": b.get("university",""),
        # Target
        "collab_score": compute_qaoa_label(a, b),
        # IDs for reference
        "id_a": a["id"],
        "id_b": b["id"],
    }


def generate_training_data():
    """
    Generate all pairwise combinations of researchers.
    Each pair becomes one training row.
    Returns a DataFrame ready for FL training.
    """
    rows = []
    for a, b in itertools.combinations(RESEARCHERS, 2):
        rows.append(build_feature_row(a, b))
        # Also add reverse pair for symmetry
        rows.append(build_feature_row(b, a))

    df = pd.DataFrame(rows)
    print(f"✅ FL training data: {len(df)} pairs from {len(RESEARCHERS)} researchers")
    return df


def split_by_university(df):
    """
    Split training data by university_a.
    Each university = one federated client node.
    This is the key difference from random IID splitting.
    """
    universities = df["university_a"].unique()
    clients = {}
    for uni in universities:
        client_df = df[df["university_a"] == uni]
        clients[uni] = client_df
        print(f"  Node [{uni}]: {len(client_df)} training pairs")
    return clients


FEATURE_COLS = [
    "methodology_overlap", "domain_proximity",
    "dataset_compatibility", "stage_complementarity",
    "a_emb_0","a_emb_1","a_emb_2","a_emb_3",
    "a_emb_4","a_emb_5","a_emb_6","a_emb_7",
    "b_emb_0","b_emb_1","b_emb_2","b_emb_3",
    "b_emb_4","b_emb_5","b_emb_6","b_emb_7",
    "irb_a","irb_b","stage_a","stage_b",
]
TARGET_COL = "collab_score"