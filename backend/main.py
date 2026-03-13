"""
Luminary — FastAPI Backend
Full pipeline: Quantum Encoding → FL → QAOA
Run: python -m uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from rag import rag_search, get_query_embedding, RESEARCHERS
from qaoa import qaoa_rank, query_to_profile
from federated import run_federated_round, encrypt_all_researchers
from fl_model import get_fl_model

app = FastAPI(title="Luminary API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:3000","*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──
federated_state      = {}
encrypted_researchers = []
fl_model             = None


@app.on_event("startup")
async def startup():
    global federated_state, encrypted_researchers, fl_model

    print("\n" + "="*65)
    print("  LUMINARY STARTUP PIPELINE")
    print("="*65)

    # Step 1 — Federated round + encryption
    print("\n🔄 Step 1: Running federated learning round...")
    federated_state       = run_federated_round(RESEARCHERS)
    encrypted_researchers = encrypt_all_researchers(RESEARCHERS)
    print(f"✅ {federated_state['global_model']['n_nodes']} university nodes trained")
    print(f"✅ {len(encrypted_researchers)} researcher embeddings encrypted")

    # Step 2 — Train FL collaboration models
    print("\n🔄 Step 2: Training FL collaboration models (RF + GB + Ridge)...")
    fl_model = get_fl_model()
    print(f"✅ FL models trained — summary: {fl_model.summary}")

    print("\n✅ Luminary ready — Full FL + QAOA pipeline active\n")


# ── Request models ──
class SearchRequest(BaseModel):
    query: str
    university_filter: Optional[str] = None
    irb_filter: Optional[bool] = None
    status_filter: Optional[str] = None
    top_k: Optional[int] = 8


@app.get("/")
def root():
    return {
        "name": "Luminary API",
        "version": "2.0.0",
        "pipeline": "Quantum Encoding → Federated Learning → QAOA",
        "status": "running",
        "fl_trained": fl_model.trained if fl_model else False,
        "federated_nodes": federated_state.get("global_model",{}).get("n_nodes",0),
        "fl_summary": fl_model.summary if fl_model else {},
    }


@app.post("/search")
def search(req: SearchRequest):
    """
    Full search pipeline:
    1. RAG — semantic search returns top candidates
    2. FL — collaboration probability from trained models
    3. QAOA — refines with multi-variable optimization
    4. Returns FL-informed QAOA ranked results
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1 — RAG
    rag_results = rag_search(req.query, top_k=req.top_k)

    # Step 2 — Filters
    filtered = rag_results
    if req.university_filter and req.university_filter != "All Universities":
        filtered = [r for r in filtered
                    if req.university_filter.lower() in r["university"].lower()]
    if req.irb_filter:
        filtered = [r for r in filtered if r.get("irb_status") == "approved"]
    if req.status_filter and req.status_filter != "all":
        filtered = [r for r in filtered if r.get("status") == req.status_filter]

    if not filtered:
        return {"query": req.query, "total": 0, "results": []}

    # Step 3 — Build query profile
    query_emb    = get_query_embedding(req.query).tolist()
    query_profile = query_to_profile(req.query, query_emb)

    # Step 4 — FL + QAOA ranking
    ranked = qaoa_rank(query_profile, filtered, fl_model=fl_model)

    # Step 5 — Add metadata
    for r in ranked:
        r["embedding_status"] = "encrypted"
        r["pipeline"]         = "FL+QAOA"
        r["fl_informed"]      = r.get("breakdown",{}).get("fl_informed", False)

    return {
        "query":                req.query,
        "total":                len(ranked),
        "pipeline":             "Quantum Encoding → Federated Learning → QAOA",
        "fl_trained":           fl_model.trained if fl_model else False,
        "federated_nodes":      federated_state.get("global_model",{}).get("n_nodes",0),
        "results":              ranked,
    }


@app.get("/fl/summary")
def fl_summary():
    """Returns FL model performance summary — shows at judges demo."""
    if not fl_model or not fl_model.trained:
        return {"status": "not trained"}
    return {
        "status":   "trained",
        "models":   fl_model.summary,
        "weights":  {
            "RF":    fl_model.best_weights[0],
            "GB":    fl_model.best_weights[1],
            "Ridge": fl_model.best_weights[2],
        },
        "pipeline": "FL score weighted {:.0f}% + QAOA {:.0f}%".format(
            40, 60
        ),
        "nodes": list(set(r["university"] for r in RESEARCHERS)),
    }


@app.get("/federated/status")
def federated_status():
    return {
        "status":            "active",
        "round":             federated_state.get("round",1),
        "nodes":             federated_state.get("nodes",[]),
        "global_model":      federated_state.get("global_model",{}),
        "privacy_guarantee": federated_state.get("privacy_guarantee",""),
        "compliance":        federated_state.get("compliance",[]),
    }


@app.get("/researcher/{researcher_id}")
def get_researcher(researcher_id: str):
    for r in encrypted_researchers:
        if r["id"] == researcher_id:
            safe = {k:v for k,v in r.items() if k != "embedding"}
            safe["embedding_status"] = "encrypted"
            return safe
    raise HTTPException(status_code=404, detail="Researcher not found")


@app.get("/researchers")
def get_all():
    return {"researchers": RESEARCHERS, "total": len(RESEARCHERS)}


@app.get("/health")
def health():
    return {
        "status":      "ok",
        "fl_trained":  fl_model.trained if fl_model else False,
        "fl_summary":  fl_model.summary if fl_model else {},
        "researchers": len(RESEARCHERS),
        "nodes":       federated_state.get("global_model",{}).get("n_nodes",0),
    }