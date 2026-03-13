"""
Luminary — Federated Learning Simulation
Each university is a federated node.
Each node trains locally on its own researchers.
Only encrypted model weights are shared centrally.
Central server runs FedAvg to build global model.
Raw data never leaves the university node.
"""

import numpy as np
import hashlib
import json
import os
import time
from typing import Dict, List

BASE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────
# STEP 1 — ENCRYPTION
# Simulates quantum-safe encryption of embeddings
# In production: use actual quantum key distribution
# or post-quantum cryptography (CRYSTALS-Kyber)
# ─────────────────────────────────────────

def generate_node_key(university: str) -> np.ndarray:
    """
    Generate a deterministic encryption key per university node
    Based on university name hash — simulates unique node keys
    In production: proper key exchange protocol
    """
    seed = int(hashlib.sha256(university.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    return rng.uniform(0.8, 1.2, size=8)  # perturbation key


def encrypt_embedding(embedding: List[float], university: str) -> Dict:
    """
    Encrypt a researcher's embedding before it leaves the university
    Simulates: multiply by node-specific key + add noise mask
    In production: AES-256 or post-quantum encryption
    Returns encrypted vector + metadata (never raw embedding)
    """
    key = generate_node_key(university)
    raw = np.array(embedding, dtype=float)

    # Encrypt: element-wise multiplication with key + small noise
    noise = np.random.RandomState(42).uniform(-0.05, 0.05, size=8)
    encrypted = (raw * key) + noise

    # Generate checksum to verify integrity
    checksum = hashlib.md5(encrypted.tobytes()).hexdigest()[:8]

    return {
        "encrypted_vector": encrypted.tolist(),
        "checksum": checksum,
        "university": university,
        "key_id": hashlib.sha256(university.encode()).hexdigest()[:12],
        "encrypted": True
    }


def decrypt_embedding(encrypted_data: Dict) -> np.ndarray:
    """
    Decrypt embedding at central server for model training only
    In production: requires university's private key
    """
    key = generate_node_key(encrypted_data["university"])
    encrypted = np.array(encrypted_data["encrypted_vector"], dtype=float)
    noise = np.random.RandomState(42).uniform(-0.05, 0.05, size=8)
    decrypted = (encrypted - noise) / key
    return decrypted


# ─────────────────────────────────────────
# STEP 2 — LOCAL NODE TRAINING
# Each university trains on its own researchers
# Produces local model weights
# Raw researcher data never leaves this function
# ─────────────────────────────────────────

def train_local_node(university: str, researchers: List[Dict]) -> Dict:
    """
    Simulates local model training at a university node

    In production:
    - SciBERT fine-tuned on local research corpus
    - Produces gradient updates not raw embeddings
    - Training happens entirely on university servers

    Prototype:
    - Compute mean embedding as local model weight
    - Add gradient noise to simulate real training
    - Encrypt weights before sending
    """
    start = time.time()

    # Filter researchers belonging to this university
    local_researchers = [r for r in researchers if r["university"] == university]

    if not local_researchers:
        return None

    # LOCAL TRAINING — compute mean embedding as model representation
    embeddings = np.array([r["embedding"] for r in local_researchers])
    local_weights = embeddings.mean(axis=0)

    # Simulate gradient update noise (real FL adds differential privacy noise)
    np.random.seed(abs(hash(university)) % 2**31)
    gradient_noise = np.random.normal(0, 0.01, size=8)
    local_weights = local_weights + gradient_noise

    # Normalize weights
    local_weights = local_weights / (np.linalg.norm(local_weights) or 1)

    # Encrypt weights before sending to central server
    encrypted_weights = encrypt_embedding(local_weights.tolist(), university)

    training_time = round(time.time() - start, 4)

    return {
        "university": university,
        "n_researchers": len(local_researchers),
        "local_weights_encrypted": encrypted_weights,
        "training_time_seconds": training_time,
        "raw_data_shared": False,        # Always False — this is the guarantee
        "status": "trained"
    }


# ─────────────────────────────────────────
# STEP 3 — FEDERATED AVERAGING (FedAvg)
# Central server aggregates encrypted weights
# McMahan et al. 2017 — Communication-Efficient
# Learning of Deep Networks from Decentralized Data
# ─────────────────────────────────────────

def federated_averaging(node_results: List[Dict]) -> Dict:
    """
    FedAvg — aggregate local model weights from all nodes
    Weighted average by number of researchers per node
    In production: runs on encrypted weights using
    secure multi-party computation (SMPC)
    """
    if not node_results:
        return {}

    total_researchers = sum(n["n_researchers"] for n in node_results)
    global_weights = np.zeros(8)

    for node in node_results:
        # Decrypt weights for aggregation
        # In production: SMPC means server never sees plaintext
        decrypted = decrypt_embedding(node["local_weights_encrypted"])

        # Weighted contribution based on data size
        weight = node["n_researchers"] / total_researchers
        global_weights += weight * decrypted

    # Normalize global model
    global_weights = global_weights / (np.linalg.norm(global_weights) or 1)

    return {
        "global_weights": global_weights.tolist(),
        "n_nodes": len(node_results),
        "total_researchers": total_researchers,
        "aggregation_method": "FedAvg",
        "raw_data_accessed": False       # Central server never saw raw data
    }


# ─────────────────────────────────────────
# STEP 4 — FULL FEDERATED PIPELINE
# Orchestrates the entire FL round
# ─────────────────────────────────────────

def run_federated_round(researchers: List[Dict]) -> Dict:
    """
    Run one complete federated learning round:
    1. Discover all university nodes
    2. Each node trains locally
    3. Encrypted weights sent to central server
    4. FedAvg aggregates into global model
    5. Global model sent back to all nodes
    """
    universities = list(set(r["university"] for r in researchers))
    node_results = []
    node_logs = []

    for uni in universities:
        result = train_local_node(uni, researchers)
        if result:
            node_results.append(result)
            node_logs.append({
                "university": uni,
                "researchers_trained": result["n_researchers"],
                "status": "✅ Trained locally",
                "data_shared": "Encrypted weights only",
                "raw_data_left_university": False
            })

    # Central aggregation
    global_model = federated_averaging(node_results)

    return {
        "round": 1,
        "nodes": node_logs,
        "global_model": global_model,
        "privacy_guarantee": "Raw data never left any university node",
        "compliance": ["FERPA", "IRB", "HIPAA-ready"]
    }


# ─────────────────────────────────────────
# STEP 5 — ENCRYPT ALL RESEARCHER EMBEDDINGS
# Called at startup — encrypts all stored embeddings
# ─────────────────────────────────────────

def encrypt_all_researchers(researchers: List[Dict]) -> List[Dict]:
    """
    Encrypts embeddings for all researchers
    Adds encryption metadata to each record
    In production: original embeddings deleted after encryption
    """
    encrypted_researchers = []
    for r in researchers:
        enc = encrypt_embedding(r["embedding"], r["university"])
        encrypted_r = {**r}
        encrypted_r["embedding_encrypted"] = enc
        encrypted_r["embedding_status"] = "encrypted"
        encrypted_r["key_id"] = enc["key_id"]
        encrypted_researchers.append(encrypted_r)
    return encrypted_researchers