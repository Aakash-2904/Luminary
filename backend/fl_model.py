"""
fl_model.py — Luminary
Federated Learning across university nodes.
Directly adapted from Federated_Learning.ipynb.

Three models:
  Model A — Random Forest    (Tree-Merging Federation)
  Model B — Gradient Boosting (Prediction-Averaging Federation)
  Model C — Ridge Regression  (FedAvg on Coefficients)

Input:  researcher pair feature vectors
Output: collaboration probability 0→1
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from fl_data import (
    generate_training_data, split_by_university,
    FEATURE_COLS, TARGET_COL, RESEARCHERS,
    build_feature_row
)

NUM_ROUNDS = 3

# ── Shared model params ──
RF_PARAMS = dict(n_estimators=40, max_depth=6,
                 min_samples_leaf=2, random_state=42, n_jobs=-1)
GB_PARAMS = dict(n_estimators=100, max_depth=4,
                 learning_rate=0.1, subsample=0.8, random_state=42)
RIDGE_ALPHA = 1.0


def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    if label:
        print(f"    {label:<40} MAE={mae:.4f}  R²={r2:.4f}")
    return mae, r2


class LuminaryFLModel:
    """
    Federated Learning model for collaboration prediction.
    Trains 3 models across university nodes using FedAvg.
    Exposes predict() for use by QAOA layer.
    """

    def __init__(self):
        self.fed_rf        = None
        self.gb_models     = None
        self.gb_sizes      = None
        self.global_coef   = None
        self.global_intercept = None
        self.best_weights  = (1/3, 1/3, 1/3)
        self.trained       = False
        self.summary       = {}

    def train(self):
        print("\n" + "="*65)
        print("  LUMINARY — FEDERATED LEARNING — COLLABORATION PREDICTION")
        print("="*65)

        # ── Generate training data ──
        df = generate_training_data()
        clients = split_by_university(df)

        # Full dataset for validation
        X_all = df[FEATURE_COLS].values.astype(float)
        y_all = df[TARGET_COL].values.astype(float)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42
        )

        print(f"\n  Training pairs : {len(X_tr)}")
        print(f"  Validation pairs: {len(X_val)}")
        print(f"  Features        : {len(FEATURE_COLS)}")
        print(f"  FL nodes        : {len(clients)} universities")
        print(f"  FL rounds       : {NUM_ROUNDS}\n")

        # Client data as arrays
        client_arrays = []
        for uni, cdf in clients.items():
            Xc = cdf[FEATURE_COLS].values.astype(float)
            yc = cdf[TARGET_COL].values.astype(float)
            client_arrays.append((uni, Xc, yc))

        client_sizes = [len(xc) for _, xc, _ in client_arrays]

        # ══════════════════════════════════════════
        # MODEL A — RANDOM FOREST (Tree-Merging)
        # ══════════════════════════════════════════
        print("─"*65)
        print("  MODEL A: Random Forest  (Tree-Merging Federation)")
        print("─"*65)

        fedrf_trees = []
        for rnd in range(1, NUM_ROUNDS + 1):
            for uni, xc, yc in client_arrays:
                local_rf = RandomForestRegressor(**RF_PARAMS)
                local_rf.fit(xc, yc)
                fedrf_trees.extend(local_rf.estimators_)

        self.fed_rf = RandomForestRegressor(**RF_PARAMS)
        self.fed_rf.fit(X_tr[:5], y_tr[:5])
        self.fed_rf.estimators_ = fedrf_trees
        self.fed_rf.n_estimators = len(fedrf_trees)

        rf_val = self.fed_rf.predict(X_val)
        mae_rf, r2_rf = evaluate(y_val, rf_val, "Random Forest (federated)")
        self.summary["RF"] = {"MAE": round(mae_rf, 4), "R2": round(r2_rf, 4)}

        # ══════════════════════════════════════════
        # MODEL B — GRADIENT BOOSTING (Prediction Averaging)
        # ══════════════════════════════════════════
        print("\n" + "─"*65)
        print("  MODEL B: Gradient Boosting  (Prediction-Averaging)")
        print("─"*65)

        gb_round_models = []
        for rnd in range(1, NUM_ROUNDS + 1):
            round_models = []
            for uni, xc, yc in client_arrays:
                local_gb = GradientBoostingRegressor(**GB_PARAMS)
                local_gb.fit(xc, yc)
                round_models.append(local_gb)
            gb_round_models = round_models

        self.gb_models = gb_round_models
        self.gb_sizes  = client_sizes

        gb_val = self._gb_predict(X_val)
        mae_gb, r2_gb = evaluate(y_val, gb_val, "Gradient Boosting (federated)")
        self.summary["GB"] = {"MAE": round(mae_gb, 4), "R2": round(r2_gb, 4)}

        # ══════════════════════════════════════════
        # MODEL C — RIDGE (FedAvg on Coefficients)
        # ══════════════════════════════════════════
        print("\n" + "─"*65)
        print("  MODEL C: Ridge Regression  (FedAvg on Coefficients)")
        print("─"*65)

        self.global_coef      = np.zeros(len(FEATURE_COLS))
        self.global_intercept = 0.0

        for rnd in range(1, NUM_ROUNDS + 1):
            local_coefs, local_ints, local_sz = [], [], []
            for uni, xc, yc in client_arrays:
                local_ridge = Ridge(alpha=RIDGE_ALPHA)
                local_ridge.coef_      = self.global_coef.copy()
                local_ridge.intercept_ = self.global_intercept
                local_ridge.fit(xc, yc)
                local_coefs.append(local_ridge.coef_)
                local_ints.append(local_ridge.intercept_)
                local_sz.append(len(xc))
            total = sum(local_sz)
            self.global_coef = np.average(
                np.stack(local_coefs), axis=0,
                weights=[s/total for s in local_sz]
            )
            self.global_intercept = np.average(
                local_ints, weights=[s/total for s in local_sz]
            )
            ridge_val = X_val @ self.global_coef + self.global_intercept
            evaluate(y_val, ridge_val, f"Ridge Round {rnd}")

        ridge_val = X_val @ self.global_coef + self.global_intercept
        mae_r, r2_r = evaluate(y_val, ridge_val, "Ridge Regression (federated)")
        self.summary["Ridge"] = {"MAE": round(mae_r, 4), "R2": round(r2_r, 4)}

        # ══════════════════════════════════════════
        # ENSEMBLE — Optimise weights on validation
        # ══════════════════════════════════════════
        print("\n" + "─"*65)
        print("  ENSEMBLE: RF + GB + Ridge (optimised weights)")
        print("─"*65)

        rf_v    = rf_val
        gb_v    = gb_val
        ridge_v = ridge_val

        best_mae, best_w = np.inf, (1/3, 1/3, 1/3)
        for w1 in np.arange(0.1, 0.9, 0.1):
            for w2 in np.arange(0.1, 0.9-w1, 0.1):
                w3 = 1 - w1 - w2
                if w3 <= 0: continue
                ens = w1*rf_v + w2*gb_v + w3*ridge_v
                m   = mean_absolute_error(y_val, ens)
                if m < best_mae:
                    best_mae, best_w = m, (w1, w2, w3)

        self.best_weights = best_w
        w1, w2, w3 = best_w
        ens_val = w1*rf_v + w2*gb_v + w3*ridge_v
        mae_e, r2_e = evaluate(y_val, ens_val, "Ensemble (federated)")
        self.summary["Ensemble"] = {"MAE": round(mae_e, 4), "R2": round(r2_e, 4)}

        print(f"\n  Optimal weights  RF={w1:.2f}  GB={w2:.2f}  Ridge={w3:.2f}")
        print("\n" + "="*65)
        print("  FL TRAINING COMPLETE")
        print("="*65 + "\n")

        self.trained = True
        return self.summary

    def _gb_predict(self, X):
        total = sum(self.gb_sizes)
        weights = [s/total for s in self.gb_sizes]
        preds = np.stack([m.predict(X) for m in self.gb_models], axis=0)
        return np.average(preds, axis=0, weights=weights)

    def predict_pair(self, researcher_a: dict, researcher_b: dict) -> dict:
        """
        Predict collaboration probability for a researcher pair.
        Returns ensemble score + individual model scores.
        Called by QAOA layer to get FL-informed prior.
        """
        if not self.trained:
            return {"fl_score": 0.5, "trained": False}

        row = build_feature_row(researcher_a, researcher_b)
        X = np.array([[row[f] for f in FEATURE_COLS]], dtype=float)

        rf_pred    = float(self.fed_rf.predict(X)[0])
        gb_pred    = float(self._gb_predict(X)[0])
        ridge_pred = float(X @ self.global_coef + self.global_intercept)

        w1, w2, w3 = self.best_weights
        ensemble   = w1*rf_pred + w2*gb_pred + w3*ridge_pred
        ensemble   = float(np.clip(ensemble, 0.0, 1.0))

        return {
            "fl_score":    round(ensemble, 4),
            "rf_score":    round(np.clip(rf_pred, 0, 1), 4),
            "gb_score":    round(np.clip(gb_pred, 0, 1), 4),
            "ridge_score": round(np.clip(ridge_pred, 0, 1), 4),
            "weights":     {"RF": w1, "GB": w2, "Ridge": w3},
            "trained":     True,
        }

    def predict_batch(self, query_profile: dict, candidates: list) -> list:
        """
        Predict FL scores for a query profile against all candidates.
        Returns list of (candidate_id, fl_score) sorted by score.
        """
        results = []
        for c in candidates:
            fl = self.predict_pair(query_profile, c)
            results.append({
                "id": c["id"],
                "fl_score": fl["fl_score"],
                "fl_breakdown": fl,
            })
        results.sort(key=lambda x: x["fl_score"], reverse=True)
        return results


# Singleton — trained once at startup
_fl_model_instance = None

def get_fl_model() -> LuminaryFLModel:
    global _fl_model_instance
    if _fl_model_instance is None:
        _fl_model_instance = LuminaryFLModel()
        _fl_model_instance.train()
    return _fl_model_instance