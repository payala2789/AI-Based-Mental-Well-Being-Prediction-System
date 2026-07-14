"""
train.py — Train RandomForest, XGBoost (early stopping), CatBoost + Ensemble.
           Final prediction = weighted blend of all models.

Usage:
    python train.py

Outputs:
    models/rf_model.joblib
    models/xgb_model.joblib
    models/cat_model.joblib       (if catboost installed)
    models/best_model.joblib      ← best individual OR ensemble
    models/ensemble_model.joblib
    models/training_report.md
"""

import os
import numpy as np
import joblib
from math import sqrt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from preprocess import run_preprocessing

MODELS_DIR   = "models"
RANDOM_STATE = 42
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Severity mapping ──────────────────────────────────────────────────────────
def phq9_severity(score: float) -> str:
    score = round(float(score))
    if score <= 4:  return "Minimal"
    if score <= 9:  return "Mild"
    if score <= 14: return "Moderate"
    if score <= 19: return "Moderately Severe"
    return "Severe"


# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate(name: str, y_true, y_pred) -> dict:
    y_pred = np.clip(y_pred, 0, 27)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{name}]")
    print(f"    MAE  : {mae:.4f}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    R²   : {r2:.4f}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2}


# ── Ensemble predictor ────────────────────────────────────────────────────────
class WeightedEnsemble:
    """Weighted average of multiple regressors. Picklable via joblib."""
    def __init__(self, models: list, weights: list):
        self.models  = models
        self.weights = np.array(weights) / sum(weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([m.predict(X) for m in self.models], axis=1)
        return np.dot(preds, self.weights)


# ── Training report ───────────────────────────────────────────────────────────
def write_report(results: list, best_name: str, ensemble_rmse: float,
                 feature_columns: list) -> None:
    lines = [
        "# Model Training Report\n\n",
        f"**Best model:** {best_name}\n\n",
        "## Evaluation Metrics\n\n",
        "| Model | MAE | RMSE | R² |\n",
        "|-------|-----|------|----|\n",
    ]
    for r in results:
        lines.append(f"| {r['name']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['r2']:.4f} |\n")
    lines.append(f"| **Ensemble** | — | **{ensemble_rmse:.4f}** | — |\n\n")
    lines += [
        "## Feature Engineering (25 total features)\n\n",
        "Base: 20 original + 5 composite:\n\n",
        "- `stress_index` — avg of burnout, job stress, financial stress\n",
        "- `lifestyle_score` — sleep + exercise − alcohol\n",
        "- `total_screen_load` — screen + social media + gaming hours\n",
        "- `isolation_index` — (10 − social support + social media) / 2\n",
        "- `work_life_ratio` — work hours / weekly sleep hours\n\n",
        "## PHQ-9 Severity Mapping\n\n",
        "| Range | Label |\n|-------|-------|\n",
        "| 0–4 | Minimal |\n| 5–9 | Mild |\n",
        "| 10–14 | Moderate |\n| 15–19 | Moderately Severe |\n| 20–27 | Severe |\n",
    ]
    path = os.path.join(MODELS_DIR, "training_report.md")
    with open(path, "w") as f: f.writelines(lines)
    print(f"\n[train] Report saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print("  Mental Health AI — Enhanced Model Training")
    print("=" * 65)

    data            = run_preprocessing()
    X_train         = data["X_train"]
    X_test          = data["X_test"]
    y_train         = data["y_train"]
    y_test          = data["y_test"]
    feature_columns = data["feature_columns"]

    print(f"\n[train] {X_train.shape[0]} train | "
          f"{X_test.shape[0]} test | "
          f"{X_train.shape[1]} features")

    trained_models = []   # (name, model, rmse)
    results        = []

    # ── 1. RandomForest ───────────────────────────────────────────────────────
    print("\n[train] Fitting RandomForest …")
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.joblib"))
    rf_res = evaluate("RandomForest", y_test, rf.predict(X_test))
    results.append(rf_res)
    trained_models.append(("RandomForest", rf, rf_res["rmse"]))

    # ── 2. XGBoost + early stopping ───────────────────────────────────────────
    print("\n[train] Fitting XGBoost (early stopping at 30 rounds) …")
    xgb = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.2,
        reg_lambda=1.5,
        min_child_weight=3,
        early_stopping_rounds=30,   # XGBoost 2.x: pass on constructor
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        eval_metric="rmse",
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  Stopped at round {xgb.best_iteration + 1} / 600")
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb_model.joblib"))
    xgb_res = evaluate("XGBoost", y_test, xgb.predict(X_test))
    results.append(xgb_res)
    trained_models.append(("XGBoost", xgb, xgb_res["rmse"]))

    # ── 3. CatBoost ───────────────────────────────────────────────────────────
    if CATBOOST_AVAILABLE:
        print("\n[train] Fitting CatBoost …")
        cat = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_seed=RANDOM_STATE,
            verbose=False,
            early_stopping_rounds=30,
            eval_metric="RMSE",
        )
        cat.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
        joblib.dump(cat, os.path.join(MODELS_DIR, "cat_model.joblib"))
        cat_res = evaluate("CatBoost", y_test, cat.predict(X_test))
        results.append(cat_res)
        trained_models.append(("CatBoost", cat, cat_res["rmse"]))
    else:
        print("\n[train] CatBoost not installed — skipping.  (pip install catboost)")

    # ── 4. Best individual model ──────────────────────────────────────────────
    best_name, best_model, best_rmse = min(trained_models, key=lambda x: x[2])
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    print(f"\n[train] Best individual → {best_name}  (RMSE={best_rmse:.4f})")

    # ── 5. Weighted Ensemble (weight = 1/RMSE) ────────────────────────────────
    models_list = [m for _, m, _ in trained_models]
    weights     = [1.0 / r for _, _, r in trained_models]
    norm_w      = np.array(weights) / sum(weights)

    ensemble = WeightedEnsemble(models_list, weights)
    ens_pred = np.clip(ensemble.predict(X_test), 0, 27)
    ens_rmse = sqrt(mean_squared_error(y_test, ens_pred))
    ens_mae  = mean_absolute_error(y_test, ens_pred)
    ens_r2   = r2_score(y_test, ens_pred)

    weight_str = " | ".join(
        f"{n}: {w:.3f}" for (n, _, _), w in zip(trained_models, norm_w)
    )
    print(f"\n  [Ensemble]  weights → {weight_str}")
    print(f"    MAE  : {ens_mae:.4f}")
    print(f"    RMSE : {ens_rmse:.4f}")
    print(f"    R²   : {ens_r2:.4f}")

    joblib.dump(ensemble, os.path.join(MODELS_DIR, "ensemble_model.joblib"))

    # Use ensemble as best_model if it wins
    if ens_rmse < best_rmse:
        joblib.dump(ensemble, os.path.join(MODELS_DIR, "best_model.joblib"))
        print(f"\n[train] ✓ Ensemble wins  (RMSE {ens_rmse:.4f} < {best_rmse:.4f})")
        final_name = "Ensemble"
        final_rmse = ens_rmse
    else:
        print(f"\n[train] ✓ {best_name} wins over ensemble")
        final_name = best_name
        final_rmse = best_rmse

    print(f"[train] best_model.joblib → {final_name}  (RMSE={final_rmse:.4f})")
    write_report(results, final_name, ens_rmse, feature_columns)
    print("\n[train] Done.")


if __name__ == "__main__":
    main()
