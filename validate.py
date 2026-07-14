"""
validate.py — Rigorous model validation for the Mental Health AI system.

Answers the key question: "Does the model actually work?"

Checks performed:
  1. K-Fold Cross-Validation (5-fold) on the full dataset
  2. Train vs Test error comparison  (overfitting detector)
  3. XGBoost staged scores           (boosting round curves)
  4. Learning curve                  (underfitting detector)
  5. Residual analysis               (bias / systematic error)
  6. Prediction distribution         (does model cover full range?)
  7. Severity classification accuracy (are bucket labels correct?)

Run AFTER train.py:
    python validate.py

Outputs:
    models/validation/cv_results.md
    models/validation/plots/  (PNG charts)
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import sqrt

from sklearn.model_selection import KFold, cross_validate, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from preprocess import run_preprocessing

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR  = "models"
VAL_DIR     = os.path.join(MODELS_DIR, "validation")
PLOTS_DIR   = os.path.join(VAL_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0b0f1a",
    "axes.facecolor":    "#111827",
    "axes.edgecolor":    "#374151",
    "axes.labelcolor":   "#9ca3af",
    "axes.titlecolor":   "#e5e7eb",
    "xtick.color":       "#6b7280",
    "ytick.color":       "#6b7280",
    "text.color":        "#e5e7eb",
    "grid.color":        "#1f2937",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

ACCENT  = "#6366f1"
GREEN   = "#10b981"
RED     = "#ef4444"
YELLOW  = "#f59e0b"
ORANGE  = "#f97316"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def phq9_severity(score: float) -> str:
    s = round(float(score))
    if s <= 4:  return "Minimal"
    if s <= 9:  return "Mild"
    if s <= 14: return "Moderate"
    if s <= 19: return "Moderately Severe"
    return "Severe"


def metrics(y_true, y_pred) -> dict:
    y_pred = np.clip(y_pred, 0, 27)
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": sqrt(mean_squared_error(y_true, y_pred)),
        "R2":   r2_score(y_true, y_pred),
    }


def save_fig(name: str) -> None:
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  1. K-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def run_kfold_cv(model, X: np.ndarray, y: np.ndarray, name: str, k: int = 5) -> dict:
    """
    Split data into k equal folds.
    Train on k-1 folds, test on the remaining fold.
    Repeat k times so every sample is tested exactly once.
    Average the scores → unbiased performance estimate.
    """
    print(f"\n[CV] {k}-Fold Cross-Validation — {name}")
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    fold_mae, fold_rmse, fold_r2 = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        preds = np.clip(model.predict(X_val), 0, 27)

        m = metrics(y_val, preds)
        fold_mae.append(m["MAE"])
        fold_rmse.append(m["RMSE"])
        fold_r2.append(m["R2"])

        print(f"  Fold {fold_idx}: MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R²={m['R2']:.4f}")

    result = {
        "MAE_mean":  np.mean(fold_mae),  "MAE_std":  np.std(fold_mae),
        "RMSE_mean": np.mean(fold_rmse), "RMSE_std": np.std(fold_rmse),
        "R2_mean":   np.mean(fold_r2),   "R2_std":   np.std(fold_r2),
        "fold_mae": fold_mae, "fold_rmse": fold_rmse, "fold_r2": fold_r2,
    }
    print(f"  ── {k}-Fold Average ──")
    print(f"  MAE  : {result['MAE_mean']:.4f} ± {result['MAE_std']:.4f}")
    print(f"  RMSE : {result['RMSE_mean']:.4f} ± {result['RMSE_std']:.4f}")
    print(f"  R²   : {result['R2_mean']:.4f} ± {result['R2_std']:.4f}")
    return result


def plot_cv_scores(rf_cv: dict, xgb_cv: dict, k: int = 5) -> None:
    """Bar chart comparing fold-level RMSE for both models."""
    folds = [f"Fold {i}" for i in range(1, k + 1)]
    x = np.arange(k)
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{k}-Fold Cross-Validation Scores", fontsize=13, fontweight="bold")

    pairs = [
        ("MAE",  rf_cv["fold_mae"],  xgb_cv["fold_mae"]),
        ("RMSE", rf_cv["fold_rmse"], xgb_cv["fold_rmse"]),
        ("R²",   rf_cv["fold_r2"],   xgb_cv["fold_r2"]),
    ]

    for ax, (metric, rf_vals, xgb_vals) in zip(axes, pairs):
        ax.bar(x - w/2, rf_vals,  w, label="RandomForest", color=ACCENT, alpha=.85)
        ax.bar(x + w/2, xgb_vals, w, label="XGBoost",      color=GREEN,  alpha=.85)
        ax.set_title(metric, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(folds, rotation=30, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.4)

    plt.tight_layout()
    save_fig("01_cv_scores_per_fold")


# ══════════════════════════════════════════════════════════════════════════════
#  2. OVERFITTING CHECK (train vs test error)
# ══════════════════════════════════════════════════════════════════════════════

def check_overfitting(model, X_train, y_train, X_test, y_test, name: str) -> dict:
    """
    Compare error on training data vs test data.

    If train error ≪ test error  → overfitting  (model memorised training data)
    If train error ≈ test error  → good fit
    If both errors are high      → underfitting (model too simple)
    """
    model.fit(X_train, y_train)
    train_m = metrics(y_train, model.predict(X_train))
    test_m  = metrics(y_test,  model.predict(X_test))

    print(f"\n[Overfit] {name}")
    print(f"  Train → MAE={train_m['MAE']:.4f}  RMSE={train_m['RMSE']:.4f}  R²={train_m['R2']:.4f}")
    print(f"  Test  → MAE={test_m['MAE']:.4f}  RMSE={test_m['RMSE']:.4f}  R²={test_m['R2']:.4f}")

    overfit_ratio = test_m["RMSE"] / (train_m["RMSE"] + 1e-9)
    verdict = "Good fit ✓" if overfit_ratio < 1.25 else ("Overfitting ⚠️" if overfit_ratio > 1.5 else "Mild overfit")
    print(f"  Test/Train RMSE ratio: {overfit_ratio:.3f} → {verdict}")
    return {"train": train_m, "test": test_m, "ratio": overfit_ratio, "verdict": verdict}


def plot_overfit_comparison(rf_of: dict, xgb_of: dict) -> None:
    labels = ["Train", "Test"]
    metrics_names = ["MAE", "RMSE", "R²"]
    x = np.arange(len(labels))
    w = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Train vs Test Error (Overfitting Check)", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, metrics_names):
        key = {"MAE": "MAE", "RMSE": "RMSE", "R²": "R2"}[metric]
        rf_vals  = [rf_of["train"][key],  rf_of["test"][key]]
        xgb_vals = [xgb_of["train"][key], xgb_of["test"][key]]

        ax.bar(x - w/2, rf_vals,  w, label="RandomForest", color=ACCENT, alpha=.85)
        ax.bar(x + w/2, xgb_vals, w, label="XGBoost",      color=GREEN,  alpha=.85)
        ax.set_title(metric, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.4)

    plt.tight_layout()
    save_fig("02_overfit_train_vs_test")


# ══════════════════════════════════════════════════════════════════════════════
#  3. XGBoost BOOSTING ROUNDS (staged evaluation = "epoch" equivalent)
# ══════════════════════════════════════════════════════════════════════════════

def plot_xgb_learning_rounds(xgb_model: XGBRegressor, X_train, y_train, X_test, y_test) -> None:
    """
    XGBoost equivalent of epoch curves.
    We evaluate RMSE at each boosting round (tree added).
    Good model: test RMSE decreases then flattens — no upward drift.
    Overfitting: test RMSE starts rising while train RMSE keeps falling.
    """
    print("\n[XGB] Evaluating per-round (staged) scores …")

    evals_result = {}
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        eval_metric="rmse",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    results = model.evals_result()
    train_rmse = results["validation_0"]["rmse"]
    test_rmse  = results["validation_1"]["rmse"]
    rounds = range(1, len(train_rmse) + 1)

    # Find best round
    best_round = int(np.argmin(test_rmse)) + 1
    best_rmse  = test_rmse[best_round - 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, train_rmse, color=ACCENT, lw=2,   label="Train RMSE", alpha=.9)
    ax.plot(rounds, test_rmse,  color=GREEN,  lw=2,   label="Test RMSE",  alpha=.9)
    ax.axvline(best_round, color=YELLOW, lw=1.5, linestyle="--", alpha=.8,
               label=f"Best round: {best_round} (RMSE={best_rmse:.3f})")
    ax.set_title("XGBoost: Boosting Rounds (equiv. of Epochs)", fontweight="bold")
    ax.set_xlabel("Boosting Round (n_estimators)")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(alpha=.4)
    plt.tight_layout()
    save_fig("03_xgb_boosting_rounds")

    print(f"  Best test RMSE at round {best_round}: {best_rmse:.4f}")
    if test_rmse[-1] > best_rmse * 1.05:
        print("  ⚠️  Late rounds show rising test RMSE — consider early stopping.")
    else:
        print("  ✓  Test RMSE is stable — no overfitting detected in boosting.")


# ══════════════════════════════════════════════════════════════════════════════
#  4. LEARNING CURVE (underfitting / data sufficiency check)
# ══════════════════════════════════════════════════════════════════════════════

def plot_learning_curve(model, X: np.ndarray, y: np.ndarray, name: str) -> None:
    """
    Trains model on increasing fractions of data (10%, 20% … 100%).
    Plots train score and cross-val score at each size.

    What to look for:
    - Both curves converge → model generalises well, enough data.
    - Gap stays large → overfitting, need more data or regularisation.
    - Both curves plateau low → underfitting, model too simple.
    """
    print(f"\n[LC] Learning curve — {name}")
    sizes = np.linspace(0.10, 1.0, 10)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=sizes,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    # Convert neg RMSE → positive
    train_mean = -train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    test_mean  = -test_scores.mean(axis=1)
    test_std   = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes_abs, train_mean, color=ACCENT, lw=2, marker="o", ms=5, label="Train RMSE")
    ax.fill_between(train_sizes_abs,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=.15, color=ACCENT)
    ax.plot(train_sizes_abs, test_mean, color=GREEN, lw=2, marker="s", ms=5, label="CV RMSE (5-fold)")
    ax.fill_between(train_sizes_abs,
                    test_mean - test_std,
                    test_mean + test_std,
                    alpha=.15, color=GREEN)
    ax.set_title(f"Learning Curve — {name}", fontweight="bold")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(alpha=.4)
    plt.tight_layout()
    tag = name.lower().replace(" ", "_")
    save_fig(f"04_learning_curve_{tag}")


# ══════════════════════════════════════════════════════════════════════════════
#  5. RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_residuals(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> None:
    """
    Residual = actual − predicted.
    Good model: residuals are random (no pattern), centred at 0.
    Bad model:  residuals follow a curve (systematic bias).
    """
    y_pred = np.clip(model.predict(X_test), 0, 27)
    residuals = y_test - y_pred

    fig = plt.figure(figsize=(13, 4))
    gs  = gridspec.GridSpec(1, 3, figure=fig)
    fig.suptitle(f"Residual Analysis — {name}", fontweight="bold")

    # (a) Predicted vs Actual
    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(y_pred, y_test, alpha=.35, color=ACCENT, s=18, edgecolors="none")
    lo, hi = 0, 27
    ax0.plot([lo, hi], [lo, hi], color=GREEN, lw=2, label="Perfect prediction")
    ax0.set_xlabel("Predicted PHQ-9"); ax0.set_ylabel("Actual PHQ-9")
    ax0.set_title("Predicted vs Actual"); ax0.legend(fontsize=8); ax0.grid(alpha=.3)

    # (b) Residuals vs Predicted (check heteroscedasticity)
    ax1 = fig.add_subplot(gs[1])
    ax1.scatter(y_pred, residuals, alpha=.35, color=ORANGE, s=18, edgecolors="none")
    ax1.axhline(0, color=GREEN, lw=2, linestyle="--")
    ax1.set_xlabel("Predicted PHQ-9"); ax1.set_ylabel("Residual (Actual − Predicted)")
    ax1.set_title("Residuals vs Predicted"); ax1.grid(alpha=.3)

    # (c) Distribution of residuals (should be ~Normal centred at 0)
    ax2 = fig.add_subplot(gs[2])
    ax2.hist(residuals, bins=30, color=ACCENT, alpha=.8, edgecolor="none")
    ax2.axvline(0, color=GREEN, lw=2, linestyle="--", label="Zero error")
    ax2.axvline(residuals.mean(), color=RED, lw=1.5, linestyle=":", label=f"Mean={residuals.mean():.2f}")
    ax2.set_xlabel("Residual"); ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution"); ax2.legend(fontsize=8); ax2.grid(alpha=.3)

    plt.tight_layout()
    tag = name.lower().replace(" ", "_")
    save_fig(f"05_residuals_{tag}")

    print(f"\n[Residuals] {name}")
    print(f"  Mean residual : {residuals.mean():.4f}  (ideal: 0)")
    print(f"  Std  residual : {residuals.std():.4f}")
    print(f"  Max abs error : {np.abs(residuals).max():.2f}")
    print(f"  % within ±2   : {(np.abs(residuals) <= 2).mean()*100:.1f}%")
    print(f"  % within ±4   : {(np.abs(residuals) <= 4).mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  6. PREDICTION DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_prediction_distribution(model, X_test, y_test, name: str) -> None:
    """
    Overlay actual vs predicted PHQ-9 distributions.
    If predicted distribution is much narrower → model is regressing to the mean.
    """
    y_pred = np.clip(model.predict(X_test), 0, 27)

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(0, 27, 28)
    ax.hist(y_test,  bins=bins, alpha=.55, color=GREEN,  label="Actual PHQ-9",    edgecolor="none")
    ax.hist(y_pred,  bins=bins, alpha=.55, color=ACCENT, label="Predicted PHQ-9", edgecolor="none")

    # Severity band backgrounds
    bands = [(0,4,"#10b981",.06), (5,9,"#f59e0b",.06),
             (10,14,"#f97316",.06), (15,19,"#ef4444",.06), (20,27,"#dc2626",.07)]
    labels_done = set()
    for lo, hi, col, a in bands:
        ax.axvspan(lo, hi+1, alpha=a, color=col,
                   label=phq9_severity(lo) if phq9_severity(lo) not in labels_done else "")
        labels_done.add(phq9_severity(lo))

    ax.set_title(f"Actual vs Predicted Distribution — {name}", fontweight="bold")
    ax.set_xlabel("PHQ-9 Score"); ax.set_ylabel("Count")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)
    plt.tight_layout()
    tag = name.lower().replace(" ", "_")
    save_fig(f"06_pred_distribution_{tag}")


# ══════════════════════════════════════════════════════════════════════════════
#  7. SEVERITY CLASSIFICATION ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

def severity_accuracy(model, X_test, y_test, name: str) -> float:
    """
    After regressing the PHQ-9 score we bucket into 5 severity classes.
    This checks how often the predicted bucket matches the true bucket.
    """
    y_pred = np.clip(model.predict(X_test), 0, 27)
    true_sev  = [phq9_severity(s) for s in y_test]
    pred_sev  = [phq9_severity(s) for s in y_pred]
    correct   = sum(t == p for t, p in zip(true_sev, pred_sev))
    acc       = correct / len(true_sev)

    # Off-by-one (adjacent bucket) accuracy
    order = ["Minimal", "Mild", "Moderate", "Moderately Severe", "Severe"]
    adj   = sum(abs(order.index(t) - order.index(p)) <= 1
                for t, p in zip(true_sev, pred_sev))
    adj_acc = adj / len(true_sev)

    print(f"\n[Severity] {name}")
    print(f"  Exact severity accuracy      : {acc*100:.1f}%")
    print(f"  Within-1-bucket accuracy     : {adj_acc*100:.1f}%")

    # Confusion-style count table
    df = pd.DataFrame({"true": true_sev, "pred": pred_sev})
    ct = pd.crosstab(df["true"], df["pred"])
    print(f"\n  Confusion table (rows=true, cols=predicted):\n{ct.to_string()}")

    return acc


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_validation_report(rf_cv, xgb_cv, rf_of, xgb_of, rf_sev_acc, xgb_sev_acc) -> None:
    lines = [
        "# Validation Report — Mental Health AI\n\n",
        "## 1. 5-Fold Cross-Validation\n\n",
        "| Metric | RandomForest | XGBoost |\n",
        "|--------|-------------|--------|\n",
        f"| MAE  | {rf_cv['MAE_mean']:.4f} ± {rf_cv['MAE_std']:.4f} | {xgb_cv['MAE_mean']:.4f} ± {xgb_cv['MAE_std']:.4f} |\n",
        f"| RMSE | {rf_cv['RMSE_mean']:.4f} ± {rf_cv['RMSE_std']:.4f} | {xgb_cv['RMSE_mean']:.4f} ± {xgb_cv['RMSE_std']:.4f} |\n",
        f"| R²   | {rf_cv['R2_mean']:.4f} ± {rf_cv['R2_std']:.4f} | {xgb_cv['R2_mean']:.4f} ± {xgb_cv['R2_std']:.4f} |\n",
        "\n> **How to read:** Mean ± Std across 5 folds. Lower std = more stable model.\n\n",

        "## 2. Overfitting Check (Train vs Test)\n\n",
        "| Model | Train RMSE | Test RMSE | Ratio | Verdict |\n",
        "|-------|-----------|----------|-------|---------|\n",
        f"| RandomForest | {rf_of['train']['RMSE']:.4f} | {rf_of['test']['RMSE']:.4f} | {rf_of['ratio']:.3f} | {rf_of['verdict']} |\n",
        f"| XGBoost      | {xgb_of['train']['RMSE']:.4f} | {xgb_of['test']['RMSE']:.4f} | {xgb_of['ratio']:.3f} | {xgb_of['verdict']} |\n",
        "\n> **Rule of thumb:** Test/Train ratio < 1.25 = good. > 1.5 = overfitting.\n\n",

        "## 3. Severity Classification Accuracy\n\n",
        "| Model | Exact Match | Within-1-Bucket |\n",
        "|-------|------------|----------------|\n",
        f"| RandomForest | {rf_sev_acc*100:.1f}% | — |\n",
        f"| XGBoost      | {xgb_sev_acc*100:.1f}% | — |\n",
        "\n> PHQ-9 score is predicted as a continuous value, then bucketed into 5 severity classes.\n\n",

        "## 4. What Each Validation Check Means\n\n",
        "| Check | What it detects |\n",
        "|-------|-----------------|\n",
        "| K-Fold CV | True generalisation — not lucky on one split |\n",
        "| Train vs Test RMSE | Overfitting (memorising training data) |\n",
        "| XGBoost round curve | Whether to stop boosting earlier |\n",
        "| Learning curve | Whether more data would help |\n",
        "| Residual plot | Systematic bias in predictions |\n",
        "| Distribution plot | Regression-to-mean / range collapse |\n",
        "| Severity accuracy | End-to-end bucket correctness |\n",
    ]
    path = os.path.join(VAL_DIR, "validation_report.md")
    with open(path, "w") as f: f.writelines(lines)
    print(f"\n[report] Validation report → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("  Mental Health AI — Model Validation Suite")
    print("=" * 65)

    # ── Load data ────────────────────────────────────────────────────────────
    data = run_preprocessing()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # ── Load saved models ────────────────────────────────────────────────────
    rf  = joblib.load(os.path.join(MODELS_DIR, "rf_model.joblib"))
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))

    # ── 1. K-Fold CV ─────────────────────────────────────────────────────────
    # Clone models with same params but fresh state for CV
    rf_fresh  = RandomForestRegressor(n_estimators=300, min_samples_split=5,
                                      min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)
    xgb_fresh = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    rf_cv  = run_kfold_cv(rf_fresh,  X_all, y_all, "RandomForest")
    xgb_cv = run_kfold_cv(xgb_fresh, X_all, y_all, "XGBoost")
    plot_cv_scores(rf_cv, xgb_cv)

    # ── 2. Overfitting check ─────────────────────────────────────────────────
    rf_of  = check_overfitting(rf_fresh,  X_train, y_train, X_test, y_test, "RandomForest")
    xgb_of = check_overfitting(xgb_fresh, X_train, y_train, X_test, y_test, "XGBoost")
    plot_overfit_comparison(rf_of, xgb_of)

    # ── 3. XGBoost round curves ──────────────────────────────────────────────
    plot_xgb_learning_rounds(xgb_fresh, X_train, y_train, X_test, y_test)

    # ── 4. Learning curves ───────────────────────────────────────────────────
    plot_learning_curve(rf_fresh,  X_all, y_all, "RandomForest")
    plot_learning_curve(xgb_fresh, X_all, y_all, "XGBoost")

    # ── 5. Residual analysis ─────────────────────────────────────────────────
    plot_residuals(rf,  X_test, y_test, "RandomForest")
    plot_residuals(xgb, X_test, y_test, "XGBoost")

    # ── 6. Prediction distributions ──────────────────────────────────────────
    plot_prediction_distribution(rf,  X_test, y_test, "RandomForest")
    plot_prediction_distribution(xgb, X_test, y_test, "XGBoost")

    # ── 7. Severity accuracy ─────────────────────────────────────────────────
    rf_sev_acc  = severity_accuracy(rf,  X_test, y_test, "RandomForest")
    xgb_sev_acc = severity_accuracy(xgb, X_test, y_test, "XGBoost")

    # ── Report ────────────────────────────────────────────────────────────────
    write_validation_report(rf_cv, xgb_cv, rf_of, xgb_of, rf_sev_acc, xgb_sev_acc)

    print("\n" + "=" * 65)
    print("  Validation complete.  Check models/validation/ for all outputs.")
    print("=" * 65)


if __name__ == "__main__":
    main()
