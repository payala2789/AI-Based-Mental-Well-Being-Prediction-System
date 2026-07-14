"""
explain.py — SHAP-based explainability for the best trained model.

Usage (after train.py has run):
    python explain.py

Outputs:
    models/explainability/shap_summary_bar.png
    models/explainability/shap_summary_beeswarm.png
    models/explainability/shap_feature_importance.csv
    models/explainability/feature_analysis_report.md
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for servers
import matplotlib.pyplot as plt
import shap

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
EXPLAIN_DIR  = os.path.join(MODELS_DIR, "explainability")
os.makedirs(EXPLAIN_DIR, exist_ok=True)


def load_artefacts() -> dict:
    """Load saved model, scaler, and data splits."""
    artefacts = {
        "model":           joblib.load(os.path.join(MODELS_DIR, "best_model.joblib")),
        "feature_columns": joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib")),
        "X_test":          np.load(os.path.join(MODELS_DIR, "X_test.npy")),
        "y_test":          np.load(os.path.join(MODELS_DIR, "y_test.npy")),
    }
    print(f"[explain] Model type : {type(artefacts['model']).__name__}")
    print(f"[explain] Test shape : {artefacts['X_test'].shape}")
    return artefacts


def build_explainer(model, X_test: np.ndarray):
    """
    Choose the right SHAP explainer:
      - TreeExplainer for tree-based models (RF, XGB) — fast & exact
      - KernelExplainer as fallback (slow, model-agnostic)

    Returns (explainer, shap_values, X_used) where X_used is the subset
    of X_test that was actually explained (may be smaller than X_test).
    """
    model_name = type(model).__name__
    if model_name in ("RandomForestRegressor", "XGBRegressor",
                      "GradientBoostingRegressor", "ExtraTreesRegressor"):
        print("[explain] Using TreeExplainer (fast).")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        X_used = X_test
    else:
        print("[explain] Using KernelExplainer (slow, model-agnostic).")
        background  = shap.sample(X_test, 100)
        explainer   = shap.KernelExplainer(model.predict, background)
        X_used      = X_test[:200]          # cap for speed
        shap_values = explainer.shap_values(X_used)

    return explainer, shap_values, X_used


def plot_summary_bar(shap_values, X_test, feature_columns, out_dir):
    """Bar plot of mean |SHAP| — global feature importance."""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_columns,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title("Global Feature Importance (mean |SHAP|)", fontsize=14, pad=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_summary_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Saved → {path}")


def plot_summary_beeswarm(shap_values, X_test, feature_columns, out_dir):
    """Beeswarm (dot) plot — shows distribution & direction of each feature."""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_columns,
        plot_type="dot",
        show=False,
        max_display=20,
    )
    plt.title("SHAP Value Distribution (beeswarm)", fontsize=14, pad=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_summary_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Saved → {path}")


def plot_individual(explainer, shap_values, X_test, feature_columns, out_dir, idx: int = 0):
    """Waterfall plot for a single test sample."""
    # Build SHAP Explanation object for modern API
    base_value = (
        explainer.expected_value
        if not hasattr(explainer.expected_value, "__len__")
        else explainer.expected_value[0]
    )
    exp = shap.Explanation(
        values=shap_values[idx],
        base_values=float(base_value),
        data=X_test[idx],
        feature_names=feature_columns,
    )
    plt.figure(figsize=(10, 5))
    shap.waterfall_plot(exp, show=False, max_display=15)
    plt.title(f"Individual Explanation — test sample #{idx}", fontsize=13, pad=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"shap_individual_sample_{idx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Saved → {path}")


def compute_feature_importance(shap_values, feature_columns) -> pd.DataFrame:
    """Return a DataFrame of mean |SHAP| sorted descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature":     feature_columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def write_feature_report(importance_df: pd.DataFrame, out_dir: str) -> None:
    """Write a markdown report of the top-10 depression risk drivers."""
    top10 = importance_df.head(10)

    lines = [
        "# Feature Analysis Report — Top Depression Risk Drivers\n\n",
        "Generated by SHAP (SHapley Additive exPlanations) on the test set.\n\n",
        "## Top 10 Contributing Factors\n\n",
        "| Rank | Feature | Mean |SHAP| |\n",
        "|------|---------|-------------|\n",
    ]
    for _, row in top10.iterrows():
        lines.append(f"| {int(row['rank'])} | {row['feature']} | {row['mean_abs_shap']:.4f} |\n")

    lines += [
        "\n## Interpretation Guide\n\n",
        "- **Higher mean |SHAP|** → greater average influence on the predicted PHQ-9 score.\n",
        "- Positive SHAP values push the prediction **higher** (more severe depression).\n",
        "- Negative SHAP values push the prediction **lower** (less severe depression).\n",
        "- See `shap_summary_beeswarm.png` for directional effects.\n\n",
        "## All Features Ranked\n\n",
        "| Rank | Feature | Mean |SHAP| |\n",
        "|------|---------|-------------|\n",
    ]
    for _, row in importance_df.iterrows():
        lines.append(f"| {int(row['rank'])} | {row['feature']} | {row['mean_abs_shap']:.4f} |\n")

    path = os.path.join(out_dir, "feature_analysis_report.md")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"[explain] Feature analysis report → {path}")

    # Also save CSV for programmatic use
    csv_path = os.path.join(out_dir, "shap_feature_importance.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"[explain] Feature importance CSV  → {csv_path}")


def main() -> None:
    print("=" * 60)
    print("  Mental Health AI — SHAP Explainability")
    print("=" * 60)

    art = load_artefacts()
    model           = art["model"]
    feature_columns = art["feature_columns"]
    X_test          = art["X_test"]

    explainer, shap_values, X_used = build_explainer(model, X_test)

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_summary_bar(shap_values, X_used, feature_columns, EXPLAIN_DIR)
    plot_summary_beeswarm(shap_values, X_used, feature_columns, EXPLAIN_DIR)
    plot_individual(explainer, shap_values, X_used, feature_columns, EXPLAIN_DIR, idx=0)
    plot_individual(explainer, shap_values, X_used, feature_columns, EXPLAIN_DIR, idx=1)

    # ── Importance table ─────────────────────────────────────────────────────
    importance_df = compute_feature_importance(shap_values, feature_columns)
    print("\n[explain] Top-10 Depression Risk Drivers:")
    print(importance_df.head(10).to_string(index=False))

    write_feature_report(importance_df, EXPLAIN_DIR)

    print("\n[explain] Done. Check models/explainability/ for all outputs.")


if __name__ == "__main__":
    main()
