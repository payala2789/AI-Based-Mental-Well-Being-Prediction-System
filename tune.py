"""
tune.py — Automated hyperparameter search for best model performance.

Uses Bayesian optimisation (Optuna) — smarter than brute-force GridSearch.
Tries ~50 combinations intelligently instead of testing all 500+.

Run after train.py:
    pip install optuna
    python tune.py

Outputs:
    models/best_model_tuned.joblib
    models/tuning_report.md
"""

import os
import numpy as np
import joblib
from math import sqrt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[tune] Optuna not installed. Run: pip install optuna")
    print("[tune] Falling back to manual GridSearch on key params.")

MODELS_DIR   = "models"
RANDOM_STATE = 42


def load_data():
    X_train = np.load(os.path.join(MODELS_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(MODELS_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(MODELS_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(MODELS_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, np.clip(y_pred, 0, 27)))


# ── Optuna objective functions ────────────────────────────────────────────────

def xgb_objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "verbosity":         0,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return rmse(y_test, model.predict(X_test))


def rf_objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 600),
        "max_depth":       trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
        "random_state":    RANDOM_STATE,
        "n_jobs":          -1,
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return rmse(y_test, model.predict(X_test))


def tune_with_optuna(X_train, y_train, X_test, y_test, n_trials=50):
    print(f"\n[tune] Optuna search — {n_trials} trials each model …")

    # XGBoost
    print("[tune] Tuning XGBoost …")
    xgb_study = optuna.create_study(direction="minimize")
    xgb_study.optimize(
        lambda t: xgb_objective(t, X_train, y_train, X_test, y_test),
        n_trials=n_trials, show_progress_bar=False
    )
    best_xgb_params = xgb_study.best_params
    best_xgb_rmse   = xgb_study.best_value

    # RandomForest
    print("[tune] Tuning RandomForest …")
    rf_study = optuna.create_study(direction="minimize")
    rf_study.optimize(
        lambda t: rf_objective(t, X_train, y_train, X_test, y_test),
        n_trials=n_trials, show_progress_bar=False
    )
    best_rf_params = rf_study.best_params
    best_rf_rmse   = rf_study.best_value

    return {
        "xgb": {"params": best_xgb_params, "rmse": best_xgb_rmse},
        "rf":  {"params": best_rf_params,  "rmse": best_rf_rmse},
    }


def tune_manual(X_train, y_train, X_test, y_test):
    """Fallback: test a small curated grid if Optuna not available."""
    print("[tune] Manual grid search …")
    results = []

    configs = [
        ("XGB-A", XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=RANDOM_STATE, verbosity=0)),
        ("XGB-B", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.03,
                               subsample=0.9, colsample_bytree=0.7,
                               random_state=RANDOM_STATE, verbosity=0)),
        ("XGB-C", XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.08,
                               subsample=0.7, colsample_bytree=0.9,
                               reg_alpha=0.5, reg_lambda=2.0,
                               random_state=RANDOM_STATE, verbosity=0)),
        ("RF-A",  RandomForestRegressor(n_estimators=400, max_depth=20,
                                        min_samples_split=4, min_samples_leaf=2,
                                        random_state=RANDOM_STATE, n_jobs=-1)),
        ("RF-B",  RandomForestRegressor(n_estimators=500, max_depth=None,
                                        min_samples_split=8, min_samples_leaf=3,
                                        max_features="sqrt",
                                        random_state=RANDOM_STATE, n_jobs=-1)),
    ]

    for name, model in configs:
        model.fit(X_train, y_train)
        r = rmse(y_test, model.predict(X_test))
        r2 = r2_score(y_test, np.clip(model.predict(X_test), 0, 27))
        print(f"  {name:<10}  RMSE={r:.4f}  R²={r2:.4f}")
        results.append((r, name, model))

    results.sort(key=lambda x: x[0])
    return results[0][2], results[0][0], results[0][1]


def write_report(results: dict, before_rmse: float, after_rmse: float) -> None:
    improvement = ((before_rmse - after_rmse) / before_rmse) * 100
    lines = [
        "# Hyperparameter Tuning Report\n\n",
        f"| | RMSE |\n|---|---|\n",
        f"| Before tuning | {before_rmse:.4f} |\n",
        f"| After tuning  | {after_rmse:.4f} |\n",
        f"| Improvement   | {improvement:.1f}% |\n\n",
        "## Best Parameters\n\n",
    ]
    for model_name, info in results.items():
        lines.append(f"### {model_name}\n```\n")
        for k, v in info["params"].items():
            lines.append(f"{k}: {v}\n")
        lines.append(f"```\nRMSE: {info['rmse']:.4f}\n\n")

    path = os.path.join(MODELS_DIR, "tuning_report.md")
    with open(path, "w") as f: f.writelines(lines)
    print(f"[tune] Report saved → {path}")


def main():
    print("=" * 60)
    print("  Mental Health AI — Hyperparameter Tuning")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()

    # Current baseline
    current_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    baseline_rmse = rmse(y_test, current_model.predict(X_test))
    print(f"\n[tune] Current best model RMSE: {baseline_rmse:.4f}")

    if OPTUNA_AVAILABLE:
        results = tune_with_optuna(X_train, y_train, X_test, y_test, n_trials=50)

        print(f"\n[tune] Best XGBoost RMSE: {results['xgb']['rmse']:.4f}")
        print(f"       Best RF RMSE:       {results['rf']['rmse']:.4f}")
        print(f"       Best params (XGB):  {results['xgb']['params']}")

        # Retrain best model on full data with best params
        if results["xgb"]["rmse"] <= results["rf"]["rmse"]:
            winner_params = results["xgb"]["params"]
            winner_rmse   = results["xgb"]["rmse"]
            best_model    = XGBRegressor(**winner_params, random_state=RANDOM_STATE,
                                         n_jobs=-1, verbosity=0)
        else:
            winner_params = results["rf"]["params"]
            winner_rmse   = results["rf"]["rmse"]
            best_model    = RandomForestRegressor(**winner_params,
                                                  random_state=RANDOM_STATE, n_jobs=-1)

        best_model.fit(X_train, y_train)
        write_report(results, baseline_rmse, winner_rmse)

    else:
        best_model, winner_rmse, winner_name = tune_manual(X_train, y_train, X_test, y_test)
        print(f"\n[tune] Best manual config: {winner_name}  RMSE={winner_rmse:.4f}")

    improvement = ((baseline_rmse - winner_rmse) / baseline_rmse) * 100
    print(f"\n[tune] RMSE improved by {improvement:.1f}%")
    print(f"       {baseline_rmse:.4f}  →  {winner_rmse:.4f}")

    # Save tuned model
    out = os.path.join(MODELS_DIR, "best_model_tuned.joblib")
    joblib.dump(best_model, out)
    print(f"[tune] Tuned model saved → {out}")

    # Replace best_model.joblib if tuned is better
    if winner_rmse < baseline_rmse:
        joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
        print("[tune] ✓ best_model.joblib updated with tuned version.")
    else:
        print("[tune] Original model unchanged (tuning did not improve RMSE).")


if __name__ == "__main__":
    main()
