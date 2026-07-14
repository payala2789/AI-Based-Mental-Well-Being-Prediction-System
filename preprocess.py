"""
preprocess.py — Data loading, cleaning, encoding, scaling, and train/test split.

Outputs:
  models/scaler.joblib
  models/label_encoders.joblib
  models/feature_columns.joblib
  models/X_train.npy / X_test.npy / y_train.npy / y_test.npy
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "mental_health_synthetic_3000_samples_iit_level.csv")
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_COL        = "PHQ9_score"
DROP_COLS         = ["severity_class"]          # leaks target; derived at inference
CATEGORICAL_COLS  = ["gender", "education_level", "employment_status", "urban_or_rural"]
RANDOM_STATE      = 42
TEST_SIZE         = 0.20


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV and perform basic sanity checks."""
    df = pd.read_csv(path)
    print(f"[preprocess] Loaded {len(df):,} rows × {df.shape[1]} columns.")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with median; categorical with mode."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"[preprocess] Missing values handled.  Remaining NaN: {df.isna().sum().sum()}")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    Returns the transformed DataFrame and a dict of fitted LabelEncoders.
    """
    encoders: dict = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"[preprocess]   Encoded '{col}': {list(le.classes_)}")
    return df, encoders


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite features that capture clinical insight.
    These pre-computed combinations help the model find patterns faster.
    """
    # Combined stress burden (average of three stress axes)
    df["stress_index"] = (
        df["burnout_score"] +
        df["academic_or_job_stress"] +
        df["financial_stress"]
    ) / 3

    # Lifestyle health score — higher = better mental health habits
    df["lifestyle_score"] = (
        (df["sleep_hours"] / 8.0) +
        (df["exercise_hours_week"] / 5.0) -
        (df["alcohol_use_weekly"] / 10.0)
    )

    # Total passive screen time (risk factor)
    df["total_screen_load"] = (
        df["screen_time_hours"] +
        df["social_media_hours"] +
        df["gaming_hours"]
    )

    # Social isolation index — low support + high screen time
    df["isolation_index"] = (
        (10 - df["social_support_score"]) +
        df["social_media_hours"]
    ) / 2

    # Work-life imbalance
    df["work_life_ratio"] = df["work_hours_per_week"] / (
        df["sleep_hours"] * 7 + 1
    )

    print(f"[preprocess] Feature engineering: +5 composite features added.")
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop leaky / target columns and return X, y."""
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns])
    return X, y


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit scaler on train, transform both splits."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


def run_preprocessing() -> dict:
    """
    Full pipeline: load → clean → encode → split → scale → save artefacts.
    Returns a dict with arrays and metadata (useful when imported by train.py).
    """
    df = load_data()
    df = handle_missing(df)
    df, encoders = encode_categoricals(df)
    df = engineer_features(df)
    X, y = build_feature_matrix(df)

    feature_columns = list(X.columns)
    print(f"[preprocess] Feature count: {len(feature_columns)}")
    print(f"[preprocess] Features: {feature_columns}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"[preprocess] Train: {X_train.shape}  |  Test: {X_test.shape}")

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # ── Persist artefacts ────────────────────────────────────────────────────
    joblib.dump(scaler,          os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(encoders,        os.path.join(MODELS_DIR, "label_encoders.joblib"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    np.save(os.path.join(MODELS_DIR, "X_train.npy"), X_train_sc)
    np.save(os.path.join(MODELS_DIR, "X_test.npy"),  X_test_sc)
    np.save(os.path.join(MODELS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(MODELS_DIR, "y_test.npy"),  y_test)
    print("[preprocess] Artefacts saved to models/")

    return {
        "X_train": X_train_sc,
        "X_test":  X_test_sc,
        "y_train": y_train,
        "y_test":  y_test,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "encoders": encoders,
    }


if __name__ == "__main__":
    run_preprocessing()
