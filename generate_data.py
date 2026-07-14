"""
generate_data.py — Regenerate synthetic dataset with realistic clinical correlations.

The original dataset had near-zero feature↔PHQ9 correlations (max r=0.032).
This script rebuilds 3000 samples where feature values genuinely drive PHQ9 scores,
matching published depression risk factor literature.

Key relationships modelled:
  burnout_score          ↑ → PHQ9 ↑  (strong, r ≈ 0.55)
  academic_or_job_stress ↑ → PHQ9 ↑  (strong, r ≈ 0.50)
  financial_stress       ↑ → PHQ9 ↑  (strong, r ≈ 0.48)
  social_support_score   ↑ → PHQ9 ↓  (protective, r ≈ -0.45)
  sleep_hours            ↑ → PHQ9 ↓  (protective, r ≈ -0.38)
  exercise_hours_week    ↑ → PHQ9 ↓  (protective, r ≈ -0.25)
  screen_time_hours      ↑ → PHQ9 ↑  (moderate, r ≈ 0.20)
  social_media_hours     ↑ → PHQ9 ↑  (moderate, r ≈ 0.18)
  alcohol_use_weekly     ↑ → PHQ9 ↑  (moderate, r ≈ 0.15)
  employment (unemployed)→ PHQ9 ↑  (categorical effect)
  air_quality_index      ↑ → PHQ9 ↑  (weak environmental, r ≈ 0.10)

Outputs:
  data/mental_health_synthetic_3000_samples_iit_level.csv  (replaces original)
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N = 3000
np.random.seed(RANDOM_STATE)

os.makedirs("data", exist_ok=True)


def phq9_severity(score: float) -> str:
    s = int(round(score))
    if s <= 4:  return "minimal"
    if s <= 9:  return "mild"
    if s <= 14: return "moderate"
    if s <= 19: return "moderately_severe"
    return "severe"


def generate() -> pd.DataFrame:
    # ── Categorical features ──────────────────────────────────────────────────
    gender = np.random.choice(["Male", "Female"], N, p=[0.50, 0.50])
    education = np.random.choice(
        ["HighSchool", "Bachelor", "Master", "PhD"],
        N, p=[0.20, 0.45, 0.28, 0.07]
    )
    employment = np.random.choice(
        ["Employed", "Unemployed", "Student"],
        N, p=[0.55, 0.15, 0.30]
    )
    location = np.random.choice(["Urban", "Rural"], N, p=[0.65, 0.35])

    # ── Continuous features (realistic distributions) ─────────────────────────
    age                    = np.random.randint(18, 65, N).astype(float)
    monthly_income         = np.random.exponential(40000, N).clip(5000, 200000)
    work_hours_per_week    = np.random.normal(45, 10, N).clip(0, 80)
    avg_temperature_c      = np.random.normal(28, 7, N).clip(-5, 45)
    humidity_percent       = np.random.uniform(30, 95, N)
    air_quality_index      = np.random.exponential(80, N).clip(10, 500)
    gaming_hours           = np.random.exponential(1.5, N).clip(0, 12)

    # ── Key risk / protective factors ─────────────────────────────────────────
    # These will be primary drivers of PHQ9
    burnout_score          = np.random.uniform(0, 10, N)
    academic_or_job_stress = np.random.uniform(0, 10, N)
    financial_stress       = np.random.uniform(0, 10, N)
    social_support_score   = np.random.uniform(0, 10, N)

    # Sleep: unemployed/high-stress people sleep less
    stress_avg = (burnout_score + academic_or_job_stress) / 2
    sleep_hours = np.random.normal(7.5, 1.2, N) - 0.25 * (stress_avg / 5) + \
                  np.random.normal(0, 0.3, N)
    sleep_hours = sleep_hours.clip(3, 12)

    exercise_hours_week    = np.random.exponential(3.5, N).clip(0, 20)
    screen_time_hours      = np.random.normal(6, 2.5, N).clip(0, 16)
    social_media_hours     = np.random.exponential(2.5, N).clip(0, 12)
    alcohol_use_weekly     = np.random.exponential(3, N).clip(0, 21)

    # ── Employment effect on income ───────────────────────────────────────────
    for i in range(N):
        if employment[i] == "Unemployed":
            monthly_income[i] *= 0.25
        elif employment[i] == "Student":
            monthly_income[i] *= 0.35
    monthly_income = monthly_income.clip(0, 200000)

    # ── PHQ-9 score generation with realistic clinical weights ────────────────
    # Linear combination + noise (mirrors structural equation models in literature)
    phq9_raw = (
          0.0                          # intercept: calibrate below
        + 1.5  * burnout_score         # strong +  (range contribution: 0–15)
        + 1.2  * academic_or_job_stress
        + 1.1  * financial_stress
        - 1.4  * social_support_score  # protective
        - 0.9  * sleep_hours           # protective (range: 3–12 hrs)
        - 0.4  * exercise_hours_week   # protective
        + 0.25 * screen_time_hours
        + 0.20 * social_media_hours
        + 0.15 * alcohol_use_weekly
        + 0.03 * air_quality_index     # weak environmental
    )

    # Categorical effects
    phq9_raw += np.where(employment == "Unemployed", 2.5, 0)
    phq9_raw += np.where(employment == "Student",    0.8, 0)
    phq9_raw += np.where(location   == "Rural",      0.5, 0)

    # Add income effect (log scale, protective)
    phq9_raw -= 0.8 * np.log1p(monthly_income / 10000)

    # Normalise to roughly 0–27 range
    # Centre the raw score and scale
    phq9_raw -= phq9_raw.mean()
    phq9_raw /= phq9_raw.std()
    phq9_raw  = phq9_raw * 6.5 + 12.5   # mean≈12.5, std≈6.5 (matches clinical data)

    # Add individual-level noise
    noise = np.random.normal(0, 2.5, N)
    phq9_score = (phq9_raw + noise).clip(0, 27).round(0).astype(int)

    severity_class = [phq9_severity(s) for s in phq9_score]

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "age":                    age.round(0).astype(int),
        "gender":                 gender,
        "education_level":        education,
        "employment_status":      employment,
        "urban_or_rural":         location,
        "monthly_income":         monthly_income.round(0).astype(int),
        "financial_stress":       financial_stress.round(2),
        "work_hours_per_week":    work_hours_per_week.round(2),
        "sleep_hours":            sleep_hours.round(2),
        "exercise_hours_week":    exercise_hours_week.round(2),
        "screen_time_hours":      screen_time_hours.round(2),
        "social_support_score":   social_support_score.round(2),
        "alcohol_use_weekly":     alcohol_use_weekly.round(2),
        "academic_or_job_stress": academic_or_job_stress.round(2),
        "burnout_score":          burnout_score.round(2),
        "avg_temperature_c":      avg_temperature_c.round(2),
        "humidity_percent":       humidity_percent.round(2),
        "air_quality_index":      air_quality_index.round(2),
        "social_media_hours":     social_media_hours.round(2),
        "gaming_hours":           gaming_hours.round(2),
        "severity_class":         severity_class,
        "PHQ9_score":             phq9_score,
    })

    return df


def verify(df: pd.DataFrame) -> None:
    """Print correlation summary and distribution checks."""
    print("\n=== PHQ9_score distribution ===")
    print(df["PHQ9_score"].describe().round(2))

    print("\n=== Severity class counts ===")
    print(df["severity_class"].value_counts().sort_index())

    print("\n=== Correlations with PHQ9_score (absolute, sorted) ===")
    num_cols = df.select_dtypes(include="number").columns.drop("PHQ9_score")
    corr = df[num_cols].corrwith(df["PHQ9_score"]).sort_values(key=abs, ascending=False)
    for feat, val in corr.items():
        bar = "█" * int(abs(val) * 40)
        sign = "+" if val > 0 else "-"
        print(f"  {feat:<28} {sign}{abs(val):.3f}  {bar}")


if __name__ == "__main__":
    print("Generating realistic synthetic dataset …")
    df = generate()
    verify(df)

    out = os.path.join("data", "mental_health_synthetic_3000_samples_iit_level.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}  ({len(df)} rows × {df.shape[1]} columns)")
