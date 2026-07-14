# Mental Health AI — PHQ-9 Risk Prediction System

> **Disclaimer:** This system is intended for research and informational purposes only. It does not provide medical diagnosis. Always consult a qualified mental health professional.

---

## Project Overview

| Component | Technology |
|-----------|-----------|
| ML Models | RandomForest + XGBoost (scikit-learn / xgboost) |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Plain HTML + JavaScript |

**Target:** Predict PHQ-9 depression score (0–27) from 20 lifestyle and environmental features, then classify into severity buckets and explain the prediction with SHAP.

---

## Folder Structure

```
mental-health-ai/
├── data/
│   └── mental_health_synthetic_3000_samples_iit_level.csv
├── notebooks/            # (optional) Jupyter exploration
├── models/               # Generated artefacts (auto-created by train.py)
│   └── explainability/   # SHAP plots + feature report
├── backend/
│   ├── __init__.py
│   ├── main.py           # FastAPI app
│   ├── schemas.py        # Pydantic request/response models
│   └── predictor.py      # Model loading + SHAP inference
├── frontend/
│   └── index.html        # Single-page assessment UI
├── preprocess.py         # Data loading, encoding, scaling
├── train.py              # Model training + evaluation + saving
├── validate.py           # K-Fold CV, overfitting check, residuals
├── explain.py            # SHAP plots + feature analysis report
├── requirements.txt
└── README.md
```

---

## Quick Start (Local)

### 1 — Prerequisites

```bash
python --version      # 3.10 or 3.11 recommended
pip install -r requirements.txt
```

### 2 — Train the models

```bash
# From the project root (mental-health-ai/)
python train.py
```

This will:
- Load and preprocess the dataset
- Train RandomForest + XGBoost regressors
- Evaluate both on the test set (MAE, RMSE, R²)
- Save the best model to `models/best_model.joblib`
- Save scaler, encoders, and feature list to `models/`
- Write `models/training_report.md`

### 3 — Validate the models

```bash
python validate.py
```

Outputs saved to `models/validation/`:

| File | What it checks |
|------|---------------|
| `validation_report.md` | Full summary |
| `plots/01_cv_scores_per_fold.png` | 5-fold CV scores per fold |
| `plots/02_overfit_train_vs_test.png` | Overfitting detector |
| `plots/03_xgb_boosting_rounds.png` | XGBoost round curves (like epochs) |
| `plots/04_learning_curve_*.png` | Data sufficiency curves |
| `plots/05_residuals_*.png` | Prediction bias analysis |
| `plots/06_pred_distribution_*.png` | Actual vs predicted coverage |

### 4 — Generate SHAP explanations

```bash
python explain.py
```

Outputs saved to `models/explainability/`:

| File | Description |
|------|-------------|
| `shap_summary_bar.png` | Global mean \|SHAP\| bar chart |
| `shap_summary_beeswarm.png` | Directional beeswarm plot |
| `shap_individual_sample_0.png` | Waterfall plot — sample 0 |
| `shap_individual_sample_1.png` | Waterfall plot — sample 1 |
| `shap_feature_importance.csv` | All features ranked by importance |
| `feature_analysis_report.md` | Human-readable markdown report |

### 5 — Run the backend API

```bash
# From the project root
uvicorn backend.main:app --reload --port 8000
```

API is live at http://localhost:8000

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Model load status |
| `/features` | GET | Expected input features |
| `/predict` | POST | PHQ-9 prediction + SHAP |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc UI |

#### Example curl request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 28,
    "gender": "Female",
    "education_level": "Bachelor",
    "employment_status": "Employed",
    "urban_or_rural": "Urban",
    "monthly_income": 45000,
    "financial_stress": 7,
    "work_hours_per_week": 55,
    "sleep_hours": 5.5,
    "exercise_hours_week": 1,
    "screen_time_hours": 9,
    "social_support_score": 3,
    "alcohol_use_weekly": 3,
    "academic_or_job_stress": 8,
    "burnout_score": 9,
    "avg_temperature_c": 32,
    "humidity_percent": 70,
    "air_quality_index": 150,
    "social_media_hours": 5,
    "gaming_hours": 1
  }'
```

#### Example response

```json
{
  "phq9_score": 17.4,
  "severity": "Moderately Severe",
  "severity_level": 4,
  "top_features": [
    {"feature": "burnout_score", "shap_value": 2.31, "direction": "increases risk"},
    {"feature": "sleep_hours",   "shap_value": -1.87, "direction": "decreases risk"},
    ...
  ],
  "recommendation": "Moderately severe symptoms detected. Please reach out to a mental health professional promptly.",
  "disclaimer": "This tool does not provide medical diagnosis..."
}
```

### 6 — Open the frontend

With the API running, open `frontend/index.html` in any browser:

```bash
open frontend/index.html          # macOS
xdg-open frontend/index.html      # Linux
start frontend/index.html         # Windows
```

Fill in the form and click **Assess My Risk**.

---

## PHQ-9 Severity Scale

| Score | Severity |
|-------|----------|
| 0–4 | Minimal |
| 5–9 | Mild |
| 10–14 | Moderate |
| 15–19 | Moderately Severe |
| 20–27 | Severe |

---

## Input Features

| Feature | Type | Range / Options |
|---------|------|----------------|
| age | numeric | 10–100 |
| gender | categorical | Female, Male, Other |
| education_level | categorical | HighSchool, Bachelor, Master, PhD, Other |
| employment_status | categorical | Employed, Unemployed, Student, Self-Employed |
| urban_or_rural | categorical | Urban, Rural |
| monthly_income | numeric | ≥ 0 |
| financial_stress | numeric | 0–10 |
| work_hours_per_week | numeric | 0–168 |
| sleep_hours | numeric | 0–24 |
| exercise_hours_week | numeric | ≥ 0 |
| screen_time_hours | numeric | 0–24 |
| social_support_score | numeric | 0–10 |
| alcohol_use_weekly | numeric | ≥ 0 |
| academic_or_job_stress | numeric | 0–10 |
| burnout_score | numeric | 0–10 |
| avg_temperature_c | numeric | any |
| humidity_percent | numeric | 0–100 |
| air_quality_index | numeric | ≥ 0 |
| social_media_hours | numeric | 0–24 |
| gaming_hours | numeric | 0–24 |

---

## Ethical Considerations

- The dataset is **synthetic** — real-world performance may differ.
- The system is **not a substitute** for clinical assessment.
- Severe predictions trigger a **helpline message** prominently in the UI.
- No personally identifiable data is stored or logged by the API.

---

## Helplines

| Country | Service | Contact |
|---------|---------|---------|
| India | iCall | 9152987821 |
| India | Vandrevala Foundation | 1860-2662-345 |
| Global | Find a Helpline | https://www.findahelpline.com |
| USA | NAMI Helpline | 1-800-950-NAMI |

---

*Built with Python 3.11 · FastAPI · scikit-learn · XGBoost · SHAP*
