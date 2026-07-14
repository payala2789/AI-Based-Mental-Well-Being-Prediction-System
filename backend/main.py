"""
main.py — FastAPI application entry-point.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /              → health check
    GET  /health        → health check (JSON)
    POST /predict       → PHQ-9 prediction + explanation
    GET  /features      → list expected input features
    GET  /docs          → Swagger UI (auto-generated)
"""

import os
import sys
import time
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.schemas import PredictionRequest, PredictionResponse, TopFeature
from backend.predictor import predictor

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mental Health Risk Prediction API",
    description=(
        "Predicts PHQ-9 depression score and severity level from lifestyle "
        "and environmental factors. Uses RandomForest / XGBoost + SHAP explanations.\n\n"
        "**Disclaimer:** This API does not provide medical diagnosis."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow frontend on any origin during development) ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: pre-load model ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Loading ML model and SHAP explainer …")
    predictor.load()
    logger.info("Model ready.")


# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Mental Health Risk Prediction API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    model_loaded = predictor._loaded
    return {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "model_type": type(predictor.model).__name__ if model_loaded else None,
    }


@app.get("/features", tags=["Info"])
async def list_features():
    """Return the ordered list of features the model expects."""
    return {
        "feature_count": len(predictor.feature_columns),
        "features": predictor.feature_columns,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict PHQ-9 depression score and severity from lifestyle/environmental inputs.

    - **phq9_score**: Continuous score 0–27
    - **severity**: Minimal | Mild | Moderate | Moderately Severe | Severe
    - **top_features**: Top 10 SHAP contributors
    - **recommendation**: Guidance based on severity level
    """
    feature_dict = request.model_dump()

    try:
        result = predictor.predict(feature_dict)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    logger.info(
        "Prediction | PHQ9=%.1f | Severity=%s | Gender=%s | Age=%s",
        result["phq9_score"], result["severity"],
        request.gender, request.age,
    )

    return PredictionResponse(
        phq9_score=result["phq9_score"],
        severity=result["severity"],
        severity_level=result["severity_level"],
        top_features=[TopFeature(**f) for f in result["top_features"]],
        recommendation=result["recommendation"],
        disclaimer=result["disclaimer"],
    )


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )
