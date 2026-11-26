"""FastAPI service exposing model health, metadata, prediction, and SHAP explanation endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    DetailedPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.config import DEFAULT_HIGH_RISK_THRESHOLD, DEFAULT_LOW_RISK_THRESHOLD
from src.predict import ChurnPredictor

_predictor: Optional[ChurnPredictor] = None


def get_predictor() -> ChurnPredictor:
    """Return initialized ChurnPredictor or raise 503 if artifacts are unavailable."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = ChurnPredictor()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Model artifact unavailable: {exc}",
            ) from exc
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload the trained pipeline on application startup when available."""
    global _predictor
    try:
        _predictor = ChurnPredictor()
    except FileNotFoundError:
        _predictor = None
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-style inference and SHAP explainability service for the "
        "Telco Customer Churn Prediction System."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_thresholds(low_threshold: float, high_threshold: float) -> None:
    if not (0.0 < low_threshold < high_threshold < 1.0):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid risk thresholds: low_threshold ({low_threshold}) must be strictly "
                f"less than high_threshold ({high_threshold}), both within (0, 1)."
            ),
        )


@app.get("/", tags=["System"])
def root():
    return {
        "service": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "predict_explain": "/predict/explain",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Report API readiness and model artifact status."""
    try:
        predictor = get_predictor()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version=predictor.model_version,
            model_name=predictor.model_name,
        )
    except HTTPException:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_version=None,
            model_name=None,
        )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info() -> ModelInfoResponse:
    """Return active model metadata, schema, operational thresholds, and test metrics."""
    predictor = get_predictor()
    meta = predictor.metadata
    if not meta:
        raise HTTPException(status_code=404, detail="Model metadata file not found.")
    return ModelInfoResponse(
        model_version=meta["model_version"],
        selected_model=meta["selected_model"],
        classifier_class=meta["classifier_class"],
        trained_at_utc=meta["trained_at_utc"],
        selection_criterion=meta["selection_criterion"],
        raw_feature_columns=meta["raw_feature_columns"],
        risk_thresholds=meta["risk_thresholds"],
        selected_model_test_metrics=meta["selected_model_test_metrics"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(
    customer: CustomerFeatures,
    decision_threshold: float = Query(default=0.50, gt=0.0, lt=1.0),
    low_threshold: float = Query(default=DEFAULT_LOW_RISK_THRESHOLD, gt=0.0, lt=1.0),
    high_threshold: float = Query(default=DEFAULT_HIGH_RISK_THRESHOLD, gt=0.0, lt=1.0),
) -> PredictionResponse:
    """Predict churn probability, binary class, and operational risk tier for a customer."""
    _validate_thresholds(low_threshold, high_threshold)
    predictor = get_predictor()
    try:
        result = predictor.predict_one(
            customer.model_dump(),
            decision_threshold=decision_threshold,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            include_explanation=False,
        )
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@app.post("/predict/explain", response_model=DetailedPredictionResponse, tags=["Inference"])
def predict_and_explain(
    customer: CustomerFeatures,
    decision_threshold: float = Query(default=0.50, gt=0.0, lt=1.0),
    low_threshold: float = Query(default=DEFAULT_LOW_RISK_THRESHOLD, gt=0.0, lt=1.0),
    high_threshold: float = Query(default=DEFAULT_HIGH_RISK_THRESHOLD, gt=0.0, lt=1.0),
    top_n_drivers: int = Query(default=8, ge=3, le=20),
) -> DetailedPredictionResponse:
    """Predict churn and attach customer-level SHAP explanations and retention considerations."""
    _validate_thresholds(low_threshold, high_threshold)
    predictor = get_predictor()
    try:
        result = predictor.predict_one(
            customer.model_dump(),
            decision_threshold=decision_threshold,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            include_explanation=True,
            top_n_drivers=top_n_drivers,
        )
        return DetailedPredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference/SHAP failed: {exc}") from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(
    payload: BatchPredictionRequest,
    decision_threshold: float = Query(default=0.50, gt=0.0, lt=1.0),
    low_threshold: float = Query(default=DEFAULT_LOW_RISK_THRESHOLD, gt=0.0, lt=1.0),
    high_threshold: float = Query(default=DEFAULT_HIGH_RISK_THRESHOLD, gt=0.0, lt=1.0),
) -> BatchPredictionResponse:
    """Predict churn for a batch of up to 500 validated customer records."""
    _validate_thresholds(low_threshold, high_threshold)
    predictor = get_predictor()
    try:
        records = [c.model_dump() for c in payload.customers]
        preds = predictor.predict_batch(
            records,
            decision_threshold=decision_threshold,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        return BatchPredictionResponse(
            count=len(preds),
            predictions=[PredictionResponse(**p) for p in preds],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {exc}") from exc
