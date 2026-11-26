"""Pydantic v2 request and response schemas for the FastAPI churn prediction service."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Validated customer feature payload matching the Telco Customer Churn schema."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 94.5,
                "TotalCharges": 283.5,
            }
        }
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 if customer is a senior citizen, else 0")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120, description="Number of months customer has stayed")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No", "Yes", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes", "No internet service"]
    OnlineBackup: Literal["No", "Yes", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport: Literal["No", "Yes", "No internet service"]
    StreamingTV: Literal["No", "Yes", "No internet service"]
    StreamingMovies: Literal["No", "Yes", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0.0, le=500.0, description="Monthly billing amount in USD")
    TotalCharges: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=50000.0,
        description="Cumulative billing amount in USD (optional when tenure is 0)",
    )


class RiskThresholds(BaseModel):
    low_max: float
    high_min: float


class PredictionResponse(BaseModel):
    churn_prediction: int = Field(description="Binary prediction (1 = Churn, 0 = Retain)")
    churn_label: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0.0, le=1.0)
    risk_category: Literal["Low", "Medium", "High"]
    decision_threshold: float
    risk_thresholds: RiskThresholds
    model_version: str
    model_name: str


class ExplanationDriver(BaseModel):
    feature: str
    display_name: str
    transformed_value: float
    shap_value: float
    abs_shap: float
    direction: Literal["increases_churn_risk", "decreases_churn_risk"]


class ShapExplanation(BaseModel):
    explainer_type: str
    base_value_log_odds: float
    top_drivers: List[ExplanationDriver]
    top_risk_increasing_factors: List[str]
    top_risk_decreasing_factors: List[str]
    retention_considerations: List[str]
    causality_disclaimer: str


class DetailedPredictionResponse(PredictionResponse):
    explanation: ShapExplanation


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_version: Optional[str] = None
    model_name: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_version: str
    selected_model: str
    classifier_class: str
    trained_at_utc: str
    selection_criterion: str
    raw_feature_columns: List[str]
    risk_thresholds: Dict[str, Any]
    selected_model_test_metrics: Dict[str, Any]
