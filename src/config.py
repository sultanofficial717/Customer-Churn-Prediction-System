"""Centralized configuration, file paths, feature schema, and operational thresholds."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "telco_churn_clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_VERSION = "v1.0.0"
MODEL_ARTIFACT_PATH = MODELS_DIR / f"churn_pipeline_{MODEL_VERSION}.joblib"
LATEST_MODEL_PATH = MODELS_DIR / "churn_pipeline_latest.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVALUATION_DIR = REPORTS_DIR / "evaluation"

EDA_SUMMARY_PATH = EVALUATION_DIR / "eda_summary.json"
EVALUATION_SUMMARY_PATH = EVALUATION_DIR / "evaluation_summary.json"
MODEL_COMPARISON_CSV_PATH = EVALUATION_DIR / "model_comparison.csv"
SHAP_SUMMARY_PATH = EVALUATION_DIR / "shap_summary.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

ID_COL = "customerID"
TARGET_COL = "Churn"

RAW_NUMERIC_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

RAW_BINARY_INT_FEATURES: List[str] = [
    "SeniorCitizen",
]

RAW_CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

RAW_FEATURE_COLUMNS: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORY_OPTIONS: Dict[str, List[str]] = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# Operational thresholds for translating churn probability into retention tiers.
# These are configurable business defaults rather than universal industry constants.
DEFAULT_LOW_RISK_THRESHOLD: float = float(os.getenv("LOW_RISK_THRESHOLD", "0.30"))
DEFAULT_HIGH_RISK_THRESHOLD: float = float(os.getenv("HIGH_RISK_THRESHOLD", "0.65"))
