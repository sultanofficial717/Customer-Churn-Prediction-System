"""Tests for model artifact loading, prediction determinism, risk bucketing, and SHAP explanations."""

from __future__ import annotations

import pytest

from src.config import LATEST_MODEL_PATH, RAW_FEATURE_COLUMNS
from src.predict import ChurnPredictor, classify_risk_category


HIGH_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
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
    "MonthlyCharges": 95.0,
    "TotalCharges": 190.0,
}

LOW_RISK_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 65,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 68.0,
    "TotalCharges": 4420.0,
}


def test_risk_category_classification_and_validation():
    assert classify_risk_category(0.15, low_threshold=0.30, high_threshold=0.65) == "Low"
    assert classify_risk_category(0.45, low_threshold=0.30, high_threshold=0.65) == "Medium"
    assert classify_risk_category(0.80, low_threshold=0.30, high_threshold=0.65) == "High"

    with pytest.raises(ValueError):
        classify_risk_category(0.50, low_threshold=0.70, high_threshold=0.60)


def test_predictor_loads_and_returns_deterministic_predictions():
    assert LATEST_MODEL_PATH.exists(), "Trained model artifact must exist."
    predictor = ChurnPredictor()

    res1 = predictor.predict_one(HIGH_RISK_CUSTOMER)
    # Shuffle key insertion order to verify strict column ordering inside inference
    shuffled_customer = {k: HIGH_RISK_CUSTOMER[k] for k in reversed(RAW_FEATURE_COLUMNS)}
    res2 = predictor.predict_one(shuffled_customer)

    assert res1["churn_probability"] == res2["churn_probability"]
    assert res1["churn_prediction"] == res2["churn_prediction"]
    assert res1["risk_category"] == res2["risk_category"]
    assert 0.0 <= res1["churn_probability"] <= 1.0


def test_high_risk_profile_scores_higher_than_loyal_profile():
    predictor = ChurnPredictor()
    high_res = predictor.predict_one(HIGH_RISK_CUSTOMER)
    low_res = predictor.predict_one(LOW_RISK_CUSTOMER)
    assert high_res["churn_probability"] > low_res["churn_probability"]


def test_shap_explanation_is_attached_and_well_formed():
    predictor = ChurnPredictor()
    result = predictor.predict_one(HIGH_RISK_CUSTOMER, include_explanation=True, top_n_drivers=6)
    exp = result["explanation"]
    assert len(exp["top_drivers"]) == 6
    assert "causality_disclaimer" in exp
    assert len(exp["retention_considerations"]) >= 1
    for driver in exp["top_drivers"]:
        assert driver["direction"] in {"increases_churn_risk", "decreases_churn_risk"}
