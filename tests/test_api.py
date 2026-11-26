"""Automated API endpoint tests for FastAPI validation, inference, SHAP, and error handling."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
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


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_model_info_endpoint():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert "selected_model" in body
    assert "selected_model_test_metrics" in body


def test_predict_endpoint_valid_payload():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["churn_prediction"] in {0, 1}
    assert body["churn_label"] in {"Yes", "No"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_category"] in {"Low", "Medium", "High"}


def test_predict_explain_endpoint():
    resp = client.post("/predict/explain?top_n_drivers=5", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert "explanation" in body
    assert len(body["explanation"]["top_drivers"]) == 5


def test_predict_batch_endpoint():
    resp = client.post("/predict/batch", json={"customers": [VALID_PAYLOAD, VALID_PAYLOAD]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["churn_probability"] == body["predictions"][1]["churn_probability"]


def test_predict_rejects_invalid_category_and_negative_charges():
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["Contract"] = "Lifetime"
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422

    bad_charges = dict(VALID_PAYLOAD)
    bad_charges["MonthlyCharges"] = -25.0
    resp2 = client.post("/predict", json=bad_charges)
    assert resp2.status_code == 422


def test_predict_rejects_invalid_thresholds():
    resp = client.post("/predict?low_threshold=0.80&high_threshold=0.40", json=VALID_PAYLOAD)
    assert resp.status_code == 400
