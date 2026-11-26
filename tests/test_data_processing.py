"""Tests for raw data cleaning, missing TotalCharges handling, and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RAW_DATA_PATH, RAW_FEATURE_COLUMNS, TARGET_COL
from src.data_processing import clean_data, load_raw_data
from src.feature_engineering import TelcoFeatureEngineer, build_preprocessor


def _sample_record(**overrides):
    base = {
        "customerID": "0001-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": "1026.0",
        "Churn": "Yes",
    }
    base.update(overrides)
    return base


def test_clean_data_handles_blank_total_charges_for_zero_tenure():
    df = pd.DataFrame(
        [
            _sample_record(customerID="A", tenure=0, TotalCharges="   ", Churn="No"),
            _sample_record(customerID="B", tenure=5, TotalCharges=" 427.5 ", Churn="Yes"),
        ]
    )
    cleaned = clean_data(df, include_target=True)
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert cleaned.loc[1, "TotalCharges"] == pytest.approx(427.5)
    assert list(cleaned[TARGET_COL]) == [0, 1]


def test_clean_data_drops_duplicates_and_validates_required_columns():
    rec = _sample_record()
    df = pd.DataFrame([rec, rec])
    cleaned = clean_data(df, include_target=True)
    assert len(cleaned) == 1

    with pytest.raises(ValueError, match="Missing required feature columns"):
        clean_data(df.drop(columns=["Contract"]), include_target=True)


def test_raw_dataset_cleans_without_remaining_nulls():
    if not RAW_DATA_PATH.exists():
        pytest.skip("Raw CSV not present in CI environment.")
    raw = load_raw_data()
    cleaned = clean_data(raw, include_target=True)
    assert len(cleaned) == 7043
    assert cleaned.isna().sum().sum() == 0
    assert set(cleaned[TARGET_COL].unique()) == {0, 1}


def test_feature_engineer_creates_expected_columns_and_handles_missing_numeric():
    df = pd.DataFrame(
        [
            _sample_record(tenure=0, TotalCharges=np.nan, PaymentMethod="Credit card (automatic)", TechSupport="Yes"),
            _sample_record(tenure=10, MonthlyCharges=80.0, TotalCharges=800.0, OnlineSecurity="No", TechSupport="No"),
        ]
    )[RAW_FEATURE_COLUMNS]

    fe = TelcoFeatureEngineer()
    transformed = fe.fit_transform(df)
    for expected in ["avg_monthly_spend", "charge_ratio", "service_count", "has_support_bundle", "is_auto_pay", "tenure_group"]:
        assert expected in transformed.columns

    assert transformed.loc[0, "TotalCharges"] == 0.0
    assert transformed.loc[0, "is_auto_pay"] == 1
    assert transformed.loc[0, "has_support_bundle"] == 1
    assert transformed.loc[1, "is_auto_pay"] == 0
    assert transformed.loc[1, "has_support_bundle"] == 0
    assert transformed.loc[1, "avg_monthly_spend"] == pytest.approx(80.0)


def test_preprocessor_outputs_finite_dense_matrix_with_reordered_columns():
    df = pd.DataFrame([_sample_record(), _sample_record(Contract="Two year", tenure=48, TotalCharges="4000")])
    X = df[RAW_FEATURE_COLUMNS]
    preprocessor = build_preprocessor(scale_numeric=True)
    X_fit = preprocessor.fit_transform(X)

    # Reverse column order at transform time; output must be identical
    X_reordered = X[list(reversed(RAW_FEATURE_COLUMNS))]
    X_trans = preprocessor.transform(X_reordered)
    assert np.allclose(X_fit, X_trans)
    assert np.isfinite(X_trans).all()
