"""Reusable, leakage-free feature engineering and preprocessing pipeline."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    RAW_BINARY_INT_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    RAW_FEATURE_COLUMNS,
    RAW_NUMERIC_FEATURES,
)

ENGINEERED_NUMERIC_FEATURES: List[str] = [
    "avg_monthly_spend",
    "charge_ratio",
    "service_count",
]

ENGINEERED_BINARY_FEATURES: List[str] = [
    "has_support_bundle",
    "is_auto_pay",
]

ENGINEERED_CATEGORICAL_FEATURES: List[str] = [
    "tenure_group",
]

ALL_NUMERIC_FEATURES: List[str] = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
ALL_BINARY_FEATURES: List[str] = RAW_BINARY_INT_FEATURES + ENGINEERED_BINARY_FEATURES
ALL_CATEGORICAL_FEATURES: List[str] = RAW_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless row-wise feature engineering transformer for Telco customer records.

    All transformations depend strictly on each customer's own row, preventing
    train/test data leakage during cross-validation and production inference.
    """

    SERVICE_YES_COLUMNS = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    def fit(self, X: pd.DataFrame, y=None) -> "TelcoFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=RAW_FEATURE_COLUMNS)

        df = X.copy()

        # Guarantee consistent column ordering and presence
        missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required feature columns: {missing}")
        df = df[RAW_FEATURE_COLUMNS].copy()

        # Defensive numeric coercion for raw inference inputs
        for col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Deterministic rule: 0-tenure customers have 0.0 TotalCharges when missing
        zero_tenure_missing = (df["tenure"] == 0) & (df["TotalCharges"].isna())
        df.loc[zero_tenure_missing, "TotalCharges"] = 0.0

        # Strip whitespace from categorical strings (pandas 2.x / 3.x safe)
        for col in RAW_CATEGORICAL_FEATURES:
            df[col] = df[col].astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
            df.loc[df[col].isin(["", "nan", "None"]), col] = np.nan

        # Engineered numeric features
        tenure_safe = df["tenure"].fillna(0.0)
        monthly_safe = df["MonthlyCharges"].fillna(0.0)
        total_safe = df["TotalCharges"].fillna(0.0)

        df["avg_monthly_spend"] = np.where(
            tenure_safe > 0,
            total_safe / np.maximum(tenure_safe, 1.0),
            monthly_safe,
        )
        df["charge_ratio"] = np.where(
            df["avg_monthly_spend"] > 0,
            monthly_safe / (df["avg_monthly_spend"] + 1e-5),
            1.0,
        )

        # Count active telecom services
        service_count = np.zeros(len(df), dtype=float)
        for col in self.SERVICE_YES_COLUMNS:
            service_count += (df[col] == "Yes").astype(float)
        service_count += (df["InternetService"].isin(["DSL", "Fiber optic"])).astype(float)
        df["service_count"] = service_count

        # Engineered binary indicators
        df["has_support_bundle"] = (
            (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
        ).astype(int)

        df["is_auto_pay"] = (
            df["PaymentMethod"].fillna("").str.contains("automatic", case=False)
        ).astype(int)

        # Tenure lifecycle bucket
        df["tenure_group"] = pd.cut(
            tenure_safe,
            bins=[-1, 12, 24, 48, 1000],
            labels=["0-12m", "13-24m", "25-48m", "49m+"],
        ).astype(str)

        return df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(
            ALL_NUMERIC_FEATURES + ALL_BINARY_FEATURES + ALL_CATEGORICAL_FEATURES,
            dtype=object,
        )


def build_column_transformer(scale_numeric: bool = True) -> ColumnTransformer:
    """Build the ColumnTransformer for engineered Telco features."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, ALL_NUMERIC_FEATURES),
            ("bin", binary_pipeline, ALL_BINARY_FEATURES),
            ("cat", categorical_pipeline, ALL_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessor(scale_numeric: bool = True) -> Pipeline:
    """Build the full scikit-learn preprocessing pipeline (feature engineering + ColumnTransformer)."""
    return Pipeline(
        steps=[
            ("feature_engineer", TelcoFeatureEngineer()),
            ("column_transformer", build_column_transformer(scale_numeric=scale_numeric)),
        ]
    )


def get_transformed_feature_names(preprocessor: Pipeline) -> List[str]:
    """Extract clean output feature names from a fitted preprocessor pipeline."""
    ct = preprocessor.named_steps["column_transformer"]
    return [str(name) for name in ct.get_feature_names_out()]
