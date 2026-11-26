"""Production inference service decoupled from FastAPI and Streamlit frontends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from src.config import (
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_LOW_RISK_THRESHOLD,
    LATEST_MODEL_PATH,
    METADATA_PATH,
    RAW_FEATURE_COLUMNS,
    SHAP_BACKGROUND_PATH,
)
from src.explain import ChurnExplainer


def classify_risk_category(
    probability: float,
    low_threshold: float = DEFAULT_LOW_RISK_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
) -> str:
    """Map predicted churn probability into configurable operational risk buckets."""
    if not (0.0 < low_threshold < high_threshold < 1.0):
        raise ValueError(
            f"Thresholds must satisfy 0 < low_threshold ({low_threshold}) "
            f"< high_threshold ({high_threshold}) < 1."
        )
    if probability < low_threshold:
        return "Low"
    if probability < high_threshold:
        return "Medium"
    return "High"


class ChurnPredictor:
    """Reusable inference & explainability facade backed by the saved Joblib pipeline."""

    def __init__(
        self,
        model_path: Path | str = LATEST_MODEL_PATH,
        metadata_path: Path | str = METADATA_PATH,
        shap_background_path: Path | str = SHAP_BACKGROUND_PATH,
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.shap_background_path = Path(shap_background_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained pipeline not found at {self.model_path}. Run `python -m src.train` first."
            )

        self.pipeline = joblib.load(self.model_path)
        self.metadata: Dict[str, Any] = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self._explainer: Optional[ChurnExplainer] = None

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("selected_model", type(self.pipeline.named_steps["classifier"]).__name__))

    def _get_explainer(self) -> ChurnExplainer:
        if self._explainer is None:
            if not self.shap_background_path.exists():
                raise FileNotFoundError(
                    f"SHAP background sample not found at {self.shap_background_path}."
                )
            bg_df = pd.read_csv(self.shap_background_path)
            self._explainer = ChurnExplainer(self.pipeline, bg_df)
        return self._explainer

    def _prepare_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        if not records:
            raise ValueError("Input records list cannot be empty.")
        df = pd.DataFrame(records)
        missing = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required customer features: {missing}")
        # Enforce strict canonical column ordering
        return df[RAW_FEATURE_COLUMNS].copy()

    def predict_batch(
        self,
        records: List[Dict[str, Any]],
        decision_threshold: float = 0.50,
        low_threshold: float = DEFAULT_LOW_RISK_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """Predict churn probabilities, labels, and risk tiers for a list of customer dicts."""
        if not (0.0 < decision_threshold < 1.0):
            raise ValueError("decision_threshold must be strictly between 0 and 1.")

        df = self._prepare_dataframe(records)
        probabilities = self.pipeline.predict_proba(df)[:, 1]

        results: List[Dict[str, Any]] = []
        for prob in probabilities:
            prob_f = float(prob)
            pred = int(prob_f >= decision_threshold)
            risk = classify_risk_category(
                prob_f, low_threshold=low_threshold, high_threshold=high_threshold
            )
            results.append(
                {
                    "churn_prediction": pred,
                    "churn_label": "Yes" if pred == 1 else "No",
                    "churn_probability": round(prob_f, 4),
                    "risk_category": risk,
                    "decision_threshold": round(float(decision_threshold), 4),
                    "risk_thresholds": {
                        "low_max": round(float(low_threshold), 4),
                        "high_min": round(float(high_threshold), 4),
                    },
                    "model_version": self.model_version,
                    "model_name": self.model_name,
                }
            )
        return results

    def predict_one(
        self,
        record: Dict[str, Any],
        decision_threshold: float = 0.50,
        low_threshold: float = DEFAULT_LOW_RISK_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
        include_explanation: bool = False,
        top_n_drivers: int = 8,
    ) -> Dict[str, Any]:
        """Predict churn for a single customer with optional personalized SHAP explanation."""
        result = self.predict_batch(
            [record],
            decision_threshold=decision_threshold,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )[0]

        if include_explanation:
            df = self._prepare_dataframe([record])
            explainer = self._get_explainer()
            result["explanation"] = explainer.explain_single(df, top_n=top_n_drivers)

        return result

    def build_local_explanation_figure(
        self,
        record: Dict[str, Any],
        churn_probability: float,
        top_n_drivers: int = 8,
    ):
        """Return a Matplotlib figure visualizing top SHAP drivers for a single customer."""
        df = self._prepare_dataframe([record])
        explainer = self._get_explainer()
        explanation = explainer.explain_single(df, top_n=top_n_drivers)
        fig = explainer.plot_local_explanation(explanation, churn_probability=churn_probability)
        return fig, explanation
