"""Streamlit entrypoint for the Customer Churn Prediction System."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EDA_SUMMARY_PATH, METADATA_PATH

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


eda_summary = _load_json(EDA_SUMMARY_PATH)
metadata = _load_json(METADATA_PATH)

st.title("📡 Telecom Customer Churn Prediction System")
st.markdown(
    "An end-to-end, explainable Machine Learning platform built on the **IBM Telco Customer Churn** "
    "dataset — combining leakage-free Scikit-learn/XGBoost pipelines, SHAP explainability, "
    "a FastAPI inference backend, and a grounded RAG analytics assistant."
)

# Executive KPI Banner
if eda_summary and metadata:
    td = eda_summary["target_distribution"]
    tm = metadata["selected_model_test_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers Analyzed", f"{td['total_customers']:,}")
    c2.metric("Observed Churn Rate", f"{td['churn_rate']:.2%}", f"{td['churned_customers']:,} churned")
    c3.metric("Selected Model", metadata["selected_model"])
    c4.metric("Test ROC-AUC / Recall", f"{tm['roc_auc']:.3f} / {tm['recall']:.1%}")
else:
    st.warning(
        "Model or EDA artifacts not found yet. Run `python -m src.data_processing`, "
        "`python -m src.train`, and `python -m src.explain` to populate live metrics."
    )

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🧭 Application Modules")
    st.markdown(
        """
        Use the sidebar navigation to explore each section:
        1. **Overview Dashboard** — Interactive churn segmentation, demographic & billing patterns, and business risk analysis.
        2. **Customer Prediction** — Real-time churn probability scoring, configurable Low/Medium/High risk thresholds, personalized SHAP risk drivers, and retention considerations.
        3. **Model Performance** — Multi-algorithm benchmark (Logistic Regression, Random Forest, XGBoost), ROC/PR curves, confusion matrix, and global SHAP explainability.
        4. **AI Assistant** — Evidence-grounded Q&A over EDA findings, evaluation metrics, SHAP drivers, and active customer predictions.
        """
    )

with col_right:
    st.subheader("🏗️ Production Architecture Highlights")
    st.markdown(
        """
        - **Leakage Prevention**: Stratified train/test split before fitting; all engineering, imputation, scaling, and encoding encapsulated inside a single Joblib `Pipeline`.
        - **Imbalance Handling**: Benchmarked class weighting and inside-fold SMOTE (via `imblearn.pipeline.Pipeline`).
        - **Decoupled Inference**: Shared `src/predict.py` & `src/explain.py` power both this Streamlit frontend and the standalone FastAPI service (`api/main.py`).
        - **Honest Explainability**: Customer-level explanations come directly from SHAP values — never hard-coded fake rules.
        """
    )

if metadata:
    with st.expander("📦 Active Model Artifact Metadata"):
        st.json(
            {
                "model_version": metadata.get("model_version"),
                "selected_model": metadata.get("selected_model"),
                "classifier_class": metadata.get("classifier_class"),
                "trained_at_utc": metadata.get("trained_at_utc"),
                "selection_criterion": metadata.get("selection_criterion"),
                "default_risk_thresholds": metadata.get("risk_thresholds"),
            }
        )
