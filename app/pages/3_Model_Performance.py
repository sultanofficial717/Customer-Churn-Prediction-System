"""Page 3: Model Benchmark, Evaluation Metrics, and Global SHAP Explainability."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EVALUATION_SUMMARY_PATH,
    FIGURES_DIR,
    MODEL_COMPARISON_CSV_PATH,
    SHAP_SUMMARY_PATH,
)

st.set_page_config(page_title="Model Performance | Churn System", page_icon="📈", layout="wide")


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


st.title("📈 Model Performance & Global Explainability")
st.caption("Cross-validated benchmarks, held-out test metrics, error trade-offs, and global SHAP diagnostics.")

eval_summary = _load_json(EVALUATION_SUMMARY_PATH)
shap_summary = _load_json(SHAP_SUMMARY_PATH)

if not eval_summary or not MODEL_COMPARISON_CSV_PATH.exists():
    st.warning("Evaluation artifacts not found. Run `python -m src.train` and `python -m src.explain` first.")
    st.stop()

comp_df = pd.read_csv(MODEL_COMPARISON_CSV_PATH)
tm = eval_summary["selected_model_test_metrics"]
cm = tm["confusion_matrix"]

# Selected Model KPI Banner
st.subheader(f"🏆 Selected Production Model: {eval_summary['selected_model']}")
st.caption(f"Selection Criterion: {eval_summary['selection_criterion']}")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Test ROC-AUC", f"{tm['roc_auc']:.4f}")
m2.metric("Test PR-AUC", f"{tm['pr_auc']:.4f}")
m3.metric("Test F1-Score", f"{tm['f1_score']:.4f}")
m4.metric("Test Recall", f"{tm['recall']:.4f}")
m5.metric("Test Precision", f"{tm['precision']:.4f}")
m6.metric("Test Accuracy", f"{tm['accuracy']:.4f}")

st.divider()

# Model Comparison Table & Interactive Chart
st.subheader("📊 Algorithm & Imbalance-Handling Comparison")
display_cols = [
    "model_name",
    "stage",
    "cv_roc_auc",
    "cv_pr_auc",
    "cv_f1",
    "roc_auc",
    "pr_auc",
    "f1_score",
    "recall",
    "precision",
    "accuracy",
]
st.dataframe(
    comp_df[display_cols].style.format(
        {c: "{:.4f}" for c in display_cols if c not in {"model_name", "stage"}}
    ),
    use_container_width=True,
    hide_index=True,
)

chart_melted = comp_df.melt(
    id_vars=["model_name"],
    value_vars=["roc_auc", "pr_auc", "f1_score", "recall", "precision"],
    var_name="Metric",
    value_name="Score",
)
fig_bar = px.bar(
    chart_melted,
    x="Metric",
    y="Score",
    color="model_name",
    barmode="group",
    title="Held-Out Test Metrics Across All Evaluated Models",
)
fig_bar.update_layout(yaxis_range=[0, 1.05])
st.plotly_chart(fig_bar, use_container_width=True)

# Diagnostic Plots
st.subheader("🔍 Classification Diagnostics (Held-Out Test Set)")
col1, col2 = st.columns(2)
with col1:
    p_roc = FIGURES_DIR / "10_roc_curves.png"
    if p_roc.exists():
        st.image(str(p_roc), caption="ROC Curves across model families", use_container_width=True)
    p_cm = FIGURES_DIR / "12_confusion_matrix_best.png"
    if p_cm.exists():
        st.image(str(p_cm), caption="Confusion Matrix for selected model", use_container_width=True)

with col2:
    p_pr = FIGURES_DIR / "11_pr_curves.png"
    if p_pr.exists():
        st.image(str(p_pr), caption="Precision-Recall Curves (critical under class imbalance)", use_container_width=True)
    p_thr = FIGURES_DIR / "13_threshold_analysis.png"
    if p_thr.exists():
        st.image(str(p_thr), caption="Threshold sensitivity: Precision vs. Recall vs. F1", use_container_width=True)

st.info(
    f"**Business Implications of Errors (Test Set at 0.50 Cutoff — TP: {cm['tp']}, FN: {cm['fn']}, FP: {cm['fp']}, TN: {cm['tn']})**:\n"
    f"- **False Negatives ({cm['fn']} customers)**: Actual churners missed by the model, resulting in lost recurring revenue without retention intervention.\n"
    f"- **False Positives ({cm['fp']} customers)**: Loyal customers flagged as at-risk, incurring outreach or retention incentive costs unnecessarily.\n"
    "- Because churn is imbalanced (~26.5% positive), accuracy alone is misleading; tuning the decision threshold lets retention teams align recall and precision with actual outreach budgets."
)

# Global SHAP Section
st.divider()
st.subheader("🧠 Global SHAP Model Explainability")
if shap_summary:
    st.caption(
        f"Explainer: `{shap_summary['explainer_type']}` | Evaluated on {shap_summary['evaluated_rows']:,} "
        "held-out test customers. Feature importance captures learned predictive contribution, not causation."
    )
    s1, s2 = st.columns(2)
    with s1:
        p_bee = FIGURES_DIR / "14_shap_summary_beeswarm.png"
        if p_bee.exists():
            st.image(str(p_bee), caption="SHAP Summary (Beeswarm) Plot", use_container_width=True)
    with s2:
        p_imp = FIGURES_DIR / "15_shap_feature_importance_bar.png"
        if p_imp.exists():
            st.image(str(p_imp), caption="Global Mean |SHAP| Feature Importance", use_container_width=True)

    p_dep = FIGURES_DIR / "16_shap_dependence_top_features.png"
    if p_dep.exists():
        st.image(str(p_dep), caption="SHAP Dependence Plots for Key Churn Predictors", use_container_width=True)

    with st.expander("View Top 15 Global SHAP Feature Importance Table"):
        st.dataframe(pd.DataFrame(shap_summary["top_global_features"]), use_container_width=True, hide_index=True)
else:
    st.warning("SHAP summary not found. Run `python -m src.explain` to generate SHAP artifacts.")
