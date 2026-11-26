"""Page 2: Real-Time Customer Churn Prediction & Personalized SHAP Explainability."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CATEGORY_OPTIONS,
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_LOW_RISK_THRESHOLD,
)
from src.predict import ChurnPredictor

st.set_page_config(page_title="Customer Prediction | Churn System", page_icon="🎯", layout="wide")


@st.cache_resource
def load_local_predictor() -> ChurnPredictor:
    return ChurnPredictor()


SAMPLE_PROFILES = {
    "High-Risk Profile (New Fiber Month-to-Month)": {
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
        "MonthlyCharges": 94.50,
        "TotalCharges": 283.50,
    },
    "Medium-Risk Profile (1-Year DSL Mid-Tenure)": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 18,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 68.00,
        "TotalCharges": 1224.00,
    },
    "Low-Risk Profile (Loyal 2-Year Auto-Pay)": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
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
        "MonthlyCharges": 72.50,
        "TotalCharges": 4350.00,
    },
}

st.title("🎯 Customer Churn Risk Prediction & SHAP Explanation")
st.caption(
    "Enter customer details to estimate churn probability, inspect customer-level SHAP drivers, "
    "and review risk-aligned retention considerations."
)

# Sidebar Controls: Operational Thresholds + Inference Mode
st.sidebar.header("Operational Risk Thresholds")
st.sidebar.info(
    "Risk cutoffs below are **configurable operational thresholds** tailored to retention budget "
    "and outreach capacity — they are not universal telecom standards."
)
low_thr = st.sidebar.slider(
    "Low / Medium Risk Cutoff",
    min_value=0.10,
    max_value=0.55,
    value=float(DEFAULT_LOW_RISK_THRESHOLD),
    step=0.05,
)
high_thr = st.sidebar.slider(
    "Medium / High Risk Cutoff",
    min_value=max(low_thr + 0.05, 0.20),
    max_value=0.90,
    value=max(float(DEFAULT_HIGH_RISK_THRESHOLD), low_thr + 0.10),
    step=0.05,
)
decision_thr = st.sidebar.slider(
    "Binary Classification Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

st.sidebar.divider()
st.sidebar.header("Inference Backend")
use_api = st.sidebar.checkbox("Call FastAPI backend instead of local pipeline", value=False)
api_url = st.sidebar.text_input(
    "FastAPI Base URL",
    value=os.getenv("API_BASE_URL", "http://localhost:8000"),
    disabled=not use_api,
)

# Preset selector
preset_name = st.selectbox("Load Sample Customer Profile (optional)", list(SAMPLE_PROFILES.keys()))
preset = SAMPLE_PROFILES[preset_name]

with st.form("customer_prediction_form"):
    st.subheader("Customer Profile")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demographics & Account**")
        gender = st.selectbox("Gender", CATEGORY_OPTIONS["gender"], index=CATEGORY_OPTIONS["gender"].index(preset["gender"]))
        senior = st.selectbox("Senior Citizen", [0, 1], index=preset["SeniorCitizen"], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        partner = st.selectbox("Partner", CATEGORY_OPTIONS["Partner"], index=CATEGORY_OPTIONS["Partner"].index(preset["Partner"]))
        dependents = st.selectbox("Dependents", CATEGORY_OPTIONS["Dependents"], index=CATEGORY_OPTIONS["Dependents"].index(preset["Dependents"]))
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=int(preset["tenure"]))
        contract = st.selectbox("Contract", CATEGORY_OPTIONS["Contract"], index=CATEGORY_OPTIONS["Contract"].index(preset["Contract"]))

    with c2:
        st.markdown("**Phone & Internet Services**")
        phone = st.selectbox("Phone Service", CATEGORY_OPTIONS["PhoneService"], index=CATEGORY_OPTIONS["PhoneService"].index(preset["PhoneService"]))
        multi_lines = st.selectbox("Multiple Lines", CATEGORY_OPTIONS["MultipleLines"], index=CATEGORY_OPTIONS["MultipleLines"].index(preset["MultipleLines"]))
        internet = st.selectbox("Internet Service", CATEGORY_OPTIONS["InternetService"], index=CATEGORY_OPTIONS["InternetService"].index(preset["InternetService"]))
        security = st.selectbox("Online Security", CATEGORY_OPTIONS["OnlineSecurity"], index=CATEGORY_OPTIONS["OnlineSecurity"].index(preset["OnlineSecurity"]))
        backup = st.selectbox("Online Backup", CATEGORY_OPTIONS["OnlineBackup"], index=CATEGORY_OPTIONS["OnlineBackup"].index(preset["OnlineBackup"]))
        device_prot = st.selectbox("Device Protection", CATEGORY_OPTIONS["DeviceProtection"], index=CATEGORY_OPTIONS["DeviceProtection"].index(preset["DeviceProtection"]))

    with c3:
        st.markdown("**Support, Streaming & Billing**")
        tech_support = st.selectbox("Tech Support", CATEGORY_OPTIONS["TechSupport"], index=CATEGORY_OPTIONS["TechSupport"].index(preset["TechSupport"]))
        streaming_tv = st.selectbox("Streaming TV", CATEGORY_OPTIONS["StreamingTV"], index=CATEGORY_OPTIONS["StreamingTV"].index(preset["StreamingTV"]))
        streaming_movies = st.selectbox("Streaming Movies", CATEGORY_OPTIONS["StreamingMovies"], index=CATEGORY_OPTIONS["StreamingMovies"].index(preset["StreamingMovies"]))
        paperless = st.selectbox("Paperless Billing", CATEGORY_OPTIONS["PaperlessBilling"], index=CATEGORY_OPTIONS["PaperlessBilling"].index(preset["PaperlessBilling"]))
        payment = st.selectbox("Payment Method", CATEGORY_OPTIONS["PaymentMethod"], index=CATEGORY_OPTIONS["PaymentMethod"].index(preset["PaymentMethod"]))
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=300.0, value=float(preset["MonthlyCharges"]), step=1.0)
        default_total = round(monthly_charges * tenure, 2) if tenure > 0 else 0.0
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=25000.0,
            value=float(preset["TotalCharges"]) if preset_name else float(default_total),
            step=10.0,
        )

    submitted = st.form_submit_button("Predict Churn Risk & Explain", type="primary", use_container_width=True)

if submitted:
    customer_payload = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multi_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        predictor = load_local_predictor()
        if use_api:
            resp = requests.post(
                f"{api_url.rstrip('/')}/predict/explain",
                params={
                    "decision_threshold": decision_thr,
                    "low_threshold": low_thr,
                    "high_threshold": high_thr,
                    "top_n_drivers": 8,
                },
                json=customer_payload,
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json()
            explanation = result["explanation"]
            fig_shap = predictor._get_explainer().plot_local_explanation(
                explanation, churn_probability=result["churn_probability"]
            )
        else:
            result = predictor.predict_one(
                customer_payload,
                decision_threshold=decision_thr,
                low_threshold=low_thr,
                high_threshold=high_thr,
                include_explanation=True,
                top_n_drivers=8,
            )
            explanation = result["explanation"]
            fig_shap = predictor._get_explainer().plot_local_explanation(
                explanation, churn_probability=result["churn_probability"]
            )

        # Persist in session state for AI Assistant page
        st.session_state["last_prediction"] = {
            "customer_profile": customer_payload,
            **result,
        }

        st.divider()
        prob = result["churn_probability"]
        risk = result["risk_category"]
        risk_color = {"Low": "#2A9D8F", "Medium": "#E9A23B", "High": "#E05A47"}[risk]

        r1, r2 = st.columns([1.1, 1.4])
        with r1:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "valueformat": ".1f"},
                    title={"text": f"Churn Probability ({risk} Risk)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": risk_color},
                        "steps": [
                            {"range": [0, low_thr * 100], "color": "#E6F4F1"},
                            {"range": [low_thr * 100, high_thr * 100], "color": "#FDF3E2"},
                            {"range": [high_thr * 100, 100], "color": "#FBE9E7"},
                        ],
                        "threshold": {
                            "line": {"color": "#222222", "width": 3},
                            "thickness": 0.75,
                            "value": decision_thr * 100,
                        },
                    },
                )
            )
            gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(gauge, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Class", "Churn (Yes)" if result["churn_prediction"] == 1 else "Retain (No)")
            m2.metric("Risk Tier", risk)
            m3.metric("Model", result["model_version"])

            st.caption(
                f"Active thresholds — Low: < {low_thr:.0%} | Medium: {low_thr:.0%}–{high_thr:.0%} | "
                f"High: >= {high_thr:.0%} | Binary cutoff: {decision_thr:.0%}"
            )

        with r2:
            st.subheader("Why did the model make this prediction? (SHAP)")
            st.pyplot(fig_shap, use_container_width=True)
            st.caption(explanation["causality_disclaimer"])

        st.subheader("📋 Top Feature Contributions & Grounded Retention Considerations")
        t1, t2 = st.columns([1.2, 1.0])
        with t1:
            drivers_df = pd.DataFrame(explanation["top_drivers"])[
                ["display_name", "shap_value", "direction"]
            ].rename(
                columns={
                    "display_name": "Feature",
                    "shap_value": "SHAP Impact (Log-Odds)",
                    "direction": "Effect",
                }
            )
            st.dataframe(drivers_df, use_container_width=True, hide_index=True)

        with t2:
            st.markdown("**Suggested Retention Considerations** *(linked to observed risk drivers)*:")
            for item in explanation["retention_considerations"]:
                st.markdown(f"- {item}")

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
