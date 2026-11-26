"""Page 1: Interactive EDA & Business Overview Dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EDA_SUMMARY_PATH, FIGURES_DIR, PROCESSED_DATA_PATH, RAW_DATA_PATH, TARGET_COL
from src.data_processing import clean_data, load_raw_data

st.set_page_config(page_title="Overview Dashboard | Churn System", page_icon="📊", layout="wide")


@st.cache_data
def load_dashboard_data() -> Optional[pd.DataFrame]:
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)
    if RAW_DATA_PATH.exists():
        return clean_data(load_raw_data(), include_target=True)
    return None


@st.cache_data
def load_eda_summary() -> dict:
    if EDA_SUMMARY_PATH.exists():
        with open(EDA_SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


st.title("📊 Customer Churn Overview Dashboard")
st.caption("Explore observed churn patterns across contracts, tenure, billing, payment methods, and services.")

df = load_dashboard_data()
eda_summary = load_eda_summary()

if df is not None:
    # Full row-level interactive mode (local or when CSV is available)
    st.sidebar.header("Filter Customer Cohort")
    contract_filter = st.sidebar.multiselect(
        "Contract Type",
        options=sorted(df["Contract"].unique().tolist()),
        default=sorted(df["Contract"].unique().tolist()),
    )
    internet_filter = st.sidebar.multiselect(
        "Internet Service",
        options=sorted(df["InternetService"].unique().tolist()),
        default=sorted(df["InternetService"].unique().tolist()),
    )
    payment_filter = st.sidebar.multiselect(
        "Payment Method",
        options=sorted(df["PaymentMethod"].unique().tolist()),
        default=sorted(df["PaymentMethod"].unique().tolist()),
    )

    filtered = df[
        df["Contract"].isin(contract_filter)
        & df["InternetService"].isin(internet_filter)
        & df["PaymentMethod"].isin(payment_filter)
    ].copy()

    if filtered.empty:
        st.warning("No customers match the current filter selection. Adjust sidebar filters.")
        st.stop()

    filtered["ChurnLabel"] = filtered[TARGET_COL].map({0: "Retained", 1: "Churned"})

    total_n = len(filtered)
    churned_n = int(filtered[TARGET_COL].sum())
    churn_rate = filtered[TARGET_COL].mean()
    avg_monthly = filtered["MonthlyCharges"].mean()
    churned_mrr = filtered.loc[filtered[TARGET_COL] == 1, "MonthlyCharges"].sum()
    total_mrr = filtered["MonthlyCharges"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers in Selection", f"{total_n:,}")
    k2.metric("Churn Rate", f"{churn_rate:.2%}", f"{churned_n:,} churned")
    k3.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
    k4.metric(
        "Churned Monthly Billing",
        f"${churned_mrr:,.0f}",
        f"{(churned_mrr / total_mrr):.1%} of filtered billing" if total_mrr > 0 else None,
    )

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        dist_df = filtered["ChurnLabel"].value_counts().reset_index()
        dist_df.columns = ["Status", "Customers"]
        fig_dist = px.pie(
            dist_df,
            names="Status",
            values="Customers",
            hole=0.45,
            color="Status",
            color_discrete_map={"Retained": "#2E86AB", "Churned": "#E05A47"},
            title="Overall Churn vs. Retention Split",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        contract_stats = (
            filtered.groupby("Contract")[TARGET_COL]
            .agg(churn_rate="mean", customers="count")
            .reset_index()
            .sort_values("churn_rate", ascending=False)
        )
        contract_stats["churn_pct"] = contract_stats["churn_rate"] * 100
        fig_contract = px.bar(
            contract_stats,
            x="Contract",
            y="churn_pct",
            text=contract_stats.apply(lambda r: f"{r['churn_pct']:.1f}% (n={r['customers']:,})", axis=1),
            color="churn_pct",
            color_continuous_scale="Reds",
            title="Churn Rate by Contract Type (%)",
            labels={"churn_pct": "Churn Rate (%)"},
        )
        fig_contract.update_traces(textposition="outside")
        fig_contract.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            yaxis_range=[0, max(55, contract_stats["churn_pct"].max() + 10)],
        )
        st.plotly_chart(fig_contract, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_tenure = px.histogram(
            filtered,
            x="tenure",
            color="ChurnLabel",
            nbins=24,
            barmode="overlay",
            opacity=0.65,
            color_discrete_map={"Retained": "#2E86AB", "Churned": "#E05A47"},
            title="Customer Tenure Distribution by Churn Status",
            labels={"tenure": "Tenure (Months)"},
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

    with c4:
        fig_charges = px.box(
            filtered,
            x="ChurnLabel",
            y="MonthlyCharges",
            color="ChurnLabel",
            color_discrete_map={"Retained": "#2E86AB", "Churned": "#E05A47"},
            title="Monthly Charges Spread: Churned vs. Retained",
            labels={"MonthlyCharges": "Monthly Charges ($)", "ChurnLabel": ""},
        )
        st.plotly_chart(fig_charges, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        pay_stats = (
            filtered.groupby("PaymentMethod")[TARGET_COL]
            .agg(churn_rate="mean", customers="count")
            .reset_index()
            .sort_values("churn_rate", ascending=True)
        )
        pay_stats["churn_pct"] = pay_stats["churn_rate"] * 100
        fig_pay = px.bar(
            pay_stats,
            y="PaymentMethod",
            x="churn_pct",
            orientation="h",
            text=pay_stats.apply(lambda r: f"{r['churn_pct']:.1f}% (n={r['customers']:,})", axis=1),
            color="churn_pct",
            color_continuous_scale="OrRd",
            title="Churn Rate by Payment Method (%)",
            labels={"churn_pct": "Churn Rate (%)", "PaymentMethod": ""},
        )
        fig_pay.update_traces(textposition="outside")
        fig_pay.update_layout(
            coloraxis_showscale=False,
            xaxis_range=[0, max(58, pay_stats["churn_pct"].max() + 15)],
        )
        st.plotly_chart(fig_pay, use_container_width=True)

    with c6:
        inet_stats = (
            filtered.groupby("InternetService")[TARGET_COL]
            .agg(churn_rate="mean", customers="count")
            .reset_index()
            .sort_values("churn_rate", ascending=False)
        )
        inet_stats["churn_pct"] = inet_stats["churn_rate"] * 100
        fig_inet = px.bar(
            inet_stats,
            x="InternetService",
            y="churn_pct",
            text=inet_stats.apply(lambda r: f"{r['churn_pct']:.1f}% (n={r['customers']:,})", axis=1),
            color="churn_pct",
            color_continuous_scale="Reds",
            title="Churn Rate by Internet Service (%)",
            labels={"churn_pct": "Churn Rate (%)"},
        )
        fig_inet.update_traces(textposition="outside")
        fig_inet.update_layout(
            coloraxis_showscale=False,
            yaxis_range=[0, max(55, inet_stats["churn_pct"].max() + 12)],
        )
        st.plotly_chart(fig_inet, use_container_width=True)

elif eda_summary:
    # Cloud-safe fallback mode using committed measured EDA artifacts (no raw CSV required)
    st.info(
        "Running in **Cloud Artifact Mode** using precomputed measured EDA summaries (`eda_summary.json`). "
        "Add the raw Telco CSV locally in `data/raw/` to enable row-level sidebar filtering."
    )
    td = eda_summary["target_distribution"]
    bm = eda_summary["measured_billing_metrics"]
    seg = eda_summary["segment_analysis"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{td['total_customers']:,}")
    k2.metric("Overall Churn Rate", f"{td['churn_rate']:.2%}", f"{td['churned_customers']:,} churned")
    k3.metric("Avg Monthly Charges", f"${bm['avg_monthly_charge_all']:.2f}")
    k4.metric(
        "Churned Monthly Billing",
        f"${bm['churned_monthly_charges_sum']:,.0f}",
        f"{bm['churned_monthly_charges_share']:.1%} of total billing",
    )

    st.divider()

    def _seg_bar(seg_key: str, title: str, orientation: str = "v"):
        sdf = pd.DataFrame(seg[seg_key])
        sdf["churn_pct"] = sdf["churn_rate"] * 100
        label_txt = sdf.apply(lambda r: f"{r['churn_pct']:.1f}% (n={r['customers']:,})", axis=1)
        if orientation == "h":
            sdf = sdf.sort_values("churn_pct", ascending=True)
            fig = px.bar(
                sdf,
                y="segment",
                x="churn_pct",
                orientation="h",
                text=label_txt,
                color="churn_pct",
                color_continuous_scale="OrRd",
                title=title,
                labels={"churn_pct": "Churn Rate (%)", "segment": ""},
            )
            fig.update_layout(coloraxis_showscale=False, xaxis_range=[0, max(58, sdf["churn_pct"].max() + 15)])
        else:
            fig = px.bar(
                sdf,
                x="segment",
                y="churn_pct",
                text=label_txt,
                color="churn_pct",
                color_continuous_scale="Reds",
                title=title,
                labels={"churn_pct": "Churn Rate (%)", "segment": ""},
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, max(55, sdf["churn_pct"].max() + 12)])
        fig.update_traces(textposition="outside")
        return fig

    c1, c2 = st.columns(2)
    with c1:
        dist_df = pd.DataFrame(
            {
                "Status": ["Retained", "Churned"],
                "Customers": [td["retained_customers"], td["churned_customers"]],
            }
        )
        fig_dist = px.pie(
            dist_df,
            names="Status",
            values="Customers",
            hole=0.45,
            color="Status",
            color_discrete_map={"Retained": "#2E86AB", "Churned": "#E05A47"},
            title="Overall Churn vs. Retention Split",
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    with c2:
        st.plotly_chart(_seg_bar("Contract", "Churn Rate by Contract Type (%)"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(_seg_bar("tenure_bucket", "Churn Rate by Tenure Cohort (%)"), use_container_width=True)
    with c4:
        st.plotly_chart(_seg_bar("monthly_charge_band", "Churn Rate by Monthly Charge Band (%)"), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(_seg_bar("PaymentMethod", "Churn Rate by Payment Method (%)", orientation="h"), use_container_width=True)
    with c6:
        st.plotly_chart(_seg_bar("InternetService", "Churn Rate by Internet Service (%)"), use_container_width=True)

    p_tenure = FIGURES_DIR / "03_churn_by_tenure.png"
    p_charges = FIGURES_DIR / "04_churn_by_monthly_charges.png"
    if p_tenure.exists() and p_charges.exists():
        f1, f2 = st.columns(2)
        f1.image(str(p_tenure), caption="Tenure Distribution by Churn Status", use_container_width=True)
        f2.image(str(p_charges), caption="Monthly Charges Density & Spread", use_container_width=True)
else:
    st.error("Neither the Telco CSV nor `reports/evaluation/eda_summary.json` was found.")
    st.stop()

# Measured Business Insights & Retention Implications
st.divider()
st.subheader("💡 Measured Business Insights & Retention Implications")

if eda_summary:
    bm = eda_summary["measured_billing_metrics"]
    seg = eda_summary["segment_analysis"]
    top_contract = seg["Contract"][0]
    low_contract = seg["Contract"][-1]
    top_tenure = seg["tenure_bucket"][0]
    top_payment = seg["PaymentMethod"][0]
    top_inet = seg["InternetService"][0]

    st.markdown(
        f"""
        **Measured Findings (IBM Telco Dataset, n = {eda_summary['cleaned_rows']:,}):**
        - **Contract Vulnerability**: `{top_contract['segment']}` accounts show **{top_contract['churn_rate']:.1%}** churn
          ({top_contract['churned']:,}/{top_contract['customers']:,}), compared to only **{low_contract['churn_rate']:.1%}** for `{low_contract['segment']}` contracts.
        - **Early-Lifecycle Attrition**: Customers in `{top_tenure['segment']}` experience **{top_tenure['churn_rate']:.1%}** churn,
          making onboarding and first-year engagement the highest-leverage window.
        - **Service & Billing Concentration**: `{top_inet['segment']}` internet subscribers churn at **{top_inet['churn_rate']:.1%}**,
          and `{top_payment['segment']}` payers churn at **{top_payment['churn_rate']:.1%}**.
        - **Measured Monthly Billing Exposure**: Churned customers pay an average of **${bm['avg_monthly_charge_churned']:.2f}/month**
          versus **${bm['avg_monthly_charge_retained']:.2f}/month** for retained customers, accounting for
          **${bm['churned_monthly_charges_sum']:,.2f}** (**{bm['churned_monthly_charges_share']:.1%}**) of total monthly charges in the dataset.

        **Operational Assumptions (Requires Company-Specific Validation):**
        - Real financial ROI from retention campaigns depends on unobserved parameters (customer acquisition cost,
          gross margin, discount cost, and actual offer conversion lift). Any campaign should be validated via
          a controlled holdout test rather than assuming 100% of predicted churners can be saved.
        """
    )
