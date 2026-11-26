"""Global and local SHAP explainability for the customer churn prediction pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import (
    EVALUATION_DIR,
    FIGURES_DIR,
    LATEST_MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_FEATURE_COLUMNS,
    SHAP_BACKGROUND_PATH,
    SHAP_SUMMARY_PATH,
    TARGET_COL,
    TEST_SIZE,
)
from src.feature_engineering import get_transformed_feature_names

PRETTY_FEATURE_MAP = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly Charges ($)",
    "TotalCharges": "Total Charges ($)",
    "avg_monthly_spend": "Avg Monthly Spend ($)",
    "charge_ratio": "Monthly-to-Avg Charge Ratio",
    "service_count": "Active Service Count",
    "SeniorCitizen": "Senior Citizen",
    "has_support_bundle": "Has Security/Tech Support",
    "is_auto_pay": "Automatic Payment",
}


def format_feature_name(raw_name: str) -> str:
    """Convert transformed feature names into readable labels."""
    if raw_name in PRETTY_FEATURE_MAP:
        return PRETTY_FEATURE_MAP[raw_name]
    if "_" in raw_name:
        prefix, val = raw_name.split("_", 1)
        return f"{prefix}: {val}"
    return raw_name


def _extract_positive_class_shap(values: Any, base_values: Any) -> tuple[np.ndarray, float]:
    """Normalize SHAP output across binary linear and tree explainers to 2D positive-class array."""
    if isinstance(values, list):
        arr = np.asarray(values[1], dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 3:
            arr = arr[:, :, 1]

    if isinstance(base_values, (list, np.ndarray)):
        bv = np.asarray(base_values, dtype=float)
        if bv.ndim == 0:
            base_scalar = float(bv)
        elif bv.ndim == 1 and len(bv) == 2:
            base_scalar = float(bv[1])
        elif bv.ndim == 2 and bv.shape[1] == 2:
            base_scalar = float(bv[0, 1])
        else:
            base_scalar = float(bv.ravel()[0])
    else:
        base_scalar = float(base_values)

    return arr, base_scalar


def derive_retention_considerations(
    customer_record: Dict[str, Any],
    top_risk_drivers: List[Dict[str, Any]],
) -> List[str]:
    """Map actual positive SHAP drivers and observed customer attributes to retention considerations.

    Strictly grounded in the customer's actual model risk drivers; framed as operational
    hypotheses to test rather than proven causal guarantees.
    """
    considerations: List[str] = []
    positive_features = {
        d["feature"] for d in top_risk_drivers if d["direction"] == "increases_churn_risk"
    }

    if (
        customer_record.get("Contract") == "Month-to-month"
        or any("Contract_Month-to-month" in f for f in positive_features)
    ):
        considerations.append(
            "Month-to-month contract is a primary risk driver: consider evaluating a targeted "
            "1-year or 2-year contract migration incentive (e.g., loyalty discount or waived upgrade fee)."
        )

    tenure_val = float(customer_record.get("tenure", 0) or 0)
    if tenure_val <= 12 or any("tenure" in f for f in positive_features):
        considerations.append(
            "Early-tenure lifecycle risk: prioritize proactive onboarding check-ins and first-year "
            "service satisfaction outreach before the initial renewal friction window."
        )

    if (
        customer_record.get("InternetService") == "Fiber optic"
        or any("InternetService_Fiber optic" in f for f in positive_features)
    ):
        considerations.append(
            "Fiber optic service shows elevated churn association: audit line quality/speed tickets "
            "and review pricing competitiveness against comparable plans."
        )

    if (
        customer_record.get("OnlineSecurity") == "No"
        or customer_record.get("TechSupport") == "No"
        or any("OnlineSecurity_No" in f or "TechSupport_No" in f or "has_support_bundle" in f for f in positive_features)
    ):
        considerations.append(
            "Missing Tech Support / Online Security add-ons increased predicted risk: test bundling "
            "a complimentary trial of technical support or security protection."
        )

    if (
        customer_record.get("PaymentMethod") == "Electronic check"
        or any("PaymentMethod_Electronic check" in f or "is_auto_pay" in f for f in positive_features)
    ):
        considerations.append(
            "Electronic check payment is associated with higher churn: consider offering a small "
            "billing credit for enrolling in automatic bank transfer or credit-card autopay."
        )

    if any("MonthlyCharges" in f or "charge_ratio" in f for f in positive_features):
        considerations.append(
            "Elevated monthly charge burden contributed to risk: review plan right-sizing options "
            "to prevent bill-shock cancellation."
        )

    if not considerations:
        considerations.append(
            "No concentrated high-friction service driver dominated this prediction; maintain standard "
            "account health monitoring and periodic satisfaction surveys."
        )

    return considerations


class ChurnExplainer:
    """Wrapper around SHAP for deterministic global and individual churn explanations."""

    def __init__(self, pipeline: Any, background_df: pd.DataFrame):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]
        self.feature_names = get_transformed_feature_names(self.preprocessor)

        bg_clean = background_df[RAW_FEATURE_COLUMNS].copy()
        self.X_bg_transformed = self.preprocessor.transform(bg_clean)

        if isinstance(self.classifier, LogisticRegression):
            self.explainer_type = "LinearExplainer"
            self.explainer = shap.LinearExplainer(self.classifier, self.X_bg_transformed)
        else:
            self.explainer_type = "TreeExplainer"
            self.explainer = shap.TreeExplainer(self.classifier)

    def compute_shap_values(self, raw_df: pd.DataFrame) -> tuple[np.ndarray, float, np.ndarray]:
        """Compute positive-class SHAP matrix, base value, and transformed feature matrix."""
        X_trans = self.preprocessor.transform(raw_df[RAW_FEATURE_COLUMNS].copy())
        if self.explainer_type == "LinearExplainer":
            raw_shap = self.explainer.shap_values(X_trans)
            base_val = self.explainer.expected_value
        else:
            raw_shap = self.explainer.shap_values(X_trans)
            base_val = self.explainer.expected_value

        shap_matrix, base_scalar = _extract_positive_class_shap(raw_shap, base_val)
        return shap_matrix, base_scalar, X_trans

    def explain_single(self, customer_df: pd.DataFrame, top_n: int = 8) -> Dict[str, Any]:
        """Return structured per-customer SHAP explanation for the first row in customer_df."""
        shap_matrix, base_val, X_trans = self.compute_shap_values(customer_df.iloc[[0]])
        row_shap = shap_matrix[0]
        row_vals = X_trans[0]

        order = np.argsort(np.abs(row_shap))[::-1][:top_n]
        drivers: List[Dict[str, Any]] = []
        for idx in order:
            sv = float(row_shap[idx])
            fname = self.feature_names[idx]
            drivers.append(
                {
                    "feature": fname,
                    "display_name": format_feature_name(fname),
                    "transformed_value": round(float(row_vals[idx]), 4),
                    "shap_value": round(sv, 4),
                    "abs_shap": round(abs(sv), 4),
                    "direction": "increases_churn_risk" if sv >= 0 else "decreases_churn_risk",
                }
            )

        record_dict = customer_df.iloc[0].to_dict()
        considerations = derive_retention_considerations(record_dict, drivers)

        increasing = [d for d in drivers if d["direction"] == "increases_churn_risk"][:3]
        decreasing = [d for d in drivers if d["direction"] == "decreases_churn_risk"][:3]

        return {
            "explainer_type": self.explainer_type,
            "base_value_log_odds": round(base_val, 4),
            "top_drivers": drivers,
            "top_risk_increasing_factors": [d["display_name"] for d in increasing],
            "top_risk_decreasing_factors": [d["display_name"] for d in decreasing],
            "retention_considerations": considerations,
            "causality_disclaimer": (
                "SHAP values explain how features contributed to this model's predicted log-odds; "
                "they represent statistical associations learned from historical data, not proven causation."
            ),
        }

    def plot_local_explanation(
        self,
        explanation: Dict[str, Any],
        churn_probability: Optional[float] = None,
    ) -> plt.Figure:
        """Create a clean horizontal contribution bar plot for an individual customer."""
        drivers = explanation["top_drivers"][::-1]
        labels = [d["display_name"] for d in drivers]
        values = [d["shap_value"] for d in drivers]
        colors = ["#E05A47" if v >= 0 else "#2E86AB" for v in values]

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        bars = ax.barh(labels, values, color=colors, height=0.65)
        ax.axvline(0, color="#333333", linewidth=1.0)
        for bar, val in zip(bars, values):
            offset = 0.015 if val >= 0 else -0.015
            ha = "left" if val >= 0 else "right"
            ax.text(
                val + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va="center",
                ha=ha,
                fontsize=9,
                fontweight="bold",
            )
        title = "Top Individual Churn Risk Drivers (SHAP Log-Odds Impact)"
        if churn_probability is not None:
            title += f"\nPredicted Churn Probability: {churn_probability:.1%}"
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("SHAP Value (+ Increases Churn Risk | - Decreases Churn Risk)")
        max_abs = max((abs(v) for v in values), default=0.5)
        ax.set_xlim(-max_abs * 1.28, max_abs * 1.28)
        plt.tight_layout()
        return fig


def generate_global_shap_artifacts() -> Dict[str, Any]:
    """Generate global SHAP plots, dependence plots, local example plot, and shap_summary.json."""
    if not LATEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {LATEST_MODEL_PATH}. Run `python -m src.train` first."
        )
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_DATA_PATH}. Run `python -m src.data_processing` first."
        )

    pipeline = joblib.load(LATEST_MODEL_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df[RAW_FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].astype(int).copy()

    # Reproduce exact held-out test set for out-of-sample global SHAP evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    bg_df = pd.read_csv(SHAP_BACKGROUND_PATH) if SHAP_BACKGROUND_PATH.exists() else X_train.sample(200, random_state=RANDOM_STATE)
    explainer = ChurnExplainer(pipeline, bg_df)

    shap_matrix, base_val, X_test_trans = explainer.compute_shap_values(X_test)
    pretty_names = [format_feature_name(f) for f in explainer.feature_names]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Global SHAP beeswarm summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_matrix,
        features=X_test_trans,
        feature_names=pretty_names,
        max_display=15,
        show=False,
    )
    plt.title("Global SHAP Summary (Beeswarm) — Held-Out Test Set", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    p_beeswarm = FIGURES_DIR / "14_shap_summary_beeswarm.png"
    plt.savefig(p_beeswarm, dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Global SHAP mean(|SHAP|) bar plot
    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    mean_signed = np.mean(shap_matrix, axis=0)
    top_idx = np.argsort(mean_abs)[::-1]

    top_n_bar = 15
    bar_idx = top_idx[:top_n_bar][::-1]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.barh(
        [pretty_names[i] for i in bar_idx],
        mean_abs[bar_idx],
        color="#2E86AB",
        height=0.68,
    )
    for idx_pos, feat_i in enumerate(bar_idx):
        ax.text(
            mean_abs[feat_i] + 0.005,
            idx_pos,
            f"{mean_abs[feat_i]:.3f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_title("Top 15 Global Feature Importances (Mean |SHAP Value|)", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Mean Absolute SHAP Impact on Log-Odds")
    ax.set_xlim(0, mean_abs[top_idx[0]] * 1.16)
    plt.tight_layout()
    p_bar = FIGURES_DIR / "15_shap_feature_importance_bar.png"
    fig.savefig(p_bar, dpi=150)
    plt.close(fig)

    # 3. SHAP dependence plots for top 3 continuous/high-impact features
    dep_candidates = [i for i in top_idx if explainer.feature_names[i] in {"tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "service_count", "Contract_Month-to-month", "InternetService_Fiber optic"}]
    dep_indices = dep_candidates[:3] if len(dep_candidates) >= 3 else list(top_idx[:3])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    X_test_df_trans = pd.DataFrame(X_test_trans, columns=pretty_names)
    for ax, f_idx in zip(axes, dep_indices):
        fname = pretty_names[f_idx]
        x_vals = X_test_df_trans[fname].values
        y_vals = shap_matrix[:, f_idx]
        color_vals = X_test["tenure"].values
        sc = ax.scatter(x_vals, y_vals, c=color_vals, cmap="coolwarm", alpha=0.55, s=16)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.9)
        ax.set_title(f"SHAP Dependence: {fname}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel(f"{fname} (Transformed)")
        ax.set_ylabel("SHAP Value")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Tenure (months)")
    fig.suptitle("SHAP Dependence Plots for Key Churn Predictors", fontsize=13, fontweight="bold", y=1.03)
    p_dep = FIGURES_DIR / "16_shap_dependence_top_features.png"
    fig.savefig(p_dep, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Individual high-risk customer example explanation
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    high_risk_pos = int(np.argmax(test_probs))
    sample_customer_df = X_test.iloc[[high_risk_pos]]
    sample_prob = float(test_probs[high_risk_pos])
    local_exp = explainer.explain_single(sample_customer_df, top_n=8)
    fig_local = explainer.plot_local_explanation(local_exp, churn_probability=sample_prob)
    p_local = FIGURES_DIR / "17_shap_local_high_risk_example.png"
    fig_local.savefig(p_local, dpi=150)
    plt.close(fig_local)

    # Build JSON summary
    global_importance = []
    for idx in top_idx[:15]:
        global_importance.append(
            {
                "feature": explainer.feature_names[idx],
                "display_name": pretty_names[idx],
                "mean_abs_shap": round(float(mean_abs[idx]), 4),
                "mean_signed_shap": round(float(mean_signed[idx]), 4),
            }
        )

    summary = {
        "explainer_type": explainer.explainer_type,
        "evaluated_rows": int(len(X_test)),
        "base_value_log_odds": round(base_val, 4),
        "top_global_features": global_importance,
        "sample_high_risk_customer": {
            "predicted_probability": round(sample_prob, 4),
            "actual_churn": int(y_test.iloc[high_risk_pos]),
            "profile": {k: (v.item() if hasattr(v, "item") else v) for k, v in sample_customer_df.iloc[0].to_dict().items()},
            "explanation": local_exp,
        },
        "generated_figures": [p_beeswarm.name, p_bar.name, p_dep.name, p_local.name],
    }

    with open(SHAP_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    out = generate_global_shap_artifacts()
    print(f"SHAP Explainer: {out['explainer_type']} (evaluated on {out['evaluated_rows']} test rows)")
    print("Top 5 Global Features by Mean |SHAP|:")
    for item in out["top_global_features"][:5]:
        print(f"  - {item['display_name']}: {item['mean_abs_shap']:.4f}")
    print(f"Saved SHAP summary to {SHAP_SUMMARY_PATH}")
