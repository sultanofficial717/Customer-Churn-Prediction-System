"""Data loading, cleaning, quality inspection, and exploratory data analysis utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    EDA_SUMMARY_PATH,
    EVALUATION_DIR,
    FIGURES_DIR,
    ID_COL,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    RAW_FEATURE_COLUMNS,
    TARGET_COL,
)

sns.set_theme(style="whitegrid", palette="muted")


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV dataset."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {csv_path}")
    return pd.read_csv(csv_path)


def inspect_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute pre-cleaning data quality metrics on the raw dataset."""
    total_charges_blank = int(
        df["TotalCharges"].astype(str).str.strip().eq("").sum()
        if "TotalCharges" in df.columns
        else 0
    )
    missing_per_col = {col: int(cnt) for col, cnt in df.isna().sum().items() if cnt > 0}
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else 0

    return {
        "raw_rows": int(df.shape[0]),
        "raw_columns": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "duplicate_customer_ids": duplicate_ids,
        "explicit_null_counts": missing_per_col,
        "blank_total_charges_count": total_charges_blank,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def clean_data(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Clean and standardize raw Telco customer records without introducing data leakage.

    Deterministic row-level operations only:
    - Strip leading/trailing whitespace from string columns.
    - Drop exact duplicate rows and duplicate customer IDs (keeping first).
    - Parse TotalCharges to numeric; where tenure == 0 and TotalCharges is blank/NaN,
      set TotalCharges = 0.0 (zero months billed).
    - Cast SeniorCitizen, tenure, MonthlyCharges, TotalCharges to numeric types.
    - Map Churn ('Yes'/'No') to binary integer (1/0) when include_target=True.
    """
    cleaned = df.copy()

    # Strip whitespace in string/object columns (compatible with pandas 2.x and 3.x)
    for col in cleaned.columns:
        if pd.api.types.is_string_dtype(cleaned[col]) or cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].astype(object)
            cleaned[col] = cleaned[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            cleaned.loc[cleaned[col].isin(["", "nan", "None"]), col] = np.nan

    # Drop exact duplicates and duplicate customerIDs if present
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    if ID_COL in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=[ID_COL], keep="first").reset_index(drop=True)

    # Validate required feature columns
    missing_required = [col for col in RAW_FEATURE_COLUMNS if col not in cleaned.columns]
    if missing_required:
        raise ValueError(f"Missing required feature columns: {missing_required}")

    # Numeric conversions
    cleaned["SeniorCitizen"] = pd.to_numeric(cleaned["SeniorCitizen"], errors="coerce").astype("Int64")
    cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce")
    cleaned["MonthlyCharges"] = pd.to_numeric(cleaned["MonthlyCharges"], errors="coerce")
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    # Deterministic business rule: customers with 0 tenure have 0.0 cumulative TotalCharges
    zero_tenure_blank = (cleaned["tenure"] == 0) & (cleaned["TotalCharges"].isna())
    cleaned.loc[zero_tenure_blank, "TotalCharges"] = 0.0

    # Cast SeniorCitizen to plain int when non-null
    if cleaned["SeniorCitizen"].notna().all():
        cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(int)

    if include_target and TARGET_COL in cleaned.columns:
        if not pd.api.types.is_numeric_dtype(cleaned[TARGET_COL]):
            target_map = {"Yes": 1, "No": 0}
            unmapped = set(cleaned[TARGET_COL].dropna().unique()) - set(target_map.keys())
            if unmapped:
                raise ValueError(f"Unexpected target values in {TARGET_COL}: {unmapped}")
            cleaned[TARGET_COL] = cleaned[TARGET_COL].map(target_map).astype(int)
        else:
            cleaned[TARGET_COL] = pd.to_numeric(cleaned[TARGET_COL], errors="raise").astype(int)

    return cleaned


def _segment_churn_table(df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    """Return churn count, customer count, churn rate, and average monthly charge per segment."""
    grouped = (
        df.groupby(group_col, observed=False)
        .agg(
            customers=(TARGET_COL, "count"),
            churned=(TARGET_COL, "sum"),
            churn_rate=(TARGET_COL, "mean"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
            avg_tenure=("tenure", "mean"),
        )
        .reset_index()
    )
    records = []
    for _, row in grouped.iterrows():
        records.append(
            {
                "segment": str(row[group_col]),
                "customers": int(row["customers"]),
                "churned": int(row["churned"]),
                "churn_rate": round(float(row["churn_rate"]), 4),
                "avg_monthly_charges": round(float(row["avg_monthly_charges"]), 2),
                "avg_tenure": round(float(row["avg_tenure"]), 2),
            }
        )
    return sorted(records, key=lambda x: x["churn_rate"], reverse=True)


def compute_eda_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive, measured EDA statistics from the cleaned dataset."""
    df = clean_df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    )
    df["monthly_charge_band"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 70, 90, 150],
        labels=["Low ($0-35)", "Mid ($35-70)", "High ($70-90)", "Premium ($90+)"],
    )
    df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({0: "Non-Senior (0)", 1: "Senior (1)"})

    total_customers = int(len(df))
    churned_customers = int(df[TARGET_COL].sum())
    retained_customers = total_customers - churned_customers
    overall_churn_rate = float(df[TARGET_COL].mean())

    # IQR outlier check for numeric columns
    numeric_summary: Dict[str, Any] = {}
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        s = df[col].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((s < lower) | (s > upper)).sum())
        numeric_summary[col] = {
            "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "q25": round(q1, 2),
            "q75": round(q3, 2),
            "max": round(float(s.max()), 2),
            "iqr_outlier_count": outlier_count,
            "mean_churned": round(float(df.loc[df[TARGET_COL] == 1, col].mean()), 2),
            "mean_retained": round(float(df.loc[df[TARGET_COL] == 0, col].mean()), 2),
        }

    # Monthly recurring revenue (measured directly from MonthlyCharges column)
    total_mrr = float(df["MonthlyCharges"].sum())
    churned_mrr = float(df.loc[df[TARGET_COL] == 1, "MonthlyCharges"].sum())

    return {
        "data_quality": inspect_raw_data(raw_df),
        "cleaned_rows": total_customers,
        "cleaned_columns": int(clean_df.shape[1]),
        "remaining_missing_values": {
            col: int(cnt) for col, cnt in clean_df.isna().sum().items() if cnt > 0
        },
        "target_distribution": {
            "total_customers": total_customers,
            "churned_customers": churned_customers,
            "retained_customers": retained_customers,
            "churn_rate": round(overall_churn_rate, 4),
            "retention_rate": round(1.0 - overall_churn_rate, 4),
            "imbalance_ratio_no_to_yes": round(retained_customers / max(churned_customers, 1), 2),
        },
        "measured_billing_metrics": {
            "total_monthly_charges_sum": round(total_mrr, 2),
            "churned_monthly_charges_sum": round(churned_mrr, 2),
            "churned_monthly_charges_share": round(churned_mrr / total_mrr, 4),
            "avg_monthly_charge_all": round(float(df["MonthlyCharges"].mean()), 2),
            "avg_monthly_charge_churned": round(
                float(df.loc[df[TARGET_COL] == 1, "MonthlyCharges"].mean()), 2
            ),
            "avg_monthly_charge_retained": round(
                float(df.loc[df[TARGET_COL] == 0, "MonthlyCharges"].mean()), 2
            ),
        },
        "numeric_summary": numeric_summary,
        "segment_analysis": {
            "Contract": _segment_churn_table(df, "Contract"),
            "tenure_bucket": _segment_churn_table(df, "tenure_bucket"),
            "monthly_charge_band": _segment_churn_table(df, "monthly_charge_band"),
            "PaymentMethod": _segment_churn_table(df, "PaymentMethod"),
            "InternetService": _segment_churn_table(df, "InternetService"),
            "TechSupport": _segment_churn_table(df, "TechSupport"),
            "OnlineSecurity": _segment_churn_table(df, "OnlineSecurity"),
            "PaperlessBilling": _segment_churn_table(df, "PaperlessBilling"),
            "SeniorCitizen": _segment_churn_table(df, "SeniorCitizenLabel"),
            "Partner": _segment_churn_table(df, "Partner"),
            "Dependents": _segment_churn_table(df, "Dependents"),
            "gender": _segment_churn_table(df, "gender"),
        },
    }


def generate_eda_figures(clean_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> list[Path]:
    """Generate and save EDA charts covering all key churn dimensions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = clean_df.copy()
    df["ChurnLabel"] = df[TARGET_COL].map({0: "Retained (No)", 1: "Churned (Yes)"})
    palette = {"Retained (No)": "#2E86AB", "Churned (Yes)": "#E05A47"}
    saved_paths: list[Path] = []

    # 1. Overall churn distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["ChurnLabel"].value_counts()
    bars = ax.bar(counts.index, counts.values, color=[palette[k] for k in counts.index], width=0.55)
    total = len(df)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{count:,}\n({count / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title("Overall Customer Churn Distribution (n = 7,043)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Customers")
    ax.set_ylim(0, counts.max() * 1.18)
    plt.tight_layout()
    p1 = output_dir / "01_churn_distribution.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved_paths.append(p1)

    # Helper for categorical churn-rate bar plots
    def _save_rate_plot(col: str, title: str, filename: str, figsize=(8, 5), rotate: int = 0) -> Path:
        stats = (
            df.groupby(col, observed=False)
            .agg(churn_rate=(TARGET_COL, "mean"), count=(TARGET_COL, "count"))
            .sort_values("churn_rate", ascending=False)
        )
        fig_, ax_ = plt.subplots(figsize=figsize)
        colors = sns.color_palette("Reds_r", n_colors=len(stats))
        bars_ = ax_.bar(stats.index.astype(str), stats["churn_rate"] * 100, color=colors, width=0.6)
        overall_pct = df[TARGET_COL].mean() * 100
        ax_.axhline(overall_pct, color="#333333", linestyle="--", linewidth=1.2, label=f"Overall Avg ({overall_pct:.1f}%)")
        for b, (_, row) in zip(bars_, stats.iterrows()):
            ax_.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 1.0,
                f"{row['churn_rate']:.1%}\n(n={int(row['count']):,})",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        ax_.set_title(title, fontsize=13, fontweight="bold")
        ax_.set_ylabel("Churn Rate (%)")
        ax_.set_ylim(0, min(100, stats["churn_rate"].max() * 100 + 12))
        if rotate:
            plt.xticks(rotation=rotate, ha="right")
        ax_.legend()
        plt.tight_layout()
        out = output_dir / filename
        fig_.savefig(out, dpi=150)
        plt.close(fig_)
        return out

    # 2. Churn by Contract Type
    saved_paths.append(
        _save_rate_plot("Contract", "Churn Rate by Contract Type", "02_churn_by_contract.png")
    )

    # 3. Churn by Tenure (Distribution + Bucket Rate)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(
        data=df,
        x="tenure",
        hue="ChurnLabel",
        bins=24,
        kde=True,
        palette=palette,
        alpha=0.55,
        ax=axes[0],
    )
    axes[0].set_title("Tenure Distribution by Churn Status", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Tenure (Months)")
    axes[0].set_ylabel("Customers")

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    )
    tb = df.groupby("tenure_bucket", observed=False)[TARGET_COL].agg(["mean", "count"])
    bars = axes[1].bar(tb.index.astype(str), tb["mean"] * 100, color="#E05A47", width=0.55)
    for b, (_, row) in zip(bars, tb.iterrows()):
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.0,
            f"{row['mean']:.1%}\n(n={int(row['count']):,})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[1].set_title("Churn Rate by Tenure Cohort", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Tenure Cohort")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_ylim(0, tb["mean"].max() * 100 + 12)
    plt.tight_layout()
    p3 = output_dir / "03_churn_by_tenure.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved_paths.append(p3)

    # 4. Churn by Monthly Charges
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.kdeplot(
        data=df,
        x="MonthlyCharges",
        hue="ChurnLabel",
        fill=True,
        common_norm=False,
        palette=palette,
        alpha=0.45,
        ax=axes[0],
    )
    axes[0].set_title("Monthly Charges Density by Churn Status", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Monthly Charges ($)")

    sns.boxplot(
        data=df,
        x="ChurnLabel",
        y="MonthlyCharges",
        hue="ChurnLabel",
        palette=palette,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Monthly Charges Spread by Churn Status", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Monthly Charges ($)")
    plt.tight_layout()
    p4 = output_dir / "04_churn_by_monthly_charges.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved_paths.append(p4)

    # 5. Churn by Payment Method
    saved_paths.append(
        _save_rate_plot(
            "PaymentMethod",
            "Churn Rate by Payment Method",
            "05_churn_by_payment_method.png",
            figsize=(9, 5),
            rotate=15,
        )
    )

    # 6. Churn by Internet Service
    saved_paths.append(
        _save_rate_plot(
            "InternetService",
            "Churn Rate by Internet Service Type",
            "06_churn_by_internet_service.png",
        )
    )

    # 7. Demographic and Support/Security Service Patterns
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    df["SeniorLabel"] = df["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
    panels = [
        ("SeniorLabel", "Churn Rate: Senior Citizen Status"),
        ("TechSupport", "Churn Rate: Tech Support"),
        ("OnlineSecurity", "Churn Rate: Online Security"),
        ("PaperlessBilling", "Churn Rate: Paperless Billing"),
    ]
    for ax, (col, title) in zip(axes.flat, panels):
        st = df.groupby(col)[TARGET_COL].agg(["mean", "count"]).sort_values("mean", ascending=False)
        bars_ = ax.bar(st.index.astype(str), st["mean"] * 100, color="#3B7EA1", width=0.55)
        for b, (_, row) in zip(bars_, st.iterrows()):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 1.0,
                f"{row['mean']:.1%}\n(n={int(row['count']):,})",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, st["mean"].max() * 100 + 13)
    plt.tight_layout()
    p7 = output_dir / "07_churn_by_demographics_and_services.png"
    fig.savefig(p7, dpi=150)
    plt.close(fig)
    saved_paths.append(p7)

    # 8. Numeric Correlation & Outlier Check
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", TARGET_COL]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0])
    axes[0].set_title("Numeric Feature Correlation Matrix", fontsize=12, fontweight="bold")

    norm_df = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
    norm_df = (norm_df - norm_df.mean()) / norm_df.std()
    sns.boxplot(data=norm_df, orient="h", color="#5DA5DA", ax=axes[1])
    axes[1].set_title("Standardized Numeric Distributions (Outlier Check)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Z-Score")
    plt.tight_layout()
    p8 = output_dir / "08_numeric_correlation_and_outliers.png"
    fig.savefig(p8, dpi=150)
    plt.close(fig)
    saved_paths.append(p8)

    return saved_paths


def run_data_pipeline() -> Dict[str, Any]:
    """Execute full cleaning + EDA summary + visualization generation pipeline."""
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df, include_target=True)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)

    summary = compute_eda_summary(raw_df, clean_df)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(EDA_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    figures = generate_eda_figures(clean_df)
    summary["generated_figures"] = [str(p.relative_to(PROCESSED_DATA_PATH.parents[2])) for p in figures]
    with open(EDA_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    result = run_data_pipeline()
    print(f"Saved cleaned dataset to: {PROCESSED_DATA_PATH}")
    print(f"Rows: {result['cleaned_rows']} | Columns: {result['cleaned_columns']}")
    print(f"Blank TotalCharges handled: {result['data_quality']['blank_total_charges_count']}")
    print(
        f"Overall churn rate: {result['target_distribution']['churn_rate']:.2%} "
        f"({result['target_distribution']['churned_customers']}/{result['target_distribution']['total_customers']})"
    )
    print(f"Generated {len(result['generated_figures'])} EDA figures and saved {EDA_SUMMARY_PATH}")
