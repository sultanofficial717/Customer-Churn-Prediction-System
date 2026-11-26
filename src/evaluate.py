"""Model evaluation metrics, threshold analysis, and visualization generators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR

sns.set_theme(style="whitegrid")


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compute binary classification metrics and confusion matrix counts."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred)), 4),
        "precision": round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4),
        "pr_auc": round(float(average_precision_score(y_true_arr, y_prob_arr)), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Plot side-by-side test metrics across trained models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["roc_auc", "pr_auc", "f1_score", "recall", "precision", "accuracy"]
    melted = comparison_df.melt(
        id_vars=["model_name"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted, x="Metric", y="Score", hue="model_name", ax=ax)
    ax.set_title("Model Performance Comparison on Held-Out Test Set", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = output_dir / "09_model_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_roc_curves(
    y_true: np.ndarray | pd.Series,
    model_probs: Dict[str, np.ndarray],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot ROC curves for all evaluated models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_val = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier (0.500)")
    ax.set_title("ROC Curves — Held-Out Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = output_dir / "10_roc_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_precision_recall_curves(
    y_true: np.ndarray | pd.Series,
    model_probs: Dict[str, np.ndarray],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot Precision-Recall curves for all evaluated models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline_rate = float(np.mean(y_true))
    for name, probs in model_probs.items():
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        ax.plot(rec, prec, linewidth=2, label=f"{name} (AP = {ap:.3f})")

    ax.axhline(
        baseline_rate,
        linestyle="--",
        color="gray",
        label=f"No-Skill Baseline ({baseline_rate:.3f})",
    )
    ax.set_title("Precision-Recall Curves — Held-Out Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = output_dir / "11_pr_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    model_name: str,
    threshold: float = 0.50,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot confusion matrix with counts and row percentages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    labels = np.array(
        [[f"{cm[i, j]:,}\n({cm_pct[i, j]:.1%})" for j in range(2)] for i in range(2)]
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted: Retain", "Predicted: Churn"],
        yticklabels=["Actual: Retain", "Actual: Churn"],
        ax=ax,
    )
    ax.set_title(
        f"Confusion Matrix — {model_name}\n(Decision Threshold = {threshold:.2f})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out = output_dir / "12_confusion_matrix_best.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_threshold_tradeoff(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot Precision, Recall, and F1-score across probability decision thresholds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(0.10, 0.90, 81)
    precisions, recalls, f1s = [], [], []
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    for thr in thresholds:
        preds = (y_prob_arr >= thr).astype(int)
        precisions.append(precision_score(y_true_arr, preds, zero_division=0))
        recalls.append(recall_score(y_true_arr, preds, zero_division=0))
        f1s.append(f1_score(y_true_arr, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, label="Precision", linewidth=2, color="#2E86AB")
    ax.plot(thresholds, recalls, label="Recall", linewidth=2, color="#E05A47")
    ax.plot(thresholds, f1s, label="F1-Score", linewidth=2.2, color="#2A9D8F")
    ax.axvline(0.50, linestyle="--", color="gray", alpha=0.8, label="Default Threshold (0.50)")
    ax.set_title(f"Precision / Recall / F1 Trade-Off — {model_name}", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.legend()
    plt.tight_layout()
    out = output_dir / "13_threshold_analysis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
