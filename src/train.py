"""End-to-end model training, cross-validation, hyperparameter tuning, and artifact persistence."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_LOW_RISK_THRESHOLD,
    EVALUATION_DIR,
    EVALUATION_SUMMARY_PATH,
    LATEST_MODEL_PATH,
    METADATA_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_COMPARISON_CSV_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_FEATURE_COLUMNS,
    SHAP_BACKGROUND_PATH,
    TARGET_COL,
    TEST_SIZE,
)
from src.data_processing import run_data_pipeline
from src.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_threshold_tradeoff,
)
from src.feature_engineering import (
    TelcoFeatureEngineer,
    build_column_transformer,
    build_preprocessor,
    get_transformed_feature_names,
)


def load_or_prepare_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the cleaned dataset (running preprocessing first if needed)."""
    if not PROCESSED_DATA_PATH.exists():
        run_data_pipeline()

    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df[RAW_FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y


def _cv_evaluate(pipeline: Any, X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> Dict[str, float]:
    """Run stratified cross-validation and return mean scores."""
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=["roc_auc", "average_precision", "f1", "recall", "precision", "accuracy"],
        n_jobs=1,
    )
    return {
        "cv_roc_auc": round(float(np.mean(scores["test_roc_auc"])), 4),
        "cv_pr_auc": round(float(np.mean(scores["test_average_precision"])), 4),
        "cv_f1": round(float(np.mean(scores["test_f1"])), 4),
        "cv_recall": round(float(np.mean(scores["test_recall"])), 4),
        "cv_precision": round(float(np.mean(scores["test_precision"])), 4),
        "cv_accuracy": round(float(np.mean(scores["test_accuracy"])), 4),
    }


def train_and_evaluate() -> Dict[str, Any]:
    """Train baseline, class-weighted, SMOTE, and tuned models; evaluate and persist best artifact."""
    X, y = load_or_prepare_dataset()

    # Stratified train/test split BEFORE fitting any transformation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    # 1. Baseline & untuned model comparisons (including imbalance strategies)
    candidate_pipelines: Dict[str, Any] = {
        "Baseline Logistic Regression (Unweighted)": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Logistic Regression (Balanced)": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Logistic Regression (SMOTE)": ImbPipeline(
            steps=[
                ("feature_engineer", TelcoFeatureEngineer()),
                ("column_transformer", build_column_transformer(scale_numeric=True)),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest (Balanced)": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Random Forest (SMOTE)": ImbPipeline(
            steps=[
                ("feature_engineer", TelcoFeatureEngineer()),
                ("column_transformer", build_column_transformer(scale_numeric=False)),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "XGBoost (Weighted)": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        tree_method="hist",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "XGBoost (SMOTE)": ImbPipeline(
            steps=[
                ("feature_engineer", TelcoFeatureEngineer()),
                ("column_transformer", build_column_transformer(scale_numeric=False)),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        tree_method="hist",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    records = []
    fitted_models: Dict[str, Any] = {}
    test_probs: Dict[str, np.ndarray] = {}

    for name, pipe in candidate_pipelines.items():
        print(f"Evaluating: {name}...", flush=True)
        cv_metrics = _cv_evaluate(pipe, X_train, y_train, cv)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        test_metrics = compute_classification_metrics(y_test, probs, threshold=0.50)

        fitted_models[name] = pipe
        test_probs[name] = probs
        records.append(
            {
                "model_name": name,
                "stage": "comparison",
                **cv_metrics,
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1_score": test_metrics["f1_score"],
                "roc_auc": test_metrics["roc_auc"],
                "pr_auc": test_metrics["pr_auc"],
                "tn": test_metrics["confusion_matrix"]["tn"],
                "fp": test_metrics["confusion_matrix"]["fp"],
                "fn": test_metrics["confusion_matrix"]["fn"],
                "tp": test_metrics["confusion_matrix"]["tp"],
            }
        )

    # 2. Hyperparameter tuning for the three model families (leakage-free via Pipeline + StratifiedKFold)
    tuning_specs = {
        "Logistic Regression (Tuned)": (
            Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(scale_numeric=True)),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "classifier__C": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
            12,
        ),
        "Random Forest (Tuned)": (
            Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(scale_numeric=False)),
                    (
                        "classifier",
                        RandomForestClassifier(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "classifier__n_estimators": [150, 250, 350],
                "classifier__max_depth": [6, 8, 10, 12, 16, None],
                "classifier__min_samples_split": [2, 5, 10, 15],
                "classifier__min_samples_leaf": [2, 4, 6, 8],
                "classifier__max_features": ["sqrt", "log2", 0.5],
            },
            14,
        ),
        "XGBoost (Tuned)": (
            Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(scale_numeric=False)),
                    (
                        "classifier",
                        XGBClassifier(
                            scale_pos_weight=scale_pos_weight,
                            tree_method="hist",
                            eval_metric="logloss",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "classifier__n_estimators": [120, 180, 250, 320],
                "classifier__max_depth": [2, 3, 4, 5],
                "classifier__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
                "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
                "classifier__colsample_bytree": [0.6, 0.75, 0.9, 1.0],
                "classifier__min_child_weight": [1, 3, 5, 8],
                "classifier__reg_lambda": [1.0, 2.0, 5.0, 10.0],
            },
            18,
        ),
    }

    tuning_results: Dict[str, Any] = {}
    for name, (pipe, param_dist, n_iter) in tuning_specs.items():
        print(f"Tuning: {name} ({n_iter} iterations)...", flush=True)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_est = search.best_estimator_

        cv_metrics = _cv_evaluate(best_est, X_train, y_train, cv)
        probs = best_est.predict_proba(X_test)[:, 1]
        test_metrics = compute_classification_metrics(y_test, probs, threshold=0.50)

        fitted_models[name] = best_est
        test_probs[name] = probs
        tuning_results[name] = {
            "best_cv_roc_auc": round(float(search.best_score_), 4),
            "best_params": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in search.best_params_.items()},
        }
        records.append(
            {
                "model_name": name,
                "stage": "tuned",
                **cv_metrics,
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1_score": test_metrics["f1_score"],
                "roc_auc": test_metrics["roc_auc"],
                "pr_auc": test_metrics["pr_auc"],
                "tn": test_metrics["confusion_matrix"]["tn"],
                "fp": test_metrics["confusion_matrix"]["fp"],
                "fn": test_metrics["confusion_matrix"]["fn"],
                "tp": test_metrics["confusion_matrix"]["tp"],
            }
        )

    comparison_df = pd.DataFrame(records).sort_values(
        by=["cv_roc_auc", "cv_f1", "cv_recall"], ascending=False
    ).reset_index(drop=True)

    # Select best tuned model using CV ROC-AUC + CV PR-AUC + CV F1 (never accuracy alone, never leaking test set into selection)
    tuned_df = comparison_df[comparison_df["stage"] == "tuned"].copy()
    tuned_df["selection_score"] = (
        0.45 * tuned_df["cv_roc_auc"]
        + 0.30 * tuned_df["cv_pr_auc"]
        + 0.25 * tuned_df["cv_f1"]
    )
    best_row = tuned_df.sort_values("selection_score", ascending=False).iloc[0]
    best_model_name = str(best_row["model_name"])
    best_pipeline = fitted_models[best_model_name]
    best_test_probs = test_probs[best_model_name]

    # Persist artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, MODEL_ARTIFACT_PATH)
    shutil.copy2(MODEL_ARTIFACT_PATH, LATEST_MODEL_PATH)

    # Save a compact stratified background sample (raw features) for fast deterministic SHAP explanations
    bg_size = min(200, len(X_train))
    bg_X, _, bg_y, _ = train_test_split(
        X_train,
        y_train,
        train_size=bg_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    bg_out = bg_X.copy()
    bg_out[TARGET_COL] = bg_y.values
    bg_out.to_csv(SHAP_BACKGROUND_PATH, index=False)

    # Save comparison table and evaluation figures
    comparison_df.to_csv(MODEL_COMPARISON_CSV_PATH, index=False)

    key_plot_models = [
        "Baseline Logistic Regression (Unweighted)",
        "Logistic Regression (Tuned)",
        "Random Forest (Tuned)",
        "XGBoost (Tuned)",
    ]
    plot_df = comparison_df[comparison_df["model_name"].isin(key_plot_models)].copy()
    plot_probs = {k: test_probs[k] for k in key_plot_models if k in test_probs}

    fig_paths = [
        plot_model_comparison(plot_df),
        plot_roc_curves(y_test, plot_probs),
        plot_precision_recall_curves(y_test, plot_probs),
        plot_confusion_matrix(y_test, best_test_probs, best_model_name, threshold=0.50),
        plot_threshold_tradeoff(y_test, best_test_probs, best_model_name),
    ]

    best_test_metrics = compute_classification_metrics(y_test, best_test_probs, threshold=0.50)
    transformed_features = get_transformed_feature_names(best_pipeline.named_steps["preprocessor"])

    metadata = {
        "model_version": MODEL_VERSION,
        "selected_model": best_model_name,
        "classifier_class": type(best_pipeline.named_steps["classifier"]).__name__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_criterion": "0.45 * CV_ROC_AUC + 0.30 * CV_PR_AUC + 0.25 * CV_F1 (selected on CV folds only; never accuracy alone)",
        "dataset_split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size": TEST_SIZE,
            "stratified": True,
            "train_churn_rate": round(float(y_train.mean()), 4),
            "test_churn_rate": round(float(y_test.mean()), 4),
            "scale_pos_weight": round(float(scale_pos_weight), 4),
        },
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "transformed_feature_count": len(transformed_features),
        "transformed_feature_names": transformed_features,
        "risk_thresholds": {
            "low_max": DEFAULT_LOW_RISK_THRESHOLD,
            "high_min": DEFAULT_HIGH_RISK_THRESHOLD,
            "note": "Operational thresholds configurable via environment variables or UI controls; not universal telecom standards.",
        },
        "best_hyperparameters": tuning_results.get(best_model_name, {}).get("best_params", {}),
        "selected_model_cv_metrics": {
            "cv_roc_auc": float(best_row["cv_roc_auc"]),
            "cv_pr_auc": float(best_row["cv_pr_auc"]),
            "cv_f1": float(best_row["cv_f1"]),
            "cv_recall": float(best_row["cv_recall"]),
            "cv_precision": float(best_row["cv_precision"]),
            "cv_accuracy": float(best_row["cv_accuracy"]),
        },
        "selected_model_test_metrics": best_test_metrics,
        "artifact_paths": {
            "versioned_pipeline": str(MODEL_ARTIFACT_PATH.name),
            "latest_pipeline": str(LATEST_MODEL_PATH.name),
            "shap_background": str(SHAP_BACKGROUND_PATH.name),
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    eval_summary = {
        "selected_model": best_model_name,
        "model_version": MODEL_VERSION,
        "selection_criterion": metadata["selection_criterion"],
        "dataset_split": metadata["dataset_split"],
        "tuning_results": tuning_results,
        "selected_model_test_metrics": best_test_metrics,
        "all_models": comparison_df.to_dict(orient="records"),
        "generated_figures": [p.name for p in fig_paths],
    }

    with open(EVALUATION_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary


if __name__ == "__main__":
    summary = train_and_evaluate()
    print(f"\nSelected Production Model: {summary['selected_model']} ({summary['model_version']})")
    tm = summary["selected_model_test_metrics"]
    print(
        f"Test Metrics -> ROC-AUC: {tm['roc_auc']:.4f} | PR-AUC: {tm['pr_auc']:.4f} | "
        f"F1: {tm['f1_score']:.4f} | Recall: {tm['recall']:.4f} | "
        f"Precision: {tm['precision']:.4f} | Accuracy: {tm['accuracy']:.4f}"
    )
    print(f"Confusion Matrix -> {tm['confusion_matrix']}")
    print(f"Saved model artifact: {MODEL_ARTIFACT_PATH}")
