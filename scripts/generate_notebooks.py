"""Generate the four structured Jupyter notebooks for EDA, feature engineering, modeling, and SHAP."""

from pathlib import Path
import nbformat as nbf


def _make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def main():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)

    # 01_data_exploration.ipynb
    nb1 = _make_nb(
        [
            nbf.v4.new_markdown_cell(
                "# 01 — Exploratory Data Analysis & Business Understanding\n\n"
                "This notebook inspects the raw **IBM Telco Customer Churn** dataset, validates data quality "
                "(missing values, blank `TotalCharges`, duplicates, outliers), quantifies class imbalance, "
                "and highlights the most churn-vulnerable customer segments."
            ),
            nbf.v4.new_code_cell(
                "import sys\nfrom pathlib import Path\n"
                "sys.path.insert(0, str(Path.cwd().parent))\n\n"
                "import pandas as pd\nimport numpy as np\n"
                "from IPython.display import Image, display\n"
                "from src.data_processing import load_raw_data, inspect_raw_data, clean_data, compute_eda_summary\n"
                "from src.config import FIGURES_DIR\n\n"
                "raw_df = load_raw_data()\n"
                "quality = inspect_raw_data(raw_df)\n"
                "print('Raw Shape:', raw_df.shape)\n"
                "print('Duplicates:', quality['duplicate_rows'])\n"
                "print('Blank TotalCharges:', quality['blank_total_charges_count'])\n"
                "raw_df.head()"
            ),
            nbf.v4.new_code_cell(
                "# Inspect rows where TotalCharges is blank\n"
                "blank_mask = raw_df['TotalCharges'].astype(str).str.strip().eq('')\n"
                "raw_df.loc[blank_mask, ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]"
            ),
            nbf.v4.new_code_cell(
                "# Clean dataset deterministically (tenure == 0 -> TotalCharges = 0.0) and inspect target imbalance\n"
                "clean_df = clean_data(raw_df, include_target=True)\n"
                "summary = compute_eda_summary(raw_df, clean_df)\n"
                "print('Target Distribution:', summary['target_distribution'])\n"
                "print('Measured Billing Metrics:', summary['measured_billing_metrics'])\n"
                "print('Numeric Summary:')\n"
                "pd.DataFrame(summary['numeric_summary']).T"
            ),
            nbf.v4.new_code_cell(
                "# Display Churn by Key Segments\n"
                "for seg_key in ['Contract', 'tenure_bucket', 'PaymentMethod', 'InternetService', 'TechSupport']:\n"
                "    print(f'\\n=== Churn by {seg_key} ===')\n"
                "    display(pd.DataFrame(summary['segment_analysis'][seg_key]))"
            ),
            nbf.v4.new_code_cell(
                "# Display generated EDA figures\n"
                "for fname in sorted(p.name for p in FIGURES_DIR.glob('0[1-8]_*.png')):\n"
                "    print(fname)\n"
                "    display(Image(filename=str(FIGURES_DIR / fname), width=750))"
            ),
            nbf.v4.new_markdown_cell(
                "## Key Business Takeaways\n"
                "- **Class Imbalance**: ~26.5% churn rate (approx. 2.77:1 retained-to-churned ratio). Accuracy alone is not a reliable metric.\n"
                "- **Highest-Risk Segments**: Month-to-month contracts, 0–12 month tenure customers, Fiber optic internet subscribers, "
                "Electronic check payers, and accounts without Tech Support / Online Security.\n"
                "- **Measured Revenue Exposure vs. Assumptions**: Churned accounts represent a disproportionately higher share of monthly "
                "charges due to higher average monthly bills. Actual ROI of retention offers depends on company-specific acquisition cost, "
                "gross margin, and offer lift (which must be measured via controlled retention experiments)."
            ),
        ]
    )

    # 02_feature_engineering.ipynb
    nb2 = _make_nb(
        [
            nbf.v4.new_markdown_cell(
                "# 02 — Leakage-Free Feature Engineering & Preprocessing Pipeline\n\n"
                "Demonstrates the reusable `TelcoFeatureEngineer` + `ColumnTransformer` pipeline from `src/feature_engineering.py`, "
                "verifying strict train/test separation, zero NaNs after transformation, and deterministic feature ordering."
            ),
            nbf.v4.new_code_cell(
                "import sys\nfrom pathlib import Path\n"
                "sys.path.insert(0, str(Path.cwd().parent))\n\n"
                "import numpy as np\nimport pandas as pd\n"
                "from sklearn.model_selection import train_test_split\n"
                "from src.config import PROCESSED_DATA_PATH, RAW_FEATURE_COLUMNS, TARGET_COL, RANDOM_STATE, TEST_SIZE\n"
                "from src.feature_engineering import TelcoFeatureEngineer, build_preprocessor, get_transformed_feature_names\n\n"
                "df = pd.read_csv(PROCESSED_DATA_PATH)\n"
                "X = df[RAW_FEATURE_COLUMNS]\n"
                "y = df[TARGET_COL]\n\n"
                "X_train, X_test, y_train, y_test = train_test_split(\n"
                "    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE\n"
                ")\n"
                "print(f'Train: {X_train.shape} (churn={y_train.mean():.3%}) | Test: {X_test.shape} (churn={y_test.mean():.3%})')"
            ),
            nbf.v4.new_code_cell(
                "# Inspect engineered features before encoding\n"
                "fe = TelcoFeatureEngineer()\n"
                "eng_sample = fe.fit_transform(X_train.head(5))\n"
                "eng_sample[['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_spend', 'charge_ratio', 'service_count', 'has_support_bundle', 'is_auto_pay', 'tenure_group']]"
            ),
            nbf.v4.new_code_cell(
                "# Fit preprocessor strictly on X_train, then transform X_test\n"
                "preprocessor = build_preprocessor(scale_numeric=True)\n"
                "X_train_t = preprocessor.fit_transform(X_train)\n"
                "X_test_t = preprocessor.transform(X_test)\n"
                "feature_names = get_transformed_feature_names(preprocessor)\n"
                "print('Transformed Train Shape:', X_train_t.shape)\n"
                "print('Transformed Test Shape:', X_test_t.shape)\n"
                "print('Any NaN/Inf in Test:', not np.isfinite(X_test_t).all())\n"
                "print('First 15 transformed features:', feature_names[:15])"
            ),
        ]
    )

    # 03_model_training.ipynb
    nb3 = _make_nb(
        [
            nbf.v4.new_markdown_cell(
                "# 03 — Model Training, Imbalance Handling, Tuning & Evaluation\n\n"
                "Reviews the benchmark results across Logistic Regression, Random Forest, and XGBoost "
                "(comparing unweighted baseline, class weighting, inside-fold SMOTE, and RandomizedSearchCV tuning)."
            ),
            nbf.v4.new_code_cell(
                "import sys, json\nfrom pathlib import Path\n"
                "sys.path.insert(0, str(Path.cwd().parent))\n\n"
                "import pandas as pd\nfrom IPython.display import Image, display\n"
                "from src.config import MODEL_COMPARISON_CSV_PATH, EVALUATION_SUMMARY_PATH, METADATA_PATH, FIGURES_DIR\n\n"
                "comp_df = pd.read_csv(MODEL_COMPARISON_CSV_PATH)\n"
                "comp_df[['model_name', 'stage', 'cv_roc_auc', 'cv_pr_auc', 'cv_f1', 'roc_auc', 'pr_auc', 'f1_score', 'recall', 'precision', 'accuracy']]"
            ),
            nbf.v4.new_code_cell(
                "with open(METADATA_PATH, 'r', encoding='utf-8') as f:\n"
                "    meta = json.load(f)\n"
                "print('Selected Model:', meta['selected_model'])\n"
                "print('Selection Criterion:', meta['selection_criterion'])\n"
                "print('Best Hyperparameters:', meta['best_hyperparameters'])\n"
                "print('Held-out Test Metrics:', meta['selected_model_test_metrics'])"
            ),
            nbf.v4.new_code_cell(
                "for fname in ['09_model_comparison.png', '10_roc_curves.png', '11_pr_curves.png', '12_confusion_matrix_best.png', '13_threshold_analysis.png']:\n"
                "    p = FIGURES_DIR / fname\n"
                "    if p.exists():\n"
                "        print(fname)\n"
                "        display(Image(filename=str(p), width=750))"
            ),
        ]
    )

    # 04_model_explainability.ipynb
    nb4 = _make_nb(
        [
            nbf.v4.new_markdown_cell(
                "# 04 — Global & Customer-Level SHAP Model Explainability\n\n"
                "Inspects global SHAP feature importance, dependence plots, and an individual customer risk breakdown "
                "using the exact serialized production pipeline."
            ),
            nbf.v4.new_code_cell(
                "import sys, json\nfrom pathlib import Path\n"
                "sys.path.insert(0, str(Path.cwd().parent))\n\n"
                "import pandas as pd\nfrom IPython.display import Image, display\n"
                "from src.config import SHAP_SUMMARY_PATH, FIGURES_DIR\n"
                "from src.predict import ChurnPredictor\n\n"
                "with open(SHAP_SUMMARY_PATH, 'r', encoding='utf-8') as f:\n"
                "    shap_summary = json.load(f)\n"
                "print('Explainer:', shap_summary['explainer_type'], '| Evaluated Rows:', shap_summary['evaluated_rows'])\n"
                "pd.DataFrame(shap_summary['top_global_features'])"
            ),
            nbf.v4.new_code_cell(
                "for fname in ['14_shap_summary_beeswarm.png', '15_shap_feature_importance_bar.png', '16_shap_dependence_top_features.png', '17_shap_local_high_risk_example.png']:\n"
                "    p = FIGURES_DIR / fname\n"
                "    if p.exists():\n"
                "        print(fname)\n"
                "        display(Image(filename=str(p), width=780))"
            ),
            nbf.v4.new_code_cell(
                "# Run live individual prediction + SHAP explanation through ChurnPredictor\n"
                "predictor = ChurnPredictor()\n"
                "sample = shap_summary['sample_high_risk_customer']['profile']\n"
                "res = predictor.predict_one(sample, include_explanation=True)\n"
                "print(f\"Predicted Churn Probability: {res['churn_probability']:.2%} ({res['risk_category']} Risk)\")\n"
                "print('Top Risk-Increasing Factors:', res['explanation']['top_risk_increasing_factors'])\n"
                "print('Retention Considerations:')\n"
                "for r in res['explanation']['retention_considerations']:\n"
                "    print(' -', r)"
            ),
        ]
    )

    nbf.write(nb1, nb_dir / "01_data_exploration.ipynb")
    nbf.write(nb2, nb_dir / "02_feature_engineering.ipynb")
    nbf.write(nb3, nb_dir / "03_model_training.ipynb")
    nbf.write(nb4, nb_dir / "04_model_explainability.ipynb")
    print("Generated 4 notebooks in notebooks/")


if __name__ == "__main__":
    main()
