# 📊 Project Results — End-to-End Customer Churn Prediction System

> **All figures, metrics, and test outcomes below are measured directly from the executed codebase, dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), trained artifacts (`v1.0.0`), and automated test suite. No values are estimated or fabricated.**

---

## 1. Executive Summary

| Category | Result |
|---|---|
| **Dataset** | IBM Telco Customer Churn (`7,043` unique customers, `21` columns) |
| **Observed Churn Rate** | **26.54%** (`1,869` churned / `5,174` retained; `2.77:1` imbalance ratio) |
| **Train / Test Split** | Stratified 80/20 (`5,634` train / `1,409` test; `26.54%` churn in both) |
| **Models Evaluated** | **10** configurations across Logistic Regression, Random Forest, and XGBoost (unweighted baseline, class-weighted, inside-fold SMOTE, and tuned) |
| **Selected Production Model** | **XGBoost (Tuned)** (`models/churn_pipeline_v1.0.0.joblib`, 299 KB) |
| **Test ROC-AUC / PR-AUC** | **0.8464** / **0.6648** |
| **Test Recall / Precision / F1** | **79.95%** (`299/374` churners caught) / **50.59%** / **0.6197** |
| **SHAP Explainer** | `TreeExplainer` evaluated across all `1,409` held-out test records |
| **Automated Test Suite** | **22 / 22 PASSED** (`pytest -v` in 5.03s) |
| **Visualizations & Notebooks** | **17** high-resolution figures (`reports/figures/`) + **4** executed Jupyter notebooks |

---

## 2. Data Quality & Cleaning Results

Source: `reports/evaluation/eda_summary.json`

- **Raw Shape**: `7,043` rows × `21` columns
- **Exact Duplicate Rows**: `0`
- **Duplicate `customerID`s**: `0`
- **Explicit Nulls**: `0`
- **Blank `TotalCharges` Strings**: `11` rows — verified that all `11` accounts have `tenure == 0`; deterministically set to `0.0` (zero train/test leakage)
- **Remaining Missing Values After Cleaning**: `0`
- **IQR Outliers (`tenure`, `MonthlyCharges`, `TotalCharges`)**: `0`

### Numeric Feature Distributions (Churned vs. Retained)

| Feature | Overall Mean | Median | Min – Max | Mean (Churned) | Mean (Retained) |
|---|---:|---:|---:|---:|---:|
| `tenure` (months) | 32.37 | 29.00 | 0.0 – 72.0 | **17.98** | **37.57** |
| `MonthlyCharges` ($) | 64.76 | 70.35 | 18.25 – 118.75 | **74.44** | **61.27** |
| `TotalCharges` ($) | 2,279.73 | 1,394.55 | 0.00 – 8,684.80 | **1,531.80** | **2,549.91** |

### Measured Monthly Billing Exposure

- **Total Monthly Charges (All 7,043 Accounts)**: `$456,116.60`
- **Monthly Charges from Churned Accounts (1,869 Accounts)**: `$139,130.85` (**30.5%** of total monthly billing)
- *Note*: Actual net financial savings from retention outreach depend on unobserved company-specific variables (acquisition cost, gross margin, incentive cost, and true causal uplift).

---

## 3. Exploratory Data Analysis — Churn by Customer Segment

### By Contract Type
| Segment | Customers | Churned | Churn Rate | Avg Monthly Charges | Avg Tenure (m) |
|---|---:|---:|---:|---:|---:|
| **Month-to-month** | 3,875 | 1,655 | **42.71%** | $66.40 | 18.04 |
| One year | 1,473 | 166 | 11.27% | $65.05 | 42.04 |
| Two year | 1,695 | 48 | **2.83%** | $60.77 | 56.74 |

### By Tenure Cohort
| Segment | Customers | Churned | Churn Rate | Avg Monthly Charges | Avg Tenure (m) |
|---|---:|---:|---:|---:|---:|
| **0–12 months** | 2,186 | 1,037 | **47.44%** | $56.10 | 4.73 |
| 13–24 months | 1,024 | 294 | 28.71% | $61.36 | 18.35 |
| 25–48 months | 1,594 | 325 | 20.39% | $65.93 | 36.19 |
| 49–72 months | 2,239 | 213 | **9.51%** | $73.95 | 63.05 |

### By Payment Method
| Segment | Customers | Churned | Churn Rate | Avg Monthly Charges | Avg Tenure (m) |
|---|---:|---:|---:|---:|---:|
| **Electronic check** | 2,365 | 1,071 | **45.29%** | $76.26 | 25.17 |
| Mailed check | 1,612 | 308 | 19.11% | $43.92 | 21.83 |
| Bank transfer (automatic) | 1,544 | 258 | 16.71% | $67.19 | 43.66 |
| Credit card (automatic) | 1,522 | 232 | **15.24%** | $66.51 | 43.27 |

### By Internet Service, Support & Demographics
| Dimension | Highest-Risk Group | Churn Rate | Lowest-Risk Group | Churn Rate |
|---|---|---:|---|---:|
| **Internet Service** | Fiber optic (n=3,096) | **41.89%** | No internet (n=1,526) | 7.40% |
| **Online Security** | No (n=3,498) | **41.77%** | No internet / Yes | 7.40% / 14.61% |
| **Tech Support** | No (n=3,473) | **41.64%** | No internet / Yes | 7.40% / 15.17% |
| **Senior Citizen** | Senior = 1 (n=1,142) | **41.68%** | Non-Senior = 0 (n=5,901) | 23.61% |
| **Paperless Billing** | Yes (n=4,171) | **33.57%** | No (n=2,872) | 16.33% |
| **Partner** | No (n=3,641) | **32.96%** | Yes (n=3,402) | 19.66% |
| **Dependents** | No (n=4,933) | **31.28%** | Yes (n=2,110) | 15.45% |
| **Gender** | Female (n=3,488) | 26.92% | Male (n=3,555) | 26.16% |

---

## 4. Machine Learning Benchmark & Tuning Results

Source: `reports/evaluation/evaluation_summary.json` & `reports/evaluation/model_comparison.csv`

- **Pipeline**: Stateless `TelcoFeatureEngineer` (6 engineered features: `avg_monthly_spend`, `charge_ratio`, `service_count`, `has_support_bundle`, `is_auto_pay`, `tenure_group`) + `ColumnTransformer` producing **54** dense features.
- **Validation Strategy**: 5-fold Stratified Cross-Validation on `5,634` training rows; SMOTE applied strictly inside training folds via `imblearn.pipeline.Pipeline`.
- **Selection Criterion**: `0.45 * CV_ROC_AUC + 0.30 * CV_PR_AUC + 0.25 * CV_F1` (selected strictly on CV folds, never raw accuracy alone).

### Full 10-Model Comparison Table

| Model | Stage | CV ROC-AUC | CV PR-AUC | CV F1 | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Accuracy | TP | FN | FP | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost (Tuned)** *(Selected)* | tuned | **0.8500** | **0.6677** | 0.6323 | **0.8464** | **0.6648** | 0.6197 | **0.7995** | 0.5059 | 0.7395 | 299 | 75 | 292 | 743 |
| Random Forest (Tuned) | tuned | 0.8477 | 0.6646 | 0.6297 | 0.8424 | 0.6503 | 0.6290 | 0.7888 | 0.5230 | 0.7530 | 295 | 79 | 269 | 766 |
| XGBoost (Weighted) | comparison | 0.8471 | 0.6618 | 0.6340 | 0.8460 | 0.6569 | 0.6378 | 0.7888 | 0.5354 | 0.7622 | 295 | 79 | 256 | 779 |
| Random Forest (Balanced) | comparison | 0.8462 | 0.6622 | 0.6337 | 0.8420 | 0.6559 | 0.6276 | 0.7727 | 0.5283 | 0.7566 | 289 | 85 | 258 | 777 |
| Logistic Regression (Tuned) | tuned | 0.8458 | 0.6617 | 0.6290 | 0.8414 | 0.6323 | 0.6123 | 0.7834 | 0.5026 | 0.7367 | 293 | 81 | 290 | 745 |
| Baseline Logistic Regression (Unweighted) | comparison | 0.8458 | 0.6630 | 0.5957 | 0.8426 | 0.6378 | 0.5804 | 0.5214 | 0.6544 | 0.7999 | 195 | 179 | 103 | 932 |
| Logistic Regression (Balanced) | comparison | 0.8456 | 0.6620 | 0.6251 | 0.8422 | 0.6363 | 0.6146 | 0.7888 | 0.5034 | 0.7374 | 295 | 79 | 291 | 744 |
| XGBoost (SMOTE) | comparison | 0.8448 | 0.6547 | 0.6039 | 0.8471 | 0.6644 | 0.6006 | 0.5749 | 0.6287 | 0.7970 | 215 | 159 | 127 | 908 |
| Logistic Regression (SMOTE) | comparison | 0.8439 | 0.6600 | 0.6306 | 0.8406 | 0.6336 | 0.6175 | 0.7941 | 0.5051 | 0.7388 | 297 | 77 | 291 | 744 |
| Random Forest (SMOTE) | comparison | 0.8423 | 0.6420 | 0.6127 | 0.8365 | 0.6361 | 0.5987 | 0.6123 | 0.5857 | 0.7821 | 229 | 145 | 162 | 873 |

### Best Hyperparameters — `XGBoost (Tuned)`
- `n_estimators`: `320`
- `max_depth`: `2`
- `learning_rate`: `0.03`
- `subsample`: `0.8`
- `colsample_bytree`: `0.75`
- `min_child_weight`: `3`
- `reg_lambda`: `5.0`
- `scale_pos_weight`: `2.7686`

### Key Modeling Takeaway
The unweighted Logistic Regression baseline achieved the highest raw accuracy (**79.99%**) but missed **47.86%** of churning customers (`179` false negatives). **XGBoost (Tuned)** reduced missed churners by **58%** (down to `75` false negatives, **79.95% recall**) while achieving the highest cross-validated ROC-AUC (**0.8500**) and PR-AUC (**0.6677**).

---

## 5. SHAP Explainability Results

Source: `reports/evaluation/shap_summary.json` (`TreeExplainer`, `1,409` held-out test rows, base log-odds = `0.0089`)

### Top 10 Global Churn Predictors (Mean `|SHAP|`)

| Rank | Feature | Mean `\|SHAP\|` |
|---:|---|---:|
| 1 | Contract: Month-to-month | 0.5297 |
| 2 | Tenure (months) | 0.4239 |
| 3 | InternetService: Fiber optic | 0.2541 |
| 4 | OnlineSecurity: No | 0.2505 |
| 5 | Contract: Two year | 0.1938 |
| 6 | TechSupport: No | 0.1648 |
| 7 | PaymentMethod: Electronic check | 0.1614 |
| 8 | Avg Monthly Spend ($) | 0.1352 |
| 9 | Monthly Charges ($) | 0.1151 |
| 10 | Total Charges ($) | 0.0892 |

### Sample Customer-Level SHAP Explanation (Held-Out Test Set)
- **Customer Profile**: Month-to-month contract, tenure = `1` month, Fiber optic internet, no Online Security/Tech Support, Electronic check, MonthlyCharges = `$95.10`, Senior Citizen = `1`
- **Predicted Churn Probability**: **95.94%** (High Risk; Actual label: `Churned = 1`)
- **Top Risk-Increasing SHAP Contributions (log-odds)**:
  1. `Tenure (months) = 1`: `+0.8418`
  2. `Contract: Month-to-month`: `+0.3810`
  3. `InternetService: Fiber optic`: `+0.2276`
  4. `OnlineSecurity: No`: `+0.2040`
  5. `Avg Monthly Spend ($95.10)`: `+0.1893`

---

## 6. Automated Testing & Verification Results

Command: `pytest -v` → **22 passed, 0 failed** (5.03s on Python 3.13.5)

| Test Suite | Tests | Status | Scope Verified |
|---|---:|---|---|
| `tests/test_data_processing.py` | 5 | PASSED | Blank `TotalCharges` handling, duplicate removal, schema validation, zero-null clean CSV, feature engineering & column-order invariance |
| `tests/test_pipeline.py` | 4 | PASSED | Risk bucketing & threshold validation, deterministic Joblib predictions with shuffled feature keys, risk monotonicity, SHAP driver structure |
| `tests/test_api.py` | 7 | PASSED | `/health`, `/model-info`, `/predict`, `/predict/explain`, `/predict/batch`, HTTP 422 validation errors, HTTP 400 invalid threshold rejection |
| `tests/test_rag.py` | 1 | PASSED | Grounded chunk indexing & evidence-separated retrieval |
| `tests/test_streamlit_smoke.py` | 5 | PASSED | All 5 Streamlit entrypoint/page scripts render without exceptions via `AppTest` |

- **Jupyter Notebooks Executed In-Place (`nbconvert --execute`)**:
  - `notebooks/01_data_exploration.ipynb` (986 KB, executed cleanly)
  - `notebooks/02_feature_engineering.ipynb` (8.6 KB, executed cleanly)
  - `notebooks/03_model_training.ipynb` (668 KB, executed cleanly)
  - `notebooks/04_model_explainability.ipynb` (724 KB, executed cleanly)

---

## 7. Generated Artifacts Inventory

- **Serialized Models (`models/`)**: `churn_pipeline_v1.0.0.joblib`, `churn_pipeline_latest.joblib`, `model_metadata.json`, `shap_background.csv`
- **Evaluation Reports (`reports/evaluation/`)**: `eda_summary.json`, `evaluation_summary.json`, `model_comparison.csv`, `shap_summary.json`, `architecture_diagram.md`, `technical_report.md`
- **Figures (`reports/figures/`)**: 17 PNG plots (`01_churn_distribution.png` through `17_shap_local_high_risk_example.png`)
- **Serving & Packaging**: FastAPI (`api/main.py`), 4-page Streamlit app (`app/streamlit_app.py`), Modular RAG (`src/rag/assistant.py`), `Dockerfile`, `docker-compose.yml`
