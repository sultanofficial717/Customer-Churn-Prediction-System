# Technical Report — End-to-End Customer Churn Prediction System

## 1. Executive Summary

This project delivers a production-style Customer Churn Prediction System built on the **IBM Telco Customer Churn** dataset (7,043 customer accounts, 21 attributes). The system predicts customer attrition probability, classifies accounts into configurable operational risk tiers (`Low`, `Medium`, `High`), explains individual and global predictions with **SHAP**, serves predictions through a validated **FastAPI** service, and provides an interactive multi-page **Streamlit** dashboard with a grounded **RAG analytics assistant**.

Using a leakage-free Scikit-learn pipeline (`TelcoFeatureEngineer` + `ColumnTransformer`) and 5-fold stratified cross-validation across Logistic Regression, Random Forest, and XGBoost (testing unweighted, class-weighted, inside-fold SMOTE, and hyperparameter-tuned variants), **XGBoost (Tuned)** achieved the strongest cross-validated composite performance (`CV ROC-AUC = 0.8500`, `CV PR-AUC = 0.6677`, `CV F1 = 0.6323`). On the held-out test set (1,409 customers), it achieved **ROC-AUC = 0.8464**, **PR-AUC = 0.6648**, **Recall = 79.95%** (identifying 299 of 374 actual churners), **Precision = 50.59%**, and **F1 = 0.6197**.

---

## 2. Data Quality & Exploratory Data Analysis (Measured Findings)

- **Dataset Size & Integrity**: 7,043 rows, 21 columns, **0 duplicate rows**, **0 duplicate `customerID`s**, and **0 IQR outliers** in `tenure`, `MonthlyCharges`, or `TotalCharges`.
- **Blank `TotalCharges` Handling**: 11 records had blank whitespace strings in `TotalCharges`. Inspection confirmed all 11 accounts had `tenure == 0` (brand-new customers not yet billed a full cycle). Applying the deterministic row-level rule (`tenure == 0 -> TotalCharges = 0.0`) resolved all 11 values with zero train/test leakage, while an inside-pipeline median `SimpleImputer` protects production inference against unexpected nulls.
- **Target Imbalance**: **1,869 churned (26.54%)** vs. **5,174 retained (73.46%)**, a **2.77:1** class imbalance.
- **Measured Billing Exposure**:
  - Total monthly charges across the dataset: **$456,116.60**
  - Monthly charges from churned accounts: **$139,130.85 (30.5% of total monthly billing)**
  - Mean monthly charge: **$74.44** (churned) vs. **$61.27** (retained)
  - Mean tenure: **17.98 months** (churned) vs. **37.57 months** (retained)

### Most Vulnerable Customer Segments (Observed Rates)

| Dimension | Highest-Churn Segment | Observed Churn Rate | Lowest-Churn Segment | Observed Churn Rate |
|---|---|---|---|---|
| **Contract** | Month-to-month (n=3,875) | **42.7%** | Two year (n=1,695) | **2.8%** |
| **Tenure Cohort** | 0–12 months (n=2,186) | **47.4%** | 49–72 months (n=2,239) | **9.5%** |
| **Payment Method** | Electronic check (n=2,365) | **45.3%** | Credit card auto-pay (n=1,522) | **15.2%** |
| **Internet Service** | Fiber optic (n=3,096) | **41.9%** | No internet (n=1,526) | **7.4%** |
| **Online Security** | No (n=3,498) | **41.8%** | Yes (n=2,019) | **14.6%** |
| **Tech Support** | No (n=3,473) | **41.6%** | Yes (n=2,044) | **15.2%** |
| **Senior Citizen** | Senior = 1 (n=1,142) | **41.7%** | Non-Senior = 0 (n=5,901) | **23.6%** |

*Distinction between measured findings and assumptions*: While churned customers account for $139,130.85/month in observed dataset billing, real financial savings from retention campaigns depend on unobserved business parameters (customer acquisition cost, gross margin, incentive cost, and true uplift rate), which must be measured via controlled A/B retention pilots.

---

## 3. Feature Engineering & Leakage Prevention

All feature engineering lives inside `TelcoFeatureEngineer` (a stateless row-wise Scikit-learn transformer) chained with a `ColumnTransformer`:
- **Engineered Features**: `avg_monthly_spend`, `charge_ratio`, `service_count`, `has_support_bundle`, `is_auto_pay`, and `tenure_group` (`0-12m`, `13-24m`, `25-48m`, `49m+`).
- **Encoding & Scaling**: `SimpleImputer(strategy="median")` + optional `StandardScaler` for numeric features; `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for categoricals (producing 54 dense transformed features).
- **Zero Leakage Guarantee**: The stratified 80/20 split (5,634 train / 1,409 test) occurs strictly before any fitting. When SMOTE is evaluated, it runs inside `imblearn.pipeline.Pipeline` strictly on training folds during 5-fold CV.

---

## 4. Model Comparison & Selection

| Model | Stage | CV ROC-AUC | CV PR-AUC | CV F1 | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost (Tuned)** | tuned | **0.8500** | **0.6677** | 0.6323 | **0.8464** | **0.6648** | 0.6197 | **0.7995** | 0.5059 | 0.7395 |
| Random Forest (Tuned) | tuned | 0.8477 | 0.6646 | 0.6297 | 0.8424 | 0.6503 | 0.6290 | 0.7888 | 0.5230 | 0.7530 |
| XGBoost (Weighted) | comparison | 0.8471 | 0.6618 | 0.6340 | 0.8460 | 0.6569 | 0.6378 | 0.7888 | 0.5354 | 0.7622 |
| Random Forest (Balanced) | comparison | 0.8462 | 0.6622 | 0.6337 | 0.8420 | 0.6559 | 0.6276 | 0.7727 | 0.5283 | 0.7566 |
| Logistic Regression (Tuned) | tuned | 0.8458 | 0.6617 | 0.6290 | 0.8414 | 0.6323 | 0.6123 | 0.7834 | 0.5026 | 0.7367 |
| Baseline Logistic Regression (Unweighted) | comparison | 0.8458 | 0.6630 | 0.5957 | 0.8426 | 0.6378 | 0.5804 | 0.5214 | 0.6544 | 0.7999 |
| Logistic Regression (Balanced) | comparison | 0.8456 | 0.6620 | 0.6251 | 0.8422 | 0.6363 | 0.6146 | 0.7888 | 0.5034 | 0.7374 |
| XGBoost (SMOTE) | comparison | 0.8448 | 0.6547 | 0.6039 | 0.8471 | 0.6644 | 0.6006 | 0.5749 | 0.6287 | 0.7970 |
| Logistic Regression (SMOTE) | comparison | 0.8439 | 0.6600 | 0.6306 | 0.8406 | 0.6336 | 0.6175 | 0.7941 | 0.5051 | 0.7388 |
| Random Forest (SMOTE) | comparison | 0.8423 | 0.6420 | 0.6127 | 0.8365 | 0.6361 | 0.5987 | 0.6123 | 0.5857 | 0.7821 |

### Why Accuracy Alone Fails & Error Trade-Offs
Notice that the unweighted baseline Logistic Regression achieves the highest raw accuracy (**79.99%**), yet misses nearly half of all churners (**52.14% recall**, 179 false negatives). By contrast, **XGBoost (Tuned)** with `scale_pos_weight = 2.7686` and shallow regularized trees (`max_depth=2`, `n_estimators=320`, `learning_rate=0.03`, `reg_lambda=5.0`) captures **79.95% of churners** (reducing false negatives from 179 down to **75**) while preserving the best overall ranking quality (`ROC-AUC = 0.8464`, `PR-AUC = 0.6648`).
- **False Negatives (75 test customers)**: Missed churners lead to preventable recurring revenue loss.
- **False Positives (292 test customers)**: Retained customers flagged as at-risk incur outreach or offer cost.
- Operational teams can adjust the decision threshold directly in the Streamlit UI or FastAPI query parameters to align outreach volume with retention capacity.

---

## 5. SHAP Explainability Findings

Using `shap.TreeExplainer` over all 1,409 held-out test records, the top global predictors by mean `|SHAP|` impact on log-odds are:

1. `Contract: Month-to-month` (`0.5297`)
2. `Tenure (months)` (`0.4239`)
3. `InternetService: Fiber optic` (`0.2541`)
4. `OnlineSecurity: No` (`0.2505`)
5. `Contract: Two year` (`0.1938`)
6. `TechSupport: No` (`0.1648`)
7. `PaymentMethod: Electronic check` (`0.1614`)
8. `Avg Monthly Spend ($)` (`0.1352`)

*Causality Limitation*: SHAP quantifies each feature's contribution to the model's statistical prediction, not causal proof that modifying a single attribute will prevent churn.

---

## 6. Limitations & Future Improvements

1. **Snapshot Dataset (No Temporal History)**: The IBM Telco dataset is a single cross-sectional snapshot without usage trends, support ticket logs, outage incidents, or NPS scores. Adding time-series behavioral features would likely improve PR-AUC substantially.
2. **Probability Calibration**: Using `scale_pos_weight` boosts recall and ranking (`ROC-AUC`) but shifts raw predicted probabilities upward relative to the true 26.5% base rate. Adding post-hoc isotonic or Platt calibration (`CalibratedClassifierCV`) is recommended if exact expected-value financial modeling is needed.
3. **Uplift Modeling**: Predicting churn risk identifies *who might leave*, not *who is persuadable by a discount*. Next-generation retention systems combine churn scoring with causal uplift modeling.
4. **Monitoring & Drift Detection**: In live production, add scheduled PSI/KS feature-drift checks and periodic retraining triggers.

---

## 7. Truthful CV / Resume Project Summary

> **End-to-End Customer Churn Prediction System** *(Python, Scikit-learn, XGBoost, SHAP, FastAPI, Streamlit, Docker, Pytest)*
> - Built a modular, leakage-free churn prediction pipeline on the IBM Telco dataset (7,043 accounts), benchmarking Logistic Regression, Random Forest, and XGBoost across class weighting, inside-fold SMOTE, and RandomizedSearchCV tuning.
> - Selected tuned XGBoost via 5-fold stratified CV (`ROC-AUC = 0.8464`, `PR-AUC = 0.6648`, `Recall = 79.95%` on held-out test set), improving churn capture over an unweighted baseline (`52.14%` recall) while preserving top ranking performance.
> - Integrated `SHAP TreeExplainer` for global feature diagnostics and real-time customer-level risk attribution, decoupling inference (`src/predict.py`) to serve both a Pydantic-validated FastAPI microservice (Dockerized) and a 4-page Streamlit analytics & RAG assistant dashboard backed by 22 automated pytest tests.
