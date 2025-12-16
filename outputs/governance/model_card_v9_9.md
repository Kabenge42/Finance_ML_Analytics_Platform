# Model Card — v9_9

**Generated:** 2025-12-14 01:04:57

## Model Overview

- **Task:** Price target regression + classification-enhanced features
- **Model Version:** v9_9
- **Model Type:** Stacking Ensemble (Gradient Boosting + Linear Meta-Learner)

## Data

- **Source:** PostgreSQL equities table / CSV files
- **Time Range:** Current snapshot
- **Snapshot Policy:** Single or time-series snapshots with proper date handling
- **Data Split:** Grouped CV by ticker, stratified by sector

## Features

- **Feature Groups:** momentum, valuation, profitability, quality, risk
- **Total Features:** N/A
- **Feature Selection:** Boruta + SHAP-based pruning (if available)
- **Safety Rails:** Winsorization (5-95th percentile), non-negativity constraints, robust loss functions

## Models

- **Base Learners:** XGBoost, LightGBM, CatBoost
- **Meta-Learner:** Linear Ridge Regression
- **Hyperparameters:** Cross-validated with GridSearchCV or Optuna
- **Stacking Strategy:** Out-of-fold base predictions as meta-features

## Validation & Metrics

- **Validation Strategy:** Grouped CV by ticker, stratified by sector
- **Overall Metrics:**
    - MAE: N/A
    - RMSE: N/A
    - MAPE: N/A%
    - R²: N/A
- **Uncertainty Coverage (80% interval):** N/A

## Fairness & Bias

- **Sector-Level Calibration:** Applied per-sector bias correction
- **Regional Balance:** Stratified sampling ensures representation across US, EU, APAC, ROTW
- **Monitoring:** Continuous tracking of sector-level performance drift

## Risk & Limitations

- **Non-Negativity:** All predictions enforced to be ≥ 0 (price targets cannot be negative)
- **Data Drift:** Model performance may degrade if market conditions change significantly
- **Missingness:** Imputation strategy (6-step) may introduce bias in sparse data
- **Outliers:** Winsorization caps extreme values but may underestimate tail risk
- **Leakage Prevention:** Grouped CV ensures no ticker appears in both train and validation

## Versioning & Reproducibility

- **Code Version:** Git SHA (if tracked)
- **Data Version:** Snapshot date-based
- **Dependencies:** See requirements.txt (Python 3.12+, scikit-learn, xgboost, lightgbm, catboost)
- **Random Seed:** 42

## Governance & Compliance

- **Model Owner:** ML Team / Data Science
- **Review Date:** 2025-12-14
- **Approval Status:** Development
- **Monitoring Plan:** Weekly performance tracking, monthly retraining cadence

## Artifacts & Documentation

- **Predictions Schema:** `outputs/regression/regression_predictions_detailed.csv`
- **Uncertainty Diagnostics:** `outputs/uncertainty/`
- **Safety Rails Reports:** `outputs/safety_rails/`
- **Calibration Metrics:** `outputs/calibration/sector_bias_calibration_v9_9.json`
- **Lineage:** `lineage.json`

## References

- Code Guidelines: `docs/code_guidelines.md` v1.2+
- Notebook: `ml_finance_model_main.ipynb`
- Improvement Plan: `docs/improvement_plan/`
