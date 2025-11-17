# Notebook Enhancements for ml_finance_model_main.ipynb

Version: v8_3 (proposed)
Last updated: 2025-10-25

Purpose

- Provide a comprehensive, actionable upgrade plan for ml_finance_model_main.ipynb focused on:
    - Data Management: unified all_stocks dataframe from PostgreSQL or CSV, with validation and quality checks
    - Feature Engineering: financial ratios, margins, volatility, revenue CAGR, categorical encodings
    - ML Models: financial event classification, sector-optimized regression, quantile models, stacking ensembles
    - Analytics: mispricing scores, ranking, interactive visuals, and Excel reporting similar to
      reports/Stock_Prediction_Analysis_Report_20250806_131704.xlsx

This plan aligns with README.md and IMPROVEMENT_PLAN.md, while staying notebook-first and reusing the finance_ml package
functions to avoid duplication.

---

High-level Flow

1) Load + validate data → all_stocks
2) EDA overview for sanity checks
3) Feature engineering (baseline + sector enrichments)
4) Multi-class event classification (probabilities as meta-features)
5) Sector-optimized price target regression (with quantiles)
6) Optional stacking ensemble of sector models
7) Analytics: mispricing, ranking, plots, Excel export
8) Save artifacts and record MODEL_VERSION

Environment and Config

- Use environment variables when possible: DATA_DIR, DB_URL, MODEL_VERSION=v8_3, RANDOM_SEED, N_JOBS
- Import from finance_ml to centralize logic; keep only notebook orchestration and visualization glue.

---

1. Data Management — Build all_stocks
   Goals

- Robustly load from PostgreSQL (preferred) or CSV fallback
- Normalize columns to pythonic snake_case and types
- Validate schema, handle missingness, deduplicate, and apply filters

Suggested Cells and Steps
A. Configuration Cell

- Set RANDOM_SEED, N_JOBS, MODEL_VERSION, output directories, feature flags

B. Data Source Selection Cell

- Use finance_ml.notebook_utils.load_stock_data which internally calls:
    - finance_ml.data.load_from_db or load_from_csv
    - finance_ml.data.normalize_columns
    - Enforce limit for quick runs

C. Validation Cell

- Run early pipeline checks:
    - finance_ml.data.validate_schema (required columns)
    - finance_ml.data.check_missing_values (Ticker, sector, last_price, target fields)
    - finance_ml.data.validate_numeric_ranges (e.g., prices > 0, sensible caps)
    - finance_ml.data.detect_outliers_iqr (summary only)
- Produce a compact validation report using finance_ml.notebook_utils.validate_and_display_data

D. Preprocessing Cell

- finance_ml.data.preprocess to:
    - Coerce numerics with pd.to_numeric(errors='coerce')
    - Quote-heavy columns → normalized snake_case
    - Drop duplicates (Ticker+Region) keeping latest or Exchange priority
    - Minimal imputation for selected features if applicable

E. Output: all_stocks

- Ensure final unified dataframe named all_stocks for further steps
- Cache to outputs/processed/all_stocks.parquet for reproducibility

Notes

- Default filters: drop rows with missing ticker, sector, last_price, and key financials used in targets
- Enforce Region in {US, EU, APAC, ROTW}

---

2. EDA (Lightweight, automated)
   Goals

- Sanity check distributions, missingness, and correlations; sector/region slices

Suggested Cells

- Use finance_ml.eval.simple_eda(all_stocks, out_dir)
- Display: counts per Region and Sector, missingness heatmap summary, distributions for market_cap, ev, pe, ebitda,
  margins
- Basic correlation heatmap and multicollinearity flags (EV vs MarketCap vs Assets)

Outputs

- outputs/eda_summary.json
- Optional: static PNGs or inline plots

---

3. Feature Engineering
   Goals

- Create robust, leakage-safe features including ratios, margins, volatility, growth, and categorical encodings

Baseline Transforms (use finance_ml.features functions where available)

- engineer_basic_ratios:
    - ev_to_ebitda = EV / EBITDA
    - net_debt_to_ebitda = NetDebt / EBITDA
    - p_e = LastPrice / EPS (or use provided P/E)
    - p_b = LastPrice / BookValuePerShare (or provided P/B)
- engineer_margin_features:
    - gross_margin, ebitda_margin, operating_margin, net_margin
- engineer_volatility_features:
    - rolling std/ATR-like proxies using historical columns if present; else use cross-sectional volatility proxies
- engineer_revenue_cagr:
    - CAGR over available historical revenue columns with robust handling

Categorical Encodings

- One-hot: sector, region, size_class
- Optional target encoding with CV for industry (avoid leakage with KFold and out-of-fold means)

Outlier Handling and Scaling

- Winsorize or robust scale per Sector (e.g., RobustScaler on continuous features)
- Nulls: impute with sector-wise medians where safe

Interactions

- Region × Sector, SizeClass × Sector interactions

Feature Selection (optional)

- Boruta or SHAP-based pruning to remove noisy features

Output

- features_df, target columns
- Persist to outputs/features/features_v8_3.parquet

---

4. Multi-class Event Classification
   Goals

- Predict financial event classes: 0=Neutral, 1=Positive catalyst, 2=Negative catalyst
- Use information available at training time only

Label Construction

- finance_ml.models.create_event_labels(all_stocks, ...):
    - Based on changes in price_target, analyst_rating, and volatility spikes
    - Parameterize windows and thresholds in the Config Cell

Models

- LightGBM/XGBoost/CatBoost classifiers
- Handle class imbalance via class_weight or scale_pos_weight

Validation

- Grouped CV by ticker or sector with stratification to prevent leakage

Outputs

- Class probabilities per row (p0, p1, p2) to be used as meta-features
- Metrics: confusion matrix, ROC/PR curves; per-sector performance

---

5. Sector-optimized Regression with Quantiles
   Goals

- Predict price targets (e.g., price_target or price_target_median) per Sector
- Add classification probabilities as meta-features

Approach

- Train base regressor per Sector using finance_ml.models.train_and_evaluate_regression_by_sector
- Quantile regression per Sector using finance_ml.models.train_quantile_regression_by_sector for uncertainty bands (
  q=0.1, 0.5, 0.9)
- Benchmarks: ElasticNet or LinearSVR as a baseline

Validation

- Grouped CV by ticker; report MAE, RMSE, MAPE, R2 overall and per Sector

Outputs

- predictions with quantile bands
- model artifacts under outputs/models/v8_3/

---

6. Stacking Ensemble (Optional, recommended)
   Goals

- Blend sector models with a meta-learner trained on out-of-fold predictions

Method

- finance_ml.models.train_stacking_ensemble_by_sector:
    - Base: Gradient boosting models (LGBM/XGB/CatBoost)
    - Meta: Ridge/Lasso or small GBM
    - Use OOF to prevent leakage

Monitoring

- Use finance_ml.models.monitor_ensemble_training hooks for logs and early stopping summaries

---

7. Analytics and Reporting
   Goals

- Identify under/overvalued stocks and visualize results

Metrics

- Mispricing score = (predicted_target - last_price) / last_price
- Rank top-N undervalued/overvalued per Sector and Region

Functions

- finance_ml.eval.calculate_mispricing_score
- finance_ml.eval.rank_undervalued_stocks, rank_overvalued_stocks, rank_stocks_by_sector
- Visuals: create_sector_heatmap, create_region_sector_heatmap, create_interactive_prediction_plot
- Export: export_predictions_to_excel to produce a report similar to
  reports/Stock_Prediction_Analysis_Report_20250806_131704.xlsx

Outputs

- outputs/analytics/regression_predictions.csv
- outputs/analytics/undervalued_topN.csv, overvalued_topN.csv
- outputs/analytics/plots/*.html (interactive) and *.png (static)
- outputs/reports/Stock_Prediction_Analysis_Report_<timestamp>.xlsx

---

8. Notebook Cell Outline (Suggested)
0. Title + Overview
1. Imports + Config (seed, n_jobs, model_version=v8_3, paths)
2. Load Data (DB or CSV auto)
3. Validation Summary (schema, missingness, numeric ranges, outliers)
4. Preprocess + all_stocks creation
5. EDA quick checks
6. Feature Engineering (baseline + sector enrichments + encodings)
7. Event Labeling (create_event_labels)
8. Event Classifier Training + Evaluation (save probs)
9. Sector Regression Training + Evaluation (with probs)
10. Quantile Regression (per sector)
11. Stacking Ensemble (optional)
12. Analytics: Mispricing, Rankings
13. Visualizations (interactive and static)
14. Excel Report Export
15. Save Artifacts + Version Log

Each cell should:

- Use finance_ml functions where possible
- Log key steps and capture parameters (for reproducibility)
- Save intermediate artifacts to outputs/ with model_version in paths

---

9. Versioning and Reproducibility

- Bump MODEL_VERSION to v8_3 for this enhancement
- Record changes in IMPROVEMENT_PLAN.md and this file
- Save config and library versions used in outputs/metadata.json
- Seed everything (NumPy, sklearn, xgboost, lightgbm, catboost where applicable)

---

10. Performance and Stability Tips

- Start with a limited feature set and scale up
- Use N_JOBS to control parallelism; monitor memory for large models
- Prefer CPU-only TensorFlow if used; otherwise, scikit-learn + GBMs are primary
- Use parquet caching for all_stocks and features
- Guard against leakage: stratified group CV, time-based splits if snapshot dates exist

---

11. Testing Hooks (Optional)

- Add small sample-based tests for feature builders and labelers
- Keep DB-dependent tests optional via DB_URL env var
- Use tests/test_*.py structure already present in the repo

---

12. Mapping to README and IMPROVEMENT_PLAN

- README: Point the Notebook section to this enhancement plan; reference CLI parity in ml_finance_model_main.py
- IMPROVEMENT_PLAN: Mark notebook roadmap items complete once implemented; keep analytics/reporting aligned with Excel
  export

---

Appendix A: Minimal Code Snippets

Config and Load

```python
from finance_ml import (
    NotebookConfig, load_stock_data, validate_and_display_data, perform_and_display_eda,
    build_features_and_target, create_event_labels,
    train_and_evaluate_regression_by_sector, train_quantile_regression_by_sector,
    calculate_mispricing_score, rank_undervalued_stocks, export_predictions_to_excel,
    )
from pathlib import Path

cfg = NotebookConfig(model_version="v8_3", random_seed=42, n_jobs=-1, out_dir=Path("outputs"))
all_stocks = load_stock_data(source="auto", db_url=None, limit=5000)
validate_and_display_data(all_stocks)
perform_and_display_eda(all_stocks, cfg.out_dir)
```

Features and Labels

```python
features_df, target = build_features_and_target(all_stocks, target_col="price_target_median")
labels = create_event_labels(all_stocks, price_change_thr=0.1, vol_spike_thr=2.0)
```

Models and Analytics

```python
metrics = train_and_evaluate_regression_by_sector(features_df, cfg.out_dir)
q_models = train_quantile_regression_by_sector(features_df, quantiles=[0.1,0.5,0.9])
all_predictions = calculate_mispricing_score(features_df, pred_col="pred", price_col="last_price")
export_predictions_to_excel(all_predictions, cfg.out_dir / "reporting")
```

---

Appendix B: Data Columns and Normalization Notes

- Prefer lowercase snake_case names (normalize_columns)
- Key columns expected:
    - ticker, region, sector, industry
    - last_price, price_target, price_target_median
    - market_cap, ev, ebitda, revenue, net_debt, eps, book_value_per_share, r_and_d
    - Any time/snapshot columns if present for proper splits

---

Change Log (for v8_3)

- Added notebook-first enhancement plan covering data, features, models, and analytics
- Integrated classification probabilities into regression
- Added quantile regression and optional stacking
- Standardized outputs and Excel export guidance
