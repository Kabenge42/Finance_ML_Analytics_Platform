Finance ML Analytics Platform — Development Guidelines

TL;DR (Quick Start)

- Python: 3.12, 3.13, or 3.14 (officially supported per pyproject.toml; 3.10-3.11 may work but are not tested).
- Create venv: python -m venv .venv && .venv\Scripts\activate
- Upgrade pip: python -m pip install --upgrade pip setuptools wheel
- Install deps: pip install -r requirements.txt
- PostgreSQL: install and start local Postgres; run create_equities_schema.sql to create the equities table.
- Data: Import the CSVs from data/ into the equities table, tagging each by Region.
- Notebook: open ml_finance_model_main.ipynb and run cells in order.
- Tests: python -m unittest -v

1) Build and Configuration Instructions
1) Build and Configuration Instructions
A. Prerequisites
- OS: Windows 10/11 (tested), macOS, or Linux.
- Python: 3.12, 3.13, or 3.14 (officially supported per pyproject.toml requires-python = ">=3.12,<3.15"). Use pyenv or
  the
  official installer. Note: Python 3.10-3.11 may work but are not officially tested. Avoid mixing Conda with venv in the
  same project.
- Git: optional but recommended for version control.
- PostgreSQL: 15+ recommended. Local DB will be used for data loading. JDBC URL in this project: jdbc:postgresql://localhost:5432/postgres
- JDK (optional): Only if you use JVM tools or JDBC from Java-based tooling.
- Jupyter: installed with pip (notebook or jupyterlab) if you’ll run the .ipynb.

B. Virtual environment setup
- Create and activate a virtual environment:
  - Windows (PowerShell):
    - python -m venv .venv
    - .venv\Scripts\Activate.ps1
  - macOS/Linux (bash):
    - python3 -m venv .venv
    - source .venv/bin/activate
- Upgrade packaging tools:
  - python -m pip install --upgrade pip setuptools wheel

C. Install Python dependencies
- Install all packages:
  - pip install -r requirements.txt
- Notes:
  - TensorFlow is heavy; CPU-only install is fine for this project. If you have an NVIDIA GPU and want acceleration, follow the official TensorFlow GPU install docs and ensure CUDA/cuDNN compatibility. If installation is problematic, you can temporarily comment out tensorflow in requirements.txt and proceed; the core workflow primarily uses scikit-learn and gradient boosting libraries.
  - Database access from Python: psycopg2-binary and SQLAlchemy are now included in requirements.txt for PostgreSQL
    connectivity. Most data movement can be done via psql or GUI tools, but these libraries enable direct database
    access from Python scripts and notebooks.

D. Environment variables
- See environment_variables.txt. You can export them in your shell or create a .env for tools that auto-load it.
- Defaults in repo:
  - TF_CPP_MIN_LOG_LEVEL=2 (reduce TensorFlow log verbosity)
- Optional environment variables (examples):
  - DATA_DIR, MODEL_DIR, CACHE_DIR, MODEL_VERSION, RANDOM_SEED, N_JOBS, MEMORY_LIMIT
  - DB_URL (SQLAlchemy connection URL for database access, e.g., postgresql+psycopg2://postgres:@localhost:5432/postgres)

E. PostgreSQL setup
- Install PostgreSQL (Windows installer from postgresql.org). Ensure psql is on PATH.
- Start PostgreSQL service and verify access with user postgres (adjust user/password if needed).
- Schema creation:
  - Run the provided SQL script to create the equities table:
    - Windows (PowerShell):
      - psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql
    - You will be prompted for the postgres password if required.
- Connection details:
  - JDBC: jdbc:postgresql://localhost:5432/postgres (user: postgres)
  - Driver: org.postgresql.Driver (42.7.x)
  - DataGrip/IDEA DataSource template (for convenience):
    - <data-source source="LOCAL" name="postgres@localhost" ... jdbc-url="jdbc:postgresql://localhost:5432/postgres" user-name="postgres" jdbc-driver="org.postgresql.Driver"/>

F. Loading data from CSVs into PostgreSQL
- The data CSV files live in data/ and represent screenings by region: screening_us.csv, screening_eu.csv, screening_apac.csv, screening_rotw.csv.
- Recommended approach: Use the comprehensive import script that handles all regions with proper NULL handling:
  - psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
  - This script uses staging tables, proper NULL handling (NULL '', ENCODING 'UTF8'), and provides validation at each step.
- Optional but recommended: Validate CSV data quality before import:
  - python validate_csv_import.py
  - This validates schema, checks for missing values, and identifies data quality issues.
- Alternative manual approach using psql and a staging table (if you need more control):
  1) Create a temporary staging table with identical columns to equities. The main equities table already includes a "Region" column.
  2) For each CSV, import with \copy (client-side) so paths work reliably on Windows:
     - Example for US region with proper NULL handling:
       - psql -h localhost -p 5432 -U postgres -d postgres -c "CREATE TEMP TABLE equities_staging (LIKE equities)"
       - psql -h localhost -p 5432 -U postgres -d postgres -c "\copy equities_staging FROM 'data/screening_us.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')"
       - psql -h localhost -p 5432 -U postgres -d postgres -c "UPDATE equities_staging SET \"Region\"='US' WHERE \"Region\" IS NULL"
       - psql -h localhost -p 5432 -U postgres -d postgres -c "INSERT INTO equities SELECT * FROM equities_staging ON CONFLICT DO NOTHING"
- Critical import parameters:
  - NULL '' - Treats empty strings as NULL values (essential to avoid import errors)
  - ENCODING 'UTF8' - Ensures proper character handling
  - HEADER true - Skips the CSV header row
- Tip: Because many column names use spaces and punctuation, always quote identifiers in SQL as shown in create_equities_schema.sql.

G. Accessing the database from Python (optional)
- Install optional deps: pip install psycopg2-binary SQLAlchemy
- Code snippet to query into pandas:
  - from sqlalchemy import create_engine
  - import pandas as pd
  - engine = create_engine('postgresql+psycopg2://postgres:@localhost:5432/postgres')  # add password if needed
  - query = 'SELECT * FROM equities WHERE "Region" IN (\'US\', \'EU\', \'APAC\', \'ROTW\')'
  - all_stocks = pd.read_sql(query, engine)
  - # Normalize column names for Pythonic usage
  - all_stocks.columns = (
  -     all_stocks.columns
  -     .str.replace('[^0-9a-zA-Z]+', '_', regex=True)
  -     .str.strip('_')
  -     .str.lower()
  - )

2) Data Pipeline aligned to the new all_stocks dataframe
   The notebook (ml_finance_model_main.ipynb) implements the pipeline below. Use all_stocks as the single, unified
   dataframe sourced from PostgreSQL across regions.

   Notebook Best Practices (docs/code_guidelines.md §8, v1.4):

- Centralized Configuration Constants: define all constants once (TARGET_COL, TARGET_COL_FALLBACK, TEST_SIZE,
  TRAIN_SIZE, CV_FOLDS, QUANTILES, MIN_SECTOR_SAMPLES, MAX_SECTOR_WEIGHT, MAX_SINGLE_POSITION, IQR_MULTIPLIER,
  ZSCORE_THRESHOLD, WINSORIZE_LOWER, WINSORIZE_UPPER, CONFIDENCE thresholds, RANDOM_SEED).
- DataFrame Stage Naming (no in-place mutation):
  all_stocks_raw → all_stocks_normalized → all_stocks_typed → all_stocks_winsorized → all_stocks_imputed →
  all_stocks_scaled → all_stocks_features → all_stocks_enhanced.
- Magic Numbers Policy: avoid embedding meaningful numeric literals in code; use named constants for splits,
  quantiles, winsorization bounds, sector/portfolio constraints, thresholds.

A. Loading and preprocessing
- Source: equities table in PostgreSQL populated from the four CSVs.
- Recommended filters: drop rows with missing Ticker, Sector, Last_Price, and critical financials needed for targets.
- Type coercion: convert numeric columns with pd.to_numeric(errors='coerce'), then handle NaNs.
- Deduplication: drop_duplicates on Ticker if needed; for multiple listings, choose Trading_Country or Exchange priority.
- Train/validation split: Follow Data Split and Leakage Policy (code_guidelines.md v1.4): time-series split if snapshot
  dates available; otherwise grouped by Ticker; else stratified by Sector or Region to maintain balance and prevent
  leakage.

B. Exploratory Data Analysis (EDA)
- Global overview: row counts per Region and Sector; missingness matrix; distributions for key metrics (Market Cap, EV, P/E, EBITDA, margins).
- Correlations: compute Pearson/Spearman; watch for multicollinearity (e.g., EV, Market Cap, Total Assets).
- Sector-level slices: summary statistics and outlier detection per Sector, Region.
- Target relationship: analyze distribution of Price_Target versus Last_Price; consider Price_Target_YTD_Ago for drift.

C. Advanced feature engineering with sector-specific optimizations
- Baseline transforms:
  - Ratios: EV/EBITDA, Net_Debt/EBITDA, P_E, P_B, Margin features, Revenue CAGR, Volatility windows.
  - Winsorize or robust scaling per Sector to mitigate outliers.
  - One-hot encode Sector, Industry; optionally target-encode Industry with CV to avoid leakage.
- Sector-specific enrichments:
  - Financials: TBV-related metrics, P_TBV, ROE.
  - Energy/Materials: Asset turnover, CAPEX intensity, EBITDA margin stability.
  - Tech/Healthcare: R&D intensity, growth proxies, volatility profile.
- Interactions: Region x Sector and Size_Class x Sector interactions.
- Feature selection: Boruta/Shap-based pruning to reduce noise.

D. Multi-class classification of financial events
- Define classes from events using available columns, for example:
  - 0: Neutral, 1: Positive catalyst, 2: Negative catalyst (construct from changes in Price_Target, Analyst_Rating, and Volatility spikes). Ensure label creation only uses information available at training time.
- Models: LightGBM/XGBoost/CatBoost classifiers with class_weight balancing or scale_pos_weight.
- Validation: Grouped CV by Sector or Ticker to prevent leakage across folds.
- Outputs: Use predicted class probabilities as meta-features for the regression stage.

E. Sector-optimized regression models enhanced with classification features
- Strategy: Train one regressor per Sector for Price_Target or Price_Target_Median using Gradient Boosting (XGBoost/LightGBM/CatBoost) and/or linear models with elastic nets as benchmarks.
- Features: Core engineered features + classification probabilities + key categorical encodings.
- Uncertainty Quantification: Use quantile regression (p10, p50, p90) + conformal calibration for 80% prediction
  intervals (code_guidelines.md v1.4). Enforce monotonicity and non-negativity constraints.
- Stacking: Optionally stack sector models with a meta-learner trained on out-of-fold predictions.

F. Model evaluation and error analysis
- Metrics: MAE, RMSE, MAPE, R2; compute per Sector and overall.
- Diagnostics: Residual plots per Sector, error histograms, performance by Market_Cap buckets.
- Ablation: Feature-importance-based ablations and SHAP analysis to validate drivers.

G. Identification of under/overvalued stocks with visualization
- Mispricing score: (Predicted_Target - Last_Price) / Last_Price.
- Ranking: Top-N undervalued/overvalued per Sector and Region.
- Visuals: bar charts, scatter plots of Predicted_Target vs Last_Price, and sector heatmaps; use matplotlib/seaborn/plotly.

H. Comprehensive analytics of prediction results
- Confusion matrices for event classifier; PR/ROC curves.
- Regression error buckets: by Sector, Region, and Volatility.
- Stability checks across snapshots (if multiple dates exist).
- Export: write CSV/Excel of predictions using Standardized Predictions Schema (code_guidelines.md v1.4). Required
  columns: ticker, isin, sector, region, last_price, y_true, y_pred, y_pred_calibrated, pred_p10, pred_p50, pred_p90,
  interval_width, abs_error, pct_error, model_version, snapshot_date. Consider xlsxwriter for formatted reports.

2.5) Running as a Python script
In addition to the notebook-first workflow, you can run a lightweight script version of the pipeline.

A. Script: ml_finance_model_main.py
- Python script with CLI for batch processing:
  - `--data-source {auto|csv|db}` — Data source selection (default: auto)
  - `--db-url <url>` — Database connection string (or use DB_URL env var)
  - `--limit <n>` — Limit rows for testing
  - `--out-dir <path>` — Output directory (default: outputs)
  - `--dry-run` — Skip model training

B. Usage examples
- Windows (PowerShell):
  - python ml_finance_model_main.py --data-source auto --limit 5000 --out-dir outputs
- macOS/Linux (bash):
  - python ml_finance_model_main.py --data-source auto --limit 5000 --out-dir outputs

C. Data source selection
- --data-source auto (default) tries DB first if DB_URL (or --db-url) is provided and SQLAlchemy is available, otherwise falls back to CSVs in data/.
- --data-source db forces DB; provide --db-url or set DB_URL env var (e.g., postgresql+psycopg2://postgres:@localhost:5432/postgres).
- --data-source csv forces CSV fallback and combines the four region files.

D. Outputs

- Artifacts are written to the outputs/ directory following standardized schema (code_guidelines.md v1.4):
  - outputs/regression/regression_predictions_detailed.csv — Standardized predictions schema with required columns
  - outputs/regression/regression_metrics_by_sector.csv — Per-sector MAE, RMSE, R², MAPE, count
  - outputs/regression/quantile_predictions.csv — Uncertainty intervals with monotonicity guarantees
  - outputs/eda/eda_summary.json — EDA summary statistics
- Environment variables recognized: DATA_DIR, MODEL_DIR, CACHE_DIR, MODEL_VERSION, RANDOM_SEED, N_JOBS, DB_URL.

3) Testing Information
A. How to run tests
- We use Python’s built-in unittest to avoid extra dependencies.
- Run all tests from project root:
  - python -m unittest -v

B. How to add tests
- Create files under tests/ named test_*.py with unittest.TestCase classes.
- Keep tests isolated from external services:
  - Prefer testing pure functions and small utilities.
  - For DB-related code, stub or mock the connection; keep integration tests optional and locally configured via env vars.
- Use small, deterministic samples; avoid loading full CSVs unless necessary.

C. Test suite overview
The project includes a comprehensive test suite with the following test modules (85 total):

- tests/test_analytics.py — Analytics and stock ranking tests
- tests/test_build_features.py — Feature building pipeline
- tests/test_classification.py — Event classification model tests
- tests/test_cli.py — Command-line interface tests
- tests/test_coverage_smoke.py — Smoke test for coverage validation
- tests/test_data_quality.py — Data validation and quality checks
- tests/test_data_splits_policy.py — Data split leakage prevention policy validation (code_guidelines.md v1.4)
- tests/test_data_types_detection.py — Schema-aware datatype detection and Phase 9.3 validation (9 tests, TDD v0.8.2)
- tests/test_eda.py — Exploratory data analysis utilities
- tests/test_enhanced_imputation.py — Phase 9.1 6-step imputation strategy tests (21 tests)
- tests/test_enhanced_imputation_phase93.py — Phase 9.3 enhanced imputation with schema alignment (8 tests, TDD v0.8.2)
- tests/test_features.py — Feature engineering functions
- tests/test_finance_ml_config.py — Configuration management tests
- tests/test_finance_ml_data.py — Data loading module tests
- tests/test_finance_ml_eval.py — Evaluation and analytics module tests
- tests/test_finance_ml_features.py — Features module tests
- tests/test_finance_ml_models.py — Models module tests
- tests/test_improvement_plan_revision.py — Development plan validation
- tests/test_integration_cli_pipeline.py — CLI pipeline integration tests
- tests/test_integration_notebook_pipeline.py — Notebook pipeline integration tests
- tests/test_integration_production_scenarios.py — Production scenario integration tests
- tests/test_loaders.py — CSV and database loading functions
- tests/test_logging.py — Logging configuration tests
- tests/test_metadata_catalog_quality.py — Metadata and quality stats validation (4 tests, TDD v0.8.2)
- tests/test_notebook_config.py — Notebook configuration tests
- tests/test_notebook_enhancements.py — Notebook enhancements validation
- tests/test_outlier_safety_rails.py — Outlier safety rails (winsorization, clipping, non-negativity) (
  code_guidelines.md v1.4)
- tests/test_phase95_nonnegative_predictions.py — Phase 9.5 non-negative prediction constraint tests
- tests/test_phase95_quick.py — Phase 9.5 quick validation tests
- tests/test_portfolio_backtesting.py — Portfolio backtesting framework (3 tests, Portfolio Phase 5)
- tests/test_portfolio_dashboards.py — Portfolio interactive dashboards (3 tests, Portfolio Phase 6)
- tests/test_portfolio_ml_prediction.py — ML-based return prediction and stock selection (9 tests, Portfolio Phases 1-2)
- tests/test_portfolio_optimization.py — Portfolio optimization tests
- tests/test_portfolio_optimization_advanced.py — Advanced optimization methods (4 tests, Portfolio Phase 3)
- tests/test_portfolio_risk_management.py — Risk management enhancements (4 tests, Portfolio Phase 4)
- tests/test_predictions_schema.py — Standardized predictions schema validation (code_guidelines.md v1.4)
- tests/test_preprocess_and_training.py — Preprocessing and training workflows
- tests/test_regression.py — Regression model evaluation
- tests/test_regression_sector_metrics.py — Sector-level metrics persistence validation (code_guidelines.md v1.4)
- tests/test_repository_setup.py — Validates repository basics (required files, SQL schema, environment config)
- tests/test_risk_metrics.py — Risk metrics calculation tests
- tests/test_sector_bias_calibration.py — Sector-specific bias calibration (code_guidelines.md v1.4)
- tests/test_setup_environment.py — Setup script validation
- tests/test_simple_eda_stringdtype.py — StringDtype compatibility validation (3 tests, TDD v0.8.2)
- tests/test_sqlite_import.py — SQLite import functionality (header removal, NULL handling, region backfilling)
- tests/test_sql_scripts.py — SQL script validation tests
- tests/test_stacking_default.py — Stacking ensemble default configuration (code_guidelines.md v1.4)
- tests/test_uncertainty_calibration.py — Uncertainty quantification with conformal prediction (code_guidelines.md v1.4)
- tests/test_validate_csv_import.py — CSV validation (schema validation, data quality checks)
- tests/test_validation_regex.py — Regex validation and pattern matching tests
- tests/test_visualizations.py — Visualization functions tests

Note: The test suite has grown to 85 modules (74 original + 4 TDD v0.8.2 + 5 Portfolio Optimization + 2 advanced
evaluation reporting).
See section 3D below for selective execution strategies.

**Recent Additions (v0.8.2, 2025-11-19):**

- **TDD Implementation (4 modules, 24 tests):** Schema-aware datatype detection, Phase 9.3 enhanced imputation,
  metadata catalog validation, StringDtype compatibility
- **Portfolio Optimization (5 modules, 23 tests):** ML-based return prediction, advanced optimization methods
  (Black-Litterman, Risk Parity, HRP), risk management enhancements, backtesting framework, interactive dashboards

New TDD modules aligned with code_guidelines.md v1.3+ standards for schema and datatype management, uncertainty
quantification, outlier safety rails, standardized predictions schema, sector metrics, data split policies, and
stacking defaults.

D. Test Execution Strategies (Avoiding Timeouts)

The full test suite (85 modules) can take significant time to execute. To avoid timeouts and speed up development, use
selective test execution:

**Fast Unit Tests** (< 100 lines, pure functions, no ML training):

- test_coverage_smoke.py — Minimal smoke test
- test_loaders.py — Data loading utilities
- test_edge_cases_*.py — Edge case validation
- test_validation_regex.py — Regex validation
- test_repository_setup.py — File existence checks
- test_improvement_plan_revision.py — Documentation validation

Run fast tests only:
python -m unittest tests.test_coverage_smoke tests.test_loaders tests.test_validation_regex tests.test_repository_setup
-v

**Medium Tests** (100-500 lines, integration, limited ML):

- test_enhanced_imputation.py — 6-step imputation (21 tests, ~2-5s)
- test_data_catalog.py — Data catalog functionality
- test_data_versioning.py — Version tracking
- test_logging.py — Logging configuration
- test_risk_metrics.py — Risk calculations
- test_portfolio_optimization.py — Portfolio optimization
- test_benchmarking.py — Sector/region benchmarking
- test_analyst_comparison.py — Prediction vs analyst analytics
- test_cli.py — Command-line interface

Run medium tests:
python -m unittest tests.test_enhanced_imputation tests.test_data_catalog tests.test_logging tests.test_risk_metrics -v

**Slow Tests** (> 500 lines, heavy ML model training, large datasets):

- test_finance_ml_eval.py — Comprehensive evaluation (1365 lines)
- test_classification_phase94.py — Classification models (1324 lines)
- test_advanced_features.py — Feature engineering (907 lines)
- test_sqlite_import.py — Database import (656 lines)
- test_analytics.py — Analytics pipeline (604 lines)
- test_advanced_models_phase95.py — Regression models (598 lines)
- test_ml_stock_prediction_notebook.py — Notebook execution (571 lines)

Run slow tests (use sparingly):
python -m unittest tests.test_classification_phase94 -v
python -m unittest tests.test_advanced_models_phase95 -v

**Recommended Workflow**:

1. During development: Run only affected module tests
  - python -m unittest tests.test_<your_module> -v
2. Before commit: Run fast + medium tests (~1-3 minutes)
  - python -m unittest discover -s tests -p "test_coverage_*.py" -v
  - python -m unittest discover -s tests -p "test_enhanced_*.py" -v
3. CI/CD: Run full suite with timeout protection (split into parallel jobs if needed)

**Test by Feature Area**:

- Data/Loading: test_finance_ml_data, test_loaders, test_sqlite_import, test_validate_csv_import
- Preprocessing: test_advanced_preprocessing, test_enhanced_imputation, test_data_quality
- Features: test_features, test_advanced_features, test_finance_ml_features
- Models: test_classification*, test_advanced_models*, test_finance_ml_models, test_regression
- Evaluation: test_finance_ml_eval, test_analytics, test_evaluation_phase96, test_valuation_phase97
- Integration: test_integration_*, test_notebook_*

3.5) Recent Enhancements (v0.8.2, 2025-11-19)

A. TDD Implementation: Data Preprocessing & Datatype Detection

**Overview:**
Implemented comprehensive data preprocessing and datatype detection features following strict Test-Driven Development (
TDD)
principles as documented in `docs/TDD_IMPLEMENTATION_SUMMARY.md`.

**New Modules:**

- **Schema Module** (`finance_ml/ml_workflow/data/schema.py`, 530 lines):
  - Centralized column schema registry derived from `create_equities_schema.sql`
  - `COLUMN_SCHEMA`: Dict mapping 350+ normalized column names to dtype and role
  - `PHASE93_FEATURE_INPUTS`: Categorization of Phase 9.3 feature engineering buckets (momentum, valuation,
    profitability, quality/risk, cash flow, growth)
  - Helper functions: `get_expected_dtype()`, `get_column_role()`, `list_numeric_feature_cols()`,
    `list_categorical_cols()`, `list_date_cols()`, `normalize_column_name()`

- **Datatype Detection Module** (`finance_ml/ml_workflow/preprocessing/dtypes.py`, 326 lines):
  - Schema-aware datatype detection, validation, and casting
  - `detect_and_cast_dtypes()`: Main function for schema-driven type casting with diagnostics
  - `_cast_to_numeric()`, `_cast_to_datetime()`: Type-specific casting with coercion tracking
  - `_infer_and_cast_unknown_column()`: Heuristic-based type inference for unknown columns
  - `validate_dtypes_against_schema()`: Post-casting validation
  - `get_dtype_summary()`: Comprehensive dtype and missing value summary

**Test Coverage:**

- 24 tests total (23 passing, 1 skipped)
- `test_data_types_detection.py` (9 tests): Schema-aware casting, coercion tracking, Phase 9.3 validation
- `test_enhanced_imputation_phase93.py` (8 tests): Sector-aware KNN, categorical/datetime strategies
- `test_metadata_catalog_quality.py` (4 tests): Metadata validation and quality stats
- `test_simple_eda_stringdtype.py` (3 tests): StringDtype compatibility validation

**Key Features:**

- Schema-driven datatype casting with comprehensive diagnostics
- Phase 9.3 feature categorization for ML pipeline
- Enhanced imputation with schema consistency checks
- Metadata catalog quality validation
- StringDtype compatibility fixes in EDA functions

B. Phase 9.3 Feature Enhancement Plan

**Overview:**
Comprehensive feature engineering enhancements documented in
`docs/improvement_plan/Phase_9.3_feature_enhancement_plan.md`
(Version 1.1, Status: ACTIVE).

**Schema Version 1.3 Expansion:**

- **310 columns total** (expanded from 262, +48 new columns)
- Technical indicators: EMAs (20D, 50D, 100D, 250D), 52W High/Low, Relative Volume
- Valuation multiples time-series: EV/Sales (11 cols), EV/EBITDA (6 cols), P/E extended (11 cols)
- Revenue forecasting estimates (4 cols)
- Dividend record information (8 cols)
- Employment metrics (2 cols)

**New Feature Categories (Schema 1.3):**

1. **Technical Analysis Integration Features**: EMA crossovers, 52W position, volume momentum
2. **Valuation Multiples Time-Series Features**: Momentum, mean reversion, forward vs trailing
3. **Revenue Forecasting & Analyst Consensus Features**: Estimate spreads, growth acceleration
4. **Dividend Reliability & Income Stock Features**: Consistency, coverage, growth
5. **Employment Dynamics & Growth Signals**: Productivity, workforce volatility

**Original Feature Categories (Version 1.0):**

1. Momentum & Technical Features
2. Quality & Risk Signals
3. Cash Flow & Capital Allocation Features
4. Market Sentiment & Analyst Features
5. Profitability Trends & Margins
6. Balance Sheet Strength & Temporal Patterns
7. Time-Series & Seasonality Features
8. Composite & Interaction Features

**Implementation Status:**

- Phase 1-5 feature groups implemented with TDD coverage
- Phase 6-8 features wired into advanced pipeline
- Phase 9 integration with classification/regression COMPLETE
- All features accessible via `finance_ml.ml_workflow.features.advanced.py`

**Model Version Target:** v9_9

C. Portfolio Optimization Enhancement Plan

**Overview:**
6-phase portfolio optimization enhancement plan documented in
`docs/improvement_plan/portfolio_optimization_enhancement_plan.md`. All phases complete as of 2025-11-17.

**Completed Phases:**

**Phase 1: Enhanced Stock Filtering & Selection** ✅

- Module: `finance_ml/ml_workflow/analytics/stock_selection.py`
- Features: Multi-metric ranking, sector-balanced selection, currency unit support

**Phase 2: ML-Based Return Prediction** ✅

- Module: `finance_ml/ml_workflow/analytics/ml_returns.py`
- Features: ML feature engineering, linear predictor, ensemble predictions

**Phase 3: Advanced Portfolio Optimization** ✅

- Module: `finance_ml/ml_workflow/analytics/portfolio.py`
- Features: Black-Litterman, Risk Parity, Hierarchical Risk Parity (HRP)

**Phase 4: Risk Management Enhancements** ✅

- Module: `finance_ml/ml_workflow/analytics/risk.py`
- Features: Expected Shortfall, tracking error, stress testing, Monte Carlo simulation

**Phase 5: Backtesting Framework** ✅

- Modules: `analytics/portfolio.py`, `analytics/attribution.py`
- Features: Vectorized backtest, walk-forward optimization, performance attribution

**Phase 6: Interactive Dashboard Expansion** ✅

- Module: `finance_ml/dashboards/portfolio_widgets.py`
- Features: Rebalancing widget, multi-period comparison, factor exposure dashboard

**Test Coverage:**

- 23 tests total (all passing)
- 5 new test modules covering all 6 phases
- Notebook integration: Section 10 structure added to `ml_finance_model_main.ipynb`

**Key Capabilities:**

- Advanced optimization methods (Black-Litterman, Risk Parity, HRP)
- Comprehensive risk management (ES, tracking error, stress tests, Monte Carlo)
- Robust backtesting framework with performance attribution
- Enhanced interactive dashboards for portfolio analytics

4) Additional Development Information
A. Code style and quality

- **Comprehensive Code Guidelines**: See `docs/code_guidelines.md` v1.4 (updated 2025-11-23) for detailed standards on:
    - Standardized function signatures and return types (train_* functions, dataset preparation)
    - Column naming schema and dataframe conventions (normalized: last_price, price_target, sector, region)
    - Typing, logging, and error handling
    - Testing conventions and reproducibility
    - Notebook and CLI alignment
  - Uncertainty and Prediction Intervals (quantile regression + conformal calibration)
  - Outlier Safety Rails Policy (winsorization, robust loss, clipping, non-negativity)
  - Data Split and Leakage Policy (time-series → grouped → stratified)
  - Standardized Predictions Schema (required columns and invariants)
  - Sector Metrics and Calibration (persistence contract, bias correction)
  - TDD Conventions and Selective Test Execution
  - NEW in v1.4: Notebook Best Practices and TDD Conventions (Section 8)
    - Centralized Configuration Constants (single source of truth)
    - DataFrame Stage Naming (8-stage pipeline): all_stocks_raw → all_stocks_normalized → all_stocks_typed →
      all_stocks_winsorized → all_stocks_imputed → all_stocks_scaled → all_stocks_features → all_stocks_enhanced
    - Magic Numbers Policy (replace meaningful literals with named constants)
- Python style: PEP 8 (black-like formatting), type hints where possible.
- Logging: prefer Python logging over prints. The notebook initializes a logger fallback if the custom one isn’t available.
- Reproducibility: set RANDOM_SEED where applicable; record library versions for experiments.

B. Notebook hygiene
- Keep data loading, feature engineering, and modeling steps modularized into functions within the notebook to ease reuse.
- Avoid hard-coding paths; use Path from pathlib and environment variables.
- Save intermediate artifacts (processed datasets, models) with versioned filenames.

C. Performance tips
- Use joblib parallelism judiciously (n_jobs) and monitor memory for large models.
- Start with a smaller feature set and scale up; use downsampling or stratification for class imbalance.

D. Development roadmap
- See IMPROVEMENT_PLAN.md for a phased development roadmap aligned to this project.
- The plan covers 8 phases including: foundations, data ingestion/validation, EDA/feature engineering, classification models, regression models, analytics/reporting, testing/CI, and packaging/modularity.
- Immediate next steps and risk mitigation strategies are documented.

E. Troubleshooting
- TensorFlow install issues: set TF_CPP_MIN_LOG_LEVEL=2; try CPU-only; verify Visual C++ Redistributable on Windows.
- Postgres import quoting: many column names have spaces; keep using double quotes in SQL to avoid errors.
- Large CSVs on Windows: prefer \copy from psql to avoid server-side file permission issues.

Appendix: IDE Data Source Settings (for JetBrains tools)
- You can configure a Data Source using the following (adapt as needed):
  #DataSourceSettings#
  #LocalDataSource: postgres@localhost
  #BEGIN#
  <data-source source="LOCAL" name="postgres@localhost" uuid="3091f81a-c800-413c-b86b-f760950418f7"><database-info product="PostgreSQL" version="18.0" jdbc-version="4.2" driver-name="PostgreSQL JDBC Driver" driver-version="42.7.3" dbms="POSTGRES" exact-version="18.0" exact-driver-version="42.7"><identifier-quote-string>&quot;</identifier-quote-string></database-info><case-sensitivity plain-identifiers="lower" quoted-identifiers="exact"/><driver-ref>postgresql</driver-ref><synchronize>true</synchronize><jdbc-driver>org.postgresql.Driver</jdbc-driver><jdbc-url>jdbc:postgresql://localhost:5432/postgres</jdbc-url><jdbc-additional-properties><property name="com.intellij.clouds.kubernetes.db.host.port"/><property name="com.intellij.clouds.kubernetes.db.enabled" value="false"/><property name="com.intellij.clouds.kubernetes.db.container.port"/></jdbc-additional-properties><secret-storage>master_key</secret-storage><user-name>postgres</user-name><schema-mapping><introspection-scope><node kind="database" qname="@"><node kind="schema" qname="@"/></node></introspection-scope></schema-mapping><working-dir>$ProjectFileDir$</working-dir></data-source>
  #END#

Versioning
- When modifying modeling behavior materially, bump MODEL_VERSION (e.g., v8_3) and document in this file what changed (features, labels, metrics).



---

Addendum: Key Features, Main Scripts, CLI, and Testing/Coverage

This addendum supplements the core guidelines with quick-reference details aligned with the README and packaging configuration. It is intended to speed up onboarding and future development.

Key Features (aligns with README)

- 📊 Data Management: Load from PostgreSQL or CSV, with validation and quality checks
- 🔧 Feature Engineering: Financial ratios, margins, volatility, revenue CAGR, and more
- 🤖 ML Models: Event classification, sector-optimized regression, quantile models, stacking ensembles
- 📈 Analytics: Mispricing scores, stock ranking, interactive visualizations
- ⚙️ Configuration: Flexible config via Anaconda (optional) or venv/pip with environment variables and CLI options;
  avoid mixing Conda with venv
- 🧪 Tested: Comprehensive unit tests with good coverage
- 🚀 CLI: Three command-line tools for different workflows (see CLI Tools below)

Finance_ML_Analytics_Platform — Main Scripts

- ml_finance_model_main.ipynb — Primary notebook for end-to-end workflow and exploration.
- ml_finance_model_main.py — Lightweight script version with a CLI for batch runs and automation.

CLI Tools (installed via pyproject’s console scripts)
- finance-ml — Primary pipeline runner (data load, preprocess, features, models, outputs).
  Example: finance-ml --data-source auto --limit 5000 --output-dir outputs
- finance-ml-analyze — EDA/analytics-only workflows.
  Example: finance-ml-analyze --data-source csv --output-dir outputs
- finance-ml-validate — Validation-only workflows (schema checks, data quality, etc.).
  Example: finance-ml-validate --data-source csv --output-dir outputs
Notes:
- The script ml_finance_model_main.py provides equivalent capabilities via Python directly:
  - python ml_finance_model_main.py --data-source auto --limit 5000 --out-dir outputs
- Data sources: --data-source auto|csv|db as described earlier. Use DB_URL or --db-url for database access.

Testing and Coverage
- Test runner (unittest):
  - python -m unittest -v
- Creating new tests:
  - Add files under tests/ named test_*.py and use unittest.TestCase.
  - Keep tests isolated from external services; prefer pure functions and mocks.
- Coverage (option A: coverage.py):
  1) pip install coverage
  2) coverage run -m unittest -v
  3) coverage report -m
  4) coverage html  # optional HTML report under htmlcov/
- Coverage (option B: pytest + pytest-cov):
  1) pip install pytest pytest-cov
  2) pytest --cov=finance_ml --cov-report=term-missing
- Demonstration test included: tests/test_coverage_smoke.py — a minimal smoke test that verifies arithmetic and imports the finance_ml package. Use it as a template for adding more tests and to confirm the coverage commands above work end-to-end.

Documentation and Artifact Update Checklist
When making significant changes, ensure the following files are updated accordingly (as applicable):
- environment_variables.txt — Add or adjust env vars; keep TF_CPP_MIN_LOG_LEVEL=2 default.
- Pipfile — Reflect dependency changes for Pipenv users.
- requirements.txt — Keep constraints aligned with pyproject.toml; consider optional extras.
- README.md — Key Features, setup, CLI usage, examples, and links.
- ml_finance_model_main.ipynb — Ensure cells align with the current pipeline functions and APIs.
- ml_finance_model_main.py — Keep CLI options and defaults in sync with package functions.
- IMPROVEMENT_PLAN.md — Log changes to features/labels/metrics; note version bumps.
- pyproject.toml — Update version, console scripts, optional dependencies, and tooling configs.
- qodana.yaml — Static analysis and quality gates configuration.
- import_equities_data.sql — Data loading procedure updates (staging, NULL handling, validation steps).
- create_equities_schema.sql — Schema changes; ensure quoted identifiers and ownership.
- .aiassistant/rules/promt_rules.md — Assistant and prompt rules if the dev workflow changes.
- LICENSE — Copyright years and ownership.

Notes on Configuration Consistency
- Prefer environment variables and CLI flags for configuration in code and notebooks.
- Avoid hard-coded paths; use pathlib.Path and env vars (DATA_DIR, MODEL_DIR, OUTPUT_DIR, etc.).
- Keep the notebook and script aligned with the finance_ml package APIs to reduce duplication.

Versioning Reminder
- When modifying modeling behavior materially, bump MODEL_VERSION (e.g., v8_3) and record changes (features, labels, metrics) in this file and IMPROVEMENT_PLAN.md.

## SQLite Local Setup and Data Load

For quick local testing without PostgreSQL, you can use SQLite.

1) Create the SQLite schema:

- sqlite3 equities.sqlite ".read create_equities_schema_sqlite.sql"

2) Import CSVs into SQLite using the provided CLI SQL script (recommended):

- sqlite3 equities.sqlite ".read import_equities_data_sqlite.sql"

This import script:

- Uses per‑region staging tables (US, EU, APAC, ROTW)
- Deletes header rows when imported as data
- Backfills missing Region values per file
- Inserts with INSERT OR IGNORE honoring the UNIQUE("Ticker","Region") index
- Prints a post‑import validation summary

3) Python alternative: chunked importer for very large CSVs:

- python tools/import_sqlite.py --db equities.sqlite --data-dir data --chunksize 2000
- python tools/import_sqlite.py --db equities.sqlite --regions US,EU

The Python importer (tools/import_sqlite.py) features:

- Configurable chunk size for memory-efficient processing of large CSVs
- Automatic header detection and removal
- NULL handling (converts empty strings to NULL)
- Per-region import with automatic Region column backfilling
- UNIQUE("Ticker","Region") constraint enforcement for deduplication
- Comprehensive test coverage via tests/test_sqlite_import.py

4) Data validation before import (recommended):

- python tools/validate_csv_import.py

This validation script:

- Checks schema compliance and critical column presence (Ticker, Sector, Last Price, etc.)
- Detects non-numeric values in numeric columns
- Identifies data quality issues before database import
- Produces detailed validation reports per region
- Comprehensive test coverage via tests/test_validate_csv_import.py

Notes:

- The Python importer uses pandas (already listed in requirements.txt). It reads CSVs with dtype=str, converts empty
  strings to NULLs, and backfills Region.
- sqlite3 is included with Python; no extra dependency is required.
- All SQLite import paths are fully tested with ≥80% coverage targets.
