# Finance ML Analytics Platform

**Version 0.3.3** — A professional, modular Python package for equity screening, feature engineering, and machine
learning models across global regions.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **TODO**: Sync version strings across files (pyproject.toml and finance_ml/__init__.py currently show 0.3.0)

## Quick Start (TL;DR)

- Python: 3.12 or 3.13
- Create and activate a virtual environment:
    - Windows (PowerShell):
        - python -m venv .venv
        - .venv\Scripts\Activate.ps1
    - macOS/Linux (bash):
        - python3 -m venv .venv
        - source .venv/bin/activate
- Upgrade pip tools: python -m pip install --upgrade pip setuptools wheel
- Install dependencies: pip install -r requirements.txt
- PostgreSQL: ensure local Postgres is running; create schema:
    - psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql
- Load data from CSVs into PostgreSQL:
    - psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
- Notebook: open **ml_finance_model_main.ipynb** and run cells in order
- Script (optional): python ml_finance_model_main.py --data-source auto --limit 5000 --out-dir outputs
- CLI tools (after pip install -e .): finance-ml --data-source auto --limit 5000
- Tests: python -m unittest -v

## Overview

Finance ML Analytics Platform is a comprehensive toolkit for quantitative equity analysis combining:
- **Unified data pipeline**: PostgreSQL integration + CSV fallback for multi-region equity data (US, EU, APAC, ROTW)
- **Modular Python package** (`finance_ml`): Clean, tested, reusable code for data loading, feature engineering, modeling, and analytics
- **Interactive notebook**: Jupyter-based workflow for exploration and prototyping
- **CLI tools**: Command-line interface for batch processing and automation
- **Production-ready**: Modern packaging, comprehensive tests, CI/CD, configuration management

### Business Objective

**Primary Goal**: Predict Stock Price Targets for all stocks in the portfolio to support investment decisions and
portfolio optimization.

**Target Variable**: "Predicted Price Target" for regression modeling

The platform implements a sophisticated 8-step ML workflow aligned with industry best practices (
see [IMPROVEMENT_PLAN.md Phase 9](IMPROVEMENT_PLAN.md#phase-9--advanced-stock-prediction-ml-system-in-progress) for
detailed implementation plan):

1. Loading and preprocessing financial data from multiple regions
2. Exploratory data analysis of financial metrics
3. Advanced feature engineering with sector-specific optimizations
4. Multi-class classification of financial events using sophisticated models
5. Training sector-optimized regression models enhanced with classification features
6. Model evaluation and error analysis
7. Identification of under/overvalued stocks with visualization
8. Comprehensive analytics: Predicted vs. Analyst Price Target comparison

### Key Features
- 📊 **Data Management**: Load from PostgreSQL or CSV, with validation and quality checks
- 🔧 **Feature Engineering**: Financial ratios, margins, volatility, revenue CAGR, sector-specific optimizations (
  Financials, Energy, Tech, Healthcare)
- 🤖 **ML Models**: Multi-class event classification, sector-optimized regression, quantile models, stacking ensembles,
  deep learning (TensorFlow/Keras)
- 📈 **Analytics**: Mispricing scores, stock ranking, interactive visualizations, Predicted vs. Analyst Target comparison
- 🎯 **Stock Prediction**: Sophisticated 8-step workflow for price target prediction with comprehensive error analysis
- 📊 **Reporting**: Excel reports (matching Stock_Prediction_Analysis_Report format), PDF generation, interactive
  dashboards
- ⚙️ **Configuration**: Flexible config via environment variables, JSON, or YAML
- 🧪 **Tested**: Comprehensive unit tests with good coverage
- 🚀 **CLI**: Three command-line tools for different workflows
- 🔍 **Model Interpretation**: SHAP, LIME for explainability and feature importance analysis

### Phase 9 Implementation Status

**Advanced Stock Prediction ML System** — 7 of 8 phases complete (87.5%)

✅ **Completed Phases:**

- **Phase 9.1**: Advanced preprocessing and data quality (619 lines) — Outlier detection, winsorization, imputation,
  scaling
- **Phase 9.2**: Advanced EDA with statistical analysis (685 lines) — Correlation, normality tests, PCA, automated
  reporting
- **Phase 9.3**: Advanced feature engineering (685 lines) — 40+ financial ratios, sector-specific features, feature
  selection
- **Phase 9.4**: Multi-class event classification (1,336 lines) — Neural networks, ensemble methods, SHAP interpretation
- **Phase 9.5**: Sector-optimized regression (1,091 lines) — 12+ regression models, hyperparameter optimization,
  stacking
- **Phase 9.6**: Model evaluation and error analysis (314 lines) — Comprehensive metrics (MAE, RMSE, MAPE, R²), residual
  analysis, cross-validation strategies
- **Phase 9.7**: Stock valuation analysis (395 lines) — Mispricing scores, valuation categories, multi-factor screening,
  interactive visualizations

🔄 **In Progress:**

- **Phase 9.8**: Prediction analytics — Model vs analyst comparison, Excel reports, tracking

See [CHANGELOG.md](CHANGELOG.md) for detailed implementation notes and version history.

## Tech Stack

### Core Technologies

- **Language**: Python 3.12 or 3.13 (required)
- **Package Manager**: pip + venv (primary), pipenv (alternative)
- **Notebook**: Jupyter Notebook or JupyterLab
- **Database**: PostgreSQL 15+ (local instance recommended)

### Key Libraries
- **ML/DS Core**: numpy, pandas, scipy, scikit-learn, statsmodels
- **Gradient Boosting**: xgboost, lightgbm, catboost
- **Deep Learning**: tensorflow, scikeras (optional)
- **Model Interpretation**: shap
- **Visualization**: matplotlib, seaborn, plotly
- **Utilities**: tqdm, joblib, xlsxwriter, psutil

See `requirements.txt` for complete list with version constraints.

### Package Installation Options
The package supports optional dependency groups via `pyproject.toml`:
- **`[dev]`**: pytest, black, flake8, mypy, isort (development tools)
- **`[tensorflow]`**: tensorflow, scikeras (deep learning)
- **`[database]`**: psycopg2-binary, SQLAlchemy, openpyxl (database access from Python)
- **`[advanced-features]`**: boruta, numba (advanced feature selection and performance)
- **`[notebook]`**: jupyter, notebook, ipykernel (Jupyter notebook support)
- **`[all]`**: All optional dependencies combined


## Requirements
- OS: Windows 10/11 (tested), macOS, or Linux
- Python: 3.12 or 3.13
- PostgreSQL: 15+ (local instance)
- Optional: Git for version control


## Setup

1) Create and activate a virtual environment
- Windows (PowerShell):
  - python -m venv .venv
  - .venv\Scripts\Activate.ps1
- macOS/Linux (bash):
  - python3 -m venv .venv
  - source .venv/bin/activate

2) Upgrade packaging tools
- python -m pip install --upgrade pip setuptools wheel

3) Install Python dependencies

**Basic installation** (includes core + optional dependencies):
- pip install -r requirements.txt
    - Note: `requirements.txt` includes TensorFlow, database libraries (psycopg2, SQLAlchemy), Jupyter, and other
      optional dependencies for a complete setup. If you encounter TensorFlow installation issues, you can comment it
      out—the core workflow uses scikit-learn and gradient boosting libraries.

**Package installation** (for CLI tools with selective dependencies):

- pip install -e . # Core package only (minimal dependencies)
- pip install -e ".[dev]"             # With development tools
- pip install -e ".[database]"        # With database client libraries
- pip install -e ".[notebook]"        # With Jupyter notebook support
- pip install -e ".[tensorflow]"      # With TensorFlow/Keras deep learning
- pip install -e ".[all]"             # With all optional dependencies

**Note**: The package installation approach allows selective installation of optional features, while `requirements.txt`
provides a batteries-included setup.

### Optional: Conda environment (Anaconda/Miniconda)

If you prefer Conda, an environment file is provided:

- Finance_ML_Analytics_Platform.yaml

Create and activate the Conda env:

- conda env create -f Finance_ML_Analytics_Platform.yaml
- conda activate Finance_ML_Analytics_Platform

Note: The primary, tested path is venv + pip. Avoid mixing Conda envs with Python venv in the same project.

### Automated setup helper (optional)

Use the setup helper to streamline local setup in one command:

- python setup_environment.py

Common flags:

- --skip-db — Skip PostgreSQL setup
- --skip-data-load — Skip CSV import into DB
- --recreate-venv — Recreate the .venv
- --install-db-libs — Also install psycopg2-binary and SQLAlchemy
- --force — Continue even if a step fails (not recommended)

See setup_environment.py --help for full usage.

## PostgreSQL Setup and Data Load
1) Ensure PostgreSQL is installed and running locally, and psql is on your PATH.

2) Create the equities table by running the SQL script in the project root:
- Windows (PowerShell):
  - psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql

The script creates the equities table and assigns ownership to postgres (ALTER TABLE equities OWNER TO postgres;). Many column names contain spaces; always use double quotes in SQL when referencing them.

3) (Optional but Recommended) Validate CSV data quality before import:
- Windows (PowerShell):
  - python validate_csv_import.py
- This script validates schema, checks for missing values, and identifies potential data quality issues before database
  import.

4) Load the regional CSVs from data/ into the equities table using the comprehensive import script:
- Windows (PowerShell):
  - psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
- This script handles all four regions (US, EU, APAC, ROTW) with proper NULL handling, staging tables, and validation.

Alternative: Manual import per region with proper NULL handling (if you need more control):
- Example for US region using staging table approach:
  - psql -h localhost -p 5432 -U postgres -d postgres -c "CREATE TEMP TABLE equities_staging (LIKE equities)"
  - psql -h localhost -p 5432 -U postgres -d postgres -c "\\copy equities_staging FROM 'data/screening_us.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')"
  - psql -h localhost -p 5432 -U postgres -d postgres -c "UPDATE equities_staging SET \"Region\"='US' WHERE \"Region\" IS NULL"
  - psql -h localhost -p 5432 -U postgres -d postgres -c "INSERT INTO equities SELECT * FROM equities_staging ON CONFLICT DO NOTHING"

Key import parameters:
- NULL '' - Treats empty strings as NULL values (critical for proper data import)
- ENCODING 'UTF8' - Ensures proper character handling
- HEADER true - Skips the CSV header row

Repeat per region (EU/APAC/ROTW) substituting the correct CSV and Region value.

### Security Note

**⚠️ Important**: Never commit database passwords or API keys to version control.

- Use environment variables or `.env` files (add `.env` to `.gitignore`)
- The `environment_variables.txt` file contains example configurations with placeholders
- For production deployments, use secure secret management (e.g., GitHub Secrets, AWS Secrets Manager)
- Review all configuration files before committing to ensure no credentials are exposed

## Environment Variables

Environment variable defaults and examples live in `environment_variables.txt`. Key items:

- **TF_CPP_MIN_LOG_LEVEL=2** — Reduces TensorFlow logging verbosity (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
- **LOG_LEVEL** — Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Optional paths**: DATA_DIR, MODEL_DIR, CACHE_DIR, OUTPUT_DIR
- **Optional modeling**: MODEL_VERSION, RANDOM_SEED
- **Optional performance**: N_JOBS, MEMORY_LIMIT
- **Database**: DB_URL, DB_SCHEMA, DB_TABLE
- **External APIs**: ALPHA_VANTAGE_API_KEY, FINANCIAL_API_KEY (if using external data sources)

Set them in your shell or via a .env file if your tools auto‑load it.
- Windows (PowerShell):
  - $env:TF_CPP_MIN_LOG_LEVEL = 2
- macOS/Linux (bash):
  - export TF_CPP_MIN_LOG_LEVEL=2

See `environment_variables.txt` for complete list with detailed descriptions.

## Running the Project

This project is notebook‑first with two main interactive notebooks.

### Interactive Notebooks

1) Start Jupyter
- jupyter notebook  (or: jupyter lab)

2) Primary notebooks:

- **ml_finance_model_main.ipynb** — Main end-to-end ML pipeline (data loading, feature engineering, modeling, analytics)
- **equities_data_explorer.ipynb** — Data exploration and preparation workflow with EDA utilities

Run cells in order. Both notebooks use environment variables and Path from pathlib to avoid hard‑coded paths.

Outputs: model diagnostics, ranking tables, visualizations, and optional CSV/Excel exports.


## Scripts and Entry Points

### Main Entry Points

- **ml_finance_model_main.ipynb** — Main Jupyter notebook for interactive exploration and modeling
- **equities_data_explorer.ipynb** — Data exploration notebook with EDA utilities and preprocessing
- **ml_finance_model_main.py** — Lightweight Python script with a minimal CLI:
  - `--data-source {auto|csv|db}` — Data source selection (default: auto)
  - `--db-url <url>` — Database connection string (or use DB_URL env var)
  - `--limit <n>` — Limit rows for testing
  - `--out-dir <path>` — Output directory (default: outputs)
  - `--dry-run` — Skip model training
  - `--n-jobs <n>` — Number of parallel jobs
  - Note: This is a lightweight wrapper. For advanced options, use the console script `finance-ml`.

### Console Scripts (installed via pyproject)

- `finance-ml` — Main pipeline (finance_ml.cli:main)
- `finance-ml-analyze` — EDA/analytics-only (finance_ml.cli:analyze_main)
- `finance-ml-validate` — Validation-only (finance_ml.cli:validate_main)

### Database Setup Scripts
- **create_equities_schema.sql** — SQL script to initialize the equities table in PostgreSQL
- **import_equities_data.sql** — Comprehensive import script for all regional CSVs with staging tables and validation

### Utility Scripts
- **setup_environment.py** — Automated environment setup utility:
  - Checks prerequisites (Python, PostgreSQL, Git)
  - Creates virtual environment
  - Installs dependencies
  - Sets up database and loads CSV data
  - Configures environment variables
  - Runs tests
  - Provides activation instructions
- **validate_csv_import.py** — CSV data quality validator (runs validation functions before import)
    - Validates schema compliance and critical column presence
    - Checks for non-numeric values in numeric columns
    - Identifies data quality issues before database import
    - Produces validation reports for each region (US, EU, APAC, ROTW)
- **load_equities_data.py** — Legacy PostgreSQL CSV importer used to insert CSV data into the equities table.
    - WARNING: Contains a hard-coded password in the sample; update it or prefer env vars/DB_URL.
    - TODO: Replace inline credentials with environment variables or remove password entirely.
- **analyze_notebook.py** — Notebook structure analyzer (counts cells, previews content, searches for functions)
- **refactor_notebook.py** — Notebook refactoring helper
- **update_notebook.py** — Notebook synchronizer (extracts TDD functions from .py and inserts into .ipynb)
- **update_notebook_imports.py** — Update notebook imports utility
- **verify_notebook.py** — Notebook verification utility (checks for presence of TDD functions)
- **verify_preprocessing_improvements.py** — Verify preprocessing pipeline improvements
- **validate_phase9_integration.py** — Automated Phase 9 integration validation script
- **tools/import_sqlite.py** — Chunked CSV-to-SQLite importer for quick local testing
    - Supports per-region imports with automatic Region backfilling
    - Handles header removal and NULL value mapping
    - Configurable chunk size for large CSV files
    - Implements UNIQUE("Ticker","Region") constraint for deduplication
- **tools/apply_improvement_plan_updates.py** — Development plan update automation
- **tools/apply_notebook_fixes.py** — Notebook maintenance utility

### Important files at a glance

- Finance_ML_Analytics_Platform.yaml — Conda environment definition (optional)
- environment_variables.txt — Default/env examples (includes TF_CPP_MIN_LOG_LEVEL=2)
- requirements.txt — Python dependencies for pip installs
- ml_finance_model_main.ipynb — Primary notebook for end-to-end workflow
- create_equities_schema_sqlite.sql — SQLite schema for equities table
- import_equities_data_sqlite.sql — SQLite CSV import script with validation
- equities.sqlite — Example/working SQLite database file for local runs
- identifier.sqlite — Auxiliary SQLite DB present in repo
    - TODO: Clarify its role and whether it’s required by any scripts/tests


## Tests
- We use Python’s built‑in unittest.
- Run all tests from the project root:
  - python -m unittest -v

Comprehensive test suite in tests/ directory (37 test modules):

- test_advanced_eda.py — Advanced EDA functions (correlation, PCA, statistical tests)
- test_advanced_features.py — Phase 9.3 advanced feature engineering tests
- test_advanced_models_phase95.py — Phase 9.5 advanced regression models and ensembles
- test_advanced_preprocessing.py — Advanced preprocessing (outlier detection, winsorization, imputation)
- test_analytics.py — Analytics and stock ranking tests
- test_build_features.py — Feature building pipeline
- test_classification.py — Event classification model tests
- test_classification_phase94.py — Phase 9.4 advanced classification models
- test_cli.py — Command-line interface tests
- test_coverage_smoke.py — Smoke test for coverage validation
- test_data_quality.py — Data validation and quality checks
- test_eda.py — Exploratory data analysis utilities
- test_evaluation_phase96.py — Phase 9.6 model evaluation and error analysis tests
- test_features.py — Feature engineering functions
- test_finance_ml_config.py — Configuration management tests
- test_finance_ml_data.py — Data loading module tests
- test_finance_ml_eval.py — Evaluation and analytics module tests
- test_finance_ml_features.py — Features module tests
- test_finance_ml_models.py — Models module tests
- test_improvement_plan_revision.py — Development plan validation
- test_loaders.py — CSV and database loading functions
- test_logging.py — Logging configuration tests
- test_notebook_config.py — Notebook configuration tests
- test_notebook_enhancements.py — Notebook enhancements validation
- test_notebook_integration.py — Notebook integration tests
- test_notebook_quality_improvements.py — Notebook quality improvements tests (config API, type validation, error
  handling)
- test_portfolio_optimization.py — Portfolio optimization tests
- test_preprocess_and_training.py — Preprocessing and training workflows
- test_regression.py — Regression model evaluation
- test_repository_setup.py — Validates repository basics (required files, SQL schema, environment config)
- test_risk_metrics.py — Risk metrics calculation tests
- test_setup_environment.py — Setup script validation
- test_sql_scripts.py — SQL script validation tests
- test_sqlite_import.py — SQLite import functionality (header removal, NULL handling, region backfilling)
- test_validate_csv_import.py — CSV validation (schema validation, data quality checks)
- test_valuation_phase97.py — Phase 9.7 stock valuation and identification tests
- test_visualizations.py — Visualization functions tests


## Project Structure

```
Finance_ML_Analytics_Platform/
├── finance_ml/                    # Main Python package (v0.3.3)
│   ├── __init__.py               # Package exports and version
│   ├── advanced_eda.py           # Advanced EDA with statistical analysis (Phase 9.2)
│   ├── advanced_features.py      # Advanced feature engineering (Phase 9.3)
│   ├── advanced_models.py        # Sector-optimized regression models (Phase 9.5)
│   ├── advanced_preprocessing.py # Advanced preprocessing and data quality (Phase 9.1)
│   ├── classification.py         # Multi-class event classification (Phase 9.4)
│   ├── cli.py                    # Command-line interface
│   ├── config.py                 # Configuration management
│   ├── data.py                   # Data loading, normalization, validation
│   ├── eval.py                   # Analytics, visualizations, reporting
│   ├── features.py               # Feature engineering functions
│   ├── logging_config.py         # Logging configuration utilities
│   ├── models.py                 # ML models (classification, regression, ensembles)
│   ├── notebook_config.py        # Notebook-specific helpers and config
│   ├── notebook_utils.py         # Notebook utility functions
│   ├── portfolio_optimization.py # Portfolio optimization utilities
│   ├── risk_metrics.py           # Risk metrics and portfolio risk analysis
│   └── verify_requirements.py    # Requirements verification utility
│
├── tests/                        # Unit tests (comprehensive test suite, 32 modules)
│   ├── test_advanced_eda.py
│   ├── test_advanced_models_phase95.py
│   ├── test_advanced_preprocessing.py
│   ├── test_analytics.py
│   ├── test_build_features.py
│   ├── test_classification.py
│   ├── test_classification_phase94.py
│   ├── test_cli.py
│   ├── test_coverage_smoke.py
│   ├── test_data_quality.py
│   ├── test_eda.py
│   ├── test_features.py
│   ├── test_finance_ml_config.py
│   ├── test_finance_ml_data.py
│   ├── test_finance_ml_eval.py
│   ├── test_finance_ml_features.py
│   ├── test_finance_ml_models.py
│   ├── test_improvement_plan_revision.py
│   ├── test_loaders.py
│   ├── test_logging.py
│   ├── test_notebook_config.py
│   ├── test_notebook_enhancements.py
│   ├── test_portfolio_optimization.py
│   ├── test_preprocess_and_training.py
│   ├── test_regression.py
│   ├── test_repository_setup.py
│   ├── test_risk_metrics.py
│   ├── test_setup_environment.py
│   ├── test_sqlite_import.py
│   ├── test_sql_scripts.py
│   ├── test_validate_csv_import.py
│   └── test_visualizations.py
│
├── data/                         # Regional equity data (CSV files)
│   ├── screening_us.csv
│   ├── screening_eu.csv
│   ├── screening_apac.csv
│   └── screening_rotw.csv
│
├── ml_finance_model_main.ipynb   # Main interactive Jupyter notebook (end-to-end ML pipeline)
├── equities_data_explorer.ipynb  # Data exploration and EDA notebook
├── ml_finance_model_main.py      # Lightweight script version with CLI (uses finance_ml package)
├── archive/                       # Archived versions (v8_2, etc.)
│
├── tools/                         # Utility scripts
│   ├── __init__.py                # Package marker
│   ├── apply_improvement_plan_updates.py  # Development plan update automation
│   ├── apply_notebook_fixes.py    # Notebook maintenance utility
│   └── import_sqlite.py           # Chunked CSV→SQLite importer
│
├── analyze_notebook.py           # Notebook analysis utility
├── refactor_notebook.py          # Notebook refactoring helper
├── update_notebook.py            # Sync functions between .py and .ipynb
├── update_notebook_imports.py    # Update notebook imports
├── verify_notebook.py            # Verify notebook functions
├── verify_preprocessing_improvements.py # Verify preprocessing pipeline improvements
├── validate_csv_import.py        # CSV data validation helper
│
├── create_equities_schema.sql           # PostgreSQL schema setup
├── import_equities_data.sql             # Data import script (PostgreSQL)
├── create_equities_schema_sqlite.sql    # SQLite schema setup
├── import_equities_data_sqlite.sql      # SQLite data import script
│
├── environment_variables.txt     # Environment configuration examples
├── requirements.txt              # Core dependencies
├── Pipfile                       # Pipenv (optional)
├── pyproject.toml                # Modern packaging configuration (console scripts)
├── setup.py                      # Legacy setup (editable installs)
├── qodana.yaml                   # Static analysis configuration
├── README.md                     # This file
├── LICENSE                       # License
├── CHANGELOG.md                  # Version history
├── IMPROVEMENT_PLAN.md           # Development roadmap
├── IMPLEMENTATION_SUMMARY.md     # Implementation notes
├── PREPROCESSING_PIPELINE_IMPROVEMENTS.md # Docs
├── NOTEBOOK_ENHANCEMENTS.md      # Docs
├── REFACTORING_COMPLETE.md       # Docs
├── REFACTORING_SUMMARY.md        # Docs
├── TDD_IMPLEMENTATION_COMPLETE.md # Docs
├── TDD_IMPLEMENTATION_SUMMARY.md  # Docs
└── reports/                      # Generated reports (if any)
```

### Package Modules

#### `finance_ml.data`
Data loading, normalization, and validation functions.
- `load_from_csv()`, `load_from_db()`: Multi-source data loading
- `preprocess()`: Data cleaning and normalization
- `validate_schema()`, `check_missing_values()`: Quality checks
- `detect_outliers_iqr()`, `validate_numeric_ranges()`: Outlier detection

#### `finance_ml.features`
Feature engineering for financial data.
- `engineer_basic_ratios()`: EV/EBITDA, P/E, P/B ratios
- `engineer_margin_features()`: Gross, operating, net margins
- `engineer_volatility_features()`: Price volatility windows
- `engineer_revenue_cagr()`: Revenue growth metrics
- `build_features_and_target()`: Complete feature pipeline

#### `finance_ml.models`
Machine learning models for classification and regression.
- `train_event_classifier()`: Multi-class event classification
- `train_and_evaluate_regression()`: Baseline regression models
- `train_and_evaluate_regression_by_sector()`: Sector-optimized models
- `train_quantile_regression()`: Uncertainty estimation
- `train_stacking_ensemble()`: Meta-learner stacking

#### `finance_ml.eval`
Analytics, visualizations, and reporting.
- `calculate_mispricing_score()`: Valuation analysis
- `rank_undervalued_stocks()`, `rank_overvalued_stocks()`: Stock ranking
- `simple_eda()`: Exploratory data analysis
- `create_sector_heatmap()`, `create_interactive_prediction_plot()`: Visualizations
- `export_predictions_to_excel()`: Excel reporting

#### `finance_ml.config`
Configuration management system.
- `FinanceMLConfig`: Configuration dataclass
- `load_config()`: Load from environment/JSON/YAML
- `get_config()`, `set_config()`: Global config management

#### `finance_ml.cli`
Command-line interface tools.
- `finance-ml`: Main analysis pipeline
- `finance-ml-analyze`: Quick data analysis
- `finance-ml-validate`: Data validation

#### `finance_ml.advanced_preprocessing`

Advanced preprocessing and data quality (Phase 9.1).

- `detect_outliers_iqr()`, `detect_outliers_zscore()`: Outlier detection methods
- `detect_outliers_isolation_forest()`: Multivariate outlier detection
- `winsorize_by_sector()`: Sector-specific winsorization
- `calculate_data_quality_score()`: Comprehensive data quality assessment
- `impute_missing_values()`: Advanced imputation strategies
- `create_scaler_pipeline()`, `scale_features()`: Feature scaling pipelines

#### `finance_ml.advanced_eda`

Advanced exploratory data analysis with statistical testing (Phase 9.2).

- `calculate_correlation_matrix()`: Pearson, Spearman, Kendall correlation analysis
- `test_normality()`: Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling tests
- `compare_sector_means()`: One-Way ANOVA, Kruskal-Wallis H-test
- `perform_pca()`: Principal Component Analysis with loadings
- `calculate_mutual_information()`: Feature importance via mutual information
- `generate_eda_report()`: Automated comprehensive EDA report generation

#### `finance_ml.advanced_features`

Advanced feature engineering with sector-specific optimizations (Phase 9.3).

- `calculate_financial_ratios()`: 40+ comprehensive financial ratios (valuation, profitability, leverage, liquidity,
  efficiency, growth)
- `calculate_sector_specific_features()`: Sector-tailored features (Financials: TBV; Tech/Healthcare: R&D intensity;
  Industrials: CAPEX)
- `create_feature_interactions()`: Pairwise interactions and polynomial features
- `calculate_relative_value_features()`: Sector-relative metrics (deviation, z-scores, percentile ranks)
- `select_features_mutual_info()`, `select_features_random_forest()`: Automated feature selection
- `build_comprehensive_features()`: One-stop orchestration for complete feature pipeline

#### `finance_ml.classification`

Multi-class event classification with advanced models (Phase 9.4).

- `create_enhanced_event_labels()`: Enhanced event labeling (price momentum, sector-adjusted thresholds)
- `train_neural_network_classifier()`: Deep neural network with batch normalization and dropout
- `train_voting_classifier()`, `train_stacking_classifier()`: Ensemble methods (voting, stacking)
- `compute_shap_values()`: SHAP-based model interpretation
- `export_classification_features()`: Export classification probabilities as meta-features for regression
- `cross_validate_classifier()`: Stratified k-fold cross-validation framework
- `compare_classifiers()`: Compare 7 classifiers (RF, XGBoost, LightGBM, CatBoost, Neural Network, Voting, Stacking)

#### `finance_ml.advanced_models`

Sector-optimized regression models with classification features (Phase 9.5).

- `prepare_regression_data_with_classification()`: Integrate classification meta-features into regression pipeline
- `train_linear_regression()`, `train_ridge_regression()`, `train_lasso_regression()`: Linear models with regularization
- `train_xgboost_regressor()`, `train_lightgbm_regressor()`, `train_catboost_regressor()`: Gradient boosting regressors
- `train_random_forest_regressor()`, `train_extra_trees_regressor()`: Tree-based ensemble regressors
- `train_neural_network_regressor()`: Deep learning regression with TensorFlow/Keras
- `train_stacking_regressor()`, `train_voting_regressor()`: Meta-learning ensembles
- `train_quantile_regressor()`: Uncertainty estimation via quantile regression
- `optimize_hyperparameters()`: Optuna-based hyperparameter optimization
- `compare_regression_models()`: Comprehensive model comparison framework
- `train_sector_specific_models()`: Train separate models per sector

#### `finance_ml.risk_metrics`

Risk metrics and portfolio risk analysis.

- `calculate_var_historical()`, `calculate_var_parametric()`: Value at Risk calculation
- `calculate_cvar()`: Conditional Value at Risk
- `calculate_sharpe_ratio()`, `calculate_sortino_ratio()`: Risk-adjusted returns
- `calculate_max_drawdown()`: Maximum drawdown analysis
- `calculate_portfolio_risk_metrics()`: Comprehensive portfolio risk

#### `finance_ml.portfolio_optimization`

Modern Portfolio Theory and optimization.

- `calculate_portfolio_return()`, `calculate_portfolio_volatility()`: Portfolio metrics
- `calculate_portfolio_sharpe_ratio()`: Portfolio Sharpe ratio
- `generate_efficient_frontier()`: Efficient frontier generation
- `optimize_portfolio_max_sharpe()`: Maximum Sharpe ratio optimization
- `optimize_portfolio_min_volatility()`: Minimum volatility optimization
- `optimize_portfolio_target_return()`: Target return optimization
- `rebalance_portfolio()`: Portfolio rebalancing

#### `finance_ml.logging_config`

Logging configuration and management.

- `setup_file_logging()`: Configure file-based logging
- `configure_logging()`: General logging configuration
- `get_logger()`: Get configured logger instance
- `add_file_handler()`, `remove_file_handlers()`: Manage log handlers
- `get_log_level()`, `set_log_level()`: Log level management

#### `finance_ml.notebook_config`

Notebook-specific configuration and helpers.

- `NotebookConfig`: Configuration dataclass for notebook environments
- Integration with main configuration system

#### `finance_ml.notebook_utils`

Notebook utility and display functions.

- `display_config_summary()`: Display configuration summary
- `load_stock_data()`: Notebook data loading helper
- `display_data_summary()`: Display data statistics
- `display_validation_results()`: Show validation results
- `display_missing_values_summary()`: Missing values report
- `validate_and_display_data()`: Combined validation and display
- `perform_and_display_eda()`: EDA execution and display

## Troubleshooting
- TensorFlow installation issues: the project primarily uses scikit‑learn and gradient boosting libraries. CPU‑only TensorFlow is fine; ensure compatible system libraries. If installation is problematic on your machine, you can proceed with non‑TF parts first.
- PostgreSQL quoting: many columns contain spaces/punctuation; always use double quotes in SQL identifiers.
- Windows CSV imports: prefer \\copy from psql to avoid server‑side file permission issues.


## Versioning
- When materially changing modeling behavior, bump MODEL_VERSION (e.g., v8_3) and document changes in this README (features, labels, metrics). Default in environment_variables.txt is MODEL_VERSION=v8_2 (commented example).


## Contributing

Contributions are welcome! This project follows standard open-source contribution practices.

### How to Contribute

1. **Fork the repository** and create a feature branch
2. **Install development dependencies**: `pip install -e ".[dev]"`
3. **Make your changes** following the code style guidelines
4. **Run tests**: `python -m unittest discover -s tests -v`
5. **Run code quality checks**:
   - Format: `black finance_ml tests`
   - Sort imports: `isort finance_ml tests`
   - Lint: `flake8 finance_ml`
   - Type check: `mypy finance_ml --ignore-missing-imports`
6. **Submit a pull request** with a clear description of changes

### Code Style

- Follow PEP 8 conventions
- Use `black` for code formatting (line length: 100)
- Use `isort` for import sorting
- Add type hints where possible
- Write docstrings for public functions and classes

### Testing

- Write unit tests for new functionality
- Maintain or improve test coverage
- Use small, deterministic test data
- Mock external dependencies (database, APIs)

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Finance ML Analytics Platform Contributors


## CLI Usage

The package provides three command-line tools after installation:

### 1. `finance-ml` — Main Analysis Pipeline

Full ML pipeline with data loading, EDA, feature engineering, and model training.

```bash
# Auto-detect data source (DB if available, else CSV)
finance-ml --data-source auto --limit 5000

# Use database
finance-ml --data-source db --db-url postgresql+psycopg2://postgres:@localhost:5432/postgres

# Use CSV files
finance-ml --data-source csv --data-dir ./data

# Dry run (skip training)
finance-ml --data-source csv --dry-run

# Use config file
finance-ml --config config.json

# Full options
finance-ml --data-source auto \
           --output-dir ./outputs \
           --n-jobs 4 \
           --seed 42 \
           --log-level INFO \
           --verbose
```

### 2. `finance-ml-analyze` — Quick Analysis

Fast EDA and data profiling.

```bash
# Analyze CSV data
finance-ml-analyze --data-source csv --data-dir ./data

# Analyze database
finance-ml-analyze --data-source db --db-url postgresql+psycopg2://postgres:@localhost:5432/postgres

# Limit rows for quick check
finance-ml-analyze --data-source csv --limit 1000 -v
```

### 3. `finance-ml-validate` — Data Validation

Validate data quality and schema.

```bash
# Validate CSV data
finance-ml-validate --data-source csv --data-dir ./data

# Validate database
finance-ml-validate --data-source db --db-url postgresql+psycopg2://postgres:@localhost:5432/postgres

# Verbose output
finance-ml-validate --data-source csv -v
```

### Lightweight Script

The Python script can be run directly without package installation:

```bash
python ml_finance_model_main.py --data-source auto --limit 5000 --out-dir outputs
python ml_finance_model_main.py --data-source csv --dry-run
python ml_finance_model_main.py --data-source db --db-url postgresql+psycopg2://user:password@localhost:5432/postgres
```

Note: This script uses the `finance_ml` package internally. For advanced options, use the `finance-ml` console command.


## Configuration Management

### Environment Variables

Set these in your shell or via `.env` file:

```bash
# macOS/Linux (bash/zsh)
# Data paths
export DATA_DIR=data
export MODEL_DIR=models
export CACHE_DIR=.cache
export OUTPUT_DIR=outputs

# Database
export DB_URL=postgresql+psycopg2://postgres:@localhost:5432/postgres

# Model settings
export MODEL_VERSION=v8_2
export RANDOM_SEED=42
export N_JOBS=-1

# Logging
export LOG_LEVEL=INFO
export TF_CPP_MIN_LOG_LEVEL=2
```

```powershell
# Windows (PowerShell)
# Data paths
$env:DATA_DIR = "data"
$env:MODEL_DIR = "models"
$env:CACHE_DIR = ".cache"
$env:OUTPUT_DIR = "outputs"

# Database
$env:DB_URL = "postgresql+psycopg2://postgres:@localhost:5432/postgres"

# Model settings
$env:MODEL_VERSION = "v8_2"
$env:RANDOM_SEED = "42"
$env:N_JOBS = "-1"

# Logging
$env:LOG_LEVEL = "INFO"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
```

### Configuration Files

Create `config.json` or `config.yaml` for project settings:

**config.json:**
```json
{
  "data_dir": "data",
  "output_dir": "outputs",
  "db_url": "postgresql+psycopg2://postgres:@localhost:5432/postgres",
  "model_version": "v8_2",
  "random_seed": 42,
  "n_jobs": -1
}
```

**config.yaml:**
```yaml
data_dir: data
output_dir: outputs
db_url: postgresql+psycopg2://postgres:@localhost:5432/postgres
model_version: v8_2
random_seed: 42
n_jobs: -1
```

Load config in Python:
```python
from finance_ml import load_config

config = load_config("config.json")
config.apply_to_env()
```


## Development

### Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run with pytest (if installed)
pytest tests/ -v --cov=finance_ml

# Run specific test module
python -m unittest tests.test_finance_ml_data
```

### Code Quality

```bash
# Format code
black finance_ml tests

# Sort imports
isort finance_ml tests

# Type checking
mypy finance_ml --ignore-missing-imports

# Linting
flake8 finance_ml
```

### Installing in Development Mode

```bash
# Install package in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install with all optional dependencies
pip install -e ".[all]"
```


## CI/CD

The project includes a comprehensive GitHub Actions workflow (`.github/workflows/tests.yml`) that runs automatically on
push and pull requests to `main` and `develop` branches.

### Workflow Jobs

1. **test** — Matrix testing across multiple platforms and Python versions
    - Platforms: Ubuntu, Windows, macOS
   - Python versions: 3.12, 3.13
    - Runs unittest suite with coverage reporting (pytest-cov and coverage.py)
    - Uploads coverage to Codecov

2. **test-optional-deps** — Tests with optional dependencies
    - Includes psycopg2-binary, SQLAlchemy, PyYAML
    - Validates database integration functionality

3. **install-test** — Package installation validation
    - Tests editable installation (`pip install -e .`)
    - Validates CLI entry points (finance-ml, finance-ml-analyze, finance-ml-validate)
    - Checks package imports and version info

4. **code-quality** — Code quality checks (continue-on-error)
    - black: Code formatting
    - isort: Import sorting
    - flake8: Linting (syntax errors and code quality)
    - mypy: Static type checking

### Running Checks Locally

```bash
# Run tests
python -m unittest discover -s tests -v

# Run with coverage
pytest tests/ -v --cov=finance_ml --cov-report=term

# Check code formatting
black --check finance_ml tests

# Check import sorting
isort --check-only finance_ml tests

# Lint code
flake8 finance_ml

# Type checking
mypy finance_ml --ignore-missing-imports
```

## What's New in v0.3.3

- ✅ **Phase 9 Workflow Integration**: Complete notebook implementation with Phase 9.3 (Advanced Feature Engineering),
  9.6 (Model Evaluation), and 9.7 (Stock Valuation) integration
- ✅ **Notebook Quality Improvements**: 12 comprehensive tests covering config API, type validation, and error handling
  with ≥80% coverage
- ✅ **Notebook Reorganization**: Improved workflow order with Phase 9.2 (Enhanced EDA) correctly positioned,
  consolidated imports, and flattened error handling
- ✅ **Configuration API Enhancement**: Added `output_dir` parameter to `load_config()` for immutable configuration (
  eliminates config mutation anti-pattern)
- ✅ **Type Safety**: Explicit `isinstance()` checks, improved error messages with actionable guidance, separated
  validation concerns
- ✅ **Enhanced Logging**: Added `exc_info=True` to logger calls for full stack traces
- ✅ **Automated Tools**: Notebook reorganization script (`implement_notebook_integration.py`)

### v0.3.2 Highlights

- Enhanced EDA with distribution analysis, outlier detection (IQR), normality tests (Shapiro-Wilk, Anderson-Darling)
- Correlation matrices with Pearson and Spearman methods
- Sector-wise statistical summaries

### v0.3.1 Highlights

- Phase 9.6 (Model Evaluation) and Phase 9.7 (Stock Valuation) complete implementations
- Integration validation script
- Notebook integration with `NotebookConfig`

### v0.3.0 Highlights

- Configuration Management with centralized config via environment, JSON, or YAML
- CLI Tools: Three console commands (`finance-ml`, `finance-ml-analyze`, `finance-ml-validate`)
- Modern Packaging: `pyproject.toml` with optional dependency groups
- CI/CD: GitHub Actions workflow for automated testing
- CHANGELOG.md for version tracking

## Development Roadmap

### Phase 9: Advanced Stock Prediction ML System (Planned)

A comprehensive development plan for a sophisticated stock prediction system is detailed
in [IMPROVEMENT_PLAN.md Phase 9](IMPROVEMENT_PLAN.md#phase-9--advanced-stock-prediction-ml-system-planned).

**Key Enhancements Planned**:

- **Sector-Specific Features**: Tailored feature engineering for Financials, Energy, Technology, Healthcare, and
  Consumer sectors
- **Advanced Classification**: Multi-class event classification with neural networks and gradient boosting ensembles
- **Enhanced Regression**: Classification-enhanced sector-optimized models with hyperparameter optimization
- **Comprehensive Analytics**: Predicted vs. Analyst Target comparison with detailed error analysis
- **Interactive Dashboards**: Real-time monitoring with Plotly Dash/Streamlit
- **Model Interpretation**: SHAP and LIME integration for explainability
- **Automated Reporting**: Excel and PDF report generation matching institutional formats

**Framework Alignment**: Phase 9 aligns with the standard 8-step ML project checklist for production-ready systems.

See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for complete version history and detailed roadmap.

## SQLite Local Setup and Data Load

If you prefer a lightweight local database for quick testing, you can use SQLite alongside or instead of PostgreSQL.

1) Create the SQLite schema (from project root):

- sqlite3 equities.sqlite ".read create_equities_schema_sqlite.sql"

2) Import CSVs into SQLite using the CLI script (recommended):

- sqlite3 equities.sqlite ".read import_equities_data_sqlite.sql"

This script:

- Uses per‑region staging tables
- Deletes header rows if they are imported as data
- Backfills missing Region values per file (US/EU/APAC/ROTW)
- Inserts with INSERT OR IGNORE honoring the UNIQUE("Ticker","Region") index
- Prints a post‑import validation summary

3) Python alternative: chunked importer (useful for very large CSVs):

- python tools/import_sqlite.py --db equities.sqlite --data-dir data --chunksize 2000
- python tools/import_sqlite.py --db equities.sqlite --regions US,EU

Notes:

- The Python importer uses pandas to read CSVs with dtype=str to preserve raw values, converts empty strings to NULL,
  and backfills Region per file.
- sqlite3 is included with Python; no additional dependency is required for SQLite.
