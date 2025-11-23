---
apply: always
---

# Finance ML Analytics Platform — Project Rules

**Version**: 0.8.3  
**Last Updated**: 2025-11-23

## Project Overview

Finance ML Analytics Platform is a professional, modular Python package for equity screening, feature engineering, and machine learning models across global regions (US, EU, APAC, ROTW).

**Key capabilities**:
- Unified data pipeline with PostgreSQL integration + CSV fallback
- Modular Python package (`finance_ml`) with clean, tested, reusable code
- Interactive Jupyter notebook for exploration and prototyping
- CLI tools for batch processing and automation
- Comprehensive unit tests with good coverage (≈85 test modules, ≥80% target)
- Configuration management via environment variables, JSON, or YAML

### Module Architecture (v9_8 - Phase 9.1-9.8)

The platform follows a **phase-aligned architecture** with dedicated subpackages:

```
finance_ml/ml_workflow/
├── preprocessing/      # Phase 9.1: 6-step imputation, outliers, scaling, quality
├── eda/               # Phase 9.2: EDA, benchmarking, reports
├── features/          # Phase 9.3: Core, advanced, selection, API
├── classification/    # Phase 9.4: Labels, tuning, models, evaluation
├── regression/        # Phase 9.5: Models, constraints, quantile, tuning, dataset, io
├── evaluation/        # Phase 9.6: Metrics, analysis
├── analytics/         # Phase 9.7: Mispricing, analyst comparison, portfolio, risk
└── reporting/         # Phase 9.8: Dashboard data, export
```

**8-Phase ML Workflow:**

1. **Phase 9.1**: Loading and preprocessing with 6-step imputation strategy
2. **Phase 9.2**: Enhanced EDA with statistical testing and benchmarking
3. **Phase 9.3**: Advanced feature engineering with sector-specific optimizations
4. **Phase 9.4**: Multi-class event classification using neural networks and ensembles
5. **Phase 9.5**: Sector-optimized regression models with hyperparameter tuning
6. **Phase 9.6**: Model evaluation and comprehensive error analysis
7. **Phase 9.7**: Identification of under/overvalued stocks with visualization
8. **Phase 9.8**: Comprehensive analytics and reporting

## Technology Stack

### Language and Runtime

- **Language**: Python 3.12–3.14 (3.13–3.14 recommended)
- **Notebook Environment**: Jupyter Notebook or JupyterLab
- **Database**: PostgreSQL 15+ (local instance)

### Package Managers
- **Primary**: pip + venv (see `requirements.txt`)
- **Alternative**: pipenv (see `Pipfile` aligned with Python 3.11)

### Core Dependencies
**Data Science**:

- numpy>=1.26.0,<2.0.0
- pandas>=2.0.0,<3.0.0
- scipy>=1.11.0,<2.0.0
- statsmodels>=0.14.0,<1.0.0

**Machine Learning**:

- scikit-learn>=1.4.0,<2.0.0
- imbalanced-learn>=0.11.0,<1.0.0
- tensorflow>=2.13.0,<3.0.0
- scikeras>=0.12.0,<1.0.0

**Gradient Boosting**:

- xgboost>=2.0.3,<3.0.0
- lightgbm>=4.0.0,<5.0.0
- catboost>=1.2.0,<2.0.0

**Feature Engineering & Explainability**:
- boruta>=0.3.0
- shap>=0.42.0,<1.0.0

**Visualization**:
- matplotlib>=3.7.0,<4.0.0
- seaborn>=0.12.0,<1.0.0
- plotly>=5.14.0,<6.0.0

**Utilities**:
- tqdm>=4.65.0,<5.0.0
- joblib>=1.3.0,<2.0.0
- numba>=0.57.0,<1.0.0
- Pillow>=10.0.0,<11.0.0
- xlsxwriter>=3.1.0,<4.0.0
- psutil>=5.9.0,<6.0.0

**Optional (for database access)**:
- psycopg2-binary (not in requirements.txt; install separately if needed)
- SQLAlchemy (not in requirements.txt; install separately if needed)

### Testing Framework
- **Built-in**: unittest (Python standard library)
- **Test Runner**: `python -m unittest -v`
- **Coverage**: `.coverage` file indicates coverage tracking is in use


## Requirements and Prerequisites

**Operating System**:
- Windows 10/11 (tested)
- macOS
- Linux

**Software**:

- Python 3.12 or 3.13 (use pyenv or official installer; avoid mixing Conda with venv)
- PostgreSQL 15+ (local instance)
- Git (optional but recommended)
- psql command-line tool (included with PostgreSQL; must be on PATH)

**Hardware**:
- TensorFlow works with CPU-only; GPU acceleration optional (CUDA/cuDNN setup required)


## Quick Start Setup

### 1. Virtual Environment
**Windows (PowerShell)**:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**macOS/Linux (bash)**:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2. Optional Database Client Libraries
If you need to query PostgreSQL from Python:
```bash
pip install psycopg2-binary SQLAlchemy
```

### 3. PostgreSQL Setup
Create the equities table:
```powershell
# Windows (PowerShell)
psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql
```

### 4. Data Import
Load regional CSVs from `data/` into PostgreSQL:
```powershell
# Windows (PowerShell)
psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
```

Optional: Validate CSV data quality before import:
```powershell
python validate_csv_import.py
```

### 5. Run Tests
```bash
python -m unittest -v
```

### 6. Start Working
**Notebook** (primary workflow):
```bash
jupyter notebook
# Open ml_finance_model_main.ipynb and run cells in order
```

**CLI**:
```bash
# Quick analysis
finance-ml-analyze --data-source auto --limit 1000 -v

# Full pipeline
finance-ml --data-source auto --limit 5000 --out-dir outputs
```


## Entry Points

### 1. Interactive Notebook (Primary)

**File**: `ml_finance_model_main.ipynb`
- Main Jupyter notebook for exploration, EDA, feature engineering, and modeling
- Imports from `finance_ml` package (modular design)
- Run cells in order
- Outputs: model diagnostics, ranking tables, optional CSV/Excel exports

### 2. Python Script (Legacy)

**File**: `ml_finance_model_main.py`
- Command-line interface for batch processing
- Now uses `finance_ml` package internally (refactored in Phase 7)

**Usage**:
```bash
python ml_finance_model_main.py --data-source auto --limit 5000 --out-dir outputs
```

**Options**:
- `--data-source {auto|csv|db}` — Data source selection (default: auto)
- `--db-url <url>` — Database connection string (or use DB_URL env var)
- `--limit <n>` — Limit rows for testing
- `--out-dir <path>` — Output directory (default: outputs)
- `--dry-run` — Skip model training

### 3. CLI Tools (Modern)
Three console commands installed via `pyproject.toml` entry points:

**a) `finance-ml`** — Main analysis pipeline
```bash
finance-ml --data-source auto --limit 5000
finance-ml --data-source db --db-url postgresql+psycopg2://postgres:@localhost:5432/postgres
finance-ml --config config.json
```

**b) `finance-ml-analyze`** — Quick EDA and data profiling
```bash
finance-ml-analyze --data-source csv --data-dir ./data
finance-ml-analyze --data-source db --limit 1000 -v
```

**c) `finance-ml-validate`** — Data validation and quality checks
```bash
finance-ml-validate --data-source csv --data-dir ./data
finance-ml-validate --data-source db -v
```


## Available Scripts and Utilities

### Database Setup Scripts
- **`create_equities_schema.sql`** — SQL script to initialize the equities table in PostgreSQL
  - Creates table with 234 columns (many with spaces/punctuation; always use double quotes in SQL)
  - Sets ownership to `postgres` user
  - Run via: `psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql`

- **`import_equities_data.sql`** — Comprehensive import script for all regional CSVs
  - Handles US, EU, APAC, ROTW regions with staging tables
  - Proper NULL handling (`NULL ''`), encoding (`ENCODING 'UTF8'`), and validation
  - Run via: `psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql`

### Data Validation Scripts
- **`validate_csv_import.py`** — CSV data quality validator
  - Validates schema, checks for missing values, identifies data quality issues
  - Run before importing data to catch problems early

### Environment Setup Scripts
- **`setup_environment.py`** — Automated environment setup utility
  - Checks prerequisites (Python, PostgreSQL, Git)
  - Creates virtual environment
  - Installs dependencies
  - Sets up database and loads CSV data
  - Configures environment variables
  - Runs tests
  - Provides activation instructions

### Notebook Utilities
- **`analyze_notebook.py`** — Notebook structure analyzer
  - Counts cells, previews content, searches for functions
  
- **`update_notebook.py`** — Notebook synchronizer
  - Extracts TDD functions from .py files and inserts into .ipynb
  
- **`verify_notebook.py`** — Notebook verification utility
  - Checks for presence of TDD functions

- **`refactor_notebook.py`** — Notebook refactoring tool
  - TODO: Document purpose and usage

- **`update_notebook_imports.py`** — Import updater
  - TODO: Document purpose and usage


## Environment Variables

See `environment_variables.txt` for complete documentation.

### Core Configuration
```bash
# Logging Configuration
TF_CPP_MIN_LOG_LEVEL=2          # TensorFlow log level: 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

# Directory Paths
DATA_DIR=data                    # Data directory
MODEL_DIR=regression                 # Saved regression directory
CACHE_DIR=.cache                 # Cache directory
OUTPUT_DIR=outputs               # Output directory

# Database Configuration
DB_URL=postgresql+psycopg2://postgres:@localhost:5432/postgres

# Model Configuration
MODEL_VERSION=v0_5_0             # Model version identifier
RANDOM_SEED=42                   # Random seed for reproducibility

# Performance Settings
N_JOBS=-1                        # Number of parallel jobs (-1=all cores)
MEMORY_LIMIT=8GB                 # Maximum memory allocation

# Logging
LOG_LEVEL=INFO                   # Logging level
```

### Setting Environment Variables

**Windows (PowerShell)**:
```powershell
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:DB_URL = "postgresql+psycopg2://postgres:@localhost:5432/postgres"
```

**macOS/Linux (bash)**:
```bash
export TF_CPP_MIN_LOG_LEVEL=2
export DB_URL=postgresql+psycopg2://postgres:@localhost:5432/postgres
```

**Using .env file**:
Create a `.env` file in the project root with the variables above. Tools that support `.env` auto-loading will pick them up.


## Database Configuration

### Connection Details
- **JDBC URL**: `jdbc:postgresql://localhost:5432/postgres`
- **Driver**: `org.postgresql.Driver` (version 42.7.x)
- **User**: `postgres`
- **Database**: `postgres`
- **Schema**: `public`
- **Table**: `equities`

### Important Notes
- Many column names contain spaces and special characters
- **Always use double quotes** when referencing column names in SQL
- Example: `SELECT "Ticker", "Last Price", "Market Cap" FROM equities WHERE "Region" = 'US'`

### Data Import Critical Parameters
- **`NULL ''`** — Treats empty strings as NULL values (essential to avoid import errors)
- **`ENCODING 'UTF8'`** — Ensures proper character handling
- **`HEADER true`** — Skips the CSV header row

### Regional Data Files
- `data/screening_us.csv` — US equities
- `data/screening_eu.csv` — European equities
- `data/screening_apac.csv` — Asia-Pacific equities
- `data/screening_rotw.csv` — Rest of World equities


## Testing

### Test Framework
- **Framework**: Python's built-in `unittest`
- **Location**: `tests/` directory
- **Count**: 85 test modules
- **Coverage**: Tracked in `.coverage` file (≥80% target for new code)

### Running Tests
```bash
# Run all tests with verbose output
python -m unittest -v

# Run specific test module
python -m unittest tests.test_finance_ml_data

# Run with pytest (if installed)
pytest tests/ -v --cov=finance_ml

# Discover and run all tests
python -m unittest discover -s tests -v
```

### Test Suite Modules (85 total)

- **`test_advanced_eda.py`** — Advanced EDA functions (correlation, PCA, statistical tests)
- **`test_advanced_features.py`** — Phase 9.3 advanced feature engineering tests
- **`test_advanced_models_phase95.py`** — Phase 9.5 advanced regression models and ensembles
- **`test_advanced_preprocessing.py`** — Advanced preprocessing (outlier detection, winsorization, imputation)
- **`test_analyst_comparison.py`** — Analyst comparison and target analysis tests
- **`test_analytics.py`** — Analytics and stock ranking tests
- **`test_benchmarking.py`** — Benchmarking and peer comparison tests
- **`test_build_features.py`** — Feature building pipeline
- **`test_classification.py`** — Event classification model tests
- **`test_classification_phase94.py`** — Phase 9.4 advanced classification models
- **`test_classification_evaluation.py`** — Classification evaluation module tests
- **`test_classification_models.py`** — Classification models module tests
- **`test_classification_phase943.py`** — Phase 9.4.3 classification tests
- **`test_cli.py`** — Command-line interface tests
- **`test_coverage_smoke.py`** — Smoke test for coverage validation
- **`test_data_catalog.py`** — Data catalog management tests
- **`test_data_quality.py`** — Data validation and quality checks
- **`test_data_splits_policy.py`** — Data split leakage prevention policy validation (code_guidelines.md v1.4)
- **`test_data_types_detection.py`** — Schema-aware datatype detection and Phase 9.3 validation (9 tests, TDD v0.8.2)
- **`test_data_versioning.py`** — Data versioning and metadata tests
- **`test_eda.py`** — Exploratory data analysis utilities
- **`test_enhanced_eda_phase92.py`** — Phase 9.2 enhanced EDA tests
- **`test_enhanced_imputation.py`** — Phase 9.16-step imputation strategy tests
- **`test_enhanced_imputation_phase93.py`** — Phase 9.3 enhanced imputation with schema alignment (8 tests, TDD v0.8.2)
- **`test_evaluation_phase96.py`** — Phase 9.6 model evaluation and error analysis tests
- **`test_features.py`** — Feature engineering functions
- **`test_finance_ml_config.py`** — Configuration management tests
- **`test_finance_ml_data.py`** — Data loading module tests
- **`test_finance_ml_eval.py`** — Evaluation and analytics module tests
- **`test_finance_ml_features.py`** — Features module tests
- **`test_finance_ml_models.py`** — Models module tests
- **`test_imputation_6step.py`** — 6-step imputation strategy tests
- **`test_improvement_plan_revision.py`** — Development plan validation
- **`test_integration_cli_pipeline.py`** — CLI pipeline integration tests
- **`test_integration_notebook_pipeline.py`** — Notebook pipeline integration tests
- **`test_integration_production_scenarios.py`** — Production scenario integration tests
- **`test_loaders.py`** — CSV and database loading functions
- **`test_logging.py`** — Logging configuration tests
- **`test_metadata_catalog_quality.py`** — Metadata and quality stats validation (4 tests, TDD v0.8.2)
- **`test_model_evaluation_advanced.py`** — Advanced model evaluation tests
- **`test_notebook_config.py`** — Notebook configuration tests
- **`test_notebook_enhancements.py`** — Notebook enhancements validation
- **`test_notebook_integration.py`** — Notebook integration tests
- **`test_notebook_quality_improvements.py`** — Notebook quality improvements tests
- **`test_onehot_encoding.py`** — One-hot encoding functionality tests
- **`test_outlier_safety_rails.py`** — Outlier safety rails (winsorization, clipping, non-negativity) (
  code_guidelines.md v1.4)
- **`test_phase91_enhancements.py`** — Phase 9.1 enhancements tests
- **`test_phase93_enhancements.py`** — Phase 9.3 enhancements tests
- **`test_phase95_error_handling.py`** — Phase 9.5 error handling tests
- **`test_phase95_nonnegative_predictions.py`** — Phase 9.5 non-negative prediction constraint tests
- **`test_phase95_quick.py`** — Phase 9.5 quick validation tests
- **`test_portfolio_backtesting.py`** — Portfolio backtesting framework (3 tests, Portfolio Phase 5)
- **`test_portfolio_dashboards.py`** — Portfolio interactive dashboards (3 tests, Portfolio Phase 6)
- **`test_portfolio_ml_prediction.py`** — ML-based return prediction and stock selection (9 tests, Portfolio Phases 1-2)
- **`test_portfolio_optimization.py`** — Portfolio optimization tests
- **`test_portfolio_optimization_advanced.py`** — Advanced optimization methods (4 tests, Portfolio Phase 3)
- **`test_portfolio_risk_management.py`** — Risk management enhancements (4 tests, Portfolio Phase 4)
- **`test_predictions_schema.py`** — Standardized predictions schema validation (code_guidelines.md v1.4)
- **`test_preprocess_and_training.py`** — Preprocessing and training workflows
- **`test_quantile_fix.py`** — Quantile regression fixes tests
- **`test_regression.py`** — Regression model evaluation
- **`test_regression_sector_metrics.py`** — Sector-level metrics persistence validation (code_guidelines.md v1.4)
- **`test_repository_setup.py`** — Validates repository basics (files, SQL schema, environment config)
- **`test_risk_metrics.py`** — Risk metrics calculation tests
- **`test_sector_bias_calibration.py`** — Sector-specific bias calibration (code_guidelines.md v1.4)
- **`test_setup_environment.py`** — Setup script validation
- **`test_simple_eda_stringdtype.py`** — StringDtype compatibility validation (3 tests, TDD v0.8.2)
- **`test_sql_scripts.py`** — SQL script validation tests
- **`test_sqlite_import.py`** — SQLite import functionality (header removal, NULL handling, region backfilling)
- **`test_stacking_default.py`** — Stacking ensemble default configuration (code_guidelines.md v1.4)
- **`test_uncertainty_calibration.py`** — Uncertainty quantification with conformal prediction (code_guidelines.md v1.4)
- **`test_validate_csv_import.py`** — CSV validation (schema validation, data quality checks)
- **`test_validation_regex.py`** — Regex validation and pattern matching tests
- **`test_valuation_phase97.py`** — Phase 9.7 stock valuation and identification tests
- **`test_visualizations.py`** — Visualization functions tests

**Note**: The test suite has grown to 85 modules (74 original + 4 TDD v0.8.2 + 5 Portfolio Optimization + 2 advanced
evaluation reporting).
Full test suite can take significant time. See guidelines.md for selective test execution strategies (fast/medium/slow
categories).

**Recent Additions (v0.8.2, 2025-11-19):**

- **TDD Implementation (4 modules, 24 tests):** Schema-aware datatype detection, Phase 9.3 enhanced imputation,
  metadata catalog validation, StringDtype compatibility
- **Portfolio Optimization (5 modules, 23 tests):** ML-based return prediction, advanced optimization methods
  (Black-Litterman, Risk Parity, HRP), risk management enhancements, backtesting framework, interactive dashboards

### Writing New Tests
- Create files under `tests/` named `test_*.py` with `unittest.TestCase` classes
- Keep tests isolated from external services
- Prefer testing pure functions and small utilities
- For DB-related code, stub or mock the connection
- Use small, deterministic samples; avoid loading full CSVs unless necessary


## Project Structure

```
Finance_ML_Analytics_Platform/
├── finance_ml/                     # Main Python package (v0.8.2)
│   ├── __init__.py                # Package exports and version
│   ├── ml_workflow/               # Phase-aligned modular workflow (v9_8)
│   │   ├── data/                  # NEW v0.8.2: schema.py (column schema registry)
│   │   ├── preprocessing/         # Phase 9.1: imputation, outliers, scaling, quality, pipeline, dtypes (NEW v0.8.2)
│   │   ├── eda/                   # Phase 9.2: eda, benchmarking, reports
│   │   ├── features/              # Phase 9.3: core, advanced, selection, api
│   │   ├── classification/        # Phase 9.4: labels, tuning, models, evaluation
│   │   ├── regression/            # Phase 9.5: models, constraints, quantile, tuning, dataset, io
│   │   ├── evaluation/            # Phase 9.6: metrics, analysis
│   │   ├── analytics/             # Phase 9.7: mispricing, analyst_comparison, portfolio, risk, eval,
│   │   │                          #            stock_selection (NEW v0.8.2), ml_returns (NEW v0.8.2), attribution (NEW v0.8.2)
│   │   └── reporting/             # Phase 9.8: dashboard_data, export
│   ├── advanced_eda.py            # Advanced EDA with statistical analysis
│   ├── advanced_features.py       # Advanced feature engineering (deprecated, use ml_workflow.features)
│   ├── advanced_models.py         # Sector-optimized regression models (deprecated, use ml_workflow.regression)
│   ├── advanced_preprocessing.py  # Advanced preprocessing (deprecated, use ml_workflow.preprocessing)
│   ├── benchmarking.py            # Comparative analysis and peer benchmarking
│   ├── classification.py          # Multi-class event classification (deprecated, use ml_workflow.classification)
│   ├── classification_enhanced.py # Enhanced classification (Phase 9.4)
│   ├── cli.py                     # Command-line interface
│   ├── config.py                  # Configuration management
│   ├── data.py                    # Data loading, normalization, validation
│   ├── data_catalog.py            # Centralized data asset registry (Phase 9.1)
│   ├── data_versioning.py         # Dataset version tracking and metadata (Phase 9.1)
│   ├── eval.py                    # Analytics, visualizations, reporting
│   ├── features.py                # Feature engineering functions (deprecated, use ml_workflow.features)
│   ├── logging_config.py          # Logging configuration utilities
│   ├── models.py                  # ML models (deprecated, use ml_workflow.regression/classification)
│   ├── notebook_config.py         # Notebook-specific helpers and config
│   ├── notebook_utils.py          # Notebook utility functions
│   ├── portfolio_optimization.py  # Portfolio optimization utilities
│   ├── risk_metrics.py            # Risk metrics and portfolio risk analysis
│   ├── transformers.py            # Scikit-learn compatible feature transformers
│   └── dashboards/                # Interactive dashboards
│       ├── dash_app.py            # Dash application
│       ├── streamlit_app.py       # Streamlit application
│       └── portfolio_widgets.py   # Portfolio interactive widgets (NEW v0.8.2)
│
├── tests/                         # Unit tests (83 test modules)
│   ├── test_finance_ml_data.py
│   ├── test_finance_ml_features.py
│   ├── test_finance_ml_models.py
│   ├── test_finance_ml_eval.py
│   ├── test_repository_setup.py
│   └── ... (see Test Suite Modules section for complete list)
│
├── data/                          # Regional equity data (CSV files)
│   ├── screening_us.csv
│   ├── screening_eu.csv
│   ├── screening_apac.csv
│   └── screening_rotw.csv
│
├── .github/workflows/             # CI/CD pipelines
│   └── tests.yml                  # Automated testing workflow
│
├── .aiassistant/rules/            # AI assistant rules
│   └── promt_rules.md             # This file
│
├── .junie/                        # Junie AI guidelines
│   └── guidelines.md              # Development guidelines
│
├── outputs/                       # Output artifacts
│   ├── eda_summary.json
│   ├── regression_predictions.csv
│   └── ... (generated files)
│
├── tools/                         # Utility scripts (34 tools)
│   ├── __init__.py
│   ├── setup_environment.py       # Automated environment setup
│   ├── validate_csv_import.py     # CSV data quality validator
│   ├── import_sqlite.py           # Chunked CSV→SQLite importer
│   ├── load_equities_data.py      # Legacy PostgreSQL CSV importer
│   ├── analyze_notebook.py        # Notebook structure analyzer
│   ├── refactor_notebook.py       # Notebook refactoring helper
│   ├── update_notebook.py         # Notebook synchronizer
│   ├── update_notebook_imports.py # Import updater
│   ├── verify_notebook.py         # Notebook verification utility
│   ├── verify_preprocessing_improvements.py # Verify preprocessing improvements
│   ├── validate_phase95_predictions.py     # Phase 9.5 prediction validation
│   ├── validate_phase9_integration.py      # Phase 9 integration validator
│   ├── apply_improvement_plan_updates.py   # Development plan automation
│   ├── apply_notebook_fixes.py    # Notebook maintenance utility
│   └── ... (20+ additional notebook and validation tools)
│
├── ml_finance_model_main.ipynb   # Interactive Jupyter notebook
├── equities_data_explorer.ipynb  # Data exploration and EDA notebook
├── ml_finance_model_main.py      # Script version (uses finance_ml package)
│
├── create_equities_schema.sql    # PostgreSQL schema setup
├── create_equities_schema_sqlite.sql # SQLite schema setup
├── import_equities_data.sql      # Data import script (PostgreSQL)
├── import_equities_data_sqlite.sql   # SQLite data import script
├── equities.sqlite               # Example SQLite database
├── identifier.sqlite             # Auxiliary SQLite database
│
├── pyproject.toml                # Modern Python packaging configuration
├── setup.py                      # Backward-compatible setup
├── requirements.txt              # Core dependencies
├── Pipfile                       # Pipenv dependencies (Python 3.11)
├── environment_variables.txt     # Environment configuration examples
│
├── CHANGELOG.md                  # Version history
├── README.md                     # Project documentation
├── IMPROVEMENT_PLAN.md           # Development roadmap (8 phases)
├── REFACTORING_COMPLETE.md       # Refactoring documentation
└── TDD_IMPLEMENTATION_COMPLETE.md # TDD implementation summary
```

### Package Modules (`finance_ml`)

#### `finance_ml.data`
Data loading, normalization, and validation functions.
- `load_from_csv()`, `load_from_db()`: Multi-source data loading
- `preprocess()`: Data cleaning and normalization
- `normalize_columns()`: Column name normalization
- `validate_schema()`, `check_missing_values()`: Quality checks
- `detect_outliers_iqr()`, `validate_numeric_ranges()`: Outlier detection
- `infer_region_from_filename()`: Region inference from CSV filenames

#### `finance_ml.features`
Feature engineering for financial data.
- `engineer_basic_ratios()`: EV/EBITDA, P/E, P/B ratios
- `engineer_margin_features()`: Gross, operating, net margins
- `engineer_volatility_features()`: Price volatility windows
- `engineer_revenue_cagr()`: Revenue growth metrics
- `build_features_and_target()`: Complete feature pipeline

#### `finance_ml.models`
Machine learning models for classification and regression.
- `create_event_labels()`: Event label creation
- `train_event_classifier()`: Multi-class event classification
- `train_and_evaluate_regression()`: Baseline regression models
- `train_and_evaluate_regression_by_sector()`: Sector-optimized models
- `train_quantile_regression()`: Uncertainty estimation
- `predict_quantile_regression()`: Quantile predictions
- `train_quantile_regression_by_sector()`: Sector-specific quantile models
- `train_stacking_ensemble()`: Meta-learner stacking
- `train_stacking_ensemble_by_sector()`: Sector-specific ensembles
- `build_regression_pipeline()`: Regression pipeline builder

#### `finance_ml.eval`
Analytics, visualizations, and reporting.
- `simple_eda()`: Exploratory data analysis
- `calculate_mispricing_score()`: Valuation analysis
- `rank_undervalued_stocks()`, `rank_overvalued_stocks()`: Stock ranking
- `rank_stocks_by_sector()`: Sector-based ranking
- `create_sector_heatmap()`: Sector performance heatmap
- `create_interactive_prediction_plot()`: Interactive visualizations
- `create_region_sector_heatmap()`: Region-sector analysis
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


## Configuration Management

### Configuration Sources (Priority Order)
1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration file (JSON/YAML)
4. Default values (lowest priority)

### Configuration File Formats

**config.json**:
```json
{
  "data_dir": "data",
  "output_dir": "outputs",
  "db_url": "postgresql+psycopg2://postgres:@localhost:5432/postgres",
  "model_version": "v0_5_0",
  "random_seed": 42,
  "n_jobs": -1
}
```

**config.yaml**:
```yaml
data_dir: data
output_dir: outputs
db_url: postgresql+psycopg2://postgres:@localhost:5432/postgres
model_version: v0_5_0
random_seed: 42
n_jobs: -1
```

### Loading Config in Python
```python
from finance_ml import load_config

config = load_config("config.json")
config.apply_to_env()
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Finance ML Analytics Platform Contributors


## Development Guidelines

See `.junie/guidelines.md` for comprehensive development guidelines covering:
- Build and configuration instructions
- Data pipeline aligned to all_stocks dataframe
- Advanced feature engineering with sector-specific optimizations
- Multi-class classification of financial events
- Sector-optimized regression models
- Model evaluation and error analysis
- Identification of under/overvalued stocks
- Comprehensive analytics of prediction results
- Testing information
- Code style and quality standards
- Troubleshooting tips

See `docs/code_guidelines.md` v1.4 (updated 2025-11-23) for detailed coding standards:

- Standardized function signatures and return types
- Column naming schema and dataframe conventions
- Highlights:
  - Uncertainty and Prediction Intervals (quantile regression + conformal calibration)
  - Outlier Safety Rails Policy (winsorization, robust loss, clipping, non-negativity)
  - Data Split and Leakage Policy (time-series → grouped → stratified)
  - Standardized Predictions Schema (required columns and invariants)
  - Sector Metrics and Calibration (persistence contract, bias correction)
  - TDD Conventions and Selective Test Execution
  - NEW in v1.4: Notebook Best Practices and TDD Conventions (Section 8)
    - Centralized Configuration Constants (single source of truth)
    - DataFrame Stage Naming (8-stage pipeline):
      all_stocks_raw → all_stocks_normalized → all_stocks_typed → all_stocks_winsorized →
      all_stocks_imputed → all_stocks_scaled → all_stocks_features → all_stocks_enhanced
    - Magic Numbers Policy (replace meaningful literals with named constants)

Portfolio Optimization Workflow

- See portfolio optimization guidelines and examples in `docs/code_guidelines.md` (v1.4) and the enhancement plan in
  docs/improvement_plan. The rules align with Phase 6 dashboard, Phase 5 backtesting, and optimization methods (
  Black-Litterman, Risk Parity, HRP).

See `IMPROVEMENT_PLAN.md` for phased development roadmap (8 phases):
1. Foundations
2. Data Ingestion/Validation
3. EDA/Feature Engineering
4. Classification Models
5. Regression Models
6. Analytics/Reporting
7. Testing/CI
8. Packaging/Modularity


## Troubleshooting

### TensorFlow Installation Issues
- Project primarily uses scikit-learn and gradient boosting libraries
- CPU-only TensorFlow is fine for this project
- If installation is problematic, you can temporarily comment out `tensorflow` in requirements.txt
- Ensure Visual C++ Redistributable is installed on Windows
- Set `TF_CPP_MIN_LOG_LEVEL=2` to reduce log verbosity

### PostgreSQL Issues
- **Column quoting**: Many columns contain spaces/punctuation; always use double quotes in SQL
- **Import errors**: Use `\copy` (client-side) from psql to avoid server-side file permission issues on Windows
- **NULL handling**: Always use `NULL ''` parameter in COPY/\copy commands to treat empty strings as NULL

### Data Import Issues
- **"relation equities does not exist"**: Run `create_equities_schema.sql` first
- **"invalid input syntax for type numeric"**: Check CSV for invalid numeric values; `NULL ''` should handle empty strings
- **"could not open file for reading"**: Ensure you're running psql from the project root directory, or use absolute paths
- **Permission denied errors**: Use `\copy` (client-side) instead of `COPY` (server-side)


## TODOs and Open Items

### Documentation
- [ ] Document purpose and usage of `refactor_notebook.py`
- [ ] Document purpose and usage of `update_notebook_imports.py`
- [ ] Create detailed `CONTRIBUTING.md` file with comprehensive contribution guidelines
- [ ] Update GitHub username/organization in badge URLs in README.md

### Nice to Have
- [ ] Document recommended IDE setup (DataGrip/PyCharm configuration)
- [ ] Add examples of config file usage in practice
- [ ] Document performance benchmarks for different data sizes
- [ ] Add troubleshooting section for common notebook issues


## Version History

- **v0.8.3** (Current, 2025-11-23) — Section 8 Notebook Best Practices added to code_guidelines.md v1.4 (centralized
  configuration constants, 8-stage DataFrame naming, magic numbers policy); Portfolio optimization guidelines aligned;
  test suite expanded to 85 modules; documentation synchronized across README.md, guidelines.md, and promt_rules.md.
- **v0.8.2** — TDD implementation (schema/dtypes modules, 24 tests), Phase 9.3 feature
  enhancements (Schema 1.3 with 310 columns, 13 feature categories), Portfolio Optimization (6 phases complete:
  stock selection, ML return prediction, advanced optimization methods [Black-Litterman, Risk Parity, HRP],
  risk management, backtesting, dashboards; 23 tests), 83 test modules total (74 original + 4 TDD + 5 Portfolio)
- **v0.6.1** — Phase 9.5 classification meta-features, enhanced imputation (6-step), modular workflow
  refactor (v9_8), 74 test modules (including 7 new TDD modules for code_guidelines.md v1.4 standards)
- **v0.6.0** — Phase 9.5 enhanced classification module, comprehensive data flow fixes, dashboard enhancements
- **v0.5.1** — Phase 9.1 6-step imputation pipeline, notebook integration, comprehensive testing
- **v0.5.0** — Complete Phase 9 implementation, 20 package modules, comprehensive ML pipeline
- **v0.4.0** — Phase 9.7 valuation analysis, Phase 9.1 data versioning and catalog
- **v0.3.0** — Modular package, CLI tools, configuration management, CI/CD
- Earlier versions documented in `CHANGELOG.md`

---

*This file is automatically applied by AI assistants to understand project structure and conventions.*
