# Finance ML Analytics Platform

**Version 0.3.0** — A professional, modular Python package for equity screening, feature engineering, and machine learning models across global regions.

[![Tests](https://github.com/yourusername/Finance_ML_Analytics_Platform/workflows/Tests/badge.svg)](https://github.com/yourusername/Finance_ML_Analytics_Platform/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Finance ML Analytics Platform is a comprehensive toolkit for quantitative equity analysis combining:
- **Unified data pipeline**: PostgreSQL integration + CSV fallback for multi-region equity data (US, EU, APAC, ROTW)
- **Modular Python package** (`finance_ml`): Clean, tested, reusable code for data loading, feature engineering, modeling, and analytics
- **Interactive notebook**: Jupyter-based workflow for exploration and prototyping
- **CLI tools**: Command-line interface for batch processing and automation
- **Production-ready**: Modern packaging, comprehensive tests, CI/CD, configuration management

### Key Features
- 📊 **Data Management**: Load from PostgreSQL or CSV, with validation and quality checks
- 🔧 **Feature Engineering**: Financial ratios, margins, volatility, revenue CAGR, and more
- 🤖 **ML Models**: Event classification, sector-optimized regression, quantile models, stacking ensembles
- 📈 **Analytics**: Mispricing scores, stock ranking, interactive visualizations
- ⚙️ **Configuration**: Flexible config via environment variables, JSON, or YAML
- 🧪 **Tested**: 144+ unit tests with comprehensive coverage
- 🚀 **CLI**: Three command-line tools for different workflows


## Tech Stack
- Language: Python (3.10–3.11 recommended)
- Notebook: Jupyter (notebook or JupyterLab)
- Database: PostgreSQL (15+ recommended)
- ML/Datascience libraries: numpy, pandas, scikit‑learn, xgboost, lightgbm, catboost, shap, etc. (see requirements.txt)

Notes on environment managers:
- Primary dependency spec is requirements.txt (pip + venv).
- A Pipfile is also provided for pipenv users, aligned with Python 3.11 and containing all packages from requirements.txt.


## Requirements
- OS: Windows 10/11 (tested), macOS, or Linux
- Python: 3.10 or 3.11
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
- pip install -r requirements.txt

Optional database client libraries for Python (only if you plan to query Postgres directly from Python):
- pip install psycopg2-binary SQLAlchemy


## PostgreSQL Setup and Data Load
1) Ensure PostgreSQL is installed and running locally, and psql is on your PATH.

2) Create the equities table by running the SQL script in the project root:
- Windows (PowerShell):
  - psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql

The script creates the equities table and assigns ownership to postgres (ALTER TABLE equities OWNER TO postgres;). Many column names contain spaces; always use double quotes in SQL when referencing them.

3) (Optional but Recommended) Validate CSV data quality before import:
- Windows (PowerShell):
  - python validate_csv_import.py
- This script validates schema, checks for missing values, and identifies potential data quality issues using the validation functions from ml_finance_model_v8_2.py.

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


## Environment Variables
Environment variable defaults and examples live in environment_variables.txt. Key items:
- TF_CPP_MIN_LOG_LEVEL=2  (reduces TensorFlow logging verbosity)
- Optional paths: DATA_DIR, MODEL_DIR, CACHE_DIR
- Optional modeling: MODEL_VERSION, RANDOM_SEED
- Optional performance: N_JOBS, MEMORY_LIMIT

Set them in your shell or via a .env file if your tools auto‑load it.
- Windows (PowerShell):
  - $env:TF_CPP_MIN_LOG_LEVEL = 2
- macOS/Linux (bash):
  - export TF_CPP_MIN_LOG_LEVEL=2


## Running the Project
This project is notebook‑first.

1) Start Jupyter
- jupyter notebook  (or: jupyter lab)

2) Open and run ml_finance_model_v8_2.ipynb
- Run cells in order. Use environment variables and Path from pathlib to avoid hard‑coded paths.

Outputs: model diagnostics, ranking tables, and optional CSV/Excel exports (if implemented in the notebook cells).


## Scripts and Entry Points

### Main Entry Points
- **ml_finance_model_v8_2.ipynb** — Main Jupyter notebook for interactive exploration and modeling
- **ml_finance_model_v8_2.py** — Python script with CLI for batch processing:
  - `--data-source {auto|csv|db}` — Data source selection (default: auto)
  - `--db-url <url>` — Database connection string (or use DB_URL env var)
  - `--limit <n>` — Limit rows for testing
  - `--out-dir <path>` — Output directory (default: outputs)
  - `--dry-run` — Skip model training
  - See "Running as a Python script" section below for examples

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
- **analyze_notebook.py** — Notebook structure analyzer (counts cells, previews content, searches for functions)
- **update_notebook.py** — Notebook synchronizer (extracts TDD functions from .py and inserts into .ipynb)
- **verify_notebook.py** — Notebook verification utility (checks for presence of TDD functions)

### TODOs
- TODO: Consider adding console_scripts entry point in setup.cfg/pyproject.toml for easier CLI access


## Tests
- We use Python’s built‑in unittest.
- Run all tests from the project root:
  - python -m unittest -v

Test suite in tests/ directory:
- test_repository_setup.py — Validates repository basics (required files, SQL schema, environment config, CSV format)
- test_data_quality.py — Data validation and quality checks
- test_loaders.py — CSV and database loading functions
- test_features.py — Feature engineering functions
- test_build_features.py — Feature building pipeline
- test_eda.py — Exploratory data analysis utilities
- test_preprocess_and_training.py — Preprocessing and training workflows
- test_regression.py — Regression model evaluation
- test_classification.py — Event classification model tests
- test_analytics.py — Analytics and stock ranking tests


## Project Structure

```
Finance_ML_Analytics_Platform/
├── finance_ml/                    # Main Python package (v0.3.0)
│   ├── __init__.py               # Package exports and version
│   ├── data.py                   # Data loading, normalization, validation
│   ├── features.py               # Feature engineering functions
│   ├── models.py                 # ML models (classification, regression, ensembles)
│   ├── eval.py                   # Analytics, visualizations, reporting
│   ├── config.py                 # Configuration management
│   └── cli.py                    # Command-line interface
│
├── tests/                        # Comprehensive test suite (144+ tests)
│   ├── test_finance_ml_data.py
│   ├── test_finance_ml_features.py
│   ├── test_finance_ml_models.py
│   ├── test_finance_ml_eval.py
│   ├── test_repository_setup.py
│   └── ... (10+ test modules)
│
├── data/                         # Regional equity data (CSV files)
│   ├── screening_us.csv
│   ├── screening_eu.csv
│   ├── screening_apac.csv
│   └── screening_rotw.csv
│
├── .github/workflows/            # CI/CD pipelines
│   └── tests.yml                 # Automated testing workflow
│
├── ml_finance_model_v8_2.ipynb  # Interactive Jupyter notebook
├── ml_finance_model_v8_2.py     # Legacy script (uses finance_ml package)
│
├── pyproject.toml               # Modern Python packaging configuration
├── setup.py                     # Backward-compatible setup
├── requirements.txt             # Core dependencies
├── CHANGELOG.md                 # Version history
├── README.md                    # This file
├── IMPROVEMENT_PLAN.md          # Development roadmap
│
├── create_equities_schema.sql   # PostgreSQL schema setup
├── import_equities_data.sql     # Data import script
├── environment_variables.txt    # Environment configuration examples
│
└── .junie/guidelines.md         # Development guidelines
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


## Troubleshooting
- TensorFlow installation issues: the project primarily uses scikit‑learn and gradient boosting libraries. CPU‑only TensorFlow is fine; ensure compatible system libraries. If installation is problematic on your machine, you can proceed with non‑TF parts first.
- PostgreSQL quoting: many columns contain spaces/punctuation; always use double quotes in SQL identifiers.
- Windows CSV imports: prefer \\copy from psql to avoid server‑side file permission issues.


## Versioning
- When materially changing modeling behavior, bump MODEL_VERSION (e.g., v8_3) and document changes in this README (features, labels, metrics). Default in environment_variables.txt is MODEL_VERSION=v8_2 (commented example).


## License
No license file is included yet.
- TODO: Add a LICENSE file (e.g., MIT, Apache‑2.0) and reference it here. Until a license is provided, assume all rights reserved.


## CLI Usage

The package provides three command-line tools after installation:

### 1. `finance-ml` — Main Analysis Pipeline

Full ML pipeline with data loading, EDA, feature engineering, and model training.

```bash
# Auto-detect data source (DB if available, else CSV)
finance-ml --data-source auto --limit 5000

# Use database
finance-ml --data-source db --db-url postgresql://postgres:@localhost/postgres

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
finance-ml-analyze --data-source db --db-url postgresql://postgres:@localhost/postgres

# Limit rows for quick check
finance-ml-analyze --data-source csv --limit 1000 -v
```

### 3. `finance-ml-validate` — Data Validation

Validate data quality and schema.

```bash
# Validate CSV data
finance-ml-validate --data-source csv --data-dir ./data

# Validate database
finance-ml-validate --data-source db --db-url postgresql://postgres:@localhost/postgres

# Verbose output
finance-ml-validate --data-source csv -v
```

### Legacy Script

The original script is still available but now uses the `finance_ml` package:

```bash
python ml_finance_model_v8_2.py --data-source auto --limit 5000 --out-dir outputs
```


## Configuration Management

### Environment Variables

Set these in your shell or via `.env` file:

```bash
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

### Configuration Files

Create `config.json` or `config.yaml` for project settings:

**config.json:**
```json
{
  "data_dir": "data",
  "output_dir": "outputs",
  "db_url": "postgresql://postgres:@localhost/postgres",
  "model_version": "v8_2",
  "random_seed": 42,
  "n_jobs": -1
}
```

**config.yaml:**
```yaml
data_dir: data
output_dir: outputs
db_url: postgresql://postgres:@localhost/postgres
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


## What's New in v0.3.0

- ✅ **Configuration Management**: Centralized config via environment, JSON, or YAML
- ✅ **CLI Tools**: Three console commands for different workflows
- ✅ **Modern Packaging**: `pyproject.toml` with optional dependency groups
- ✅ **CI/CD**: GitHub Actions workflow for automated testing
- ✅ **Updated Notebook**: Now imports from `finance_ml` package
- ✅ **CHANGELOG.md**: Track version history

See [CHANGELOG.md](CHANGELOG.md) for complete version history.
