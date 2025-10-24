# Finance ML Analytics Platform

**Version 0.3.0** — A professional, modular Python package for equity screening, feature engineering, and machine learning models across global regions.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Quick Start (TL;DR)

- Python: 3.10 or 3.11
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
- Notebook: open ml_finance_model_v8_2.ipynb and run cells in order
- Script (optional): python ml_finance_model_v8_2.py --data-source auto --limit 5000 --out-dir outputs
- Tests: python -m unittest -v

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
- 🧪 **Tested**: Comprehensive unit tests with good coverage
- 🚀 **CLI**: Three command-line tools for different workflows


## Tech Stack

### Core Technologies
- **Language**: Python 3.10 or 3.11 (required)
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
- **`[database]`**: psycopg2-binary, SQLAlchemy (database access from Python)
- **`[advanced-features]`**: boruta, numba (advanced feature selection and performance)
- **`[all]`**: All optional dependencies combined


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

**Basic installation** (core dependencies only):
- pip install -r requirements.txt

**Package installation** (for CLI tools and optional dependencies):
- pip install -e .                    # Core package only
- pip install -e ".[dev]"             # With development tools
- pip install -e ".[database]"        # With database client libraries
- pip install -e ".[all]"             # With all optional dependencies

**Note**: TensorFlow is optional. If you encounter installation issues, you can skip it—the core workflow uses scikit-learn and gradient boosting libraries.


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
- **ml_finance_model_v8_2.py** — Lightweight Python script with a minimal CLI:
  - `--data-source {auto|csv|db}` — Data source selection (default: auto)
  - `--db-url <url>` — Database connection string (or use DB_URL env var)
  - `--limit <n>` — Limit rows for testing
  - `--out-dir <path>` — Output directory (default: outputs)
  - `--dry-run` — Skip model training
  - Note: Advanced options (skip-eda, per-sector, etc.) are available via the console script `finance-ml`.
  - See "Legacy Script" example in CLI Usage below

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
- **analyze_notebook.py** — Notebook structure analyzer (counts cells, previews content, searches for functions)
- **update_notebook.py** — Notebook synchronizer (extracts TDD functions from .py and inserts into .ipynb)
- **verify_notebook.py** — Notebook verification utility (checks for presence of TDD functions)


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
│   ├── cli.py                    # Command-line interface
│   ├── risk_metrics.py           # Risk metrics and portfolio risk analysis
│   └── portfolio_optimization.py # Portfolio optimization utilities
│
├── tests/                        # Unit tests (see tests/ for modules)
│   ├── test_repository_setup.py
│   ├── test_finance_ml_data.py
│   ├── test_features.py
│   ├── test_build_features.py
│   ├── test_eda.py
│   ├── test_preprocess_and_training.py
│   ├── test_regression.py
│   ├── test_classification.py
│   ├── test_analytics.py
│   └── ...
│
├── data/                         # Regional equity data (CSV files)
│   ├── screening_us.csv
│   ├── screening_eu.csv
│   ├── screening_apac.csv
│   └── screening_rotw.csv
│
├── ml_finance_model_v8_2.ipynb   # Interactive Jupyter notebook
├── ml_finance_model_v8_2.py      # Script version (uses finance_ml package)
│
├── analyze_notebook.py           # Notebook analysis utility
├── refactor_notebook.py          # Notebook refactoring helper
├── update_notebook.py            # Sync functions between .py and .ipynb
├── update_notebook_imports.py    # Update notebook imports
├── verify_notebook.py            # Verify notebook functions
├── verify_preprocessing_improvements.py # Verify preprocessing pipeline improvements
├── validate_csv_import.py        # CSV data validation helper
│
├── create_equities_schema.sql    # PostgreSQL schema setup
├── import_equities_data.sql      # Data import script
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

### Legacy Script

The original script is still available but now uses the `finance_ml` package:

```bash
python ml_finance_model_v8_2.py --data-source auto --limit 5000 --out-dir outputs
```


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

Currently, no CI workflows are included in this repository (no .github/workflows directory).

TODO:

- Add a GitHub Actions workflow for automated tests across Windows, Ubuntu, and macOS on Python 3.10 and 3.11.
- Optionally add code quality checks (black, isort, flake8, mypy) and coverage reporting.

You can run equivalent checks locally:

```bash
# Run tests
python -m unittest discover -s tests -v

# Check code formatting
black --check finance_ml tests

# Check import sorting
isort --check-only finance_ml tests

# Lint code
flake8 finance_ml

# Type checking
mypy finance_ml --ignore-missing-imports
```


## What's New in v0.3.0

- ✅ **Configuration Management**: Centralized config via environment, JSON, or YAML
- ✅ **CLI Tools**: Three console commands for different workflows
- ✅ **Modern Packaging**: `pyproject.toml` with optional dependency groups
- ✅ **CI/CD**: GitHub Actions workflow for automated testing
- ✅ **Updated Notebook**: Now imports from `finance_ml` package
- ✅ **CHANGELOG.md**: Track version history

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

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
