# Finance ML Analytics Platform

A lightweight, notebook-centric workflow for equity screening, feature engineering, and machine learning models across
global regions.

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.9.5-green)](https://github.com/Kabenge42/Finance_ML_Analytics_Platform)

## Overview

The Finance ML Analytics Platform is a comprehensive solution for financial data analysis, machine learning modeling,
and portfolio optimization. It implements a structured 8-phase workflow (Phases 9.1–9.8) to ensure data quality, robust
feature engineering, and reliable model evaluation, followed by a Portfolio Optimization module.

### Key Features

- **8-Phase ML Workflow**: Covers data ingestion, preprocessing (winsorization, imputation), feature engineering,
  feature selection, model training, and evaluation.
- **Unified Schema Module**: Single source of truth (`finance_ml.core.schema`) for 500+ financial columns, ensuring
  alignment between SQL and Python.
- **Modular Feature Engineering**: High-performance feature generators for valuation, profitability, momentum, and
  quality. Phase 9.3 v1.14 delivers 350 schema-aligned features across 21 categories, including new temporal,
  sentiment, dividend timing, cash flow, and EPS trajectory signals.
- **Unified ETL Pipeline**: Decoupled ETL configuration and execution handling CSV and SQL sources.
- **Portfolio Optimization**: Risk-adjusted return maximization, efficient frontier visualization, and backtesting.
- **Interactive Dashboards**: Streamlit and Dash applications for earnings analytics, market monitoring, and model
  performance.
- **Production-Ready**: CLI entry points and optimized scripts for automated execution.

## Tech Stack

| Category            | Technologies                                                               |
|:--------------------|:---------------------------------------------------------------------------|
| **Language**        | Python 3.12 / 3.13 / 3.14                                                  |
| **Package Manager** | `pip`, `Pipfile` (pipenv), `pyproject.toml` (setuptools)                   |
| **ML Frameworks**   | `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `TensorFlow` (optional) |
| **Data Processing** | `pandas`, `NumPy`, `SciPy`, `statsmodels`                                  |
| **Visualization**   | `Plotly`, `Matplotlib`, `Seaborn`                                          |
| **Dashboards**      | `Streamlit`, `Dash`                                                        |
| **Database**        | `PostgreSQL` (psycopg2), `SQLAlchemy`, `SQLite`                            |
| **Testing**         | `pytest`, `unittest`                                                       |
| **Code Quality**    | `Black`, `Flake8`, `isort`, `Mypy`                                         |
| **Utilities**       | `tqdm`, `joblib`, `numba`, `forex-python`                                  |

## Requirements

- **Python**: Version 3.12, 3.13, or 3.14 (`>=3.12,<3.15`).
- **Operating System**: Windows (primary support via PowerShell), Linux, or macOS.
- **Dependencies**: Managed via `requirements.txt`, `Pipfile`, or `pyproject.toml`.

## Setup

### Automated Setup (Recommended)

Use the provided setup script to initialize the environment:

```powershell
python tools\setup_environment.py
```

This script detects your Python version, creates a virtual environment, installs dependencies, and configures
environment variables.

### Manual Setup

1. **Create a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **(Optional) Install as a package:**
   ```powershell
   pip install -e .
   ```

## Running the Platform

### Main Workflow

To execute the complete 8-phase ML workflow and portfolio optimization:

```powershell
python finance_ml_analytics_platform.py
```

### Entry Points (CLI)

The package provides several command-line entry points:

| Command               | Description                                |
|:----------------------|:-------------------------------------------|
| `finance-ml`          | Main analysis pipeline execution           |
| `finance-ml-analyze`  | Quick data analysis and exploration        |
| `finance-ml-validate` | Data validation and schema alignment check |

### Dashboards

Interactive dashboards for visualizing results and monitoring markets:

| Dashboard              | Run Command                                              |
|:-----------------------|:---------------------------------------------------------|
| **Streamlit App**      | `streamlit run finance_ml/dashboards/streamlit_app.py`   |
| **Equities Dashboard** | `python finance_ml/dashboards/equities_dashboard_app.py` |
| **Dash App**           | `python finance_ml/dashboards/dash_app.py`               |

### Jupyter Notebooks

- `ml_finance_model_main.ipynb`: Core ML pipeline execution and experimentation.
- `etl_data_explorer.ipynb`: Interactive data exploration and ETL testing.
- `stock_analytics.ipynb`: Comprehensive stock market analysis.
- `stock_price_target_prediction.ipynb`: Price target prediction modeling.
- `portfolio_optimization_risk_management.ipynb`: Portfolio optimization and risk analysis.

## Scripts & Utilities

Useful utility scripts located in `tools/`:

| Script                         | Purpose                                            |
|:-------------------------------|:---------------------------------------------------|
| `setup_environment.py`         | Full environment and dependency setup              |
| `run_fast_tests.py`            | Quick verification of utility modules (Priority 6) |
| `run_earnings_monitor.py`      | Monitors and visualizes company earnings events    |
| `setup_dashboard_assets.py`    | Syncs pipeline results with dashboard interface    |
| `load_equities_data.py`        | Loads and processes equities data                  |
| `validate_schema_alignment.py` | Validates SQL/Python schema alignment              |
| `extract_schema.py`            | Extracts schema definitions from database          |
| `cleanup_environments.py`      | Cleans up virtual environments and caches          |
| `import_sqlite.py`             | Imports data into SQLite database                  |

## Environment Variables

Configuration options can be set via environment variables or in `environment_variables.txt`:

| Variable               | Description                                     | Default    |
|:-----------------------|:------------------------------------------------|:-----------|
| `LOG_LEVEL`            | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | `INFO`     |
| `TF_CPP_MIN_LOG_LEVEL` | TensorFlow log level (0=DEBUG to 3=ERROR)       | `2`        |
| `DATA_DIR`             | Directory for input data                        | `data`     |
| `MODEL_DIR`            | Directory for saved models                      | `models`   |
| `CACHE_DIR`            | Directory for cached files                      | `.cache`   |
| `OUTPUT_DIR`           | Directory for generated reports and artifacts   | `outputs`  |
| `DB_URL`               | SQLAlchemy connection URL                       | —          |
| `DB_SCHEMA`            | Database schema name                            | `public`   |
| `DB_TABLE`             | Database table name                             | `equities` |
| `MODEL_VERSION`        | Identifier for current model version            | `v9_10`    |
| `RANDOM_SEED`          | Random seed for reproducibility                 | `42`       |
| `N_JOBS`               | Number of parallel jobs (-1 for all cores)      | `-1`       |

## Tests

The project uses `pytest` for comprehensive testing and a custom fast test runner for quick verification.

```powershell
# Run all tests
pytest

# Run fast tests (utility modules only)
python tools\run_fast_tests.py

# Run with coverage
pytest --cov=finance_ml --cov-report=html
```

Test files are located in `tests/` and follow the naming convention `test_*.py`.

## Project Structure

```text
Finance_Analytics_Platform/
├── finance_ml/                 # Core Python package
│   ├── core/                   # Shared constants and unified schema
│   ├── etl/                    # ETL pipeline and data transformation
│   ├── features/               # Feature engineering (advanced, basic)
│   ├── ml_workflow/            # 8-phase ML workflow implementation
│   │   ├── preprocessing/      # Data cleaning, imputation, scaling
│   │   ├── eda/                # Exploratory Data Analysis
│   │   └── models/             # Model training and evaluation
│   ├── dashboards/             # Streamlit, Dash, and Equities dashboards
│   └── cli.py                  # CLI entry point definitions
├── tests/                      # Unit and integration tests
├── tools/                      # Utility and setup scripts
├── docs/                       # Documentation and code guidelines
├── data/                       # Local data storage
├── models/                     # Saved model artifacts
├── outputs/                    # Generated reports and visualizations
├── pyproject.toml              # Build system and project metadata
├── Pipfile                     # Pipenv dependency management
└── requirements.txt            # Unified dependency list
```

## Database Setup (Optional)

For PostgreSQL database support:

1. **Create the schema**:
   ```powershell
   psql -U postgres -d your_database -f create_equities_schema.sql
   ```
2. **Import data**:
   ```powershell
   psql -U postgres -d your_database -f import_equities_data.sql
   ```
3. **Configure environment**:
   Update `DB_URL` in `environment_variables.txt` or set the environment variable directly.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- **Repository**: [GitHub](https://github.com/Kabenge42/Finance_ML_Analytics_Platform)
- **Guidelines**: [docs/code_guidelines.md](docs/code_guidelines.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## TODOs

- [ ] **Phase 9.8**: Enhance model governance and lineage tracking.
- [ ] **Data Support**: Implement real-time data ingestion via WebSocket APIs.
- [ ] **Deployment**: Add Docker Compose for one-click environment deployment.
- [ ] **Docs**: Expand KDoc-style documentation for the `finance_ml.ml_workflow` sub-packages.
- [ ] **Testing**: Increase coverage for `finance_ml.features.advanced` modules.
- [ ] **CI/CD**: Add GitHub Actions workflow for automated testing.
