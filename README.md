# Finance ML Analytics Platform

A lightweight, notebook-centric workflow for equity screening, feature engineering, and machine learning models across
global regions.

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.9.5-green)](https://github.com/Kabenge42/Finance_ML_Analytics_Platform)

## Overview

The Finance ML Analytics Platform is a comprehensive solution for financial data analysis, machine learning modeling,
and portfolio optimization. It implements a structured workflow (Phases 9.1–9.8) to ensure data quality, robust feature
engineering, and reliable model evaluation, followed by a Portfolio Optimization module (Section 18).

### Key Features

- **8-Phase ML Workflow**: Covers data ingestion, preprocessing (winsorization, imputation), feature engineering,
  feature selection, model training, and evaluation.
- **Unified Schema Module**: Single source of truth (`finance_ml.core.schema`) for 500+ financial columns, ensuring
  alignment between SQL and Python.
- **Modular Feature Engineering**: High-performance feature generators for valuation, profitability, momentum, and
  quality in `finance_ml/features/advanced/`.
- **Unified ETL Pipeline**: Decoupled ETL configuration and execution handling CSV and SQL sources.
- **Portfolio Optimization**: Risk-adjusted return maximization, efficient frontier visualization, and backtesting (
  Section 18).
- **Interactive Dashboards**: Streamlit and Dash applications for earnings analytics, market monitoring, and model
  performance.
- **Production-Ready**: CLI entry points and optimized scripts for automated execution.

## Tech Stack

| Category            | Technologies                                                     |
|:--------------------|:-----------------------------------------------------------------|
| **Language**        | Python 3.12 / 3.13 / 3.14                                        |
| **Package Manager** | pip, Pipfile (pipenv), pyproject.toml (setuptools)               |
| **ML Frameworks**   | scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow (optional) |
| **Data Processing** | pandas, NumPy, SciPy, statsmodels                                |
| **Visualization**   | Plotly, Matplotlib, Seaborn                                      |
| **Dashboards**      | Streamlit, Dash                                                  |
| **Database**        | PostgreSQL (psycopg2), SQLAlchemy, SQLite                        |
| **Testing**         | pytest, unittest                                                 |
| **Code Quality**    | Black, Flake8, isort, Mypy                                       |

## Requirements

- **Python**: Version 3.12, 3.13, or 3.14 (`>=3.12,<3.15`).
- **Operating System**: Windows (primary support via PowerShell), Linux, or macOS.
- **Dependencies**: Managed via `requirements.txt`, `Pipfile`, or `pyproject.toml`. Key libraries include:
    - `pandas`, `numpy`, `scipy`, `statsmodels`
    - `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
  - `streamlit`, `dash`, `plotly`
  - `sqlalchemy`, `psycopg2-binary`

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
4. **(Optional) Install with all extras:**
   ```powershell
   pip install -e ".[all]"
   ```

## Running the Platform

### Main Workflow

To execute the complete 8-phase ML workflow and portfolio optimization:

```powershell
python finance_ml_analytics_platform.py
```

### Entry Points

| Entry Point                        | Description                                    |
|:-----------------------------------|:-----------------------------------------------|
| `finance_ml_analytics_platform.py` | Main orchestration script for full ML pipeline |
| `ml_finance_model_main.py`         | Core ML pipeline execution                     |

### Jupyter Notebooks

- `ml_finance_model_main.ipynb`: Core ML pipeline execution and experimentation.
- `etl_data_explorer.ipynb`: Interactive data exploration and ETL testing.
- `stock_analytics.ipynb`: Comprehensive stock market analysis.
- `stock_price_target_prediction.ipynb`: Price target prediction modeling.
- `portfolio_optimization_risk_management.ipynb`: Portfolio optimization and risk analysis.

### Interactive Dashboards

First, sync the dashboard assets:
```powershell
python tools\setup_dashboard_assets.py
```

- **Streamlit App (Recommended)**:
  ```powershell
  streamlit run finance_ml/dashboards/streamlit_app.py
  ```
- **Dash App**:
  ```powershell
  python finance_ml/dashboards/dash_app.py
  ```

### CLI Commands

The package exposes the following CLI commands (after `pip install -e .`):

| Command               | Description                        |
|:----------------------|:-----------------------------------|
| `finance-ml`          | Main analysis pipeline             |
| `finance-ml-analyze`  | Quick data analysis                |
| `finance-ml-validate` | Data schema and quality validation |

## Scripts

Useful utility scripts located in `tools/`:

| Script                         | Purpose                                         |
|:-------------------------------|:------------------------------------------------|
| `setup_environment.py`         | Full environment and dependency setup           |
| `run_fast_tests.py`            | Quick verification of utility modules           |
| `run_earnings_monitor.py`      | Monitors and visualizes company earnings events |
| `setup_dashboard_assets.py`    | Syncs pipeline results with dashboard interface |
| `load_equities_data.py`        | Loads and processes equities data               |
| `validate_schema_alignment.py` | Validates SQL/Python schema alignment           |
| `extract_schema.py`            | Extracts schema definitions from database       |
| `cleanup_environments.py`      | Cleans up virtual environments and caches       |
| `import_sqlite.py`             | Imports data into SQLite database               |

## Environment Variables

Configuration options can be set via environment variables or in `environment_variables.txt`:

| Variable                   | Description                             | Default    |
|:---------------------------|:----------------------------------------|:-----------|
| `OUTPUT_DIR`               | Directory for generated reports         | `outputs`  |
| `DATA_DIR`                 | Directory for input data                | `data`     |
| `MODEL_DIR`                | Directory for saved models              | `models`   |
| `DB_URL`                   | SQLAlchemy connection URL               | —          |
| `DB_SCHEMA`                | Database schema name                    | `public`   |
| `DB_TABLE`                 | Database table name                     | `equities` |
| `MODEL_VERSION`            | Identifier for current model            | `v9_10`    |
| `LOG_LEVEL`                | Logging verbosity                       | `INFO`     |
| `TF_CPP_MIN_LOG_LEVEL`     | TensorFlow log level (0-3)              | `2`        |
| `N_JOBS`                   | Parallel jobs (-1 for all cores)        | `-1`       |
| `RANDOM_SEED`              | Random seed for reproducibility         | `42`       |
| `ENABLE_INTERACTIVE_PLOTS` | Enable interactive visualizations       | `true`     |
| `REPORT_FORMAT`            | Report output format (html, pdf, excel) | `html`     |

## Tests

Run the full test suite or fast validation:

```powershell
# Run all tests (requires pytest)
pytest

# Run fast tests only (skips heavy ML tests)
python tools\run_fast_tests.py

# Run specific test file
pytest tests\test_etl_unified_pipeline.py

# Run with coverage
pytest --cov=finance_ml --cov-report=html
```

Test files are located in `tests/` and follow the naming convention `test_*.py`.

## Project Structure

```text
Finance_Analytics_Platform/
├── finance_ml/                 # Core Python package
│   ├── core/                   # Shared constants, schema, utilities
│   │   ├── schema.py           # Unified column schema (500+ columns)
│   │   └── constants.py        # Global constants and configurations
│   ├── etl/                    # ETL pipeline modules
│   │   ├── config.py           # ETL configuration
│   │   └── stages/             # Pipeline stages (imputation, validation)
│   ├── features/               # Feature engineering
│   │   └── advanced/           # Domain-specific features (valuation, momentum)
│   ├── ml_workflow/            # ML phases (9.1-9.8)
│   │   └── preprocessing/      # Data preprocessing modules
│   ├── dashboards/             # Interactive dashboards
│   │   ├── streamlit_app.py    # Streamlit application
│   │   ├── dash_app.py         # Dash application
│   │   └── widgets/            # Dashboard components
│   └── cli.py                  # Command-line interface
├── tests/                      # Unit and integration tests
├── tools/                      # Utility scripts
├── docs/                       # Documentation and guidelines
│   └── code_guidelines.md      # Development guidelines
├── data/                       # Local data storage (CSV, etc.)
├── models/                     # Serialized ML models
├── outputs/                    # Generated reports, visualizations, logs
├── create_equities_schema.sql  # PostgreSQL schema definition
├── import_equities_data.sql    # Data import SQL scripts
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata and build config
├── Pipfile                     # Pipenv dependencies
└── LICENSE                     # MIT License
```

## Database Setup (Optional)

For PostgreSQL database support:

1. Create the schema:
   ```powershell
   psql -U postgres -d your_database -f create_equities_schema.sql
   ```

2. Import data:
   ```powershell
   psql -U postgres -d your_database -f import_equities_data.sql
   ```

3. Set the connection URL:
   ```powershell
   $env:DB_URL = "postgresql+psycopg2://user:password@localhost:5432/database"
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- **Repository**: [GitHub](https://github.com/Kabenge42/Finance_ML_Analytics_Platform)
- **Issues**: [GitHub Issues](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/issues)
- **Documentation**: [docs/code_guidelines.md](docs/code_guidelines.md)

## TODOs

- [ ] **Phase 9.8**: Enhance model governance and lineage tracking.
- [ ] **Data Support**: Implement real-time data ingestion via WebSocket APIs.
- [ ] **Deployment**: Add Docker Compose for one-click environment deployment.
- [ ] **Docs**: Expand KDoc-style documentation for the `finance_ml.ml_workflow` sub-packages.
- [ ] **Testing**: Increase coverage for `finance_ml.features.advanced` modules.
- [ ] **CI/CD**: Add GitHub Actions workflow for automated testing.
