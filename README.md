# Finance ML Analytics Platform

A lightweight, notebook-centric workflow for equity screening, feature engineering, and machine learning models across
global regions.

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## Requirements

- **Python**: Version 3.12, 3.13, or 3.14.
- **Operating System**: Windows (primary support via PowerShell), Linux, or macOS.
- **Dependencies**: Managed via `requirements.txt`. Key libraries include:
    - `pandas`, `numpy`, `scipy`, `statsmodels`
    - `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
    - `streamlit`, `dash`
    - `sqlalchemy`

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

### Jupyter Notebooks

- `ml_finance_model_main.py` / `.ipynb`: Core ML pipeline execution and experimentation.
- `etl_data_explorer.ipynb`: Interactive data exploration and ETL testing.
- `stock_analytics.ipynb`: Comprehensive stock market analysis.
- `stock_price_target_prediction.ipynb`: Price target prediction modeling.

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
- `finance-ml`: Main analysis pipeline.
- `finance-ml-analyze`: Quick data analysis.
- `finance-ml-validate`: Data schema and quality validation.

## Scripts

Useful utility scripts located in `tools/`:

| Script                                | Purpose                                          |
|:--------------------------------------|:-------------------------------------------------|
| `tools\setup_environment.py`          | Full environment and dependency setup.           |
| `tools\run_fast_tests.py`             | Quick verification of utility modules.           |
| `tools\run_earnings_monitor.py`       | Monitors and visualizes company earnings events. |
| `tools\setup_dashboard_assets.py`     | Syncs pipeline results with dashboard interface. |
| `tools\load_equities_data.py`         | Loads and processes equities data.               |
| `tools\check_source_csv_zero_cols.py` | Validates source CSVs for empty columns.         |

## Environment Variables

Key configuration options (see `environment_variables.txt` for full list):

| Variable        | Description                      | Default   |
|:----------------|:---------------------------------|:----------|
| `OUTPUT_DIR`    | Directory for generated reports. | `outputs` |
| `DB_URL`        | SQLAlchemy connection URL.       | -         |
| `MODEL_VERSION` | Identifier for current model.    | `v9_10`   |
| `LOG_LEVEL`     | Logging verbosity.               | `INFO`    |

## Tests

Run the full test suite or fast validation:

```powershell
# Run all tests (requires pytest)
pytest

# Run fast tests only (skips heavy ML tests)
python tools\run_fast_tests.py
```

## Project Structure

```text
finance_ml_analytics_platform/
├── finance_ml/               # Core package
│   ├── core/                 # Shared constants & schema
│   ├── etl/                  # ETL pipeline & config
│   ├── features/             # Feature engineering
│   ├── ml_workflow/          # ML phases (9.1-9.8)
│   └── dashboards/           # Streamlit/Dash apps
├── tests/                    # Unit & integration tests
├── tools/                    # Utility scripts
├── docs/                     # Documentation & guidelines
├── data/                     # Local data storage
├── models/                   # Serialized models
└── outputs/                  # Generated reports & logs
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## TODOs

- [ ] **Phase 9.8**: Enhance model governance and lineage tracking.
- [ ] **Data Support**: Implement real-time data ingestion via WebSocket APIs.
- [ ] **Deployment**: Add Docker Compose for one-click environment deployment.
- [ ] **Docs**: Expand KDoc-style documentation for the `finance_ml.ml_workflow` sub-packages.
- [ ] **Testing**: Increase coverage for `finance_ml.features.advanced` modules.
