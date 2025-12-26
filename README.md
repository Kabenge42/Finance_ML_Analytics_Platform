# Finance ML Analytics Platform

A lightweight, notebook-centric workflow for equity screening, feature engineering, and machine learning models across
global regions.

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The Finance ML Analytics Platform is a comprehensive solution for financial data analysis, machine learning modeling,
and portfolio optimization. It implements a structured 8-phase workflow to ensure data quality, robust feature
engineering, and reliable model evaluation.

### Key Features

- **8-Phase ML Workflow**: Phases 9.1–9.8 covering data ingestion, preprocessing (winsorization, imputation, scaling),
  feature engineering, feature selection, model training, uncertainty estimation, and evaluation.
- **Unified Schema Module**: Single source of truth for 500+ financial columns, ensuring alignment between SQL and
  Python.
- **Modular Feature Engineering**: High-performance feature generators for valuation, profitability, momentum, quality,
  and more.
- **Unified ETL Pipeline**: Seamlessly handles data from CSV files or SQL databases (PostgreSQL/SQLite) with strict
  schema validation.
- **Advanced Feature Engineering**: Automated generation of 100+ financial metrics including valuation ratios,
  profitability margins, growth rates, and leverage stats.
- **Portfolio Optimization**: Section 18 workflow for risk-adjusted returns, efficient frontier visualization, and
  backtesting.
- **Interactive Dashboards**: Advanced visualizations for earnings analytics, market movers, and model performance using
  Streamlit and Dash.
- **Production-Ready**: Includes a CLI and high-performance scripts for automated execution.

## Requirements

- **Python**: 3.12, 3.13, or 3.14.
- **Operating System**: Windows (primary support via PowerShell), Linux, or macOS.
- **Memory**: 8GB RAM minimum (16GB recommended for large datasets).

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
3. **Optional: Install as a package:**
   ```powershell
   pip install -e .
   ```

## Running the Platform

### Main Workflow

1. **Production Script**:
   ```powershell
   python finance_ml_analytics_platform.py
   ```
2. **Jupyter Notebooks**:
    - `ml_finance_model_main.ipynb`: Core ML pipeline execution.
    - `etl_data_explorer.ipynb`: Interactive data exploration and ETL testing.
    - `stock_analytics.ipynb`: Comprehensive stock market analysis.

### Interactive Dashboards

Before running dashboards, ensure you have generated results and synced assets:
```powershell
python tools\setup_dashboard_assets.py
```

- **Streamlit (Recommended)**:
  ```powershell
  streamlit run finance_ml/dashboards/streamlit_app.py
  ```
- **Dash**:
  ```powershell
  python finance_ml/dashboards/dash_app.py
  ```

### CLI Commands

- `finance-ml`: Main analysis pipeline.
- `finance-ml-analyze`: Quick data analysis.
- `finance-ml-validate`: Data schema and quality validation.

## Scripts

| Script                            | Purpose                                              |
|:----------------------------------|:-----------------------------------------------------|
| `tools\setup_environment.py`      | Full environment and dependency setup.               |
| `tools\run_fast_tests.py`         | Quick verification of utility modules.               |
| `tools\run_earnings_monitor.py`   | Monitors and visualizes company earnings events.     |
| `tools\setup_dashboard_assets.py` | Syncs pipeline results with the dashboard interface. |

## Environment Variables

Key configuration options (see `environment_variables.txt` for all options):

| Variable        | Description                                    | Default   |
|:----------------|:-----------------------------------------------|:----------|
| `OUTPUT_DIR`    | Directory for generated reports and artifacts. | `outputs` |
| `DB_URL`        | SQLAlchemy connection URL (PostgreSQL/SQLite). | -         |
| `MODEL_VERSION` | Identifier for the current model iteration.    | `v9_10`   |
| `LOG_LEVEL`     | Logging verbosity (DEBUG, INFO, etc.).         | `INFO`    |

## Tests

Run the full test suite or fast validation:

```powershell
# Run all tests
pytest

# Run fast tests only
python tools\run_fast_tests.py
```

## Project Structure

- `finance_ml/`: Core package including ETL, ML models, and dashboards.
- `tests/`: Comprehensive unit and integration test suite.
- `docs/`: Technical guidelines and workflow documentation.
- `tools/`: Utility scripts for environment setup and maintenance.
- `data/`: Local data storage (CSV format).
- `models/`: Serialized model files and metadata.
- `outputs/`: Generated reports, visualizations, and logs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## TODOs

- [ ] **Phase 9.8**: Enhance model governance and lineage tracking.
- [ ] **Data Support**: Implement real-time data ingestion via WebSocket APIs.
- [ ] **Deployment**: Add Docker Compose for one-click environment deployment.
- [ ] **Docs**: Expand KDoc-style documentation for the `finance_ml.ml_workflow` sub-packages.
