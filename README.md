# Finance ML Analytics Platform

A comprehensive, production-ready platform for equity screening, feature engineering, and machine learning modeling
across global financial markets.

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Package Version](https://img.shields.io/badge/version-0.9.5-green)](https://github.com/Kabenge42/Finance_ML_Analytics_Platform)

## Overview

The Finance ML Analytics Platform is a robust solution for financial data analysis, machine learning modeling, and
portfolio optimization. It implements a structured **8-Phase ML Workflow** (Phases 9.1–9.8) to ensure data quality,
advanced feature engineering, and reliable model evaluation, followed by a **7-Phase Portfolio Optimization** module.

The platform has been enhanced with a refactored **Market Analytics** implementation (`finance_ml.analytics`), providing
modular, high-performance tools for Bayesian statistical analysis, multi-factor screening, and interactive
visualizations.

### Key Features

- **8-Phase ML Workflow**: Covers data ingestion, preprocessing (winsorization, 6-step imputation), exploratory data
  analysis (EDA), advanced feature engineering, feature selection, model training (regression/classification), and error
  analysis.
- **7-Phase Portfolio Optimization**: Implements stock selection, return prediction, risk-adjusted optimization (
  Efficient Frontier), backtesting, and interactive dashboards.
- **Advanced Market Analytics**: Modular implementation for Bayesian parameter estimation, MCMC sampling, Kalman
  filtering, and multi-factor screening.
- **Unified Schema Module**: Single source of truth (`finance_ml.core.schema`) for 500+ financial columns, ensuring
  strict alignment between SQL databases and Python data structures.
- **Advanced Feature Engineering**: Delivers 350+ schema-aligned features across 21 categories, including price
  dynamics, fiscal calendar, dividend timing, EPS trajectory, and cash flow temporal signals.
- **Flexible ETL Pipeline**: Decoupled configuration handling multiple data sources (CSV, SQL) with built-in currency
  conversion and data validation.
- **Interactive Dashboards**: Integrated Streamlit, Dash, and Plotly applications for market monitoring, earnings
  analytics, and portfolio visualization.

## Tech Stack

| Category            | Technologies                                                                                 |
|:--------------------|:---------------------------------------------------------------------------------------------|
| **Language**        | Python 3.12 / 3.13 / 3.14                                                                    |
| **Package Manager** | `pip`, `Pipfile` (pipenv), `pyproject.toml` (setuptools)                                     |
| **ML Frameworks**   | `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `Optuna`, `SHAP`, `TensorFlow` (optional) |
| **Data Processing** | `pandas`, `NumPy`, `SciPy`, `statsmodels`, `numba`, `imbalanced-learn`                       |
| **Visualization**   | `Plotly`, `Matplotlib`, `Seaborn`                                                            |
| **Dashboards**      | `Streamlit`, `Dash`                                                                          |
| **Database**        | `PostgreSQL` (psycopg2), `SQLAlchemy`, `SQLite`                                              |
| **Testing**         | `pytest`, `unittest`                                                                         |
| **Code Quality**    | `Black`, `Flake8`, `isort`, `Mypy`                                                           |
| **Utilities**       | `tqdm`, `joblib`, `xlsxwriter`, `psutil`                                                     |

## Requirements

- **Python**: Version 3.12, 3.13, or 3.14 (`>=3.12,<3.15`).
- **Operating System**: Windows (primary support via PowerShell), Linux, or macOS.
- **Dependencies**: Managed via `requirements.txt`, `Pipfile`, or `pyproject.toml`.

## Setup

### Automated Setup (Recommended)

Use the provided setup script to initialize the complete environment:

```powershell
python tools\setup_environment.py
```

This script detects your Python version, creates a virtual environment, installs all dependencies, and configures
environment variables.

### Manual Setup

1. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. **Install core dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **(Optional) Install development and dashboard tools:**
   ```powershell
   pip install -e ".[dev,dashboards,database]"
   ```

## Execution & Entry Points

### CLI Entry Points

The package provides several command-line entry points:

| Command               | Description                                          |
|:----------------------|:-----------------------------------------------------|
| `finance-ml`          | Main analysis pipeline execution (loading to models) |
| `finance-ml-analyze`  | Quick data analysis and exploratory profiling        |
| `finance-ml-validate` | Data validation and schema alignment check           |

### Main Execution Scripts

| Script                             | Description                                              |
|:-----------------------------------|:---------------------------------------------------------|
| `finance_ml_analytics_platform.py` | Complete 8-phase ML workflow and portfolio optimization  |
| `market_analytics.py`              | Demonstration of the refactored modular market analytics |
| `ml_finance_model_main.py`         | Alternative ML pipeline execution script                 |

### Interactive Dashboards

| Dashboard              | Run Command                                              |
|:-----------------------|:---------------------------------------------------------|
| **Streamlit App**      | `streamlit run finance_ml/dashboards/streamlit_app.py`   |
| **Equities Dashboard** | `python finance_ml/dashboards/equities_dashboard_app.py` |
| **Dash App**           | `python finance_ml/dashboards/dash_app.py`               |

### Jupyter Notebooks

Key notebooks for experimentation and exploration:
- `ml_finance_model_main.ipynb`: Core ML pipeline execution and experimentation.
- `financial_market_statistical_analysis.ipynb`: Advanced statistical analysis using the new analytics modules.
- `etl_data_explorer.ipynb`: Interactive data exploration and ETL testing.
- `stock_analytics.ipynb`: Comprehensive stock market analysis.
- `portfolio_optimization_risk_management.ipynb`: Portfolio optimization and risk analysis.

## Market Analytics Implementation

The market analytics logic has been refactored into a modular structure within `finance_ml.analytics`. This architecture
improves maintainability, performance, and reusability.

### Module Architecture

- **`data_utils`**: Data loading from PostgreSQL/materialized views, backfilling feature columns, and schema validation.
- **`statistical_analysis`**: Advanced methods including Bayesian parameter estimation, MCMC (Metropolis-Hastings,
  Student-t), Hierarchical Bayesian modeling, and Kalman filtering.
- **`screening`**: Multi-factor stock screening (Quality, Value, Growth, Momentum, Dividends) and composite ranking.
- **`feature_analytics`**: Core interactive Plotly dashboards and price target simulations.
- **`optimized_ops`**: Performance-critical operations using Numba-acceleration (Monte Carlo) and vectorized
  computations.
- **`visualizations/`**: Category-specific charting modules (Profitability, Technical, Temporal Analysis).

For more details, see [Market Analytics Refactoring Guide](docs/improvement_plan/market_analysis_refactoring_guide.md).

## Scripts & Utilities

Useful utility scripts located in `tools/`:

| Script                         | Purpose                                               |
|:-------------------------------|:------------------------------------------------------|
| `setup_environment.py`         | Full environment and dependency setup                 |
| `run_fast_tests.py`            | Quick verification of utility modules                 |
| `run_earnings_monitor.py`      | Monitors and visualizes company earnings events       |
| `setup_dashboard_assets.py`    | Syncs pipeline results with dashboard interface       |
| `load_equities_data.py`        | Loads and processes equities data into the system     |
| `validate_schema_alignment.py` | Validates SQL/Python schema alignment                 |
| `import_sqlite.py`             | Imports data into SQLite for local development        |
| `update_notebooks.py`          | Updates notebooks to align with latest schema changes |
| `cleanup_environments.py`      | Cleans up virtual environments and temporary caches   |

## Environment Variables

Configuration is managed via environment variables or the `environment_variables.txt` file:

| Variable               | Description                                     | Default   |
|:-----------------------|:------------------------------------------------|:----------|
| `LOG_LEVEL`            | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | `INFO`    |
| `TF_CPP_MIN_LOG_LEVEL` | TensorFlow log level (0=DEBUG to 3=ERROR)       | `2`       |
| `DATA_DIR`             | Directory for input data                        | `data`    |
| `MODEL_DIR`            | Directory for saved models                      | `models`  |
| `OUTPUT_DIR`           | Directory for generated reports                 | `outputs` |
| `DB_URL`               | SQLAlchemy connection URL                       | —         |
| `MODEL_VERSION`        | Identifier for current model version            | `v9_11`   |
| `RANDOM_SEED`          | Random seed for reproducibility                 | `42`      |

## Testing

The project uses `pytest` for comprehensive testing.

```powershell
# Run all tests
pytest

# Run fast tests (utility modules only)
python tools\run_fast_tests.py

# Run with coverage report
pytest --cov=finance_ml --cov-report=html
```

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
│   │   ├── models/             # Model training and evaluation
│   │   └── analytics/          # Phase 9.7 stock selection and mispricing
│   ├── analytics/              # NEW: Modular market analytics implementation
│   │   ├── statistical_analysis.py
│   │   ├── screening.py
│   │   └── visualizations/     # Specialized charting modules
│   └── dashboards/             # Streamlit, Dash, and Equities dashboards
├── tests/                      # Unit, integration, and regression tests
├── tools/                      # Utility and setup scripts
├── docs/                       # Documentation and guidelines
├── data/                       # Local data storage
├── models/                     # Saved model artifacts
├── outputs/                    # Generated reports and visualizations
├── pyproject.toml              # Build system and project metadata
├── Pipfile                     # Pipenv dependency management
└── requirements.txt            # Unified dependency list
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## TODOs

- [ ] **Docker Support**: Add Dockerfile and Docker Compose for one-click deployment.
- [ ] **Real-time Data**: Implement WebSocket support for live market data ingestion.
- [ ] **API Documentation**: Generate and host KDoc-style API documentation.
- [ ] **REST API**: Develop a FastAPI wrapper for the screening and analytics modules.
- [ ] **Test Coverage**: Increase coverage for the new `finance_ml.analytics` sub-modules.
