# Finance ML Analytics Platform

A comprehensive, production-ready platform for equity screening, feature engineering, and machine learning modeling across global financial markets.

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Package Version](https://img.shields.io/badge/version-0.9.5-green)](https://github.com/finance-ml/analytics-platform)

## Overview

The Finance ML Analytics Platform is a robust solution for financial data analysis, machine learning modeling, and portfolio optimization. It implements a structured **8-Phase ML Workflow** (Phases 9.1–9.8) to ensure data quality, advanced feature engineering, and reliable model evaluation, followed by a **7-Phase Portfolio Optimization** module.

### Key Features

- **8-Phase ML Workflow**: Covers data ingestion, preprocessing (winsorization, 6-step imputation), exploratory data analysis (EDA), advanced feature engineering, feature selection, model training (regression/classification), and error analysis.
- **7-Phase Portfolio Optimization**: Implements stock selection, return prediction, risk-adjusted optimization (Efficient Frontier), backtesting, and interactive dashboards.
- **Probabilistic ML Models**: Bayesian parameter estimation, MCMC sampling, Kalman filtering, Monte Carlo simulation, credit risk, dividend safety, and accounting anomaly detection.
- **Unified Schema Module**: Single source of truth for financial columns, ensuring alignment between SQL databases and Python data structures.
- **Advanced Feature Engineering**: Delivers 350+ schema-aligned features across 21 categories, including price dynamics, fiscal calendar, dividend timing, EPS trajectory, and cash flow temporal signals.
- **Flexible ETL Pipeline**: Decoupled configuration handling multiple data sources (CSV, SQL) with built-in currency conversion and data validation.
- **Interactive Dashboards**: Integrated Streamlit, Dash, and Plotly applications for market monitoring, earnings analytics, and portfolio visualization.

## Tech Stack

| Category            | Technologies                                                                       |
|:--------------------|:-----------------------------------------------------------------------------------|
| **Language**        | Python 3.12 / 3.13 / 3.14                                                          |
| **Package Manager** | `pip`, `pipenv` (`Pipfile`), `setuptools` (`pyproject.toml`)                       |
| **ML Frameworks**   | `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `Optuna`, `SHAP`, `TensorFlow` |
| **Bayesian**        | `PyMC`, `PyTensor`, `ArviZ`, `xarray`                                              |
| **Data Processing** | `pandas`, `NumPy`, `SciPy`, `statsmodels`, `numba`, `imbalanced-learn`             |
| **Visualization**   | `Plotly`, `Matplotlib`, `Seaborn`                                                  |
| **Dashboards**      | `Streamlit`, `Dash`, `dash-bootstrap-components`                                   |
| **Database**        | `PostgreSQL` (`psycopg2`), `SQLAlchemy`, `SQLite`                                  |
| **Testing**         | `pytest`, `unittest`                                                               |
| **Code Quality**    | `Black`, `Flake8`, `isort`, `Mypy`                                                 |
| **Utilities**       | `tqdm`, `joblib`, `xlsxwriter`, `psutil`, `forex-python`, `python-dotenv`          |

## Requirements

- **Python**: Version 3.12, 3.13, or 3.14 (`>=3.12,<3.15`).
- **Operating System**: Windows (primary support via PowerShell), Linux, or macOS.
- **Dependencies**: Managed via `requirements.txt`, `Pipfile`, or `pyproject.toml`.

### Python-Version-Gated Dependencies

Some packages are restricted to `python_version < '3.14'`:
`catboost`, `shap`, `streamlit`, `tensorflow`, `scikeras`, `numba`.

## Setup

### Quick Setup

```powershell
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Set environment variables for the current session
. .\set_env.ps1
```

### Editable Install (Recommended for Development)

```powershell
# Install the package in editable mode
pip install -e .

# Install with optional dependency groups
pip install -e ".[dev,dashboards,database,performance,tensorflow]"
```

### Optional Dependency Groups (`pyproject.toml`)

| Group          | Packages                                                  |
|:---------------|:----------------------------------------------------------|
| `dev`          | pytest, pytest-cov, black, flake8, mypy, isort, pip-tools |
| `dashboards`   | streamlit, dash                                           |
| `database`     | psycopg2-binary, SQLAlchemy                               |
| `tensorflow`   | tensorflow, scikeras                                      |
| `performance`  | numba                                                     |

## Execution & Entry Points

### CLI Entry Points

The package provides command-line entry points (defined in `pyproject.toml`):

| Command               | Target           | Description                          |
|:----------------------|:-----------------|:-------------------------------------|
| `finance-ml`          | `cli:main`       | Main analysis pipeline               |
| `finance-ml-analyze`  | `cli:analyze`    | Quick data analysis and exploration  |
| `finance-ml-validate` | `cli:validate`   | Data validation and schema check     |

> **Note:** Some CLI entry points may require fixes due to recent refactoring — see [TODOs](#todos).

### Main Execution Scripts

| Script                     | Description                                                   |
|:---------------------------|:--------------------------------------------------------------|
| `expected_returns_v3.py`   | Automated pipeline for expected returns analysis (v3.1)       |
| `market_analytics.py`      | Demonstration of the refactored modular market analytics      |
| `feature_analytics.py`     | Core interactive Plotly dashboards and price target simulations|
| `CurrencyDataProcessing.py`| Currency data transformation and DB import                    |
| `expected_returns.py`      | Legacy expected returns analysis script (v2.5)                |

### Interactive Dashboards

| Dashboard              | Run Command                                                    |
|:-----------------------|:---------------------------------------------------------------|
| **GEIB Dashboard**     | `python finance_ml/dashboards/geib_dash_app.py`               |
| **Streamlit App**      | `streamlit run finance_ml/dashboards/streamlit_app.py`         |
| **Equities Dashboard** | `python finance_ml/dashboards/equities_dashboard_app.py`       |
| **Dash App**           | `python finance_ml/dashboards/dash_app.py`                     |

### Jupyter Notebooks

Key notebooks for experimentation and exploration:

- `ExpectedReturnsAnalytics.ipynb` — Analysis of expected returns using the v3 analytics module.
- `stock_price_target_prediction.ipynb` — Phase 9.3+ ML modeling.
- `financial_market_statistical_analysis.ipynb` — Advanced statistical analysis.
- `etl_data_explorer.ipynb` — Interactive data exploration and ETL testing.
- `stock_analytics.ipynb` — Comprehensive stock market analysis.
- `portfolio_optimization_risk_management.ipynb` — Portfolio optimization and risk analysis.
- `accounting_anomaly_analysis.ipynb` — Accounting anomaly detection workflows.

## Environment Variables

Configuration is managed via `set_env.ps1` (dot-source to persist: `. .\set_env.ps1`).
Reference values are listed in `environment_variables.txt`.

| Variable                   | Description                              | Default / Example                                          |
|:---------------------------|:-----------------------------------------|:-----------------------------------------------------------|
| `LOG_LEVEL`                | Python logging level                     | `INFO`                                                     |
| `DATA_DIR`                 | Local data storage directory             | `data`                                                     |
| `MODEL_DIR`                | Saved model artifacts directory          | `regression`                                               |
| `CACHE_DIR`                | Cache directory                          | `.cache`                                                   |
| `OUTPUT_DIR`               | Generated reports / visualizations       | `outputs`                                                  |
| `DB_URL`                   | SQLAlchemy database connection URL       | `postgresql+psycopg2://user:pass@localhost:5432/postgres`  |
| `DB_EQUITIES_SCHEMA`       | PostgreSQL schema name                   | `public`                                                   |
| `DB_TABLE`                 | Database table name                      | `equities`                                                 |
| `DB_ANALYTICS_SCHEMA`      | PostgreSQL analytics schema              | `analytics`                                                |
| `MODEL_VERSION`            | Active model version tag                 | `v9_11`                                                    |
| `RANDOM_SEED`              | Reproducibility seed                     | `42`                                                       |
| `N_JOBS`                   | Number of parallel jobs (`-1` = all)     | `-1`                                                       |
| `GEIB_DASHBOARD`           | Enable GEIB Dashboard features           | `true`                                                     |
| `ENABLE_INTERACTIVE_PLOTS` | Toggle interactive visualizations        | `true`                                                     |

## Testing

The project uses `pytest` as the primary testing framework (configured in `pyproject.toml`), with `unittest.TestCase` style also supported. The `tests/` directory contains **159 test files**.

```powershell
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=finance_ml --cov-report=html

# Run a specific test file
pytest tests\test_screening.py

# Fast subset (avoids heavy ML training)
python tools\run_fast_tests.py
```

### Adding New Tests

- Place tests in the `tests/` directory; name files `test_*.py`.
- Use small, deterministic samples; mock external services.
- Aim for ≥ 80% coverage on new code.

## Project Structure

```text
Finance_Analytics_Platform/
├── finance_ml/                         # Core Python package
│   ├── analytics/                      # Modular market analytics (screening, stats, viz)
│   │   ├── data_utils.py               # Data loading, preprocessing, export
│   │   ├── statistical_analysis.py     # Bayesian, MCMC, Kalman, Copula
│   │   ├── screening.py               # Multi-factor stock screeners
│   │   ├── feature_analytics.py       # Interactive Plotly dashboards
│   │   ├── probability_analytics.py   # Probability models
│   │   ├── inference_schema.py        # ArviZ / xarray InferenceData bridge
│   │   ├── optimized_ops.py           # Performance optimizations
│   │   └── visualizations/            # Modular visualization sub-package
│   ├── core/                           # Shared constants and unified schema
│   │   ├── schema.py                  # COLUMN_SCHEMA — single source of truth
│   │   ├── constants.py               # Project-wide constants
│   │   └── schema_helpers.py          # Schema utility functions
│   ├── dashboards/                     # Streamlit, Dash, and GEIB dashboards
│   ├── etl/                            # ETL pipeline and data transformation
│   ├── features/                       # Feature engineering (advanced, basic)
│   ├── ml_workflow/                    # 8-phase ML workflow implementation
│   │   ├── preprocessing/             # Data cleaning, imputation, scaling
│   │   ├── eda/                       # Exploratory Data Analysis
│   │   ├── classification/            # Phase 9.4 Classification
│   │   ├── regression/                # Phase 9.5 Regression
│   │   ├── evaluation/                # Phase 9.6 Model Evaluation
│   │   ├── analytics/                 # Phase 9.7 Stock selection and mispricing
│   │   ├── reporting/                 # Phase 9.8 Reporting and export
│   │   ├── quality/                   # Data quality checks
│   │   └── validation/                # Validation utilities
│   ├── probabilistic_ml_model/         # Bayesian / probabilistic ML models
│   │   ├── pml_models/                # Model implementations (Monte Carlo, Kalman, DCF, etc.)
│   │   ├── data_utils/                # Data loading and inference schema
│   │   ├── statistical_functions/     # Probability analytics, screening, statistical analysis
│   │   └── visualizations/            # ArviZ diagnostics, earnings quality charts
│   └── cli.py                          # CLI entry point definitions
├── feature_factory/                    # Feature calculation utilities (Beta/CAPM, DCF, Monte Carlo)
├── tests/                              # Unit, integration, and regression tests (159 files)
├── tools/                              # Utility and automation scripts
├── docs/                               # Documentation and guidelines
├── data/                               # Local data storage
├── models/                             # Saved model artifacts
├── outputs/                            # Generated reports and visualizations
├── expected_returns_v3.py              # Main expected-returns pipeline (v3.1)
├── market_analytics.py                 # Modular market analytics demo
├── feature_analytics.py                # Interactive dashboards and simulations
├── CurrencyDataProcessing.py           # Currency data transformation
├── set_env.ps1                         # PowerShell env var setup script
├── environment_variables.txt           # Environment variable reference
├── pyproject.toml                      # Build system and project metadata (v0.9.5)
├── Pipfile                             # Pipenv dependency management
├── requirements.txt                    # Unified dependency list
└── LICENSE                             # MIT License
```

### Critical Imports

```python
# Schema (single source of truth for column definitions)
from finance_ml.core.schema import COLUMN_SCHEMA, normalize_column_name, list_price_cols

# ETL pipeline
from finance_ml.etl import run_etl_pipeline, ETLConfig

# Feature engineering
from finance_ml.ml_workflow.features.api import build_features

# Analytics
from finance_ml.analytics.data_utils import load_feature_data_from_db, load_all_feature_views
from finance_ml.analytics.screening import create_enhanced_screener
from finance_ml.analytics.statistical_analysis import bayesian_category_analysis
```

## 8-Phase ML Workflow

| Phase   | Description                                      | Key Module                              |
|:--------|:-------------------------------------------------|:----------------------------------------|
| **9.1** | Loading and preprocessing with 6-step imputation | `finance_ml.etl`                        |
| **9.2** | Enhanced exploratory data analysis               | `finance_ml.ml_workflow.eda`            |
| **9.3** | Advanced feature engineering                     | `finance_ml.ml_workflow.features`       |
| **9.4** | Multi-class event classification                 | `finance_ml.ml_workflow.classification` |
| **9.5** | Sector-optimized regression with quantile models | `finance_ml.ml_workflow.regression`     |
| **9.6** | Model evaluation and error analysis              | `finance_ml.ml_workflow.evaluation`     |
| **9.7** | Identification of under/overvalued stocks        | `finance_ml.ml_workflow.analytics`      |
| **9.8** | Comprehensive analytics and reporting            | `finance_ml.ml_workflow.reporting`      |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## TODOs

- [ ] **CLI Fix**: Resolve import issues in `finance_ml/cli.py` and ensure consistency between `pyproject.toml` entry points and actual module paths.
- [ ] **Restore Missing Scripts**: Locate or re-implement `finance_ml_analytics_platform.py` and `ml_finance_model_main.py` (previously in project root).
- [ ] **Recreate Setup Script**: Re-implement `tools/setup_environment.py` for automated environment initialization.
- [ ] **Docker Support**: Add Dockerfile and Docker Compose for one-click deployment.
- [ ] **API Documentation**: Generate and host API documentation.
- [ ] **REST API**: Develop a FastAPI wrapper for the screening and analytics modules.
- [ ] **Test Coverage**: Increase coverage for the `finance_ml.analytics` and `finance_ml.probabilistic_ml_model` sub-modules.
