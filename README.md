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
- **Advanced Market Analytics**: Modular implementation for Bayesian parameter estimation, MCMC sampling, Kalman filtering, and multi-factor screening.
- **Unified Schema Module**: Single source of truth for financial columns, ensuring alignment between SQL databases and Python data structures.
- **Advanced Feature Engineering**: Delivers 350+ schema-aligned features across 21 categories, including price dynamics, fiscal calendar, dividend timing, EPS trajectory, and cash flow temporal signals.
- **Flexible ETL Pipeline**: Decoupled configuration handling multiple data sources (CSV, SQL) with built-in currency conversion and data validation.
- **Interactive Dashboards**: Integrated Streamlit, Dash, and Plotly applications for market monitoring, earnings analytics, and portfolio visualization.

## Tech Stack

| Category            | Technologies                                                                      |
|:--------------------|:----------------------------------------------------------------------------------|
| **Language**        | Python 3.12 / 3.13 / 3.14                                                         |
| **Package Manager** | `pip`, `pipenv` (`Pipfile`), `setuptools` (`pyproject.toml`)                      |
| **ML Frameworks**   | `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `Optuna`, `SHAP`, `TensorFlow` |
| **Data Processing** | `pandas`, `NumPy`, `SciPy`, `statsmodels`, `numba`, `imbalanced-learn`            |
| **Visualization**   | `Plotly`, `Matplotlib`, `Seaborn`                                                 |
| **Dashboards**      | `Streamlit`, `Dash`                                                               |
| **Database**        | `PostgreSQL` (`psycopg2`), `SQLAlchemy`, `SQLite`                                 |
| **Testing**         | `pytest`, `unittest`                                                              |
| **Code Quality**    | `Black`, `Flake8`, `isort`, `Mypy`                                                |
| **Utilities**       | `tqdm`, `joblib`, `xlsxwriter`, `psutil`, `arviz`                                 |

## Requirements

- **Python**: Version 3.12, 3.13, or 3.14 (`>=3.12,<3.15`).
- **Operating System**: Windows (primary support via PowerShell), Linux, or macOS.
- **Dependencies**: Managed via `requirements.txt`, `Pipfile`, or `pyproject.toml`.

## Setup

### Manual Setup (Recommended)

1. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. **Install core dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Install the package in editable mode:**
   ```powershell
   pip install -e .
   ```
4. **(Optional) Install development and dashboard tools:**
   ```powershell
   pip install -e ".[dev,dashboards,database,performance,tensorflow]"
   ```

## Execution & Entry Points

### CLI Entry Points

The package provides several command-line entry points (defined in `pyproject.toml`).
*Note: Some entry points may currently have issues due to recent refactoring.*

| Command | Description |
|:---|:---|
| `finance-ml` | Main analysis pipeline execution (loading to models) |
| `finance-ml-analyze` | Quick data analysis and exploratory profiling |
| `finance-ml-validate` | Data validation and schema alignment check |

### Main Execution Scripts

| Script | Description |
|:---|:---|
| `expected_returns_v3.py` | Automated pipeline for expected returns analysis (v3.0+ platform) |
| `market_analytics.py` | Demonstration of the refactored modular market analytics |
| `feature_analytics.py` | Core interactive Plotly dashboards and price target simulations |
| `CurrencyDataProcessing.py` | Currency data transformation and DB import |
| `expected_returns.py` | Legacy expected returns analysis script (v2.5) |

### Interactive Dashboards

| Dashboard | Run Command |
|:---|:---|
| **GEIB Dashboard** | `python finance_ml/dashboards/geib_dash_app.py` |
| **Streamlit App** | `streamlit run finance_ml/dashboards/streamlit_app.py` |
| **Equities Dashboard** | `python finance_ml/dashboards/equities_dashboard_app.py` |
| **Dash App** | `python finance_ml/dashboards/dash_app.py` |

### Jupyter Notebooks

Key notebooks for experimentation and exploration:
- `ExpectedReturnsAnalytics.ipynb`: Analysis of expected returns using the v3 analytics module.
- `stock_price_target_prediction.ipynb`: Phase 9.3 and beyond ML modeling.
- `financial_market_statistical_analysis.ipynb`: Advanced statistical analysis using the new analytics modules.
- `etl_data_explorer.ipynb`: Interactive data exploration and ETL testing.
- `stock_analytics.ipynb`: Comprehensive stock market analysis.
- `portfolio_optimization_risk_management.ipynb`: Portfolio optimization and risk analysis.

## Market Analytics Implementation

The market analytics logic is organized into a modular structure within `finance_ml.analytics`.

### Module Architecture

- **`data_utils`**: Data loading, backfilling feature columns, and schema validation.
- **`statistical_analysis`**: Advanced methods including Bayesian parameter estimation, MCMC, and Kalman filtering.
- **`probability_analytics`**: Bayesian earnings beat probability models, credit risk, and dividend safety analysis.
- **`screening`**: Multi-factor stock screening (Quality, Value, Growth, Momentum, Dividends) and composite ranking.
- **`feature_analytics`**: Core interactive Plotly dashboards and price target simulations.
- **`visualizations/`**: Category-specific charting modules (Profitability, Technical, Temporal Analysis).

## Environment Variables

Configuration is managed via environment variables or an `environment_variables.txt` file:

| Variable | Description | Default |
|:---|:---|:---|
| `DB_URL` | SQLAlchemy connection URL | — |
| `DB_EQUITIES_SCHEMA` | PostgreSQL schema name | `public` |
| `DB_ANALYTICS_SCHEMA` | PostgreSQL analytics schema | `analytics` |
| `LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `TF_CPP_MIN_LOG_LEVEL` | TensorFlow log level (0=DEBUG to 3=ERROR) | `2` |
| `DATA_DIR` | Directory for input data | `data` |
| `MODEL_DIR` | Directory for saved models | `models` |
| `OUTPUT_DIR` | Directory for generated reports | `outputs` |
| `MODEL_VERSION` | Identifier for current model version | `v9_11` |
| `RANDOM_SEED` | Random seed for reproducibility | `42` |
| `N_JOBS` | Number of parallel jobs | `-1` |
| `GEIB_DASHBOARD` | Enable GEIB Dashboard features | `true` |

## Testing

The project uses `pytest` for comprehensive testing.

```powershell
# Run all tests
pytest

# Run with coverage report
pytest --cov=finance_ml --cov-report=html
```

## Project Structure

```text
Finance_Analytics_Platform/
├── finance_ml/                 # Core Python package
│   ├── analytics/              # Modular market analytics implementation
│   ├── core/                   # Shared constants and unified schema
│   ├── dashboards/             # Streamlit, Dash, and GEIB dashboards
│   ├── etl/                    # ETL pipeline and data transformation
│   ├── features/               # Feature engineering (advanced, basic)
│   ├── ml_workflow/            # 8-phase ML workflow implementation
│   │   ├── preprocessing/      # Data cleaning, imputation, scaling
│   │   ├── eda/                # Exploratory Data Analysis
│   │   ├── classification/     # Phase 9.4 Classification
│   │   ├── regression/         # Phase 9.5 Regression
│   │   ├── evaluation/         # Phase 9.6 Model Evaluation
│   │   ├── analytics/          # Phase 9.7 Stock selection and mispricing
│   │   └── reporting/          # Phase 9.8 Reporting and export
│   └── cli.py                  # CLI entry point definitions
├── tests/                      # Unit, integration, and regression tests
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

- [ ] **CLI Fix**: Resolve non-existent `finance_ml.ml_workflow.models` and `archive` imports in `finance_ml/cli.py` and ensure consistency between `pyproject.toml` and `cli.py`.
- [ ] **Restore Missing Scripts**: Locate or re-implement `finance_ml_analytics_platform.py` and `ml_finance_model_main.py` (previously in project root).
- [ ] **Recreate Setup Script**: Re-implement `tools/setup_environment.py` for automated environment initialization.
- [ ] **Docker Support**: Add Dockerfile and Docker Compose for one-click deployment.
- [ ] **API Documentation**: Generate and host KDoc-style API documentation.
- [ ] **REST API**: Develop a FastAPI wrapper for the screening and analytics modules.
- [ ] **Test Coverage**: Increase coverage for the new `finance_ml.analytics` sub-modules.
