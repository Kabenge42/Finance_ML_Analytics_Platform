# Finance ML Analytics Platform

**Version 0.9.5** — Comprehensive ML Platform for Equity Analysis and Price Target Prediction

> **Documentation Last Updated:** 2025-12-18  
> **Latest Release**: v0.9.5 (see CHANGELOG.md)  
> **Model Version**: v9_10  
> **Note**: Package versions are synchronized across pyproject.toml, CHANGELOG.md, and environment_variables.txt where
> applicable.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%20|%203.13%20|%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Database Setup](#database-setup)
- [Usage](#usage)
    - [Jupyter Notebook Workflow](#jupyter-notebook-workflow)
    - [Python Script](#python-script)
    - [CLI Tools](#cli-tools)
    - [Interactive Dashboards](#interactive-dashboards)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Scripts and Tools](#scripts-and-tools)
- [Recent Updates](#recent-updates)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Finance ML Analytics Platform is a comprehensive toolkit for quantitative equity analysis combining unified data
pipelines, modular Python packages, interactive notebooks, and production-ready CLI tools.

### Business Objective

**Primary Goal**: Predict Stock Price Targets for all stocks in the portfolio to support investment decisions and
portfolio optimization.

**Target Variable**: "Predicted Price Target" for regression modeling

The platform implements a sophisticated **8-phase ML workflow** (Phase 9.1 - 9.8) aligned with industry best practices:

1. **Phase 9.1**: Unified ETL pipeline with semantic transformations and 6-step imputation
    - Single entry point: `etl_with_features()` for complete ETL + feature engineering
    - Semantic-aware transformations (price column preservation, log-transforms for market values)
   - See `docs/code_guidelines.md` v1.10 for code conventions
   - See `docs/ml_workflow_guidelines.md` for detailed acceptance criteria, success metrics, and validation checkpoints
2. **Phase 9.2**: Enhanced exploratory data analysis with statistical testing and benchmarking
3. **Phase 9.3**: Advanced feature engineering with sector-specific optimizations
4. **Phase 9.4**: Multi-class event classification using neural networks and ensembles
5. **Phase 9.5**: Sector-optimized regression models with hyperparameter tuning and quantile models
6. **Phase 9.6**: Model evaluation and comprehensive error analysis
7. **Phase 9.7**: Identification of under/overvalued stocks with visualization and analyst comparison
8. **Phase 9.8**: Comprehensive analytics and reporting

---

## Stack & Entry Points (Quick)

- Language: Python 3.12–3.14
- Package manager: pip (requirements.txt) + pyproject.toml (PEP 621)
- Core frameworks: pandas, scikit-learn, XGBoost/LightGBM/CatBoost, plotly/matplotlib, SHAP
- Optional: TensorFlow/Keras, SQLAlchemy + psycopg2 (PostgreSQL)
- Database: PostgreSQL (primary), SQLite (local quick start)
- CLI entry points (from pyproject.toml):
  - finance-ml — full pipeline
  - finance-ml-analyze — EDA/analytics only
  - finance-ml-validate — validation-only
- Main notebook: ml_finance_model_main.ipynb
- Script alternative: ml_finance_model_main.py

Note: If something appears missing for your environment, see environment_variables.txt and the Testing section’s
selective execution tips. TODO: Add containerization instructions (Docker) if/when available.

---

## Key Features

- 📊 **Data Management**: PostgreSQL/SQLite integration with unified all_stocks table (318 columns) combining four
  regional screening tables + CSV fallback for multi-region equity data (US, EU, APAC, ROTW)
- 🧹 **Data Quality**: 6-step imputation pipeline (zero-fill, KNN, price-based, median) with validation; outlier safety
  rails (winsorization, robust loss, clipping); schema-aware datatype detection
- 🔧 **Feature Engineering**: Phase 9.3 Schema Version 1.4 (343 columns, +33 new): Financial ratios, margins, volatility,
  technical indicators (EMA 20D/50D/100D/250D, 52W High/Low), valuation time-series (EV/Sales, EV/EBITDA, P/E extended),
  revenue forecasts (NTM, FY1E), dividend reliability (frequency, streak, coverage), employment dynamics (FY/FQ
  metrics), sector-specific features, **NEW: Earnings Quality Analytics** (33 features for EPS/revenue surprise,
  GAAP vs. Adjusted metrics, earnings quality scoring)
- 🤖 **ML Models**: Event classification, sector-optimized regression, quantile models with conformal calibration,
  stacking ensembles
- 📈 **Analytics**: Mispricing scores, stock ranking, analyst comparison, benchmarking, risk metrics
- 💼 **Portfolio Optimization**: Advanced methods (Black-Litterman, Risk Parity, Hierarchical Risk Parity), efficient
  frontier, maximum Sharpe ratio, minimum volatility, backtesting framework, performance attribution
- 📉 **Risk Metrics**: VaR, CVaR (Expected Shortfall), Sharpe ratio, Sortino ratio, maximum drawdown, tracking error,
  stress testing, Monte Carlo simulation
- 📊 **Interactive Dashboards**: Streamlit and Dash applications with portfolio & risk metrics visualization
- 🎯 **Stock Prediction**: End-to-end 8-phase workflow for price target prediction with standardized predictions schema
- 🔬 **Uncertainty Quantification**: Quantile regression + conformal prediction for calibrated 80% prediction intervals;
  interval coverage diagnostics, reliability diagrams, sector-level uncertainty analysis
- 🛡️ **Safety Rails & Monitoring**: Winsorization effect tracking, non-negativity constraint validation, outlier
  detection sensitivity analysis, interactive safety rails dashboards
- 🔍 **Data Split Validation**: CV fold overlap analysis, grouped stratification balance metrics, time-based leakage
  detection with severity assessment
- 📊 **Sector Bias Calibration**: Pre/post calibration bias estimation, versioned metrics persistence, MAE/MAPE trend
  visualization, interactive sector bias dashboards
- 📋 **Model Governance**: Stacking ensemble diagnostics, base model contribution analysis, auto-generated model cards
  (markdown), complete lineage tracking (datasets → features → models → artifacts), meta-learner error maps
- 📄 **Reporting**: Excel/PDF reports, interactive Plotly visualizations, valuation analysis, standardized predictions
  output, 30+ artifacts across 5 governance directories (uncertainty/, safety_rails/, splits/, calibration/,
  governance/)
- ⚙️ **Configuration**: Flexible config via environment variables and CLI options
- 🧪 **Tested**: Comprehensive unittest suite (128 test modules) with extensive coverage (≥80% target for new code);
  TDD conventions for uncertainty, safety rails, splits validation, calibration, governance, schema validation, and
  datatype detection
- 🚀 **CLI**: Three command-line tools for different workflows
- 🔍 **Model Interpretation**: SHAP analysis for explainability (with permutation importance fallback)

---

## Module Structure (v9_8 - Phase 9.1-9.8 Refactor)

The codebase follows a **phase-aligned architecture** with dedicated subpackages for each development phase:

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

### Phase-to-Module Mapping

| Phase   | Subpackage        | Purpose                                      | Import Prefix              | Key Functions                                                                                                                                 |
|---------|-------------------|----------------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| **9.1** | `preprocessing/`  | Data quality, imputation, outliers, scaling  | `preprocessing_*`          | `apply_enhanced_imputation_strategy_4step`, `preprocessing_calculate_quality`, `detect_outliers_iqr`, `winsorize_by_sector`, `scale_features` |
| **9.2** | `eda/`            | EDA reports, benchmarking, statistical tests | `generate_*`, `compare_*`  | `generate_eda_report`, `generate_benchmarking_report`, `compare_sector_distributions`                                                         |
| **9.3** | `features/`       | Feature engineering, importance, selection   | `features_*`, `engineer_*` | `features_build_comprehensive`, `features_importance_rf`, `engineer_valuation_ratios`, `engineer_analyst_quality_features`                    |
| **9.4** | `classification/` | Event labels, hyperparameter tuning, models  | `classification_*`         | `classification_create_enhanced_event_labels`, `classification_optimize_hyperparameters`                                                      |
| **9.5** | `regression/`     | Model training, quantile, constraints, I/O   | `regression_*`             | `regression_train_xgboost`, `regression_train_lightgbm`, `regression_compare_regressors`, `regression_save_model`                             |
| **9.6** | `evaluation/`     | Metrics, error analysis, segmentation        | `evaluation_*`             | `evaluation_comprehensive_metrics`, `evaluation_metrics_by_segment`                                                                           |
| **9.7** | `analytics/`      | Mispricing, rankings, portfolio, risk        | `analytics_*`              | `analytics_calculate_mispricing`, `analytics_rank_undervalued`, `analytics_rank_by_sector`                                                    |
| **9.8** | `reporting/`      | Dashboard data, quality alerts, exports      | `reporting_*`              | `reporting_financial_metrics`, `reporting_quality_alerts`, `reporting_plotly_data`                                                            |

### Import Examples

**Recommended Pattern (Package-Level Imports):**

```python
# All Phase 9.1-9.8 functions are available at package level with descriptive prefixes
from finance_ml import (
    # Phase 9.1: Preprocessing
    apply_enhanced_imputation_strategy_4step,
    preprocessing_calculate_quality,
    detect_outliers_iqr,
    winsorize_by_sector,
    scale_features,

    # Phase 9.2: EDA
    generate_eda_report,
    generate_benchmarking_report,

    # Phase 9.3: Features
    features_build_comprehensive,
    features_importance_rf,
    engineer_analyst_quality_features,

    # Phase 9.4: Classification
    classification_create_enhanced_event_labels,
    classification_optimize_hyperparameters,

    # Phase 9.5: Regression
    regression_train_xgboost,
    regression_compare_regressors,
    regression_save_model,

    # Phase 9.6: Evaluation
    evaluation_comprehensive_metrics,
    evaluation_metrics_by_segment,

    # Phase 9.7: Analytics
    analytics_calculate_mispricing,
    analytics_rank_undervalued,

    # Phase 9.8: Reporting
    reporting_financial_metrics,
    reporting_quality_alerts,
    )
```

**Alternative Pattern (Direct Subpackage Imports):**

```python
# Import directly from subpackages for more explicit organization
from finance_ml.ml_workflow.preprocessing import imputation, outliers, scaling
from finance_ml.ml_workflow.eda import eda, benchmarking
from finance_ml.ml_workflow.features import core, advanced, selection
from finance_ml.ml_workflow.classification import labels, tuning, models
from finance_ml.ml_workflow.regression import models, quantile, constraints
from finance_ml.ml_workflow.evaluation import metrics
from finance_ml.ml_workflow.analytics import mispricing
from finance_ml.ml_workflow.reporting import dashboard_data
```

**Unified ETL Pipeline (Recommended for Complete Preprocessing):**

```python
# Single entry point for all preprocessing + financial metrics (NEW in v0.9.2)
from finance_ml.ml_workflow.preprocessing.etl import (
    run_etl_pipeline,  # Core ETL pipeline
    etl_with_financial_metrics,  # Complete ETL + financial metrics in one call
    ETLConfig,  # Configuration dataclass
    ETLMetrics,  # Metrics tracking
    )

# Option 1: Complete ETL + financial metrics (recommended)
all_stocks_preprocessed, metrics = etl_with_financial_metrics(
        source='csv',
        data_dir='data/',
        compute_all_metrics=True,  # Valuation, profitability, growth, leverage
        output_dir='outputs/financial_metrics',
        return_metrics=True,
        )

# Option 2: Fine-grained control via ETLConfig
config = ETLConfig(
        apply_imputation=True,
        imputation_strategy='6step',
        compute_valuation_metrics=True,
        compute_profitability_metrics=True,
        compute_growth_metrics=True,
        compute_leverage_metrics=True,
        compute_target_vs_price=True,
        handle_sector_specific_metrics=True,
        generate_quality_alerts=True,
        generate_metrics_dashboard=True,
        )
all_stocks_preprocessed, metrics = run_etl_pipeline(
        source='csv', data_dir='data/', config=config, return_metrics=True
        )
```

### Migration Guide

**Old Pattern (Deprecated):**

```python
# These still work but trigger deprecation warnings
from finance_ml.ml_workflow.advanced_preprocessing import detect_outliers_iqr
from finance_ml.ml_workflow.advanced_features import build_comprehensive_features
from finance_ml.ml_workflow.advanced_models import train_xgboost_regressor
# Note: eval.py moved to analytics/eval.py (Phase 9.7)
from finance_ml.ml_workflow.analytics.eval import calculate_mispricing_score

# DEPRECATED in v0.9.2: financial_metrics_etl module consolidated into etl.py
from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
    run_financial_metrics_etl,  # ⚠️ Use etl_with_financial_metrics() instead
    FinancialMetricsETLConfig,  # ⚠️ Use ETLConfig with financial metrics flags
    )
```

**New Pattern (Recommended):**

```python
# Use package-level imports with descriptive prefixes
from finance_ml import (
    detect_outliers_iqr,  # Phase 9.1
    features_build_comprehensive,  # Phase 9.3
    regression_train_xgboost,  # Phase 9.5
    analytics_calculate_mispricing,  # Phase 9.7
    )
```

### Key Design Principles

✅ **Phase alignment**: Each subpackage maps directly to a business phase (9.1–9.8)  
✅ **Backward compatibility**: Old imports still work with deprecation warnings for 1-2 releases  
✅ **Clean imports**: All functions imported once at package level  
✅ **Descriptive prefixes**: Function names indicate their module (preprocessing_*, features_*, etc.)  
✅ **Consolidation**: Eliminated duplication across `features.py`/`advanced_features.py`, `models.py`/
`advanced_models.py`, etc.  
✅ **Testability**: Isolated modules are easier to unit test  
✅ **Maintainability**: Clear module boundaries and responsibilities

See `docs/improvement_plan/finance_ml_improvement_plan.md` for detailed migration guide and complete API reference.

---

## Technology Stack

### Language & Runtime

- **Python**: 3.12, 3.13, or 3.14 (officially supported per `pyproject.toml`; 3.10-3.11 may work but untested)
- **Package Manager**: pip with `requirements.txt` and `pyproject.toml`

### Core Libraries

- **Data**: pandas, numpy, scipy, statsmodels
- **ML Frameworks**: scikit-learn, imbalanced-learn
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Deep Learning** (optional): TensorFlow/Keras, scikeras
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboards**: streamlit, dash
- **Explainability**: SHAP
- **Utilities**: joblib, tqdm, xlsxwriter, psutil

### Database

- **Primary**: PostgreSQL 15+ (recommended for production)
- **Alternative**: SQLite 3 (for quick local testing)
- **Drivers**: psycopg2-binary, SQLAlchemy

### Development Tools

- **Testing**: unittest (built-in), pytest (optional), coverage
- **Code Quality**: black, flake8, mypy, isort
- **Notebooks**: Jupyter, notebook, ipykernel

---

## Requirements

### System Requirements

- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.12, 3.13, or 3.14 (officially supported)
- **PostgreSQL**: 15+ (optional; SQLite works for local testing)
- **Memory**: 8GB+ recommended for full dataset
- **Disk**: 2GB+ for project, data, and models

### Python Dependencies

Core dependencies are managed in `requirements.txt`. Optional extras include:

- **tensorflow**: TensorFlow/Keras for deep learning (optional)
- **database**: PostgreSQL drivers (psycopg2-binary, SQLAlchemy)
- **dev**: Testing and code quality tools (pytest, black, mypy, etc.)
- **notebook**: Jupyter notebook support
- **dashboards**: Streamlit and Dash for interactive dashboards

---

## Quick Start

```powershell
# 1. Clone repository (or download)
git clone https://github.com/Kabenge42/Finance_ML_Analytics_Platform.git
cd Finance_ML_Analytics_Platform

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Set up database (PostgreSQL or SQLite)
# PostgreSQL:
psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql
psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql

# PostgreSQL - Create unified all_stocks table (RECOMMENDED):
psql -h localhost -p 5432 -U postgres -d postgres -f all_stocks/all_stocks.sql
# This creates a unified 318-column table combining all four regional screening tables

# SQLite (alternative):
sqlite3 equities.sqlite ".read create_equities_schema_sqlite.sql"
sqlite3 equities.sqlite ".read import_equities_data_sqlite.sql"

# 5. Configure environment (optional)
# Edit environment_variables.txt and export or create .env

# 6. Run main notebook
jupyter notebook ml_finance_model_main.ipynb

# Or run as Python script
python ml_finance_model_main.py --data-source auto --limit 5000

# Or use CLI
finance-ml --data-source auto --output-dir outputs

# Or launch interactive dashboard
streamlit run finance_ml/dashboards/streamlit_app.py
```

### Data Loading Options

The platform supports three data loading methods:

```python
from finance_ml.ml_workflow.preprocessing import (
    load_from_csv,           # Load from CSV files in data/ directory
    load_from_db,            # Load from equities table with Region filter
    load_from_all_stocks,    # Load from unified all_stocks table (RECOMMENDED)
)

# Option 1: Load from unified all_stocks table (RECOMMENDED)
# This is the fastest and most efficient method after running all_stocks/all_stocks.sql
db_url = "postgresql+psycopg2://postgres:@localhost:5432/postgres"
all_stocks_df = load_from_all_stocks(db_url, limit=10000)
print(f"Loaded {len(all_stocks_df)} stocks from unified all_stocks table")
print(all_stocks_df['region'].value_counts())

# Option 2: Load from regional equities table
all_stocks_df = load_from_db(db_url, limit=10000)

# Option 3: Load from CSV files (fallback when database not available)
from pathlib import Path
data_dir = Path("data")
all_stocks_df = load_from_csv(data_dir, limit=10000)
```

**Unified all_stocks Table Benefits:**

- Single query instead of UNION ALL across four regional tables
- Faster query performance with optimized indexes
- Simplified data pipeline code
- Pre-joined 318-column schema (262 original + 48 Phase 9.3 additions)
- Primary key: (Ticker, Region) ensures data integrity

---

## Installation

### 1. Prerequisites

Ensure you have Python 3.12, 3.13, or 3.14 installed:

```powershell
python --version
```

### 2. Virtual Environment Setup

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux (bash):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```powershell
# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install all dependencies
pip install -r requirements.txt
```

**Optional**: Install with specific extras:

```powershell
# Install package in editable mode with extras
pip install -e ".[dev,database,notebook]"
```

#### Python 3.14 Compatibility Notes

As of 2025-11-21, Python 3.14 is officially supported by this project (requires-python >=3.12,<3.15). Some optional
packages still lag behind with prebuilt wheels on Windows for 3.14. We applied safe defaults in our dependency files to
ensure a smooth install:

- NumPy 2.x is used automatically on Python 3.14+; NumPy 1.26.x is used on earlier supported versions.
- SHAP 0.50.0 (with enhanced explainability features and improved performance) is gated on Python < 3.14 due to its
  `numba` dependency not publishing 3.14 wheels yet.
- CatBoost is gated on Python < 3.14 because cp314 wheels are not yet available on Windows; this avoids slow/fragile
  source builds.
- Streamlit is gated on Python < 3.14 to avoid `pyarrow` source builds until cp314 wheels are broadly available.
- TensorFlow/Keras extras are gated on Python < 3.14 pending official 3.14 wheel availability.

What this means for you:

- On Python 3.14, `pip install -r requirements.txt` installs the full core stack (
  NumPy/Pandas/SciPy/Scikit-Learn/XGBoost/LightGBM/etc.) without attempting to build problematic optional packages.
- If you need any gated optional package immediately, use Python 3.13 for that environment or install the package later
  once official 3.14 wheels are released.

Quick Windows setup for Python 3.14:

```powershell
py -3.14 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip cache purge
pip install -r requirements.txt
```

### 4. Verify Installation

```powershell
# Run smoke test
python -m unittest tests.test_coverage_smoke -v

# Check CLI tools
finance-ml --help
```

---

## Database Setup

### Option 1: PostgreSQL (Recommended)

1. **Install PostgreSQL** (version 15+):
    - Download from [postgresql.org](https://www.postgresql.org/download/)
    - Ensure `psql` is on PATH

2. **Start PostgreSQL service** and verify access with user `postgres`

3. **Create schema and table**:

```powershell
psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql
```

4. **Import CSV data**:

The comprehensive import script handles all regions with proper NULL handling:

```powershell
psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
```

**Optional**: Validate CSV data before import:

```powershell
python tools/validate_csv_import.py
```

5. **Connection details**:
    - JDBC URL: `jdbc:postgresql://localhost:5432/postgres`
    - SQLAlchemy URL: `postgresql+psycopg2://postgres:password@localhost:5432/postgres`

### Option 2: SQLite (Quick Local Testing)

1. **Create schema**:

```powershell
sqlite3 equities.sqlite ".read create_equities_schema_sqlite.sql"
```

2. **Import data**:

```powershell
# Using SQL script (recommended)
sqlite3 equities.sqlite ".read import_equities_data_sqlite.sql"

# Or using Python importer
python tools/import_sqlite.py --db equities.sqlite --data-dir data
```

3. **Connection**:
    - SQLAlchemy URL: `sqlite:///equities.sqlite`

---

## Usage

### Jupyter Notebook Workflow

The primary workflow is notebook-based for exploration and experimentation.

**Main Notebook**: `ml_finance_model_main.ipynb`

```powershell
# Launch Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

Open `ml_finance_model_main.ipynb` and run cells sequentially. The notebook includes:

- Configuration and setup
- Data loading from database or CSV
- 6-step imputation pipeline
- EDA with visualizations
- Feature engineering
- Classification models
- Regression models (sector-optimized)
- Evaluation and error analysis
- Stock ranking and valuation
- Report generation

Note on Notebook Best Practices (see docs/code_guidelines.md §8):

- Centralized Configuration Constants: define all constants once (TARGET_COL, TEST_SIZE, CV_FOLDS, QUANTILES,
  RANDOM_SEED, winsorization bounds, sector/portfolio constraints).
- DataFrame Stage Naming (no in-place mutation):
  all_stocks_raw → all_stocks_normalized → all_stocks_typed → all_stocks_winsorized → all_stocks_imputed →
  all_stocks_scaled → all_stocks_features → all_stocks_enhanced.
- Magic Numbers Policy: replace meaningful numeric literals (e.g., 0.2 test_size, 0.25 max sector weight) with named
  constants.

**Other Notebooks**:

- `etl_data_explorer.ipynb` — ETL pipeline exploration and data analysis
- `stock_analytics.ipynb` — Stock analytics and visualization
- `portfolio_optimization_risk_management.ipynb` — Portfolio optimization and risk management
- `stock_price_target_prediction.ipynb` — Stock price target prediction workflow
- `ml_finance_model_main2_0.ipynb` — Alternative notebook version

### Python Script

Run the pipeline as a standalone Python script with CLI arguments.

**Script**: `ml_finance_model_main.py`

```powershell
# Run with auto data source selection
python ml_finance_model_main.py --data-source auto --limit 5000 --out-dir outputs

# Force database
python ml_finance_model_main.py --data-source db --db-url "postgresql+psycopg2://postgres:@localhost:5432/postgres"

# Force CSV
python ml_finance_model_main.py --data-source csv --out-dir outputs

# Dry run (skip training)
python ml_finance_model_main.py --data-source auto --dry-run
```

**Arguments**:

- `--data-source {auto|csv|db}` — Data source (default: auto)
- `--db-url URL` — Database connection string (or use `DB_URL` env var)
- `--limit N` — Limit rows for testing
- `--out-dir PATH` — Output directory (default: outputs)
- `--dry-run` — Skip model training

### CLI Tools

Three command-line tools are available via `pyproject.toml` console scripts:

#### 1. `finance-ml` — Full Pipeline

Run the complete ML pipeline (data load, preprocess, features, models, outputs).

```powershell
finance-ml --data-source auto --limit 5000 --output-dir outputs
finance-ml --data-source db --db-url "postgresql+psycopg2://postgres:@localhost:5432/postgres"
finance-ml --help
```

#### 2. `finance-ml-analyze` — EDA/Analytics Only

Run exploratory data analysis and analytics workflows without training models.

```powershell
finance-ml-analyze --data-source csv --output-dir outputs
finance-ml-analyze --data-source auto
```

#### 3. `finance-ml-validate` — Validation Only

Run data validation workflows (schema checks, data quality, etc.).

```powershell
finance-ml-validate --data-source csv --output-dir outputs
finance-ml-validate --data-source db
```

**Note**: CLI tools are defined in `finance_ml/cli.py` with entry points:

- `finance-ml` → `finance_ml.cli:main`
- `finance-ml-analyze` → `finance_ml.cli:analyze_main`
- `finance-ml-validate` → `finance_ml.cli:validate_main`

### Interactive Dashboards

Two dashboard applications are available for interactive visualization and exploration.

#### Streamlit Dashboard

**Features**: Multi-page layout, KPI cards, interactive filters, stock rankings, data quality monitoring, model
performance analytics, portfolio optimization & risk metrics visualization.

```powershell
streamlit run finance_ml/dashboards/streamlit_app.py
```

**Tabs**: Overview, Data Quality, Model Performance, Predictions Explorer, Sector Analysis, Portfolio & Risk Metrics.

Upload a predictions CSV file with columns: `ticker`,`name`,`exchange`, `sector`, `region`, `last_price`,`price_target`,
`predicted_price_target`,
`market_cap`, `mispricing_score`.

#### Dash Dashboard

**Features**: Interactive filters (sector/region), scatter plots, heatmaps, top undervalued stocks table, portfolio
optimization & risk metrics visualization, reactive callbacks.

```powershell
python finance_ml/dashboards/dash_app.py
```

**Tabs**: Predictions, Data Quality, Model Performance, Portfolio & Risk Metrics.

Access at [http://localhost:8050](http://localhost:8050)

**Programmatic Usage**:

```python
# Evaluation and analytics helpers
# Updated path: eval.py moved to analytics/eval.py (Phase 9.7)
from finance_ml.ml_workflow.analytics.eval import (
    calculate_mispricing_score,
    rank_stocks_by_sector,
    calculate_financial_metrics_dashboard,
    generate_data_quality_alerts,
    prepare_plotly_dashboard_data,
    )

import pandas as pd

# Example DataFrame (replace with your predictions dataframe)
df = pd.DataFrame({
    'ticker': ['AAA', 'BBB', 'CCC'],
    'sector': ['Tech', 'Health', 'Energy'],
    'region': ['US', 'EU', 'APAC'],
    'last_price': [100.0, 50.0, 75.0],
    'predicted_price_target': [120.0, 45.0, 90.0],
    'market_cap': [1e11, 5e10, 3e10],
    })

# Calculate mispricing
mispricing = calculate_mispricing_score(df)
df_with_scores = df.assign(mispricing_score=mispricing)

# Get top undervalued stocks by sector
rankings = rank_stocks_by_sector(df_with_scores, top_n=10)

# Generate financial metrics
metrics = calculate_financial_metrics_dashboard(df_with_scores, group_by='sector')

# Check data quality
alerts = generate_data_quality_alerts(df_with_scores)

# Prepare Plotly chart data
plotly_data = prepare_plotly_dashboard_data(df_with_scores)
```

---

## Environment Variables

Environment variables can be set in your shell or in a `.env` file. See `environment_variables.txt` for the full
reference.

### Core Variables

```bash
# Logging
TF_CPP_MIN_LOG_LEVEL=2          # TensorFlow log level (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
LOG_LEVEL=INFO                   # Python logging level

# Directories
DATA_DIR=data                    # Data directory
MODEL_DIR=models                 # Model output directory
OUTPUT_DIR=outputs               # General output directory
CACHE_DIR=.cache                 # Cache directory

# Database
DB_URL=postgresql+psycopg2://postgres:password@localhost:5432/postgres  # SQLAlchemy URL
DB_SCHEMA=public                 # Database schema
DB_TABLE=equities                # Table name

# Model Configuration
MODEL_VERSION=v9_10              # Model version identifier (current: v9_10 as of v0.9.5)
# TODO: Keep MODEL_VERSION synchronized with releases (see CHANGELOG.md)
RANDOM_SEED=42                   # Random seed for reproducibility

# Performance
N_JOBS=-1                        # Parallel jobs (-1=all cores)
MEMORY_LIMIT=8GB                 # Memory limit

# Analytics
ENABLE_INTERACTIVE_PLOTS=true    # Enable interactive visualizations
REPORT_FORMAT=html               # Report format (html, pdf, excel)
ENABLE_BENCHMARKING=true         # Enable benchmarking analysis (Phase 9.2)
```

### Windows (PowerShell)

```powershell
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:DB_URL = "postgresql+psycopg2://postgres:password@localhost:5432/postgres"
```

### macOS/Linux (bash)

```bash
export TF_CPP_MIN_LOG_LEVEL=2
export DB_URL="postgresql+psycopg2://postgres:password@localhost:5432/postgres"
```

---

## Testing

The project uses Python's built-in `unittest` framework with 128 test modules covering data loading, preprocessing,
features, models, evaluation, portfolio optimization, and integration.
See [docs/code_guidelines.md](docs/code_guidelines.md) v1.7 for TDD
conventions and standards. The full suite can be slow; prefer selective execution during development (see below).

### Run All Tests

```powershell
python -m unittest -v
```

### Run Specific Test Modules (Selective Execution)

```powershell
# Fast unit tests (<100 lines, pure functions)
python -m unittest tests.test_coverage_smoke tests.test_loaders tests.test_validation_regex tests.test_repository_setup -v

# Medium tests (100–500 lines, limited ML)
python -m unittest tests.test_enhanced_imputation tests.test_logging tests.test_risk_metrics tests.test_portfolio_optimization -v

# Targeted standards (code_guidelines.md v1.7)
python -m unittest tests.test_uncertainty_calibration -v        # Uncertainty quantification
python -m unittest tests.test_predictions_schema -v             # Standardized predictions schema
python -m unittest tests.test_regression_sector_metrics -v      # Sector metrics validation
python -m unittest tests.test_data_splits_policy -v             # Data split leakage prevention
python -m unittest tests.test_outlier_safety_rails -v           # Outlier safety rails
python -m unittest tests.test_stacking_default -v               # Stacking defaults

# Representative feature areas
python -m unittest tests.test_finance_ml_data -v        # Data loading
python -m unittest tests.test_features -v               # Feature engineering
python -m unittest tests.test_classification -v         # Classification models
python -m unittest tests.test_regression -v             # Regression models
```

### Fast Helper Tests (Selective)

For a quick verification of the lightweight helper modules added in Model Optimization work (conformal uncertainty,
robust outlier safety, sector features, sector calibration), use the fast test runner:

```powershell
python tools\run_fast_tests.py
```

This runs only small, dependency-light unit tests and completes in milliseconds.

### Coverage Analysis

**Option A: coverage.py**

```powershell
pip install coverage
coverage run -m unittest -v
coverage report -m
coverage html  # Generates htmlcov/index.html
```

**Option B: pytest + pytest-cov**

```powershell
pip install pytest pytest-cov
pytest --cov=finance_ml --cov-report=term-missing
```

### Test Organization

Tests are organized by feature area under `tests/`:

- **Data/Loading**: `test_finance_ml_data`, `test_loaders`, `test_sqlite_import`, `test_validate_csv_import`
- **Preprocessing**: `test_advanced_preprocessing`, `test_enhanced_imputation`, `test_enhanced_imputation_phase93`,
  `test_data_quality`, `test_data_types_detection`
- **Schema & Metadata**: `test_metadata_catalog_quality`, `test_simple_eda_stringdtype`
- **Features**: `test_features`, `test_advanced_features`, `test_finance_ml_features`
- **Models**: `test_classification*`, `test_advanced_models*`, `test_finance_ml_models`, `test_regression`
- **Evaluation**: `test_finance_ml_eval`, `test_analytics`, `test_evaluation_phase96`, `test_valuation_phase97`
- **Portfolio Optimization**: `test_portfolio_ml_prediction`, `test_portfolio_optimization_advanced`,
  `test_portfolio_risk_management`, `test_portfolio_backtesting`, `test_portfolio_dashboards`
- **Integration**: `test_integration_*`, `test_notebook_*`
- **Standards & Policy**: `test_uncertainty_calibration`, `test_predictions_schema`, `test_regression_sector_metrics`,
  `test_data_splits_policy`, `test_outlier_safety_rails`, `test_stacking_default`

**Note**: Some test modules are large (500+ lines) and involve heavy ML training. For faster development iterations, run
smaller test modules or use test discovery patterns.

---

## Project Structure

```
Finance_ML_Analytics_Platform/
├── finance_ml/                    # Main Python package
│   ├── __init__.py               # Package exports
│   ├── cli.py                    # CLI entry points (finance-ml, finance-ml-analyze, finance-ml-validate)
│   ├── data.py                   # Data loading utilities (legacy)
│   ├── features.py               # Feature engineering (legacy, deprecated)
│   ├── models.py                 # Model training utilities (legacy, deprecated)
│   ├── advanced_models.py        # Sector-optimized regression models (legacy, deprecated)
│   ├── classification.py         # Event classification models (legacy, deprecated)
│   ├── classification_enhanced.py # Enhanced classification (legacy, deprecated)
│   ├── advanced_eda.py           # Enhanced EDA utilities (legacy)
│   ├── benchmarking.py           # Benchmarking module (legacy, moved to ml_workflow/eda/)
│   ├── risk_metrics.py           # Risk analytics (legacy, moved to ml_workflow/analytics/)
│   ├── portfolio_optimization.py # Portfolio optimization (legacy, moved to ml_workflow/analytics/)
│   ├── data_catalog.py           # Data catalog and versioning
│   ├── data_versioning.py        # Version tracking
│   ├── dashboards/               # Interactive dashboard applications
│   │   ├── streamlit_app.py      # Streamlit dashboard
│   │   └── dash_app.py           # Dash dashboard
│   └── ml_workflow/              # Phase 9.1-9.8 modular architecture (v9_8)
│       ├── core/                 # Core utilities (config, types, utils)
│       ├── preprocessing/        # Phase 9.1: Imputation, outliers, scaling, quality
│       ├── eda/                  # Phase 9.2: EDA, benchmarking, reports
│       ├── features/             # Phase 9.3: Core, advanced, selection, API
│       ├── classification/       # Phase 9.4: Labels, tuning, models, evaluation
│       ├── regression/           # Phase 9.5: Models, constraints, quantile, tuning, dataset, io
│       ├── evaluation/           # Phase 9.6: Metrics, analysis
│       ├── analytics/            # Phase 9.7: Mispricing, analyst comparison, portfolio, risk
│       └── reporting/            # Phase 9.8: Dashboard data, export
├── tests/                         # Test suite (128 modules)
│   ├── test_*.py                 # Unit and integration tests
│   └── ...
├── tools/                         # Utility scripts and automation
│   ├── import_sqlite.py          # SQLite import utility
│   ├── validate_csv_import.py    # CSV validation
│   ├── analyze_notebook.py       # Notebook analysis
│   └── ...
├── data/                          # CSV data files
│   ├── screening_us.csv          # US equity data
│   ├── screening_eu.csv          # EU equity data
│   ├── screening_apac.csv        # APAC equity data
│   └── screening_rotw.csv        # ROTW equity data
├── outputs/                       # Generated outputs (phase-aligned structure)
│   ├── analytics/                # Analytics reports and rankings
│   ├── catalog/                  # Data catalog metadata
│   ├── classification/           # Classification model outputs
│   ├── dashboards/               # Dashboard data exports
│   ├── eda/                      # EDA visualizations and reports
│   ├── evaluation/               # Model evaluation results
│   ├── features/                 # Feature engineering artifacts
│   ├── plots/                    # Visualization outputs (PNG, HTML)
│   ├── preprocessing/            # Preprocessing artifacts and quality reports
│   ├── regression/               # Regression model outputs and predictions
│   └── reporting/                # Final reports (Excel, PDF)
├── docs/                          # Documentation
│   ├── improvement_plan/         # Development roadmap and phase documentation
│   └── summaries/                # Implementation summaries
├── ml_finance_model_main.ipynb   # Main notebook (Phase 9.1-9.8)
├── ml_finance_model_main2_0.ipynb # Alternative notebook version
├── ml_finance_model_main.py      # Python script version
├── etl_data_explorer.ipynb       # ETL pipeline exploration and data analysis
├── stock_analytics.ipynb         # Stock analytics and visualization
├── portfolio_optimization_risk_management.ipynb # Portfolio optimization
├── stock_price_target_prediction.ipynb # Price target prediction workflow
├── create_equities_schema.sql        # PostgreSQL schema
├── import_equities_data.sql          # PostgreSQL data import (staging + validation)
├── create_equities_schema_sqlite.sql # SQLite schema
├── import_equities_data_sqlite.sql   # SQLite data import
├── pipeline/                          # Pipeline data directory
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Package metadata and build config
├── environment_variables.txt     # Environment configuration reference
├── CHANGELOG.md                  # Version history
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## Scripts and Tools

### Main Entry Points

| Script/Tool                   | Description                   | Usage                                                  |
|-------------------------------|-------------------------------|--------------------------------------------------------|
| `ml_finance_model_main.ipynb` | Main notebook (Phase 9.1-9.8) | `jupyter notebook ml_finance_model_main.ipynb`         |
| `ml_finance_model_main.py`    | Python script version         | `python ml_finance_model_main.py --data-source auto`   |
| `finance-ml`                  | CLI: Full pipeline            | `finance-ml --data-source auto --output-dir outputs`   |
| `finance-ml-analyze`          | CLI: EDA/analytics only       | `finance-ml-analyze --data-source csv`                 |
| `finance-ml-validate`         | CLI: Validation only          | `finance-ml-validate --data-source db`                 |
| `streamlit_app.py`            | Streamlit dashboard           | `streamlit run finance_ml/dashboards/streamlit_app.py` |
| `dash_app.py`                 | Dash dashboard                | `python finance_ml/dashboards/dash_app.py`             |

### Utility Scripts (tools/)

| Script                    | Description                                       |
|---------------------------|---------------------------------------------------|
| `import_sqlite.py`        | Import CSVs into SQLite with chunked processing   |
| `validate_csv_import.py`  | Validate CSV data quality before import           |
| `run_earnings_monitor.py` | Generate earnings dashboards + alerts (scheduled) |
| `analyze_notebook.py`     | Analyze notebook structure and cells              |
| `analyze_predictions.py`  | Analyze model prediction outputs                  |
| `run_fast_tests.py`       | Run fast helper unit tests (no heavy training)    |
| `verify_outputs.py`       | Verify expected output files exist and non-empty  |

#### Scheduled Earnings Monitoring (Windows Task Scheduler)

You can run earnings monitoring on a daily/weekly schedule without opening notebooks:

```powershell
python tools\run_earnings_monitor.py --data-source auto --out-dir outputs\eda\earnings_analytics
```

- `--data-source auto` uses DB when `DB_URL` (or `--db-url`) is provided, otherwise falls back to CSV.
- Thresholds are configurable via CLI flags (e.g. `--eps-miss-threshold-pct 25`, `--target-spread-threshold-pct 40`).

### Database Scripts

| Script                                | Description                          |
|---------------------------------------|--------------------------------------|
| `pipeline/create_equities_schema.sql` | PostgreSQL schema creation           |
| `pipeline/import_equities_data.sql`   | PostgreSQL data import (all regions) |
| `create_equities_schema_sqlite.sql`   | SQLite schema creation               |
| `import_equities_data_sqlite.sql`     | SQLite data import (all regions)     |

---

## Recent Updates

### Version 0.9.5 (Current Release - 2025-12-18)

**Feature Engineering Enhancements: Earnings Quality Analytics**:

- **NEW: Earnings Quality Category** (33 features):
    - `engineer_estimated_vs_actual_analytics()`: 11 features for EPS/revenue surprise analysis
        - `eps_surprise_pct`, `eps_surprise_magnitude`, `revenue_surprise_pct`, `revenue_beat_indicator`
        - `ebitda_surprise_pct`, `earnings_beat_indicator`, `surprise_momentum_score`
        - `positive_revision_momentum`, `consensus_uncertainty_score`, `estimate_revision_acceleration`
    - `engineer_gaap_vs_adjusted_analytics()`: 22 features for GAAP vs. Adjusted earnings comparison
        - EPS/Net Income/EBITDA/EBIT adjustment metrics (spread, ratio, percentage)
        - Quality flags: `eps_quality_flag_ltm`, `earnings_quality_warning_flag`
        - Composite scores: `adjustment_consistency_score`, `earnings_quality_score` (0-100)
- **ETL Integration**: New `engineer_earnings_analytics` flag in `FeatureEngineeringConfig`
- **Schema Updates**: 33 new column definitions in `COLUMN_SCHEMA`, new `"earnings_quality"` category in
  `PHASE93_FEATURE_CATEGORIES`
- **Phase 9.3 Feature Categories**: Updated from 196 to **229 features** across **17 categories**
- **Documentation**: `docs/code_guidelines.md` v1.12 with complete usage examples

**Usage Example**:

```python
from finance_ml.ml_workflow.features.advanced import (
    engineer_estimated_vs_actual_analytics,
    engineer_gaap_vs_adjusted_analytics
)

# Apply earnings analytics
df_earnings = engineer_estimated_vs_actual_analytics(all_stocks_features)
df_earnings = engineer_gaap_vs_adjusted_analytics(df_earnings)

# Filter for high-quality earnings beats
quality_beats = df_earnings[
    (df_earnings['earnings_beat_indicator'] == True) &
    (df_earnings['earnings_quality_score'] > 80)
]
```

See [CHANGELOG.md](CHANGELOG.md) for complete details.

### Version 0.9.4 (Previous Release - 2025-12-10)

**Phase 9.5 Notebook Integration and ETL Pipeline Test Coverage**:

- **Phase 9.5 Notebook Integration Guides**: Comprehensive guides for automated stacking hyperparameter tuning and
  feature alignment validation
- **ETL Unified Pipeline Test Coverage**: New `test_etl_unified_pipeline.py` with 63 comprehensive tests validating
  ETLConfig, ETLMetrics, ETLPipeline, and semantic transformations
- **Documentation**: New guides in `docs/guides/` and completion summaries for Priority 1-3 tasks
- **Type Hints and Code Quality**: Enhanced type safety with Literal type hints and improved import organization

See [CHANGELOG.md](CHANGELOG.md) for complete details.

### Version 0.9.3 (Previous Release - 2025-12-08)

**Unified ETL Pipeline with Semantic Transformations**:

- **Single Entry Point**: `etl_with_features()` consolidates schema.py, column_semantics.py, and features/api.py into
  one unified call for complete ETL + feature engineering
- **Semantic-Aware Transformations**:
    - Price column preservation (21 columns protected from transformation)
    - Log-transforms for skewed market value columns (19 columns)
    - Ratio/percentage column exclusion from winsorization
- **ETLConfig Semantic Attributes**: `use_semantic_column_classification`, `preserve_price_columns`,
  `log_transform_market_values`, `apply_feature_engineering`, `feature_preset`
- **ETLMetrics Tracking**: `semantic_classification_applied`, `price_columns_count`, `log_transformed_columns`,
  `features_added`, `feature_preset_used`
- **Feature Engineering Presets**: basic, momentum, quality, standard, comprehensive (196 features)
- **Test Coverage**: 63 new tests in `test_etl_unified_pipeline.py`
- **Documentation**: `docs/code_guidelines.md` v1.10 with STANDARD/OPTIONAL import patterns

**Usage Example**:

```python
from finance_ml.ml_workflow.preprocessing import etl_with_features

all_stocks, metrics = etl_with_features(
    source='csv',
    data_dir='data/',
    feature_preset='comprehensive',
    return_metrics=True,
)
print(metrics.summary())
```

### Version 0.8.3 (Previous Release - 2025-11-22)

- Packaging: pyproject.toml bumped to 0.8.3; CLI entry points unchanged (finance-ml, finance-ml-analyze,
  finance-ml-validate)
- Notebook: Output directory configuration standardized; added missing subdirectories and unified OUTPUT_DIR usage
  across cells (see NOTEBOOK_OUTPUT_DIR_FIX_SUMMARY.md)
- Notebook: Integrated Phases 9.5–9.8 sections with artifacts and structural tests passing (see
  IMPLEMENTATION_SUMMARY_PHASE95_98.md)
- Documentation: This README updated for alignment with pyproject.toml and latest guidance
- TODO: Confirm CHANGELOG.md includes 0.8.3 entry and update environment_variables.txt version header accordingly

### Version 0.8.2 (2025-11-19)

**TDD Implementation: Data Preprocessing & Datatype Detection**:

- **Schema-Aware Datatype Detection**: New `finance_ml.ml_workflow.preprocessing.dtypes` module (326 lines) with
  schema-driven type casting, validation, and comprehensive diagnostics
- **Centralized Schema Registry**: `finance_ml.ml_workflow.data.schema` module (530 lines) with 350+ normalized column
  mappings derived from SQL schema
- **Phase 9.3 Feature Categorization**: Organized feature inputs into 6 buckets (momentum, valuation, profitability,
  quality/risk, cash flow, growth) for ML pipeline
- **Enhanced Imputation**: Phase 9.3 enhanced imputation with schema consistency checks, sector-aware KNN,
  categorical/datetime strategies
- **Test Coverage**: 24 new TDD tests (23 passing, 1 skipped) across 4 modules:
  - `test_data_types_detection.py` (9 tests): Schema-aware casting, coercion tracking, Phase 9.3 validation
  - `test_enhanced_imputation_phase93.py` (8 tests): Sector-aware KNN, categorical/datetime strategies
  - `test_metadata_catalog_quality.py` (4 tests): Metadata validation and quality stats
  - `test_simple_eda_stringdtype.py` (3 tests): StringDtype compatibility validation
- **Documentation**: `docs/TDD_IMPLEMENTATION_SUMMARY.md` with complete implementation details

**Portfolio Optimization Enhancement Plan (6 Phases Complete)**:

- **Phase 1: Enhanced Stock Filtering & Selection** ✓
  - Module: `finance_ml/ml_workflow/analytics/stock_selection.py`
  - Multi-metric ranking, sector-balanced selection, currency unit support
- **Phase 2: ML-Based Return Prediction** ✓
  - Module: `finance_ml/ml_workflow/analytics/ml_returns.py`
  - ML feature engineering, linear predictor, ensemble predictions
- **Phase 3: Advanced Portfolio Optimization** ✓
  - Module: `finance_ml/ml_workflow/analytics/portfolio.py`
  - Black-Litterman, Risk Parity, Hierarchical Risk Parity (HRP)
- **Phase 4: Risk Management Enhancements** ✓
  - Module: `finance_ml/ml_workflow/analytics/risk.py`
  - Expected Shortfall (CVaR), tracking error, stress testing, Monte Carlo simulation
- **Phase 5: Backtesting Framework** ✓
  - Modules: `analytics/portfolio.py`, `analytics/attribution.py`
  - Vectorized backtest, walk-forward optimization, performance attribution
- **Phase 6: Interactive Dashboard Expansion** ✓
  - Module: `finance_ml/dashboards/portfolio_widgets.py`
  - Rebalancing widget, multi-period comparison, factor exposure dashboard
- **Test Coverage**: 23 new tests across 5 modules covering all 6 phases
- **Notebook Integration**: Section 10 structure added to `ml_finance_model_main.ipynb`
- **Documentation**: `docs/improvement_plan/portfolio_optimization_enhancement_plan.md`

**Phase 9.3 Feature Enhancement Plan (Schema Version 1.3)**:

- **Schema Expansion**: 310 columns total (expanded from 262, +48 new columns, +18.3%)
- **New Feature Categories**:
  - Technical indicators: EMAs (20D, 50D, 100D, 250D), 52W High/Low, Relative Volume
  - Valuation multiples time-series: EV/Sales (11 cols), EV/EBITDA (6 cols), P/E extended (11 cols)
  - Revenue forecasting estimates: NTM, FY1E (4 cols)
  - Dividend record information: Frequency, streak, coverage (8 cols)
  - Employment metrics: FY/FQ employee counts (2 cols)
- **Implementation Status**: All features wired into advanced pipeline, accessible via
  `finance_ml.ml_workflow.features.advanced.py`
- **Model Version Target**: v9_10
- **Documentation**: `docs/improvement_plan/Phase_9.3_feature_enhancement_plan.md` (Version 1.1)

**Test Suite Expansion**:

- **Total Test Modules**: 128 (expanded significantly with preprocessing, semantic column, and feature tests)
- **New Test Categories**:
  - TDD Implementation: 4 modules, 24 tests
  - Portfolio Optimization: 5 modules, 23 tests
- **Test Execution Strategies**: Fast/medium/slow test categorization for development efficiency (see Testing section)

### Version 0.8.1 (Previous Release - 2025-11-14)

**LightGBM Preprocessing Test Suite**:

- New comprehensive test suite for LightGBM preprocessing validation (`tests/test_preprocess_lightgbm.py`)
- Ensures consistent feature engineering across training and prediction pipelines
- Validates categorical encoding, datetime feature extraction, and column alignment

**Major Bug Fixes**:

- **Feature Mismatch Error**: Resolved critical LightGBM prediction error (461 vs 941 features)
  - Root cause: Using feature names from wrong model instance after reassignment
  - Solution: Re-extract feature names from correct model after hyperparameter optimization
  - Added support for CatBoost, XGBoost, and LightGBM feature name extraction
- **Column Selection and Alignment**: Fixed shape mismatch issues in data preprocessing
  - Replaced list comprehension with `.reindex()` for exact column matching
  - Enhanced SHAP computation with proper data alignment
- **Unicode Encoding**: Replaced all Unicode emojis with ASCII for universal terminal compatibility
- **Model-Agnostic Scoring**: Fixed CatBoost-specific Pool scoring to work with all model types

**Documentation and Validation Enhancements**:

- Three detailed fix summary documents: `FIX_FEATURE_MISMATCH_FINAL.md`, `FIX_SUMMARY_SHAPE_MISMATCH.md`,
  `FIX_EMOJI_AND_POOL_ISSUES.md`
- Enhanced validation tools: `validate_clipping_fix.py` and `validate_zero_predictions_fix.py`
- Updated core modules: `features/core.py`, `classification/models.py`, `classification/tuning.py`, `analytics/eval.py`

### Version 0.8.0 (Previous Release - 2025-11-13)

**Phase 10 Integration - Prediction Confidence Scoring**:

- Comprehensive confidence scoring and outlier detection system with three approaches:
    - Ensemble-based confidence: measures agreement across multiple models
    - Residual-based confidence: uses historical prediction error patterns
    - Quantile-based confidence: leverages prediction interval width
- New `finance_ml.ml_workflow.evaluation.confidence` module with confidence scoring methods
- Sector-specific model training utilities in `finance_ml.ml_workflow.regression.sector_models`
- Enhanced uncertainty quantification with isotonic bias correction and quantile calibration
- Documentation: `docs/summaries/phase10_integration_summary.md` (526 lines)
- Comprehensive test suite: `test_bias_correction_isotonic.py`, `test_outlier_prediction_filtering.py`,
  `test_quantile_calibration_coverage.py`, `test_sector_specific_models.py`

**Intelligent Train/Test Splitting Utilities**:

- Time-series cross-validation with sector stratification to prevent data leakage
- Grouped splitting by ticker to ensure no ticker appears in both train and test sets
- Leakage-prevention utilities aligned with code_guidelines.md v1.7 Data Split Policy
- Enhanced documentation in multiple summary files

**Prediction Bound Fixes**:

- **Upper Bound Fix**: Replaced statistical clipping (mean±3σ) with percentile-based clipping (1.5x p99.5)
    - Reduced high-value stock prediction error by 83.3%
- **Lower Bound Fix**: Replaced hard zero with adaptive lower bound (0.5x p0.5, min $0.10)
    - Eliminated 348 zero predictions (24.75% reduction) while preserving low-value stock predictions
- Validation tools: `tools/validate_clipping_fix.py`, `tools/validate_zero_predictions_fix.py`
- Documentation: `docs/summaries/PREDICTION_CAPPING_FIX.md`, `docs/summaries/ZERO_PREDICTIONS_FIX.md`

**Model Optimization Completion**:

- Enhanced `ml_finance_model_main.ipynb` with time-series cross-validation, quantile regression export, feature
  importance analysis
- Default stacking ensemble usage with improved safety checks for missing data columns
- Generated comprehensive output CSV files for model evaluation
- Added fast helper test runner and output verification utility (16 tests passing)

**Enhanced Uncertainty Quantification**:

- Improved `finance_ml.ml_workflow.regression.calibration` (679 lines) with isotonic regression
- Enhanced `finance_ml.ml_workflow.regression.uncertainty` (525 lines) with conformal prediction
- Updated code guidelines with uncertainty quantification standards

**Regression and Classification Integration**:

- Complete workflow integration with error handling and output persistence
- Comprehensive documentation: `docs/summaries/REGRESSION_INTEGRATION_SUMMARY.md` (523 lines)
- Classification integration: `docs/summaries/CLASSIFICATION_INTEGRATION_SUMMARY.md` (209 lines)

### Version 0.7.1 (Previous Release - 2025-11-11)

**Portfolio Optimization & Risk Metrics Visualization** (2025-11-11):

- Interactive Plotly visualizations for portfolio optimization and risk analysis
- **Notebook Section 10**: Three comprehensive visualizations added to `ml_finance_model_main.ipynb`:
    - Efficient Frontier with Maximum Sharpe Ratio and Minimum Volatility portfolios highlighted
    - Risk Metrics Dashboard with VaR, CVaR, Sharpe, Sortino ratios, and gauge charts
    - Portfolio Drawdown Analysis with time series visualization
- **Dashboard Integration**: Portfolio & Risk Metrics tab added to both Streamlit and Dash applications
- **Output Files**: 6 new visualization files saved to `outputs/analytics/` (3 HTML + 3 PNG)
- **Functions Used**: Portfolio optimization (`finance_ml/ml_workflow/analytics/portfolio.py`) and risk metrics (
  `finance_ml/ml_workflow/analytics/risk.py`)
- See `docs/PORTFOLIO_VISUALIZATION_IMPLEMENTATION.md` for complete implementation details

**Phase 9.3 Feature Engineering API**:

- Public API with feature engineering presets (`finance_ml/ml_workflow/features/api.py`):
    - `basic`: Core ratios, margins, volatility, and revenue CAGR
    - `momentum`: Price momentum and technical indicators
    - `quality`: Accounting quality and financial distress signals
    - `comprehensive`: Full advanced feature set
- Enhanced `build_comprehensive_features()` with optional preset parameter for flexible feature selection
- Backward compatible: default behavior unchanged (comprehensive mode)

**Test Infrastructure & Quality**:

- Phase 9.3 test infrastructure with comprehensive fixtures:
    - `tests/fixtures/feature_engineering_samples.py`: Sample DataFrames with edge cases
    - `tests/utils/feature_test_helpers.py`: Validation utilities (assert_no_inf, assert_nan_ratio_below, time_block)
    - `tests/test_feature_infra_phase93.py`: Infrastructure validation tests
    - `tests/test_features_api_phase93.py`: API preset tests with TDD methodology
- Fixed test import paths: `tests/test_risk_metrics.py` updated to Phase 9.7 module path

**Model Version & Configuration**:

- MODEL_VERSION bump to v9_10 (synchronized across all configuration files)
- Updated `finance_ml/config.py`, test assertions, and environment files
- Production readiness: Cross-validation with 138+ tests passing (fast and medium test suites)

**Phase 9.5.1 Outputs and Validation Enhancements**:

- Enhanced regression outputs with diagnostics and sector metadata
- Added Time-Series Cross-Validation (guarded) metrics export when a date column is present
- New scripts:
    - `tools/run_fast_tests.py` — fast helper tests runner
    - `tools/verify_outputs.py` — output artifact verification (see below)

Expected output files (after running the notebook or script):

- `outputs/regression/regression_predictions_detailed.csv` — Standardized predictions schema (
  see [code_guidelines.md](docs/code_guidelines.md) v1.7)
    - Required columns: ticker, isin, sector, region, last_price, y_true, y_pred, y_pred_calibrated, pred_p10, pred_p50,
      pred_p90, interval_width, abs_error, pct_error, model_version, snapshot_date
- `outputs/regression/regression_metrics_by_sector.csv` — Per-sector MAE, RMSE, R², MAPE, count
- `outputs/regression/quantile_predictions.csv` — Uncertainty intervals with monotonicity guarantees
- `outputs/regression/feature_importance.csv`
- `outputs/evaluation/tscv_metrics.csv` (when TimeSeriesSplit is applicable)

Verify these with:

```powershell
python tools\verify_outputs.py
```

### Version 0.6.1 (Previous Release - 2025-11-09)

**Phase 9.5 Classification Meta-Features & Enhanced Imputation**:

- Classification meta-feature extraction (`extract_classification_features`) to enhance regression models
- New classification module structure with dedicated evaluation and models submodules
- Enhanced 6-step imputation strategy with comprehensive test coverage
- Modular regression pipelines including Ridge, Lasso, ElasticNet, Bayesian Ridge, and Gradient Boosting models

### Version 0.5.1

**Phase 9.1-9.8 Module Structure (v9_8)**:

- Complete subpackage refactoring into phase-aligned architecture
- Dedicated subpackages: preprocessing/, eda/, features/, classification/, regression/, evaluation/, analytics/,
  reporting/
- Backward-compatible imports with deprecation warnings
- Package-level exports with descriptive function prefixes

**All Phases Implementation**:

- Phase 9.1: 6-step imputation pipeline (zero, KNN, price-based, median)
- Phase 9.2: Enhanced EDA with benchmarking and statistical testing
- Phase 9.3: Advanced feature engineering with sector-specific optimizations
- Phase 9.4: Multi-class event classification with neural networks
- Phase 9.5: Sector-optimized regression with quantile models and ensembles
- Phase 9.6: Comprehensive evaluation and error analysis
- Phase 9.7: Stock valuation, mispricing analysis, and analyst comparison
- Phase 9.8: Interactive dashboards and reporting

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow PEP 8** code style (use `black`, `isort`)
3. **Follow coding standards** in [docs/code_guidelines.md](docs/code_guidelines.md) v1.7 (function signatures, return
   types, TDD conventions)
4. **Write tests** for new functionality (TDD preferred; see code_guidelines.md for uncertainty, safety rails, and
   schema validation standards)
5. **Run test suite** before submitting: `python -m unittest -v`
6. **Update documentation** (README, docstrings, CHANGELOG)
7. **Submit a pull request** with clear description

### Development Workflow

```powershell
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes and add tests
# Edit code in finance_ml/
# Add tests in tests/test_your_feature.py

# 3. Run tests
python -m unittest tests.test_your_feature -v

# 4. Run full test suite
python -m unittest -v

# 5. Check coverage
coverage run -m unittest -v
coverage report

# 6. Format code
black finance_ml/ tests/
isort finance_ml/ tests/

# 7. Commit and push
git add .
git commit -m "Add your feature"
git push origin feature/your-feature

# 8. Create pull request on GitHub
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 Finance ML Analytics Platform Contributors**

---

## Support and Contact

- **Issues**: [GitHub Issues](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/issues)
- **Documentation**: [README.md](README.md) and [docs/](docs/)
- **Repository**: [GitHub](https://github.com/Kabenge42/Finance_ML_Analytics_Platform)

---

## Acknowledgments

- Built with Python 3.12+ and modern ML libraries (scikit-learn, XGBoost, LightGBM, CatBoost)
- Interactive dashboards powered by Streamlit and Dash
- Data management with PostgreSQL and SQLite
- Comprehensive testing with unittest and pytest

---

## Known Issues and TODOs

### Optional Dependencies

**Dashboards**: Streamlit and Dash are included in `requirements.txt` for interactive dashboard functionality. If you
encounter issues or prefer to install them separately:

```powershell
pip install streamlit dash plotly
```

**Note**: Dashboard dependencies (streamlit>=1.30.0, dash>=2.14.0) are included by default in the unified
requirements.txt but are optional in pyproject.toml.

### TensorFlow Installation

TensorFlow is optional and CPU-only installation is sufficient for this project. For GPU acceleration:

- Follow official [TensorFlow GPU installation guide](https://www.tensorflow.org/install/gpu)
- Ensure CUDA/cuDNN compatibility with your GPU and TensorFlow version
- If installation issues occur, TensorFlow can be temporarily removed from `requirements.txt`

The core workflow uses scikit-learn and gradient boosting libraries (XGBoost, LightGBM, CatBoost) and will function
without TensorFlow.

### Version Numbering Convention

**Package Version** (e.g., `0.7.1`):

- Follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`
- **MAJOR**: Breaking API changes or major architectural updates
- **MINOR**: New features, backward-compatible changes
- **PATCH**: Bug fixes, documentation updates
- Updated in: `pyproject.toml`, `README.md`, `CHANGELOG.md`, `environment_variables.txt`

**Model Version** (e.g., `v9_10`):

- Format: `v{PHASE}_{ITERATION}` (e.g., `v9_10` = Phase 9, Iteration 10)
- **PHASE**: Major modeling phase (Phase 9 = modular refactor, Phase 10 = optimization)
- **ITERATION**: Minor updates within phase (features, hyperparameters, calibration)
- Updated in: `finance_ml/config.py`, `ml_finance_model_main.ipynb`, `README.md`

**Alignment Requirements**:

- **Notebook** (`ml_finance_model_main.ipynb`) and **Package** (`finance_ml/config.py`) MODEL_VERSION **must match**
- Package version increments with each release; MODEL_VERSION increments with modeling changes
- Example: Package v0.9.5 can have MODEL_VERSION v9_10 (no modeling changes from v0.9.4)

**Current Versions** (as of 2025-12-18):

- Package: `0.9.5`
- Model: `v9_10`
- Status: ✓ Aligned (notebook and package both use v9_10)

**Version Update Checklist**:

- [x] Package version change: Update `pyproject.toml`, `README.md`, `CHANGELOG.md`, `environment_variables.txt`
- [x] Model version change: Update `finance_ml/config.py` and `ml_finance_model_main.ipynb` (must match!)
- [x] Document changes in `CHANGELOG.md` with clear explanation of what changed

### Version Synchronization

**Current Status** (as of 2025-12-18):

- **pyproject.toml**: Version = "0.9.5" ✓
- **README.md**: Updated to 0.9.5 ✓
- **CHANGELOG.md**: See CHANGELOG.md for version history
- **environment_variables.txt**: Version = 0.9.5, MODEL_VERSION=v9_10 ✓
- **finance_ml/config.py**: MODEL_VERSION v9_10 ✓

**Status**: ✓ Aligned — all version references synchronized at 0.9.5 with MODEL_VERSION v9_10

### Future Enhancements

**TODO**: Consider adding the following enhancements:

- Expand dashboard functionality with more interactive features
- Add time-series forecasting capabilities
- Implement automated hyperparameter tuning for all model types
- Add support for real-time data streaming
- Create Docker containerization for easier deployment

See `docs/improvement_plan/finance_ml_improvement_plan.md` for detailed development roadmap.

---

**Last Updated**: 2025-12-15  
**README Version**: 4.6 (aligned to v0.9.4; 128 test modules; code_guidelines.md v1.10; unified ETL pipeline; corrected
notebook/script paths)
