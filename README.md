# Finance ML Analytics Platform

**Version 0.6.0** — Comprehensive ML Platform for Equity Analysis and Price Target Prediction

> **Documentation Last Updated:** 2025-11-06  
> **Latest Release**: v0.6.0 - Phase 9.5 & 9.7 Enhanced (Classification + Analyst Comparison)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
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

1. **Phase 9.1**: Loading and preprocessing with 4-step imputation strategy
2. **Phase 9.2**: Enhanced exploratory data analysis with statistical testing and benchmarking
3. **Phase 9.3**: Advanced feature engineering with sector-specific optimizations
4. **Phase 9.4**: Multi-class event classification using neural networks and ensembles
5. **Phase 9.5**: Sector-optimized regression models with hyperparameter tuning and quantile models
6. **Phase 9.6**: Model evaluation and comprehensive error analysis
7. **Phase 9.7**: Identification of under/overvalued stocks with visualization and analyst comparison
8. **Phase 9.8**: Comprehensive analytics and reporting

---

## Key Features

- 📊 **Data Management**: PostgreSQL/SQLite integration + CSV fallback for multi-region equity data (US, EU, APAC, ROTW)
- 🧹 **Data Quality**: 4-step imputation pipeline (zero-fill, KNN, price-based, median) with validation
- 🔧 **Feature Engineering**: Financial ratios, margins, volatility, revenue CAGR, sector-specific features
- 🤖 **ML Models**: Event classification, sector-optimized regression, quantile models, stacking ensembles
- 📈 **Analytics**: Mispricing scores, stock ranking, analyst comparison, benchmarking, risk metrics
- 📊 **Interactive Dashboards**: Streamlit and Dash applications for real-time visualization
- 🎯 **Stock Prediction**: End-to-end 8-phase workflow for price target prediction
- 📄 **Reporting**: Excel/PDF reports, interactive visualizations, valuation analysis
- ⚙️ **Configuration**: Flexible config via environment variables and CLI options
- 🧪 **Tested**: 67+ test modules with comprehensive coverage (≥80% target for new code)
- 🚀 **CLI**: Three command-line tools for different workflows
- 🔍 **Model Interpretation**: SHAP analysis for explainability

---

## Technology Stack

### Language & Runtime

- **Python**: 3.12 or 3.13 (officially supported per `pyproject.toml`; 3.10-3.11 may work but untested)
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
- **Python**: 3.12 or 3.13 (officially supported)
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

# SQLite (alternative):
sqlite3 equities.sqlite ".read create_equities_schema_sqlite.sql"
sqlite3 equities.sqlite ".read import_equities_data_sqlite.sql"

# 5. Configure environment (optional)
# Edit environment_variables.txt and export or create .env

# 6. Run main notebook
jupyter notebook ml_finance_model_main_v9.ipynb

# Or run as Python script
python ml_finance_model_main.py --data-source auto --limit 5000

# Or use CLI
finance-ml --data-source auto --output-dir outputs

# Or launch interactive dashboard
streamlit run finance_ml/dashboards/streamlit_app.py
```

---

## Installation

### 1. Prerequisites

Ensure you have Python 3.12 or 3.13 installed:

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

**Main Notebook**: `ml_finance_model_main_v9.ipynb`

```powershell
# Launch Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

Open `ml_finance_model_main_v9.ipynb` and run cells sequentially. The notebook includes:

- Configuration and setup
- Data loading from database or CSV
- 4-step imputation pipeline
- EDA with visualizations
- Feature engineering
- Classification models
- Regression models (sector-optimized)
- Evaluation and error analysis
- Stock ranking and valuation
- Report generation

**Backup Notebooks**:

- `ml_finance_model_main.ipynb` — Previous stable version
- `ml_finance_model_main_backup.ipynb` — Historical backup

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
performance analytics.

```powershell
streamlit run finance_ml/dashboards/streamlit_app.py
```

Upload a predictions CSV file with columns: `ticker`, `sector`, `region`, `last_price`, `predicted_price_target`,
`market_cap`, `mispricing_score`.

#### Dash Dashboard

**Features**: Interactive filters (sector/region), scatter plots, heatmaps, top undervalued stocks table, reactive
callbacks.

```powershell
python finance_ml/dashboards/dash_app.py
```

Access at [http://localhost:8050](http://localhost:8050)

**Programmatic Usage**:

```python
from finance_ml.eval import (
    calculate_mispricing_score,
    rank_stocks_by_sector,
    calculate_financial_metrics_dashboard,
    generate_data_quality_alerts,
    prepare_plotly_dashboard_data,
    )

# Calculate mispricing
df_with_scores = calculate_mispricing_score(df)

# Get top undervalued stocks by sector
rankings = rank_stocks_by_sector(df_with_scores, top_n=10)

# Generate financial metrics
metrics = calculate_financial_metrics_dashboard(df, group_by='sector')

# Check data quality
alerts = generate_data_quality_alerts(df)

# Prepare Plotly chart data
plotly_data = prepare_plotly_dashboard_data(df)
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
MODEL_VERSION=v0_6_0             # Model version identifier
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

The project uses Python's built-in `unittest` framework with 67+ test modules covering data loading, preprocessing,
features, models, evaluation, and integration.

### Run All Tests

```powershell
python -m unittest -v
```

### Run Specific Test Modules

```powershell
# Fast unit tests (pure functions, no ML training)
python -m unittest tests.test_coverage_smoke tests.test_loaders tests.test_validation_regex -v

# Medium tests (integration, limited ML)
python -m unittest tests.test_enhanced_imputation tests.test_data_catalog tests.test_logging -v

# Specific feature areas
python -m unittest tests.test_finance_ml_data -v        # Data loading
python -m unittest tests.test_features -v               # Feature engineering
python -m unittest tests.test_classification -v         # Classification models
python -m unittest tests.test_regression -v             # Regression models
python -m unittest tests.test_dashboard_helpers -v      # Dashboard helpers
```

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
- **Preprocessing**: `test_advanced_preprocessing`, `test_enhanced_imputation`, `test_data_quality`
- **Features**: `test_features`, `test_advanced_features`, `test_finance_ml_features`
- **Models**: `test_classification*`, `test_advanced_models*`, `test_finance_ml_models`, `test_regression`
- **Evaluation**: `test_finance_ml_eval`, `test_analytics`, `test_evaluation_phase96`, `test_valuation_phase97`
- **Integration**: `test_integration_*`, `test_notebook_*`
- **Dashboards**: `test_dashboard_helpers*`, `test_streamlit_dashboard`, `test_dash_dashboard`

**Note**: Some test modules are large (500+ lines) and involve heavy ML training. For faster development iterations, run
smaller test modules or use test discovery patterns.

---

## Project Structure

```
Finance_ML_Analytics_Platform/
├── finance_ml/                    # Main Python package
│   ├── __init__.py               # Package exports
│   ├── cli.py                    # CLI entry points (finance-ml, finance-ml-analyze, finance-ml-validate)
│   ├── data.py                   # Data loading utilities
│   ├── features.py               # Feature engineering
│   ├── models.py                 # Model training utilities
│   ├── eval.py                   # Evaluation and analytics (7751 lines)
│   ├── advanced_models.py        # Sector-optimized regression models
│   ├── classification.py         # Event classification models
│   ├── classification_enhanced.py # Enhanced classification (Phase 9.5)
│   ├── advanced_eda.py           # Enhanced EDA utilities
│   ├── benchmarking.py           # Benchmarking module (Phase 9.2)
│   ├── risk_metrics.py           # Risk analytics
│   ├── portfolio_optimization.py # Portfolio optimization
│   ├── data_catalog.py           # Data catalog and versioning
│   ├── data_versioning.py        # Version tracking
│   └── dashboards/               # Interactive dashboard applications
│       ├── streamlit_app.py      # Streamlit dashboard
│       └── dash_app.py           # Dash dashboard
├── tests/                         # Test suite (67+ modules)
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
├── outputs/                       # Generated outputs
│   ├── eda/                      # EDA visualizations
│   ├── models/                   # Trained models and predictions
│   └── analytics/                # Analytics reports
├── docs/                          # Documentation
│   ├── improvement_plan/         # Development roadmap and phase documentation
│   └── summaries/                # Implementation summaries
├── ml_finance_model_main_v9.ipynb # Main notebook (Phase 9.1-9.8)
├── ml_finance_model_main.ipynb   # Previous stable notebook
├── ml_finance_model_main.py      # Python script version
├── create_equities_schema.sql    # PostgreSQL schema
├── import_equities_data.sql      # PostgreSQL data import
├── create_equities_schema_sqlite.sql # SQLite schema
├── import_equities_data_sqlite.sql   # SQLite data import
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

| Script/Tool                      | Description                   | Usage                                                  |
|----------------------------------|-------------------------------|--------------------------------------------------------|
| `ml_finance_model_main_v9.ipynb` | Main notebook (Phase 9.1-9.8) | `jupyter notebook ml_finance_model_main_v9.ipynb`      |
| `ml_finance_model_main.py`       | Python script version         | `python ml_finance_model_main.py --data-source auto`   |
| `finance-ml`                     | CLI: Full pipeline            | `finance-ml --data-source auto --output-dir outputs`   |
| `finance-ml-analyze`             | CLI: EDA/analytics only       | `finance-ml-analyze --data-source csv`                 |
| `finance-ml-validate`            | CLI: Validation only          | `finance-ml-validate --data-source db`                 |
| `streamlit_app.py`               | Streamlit dashboard           | `streamlit run finance_ml/dashboards/streamlit_app.py` |
| `dash_app.py`                    | Dash dashboard                | `python finance_ml/dashboards/dash_app.py`             |

### Utility Scripts (tools/)

| Script                   | Description                                     |
|--------------------------|-------------------------------------------------|
| `import_sqlite.py`       | Import CSVs into SQLite with chunked processing |
| `validate_csv_import.py` | Validate CSV data quality before import         |
| `analyze_notebook.py`    | Analyze notebook structure and cells            |
| `analyze_predictions.py` | Analyze model prediction outputs                |

### Database Scripts

| Script                              | Description                          |
|-------------------------------------|--------------------------------------|
| `create_equities_schema.sql`        | PostgreSQL schema creation           |
| `import_equities_data.sql`          | PostgreSQL data import (all regions) |
| `create_equities_schema_sqlite.sql` | SQLite schema creation               |
| `import_equities_data_sqlite.sql`   | SQLite data import (all regions)     |

---

## Recent Updates

### Version 0.6.0 (2025-11-06)

**Added**:

- Phase 9.5 enhanced classification module (`finance_ml/classification_enhanced.py`)
- Comprehensive Phase 9.5 data flow fix documentation
- Phase 9.7 analyst comparison enhancements
- New test suites for enhanced functionality (classification, dashboards, restructuring)

**Changed**:

- Major notebook reorganization (`ml_finance_model_main_v9.ipynb`) with Phase 9.5 and 9.7 integration
- Enhanced core modules with improved data flow and error handling
- Updated `finance_ml/__init__.py` exports

**Fixed**:

- Resolved Phase 9.5 data flow issues with tuple unpacking and data pipeline fixes
- Fixed classification module data handling
- Improved notebook phase ordering and cell organization

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

### Version 0.5.1 (2025-11-05)

- Phase 9.1 comprehensive 4-step imputation pipeline
- Comprehensive TDD test suite (21 tests, ≥80% coverage)
- Phase 9.1 notebook integration with visualizations
- Implementation guide documentation

### Version 0.5.0 (2025-11-02)

- Phase 9.2 enhanced EDA with 7 new analysis functions
- Benchmarking module with sector/regional comparisons
- 59 new tests (36 EDA + 23 benchmarking)
- Enhanced notebook with schema validation and error handling

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow PEP 8** code style (use `black`, `isort`)
3. **Write tests** for new functionality (TDD preferred)
4. **Run test suite** before submitting: `python -m unittest -v`
5. **Update documentation** (README, docstrings, CHANGELOG)
6. **Submit a pull request** with clear description

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

<!-- Version conflicts detected during README update (2025-11-06):
- pyproject.toml: version 0.5.1
- CHANGELOG.md: version 0.6.0
- README (old): version 2.1.0
TODO: Align version numbering across all files (recommend following CHANGELOG.md) -->

### Version Alignment

**TODO**: Resolve version conflicts across project files:

- `pyproject.toml` currently shows version `0.5.1`
- `CHANGELOG.md` shows latest version `0.6.0`
- Previous README showed version `2.1.0`
- **Recommendation**: Use CHANGELOG.md version (0.6.0) as source of truth and update pyproject.toml

### Optional Dependencies

Some optional dependencies (streamlit, dash) are not in `requirements.txt`. To use dashboards:

```powershell
pip install streamlit dash
```

Or add to `requirements.txt` for easier installation.

### TensorFlow Installation

TensorFlow is heavy and CPU-only install is sufficient for this project. If GPU acceleration is needed, follow official
TensorFlow GPU installation docs and ensure CUDA/cuDNN compatibility. If installation issues occur, TensorFlow can be
temporarily commented out in `requirements.txt`.

---

**Last Updated**: 2025-11-06  
**README Version**: 1.0 (aligned with CHANGELOG v0.6.0)
