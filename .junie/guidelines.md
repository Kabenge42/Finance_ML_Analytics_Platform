# Finance ML Analytics Platform — Development Guidelines

This document provides essential information for setting up, developing, and testing the Finance ML Analytics Platform.
It aligns with `README.md` and `docs/code_guidelines.md`.

## 1. Build/Configuration Instructions

### Prerequisites

- **Python**: Version 3.12, 3.13, or 3.14 is required.
- **Operating System**: Windows (primary support via PowerShell), Linux, or macOS.
- **Dependencies**: Managed via `requirements.txt`, `Pipfile`, and `pyproject.toml`.

### Tech Stack

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

### Automated Environment Setup

The easiest way to set up the development environment is by using the provided setup script:

```powershell
python tools\setup_environment.py
```

This script performs the following:

- Detects the Python version.
- Creates a virtual environment (`venv`).
- Upgrades `pip`.
- Installs all necessary dependencies from `requirements.txt`.
- Configures environment variables.
- (Optional) Sets up the PostgreSQL database and loads initial data.

### Manual Setup

If you prefer manual setup:

1. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```
2. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## 2. Execution and Entry Points

### Main Workflow

To execute the complete 8-phase ML workflow and portfolio optimization:

```powershell
python finance_ml_analytics_platform.py
```

### CLI Entry Points

The package provides several command-line entry points:

| Command               | Description                                |
|:----------------------|:-------------------------------------------|
| `finance-ml`          | Main analysis pipeline execution           |
| `finance-ml-analyze`  | Quick data analysis and exploration        |
| `finance-ml-validate` | Data validation and schema alignment check |

### Interactive Dashboards

| Dashboard              | Run Command                                              |
|:-----------------------|:---------------------------------------------------------|
| **Streamlit App**      | `streamlit run finance_ml/dashboards/streamlit_app.py`   |
| **Equities Dashboard** | `python finance_ml/dashboards/equities_dashboard_app.py` |
| **Dash App**           | `python finance_ml/dashboards/dash_app.py`               |

## 3. Testing Information

### Configuring and Running Tests

The project uses `pytest` as the primary testing framework, along with the built-in `unittest` module.

#### Running All Tests

To run the full test suite (which may take some time):

```powershell
pytest
```

#### Running Fast Tests

For quick verification during development, use the fast test runner:

```powershell
python tools\run_fast_tests.py
```

This script runs a subset of tests focused on utility modules, avoiding heavy ML training.

#### Running Specific Tests

You can run individual test files using `pytest` or `python`:

```powershell
pytest tests\test_etl_unified_pipeline.py
python tests\test_junie_demo.py
```

### Guidelines for Adding New Tests

- **Location**: Place new tests in the `tests/` directory.
- **Naming**: Test files should be named `test_*.py`. Test classes should inherit from `unittest.TestCase` or use
  `pytest` style functions.
- **Isolation**: Use small, deterministic samples. Avoid external service dependencies; use mocks or stubs where
  necessary.
- **Coverage**: Aim for ≥80% test coverage for new code.

### Testing Demonstration

A simple test `tests\test_junie_demo.py` was created to demonstrate the process:

```python
import unittest
from pathlib import Path

class TestJunieDemo(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(1 + 1, 2)
        
    def test_project_structure(self):
        project_root = Path(__file__).resolve().parent.parent
        self.assertTrue((project_root / "finance_ml").exists())
```

**Execution Result:**

```
> python tests\test_junie_demo.py
..
----------------------------------------------------------------------
Ran 2 tests in 0.000s
OK
```

## 4. Additional Development Information

### Code Style

The project follows standard Python styling conventions, enforced by:

- **Black**: Code formatting.
- **Flake8**: Linting and style guide enforcement.
- **Isort**: Import sorting.
- **Mypy**: Static type checking.

### Quick Reference Card

| Task                     | Code                                          |
|:-------------------------|:----------------------------------------------|
| **Load with ETL**        | `df, m = run_etl_pipeline(source='csv', ...)` |
| **Normalize column**     | `normalize_column_name(col)`                  |
| **Get column dtype**     | `get_expected_dtype('last_price')`            |
| **List price columns**   | `list_price_cols()`                           |
| **Build features**       | `build_features(df, preset='comprehensive')`  |
| **Validate predictions** | `validate_predictions_schema(df)`             |

### Architecture

The platform follows a structured multi-phase workflow for machine learning and portfolio optimization.

#### 8-Phase ML Workflow

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

#### 7-Phase Portfolio Optimization

Implements risk-adjusted return maximization, efficient frontier visualization, and backtesting.

#### Design Principles

- **Unified Schema Module**: Use `finance_ml.core.schema` as the single source of truth for column definitions (
  `COLUMN_SCHEMA`).
- **Modular Feature Engineering**: Feature functions are organized by domain in `finance_ml/features/advanced/`.
- **Modular ETL**: ETL configuration and pipeline stages are decoupled in `finance_ml/etl/`.
- **Module Support**: Reusable logic is extracted into the `finance_ml/` package, with `finance_ml.core` containing
  shared constants and schema.

### Critical Imports

```python
# Schema
from finance_ml.core.schema import COLUMN_SCHEMA, normalize_column_name, list_price_cols

# ETL
from finance_ml.etl import run_etl_pipeline, ETLConfig

# Features
from finance_ml.ml_workflow.features.api import build_features
```

### Project Structure

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
