# Finance ML Analytics Platform — Code Guidelines

**Version:** 1.8  
**Last Updated:** 2025-11-26  
**Package Version:** 0.9.1  
**Model Version:** v9_9

These guidelines codify conventions for the Finance ML Analytics Platform, covering technology stack, configuration,
architecture, function signatures, column naming, and best practices. They align with the project's 8-phase ML
workflow (Phase 9.1-9.8) and 7-phase Portfolio Optimization workflow.

**Recent Updates (v1.8):**

- **NEW:** Added Section 18 Portfolio Optimization Workflow (2025-11-26)
    - **Section 18.1**: Workflow Overview — 7-phase architecture with module mapping
    - **Section 18.2**: Return Calculation Best Practices — Critical policy for expected return bounds (MAX=0.29,
      MIN=-0.50)
    - **Section 18.3**: Price Column Integration — PRICE_COLUMNS registry (21 columns, 4 categories)
    - **Section 18.4**: Phase 9.3 Feature Integration — 196 engineered features for return prediction
    - **Section 18.5**: Ensemble Model Best Practices — Multi-model and dynamic weighting
    - **Section 18.6**: Black-Litterman ML Integration — ML-derived views and regime detection
    - **Section 18.7**: Robust Covariance Estimation — Ledoit-Wolf shrinkage, EWM methods
    - **Section 18.8**: Portfolio Validation Diagnostics — Return and Sharpe ratio validation
    - **Section 18.9**: Configuration Constants Summary — Centralized constants reference
    - **Section 18.10**: Test Coverage Requirements — 90+ tests for portfolio optimization
    - Implementation details in `portfolio_optimization_enhancement_plan.md` Phase 7

**Previous Updates (v1.7):**

- Added Section 8.5 Preprocessing Stage Naming and Semantic Column Classification (2025-11-25)
    - **Section 8.5.1**: Column Semantic Classification — Five semantic categories (price, market value, ratio,
      percentage, count)
    - **Section 8.5.2**: Price Column Preservation Policy — Price columns must never be winsorized, scaled, or
      transformed in place
    - **Section 8.5.3**: Alternative Transformations for Skewed Data — Use log-transforms instead of winsorization for
      market value columns
    - CI/CD validation via 36 tests in test_column_semantics.py, test_selective_winsorization.py,
      test_log_transforms.py, test_selective_scaling.py
    - New modules: column_semantics.py (324 lines), transforms.py (214 lines)
    - Updated functions: outliers.py (winsorize_by_sector), scaling.py (scale_features) with exclude_price_columns=True
      defaults
    - **Business Impact**: Protects core valuation metric `(Predicted_Target - Last_Price) / Last_Price` from corruption

**Previous Updates (v1.6):**

- Added Section 5.5 Column Normalization Consistency Policy (2025-11-24)
- Updated technology stack from pyproject.toml (Python 3.12-3.14, setuptools build system)
- Clarified CLI entry points: finance-ml, finance-ml-analyze, finance-ml-validate
- Updated package architecture with 14 ml_workflow submodules
- Confirmed Schema v1.3 with 318 columns (262 original + 48 Phase 9.3 + 8 additional)
- Added Python Script/Module Review Checklist (Section 6.2) with AST-based static analysis

---

## Table of Contents

1. [Overview and Technology Stack](#1-overview-and-technology-stack)
2. [Configuration Constants](#2-configuration-constants)
3. [Main Scripts and Entry Points](#3-main-scripts-and-entry-points)
4. [Finance_ML Package Architecture](#4-finance_ml-package-architecture)
5. [Column Naming and Mapping](#5-column-naming-and-mapping)
6. [Code Review Checklist](#6-code-review-checklist)
7. [Standardized Function Signatures](#7-standardized-function-signatures)
8. [Notebook Best Practices and TDD Conventions](#8-notebook-best-practices-and-tdd-conventions)
9. [Column Schema and DataFrame Conventions](#9-column-schema-and-dataframe-conventions)
10. [Data Split and Leakage Policy](#10-data-split-and-leakage-policy)
11. [Standardized Predictions Schema](#11-standardized-predictions-schema)
12. [Sector Metrics and Calibration](#12-sector-metrics-and-calibration)
13. [Outlier Safety Rails Policy](#13-outlier-safety-rails-policy)
14. [Uncertainty and Prediction Intervals](#14-uncertainty-and-prediction-intervals)
15. [Jupyter Notebook Guidelines](#15-jupyter-notebook-guidelines)
16. [Model Optimization and Performance](#16-model-optimization-and-performance)
17. [Styles Guides for Visual Elements](#17-styles-guides-for-visual-elements)
18. [Portfolio Optimization Workflow](#18-portfolio-optimization-workflow)

---

## 1. Overview and Technology Stack

### 1.1 Project Overview

Finance ML Analytics Platform is a comprehensive toolkit for quantitative equity analysis combining unified data
pipelines, modular Python packages, interactive notebooks, and production-ready CLI tools.

**Business Objective:** Predict Stock Price Targets for all stocks in the portfolio to support investment decisions and
portfolio optimization.

**Target Variable:** "Predicted Price Target" for regression modeling

The platform implements a sophisticated **8-phase ML workflow**:

1. **Phase 9.1**: Loading and preprocessing with 6-step imputation strategy
2. **Phase 9.2**: Enhanced exploratory data analysis with statistical testing
3. **Phase 9.3**: Advanced feature engineering (Schema v1.3, 318 columns)
4. **Phase 9.4**: Multi-class event classification
5. **Phase 9.5**: Sector-optimized regression with quantile models
6. **Phase 9.6**: Model evaluation and error analysis
7. **Phase 9.7**: Identification of under/overvalued stocks
8. **Phase 9.8**: Comprehensive analytics and reporting

### 1.2 Technology Stack

**Language & Runtime:**

- Python: 3.12, 3.13, or 3.14 (officially supported per `pyproject.toml`)
- Package Manager: pip with `requirements.txt` and `pyproject.toml` (PEP 621)
- Build System: setuptools ≥68.0

**Core Libraries:**

- **Data**: pandas ≥2.0.0, numpy (1.26+ for py<3.14, 2.0+ for py≥3.14), scipy ≥1.11.0, statsmodels ≥0.14.0
- **ML Core**: scikit-learn ≥1.4.0, imbalanced-learn ≥0.11.0
- **Gradient Boosting**: XGBoost ≥2.0.3, LightGBM ≥4.0.0, CatBoost ≥1.2.0 (py<3.14)
- **Explainability**: SHAP 0.50.0 (py<3.14)
- **Visualization**: matplotlib ≥3.7.0, seaborn ≥0.12.0, plotly ≥5.14.0
- **Dashboards**: streamlit, dash
- **Utilities**: joblib ≥1.3.0, tqdm ≥4.65.0, xlsxwriter ≥3.1.0, psutil ≥5.9.0

**Optional Dependencies:**

- **Deep Learning**: TensorFlow ≥2.13.0 (py<3.14), scikeras ≥0.12.0
- **Database**: PostgreSQL (psycopg2-binary ≥2.9.0, SQLAlchemy ≥2.0.0), SQLite 3
- **Development**: pytest ≥7.4.0, pytest-cov ≥4.1.0, black ≥23.0.0, flake8 ≥6.0.0, mypy ≥1.5.0, isort ≥5.12.0
- **Advanced Features**: boruta ≥0.3.0

**Database Systems:**

- **Primary**: PostgreSQL 15+ (recommended for production)
- **Alternative**: SQLite 3 (for quick local testing)
- **Schema**: 318 columns in equities/all_stocks tables

**Development Tools:**

- Testing: unittest (built-in), pytest (optional), coverage
- Code Quality: black (line-length 100), flake8, mypy, isort
- Notebooks: Jupyter, notebook ≥7.0.0, ipykernel ≥6.25.0

---

## 2. Configuration Constants

All configuration constants are defined in the notebook and Python scripts following the **Single Source of Truth**
principle. Constants are defined once and validated at initialization.

### 2.1 Core Constants

```python
import os

# Target columns (code_guidelines.md Section 8.2)
TARGET_COL = 'price_target'  # Canonical target
TARGET_COL_FALLBACK = 'last_price'  # Fallback target

# Data splits
TEST_SIZE = 0.2
TRAIN_SIZE = 1 - TEST_SIZE
CV_FOLDS = 5

# Quantile regression
QUANTILES = [0.1, 0.5, 0.9]
LOWER_QUANTILE = QUANTILES[0]
MEDIAN_QUANTILE = QUANTILES[1]
UPPER_QUANTILE = QUANTILES[2]

# Sector constraints
MIN_SECTOR_SAMPLES = 20

# Portfolio constraints
MAX_SECTOR_WEIGHT = 0.25
MAX_SINGLE_POSITION = 0.10

# Outlier thresholds (aligned with ml_finance_model_main.ipynb)
IQR_MULTIPLIER = 2.5  # More conservative to preserve valid extreme values
ZSCORE_THRESHOLD = 3.0
WINSORIZE_LOWER = 0.10  # 10th percentile (less aggressive)
WINSORIZE_UPPER = 0.90  # 90th percentile (less aggressive)

# Confidence scoring
CONFIDENCE_LOW_THRESHOLD = 0.50
CONFIDENCE_MEDIUM_THRESHOLD = 0.75

# Random seed and versioning
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v9_9')
```

> **Note:** The winsorization bounds (0.10/0.90) are intentionally less aggressive than traditional (0.01/0.99) to
> preserve more valid extreme values in financial data (e.g., high-growth stocks, mega-cap companies). Combined with
> the Price Column Preservation Policy (Section 8.5.2), this ensures business-critical valuation metrics remain
> accurate.

### 2.2 Environment Variables

Environment variables provide runtime configuration overrides (see `environment_variables.txt`):

```python
import os

# Required
TF_CPP_MIN_LOG_LEVEL = '2'  # Reduce TensorFlow verbosity

# Optional
DATA_DIR = os.getenv('DATA_DIR', 'data')
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'outputs')
CACHE_DIR = os.getenv('CACHE_DIR', '.cache')
DB_URL = os.getenv('DB_URL', 'postgresql+psycopg2://postgres:@localhost:5432/postgres')
RANDOM_SEED = os.getenv('RANDOM_SEED', '42')
N_JOBS = int(os.getenv('N_JOBS', '-1'))
MEMORY_LIMIT = os.getenv('MEMORY_LIMIT', '8GB')
```

### 2.3 Configuration Validation

All configurations should be validated at initialization:

```python
def validate_configuration():
   """Validate notebook/script configuration constants."""
   # Validate target columns
   if not TARGET_COL or not isinstance(TARGET_COL, str):
      raise ValueError(f"TARGET_COL must be non-empty string: {TARGET_COL}")

   # Validate test size
   if not (0 < TEST_SIZE < 1):
      raise ValueError(f"TEST_SIZE must be between 0 and 1: {TEST_SIZE}")

   # Validate CV folds
   if CV_FOLDS < 2:
      raise ValueError(f"CV_FOLDS must be >= 2: {CV_FOLDS}")

   # Validate quantiles
   if not all(0 < q < 1 for q in QUANTILES):
      raise ValueError(f"All QUANTILES must be between 0 and 1: {QUANTILES}")

   # Validate monotonicity
   if QUANTILES != sorted(QUANTILES):
      raise ValueError(f"QUANTILES must be monotonically increasing: {QUANTILES}")

   return True
```

---

## 3. Main Scripts and Entry Points

### 3.1 Main Entry Points

| Script/Tool                   | Description                   | Usage                                                 |
|-------------------------------|-------------------------------|-------------------------------------------------------|
| `ml_finance_model_main.ipynb` | Main notebook (Phase 9.1-9.8) | `jupyter notebook ml_finance_model_main.ipynb`        |
| `ml_finance_model_main.py`    | Python script version         | `python ml_finance_model_main.py --data-source auto`  |
| `finance-ml`                  | CLI: Full pipeline            | `finance-ml --data-source auto --output-dir outputs`  |
| `finance-ml-analyze`          | CLI: EDA/analytics only       | `finance-ml-analyze --data-source csv`                |
| `finance-ml-validate`         | CLI: Validation only          | `finance-ml-validate --data-source db --db-url <url>` |

### 3.2 CLI Entry Points (from pyproject.toml)

```toml
[project.scripts]
finance-ml = "finance_ml.cli:main"
finance-ml-analyze = "finance_ml.cli:analyze_main"
finance-ml-validate = "finance_ml.cli:validate_main"
```

### 3.3 Dashboard Applications

| Script             | Description         | Usage                                                  |
|--------------------|---------------------|--------------------------------------------------------|
| `streamlit_app.py` | Streamlit dashboard | `streamlit run finance_ml/dashboards/streamlit_app.py` |
| `dash_app.py`      | Dash dashboard      | `python finance_ml/dashboards/dash_app.py`             |

### 3.4 Utility Scripts (tools/)

| Script                   | Description                                      |
|--------------------------|--------------------------------------------------|
| `import_sqlite.py`       | Import CSVs into SQLite with chunked processing  |
| `validate_csv_import.py` | Validate CSV data quality before import          |
| `analyze_notebook.py`    | Analyze notebook structure and cells             |
| `analyze_predictions.py` | Analyze model prediction outputs                 |
| `run_fast_tests.py`      | Run fast unit tests (no heavy training)          |
| `verify_outputs.py`      | Verify expected output files exist and non-empty |

### 3.5 Database Scripts

| Script                              | Description                              |
|-------------------------------------|------------------------------------------|
| `create_equities_schema.sql`        | PostgreSQL schema creation (318 columns) |
| `import_equities_data.sql`          | PostgreSQL data import (all regions)     |
| `create_equities_schema_sqlite.sql` | SQLite schema creation                   |
| `import_equities_data_sqlite.sql`   | SQLite data import (all regions)         |

---

## 4. Finance_ML Package Architecture

### 4.1 Package Structure

The `finance_ml` package follows a phase-aligned architecture with 13 ml_workflow subpackages:

```
finance_ml/
├── __init__.py                    # Package-level exports (v0.8.3)
├── cli.py                         # CLI entry points (main, analyze_main, validate_main)
├── ml_workflow/                   # Main ML workflow package (Phase 9.1-9.8)
│   ├── __init__.py
│   ├── preprocessing/             # Phase 9.1: Data preprocessing (10 modules)
│   │   ├── __init__.py            # Public API exports
│   │   ├── imputation.py          # 6-step imputation strategy (4step for backward compat)
│   │   ├── outliers.py            # Outlier detection and winsorization
│   │   ├── scaling.py             # Feature scaling with price column exclusion
│   │   ├── dtypes.py              # Schema-aware datatype detection
│   │   ├── pipeline.py            # Preprocessing pipeline orchestration
│   │   ├── column_semantics.py    # Semantic column classification (5 categories)
│   │   ├── transforms.py          # Log-transforms for skewed data
│   │   ├── data.py                # Data loading and normalization
│   │   └── quality.py             # Data quality metrics
│   ├── eda/                       # Phase 9.2: Exploratory Data Analysis
│   │   ├── __init__.py
│   │   ├── eda.py                 # Core EDA functions
│   │   ├── benchmarking.py        # Sector/region benchmarking
│   │   └── statistical_tests.py   # Statistical testing
│   ├── features/                  # Phase 9.3: Feature Engineering
│   │   ├── __init__.py
│   │   ├── core.py                # Core feature functions
│   │   ├── advanced.py            # Advanced feature engineering
│   │   ├── selection.py           # Feature selection (RF, MI)
│   │   └── api.py                 # High-level API with presets
│   ├── classification/            # Phase 9.4: Event Classification
│   │   ├── __init__.py
│   │   ├── labels.py              # Label generation (13 methods)
│   │   ├── models.py              # Classification models
│   │   ├── tuning.py              # Hyperparameter tuning
│   │   └── evaluation.py          # Classification evaluation
│   ├── regression/                # Phase 9.5: Regression Models
│   │   ├── __init__.py
│   │   ├── models.py              # Regression models (XGBoost, LightGBM, CatBoost)
│   │   ├── quantile.py            # Quantile regression
│   │   ├── constraints.py         # Non-negativity constraints
│   │   ├── stacking.py            # Ensemble stacking
│   │   └── safety_rails.py        # Prediction safety rails
│   ├── evaluation/                # Phase 9.6: Model Evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py             # Evaluation metrics
│   │   ├── uncertainty.py         # Uncertainty quantification
│   │   ├── safety_rails.py        # Outlier safety rails
│   │   └── calibration.py         # Model calibration
│   ├── analytics/                 # Phase 9.7: Analytics
│   │   ├── __init__.py
│   │   ├── mispricing.py          # Mispricing score calculation
│   │   ├── stock_selection.py     # Stock ranking and selection
│   │   ├── portfolio.py           # Portfolio optimization
│   │   ├── risk.py                # Risk metrics (VaR, CVaR, etc.)
│   │   ├── ml_returns.py          # ML-based return prediction
│   │   ├── attribution.py         # Performance attribution
│   │   ├── analyst_comparison.py  # Analyst comparison
│   │   ├── eval.py                # Legacy eval functions
│   │   └── portfolio_reporting.py # Portfolio reporting
│   ├── reporting/                 # Phase 9.8: Reporting
│   │   ├── __init__.py
│   │   ├── dashboard_data.py      # Dashboard data preparation
│   │   └── quality_alerts.py      # Data quality alerts
│   ├── data/                      # Data loading and schema
│   │   ├── __init__.py
│   │   ├── loaders.py             # CSV and DB loaders
│   │   └── schema.py              # Column schema registry (318 columns)
│   ├── config/                    # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py            # Configuration settings
│   ├── core/                      # Core utilities
│   │   ├── __init__.py
│   │   └── utils.py               # Utility functions
│   ├── quality/                   # Code quality tools
│   │   ├── __init__.py
│   │   ├── script_review.py       # AST-based static analysis (Section 6.2)
│   │   └── notebook_review.py     # Notebook review tools
│   └── validation/                # Data validation
│       ├── __init__.py
│       └── validators.py          # Validation functions
└── dashboards/                    # Interactive dashboards
    ├── streamlit_app.py           # Streamlit application
    ├── dash_app.py                # Dash application
    └── portfolio_widgets.py       # Portfolio dashboard widgets
```

> **Note:** The package also contains legacy modules at the `ml_workflow/` level (e.g., `advanced_preprocessing.py`,
> `advanced_models.py`, `advanced_features.py`) for backward compatibility. New code should use the subpackage imports.

### 4.2 Phase Alignment

Each subpackage maps directly to a business phase:

| Phase | Subpackage        | Description                                           |
|-------|-------------------|-------------------------------------------------------|
| 9.1   | `preprocessing/`  | Data loading, imputation, scaling, outlier handling   |
| 9.2   | `eda/`            | Exploratory analysis, benchmarking, statistical tests |
| 9.3   | `features/`       | Feature engineering (318 columns, Schema v1.3)        |
| 9.4   | `classification/` | Event classification (13 label methods)               |
| 9.5   | `regression/`     | Regression models, quantile, stacking                 |
| 9.6   | `evaluation/`     | Metrics, uncertainty, calibration, safety rails       |
| 9.7   | `analytics/`      | Mispricing, portfolio optimization, risk metrics      |
| 9.8   | `reporting/`      | Dashboard data, quality alerts, reporting             |

### 4.3 Import Patterns

**Recommended Pattern (Module-level imports):**

```python
# Phase 9.1: Preprocessing
from finance_ml.ml_workflow.preprocessing import (
    impute_missing_values,
    detect_outliers,
    winsorize_features,
    scale_features,
    detect_and_cast_dtypes,
)

# Phase 9.2: EDA
from finance_ml.ml_workflow.eda import (
    compute_descriptive_stats,
    plot_distributions,
    compute_correlation_matrix,
)

# Phase 9.3: Features
from finance_ml.ml_workflow.features import (
    build_valuation_features,
    build_momentum_features,
    build_quality_features,
    select_features_rf,
)

# Phase 9.4: Classification
from finance_ml.ml_workflow.classification import (
    create_event_labels,
    train_event_classifier,
    tune_classifier_hyperparameters,
)

# Phase 9.5: Regression
from finance_ml.ml_workflow.regression import (
    train_sector_models,
    train_quantile_regressor,
    apply_nonnegative_constraint,
)

# Phase 9.6: Evaluation
from finance_ml.ml_workflow.evaluation import (
    compute_regression_metrics,
    build_quantile_diagnostics,
    track_constraint_violations,
    calibrate_sector_bias,
)

# Phase 9.7: Analytics
from finance_ml.ml_workflow.analytics import (
    calculate_mispricing_scores,
    rank_stocks,
    optimize_portfolio,
    compute_risk_metrics,
)

# Phase 9.8: Reporting
from finance_ml.ml_workflow.reporting import (
    generate_dashboard_data,
    create_quality_alerts,
)
```

**Data Schema Access:**

```python
from finance_ml.ml_workflow.data.schema import (
   COLUMN_SCHEMA,
   PHASE93_FEATURE_INPUTS,
   get_expected_dtype,
   get_column_role,
   normalize_column_name,
   )
```

**Configuration:**

```python
from finance_ml.ml_workflow.config import (
   get_default_config,
   validate_config,
   )
```

---

## 5. Column Naming and Mapping

### 5.1 Normalization Rules

**SQL to Python Column Name Mapping:**

SQL columns (mixed-case with spaces) are normalized to Python (lowercase with underscores):

```python
# Normalization function
def normalize_column_name(col: str) -> str:
   """Normalize column name: lowercase, replace non-alphanumeric with underscore."""
   import re
   normalized = re.sub(r'[^0-9a-zA-Z]+', '_', col)
   normalized = normalized.strip('_').lower()
   return normalized
```

### 5.2 Common Column Mappings

| SQL Column Name            | Normalized Python Name | Role            | DType    |
|----------------------------|------------------------|-----------------|----------|
| `"Ticker"`                 | `ticker`               | id              | string   |
| `"ISIN"`                   | `isin`                 | id              | string   |
| `"Sector"`                 | `sector`               | categorical     | category |
| `"Industry"`               | `industry`             | categorical     | category |
| `"Region"`                 | `region`               | categorical     | category |
| `"Country"`                | `country`              | categorical     | category |
| `"Last Price"`             | `last_price`           | feature         | float    |
| `"Price Target"`           | `price_target`         | target          | float    |
| `"Price Target - Median"`  | `price_target_median`  | target_fallback | float    |
| `"Price Target (YTD Ago)"` | `price_target_ytd_ago` | target_fallback | float    |
| `"Market Cap"`             | `market_cap`           | feature         | float    |
| `"Enterprise Value"`       | `enterprise_value`     | feature         | float    |
| `"P/E (NTM)"`              | `p_e_ntm`              | feature         | float    |
| `"P/E (LTM)"`              | `p_e_ltm`              | feature         | float    |
| `"P/B (LTM)"`              | `p_b_ltm`              | feature         | float    |
| `"EV/Sales (LTM)"`         | `ev_sales_ltm`         | feature         | float    |
| `"EV/EBITDA (LTM)"`        | `ev_ebitda_ltm`        | feature         | float    |
| `"EBITDA (LTM)"`           | `ebitda_ltm`           | feature         | float    |
| `"Total Revenues (LTM)"`   | `total_revenues_ltm`   | feature         | float    |
| `"Net Income/Adj. (LTM)"`  | `net_income_adj_ltm`   | feature         | float    |
| `"Volatility (1M)"`        | `volatility_1m`        | feature         | float    |
| `"Beta (5Y)"`              | `beta_5y`              | feature         | float    |
| `"Analyst Rating"`         | `analyst_rating`       | feature         | float    |
| `"Price (5D Ago)"`         | `price_5d_ago`         | feature         | float    |
| `"Price (1W Ago)"`         | `price_1w_ago`         | feature         | float    |
| `"Price (1M Ago)"`         | `price_1m_ago`         | feature         | float    |
| `"Price (3M Ago)"`         | `price_3m_ago`         | feature         | float    |
| `"Price (6M Ago)"`         | `price_6m_ago`         | feature         | float    |
| `"Price (1Y Ago)"`         | `price_1y_ago`         | feature         | float    |
| `"Price (3Y Ago)"`         | `price_3y_ago`         | feature         | float    |
| `"Price (5Y Ago)"`         | `price_5y_ago`         | feature         | float    |
| `"Price (QTD Ago)"`        | `price_qtd_ago`        | feature         | float    |
| `"52W High/Adj."`          | `52w_high_adj`         | feature         | float    |
| `"52W Low/Adj."`           | `52w_low_adj`          | feature         | float    |
| `"EMA (20D)"`              | `ema_20d`              | feature         | float    |
| `"EMA (50D)"`              | `ema_50d`              | feature         | float    |
| `"EMA (100D)"`             | `ema_100d`             | feature         | float    |
| `"EMA (250D)"`             | `ema_250d`             | feature         | float    |

### 5.3 Schema Registry

The authoritative column schema is defined in `finance_ml/ml_workflow/data/schema.py`:

```python
from finance_ml.ml_workflow.data.schema import (
   COLUMN_SCHEMA,  # Dict[str, Dict[str, str]] - 318 columns
   get_expected_dtype,  # Get dtype for a column
   get_column_role,  # Get role for a column
   list_numeric_feature_cols,  # List all numeric features
   list_categorical_cols,  # List all categorical columns
   list_date_cols,  # List all date columns
   normalize_column_name,  # Normalize a column name
   )

# Example usage
dtype = get_expected_dtype('last_price')  # Returns 'float'
role = get_column_role('sector')  # Returns 'categorical'
numeric_cols = list_numeric_feature_cols()  # Returns list of numeric feature columns
```

### 5.4 Phase 9.3 Feature Categories

Schema v1.3 organizes features into categories (defined in `PHASE93_FEATURE_INPUTS`):

- **Momentum**: price_momentum_1m/3m/6m, rsi_14d/30d, ma_crossover_signal, return_stability_score
    - **Input Columns** (from PRICE_COLUMNS): price_5d_ago, price_1w_ago, price_1m_ago, price_3m_ago, price_6m_ago,
      price_1y_ago, price_3y_ago, price_5y_ago, price_qtd_ago
- **Technical Indicators**: ema_crossover_signal, price_vs_52w_range, ema_deviation
    - **Input Columns** (from PRICE_COLUMNS): 52w_high_adj, 52w_low_adj, ema_20d, ema_50d, ema_100d, ema_250d
- **Valuation**: p_e_ratio, p_b_ratio, p_s_ratio, ev_ebitda_ratio, peg_ratio, price_to_fcf
- **Profitability**: gross_margin_pct, operating_margin_pct, net_margin_pct, roe, roa, roic
- **Quality/Risk**: altman_z_score, debt_to_equity, current_ratio, interest_coverage, leverage_ratio
- **Cash Flow**: fcf_yield, ocf_to_sales, capex_intensity, fcf_growth
- **Growth**: revenue_growth_yoy, earnings_growth_yoy, sales_cagr_3y, ebitda_growth

### 5.5 Column Normalization Consistency Policy

**Version:** 1.6 (added 2025-11-24)  
**Status:** ENFORCED via CI/CD tests

**Canonical Normalization Function:**

All column name normalization MUST use `normalize_column_name()` from `finance_ml.ml_workflow.data.schema`:

```python
from finance_ml.ml_workflow.data.schema import normalize_column_name

# Correct usage
normalized = normalize_column_name("# Strong Sell Ratings")  # → "num_strong_sell_ratings"
normalized = normalize_column_name("Selling General & Admin Expenses/Total (FQ)")  # → "selling_general_and_admin_expenses_total_fq"
normalized = normalize_column_name("1-Day %")  # → "1_day_pct"
```

**Normalization Rules:**

The canonical function applies these transformations in order:

1. `#` → `num` (analyst rating counts)
2. `%` → `pct` (percentages)
3. `&` → `and` (conjunctions)
4. `/` → `_` (ratios, divisions)
5. `(`, `)` → removed (parentheses)
6. `-` → `_` (hyphens, negative indicators)
7. Multiple spaces → single `_`
8. Multiple underscores → single `_`
9. Leading/trailing `_` → stripped
10. Lowercase conversion

**Enforcement Rules:**

1. **ALL** column name normalization MUST use `normalize_column_name()` from schema.py
2. **NO** alternative normalization functions allowed in data loading or preprocessing
3. All `COLUMN_SCHEMA` keys MUST be producible via `normalize_column_name()` from SQL schema column names
4. Test coverage REQUIRED: Any PR touching normalization must include round-trip tests
5. CI pipeline runs `test_schema_normalization.py` to prevent drift

**Examples of Correct vs. Incorrect:**

```python
# ✅ CORRECT: Use canonical function
from finance_ml.ml_workflow.data.schema import normalize_column_name
df.columns = [normalize_column_name(col) for col in df.columns]

# ❌ INCORRECT: Custom regex normalization
df.columns = df.columns.str.replace(r"[^0-9a-zA-Z]+", "_", regex=True).str.strip("_").str.lower()
# This produces "strong_sell_ratings" instead of "num_strong_sell_ratings"

# ❌ INCORRECT: Manual string operations
df.columns = [col.lower().replace(" ", "_").replace("#", "") for col in df.columns]
# Missing semantic transformations (# → num, & → and, etc.)
```

**Critical Columns Affected:**

- **Analyst Ratings** (5 columns): Must have `num_` prefix
    - `# Strong Sell Ratings` → `num_strong_sell_ratings`
    - `# Strong Buys Ratings` → `num_strong_buys_ratings`
    - `# Hold Ratings` → `num_hold_ratings`
    - `# Buys Ratings` → `num_buys_ratings`
    - `# Sell Ratings` → `num_sell_ratings`

- **SG&A Expenses** (4 columns): Must include `and` connector
    - `Selling General & Admin Expenses/Total (FQ)` → `selling_general_and_admin_expenses_total_fq`
    - (Similar for FY, -1FY, 5YAVGFQ variants)

- **Percentage Columns**: Must use `pct` suffix
    - `1-Day %` → `1_day_pct`
    - `Short Int. (%)` → `short_int_pct` (deprecated, removed from schema)

**Validation:**

CI/CD enforces normalization consistency via:

- `tests/test_schema_normalization.py` - 16 tests covering normalization rules and round-trip validation
- `tests/test_schema_completeness.py` - 17 tests validating COLUMN_SCHEMA integrity
- `tests/test_data_loading_normalization.py` - 7 integration tests for CSV/DB loading

**References:**

- Schema Registry: `finance_ml/ml_workflow/data/schema.py` (COLUMN_SCHEMA, normalize_column_name)
- Data Loading: `finance_ml/ml_workflow/preprocessing/data.py` (normalize_columns, load_from_csv)
- Test Suite: `tests/test_schema_normalization.py`, `tests/test_data_loading_normalization.py`
- Documentation: `docs/improvement_plan/data_preprocessing improvement_plan.md` (Section 0, TDD plan)

---

## 6. Code Review Checklist

### 6.1 Jupyter Notebook Review Checklist

**Configuration and Setup:**

- [ ] Configuration constants defined at top (Section 2)
- [ ] `validate_configuration()` called and passes
- [ ] Environment variables documented and used consistently
- [ ] Random seed set for reproducibility: `np.random.seed(RANDOM_SEED)`
- [ ] Output directories defined using pathlib: `OUTPUT_DIR = Path('outputs')`

**Data Loading and Preprocessing:**

- [ ] Data source clearly specified (CSV, PostgreSQL, SQLite)
- [ ] Column names normalized using `normalize_column_name()`
- [ ] Missing value handling uses 6-step imputation strategy
- [ ] Outliers detected and handled (winsorization, clipping)
- [ ] Data types validated against schema: `validate_dtypes_against_schema()`
- [ ] Critical date columns included: `last_updated`, `income_statement_report_date`, `next_earnings`,
  `dividend_record_*`

**Feature Engineering:**

- [ ] Features aligned with Phase 9.3 Schema v1.3 (318 columns)
- [ ] Feature preset used or documented: "basic", "momentum", "quality", "comprehensive"
- [ ] No target leakage in feature construction
- [ ] Feature importance analyzed and documented

**Model Training:**

- [ ] Train/test split follows Data Split Policy (Section 9)
- [ ] Cross-validation uses grouped or stratified strategy (no leakage)
- [ ] Hyperparameters documented and versioned
- [ ] Model artifacts saved with version: `MODEL_VERSION`

**Evaluation and Outputs:**

- [ ] Predictions follow Standardized Predictions Schema (Section 10)
- [ ] Quantile predictions satisfy monotonicity: `pred_p10 ≤ pred_p50 ≤ pred_p90`
- [ ] Non-negativity enforced for price predictions
- [ ] Sector metrics calculated and persisted
- [ ] Required output files exist and non-empty

**Code Quality:**

- [ ] No hard-coded paths (use environment variables and pathlib)
- [ ] Functions modularized (avoid monolithic cells)
- [ ] Execution order dependencies documented
- [ ] Cell outputs cleared before commit (for version control)
- [ ] Markdown cells provide context and explanations

### 6.2 Python Script/Module Review Checklist

**Code Structure:**

- [ ] Type hints used for function signatures
- [ ] Docstrings follow NumPy/Google style
- [ ] Imports organized: stdlib → third-party → local
- [ ] Functions follow single responsibility principle
- [ ] No global mutable state

**Function Signatures:**

- [ ] Training functions return dict with keys: `model`, `metrics`, `y_pred`, `y_proba`, `artifacts`
- [ ] Dataset prep functions return 5-tuple or `DatasetSplit` dataclass
- [ ] Column names use normalized schema (lowercase, underscores)

**Error Handling:**

- [ ] Input validation with clear error messages
- [ ] Graceful degradation for missing optional dependencies
- [ ] Logging used instead of print statements
- [ ] Exceptions documented in docstrings

**Testing:**

- [ ] Unit tests cover core functionality
- [ ] Tests use small deterministic samples
- [ ] Tests isolated from external services (mocks/stubs)
- [ ] Test coverage ≥80% for new code

**Documentation:**

- [ ] README updated if adding new scripts/tools
- [ ] CHANGELOG.md updated for significant changes
- [ ] Configuration documented in environment_variables.txt
- [ ] API documented in code_guidelines.md

**Code Quality:**

- [ ] Black formatted (line-length 100)
- [ ] Flake8 compliant
- [ ] Mypy type checks pass (where applicable)
- [ ] No unused imports or variables

### 6.2.2 Common Parameter Naming Conventions

To prevent parameter mismatch TypeErrors and maintain consistency across the codebase, follow these naming conventions:

**Data Parameters:**

- `data_df` or `df` — Full DataFrame input (raw or intermediate data)
- `features_df` — Feature matrix as DataFrame
- `predictions_df` — Predictions DataFrame with metadata (ticker, sector, y_true, y_pred, etc.)
- `X_train`, `X_test` — Feature arrays/DataFrames (NOT `X_tr`, `X_tst`)
- `y_train`, `y_test` — Target arrays/Series (NOT `y_tr`, `y_tst`)

**Column Name Parameters:**

- Use `*_col` suffix: `target_col`, `sector_col`, `date_col`, `region_col`
- NOT: `target_column`, `sector_name`, `date_field`, `region_colname`

**Output Parameters:**

- `output_dir` — Always Path or str for output directory (NOT `out_dir`, `save_dir`, `results_dir`)

**Model Parameters:**

- `model_info` — Dictionary containing model metadata (datasets, features, models, artifacts, metrics)
- NOT: Separate parameters like `datasets=`, `features=`, `models=`, etc. when a dict is expected

**Examples:**

```python
# ✅ CORRECT
safety_rails_sensitivity_app(
    data_df=all_stocks_preprocessed,
    output_dir=safety_rails_dir,
    thresholds=[0.01, 0.05, 0.1]
)

estimate_sector_bias(
    predictions_df=predictions_df,
    output_dir=calibration_dir,
    model_version=MODEL_VERSION
)

plot_metrics_by_sector_time(
    predictions_df=metrics_history_df,
    output_dir=calibration_dir,
    date_col="snapshot_date"
)

build_lineage_json(
    model_info={
        'datasets': datasets,
        'features': features,
        'models': models
    },
    output_dir=governance_dir,
    model_version=MODEL_VERSION
)

# ❌ INCORRECT
safety_rails_sensitivity_app(
    df_raw=all_stocks_preprocessed,  # Wrong: should be data_df
    out_dir=safety_rails_dir  # Wrong: should be output_dir
)

estimate_sector_bias(
    predictions_df=predictions_df,
    y_true_col="y_true",  # Wrong: function uses hardcoded column names
    sector_col="sector"
)

plot_metrics_by_sector_time(
    metrics_history=df,  # Wrong: should be predictions_df
    snapshot_date_col="date"  # Wrong: should be date_col
)

build_lineage_json(
    datasets=datasets,  # Wrong: should be in model_info dict
    features=features,
    models=models
)
```

**Validation:**

Use the static analyzer to check notebooks and scripts for parameter mismatches:

```bash
# Check notebook function signatures
python -m finance_ml.ml_workflow.quality.notebook_review ml_finance_model_main.ipynb

# Check Python script
python -m finance_ml.ml_workflow.quality.script_review path/to/script.py
```

Run `tests/test_evaluation_function_signatures.py` to validate evaluation module function signatures.

---

## 7. Standardized Function Signatures

### 7.1 Training Functions (train_*)

**Contract:** All model-training functions return a dict with these keys:

```python
{
   "model": fitted_estimator_or_pipeline,
   "metrics": Dict[str, float],  # e.g., accuracy, f1_macro, mae, rmse, r2
   "y_pred": array_like,  # Predictions aligned to input indices
   "y_proba": Optional[array_like],  # Class probabilities (classification only)
   "artifacts": Optional[Dict[str, Any]]  # feature_importance, confusion_matrix, etc.
   }
```

**Examples:**

```python
# Classification
from finance_ml.ml_workflow.classification import train_event_classifier

res = train_event_classifier(X, y, model="lightgbm")
assert set(res).issuperset({"model", "metrics", "y_pred"})
acc = res["metrics"].get("accuracy")
f1m = res["metrics"].get("f1_macro")
y_proba = res.get("y_proba")  # May be None if no predict_proba

# Regression
from finance_ml.ml_workflow.regression import train_and_evaluate_regression

res = train_and_evaluate_regression(df)
mae = res["metrics"].get("mae")
r2 = res["metrics"].get("r2")
y_pred = res["y_pred"]  # Series/DataFrame aligned to df index
```

**Backward Compatibility:** Legacy code expecting top-level metric keys (e.g., `res["mae"]`) should use shims during
transition. New code must use `res["metrics"]["mae"]`.

### 7.2 Dataset Preparation Functions

**Contract:** Dataset prep functions return a 5-tuple or dataclass:

```python
(X_train, X_test, y_train, y_test, meta)
```

Where `meta` is a dict including: `feature_names`, `categorical_features`, `target_name`, `indices`, and optional
`scalers`/`encoders`.

**Dataclass Option:**

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DatasetSplit:
   X_train: Any
   X_test: Any
   y_train: Any
   y_test: Any
   meta: Dict[str, Any]
```

### 7.3 Comprehensive Function Signatures by Phase

**Phase 9.1 — Preprocessing**

```python
from finance_ml.ml_workflow.preprocessing.pipeline import prepare_phase91_data

prepared_df, quality_stats = prepare_phase91_data(
        df: pd.DataFrame,
sector_column: str = "sector",
price_column: str = "last_price",
n_neighbors: int = 5,
return_stats: bool = True
)
# Returns: (preprocessed_df, quality_statistics_dict)

from finance_ml.ml_workflow.preprocessing.imputation import (
   apply_enhanced_imputation_strategy_6step,  # Current 6-step imputation (recommended)
   apply_zero_imputation,
   apply_knn_imputation_enhanced,
   apply_price_imputation,
   apply_median_imputation
   )

df_imputed = apply_enhanced_imputation_strategy_6step(
        df: pd.DataFrame,
zero_fill_columns: Optional[List[str]] = None,
knn_neighbors: int = 5,
price_columns: Optional[List[str]] = None
)
# Returns: DataFrame with all missing values imputed
```

**Phase 9.3 — Features**

```python
from finance_ml.ml_workflow.features.api import build_features

features_df = build_features(
        df: pd.DataFrame,
preset: str = "comprehensive"
)
# Presets: "basic", "momentum", "quality", "comprehensive"

from finance_ml.ml_workflow.features.advanced import (
   engineer_valuation_ratios,
   engineer_momentum_features,
   build_comprehensive_features
   )
```

**Phase 9.4 — Classification**

```python
from finance_ml.ml_workflow.classification.labels import create_event_labels

labels = create_event_labels(
        df: pd.DataFrame,
method: str = "price_momentum"
)
# Methods: price_momentum, valuation, fundamental, volatility, analyst_rating, market_events
# Returns: array of labels (0-4 scale)
```

**Phase 9.5 — Regression**

```python
from finance_ml.ml_workflow.regression.models import (
   train_xgboost_regressor,
   train_sector_optimized_regressors
   )

from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor

quantile_result = train_quantile_regressor(
        X_train, y_train, X_test, y_test,
        quantiles: List[float] = [0.1, 0.5, 0.9]
)
# Returns: {"model", "metrics", "quantile_predictions": {q: pred_array}}
```

**Phase 9.6 — Evaluation**

```python
from finance_ml.ml_workflow.evaluation.metrics import (
   calculate_regression_metrics,
   calculate_sector_metrics
   )

metrics = calculate_regression_metrics(y_true, y_pred, include_mape=True)
# Returns: {"mae", "rmse", "r2", "mape"}

sector_metrics_df = calculate_sector_metrics(df, y_true_col, y_pred_col, sector_col)
# Returns: DataFrame with sector-level MAE, RMSE, R2, MAPE, count
```

**Phase 9.7 — Analytics**

```python
from finance_ml.ml_workflow.analytics import (
   calculate_mispricing_score,
   rank_undervalued_stocks
   )

mispricing = calculate_mispricing_score(last_price, predicted_target)
# Returns: (predicted_target - last_price) / last_price

undervalued_df = rank_undervalued_stocks(df, mispricing_col='mispricing_score', top_n=20)
# Returns: Top N undervalued stocks
```

---

## 8. Notebook Best Practices and TDD Conventions

This section establishes formal standards for notebook development following Test-Driven Development (TDD) principles,
ensuring maintainability, testability, and consistency across the project. All policies are validated via
`tests/test_notebook_tdd_compliance.py` (24 tests passing).

### 8.1 Centralized Configuration Constants (Single Source of Truth)

**Policy**: All configuration constants must be defined once in a dedicated configuration cell at the top of the
notebook. Never use magic numbers or duplicate constant definitions across cells.

**Required Constants**:

```python
# Target Configuration
TARGET_COL = 'price_target'  # Canonical target (code_guidelines.md Section 2.2)
TARGET_COL_FALLBACK = 'last_price'  # Canonical fallback target

# Data Split Configuration
TEST_SIZE = 0.2
TRAIN_SIZE = 1 - TEST_SIZE
CV_FOLDS = 5

# Quantile Regression Configuration
QUANTILES = [0.1, 0.5, 0.9]
LOWER_QUANTILE = QUANTILES[0]
MEDIAN_QUANTILE = QUANTILES[1]
UPPER_QUANTILE = QUANTILES[2]

# Sector Analysis Configuration
MIN_SECTOR_SAMPLES = 20
MAX_SECTOR_WEIGHT = 0.25
MAX_SINGLE_POSITION = 0.10

# Outlier Detection Configuration
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99

# Confidence Thresholds
CONFIDENCE_LEVEL = 0.80
ALPHA = 1 - CONFIDENCE_LEVEL

# Reproducibility
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
np.random.seed(RANDOM_SEED)
```

**Validation Pattern**:

Every notebook should include a `validate_configuration()` function to enforce invariants:

```python
def validate_configuration():
   """
   Validate all configuration constants meet required constraints.
   
   Raises:
       ValueError: If any configuration constant is invalid
   """
   # Validate target columns
   if not TARGET_COL or not isinstance(TARGET_COL, str):
      raise ValueError(f"TARGET_COL must be a non-empty string, got: {TARGET_COL}")
   if not TARGET_COL_FALLBACK or not isinstance(TARGET_COL_FALLBACK, str):
      raise ValueError(f"TARGET_COL_FALLBACK must be a non-empty string, got: {TARGET_COL_FALLBACK}")

   # Validate split configuration
   if not (0 < TEST_SIZE < 1):
      raise ValueError(f"TEST_SIZE must be between 0 and 1, got: {TEST_SIZE}")
   if not (0 < TRAIN_SIZE < 1):
      raise ValueError(f"TRAIN_SIZE must be between 0 and 1, got: {TRAIN_SIZE}")
   if not abs((TRAIN_SIZE + TEST_SIZE) - 1.0) < 0.01:
      raise ValueError(f"TRAIN_SIZE + TEST_SIZE must equal 1.0, got: {TRAIN_SIZE + TEST_SIZE}")

   # Validate CV folds
   if not isinstance(CV_FOLDS, int) or CV_FOLDS < 2:
      raise ValueError(f"CV_FOLDS must be an integer >= 2, got: {CV_FOLDS}")

   # Validate quantiles
   if not QUANTILES or not isinstance(QUANTILES, list):
      raise ValueError(f"QUANTILES must be a non-empty list, got: {QUANTILES}")
   for q in QUANTILES:
      if not (0 <= q <= 1):
         raise ValueError(f"All quantiles must be between 0 and 1, got: {q}")
   if len(QUANTILES) != len(set(QUANTILES)):
      raise ValueError(f"QUANTILES must not contain duplicates, got: {QUANTILES}")

   # Validate sector configuration
   if not isinstance(MIN_SECTOR_SAMPLES, int) or MIN_SECTOR_SAMPLES < 1:
      raise ValueError(f"MIN_SECTOR_SAMPLES must be a positive integer, got: {MIN_SECTOR_SAMPLES}")
   if not (0 < MAX_SECTOR_WEIGHT <= 1):
      raise ValueError(f"MAX_SECTOR_WEIGHT must be between 0 and 1, got: {MAX_SECTOR_WEIGHT}")
   if not (0 < MAX_SINGLE_POSITION <= 1):
      raise ValueError(f"MAX_SINGLE_POSITION must be between 0 and 1, got: {MAX_SINGLE_POSITION}")

   # Validate outlier detection
   if IQR_MULTIPLIER <= 0:
      raise ValueError(f"IQR_MULTIPLIER must be positive, got: {IQR_MULTIPLIER}")
   if ZSCORE_THRESHOLD <= 0:
      raise ValueError(f"ZSCORE_THRESHOLD must be positive, got: {ZSCORE_THRESHOLD}")
   if not (0 <= WINSORIZE_LOWER < 0.5):
      raise ValueError(f"WINSORIZE_LOWER must be between 0 and 0.5, got: {WINSORIZE_LOWER}")
   if not (0.5 < WINSORIZE_UPPER <= 1):
      raise ValueError(f"WINSORIZE_UPPER must be between 0.5 and 1, got: {WINSORIZE_UPPER}")

   # Validate confidence configuration
   if not (0 < CONFIDENCE_LEVEL < 1):
      raise ValueError(f"CONFIDENCE_LEVEL must be between 0 and 1, got: {CONFIDENCE_LEVEL}")
   if not abs(ALPHA - (1 - CONFIDENCE_LEVEL)) < 0.01:
      raise ValueError(f"ALPHA must equal (1 - CONFIDENCE_LEVEL), got: {ALPHA}")

   print("✓ All configuration constants validated successfully")


# Run validation immediately after defining constants
validate_configuration()
```

**Examples**:

✅ **Correct Usage** (Single Source of Truth):

```python
# Configuration cell (execute once)
TEST_SIZE = 0.2
TRAIN_SIZE = 1 - TEST_SIZE

# Later cells reference constants
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
```

❌ **Violation** (Magic numbers):

```python
# Bad: Magic number duplicated across cells
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)  # ❌ Magic numbers
```

❌ **Violation** (Duplicate definitions):

```python
# Cell 1
TEST_SIZE = 0.2

# Cell 10 (much later)
TEST_SIZE = 0.25  # ❌ Redefining constant creates confusion
```

### 8.2 DataFrame Stage Naming Convention

**Policy**: Use descriptive stage-based naming instead of in-place mutations. Each transformation stage produces a new
DataFrame with a descriptive suffix indicating the preprocessing stage.

**Required Stage Names** (4-stage pipeline):

The pipeline is organized into four high-level stages that align with the unified ETL pipeline
(`finance_ml.ml_workflow.preprocessing.etl`):

1. **`all_stocks_preprocessed`** — ETL pipeline output: extraction, normalization, validation, sanitization,
   imputation (6-step), and optional scaling. This consolidates the low-level preprocessing steps into a single
   ETL call using `run_etl_pipeline()` or `etl_with_imputation()`.

2. **`all_stocks_features`** — DataFrame enhanced with engineered features (Phase 9.3 feature categories:
   momentum, valuation, profitability, quality/risk, cash flow, growth).

3. **`all_stocks_classification`** — DataFrame enhanced with classification model outputs (event probabilities,
   predicted classes) for use as meta-features in regression.

4. **`all_stocks_enhanced`** — Final Phase 9.5 regression-ready dataset with all transformations including
   classification meta-features.

**ETL Pipeline Internal Stages** (handled automatically by `run_etl_pipeline()`):

The ETL pipeline internally handles these preprocessing steps in sequence:

- Stage 1: Column normalization (lowercase, underscores)
- Stage 2: Schema validation
- Stage 3: Drop invalid rows (missing ticker, sector, last_price)
- Stage 4: Data sanitization (inf, nan, extremes)
- Stage 5: Imputation (6-step: zero, sector-KNN, price, median, categorical, datetime)
- Stage 6: Log transforms (optional)
- Stage 7: Feature scaling (optional, excludes price columns)

**Benefits**:

- **Simplified Pipeline**: ETL handles low-level preprocessing; notebook focuses on ML stages
- **Debugging**: Inspect intermediate stages without re-running expensive operations
- **Rollback**: Revert to earlier stage if downstream transformation fails
- **Metrics Tracking**: ETL returns `ETLMetrics` with imputation/scaling statistics
- **Self-documenting**: Stage names clearly indicate transformation history

**Implementation Pattern**:

```python
from finance_ml.ml_workflow.preprocessing.etl import run_etl_pipeline, ETLConfig

# Stage 1: ETL Pipeline (preprocessing)
config = ETLConfig(
    apply_imputation=True,
    imputation_strategy='6step',
    apply_scaling=False,  # Scale later if needed
    validate_quality=True,
)
all_stocks_preprocessed, etl_metrics = run_etl_pipeline(
    source='csv',  # or 'db', 'all_stocks'
    data_dir='data/',
    config=config,
    return_metrics=True,
)
print(f"✓ Stage 1 (preprocessed): {all_stocks_preprocessed.shape}")
print(f"  Imputation: {etl_metrics.missing_values_before_imputation} → "
      f"{etl_metrics.missing_values_after_imputation} missing values")

# Validation checkpoint
assert not all_stocks_preprocessed.empty, "Preprocessed data must not be empty"
assert etl_metrics.imputation_completeness, "Imputation must be complete"

# Stage 2: Feature Engineering
from finance_ml.ml_workflow.features.advanced import build_comprehensive_features

all_stocks_features = build_comprehensive_features(
    all_stocks_preprocessed,
    phase93_categories=['momentum', 'valuation', 'profitability', 'quality_risk', 'cash_flow', 'growth']
)
print(f"✓ Stage 2 (features): {all_stocks_features.shape}")

# Validation checkpoint
new_features = all_stocks_features.shape[1] - all_stocks_preprocessed.shape[1]
assert new_features > 0, "Feature engineering must add new columns"
print(f"  New features added: {new_features}")

# Stage 3: Classification (event prediction)
from finance_ml.ml_workflow.models.classification import train_event_classifier

X_class = all_stocks_features[feature_cols]
y_class = all_stocks_features['event_label']  # Derived event labels

classifier_result = train_event_classifier(X_class, y_class, model_type='lightgbm')
class_probs = classifier_result['model'].predict_proba(X_class)

all_stocks_classification = all_stocks_features.copy()
all_stocks_classification['class_prob_positive'] = class_probs[:, 1]
all_stocks_classification['class_prob_negative'] = class_probs[:, 2] if class_probs.shape[1] > 2 else 0
print(f"✓ Stage 3 (classification): {all_stocks_classification.shape}")

# Validation checkpoint
assert 'class_prob_positive' in all_stocks_classification.columns, "Classification probabilities required"

# Stage 4: Final Enhanced Dataset (regression-ready)
all_stocks_enhanced = all_stocks_classification.copy()
# Add any final composite features or interactions
print(f"✓ Stage 4 (enhanced): {all_stocks_enhanced.shape}")

# Final validation checkpoint
assert all_stocks_enhanced.shape[0] == all_stocks_preprocessed.shape[0], "Row count must remain constant"
print(f"✓ Pipeline complete: {all_stocks_enhanced.shape[0]} stocks, {all_stocks_enhanced.shape[1]} features")
```

**Examples**:

✅ **Correct Usage** (Stage-based naming with ETL pipeline):

```python
from finance_ml.ml_workflow.preprocessing.etl import run_etl_pipeline

# Stage 1: Preprocessing via ETL
all_stocks_preprocessed, metrics = run_etl_pipeline(source='csv', data_dir='data/', return_metrics=True)

# Stage 2: Feature engineering
all_stocks_features = build_comprehensive_features(all_stocks_preprocessed)

# Stage 3: Classification meta-features
all_stocks_classification = add_classification_features(all_stocks_features)

# Stage 4: Final enhanced dataset
all_stocks_enhanced = all_stocks_classification.copy()
# Each stage preserves history and enables rollback
```

❌ **Violation** (In-place mutation):

```python
all_stocks = run_etl_pipeline(source='csv', data_dir='data/')
all_stocks = build_comprehensive_features(all_stocks)  # ❌ Overwrites original, no rollback
all_stocks = add_classification_features(all_stocks)  # ❌ Cannot inspect intermediate stages
```

❌ **Violation** (Unclear naming):

```python
df1 = run_etl_pipeline(source='csv', data_dir='data/')
df2 = build_comprehensive_features(df1)  # ❌ Generic names don't indicate transformation
df3 = add_classification_features(df2)  # ❌ What does df3 represent?
```

### 8.3 Magic Numbers Policy

**Policy**: All numeric literals with semantic meaning must be named constants. Magic numbers make code harder to
maintain, test, and understand.

**Prohibited Magic Numbers**:

- `random_state=42` → Use `RANDOM_SEED`
- `test_size=0.2` → Use `TEST_SIZE`
- `0.8` for train size → Use `TRAIN_SIZE`
- `max_sector_weight=0.25` → Use `MAX_SECTOR_WEIGHT`
- `quantiles=[0.1, 0.5, 0.9]` → Use `QUANTILES`
- `lower=0.01, upper=0.99` → Use `WINSORIZE_LOWER`, `WINSORIZE_UPPER`
- `threshold=1.5` for IQR → Use `IQR_MULTIPLIER`
- `n_splits=5` → Use `CV_FOLDS`

**Allowed Inline Literals**:

- **Universal constants**: `0`, `1`, `100` (for percentage calculations)
- **Highly localized single-use values**: Loop indices, array dimensions in small functions
- **Algorithm parameters with clear context**: `np.clip(x, 0, 1)` when enforcing probability bounds

**Special Case — Correlation Matrix Construction**:

When constructing correlation matrices, algorithm-specific parameters may remain inline if clearly documented:

```python
# ✅ Allowed: Algorithm parameters with clear context
corr_matrix = df.corr(method='pearson', min_periods=10)  # min_periods is sklearn default
```

**Examples**:

✅ **Correct Usage** (Named constants):

```python
# Configuration cell
TEST_SIZE = 0.2
RANDOM_SEED = 42
CV_FOLDS = 5

# Usage
train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
gkf = GroupKFold(n_splits=CV_FOLDS)
```

❌ **Violation** (Magic numbers):

```python
train, test = train_test_split(df, test_size=0.2, random_state=42)  # ❌ What do these mean?
gkf = GroupKFold(n_splits=5)  # ❌ Why 5? Can it change?
```

✅ **Correct Usage** (Portfolio optimization):

```python
# Configuration cell
MAX_SECTOR_WEIGHT = 0.25
MAX_SINGLE_POSITION = 0.10
MIN_WEIGHT = 0.01

# Usage
constraints = [
   {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # ✅ Universal constant (sum to 1)
   {'type': 'ineq', 'fun': lambda w: MAX_SECTOR_WEIGHT - sector_exposure(w)},
   {'type': 'ineq', 'fun': lambda w: w - MIN_WEIGHT}  # No position below 1%
   ]
```

❌ **Violation** (Portfolio optimization with magic numbers):

```python
constraints = [
   {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # ✅ OK
   {'type': 'ineq', 'fun': lambda w: 0.25 - sector_exposure(w)},  # ❌ What is 0.25?
   {'type': 'ineq', 'fun': lambda w: w - 0.01}  # ❌ What is 0.01?
   ]
```

✅ **Correct Usage** (Percentage calculations):

```python
# Universal constants for percentage conversion
pct_error = (predicted - actual) / actual * 100  # ✅ 100 is universal
success_rate = correct / total * 100  # ✅ 100 is universal
```

**Validation**:

All notebook cells are validated against these policies via `tests/test_notebook_tdd_compliance.py`. The test suite
includes:

- `test_configuration_constants_defined()` — Verifies all required constants exist
- `test_no_magic_numbers()` — Detects inline magic numbers in code cells
- `test_dataframe_stage_naming()` — Validates stage-based naming convention
- `test_validation_checkpoints()` — Ensures assertions exist after each stage

**References**:

- Implementation: `ml_finance_model_main.ipynb` (all 24 compliance tests passing)
- Test suite: `tests/test_notebook_tdd_compliance.py`
- Configuration guide: `docs/code_guidelines.md` Section 2

---

### 8.5 Preprocessing Stage Naming and Semantic Column Classification

This section establishes formal standards for preprocessing stages 4-8 (winsorization, imputation, scaling, feature
engineering) to ensure business-critical columns (prices, valuations) are handled correctly. All policies are validated
via `tests/test_column_semantics.py`, `tests/test_selective_winsorization.py`, `tests/test_log_transforms.py`, and
`tests/test_selective_scaling.py` (36 tests passing).

#### 8.5.1 Column Semantic Classification

**Policy**: All preprocessing functions must respect semantic column types defined in
`finance_ml/ml_workflow/preprocessing/column_semantics.py`.

**Five Semantic Categories**:

1. **Price Columns** (`PRICE_COLUMNS`): Never transform, clip, or scale
    - `last_price`, `price_target`, `price_target_median`, `price_target_ytd_ago`, `price_target_12m_ago`

2. **Market Value Columns** (`MARKET_VALUE_COLUMNS`): Apply log-transforms instead of winsorization
    - `market_cap`, `ev`, `total_assets`, `revenue`, `total_debt`, `ebitda`, `operating_income`, `net_income`,
      `cash_and_equivalents`

3. **Ratio Columns** (`RATIO_COLUMNS`): Pre-normalized, exclude from winsorization
    - `p_e`, `p_b`, `p_s`, `ev_ebitda`, `ev_sales`, `roe`, `roa`, `roic`, `debt_equity`, `current_ratio`, `quick_ratio`

4. **Percentage Columns** (`PERCENTAGE_COLUMNS`): Bounded [0, 100], exclude from winsorization
    - `gross_margin`, `operating_margin`, `net_margin`, `ebitda_margin`, `revenue_growth_yoy`, `volatility_20d`,
      `volatility_60d`

5. **Count Columns** (`COUNT_COLUMNS`): Discrete integers, inappropriate for continuous scaling
    - `num_analysts`, `num_employees`, `num_strong_buy_ratings`, `num_buy_ratings`, `num_hold_ratings`

**Helper Functions**:

```python
from finance_ml.ml_workflow.preprocessing.column_semantics import (
    classify_columns,           # Classify all columns by semantic type
    get_winsorizable_columns,   # Get columns safe for winsorization
    get_log_transform_columns,  # Get columns requiring log-transform
    get_scalable_columns,       # Get columns safe for scaling
)

# Example: Semantic-aware preprocessing
winsorizable = get_winsorizable_columns(df.columns.tolist())
df_winsorized = winsorize_by_sector(df, columns=winsorizable, exclude_price_columns=True)
```

**Rationale**: The core business metric `(Predicted_Target - Last_Price) / Last_Price` requires original price scale.
Winsorizing or scaling price columns destroys interpretability and invalidates valuation analysis.

#### 8.5.2 Price Column Preservation Policy

**Policy**: All 21 price columns (current prices, targets, historical prices, 52w bounds, EMAs) must **NEVER** be:

1. **Winsorized** — Capping extreme prices corrupts valid high-growth stock valuations
2. **Scaled** (StandardScaler, RobustScaler, MinMaxScaler) — Destroys dollar interpretability
3. **Log-transformed** in place — Only create new columns (`log_market_cap`) while preserving originals
4. **Clipped or capped** — Valid extreme values must be preserved

**Complete PRICE_COLUMNS List (21 columns)**:

- Current: `last_price`, `price_target`, `price_target_median`, `price_target_ytd_ago`, `price_target_low`,
  `price_target_high` (6 columns)
- Historical: `price_5d_ago`, `price_1w_ago`, `price_1m_ago`, `price_3m_ago`, `price_6m_ago`, `price_1y_ago`,
  `price_3y_ago`, `price_5y_ago`, `price_qtd_ago` (9 columns)
- 52w Bounds: `52w_high_adj`, `52w_low_adj` (2 columns)
- EMAs: `ema_20d`, `ema_50d`, `ema_100d`, `ema_250d` (4 columns)

**Enforcement**:

All preprocessing functions default to `exclude_price_columns=True`:

```python
# ✅ CORRECT: Price columns excluded by default
df_winsorized = winsorize_by_sector(
    df,
    columns=numeric_cols,
    exclude_price_columns=True,  # Default: True
    exclude_ratio_columns=True    # Default: True
)

df_scaled = scale_features(
    df,
    scaler_type='robust',
    exclude_price_columns=True    # Default: True
)

# ❌ INCORRECT: Treating all numeric columns uniformly
df_corrupted = winsorize_by_sector(df)  # Corrupts price columns if not excluded!
df_corrupted = scale_features(df)        # Destroys price interpretability!
```

**Validation**:

```python
# Verify price columns unchanged after preprocessing
from finance_ml.ml_workflow.preprocessing.column_semantics import PRICE_COLUMNS

price_cols = [col for col in PRICE_COLUMNS if col in df_processed.columns]
for col in price_cols:
    if col in df_original.columns:
        assert df_processed[col].equals(df_original[col]), f"{col} must not be modified"

# Expected: 21 price columns preserved
print(f"Verified {len(price_cols)} price columns unchanged")
```

**Examples of Protected Use Cases**:

✅ **Historical Prices** (momentum calculations):

```python
# Momentum feature requires original dollar scale
df['price_momentum_1m'] = (df['last_price'] - df['price_1m_ago']) / df['price_1m_ago']
# Winsorizing price_1m_ago would corrupt this calculation
```

✅ **52-Week Bounds** (relative positioning):

```python
# Requires original price scale for meaningful positioning
df['price_vs_52w_range'] = (df['last_price'] - df['52w_low_adj']) / (df['52w_high_adj'] - df['52w_low_adj'])
# Scaling 52w_high_adj/52w_low_adj destroys cross-stock comparability
```

✅ **EMAs** (technical analysis):

```python
# EMA deviation calculation requires same dollar scale
df['ema_50d_deviation'] = (df['last_price'] - df['ema_50d']) / df['ema_50d']
# Transforming ema_50d invalidates technical signals
```

**Rationale**: The core business objective (stock valuation and mispricing detection) depends on comparing predicted
targets to actual prices in original dollar units. This extends to:

- **Momentum features**: Historical price comparisons require consistent scale
- **Technical indicators**: 52w bounds and EMAs are price-derived and must maintain dollar interpretability
- **Cross-stock analysis**: Relative price metrics depend on absolute price preservation

Any transformation of these columns invalidates valuation, momentum, and technical analysis.

#### 8.5.3 Alternative Transformations for Skewed Data

**Policy**: Use log-transforms instead of winsorization for highly skewed market value columns (market_cap, revenue,
total_assets) to preserve information about extreme but valid values (e.g., mega-cap stocks).

**Log-Transform Methods**:

```python
from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms

# Method 1: log1p for non-negative values (handles zeros)
df = apply_log_transforms(df, method='log1p')
# Creates: log_market_cap, log_revenue, log_total_assets, etc.
# Formula: log(1 + x)

# Method 2: signed_log for values that can be negative (debt, income)
df = apply_log_transforms(df, method='signed_log')
# Creates: log_market_cap, log_revenue, log_net_income, etc.
# Formula: sign(x) * log(1 + |x|)
```

**Benefits**:

1. **Reduces skewness** by ≥50% while preserving rank order
2. **Preserves information** about extreme values (mega-cap stocks, high-revenue companies)
3. **Handles zeros and negatives** appropriately
4. **Reversible** via `inverse_log_transform()` for interpretability

**ETL Integration Note**:

> **As of v0.8.3**: Log transforms and winsorization are now handled internally by the unified ETL pipeline
> (`run_etl_pipeline()`) via the `apply_log_transforms` and `apply_scaling` configuration options.
> The examples below are provided for understanding the internal operations and for advanced users who
> need fine-grained control outside the ETL pipeline.

**Manual Pipeline** (for advanced use cases):

```python
# Note: For most use cases, use run_etl_pipeline() instead (see Section 8.2)
# Step 1: Apply log-transforms to skewed market value columns
from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms
from finance_ml.ml_workflow.preprocessing.column_semantics import get_winsorizable_columns

all_stocks_log_transformed = apply_log_transforms(
    all_stocks_preprocessed,  # Output from ETL pipeline
    method='signed_log'  # Handles negative values (debt, income)
)

# Step 2: Selective winsorization (excludes prices, ratios, percentages)
winsorizable_cols = get_winsorizable_columns(all_stocks_log_transformed.columns.tolist())
all_stocks_winsorized = winsorize_by_sector(
    all_stocks_log_transformed,
    columns=winsorizable_cols,
    lower_percentile=WINSORIZE_LOWER,
    upper_percentile=WINSORIZE_UPPER,
    by_sector=True,
    exclude_price_columns=True,
    exclude_ratio_columns=True
)

print(f"✓ Log-transformed {len([c for c in all_stocks_winsorized.columns if c.startswith('log_')])} columns")
print(f"✓ Winsorized {len(winsorizable_cols)} columns (excluded price/ratio columns)")
```

**Example: Comparing Approaches**:

```python
# ❌ BAD: Winsorization loses information
df['market_cap_winsorized'] = df['market_cap'].clip(lower=p1, upper=p99)
# Result: Apple ($3T) and Nvidia ($2T) capped at p99 (~$500B), losing $2.5T information

# ✅ GOOD: Log-transform preserves information
df['log_market_cap'] = np.sign(df['market_cap']) * np.log1p(np.abs(df['market_cap']))
# Result: Apple (28.7) and Nvidia (28.3) maintain relative ordering and magnitude information
```

**Validation**:

Test coverage for log-transforms (`tests/test_log_transforms.py`, 9 tests):

- Skewness reduction (≥50% improvement)
- Zero and negative value handling
- Null preservation
- Reversibility via `inverse_log_transform()`

**References**:

- Implementation: `finance_ml/ml_workflow/preprocessing/transforms.py` (214 lines)
- Column semantics: `finance_ml/ml_workflow/preprocessing/column_semantics.py` (324 lines)
- Updated functions: `outliers.py` (winsorize_by_sector), `scaling.py` (scale_features)
- Test suite: 36 tests passing (column_semantics, selective_winsorization, log_transforms, selective_scaling)
- Improvement plan: `docs/improvement_plan/preprocessing_stages_4-8_improvement_plan.md`

---

## 9. Column Schema and DataFrame Conventions

### 9.1 Canonical Column Names

**Target Columns:**

- Primary: `price_target` (from "Price Target")
- Fallback: `price_target_median` (from "Price Target - Median")
- Basis: `last_price` (from "Last Price")

**Identifier Columns:**

- `ticker` (from "Ticker")
- `isin` (from "ISIN")
- `sector` (from "Sector")
- `region` (from "Region")

**Feature Columns:** Use normalized names (lowercase, underscores) as defined in `COLUMN_SCHEMA`.

### 9.2 DataFrame Conventions

**Index:**

- Use `ticker` as index for stock-level DataFrames
- Use `(ticker, region)` as multi-index when combining regions
- Reset index before saving to CSV: `df.reset_index().to_csv(...)`

**Column Order:**

- Identifiers first: ticker, isin, sector, region, country
- Target columns: last_price, price_target, price_target_median
- Features: alphabetical or grouped by category
- Predictions: y_true, y_pred, y_pred_calibrated, pred_p10, pred_p50, pred_p90

**Missing Values:**

- Represented as `np.nan` or `pd.NA`
- Never use 0, -1, or empty string for missing numeric values
- Apply imputation before modeling

---

## 10. Data Split and Leakage Policy

### 10.1 Split Strategies (in priority order)

1. **Time-Series Split** (preferred if temporal data available):
   ```python
   df_sorted = df.sort_values('last_updated')
   split_idx = int(len(df) * 0.8)
   train_df = df_sorted.iloc[:split_idx]
   test_df = df_sorted.iloc[split_idx:]
   ```

2. **Grouped Split** (prevent leakage across tickers):
   ```python
   from sklearn.model_selection import GroupShuffleSplit
   gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
   train_idx, test_idx = next(gss.split(X, y, groups=df['ticker']))
   ```

3. **Stratified Split** (maintain sector/region balance):
   ```python
   from sklearn.model_selection import train_test_split
   train, test = train_test_split(df, test_size=0.2, stratify=df['sector'], random_state=RANDOM_SEED)
   ```

### 10.2 Cross-Validation Strategy

**Grouped CV** (prevent same ticker in train and validation):

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=df['ticker']):
# Training fold...
```

**Stratified CV** (when grouped not feasible):

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
```

### 10.3 Leakage Prevention Rules

- **No future information**: Features must only use data available at prediction time
- **No target leakage**: Features cannot be derived from target variable
- **No data from test set**: Scalers, encoders, imputers fit only on train set
- **No group mixing**: Same ticker should not appear in both train and validation in CV

---

## 11. Standardized Predictions Schema

### 11.1 Required Columns

All prediction outputs must include these columns:

```python
REQUIRED_COLUMNS = [
   'ticker', 'isin', 'sector', 'region',
   'last_price', 'y_true', 'y_pred', 'y_pred_calibrated',
   'pred_p10', 'pred_p50', 'pred_p90',
   'interval_width', 'abs_error', 'pct_error',
   'model_version', 'snapshot_date'
   ]
```

### 11.2 Column Definitions

- `y_true`: Actual target value (price_target or last_price)
- `y_pred`: Raw model prediction
- `y_pred_calibrated`: Sector-bias-corrected prediction
- `pred_p10`, `pred_p50`, `pred_p90`: Quantile predictions (10th, 50th, 90th percentiles)
- `interval_width`: pred_p90 - pred_p10
- `abs_error`: abs(y_pred - y_true)
- `pct_error`: 100 * (y_pred - y_true) / y_true
- `model_version`: e.g., "v9_9"
- `snapshot_date`: Date of prediction run

### 11.3 Invariants

- **Monotonicity**: `pred_p10 ≤ pred_p50 ≤ pred_p90`
- **Non-negativity**: All price predictions ≥ 0
- **Interval coverage**: Target 80% of actual values within [pred_p10, pred_p90]

### 11.4 Output Files

- `outputs/regression/regression_predictions_detailed.csv` — Full predictions schema
- `outputs/regression/quantile_predictions.csv` — Quantile-specific outputs
- `outputs/regression/regression_metrics_by_sector.csv` — Sector-level metrics

---

## 12. Sector Metrics and Calibration

### 12.1 Sector-Level Metrics Calculation

```python
sector_metrics = df.groupby('sector').apply(lambda g: pd.Series({
   'mae': (g['y_pred'] - g['y_true']).abs().mean(),
   'rmse': np.sqrt(((g['y_pred'] - g['y_true']) ** 2).mean()),
   'r2': r2_score(g['y_true'], g['y_pred']),
   'mape': 100 * (g['y_pred'] - g['y_true']).abs().div(g['y_true']).mean(),
   'bias': (g['y_pred'] - g['y_true']).mean(),
   'count': len(g)
   }))
```

### 12.2 Sector Bias Calibration

**Additive Correction:**

```python
sector_bias = val_df.groupby('sector').apply(
        lambda x: (x['y_pred'] - x['y_true']).mean()
        )
df['y_pred_calibrated'] = df.apply(
        lambda row: row['y_pred'] - sector_bias.get(row['sector'], 0),
        axis=1
        )
```

**Isotonic Regression Calibration:**

```python
from sklearn.isotonic import IsotonicRegression

isotonic_models = {}
for sector in sectors:
   sector_data = val_df[val_df['sector'] == sector]
   iso = IsotonicRegression(out_of_bounds='clip')
   iso.fit(sector_data['y_pred'], sector_data['y_true'])
   isotonic_models[sector] = iso

df['y_pred_calibrated'] = df.apply(
        lambda row: isotonic_models[row['sector']].transform([row['y_pred']])[0],
        axis=1
        )
```

### 12.3 Metrics Persistence

Sector metrics must be persisted to `outputs/regression/regression_metrics_by_sector.csv` with columns:

- sector, mae, rmse, r2, mape, bias, count, model_version, timestamp

---

## 13. Outlier Safety Rails Policy

### 13.1 Winsorization

Apply winsorization to extreme values before modeling:

```python
from scipy.stats.mstats import winsorize

df['market_cap_winsorized'] = winsorize(df['market_cap'], limits=[0.01, 0.01])
# Clips values at 1st and 99th percentiles
```

### 13.2 Robust Loss Functions

Use robust loss functions in gradient boosting models:

```python
# XGBoost with Huber loss
xgb_model = XGBRegressor(objective='reg:pseudohuber', huber_slope=1.0)

# LightGBM with MAE loss
lgb_model = LGBMRegressor(objective='mae')
```

### 13.3 Non-Negativity Constraints

Enforce non-negative predictions for prices:

```python
df['y_pred'] = df['y_pred'].clip(lower=0)
df['pred_p10'] = df['pred_p10'].clip(lower=0)
df['pred_p50'] = df['pred_p50'].clip(lower=0)
df['pred_p90'] = df['pred_p90'].clip(lower=0)
```

### 13.4 Outlier Detection Thresholds

- **IQR Method**: Values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- **Z-Score Method**: |z-score| > 3
- **Domain-specific**: Market cap > $1T, P/E > 100, volatility > 100%

---

## 14. Uncertainty and Prediction Intervals

### 14.1 Quantile Regression

Train quantile regressors for uncertainty bounds:

```python
from sklearn.ensemble import GradientBoostingRegressor

quantile_models = {}
for q in [0.1, 0.5, 0.9]:
   model = GradientBoostingRegressor(loss='quantile', alpha=q)
   model.fit(X_train, y_train)
   quantile_models[q] = model

df['pred_p10'] = quantile_models[0.1].predict(X)
df['pred_p50'] = quantile_models[0.5].predict(X)
df['pred_p90'] = quantile_models[0.9].predict(X)
```

### 14.2 Conformal Prediction

Apply conformal calibration for coverage guarantees:

```python
# Calculate calibration residuals on validation set
val_residuals = np.abs(val_df['y_true'] - val_df['pred_p50'])
q = np.quantile(val_residuals, 0.8)  # 80% coverage

# Adjust intervals
df['pred_p10_calibrated'] = df['pred_p50'] - q
df['pred_p90_calibrated'] = df['pred_p50'] + q
```

### 14.3 Coverage Diagnostics

```python
coverage = ((df['y_true'] >= df['pred_p10']) &
            (df['y_true'] <= df['pred_p90'])).mean()
print(f"Interval coverage: {coverage:.1%}")  # Target: 75-85%
```

### 14.4 Sector-Level Uncertainty

Calculate uncertainty metrics by sector:

```python
sector_uncertainty = df.groupby('sector').agg({
   'interval_width': 'mean',
   'abs_error': 'mean',
   'pct_error': lambda x: x.abs().mean()
   })
```

---

## 15. Jupyter Notebook Guidelines

### 15.1 Notebook Structure

**Required Sections:**

1. Configuration and Setup
2. Data Loading and Preprocessing (Phase 9.1)
3. Exploratory Data Analysis (Phase 9.2)
4. Feature Engineering (Phase 9.3)
5. Classification (Phase 9.4)
6. Regression (Phase 9.5)
7. Evaluation (Phase 9.6)
8. Analytics (Phase 9.7)
9. Reporting (Phase 9.8)
10. Portfolio Optimization (optional)

### 15.2 Cell Organization

- **One logical unit per cell**: Don't mix data loading and feature engineering
- **Markdown documentation**: Each section starts with markdown cell explaining purpose
- **Output management**: Clear large outputs before committing: `Cell → All Output → Clear`
- **Error handling**: Use try-except for data loading and model training

### 15.3 Configuration Cell (First Cell)

```python
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuration constants (Section 2)
TARGET_COL = 'price_target'
TARGET_COL_FALLBACK = 'last_price'
TEST_SIZE = 0.2
CV_FOLDS = 5
QUANTILES = [0.1, 0.5, 0.9]
MIN_SECTOR_SAMPLES = 20
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v9_9')

np.random.seed(RANDOM_SEED)

# Output directories
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
for subdir in ['eda', 'features', 'classification', 'regression', 'analytics', 'plots']:
   (OUTPUT_DIR / subdir).mkdir(exist_ok=True)

# Validate configuration
validate_configuration()  # From Section 2.3
```

### 15.4 Import Organization

Follow Phase 9.1-9.8 structure (Section 4.3):

```python
# Phase 9.1: Preprocessing
from finance_ml.ml_workflow.preprocessing import imputation, outliers, scaling

# Phase 9.3: Features
from finance_ml.ml_workflow.features import advanced, selection

# Phase 9.5: Regression
from finance_ml.ml_workflow.regression import models, quantile

# Phase 9.7: Analytics
from finance_ml.ml_workflow.analytics import mispricing, portfolio
```

### 15.5 Version Tracking

Add version cell at end of notebook:

```python
print(f"Notebook execution completed")
print(f"Model Version: {MODEL_VERSION}")
print(f"Random Seed: {RANDOM_SEED}")
print(f"Timestamp: {pd.Timestamp.now()}")
```

---

## 16. Model Optimization and Performance

### 16.1 Hyperparameter Tuning

Use Optuna for efficient hyperparameter search:

```python
import optuna


def objective(trial):
   params = {
      'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
      'max_depth': trial.suggest_int('max_depth', 3, 10),
      'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
      'subsample': trial.suggest_float('subsample', 0.6, 1.0),
      }
   model = XGBRegressor(**params, random_state=RANDOM_SEED)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_val)
   return mean_absolute_error(y_val, y_pred)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
```

### 16.2 Feature Selection

Use feature importance and recursive elimination:

```python
from sklearn.feature_selection import RFECV

rfe = RFECV(estimator=XGBRegressor(), cv=5, scoring='neg_mean_absolute_error')
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]
```

### 16.3 Model Stacking

Combine multiple models for better predictions:

```python
from sklearn.ensemble import StackingRegressor

base_models = [
   ('xgb', XGBRegressor(n_estimators=500)),
   ('lgb', LGBMRegressor(n_estimators=500)),
   ('cat', CatBoostRegressor(n_estimators=500, verbose=0))
   ]

stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        cv=5
        )
stacking_model.fit(X_train, y_train)
```

### 16.4 Performance Thresholds

**Regression Performance Targets:**

- **Excellent**: MAE < 20%, R² > 0.7
- **Good**: MAE 20-40%, R² 0.5-0.7
- **Acceptable**: MAE 40-60%, R² 0.3-0.5
- **Needs Improvement**: MAE > 60%, R² < 0.3

**Sector-Specific Thresholds:**

- **Technology, Healthcare**: MAE < 40%
- **Financials, Industrials**: MAE < 50%
- **Real Estate, Energy**: MAE < 60% (higher volatility sectors)

---

**Document Version:** 1.7  
**Last Updated:** 2025-11-26  
**Package Version:** 0.8.3  
**Model Version:** v9_9  
**Synchronized with:** README.md v0.8.3, CHANGELOG.md v0.8.3, pyproject.toml v0.8.3

---

## 17. Styles Guides for Visual Elements

### 17.1 Plot Formatting and Labeling

Standardize all visualizations (Plotly, Matplotlib, Seaborn) to ensure consistency across dashboards and reports.

**General Principles:**

- **Theme:** Use Dark Mode compatible themes (`template="plotly_dark"` for Plotly).
- **Font:** Use a standard sans-serif font (e.g., Arial, Roboto) for legibility.
- **Titles:** Clear, descriptive titles with consistent sizing (H3 equivalent).
- **Labels:** Always label axes with units (e.g., "Price ($)", "Market Cap (Billion $)", "Return (%)").
- **Tooltips:** Include detailed hover information (Ticker, Name, Sector, Metric Value).

**Color Palette:**

- **Primary:** `#375a7f` (Blue/Primary)
- **Success:** `#00bc8c` (Green/Positive)
- **Warning:** `#f39c12` (Orange/Warning)
- **Danger:** `#e74c3c` (Red/Negative/Error)
- **Info:** `#3498db` (Light Blue/Info)
- **Neutral:** `#adb5bd` (Gray)

**Heatmaps and Conditional Formatting:**

- Use Diverging color scales for metrics centered around zero (e.g., `RdYlGn` for correlation or errors).
- Use Sequential color scales for magnitude (e.g., `Viridis` or `Blues`).
- Always include value annotations (`text_auto=True` or formatted text) for readability.
- Ensure sufficient contrast between text and background.

### 17.2 Dashboard Layout and Structure

**Dashboards (Streamlit & Dash):**

- **Header:** Clear application title and status indicators.
- **Navigation:** Logical tab-based structure grouped by business function (Overview, Analysis, Governance, Portfolio).
- **Filters:**
    - Use a dedicated sidebar or top filter bar.
    - Use distinct styles for active vs. inactive filters.
    - Enable multi-select for categorical fields (Sector, Region).
    - Implement "Dark Mode" styling for dropdowns and inputs (`custom.css` for Dash).
- **KPI Cards:** Use summary cards at the top for high-level metrics.
- **Responsiveness:** Ensure plots resize dynamically (`width='stretch'` in Streamlit, Flexbox in Dash).

### 17.3 Table Structure

- **Headers:** Bold, sentence case.
- **Numbers:**
    - Currency: `$1,234.56`
    - Percentages: `12.34%`
    - Decimals: limit to 2-4 decimal places.
- **Conditional Formatting:** Highlight outliers or significant values (e.g., top/bottom 10%).
- **Pagination:** Use pagination for tables with >20 rows.

### 17.4 Font Style

- **Family:** System sans-serif preference (Segoe UI, Roboto, Helvetica Neue, Arial).
- **Size:**
    - H1: 2rem (32px)
    - H2: 1.5rem (24px)
    - H3: 1.25rem (20px)
    - Body: 1rem (16px)
    - Caption/Label: 0.875rem (14px)
- **Color:**
    - Primary Text: `#ffffff` (on dark), `#333333` (on light)
    - Secondary Text: `#aaaaaa` (on dark), `#666666` (on light)

---

## 18. Portfolio Optimization Workflow

The platform includes a comprehensive **7-phase Portfolio Optimization Workflow** that extends the core ML pipeline
(Phase 9.1-9.8) with advanced portfolio construction, risk management, and backtesting capabilities.

### 18.1 Workflow Overview

The Portfolio Optimization workflow is implemented in `portfolio_optimization_risk_management.ipynb` (Section 10) and
integrates with the main ML pipeline through the `finance_ml.ml_workflow.analytics` module.

**7-Phase Architecture:**

| Phase | Description                | Module                            | Key Functions                                                                          |
|-------|----------------------------|-----------------------------------|----------------------------------------------------------------------------------------|
| 1     | Enhanced Stock Selection   | `stock_selection.py`              | `select_portfolio_candidates()`, `rank_stocks_multi_metric()`                          |
| 2     | ML-Based Return Prediction | `ml_returns.py`                   | `create_ml_return_features()`, `train_linear_return_predictor()`                       |
| 3     | Advanced Optimization      | `portfolio.py`                    | `optimize_black_litterman()`, `optimize_risk_parity()`, `optimize_hrp()`               |
| 4     | Risk Management            | `risk.py`                         | `calculate_expected_shortfall()`, `run_stress_tests()`, `run_monte_carlo_simulation()` |
| 5     | Backtesting Framework      | `portfolio.py`, `attribution.py`  | `run_vectorized_backtest()`, `calculate_performance_attribution()`                     |
| 6     | Interactive Dashboards     | `dashboards/portfolio_widgets.py` | `PortfolioRebalanceWidget`, `create_factor_exposure_dashboard()`                       |
| 7     | Enhanced ML & Validation   | `ml_returns.py`                   | `clip_expected_returns()`, `validate_expected_returns()`, `create_return_ensemble()`   |

### 18.2 Return Calculation Best Practices

**Critical Policy: Expected Return Bounds**

Expected returns must be bounded to prevent unrealistic optimization outputs:

```python
from finance_ml.ml_workflow.config import (
    MAX_EXPECTED_RETURN,  # 0.29 (29% annual cap)
    MIN_EXPECTED_RETURN,  # -0.50 (-50% annual floor)
    REALISTIC_RETURN_MEAN_THRESHOLD,  # 0.30 (30% mean threshold)
)

# Always clip returns before portfolio optimization
from finance_ml.ml_workflow.analytics import clip_expected_returns

expected_returns = clip_expected_returns(raw_returns)
assert expected_returns.mean() < 0.30, "Mean return exceeds realistic threshold"
```

**Rationale:**

- Long-term equity market returns average 7-10% annually
- Even high-growth stocks rarely sustain >30% annual returns
- Unbounded returns lead to inflated Sharpe ratios (e.g., 42.4 instead of <3.0)

**Return Validation:**

```python
from finance_ml.ml_workflow.analytics import validate_expected_returns

diagnostics = validate_expected_returns(expected_returns)
if not diagnostics['is_realistic']:
    for warning in diagnostics['warnings']:
        logger.warning(warning)
```

### 18.3 Price Column Integration

Use the `PRICE_COLUMNS` registry for historical return calculation:

```python
from finance_ml.ml_workflow.config import PRICE_COLUMNS

# 4 categories, 21 columns total
PRICE_COLUMNS = {
    'current': ['last_price', 'price_target', 'price_target_median', ...],
    'historical': ['price_5d_ago', 'price_1w_ago', 'price_1m_ago', 'price_3m_ago', 'price_6m_ago', 'price_1y_ago', ...],
    '52w_bounds': ['52w_high_adj', '52w_low_adj', ...],
    'emas': ['ema_20d', 'ema_50d', 'ema_100d', 'ema_250d'],
}

# Calculate historical returns from price columns
from finance_ml.ml_workflow.analytics import calculate_historical_returns

df_with_returns = calculate_historical_returns(df, current_price_col='last_price')
# Creates: return_1w, return_1m, return_3m, return_6m, return_1y
```

### 18.4 Phase 9.3 Feature Integration

Leverage 196 Phase 9.3 engineered features for enhanced return prediction:

```python
from finance_ml.ml_workflow.analytics import (
    get_phase93_return_features,
    create_ml_return_features_enhanced,
)

# Get high-relevance feature categories
categories = get_phase93_return_features()
# Returns: Momentum & Technical, Valuation Ratios, Growth Metrics,
#          Analyst Sentiment, Quality & Risk, Profitability

# Create enhanced features
enhanced_df = create_ml_return_features_enhanced(
    df,
    include_phase93=True,
    include_historical_returns=True,
)
```

### 18.5 Ensemble Model Best Practices

**Multi-Model Ensemble:**

```python
from finance_ml.ml_workflow.analytics import create_return_ensemble

# Create ensemble with multiple model types
ensemble = create_return_ensemble(
        X_train, y_train,
        models=['ridge', 'random_forest', 'gradient_boosting'],  # Add 'dnn' if TensorFlow available
        cv_folds=5,
        )

# Get predictions
predictions = ensemble.predict(X_test)
weights = ensemble.get_model_weights()  # View model contributions
```

**Dynamic Weighting:**

```python
from finance_ml.ml_workflow.analytics import create_dynamic_ensemble

# Weights based on validation performance
ensemble = create_dynamic_ensemble(
    X_train, y_train,
    models=['ridge', 'random_forest', 'gradient_boosting'],
    weighting_method='inverse_mse',  # Options: 'inverse_mse', 'softmax', 'equal'
    validation_data=(X_val, y_val),
)
```

### 18.6 Black-Litterman ML Integration

Integrate ML predictions as views in Black-Litterman optimization:

```python
from finance_ml.ml_workflow.analytics import (
    create_bl_views_from_ml,
    detect_market_regime,
    optimize_black_litterman,
)

# Create views from ML predictions
views, confidences = create_bl_views_from_ml(
    ml_predictions,
    tickers=ticker_list,
    confidence_method='prediction_interval',  # or 'uniform'
    min_confidence=0.3,
    max_confidence=0.9,
)

# Detect market regime for parameter adjustment
regime = detect_market_regime(returns, method='volatility')
# Returns: 'low_volatility', 'normal', or 'high_volatility'

# Optimize with ML-derived views
result = optimize_black_litterman(
    returns=expected_returns,
    cov_matrix=cov_matrix,
    market_weights=market_weights,
    views=views,
    view_confidences=confidences,
)
```

### 18.7 Robust Covariance Estimation

Use shrinkage methods for ill-conditioned covariance matrices:

```python
from finance_ml.ml_workflow.analytics import (
    estimate_covariance_shrinkage,
    estimate_covariance_ewm,
)

# Ledoit-Wolf shrinkage (recommended for n_assets > n_observations)
cov_shrunk = estimate_covariance_shrinkage(returns, method='ledoit_wolf')

# Exponentially weighted (for recency bias)
cov_ewm = estimate_covariance_ewm(returns, halflife=60, min_periods=30)

# Check condition number
eigenvalues = np.linalg.eigvalsh(cov_shrunk)
condition_number = eigenvalues.max() / eigenvalues.min()
assert condition_number < 1e6, "Covariance matrix ill-conditioned"
```

### 18.8 Portfolio Validation Diagnostics

**Return Prediction Diagnostics:**

```python
from finance_ml.ml_workflow.analytics import calculate_return_prediction_diagnostics

diagnostics = calculate_return_prediction_diagnostics(
    y_true, y_pred,
    include_distribution_tests=True,
    include_autocorrelation=True,
)
# Returns: mse, mae, r2, ic, residual_normality_pvalue, residual_skewness, residual_acf_lag1
```

**Portfolio Metrics Validation:**

```python
from finance_ml.ml_workflow.analytics import validate_portfolio_metrics

validation = validate_portfolio_metrics(
    weights=portfolio_weights,
    returns=historical_returns,
    risk_free_rate=0.03,
    max_sharpe_threshold=3.0,  # Flag if Sharpe > 3.0
    max_return_threshold=1.0,  # Flag if return > 100%
)

if not validation['sharpe_ratio_valid']:
    logger.warning(f"Unrealistic Sharpe: {validation['sharpe_ratio']:.2f}")
```

### 18.9 Configuration Constants Summary

All portfolio optimization constants are centralized in `finance_ml/ml_workflow/config/ml_returns_config.py`:

| Constant                            | Value | Description                                     |
|-------------------------------------|-------|-------------------------------------------------|
| `MAX_EXPECTED_RETURN`               | 0.29  | Maximum expected annual return (29%)            |
| `MIN_EXPECTED_RETURN`               | -0.50 | Minimum expected annual return (-50%)           |
| `REALISTIC_RETURN_MEAN_THRESHOLD`   | 0.30  | Threshold for flagging unrealistic mean returns |
| `PRICE_COLUMNS`                     | dict  | Registry of 21 price columns in 4 categories    |
| `PHASE93_RETURN_FEATURE_CATEGORIES` | list  | 6 feature categories for return prediction      |
| `DEFAULT_EXPECTED_RETURN`           | 0.08  | Default 8% return when data unavailable         |
| `TRAIN_SIZE`                        | 0.80  | 80% training split                              |

### 18.10 Test Coverage Requirements

Portfolio optimization tests follow the TDD convention with 90+ tests:

```
tests/
├── test_phase7_ml_returns_enhanced.py   # 26 tests - Return bounds, clipping, Phase 9.3
├── test_phase7_dnn_ensemble.py          # 30 tests - DNN, ensemble, BL, covariance
├── test_portfolio_ml_prediction.py      # 34 tests - Portfolio ML integration
└── test_ml_returns_config_compliance.py # Configuration compliance tests
```

**Test Categories:**

- **Fast Tests** (<1s): Return bounds, configuration constants, function existence
- **Medium Tests** (1-10s): Ensemble training, covariance estimation
- **Slow Tests** (>10s): DNN training (skip if TensorFlow unavailable)

<!-- v1.4.1 (2025-11-27) Regression Workflow Integration Updates -->

# Addendum v1.4.1 — Regression Workflow Integration and Safety Rails

This addendum documents the finalized regression workflow implementations and package structure aligned with Section
16.4 performance targets and recent Phase 9.1/9.3/9.5/9.8 improvements. It complements, not replaces, existing sections
in this document.

1) Standardized Predictions Schema (Phase 9.5)

- Required base columns: [y_true, y_pred, abs_error, pct_error]
- Quantiles: QUANTILES = [0.1, 0.5, 0.9] for 80% intervals
- When quantiles are present:
    - Required columns: pred_p10, pred_p50, pred_p90, interval_width
    - Monotonicity invariant: p10 ≤ p50 ≤ p90 (row-wise)
    - Non-negativity: all four columns must be ≥ 0
- Centralized helpers (finance_ml.ml_workflow.regression.io):
    - build_predictions_frame(): constructs schema-compliant frame and auto-adds interval_width
    - validate_predictions_schema(): enforces invariants; raises if violated
- Safety Rails: Negative predictions are clamped to 0.0 before error computation in build_predictions_frame.

2) Shared Data Split & Leakage Policy (Phase 9.9)

- Priority order: time-aware by snapshot_date → grouped by ticker → stratified by sector → random fallback
- Function: finance_ml.ml_workflow.validation.splits.create_train_test_split()
- Integration: finance_ml.ml_workflow.regression.dataset.prepare_regression_data() uses the shared policy when policy
  columns exist in the input dataframe.

3) Phase 9.3 Feature Engineering Review & Sector Interactions

- Utilities (finance_ml.ml_workflow.features):
    - validate_feature_coverage(X, expected=318): quick coverage check by count or names
    - prune_low_importance_features(X_train, X_test, feature_importance_df, threshold=0.01): drops features <1%
      importance while preserving classification probability features
    - save_feature_list(features, path): persist lists for auditability
- Integration point: prepare_regression_data() optionally prunes based on outputs/regression/feature_importance.csv and
  records a structured report in feature_info/meta.
- Sector-specific interaction features (default ON): one-hot(sector) × curated base
  columns [p_e_ratio, ev_ebitda_ratio, gross_margin, market_cap, beta_5y].
    - Toggle via environment variable FEATURE_SECTOR_INTERACTIONS ("1"/"0").
    - Pruning threshold configurable via FEATURE_IMPORTANCE_THRESHOLD (default 0.01).

4) Phase 9.1 Data Quality Validation

- Lightweight validators (finance_ml.ml_workflow.preprocessing):
    - check_nan_inf(df): returns NaN/Inf counts; raises if any Inf present (post-imputation guard)
    - validate_winsorization_bounds(df, lower=0.10, upper=0.90, exclude=[price columns]): reports median 10th/90th
      percentiles for numeric columns to validate winsorization bounds
- Recommended notebook hooks: call immediately after 6-step imputation and after winsorization.

5) Stacking Ensemble and Baseline Models (Phase 16.4 Optimizations)

- Stacking base learners and hyperparameters (finance_ml.ml_workflow.regression.models.train_stacking_regressor):
    - RandomForestRegressor: n_estimators=200, max_depth=15, min_samples_split=5, max_features="sqrt"
    - ExtraTreesRegressor: n_estimators=200, max_depth=15, min_samples_split=5
    - GradientBoostingRegressor: n_estimators=150, max_depth=6, learning_rate=0.05, subsample=0.8
    - XGBoost (optional): n_estimators=150, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
    - Meta-learner: Ridge(alpha=1.0)
- Model comparison defaults (compare_regressors): 2×-increased estimators and depth controls for RF/ET/GB/HGB.

6) Notebook and Script Alignment

- QUANTILES are standardized to [0.1, 0.5, 0.9]; ensure notebooks use these values.
- ml_finance_model_main.ipynb integrates:
    - prepare_regression_data() meta summary prints (feature coverage, pruned features saved under
      outputs/regression/pruned_features.txt, sector interactions count)
    - Phase 9.8 stacking governance enabled by preparing base_predictions and y_pred_meta

7) Package Architecture Exports (for stable imports)

- finance_ml.ml_workflow.features now exports:
    - validate_feature_coverage, prune_low_importance_features, save_feature_list
- finance_ml.ml_workflow.preprocessing now exports:
    - check_nan_inf, validate_winsorization_bounds

8) Environment Variables (centralized toggles)

- FEATURE_IMPORTANCE_THRESHOLD: float threshold for pruning (default 0.01)
- FEATURE_SECTOR_INTERACTIONS: enable sector interaction features (default enabled)
- DB_URL, DATA_DIR, MODEL_DIR, CACHE_DIR, MODEL_VERSION, RANDOM_SEED, N_JOBS (as documented elsewhere in this file)

9) Performance Targets (Section 16.4)

- Targets remain unchanged; these implementations are designed to help meet:
    - Overall: R² > 0.7 and MAE < 40%
    - Sector-specific MAE thresholds as already defined

Refer to docs/summaries/MODEL_OPTIMIZATION_PHASE16_4_SUMMARY.md and CHANGELOG.md for implementation details and test
coverage.
