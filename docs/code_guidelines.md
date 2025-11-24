# Finance ML Analytics Platform — Code Guidelines

**Version:** 1.5  
**Last Updated:** 2025-11-23  
**Package Version:** 0.8.3  
**Model Version:** v9_9

These guidelines codify conventions for the Finance ML Analytics Platform, covering technology stack, configuration,
architecture, function signatures, column naming, and best practices. They align with the project's 8-phase ML
workflow (Phase 9.1-9.8) and business objectives.

**Recent Updates (v1.5):**

- Updated technology stack from pyproject.toml (Python 3.12-3.14, setuptools build system)
- Clarified CLI entry points: finance-ml, finance-ml-analyze, finance-ml-validate
- Updated package architecture with 14 ml_workflow submodules
- Confirmed Schema v1.3 with 318 columns (262 original + 48 Phase 9.3 + 8 additional)
- Added Python Script/Module Review Checklist (Section 6.2) with AST-based static analysis
- Updated code examples to remove unresolved reference errors
- Aligned with CHANGELOG.md Phase 9.4-9.8 integration and recent enhancements

---

## Table of Contents

1. [Overview and Technology Stack](#1-overview-and-technology-stack)
2. [Configuration Constants](#2-configuration-constants)
3. [Main Scripts and Entry Points](#3-main-scripts-and-entry-points)
4. [Finance_ML Package Architecture](#4-finance_ml-package-architecture)
5. [Column Naming and Mapping](#5-column-naming-and-mapping)
6. [Code Review Checklist](#6-code-review-checklist)
7. [Standardized Function Signatures](#7-standardized-function-signatures)
8. [Column Schema and DataFrame Conventions](#8-column-schema-and-dataframe-conventions)
9. [Data Split and Leakage Policy](#9-data-split-and-leakage-policy)
10. [Standardized Predictions Schema](#10-standardized-predictions-schema)
11. [Sector Metrics and Calibration](#11-sector-metrics-and-calibration)
12. [Outlier Safety Rails Policy](#12-outlier-safety-rails-policy)
13. [Uncertainty and Prediction Intervals](#13-uncertainty-and-prediction-intervals)
14. [Jupyter Notebook Guidelines](#14-jupyter-notebook-guidelines)
15. [Model Optimization and Performance](#15-model-optimization-and-performance)

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

# Outlier thresholds
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99

# Confidence scoring
CONFIDENCE_LOW_THRESHOLD = 0.50
CONFIDENCE_MEDIUM_THRESHOLD = 0.75

# Random seed and versioning
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v9_9')
```

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

```
finance_ml/
├── __init__.py                    # Package-level exports
├── cli.py                         # CLI entry points (main, analyze_main, validate_main)
├── ml_workflow/                   # Main ML workflow package (Phase 9.1-9.8)
│   ├── __init__.py
│   ├── preprocessing/             # Phase 9.1: Data preprocessing
│   │   ├── imputation.py          # 6-step imputation strategy
│   │   ├── outliers.py            # Outlier detection and handling
│   │   ├── scaling.py             # Feature scaling
│   │   ├── dtypes.py              # Schema-aware datatype detection
│   │   └── pipeline.py            # Preprocessing pipeline
│   ├── eda/                       # Phase 9.2: Exploratory Data Analysis
│   │   ├── eda.py                 # Core EDA functions
│   │   ├── benchmarking.py        # Sector/region benchmarking
│   │   └── statistical_tests.py   # Statistical testing
│   ├── features/                  # Phase 9.3: Feature Engineering
│   │   ├── core.py                # Core feature functions
│   │   ├── advanced.py            # Advanced feature engineering
│   │   ├── selection.py           # Feature selection
│   │   └── api.py                 # High-level API with presets
│   ├── classification/            # Phase 9.4: Event Classification
│   │   ├── labels.py              # Label generation (13 methods)
│   │   ├── models.py              # Classification models
│   │   └── tuning.py              # Hyperparameter tuning
│   ├── regression/                # Phase 9.5: Regression Models
│   │   ├── models.py              # Regression models (XGBoost, LightGBM, CatBoost)
│   │   ├── quantile.py            # Quantile regression
│   │   ├── constraints.py         # Non-negativity constraints
│   │   └── stacking.py            # Ensemble stacking
│   ├── evaluation/                # Phase 9.6: Model Evaluation
│   │   ├── metrics.py             # Evaluation metrics
│   │   ├── uncertainty.py         # Uncertainty quantification
│   │   ├── safety_rails.py        # Outlier safety rails
│   │   └── calibration.py         # Model calibration
│   ├── analytics/                 # Phase 9.7: Analytics
│   │   ├── mispricing.py          # Mispricing score calculation
│   │   ├── stock_selection.py     # Stock ranking and selection
│   │   ├── portfolio.py           # Portfolio optimization
│   │   ├── risk.py                # Risk metrics
│   │   ├── ml_returns.py          # ML-based return prediction
│   │   ├── attribution.py         # Performance attribution
│   │   └── analyst_comparison.py  # Analyst comparison
│   ├── reporting/                 # Phase 9.8: Reporting
│   │   ├── dashboard_data.py      # Dashboard data preparation
│   │   └── quality_alerts.py      # Data quality alerts
│   ├── data/                      # Data loading and schema
│   │   ├── loaders.py             # CSV and DB loaders
│   │   └── schema.py              # Column schema registry (318 columns)
│   ├── config/                    # Configuration management
│   │   └── settings.py            # Configuration settings
│   ├── core/                      # Core utilities
│   │   └── utils.py               # Utility functions
│   └── validation/                # Data validation
│       └── validators.py          # Validation functions
└── dashboards/                    # Interactive dashboards
    ├── streamlit_app.py           # Streamlit application
    ├── dash_app.py                # Dash application
    └── portfolio_widgets.py       # Portfolio dashboard widgets
```

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
- **Valuation**: p_e_ratio, p_b_ratio, p_s_ratio, ev_ebitda_ratio, peg_ratio, price_to_fcf
- **Profitability**: gross_margin_pct, operating_margin_pct, net_margin_pct, roe, roa, roic
- **Quality/Risk**: altman_z_score, debt_to_equity, current_ratio, interest_coverage, leverage_ratio
- **Cash Flow**: fcf_yield, ocf_to_sales, capex_intensity, fcf_growth
- **Growth**: revenue_growth_yoy, earnings_growth_yoy, sales_cagr_3y, ebitda_growth

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
    data_df=all_stocks_raw,
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
    df_raw=all_stocks_raw,  # Wrong: should be data_df
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
   apply_enhanced_imputation_strategy_4step,  # 6-step imputation
   apply_zero_imputation,
   apply_knn_imputation_enhanced,
   apply_price_imputation,
   apply_median_imputation
   )

df_imputed = apply_enhanced_imputation_strategy_4step(
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

**Required Stage Names** (8-stage pipeline):

1. **`all_stocks_raw`** — Initial data load (from database or CSV)
2. **`all_stocks_normalized`** — Column names normalized (lowercase, underscores)
3. **`all_stocks_typed`** — Schema-aware datatype casting with diagnostics
4. **`all_stocks_winsorized`** — Outlier handling via winsorization
5. **`all_stocks_imputed`** — 6-step enhanced imputation strategy applied
6. **`all_stocks_scaled`** — Feature scaling applied
7. **`all_stocks_features`** — Advanced feature engineering applied
8. **`all_stocks_enhanced`** — Final dataset with all transformations

**Benefits**:

- **Debugging**: Inspect intermediate stages without re-running expensive operations
- **Rollback**: Revert to earlier stage if downstream transformation fails
- **Testing**: Validate each stage independently with assertions
- **Self-documenting**: Stage names clearly indicate transformation history

**Implementation Pattern**:

```python
# Stage 1: Load raw data
all_stocks_raw = load_from_db(DB_URL, limit=None)
print(f"✓ Stage 1 (raw): {all_stocks_raw.shape}")

# Validation checkpoint
assert not all_stocks_raw.empty, "Raw data must not be empty"
assert 'Ticker' in all_stocks_raw.columns, "Ticker column required"

# Stage 2: Normalize column names
all_stocks_normalized = normalize_columns(all_stocks_raw)
print(f"✓ Stage 2 (normalized): {all_stocks_normalized.shape}")

# Validation checkpoint
validate_schema(all_stocks_normalized, require_target=True)
assert all_stocks_normalized.columns.str.islower().all(), "All columns must be lowercase"

# Stage 3: Cast datatypes with schema awareness
all_stocks_typed, dtype_diagnostics = detect_and_cast_dtypes(all_stocks_normalized)
print(f"✓ Stage 3 (typed): {all_stocks_typed.shape}")

# Validation checkpoint
assert dtype_diagnostics['coercion_count'] < len(all_stocks_typed) * 0.05,
"Excessive type coercions detected (>5%)"
print(f"  Coercion rate: {dtype_diagnostics['coercion_count'] / len(all_stocks_typed):.2%}")

# Stage 4: Winsorize outliers
all_stocks_winsorized = winsorize_by_sector(
        all_stocks_typed,
        lower=WINSORIZE_LOWER,
        upper=WINSORIZE_UPPER,
        group_col='sector'
        )
print(f"✓ Stage 4 (winsorized): {all_stocks_winsorized.shape}")

# Validation checkpoint
quality_report = preprocessing_calculate_quality(all_stocks_winsorized)
assert quality_report['outlier_rate'] < 0.10, "Excessive outliers remaining (>10%)"

# Stage 5: Impute missing values
all_stocks_imputed = apply_enhanced_imputation_strategy_6step(
        all_stocks_winsorized,
        target_col=TARGET_COL,
        strategy='sector_knn',
        sector_col='sector'
        )
print(f"✓ Stage 5 (imputed): {all_stocks_imputed.shape}")

# Validation checkpoint
remaining_nulls = all_stocks_imputed.isnull().sum().sum()
assert remaining_nulls == 0, f"Imputation incomplete: {remaining_nulls} nulls remaining"

# Stage 6: Scale features
all_stocks_scaled = scale_features(
        all_stocks_imputed.copy(),
        method='robust',
        exclude_cols=['ticker', 'sector', 'region']
        )
print(f"✓ Stage 6 (scaled): {all_stocks_scaled.shape}")

# Validation checkpoint
numeric_cols = all_stocks_scaled.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
   if col not in ['ticker', 'sector', 'region']:
      assert all_stocks_scaled[col].std() > 0, f"Column {col} has zero variance"

# Stage 7: Build advanced features
all_stocks_features = build_comprehensive_features(
        all_stocks_scaled,
        phase93_categories=['momentum', 'valuation', 'profitability', 'quality_risk', 'cash_flow', 'growth']
        )
print(f"✓ Stage 7 (features): {all_stocks_features.shape}")

# Validation checkpoint
assert all_stocks_features.shape[1] > all_stocks_scaled.shape[1],
"Feature engineering must add new columns"
print(f"  New features added: {all_stocks_features.shape[1] - all_stocks_scaled.shape[1]}")

# Stage 8: Final enhancements (composite scores, interactions)
all_stocks_enhanced = add_composite_features(
        all_stocks_features,
        interaction_terms=True,
        polynomial_degree=2
        )
print(f"✓ Stage 8 (enhanced): {all_stocks_enhanced.shape}")

# Final validation checkpoint
assert all_stocks_enhanced.shape[0] == all_stocks_raw.shape[0],
"Row count must remain constant through pipeline"
print(f"✓ Pipeline complete: {all_stocks_enhanced.shape[0]} stocks, {all_stocks_enhanced.shape[1]} features")
```

**Examples**:

✅ **Correct Usage** (Stage-based naming):

```python
all_stocks_raw = load_from_csv(Path("data"))
all_stocks_normalized = normalize_columns(all_stocks_raw)
all_stocks_typed = detect_and_cast_dtypes(all_stocks_normalized)
# Each stage preserves history and enables rollback
```

❌ **Violation** (In-place mutation):

```python
all_stocks = load_from_csv(Path("data"))
all_stocks = normalize_columns(all_stocks)  # ❌ Overwrites original, no rollback
all_stocks = detect_and_cast_dtypes(all_stocks)  # ❌ Cannot inspect intermediate stages
```

❌ **Violation** (Unclear naming):

```python
df1 = load_from_csv(Path("data"))
df2 = normalize_columns(df1)  # ❌ Generic names don't indicate transformation
df3 = detect_and_cast_dtypes(df2)  # ❌ What does df3 represent?
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

## 9. Column Schema and DataFrame Conventions

### 10.1 Canonical Column Names

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

### 10.2 DataFrame Conventions

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

**Document Version:** 1.4  
**Last Updated:** 2025-11-23  
**Package Version:** 0.8.3  
**Model Version:** v9_9  
**Synchronized with:** README.md v0.8.3, CHANGELOG.md v0.8.3, pyproject.toml v0.8.3

