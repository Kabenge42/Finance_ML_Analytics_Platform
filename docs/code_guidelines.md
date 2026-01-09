# Finance ML Analytics Platform — Code Guidelines

**Version:** 2.0.0  
**Last Updated:** 2026-01-07  
**Package Version:** 0.9.7  
**Model Version:** v9_11

These guidelines codify conventions for the Finance ML Analytics Platform, covering technology stack, configuration,
architecture, function signatures, column naming, and best practices. They align with the project's 8-phase ML
workflow (Phase 9.1-9.8) and 7-phase Portfolio Optimization workflow.

**Related Documentation:**

- **[ML Workflow Guidelines](ml_workflow_guidelines.md)**: Comprehensive guidelines for the 8-phase ML workflow
- **[Changelog](docs/code_guidelines_changelog.md)**: Complete version history (v1.0-v2.0)

---

## Quick Reference Card

### Most Common Operations

| Task                 | Code                                          | Section                                   |
|----------------------|-----------------------------------------------|-------------------------------------------|
| Load with ETL        | `df, m = run_etl_pipeline(source='csv', ...)` | [7.5](#75-etl-pipeline-functions)         |
| Normalize column     | `normalize_column_name(col)`                  | [5.1](#51-normalization-rules)            |
| Get column dtype     | `get_expected_dtype('last_price')`            | [5.3](#53-schema-utility-functions)       |
| List price columns   | `list_price_cols()`                           | [5.3](#53-schema-utility-functions)       |
| Build features       | `build_features(df, preset='comprehensive')`  | [9.3](#93-phase-93-feature-categories)    |
| Validate predictions | `validate_predictions_schema(df)`             | [11](#11-standardized-predictions-schema) |

### Critical Imports

```python
# Schema (Section 5)
from finance_ml.core.schema import COLUMN_SCHEMA, normalize_column_name, list_price_cols

# ETL (Section 7.5)
from finance_ml.etl import run_etl_pipeline, ETLConfig

# Features (Section 9.3)
from finance_ml.features.api import build_features
```
---

## Table of Contents

1. [Overview and Technology Stack](#1-overview-and-technology-stack)
2. [Configuration Constants](#2-configuration-constants)
3. [Main Scripts and Entry Points](#3-main-scripts-and-entry-points)
4. [Finance_ML Package Architecture](#4-finance_ml-package-architecture)
5. [Schema Reference (Canonical)](#5-schema-reference-canonical)
6. [Code Review Checklist](#6-code-review-checklist)
7. [Standardized Function Signatures](#7-standardized-function-signatures)
8. [Notebook Best Practices and TDD Conventions](#8-notebook-best-practices-and-tdd-conventions)
9. [DataFrame Schema and Feature Engineering](#9-dataframe-schema-and-feature-engineering)
10. [Data Split and Leakage Policy](#10-data-split-and-leakage-policy)
11. [Standardized Predictions Schema](#11-standardized-predictions-schema)
12. [Sector Metrics and Calibration](#12-sector-metrics-and-calibration)
13. [Outlier Safety Rails Policy](#13-outlier-safety-rails-policy)
14. [Uncertainty and Prediction Intervals](#14-uncertainty-and-prediction-intervals)
15. [Jupyter Notebook Guidelines](#15-jupyter-notebook-guidelines)
16. [Model Optimization and Performance](#16-model-optimization-and-performance)
17. [Style Guides for Visual Elements](#17-style-guides-for-visual-elements)
18. [Portfolio Optimization Workflow](#18-portfolio-optimization-workflow)
19. [Data Quality Validation Checkpoints](#19-data-quality-validation-checkpoints)
20. [Output Artifact Standards](#20-output-artifact-standards)

---

## 1. Overview and Technology Stack

### 1.1 Project Overview

Finance ML Analytics Platform is a comprehensive toolkit for quantitative equity analysis combining unified data
pipelines, modular Python packages, interactive notebooks, and production-ready CLI tools.

**Business Objective:** Predict Stock Price Targets for all stocks in the portfolio to support investment decisions and
portfolio optimization.

**Target Variable:** `price_target` for regression modeling (see [Section 5.4](#54-column-roles-and-semantics) for role
definitions)

**8-Phase ML Workflow:**

| Phase | Description                                      | Key Module                              |
|-------|--------------------------------------------------|-----------------------------------------|
| 9.1   | Loading and preprocessing with 6-step imputation | `finance_ml.etl`                        |
| 9.2   | Enhanced exploratory data analysis               | `finance_ml.ml_workflow.eda`            |
| 9.3   | Advanced feature engineering                     | `finance_ml.features`                   |
| 9.4   | Multi-class event classification                 | `finance_ml.ml_workflow.classification` |
| 9.5   | Sector-optimized regression with quantile models | `finance_ml.ml_workflow.regression`     |
| 9.6   | Model evaluation and error analysis              | `finance_ml.ml_workflow.evaluation`     |
| 9.7   | Identification of under/overvalued stocks        | `finance_ml.ml_workflow.analytics`      |
| 9.8   | Comprehensive analytics and reporting            | `finance_ml.ml_workflow.reporting`      |

### 1.2 Technology Stack

**Language & Runtime:**

- Python: 3.12, 3.13, or 3.14 (per `pyproject.toml`)
- Package Manager: pip with `requirements.txt` and `pyproject.toml` (PEP 621)
- Build System: setuptools ≥68.0

**Core Libraries:**

| Category              | Libraries                                                                                     |
|-----------------------|-----------------------------------------------------------------------------------------------|
| **Data**              | pandas ≥2.2.0, numpy ≥1.26.0 (py<3.14) / ≥2.1.0 (py≥3.14), scipy ≥1.12.0, statsmodels ≥0.14.1 |
| **ML Core**           | scikit-learn ≥1.5.0, imbalanced-learn ≥0.12.0                                                 |
| **Gradient Boosting** | XGBoost ≥2.1.0, LightGBM ≥4.3.0, CatBoost ≥1.2.2 (py<3.14)                                    |
| **Explainability**    | SHAP ≥0.45.0 (py<3.14)                                                                        |
| **Visualization**     | matplotlib ≥3.8.0, seaborn ≥0.13.0, plotly ≥5.18.0                                            |
| **Dashboards**        | streamlit, dash                                                                               |
| **Utilities**         | joblib ≥1.3.0, tqdm ≥4.65.0, xlsxwriter ≥3.1.0, psutil ≥5.9.0                                 |

**Optional Dependencies:**

| Category          | Libraries                                                                 |
|-------------------|---------------------------------------------------------------------------|
| **Deep Learning** | TensorFlow ≥2.15.0 (py<3.14), scikeras ≥0.13.0                            |
| **Database**      | PostgreSQL (psycopg2-binary ≥2.9.0, SQLAlchemy ≥2.0.0), SQLite 3          |
| **Development**   | pytest ≥8.0.0, pytest-cov ≥4.1.0, black ≥24.0.0, ruff ≥0.2.0, mypy ≥1.8.0 |

**Database Systems:**

- **Primary**: PostgreSQL 15+ (recommended for production)
- **Alternative**: SQLite 3 (for local testing)
- **Schema**: See [Section 5](#5-schema-reference-canonical) for complete column definitions

---

## 2. Configuration Constants

All configuration constants follow the **Single Source of Truth** principle. Constants are defined once in this section
and referenced throughout the codebase.

### 2.1 Core Constants
```python
import os
from pathlib import Path

# === TARGET COLUMNS (Section 5.4 Role: target, target_fallback) ===
TARGET_COL = 'price_target'
TARGET_COL_FALLBACK = 'last_price'

# === DATA SPLITS ===
TEST_SIZE = 0.2
TRAIN_SIZE = 1 - TEST_SIZE
CV_FOLDS = 5

# === QUANTILE REGRESSION ===
QUANTILES = [0.1, 0.5, 0.9]
LOWER_QUANTILE = QUANTILES[0]
MEDIAN_QUANTILE = QUANTILES[1]
UPPER_QUANTILE = QUANTILES[2]

# === SECTOR CONSTRAINTS ===
MIN_SECTOR_SAMPLES = 20

# === PORTFOLIO CONSTRAINTS ===
MAX_SECTOR_WEIGHT = 0.25
MAX_SINGLE_POSITION = 0.10

# === OUTLIER THRESHOLDS ===
IQR_MULTIPLIER = 2.5
ZSCORE_THRESHOLD = 3.0
WINSORIZE_LOWER = 0.10  # Conservative (see Section 13)
WINSORIZE_UPPER = 0.90

# === CONFIDENCE SCORING ===
CONFIDENCE_LOW_THRESHOLD = 0.50
CONFIDENCE_MEDIUM_THRESHOLD = 0.75

# === REPRODUCIBILITY ===
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v9_11')

# === DIRECTORIES ===
DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
MODEL_DIR = Path(os.getenv('MODEL_DIR', 'models'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', 'outputs'))
CACHE_DIR = Path(os.getenv('CACHE_DIR', '.cache'))
```
### 2.2 Environment Variables
```python
# Required
TF_CPP_MIN_LOG_LEVEL = '2'  # Reduce TensorFlow verbosity

# Optional overrides
DB_URL = os.getenv('DB_URL', 'postgresql+psycopg2://postgres:@localhost:5432/postgres')
N_JOBS = int(os.getenv('N_JOBS', '-1'))
MEMORY_LIMIT = os.getenv('MEMORY_LIMIT', '8GB')

# Feature engineering toggles
FEATURE_SECTOR_INTERACTIONS = os.getenv('FEATURE_SECTOR_INTERACTIONS', '1') == '1'
FEATURE_IMPORTANCE_THRESHOLD = float(os.getenv('FEATURE_IMPORTANCE_THRESHOLD', '0.01'))
```
### 2.3 Configuration Validation
```python
def validate_configuration() -> bool:
    """Validate all configuration constants meet required constraints."""
    # Target columns
    assert TARGET_COL and isinstance(TARGET_COL, str), f"Invalid TARGET_COL: {TARGET_COL}"
    assert TARGET_COL_FALLBACK and isinstance(TARGET_COL_FALLBACK, str)
    
    # Split configuration
    assert 0 < TEST_SIZE < 1, f"TEST_SIZE must be in (0, 1): {TEST_SIZE}"
    assert abs((TRAIN_SIZE + TEST_SIZE) - 1.0) < 0.01
    
    # CV folds
    assert isinstance(CV_FOLDS, int) and CV_FOLDS >= 2
    
    # Quantiles
    assert all(0 < q < 1 for q in QUANTILES)
    assert QUANTILES == sorted(QUANTILES), "QUANTILES must be monotonically increasing"
    
    # Sector constraints
    assert MIN_SECTOR_SAMPLES >= 1
    assert 0 < MAX_SECTOR_WEIGHT <= 1
    assert 0 < MAX_SINGLE_POSITION <= 1
    
    # Winsorization bounds
    assert 0 <= WINSORIZE_LOWER < 0.5
    assert 0.5 < WINSORIZE_UPPER <= 1
    
    print("✓ All configuration constants validated")
    return True
```
### 2.4 Business-Driven Configuration Rationale

| Constant                | Value             | Business Rationale                                          |
|-------------------------|-------------------|-------------------------------------------------------------|
| `TARGET_COL`            | `'price_target'`  | Core prediction target for investment decisions             |
| `TEST_SIZE`             | `0.2`             | Balance between training quality (80%) and validation (20%) |
| `QUANTILES`             | `[0.1, 0.5, 0.9]` | 80% prediction interval for risk assessment                 |
| `MIN_SECTOR_SAMPLES`    | `20`              | Minimum for statistically meaningful sector models          |
| `MAX_SECTOR_WEIGHT`     | `0.25`            | Portfolio diversification constraint                        |
| `WINSORIZE_LOWER/UPPER` | `0.10/0.90`       | Conservative bounds preserving valid extremes               |
| `RANDOM_SEED`           | `42`              | Reproducibility for regulatory compliance                   |

---

## 3. Main Scripts and Entry Points

### 3.1 Main Entry Points

| Script/Tool                   | Description                   | Usage                                                |
|-------------------------------|-------------------------------|------------------------------------------------------|
| `ml_finance_model_main.ipynb` | Main notebook (Phase 9.1-9.8) | `jupyter notebook ml_finance_model_main.ipynb`       |
| `ml_finance_model_main.py`    | Python script version         | `python ml_finance_model_main.py --data-source auto` |
| `finance-ml`                  | CLI: Full pipeline            | `finance-ml --data-source auto --output-dir outputs` |
| `finance-ml-analyze`          | CLI: EDA/analytics only       | `finance-ml-analyze --data-source csv`               |
| `finance-ml-validate`         | CLI: Validation only          | `finance-ml-validate --data-source db`               |

### 3.2 CLI Entry Points
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

### 3.4 Utility Scripts

| Script                          | Purpose                                 |
|---------------------------------|-----------------------------------------|
| `tools/setup_environment.py`    | Full environment and dependency setup   |
| `tools/run_fast_tests.py`       | Quick verification of utility modules   |
| `tools/run_earnings_monitor.py` | Monitors and visualizes earnings events |
| `tools/load_equities_data.py`   | Loads and processes equities data       |

### 3.5 Database Scripts

| Script                       | Description                |
|------------------------------|----------------------------|
| `create_equities_schema.sql` | PostgreSQL schema creation |
| `import_equities_data.sql`   | PostgreSQL data import     |

---

## 4. Finance_ML Package Architecture

### 4.1 Package Structure
```
finance_ml/
├── core/                     # Shared constants & schema
│   ├── schema.py             # ← CANONICAL SCHEMA (Section 5)
│   └── constants.py          # Global constants
├── etl/                      # Unified ETL Pipeline
│   ├── config.py             # Configuration dataclasses
│   ├── pipeline.py           # Pipeline orchestration
│   ├── currency.py           # FOREX conversion
│   └── stages/               # ETL stages
├── features/                 # Feature Engineering
│   ├── api.py                # Public API with presets
│   ├── core.py               # Core feature functions
│   └── advanced/             # Domain-specific modules
│       ├── valuation.py
│       ├── profitability.py
│       ├── momentum.py
│       ├── earnings.py
│       └── ...
├── ml_workflow/              # ML Workflow Phases (9.1-9.8)
│   ├── preprocessing/        # Phase 9.1
│   ├── eda/                  # Phase 9.2
│   ├── classification/       # Phase 9.4
│   ├── regression/           # Phase 9.5
│   ├── evaluation/           # Phase 9.6
│   ├── analytics/            # Phase 9.7
│   └── reporting/            # Phase 9.8
└── dashboards/               # Interactive Applications
```
### 4.2 Phase Alignment

| Phase | Subpackage                    | Entry Point          | Description                                      |
|-------|-------------------------------|----------------------|--------------------------------------------------|
| 9.1   | `etl/`                        | `run_etl_pipeline()` | Unified ETL with imputation, scaling, transforms |
| 9.2   | `ml_workflow/eda/`            | —                    | Exploratory analysis, statistical tests          |
| 9.3   | `features/`                   | `build_features()`   | Feature engineering (Schema-driven)              |
| 9.4   | `ml_workflow/classification/` | —                    | Event classification                             |
| 9.5   | `ml_workflow/regression/`     | —                    | Regression models, quantile, stacking            |
| 9.6   | `ml_workflow/evaluation/`     | —                    | Metrics, calibration, safety rails               |
| 9.7   | `ml_workflow/analytics/`      | —                    | Mispricing, portfolio optimization               |
| 9.8   | `ml_workflow/reporting/`      | —                    | Dashboard data, reporting                        |

### 4.3 Canonical Import Patterns

> **Principle**: Always import from the most specific subpackage. Use `finance_ml.core.schema` for all column-related
> operations.

**Tier 1: Core Schema (Always Use)**

```python
# ✅ CORRECT: All schema operations from core.schema
from finance_ml.core.schema import (
    COLUMN_SCHEMA,
    PHASE93_FEATURE_CATEGORIES,
    DType,
    Role,
    ColumnMeta,
    normalize_column_name,
    get_expected_dtype,
    list_price_cols,
    list_numeric_feature_cols,
    list_categorical_cols,
    list_date_cols,
    list_count_cols,
    list_non_recurring_cols,
    list_knn_imputable_cols,
    list_required_schema_columns_for_etl,
)
```

**Tier 2: ETL Pipeline**

```python
# ✅ CORRECT: ETL from etl subpackage
from finance_ml.etl import (
    ETLConfig,
    ETLPipeline,
    run_etl_pipeline,
    CurrencyConversionConfig,
    ImputationConfig,
)
from finance_ml.etl.currency import CurrencyConverter, convert_to_usd
```

**Tier 3: Features**
```python
# ✅ CORRECT: Features API
from finance_ml.features.api import build_features
from finance_ml.features.advanced import valuation, profitability, momentum
```

**Tier 4: ML Workflow Phases**

```python
# ✅ CORRECT: Direct phase imports
from finance_ml.ml_workflow.regression import models, quantile
from finance_ml.ml_workflow.evaluation import metrics, calibration
from finance_ml.ml_workflow.analytics import portfolio, mispricing
```

### 4.4 Deprecated Import Paths

> ⚠️ **Do Not Use**: These paths are deprecated and will be removed in v3.0.

```python
# ❌ DEPRECATED
from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA  # Use core.schema
from finance_ml.ml_workflow.preprocessing.schema import normalize_column_name  # Use core.schema
from finance_ml.ml_workflow.preprocessing.etl import etl_with_financial_metrics  # Use etl module
```

---

## 5. Schema Reference (Canonical)

> **This is the single source of truth for all column definitions, naming conventions, and schema utilities.**

### 5.1 Normalization Rules

All SQL column names are normalized to Python-compatible names using `normalize_column_name()`:

```python
from finance_ml.core.schema import normalize_column_name

# Transformation rules (applied in order):
# 1. '#' → 'num'     (analyst rating counts)
# 2. '%' → 'pct'     (percentages)
# 3. '&' → 'and'     (conjunctions)
# 4. '/' → '_'       (ratios)
# 5. '(', ')' → ''   (parentheses removed)
# 6. '-' → '_'       (hyphens)
# 7. Spaces → '_'    (spaces)
# 8. Collapse multiple '_'
# 9. Strip leading/trailing '_'
# 10. Lowercase

# Examples:
normalize_column_name("# Strong Buy Ratings")  # → 'num_strong_buy_ratings'
normalize_column_name("P/E (LTM)")             # → 'p_e_ltm'
normalize_column_name("1-Day %")               # → '1_day_pct'
normalize_column_name("R&D Expenses")          # → 'r_and_d_expenses'
```

### 5.2 Schema Registry

**Version:** 2.0 (Updated 2026-01-07)

The authoritative column schema is defined in `finance_ml/core/schema.py`:

```python
from finance_ml.core.schema import COLUMN_SCHEMA

# COLUMN_SCHEMA is Dict[str, ColumnMeta]
# Query current count:
print(f"Total columns: {len(COLUMN_SCHEMA)}")  # ~599 columns
```

**Schema Composition:**

| Category             | Count | Description                             |
|----------------------|-------|-----------------------------------------|
| Source columns       | 330   | From CSV/SQL schema                     |
| Log-transformed      | 61    | ETL-generated (log1p of market values)  |
| Legacy aliases       | 43    | Backward compatibility (role=auxiliary) |
| Generic base         | 36    | No time suffix                          |
| Conditional metrics  | 34    | With `_applicable` flags                |
| Derived ratios       | 26    | ETL semantic transforms                 |
| Phase 9.3 composites | 4     | Quality scores                          |

### 5.3 Schema Utility Functions

All functions are imported from `finance_ml.core.schema`:

| Function                                 | Purpose                        | Returns     |
|------------------------------------------|--------------------------------|-------------|
| `get_sql_column_name(col)`               | Get SQL-compatible name        | `str`       |
| `normalize_column_name(col)`             | Normalize to Python name       | `str`       |
| `generate_sql_schema()`                  | Generate CREATE TABLE SQL      | `str`       |
| `get_expected_dtype(col)`                | Get expected pandas dtype      | `DType`     |
| `get_pandas_nullable_dtype(col)`         | Get nullable pandas dtype      | `str`       |
| `get_numpy_dtype(col)`                   | Get numpy dtype                | `str`       |
| `list_numeric_feature_cols()`            | List numeric features          | `List[str]` |
| `list_categorical_cols()`                | List categorical columns       | `List[str]` |
| `list_date_cols()`                       | List date/datetime columns     | `List[str]` |
| `list_price_cols()`                      | List price columns (protected) | `List[str]` |
| `list_count_cols()`                      | List count-type columns        | `List[str]` |
| `list_non_recurring_cols()`              | List non-recurring items       | `List[str]` |
| `list_knn_imputable_cols()`              | List KNN-imputable columns     | `List[str]` |
| `list_required_schema_columns_for_etl()` | Get required ETL columns       | `List[str]` |
| `list_etl_generated_column_patterns()`   | Get ETL-generated patterns     | `List[str]` |

**Usage Examples:**
```python
from finance_ml.core.schema import (
    get_expected_dtype,
    list_price_cols,
    list_numeric_feature_cols,
    list_required_schema_columns_for_etl,
)

# Get dtype for validation
dtype = get_expected_dtype('last_price')  # Returns DType.FLOAT

# Get protected price columns (never transform these)
price_cols = list_price_cols()
# ['last_price', 'price_target', 'price_target_median', 'price_target_low', ...]

# Get numeric features for modeling
numeric_cols = list_numeric_feature_cols()

# Get required columns for ETL validation
required = list_required_schema_columns_for_etl()
# ['ticker', 'isin', 'sector', 'region', 'last_price', 'price_target', ...]
```

### 5.4 Column Roles and Semantics

The `ColumnMeta` dataclass defines column metadata:

```python
from dataclasses import dataclass
from finance_ml.core.schema import ColumnMeta, DType, Role

@dataclass
class ColumnMeta:
    dtype: DType          # Data type
    role: Role            # Semantic role
    sql_name: str         # SQL column name
    description: str      # Human-readable description
```

**DType Enum:**

| Value      | Description            | Pandas Type        |
|------------|------------------------|--------------------|
| `FLOAT`    | Floating-point numbers | `float64`          |
| `INT`      | Integers               | `Int64` (nullable) |
| `STRING`   | Text strings           | `string`           |
| `CATEGORY` | Categorical values     | `category`         |
| `DATETIME` | Date/time values       | `datetime64[ns]`   |
| `BOOL`     | Boolean flags          | `boolean`          |

**Role Enum:**

| Value                 | Description         | Treatment                           |
|-----------------------|---------------------|-------------------------------------|
| `feature`             | ML features         | Include in modeling                 |
| `target`              | Prediction target   | `price_target`                      |
| `target_fallback`     | Alternative targets | `last_price`, `price_target_median` |
| `id`                  | Identifiers         | `ticker`, `isin`                    |
| `categorical`         | Grouping columns    | `sector`, `region`, `country`       |
| `date`                | Date columns        | `last_updated`, `next_earnings`     |
| `auxiliary`           | Legacy/optional     | Excluded from validation            |
| `market`              | Market prices       | Protected from transforms           |
| `financial_statement` | P&L items           | Zero-fill on missing                |
| `balance_sheet`       | Balance sheet items | Zero-fill on missing                |
| `cash_flow`           | Cash flow items     | Zero-fill on missing                |
| `count`               | Count metrics       | Integer, zero-fill                  |
| `non_recurring`       | Exceptional items   | Zero-fill on missing                |

### 5.5 Common Column Mappings

| SQL Column Name           | Normalized Python Name | Role            | DType    |
|---------------------------|------------------------|-----------------|----------|
| `"Ticker"`                | `ticker`               | id              | string   |
| `"ISIN"`                  | `isin`                 | id              | string   |
| `"Sector"`                | `sector`               | categorical     | category |
| `"Industry"`              | `industry`             | categorical     | category |
| `"Region"`                | `region`               | categorical     | category |
| `"Last Price"`            | `last_price`           | market          | float    |
| `"Price Target"`          | `price_target`         | target          | float    |
| `"Price Target - Median"` | `price_target_median`  | target_fallback | float    |
| `"Market Cap"`            | `market_cap`           | feature         | float    |
| `"P/E (LTM)"`             | `p_e_ltm`              | feature         | float    |
| `"EV/EBITDA (LTM)"`       | `ev_ebitda_ltm`        | feature         | float    |

### 5.6 Price Columns (Protected)

> **Critical Policy**: Price columns must **NEVER** be winsorized, scaled, or transformed in place.

The `list_price_cols()` function returns all 21 protected price columns:

| Category           | Columns                                                                                                                                         |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| **Current** (6)    | `last_price`, `price_target`, `price_target_median`, `price_target_low`, `price_target_high`, `price_target_ytd_ago`                            |
| **Historical** (9) | `price_5d_ago`, `price_1w_ago`, `price_1m_ago`, `price_3m_ago`, `price_6m_ago`, `price_1y_ago`, `price_3y_ago`, `price_5y_ago`, `price_qtd_ago` |
| **52W Bounds** (2) | `52w_high_adj`, `52w_low_adj`                                                                                                                   |
| **EMAs** (4)       | `ema_20d`, `ema_50d`, `ema_100d`, `ema_250d`                                                                                                    |

**Rationale**: The core valuation metric `(Predicted_Target - Last_Price) / Last_Price` requires original price scale.

### 5.7 ETL-Required Columns

The `list_required_schema_columns_for_etl()` function provides validation targets:

**Core Required (12 columns):**

- **Identifiers**: `ticker`, `isin`
- **Group Keys**: `sector`, `region`, `country`, `trading_country`
- **Prices/Targets**: `last_price`, `price_target`, `price_target_median`, `price_target_ytd_ago`
- **Market Values**: `market_cap`, `enterprise_value`

**Extended Financials (optional, 6 columns):**

- `total_revenues_ltm`, `ebitda_ltm`, `net_income_is_ltm`, `total_assets_ltm`, `total_debt_ltm`, `total_equity_ltm`

### 5.8 Derived ETL Columns

Columns created by ETL semantic transformations:

**Log-Transformed (13 columns):**
```python
log_columns = [
    'log_market_cap', 'log_enterprise_value', 'log_revenue', 'log_ebitda',
    'log_net_income', 'log_total_assets', 'log_total_equity', 'log_total_debt',
    'log_gross_profit', 'log_operating_income', 'log_operating_cash_flow',
    'log_capex', 'log_cash_and_equivalents'
]
```

**Derived Ratios (17 columns):**
```python
derived_ratios = [
    'p_e_ratio', 'p_s_ratio', 'ev_ebitda_ratio', 'ev_sales_ratio',
    'gross_margin_pct', 'operating_margin_pct', 'net_margin_pct',
    'roe', 'roa', 'roic', 'debt_to_equity', 'debt_to_assets',
    'revenue_growth', 'ebitda_growth', 'earnings_growth',
    'target_vs_price', 'peg_ratio'
]
```

### 5.9 Schema Validation

The ETL pipeline includes automated schema validation (Stage 11):

```python
from finance_ml.etl import ETLPipeline, ETLConfig

config = ETLConfig(validate_quality=True)
pipeline = ETLPipeline(config=config)
df_transformed = pipeline.transform(df_extracted)

# Access validation metrics
print(f"Schema alignment: {pipeline.metrics.schema_alignment_score:.2%}")
print(f"Unknown columns: {pipeline.metrics.unknown_columns_count}")
print(f"Missing expected: {pipeline.metrics.missing_expected_columns_count}")
print(f"Dtype mismatches: {pipeline.metrics.dtype_mismatches_count}")
```
**Validation Checks:**

1. **Unknown Columns**: In DataFrame but not in `COLUMN_SCHEMA`
2. **Missing Expected**: In `COLUMN_SCHEMA` but not in DataFrame
3. **Dtype Mismatches**: Actual dtype differs from expected
4. **Alignment Score**: `1.0 - (unknown + missing + mismatch) / total_expected`

**Warning Thresholds:**

- Alignment score < 95%: Warning logged
- Unknown columns > 10: Warning logged

### 5.10 Column Normalization Enforcement

> **Policy**: All column normalization **MUST** use `normalize_column_name()` from `finance_ml.core.schema`.
```python
# ✅ CORRECT
from finance_ml.core.schema import normalize_column_name
df.columns = [normalize_column_name(col) for col in df.columns]

# ❌ INCORRECT: Custom regex
df.columns = df.columns.str.replace(r"[^0-9a-zA-Z]+", "_", regex=True).str.lower()
# Produces 'strong_sell_ratings' instead of 'num_strong_sell_ratings'
```

**CI/CD Enforcement:**

- `tests/test_schema_normalization.py` — Normalization rules
- `tests/test_schema_completeness.py` — Schema integrity
- `tests/test_data_loading_normalization.py` — Integration tests

---

## 6. Code Review Checklist

### 6.1 Jupyter Notebook Checklist

**Configuration and Setup:**

- [ ] Constants from Section 2 defined at top
- [ ] `validate_configuration()` called and passes
- [ ] `RANDOM_SEED` set: `np.random.seed(RANDOM_SEED)`
- [ ] Output directories use pathlib

**Data Loading (uses Schema Section 5):**

- [ ] ETL pipeline used: `run_etl_pipeline()` (Section 7.5)
- [ ] Column normalization via `normalize_column_name()` (Section 5.1)
- [ ] Required columns validated via `list_required_schema_columns_for_etl()` (Section 5.7)
- [ ] Data types validated via `get_expected_dtype()` (Section 5.3)
- [ ] Price columns protected via `list_price_cols()` (Section 5.6)

**Post-ETL Validation (Section 19):**

- [ ] DataFrame not empty
- [ ] Critical columns present
- [ ] No missing values after imputation
- [ ] Schema alignment score ≥ 95%

**Feature Engineering:**

- [ ] Features use `PHASE93_FEATURE_CATEGORIES` from schema
- [ ] No target leakage in construction
- [ ] Feature importance analyzed

**Model Training:**

- [ ] Train/test split follows Section 10
- [ ] Cross-validation uses grouped/stratified strategy
- [ ] Model artifacts saved with `MODEL_VERSION`

**Predictions:**

- [ ] Schema follows Section 11
- [ ] Monotonicity: `pred_p10 ≤ pred_p50 ≤ pred_p90`
- [ ] Non-negativity enforced

### 6.2 Python Script Checklist

**Code Structure:**

- [ ] Type hints for function signatures
- [ ] Docstrings (NumPy/Google style)
- [ ] Imports: stdlib → third-party → local
- [ ] Single responsibility functions
- [ ] No global mutable state

**Schema Usage:**

- [ ] All column operations use `finance_ml.core.schema`
- [ ] No hardcoded column lists (use schema functions)
- [ ] Normalization uses `normalize_column_name()`

**Error Handling:**

- [ ] Input validation with clear messages
- [ ] Graceful degradation for optional dependencies
- [ ] Logging instead of print statements

**Testing:**

- [ ] Unit tests cover core functionality
- [ ] Test coverage ≥ 80%
- [ ] Tests isolated from external services

### 6.3 Parameter Naming Conventions

| Parameter Type | Convention                                       | Examples                               |
|----------------|--------------------------------------------------|----------------------------------------|
| DataFrames     | `df`, `data_df`, `features_df`, `predictions_df` | NOT `df1`, `data`                      |
| Train/Test     | `X_train`, `X_test`, `y_train`, `y_test`         | NOT `X_tr`, `y_tst`                    |
| Columns        | `*_col` suffix                                   | `target_col`, `sector_col`, `date_col` |
| Output         | `output_dir`                                     | NOT `out_dir`, `save_dir`              |
| Model info     | `model_info` dict                                | NOT separate params                    |

---

## 7. Standardized Function Signatures

### 7.1 Training Functions

**Contract**: All `train_*` functions return a dict with these keys:
```python
{
    "model": fitted_estimator,
    "metrics": Dict[str, float],      # mae, rmse, r2, etc.
    "y_pred": array_like,             # Predictions
    "y_proba": Optional[array_like],  # Class probabilities
    "artifacts": Optional[Dict],      # Feature importance, etc.
}
```
### 7.2 Dataset Preparation Functions

**Contract**: Return 5-tuple or `DatasetSplit` dataclass:
```python
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class DatasetSplit:
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    meta: Dict[str, Any]  # feature_names, categorical_features, etc.
```

### 7.3 Phase-Specific Signatures

**Phase 9.1 — Preprocessing:**
```python
from finance_ml.etl import run_etl_pipeline, ETLConfig

df, metrics = run_etl_pipeline(
    source='csv',  # or 'db', 'all_stocks'
    data_dir='data/',
    config=ETLConfig(
        apply_imputation=True,
        imputation_strategy='6step',
    ),
    return_metrics=True,
)
```

**Phase 9.3 — Features:**
```python
from finance_ml.features.api import build_features

features_df = build_features(
    df,
    preset="comprehensive",  # basic, momentum, quality, comprehensive
)
```

**Phase 9.5 — Regression:**
```python
from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor

result = train_quantile_regressor(
    X_train, y_train, X_test, y_test,
    quantiles=[0.1, 0.5, 0.9],
)
# Returns: {"model", "metrics", "quantile_predictions": {q: array}}
```

**Phase 9.6 — Evaluation:**
```python
from finance_ml.ml_workflow.evaluation.metrics import (
    calculate_regression_metrics,
    calculate_sector_metrics,
)

metrics = calculate_regression_metrics(y_true, y_pred, include_mape=True)
sector_metrics = calculate_sector_metrics(df, 'y_true', 'y_pred', 'sector')
```

### 7.4 Feature Alignment
```python
from finance_ml.ml_workflow.regression.dataset import (
    align_features_to_model,
    predict_with_model,
)

# Align features to model's expected columns
X_aligned = align_features_to_model(X_test, model, fill_value=0.0)

# Or use wrapper for safe prediction
predictions = predict_with_model(model, X_test, fill_missing=0.0)
```
### 7.5 ETL Pipeline Functions

The ETL module provides a 12-stage pipeline:

| Stage | Description               |
|-------|---------------------------|
| 1     | Extraction (CSV/Database) |
| 2     | Dtype Casting             |
| 3     | Semantic Classification   |
| 4     | Validation                |
| 5     | Row Dropping              |
| 6     | Sanitization              |
| 7     | Imputation (6-step)       |
| 8     | Currency Conversion       |
| 9     | Semantic Transforms       |
| 10    | Scaling                   |
| 11    | Financial Metrics         |
| 12    | Feature Engineering       |

**Primary Interface:**
```python
from finance_ml.etl import (
    ETLConfig,
    ETLPipeline,
    run_etl_pipeline,
    CurrencyConversionConfig,
    ImputationConfig,
)

# Basic usage
df, metrics = run_etl_pipeline(
    source='all_stocks',
    db_url='postgresql://user:pass@localhost/postgres',
    return_metrics=True,
)

# Advanced configuration
config = ETLConfig(
    currency_conversion=CurrencyConversionConfig(
        enabled=True,
        target_currency="USD",
        use_business_day_fallback=True,
    ),
    imputation=ImputationConfig(
        strategy="6step",
        apply_dividend_zero_fill=True,
    ),
)
df, metrics = run_etl_pipeline(source='csv', data_dir='data/', config=config)
```

**Currency Conversion (Stage 8):**
```python
from finance_ml.etl.currency import CurrencyConverter, convert_to_usd

# Standard conversion with fallback
df_converted = convert_to_usd(df, use_fallback=True)

# Class-based with metrics
converter = CurrencyConverter(
    target_currency="USD",
    max_fallback_days=7,
)
df_converted = converter.convert_dataframe(df)
print(converter.get_metrics().summary())
```
---

## 8. Notebook Best Practices and TDD Conventions

### 8.1 Configuration Constants

> **Reference**: Use constants from Section 2. Do not redefine.
```python
# Cell 1: Configuration
import os
import numpy as np
from pathlib import Path

# Import from shared module OR define per Section 2.1
TARGET_COL = 'price_target'
TARGET_COL_FALLBACK = 'last_price'
TEST_SIZE = 0.2
# ... etc.

np.random.seed(RANDOM_SEED)
validate_configuration()
```

### 8.2 DataFrame Stage Naming

**Required Stage Names (6-stage pipeline):**

| Stage | Name                        | Description                        | Shape     |
|-------|-----------------------------|------------------------------------|-----------|
| 1     | `all_stocks_preprocessed`   | ETL output                         | ~(N, 655) |
| 2     | `all_stocks_features`       | With Phase 9.3 features            | ~(N, 700) |
| 3     | `all_stocks_classification` | With classification outputs        | ~(N, 710) |
| 4     | `all_stocks_enhanced`       | Final regression-ready             | ~(N, 928) |
| 5     | `all_stocks_selected`       | After feature selection (optional) | ~(N, 400) |
| 6     | `all_stocks_balanced`       | SMOTE-balanced (optional)          | Varies    |

**Deprecated Names (Do Not Use):**

| Deprecated                       | Replacement                 |
|----------------------------------|-----------------------------|
| `all_stocks_typed`               | `all_stocks_preprocessed`   |
| `all_stocks_winsorized`          | `all_stocks_preprocessed`   |
| `all_stocks_imputed`             | `all_stocks_preprocessed`   |
| `all_stocks_scaled`              | `all_stocks_preprocessed`   |
| `all_stocks_with_classification` | `all_stocks_classification` |

**Implementation Pattern:**
```python
from finance_ml.etl import run_etl_pipeline
from finance_ml.features.api import build_features

# Stage 1: ETL
all_stocks_preprocessed, metrics = run_etl_pipeline(
    source='csv', data_dir='data/', return_metrics=True
)
print(f"✓ Stage 1: {all_stocks_preprocessed.shape}")

# Stage 2: Features
all_stocks_features = build_features(all_stocks_preprocessed, preset='comprehensive')
print(f"✓ Stage 2: {all_stocks_features.shape}")

# ... continue stages
```
### 8.3 Magic Numbers Policy

> **Policy**: All numeric literals with semantic meaning must be named constants.

**Prohibited:**
```python
# ❌ Magic numbers
train, test = train_test_split(df, test_size=0.2, random_state=42)
```

**Required:**
```python
# ✅ Named constants
train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
```

**Allowed Inline:**

- Universal constants: `0`, `1`, `100` (percentage conversion)
- Algorithm parameters with clear context: `np.clip(x, 0, 1)`

### 8.4 Semantic Column Classification

> **Reference**: Use schema functions from Section 5.3 for column classification.
```python
from finance_ml.core.schema import (
    list_price_cols,
    list_numeric_feature_cols,
    list_categorical_cols,
    list_count_cols,
)

# Get protected columns
protected = list_price_cols()

# Get columns for scaling
scalable = [c for c in list_numeric_feature_cols() if c not in protected]
```

### 8.5 ETL Best Practices

**Recommended Entry Points:**

| Function             | Use Case                  |
|----------------------|---------------------------|
| `run_etl_pipeline()` | Default for all workflows |
| `ETLPipeline` class  | Fine-grained control      |

**Required Validation After ETL:**
```python
# Validation checkpoint
assert not all_stocks_preprocessed.empty
assert 'ticker' in all_stocks_preprocessed.columns
assert 'sector' in all_stocks_preprocessed.columns
assert all_stocks_preprocessed.isna().sum().sum() == 0
print(f"✓ ETL validation passed")
```
---

## 9. DataFrame Schema and Feature Engineering

### 9.1 Canonical Column Names

> **Reference**: All column names follow normalization rules in Section 5.1.

**Target Columns:**

- Primary: `price_target` (Role: target)
- Fallback: `price_target_median`, `last_price` (Role: target_fallback)

**Identifier Columns:**

- `ticker`, `isin` (Role: id)
- `sector`, `region`, `country` (Role: categorical)

### 9.2 DataFrame Conventions

**Index:**

- Use `ticker` as index for stock-level DataFrames
- Reset index before saving: `df.reset_index().to_csv(...)`

**Column Order:**

1. Identifiers: `ticker`, `isin`, `sector`, `region`
2. Targets: `last_price`, `price_target`
3. Features: alphabetical or by category
4. Predictions: `y_true`, `y_pred`, `pred_p10`, `pred_p50`, `pred_p90`

**Missing Values:**

- Represented as `np.nan` or `pd.NA`
- Never use `0`, `-1`, or empty string for missing
- Apply imputation before modeling

### 9.3 Phase 9.3 Feature Categories

> **Reference**: Feature categories are defined in `PHASE93_FEATURE_CATEGORIES` (Section 5).

```python
from finance_ml.core.schema import PHASE93_FEATURE_CATEGORIES

# Get current feature count
total = sum(len(cols) for cols in PHASE93_FEATURE_CATEGORIES.values())
print(f"Total Phase 9.3 features: {total}")

# List categories
for category, cols in PHASE93_FEATURE_CATEGORIES.items():
    print(f"{category}: {len(cols)} features")
```
**Category Overview:**

| Category             | Features | Description                          |
|----------------------|----------|--------------------------------------|
| Momentum & Technical | 25       | EMA crossovers, RSI, price momentum  |
| Valuation Ratios     | 25       | P/E, P/B, EV/EBITDA, PEG             |
| Profitability        | 16       | Margins, ROE, ROA, ROIC              |
| Quality & Risk       | 18       | Altman Z, Piotroski F, volatility    |
| Cash Flow            | 5        | FCF yield, OCF/Sales                 |
| Growth Metrics       | 9        | Revenue, EBITDA, earnings growth     |
| Earnings Quality     | 33       | Surprise analysis, GAAP vs. Adjusted |
| ...                  | ...      | ...                                  |

**Building Features:**
```python
from finance_ml.features.api import build_features

# Build with preset
df_features = build_features(df, preset='comprehensive')

# Or specific categories
df_momentum = build_features(df, preset='momentum')
```

### 9.4 Temporal Calculation Standards

> **Policy**: All temporal calculations use `reference_date` parameter.
```python
from finance_ml.features.advanced import engineer_temporal_features
import pandas as pd

# ✅ CORRECT: Explicit reference date
df_temporal = engineer_temporal_features(
    df,
    reference_date=pd.Timestamp('2026-01-07'),
)

# Creates: days_to_earnings, earnings_report_recency
```

### 9.5 Multi-Label Classification
```python
from finance_ml.ml_workflow.classification.labels import create_multilabel_event_labels

labels = create_multilabel_event_labels(
    df,
    categories=['valuation', 'momentum', 'quality'],
    sector_adjusted=True,
    threshold_percentile=0.6,
)
# Returns: label_valuation, label_momentum, label_quality (0/1)
```
---

## 10. Data Split and Leakage Policy

### 10.1 Split Strategies

**Priority Order:**

1. **Time-Series Split** (preferred):
   ```python
   df_sorted = df.sort_values('last_updated')
   split_idx = int(len(df) * TRAIN_SIZE)
   train_df, test_df = df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]
   ```

2. **Grouped Split** (prevent ticker leakage):
   ```python
   from sklearn.model_selection import GroupShuffleSplit
   gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
   train_idx, test_idx = next(gss.split(X, y, groups=df['ticker']))
   ```

3. **Stratified Split** (maintain sector balance):
   ```python
   from sklearn.model_selection import train_test_split
   train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df['sector'])
   ```

### 10.2 Cross-Validation

**Automated CV Selection:**
```python
from finance_ml.ml_workflow.classification.models import determine_cv_strategy

cv_strategy, cv_obj = determine_cv_strategy(
    df, target=y, n_splits=CV_FOLDS,
    date_column='snapshot_date',
    group_column='ticker',
)
# Returns: 'TimeSeriesSplit', 'GroupKFold', 'StratifiedKFold', or 'KFold'
```

### 10.3 Leakage Prevention

- **No future information**: Features use only data available at prediction time
- **No target leakage**: Features cannot derive from target
- **No test data**: Scalers/encoders fit only on train set
- **No group mixing**: Same ticker not in both train and validation

### 10.4 Feature Alignment
```python
from finance_ml.ml_workflow.regression.dataset import predict_with_model

# Safe prediction with automatic alignment
predictions = predict_with_model(model, X_test, fill_missing=0.0)
```
---

## 11. Standardized Predictions Schema

### 11.1 Required Columns
```python
REQUIRED_COLUMNS = [
    'ticker', 'isin', 'sector', 'region',
    'last_price', 'y_true', 'y_pred', 'y_pred_calibrated',
    'pred_p10', 'pred_p50', 'pred_p90',
    'interval_width', 'abs_error', 'pct_error',
    'model_version', 'snapshot_date',
]
```
### 11.2 Column Definitions

| Column              | Description                        |
|---------------------|------------------------------------|
| `y_true`            | Actual target value                |
| `y_pred`            | Raw model prediction               |
| `y_pred_calibrated` | Sector-bias-corrected prediction   |
| `pred_p10/p50/p90`  | Quantile predictions               |
| `interval_width`    | `pred_p90 - pred_p10`              |
| `abs_error`         | `abs(y_pred - y_true)`             |
| `pct_error`         | `100 * (y_pred - y_true) / y_true` |

### 11.3 Invariants

- **Monotonicity**: `pred_p10 ≤ pred_p50 ≤ pred_p90`
- **Non-negativity**: All price predictions ≥ 0
- **Coverage**: 80% of actuals within `[pred_p10, pred_p90]`

### 11.4 Validation

```python
from finance_ml.ml_workflow.regression.io import (
    build_predictions_frame,
    validate_predictions_schema,
)

predictions_df = build_predictions_frame(
    df, y_true, y_pred, quantile_preds={0.1: p10, 0.5: p50, 0.9: p90}
)
validate_predictions_schema(predictions_df)  # Raises if invalid
```
---

## 12. Sector Metrics and Calibration

### 12.1 Sector-Level Metrics
```python
from finance_ml.ml_workflow.evaluation.metrics import calculate_sector_metrics

sector_metrics = calculate_sector_metrics(df, 'y_true', 'y_pred', 'sector')
# Returns DataFrame: sector, mae, rmse, r2, mape, bias, count
```
### 12.2 Sector Bias Calibration

**Additive Correction:**
```python
sector_bias = val_df.groupby('sector').apply(
    lambda x: (x['y_pred'] - x['y_true']).mean()
)
df['y_pred_calibrated'] = df.apply(
    lambda row: row['y_pred'] - sector_bias.get(row['sector'], 0), axis=1
)
```

**Isotonic Regression:**
```python
from sklearn.isotonic import IsotonicRegression

isotonic_models = {}
for sector in sectors:
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_df[val_df['sector'] == sector]['y_pred'],
            val_df[val_df['sector'] == sector]['y_true'])
    isotonic_models[sector] = iso
```
---

## 13. Outlier Safety Rails Policy

### 13.1 Winsorization

> **Canonical Reference**: This section is the single source of truth for winsorization policy.

**Two Strategies:**

| Strategy                   | Bounds    | Use Case                        |
|----------------------------|-----------|---------------------------------|
| **Conservative** (Default) | 0.10/0.90 | Production, preserves mega-caps |
| **Aggressive**             | 0.01/0.99 | Exploratory, high-noise data    |

**Implementation (uses Schema Section 5):**
```python
from finance_ml.ml_workflow.preprocessing.outliers import winsorize_by_sector
from finance_ml.core.schema import list_price_cols

# Get protected columns from schema
protected = list_price_cols()

df_winsorized = winsorize_by_sector(
    df,
    columns=numeric_cols,
    lower=WINSORIZE_LOWER,  # 0.10 (Section 2.1)
    upper=WINSORIZE_UPPER,  # 0.90
    exclude_columns=protected,
)
```

### 13.2 Log Transforms (Alternative)

> **Reference**: Use log transforms instead of winsorization for skewed market values.
```python
from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms

df = apply_log_transforms(df, method='signed_log')
# Creates: log_market_cap, log_revenue, log_total_assets, etc.
```
### 13.3 Non-Negativity Constraints
```python
df['y_pred'] = df['y_pred'].clip(lower=0)
df['pred_p10'] = df['pred_p10'].clip(lower=0)
df['pred_p50'] = df['pred_p50'].clip(lower=0)
df['pred_p90'] = df['pred_p90'].clip(lower=0)
```
---

## 14. Uncertainty and Prediction Intervals

### 14.1 Quantile Regression
```python
from sklearn.ensemble import GradientBoostingRegressor

quantile_models = {}
for q in QUANTILES:  # [0.1, 0.5, 0.9]
    model = GradientBoostingRegressor(loss='quantile', alpha=q)
    model.fit(X_train, y_train)
    quantile_models[q] = model
```
### 14.2 Conformal Prediction
```python
# Calibration on validation set
val_residuals = np.abs(val_df['y_true'] - val_df['pred_p50'])
q = np.quantile(val_residuals, 0.8)  # 80% coverage

df['pred_p10_calibrated'] = df['pred_p50'] - q
df['pred_p90_calibrated'] = df['pred_p50'] + q
```
### 14.3 Coverage Diagnostics
```python
coverage = (
    (df['y_true'] >= df['pred_p10']) & 
    (df['y_true'] <= df['pred_p90'])
).mean()
print(f"Interval coverage: {coverage:.1%}")  # Target: 75-85%
```
---

## 15. Jupyter Notebook Guidelines

### 15.1 Required Sections

1. Configuration and Setup
2. Data Loading (Phase 9.1)
3. EDA (Phase 9.2)
4. Feature Engineering (Phase 9.3)
5. Classification (Phase 9.4)
6. Regression (Phase 9.5)
7. Evaluation (Phase 9.6)
8. Analytics (Phase 9.7)
9. Reporting (Phase 9.8)

### 15.2 Cell Organization

- One logical unit per cell
- Markdown documentation for each section
- Clear large outputs before committing
- Use try-except for data loading

### 15.3 Configuration Cell Template
```python
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Use constants from Section 2
from config import (
    TARGET_COL, TEST_SIZE, CV_FOLDS, QUANTILES,
    MIN_SECTOR_SAMPLES, RANDOM_SEED, MODEL_VERSION,
)

np.random.seed(RANDOM_SEED)

# Output directories
OUTPUT_DIR = Path('outputs')
for subdir in ['eda', 'features', 'regression', 'analytics', 'plots']:
    (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

validate_configuration()
```
### 15.4 Import Organization
```python
# Phase 9.1
from finance_ml.etl import run_etl_pipeline, ETLConfig

# Schema utilities (Section 5)
from finance_ml.core.schema import (
    normalize_column_name, list_price_cols, get_expected_dtype
)

# Phase 9.3
from finance_ml.features.api import build_features

# Phase 9.5
from finance_ml.ml_workflow.regression import models, quantile
```
---

## 16. Model Optimization and Performance

### 16.1 Hyperparameter Tuning
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    }
    model = XGBRegressor(**params, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return mean_absolute_error(y_val, model.predict(X_val))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### 16.2 Model Stacking
```python
from sklearn.ensemble import StackingRegressor

base_models = [
    ('xgb', XGBRegressor(n_estimators=500)),
    ('lgb', LGBMRegressor(n_estimators=500)),
    ('cat', CatBoostRegressor(n_estimators=500, verbose=0)),
]

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    cv=CV_FOLDS,
)
```

### 16.3 Performance Thresholds

| Level             | MAE    | R²      |
|-------------------|--------|---------|
| Excellent         | < 20%  | > 0.7   |
| Good              | 20-40% | 0.5-0.7 |
| Acceptable        | 40-60% | 0.3-0.5 |
| Needs Improvement | > 60%  | < 0.3   |

---

## 17. Style Guides for Visual Elements

### 17.1 Plot Formatting

**Theme**: Dark Mode (`template="plotly_dark"`)

**Color Palette:**

| Name    | Hex       | Use            |
|---------|-----------|----------------|
| Primary | `#375a7f` | Blue/Primary   |
| Success | `#00bc8c` | Green/Positive |
| Warning | `#f39c12` | Orange/Warning |
| Danger  | `#e74c3c` | Red/Negative   |
| Info    | `#3498db` | Light Blue     |
| Neutral | `#adb5bd` | Gray           |

### 17.2 Plotly Configuration
```python
import plotly.express as px

PLOTLY_TEMPLATE = 'plotly_dark'
px.defaults.template = PLOTLY_TEMPLATE

fig = px.scatter(
    df, x='last_price', y='price_target', color='sector',
    hover_data=['ticker', 'sector', 'region'],
    labels={'last_price': 'Last Price ($)', 'price_target': 'Price Target ($)'},
)
fig.update_layout(font=dict(family='Arial', size=14))
fig.write_html('outputs/plots/scatter.html')
```

### 17.3 Tables

- Headers: Bold, sentence case
- Currency: `$1,234.56`
- Percentages: `12.34%`
- Decimals: 2-4 places
- Pagination for > 20 rows

---

## 18. Portfolio Optimization Workflow

### 18.1 7-Phase Architecture

| Phase | Description                | Module                 |
|-------|----------------------------|------------------------|
| 1     | Enhanced Stock Selection   | `stock_selection.py`   |
| 2     | ML-Based Return Prediction | `ml_returns.py`        |
| 3     | Advanced Optimization      | `portfolio.py`         |
| 4     | Risk Management            | `risk.py`              |
| 5     | Backtesting Framework      | `portfolio.py`         |
| 6     | Interactive Dashboards     | `portfolio_widgets.py` |
| 7     | Enhanced ML & Validation   | `ml_returns.py`        |

### 18.2 Expected Return Bounds

> **Critical Policy**: Bound expected returns to prevent unrealistic optimization.
```python
from finance_ml.ml_workflow.analytics import clip_expected_returns

MAX_EXPECTED_RETURN = 0.29   # 29% cap
MIN_EXPECTED_RETURN = -0.50  # -50% floor

expected_returns = clip_expected_returns(raw_returns)
assert expected_returns.mean() < 0.30
```
### 18.3 Price Column Integration

> **Reference**: Use `list_price_cols()` from Section 5.6.
```python
from finance_ml.core.schema import list_price_cols
from finance_ml.ml_workflow.analytics import calculate_historical_returns

price_cols = list_price_cols()
df_returns = calculate_historical_returns(df, current_price_col='last_price')
# Creates: return_1w, return_1m, return_3m, return_6m, return_1y
```

### 18.4 Black-Litterman Integration
```python
from finance_ml.ml_workflow.analytics import (
    create_bl_views_from_ml,
    optimize_black_litterman,
)

views, confidences = create_bl_views_from_ml(ml_predictions, tickers=ticker_list)
result = optimize_black_litterman(
    returns=expected_returns,
    cov_matrix=cov_matrix,
    views=views,
    view_confidences=confidences,
)
```
---

## 19. Data Quality Validation Checkpoints

### 19.1 Post-ETL Validation
```python
# Required assertions
assert not df.empty, "DataFrame must not be empty"
assert 'ticker' in df.columns
assert 'sector' in df.columns
assert 'last_price' in df.columns

# Validate against schema
from finance_ml.core.schema import list_required_schema_columns_for_etl
required = list_required_schema_columns_for_etl()
missing = [c for c in required if c not in df.columns]
assert not missing, f"Missing required columns: {missing}"

# Quality metrics
assert df.isna().sum().sum() == 0, "No missing values after imputation"
assert df['last_price'].min() > 0, "Prices must be positive"

print("✓ ETL validation passed")
```
### 19.2 Post-Feature Engineering Validation
```python
from finance_ml.core.schema import PHASE93_FEATURE_CATEGORIES

total_expected = sum(len(c) for c in PHASE93_FEATURE_CATEGORIES.values())
present = sum(1 for c in df.columns if c in 
              [col for cats in PHASE93_FEATURE_CATEGORIES.values() for col in cats])
coverage = present / total_expected * 100

assert coverage >= 90, f"Phase 9.3 coverage: {coverage:.1f}% (minimum 90%)"
print(f"✓ Feature validation passed: {coverage:.1f}% coverage")
```
### 19.3 Pre-Modeling Validation
```python
# No infinity values
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
assert inf_count == 0, f"Infinity values: {inf_count}"

# Target validation
assert df[TARGET_COL].notna().all()
assert df[TARGET_COL].min() > 0
```
### 19.4 Post-Prediction Validation
```python
from finance_ml.ml_workflow.regression.io import validate_predictions_schema

validate_predictions_schema(predictions_df)

# Monotonicity check
if all(c in predictions_df.columns for c in ['pred_p10', 'pred_p50', 'pred_p90']):
    violations = (
        (predictions_df['pred_p10'] > predictions_df['pred_p50']) |
        (predictions_df['pred_p50'] > predictions_df['pred_p90'])
    ).sum()
    assert violations == 0, f"Monotonicity violated: {violations} rows"
```
---

## 20. Output Artifact Standards

### 20.1 Directory Structure
```
outputs/
├── preprocessing/
│   ├── etl_metrics.json
│   ├── dtype_diagnostics.json
│   └── imputation_summary.json
├── eda/
│   ├── financial_metrics/
│   └── phase93_feature_categories/
├── regression/
│   ├── regression_predictions_detailed.csv
│   ├── regression_metrics_by_sector.csv
│   └── sector_models/
├── evaluation/
│   ├── calibration_report.json
│   └── quantile_diagnostics.csv
├── analytics/
│   ├── mispricing_scores.csv
│   └── portfolio_weights.csv
├── governance/
│   ├── model_card.json
│   └── lineage.json
└── plots/
    └── *.html, *.png
```

### 20.2 JSON Artifact Format

**ETL Metrics:**
```json
{
  "rows_initial": 5234,
  "rows_after_etl": 5234,
  "missing_before_imputation": 12045,
  "missing_after_imputation": 0,
  "schema_alignment_score": 0.98,
  "timestamp": "2026-01-07T10:30:00",
  "model_version": "v9_11"
}
```

**Model Card:**
```json
{
  "model_version": "v9_11",
  "model_type": "Stacking Ensemble",
  "training_date": "2026-01-07",
  "metrics": {"mae": 8.45, "rmse": 12.32, "r2": 0.78},
  "features_used": 600,
  "phase93_coverage": 92.8
}
```

### 20.3 CSV Standards

**Predictions Output:**

- Follow schema from Section 11
- Include all required columns
- Reset index before saving

**Sector Metrics:**

- Columns: `sector`, `mae`, `rmse`, `r2`, `mape`, `bias`, `count`, `model_version`, `timestamp`

---

## Appendix: Migration Guide

### From v1.x to v2.0

1. **Update imports** to use `finance_ml.core.schema` (Section 4.3)
2. **Remove deprecated function calls** (Section 4.4)
3. **Use new ETL module** instead of `etl_with_financial_metrics()` (Section 7.5)
4. **Align DataFrame names** with Section 8.2
5. **Validate schema alignment** using Section 5.9

### Deprecated → Current Mapping

| Deprecated                                   | Current                       |
|----------------------------------------------|-------------------------------|
| `finance_ml.ml_workflow.data.schema`         | `finance_ml.core.schema`      |
| `etl_with_financial_metrics()`               | `run_etl_pipeline()`          |
| `apply_enhanced_imputation_strategy_6step()` | `apply_imputation_pipeline()` |
| `all_stocks_imputed`                         | `all_stocks_preprocessed`     |

---

*Document Version: 2.0.0 | Last Updated: 2026-01-07*