# Finance ML Analytics Platform — Code Guidelines

**Version:** 1.11  
**Last Updated:** 2025-12-14  
**Package Version:** 0.9.4  
**Model Version:** v9_9

These guidelines codify conventions for the Finance ML Analytics Platform, covering technology stack, configuration,
architecture, function signatures, column naming, and best practices. They align with the project's 8-phase ML
workflow (Phase 9.1-9.8) and 7-phase Portfolio Optimization workflow.

**Related Documentation:**

- **[ML Workflow Guidelines](ml_workflow_guidelines.md)**: Comprehensive guidelines for the 8-phase ML workflow with
  acceptance criteria, success metrics, and validation checkpoints for each phase. Includes critical issues analysis
  and recommended fixes.

**Recent Updates (v1.11):**

- **Schema Alignment Validation Enhancement** (2025-12-14)
    - **Section 5.3**: Updated COLUMN_SCHEMA to 503 columns (up from 447)
        - Added 48 log-transformed columns (log_gross_profit_previous_year, log_operating_income_fq, etc.)
        - Added 4 Phase 9.3 composite scores (altman_z_score, beneish_m_score, composite_quality_score, momentum_score)
        - Updated schema structure breakdown with detailed categorization
    - **Section 5.3.4**: NEW Schema Alignment Validation subsection
        - Automated validation in ETL Stage 11 (when validate_quality=True)
        - Four validation checks: unknown columns, missing expected columns, dtype mismatches, alignment score
        - ETLMetrics enhancements: schema_alignment_score, unknown_columns_count, missing_expected_columns_count,
          dtype_mismatches_count
        - Warning triggers: alignment < 95% or unknown columns > 10
        - Integration with dtype_diagnostics.json for comprehensive schema monitoring
    - **normalize_column_name()**: Enhanced with special R&D handling (R&D → randd, not r_and_d)
    - **ETL Pipeline**: Added _validate_schema_alignment() method in etl.py
    - **Business Impact**: Real-time schema drift detection and data quality monitoring

- **Alignment with ml_workflow_guidelines.md v1.1** (2025-12-14)
    - **Section 6**: Fixed section cross-references (Data Split Policy → Section 10, Predictions Schema → Section 11)
    - **Section 8.2**: DataFrame Stage Naming now aligned with ml_workflow_guidelines.md Appendix (6-stage pipeline)
    - **Section 16**: Removed outdated footer with stale version information
    - **Configuration Constants**: Verified alignment with ml_workflow_guidelines.md "Single Source of Truth" section
    - **Test Coverage**: Updated test module references to match ml_workflow_guidelines.md requirements table
    - **Document Cleanup**: Consolidated redundant version metadata, removed conflicting information

**Previous Updates (v1.10):**

- **NEW:** Unified ETL Pipeline with Semantic Transformations and Feature Engineering (2025-12-08)
    - **Section 7.5**: Added `etl_with_features()` — Single entry point consolidating schema.py, column_semantics.py,
      and features/api.py functionality
    - **ETLConfig**: New semantic-aware attributes (`use_semantic_column_classification`, `preserve_price_columns`,
      `log_transform_market_values`, `apply_feature_engineering`, `feature_preset`)
    - **ETLMetrics**: New tracking attributes for semantic classification and feature engineering
      (`semantic_classification_applied`, `price_columns_count`, `log_transformed_columns`, `features_added`)
    - **Pipeline Stages**: 9-stage unified pipeline (Extract → Normalize → Dtype Cast → Semantic Classification →
      Imputation → Semantic Transforms → Winsorization → Feature Engineering → Quality Validation)
    - **Migration Guide**: From 7-10 scattered preprocessing cells to single `etl_with_features()` call
    - **Feature Presets**: basic, momentum, quality, standard, comprehensive (196 features)
    - **Test Coverage**: 51 tests in test_etl_unified_pipeline.py validating all new functionality
    - **Business Impact**: Simplified notebook workflow while preserving price column integrity for valuation metrics

**Previous Updates (v1.9):**

- **NEW:** Documentation alignment with implementation audit (2025-12-05)
    - **Section 2.4**: Business-Driven Configuration Rationale — Links technical constants to business objectives
    - **Section 4.4**: Deprecated Modules — Clear migration paths for legacy code
    - **Section 7.5**: ETL Function Signatures — Complete signatures for `etl_with_financial_metrics()` and
      `run_etl_pipeline()`
    - **Section 8.6**: ETL Pipeline Best Practices — Unified entry point patterns and validation checkpoints
    - **Section 9.3**: Phase 9.3 Feature Categories — Documentation of 196 features across 16 categories
    - **Section 17.2**: Expanded Interactive Visualization Guidelines — Plotly configuration and required visualizations
    - **Section 19**: Data Quality Validation Checkpoints — Post-ETL and post-feature-engineering assertions
    - **Section 20**: Output Artifact Standards — Directory structure and required JSON artifacts
    - **Updated Section 4.3**: Fixed imputation strategy references (4-step → 6-step), confirmed evaluation imports
    - **Updated Section 13**: Added winsorization policy note reconciling 0.01/0.99 vs 0.10/0.90 approaches
    - Business objective alignment throughout: Stock price target prediction for portfolio optimization

**Previous Updates (v1.8):**

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
19. [Data Quality Validation Checkpoints](#19-data-quality-validation-checkpoints)
20. [Output Artifact Standards](#20-output-artifact-standards)

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

### 2.4 Business-Driven Configuration Rationale

All configuration constants are designed to support the **primary business objective** stated in `README.md`:

> **Primary Goal**: Predict Stock Price Targets for all stocks in the portfolio to support investment decisions and
> portfolio optimization.

| Constant              | Value             | Business Rationale                                                                    |
|-----------------------|-------------------|---------------------------------------------------------------------------------------|
| `TARGET_COL`          | `'price_target'`  | Core prediction target for investment decisions; directly supports valuation analysis |
| `TARGET_COL_FALLBACK` | `'last_price'`    | Ensures models can train even when analyst targets are unavailable                    |
| `TEST_SIZE`           | `0.2`             | Balance between training data quality (80%) and robust validation (20%)               |
| `CV_FOLDS`            | `5`               | Standard cross-validation setup providing reliable performance estimates              |
| `QUANTILES`           | `[0.1, 0.5, 0.9]` | 80% prediction interval for risk assessment and portfolio construction                |
| `MIN_SECTOR_SAMPLES`  | `20`              | Minimum sample size for statistically meaningful sector-specific models               |
| `MAX_SECTOR_WEIGHT`   | `0.25`            | Portfolio diversification constraint to limit sector concentration risk               |
| `MAX_SINGLE_POSITION` | `0.10`            | Position sizing limit to prevent overexposure to individual securities                |
| `WINSORIZE_LOWER`     | `0.10`            | Conservative outlier handling preserving valid extreme values (high-growth stocks)    |
| `WINSORIZE_UPPER`     | `0.90`            | Conservative outlier handling preserving valid extreme values (mega-cap companies)    |
| `RANDOM_SEED`         | `42`              | Reproducibility for regulatory compliance and model governance                        |
| `MODEL_VERSION`       | `v9_9`            | Version tracking for audit trails and model comparison                                |

**Key Design Principles:**

1. **Prediction Accuracy**: Constants optimize for reliable price target predictions
2. **Risk Management**: Quantile predictions and portfolio constraints support risk-adjusted decision making
3. **Statistical Validity**: Minimum sample sizes ensure sector models are statistically sound
4. **Reproducibility**: Fixed random seeds enable consistent model evaluation and debugging
5. **Regulatory Compliance**: Version tracking and audit trails support governance requirements

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

| Phase | Subpackage        | Entry Point | Description                                           |
|-------|-------------------|-------------|-------------------------------------------------------|
| 9.1   | `preprocessing/`  | `etl.py`    | Unified ETL: loading, imputation, scaling, outliers   |
| 9.2   | `eda/`            | —           | Exploratory analysis, benchmarking, statistical tests |
| 9.3   | `features/`       | `api.py`    | Feature engineering (318 columns, Schema v1.3)        |
| 9.4   | `classification/` | —           | Event classification (13 label methods)               |
| 9.5   | `regression/`     | —           | Regression models, quantile, stacking                 |
| 9.6   | `evaluation/`     | —           | Metrics, uncertainty, calibration, safety rails       |
| 9.7   | `analytics/`      | —           | Mispricing, portfolio optimization, risk metrics      |
| 9.8   | `reporting/`      | —           | Dashboard data, quality alerts, reporting             |

> **Note:** Phase 9.1 uses `etl.py` as the unified entry point that consolidates data loading,
> imputation, scaling, semantic transformations, and feature engineering into a single pipeline.
> See Section 7.5 for detailed API documentation.

### 4.3 Import Patterns

**STANDARD Pattern (Unified ETL Pipeline):**

Use the unified ETL pipeline as the recommended entry point for Phase 9.1-9.3:

```python
# Phase 9.1-9.3: Unified ETL Pipeline (STANDARD - Use This)
from finance_ml.ml_workflow.preprocessing import (
    # Primary entry points
    etl_with_features,  # Complete ETL + feature engineering (recommended)
    etl_with_financial_metrics,  # ETL + financial metrics (for analysis workflows)
    run_etl_pipeline,  # Low-level ETL with custom config
    # Configuration classes
    ETLConfig,  # Pipeline configuration
    ETLMetrics,  # Pipeline metrics tracking
    )

# Example usage:
from pathlib import Path

df, metrics = etl_with_features(
        source='csv',
        data_dir=Path('data'),
        feature_preset='comprehensive',
        return_metrics=True,
        )
print(metrics.summary())
```

**OPTIONAL Pattern (Module-level imports for fine-grained control):**

Use these imports when you need direct access to individual functions:

```python
# Phase 9.1: Preprocessing - Imputation (OPTIONAL)
from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_enhanced_imputation_strategy_6step,
    validate_imputation_completeness,
    apply_zero_imputation,
    apply_knn_imputation_enhanced,
    apply_price_imputation,
    apply_median_imputation,
    apply_categorical_imputation,
    apply_datetime_imputation,
)

# Phase 9.1: Preprocessing - Other utilities (OPTIONAL)
from finance_ml.ml_workflow.preprocessing import (
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

# Phase 9.2: EDA - Phase 9.3 Categories (OPTIONAL)
from finance_ml.ml_workflow.eda.phase93_categories import (
    PHASE93_FEATURE_CATEGORIES,
    categorize_dataframe_columns,
    get_phase93_coverage_stats,
)

# Phase 9.3: Features - Individual builders (OPTIONAL)
from finance_ml.ml_workflow.features import (
    build_valuation_features,
    build_momentum_features,
    build_quality_features,
    select_features_rf,
)

# Phase 9.3: Features - Unified API (OPTIONAL when using etl_with_features)
from finance_ml.ml_workflow.features.api import (
    build_features,  # Unified entry point
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

# Phase 9.6: Evaluation (Complete Import List)
from finance_ml.ml_workflow.evaluation import (
    # Metrics functions
    comprehensive_regression_metrics,
    compute_metrics_by_segment,
    compute_sector_region_metrics,
    # Analysis functions
    residual_analysis,
    error_analysis,
    model_diagnostics,
    prediction_intervals,
    cross_validation_analysis,
    # Phase 9.4 - Uncertainty Quantification
    build_quantile_diagnostics,
    plot_interval_coverage,
    plot_reliability_diagram,
    # Phase 9.5 - Safety Rails
    summarize_winsorization_effects,
    track_constraint_violations,
    safety_rails_sensitivity_app,
    # Phase 9.6 - Data Splits & Leakage
    compute_fold_overlap,
    summarize_grouped_cv_balance,
    time_leakage_checks,
    # Phase 9.7 - Sector Bias Calibration
    estimate_sector_bias,
    plot_metrics_by_sector_time,
    create_sector_bias_dashboard,
    # Phase 9.8 - Stacking & Governance
    compute_stacking_contributions,
    meta_error_maps,
    generate_model_card,
    build_lineage_json,
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

> **Important**: Always use `apply_enhanced_imputation_strategy_6step` for imputation. The 4-step version is
> deprecated (see Section 4.4).

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

### 4.4 Deprecated Modules

The following modules exist for backward compatibility but should **NOT** be used in new code. Use the recommended
replacements instead.

| Deprecated Import                               | Status        | Replacement                                | Notes                                          |
|-------------------------------------------------|---------------|--------------------------------------------|------------------------------------------------|
| `finance_ml.ml_workflow.advanced_preprocessing` | ⚠️ Legacy     | `finance_ml.ml_workflow.preprocessing`     | Consolidated into preprocessing module         |
| `finance_ml.ml_workflow.advanced_features`      | ⚠️ Legacy     | `finance_ml.ml_workflow.features`          | Use `features.api.build_features()`            |
| `finance_ml.ml_workflow.advanced_models`        | ⚠️ Legacy     | `finance_ml.ml_workflow.regression`        | Use regression module                          |
| `finance_ml.ml_workflow.financial_metrics_etl`  | ⚠️ Legacy     | `finance_ml.ml_workflow.preprocessing.etl` | Use `etl_with_financial_metrics()`             |
| `apply_enhanced_imputation_strategy_4step`      | ⚠️ Deprecated | `apply_enhanced_imputation_strategy_6step` | 4-step incomplete for categorical/date columns |

**Migration Guide:**

```python
# OLD (deprecated):
from finance_ml.ml_workflow.advanced_preprocessing import preprocess_data
from finance_ml.ml_workflow.advanced_features import engineer_features
from finance_ml.ml_workflow.advanced_models import train_model
from finance_ml.ml_workflow.preprocessing.imputation import apply_enhanced_imputation_strategy_4step

# NEW (recommended):
from finance_ml.ml_workflow.preprocessing.etl import etl_with_financial_metrics
from finance_ml.ml_workflow.features.api import build_features
from finance_ml.ml_workflow.regression import train_sector_models
from finance_ml.ml_workflow.preprocessing.imputation import apply_enhanced_imputation_strategy_6step
```

**Breaking Changes:**

- **4-step → 6-step imputation**: The 6-step strategy adds Steps 5 (categorical imputation) and 6 (datetime imputation),
  ensuring **zero missing values** across all data types. The 4-step version only handles numeric columns and is
  incomplete.

**Policy:** All new code must use 6-step imputation. The 4-step function will be removed in v0.10.0.

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

**Version:** 1.11 (Updated 2025-12-14)  
**Total Columns:** 503 (up from 447)

The authoritative column schema is defined in `finance_ml/ml_workflow/data/schema.py`:

**Schema Structure (v1.11):**

- 299 source columns (from CSV/SQL schema)
- 61 log-transformed columns (ETL-generated, log1p of market values)
- 43 legacy aliases (role=auxiliary, for backward compatibility)
- 36 generic base columns (no time suffix)
- 34 conditional metrics (with _applicable flags)
- 26 derived ratios and percentage metrics (ETL semantic transforms)
- 4 Phase 9.3 composite quality scores (altman_z_score, beneish_m_score, composite_quality_score, momentum_score)

```python
from finance_ml.ml_workflow.data.schema import (
   COLUMN_SCHEMA,  # Dict[str, Dict[str, str]] - 503 columns (includes derived ETL columns)
   get_expected_dtype,  # Get dtype for a column
   get_column_role,  # Get role for a column
   list_numeric_feature_cols,  # List all numeric features
   list_categorical_cols,  # List all categorical columns
   list_date_cols,  # List all date columns
   normalize_column_name,  # Normalize a column name (handles R&D → randd special case)
   list_required_schema_columns_for_etl,  # Get canonical ETL-required columns (v0.9.3+)
   )

# Example usage
dtype = get_expected_dtype('last_price')  # Returns 'float'
role = get_column_role('sector')  # Returns 'categorical'
numeric_cols = list_numeric_feature_cols()  # Returns list of numeric feature columns

# ETL-required columns validation (v0.9.3+)
required = list_required_schema_columns_for_etl()  # Core columns: ticker, sector, region, last_price, etc.
required_ext = list_required_schema_columns_for_etl(include_extended_financials=True)  # + ebitda_ltm, etc.
```

#### 5.3.1 ETL-Required Columns (v0.9.3+)

The `list_required_schema_columns_for_etl()` function provides a canonical list of columns required for the unified ETL
pipeline:

**Core Required Columns (12 columns):**

- **Identifiers**: `ticker`, `isin`
- **Group Keys**: `sector`, `region`, `country`, `trading_country`
- **Price/Targets**: `last_price`, `price_target`, `price_target_median`, `price_target_ytd_ago`
- **Market Values**: `market_cap`, `enterprise_value`

**Extended Financials (optional, 6 additional columns):**

- `total_revenues_ltm`, `ebitda_ltm`, `net_income_is_ltm`, `total_assets_ltm`, `total_debt_ltm`, `total_equity_ltm`

**Usage with dtype diagnostics:**

```python
from finance_ml.ml_workflow.preprocessing.dtypes import (
    detect_and_cast_dtypes,
    get_critical_missing_columns,
)

df_cast, diagnostics = detect_and_cast_dtypes(df)
# Check for truly critical missing columns (not just optional features)
critical = get_critical_missing_columns(diagnostics)
if critical:
    raise ValueError(f"Missing required ETL columns: {critical}")
```

#### 5.3.2 Schema Column Roles

Columns in `COLUMN_SCHEMA` are assigned roles that determine their treatment:

| Role              | Description                      | Included in `missing_expected_columns` |
|-------------------|----------------------------------|----------------------------------------|
| `feature`         | ML features used in modeling     | Yes                                    |
| `target`          | Target variables for prediction  | Yes                                    |
| `target_fallback` | Alternative targets              | Yes                                    |
| `id`              | Identifier columns               | Yes                                    |
| `date`            | Date/timestamp columns           | Yes                                    |
| `categorical`     | Categorical grouping columns     | Yes                                    |
| `auxiliary`       | Legacy aliases, optional columns | **No**                                 |

**Note:** Legacy/alias column names (e.g., `price_target_num`, `1_day_pct`, `shrs_out`, `sga_expenses_*`) have been
demoted to `role: "auxiliary"` so they no longer appear in `missing_expected_columns` diagnostics. This keeps
diagnostics focused on truly required columns.

#### 5.3.3 Derived ETL Columns

The schema includes columns created by the ETL semantic transformation stage:

**Log-Transformed Market Values (13 columns):**

- `log_market_cap`, `log_enterprise_value`, `log_revenue`, `log_ebitda`, `log_net_income`
- `log_total_assets`, `log_total_equity`, `log_total_debt`, `log_gross_profit`
- `log_operating_income`, `log_operating_cash_flow`, `log_capex`, `log_cash_and_equivalents`

**Valuation/Profitability Ratios (17 columns):**

- `p_e_ratio`, `p_s_ratio`, `ev_ebitda_ratio`, `ev_sales_ratio`
- `gross_margin_pct`, `operating_margin_pct`, `net_margin_pct`
- `roe`, `roa`, `roic`, `debt_to_equity`, `debt_to_assets`
- `revenue_growth`, `ebitda_growth`, `earnings_growth`
- `target_vs_price`, `target_vs_price_median`, `peg_ratio`, `dividend_yield`

#### 5.3.4 Schema Alignment Validation (v1.11)

**Added:** 2025-12-14

The ETL pipeline now includes automated schema alignment validation to ensure DataFrame columns match the COLUMN_SCHEMA
registry. This validation runs as Stage 11 in the unified ETL pipeline when `validate_quality=True`.

**Validation Method:**

```python
from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline

# Schema validation runs automatically in Stage 11
pipeline = ETLPipeline(config=ETLConfig(validate_quality=True))
df_transformed = pipeline.transform(df_extracted)

# Access validation metrics
print(f"Schema alignment: {pipeline.metrics.schema_alignment_score:.2%}")
print(f"Unknown columns: {pipeline.metrics.unknown_columns_count}")
print(f"Missing expected: {pipeline.metrics.missing_expected_columns_count}")
print(f"Dtype mismatches: {pipeline.metrics.dtype_mismatches_count}")
```

**Validation Checks:**

1. **Unknown Columns**: Columns present in DataFrame but not in COLUMN_SCHEMA
    - Excludes auxiliary/legacy columns (role="auxiliary")
    - Logged as warnings if count > 10

2. **Missing Expected Columns**: Columns in COLUMN_SCHEMA but not in DataFrame
    - Only includes columns with roles: feature, target, target_fallback, id, date, categorical
    - Excludes auxiliary columns to reduce false positives

3. **Dtype Mismatches**: Columns with dtype different from COLUMN_SCHEMA expectation
    - Compares actual dtype against expected dtype from schema
    - Logged with details for manual review

4. **Alignment Score**: Overall schema quality metric [0.0-1.0]
    - Formula: `1.0 - (unknown_count + missing_count + mismatch_count) / total_expected_columns`
    - Warning threshold: < 0.95 (95% alignment)

**ETLMetrics Schema Validation Fields:**

```python
@dataclass
class ETLMetrics:
    # ... existing fields ...
    
    # Schema validation metrics (v1.11)
    schema_alignment_score: float = 1.0  # Schema alignment quality [0.0-1.0]
    unknown_columns_count: int = 0  # Columns in df but not in COLUMN_SCHEMA
    missing_expected_columns_count: int = 0  # Expected columns not in df
    dtype_mismatches_count: int = 0  # Columns with dtype mismatches
```

**Validation Output Example:**

```
Stage 11: Validating schema alignment
Schema Validation: ✓ (alignment: 98.50%, unknown: 3, missing: 2, dtype mismatches: 0)
```

**Warning Triggers:**

- Alignment score < 95%: `"Schema alignment below 95%: 92.3%"`
- Unknown columns > 10: `"Found 15 unknown columns"`

**Integration with dtype_diagnostics.json:**

The schema validation complements dtype diagnostics by providing:

- Real-time validation during ETL execution
- Alignment score for quality monitoring
- Separation of critical vs. auxiliary column gaps

**Best Practices:**

1. **Review unknown columns**: May indicate new data sources or schema drift
2. **Investigate missing expected columns**: May require schema updates or data source fixes
3. **Monitor alignment score trends**: Declining scores indicate schema/data misalignment
4. **Update COLUMN_SCHEMA**: Add new columns with appropriate roles when data sources expand

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

- [ ] ETL pipeline used: `etl_with_financial_metrics()` (Section 8.6)
- [ ] Data source clearly specified: `source='csv'`, `'db'`, or `'all_stocks'`
- [ ] ETL configuration documented: `compute_all_metrics=True`, `output_dir` set
- [ ] ETL metrics returned and validated: `return_metrics=True`
- [ ] Post-ETL validation passed (Section 19.1):
    - [ ] DataFrame not empty: `assert not df.empty`
    - [ ] Critical columns present: `ticker`, `sector`, `last_price`, `price_target`
    - [ ] No missing values after imputation: `df.isna().sum().sum() == 0`
- [ ] 6-step imputation strategy applied via ETL pipeline
- [ ] Data types validated against schema: `validate_dtypes_against_schema()`
- [ ] Outliers handled via ETL: winsorization (0.01/0.99), clipping, non-negativity enforced
- [ ] Critical date columns validated: `last_updated`, `income_statement_report_date`, `next_earnings`,
  `dividend_record_*`

**Feature Engineering:**

- [ ] Features aligned with Phase 9.3 Schema v1.3 (318 columns)
- [ ] Feature preset used or documented: "basic", "momentum", "quality", "comprehensive"
- [ ] No target leakage in feature construction
- [ ] Feature importance analyzed and documented

**Model Training:**

- [ ] Train/test split follows Data Split Policy (Section 10)
- [ ] Cross-validation uses grouped or stratified strategy (no leakage)
- [ ] Hyperparameters documented and versioned
- [ ] Model artifacts saved with version: `MODEL_VERSION`

**Evaluation and Outputs:**

- [ ] Predictions follow Standardized Predictions Schema (Section 11)
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
        df,
        sector_column="sector",
        price_column="last_price",
        n_neighbors=5,
        return_stats=True,
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
        df,
        zero_fill_columns=None,
        knn_neighbors=5,
        price_columns=None,
)
# Returns: DataFrame with all missing values imputed
```

**Phase 9.3 — Features**

```python
from finance_ml.ml_workflow.features.api import build_features

features_df = build_features(
        df,
        preset="comprehensive",
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
        df,
        method="price_momentum",
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
        X_train,
        y_train,
        X_test,
        y_test,
        quantiles=[0.1, 0.5, 0.9],
)
# Returns: {"model", "metrics", "quantile_predictions": {q: pred_array}}

# Phase 9.5 Feature Alignment (Task 7 - High Priority)
from finance_ml.ml_workflow.regression.dataset import (
    align_features_to_model,
    predict_with_model
)

X_test_aligned = align_features_to_model(
    X_test,
    model,
    fill_value=0.0,
)
# Returns: DataFrame with columns aligned to model.feature_names_in_
# Adds missing features (filled with fill_value), drops extra features
# Preserves original column order where possible

predictions = predict_with_model(
    model,
    X_test,
    fill_missing=0.0,
)
# Returns: np.ndarray of predictions
# Wraps align_features_to_model + model.predict for safe inference

# Phase 9.5 Stacking Hyperparameter Tuning (Task 6 - Low Priority)
from finance_ml.ml_workflow.regression.models import (
    tune_stacking_hyperparameters,
    select_stacking_base_models,
    select_meta_learner
)

best_config = tune_stacking_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    timeout: Optional[int] = 1800,
    cv_folds: int = 3,
    random_state: int = 42
)
# Returns: dict with "base_models", "meta_learner", "best_score", "study"
# Uses Optuna for Bayesian hyperparameter optimization

base_models = select_stacking_base_models(trial: optuna.Trial)
# Returns: list of (name, estimator) tuples for StackingRegressor
# Optimizes: XGBoost (n_estimators, max_depth, learning_rate, subsample),
#            LightGBM (n_estimators, num_leaves, learning_rate),
#            Ridge (alpha), Lasso (alpha)

meta_learner = select_meta_learner(trial: optuna.Trial)
# Returns: meta-learner estimator (Ridge or HuberRegressor)
# Optimizes: Ridge (alpha) or Huber (epsilon, alpha)
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

### 7.5 ETL Pipeline Functions

The ETL module provides unified entry points for complete Extract-Transform-Load workflows with financial metrics
computation.

**Primary Function: `etl_with_financial_metrics()`**

```python
from finance_ml.ml_workflow.preprocessing.etl import etl_with_financial_metrics

def etl_with_financial_metrics(
    source: Literal["csv", "db", "all_stocks"],
    data_dir: Optional[Path | str] = None,
    db_url: Optional[str] = None,
    compute_all_metrics: bool = True,
    output_dir: Optional[Path] = None,
    return_metrics: bool = True,
) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
    """
    Complete ETL pipeline with financial metrics computation.
    
    Unified entry point for: Extract → Transform (with 6-step imputation + 
    scaling + financial metrics) → Load
    
    Args:
        source: Data source type
            - 'csv': Load from regional CSV files
            - 'db': Load from PostgreSQL equities table
            - 'all_stocks': Load from unified all_stocks table (recommended)
        data_dir: Directory for CSV files (required if source='csv')
        db_url: Database URL (required if source='db' or 'all_stocks')
        compute_all_metrics: Enable all financial metrics computation (default: True)
            When True, computes:
            - Valuation metrics (P/E, P/S, EV/EBITDA, EV/Sales)
            - Profitability metrics (margins, ROE, ROA)
            - Growth metrics (revenue, EBITDA, earnings growth)
            - Leverage metrics (debt ratios)
            - Target vs price metrics
            - Sector-specific ratios
        output_dir: Optional directory for quality alerts and dashboard JSON files
        return_metrics: Return ETLMetrics for monitoring (default: True)
    
    Returns:
        (DataFrame, ETLMetrics) tuple if return_metrics=True
        DataFrame only if return_metrics=False
    """
```

**Usage Examples:**

```python
# Basic usage with all metrics
from finance_ml.ml_workflow.preprocessing.etl import etl_with_financial_metrics

df, metrics = etl_with_financial_metrics(
    source='csv',
    data_dir='data/',
    compute_all_metrics=True,
    output_dir='outputs/eda/financial_metrics',
    return_metrics=True
)

print(metrics.summary())
print(f"Valuation metrics added: {metrics.valuation_metrics_added}")

# Database source
df, metrics = etl_with_financial_metrics(
    source='all_stocks',
    db_url='postgresql+psycopg2://postgres:@localhost:5432/postgres',
    compute_all_metrics=True,
    return_metrics=True
)
```

**Advanced Function: `run_etl_pipeline()`**

The ETL pipeline now uses stage-aligned configuration dataclasses to match the 11-stage workflow (Extract → Normalize →
Dtype Cast → Semantic Classification → Imputation → Semantic Transforms → Winsorization → Scaling → Feature
Engineering → Post-Feature Imputation → Schema Validation).

```python
from finance_ml.ml_workflow.preprocessing.etl import (
    DataExtractionConfig,
    DataSanitizationConfig,
    DtypeCastingConfig,
    ETLConfig,
    FeatureEngineeringConfig,
    FeatureSelectionConfig,
    FinancialMetricsConfig,
    ImputationConfig,
    ScalingConfig,
    SchemaValidationConfig,
    SemanticClassificationConfig,
    SemanticTransformConfig,
    etl_with_features,
)

etl_config = ETLConfig(
    extraction=DataExtractionConfig(normalize_column_names=True),
    validation=SchemaValidationConfig(
        validate_schema=True,
        require_target_column=False,
        drop_rows_with_missing_critical_fields=True,
        validate_schema_alignment=True,
        schema_alignment_threshold=0.95,
    ),
    dtype_casting=DtypeCastingConfig(apply_dtype_casting=True, track_diagnostics=True),
    semantic_classification=SemanticClassificationConfig(enabled=True, preserve_price_columns=True),
    imputation=ImputationConfig(
        apply_imputation=True,
        strategy="6step",
        knn_neighbors=5,
        sector_column="sector",
        reference_price_column="last_price",
        impute_categorical_columns=True,
        impute_datetime_columns=True,
    ),
    semantic_transform=SemanticTransformConfig(
        apply_log_transforms=True,
        log_transform_method="log1p",
        log_transform_market_values=True,
        exclude_ratios_from_winsorization=True,
        exclude_percentages_from_winsorization=True,
        exclude_counts_from_scaling=False,
    ),
    sanitization=DataSanitizationConfig(
        sanitize_data=True,
        apply_winsorization=False,
        winsorize_lower_percentile=0.10,
        winsorize_upper_percentile=0.90,
    ),
    scaling=ScalingConfig(
        enabled=False,
        scaler_type="robust",
        scale_by_sector=True,
        exclude_price_columns=True,
    ),
    feature_engineering=FeatureEngineeringConfig(enabled=True, preset="comprehensive"),
    feature_selection=FeatureSelectionConfig(
        enabled=False,
        method="mutual_info",
        min_importance_threshold=0.01,
        max_correlation_threshold=0.95,
    ),
    financial_metrics=FinancialMetricsConfig(
        compute_valuation_metrics=True,
        compute_profitability_metrics=True,
        compute_growth_metrics=True,
        compute_leverage_metrics=True,
        compute_target_vs_price_metrics=True,
        compute_sector_specific_metrics=False,
    ),
)

df, metrics = etl_with_features(
    source="csv",
    data_dir="data/",
    feature_preset="comprehensive",
    config=etl_config,
    return_metrics=True,
)
```

**ETLConfig Stage Components (v1.11):**

- `DataExtractionConfig`: column normalization, row limits
- `SchemaValidationConfig`: schema/critical column checks, alignment threshold
- `DtypeCastingConfig`: schema-aware dtype casting and diagnostics
- `SemanticClassificationConfig`: price preservation and column categories
- `ImputationConfig`: 6-step imputation (sector-aware KNN, price-based, categorical, datetime)
- `SemanticTransformConfig`: log transforms and semantic exclusions for winsorization/scaling
- `DataSanitizationConfig`: sanitization and optional winsorization bounds
- `ScalingConfig`: scaler selection, sector-aware scaling, price exclusion
- `FeatureEngineeringConfig`: Phase 9.3 presets/categories
- `FeatureSelectionConfig`: mutual information/correlation thresholds
- `FinancialMetricsConfig`: valuation/profitability/growth/leverage/target-vs-price metrics

**Migration Note:**

The two-step approach is deprecated:

```python
# OLD (deprecated):
# df = run_etl_pipeline(source='csv', data_dir='data/')
# df, metrics = run_financial_metrics_etl(df, output_dir=output_dir)

# NEW (recommended):
df, metrics = etl_with_financial_metrics(
    source='csv',
    data_dir='data/',
    output_dir=output_dir
)
```

**Unified ETL with Feature Engineering: `etl_with_features()`** *(NEW in v1.10)*

The `etl_with_features()` function provides a single entry point that consolidates schema.py, column_semantics.py,
and features/api.py functionality into one unified call. This is the **recommended entry point** for complete
ETL with semantic-aware transformations and Phase 9.3 feature engineering.

```python
from finance_ml.ml_workflow.preprocessing.etl import etl_with_features, ETLConfig


def etl_with_features(
        source: Literal["csv", "db", "all_stocks"],
        data_dir: Optional[Path | str] = None,
        db_url: Optional[str] = None,
        feature_preset: str = "standard",
        feature_categories: Optional[List[str]] = None,
        config: Optional[ETLConfig] = None,
        return_metrics: bool = True,
        ) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
    """
    Complete ETL pipeline with integrated feature engineering.
    
    Consolidates schema.py, column_semantics.py, and api.py functionality
    into a single entry point (Section 8.6, Section 9.3).
    
    Pipeline Stages:
    1. Extract from source (CSV or database)
    2. Column normalization and dtype casting
    3. Semantic column classification (price, market_value, ratio, percentage, count)
    4. 6-step imputation strategy
    5. Semantic-aware transformations (log-transforms for market values)
    6. Winsorization (excluding price/ratio/percentage columns)
    7. Feature engineering (Phase 9.3 features via build_features API)
    8. Financial metrics computation
    9. Quality validation
    
    Args:
        source: Data source ('csv', 'db', 'all_stocks')
        data_dir: Directory for CSV files
        db_url: Database connection URL
        feature_preset: Feature engineering preset
            - 'basic': Core ratios, margins, volatility, revenue CAGR
            - 'momentum': Momentum & technical indicators only
            - 'quality': Accounting quality and financial distress signals
            - 'standard': Balanced feature set (default)
            - 'comprehensive': Full advanced feature set (196 features)
        feature_categories: Specific feature categories to engineer (optional)
        config: Optional ETLConfig override
        return_metrics: Whether to return ETLMetrics
    
    Returns:
        DataFrame with all features, optionally with ETLMetrics
    """
```

**Usage Examples:**

```python
# Recommended: Complete ETL with comprehensive features
from finance_ml.ml_workflow.preprocessing.etl import etl_with_features

df, metrics = etl_with_features(
        source='csv',
        data_dir='data/',
        feature_preset='comprehensive',
        return_metrics=True
        )

print(metrics.summary())
# ETL Pipeline Summary:
#   Source: csv
#   Duration: 2.34s
#   Semantic Classification: ✓ (Price Columns: 21, Market Value: 19, Log-Transformed: 19)
#   Feature Engineering: comprehensive (196 features added)

# With custom ETLConfig for fine-grained control
from finance_ml.ml_workflow.preprocessing.etl import etl_with_features, ETLConfig

config = ETLConfig(
        use_semantic_column_classification=True,
        preserve_price_columns=True,  # CRITICAL: Never transform price columns
        log_transform_market_values=True,
        apply_feature_engineering=True,
        feature_preset="comprehensive",
        )

df, metrics = etl_with_features(
        source='csv',
        data_dir='data/',
        config=config,
        return_metrics=True
        )

# Verify price columns were protected
assert metrics.price_columns_count > 0, "Price columns should be detected"
assert 'last_price' in df.columns, "last_price must be preserved"
```

**ETLConfig Semantic Attributes** *(NEW in v1.10)*:

```python
@dataclass
class ETLConfig:
    # ... existing attributes ...

    # Semantic-aware transformation flags (Section 8.5)
    use_semantic_column_classification: bool = True  # Enable semantic classification
    preserve_price_columns: bool = True  # CRITICAL: Never transform price columns
    log_transform_market_values: bool = True  # Apply log-transforms to skewed columns
    exclude_ratios_from_winsorization: bool = True  # Ratios are pre-normalized
    exclude_percentages_from_winsorization: bool = True  # Percentages are bounded
    exclude_counts_from_scaling: bool = False  # Optionally exclude discrete counts

    # Feature engineering integration (Section 9.3)
    apply_feature_engineering: bool = False  # Default OFF for backward compatibility
    feature_preset: str = "standard"  # Options: "basic", "momentum", "quality", "comprehensive"
    feature_categories: Optional[List[str]] = None  # Specific categories to engineer
```

**ETLMetrics Semantic Tracking** *(NEW in v1.10)*:

```python
@dataclass
class ETLMetrics:
    # ... existing attributes ...
    
    # Semantic transformation metrics
    semantic_classification_applied: bool = False
    price_columns_count: int = 0  # Number of protected price columns (21 total)
    market_value_columns_count: int = 0  # Market value columns (19 total)
    ratio_columns_count: int = 0  # Pre-normalized ratio columns
    percentage_columns_count: int = 0  # Bounded percentage columns
    count_columns_count: int = 0  # Discrete count columns
    log_transformed_columns: int = 0  # Columns with log-transforms applied
    
    # Feature engineering metrics
    feature_engineering_applied: bool = False
    feature_preset_used: str = ""
    features_added: int = 0
    feature_categories_applied: List[str] = field(default_factory=list)
```

**Migration from Multiple Cells to Single Entry Point:**

```python
# OLD (7-10 cells):
all_stocks = load_from_csv(data_dir)
all_stocks.columns = normalize_columns(all_stocks.columns)
all_stocks = detect_and_cast_dtypes(all_stocks)
all_stocks = apply_enhanced_imputation_strategy_6step(all_stocks)
all_stocks = winsorize_by_sector(all_stocks)
all_stocks = scale_features(all_stocks)
all_stocks = build_features(all_stocks, preset='comprehensive')
# ... more steps ...

# NEW (1 cell - recommended):
from finance_ml.ml_workflow.preprocessing import etl_with_features

df, metrics = etl_with_features(
        source='csv',
        data_dir='data/',
        feature_preset='comprehensive',
        return_metrics=True
        )

# Verify pipeline completed successfully
assert metrics.semantic_classification_applied, "Semantic classification should be applied"
assert metrics.feature_engineering_applied, "Feature engineering should be applied"
assert metrics.price_columns_count >= 21, "All 21 price columns should be protected"
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

**Required Stage Names** (6-stage ML pipeline):

The pipeline is organized into six stages for the complete ML workflow, with four core stages and two optional
ML-specific stages. This aligns with the unified ETL pipeline (`finance_ml.ml_workflow.preprocessing.etl`):

#### Core Pipeline Stages (Required)

1. **`all_stocks_preprocessed`** — ETL pipeline output: extraction, normalization, validation, sanitization,
   imputation (6-step), optional scaling, and optional financial metrics computation. This consolidates all
   preprocessing steps into a single ETL call using `run_etl_pipeline()`, `etl_with_features()`, or
   `etl_with_financial_metrics()`. Shape: ~(N, 655) columns after ETL.

2. **`all_stocks_features`** — DataFrame enhanced with engineered features (Phase 9.3 feature categories:
   momentum, valuation, profitability, quality/risk, cash flow, growth). Shape: ~(N, 656) columns.

3. **`all_stocks_classification`** — DataFrame enhanced with classification model outputs (event probabilities,
   predicted classes) for use as meta-features in regression. Shape: ~(N, 663) columns.

4. **`all_stocks_enhanced`** — Final Phase 9.5 regression-ready dataset with all transformations including
   classification meta-features and interaction terms. Shape: ~(N, 928) columns.

#### Optional ML-Specific Stages

5. **`all_stocks_selected`** — DataFrame after feature selection (importance/correlation filtering).
   Use when reducing dimensionality for model training. Shape: ~(N, 392) columns (selected features only).

6. **`all_stocks_balanced`** — SMOTE-balanced DataFrame for classification training. Only used for
   class-imbalanced event classification, not for regression. Shape varies based on balancing strategy.

#### Auxiliary DataFrames (Not Pipeline Stages)

These are supporting DataFrames that store specific outputs, not pipeline stages:

- **`all_stocks_multilabel`** — Multi-label target matrix (8 label columns only: label_momentum,
  label_valuation, label_quality, label_profitability, label_growth, label_efficiency, label_cash_flow,
  label_leverage). Used for multi-label classification experiments.

#### Deprecated Stage Names (Do Not Use)

The following stage names were used in legacy implementations but are now handled internally by the ETL pipeline.
**Do not create these as separate DataFrames**:

- ~~`all_stocks_typed`~~ → Handled by ETL Stage 1 (column normalization)
- ~~`all_stocks_winsorized`~~ → Handled by ETL Stage 4 (sanitization) with `apply_winsorization=True`
- ~~`all_stocks_imputed`~~ → Handled by ETL Stage 5 (imputation)
- ~~`all_stocks_scaled`~~ → Handled by ETL Stage 7 (scaling) with `apply_scaling=True`
- ~~`all_stocks_normalized`~~ → Consolidated into `all_stocks_preprocessed`
- ~~`all_stocks_with_classification`~~ → Renamed to `all_stocks_classification` for consistency

**ETL Pipeline Internal Stages** (handled automatically by `run_etl_pipeline()`):

The ETL pipeline internally handles these preprocessing steps in sequence:

- Internal Stage 1: Column normalization (lowercase, underscores)
- Internal Stage 2: Schema validation
- Internal Stage 3: Drop invalid rows (missing ticker, sector, last_price)
- Internal Stage 4: Data sanitization (inf, nan, extremes, winsorization)
- Internal Stage 5: Imputation (6-step: zero, sector-KNN, price, median, categorical, datetime)
- Internal Stage 6: Log transforms (optional, for skewed market values)
- Internal Stage 7: Feature scaling (optional, excludes price columns)
- Internal Stage 8: Financial metrics computation (optional, via `etl_with_financial_metrics()` or ETLConfig flags):
    - Valuation metrics: P/E, P/S, EV/EBITDA, EV/Sales ratios
    - Profitability metrics: gross/operating/net margins, ROE, ROA
    - Growth metrics: revenue, EBITDA, earnings YoY growth
    - Leverage metrics: debt-to-equity, debt-to-assets
    - Target vs price metrics: analyst target upside/downside
    - Sector-specific metrics: P/TBV (financials), R&D intensity (tech/healthcare), Rule of 40 (SaaS)

**Benefits**:

- **Simplified Pipeline**: ETL handles low-level preprocessing; notebook focuses on ML stages
- **Debugging**: Inspect intermediate stages without re-running expensive operations
- **Rollback**: Revert to earlier stage if downstream transformation fails
- **Metrics Tracking**: ETL returns `ETLMetrics` with imputation/scaling statistics
- **Self-documenting**: Stage names clearly indicate transformation history
- **Reduced Memory**: Fewer intermediate DataFrames means lower memory footprint

#### TDD Improvement Tasks

The following tasks should be implemented with Test-Driven Development:

**Task 1: Deprecation Warnings for Legacy Stage Names**

- Add `DeprecationWarning` when `all_stocks_typed`, `all_stocks_winsorized`, `all_stocks_imputed`,
  `all_stocks_scaled`, or `all_stocks_with_classification` are created
- Test: `test_deprecated_stage_names.py` — verify warnings are raised

**Task 2: Pipeline Stage Validator**

- Create `validate_pipeline_stages(globals_dict)` function to check stage naming compliance
- Test: `test_pipeline_stage_validator.py` — verify correct/incorrect stage detection

**Task 3: ETL Metrics for All Internal Stages**

- Extend `ETLMetrics` to track row counts at each internal stage
- Test: `test_etl_internal_stage_metrics.py` — verify metrics accuracy

**Task 4: Notebook Refactoring Validation**

- Create notebook cell that validates all DataFrame stages follow convention
- Test: `test_notebook_stage_compliance.py` — validate notebook follows guidelines

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

✅ **Correct Usage** (Unified ETL with financial metrics - recommended):

```python
from finance_ml.ml_workflow.preprocessing.etl import etl_with_financial_metrics

# Stage 1: Complete ETL + financial metrics in one call
all_stocks_preprocessed, etl_metrics = etl_with_financial_metrics(
    source='csv',
    data_dir='data/',
    compute_all_metrics=True,  # Valuation, profitability, growth, leverage
    output_dir='outputs/financial_metrics',  # Optional: saves quality alerts and dashboard
    return_metrics=True,
)
print(f"✓ Preprocessed with financial metrics: {all_stocks_preprocessed.shape}")
print(f"  Valuation metrics added: {etl_metrics.valuation_metrics_added}")
print(f"  Profitability metrics added: {etl_metrics.profitability_metrics_added}")
print(f"  Growth metrics added: {etl_metrics.growth_metrics_added}")
print(f"  Leverage metrics added: {etl_metrics.leverage_metrics_added}")

# Stage 2: Feature engineering (builds on financial metrics)
all_stocks_features = build_comprehensive_features(all_stocks_preprocessed)
```

✅ **Correct Usage** (Fine-grained control via ETLConfig):

```python
from finance_ml.ml_workflow.preprocessing.etl import run_etl_pipeline, ETLConfig

# Selective financial metrics computation
config = ETLConfig(
    apply_imputation=True,
    imputation_strategy='6step',
    # Financial metrics flags (new in v1.5)
    compute_valuation_metrics=True,
    compute_profitability_metrics=True,
    compute_growth_metrics=False,  # Skip growth metrics
    compute_leverage_metrics=True,
    compute_target_vs_price=True,
    handle_sector_specific_metrics=True,  # P/TBV, R&D intensity, etc.
    generate_quality_alerts=True,
    generate_metrics_dashboard=True,
)
all_stocks_preprocessed, etl_metrics = run_etl_pipeline(
    source='csv', data_dir='data/', config=config, return_metrics=True
)
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
   - `full_time_employees_fq`, `full_time_employees_fy`, `full_time_employees_1fy`, `full_time_employees_2fy`,
     `full_time_employees_3fy`

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

#### 8.5.4 Semantic Classification Enhancements (Phase 9.3 Task 3)

**Overview:**

Enhanced semantic classification reduces "OTHER" category from 487 to 27 columns (93.8% coverage) using pattern-based
and schema fallback methods. Implemented in `column_semantics.py` with 3 new functions.

**Three-Stage Classification Pipeline:**

```python
from finance_ml.ml_workflow.preprocessing.column_semantics import (
    classify_columns,  # Main entry point (3-stage pipeline)
    classify_columns_with_patterns,  # Stage 1: Pattern-based inference
    classify_columns_with_schema_fallback,  # Stage 2: Schema dtype fallback
    )

# Stage 1: Hardcoded sets (PRICE_COLUMNS, MARKET_VALUE_COLUMNS, etc.)
# Stage 2: Pattern-based inference using SUFFIX_PATTERNS
# Stage 3: Schema fallback using COLUMN_SCHEMA dtypes

result = classify_columns(all_columns)
# Returns: {'price': set(...), 'market_value': set(...), 'ratio': set(...), 
#           'percentage': set(...), 'count': set(...), 'other': set(...)}
```

**Pattern-Based Classification (Stage 2):**

SUFFIX_PATTERNS dictionary maps common financial suffixes to semantic categories:

```python
SUFFIX_PATTERNS = {
    'RATIO': ['_ratio', '_yield', '_coverage', '_margin_ratio', '_rate'],
    'PERCENTAGE': ['_margin', '_growth', '_return', '_pct', 'volatility_'],
    'MARKET_VALUE': ['_ltm', '_fy', '_fq', 'revenue_', 'income_', 'ebitda_'],
    'COUNT': ['num_', '_count', '_employees']
}

# Example classifications:
# 'debt_to_equity_ltm' → RATIO (matches '_ltm' pattern)
# 'operating_margin_fy' → PERCENTAGE (matches '_margin' pattern)
# 'total_revenues_fy' → MARKET_VALUE (matches '_fy' pattern)
# 'num_analysts' → COUNT (matches 'num_' pattern)
```

**Schema Fallback (Stage 3):**

For remaining unclassified columns, infer category from COLUMN_SCHEMA dtype:

```python
from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA

# dtype → semantic category mapping:
# 'float64'/'float32' → RATIO (default for unknown numeric)
# 'int64'/'int32' → COUNT
# 'object'/'string' → CATEGORICAL (not tracked in classify_columns)

classifications = classify_columns_with_schema_fallback(unclassified_columns)
```

**Performance Metrics:**

- **Coverage**: 93.8% (410/437 columns classified)
- **OTHER category**: 27 columns (down from 487, 94.4% reduction)
- **Test coverage**: 3 tests in `test_semantic_classification.py` (all passing)

**Integration:**

Semantic classification runs automatically in ETL pipeline Stage 1.6 when
`ETLConfig.use_semantic_column_classification=True` (default).

**References:**

- Implementation: `finance_ml/ml_workflow/preprocessing/column_semantics.py` (lines 50-120)
- Schema: `finance_ml/ml_workflow/data/schema.py` (437 columns)
- Tests: `tests/test_semantic_classification.py` (3 tests, Phase 9.3 Task 3)

### 8.6 Unified ETL Pipeline Best Practices

The ETL pipeline is the **primary entry point** for data processing. Choose the appropriate entry point based on your
workflow needs:

| Function                       | Use Case                                | Features                                 |
|--------------------------------|-----------------------------------------|------------------------------------------|
| `etl_with_features()`          | **ML modeling workflows** (RECOMMENDED) | Semantic transforms + Phase 9.3 features |
| `etl_with_financial_metrics()` | Financial metrics analysis              | Financial ratios + quality alerts        |
| `run_etl_pipeline()`           | Custom configurations                   | Fine-grained control via ETLConfig       |

**Recommended Entry Point for ML Workflows** *(NEW in v1.10)*:

```python
from finance_ml.ml_workflow.preprocessing.etl import (
    etl_with_features,
    ETLConfig,
    ETLMetrics,
    )

# Complete ETL with semantic transforms + feature engineering (RECOMMENDED)
all_stocks_preprocessed, metrics = etl_with_features(
        source='csv',  # or 'db', 'all_stocks'
        data_dir=Path("data"),
        feature_preset='comprehensive',  # 196 Phase 9.3 features
        return_metrics=True
        )

# Inspect ETL metrics
print(metrics.summary())
print(f"Price columns protected: {metrics.price_columns_count}")
print(f"Features added: {metrics.features_added}")
print(f"Log-transformed columns: {metrics.log_transformed_columns}")

# Verify semantic transformations applied
assert metrics.semantic_classification_applied, "Semantic classification should be applied"
assert metrics.price_columns_count >= 21, "All 21 price columns should be protected"
```

**Alternative: Financial Metrics Only:**

```python
from finance_ml.ml_workflow.preprocessing.etl import (
    etl_with_financial_metrics,
    ETLConfig,
    ETLMetrics,
)

# Complete ETL with financial metrics (for analysis workflows)
all_stocks_preprocessed, metrics = etl_with_financial_metrics(
    source='csv',  # or 'db', 'all_stocks'
    data_dir=Path("data"),
    compute_all_metrics=True,
    output_dir=Path("outputs/eda/financial_metrics"),
    return_metrics=True
)

# Inspect ETL metrics
print(f"Rows processed: {metrics.rows_output}")
print(f"Valuation metrics added: {metrics.valuation_metrics_added}")
print(f"Missing values after imputation: {metrics.missing_values_after_imputation}")
```

**Validation Checkpoints (REQUIRED after ETL):**

```python
# Required assertions after ETL
assert not all_stocks_preprocessed.empty, "Preprocessed data must not be empty"
assert 'ticker' in all_stocks_preprocessed.columns, "ticker column required"
assert 'sector' in all_stocks_preprocessed.columns, "sector column required"
assert 'last_price' in all_stocks_preprocessed.columns, "last_price column required"

# Validate no missing values after 6-step imputation
missing_total = all_stocks_preprocessed.isna().sum().sum()
assert missing_total == 0, f"No missing values allowed after 6-step imputation, found {missing_total}"

# Validate data quality
assert len(all_stocks_preprocessed) > 100, f"Insufficient data: {len(all_stocks_preprocessed)} rows"
assert all_stocks_preprocessed['last_price'].min() > 0, "last_price must be positive"
```

**Configuration for Different Workflows:**

```python
# 1. Full ETL with all metrics (notebooks, production)
df, metrics = etl_with_financial_metrics(
    source='csv',
    data_dir='data/',
    compute_all_metrics=True,
    output_dir='outputs/eda/financial_metrics'
)

# 2. Quick ETL without metrics (EDA, testing)
from finance_ml.ml_workflow.preprocessing.etl import run_etl_pipeline_quick

df = run_etl_pipeline_quick(
    source='csv',
    data_dir='data/',
    apply_scaling=False,
    scaler_type='standard'
)

# 3. Custom configuration (advanced users)
config = ETLConfig(
    apply_imputation=True,
    imputation_strategy="6step",
    compute_valuation_metrics=True,
    compute_profitability_metrics=True,
    generate_quality_alerts=True
)

df, metrics = run_etl_pipeline(
    source='csv',
    data_dir='data/',
    config=config,
    return_metrics=True
)
```

**Data Source Selection:**

| Source         | Use Case                    | Requirements                       |
|----------------|-----------------------------|------------------------------------|
| `'csv'`        | Local development, testing  | `data_dir` pointing to CSV files   |
| `'db'`         | Production, large datasets  | `db_url` for PostgreSQL connection |
| `'all_stocks'` | Unified table (recommended) | `db_url` with all_stocks table     |

**Output Artifacts:**

When `output_dir` is specified, ETL generates:

```
outputs/eda/financial_metrics/
├── data_quality_alerts.json       # Quality issues and warnings
├── metrics_dashboard.json         # Summary statistics
└── dtype_diagnostics.json         # Datatype casting report
```

**Best Practices:**

1. **Always use 6-step imputation** (handles numeric, categorical, datetime)
2. **Validate output immediately** with assertions
3. **Inspect ETLMetrics** to understand data transformations
4. **Use output_dir** to generate quality alerts for monitoring
5. **Prefer `etl_with_financial_metrics()`** over manual two-step processes

**Integration with Notebooks:**

```python
# Cell 1: Configuration
from pathlib import Path
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/eda/financial_metrics")

# Cell 2: ETL execution
all_stocks_preprocessed, etl_metrics = etl_with_financial_metrics(
    source='csv',
    data_dir=DATA_DIR,
    compute_all_metrics=True,
    output_dir=OUTPUT_DIR,
    return_metrics=True
)

# Cell 3: Validation (REQUIRED)
assert not all_stocks_preprocessed.empty
assert all_stocks_preprocessed.isna().sum().sum() == 0
print(f"✓ ETL complete: {len(all_stocks_preprocessed)} rows, 0 missing values")
```

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

### 9.3 Phase 9.3 Feature Categories (196 Features)

The platform organizes engineered features into **16 semantic categories** for analysis, tracking, and model
interpretation. All features are generated by `finance_ml.ml_workflow.features.api.build_features()` and cataloged in
`finance_ml.ml_workflow.eda.phase93_categories.py`.

**Category Overview:**

| Category                   | Features | Description                                                    |
|----------------------------|----------|----------------------------------------------------------------|
| **Momentum & Technical**   | 27       | EMA crossovers, RSI, 52W High/Low position, price momentum     |
| **Valuation Ratios**       | 23       | P/E, P/B, EV/EBITDA, EV/Sales, PEG ratio, valuation trends     |
| **Profitability**          | 12       | Operating margin, net margin, ROE, ROA, ROIC, earnings quality |
| **Quality & Risk**         | 18       | Altman Z-Score, Piotroski F-Score, accruals ratio, volatility  |
| **Cash Flow**              | 5        | FCF yield, OCF/Sales, cash conversion                          |
| **Capital Allocation**     | 23       | Buyback yield, dividend coverage, payout ratios                |
| **Analyst Sentiment**      | 10       | Analyst rating changes, target revisions, consensus            |
| **Market Sentiment**       | 5        | Relative strength, volume trends, market cap percentile        |
| **Leverage & Liquidity**   | 9        | Debt ratios, current ratio, interest coverage                  |
| **Temporal Patterns**      | 16       | Seasonality, day-of-week effects, quarter-end patterns         |
| **Composite Scores**       | 5        | Combined quality, value, momentum scores                       |
| **Growth Metrics**         | 6        | Revenue growth, EBITDA growth, earnings CAGR                   |
| **Efficiency Ratios**      | 4        | Asset turnover, inventory turnover, receivables days           |
| **Employee Productivity**  | 16       | Revenue per employee, productivity trends                      |
| **Balance Sheet Dynamics** | 8        | Working capital trends, asset quality                          |
| **Revenue Forecasting**    | 9        | Analyst estimate spreads, revision momentum                    |

**Total: 196 features** registered in `PHASE93_FEATURE_CATEGORIES`

**Usage:**

```python
from finance_ml.ml_workflow.features.api import build_features

# Build all 196 features with 'comprehensive' preset
all_stocks_features = build_features(
    all_stocks_preprocessed,
    preset='comprehensive',  # Enables all Phase 9.3 features
    include_interactions=True,
    include_relative=True,
    sector_col='sector'
)

# Selective feature building
momentum_features = build_features(
    all_stocks_preprocessed,
    preset='momentum',  # Only momentum & technical features
    sector_col='sector'
)
```

**Feature Coverage Validation:**

```python
from finance_ml.ml_workflow.eda.phase93_categories import (
    PHASE93_FEATURE_CATEGORIES,
    get_phase93_coverage_stats,
    categorize_dataframe_columns,
)

# Validate feature coverage
coverage = get_phase93_coverage_stats(all_stocks_features)
total_features = sum(coverage.values())
coverage_pct = (total_features / 196) * 100

# Target: ≥90% coverage (182/196 features)
assert coverage_pct >= 90, f"Phase 9.3 coverage must be ≥90%, got {coverage_pct:.1f}%"

# Breakdown by category
for category, count in coverage.items():
    expected = len(PHASE93_FEATURE_CATEGORIES[category])
    print(f"{category}: {count}/{expected} features ({count/expected*100:.1f}%)")
```

**Category-Specific Analysis:**

```python
# Categorize columns in your dataframe
category_mapping = categorize_dataframe_columns(all_stocks_features)

# Analyze specific category
momentum_cols = [
    col for col, cat in category_mapping.items() 
    if cat == "Momentum & Technical"
]

print(f"Momentum features present: {len(momentum_cols)}/27")
print(f"Columns: {momentum_cols[:5]}...")  # Show first 5
```

**Integration with Business Objective:**

Phase 9.3 features directly support **stock price target prediction** by providing:

1. **Valuation context**: P/E, EV/EBITDA ratios indicate over/undervaluation
2. **Momentum signals**: Price trends and technical indicators predict short-term movements
3. **Quality assessment**: Altman Z, Piotroski F identify financially healthy companies
4. **Growth indicators**: Revenue/earnings growth rates justify valuation multiples
5. **Risk metrics**: Volatility and leverage ratios inform prediction uncertainty

**Output Artifacts:**

Feature engineering generates validation artifacts:

```
outputs/eda/phase93_feature_categories/
├── phase93_feature_viz_summary.json      # Feature coverage summary
├── phase93_coverage_stats.json           # Coverage by category
└── phase93_category_analysis_report.xlsx # Detailed Excel report
```

**Best Practices:**

1. **Use 'comprehensive' preset** for production models to access all 196 features
2. **Validate ≥90% coverage** after feature engineering (182/196 features minimum)
3. **Monitor coverage trends** across data snapshots
4. **Use category grouping** for feature importance analysis and interpretation
5. **Document missing features** when coverage < 90% (e.g., due to missing source columns)

#### 9.3.1 Automated Feature Selection (Phase 9.3 Task 1)

**Overview:**

Automated feature selection reduces dimensionality by removing low-importance and correlated features while preserving
PRICE_COLUMNS and model interpretability. Integrated into `etl_with_features()` as optional Stage 10.

**API:**

```python
from finance_ml.ml_workflow.features.selection import (
    select_features_auto,
    select_features_by_category
)

# Importance-based selection
X_selected = select_features_auto(
    X, y,
    importance_threshold=0.01,      # Min mutual information score
    correlation_threshold=0.95,      # Max correlation before deduplication
    method='mutual_info'             # or 'correlation', 'both'
)

# Category-based selection
X_momentum = select_features_by_category(
    X,
    categories=['momentum', 'technical']  # Select specific Phase 9.3 categories
)
```

**ETL Integration:**

```python
from finance_ml.ml_workflow.preprocessing import etl_with_features

# Basic usage (no feature selection)
df, metrics = etl_with_features(
    source='csv',
    data_dir='data/',
    feature_preset='comprehensive',
    return_metrics=True
)

# With automated feature selection (optional)
df_selected, metrics = etl_with_features(
    source='csv',
    data_dir='data/',
    feature_preset='comprehensive',
    auto_feature_selection=True,        # Enable Stage 10
    importance_threshold=0.05,          # Stricter threshold
    correlation_threshold=0.95,
    return_metrics=True
)

# Metrics tracking
print(f"Features: {metrics.features_before_selection} → {metrics.features_after_selection}")
print(f"Removed: {metrics.features_removed_by_selection} ({reduction_pct:.1f}%)")
```

**ETLConfig Parameters:**

```python
from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

config = ETLConfig(
    apply_feature_selection=True,                        # Enable feature selection
    feature_selection_method='mutual_info',              # or 'correlation', 'both'
    importance_threshold=0.01,                           # Min importance score
    correlation_threshold=0.95,                          # Max correlation
    feature_selection_categories=['momentum', 'quality'] # Category filter (optional)
)
```

**Preservation Policy:**

- **PRICE_COLUMNS** (last_price, price_target, price_target_median) are **NEVER removed**
- Target column automatically excluded from selection
- Selection applied after feature engineering (Stage 10)

**Performance Targets:**

- Execution time: <5 seconds for 6974 rows × 591 columns
- Dimensionality reduction: 20-30% while maintaining R² > 0.90 of full model
- Test coverage: 100% (4 tests in `test_feature_selection_auto.py`)

---

### 9.4 Multi-Label Classification Support (Phase 9.4)

**Purpose**: Enable simultaneous signal detection across multiple Phase 9.3 feature categories for granular
sector-specific investment strategies.

**Key Function**: `create_multilabel_event_labels()`

**Module**: `finance_ml.ml_workflow.classification.labels`

**Signature**:

```python
def create_multilabel_event_labels(
        df: pd.DataFrame,
        label_mode: str = "multilabel",
        categories: Optional[list] = None,
        sector_adjusted: bool = False,
        threshold_percentile: float = 0.6,
        ) -> pd.DataFrame
```

**Parameters**:

- `df`: Stock data with Phase 9.3 features
- `label_mode`: Must be 'multilabel' (single mode supported)
- `categories`: List of categories (e.g., `['valuation', 'momentum', 'quality']`). If None, uses all 8 default
  categories.
- `sector_adjusted`: If True, use sector-specific percentile thresholds
- `threshold_percentile`: Percentile for positive signal (0.6 = top 40%)

**Returns**: DataFrame with binary label columns `label_<category>` (0/1 per category)

**Supported Categories**:

1. `valuation`: price_target, p_e_ltm, ev_ebitda, p_b_ltm
2. `momentum`: momentum_rsi, price_change_1m, price_momentum_1m, ema_20d
3. `quality`: quality_altman_z, roe_ltm, quality_score
4. `profitability`: net_margin_ltm, operating_margin_ltm, gross_margin_ltm, roe_ltm
5. `growth`: revenue_growth_yoy, earnings_growth_yoy, revenue_growth_3y_cagr
6. `leverage`: debt_to_equity, net_debt_ebitda, current_ratio
7. `efficiency`: asset_turnover, inventory_turnover
8. `cash_flow`: fcf_margin, operating_cash_flow

**Example Usage**:

```python
from finance_ml.ml_workflow.classification.labels import create_multilabel_event_labels

# Basic multi-label classification
labels = create_multilabel_event_labels(
        df,
        categories=['valuation', 'momentum', 'quality']
        )
# Returns: label_valuation, label_momentum, label_quality columns (0/1)

# Sector-adjusted thresholds
labels = create_multilabel_event_labels(
        df,
        categories=['valuation'],
        sector_adjusted=True,
        threshold_percentile=0.7  # Top 30% = positive signal
        )
```

**Business Value**:

- **Independent signals**: Stock can be positive on valuation but negative on momentum
- **Sector-specific strategies**: Tech stocks use different valuation thresholds than Utilities
- **Granular analysis**: Identify stocks strong in quality but weak in growth

**Integration with Training**:

```python
# Train separate model per category
for category in ['valuation', 'momentum', 'quality']:
    labels = create_multilabel_event_labels(df, categories=[category])
    y = labels[f'label_{category}']

    model = train_classification_model(X, y, model='xgboost')
    # Each model specializes in one signal dimension
```

**Test Coverage**: 100% (3 tests in `test_multilabel_classification.py`)

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

**Automated CV Policy Enforcement (Phase 9.4 - RECOMMENDED)**:

```python
from finance_ml.ml_workflow.classification.models import determine_cv_strategy

# Automatically select best CV strategy based on data characteristics
cv_strategy, cv_object = determine_cv_strategy(
    df,
    target=y,
    n_splits=5,
    date_column='snapshot_date',  # default
    group_column='ticker',         # default
    random_state=42
)

# Use the returned CV object directly
for train_idx, val_idx in cv_object.split(df, y, groups=df.get('ticker')):
    # Training fold with correct strategy
    pass
```

**Strategy Selection Hierarchy**:

1. **TimeSeriesSplit**: If `date_column` exists in df → prevents look-ahead bias
2. **GroupKFold**: If `group_column` exists and has ≥n_splits unique groups → prevents ticker leakage
3. **StratifiedKFold**: If target is categorical with sufficient samples per class → maintains class balance
4. **KFold**: Fallback when above conditions not met

**Manual CV Options** (legacy, use automated method above):

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

**Integration Example**:

```python
# Determine strategy once
cv_strategy, cv_obj = determine_cv_strategy(df, target=y, n_splits=5)
logger.info(f"Using CV strategy: {cv_strategy}")

# Use in model training
scores = cross_val_score(model, X, y, cv=cv_obj, scoring='f1_weighted')
```

**Test Coverage**: 100% (3 tests in `test_cv_policy_enforcement.py`)

### 10.3 Leakage Prevention Rules

- **No future information**: Features must only use data available at prediction time
- **No target leakage**: Features cannot be derived from target variable
- **No data from test set**: Scalers, encoders, imputers fit only on train set
- **No group mixing**: Same ticker should not appear in both train and validation in CV

### 10.4 Feature Alignment Policy

**Problem**: Train/test feature misalignment causes prediction errors when:

- Features are added/removed between training and inference
- Feature engineering is regenerated with different parameters
- Interaction features change due to categorical encoding differences

**Solution**: Use `align_features_to_model()` before prediction (Phase 9.5 Task 7):

```python
from finance_ml.ml_workflow.regression.dataset import align_features_to_model, predict_with_model

# Option 1: Explicit alignment
X_test_aligned = align_features_to_model(X_test, model, fill_value=0.0)
predictions = model.predict(X_test_aligned)

# Option 2: Wrapper function (recommended)
predictions = predict_with_model(model, X_test, fill_missing=0.0)
```

**Alignment Strategy**:

1. **Missing features**: Added and filled with `fill_value` (default: 0.0)
    - Rationale: Zero is safe default for standardized features
    - Alternative: Use median from training set for critical features

2. **Extra features**: Dropped silently
    - Rationale: Model was not trained on these features

3. **Column order**: Reordered to match `model.feature_names_in_`
    - Rationale: Some models (neural nets) are order-sensitive

**Best Practices**:

- **Always align before prediction**: Use `predict_with_model()` wrapper in production
- **Log alignment statistics**: Track missing/extra features for monitoring
- **Validate alignment in tests**: Assert `X_test_aligned.columns.tolist() == model.feature_names_in_.tolist()`
- **Preserve feature engineering**: Save feature engineering parameters with model artifacts

**Integration with Notebook Workflow**:

```python
# Section 6.4: Prediction on Test Set
from finance_ml.ml_workflow.regression.dataset import predict_with_model

# Safe prediction with automatic alignment
y_pred_test = predict_with_model(stacking_model, X_test_scaled, fill_missing=0.0)

# Log alignment info
aligned_features = align_features_to_model(X_test_scaled, stacking_model)
logger.info(f"Features aligned: {X_test_scaled.shape[1]} → {aligned_features.shape[1]}")
```

**Test Coverage**: 100% (4 tests in `test_feature_alignment.py`)

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

Apply winsorization to extreme values before modeling. The platform supports two winsorization strategies:

**Aggressive Winsorization (0.01/0.99 - Default for Notebooks):**

```python
from scipy.stats.mstats import winsorize

df['market_cap_winsorized'] = winsorize(df['market_cap'], limits=[0.01, 0.01])
# Clips values at 1st and 99th percentiles (aggressive)
```

**Conservative Winsorization (0.10/0.90 - Default for Package):**

```python
from finance_ml.ml_workflow.preprocessing.outliers import winsorize_by_sector

df_winsorized = winsorize_by_sector(
    df,
    columns=['market_cap', 'ev', 'total_assets'],
    lower=0.10,  # 10th percentile
    upper=0.90,  # 90th percentile
    sector_col='sector'
)
# Less aggressive, preserves more extreme values
```

**Policy Decision - When to Use Each:**

| Approach         | Bounds    | Use Case                                         | Rationale                                                                                                 |
|------------------|-----------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **Aggressive**   | 0.01/0.99 | Notebooks, exploratory analysis, high-noise data | Removes more outliers; reduces impact of data quality issues                                              |
| **Conservative** | 0.10/0.90 | Production models, financial data, mega-caps     | Preserves valid extreme values (e.g., Apple, Nvidia market caps); maintains business-critical information |

**Recommended: Conservative (0.10/0.90)** for production models to avoid corrupting valuation metrics for legitimate
high-growth stocks and mega-cap companies.

> **Note**: The notebooks (`ml_finance_model_main2_0.ipynb`) use aggressive bounds (0.01/0.99) for data exploration,
> while the package defaults to conservative bounds (0.10/0.90) as defined in Section 2.1. This is intentional -
> notebooks
> prioritize noise reduction for exploration, while production code prioritizes preserving valid extreme values for
> accurate predictions.

**Price Column Protection:**

Per Section 8.5.2, price columns (`last_price`, `price_target`, etc.) must **never** be winsorized to preserve the core
valuation metric: `(Predicted_Target - Last_Price) / Last_Price`.

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

### 17.2 Interactive Plotly Visualizations

The platform uses Plotly extensively for interactive visualizations in notebooks and dashboards. Follow these standards
for consistency.

**Standard Plotly Configuration:**

```python
import plotly.express as px
import plotly.graph_objects as go

# Template selection
PLOTLY_TEMPLATE = 'plotly_dark'  # or 'seaborn' for light mode

# Standard color palette (aligned with Section 17.1)
COLOR_PALETTE = {
    'primary': '#375a7f',
    'secondary': '#6c757d',
    'success': '#00bc8c',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#3498db',
    'neutral': '#adb5bd',
}

# Apply template to all Plotly figures
px.defaults.template = PLOTLY_TEMPLATE
```

**Required Visualizations by Phase:**

| Phase              | Visualization            | Function/Module                                                | Purpose                          |
|--------------------|--------------------------|----------------------------------------------------------------|----------------------------------|
| **9.2 EDA**        | Correlation Heatmap      | `finance_ml.ml_workflow.eda.correlation_analysis()`            | Identify multicollinearity       |
| **9.2 EDA**        | Distribution Plots       | `px.histogram()`, `px.box()`                                   | Assess feature distributions     |
| **9.3 Features**   | Feature Category Treemap | Custom Plotly treemap                                          | Visualize Phase 9.3 coverage     |
| **9.5 Regression** | Residual Analysis        | `finance_ml.ml_workflow.evaluation.plot_residual_analysis()`   | Check model assumptions          |
| **9.6 Evaluation** | Reliability Diagram      | `finance_ml.ml_workflow.evaluation.plot_reliability_diagram()` | Validate uncertainty calibration |
| **9.6 Evaluation** | Interval Coverage        | `finance_ml.ml_workflow.evaluation.plot_interval_coverage()`   | Assess prediction intervals      |
| **9.7 Analytics**  | Stock Rankings           | `finance_ml.ml_workflow.analytics.plot_rankings_interactive()` | Identify under/overvalued stocks |
| **9.7 Portfolio**  | Efficient Frontier       | `finance_ml.ml_workflow.analytics.plot_efficient_frontier()`   | Portfolio optimization results   |

**Standard Plot Configuration:**

```python
# Scatter plot example with standard configuration
fig = px.scatter(
    df,
    x='last_price',
    y='price_target',
    color='sector',
    hover_data=['ticker', 'isin', 'region'],
    template=PLOTLY_TEMPLATE,
    title='Price Target vs Last Price',
    labels={'last_price': 'Last Price ($)', 'price_target': 'Price Target ($)'}
)

# Update layout for consistency
fig.update_layout(
    font=dict(family='Arial, sans-serif', size=14),
    title_font_size=20,
    showlegend=True,
    legend=dict(orientation='v', yanchor='top', xanchor='right', x=1.02, y=1),
    hovermode='closest',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

# Save as HTML for sharing
fig.write_html('outputs/plots/prediction_scatter_interactive.html')
```

**Heatmap Configuration:**

```python
# Correlation heatmap with consistent styling
fig = px.imshow(
    correlation_matrix,
    template=PLOTLY_TEMPLATE,
    color_continuous_scale='RdBu_r',  # Diverging scale centered at 0
    zmin=-1, zmax=1,
    text_auto='.2f',
    labels=dict(color='Correlation'),
    title='Feature Correlation Matrix'
)

fig.update_layout(
    width=1000,
    height=800,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
)
```

**Best Practices:**

1. **Always set template**: Use `PLOTLY_TEMPLATE` for consistency
2. **Include hover data**: Add `ticker`, `sector`, `region` for context
3. **Label axes with units**: e.g., "Price ($)", "Market Cap (Billion $)"
4. **Save interactive plots**: Use `fig.write_html()` for sharing and archiving
5. **Use color semantics**: Green for positive, red for negative, blue for neutral
6. **Enable zoom and pan**: Default Plotly interactions support exploration

**Dashboard Layout and Structure:**

**Dashboards (Streamlit & Dash):**

- **Header:** Clear application title and status indicators
- **Navigation:** Logical tab-based structure grouped by business function (Overview, Analysis, Governance, Portfolio)
- **Filters:**
    - Use a dedicated sidebar or top filter bar
    - Use distinct styles for active vs. inactive filters
    - Enable multi-select for categorical fields (Sector, Region)
    - Implement "Dark Mode" styling for dropdowns and inputs (`custom.css` for Dash)
- **KPI Cards:** Use summary cards at the top for high-level metrics
- **Responsiveness:** Ensure plots resize dynamically (`width='stretch'` in Streamlit, Flexbox in Dash)

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

---

## 19. Data Quality Validation Checkpoints

All data transformations must be validated at critical checkpoints to ensure pipeline integrity. These assertions serve
as guardrails against data quality issues and transformation errors.

### 19.1 Post-ETL Validation (REQUIRED)

After running `etl_with_financial_metrics()` or `run_etl_pipeline()`, validate the output immediately:

```python
# Required assertions after ETL
assert not df.empty, "DataFrame must not be empty"
assert 'ticker' in df.columns, "ticker column required"
assert 'sector' in df.columns, "sector column required"
assert 'last_price' in df.columns, "last_price column required"

# Validate target columns
target_cols = ['price_target', 'price_target_median', 'last_price']
has_target = any(col in df.columns for col in target_cols)
assert has_target, f"At least one target column required: {target_cols}"

# Quality metrics
missing_pct = df.isna().sum().sum() / df.size * 100
assert missing_pct == 0, f"No missing values allowed after 6-step imputation, found {missing_pct:.2f}%"

# Data sufficiency
assert len(df) > 100, f"Insufficient data: {len(df)} rows (minimum 100)"
assert df['last_price'].min() > 0, "last_price must be positive"

print(f"✓ ETL validation passed: {len(df)} rows, 0 missing values")
```

### 19.2 Post-Feature Engineering Validation

After building Phase 9.3 features, validate coverage:

```python
from finance_ml.ml_workflow.eda.phase93_categories import get_phase93_coverage_stats

# Phase 9.3 coverage target: ≥90% (182/196 features)
coverage_stats = get_phase93_coverage_stats(df)
total_features = sum(coverage_stats.values())
coverage_pct = (total_features / 196) * 100

assert coverage_pct >= 90, f"Phase 9.3 coverage must be ≥90%, got {coverage_pct:.1f}%"

print(f"✓ Feature engineering validation passed: {coverage_pct:.1f}% coverage ({total_features}/196 features)")

# Breakdown by category
for category, count in coverage_stats.items():
    expected = len(PHASE93_FEATURE_CATEGORIES[category])
    print(f"  {category}: {count}/{expected} features")
```

### 19.3 Pre-Modeling Validation

Before training models, validate data types and schema:

```python
# Validate data types
numeric_cols = df.select_dtypes(include=[np.number]).columns
assert len(numeric_cols) > 50, f"Insufficient numeric features: {len(numeric_cols)}"

# Check for infinity values (should not exist after imputation)
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
assert inf_count == 0, f"Infinity values detected: {inf_count}"

# Validate target variable distribution
target_col = 'price_target' if 'price_target' in df.columns else 'last_price'
assert df[target_col].notna().all(), f"Target column {target_col} contains NaN"
assert df[target_col].min() > 0, f"Target column {target_col} must be positive"

print(f"✓ Pre-modeling validation passed: {len(numeric_cols)} numeric features, no NaN/Inf")
```

### 19.4 Post-Prediction Validation

After generating predictions, validate schema and invariants:

```python
# Required columns (Section 11)
required_cols = ['ticker', 'sector', 'y_true', 'y_pred']
missing_cols = [col for col in required_cols if col not in predictions_df.columns]
assert not missing_cols, f"Missing required columns: {missing_cols}"

# Non-negativity constraint (prices must be ≥ 0)
price_cols = ['y_pred', 'pred_p10', 'pred_p50', 'pred_p90']
for col in price_cols:
    if col in predictions_df.columns:
        assert predictions_df[col].min() >= 0, f"{col} contains negative values"

# Monotonicity constraint (quantile predictions)
if all(col in predictions_df.columns for col in ['pred_p10', 'pred_p50', 'pred_p90']):
    violations = (
        (predictions_df['pred_p10'] > predictions_df['pred_p50']) |
        (predictions_df['pred_p50'] > predictions_df['pred_p90'])
    ).sum()
    assert violations == 0, f"Quantile monotonicity violated in {violations} rows"

print(f"✓ Post-prediction validation passed: {len(predictions_df)} predictions, all constraints satisfied")
```

### 19.5 Validation Utilities

Use built-in validation utilities for consistency:

```python
# ETL validation
from finance_ml.ml_workflow.preprocessing.imputation import validate_imputation_completeness

result = validate_imputation_completeness(df)
assert result['is_complete'], f"Imputation incomplete: {result['missing_total']} missing values"

# Schema validation
from finance_ml.ml_workflow.preprocessing.data import validate_schema

schema_issues = validate_schema(df, required_cols=['ticker', 'sector', 'last_price'])
assert not schema_issues, f"Schema validation failed: {schema_issues}"

# Predictions schema validation
from finance_ml.ml_workflow.regression.io import validate_predictions_schema

validate_predictions_schema(predictions_df)  # Raises if invalid
```

**Best Practices:**

1. **Fail fast**: Place assertions immediately after transformations
2. **Descriptive messages**: Include actual values and thresholds in error messages
3. **Log successes**: Print confirmation messages for passed validations
4. **Use utilities**: Prefer built-in validation functions over manual checks
5. **Document exceptions**: If skipping a validation, document why with a comment

---

## 20. Output Artifact Standards

The platform generates structured output artifacts for monitoring, governance, and reproducibility. All outputs follow
standardized directory structure and naming conventions.

### 20.1 Required Output Directories

```
outputs/
├── eda/                          # Phase 9.2 EDA outputs
│   ├── financial_metrics/        # Quality alerts, dashboards (Section 8.6)
│   └── phase93_feature_categories/  # Feature visualizations (Section 9.3)
├── preprocessing/                # Phase 9.1 preprocessing artifacts
│   ├── etl_metrics.json          # ETL pipeline metrics
│   ├── dtype_diagnostics.json    # Datatype casting report
│   └── imputation_summary.json   # Imputation statistics
├── regression/                   # Phase 9.5 trained models
│   ├── sector_models/            # Per-sector model files
│   ├── quantile_q1_phase95.joblib
│   ├── quantile_q5_phase95.joblib
│   ├── quantile_q9_phase95.joblib
│   └── stacking_ensemble_phase95.joblib
├── evaluation/                   # Phase 9.6 metrics, calibration
│   ├── regression_metrics_by_sector.csv  # Sector-level metrics
│   ├── quantile_diagnostics.csv          # Uncertainty quantification
│   └── calibration_report.json           # Sector bias calibration
├── analytics/                    # Phase 9.7 rankings, portfolio
│   ├── mispricing_scores.csv     # Stock rankings
│   ├── portfolio_weights.csv     # Optimized portfolio
│   └── risk_metrics.json         # Risk analysis
├── reporting/                    # Phase 9.8 final reports
│   ├── model_card.json           # Model governance
│   ├── lineage.json              # Data provenance
│   └── executive_summary.xlsx    # Business report
├── plots/                        # Visualizations
│   ├── prediction_scatter_interactive.html
│   ├── residual_analysis_interactive.png
│   └── correlation_heatmap.html
└── governance/                   # Model governance artifacts
    ├── model_card.json
    ├── lineage.json
    └── audit_trail.log
```

### 20.2 Required JSON Artifacts

**ETL Metrics (`outputs/preprocessing/etl_metrics.json`):**

```json
{
  "rows_initial": 5234,
  "rows_after_etl": 5234,
  "missing_before_imputation": 12045,
  "missing_after_imputation": 0,
  "valuation_metrics_added": 24,
  "profitability_metrics_added": 18,
  "timestamp": "2025-12-05T15:30:00",
  "model_version": "v9_9"
}
```

**EDA Summary (`outputs/eda/eda_summary.json`):**

```json
{
  "total_rows": 5234,
  "total_columns": 318,
  "sectors": ["Technology", "Healthcare", "Financials", "..."],
  "sector_counts": {"Technology": 1245, "Healthcare": 892, "..."},
  "phase93_coverage": {"total": 182, "percentage": 92.8},
  "timestamp": "2025-12-05T15:35:00"
}
```

**Phase 9.3 Benchmarking (`outputs/eda/phase93_benchmarking.json`):**

```json
{
  "coverage_by_category": {
    "Momentum & Technical": 27,
    "Valuation Ratios": 23,
    "Profitability": 12,
    "...": "..."
  },
  "total_coverage": 182,
  "target_coverage": 196,
  "coverage_percentage": 92.8,
  "missing_features": ["feature1", "feature2", "..."]
}
```

**Model Card (`outputs/governance/model_card.json`):**

```json
{
  "model_version": "v9_9",
  "model_type": "Stacking Ensemble (RF + GB + XGB)",
  "training_date": "2025-12-05",
  "metrics": {
    "overall": {"mae": 8.45, "rmse": 12.32, "r2": 0.78},
    "by_sector": {"Technology": {"mae": 7.21, "r2": 0.82}, "...": "..."}
  },
  "features_used": 318,
  "phase93_coverage": 92.8,
  "data_source": "csv",
  "imputation_strategy": "6step"
}
```

### 20.3 CSV Output Standards

**Predictions (`outputs/regression/regression_predictions_detailed.csv`):**

Required columns (Section 11):

- ticker, isin, sector, region
- last_price, y_true, y_pred, y_pred_calibrated
- pred_p10, pred_p50, pred_p90, interval_width
- abs_error, pct_error
- model_version, snapshot_date

**Sector Metrics (`outputs/regression/regression_metrics_by_sector.csv`):**

Required columns (Section 12):

- sector, mae, rmse, r2, mape, bias, count
- model_version, timestamp

### 20.4 Artifact Generation Functions

```python
# Generate ETL metrics
from finance_ml.ml_workflow.preprocessing.etl import etl_with_financial_metrics

df, etl_metrics = etl_with_financial_metrics(
    source='csv',
    data_dir='data/',
    output_dir='outputs/eda/financial_metrics',  # Auto-generates artifacts
    return_metrics=True
)

# Save predictions
from finance_ml.ml_workflow.regression.io import save_predictions

save_predictions(
    predictions_df,
    output_path='outputs/regression/regression_predictions_detailed.csv',
    schema='standardized'  # Enforces Section 11 schema
)

# Generate model card
from finance_ml.ml_workflow.evaluation import generate_model_card

model_card = generate_model_card(
    model=stacking_model,
    metrics=metrics_dict,
    output_path='outputs/governance/model_card.json'
)
```

**Best Practices:**

1. **Use standard paths**: Follow the directory structure exactly
2. **Include timestamps**: Add ISO 8601 timestamps to all JSON artifacts
3. **Version everything**: Include `model_version` in all outputs
4. **Validate before saving**: Use schema validation functions
5. **Generate automatically**: Use provided generation functions rather than manual file writes

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
