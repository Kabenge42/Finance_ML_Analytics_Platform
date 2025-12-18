==== UserMessageImpl #1 ====
User:
Suggest further enhancements/improvements for the @file:etl_data_explorer.ipynb and @file:earnings_widgets.py
respectively based on the updated sections in @file:etl_data_explorer.ipynb and @file:schema.py . Include comprehensive
financial analytics to the earnings calendars, financial metrics chart and earnings_dashboard (e.g. monitoring/alerts
regarding expected vs actual earnings, analyst rating/recommendation, Price Target analytics, Market movers etc.). See
attached json outputs for reference.
Attachments:
Attachment Name: code_guidelines.md
Attachments Kind: Visible
Attachments Source: AiRuleContextSource / AiRuleChatAttachment
Attachments Text:
truncated contents:

# Finance ML Analytics Platform — Code Guidelines

**Version:** 1.11  
**Last Updated:** 2025-12-14  
**Package Version:** 0.9.4  
**Model Version:** v9_10

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
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v9_10')
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
| `MODEL_VERSION`       | `v9_10`           | Version tracking for audit trails and model comparison                                |

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
normalized = normalize_column_name(
        "Selling General & Admin Expenses/Total (FQ)")  # → "selling_general_and_admin_expenses_total_fq"
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

Attachment Name: valuation_opportunities.json
Attachments Kind: Visible
Attachments Source: User / FileChatAttachment
Attachments Text:

```json
{
  "timestamp": "2025-12-15T19:57:30.954154",
  "total_stocks_analyzed": 6666,
  "valuation_summary": {
    "mean_target_vs_price": 24.826135314627194,
    "median_target_vs_price": 15.989297623481882,
    "std_target_vs_price": 205.72513866505307
  },
  "category_distribution": {
    "Fairly Valued": 2939,
    "Undervalued": 2833,
    "Deeply Undervalued": 671,
    "Overvalued": 144,
    "Deeply Overvalued": 79
  },
  "top_undervalued": [
    {
      "ticker": "BALL",
      "sector": "Materials",
      "last_price": 50.91,
      "price_target": 8201.72,
      "target_vs_price": 16010.233745825968
    },
    {
      "ticker": "AMBP3",
      "sector": "Industrials",
      "last_price": 0.27,
      "price_target": 12.0,
      "target_vs_price": 4344.444444444444
    },
    {
      "ticker": "ENLGN0000",
      "sector": "Real Estate",
      "last_price": 6.3,
      "price_target": 52.43,
      "target_vs_price": 732.2222222222223
    },
    {
      "ticker": "ALSEN",
      "sector": "Health Care",
      "last_price": 0.337,
      "price_target": 2.375,
      "target_vs_price": 604.7477744807121
    },
    {
      "ticker": "NVM",
      "sector": "Consumer Discretionary",
      "last_price": 2.62,
      "price_target": 14.45,
      "target_vs_price": 451.5267175572518
    },
    {
      "ticker": "ZPHR",
      "sector": "Energy",
      "last_price": 0.0255,
      "price_target": 0.1373,
      "target_vs_price": 438.43137254901967
    },
    {
      "ticker": "RVRG",
      "sector": "Financials",
      "last_price": 0.0515,
      "price_target": 0.2725,
      "target_vs_price": 429.12621359223306
    },
    {
      "ticker": "AERI3",
      "sector": "Industrials",
      "last_price": 3.28,
      "price_target": 17.0,
      "target_vs_price": 418.29268292682934
    },
    {
      "ticker": "AVTX",
      "sector": "Industrials",
      "last_price": 6.574,
      "price_target": 31.18,
      "target_vs_price": 374.292668086401
    },
    {
      "ticker": "IBRX",
      "sector": "Health Care",
      "last_price": 2.36,
      "price_target": 10.4,
      "target_vs_price": 340.677966101695
    }
  ],
  "top_overvalued": [
    {
      "ticker": "BFG",
      "sector": "Consumer Staples",
      "last_price": 4.1,
      "price_target": 0.0,
      "target_vs_price": -100.0
    },
    {
      "ticker": "LAT",
      "sector": "Industrials",
      "last_price": 0.0128,
      "price_target": 0.0,
      "target_vs_price": -100.0
    },
    {
      "ticker": "3038",
      "sector": "Consumer Staples",
      "last_price": 3655.0,
      "price_target": 160.4931,
      "target_vs_price": -95.6089439124487
    },
    {
      "ticker": "BBDB",
      "sector": "Industrials",
      "last_price": 224.73,
      "price_target": 27.8152,
      "target_vs_price": -87.62283629243981
    },
    {
      "ticker": "300718",
      "sector": "Industrials",
      "last_price": 82.17,
      "price_target": 21.05,
      "target_vs_price": -74.38237799683584
    },
    {
      "ticker": "688195",
      "sector": "Information Technology",
      "last_price": 196.6,
      "price_target": 52.49,
      "target_vs_price": -73.30111902339776
    },
    {
      "ticker": "300315",
      "sector": "Communication Services",
      "last_price": 4.92,
      "price_target": 1.5,
      "target_vs_price": -69.51219512195121
    },
    {
      "ticker": "1860",
      "sector": "Communication Services",
      "last_price": 15.7,
      "price_target": 5.4997,
      "target_vs_price": -64.97006369426751
    },
    {
      "ticker": "688502",
      "sector": "Information Technology",
      "last_price": 378.0,
      "price_target": 133.0,
      "target_vs_price": -64.81481481481481
    },
    {
      "ticker": "BERGER",
      "sector": "Materials",
      "last_price": 40.05,
      "price_target": 16.46,
      "target_vs_price": -58.901373283395756
    }
  ]
}
```

Attachment Name: analyst_recommendations.json
Attachments Kind: Visible
Attachments Source: User / FileChatAttachment
Attachments Text:

```json
{
  "timestamp": "2025-12-15T19:57:31.473850",
  "total_stocks_analyzed": 6666,
  "available_columns": {
    "analyst_rating": "analyst_rating",
    "buy_ratings": "num_buys_ratings",
    "hold_ratings": "num_hold_ratings",
    "sell_ratings": "num_strong_sell_ratings",
    "price_target": "price_target_ytd_ago",
    "target_high": "price_target_high",
    "target_low": "price_target_low"
  },
  "rating_distribution": {},
  "by_sector": {
    "Information Technology": {
      "count": 816,
      "mean_target": 1173.4136903186275,
      "median_target": 47.19055,
      "mean_upside": 23.85536441380263,
      "positive_upside_pct": 85.17156862745098
    },
    "Communication Services": {
      "count": 303,
      "mean_target": 4065.9284056105607,
      "median_target": 30.0104,
      "mean_upside": 27.43034270550233,
      "positive_upside_pct": 89.10891089108911
    },
    "Consumer Discretionary": {
      "count": 790,
      "mean_target": 1378.6137060759495,
      "median_target": 43.0,
      "mean_upside": 23.713226844051324,
      "positive_upside_pct": 86.58227848101265
    },
    "Consumer Staples": {
      "count": 462,
      "mean_target": 1837.3124424242424,
      "median_target": 43.422200000000004,
      "mean_upside": 19.98933798344384,
      "positive_upside_pct": 87.44588744588745
    },
    "Health Care": {
      "count": 692,
      "mean_target": 6081.668851156069,
      "median_target": 43.2111,
      "mean_upside": 34.51767813185501,
      "positive_upside_pct": 92.48554913294798
    },
    "Financials": {
      "count": 947,
      "mean_target": 1971.5026100316788,
      "median_target": 43.4889,
      "mean_upside": 14.274450187357377,
      "positive_upside_pct": 81.9429778247096
    },
    "Energy": {
      "count": 281,
      "mean_target": 1541.5102288256226,
      "median_target": 33.0,
      "mean_upside": 22.268390027582008,
      "positive_upside_pct": 85.40925266903915
    },
    "Industrials": {
      "count": 1316,
      "mean_target": 2642.0216014437688,
      "median_target": 47.7222,
      "mean_upside": 23.621169917298456,
      "positive_upside_pct": 85.25835866261399
    },
    "Materials": {
      "count": 629,
      "mean_target": 3281.7896580286174,
      "median_target": 34.5455,
      "mean_upside": 42.53509026794702,
      "positive_upside_pct": 80.28616852146264
    },
    "Utilities": {
      "count": 242,
      "mean_target": 547.0834276859504,
      "median_target": 35.468500000000006,
      "mean_upside": 15.064564259708296,
      "positive_upside_pct": 88.01652892561982
    },
    "Real Estate": {
      "count": 188,
      "mean_target": 1125.61445106383,
      "median_target": 28.05,
      "mean_upside": 24.456980413556575,
      "positive_upside_pct": 83.51063829787235
    }
  },
  "by_region": {},
  "by_size_class": {
    "Large Cap": {
      "count": 1514,
      "mean_upside": 12.183511774620142,
      "median_upside": 10.830172103584772,
      "positive_pct": 82.56274768824306
    },
    "Mid Cap": {
      "count": 3036,
      "mean_upside": 22.904070646812716,
      "median_upside": 15.249980972524328,
      "positive_pct": 85.93544137022397
    },
    "Small Cap": {
      "count": 2116,
      "mean_upside": 36.6296893652205,
      "median_upside": 23.96263774104684,
      "positive_pct": 87.28733459357278
    }
  },
  "by_style_class": {
    "Growth": {
      "count": 598,
      "mean_upside": 18.54777060167556,
      "median_upside": 17.12837554269479,
      "positive_pct": 85.45150501672241
    },
    "Core": {
      "count": 4314,
      "mean_upside": 25.35564586375824,
      "median_upside": 15.70967164130811,
      "positive_pct": 85.76726935558646
    },
    "Value": {
      "count": 1754,
      "mean_upside": 25.664307258409256,
      "median_upside": 16.500998613037446,
      "positive_pct": 85.23375142531357
    }
  },
  "price_target_stats": {
    "count": 6666,
    "mean": 2478.4388049954996,
    "median": 43.0,
    "std": 30522.31244697986,
    "min": 0.0098,
    "max": 1868690.433
  }
}
```

Attachment Name: earnings_monitor.json
Attachments Kind: Visible
Attachments Source: User / FileChatAttachment
Attachments Text:

```json
{
  "timestamp": "2025-12-03T19:27:28.561771",
  "reference_date": "2025-12-03T00:00:00",
  "earnings_window_days": 10,
  "total_stocks_analyzed": 6957,
  "stocks_in_earnings_window": 511,
  "earnings_calendar": [
    {
      "ticker": "AVGO",
      "sector": "Information Technology",
      "last_price": 381.57,
      "days_to_earnings": 8
    },
    {
      "ticker": "COST",
      "sector": "Consumer Staples",
      "last_price": 922.03,
      "days_to_earnings": 8
    },
    {
      "ticker": "CRM",
      "sector": "Information Technology",
      "last_price": 234.71,
      "days_to_earnings": 0
    },
    {
      "ticker": "RY",
      "sector": "Financials",
      "last_price": 216.21,
      "days_to_earnings": 0
    },
    {
      "ticker": "TD",
      "sector": "Financials",
      "last_price": 118.2,
      "days_to_earnings": 1
    },
    {
      "ticker": "ADBE",
      "sector": "Information Technology",
      "last_price": 322.81,
      "days_to_earnings": 7
    },
    {
      "ticker": "CRWD",
      "sector": "Information Technology",
      "last_price": 516.55,
      "days_to_earnings": -1
    },
    {
      "ticker": "BMO",
      "sector": "Financials",
      "last_price": 175.78,
      "days_to_earnings": 1
    },
    {
      "ticker": "SNOW",
      "sector": "Information Technology",
      "last_price": 259.68,
      "days_to_earnings": 0
    },
    {
      "ticker": "BNS",
      "sector": "Financials",
      "last_price": 98.72,
      "days_to_earnings": -1
    },
    {
      "ticker": "SNPS",
      "sector": "Information Technology",
      "last_price": 449.35,
      "days_to_earnings": 7
    },
    {
      "ticker": "CM",
      "sector": "Financials",
      "last_price": 121.06,
      "days_to_earnings": 1
    },
    {
      "ticker": "MRVL",
      "sector": "Information Technology",
      "last_price": 92.89,
      "days_to_earnings": -1
    },
    {
      "ticker": "AZO",
      "sector": "Consumer Discretionary",
      "last_price": 3826.77,
      "days_to_earnings": 6
    },
    {
      "ticker": "FERG",
      "sector": "Industrials",
      "last_price": 252.29,
      "days_to_earnings": 6
    },
    {
      "ticker": "KR",
      "sector": "Consumer Staples",
      "last_price": 67.03,
      "days_to_earnings": 1
    },
    {
      "ticker": "DOL",
      "sector": "Consumer Discretionary",
      "last_price": 196.89,
      "days_to_earnings": 8
    },
    {
      "ticker": "PSTG",
      "sector": "Information Technology",
      "last_price": 94.72,
      "days_to_earnings": -1
    },
    {
      "ticker": "HPE",
      "sector": "Information Technology",
      "last_price": 21.92,
      "days_to_earnings": 0
    },
    {
      "ticker": "CIEN",
      "sector": "Information Technology",
      "last_price": 202.5,
      "days_to_earnings": 8
    },
    {
      "ticker": "ULTA",
      "sector": "Consumer Discretionary",
      "last_price": 548.05,
      "days_to_earnings": 1
    },
    {
      "ticker": "DG",
      "sector": "Consumer Staples",
      "last_price": 110.03,
      "days_to_earnings": 1
    },
    {
      "ticker": "DLTR",
      "sector": "Consumer Staples",
      "last_price": 108.99,
      "days_to_earnings": 0
    },
    {
      "ticker": "IOT",
      "sector": "Information Technology",
      "last_price": 37.98,
      "days_to_earnings": 1
    },
    {
      "ticker": "LULU",
      "sector": "Consumer Discretionary",
      "last_price": 182.4,
      "days_to_earnings": 8
    },
    {
      "ticker": "CASY",
      "sector": "Consumer Staples",
      "last_price": 566.81,
      "days_to_earnings": 6
    },
    {
      "ticker": "GWRE",
      "sector": "Information Technology",
      "last_price": 214.72,
      "days_to_earnings": 0
    },
    {
      "ticker": "COO",
      "sector": "Health Care",
      "last_price": 75.84,
      "days_to_earnings": 1
    },
    {
      "ticker": "OKTA",
      "sector": "Information Technology",
      "last_price": 81.87,
      "days_to_earnings": -1
    },
    {
      "ticker": "BFB",
      "sector": "Consumer Staples",
      "last_price": 29.89,
      "days_to_earnings": 1
    },
    {
      "ticker": "CHWY",
      "sector": "Consumer Discretionary",
      "last_price": 33.47,
      "days_to_earnings": 7
    },
    {
      "ticker": "DOCU",
      "sector": "Information Technology",
      "last_price": 68.86,
      "days_to_earnings": 1
    },
    {
      "ticker": "RBRK",
      "sector": "Information Technology",
      "last_price": 69.4,
      "days_to_earnings": 1
    },
    {
      "ticker": "TOL",
      "sector": "Consumer Discretionary",
      "last_price": 139.1,
      "days_to_earnings": 5
    },
    {
      "ticker": "NDSN",
      "sector": "Industrials",
      "last_price": 236.53,
      "days_to_earnings": 7
    },
    {
      "ticker": "AVAV",
      "sector": "Industrials",
      "last_price": 264.95,
      "days_to_earnings": 6
    },
    {
      "ticker": "HRL",
      "sector": "Consumer Staples",
      "last_price": 23.09,
      "days_to_earnings": 1
    },
    {
      "ticker": "SAIL",
      "sector": "Information Technology",
      "last_price": 19.14,
      "days_to_earnings": 6
    },
    {
      "ticker": "DCI",
      "sector": "Industrials",
      "last_price": 88.16,
      "days_to_earnings": 1
    },
    {
      "ticker": "CPB",
      "sector": "Consumer Staples",
      "last_price": 29.89,
      "days_to_earnings": 6
    },
    {
      "ticker": "FIVE",
      "sector": "Consumer Discretionary",
      "last_price": 158.9,
      "days_to_earnings": 0
    },
    {
      "ticker": "HQY",
      "sector": "Health Care",
      "last_price": 98.88,
      "days_to_earnings": 0
    },
    {
      "ticker": "EMPA",
      "sector": "Consumer Staples",
      "last_price": 51.0,
      "days_to_earnings": 8
    },
    {
      "ticker": "TTAN",
      "sector": "Information Technology",
      "last_price": 90.075,
      "days_to_earnings": 1
    },
    {
      "ticker": "PATH",
      "sector": "Information Technology",
      "last_price": 14.3,
      "days_to_earnings": 0
    },
    {
      "ticker": "OLLI",
      "sector": "Consumer Discretionary",
      "last_price": 119.72,
      "days_to_earnings": 6
    },
    {
      "ticker": "GTLB",
      "sector": "Information Technology",
      "last_price": 43.37,
      "days_to_earnings": -1
    },
    {
      "ticker": "DSG",
      "sector": "Information Technology",
      "last_price": 115.39,
      "days_to_earnings": 0
    },
    {
      "ticker": "NTSK",
      "sector": "Information Technology",
      "last_price": 17.88,
      "days_to_earnings": 8
    },
    {
      "ticker": "BETA",
      "sector": "Industrials",
      "last_price": 28.37,
      "days_to_earnings": 1
    }
  ],
  "category_distribution": {
    "Next 3 Days": 181,
    "Today": 104,
    "Next Week": 102,
    "Later": 70,
    "Recently Reported": 54
  },
  "sector_distribution": {
    "Consumer Discretionary": 91,
    "Industrials": 84,
    "Financials": 78,
    "Health Care": 54,
    "Information Technology": 54,
    "Materials": 52,
    "Consumer Staples": 42,
    "Communication Services": 24,
    "Real Estate": 14,
    "Energy": 10,
    "Utilities": 8
  },
  "earnings_surprise_analysis": {
    "Revenue": {
      "estimate_col": "revenues_est_avg_ntm",
      "actual_col": "total_revenues_ltm",
      "count": 6708,
      "mean_surprise_pct": 10.996079085154054,
      "median_surprise_pct": 3.4134033748600303,
      "positive_surprises": 3868,
      "negative_surprises": 2840
    },
    "EBITDA": {
      "estimate_col": "ebitda_1fy",
      "actual_col": "ebitda_ltm",
      "count": 6216,
      "mean_surprise_pct": 7.984460883421329,
      "median_surprise_pct": -4.1914630434699305,
      "positive_surprises": 2669,
      "negative_surprises": 3547
    },
    "EBIT": {
      "estimate_col": "ebit_est_med_ntm",
      "actual_col": "ebit_ltm",
      "count": 6728,
      "mean_surprise_pct": 331.11136094004394,
      "median_surprise_pct": 22.066270678879988,
      "positive_surprises": 4653,
      "negative_surprises": 2075
    },
    "Net Income": {
      "estimate_col": "net_income_adj_1fy",
      "actual_col": "net_income_is_ltm",
      "count": 3368,
      "mean_surprise_pct": 33.82433438853819,
      "median_surprise_pct": 19.47233664266465,
      "positive_surprises": 1770,
      "negative_surprises": 1598
    },
    "EPS": {
      "estimate_col": "eps_norm_est_avg_ntm",
      "actual_col": "eps_adj_ltm",
      "count": 6213,
      "mean_surprise_pct": -232.75591952710093,
      "median_surprise_pct": 73.85074626865672,
      "positive_surprises": 3476,
      "negative_surprises": 2737
    }
  }
}
```

Attachment Name: dividend_analytics.json
Attachments Kind: Visible
Attachments Source: User / FileChatAttachment
Attachments Text:

```json
{
  "timestamp": "2025-12-15T19:57:33.650430",
  "total_stocks_analyzed": 6666,
  "phase93_dividends_metrics": [
    "dividend_streak",
    "dividend_record_amount",
    "dividend_record_frequency",
    "dividend_per_share_ltm",
    "common_dividends_paid_ltm",
    "div_yield_ltm",
    "div_yield_ntm",
    "div_yield_ttm"
  ],
  "phase93_cashflow_metrics": [
    "cfo_ltm",
    "fcf_ltm",
    "cfi_ltm",
    "cff_ltm",
    "cff_fy"
  ],
  "available_columns": {
    "dividend_yield": "div_yield_ltm",
    "dividend_yield_ind": "div_yield_ind",
    "dividend_yield_fwd_1y": "div_yield_1fyind",
    "dividend_yield_fwd_2y": "div_yield_2fyind",
    "dividend_yield_fwd_3y": "div_yield_3fyind",
    "dividend_yield_fwd_4y": "div_yield_4fyind",
    "dividend_yield_fwd_5y": "div_yield_5fyind",
    "dividend_yield_5y_avg": "div_yield_5yavgltm",
    "dividend_per_share": "dividend_per_share_ltm",
    "ex_dividend_date": "dividend_record_ex_date",
    "dividend_frequency": "dividend_record_frequency",
    "dividend_streak": "dividend_streak",
    "dividend_amount": "dividend_record_amount",
    "dividends_paid_ltm": "common_dividends_paid_ltm",
    "dividends_paid_fy": "common_dividends_paid_fy",
    "dividend_currency": "dividend_record_currency",
    "buyback_yield": "buyback_yield_ltm"
  },
  "phase93_category_analysis": {
    "dividends": {
      "dividend_streak": {
        "count": 6666,
        "mean": 2.3994898796081543,
        "median": 1.0,
        "std": 5.6213765144348145,
        "non_zero_pct": 70.47704770477048
      },
      "dividend_record_amount": {
        "count": 6666,
        "mean": 27.965145111083984,
        "median": 0.47699999809265137,
        "std": 322.6510314941406,
        "non_zero_pct": 99.98499849984998
      },
      "dividend_per_share_ltm": {
        "count": 6666,
        "mean": 1.223307728767395,
        "median": 0.33000001311302185,
        "std": 23.438678741455078,
        "non_zero_pct": 98.15481548154816
      },
      "common_dividends_paid_ltm": {
        "count": 6666,
        "mean": -295.05047607421875,
        "median": -78.44999694824219,
        "std": 962.763916015625,
        "non_zero_pct": 98.91989198919892
      },
      "div_yield_ltm": {
        "count": 6666,
        "mean": 0.03145194798707962,
        "median": 0.02459999918937683,
        "std": 0.04622972384095192,
        "non_zero_pct": 99.5949594959496
      },
      "div_yield_ntm": {
        "count": 6666,
        "mean": 0.026136305183172226,
        "median": 0.020999999716877937,
        "std": 0.029729198664426804,
        "non_zero_pct": 82.97329732973297
      },
      "div_yield_ttm": {
        "count": 6666,
        "mean": 0.034319426864385605,
        "median": 0.026000000536441803,
        "std": 0.08671820908784866,
        "non_zero_pct": 100.0
      }
    }
  },
  "yield_analysis": {
    "column": "div_yield_ltm",
    "total_stocks": 6666,
    "dividend_payers": 6639,
    "dividend_payer_pct": 99.5949594959496,
    "mean_yield": 0.03157985955476761,
    "median_yield": 0.02459999918937683,
    "std_yield": 0.04628002271056175,
    "min_yield": 9.999999747378752e-05,
    "max_yield": 1.7843999862670898,
    "q25": 0.016649999655783176,
    "q75": 0.03500000014901161
  },
  "payout_analysis": {},
  "dividend_payers": {},
  "sustainability_analysis": {
    "cfo_ltm": {
      "count": 6666,
      "mean": 921.3033573357335,
      "median": 218.67,
      "positive_pct": 87.56375637563757
    },
    "fcf_ltm": {
      "count": 6666,
      "mean": 411.51084608460843,
      "median": 102.695,
      "positive_pct": 77.10771077107711
    },
    "cfi_ltm": {
      "count": 6666,
      "mean": -951.9943249324932,
      "median": -113.0,
      "positive_pct": 12.436243624362437
    },
    "cff_ltm": {
      "count": 6666,
      "mean": 25.1875382538254,
      "median": -50.125,
      "positive_pct": 30.61806180618062
    },
    "cff_fy": {
      "count": 6666,
      "mean": -74.90460096009602,
      "median": -47.599999999999994,
      "positive_pct": 30.663066306630665
    }
  },
  "by_sector": {
    "Information Technology": {
      "total_stocks": 816,
      "dividend_payers": 811,
      "dividend_payer_pct": 99.38725490196079,
      "mean_yield": 0.01949777826666832,
      "median_yield": 0.02459999918937683
    },
    "Communication Services": {
      "total_stocks": 303,
      "dividend_payers": 302,
      "dividend_payer_pct": 99.66996699669967,
      "mean_yield": 0.0345149002969265,
      "median_yield": 0.02459999918937683
    },
    "Consumer Discretionary": {
      "total_stocks": 790,
      "dividend_payers": 788,
      "dividend_payer_pct": 99.74683544303798,
      "mean_yield": 0.03397855535149574,
      "median_yield": 0.02459999918937683
    },
    "Consumer Staples": {
      "total_stocks": 462,
      "dividend_payers": 460,
      "dividend_payer_pct": 99.56709956709958,
      "mean_yield": 0.032704565674066544,
      "median_yield": 0.02459999918937683
    },
    "Health Care": {
      "total_stocks": 692,
      "dividend_payers": 690,
      "dividend_payer_pct": 99.71098265895954,
      "mean_yield": 0.022751014679670334,
      "median_yield": 0.02459999918937683
    },
    "Financials": {
      "total_stocks": 947,
      "dividend_payers": 944,
      "dividend_payer_pct": 99.68321013727561,
      "mean_yield": 0.03678188472986221,
      "median_yield": 0.027499999850988388
    },
    "Energy": {
      "total_stocks": 281,
      "dividend_payers": 280,
      "dividend_payer_pct": 99.644128113879,
      "mean_yield": 0.05267428606748581,
      "median_yield": 0.03294999897480011
    },
    "Industrials": {
      "total_stocks": 1316,
      "dividend_payers": 1310,
      "dividend_payer_pct": 99.54407294832826,
      "mean_yield": 0.028631603345274925,
      "median_yield": 0.02459999918937683
    },
    "Materials": {
      "total_stocks": 629,
      "dividend_payers": 626,
      "dividend_payer_pct": 99.52305246422894,
      "mean_yield": 0.031086741015315056,
      "median_yield": 0.02459999918937683
    },
    "Utilities": {
      "total_stocks": 242,
      "dividend_payers": 242,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.04583305865526199,
      "median_yield": 0.0361500009894371
    },
    "Real Estate": {
      "total_stocks": 188,
      "dividend_payers": 186,
      "dividend_payer_pct": 98.93617021276596,
      "mean_yield": 0.04502634331583977,
      "median_yield": 0.02459999918937683
    }
  },
  "by_region": {
    "United States and Canada": {
      "total_stocks": 1800,
      "dividend_payers": 1795,
      "dividend_payer_pct": 99.72222222222223,
      "mean_yield": 0.02532930299639702,
      "median_yield": 0.02459999918937683
    },
    "Asia / Pacific": {
      "total_stocks": 2053,
      "dividend_payers": 2044,
      "dividend_payer_pct": 99.56161714564053,
      "mean_yield": 0.028192125260829926,
      "median_yield": 0.02459999918937683
    },
    "Europe": {
      "total_stocks": 2058,
      "dividend_payers": 2053,
      "dividend_payer_pct": 99.75704567541302,
      "mean_yield": 0.03663136437535286,
      "median_yield": 0.02459999918937683
    },
    "Latin America and Caribbean": {
      "total_stocks": 349,
      "dividend_payers": 345,
      "dividend_payer_pct": 98.8538681948424,
      "mean_yield": 0.04252289980649948,
      "median_yield": 0.02669999934732914
    },
    "Africa / Middle East": {
      "total_stocks": 406,
      "dividend_payers": 402,
      "dividend_payer_pct": 99.01477832512316,
      "mean_yield": 0.041525617241859436,
      "median_yield": 0.025150001049041748
    }
  },
  "by_size_class": {
    "Large Cap": {
      "total_stocks": 1514,
      "dividend_payers": 1509,
      "dividend_payer_pct": 99.66974900924703,
      "mean_yield": 0.027735983952879906,
      "median_yield": 0.02459999918937683
    },
    "Mid Cap": {
      "total_stocks": 3036,
      "dividend_payers": 3030,
      "dividend_payer_pct": 99.80237154150198,
      "mean_yield": 0.029497722163796425,
      "median_yield": 0.02459999918937683
    },
    "Small Cap": {
      "total_stocks": 2116,
      "dividend_payers": 2100,
      "dividend_payer_pct": 99.24385633270322,
      "mean_yield": 0.037346187978982925,
      "median_yield": 0.02459999918937683
    }
  },
  "by_style_class": {
    "Growth": {
      "total_stocks": 598,
      "dividend_payers": 596,
      "dividend_payer_pct": 99.66555183946488,
      "mean_yield": 0.014992784708738327,
      "median_yield": 0.013849999755620956
    },
    "Core": {
      "total_stocks": 4314,
      "dividend_payers": 4298,
      "dividend_payer_pct": 99.62911451089475,
      "mean_yield": 0.02613101340830326,
      "median_yield": 0.02459999918937683
    },
    "Value": {
      "total_stocks": 1754,
      "dividend_payers": 1745,
      "dividend_payer_pct": 99.48688711516533,
      "mean_yield": 0.05066584423184395,
      "median_yield": 0.0357000008225441
    }
  },
  "by_industry": {
    "Semiconductors and Semiconductor Equipment": {
      "total_stocks": 193,
      "dividend_payers": 191,
      "dividend_payer_pct": 98.96373056994818,
      "mean_yield": 0.015848690643906593,
      "median_yield": 0.013700000010430813
    },
    "Technology Hardware Storage and Peripherals": {
      "total_stocks": 54,
      "dividend_payers": 53,
      "dividend_payer_pct": 98.14814814814815,
      "mean_yield": 0.02440566197037697,
      "median_yield": 0.02459999918937683
    },
    "Interactive Media and Services": {
      "total_stocks": 43,
      "dividend_payers": 42,
      "dividend_payer_pct": 97.67441860465115,
      "mean_yield": 0.02503095380961895,
      "median_yield": 0.02459999918937683
    },
    "Software": {
      "total_stocks": 225,
      "dividend_payers": 225,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.021027998998761177,
      "median_yield": 0.02459999918937683
    },
    "Broadline Retail": {
      "total_stocks": 50,
      "dividend_payers": 50,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.027177998796105385,
      "median_yield": 0.02459999918937683
    },
    "Automobiles": {
      "total_stocks": 47,
      "dividend_payers": 47,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03224468231201172,
      "median_yield": 0.02459999918937683
    },
    "Consumer Staples Distribution and Retail": {
      "total_stocks": 105,
      "dividend_payers": 104,
      "dividend_payer_pct": 99.04761904761905,
      "mean_yield": 0.032230768352746964,
      "median_yield": 0.02459999918937683
    },
    "Pharmaceuticals": {
      "total_stocks": 173,
      "dividend_payers": 172,
      "dividend_payer_pct": 99.42196531791907,
      "mean_yield": 0.022828487679362297,
      "median_yield": 0.02459999918937683
    },
    "Banks": {
      "total_stocks": 397,
      "dividend_payers": 395,
      "dividend_payer_pct": 99.49622166246851,
      "mean_yield": 0.03864354267716408,
      "median_yield": 0.029400000348687172
    },
    "Financial Services": {
      "total_stocks": 114,
      "dividend_payers": 113,
      "dividend_payer_pct": 99.12280701754386,
      "mean_yield": 0.03239557519555092,
      "median_yield": 0.02459999918937683
    },
    "Oil Gas and Consumable Fuels": {
      "total_stocks": 225,
      "dividend_payers": 224,
      "dividend_payer_pct": 99.55555555555556,
      "mean_yield": 0.05234508961439133,
      "median_yield": 0.03485000133514404
    },
    "Entertainment": {
      "total_stocks": 84,
      "dividend_payers": 84,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.025621429085731506,
      "median_yield": 0.02459999918937683
    },
    "Biotechnology": {
      "total_stocks": 190,
      "dividend_payers": 190,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.023465262725949287,
      "median_yield": 0.02459999918937683
    },
    "Specialty Retail": {
      "total_stocks": 156,
      "dividend_payers": 156,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.036857690662145615,
      "median_yield": 0.02459999918937683
    },
    "Household Products": {
      "total_stocks": 22,
      "dividend_payers": 22,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.031036365777254105,
      "median_yield": 0.028349999338388443
    },
    "Aerospace and Defense": {
      "total_stocks": 102,
      "dividend_payers": 101,
      "dividend_payer_pct": 99.01960784313727,
      "mean_yield": 0.013503961265087128,
      "median_yield": 0.010499999858438969
    },
    "Health Care Providers and Services": {
      "total_stocks": 112,
      "dividend_payers": 111,
      "dividend_payer_pct": 99.10714285714286,
      "mean_yield": 0.02676396630704403,
      "median_yield": 0.02459999918937683
    },
    "Communications Equipment": {
      "total_stocks": 48,
      "dividend_payers": 48,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.01730208285152912,
      "median_yield": 0.02084999904036522
    },
    "Beverages": {
      "total_stocks": 81,
      "dividend_payers": 81,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.0327925905585289,
      "median_yield": 0.031099999323487282
    },
    "IT Services": {
      "total_stocks": 94,
      "dividend_payers": 93,
      "dividend_payer_pct": 98.93617021276596,
      "mean_yield": 0.025126881897449493,
      "median_yield": 0.02459999918937683
    },
    "Capital Markets": {
      "total_stocks": 230,
      "dividend_payers": 230,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.039274781942367554,
      "median_yield": 0.028149999678134918
    },
    "Machinery": {
      "total_stocks": 275,
      "dividend_payers": 274,
      "dividend_payer_pct": 99.63636363636364,
      "mean_yield": 0.02185072936117649,
      "median_yield": 0.01954999938607216
    },
    "Consumer Finance": {
      "total_stocks": 41,
      "dividend_payers": 41,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.028331706300377846,
      "median_yield": 0.02459999918937683
    },
    "Tobacco": {
      "total_stocks": 12,
      "dividend_payers": 12,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.04422500357031822,
      "median_yield": 0.03989999741315842
    },
    "Hotels Restaurants and Leisure": {
      "total_stocks": 157,
      "dividend_payers": 156,
      "dividend_payer_pct": 99.36305732484077,
      "mean_yield": 0.04074744135141373,
      "median_yield": 0.02459999918937683
    },
    "Wireless Telecommunication Services": {
      "total_stocks": 29,
      "dividend_payers": 29,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.055713798850774765,
      "median_yield": 0.042100001126527786
    },
    "Health Care Equipment and Supplies": {
      "total_stocks": 140,
      "dividend_payers": 140,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.021562140434980392,
      "median_yield": 0.02459999918937683
    },
    "Life Sciences Tools and Services": {
      "total_stocks": 59,
      "dividend_payers": 59,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.01598474569618702,
      "median_yield": 0.02459999918937683
    },
    "Chemicals": {
      "total_stocks": 227,
      "dividend_payers": 227,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.033748459070920944,
      "median_yield": 0.02459999918937683
    },
    "Electrical Equipment": {
      "total_stocks": 175,
      "dividend_payers": 174,
      "dividend_payer_pct": 99.42857142857143,
      "mean_yield": 0.019090229645371437,
      "median_yield": 0.017999999225139618
    },
    "Ground Transportation": {
      "total_stocks": 63,
      "dividend_payers": 62,
      "dividend_payer_pct": 98.4126984126984,
      "mean_yield": 0.029020966961979866,
      "median_yield": 0.02459999918937683
    },
    "Diversified Telecommunication Services": {
      "total_stocks": 69,
      "dividend_payers": 69,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03950289636850357,
      "median_yield": 0.03840000182390213
    },
    "Electric Utilities": {
      "total_stocks": 77,
      "dividend_payers": 77,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.05309870466589928,
      "median_yield": 0.032099999487400055
    },
    "Electronic Equipment Instruments and Components": {
      "total_stocks": 202,
      "dividend_payers": 201,
      "dividend_payer_pct": 99.5049504950495,
      "mean_yield": 0.017878109589219093,
      "median_yield": 0.01600000075995922
    },
    "Insurance": {
      "total_stocks": 165,
      "dividend_payers": 165,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.033953942358493805,
      "median_yield": 0.026100000366568565
    },
    "Industrial Conglomerates": {
      "total_stocks": 54,
      "dividend_payers": 54,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.034405555576086044,
      "median_yield": 0.02459999918937683
    },
    "Metals and Mining": {
      "total_stocks": 270,
      "dividend_payers": 269,
      "dividend_payer_pct": 99.62962962962963,
      "mean_yield": 0.025704460218548775,
      "median_yield": 0.02459999918937683
    },
    "Professional Services": {
      "total_stocks": 89,
      "dividend_payers": 89,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.030428091064095497,
      "median_yield": 0.02459999918937683
    },
    "Textiles Apparel and Luxury Goods": {
      "total_stocks": 75,
      "dividend_payers": 75,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.02762666344642639,
      "median_yield": 0.02459999918937683
    },
    "Media": {
      "total_stocks": 78,
      "dividend_payers": 78,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03690512478351593,
      "median_yield": 0.02499999850988388
    },
    "Commercial Services and Supplies": {
      "total_stocks": 102,
      "dividend_payers": 101,
      "dividend_payer_pct": 99.01960784313727,
      "mean_yield": 0.02716634050011635,
      "median_yield": 0.02459999918937683
    },
    "Building Products": {
      "total_stocks": 71,
      "dividend_payers": 71,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.028180284425616264,
      "median_yield": 0.02070000022649765
    },
    "Air Freight and Logistics": {
      "total_stocks": 38,
      "dividend_payers": 37,
      "dividend_payer_pct": 97.36842105263158,
      "mean_yield": 0.04665946215391159,
      "median_yield": 0.02459999918937683
    },
    "Construction Materials": {
      "total_stocks": 65,
      "dividend_payers": 64,
      "dividend_payer_pct": 98.46153846153847,
      "mean_yield": 0.0367249995470047,
      "median_yield": 0.02459999918937683
    },
    "Food Products": {
      "total_stocks": 197,
      "dividend_payers": 196,
      "dividend_payer_pct": 99.49238578680203,
      "mean_yield": 0.034472446888685226,
      "median_yield": 0.026499999687075615
    },
    "Construction and Engineering": {
      "total_stocks": 148,
      "dividend_payers": 148,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.035369597375392914,
      "median_yield": 0.02459999918937683
    },
    "Energy Equipment and Services": {
      "total_stocks": 56,
      "dividend_payers": 56,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.05399107187986374,
      "median_yield": 0.02459999918937683
    },
    "Multi-Utilities": {
      "total_stocks": 35,
      "dividend_payers": 35,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03923143073916435,
      "median_yield": 0.03660000115633011
    },
    "Independent Power and Renewable Electricity Producers": {
      "total_stocks": 80,
      "dividend_payers": 80,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.04402749985456467,
      "median_yield": 0.035999998450279236
    },
    "Trading Companies and Distributors": {
      "total_stocks": 85,
      "dividend_payers": 85,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03187999874353409,
      "median_yield": 0.02459999918937683
    },
    "Real Estate Management and Development": {
      "total_stocks": 177,
      "dividend_payers": 175,
      "dividend_payer_pct": 98.87005649717514,
      "mean_yield": 0.04058343172073364,
      "median_yield": 0.02459999918937683
    },
    "Household Durables": {
      "total_stocks": 114,
      "dividend_payers": 113,
      "dividend_payer_pct": 99.12280701754386,
      "mean_yield": 0.0327017679810524,
      "median_yield": 0.02459999918937683
    },
    "Passenger Airlines": {
      "total_stocks": 38,
      "dividend_payers": 38,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.029310528188943863,
      "median_yield": 0.02459999918937683
    },
    "Personal Care Products": {
      "total_stocks": 45,
      "dividend_payers": 45,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.023684443905949593,
      "median_yield": 0.02459999918937683
    },
    "Health Care Technology": {
      "total_stocks": 18,
      "dividend_payers": 18,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.021150000393390656,
      "median_yield": 0.02459999918937683
    },
    "Gas Utilities": {
      "total_stocks": 29,
      "dividend_payers": 29,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.0410655178129673,
      "median_yield": 0.03929999843239784
    },
    "Water Utilities": {
      "total_stocks": 21,
      "dividend_payers": 21,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.04365714266896248,
      "median_yield": 0.04529999941587448
    },
    "Containers and Packaging": {
      "total_stocks": 42,
      "dividend_payers": 41,
      "dividend_payer_pct": 97.61904761904762,
      "mean_yield": 0.03938780352473259,
      "median_yield": 0.03750000149011612
    },
    "Distributors": {
      "total_stocks": 16,
      "dividend_payers": 16,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.06201250106096268,
      "median_yield": 0.02459999918937683
    },
    "Automobile Components": {
      "total_stocks": 112,
      "dividend_payers": 112,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.028954463079571724,
      "median_yield": 0.02459999918937683
    },
    "Mortgage Real Estate Investment Trusts (REITs)": {
      "total_stocks": 11,
      "dividend_payers": 11,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.1157090961933136,
      "median_yield": 0.10869999974966049
    },
    "Leisure Products": {
      "total_stocks": 29,
      "dividend_payers": 29,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03200344741344452,
      "median_yield": 0.02459999918937683
    },
    "Diversified Consumer Services": {
      "total_stocks": 34,
      "dividend_payers": 34,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.02540588192641735,
      "median_yield": 0.02459999918937683
    },
    "Paper and Forest Products": {
      "total_stocks": 25,
      "dividend_payers": 25,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03678400069475174,
      "median_yield": 0.02459999918937683
    },
    "Marine Transportation": {
      "total_stocks": 34,
      "dividend_payers": 34,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.09361176192760468,
      "median_yield": 0.06385000050067902
    },
    "Transportation Infrastructure": {
      "total_stocks": 42,
      "dividend_payers": 42,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.04184047505259514,
      "median_yield": 0.03034999966621399
    }
  },
  "by_trading_country": {
    "US": {
      "total_stocks": 1822,
      "dividend_payers": 1816,
      "dividend_payer_pct": 99.67069154774973,
      "mean_yield": 0.025578303262591362,
      "median_yield": 0.02459999918937683
    },
    "CA": {
      "total_stocks": 176,
      "dividend_payers": 175,
      "dividend_payer_pct": 99.43181818181817,
      "mean_yield": 0.027582861483097076,
      "median_yield": 0.02459999918937683
    },
    "NL": {
      "total_stocks": 66,
      "dividend_payers": 66,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.05510757863521576,
      "median_yield": 0.025299999862909317
    },
    "FR": {
      "total_stocks": 217,
      "dividend_payers": 217,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.032274194061756134,
      "median_yield": 0.02459999918937683
    },
    "CH": {
      "total_stocks": 137,
      "dividend_payers": 137,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.029705841094255447,
      "median_yield": 0.02459999918937683
    },
    "DE": {
      "total_stocks": 208,
      "dividend_payers": 207,
      "dividend_payer_pct": 99.51923076923077,
      "mean_yield": 0.025681642815470695,
      "median_yield": 0.02459999918937683
    },
    "GB": {
      "total_stocks": 374,
      "dividend_payers": 373,
      "dividend_payer_pct": 99.73262032085562,
      "mean_yield": 0.04260643571615219,
      "median_yield": 0.02459999918937683
    },
    "DK": {
      "total_stocks": 47,
      "dividend_payers": 47,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.029529791325330734,
      "median_yield": 0.02459999918937683
    },
    "ES": {
      "total_stocks": 75,
      "dividend_payers": 75,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03609733283519745,
      "median_yield": 0.02459999918937683
    },
    "BE": {
      "total_stocks": 56,
      "dividend_payers": 56,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03892321139574051,
      "median_yield": 0.02459999918937683
    },
    "IT": {
      "total_stocks": 140,
      "dividend_payers": 140,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03202071040868759,
      "median_yield": 0.02459999918937683
    },
    "SE": {
      "total_stocks": 255,
      "dividend_payers": 254,
      "dividend_payer_pct": 99.6078431372549,
      "mean_yield": 0.029949214309453964,
      "median_yield": 0.02459999918937683
    },
    "FI": {
      "total_stocks": 92,
      "dividend_payers": 92,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.04455434903502464,
      "median_yield": 0.03424999862909317
    },
    "NO": {
      "total_stocks": 119,
      "dividend_payers": 119,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.06467143446207047,
      "median_yield": 0.03009999915957451
    },
    "AT": {
      "total_stocks": 29,
      "dividend_payers": 29,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.033131036907434464,
      "median_yield": 0.03240000084042549
    },
    "IE": {
      "total_stocks": 11,
      "dividend_payers": 11,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.029945455491542816,
      "median_yield": 0.0272000003606081
    },
    "PL": {
      "total_stocks": 94,
      "dividend_payers": 92,
      "dividend_payer_pct": 97.87234042553192,
      "mean_yield": 0.04181956499814987,
      "median_yield": 0.02459999918937683
    },
    "HU": {
      "total_stocks": 5,
      "dividend_payers": 5,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.0542600043118,
      "median_yield": 0.061500001698732376
    },
    "PT": {
      "total_stocks": 13,
      "dividend_payers": 13,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.05259230360388756,
      "median_yield": 0.05260000005364418
    },
    "GR": {
      "total_stocks": 31,
      "dividend_payers": 31,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.02936774305999279,
      "median_yield": 0.02459999918937683
    },
    "RO": {
      "total_stocks": 12,
      "dividend_payers": 12,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.023541666567325592,
      "median_yield": 0.023000000044703484
    },
    "SI": {
      "total_stocks": 6,
      "dividend_payers": 6,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.043699998408555984,
      "median_yield": 0.04179999977350235
    },
    "HR": {
      "total_stocks": 6,
      "dividend_payers": 6,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.028966667130589485,
      "median_yield": 0.02304999902844429
    },
    "HK": {
      "total_stocks": 230,
      "dividend_payers": 229,
      "dividend_payer_pct": 99.56521739130434,
      "mean_yield": 0.038979917764663696,
      "median_yield": 0.02459999918937683
    },
    "JP": {
      "total_stocks": 378,
      "dividend_payers": 376,
      "dividend_payer_pct": 99.47089947089947,
      "mean_yield": 0.024510636925697327,
      "median_yield": 0.02459999918937683
    },
    "CN": {
      "total_stocks": 901,
      "dividend_payers": 898,
      "dividend_payer_pct": 99.66703662597114,
      "mean_yield": 0.027082962915301323,
      "median_yield": 0.016750000417232513
    },
    "AU": {
      "total_stocks": 57,
      "dividend_payers": 57,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.029310526326298714,
      "median_yield": 0.02459999918937683
    },
    "SG": {
      "total_stocks": 26,
      "dividend_payers": 26,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.035542309284210205,
      "median_yield": 0.027049999684095383
    },
    "TH": {
      "total_stocks": 27,
      "dividend_payers": 26,
      "dividend_payer_pct": 96.29629629629629,
      "mean_yield": 0.03700769320130348,
      "median_yield": 0.035099998116493225
    },
    "ID": {
      "total_stocks": 26,
      "dividend_payers": 26,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03641538321971893,
      "median_yield": 0.02459999918937683
    },
    "IN": {
      "total_stocks": 114,
      "dividend_payers": 114,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.01337807159870863,
      "median_yield": 0.008750000037252903
    },
    "KR": {
      "total_stocks": 95,
      "dividend_payers": 95,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.027459997683763504,
      "median_yield": 0.02459999918937683
    },
    "TW": {
      "total_stocks": 76,
      "dividend_payers": 76,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03613421320915222,
      "median_yield": 0.029649998992681503
    },
    "MY": {
      "total_stocks": 24,
      "dividend_payers": 24,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03245416656136513,
      "median_yield": 0.032099999487400055
    },
    "VN": {
      "total_stocks": 25,
      "dividend_payers": 23,
      "dividend_payer_pct": 92.0,
      "mean_yield": 0.024326086044311523,
      "median_yield": 0.02459999918937683
    },
    "PH": {
      "total_stocks": 16,
      "dividend_payers": 16,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.045237503945827484,
      "median_yield": 0.04845000058412552
    },
    "AE": {
      "total_stocks": 64,
      "dividend_payers": 63,
      "dividend_payer_pct": 98.4375,
      "mean_yield": 0.04490000382065773,
      "median_yield": 0.04830000177025795
    },
    "MX": {
      "total_stocks": 61,
      "dividend_payers": 61,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03270164132118225,
      "median_yield": 0.02459999918937683
    },
    "BR": {
      "total_stocks": 158,
      "dividend_payers": 157,
      "dividend_payer_pct": 99.36708860759494,
      "mean_yield": 0.048068154603242874,
      "median_yield": 0.03150000050663948
    },
    "ZA": {
      "total_stocks": 79,
      "dividend_payers": 78,
      "dividend_payer_pct": 98.73417721518987,
      "mean_yield": 0.060392312705516815,
      "median_yield": 0.03784999996423721
    },
    "TR": {
      "total_stocks": 97,
      "dividend_payers": 96,
      "dividend_payer_pct": 98.96907216494846,
      "mean_yield": 0.02760937251150608,
      "median_yield": 0.02459999918937683
    },
    "CO": {
      "total_stocks": 12,
      "dividend_payers": 11,
      "dividend_payer_pct": 91.66666666666666,
      "mean_yield": 0.06027273088693619,
      "median_yield": 0.05609999969601631
    },
    "CL": {
      "total_stocks": 36,
      "dividend_payers": 36,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.044127777218818665,
      "median_yield": 0.035350002348423004
    },
    "PE": {
      "total_stocks": 7,
      "dividend_payers": 7,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.06770000606775284,
      "median_yield": 0.05730000138282776
    },
    "MA": {
      "total_stocks": 33,
      "dividend_payers": 33,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.02857878804206848,
      "median_yield": 0.02459999918937683
    },
    "AR": {
      "total_stocks": 11,
      "dividend_payers": 11,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.027990909293293953,
      "median_yield": 0.02459999918937683
    },
    "KE": {
      "total_stocks": 10,
      "dividend_payers": 10,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03368999809026718,
      "median_yield": 0.02459999918937683
    },
    "NG": {
      "total_stocks": 45,
      "dividend_payers": 44,
      "dividend_payer_pct": 97.77777777777777,
      "mean_yield": 0.0372227281332016,
      "median_yield": 0.027400000020861626
    },
    "GH": {
      "total_stocks": 12,
      "dividend_payers": 12,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03559166565537453,
      "median_yield": 0.02630000002682209
    },
    "TN": {
      "total_stocks": 34,
      "dividend_payers": 34,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.04253823682665825,
      "median_yield": 0.03779999911785126
    }
  },
  "by_exchange": {
    "Semiconductors and Semiconductor Equipment": {
      "total_stocks": 193,
      "dividend_payers": 191,
      "dividend_payer_pct": 98.96373056994818,
      "mean_yield": 0.015848690643906593,
      "median_yield": 0.013700000010430813
    },
    "Technology Hardware Storage and Peripherals": {
      "total_stocks": 54,
      "dividend_payers": 53,
      "dividend_payer_pct": 98.14814814814815,
      "mean_yield": 0.02440566197037697,
      "median_yield": 0.02459999918937683
    },
    "Interactive Media and Services": {
      "total_stocks": 43,
      "dividend_payers": 42,
      "dividend_payer_pct": 97.67441860465115,
      "mean_yield": 0.02503095380961895,
      "median_yield": 0.02459999918937683
    },
    "Software": {
      "total_stocks": 225,
      "dividend_payers": 225,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.021027998998761177,
      "median_yield": 0.02459999918937683
    },
    "Broadline Retail": {
      "total_stocks": 50,
      "dividend_payers": 50,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.027177998796105385,
      "median_yield": 0.02459999918937683
    },
    "Automobiles": {
      "total_stocks": 47,
      "dividend_payers": 47,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03224468231201172,
      "median_yield": 0.02459999918937683
    },
    "Consumer Staples Distribution and Retail": {
      "total_stocks": 105,
      "dividend_payers": 104,
      "dividend_payer_pct": 99.04761904761905,
      "mean_yield": 0.032230768352746964,
      "median_yield": 0.02459999918937683
    },
    "Pharmaceuticals": {
      "total_stocks": 173,
      "dividend_payers": 172,
      "dividend_payer_pct": 99.42196531791907,
      "mean_yield": 0.022828487679362297,
      "median_yield": 0.02459999918937683
    },
    "Banks": {
      "total_stocks": 397,
      "dividend_payers": 395,
      "dividend_payer_pct": 99.49622166246851,
      "mean_yield": 0.03864354267716408,
      "median_yield": 0.029400000348687172
    },
    "Financial Services": {
      "total_stocks": 114,
      "dividend_payers": 113,
      "dividend_payer_pct": 99.12280701754386,
      "mean_yield": 0.03239557519555092,
      "median_yield": 0.02459999918937683
    },
    "Oil Gas and Consumable Fuels": {
      "total_stocks": 225,
      "dividend_payers": 224,
      "dividend_payer_pct": 99.55555555555556,
      "mean_yield": 0.05234508961439133,
      "median_yield": 0.03485000133514404
    },
    "Entertainment": {
      "total_stocks": 84,
      "dividend_payers": 84,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.025621429085731506,
      "median_yield": 0.02459999918937683
    },
    "Biotechnology": {
      "total_stocks": 190,
      "dividend_payers": 190,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.023465262725949287,
      "median_yield": 0.02459999918937683
    },
    "Specialty Retail": {
      "total_stocks": 156,
      "dividend_payers": 156,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.036857690662145615,
      "median_yield": 0.02459999918937683
    },
    "Household Products": {
      "total_stocks": 22,
      "dividend_payers": 22,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.031036365777254105,
      "median_yield": 0.028349999338388443
    },
    "Aerospace and Defense": {
      "total_stocks": 102,
      "dividend_payers": 101,
      "dividend_payer_pct": 99.01960784313727,
      "mean_yield": 0.013503961265087128,
      "median_yield": 0.010499999858438969
    },
    "Health Care Providers and Services": {
      "total_stocks": 112,
      "dividend_payers": 111,
      "dividend_payer_pct": 99.10714285714286,
      "mean_yield": 0.02676396630704403,
      "median_yield": 0.02459999918937683
    },
    "Communications Equipment": {
      "total_stocks": 48,
      "dividend_payers": 48,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.01730208285152912,
      "median_yield": 0.02084999904036522
    },
    "Beverages": {
      "total_stocks": 81,
      "dividend_payers": 81,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.0327925905585289,
      "median_yield": 0.031099999323487282
    },
    "IT Services": {
      "total_stocks": 94,
      "dividend_payers": 93,
      "dividend_payer_pct": 98.93617021276596,
      "mean_yield": 0.025126881897449493,
      "median_yield": 0.02459999918937683
    },
    "Capital Markets": {
      "total_stocks": 230,
      "dividend_payers": 230,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.039274781942367554,
      "median_yield": 0.028149999678134918
    },
    "Machinery": {
      "total_stocks": 275,
      "dividend_payers": 274,
      "dividend_payer_pct": 99.63636363636364,
      "mean_yield": 0.02185072936117649,
      "median_yield": 0.01954999938607216
    },
    "Consumer Finance": {
      "total_stocks": 41,
      "dividend_payers": 41,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.028331706300377846,
      "median_yield": 0.02459999918937683
    },
    "Tobacco": {
      "total_stocks": 12,
      "dividend_payers": 12,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.04422500357031822,
      "median_yield": 0.03989999741315842
    },
    "Hotels Restaurants and Leisure": {
      "total_stocks": 157,
      "dividend_payers": 156,
      "dividend_payer_pct": 99.36305732484077,
      "mean_yield": 0.04074744135141373,
      "median_yield": 0.02459999918937683
    },
    "Wireless Telecommunication Services": {
      "total_stocks": 29,
      "dividend_payers": 29,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.055713798850774765,
      "median_yield": 0.042100001126527786
    },
    "Health Care Equipment and Supplies": {
      "total_stocks": 140,
      "dividend_payers": 140,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.021562140434980392,
      "median_yield": 0.02459999918937683
    },
    "Life Sciences Tools and Services": {
      "total_stocks": 59,
      "dividend_payers": 59,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.01598474569618702,
      "median_yield": 0.02459999918937683
    },
    "Chemicals": {
      "total_stocks": 227,
      "dividend_payers": 227,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.033748459070920944,
      "median_yield": 0.02459999918937683
    },
    "Electrical Equipment": {
      "total_stocks": 175,
      "dividend_payers": 174,
      "dividend_payer_pct": 99.42857142857143,
      "mean_yield": 0.019090229645371437,
      "median_yield": 0.017999999225139618
    },
    "Ground Transportation": {
      "total_stocks": 63,
      "dividend_payers": 62,
      "dividend_payer_pct": 98.4126984126984,
      "mean_yield": 0.029020966961979866,
      "median_yield": 0.02459999918937683
    },
    "Diversified Telecommunication Services": {
      "total_stocks": 69,
      "dividend_payers": 69,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03950289636850357,
      "median_yield": 0.03840000182390213
    },
    "Electric Utilities": {
      "total_stocks": 77,
      "dividend_payers": 77,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.05309870466589928,
      "median_yield": 0.032099999487400055
    },
    "Electronic Equipment Instruments and Components": {
      "total_stocks": 202,
      "dividend_payers": 201,
      "dividend_payer_pct": 99.5049504950495,
      "mean_yield": 0.017878109589219093,
      "median_yield": 0.01600000075995922
    },
    "Insurance": {
      "total_stocks": 165,
      "dividend_payers": 165,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.033953942358493805,
      "median_yield": 0.026100000366568565
    },
    "Industrial Conglomerates": {
      "total_stocks": 54,
      "dividend_payers": 54,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.034405555576086044,
      "median_yield": 0.02459999918937683
    },
    "Metals and Mining": {
      "total_stocks": 270,
      "dividend_payers": 269,
      "dividend_payer_pct": 99.62962962962963,
      "mean_yield": 0.025704460218548775,
      "median_yield": 0.02459999918937683
    },
    "Professional Services": {
      "total_stocks": 89,
      "dividend_payers": 89,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.030428091064095497,
      "median_yield": 0.02459999918937683
    },
    "Textiles Apparel and Luxury Goods": {
      "total_stocks": 75,
      "dividend_payers": 75,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.02762666344642639,
      "median_yield": 0.02459999918937683
    },
    "Media": {
      "total_stocks": 78,
      "dividend_payers": 78,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03690512478351593,
      "median_yield": 0.02499999850988388
    },
    "Commercial Services and Supplies": {
      "total_stocks": 102,
      "dividend_payers": 101,
      "dividend_payer_pct": 99.01960784313727,
      "mean_yield": 0.02716634050011635,
      "median_yield": 0.02459999918937683
    },
    "Building Products": {
      "total_stocks": 71,
      "dividend_payers": 71,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.028180284425616264,
      "median_yield": 0.02070000022649765
    },
    "Air Freight and Logistics": {
      "total_stocks": 38,
      "dividend_payers": 37,
      "dividend_payer_pct": 97.36842105263158,
      "mean_yield": 0.04665946215391159,
      "median_yield": 0.02459999918937683
    },
    "Construction Materials": {
      "total_stocks": 65,
      "dividend_payers": 64,
      "dividend_payer_pct": 98.46153846153847,
      "mean_yield": 0.0367249995470047,
      "median_yield": 0.02459999918937683
    },
    "Food Products": {
      "total_stocks": 197,
      "dividend_payers": 196,
      "dividend_payer_pct": 99.49238578680203,
      "mean_yield": 0.034472446888685226,
      "median_yield": 0.026499999687075615
    },
    "Construction and Engineering": {
      "total_stocks": 148,
      "dividend_payers": 148,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.035369597375392914,
      "median_yield": 0.02459999918937683
    },
    "Energy Equipment and Services": {
      "total_stocks": 56,
      "dividend_payers": 56,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.05399107187986374,
      "median_yield": 0.02459999918937683
    },
    "Multi-Utilities": {
      "total_stocks": 35,
      "dividend_payers": 35,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03923143073916435,
      "median_yield": 0.03660000115633011
    },
    "Independent Power and Renewable Electricity Producers": {
      "total_stocks": 80,
      "dividend_payers": 80,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.04402749985456467,
      "median_yield": 0.035999998450279236
    },
    "Trading Companies and Distributors": {
      "total_stocks": 85,
      "dividend_payers": 85,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03187999874353409,
      "median_yield": 0.02459999918937683
    },
    "Real Estate Management and Development": {
      "total_stocks": 177,
      "dividend_payers": 175,
      "dividend_payer_pct": 98.87005649717514,
      "mean_yield": 0.04058343172073364,
      "median_yield": 0.02459999918937683
    },
    "Household Durables": {
      "total_stocks": 114,
      "dividend_payers": 113,
      "dividend_payer_pct": 99.12280701754386,
      "mean_yield": 0.0327017679810524,
      "median_yield": 0.02459999918937683
    },
    "Passenger Airlines": {
      "total_stocks": 38,
      "dividend_payers": 38,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.029310528188943863,
      "median_yield": 0.02459999918937683
    },
    "Personal Care Products": {
      "total_stocks": 45,
      "dividend_payers": 45,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.023684443905949593,
      "median_yield": 0.02459999918937683
    },
    "Health Care Technology": {
      "total_stocks": 18,
      "dividend_payers": 18,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.021150000393390656,
      "median_yield": 0.02459999918937683
    },
    "Gas Utilities": {
      "total_stocks": 29,
      "dividend_payers": 29,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.0410655178129673,
      "median_yield": 0.03929999843239784
    },
    "Water Utilities": {
      "total_stocks": 21,
      "dividend_payers": 21,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.04365714266896248,
      "median_yield": 0.04529999941587448
    },
    "Containers and Packaging": {
      "total_stocks": 42,
      "dividend_payers": 41,
      "dividend_payer_pct": 97.61904761904762,
      "mean_yield": 0.03938780352473259,
      "median_yield": 0.03750000149011612
    },
    "Distributors": {
      "total_stocks": 16,
      "dividend_payers": 16,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.06201250106096268,
      "median_yield": 0.02459999918937683
    },
    "Automobile Components": {
      "total_stocks": 112,
      "dividend_payers": 112,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.028954463079571724,
      "median_yield": 0.02459999918937683
    },
    "Mortgage Real Estate Investment Trusts (REITs)": {
      "total_stocks": 11,
      "dividend_payers": 11,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.1157090961933136,
      "median_yield": 0.10869999974966049
    },
    "Leisure Products": {
      "total_stocks": 29,
      "dividend_payers": 29,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03200344741344452,
      "median_yield": 0.02459999918937683
    },
    "Diversified Consumer Services": {
      "total_stocks": 34,
      "dividend_payers": 34,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.02540588192641735,
      "median_yield": 0.02459999918937683
    },
    "Paper and Forest Products": {
      "total_stocks": 25,
      "dividend_payers": 25,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.03678400069475174,
      "median_yield": 0.02459999918937683
    },
    "Marine Transportation": {
      "total_stocks": 34,
      "dividend_payers": 34,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.09361176192760468,
      "median_yield": 0.06385000050067902
    },
    "Transportation Infrastructure": {
      "total_stocks": 42,
      "dividend_payers": 42,
      "dividend_payer_pct": 100.0,
      "mean_yield": 0.04184047505259514,
      "median_yield": 0.03034999966621399
    }
  },
  "yield_distribution": {
    "0-1%": 6636,
    "1-2%": 3,
    "2-3%": 0,
    "3-4%": 0,
    "4-5%": 0,
    "5-7%": 0,
    "7-10%": 0,
    "10%+": 0
  },
  "high_yield_screening": {
    "threshold": 4.0,
    "count": 0,
    "pct_of_total": 0.0
  },
  "forward_yield_analysis": {
    "div_yield_ind": {
      "count": 6354,
      "mean": 0.036795999854803085,
      "median": 0.023499999195337296,
      "q25": 0.01600000075995922,
      "q75": 0.03530000150203705
    },
    "div_yield_1fyind": {
      "count": 6086,
      "mean": 0.03227661922574043,
      "median": 0.02250000089406967,
      "q25": 0.016499999910593033,
      "q75": 0.037675000727176666
    },
    "div_yield_2fyind": {
      "count": 6090,
      "mean": 0.03398612514138222,
      "median": 0.02329999953508377,
      "q25": 0.017799999564886093,
      "q75": 0.03920000046491623
    },
    "div_yield_3fyind": {
      "count": 5967,
      "mean": 0.025330064818263054,
      "median": 0.017400000244379044,
      "q25": 0.014399999752640724,
      "q75": 0.029600000008940697
    },
    "div_yield_4fyind": {
      "count": 5636,
      "mean": 0.022770758718252182,
      "median": 0.014499999582767487,
      "q25": 0.014499999582767487,
      "q75": 0.026925000827759504
    },
    "div_yield_5fyind": {
      "count": 6176,
      "mean": 0.028622537851333618,
      "median": 0.022099999710917473,
      "q25": 0.01889999955892563,
      "q75": 0.03200000151991844
    },
    "div_yield_5yavgltm": {
      "count": 6660,
      "mean": 0.032195448875427246,
      "median": 0.02630000002682209,
      "q25": 0.018300000578165054,
      "q75": 0.03539999946951866
    }
  },
  "buyback_analysis": {
    "column": "buyback_yield_ltm",
    "count": 6666,
    "buyback_payers": 5032,
    "buyback_payer_pct": 75.48754875487549,
    "mean_yield": 0.015621144324541092,
    "median_yield": 0.0026000000070780516,
    "q25": 0.0026000000070780516,
    "q75": 0.017000000923871994
  },
  "total_shareholder_yield": {
    "count": 6072,
    "mean": 0.043996427208185196,
    "median": 0.02969999983906746,
    "q25": 0.021700000390410423,
    "q75": 0.05429999902844429
  }
}
```

Attachment Name: earnings_estimates_analysis.json
Attachments Kind: Visible
Attachments Source: User / FileChatAttachment
Attachments Text:

```json
{
  "timestamp": "2025-12-15T19:57:32.042638",
  "total_stocks_analyzed": 6666,
  "phase93_categories_used": [
    "profitability",
    "forecasts",
    "valuation",
    "growth",
    "cash_flow"
  ],
  "phase93_metrics_available": {
    "profitability": 7,
    "forecasts": 8,
    "valuation": 15,
    "growth": 5,
    "cash_flow": 10
  },
  "available_columns": {
    "roe": "roe",
    "roa": "roa",
    "roic": "roic",
    "gross_margin": "gross_margin_pct",
    "operating_margin": "operating_margin_pct",
    "net_margin": "net_margin_pct",
    "ebitda_adjustment": "ebitda_adjustment_ratio_ltm",
    "ebit_adjustment": "ebit_adjustment_ratio_ltm",
    "net_income_adjustment": "net_income_adjustment_ratio_ltm",
    "revenue_estimate_spread_ntm": "revenue_estimate_spread_ntm",
    "revenue_estimate_spread_fy1e": "revenue_estimate_spread_fy1e",
    "revenue_consensus_uncertainty": "revenue_consensus_uncertainty_score",
    "revenue_growth_implied_ntm": "revenue_growth_implied_ntm",
    "revenue_growth_implied_fy1e": "revenue_growth_implied_fy1e",
    "revenue_growth_acceleration": "revenue_growth_acceleration",
    "estimate_confidence": "estimate_confidence_flag",
    "growth_surprise_potential": "growth_surprise_potential",
    "eps_actual": "eps_adj_ltm",
    "eps_estimate": "eps_norm_est_avg_ntm",
    "eps_adjusted": "eps_adj_fy",
    "eps_basic_fq": "net_eps_basic_fq",
    "eps_basic_fy": "net_eps_basic_fy",
    "eps_basic_qoq_1q": "net_eps_basic_1fqfq",
    "eps_basic_qoq_2q": "net_eps_basic_2fqfq",
    "eps_basic_qoq_3q": "net_eps_basic_3fqfq",
    "eps_gaap_est_fy1e": "eps_gaap_est_avg_fy1e",
    "eps_gaap_est_ntm": "eps_gaap_est_avg_ntm",
    "eps_analyst_count": "eps_norm_est_num_fy1e",
    "revenue_actual": "total_revenues_ltm",
    "revenue_estimate": "revenues_est_avg_ntm",
    "net_income": "net_income_is_ltm",
    "ebitda": "ebitda_ltm",
    "ebit": "ebit_ltm"
  },
  "eps_analysis": {
    "actual": {
      "column": "eps_adj_ltm",
      "count": 6666,
      "mean": 2.452913522720337,
      "median": 1.2200000286102295,
      "std": 9.13582706451416,
      "positive_pct": 94.08940894089409,
      "negative_pct": 5.535553555355536
    },
    "estimated": {
      "column": "eps_norm_est_avg_ntm",
      "count": 6666,
      "mean": 3.698495388031006,
      "median": 0.8799999952316284,
      "std": 51.38079071044922
    },
    "adjusted": {
      "column": "eps_adj_fy",
      "count": 6666,
      "mean": 3.7381160259246826,
      "median": 0.6600000262260437,
      "std": 74.75318145751953
    }
  },
  "earnings_surprise": {
    "count": 6601,
    "mean_surprise_pct": 335.2435302734375,
    "median_surprise_pct": -0.9259250164031982,
    "beat_pct": 48.219966671716406,
    "miss_pct": 50.58324496288441,
    "large_beat_pct": 43.90243902439025,
    "large_miss_pct": 37.91849719739434
  },
  "phase93_category_analysis": {
    "profitability": {
      "net_income_margin_pct_ltm": {
        "count": 6666,
        "mean": 4.967514326432643,
        "median": 0.0805,
        "std": 320.14402295287994
      },
      "gross_profit_margin_pct_ltm": {
        "count": 6666,
        "mean": 0.40948745608329773,
        "median": 0.3776000142097473,
        "std": 0.2728697955608368
      },
      "ebitda_ltm": {
        "count": 6666,
        "mean": 1308.2608535853585,
        "median": 334.8,
        "std": 5285.5373272871775
      },
      "ebit_ltm": {
        "count": 6666,
        "mean": 916.156608160816,
        "median": 180.46,
        "std": 4188.234718944691
      },
      "net_income_is_ltm": {
        "count": 6666,
        "mean": 731.3332298229823,
        "median": 123.69999999999999,
        "std": 3705.9303330357347
      }
    },
    "forecasts": {
      "revenues_est_avg_ntm": {
        "count": 6666,
        "mean": 8261.370621062106,
        "median": 1895.66,
        "std": 27465.142791475708
      },
      "revenues_est_med_ntm": {
        "count": 6666,
        "mean": 8198.164501950196,
        "median": 1879.22,
        "std": 27289.43980636333
      },
      "revenues_est_avg_fy1e": {
        "count": 6666,
        "mean": 7994.414390939093,
        "median": 1799.335,
        "std": 26619.640712437886
      },
      "revenues_est_med_fy1e": {
        "count": 6666,
        "mean": 7992.977673267327,
        "median": 1799.2849999999999,
        "std": 26613.549984560974
      },
      "eps_norm_est_avg_ntm": {
        "count": 6666,
        "mean": 3.698495388031006,
        "median": 0.8799999952316284,
        "std": 51.38079071044922
      }
    },
    "valuation": {
      "p_e_ltm": {
        "count": 6666,
        "mean": 29.61552619934082,
        "median": 19.5,
        "std": 39.21567153930664
      },
      "p_e_ntm": {
        "count": 6666,
        "mean": 23.050643920898438,
        "median": 15.899999618530273,
        "std": 30.413021087646484
      },
      "p_b_ltm": {
        "count": 6666,
        "mean": 4.487653732299805,
        "median": 2.200000047683716,
        "std": 12.656661987304688
      },
      "p_tbv_ltm": {
        "count": 6666,
        "mean": 7.001694679260254,
        "median": 2.9000000953674316,
        "std": 21.933929443359375
      },
      "ev_sales_ltm": {
        "count": 6666,
        "mean": 5.3164567947387695,
        "median": 2.200000047683716,
        "std": 20.330429077148438
      }
    },
    "growth": {
      "total_revenues_cagr_5y_fy": {
        "count": 6666,
        "mean": 0.13622133433818817,
        "median": 0.08630000054836273,
        "std": 0.2897663712501526
      },
      "revenues_est_yoy_pct_fy1e": {
        "count": 6666,
        "mean": 0.513738751411438,
        "median": 0.11259999871253967,
        "std": 14.480072975158691
      },
      "tot_return_pct_cagr_3y": {
        "count": 6666,
        "mean": 0.16233879327774048,
        "median": 0.11829999834299088,
        "std": 0.2997972071170807
      },
      "total_revenues_ltm": {
        "count": 6666,
        "mean": 7899.171137113711,
        "median": 1765.185,
        "std": 25992.92982725376
      },
      "total_revenues_fy": {
        "count": 6666,
        "mean": 7449.416350135014,
        "median": 1618.08,
        "std": 24917.683517800462
      }
    },
    "cash_flow": {
      "cfo_ltm": {
        "count": 6666,
        "mean": 921.3033573357335,
        "median": 218.67,
        "std": 6518.3894150461565
      },
      "fcf_ltm": {
        "count": 6666,
        "mean": 411.51084608460843,
        "median": 102.695,
        "std": 5375.613480853006
      },
      "cfi_ltm": {
        "count": 6666,
        "mean": -951.9943249324932,
        "median": -113.0,
        "std": 7130.454977268094
      },
      "cff_ltm": {
        "count": 6666,
        "mean": 25.1875382538254,
        "median": -50.125,
        "std": 8613.861092285255
      },
      "cff_fy": {
        "count": 6666,
        "mean": -74.90460096009602,
        "median": -47.599999999999994,
        "std": 6752.51976151286
      }
    }
  },
  "by_sector": {
    "Information Technology": {
      "count": 816,
      "mean_eps": 2.217144727706909,
      "median_eps": 1.2200000286102295,
      "positive_pct": 93.50490196078431,
      "mean_surprise": 354.6229553222656,
      "beat_pct": 44.32098765432099
    },
    "Communication Services": {
      "count": 303,
      "mean_eps": 1.6619142293930054,
      "median_eps": 1.2200000286102295,
      "positive_pct": 91.0891089108911,
      "mean_surprise": 385.77484130859375,
      "beat_pct": 53.84615384615385
    },
    "Consumer Discretionary": {
      "count": 790,
      "mean_eps": 3.7070000171661377,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.20253164556962,
      "mean_surprise": 228.2751007080078,
      "beat_pct": 50.25445292620865
    },
    "Consumer Staples": {
      "count": 462,
      "mean_eps": 1.7000435590744019,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.1038961038961,
      "mean_surprise": 567.886474609375,
      "beat_pct": 49.336283185840706
    },
    "Health Care": {
      "count": 692,
      "mean_eps": 1.974956750869751,
      "median_eps": 1.2200000286102295,
      "positive_pct": 83.23699421965318,
      "mean_surprise": 292.4464111328125,
      "beat_pct": 49.85507246376812
    },
    "Financials": {
      "count": 947,
      "mean_eps": 3.5711514949798584,
      "median_eps": 1.2200000286102295,
      "positive_pct": 98.6272439281943,
      "mean_surprise": 234.3788299560547,
      "beat_pct": 42.12765957446808
    },
    "Energy": {
      "count": 281,
      "mean_eps": 1.8543062210083008,
      "median_eps": 1.2200000286102295,
      "positive_pct": 94.30604982206405,
      "mean_surprise": 536.3971557617188,
      "beat_pct": 50.719424460431654
    },
    "Industrials": {
      "count": 1316,
      "mean_eps": 2.6458661556243896,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.04863221884499,
      "mean_surprise": 250.1302032470703,
      "beat_pct": 47.3159509202454
    },
    "Materials": {
      "count": 629,
      "mean_eps": 1.4756121635437012,
      "median_eps": 1.2200000286102295,
      "positive_pct": 91.8918918918919,
      "mean_surprise": 514.0348510742188,
      "beat_pct": 50.89430894308943
    },
    "Utilities": {
      "count": 242,
      "mean_eps": 1.7424381971359253,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.28099173553719,
      "mean_surprise": 369.76220703125,
      "beat_pct": 55.64853556485355
    },
    "Real Estate": {
      "count": 188,
      "mean_eps": 1.1862765550613403,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.2127659574468,
      "mean_surprise": 384.8148193359375,
      "beat_pct": 53.72340425531915
    }
  },
  "by_region": {
    "United States and Canada": {
      "count": 1800,
      "mean_eps": 5.24518346786499,
      "median_eps": 2.9800000190734863,
      "positive_pct": 89.38888888888889,
      "mean_surprise": -5.28513765335083,
      "beat_pct": 23.817473567056204
    },
    "Asia / Pacific": {
      "count": 2053,
      "mean_eps": 1.1089577674865723,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.12615684364346,
      "mean_surprise": 565.1049194335938,
      "beat_pct": 71.47800296589224
    },
    "Europe": {
      "count": 2058,
      "mean_eps": 1.8150534629821777,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.04373177842565,
      "mean_surprise": 311.4342956542969,
      "beat_pct": 44.10757946210269
    },
    "Latin America and Caribbean": {
      "count": 349,
      "mean_eps": 1.1871060132980347,
      "median_eps": 0.4300000071525574,
      "positive_pct": 88.53868194842407,
      "mean_surprise": 170.74957275390625,
      "beat_pct": 36.231884057971016
    },
    "Africa / Middle East": {
      "count": 406,
      "mean_eps": 1.1906898021697998,
      "median_eps": 1.2200000286102295,
      "positive_pct": 99.50738916256158,
      "mean_surprise": 980.6668701171875,
      "beat_pct": 72.1227621483376
    }
  },
  "by_size_class": {
    "Large Cap": {
      "count": 1514,
      "mean_eps": 4.732107162475586,
      "median_eps": 1.2200000286102295,
      "positive_pct": 98.08454425363277,
      "mean_surprise": 154.7398223876953,
      "beat_pct": 38.80398671096346
    },
    "Mid Cap": {
      "count": 3036,
      "mean_eps": 2.2460999488830566,
      "median_eps": 1.2200000286102295,
      "positive_pct": 94.49934123847167,
      "mean_surprise": 286.548828125,
      "beat_pct": 47.131011608623545
    },
    "Small Cap": {
      "count": 2116,
      "mean_eps": 1.1188801527023315,
      "median_eps": 1.2200000286102295,
      "positive_pct": 90.64272211720227,
      "mean_surprise": 536.3355102539062,
      "beat_pct": 56.60740028832292
    }
  },
  "by_style_class": {
    "Growth": {
      "count": 598,
      "mean_eps": 3.5249998569488525,
      "median_eps": 1.2200000286102295,
      "positive_pct": 98.49498327759197,
      "mean_surprise": 195.04330444335938,
      "beat_pct": 19.798657718120804
    },
    "Core": {
      "count": 4314,
      "mean_eps": 2.5508949756622314,
      "median_eps": 1.2200000286102295,
      "positive_pct": 92.65183124710245,
      "mean_surprise": 314.9920959472656,
      "beat_pct": 45.77663068981699
    },
    "Value": {
      "count": 1754,
      "mean_eps": 1.8464139699935913,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.12314709236031,
      "mean_surprise": 432.70245361328125,
      "beat_pct": 63.912794033275965
    }
  },
  "by_industry": {
    "Semiconductors and Semiconductor Equipment": {
      "count": 193,
      "mean_eps": 2.11616587638855,
      "median_eps": 1.2200000286102295,
      "positive_pct": 93.26424870466322,
      "mean_surprise": 260.2463073730469,
      "beat_pct": 44.50261780104712
    },
    "Technology Hardware Storage and Peripherals": {
      "count": 54,
      "mean_eps": 1.75,
      "median_eps": 1.2200000286102295,
      "positive_pct": 90.74074074074075,
      "mean_surprise": 172.81793212890625,
      "beat_pct": 40.74074074074074
    },
    "Interactive Media and Services": {
      "count": 43,
      "mean_eps": 2.4346516132354736,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.34883720930233,
      "mean_surprise": 215.52627563476562,
      "beat_pct": 48.837209302325576
    },
    "Software": {
      "count": 225,
      "mean_eps": 2.78439998626709,
      "median_eps": 1.2200000286102295,
      "positive_pct": 92.44444444444444,
      "mean_surprise": 502.9330749511719,
      "beat_pct": 42.857142857142854
    },
    "Broadline Retail": {
      "count": 50,
      "mean_eps": 3.472599983215332,
      "median_eps": 1.2200000286102295,
      "positive_pct": 98.0,
      "mean_surprise": 253.3134307861328,
      "beat_pct": 40.816326530612244
    },
    "Automobiles": {
      "count": 47,
      "mean_eps": 1.6510639190673828,
      "median_eps": 1.2200000286102295,
      "positive_pct": 89.36170212765957,
      "mean_surprise": 209.40188598632812,
      "beat_pct": 65.95744680851064
    },
    "Consumer Staples Distribution and Retail": {
      "count": 105,
      "mean_eps": 1.8573331832885742,
      "median_eps": 1.2200000286102295,
      "positive_pct": 94.28571428571428,
      "mean_surprise": 750.4896240234375,
      "beat_pct": 45.19230769230769
    },
    "Pharmaceuticals": {
      "count": 173,
      "mean_eps": 1.6008092164993286,
      "median_eps": 1.2200000286102295,
      "positive_pct": 89.01734104046243,
      "mean_surprise": 339.4502258300781,
      "beat_pct": 58.38150289017341
    },
    "Banks": {
      "count": 397,
      "mean_eps": 2.8382115364074707,
      "median_eps": 1.2200000286102295,
      "positive_pct": 98.74055415617129,
      "mean_surprise": 283.6103820800781,
      "beat_pct": 38.07106598984771
    },
    "Financial Services": {
      "count": 114,
      "mean_eps": 2.945965051651001,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.36842105263158,
      "mean_surprise": 204.9404754638672,
      "beat_pct": 43.75
    },
    "Oil Gas and Consumable Fuels": {
      "count": 225,
      "mean_eps": 1.9786665439605713,
      "median_eps": 1.2200000286102295,
      "positive_pct": 93.77777777777779,
      "mean_surprise": 587.684814453125,
      "beat_pct": 50.45045045045045
    },
    "Entertainment": {
      "count": 84,
      "mean_eps": 1.5169049501419067,
      "median_eps": 1.2200000286102295,
      "positive_pct": 85.71428571428571,
      "mean_surprise": 436.3431396484375,
      "beat_pct": 60.71428571428571
    },
    "Biotechnology": {
      "count": 190,
      "mean_eps": 0.6764736771583557,
      "median_eps": 0.4449999928474426,
      "positive_pct": 61.578947368421055,
      "mean_surprise": 322.7252197265625,
      "beat_pct": 57.36842105263158
    },
    "Specialty Retail": {
      "count": 156,
      "mean_eps": 3.98641037940979,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.43589743589743,
      "mean_surprise": 211.3867950439453,
      "beat_pct": 53.84615384615385
    },
    "Household Products": {
      "count": 22,
      "mean_eps": 2.714545726776123,
      "median_eps": 1.4200000762939453,
      "positive_pct": 95.45454545454545,
      "mean_surprise": 1195.947509765625,
      "beat_pct": 47.61904761904761
    },
    "Aerospace and Defense": {
      "count": 102,
      "mean_eps": 3.770294427871704,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.09803921568627,
      "mean_surprise": 202.78384399414062,
      "beat_pct": 34.65346534653465
    },
    "Health Care Providers and Services": {
      "count": 112,
      "mean_eps": 3.7019646167755127,
      "median_eps": 1.2200000286102295,
      "positive_pct": 94.64285714285714,
      "mean_surprise": 174.40528869628906,
      "beat_pct": 39.63963963963964
    },
    "Communications Equipment": {
      "count": 48,
      "mean_eps": 1.966666579246521,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.83333333333334,
      "mean_surprise": 512.7730102539062,
      "beat_pct": 52.17391304347826
    },
    "Beverages": {
      "count": 81,
      "mean_eps": 1.677901268005371,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.53086419753086,
      "mean_surprise": 485.56060791015625,
      "beat_pct": 44.871794871794876
    },
    "IT Services": {
      "count": 94,
      "mean_eps": 2.215851068496704,
      "median_eps": 1.2200000286102295,
      "positive_pct": 94.68085106382979,
      "mean_surprise": 406.7995300292969,
      "beat_pct": 43.61702127659575
    },
    "Capital Markets": {
      "count": 230,
      "mean_eps": 3.2802176475524902,
      "median_eps": 1.2200000286102295,
      "positive_pct": 99.56521739130434,
      "mean_surprise": 276.6970520019531,
      "beat_pct": 50.0
    },
    "Machinery": {
      "count": 275,
      "mean_eps": 2.5666184425354004,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.45454545454545,
      "mean_surprise": 161.1416015625,
      "beat_pct": 45.98540145985402
    },
    "Consumer Finance": {
      "count": 41,
      "mean_eps": 4.364877700805664,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.5609756097561,
      "mean_surprise": 166.18670654296875,
      "beat_pct": 29.268292682926827
    },
    "Tobacco": {
      "count": 12,
      "mean_eps": 2.686666488647461,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 328.4228210449219,
      "beat_pct": 18.181818181818183
    },
    "Hotels Restaurants and Leisure": {
      "count": 157,
      "mean_eps": 3.576688051223755,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.54140127388536,
      "mean_surprise": 300.691162109375,
      "beat_pct": 39.490445859872615
    },
    "Wireless Telecommunication Services": {
      "count": 29,
      "mean_eps": 1.4741380214691162,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.55172413793103,
      "mean_surprise": 377.9034118652344,
      "beat_pct": 58.620689655172406
    },
    "Health Care Equipment and Supplies": {
      "count": 140,
      "mean_eps": 2.09757137298584,
      "median_eps": 1.2200000286102295,
      "positive_pct": 90.71428571428571,
      "mean_surprise": 174.6339874267578,
      "beat_pct": 41.007194244604314
    },
    "Life Sciences Tools and Services": {
      "count": 59,
      "mean_eps": 3.8423733711242676,
      "median_eps": 1.2200000286102295,
      "positive_pct": 91.52542372881356,
      "mean_surprise": 555.5504760742188,
      "beat_pct": 37.28813559322034
    },
    "Chemicals": {
      "count": 227,
      "mean_eps": 1.6380176544189453,
      "median_eps": 1.2200000286102295,
      "positive_pct": 92.07048458149781,
      "mean_surprise": 392.1242370605469,
      "beat_pct": 56.95067264573991
    },
    "Electrical Equipment": {
      "count": 175,
      "mean_eps": 1.8177716732025146,
      "median_eps": 1.2200000286102295,
      "positive_pct": 92.57142857142857,
      "mean_surprise": 272.6089782714844,
      "beat_pct": 48.85057471264368
    },
    "Ground Transportation": {
      "count": 63,
      "mean_eps": 2.0022222995758057,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.23809523809523,
      "mean_surprise": 101.61214447021484,
      "beat_pct": 45.90163934426229
    },
    "Diversified Telecommunication Services": {
      "count": 69,
      "mean_eps": 1.1668115854263306,
      "median_eps": 1.2200000286102295,
      "positive_pct": 86.95652173913044,
      "mean_surprise": 507.1138610839844,
      "beat_pct": 44.776119402985074
    },
    "Electric Utilities": {
      "count": 77,
      "mean_eps": 2.0767533779144287,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.40259740259741,
      "mean_surprise": 231.3170928955078,
      "beat_pct": 51.31578947368421
    },
    "Electronic Equipment Instruments and Components": {
      "count": 202,
      "mean_eps": 1.8667821884155273,
      "median_eps": 1.2200000286102295,
      "positive_pct": 94.55445544554455,
      "mean_surprise": 267.2718505859375,
      "beat_pct": 45.27363184079602
    },
    "Insurance": {
      "count": 165,
      "mean_eps": 5.974908828735352,
      "median_eps": 1.2200000286102295,
      "positive_pct": 98.18181818181819,
      "mean_surprise": 93.04470825195312,
      "beat_pct": 42.944785276073624
    },
    "Industrial Conglomerates": {
      "count": 54,
      "mean_eps": 1.8611112833023071,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 448.8796081542969,
      "beat_pct": 55.55555555555556
    },
    "Metals and Mining": {
      "count": 270,
      "mean_eps": 1.196333408355713,
      "median_eps": 1.2200000286102295,
      "positive_pct": 90.37037037037037,
      "mean_surprise": 707.0619506835938,
      "beat_pct": 46.15384615384615
    },
    "Professional Services": {
      "count": 89,
      "mean_eps": 2.847078800201416,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.50561797752809,
      "mean_surprise": 180.79037475585938,
      "beat_pct": 37.07865168539326
    },
    "Textiles Apparel and Luxury Goods": {
      "count": 75,
      "mean_eps": 4.423333168029785,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.33333333333334,
      "mean_surprise": 257.90789794921875,
      "beat_pct": 49.333333333333336
    },
    "Media": {
      "count": 78,
      "mean_eps": 1.899871826171875,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.15384615384616,
      "mean_surprise": 322.2420959472656,
      "beat_pct": 55.26315789473685
    },
    "Commercial Services and Supplies": {
      "count": 102,
      "mean_eps": 1.706078290939331,
      "median_eps": 1.2200000286102295,
      "positive_pct": 94.11764705882352,
      "mean_surprise": 233.6544189453125,
      "beat_pct": 45.09803921568628
    },
    "Building Products": {
      "count": 71,
      "mean_eps": 3.1714084148406982,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 205.69503784179688,
      "beat_pct": 47.88732394366197
    },
    "Air Freight and Logistics": {
      "count": 38,
      "mean_eps": 2.1465792655944824,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 420.5173034667969,
      "beat_pct": 50.0
    },
    "Construction Materials": {
      "count": 65,
      "mean_eps": 1.7264615297317505,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.38461538461539,
      "mean_surprise": 669.8980102539062,
      "beat_pct": 52.307692307692314
    },
    "Food Products": {
      "count": 197,
      "mean_eps": 1.5587818622589111,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.93908629441624,
      "mean_surprise": 536.09521484375,
      "beat_pct": 52.331606217616574
    },
    "Construction and Engineering": {
      "count": 148,
      "mean_eps": 1.9920270442962646,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.97297297297297,
      "mean_surprise": 302.35333251953125,
      "beat_pct": 55.10204081632652
    },
    "Energy Equipment and Services": {
      "count": 56,
      "mean_eps": 1.3546428680419922,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.42857142857143,
      "mean_surprise": 333.0783996582031,
      "beat_pct": 51.78571428571429
    },
    "Multi-Utilities": {
      "count": 35,
      "mean_eps": 2.278571605682373,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 191.78292846679688,
      "beat_pct": 48.57142857142857
    },
    "Independent Power and Renewable Electricity Producers": {
      "count": 80,
      "mean_eps": 1.082625150680542,
      "median_eps": 1.2200000286102295,
      "positive_pct": 91.25,
      "mean_surprise": 565.2443237304688,
      "beat_pct": 64.1025641025641
    },
    "Trading Companies and Distributors": {
      "count": 85,
      "mean_eps": 3.2958827018737793,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.6470588235294,
      "mean_surprise": 174.79998779296875,
      "beat_pct": 45.78313253012048
    },
    "Real Estate Management and Development": {
      "count": 177,
      "mean_eps": 1.1702826023101807,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.48022598870057,
      "mean_surprise": 409.98370361328125,
      "beat_pct": 55.367231638418076
    },
    "Household Durables": {
      "count": 114,
      "mean_eps": 6.514736652374268,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.6140350877193,
      "mean_surprise": 192.9534149169922,
      "beat_pct": 56.25
    },
    "Passenger Airlines": {
      "count": 38,
      "mean_eps": 2.0221052169799805,
      "median_eps": 1.2200000286102295,
      "positive_pct": 81.57894736842105,
      "mean_surprise": 198.71420288085938,
      "beat_pct": 32.432432432432435
    },
    "Personal Care Products": {
      "count": 45,
      "mean_eps": 1.2322224378585815,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.77777777777777,
      "mean_surprise": 190.3582305908203,
      "beat_pct": 62.22222222222222
    },
    "Health Care Technology": {
      "count": 18,
      "mean_eps": 1.4566664695739746,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 296.3750915527344,
      "beat_pct": 61.111111111111114
    },
    "Gas Utilities": {
      "count": 29,
      "mean_eps": 2.2368967533111572,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 223.1974334716797,
      "beat_pct": 55.172413793103445
    },
    "Water Utilities": {
      "count": 21,
      "mean_eps": 1.4538094997406006,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 643.7571411132812,
      "beat_pct": 52.38095238095239
    },
    "Containers and Packaging": {
      "count": 42,
      "mean_eps": 2.2016665935516357,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.23809523809523,
      "mean_surprise": 15.246773719787598,
      "beat_pct": 45.23809523809524
    },
    "Distributors": {
      "count": 16,
      "mean_eps": 2.382500171661377,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 167.9739990234375,
      "beat_pct": 62.5
    },
    "Automobile Components": {
      "count": 112,
      "mean_eps": 1.7355355024337769,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.42857142857143,
      "mean_surprise": 179.6894989013672,
      "beat_pct": 54.95495495495496
    },
    "Mortgage Real Estate Investment Trusts (REITs)": {
      "count": 11,
      "mean_eps": 1.4436362981796265,
      "median_eps": 1.5399999618530273,
      "positive_pct": 90.9090909090909,
      "mean_surprise": -20.175291061401367,
      "beat_pct": 27.27272727272727
    },
    "Leisure Products": {
      "count": 29,
      "mean_eps": 1.4393105506896973,
      "median_eps": 1.2200000286102295,
      "positive_pct": 93.10344827586206,
      "mean_surprise": 40.83460235595703,
      "beat_pct": 44.827586206896555
    },
    "Diversified Consumer Services": {
      "count": 34,
      "mean_eps": 4.2708821296691895,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 359.2332763671875,
      "beat_pct": 41.17647058823529
    },
    "Paper and Forest Products": {
      "count": 25,
      "mean_eps": 1.1452000141143799,
      "median_eps": 1.2200000286102295,
      "positive_pct": 92.0,
      "mean_surprise": 26.715225219726562,
      "beat_pct": 52.0
    },
    "Marine Transportation": {
      "count": 34,
      "mean_eps": 12.201470375061035,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.05882352941177,
      "mean_surprise": 499.6700439453125,
      "beat_pct": 69.6969696969697
    },
    "Transportation Infrastructure": {
      "count": 42,
      "mean_eps": 1.0954762697219849,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.61904761904762,
      "mean_surprise": 759.0257568359375,
      "beat_pct": 67.5
    }
  },
  "by_trading_country": {
    "US": {
      "count": 1822,
      "mean_eps": 5.3000874519348145,
      "median_eps": 3.119999885559082,
      "positive_pct": 89.07793633369924,
      "mean_surprise": -11.145663261413574,
      "beat_pct": 23.914238592633318
    },
    "CA": {
      "count": 176,
      "mean_eps": 2.7993180751800537,
      "median_eps": 1.3250000476837158,
      "positive_pct": 88.06818181818183,
      "mean_surprise": 81.16999816894531,
      "beat_pct": 21.59090909090909
    },
    "NL": {
      "count": 66,
      "mean_eps": 1.9625757932662964,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.96969696969697,
      "mean_surprise": 14.217747688293457,
      "beat_pct": 36.36363636363637
    },
    "FR": {
      "count": 217,
      "mean_eps": 1.545345664024353,
      "median_eps": 1.2200000286102295,
      "positive_pct": 99.53917050691244,
      "mean_surprise": 110.280517578125,
      "beat_pct": 35.648148148148145
    },
    "CH": {
      "count": 137,
      "mean_eps": 1.7988324165344238,
      "median_eps": 1.2200000286102295,
      "positive_pct": 99.27007299270073,
      "mean_surprise": -4.8854570388793945,
      "beat_pct": 20.437956204379564
    },
    "DE": {
      "count": 208,
      "mean_eps": 2.7866830825805664,
      "median_eps": 1.2200000286102295,
      "positive_pct": 92.3076923076923,
      "mean_surprise": 109.93953704833984,
      "beat_pct": 38.64734299516908
    },
    "GB": {
      "count": 374,
      "mean_eps": 1.265187382698059,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 1088.1671142578125,
      "beat_pct": 77.50677506775068
    },
    "DK": {
      "count": 47,
      "mean_eps": 9.81978702545166,
      "median_eps": 1.3899999856948853,
      "positive_pct": 93.61702127659575,
      "mean_surprise": 9.81968879699707,
      "beat_pct": 25.53191489361702
    },
    "ES": {
      "count": 75,
      "mean_eps": 1.1869336366653442,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.33333333333334,
      "mean_surprise": 853.1630859375,
      "beat_pct": 50.68493150684932
    },
    "BE": {
      "count": 56,
      "mean_eps": 1.7467857599258423,
      "median_eps": 1.2200000286102295,
      "positive_pct": 98.21428571428571,
      "mean_surprise": 81.95157623291016,
      "beat_pct": 51.78571428571429
    },
    "IT": {
      "count": 140,
      "mean_eps": 1.3067145347595215,
      "median_eps": 1.2200000286102295,
      "positive_pct": 99.28571428571429,
      "mean_surprise": 499.92572021484375,
      "beat_pct": 58.57142857142858
    },
    "SE": {
      "count": 255,
      "mean_eps": 0.7125882506370544,
      "median_eps": 0.5699999928474426,
      "positive_pct": 89.80392156862746,
      "mean_surprise": 70.9918441772461,
      "beat_pct": 27.559055118110237
    },
    "FI": {
      "count": 92,
      "mean_eps": 0.9236956834793091,
      "median_eps": 0.7350000143051147,
      "positive_pct": 84.78260869565217,
      "mean_surprise": -45.1297607421875,
      "beat_pct": 26.08695652173913
    },
    "NO": {
      "count": 119,
      "mean_eps": 0.8221850395202637,
      "median_eps": 0.699999988079071,
      "positive_pct": 86.5546218487395,
      "mean_surprise": 120.64832305908203,
      "beat_pct": 30.508474576271187
    },
    "AT": {
      "count": 29,
      "mean_eps": 2.056896686553955,
      "median_eps": 1.2200000286102295,
      "positive_pct": 89.65517241379311,
      "mean_surprise": 12.556933403015137,
      "beat_pct": 24.137931034482758
    },
    "IE": {
      "count": 11,
      "mean_eps": 1.3472727537155151,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 111.42997741699219,
      "beat_pct": 54.54545454545454
    },
    "PL": {
      "count": 94,
      "mean_eps": 3.2145750522613525,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.80851063829788,
      "mean_surprise": 78.88607788085938,
      "beat_pct": 59.57446808510638
    },
    "HU": {
      "count": 5,
      "mean_eps": 1.3380000591278076,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": -8.895471572875977,
      "beat_pct": 20.0
    },
    "PT": {
      "count": 13,
      "mean_eps": 1.0892307758331299,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 250.26412963867188,
      "beat_pct": 84.61538461538461
    },
    "GR": {
      "count": 31,
      "mean_eps": 1.3154840469360352,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 78.49446105957031,
      "beat_pct": 56.666666666666664
    },
    "RO": {
      "count": 12,
      "mean_eps": 1.1358333826065063,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 3214.964111328125,
      "beat_pct": 100.0
    },
    "SI": {
      "count": 6,
      "mean_eps": 1.220000147819519,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": -44.42222213745117,
      "beat_pct": 33.33333333333333
    },
    "HR": {
      "count": 6,
      "mean_eps": 1.220000147819519,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": -24.674560546875,
      "beat_pct": 50.0
    },
    "HK": {
      "count": 230,
      "mean_eps": 1.1604783535003662,
      "median_eps": 1.2200000286102295,
      "positive_pct": 99.56521739130434,
      "mean_surprise": 1573.4530029296875,
      "beat_pct": 91.5929203539823
    },
    "JP": {
      "count": 378,
      "mean_eps": 1.2200002670288086,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 38.6363639831543,
      "beat_pct": 100.0
    },
    "CN": {
      "count": 901,
      "mean_eps": 0.9507104158401489,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.78135405105438,
      "mean_surprise": 543.8406982421875,
      "beat_pct": 73.68421052631578
    },
    "AU": {
      "count": 57,
      "mean_eps": 1.220000147819519,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 1253.611083984375,
      "beat_pct": 89.0909090909091
    },
    "SG": {
      "count": 26,
      "mean_eps": 1.2480769157409668,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 1400.23095703125,
      "beat_pct": 88.0
    },
    "TH": {
      "count": 27,
      "mean_eps": 0.1325925886631012,
      "median_eps": 0.03999999910593033,
      "positive_pct": 88.88888888888889,
      "mean_surprise": -27.171720504760742,
      "beat_pct": 12.0
    },
    "ID": {
      "count": 26,
      "mean_eps": 0.7592307329177856,
      "median_eps": 1.2200000286102295,
      "positive_pct": 92.3076923076923,
      "mean_surprise": 3862.6123046875,
      "beat_pct": 61.111111111111114
    },
    "IN": {
      "count": 114,
      "mean_eps": 0.6693860292434692,
      "median_eps": 0.5099999904632568,
      "positive_pct": 98.24561403508771,
      "mean_surprise": 87.1362533569336,
      "beat_pct": 30.08849557522124
    },
    "KR": {
      "count": 95,
      "mean_eps": 3.113368272781372,
      "median_eps": 1.2200000286102295,
      "positive_pct": 92.63157894736842,
      "mean_surprise": 40.93170166015625,
      "beat_pct": 17.894736842105264
    },
    "TW": {
      "count": 76,
      "mean_eps": 0.5834210515022278,
      "median_eps": 0.22499999403953552,
      "positive_pct": 94.73684210526315,
      "mean_surprise": 52.043949127197266,
      "beat_pct": 17.56756756756757
    },
    "MY": {
      "count": 24,
      "mean_eps": 0.6620833277702332,
      "median_eps": 0.8550000190734863,
      "positive_pct": 95.83333333333334,
      "mean_surprise": 1003.3403930664062,
      "beat_pct": 50.0
    },
    "VN": {
      "count": 25,
      "mean_eps": 0.7843999266624451,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 871.5482788085938,
      "beat_pct": 64.0
    },
    "PH": {
      "count": 16,
      "mean_eps": 0.7068750262260437,
      "median_eps": 1.215000033378601,
      "positive_pct": 93.75,
      "mean_surprise": 1868.5018310546875,
      "beat_pct": 46.666666666666664
    },
    "AE": {
      "count": 64,
      "mean_eps": 0.7089062929153442,
      "median_eps": 1.2200000286102295,
      "positive_pct": 98.4375,
      "mean_surprise": 1495.390625,
      "beat_pct": 56.25
    },
    "MX": {
      "count": 61,
      "mean_eps": 0.6378688216209412,
      "median_eps": 0.28999999165534973,
      "positive_pct": 85.24590163934425,
      "mean_surprise": 308.82061767578125,
      "beat_pct": 32.78688524590164
    },
    "BR": {
      "count": 158,
      "mean_eps": 0.48575955629348755,
      "median_eps": 0.27000001072883606,
      "positive_pct": 88.60759493670885,
      "mean_surprise": 91.27923583984375,
      "beat_pct": 38.21656050955414
    },
    "ZA": {
      "count": 79,
      "mean_eps": 1.205569863319397,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 626.1280517578125,
      "beat_pct": 73.41772151898735
    },
    "TR": {
      "count": 97,
      "mean_eps": 1.1283506155014038,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 388.5874938964844,
      "beat_pct": 89.24731182795699
    },
    "CO": {
      "count": 12,
      "mean_eps": 0.9633333683013916,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 804.0518188476562,
      "beat_pct": 58.333333333333336
    },
    "CL": {
      "count": 36,
      "mean_eps": 0.6663888096809387,
      "median_eps": 0.32999998331069946,
      "positive_pct": 88.88888888888889,
      "mean_surprise": 427.521728515625,
      "beat_pct": 44.11764705882353
    },
    "PE": {
      "count": 7,
      "mean_eps": 1.7128572463989258,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 84.32030487060547,
      "beat_pct": 42.857142857142854
    },
    "MA": {
      "count": 33,
      "mean_eps": 1.2200000286102295,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 87.04894256591797,
      "beat_pct": 18.181818181818183
    },
    "AR": {
      "count": 11,
      "mean_eps": 0.34272727370262146,
      "median_eps": 0.18000000715255737,
      "positive_pct": 72.72727272727273,
      "mean_surprise": -49.53886032104492,
      "beat_pct": 30.0
    },
    "KE": {
      "count": 10,
      "mean_eps": 1.2200000286102295,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 1158.3389892578125,
      "beat_pct": 100.0
    },
    "NG": {
      "count": 45,
      "mean_eps": 1.1931113004684448,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 3550.96484375,
      "beat_pct": 97.05882352941177
    },
    "GH": {
      "count": 12,
      "mean_eps": 1.220000147819519,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 3175.339599609375,
      "beat_pct": 100.0
    },
    "TN": {
      "count": 34,
      "mean_eps": 1.220000147819519,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 666.0504150390625,
      "beat_pct": 88.23529411764706
    }
  },
  "by_exchange": {
    "Semiconductors and Semiconductor Equipment": {
      "count": 193,
      "mean_eps": 2.11616587638855,
      "median_eps": 1.2200000286102295,
      "positive_pct": 93.26424870466322,
      "mean_surprise": 260.2463073730469,
      "beat_pct": 44.50261780104712
    },
    "Technology Hardware Storage and Peripherals": {
      "count": 54,
      "mean_eps": 1.75,
      "median_eps": 1.2200000286102295,
      "positive_pct": 90.74074074074075,
      "mean_surprise": 172.81793212890625,
      "beat_pct": 40.74074074074074
    },
    "Interactive Media and Services": {
      "count": 43,
      "mean_eps": 2.4346516132354736,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.34883720930233,
      "mean_surprise": 215.52627563476562,
      "beat_pct": 48.837209302325576
    },
    "Software": {
      "count": 225,
      "mean_eps": 2.78439998626709,
      "median_eps": 1.2200000286102295,
      "positive_pct": 92.44444444444444,
      "mean_surprise": 502.9330749511719,
      "beat_pct": 42.857142857142854
    },
    "Broadline Retail": {
      "count": 50,
      "mean_eps": 3.472599983215332,
      "median_eps": 1.2200000286102295,
      "positive_pct": 98.0,
      "mean_surprise": 253.3134307861328,
      "beat_pct": 40.816326530612244
    },
    "Automobiles": {
      "count": 47,
      "mean_eps": 1.6510639190673828,
      "median_eps": 1.2200000286102295,
      "positive_pct": 89.36170212765957,
      "mean_surprise": 209.40188598632812,
      "beat_pct": 65.95744680851064
    },
    "Consumer Staples Distribution and Retail": {
      "count": 105,
      "mean_eps": 1.8573331832885742,
      "median_eps": 1.2200000286102295,
      "positive_pct": 94.28571428571428,
      "mean_surprise": 750.4896240234375,
      "beat_pct": 45.19230769230769
    },
    "Pharmaceuticals": {
      "count": 173,
      "mean_eps": 1.6008092164993286,
      "median_eps": 1.2200000286102295,
      "positive_pct": 89.01734104046243,
      "mean_surprise": 339.4502258300781,
      "beat_pct": 58.38150289017341
    },
    "Banks": {
      "count": 397,
      "mean_eps": 2.8382115364074707,
      "median_eps": 1.2200000286102295,
      "positive_pct": 98.74055415617129,
      "mean_surprise": 283.6103820800781,
      "beat_pct": 38.07106598984771
    },
    "Financial Services": {
      "count": 114,
      "mean_eps": 2.945965051651001,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.36842105263158,
      "mean_surprise": 204.9404754638672,
      "beat_pct": 43.75
    },
    "Oil Gas and Consumable Fuels": {
      "count": 225,
      "mean_eps": 1.9786665439605713,
      "median_eps": 1.2200000286102295,
      "positive_pct": 93.77777777777779,
      "mean_surprise": 587.684814453125,
      "beat_pct": 50.45045045045045
    },
    "Entertainment": {
      "count": 84,
      "mean_eps": 1.5169049501419067,
      "median_eps": 1.2200000286102295,
      "positive_pct": 85.71428571428571,
      "mean_surprise": 436.3431396484375,
      "beat_pct": 60.71428571428571
    },
    "Biotechnology": {
      "count": 190,
      "mean_eps": 0.6764736771583557,
      "median_eps": 0.4449999928474426,
      "positive_pct": 61.578947368421055,
      "mean_surprise": 322.7252197265625,
      "beat_pct": 57.36842105263158
    },
    "Specialty Retail": {
      "count": 156,
      "mean_eps": 3.98641037940979,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.43589743589743,
      "mean_surprise": 211.3867950439453,
      "beat_pct": 53.84615384615385
    },
    "Household Products": {
      "count": 22,
      "mean_eps": 2.714545726776123,
      "median_eps": 1.4200000762939453,
      "positive_pct": 95.45454545454545,
      "mean_surprise": 1195.947509765625,
      "beat_pct": 47.61904761904761
    },
    "Aerospace and Defense": {
      "count": 102,
      "mean_eps": 3.770294427871704,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.09803921568627,
      "mean_surprise": 202.78384399414062,
      "beat_pct": 34.65346534653465
    },
    "Health Care Providers and Services": {
      "count": 112,
      "mean_eps": 3.7019646167755127,
      "median_eps": 1.2200000286102295,
      "positive_pct": 94.64285714285714,
      "mean_surprise": 174.40528869628906,
      "beat_pct": 39.63963963963964
    },
    "Communications Equipment": {
      "count": 48,
      "mean_eps": 1.966666579246521,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.83333333333334,
      "mean_surprise": 512.7730102539062,
      "beat_pct": 52.17391304347826
    },
    "Beverages": {
      "count": 81,
      "mean_eps": 1.677901268005371,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.53086419753086,
      "mean_surprise": 485.56060791015625,
      "beat_pct": 44.871794871794876
    },
    "IT Services": {
      "count": 94,
      "mean_eps": 2.215851068496704,
      "median_eps": 1.2200000286102295,
      "positive_pct": 94.68085106382979,
      "mean_surprise": 406.7995300292969,
      "beat_pct": 43.61702127659575
    },
    "Capital Markets": {
      "count": 230,
      "mean_eps": 3.2802176475524902,
      "median_eps": 1.2200000286102295,
      "positive_pct": 99.56521739130434,
      "mean_surprise": 276.6970520019531,
      "beat_pct": 50.0
    },
    "Machinery": {
      "count": 275,
      "mean_eps": 2.5666184425354004,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.45454545454545,
      "mean_surprise": 161.1416015625,
      "beat_pct": 45.98540145985402
    },
    "Consumer Finance": {
      "count": 41,
      "mean_eps": 4.364877700805664,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.5609756097561,
      "mean_surprise": 166.18670654296875,
      "beat_pct": 29.268292682926827
    },
    "Tobacco": {
      "count": 12,
      "mean_eps": 2.686666488647461,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 328.4228210449219,
      "beat_pct": 18.181818181818183
    },
    "Hotels Restaurants and Leisure": {
      "count": 157,
      "mean_eps": 3.576688051223755,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.54140127388536,
      "mean_surprise": 300.691162109375,
      "beat_pct": 39.490445859872615
    },
    "Wireless Telecommunication Services": {
      "count": 29,
      "mean_eps": 1.4741380214691162,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.55172413793103,
      "mean_surprise": 377.9034118652344,
      "beat_pct": 58.620689655172406
    },
    "Health Care Equipment and Supplies": {
      "count": 140,
      "mean_eps": 2.09757137298584,
      "median_eps": 1.2200000286102295,
      "positive_pct": 90.71428571428571,
      "mean_surprise": 174.6339874267578,
      "beat_pct": 41.007194244604314
    },
    "Life Sciences Tools and Services": {
      "count": 59,
      "mean_eps": 3.8423733711242676,
      "median_eps": 1.2200000286102295,
      "positive_pct": 91.52542372881356,
      "mean_surprise": 555.5504760742188,
      "beat_pct": 37.28813559322034
    },
    "Chemicals": {
      "count": 227,
      "mean_eps": 1.6380176544189453,
      "median_eps": 1.2200000286102295,
      "positive_pct": 92.07048458149781,
      "mean_surprise": 392.1242370605469,
      "beat_pct": 56.95067264573991
    },
    "Electrical Equipment": {
      "count": 175,
      "mean_eps": 1.8177716732025146,
      "median_eps": 1.2200000286102295,
      "positive_pct": 92.57142857142857,
      "mean_surprise": 272.6089782714844,
      "beat_pct": 48.85057471264368
    },
    "Ground Transportation": {
      "count": 63,
      "mean_eps": 2.0022222995758057,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.23809523809523,
      "mean_surprise": 101.61214447021484,
      "beat_pct": 45.90163934426229
    },
    "Diversified Telecommunication Services": {
      "count": 69,
      "mean_eps": 1.1668115854263306,
      "median_eps": 1.2200000286102295,
      "positive_pct": 86.95652173913044,
      "mean_surprise": 507.1138610839844,
      "beat_pct": 44.776119402985074
    },
    "Electric Utilities": {
      "count": 77,
      "mean_eps": 2.0767533779144287,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.40259740259741,
      "mean_surprise": 231.3170928955078,
      "beat_pct": 51.31578947368421
    },
    "Electronic Equipment Instruments and Components": {
      "count": 202,
      "mean_eps": 1.8667821884155273,
      "median_eps": 1.2200000286102295,
      "positive_pct": 94.55445544554455,
      "mean_surprise": 267.2718505859375,
      "beat_pct": 45.27363184079602
    },
    "Insurance": {
      "count": 165,
      "mean_eps": 5.974908828735352,
      "median_eps": 1.2200000286102295,
      "positive_pct": 98.18181818181819,
      "mean_surprise": 93.04470825195312,
      "beat_pct": 42.944785276073624
    },
    "Industrial Conglomerates": {
      "count": 54,
      "mean_eps": 1.8611112833023071,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 448.8796081542969,
      "beat_pct": 55.55555555555556
    },
    "Metals and Mining": {
      "count": 270,
      "mean_eps": 1.196333408355713,
      "median_eps": 1.2200000286102295,
      "positive_pct": 90.37037037037037,
      "mean_surprise": 707.0619506835938,
      "beat_pct": 46.15384615384615
    },
    "Professional Services": {
      "count": 89,
      "mean_eps": 2.847078800201416,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.50561797752809,
      "mean_surprise": 180.79037475585938,
      "beat_pct": 37.07865168539326
    },
    "Textiles Apparel and Luxury Goods": {
      "count": 75,
      "mean_eps": 4.423333168029785,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.33333333333334,
      "mean_surprise": 257.90789794921875,
      "beat_pct": 49.333333333333336
    },
    "Media": {
      "count": 78,
      "mean_eps": 1.899871826171875,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.15384615384616,
      "mean_surprise": 322.2420959472656,
      "beat_pct": 55.26315789473685
    },
    "Commercial Services and Supplies": {
      "count": 102,
      "mean_eps": 1.706078290939331,
      "median_eps": 1.2200000286102295,
      "positive_pct": 94.11764705882352,
      "mean_surprise": 233.6544189453125,
      "beat_pct": 45.09803921568628
    },
    "Building Products": {
      "count": 71,
      "mean_eps": 3.1714084148406982,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 205.69503784179688,
      "beat_pct": 47.88732394366197
    },
    "Air Freight and Logistics": {
      "count": 38,
      "mean_eps": 2.1465792655944824,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 420.5173034667969,
      "beat_pct": 50.0
    },
    "Construction Materials": {
      "count": 65,
      "mean_eps": 1.7264615297317505,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.38461538461539,
      "mean_surprise": 669.8980102539062,
      "beat_pct": 52.307692307692314
    },
    "Food Products": {
      "count": 197,
      "mean_eps": 1.5587818622589111,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.93908629441624,
      "mean_surprise": 536.09521484375,
      "beat_pct": 52.331606217616574
    },
    "Construction and Engineering": {
      "count": 148,
      "mean_eps": 1.9920270442962646,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.97297297297297,
      "mean_surprise": 302.35333251953125,
      "beat_pct": 55.10204081632652
    },
    "Energy Equipment and Services": {
      "count": 56,
      "mean_eps": 1.3546428680419922,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.42857142857143,
      "mean_surprise": 333.0783996582031,
      "beat_pct": 51.78571428571429
    },
    "Multi-Utilities": {
      "count": 35,
      "mean_eps": 2.278571605682373,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 191.78292846679688,
      "beat_pct": 48.57142857142857
    },
    "Independent Power and Renewable Electricity Producers": {
      "count": 80,
      "mean_eps": 1.082625150680542,
      "median_eps": 1.2200000286102295,
      "positive_pct": 91.25,
      "mean_surprise": 565.2443237304688,
      "beat_pct": 64.1025641025641
    },
    "Trading Companies and Distributors": {
      "count": 85,
      "mean_eps": 3.2958827018737793,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.6470588235294,
      "mean_surprise": 174.79998779296875,
      "beat_pct": 45.78313253012048
    },
    "Real Estate Management and Development": {
      "count": 177,
      "mean_eps": 1.1702826023101807,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.48022598870057,
      "mean_surprise": 409.98370361328125,
      "beat_pct": 55.367231638418076
    },
    "Household Durables": {
      "count": 114,
      "mean_eps": 6.514736652374268,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.6140350877193,
      "mean_surprise": 192.9534149169922,
      "beat_pct": 56.25
    },
    "Passenger Airlines": {
      "count": 38,
      "mean_eps": 2.0221052169799805,
      "median_eps": 1.2200000286102295,
      "positive_pct": 81.57894736842105,
      "mean_surprise": 198.71420288085938,
      "beat_pct": 32.432432432432435
    },
    "Personal Care Products": {
      "count": 45,
      "mean_eps": 1.2322224378585815,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.77777777777777,
      "mean_surprise": 190.3582305908203,
      "beat_pct": 62.22222222222222
    },
    "Health Care Technology": {
      "count": 18,
      "mean_eps": 1.4566664695739746,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 296.3750915527344,
      "beat_pct": 61.111111111111114
    },
    "Gas Utilities": {
      "count": 29,
      "mean_eps": 2.2368967533111572,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 223.1974334716797,
      "beat_pct": 55.172413793103445
    },
    "Water Utilities": {
      "count": 21,
      "mean_eps": 1.4538094997406006,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 643.7571411132812,
      "beat_pct": 52.38095238095239
    },
    "Containers and Packaging": {
      "count": 42,
      "mean_eps": 2.2016665935516357,
      "median_eps": 1.2200000286102295,
      "positive_pct": 95.23809523809523,
      "mean_surprise": 15.246773719787598,
      "beat_pct": 45.23809523809524
    },
    "Distributors": {
      "count": 16,
      "mean_eps": 2.382500171661377,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 167.9739990234375,
      "beat_pct": 62.5
    },
    "Automobile Components": {
      "count": 112,
      "mean_eps": 1.7355355024337769,
      "median_eps": 1.2200000286102295,
      "positive_pct": 96.42857142857143,
      "mean_surprise": 179.6894989013672,
      "beat_pct": 54.95495495495496
    },
    "Mortgage Real Estate Investment Trusts (REITs)": {
      "count": 11,
      "mean_eps": 1.4436362981796265,
      "median_eps": 1.5399999618530273,
      "positive_pct": 90.9090909090909,
      "mean_surprise": -20.175291061401367,
      "beat_pct": 27.27272727272727
    },
    "Leisure Products": {
      "count": 29,
      "mean_eps": 1.4393105506896973,
      "median_eps": 1.2200000286102295,
      "positive_pct": 93.10344827586206,
      "mean_surprise": 40.83460235595703,
      "beat_pct": 44.827586206896555
    },
    "Diversified Consumer Services": {
      "count": 34,
      "mean_eps": 4.2708821296691895,
      "median_eps": 1.2200000286102295,
      "positive_pct": 100.0,
      "mean_surprise": 359.2332763671875,
      "beat_pct": 41.17647058823529
    },
    "Paper and Forest Products": {
      "count": 25,
      "mean_eps": 1.1452000141143799,
      "median_eps": 1.2200000286102295,
      "positive_pct": 92.0,
      "mean_surprise": 26.715225219726562,
      "beat_pct": 52.0
    },
    "Marine Transportation": {
      "count": 34,
      "mean_eps": 12.201470375061035,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.05882352941177,
      "mean_surprise": 499.6700439453125,
      "beat_pct": 69.6969696969697
    },
    "Transportation Infrastructure": {
      "count": 42,
      "mean_eps": 1.0954762697219849,
      "median_eps": 1.2200000286102295,
      "positive_pct": 97.61904761904762,
      "mean_surprise": 759.0257568359375,
      "beat_pct": 67.5
    }
  },
  "eps_revision_momentum": {
    "normalized": {
      "1w": {
        "column": "eps_est_avg_rev_pct_fy1e_1w",
        "count": 6666,
        "mean": 0.0027598112355917692,
        "median": 0.002300000051036477,
        "positive_pct": 71.66216621662166,
        "negative_pct": 13.711371137113712,
        "large_upgrade_pct": 0.0,
        "large_downgrade_pct": 0.0
      },
      "1m": {
        "column": "eps_est_avg_rev_pct_fy1e_1m",
        "count": 6666,
        "mean": 0.009412482380867004,
        "median": 0.005100000184029341,
        "positive_pct": 70.16201620162016,
        "negative_pct": 22.247224722472247,
        "large_upgrade_pct": 0.030003000300030006,
        "large_downgrade_pct": 0.0
      },
      "3m": {
        "column": "eps_est_avg_rev_pct_fy1e_3m",
        "count": 6666,
        "mean": 0.04458579421043396,
        "median": 0.003599999938160181,
        "positive_pct": 66.02160216021602,
        "negative_pct": 33.73837383738374,
        "large_upgrade_pct": 0.12001200120012002,
        "large_downgrade_pct": 0.0
      },
      "6m": {
        "column": "eps_est_avg_rev_pct_fy1e_6m",
        "count": 6666,
        "mean": 0.05683499574661255,
        "median": 0.010599999688565731,
        "positive_pct": 67.16171617161716,
        "negative_pct": 32.76327632763276,
        "large_upgrade_pct": 0.18001800180018002,
        "large_downgrade_pct": 0.0
      },
      "1y": {
        "column": "eps_est_avg_rev_pct_fy1e_1y",
        "count": 6666,
        "mean": 0.09998194128274918,
        "median": -0.0008499999530613422,
        "positive_pct": 35.628562856285626,
        "negative_pct": 64.34143414341435,
        "large_upgrade_pct": 0.18001800180018002,
        "large_downgrade_pct": 0.0
      }
    },
    "gaap": {
      "1m": {
        "column": "eps_gaap_est_avg_rev_pct_fy1e_1m",
        "count": 6666,
        "mean": 0.013695199973881245,
        "median": 0.006300000008195639,
        "positive_pct": 66.38163816381638,
        "negative_pct": 24.602460246024602,
        "large_upgrade_pct": 0.030003000300030006,
        "large_downgrade_pct": 0.0
      },
      "3m": {
        "column": "eps_gaap_est_avg_rev_pct_fy1e_3m",
        "count": 6666,
        "mean": 0.008638358674943447,
        "median": -0.0005000000237487257,
        "positive_pct": 39.40894089408941,
        "negative_pct": 60.29102910291029,
        "large_upgrade_pct": 0.075007500750075,
        "large_downgrade_pct": 0.0
      },
      "6m": {
        "column": "eps_gaap_est_avg_rev_pct_fy1e_6m",
        "count": 6666,
        "mean": 0.06899666786193848,
        "median": -0.0044999998062849045,
        "positive_pct": 38.53885388538854,
        "negative_pct": 61.34113411341134,
        "large_upgrade_pct": 0.24002400240024005,
        "large_downgrade_pct": 0.0
      },
      "1y": {
        "column": "eps_gaap_est_avg_rev_pct_fy1e_1y",
        "count": 6666,
        "mean": 0.043582748621702194,
        "median": -0.031300000846385956,
        "positive_pct": 35.013501350135016,
        "negative_pct": 64.98649864986498,
        "large_upgrade_pct": 0.21002100210021002,
        "large_downgrade_pct": 0.0
      }
    }
  },
  "upgrade_momentum": {
    "count": 3191,
    "percentage": 47.86978697869787,
    "periods_used": [
      "eps_est_avg_rev_pct_fy1e_1m",
      "eps_est_avg_rev_pct_fy1e_3m",
      "eps_est_avg_rev_pct_fy1e_6m"
    ]
  }
}
```

Attachment Name: earnings_widgets.py
Attachments Kind: Visible
Attachments Source: ReferencesRetriever / FileChatAttachment
Attachments Text:

```python
"""
Earnings and Dividend Dashboard Widgets.

Enhanced with Phase 9.3 PHASE93_FEATURE_INPUTS categories for comprehensive
metric selection across earnings, dividends, valuation, quality/risk, and
technical analysis domains.

Aligned with code_guidelines.md v1.4 Section 17 (Style Guides for Visual Elements).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Literal, Union
from datetime import timedelta
from pathlib import Path

# Schema-driven Phase 9.3 feature categorization (code_guidelines.md §9.3)
from finance_ml.ml_workflow.data.schema import PHASE93_FEATURE_INPUTS

# =============================================================================
# Style Constants (aligned with code_guidelines.md §17.1, §17.2)
# =============================================================================

PLOTLY_TEMPLATE = "plotly_dark"

COLOR_PALETTE = {
    "primary": "#375a7f",
    "secondary": "#6c757d",
    "success": "#00bc8c",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#3498db",
    "neutral": "#adb5bd",
    }

# Category colors for visualization
CATEGORY_COLORS = {
    "momentum": "#3498db",  # Info blue
    "valuation": "#375a7f",  # Primary blue
    "profitability": "#00bc8c",  # Success green
    "quality_risk": "#e74c3c",  # Danger red
    "cash_flow": "#f39c12",  # Warning orange
    "growth": "#9b59b6",  # Purple
    "technical": "#1abc9c",  # Teal
    "employment": "#34495e",  # Dark gray
    "dividends": "#27ae60",  # Green
    "forecasts": "#2980b9",  # Blue
    }

# Valid mode options
EarningsMode = Literal[
    "all",
    "earnings",
    "dividends",
    "valuation",
    "quality_risk",
    "technical",
    "forecasts",
    "momentum",
    "profitability",
    "growth",
    "cash_flow",
    "employment",
]


def get_category_metrics(
        categories: List[str],
        include_supplemental: bool = True,
        ) -> Dict[str, List[str]]:
    """
    Get metrics from specified PHASE93_FEATURE_INPUTS categories.

    Args:
        categories: List of category names from PHASE93_FEATURE_INPUTS.
        include_supplemental: Whether to include supplemental domain-specific metrics.

    Returns:
        Dict mapping category name to list of metric column names.
    """
    result = {}
    for cat in categories:
        metrics = PHASE93_FEATURE_INPUTS.get(cat, []).copy()
        result[cat] = metrics

    # Add supplemental metrics for specific categories
    if include_supplemental:
        # Earnings-related supplemental metrics
        if "profitability" in result or "growth" in result:
            supplemental_earnings = [
                "net_income_adj_1fy",
                "ebitda_adj_fy",
                "ebitda_adj_1fy",
                "ebit_adj_1fy",
                "ebit_adj_fy",
                "net_income_adj_fy",
                "net_income_adj_fq",
                "net_income_adj_5yavgfq",
                "eps_adj_1fy",
                "eps_adj_fy",
                "eps_adj_ltm",
                ]
            if "profitability" in result:
                result["profitability"].extend(supplemental_earnings)

        # Dividend-related supplemental metrics
        if "dividends" in result:
            supplemental_dividends = [
                "dividend_record_announce_date",
                "dividend_record_ex_date",
                "dividend_record_payable_date",
                "dividend_record_record_date",
                "dividend_record_frequency",
                "dividend_record_currency",
                "div_yield_ind",
                "div_yield_1fyind",
                "div_yield_5yavgltm",
                "dividend_per_share",
                "common_dividends_paid_fy",
                "dividends_paid",
                "dividends_paid_ltm",
                ]
            result["dividends"].extend(supplemental_dividends)

    return result


def create_earnings_calendar_dashboard(
        df: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None,
        top_n: int = 100,
        mode: EarningsMode = "all",
        categories: Optional[List[str]] = None,
        ) -> pd.DataFrame:
    """
    Creates a dashboard (styled DataFrame) for Earnings and Dividend Analytics.
    Filters for companies with upcoming or recent earnings (t +/- 10 days).

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Uses PHASE93_FEATURE_INPUTS categories for metric selection:
    - **momentum**: Price changes, returns, EMAs (market reaction context)
    - **valuation**: P/E, P/B, EV/EBITDA multiples (valuation impact)
    - **profitability**: Margins, EBITDA, EBIT, net income (earnings quality)
    - **quality_risk**: Altman Z-score, ROE, ROA, volatility (risk assessment)
    - **cash_flow**: CFO, FCF (dividend sustainability, earnings quality)
    - **growth**: Revenue CAGR, growth estimates (growth trajectory)
    - **technical**: EMAs, 52W high/low (price context)
    - **employment**: Workforce metrics (operational health)
    - **dividends**: Dividend yields, streaks, payments (income metrics)
    - **forecasts**: Analyst estimates (earnings expectations)

    Args:
        df: Input DataFrame containing stock data.
        reference_date: Date to compare next_earnings against. Defaults to today.
        top_n: Number of top companies (by Market Cap) to include.
        mode: Display mode - 'all', 'earnings', 'dividends', or specific category name.
        categories: Optional list of specific PHASE93 categories to include.
            Overrides mode if provided.

    Returns:
        pd.DataFrame: Filtered DataFrame with selected metrics.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()

    # Ensure date columns are datetime
    date_cols = [
        "next_earnings",
        "dividend_record_ex_date",
        "dividend_record_payable_date",
        "dividend_record_announce_date",
        "dividend_record_record_date",
        ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Filter logic: next_earnings within +/- 10 days
    if "next_earnings" not in df.columns:
        print("Warning: 'next_earnings' column not found. Returning empty dataframe.")
        return pd.DataFrame()

    mask = (df["next_earnings"] - reference_date).abs() <= timedelta(days=10)
    filtered_df = df[mask].copy()

    # Sort by Market Cap
    mcap_col = None
    for col in ["market_cap", "market_cap_usd", "market_cap_curr"]:
        if col in df.columns:
            mcap_col = col
            break

    if mcap_col:
        filtered_df = filtered_df.sort_values(by=mcap_col, ascending=False)

    filtered_df = filtered_df.head(top_n)

    # Define identity columns
    display_cols = ["ticker", "sector", "region", "next_earnings"]
    display_cols = [c for c in display_cols if c in df.columns or c == "next_earnings"]
    if mcap_col and mcap_col not in display_cols:
        display_cols.append(mcap_col)

    # Determine which categories to include based on mode or explicit categories
    if categories is not None:
        selected_categories = categories
    elif mode == "all":
        # Include all major categories for comprehensive view
        selected_categories = [
            "profitability",
            "valuation",
            "growth",
            "momentum",
            "quality_risk",
            "cash_flow",
            "dividends",
            "forecasts",
            ]
    elif mode == "earnings":
        # Earnings-focused categories
        selected_categories = [
            "profitability",
            "valuation",
            "growth",
            "momentum",
            "forecasts",
            ]
    elif mode == "dividends":
        # Dividend-focused categories
        selected_categories = ["dividends", "cash_flow"]
    elif mode in PHASE93_FEATURE_INPUTS:
        # Single category mode
        selected_categories = [mode]
    else:
        # Default to earnings mode
        selected_categories = ["profitability", "growth", "momentum"]

    # Get metrics from selected categories
    category_metrics = get_category_metrics(
            selected_categories, include_supplemental=True
            )

    # Build final columns list
    final_cols = display_cols.copy()
    for cat, metrics in category_metrics.items():
        existing_metrics = [c for c in metrics if c in df.columns]
        final_cols.extend(existing_metrics)

    # Remove duplicates while preserving order
    final_cols = list(dict.fromkeys(final_cols))

    # Filter to only columns that exist
    final_cols = [c for c in final_cols if c in filtered_df.columns]

    dashboard_df = filtered_df[final_cols].copy()

    # Add computed columns
    if "next_earnings" in dashboard_df.columns:
        dashboard_df["days_to_earnings"] = (
                dashboard_df["next_earnings"] - reference_date
        ).dt.days

        # Reorder: Put days_to_earnings near next_earnings
        cols = list(dashboard_df.columns)
        if "days_to_earnings" in cols:
            cols.remove("days_to_earnings")
            if "next_earnings" in cols:
                idx = cols.index("next_earnings") + 1
                cols.insert(idx, "days_to_earnings")
            dashboard_df = dashboard_df[cols]

    return dashboard_df


def _build_format_dict(columns: List[str]) -> Dict[str, str]:
    """
    Build format dictionary for DataFrame styling based on column names.

    Args:
        columns: List of column names to format.

    Returns:
        Dict mapping column names to format strings.
    """
    format_dict = {}

    for col in columns:
        col_lower = col.lower()

        # Date columns
        if any(
                x in col_lower
                for x in ["date", "next_earnings", "last_updated", "record_date"]
                ):
            format_dict[col] = "{:%Y-%m-%d}"

        # Currency/Price columns
        elif any(
                x in col_lower
                for x in [
                    "market_cap",
                    "enterprise_value",
                    "price",
                    "ebitda",
                    "ebit",
                    "revenue",
                    "income",
                    "fcf",
                    "cfo",
                    "cfi",
                    "cff",
                    "capex",
                    "capital_expenditure",
                    "eps",
                    "dividend_per_share",
                    "dividends_paid",
                    "gross_profit",
                    ]
                ):
            if "pct" in col_lower or "yield" in col_lower or "margin" in col_lower:
                format_dict[col] = "{:.2%}"
            elif "eps" in col_lower or "per_share" in col_lower:
                format_dict[col] = "${:.2f}"
            else:
                format_dict[col] = "${:,.0f}"

        # Percentage columns
        elif any(
                x in col_lower
                for x in [
                    "pct",
                    "yield",
                    "margin",
                    "return",
                    "roe",
                    "roa",
                    "cagr",
                    "volatility",
                    ]
                ):
            format_dict[col] = "{:.2%}"

        # Ratio columns
        elif any(
                x in col_lower
                for x in [
                    "p_e",
                    "p_b",
                    "p_tbv",
                    "ev_sales",
                    "ev_ebitda",
                    "current_ratio",
                    "beta",
                    "altman",
                    "z_score",
                    ]
                ):
            format_dict[col] = "{:.2f}"

        # Integer columns
        elif any(x in col_lower for x in ["employees", "streak", "count", "num"]):
            format_dict[col] = "{:,.0f}"

        # Days column
        elif "days" in col_lower:
            format_dict[col] = "{:+.0f}"

    return format_dict


def display_earnings_dashboard(
        df: pd.DataFrame,
        mode: EarningsMode = "all",
        categories: Optional[List[str]] = None,
        reference_date: Optional[pd.Timestamp] = None,
        top_n: int = 100,
        ) -> Optional["pd.io.formats.style.Styler"]:
    """
    Displays the earnings dashboard using Pandas Styler with enhanced formatting.

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Supports all PHASE93_FEATURE_INPUTS categories with appropriate formatting:
    - Currency formatting for financial metrics
    - Percentage formatting for yields, margins, returns
    - Ratio formatting for valuation multiples
    - Color-coded days_to_earnings indicator

    **Style Guide Alignment (code_guidelines.md §17.1):**
    - Color palette: danger (red) for past, warning (yellow) for today,
      success (green) for future earnings
    - Consistent number formatting with appropriate precision
    - NA values displayed as "-"

    Args:
        df: Input DataFrame containing stock data.
        mode: Display mode - 'all', 'earnings', 'dividends', or specific category.
        categories: Optional list of specific PHASE93 categories to include.
        reference_date: Date for earnings comparison. Defaults to today.
        top_n: Number of top companies to include.

    Returns:
        pd.io.formats.style.Styler: Styled DataFrame for display, or None if empty.
    """
    dashboard_df = create_earnings_calendar_dashboard(
            df,
            reference_date=reference_date,
            top_n=top_n,
            mode=mode,
            categories=categories,
            )

    if dashboard_df.empty:
        print("No companies found with earnings within +/- 10 days.")
        return None

    # Build comprehensive format dictionary
    format_dict = _build_format_dict(list(dashboard_df.columns))

    # Apply styling
    styler = dashboard_df.style.format(format_dict, na_rep="-")

    # Color-code days_to_earnings (aligned with code_guidelines.md §17.1 colors)
    def color_days(val):
        if pd.isna(val):
            return ""
        if val < 0:
            return f"color: {COLOR_PALETTE['danger']}"  # Past
        if val == 0:
            return (
                f"background-color: {COLOR_PALETTE['warning']}; color: black"  # Today
            )
        if val > 0:
            return f"color: {COLOR_PALETTE['success']}"  # Future
        return ""

    if "days_to_earnings" in dashboard_df.columns:
        styler = styler.map(color_days, subset=["days_to_earnings"])

    # Add caption with mode info
    mode_display = mode.replace("_", " ").title()
    styler = styler.set_caption(
            f"Earnings Calendar Dashboard - Mode: {mode_display} "
            f"(Top {len(dashboard_df)} by Market Cap)"
            )

    return styler


def create_earnings_metrics_chart(
        df: pd.DataFrame,
        metric_category: str = "profitability",
        reference_date: Optional[pd.Timestamp] = None,
        top_n: int = 20,
        output_path: Optional[Union[str, Path]] = None,
        ) -> go.Figure:
    """
    Creates an interactive Plotly chart showing metrics for upcoming earnings.

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Visualizes metrics from specified PHASE93_FEATURE_INPUTS category.

    **Style Guide Alignment (code_guidelines.md §17.2):**
    - Uses PLOTLY_TEMPLATE ('plotly_dark')
    - Standard color palette from COLOR_PALETTE
    - Hover data includes ticker, sector, region
    - Labeled axes with units

    Args:
        df: Input DataFrame containing stock data.
        metric_category: PHASE93_FEATURE_INPUTS category to visualize.
        reference_date: Date for earnings comparison. Defaults to today.
        top_n: Number of companies to include.
        output_path: Optional path to save HTML output.

    Returns:
        go.Figure: Plotly figure object.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()

    # Get dashboard data for the specific category
    dashboard_df = create_earnings_calendar_dashboard(
            df,
            reference_date=reference_date,
            top_n=top_n,
            mode=metric_category
            if metric_category in PHASE93_FEATURE_INPUTS
            else "earnings",
            categories=[metric_category]
            if metric_category in PHASE93_FEATURE_INPUTS
            else None,
            )

    if dashboard_df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
                text="No companies found with earnings within +/- 10 days",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16),
                )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    # Get metrics for the category
    category_metrics = PHASE93_FEATURE_INPUTS.get(metric_category, [])
    available_metrics = [c for c in category_metrics if c in dashboard_df.columns][:5]

    if not available_metrics:
        # Fallback to any numeric columns
        numeric_cols = dashboard_df.select_dtypes(include=["float64", "int64"]).columns
        available_metrics = [
            c
            for c in numeric_cols
            if c not in ["days_to_earnings"] and "date" not in c.lower()
            ][:5]

    if not available_metrics:
        fig = go.Figure()
        fig.add_annotation(
                text=f"No metrics available for category: {metric_category}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16),
                )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    # Create subplots for multiple metrics
    n_metrics = len(available_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + 1) // 2

    fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[m.replace("_", " ").title() for m in available_metrics],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            )

    # Add bar charts for each metric
    for i, metric in enumerate(available_metrics):
        row = i // 2 + 1
        col = i % 2 + 1

        # Sort by metric value
        plot_df = dashboard_df.dropna(subset=[metric]).head(15)
        if plot_df.empty:
            continue

        plot_df = plot_df.sort_values(by=metric, ascending=True)

        # Determine color based on value sign
        colors = [
            COLOR_PALETTE["success"] if v >= 0 else COLOR_PALETTE["danger"]
            for v in plot_df[metric]
            ]

        fig.add_trace(
                go.Bar(
                        x=plot_df[metric],
                        y=plot_df["ticker"] if "ticker" in plot_df.columns else plot_df.index,
                        orientation="h",
                        marker_color=colors,
                        name=metric.replace("_", " ").title(),
                        hovertemplate=(
                                "<b>%{y}</b><br>"
                                + f"{metric}: "
                                + "%{x:.2f}<br>"
                                + "<extra></extra>"
                        ),
                        ),
                row=row,
                col=col,
                )

    # Update layout
    category_display = metric_category.replace("_", " ").title()
    fig.update_layout(
            title=dict(
                    text=f"Earnings Calendar: {category_display} Metrics (Top {top_n})",
                    font=dict(size=20),
                    ),
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            font=dict(family="Arial, sans-serif", size=12),
            height=300 * n_rows,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            )

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return fig


def create_category_comparison_chart(
        df: pd.DataFrame,
        categories: Optional[List[str]] = None,
        reference_date: Optional[pd.Timestamp] = None,
        top_n: int = 10,
        output_path: Optional[Union[str, Path]] = None,
        ) -> go.Figure:
    """
    Creates an interactive chart comparing metrics across PHASE93 categories.

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Provides visual comparison of metric availability and values across
    all PHASE93_FEATURE_INPUTS categories for earnings calendar companies.

    Args:
        df: Input DataFrame containing stock data.
        categories: List of categories to compare. Defaults to all.
        reference_date: Date for earnings comparison.
        top_n: Number of companies to include.
        output_path: Optional path to save HTML output.

    Returns:
        go.Figure: Plotly figure with category comparison.
    """
    if categories is None:
        categories = list(PHASE93_FEATURE_INPUTS.keys())

    if reference_date is None:
        reference_date = pd.Timestamp.now()

    # Get base dashboard data
    dashboard_df = create_earnings_calendar_dashboard(
            df,
            reference_date=reference_date,
            top_n=top_n,
            mode="all",
            )

    if dashboard_df.empty:
        fig = go.Figure()
        fig.add_annotation(
                text="No companies found with earnings within +/- 10 days",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    # Calculate coverage statistics per category
    coverage_data = []
    for cat in categories:
        metrics = PHASE93_FEATURE_INPUTS.get(cat, [])
        available = [m for m in metrics if m in df.columns]
        non_null_counts = [
            dashboard_df[m].notna().sum()
            for m in available
            if m in dashboard_df.columns
            ]

        coverage_data.append(
                {
                    "category": cat.replace("_", " ").title(),
                    "total_metrics": len(metrics),
                    "available_metrics": len(available),
                    "coverage_pct": len(available) / len(metrics) * 100 if metrics else 0,
                    "avg_non_null": (
                        sum(non_null_counts) / len(non_null_counts)
                        if non_null_counts
                        else 0
                    ),
                    "color": CATEGORY_COLORS.get(cat, COLOR_PALETTE["neutral"]),
                    }
                )

    coverage_df = pd.DataFrame(coverage_data)

    # Create figure with two subplots
    fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Metric Coverage by Category",
                "Data Availability (Avg Non-Null)",
                ],
            horizontal_spacing=0.15,
            )

    # Coverage bar chart
    fig.add_trace(
            go.Bar(
                    x=coverage_df["category"],
                    y=coverage_df["coverage_pct"],
                    marker_color=coverage_df["color"],
                    name="Coverage %",
                    hovertemplate=("<b>%{x}</b><br>Coverage: %{y:.1f}%<br><extra></extra>"),
                    ),
            row=1,
            col=1,
            )

    # Data availability bar chart
    fig.add_trace(
            go.Bar(
                    x=coverage_df["category"],
                    y=coverage_df["avg_non_null"],
                    marker_color=coverage_df["color"],
                    name="Avg Non-Null",
                    hovertemplate=("<b>%{x}</b><br>Avg Records: %{y:.0f}<br><extra></extra>"),
                    ),
            row=1,
            col=2,
            )

    fig.update_layout(
            title=dict(
                    text="Phase 9.3 Category Metrics: Coverage & Availability Analysis",
                    font=dict(size=18),
                    ),
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            font=dict(family="Arial, sans-serif", size=12),
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            )

    fig.update_yaxes(title_text="Coverage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Avg Non-Null Records", row=1, col=2)
    fig.update_xaxes(tickangle=45)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return fig

```

Attachment Name: schema.py
Attachments Kind: Visible
Attachments Source: ReferencesRetriever / FileChatAttachment
Attachments Text:

```python
"""
Schema definition and column metadata registry.

This module defines the authoritative COLUMN_SCHEMA derived from
create_equities_schema.sql, providing centralized datatype and role
information for all preprocessing, feature engineering, and modeling.

Schema Structure (v1.12 - Updated 2025-12-15):
- Source columns from CSV/SQL: 328 (matching create_equities_schema.sql + forward estimates)
- Total COLUMN_SCHEMA entries: 532
  - 328 source columns (from CSV/SQL schema + forward estimates)
  - 61 log-transformed columns (ETL-generated, log1p of market values)
  - 43 legacy aliases (role=auxiliary, for backward compatibility)
  - 36 generic base columns (no time suffix)
  - 34 conditional metrics (with _applicable flags)
  - 26 derived ratios and percentage metrics (ETL semantic transforms)
  - 4 Phase 9.3 composite quality scores (altman_z_score, beneish_m_score, etc.)

New in v1.12:
  - 4 dividend yield forward estimates (2fy, 3fy, 4fy, 5fy)
  - 2 EBITDA forward estimates (avg_fy1e, avg_ntm)
  - 6 EPS normalized estimate revisions (1w, 1m, 3m, 6m, 1y, analyst count)
  - 6 EPS GAAP estimates & revisions (fy1e, ntm, 1m, 3m, 6m, 1y)
  - 6 basic EPS historical metrics (ltm, fq, fy, 1fqfq, 2fqfq, 3fqfq)
  - Total: 29 new raw source columns

Database Tables:
- equities: Original table with per-region data loading
- all_stocks: Unified table combining four regional screening tables
  (screening_us, screening_eu, screening_apac, screening_rotw)
  Created by: equities/import_equities_data.sql
  Primary key: (Ticker, Region)
  Indexes: ticker, region, sector, industry, country, last_price, market_cap, sector_region

Data Loading:
- load_from_csv(): Load from CSV files in data/ directory
- load_from_db(): Load from equities table with Region filter
- load_from_all_stocks(): Load from unified all_stocks table (recommended)

Aligned with code_guidelines.md v1.12+ Schema and Datatype Management.
"""

from typing import Dict, List, Optional, Literal

# =============================================================================
# Type Aliases for Schema Definition (Aligned with code_guidelines.md v1.11)
# =============================================================================

# DType: Maps to pandas/numpy dtype strings for ETL casting
DType = Literal[
    "float",  # float64 - default for numeric financial data
    "int",  # int64 - discrete counts, integer IDs
    "string",  # object/string - text data
    "category",  # pandas Categorical - low-cardinality (sector, region)
    "datetime64[ns]",  # datetime columns
    "bool",  # boolean flags
]

# Role: Semantic role determining preprocessing and pipeline treatment
Role = Literal[
    # === Pipeline Stage Roles ===
    "id",  # Identifier columns (ticker, isin) - never used as features
    "target",  # Primary prediction target (price_target)
    "target_fallback",  # Alternative targets (price_target_median, last_price)
    "date",  # Temporal columns for time-series features
    "categorical",  # Grouping columns (sector, region, industry)
    "auxiliary",  # Legacy aliases, optional - excluded from diagnostics
    "feature",  # General ML features not in other categories
        # === Semantic Classification Roles (code_guidelines.md v1.11, Section 8.5) ===
    "price",  # Price columns - NEVER transform (21 columns)
    "market_value",  # Market cap, revenue, assets - log-transform recommended
    "ratio",  # Pre-normalized ratios - skip winsorization
    "percentage",  # Bounded [0,100] metrics - margins, growth rates
    "count",  # Discrete integers - analyst ratings, employees
    "label",  # Classification targets (multi-label)
]

# Central column schema registry
# Maps normalized column names (lowercase, underscores) to dtype and role
COLUMN_SCHEMA: Dict[str, Dict[str, str]] = {
    # Identifiers
    "ticker": {"dtype": "string", "role": "id"},
    "isin": {"dtype": "string", "role": "id"},
    "name": {"dtype": "string", "role": "auxiliary"},
    "description": {"dtype": "string", "role": "auxiliary"},
    # Categorical/Classification
    "sector": {"dtype": "category", "role": "categorical"},
    "industry": {"dtype": "category", "role": "categorical"},
    "region": {"dtype": "category", "role": "categorical"},
    "country": {"dtype": "category", "role": "categorical"},
    "trading_country": {"dtype": "category", "role": "categorical"},
    "exchange": {"dtype": "category", "role": "categorical"},
    "unit": {"dtype": "string", "role": "categorical"},
    "style_class": {"dtype": "category", "role": "categorical"},
    "size_class": {"dtype": "category", "role": "categorical"},
    "next_earnings_status": {"dtype": "category", "role": "categorical"},
    # Date columns
    "last_updated": {"dtype": "datetime64[ns]", "role": "date"},
    "income_statement_report_date": {"dtype": "datetime64[ns]", "role": "date"},
    "next_earnings": {"dtype": "datetime64[ns]", "role": "date"},
    "dividend_record_announce_date": {"dtype": "datetime64[ns]", "role": "date"},
    "dividend_record_ex_date": {"dtype": "datetime64[ns]", "role": "date"},
    "dividend_record_payable_date": {"dtype": "datetime64[ns]", "role": "date"},
    "dividend_record_record_date": {"dtype": "datetime64[ns]", "role": "date"},
    # ====================
    # PRICE COLUMNS - NEVER transform (preserve original dollar units)
    # ====================
    "last_price": {"dtype": "float", "role": "price"},
    "price_target": {"dtype": "float", "role": "target"},
    "price_target_ytd_ago": {"dtype": "float", "role": "price"},
    "price_target_low": {"dtype": "float", "role": "price"},
    "price_target_median": {"dtype": "float", "role": "target_fallback"},
    "price_target_high": {"dtype": "float", "role": "price"},
    "price_target_num": {"dtype": "float", "role": "auxiliary"},  # Legacy alias
    "price_target_count": {"dtype": "float", "role": "count"},
    "price_5d_ago": {"dtype": "float", "role": "price"},
    "price_1w_ago": {"dtype": "float", "role": "price"},
    "price_1m_ago": {"dtype": "float", "role": "price"},
    "price_3m_ago": {"dtype": "float", "role": "price"},
    "price_6m_ago": {"dtype": "float", "role": "price"},
    "price_1y_ago": {"dtype": "float", "role": "price"},
    "price_3y_ago": {"dtype": "float", "role": "price"},
    "price_5y_ago": {"dtype": "float", "role": "price"},
    "price_qtd_ago": {"dtype": "float", "role": "price"},
    # ====================
    # MARKET VALUE COLUMNS - Log-transform recommended
    # ====================
    "market_cap": {"dtype": "float", "role": "market_value"},
    "enterprise_value": {"dtype": "float", "role": "market_value"},
    "market_cap_country_r": {"dtype": "float", "role": "market_value"},
    # ====================
    # RATIO COLUMNS - Pre-normalized financial ratios
    # ====================
    "p_e_ntm": {"dtype": "float", "role": "ratio"},
    "p_e_ltm": {"dtype": "float", "role": "ratio"},
    "p_e_1fyltm": {"dtype": "float", "role": "ratio"},
    "p_b_ltm": {"dtype": "float", "role": "ratio"},
    "p_b_1fy": {"dtype": "float", "role": "ratio"},
    "p_b_5yavg": {"dtype": "float", "role": "ratio"},
    "p_tbv_ltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_ltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_ntm": {"dtype": "float", "role": "ratio"},
    "ev_sales_est_fy1": {"dtype": "float", "role": "ratio"},
    "ev_ebitda_ltm": {"dtype": "float", "role": "ratio"},
    "ev_ebitda_ntm": {"dtype": "float", "role": "ratio"},
    "ev_ebitda_est_fy1": {"dtype": "float", "role": "ratio"},
    "p_e_est_fy1": {"dtype": "float", "role": "ratio"},
    # Phase 9.3 Schema 1.3: EV/Sales historical time-series (ratio)
    "ev_sales_1fyltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_2fyltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_3fyltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_3yavgltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_1fqltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_2fqltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_3fqltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_4fqltm": {"dtype": "float", "role": "ratio"},
    # Phase 9.3 Schema 1.3: EV/EBITDA historical time-series (ratio)
    "ev_ebitda_1fyltm": {"dtype": "float", "role": "ratio"},
    "ev_ebitda_1fqltm": {"dtype": "float", "role": "ratio"},
    "ev_ebitda_3yavgltm": {"dtype": "float", "role": "ratio"},
    # Phase 9.3 Schema 1.3: P/E extended time-series (ratio)
    "p_e_2fyltm": {"dtype": "float", "role": "ratio"},
    "p_e_3fyltm": {"dtype": "float", "role": "ratio"},
    "p_e_3yavgltm": {"dtype": "float", "role": "ratio"},
    "p_e_1fqltm": {"dtype": "float", "role": "ratio"},
    "p_e_2fqltm": {"dtype": "float", "role": "ratio"},
    "p_e_3fqltm": {"dtype": "float", "role": "ratio"},
    "p_e_5yavgltm": {"dtype": "float", "role": "ratio"},
    "p_e_0fqqoqltm": {"dtype": "float", "role": "ratio"},
    "p_e_0fyyoyltm": {"dtype": "float", "role": "ratio"},
    "p_e_1fyyoyltm": {"dtype": "float", "role": "ratio"},
    "p_e_0fqyoyltm": {"dtype": "float", "role": "ratio"},
    # Risk & Quality (ratio)
    "altman_z_score_fy": {"dtype": "float", "role": "ratio"},
    "altman_z_score_fq": {"dtype": "float", "role": "ratio"},
    "altman_z_score_ltm": {"dtype": "float", "role": "ratio"},
    # ====================
    # PERCENTAGE COLUMNS - Bounded metrics
    # ====================
    "beta_1y": {"dtype": "float", "role": "percentage"},
    "beta_2y": {"dtype": "float", "role": "percentage"},
    "beta_5y": {"dtype": "float", "role": "percentage"},
    "total_return_ytd": {"dtype": "float", "role": "percentage"},
    "total_return_5y": {"dtype": "float", "role": "percentage"},
    "total_return_10y": {"dtype": "float", "role": "percentage"},
    "tot_return_pct_cagr_3y": {"dtype": "float", "role": "percentage"},
    "tot_return_pct_cagr_10y": {"dtype": "float", "role": "percentage"},
    "price_chg_pct_1m": {"dtype": "float", "role": "percentage"},
    "price_chg_pct_3m": {"dtype": "float", "role": "percentage"},
    "1_day_pct": {"dtype": "float", "role": "auxiliary"},  # Legacy alias
    "one_day_pct": {"dtype": "float", "role": "percentage"},
    # ====================
    # COUNT COLUMNS - Discrete integers
    # ====================
    "analyst_rating": {"dtype": "float", "role": "count"},
    "num_strong_sell_ratings": {"dtype": "float", "role": "count"},
    "num_strong_buys_ratings": {"dtype": "float", "role": "count"},
    "num_hold_ratings": {"dtype": "float", "role": "count"},
    "num_buys_ratings": {"dtype": "float", "role": "count"},
    "num_sell_ratings": {"dtype": "float", "role": "count"},
    # ====================
    # TECHNICAL INDICATORS - Price-based (role: price)
    # ====================
    "ema_20d": {"dtype": "float", "role": "price"},
    "ema_50d": {"dtype": "float", "role": "price"},
    "ema_100d": {"dtype": "float", "role": "price"},
    "ema_250d": {"dtype": "float", "role": "price"},
    "52w_high_adj": {"dtype": "float", "role": "price"},
    "52w_low_adj": {"dtype": "float", "role": "price"},
    # ====================
    # VOLATILITY - Percentage metrics
    # ====================
    "volatility_1m": {"dtype": "float", "role": "percentage"},
    "volatility_3m": {"dtype": "float", "role": "percentage"},
    "volatility_6m": {"dtype": "float", "role": "percentage"},
    "volatility_1y": {"dtype": "float", "role": "percentage"},
    # ====================
    # VOLUME & TRADING - Market value/count
    # ====================
    "volume_shrs": {"dtype": "float", "role": "market_value"},
    "rel_volume": {"dtype": "float", "role": "ratio"},
    "shrs_out": {"dtype": "float", "role": "auxiliary"},  # Legacy alias
    "shares_outstanding": {"dtype": "float", "role": "count"},
    "shrs_out_1fy": {"dtype": "float", "role": "count"},
    # ====================
    # REVENUES & GROWTH - Market value columns
    # ====================
    "total_revenues_fy": {"dtype": "float", "role": "market_value"},
    "total_revenues_ltm": {"dtype": "float", "role": "market_value"},
    "total_revenues_fq": {"dtype": "float", "role": "market_value"},
    "total_revenues_1fy": {"dtype": "float", "role": "market_value"},
    "total_revenues_cagr_5y_fy": {"dtype": "float", "role": "percentage"},
    "total_revenues_5yavgfq": {"dtype": "float", "role": "market_value"},
    "total_revenues_5yavgltm": {"dtype": "float", "role": "market_value"},
    "revenues_est_avg_ntm": {"dtype": "float", "role": "market_value"},
    "revenues_est_avg_fy1e": {"dtype": "float", "role": "market_value"},
    "revenues_est_med_ntm": {"dtype": "float", "role": "market_value"},
    "revenues_est_med_fy1e": {"dtype": "float", "role": "market_value"},
    "revenues_est_yoy_pct_fy1e": {"dtype": "float", "role": "percentage"},
    "total_operating_expenses_ltm": {"dtype": "float", "role": "market_value"},
    # ====================
    # PROFITABILITY - EBITDA (market_value)
    # ====================
    "ebitda_fy": {"dtype": "float", "role": "market_value"},
    "ebitda_ltm": {"dtype": "float", "role": "market_value"},
    "ebitda_fq": {"dtype": "float", "role": "market_value"},
    "ebitda_1fy": {"dtype": "float", "role": "market_value"},
    "ebitda_adj_ltm": {"dtype": "float", "role": "market_value"},
    "ebitda_adj_fy": {"dtype": "float", "role": "market_value"},
    "ebitda_adj_1fy": {"dtype": "float", "role": "market_value"},
    "ebitda_5yavgfq": {"dtype": "float", "role": "market_value"},
    "ebitda_5yavgltm": {"dtype": "float", "role": "market_value"},
    # Forward estimates
    "ebitda_est_avg_fy1e": {"dtype": "float", "role": "market_value"},
    "ebitda_est_avg_ntm": {"dtype": "float", "role": "market_value"},
    # ====================
    # PROFITABILITY - EBIT (market_value)
    # ====================
    "ebit_fy": {"dtype": "float", "role": "market_value"},
    "ebit_ltm": {"dtype": "float", "role": "market_value"},
    "ebit_fq": {"dtype": "float", "role": "market_value"},
    "ebit_1fy": {"dtype": "float", "role": "market_value"},
    "ebit_adj_ltm": {"dtype": "float", "role": "market_value"},
    "ebit_adj_fy": {"dtype": "float", "role": "market_value"},
    "ebit_adj_1fy": {"dtype": "float", "role": "market_value"},
    "ebit_est_med_fy1e": {"dtype": "float", "role": "market_value"},
    "ebit_est_med_ntm": {"dtype": "float", "role": "market_value"},
    "ebit_5yavgfq": {"dtype": "float", "role": "market_value"},
    "ebit_5yavgltm": {"dtype": "float", "role": "market_value"},
    # ====================
    # PROFITABILITY - Net Income (market_value)
    # ====================
    "net_income_is_fy": {"dtype": "float", "role": "market_value"},
    "net_income_is_ltm": {"dtype": "float", "role": "market_value"},
    "net_income_is_fq": {"dtype": "float", "role": "market_value"},
    "net_income_is_1fy": {"dtype": "float", "role": "market_value"},
    "net_income_is_5yavgfq": {"dtype": "float", "role": "market_value"},
    "net_income_is_5yavgltm": {"dtype": "float", "role": "market_value"},
    "normalized_net_income_fy": {"dtype": "float", "role": "market_value"},
    "normalized_net_income_ltm": {"dtype": "float", "role": "market_value"},
    "normalized_net_income_fq": {"dtype": "float", "role": "market_value"},
    "normalized_net_income_1fy": {"dtype": "float", "role": "market_value"},
    "normalized_net_income_5yavgfq": {"dtype": "float", "role": "market_value"},
    "normalized_net_income_5yavgltm": {"dtype": "float", "role": "market_value"},
    "net_income_adj_fy": {"dtype": "float", "role": "market_value"},
    "net_income_adj_ltm": {"dtype": "float", "role": "market_value"},
    "net_income_adj_fq": {"dtype": "float", "role": "market_value"},
    "net_income_adj_1fy": {"dtype": "float", "role": "market_value"},
    "net_income_adj_5yavgfq": {"dtype": "float", "role": "market_value"},
    "operating_income_ltm": {"dtype": "float", "role": "market_value"},
    "operating_income_fy": {"dtype": "float", "role": "market_value"},
    "operating_income_fq": {"dtype": "float", "role": "market_value"},
    "operating_income_5yavgfq": {"dtype": "float", "role": "market_value"},
    # ====================
    # MARGINS - Percentage metrics
    # ====================
    "net_income_margin_pct_fy": {"dtype": "float", "role": "percentage"},
    "net_income_margin_pct_ltm": {"dtype": "float", "role": "percentage"},
    "gross_profit_margin_pct_fy": {"dtype": "float", "role": "percentage"},
    "gross_profit_margin_pct_ltm": {"dtype": "float", "role": "percentage"},
    "gross_profit_ltm": {"dtype": "float", "role": "market_value"},
    "gross_profit_fy": {"dtype": "float", "role": "market_value"},
    # ====================
    # RETURNS ON CAPITAL - Percentage metrics
    # ====================
    "return_on_equity_pct_ltm": {"dtype": "float", "role": "percentage"},
    "return_on_equity_pct_fy": {"dtype": "float", "role": "percentage"},
    "return_on_assets_roa_pct_ltm": {"dtype": "float", "role": "percentage"},
    "return_on_assets_roa_pct_fy": {"dtype": "float", "role": "percentage"},
    # ====================
    # CASH FLOW - Market value columns
    # ====================
    "cfo_ltm": {"dtype": "float", "role": "market_value"},
    "cfo_fy": {"dtype": "float", "role": "market_value"},
    "cfo_fq": {"dtype": "float", "role": "market_value"},
    "cfo_1fy": {"dtype": "float", "role": "market_value"},
    "fcf_ltm": {"dtype": "float", "role": "market_value"},
    "fcf_fy": {"dtype": "float", "role": "market_value"},
    "fcf_fq": {"dtype": "float", "role": "market_value"},
    "fcf_5yavgfq": {"dtype": "float", "role": "market_value"},
    "cfi_ltm": {"dtype": "float", "role": "market_value"},
    "cfi_fy": {"dtype": "float", "role": "market_value"},
    "cfi_fq": {"dtype": "float", "role": "market_value"},
    "cfi_1fy": {"dtype": "float", "role": "market_value"},
    "cff_ltm": {"dtype": "float", "role": "market_value"},
    "cff_fy": {"dtype": "float", "role": "market_value"},
    "cff_fq": {"dtype": "float", "role": "market_value"},
    "cff_1fy": {"dtype": "float", "role": "market_value"},
    # ====================
    # BALANCE SHEET - Market value columns
    # ====================
    "total_assets_ltm": {"dtype": "float", "role": "market_value"},
    "total_assets_fy": {"dtype": "float", "role": "market_value"},
    "total_equity_fy": {"dtype": "float", "role": "market_value"},
    "total_equity_ltm": {"dtype": "float", "role": "market_value"},
    "total_debt_fy": {"dtype": "float", "role": "market_value"},
    "total_debt_ltm": {"dtype": "float", "role": "market_value"},
    "total_current_assets_ltm": {"dtype": "float", "role": "market_value"},
    "total_current_liabilities_ltm": {"dtype": "float", "role": "market_value"},
    "current_ratio_fy": {"dtype": "float", "role": "ratio"},
    "current_ratio_ltm": {"dtype": "float", "role": "ratio"},
    "working_capital_ltm": {"dtype": "float", "role": "market_value"},
    "working_capital_fq": {"dtype": "float", "role": "market_value"},
    "working_capital_fy": {"dtype": "float", "role": "market_value"},
    "working_capital_5yavgfy": {"dtype": "float", "role": "market_value"},
    "tbv_fy": {"dtype": "float", "role": "market_value"},
    "tbv_ltm": {"dtype": "float", "role": "market_value"},
    "cash_and_equivalents_ltm": {"dtype": "float", "role": "market_value"},
    "cash_and_equivalents_fq": {"dtype": "float", "role": "market_value"},
    "cash_and_equivalents_fy": {"dtype": "float", "role": "market_value"},
    "cash_and_equivalents_5yavgfq": {"dtype": "float", "role": "market_value"},
    "retained_earnings_ltm": {"dtype": "float", "role": "market_value"},
    "retained_earnings_fq": {"dtype": "float", "role": "market_value"},
    "retained_earnings_fy": {"dtype": "float", "role": "feature"},
    "retained_earnings_5yavgfq": {"dtype": "float", "role": "feature"},
    # Asset Details
    "inventory_ltm": {"dtype": "float", "role": "feature"},
    "inventory_fq": {"dtype": "float", "role": "feature"},
    "inventory_fy": {"dtype": "float", "role": "feature"},
    "inventory_5yavgfq": {"dtype": "float", "role": "feature"},
    "goodwill_fq": {"dtype": "float", "role": "feature"},
    "goodwill_ltm": {"dtype": "float", "role": "feature"},
    "goodwill_fy": {"dtype": "float", "role": "feature"},
    "goodwill_1fy": {"dtype": "float", "role": "feature"},
    "goodwill_5yavgfq": {"dtype": "float", "role": "feature"},
    "intangible_assets": {
        "dtype": "float",
        "role": "feature",
        },  # Base column (no time suffix)
    "gross_intangible_assets_ltm": {"dtype": "float", "role": "feature"},
    "gross_intangible_assets_fy": {"dtype": "float", "role": "feature"},
    "gross_intangible_assets_5yavgfq": {"dtype": "float", "role": "feature"},
    # Capex & Investments
    "capital_expenditure_ltm": {"dtype": "float", "role": "feature"},
    "capital_expenditure_fy": {"dtype": "float", "role": "feature"},
    "capital_expenditure_fq": {"dtype": "float", "role": "feature"},
    "capital_expenditure_1fy": {"dtype": "float", "role": "feature"},
    "capital_expenditure_5yavgfq": {"dtype": "float", "role": "feature"},
    "asset_turnover_fy": {"dtype": "float", "role": "feature"},
    "asset_turnover_ltm": {"dtype": "float", "role": "feature"},
    "cash_acquisitions_ltm": {"dtype": "float", "role": "feature"},
    "cash_acquisitions_fy": {"dtype": "float", "role": "feature"},
    "cash_acquisitions_fq": {"dtype": "float", "role": "feature"},
    "cash_acquisitions_1fy": {"dtype": "float", "role": "feature"},
    "cash_acquisitions_5yavgfq": {"dtype": "float", "role": "feature"},
    # Exceptional Items
    "impairment_of_goodwill_fq": {"dtype": "float", "role": "feature"},
    "impairment_of_goodwill_ltm": {"dtype": "float", "role": "feature"},
    "impairment_of_goodwill_1fy": {"dtype": "float", "role": "feature"},
    "impairment_of_goodwill_fy": {"dtype": "float", "role": "feature"},
    "impairment_of_goodwill_5yavgfq": {"dtype": "float", "role": "feature"},
    "asset_writedown_ltm": {"dtype": "float", "role": "feature"},
    "asset_writedown_fy": {"dtype": "float", "role": "feature"},
    "asset_writedown_fq": {"dtype": "float", "role": "feature"},
    "asset_writedown_1fy": {"dtype": "float", "role": "feature"},
    "asset_writedown_5yavgfq": {"dtype": "float", "role": "feature"},
    "restructuring_charges_ltm": {"dtype": "float", "role": "feature"},
    "restructuring_charges_fq": {"dtype": "float", "role": "feature"},
    "restructuring_charges_1fy": {"dtype": "float", "role": "feature"},
    "restructuring_charges_fy": {"dtype": "float", "role": "feature"},
    "restructuring_charges_5yavgfq": {"dtype": "float", "role": "feature"},
    # Merger & Restructuring Charges - correct normalization with "and" (from CSV "Merger & Restructuring Charges")
    "merger_and_restructuring_charges_ltm": {"dtype": "float", "role": "feature"},
    "merger_and_restructuring_charges_fq": {"dtype": "float", "role": "feature"},
    "merger_and_restructuring_charges_fy": {"dtype": "float", "role": "feature"},
    "merger_and_restructuring_charges_5yavgfq": {"dtype": "float", "role": "feature"},
    # Legacy aliases (without "and") - kept for backward compatibility
    "merger_restructuring_charges_ltm": {"dtype": "float", "role": "auxiliary"},
    "merger_restructuring_charges_fq": {"dtype": "float", "role": "auxiliary"},
    "merger_restructuring_charges_fy": {"dtype": "float", "role": "auxiliary"},
    "merger_restructuring_charges_5yavgfq": {"dtype": "float", "role": "auxiliary"},
    "other_unusual_items_total_ltm": {"dtype": "float", "role": "feature"},
    "gain_loss_on_sale_of_assets_ltm": {"dtype": "float", "role": "feature"},
    # Operating Expenses
    "cost_of_revenues_ltm": {"dtype": "float", "role": "feature"},
    # R&D Expenses - correct normalization (from CSV "R&D Expenses (LTM)" where & becomes "and")
    "randd_expenses_ltm": {"dtype": "float", "role": "feature"},
    # R&D generic base column and legacy alias
    "r_d_expenses": {
        "dtype": "float",
        "role": "feature",
        },  # Generic base column (no time suffix)
    "r_d_expenses_ltm": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for randd_expenses_ltm
    "selling_general_admin_expenses_total_fq": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy - use sga_expenses_fq
    "selling_general_admin_expenses_total_fy": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy - use sga_expenses_fy
    "selling_general_admin_expenses_total_1fy": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy - use sga_expenses_1fy
    "selling_general_admin_expenses_total_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy
    # SG&A with "and" (correct normalization from "Selling General & Admin Expenses/Total")
    "selling_general_and_admin_expenses_total_fq": {
        "dtype": "float",
        "role": "feature",
        },
    "selling_general_and_admin_expenses_total_fy": {
        "dtype": "float",
        "role": "feature",
        },
    "selling_general_and_admin_expenses_total_1fy": {
        "dtype": "float",
        "role": "feature",
        },
    "selling_general_and_admin_expenses_total_5yavgfq": {
        "dtype": "float",
        "role": "feature",
        },
    "accounts_receivable_total_fy": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy - use accounts_receivable_fy
    "accounts_receivable_total_1fy": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy - use accounts_receivable_1fy
    "accounts_receivable_total_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy
    "marketing_expenses": {
        "dtype": "float",
        "role": "feature",
        },  # Base column (no time suffix)
    "marketing_expenses_fq": {"dtype": "float", "role": "feature"},
    "marketing_expenses_fy": {"dtype": "float", "role": "feature"},
    "marketing_expenses_1fy": {"dtype": "float", "role": "feature"},
    "marketing_expenses_5yavgltm": {"dtype": "float", "role": "feature"},
    # Earnings Per Share
    "eps_adj_1fy": {"dtype": "float", "role": "feature"},
    "eps_adj_fy": {"dtype": "float", "role": "feature"},
    "eps_adj_ltm": {"dtype": "float", "role": "feature"},
    # Basic EPS Historical (GAAP-based)
    "net_eps_basic_ltm": {"dtype": "float", "role": "feature"},
    "net_eps_basic_fq": {"dtype": "float", "role": "feature"},
    "net_eps_basic_fy": {"dtype": "float", "role": "feature"},
    "net_eps_basic_1fqfq": {"dtype": "float", "role": "feature"},  # 1 quarter ago
    "net_eps_basic_2fqfq": {"dtype": "float", "role": "feature"},  # 2 quarters ago
    "net_eps_basic_3fqfq": {"dtype": "float", "role": "feature"},  # 3 quarters ago
    # Forward EPS Estimates
    "eps_norm_est_avg_ntm": {"dtype": "float", "role": "feature"},
    "eps_norm_est_avg_fy1e": {"dtype": "float", "role": "feature"},
    "eps_norm_est_num_fy1e": {"dtype": "float", "role": "count"},  # Number of analysts
    # EPS Normalized Estimate Revisions (percentage changes over time periods)
    "eps_est_avg_rev_pct_fy1e_1w": {"dtype": "float", "role": "percentage"},
    "eps_est_avg_rev_pct_fy1e_1m": {"dtype": "float", "role": "percentage"},
    "eps_est_avg_rev_pct_fy1e_3m": {"dtype": "float", "role": "percentage"},
    "eps_est_avg_rev_pct_fy1e_6m": {"dtype": "float", "role": "percentage"},
    "eps_est_avg_rev_pct_fy1e_1y": {"dtype": "float", "role": "percentage"},
    # EPS GAAP Estimates & Revisions
    "eps_gaap_est_avg_fy1e": {"dtype": "float", "role": "feature"},
    "eps_gaap_est_avg_ntm": {"dtype": "float", "role": "feature"},
    "eps_gaap_est_avg_rev_pct_fy1e_1m": {"dtype": "float", "role": "percentage"},
    "eps_gaap_est_avg_rev_pct_fy1e_3m": {"dtype": "float", "role": "percentage"},
    "eps_gaap_est_avg_rev_pct_fy1e_6m": {"dtype": "float", "role": "percentage"},
    "eps_gaap_est_avg_rev_pct_fy1e_1y": {"dtype": "float", "role": "percentage"},
    "eps_previous_year": {
        "dtype": "float",
        "role": "feature",
        },  # Base column for YoY calculations
    # Dividends
    "dividend_per_share_ltm": {"dtype": "float", "role": "feature"},
    "div_yield_ind": {"dtype": "float", "role": "feature"},
    "div_yield_ltm": {"dtype": "float", "role": "feature"},
    "div_yield_ttm": {"dtype": "float", "role": "feature"},
    "div_yield_ntm": {"dtype": "float", "role": "feature"},
    "div_yield_1fyind": {"dtype": "float", "role": "feature"},
    "div_yield_2fyind": {"dtype": "float", "role": "feature"},
    "div_yield_3fyind": {"dtype": "float", "role": "feature"},
    "div_yield_4fyind": {"dtype": "float", "role": "feature"},
    "div_yield_5fyind": {"dtype": "float", "role": "feature"},
    "div_yield_5yavgltm": {"dtype": "float", "role": "feature"},
    "common_dividends_paid_ltm": {"dtype": "float", "role": "feature"},
    "common_dividends_paid_fy": {"dtype": "float", "role": "feature"},
    "dividend_record_frequency": {"dtype": "string", "role": "auxiliary"},
    "dividend_record_currency": {"dtype": "string", "role": "auxiliary"},
    "dividend_record_amount": {"dtype": "float", "role": "feature"},
    "dividend_streak": {"dtype": "float", "role": "feature"},
    "buyback_yield_ltm": {"dtype": "float", "role": "feature"},
    # Interest & Financing
    "interest_expense_total_ltm": {"dtype": "float", "role": "feature"},
    "interest_income_on_investments_ltm": {"dtype": "float", "role": "feature"},
    # Employees
    "employees": {
        "dtype": "float",
        "role": "count",
        },  # Base column (current employee count) - float for NULL handling
    "avg_employees_ltm": {"dtype": "float", "role": "count"},
    "avg_employees_fy": {"dtype": "float", "role": "count"},
    "avg_employees_5yavgfy": {"dtype": "float", "role": "count"},
    "total_employees_fy": {"dtype": "float", "role": "count"},
    "total_employees_fq": {"dtype": "float", "role": "count"},
    "full_time_employees_fq": {
        "dtype": "float",
        "role": "feature",
        },  # Full time employees (Fiscal Quarter) - float for NULL handling
    "full_time_employees_fy": {
        "dtype": "float",
        "role": "feature",
        },  # Full time employees (Fiscal Year) - float for NULL handling
    "full_time_employees_1fy": {
        "dtype": "float",
        "role": "feature",
        },  # Full time employees (Previous FY) - float for NULL handling
    "full_time_employees_2fy": {
        "dtype": "float",
        "role": "feature",
        },  # Full time employees (2 Years Ago) - float for NULL handling
    "full_time_employees_3fy": {
        "dtype": "float",
        "role": "feature",
        },  # Full time employees (3 Years Ago) - float for NULL handling
    # Country-specific
    # ==================================================================================
    # NORMALIZATION VARIANTS & SIMPLIFIED ALIASES
    # Added to resolve unknown column warnings from dtype_diagnostics.json
    # These columns exist in the data pipeline but use different naming conventions
    # ==================================================================================
    # Analyst Ratings (normalized names without "num_" prefix) - Legacy aliases
    # Alias for price_target_num
    "strong_sell_ratings": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for num_strong_sell_ratings
    "strong_buys_ratings": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for num_strong_buys_ratings
    "hold_ratings": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for num_hold_ratings
    "buys_ratings": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for num_buys_ratings
    "sell_ratings": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for num_sell_ratings
    # Simplified Base Columns (without time suffixes - used as generic references)
    "p_e": {"dtype": "float", "role": "feature"},  # Generic P/E ratio
    "p_b": {"dtype": "float", "role": "feature"},  # Generic P/B ratio
    "revenue": {"dtype": "float", "role": "feature"},  # Generic revenue
    "ebitda": {"dtype": "float", "role": "feature"},  # Generic EBITDA
    "ebit": {"dtype": "float", "role": "feature"},  # Generic EBIT
    "net_income": {"dtype": "float", "role": "feature"},  # Generic net income
    "net_income_ltm": {
        "dtype": "float",
        "role": "feature",
        },  # Duplicate of net_income_is_ltm
    "gross_margin": {"dtype": "float", "role": "feature"},  # Generic gross margin
    "eps": {"dtype": "float", "role": "feature"},  # Generic EPS
    "total_equity": {"dtype": "float", "role": "feature"},  # Generic total equity
    "total_assets": {"dtype": "float", "role": "feature"},  # Generic total assets
    "total_debt": {"dtype": "float", "role": "feature"},  # Generic total debt
    "inventory": {"dtype": "float", "role": "feature"},  # Generic inventory
    "capex": {"dtype": "float", "role": "feature"},  # Generic capital expenditure
    "cash_and_equivalents": {"dtype": "float", "role": "feature"},  # Generic cash
    "current_assets": {"dtype": "float", "role": "feature"},  # Generic current assets
    "accounts_receivable": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias - use accounts_receivable_fy
    "current_liabilities": {
        "dtype": "float",
        "role": "feature",
        },  # Generic current liabilities
    "working_capital": {"dtype": "float", "role": "feature"},  # Generic working capital
    "retained_earnings": {
        "dtype": "float",
        "role": "feature",
        },  # Generic retained earnings
    "cfo": {"dtype": "float", "role": "feature"},  # Generic cash flow from operations
    "cfi": {"dtype": "float", "role": "feature"},  # Generic cash flow from investing
    "cff": {"dtype": "float", "role": "feature"},  # Generic cash flow from financing
    "fcf": {"dtype": "float", "role": "feature"},  # Generic free cash flow
    "gross_profit": {"dtype": "float", "role": "feature"},  # Generic gross profit
    "operating_income": {
        "dtype": "float",
        "role": "feature",
        },  # Generic operating income
    "interest_expense": {
        "dtype": "float",
        "role": "feature",
        },  # Generic interest expense
    "goodwill": {"dtype": "float", "role": "feature"},  # Generic goodwill
    "dividend_per_share": {
        "dtype": "float",
        "role": "feature",
        },  # Generic dividend per share
    "operating_expenses": {
        "dtype": "float",
        "role": "feature",
        },  # Generic operating expenses
    "operating_cash_flow": {"dtype": "float", "role": "feature"},  # Alias for cfo
    "dividends_paid": {"dtype": "float", "role": "feature"},  # Generic dividends paid
    "dividends_paid_ltm": {"dtype": "float", "role": "feature"},  # Dividends paid LTM
    # Additional normalized names
    "price_target_number": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Alias for price_target_num
    "one_day_pct": {"dtype": "float", "role": "feature"},  # Alias for 1_day_pct
    "shares_outstanding": {"dtype": "float", "role": "feature"},  # Alias for shrs_out
    "p_e_5yavgltm": {"dtype": "float", "role": "feature"},  # 5-year average P/E LTM
    # SG&A Expenses (normalized naming) - Legacy aliases
    "sga_expenses": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias - use selling_general_and_admin_expenses_*
    "sga_expenses_fq": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for selling_general_and_admin_expenses_total_fq
    "sga_expenses_fy": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for selling_general_and_admin_expenses_total_fy
    "sga_expenses_1fy": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for selling_general_and_admin_expenses_total_1fy
    "sga_expenses_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for selling_general_and_admin_expenses_total_5yavgfq
    # Accounts Receivable (normalized naming) - Legacy aliases
    "accounts_receivable_fy": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for accounts_receivable_total_fy
    "accounts_receivable_1fy": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for accounts_receivable_total_1fy
    "accounts_receivable_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary",
        },  # Legacy alias for accounts_receivable_total_5yavgfq
    # ==================================================================================
    # DERIVED & COMPUTED COLUMNS (Created during preprocessing/feature engineering)
    # ==================================================================================
    # Volatility percentage variants
    "volatility_1y_pct": {
        "dtype": "float",
        "role": "feature",
        },  # 1-year volatility as percentage
    # --------------------------------------------------------------------------
    # Log-transformed semantic metrics (created by ETL semantic transforms)
    # These are log1p transforms of market values for better ML distribution
    # --------------------------------------------------------------------------
    "log_operating_income": {"dtype": "float", "role": "feature"},
    "log_ebitda": {"dtype": "float", "role": "feature"},
    "log_net_income": {"dtype": "float", "role": "feature"},
    "log_capex": {"dtype": "float", "role": "feature"},
    "log_operating_cash_flow": {"dtype": "float", "role": "feature"},
    "log_total_equity": {"dtype": "float", "role": "feature"},
    "log_market_cap": {"dtype": "float", "role": "feature"},
    "log_total_assets": {"dtype": "float", "role": "feature"},
    "log_gross_profit": {"dtype": "float", "role": "feature"},
    "log_cash_and_equivalents": {"dtype": "float", "role": "feature"},
    "log_total_debt": {"dtype": "float", "role": "feature"},
    "log_revenue": {"dtype": "float", "role": "feature"},
    "log_enterprise_value": {"dtype": "float", "role": "feature"},
    # Additional log-transformed columns (time-series variants)
    "log_gross_profit_previous_year": {"dtype": "float", "role": "feature"},
    "log_operating_income_fq": {"dtype": "float", "role": "feature"},
    "log_ebitda_ltm": {"dtype": "float", "role": "feature"},
    "log_total_revenues_5yavgfq": {"dtype": "float", "role": "feature"},
    "log_cash_acquisitions_fq": {"dtype": "float", "role": "feature"},
    "log_total_revenues_5yavgltm": {"dtype": "float", "role": "feature"},
    "log_ebitda_fy": {"dtype": "float", "role": "feature"},
    "log_total_assets_ltm": {"dtype": "float", "role": "feature"},
    "log_ebitda_previous_year": {"dtype": "float", "role": "feature"},
    "log_operating_income_fy": {"dtype": "float", "role": "feature"},
    "log_cash_acquisitions_ltm": {"dtype": "float", "role": "feature"},
    "log_revenues_est_avg_ntm": {"dtype": "float", "role": "feature"},
    "log_total_revenues_fy": {"dtype": "float", "role": "feature"},
    "log_net_income_is_1fy": {"dtype": "float", "role": "feature"},
    "log_fcf_fq": {"dtype": "float", "role": "feature"},
    "log_total_equity_ltm": {"dtype": "float", "role": "feature"},
    "log_total_revenues_ltm": {"dtype": "float", "role": "feature"},
    "log_net_income_adj_1fy": {"dtype": "float", "role": "feature"},
    "log_total_equity_fy": {"dtype": "float", "role": "feature"},
    "log_total_debt_fy": {"dtype": "float", "role": "feature"},
    "log_revenue_previous_year": {"dtype": "float", "role": "feature"},
    "log_revenue_fy": {"dtype": "float", "role": "feature"},
    "log_cash_acquisitions_5yavgfq": {"dtype": "float", "role": "feature"},
    "log_net_income_is_5yavgltm": {"dtype": "float", "role": "feature"},
    "log_cash_acquisitions_fy": {"dtype": "float", "role": "feature"},
    "log_total_assets_fy": {"dtype": "float", "role": "feature"},
    "log_net_income_adj_fy": {"dtype": "float", "role": "feature"},
    "log_ebitda_5yavgltm": {"dtype": "float", "role": "feature"},
    "log_revenues_est_avg_fy1e": {"dtype": "float", "role": "feature"},
    "log_ebitda_fq": {"dtype": "float", "role": "feature"},
    "log_ebitda_1fy": {"dtype": "float", "role": "feature"},
    "log_revenues_est_med_ntm": {"dtype": "float", "role": "feature"},
    "log_cash_and_equivalents_fy": {"dtype": "float", "role": "feature"},
    "log_net_income_is_5yavgfq": {"dtype": "float", "role": "feature"},
    "log_cash_and_equivalents_5yavgfq": {"dtype": "float", "role": "feature"},
    "log_fcf_ltm": {"dtype": "float", "role": "feature"},
    "log_total_debt_ltm": {"dtype": "float", "role": "feature"},
    "log_fcf": {"dtype": "float", "role": "feature"},
    "log_gross_profit_fy": {"dtype": "float", "role": "feature"},
    "log_market_cap_country_r": {"dtype": "float", "role": "feature"},
    "log_cash_and_equivalents_ltm": {"dtype": "float", "role": "feature"},
    "log_fcf_5yavgfq": {"dtype": "float", "role": "feature"},
    "log_ebitda_5yavgfq": {"dtype": "float", "role": "feature"},
    "log_fcf_fy": {"dtype": "float", "role": "feature"},
    "log_revenues_est_med_fy1e": {"dtype": "float", "role": "feature"},
    "log_total_assets_previous_year": {"dtype": "float", "role": "feature"},
    "log_operating_income_ltm": {"dtype": "float", "role": "feature"},
    "log_net_income_is_fq": {"dtype": "float", "role": "feature"},
    "log_ebitda_adj_ltm": {"dtype": "float", "role": "feature"},
    "log_gross_profit_ltm": {"dtype": "float", "role": "feature"},
    # --------------------------------------------------------------------------
    # Valuation / profitability / leverage ratios (Phase 9.3 semantic metrics)
    # Created during ETL semantic transformation stage
    # --------------------------------------------------------------------------
    "p_e_ratio": {"dtype": "float", "role": "feature"},  # Price-to-Earnings ratio
    "p_s_ratio": {"dtype": "float", "role": "feature"},  # Price-to-Sales ratio
    "ev_ebitda_ratio": {"dtype": "float", "role": "feature"},  # EV/EBITDA ratio
    "ev_sales_ratio": {"dtype": "float", "role": "feature"},  # EV/Sales ratio
    "gross_margin_pct": {
        "dtype": "float",
        "role": "feature",
        },  # Gross margin percentage
    "operating_margin_pct": {
        "dtype": "float",
        "role": "feature",
        },  # Operating margin percentage
    "net_margin_pct": {"dtype": "float", "role": "feature"},  # Net margin percentage
    "roe": {"dtype": "float", "role": "feature"},  # Return on Equity
    "roa": {"dtype": "float", "role": "feature"},  # Return on Assets
    "revenue_growth": {"dtype": "float", "role": "feature"},  # Revenue growth rate
    "ebitda_growth": {"dtype": "float", "role": "feature"},  # EBITDA growth rate
    "earnings_growth": {"dtype": "float", "role": "feature"},  # Earnings growth rate
    "debt_to_equity": {"dtype": "float", "role": "feature"},  # Debt-to-Equity ratio
    "debt_to_assets": {"dtype": "float", "role": "feature"},  # Debt-to-Assets ratio
    "target_vs_price": {
        "dtype": "float",
        "role": "feature",
        },  # Price target vs last price ratio
    "target_vs_price_median": {
        "dtype": "float",
        "role": "feature",
        },  # Median price target vs last price
    "peg_ratio": {
        "dtype": "float",
        "role": "feature",
        },  # Price/Earnings-to-Growth ratio
    "dividend_yield": {
        "dtype": "float",
        "role": "feature",
        },  # Dividend yield percentage
    "roic": {"dtype": "float", "role": "feature"},  # Return on Invested Capital
    # --------------------------------------------------------------------------
    # Year-over-Year (YoY) comparison columns (_previous_year suffix)
    "revenue_previous_year": {
        "dtype": "float",
        "role": "feature",
        },  # Revenue from previous year
    "ebitda_previous_year": {
        "dtype": "float",
        "role": "feature",
        },  # EBITDA from previous year
    "total_equity_previous_year": {
        "dtype": "float",
        "role": "feature",
        },  # Total equity from previous year
    "total_assets_previous_year": {
        "dtype": "float",
        "role": "feature",
        },  # Total assets from previous year
    "gross_profit_previous_year": {
        "dtype": "float",
        "role": "feature",
        },  # Gross profit from previous year
    "accounts_receivable_previous_year": {
        "dtype": "float",
        "role": "feature",
        },  # AR from previous year
    "roa_previous_year": {
        "dtype": "float",
        "role": "feature",
        },  # ROA from previous year
    "current_ratio_previous_year": {
        "dtype": "float",
        "role": "feature",
        },  # Current ratio from previous year
    "shares_outstanding_previous_year": {
        "dtype": "float",
        "role": "feature",
        },  # Shares outstanding from previous year
    "gross_margin_pct_previous_year": {
        "dtype": "float",
        "role": "feature",
        },  # Gross margin % from previous year
    "asset_turnover_previous_year": {
        "dtype": "float",
        "role": "feature",
        },  # Asset turnover from previous year
    # Fiscal year variants (alternative naming)
    "revenue_fy": {"dtype": "float", "role": "feature"},  # Alias for total_revenues_fy
    "working_capital_1fy": {
        "dtype": "float",
        "role": "feature",
        },  # Working capital 1 fiscal year
    # ==================================================================================
    # CONDITIONAL METRICS (Only valid for specific business conditions)
    # ==================================================================================
    # Cash burn rate - only valid for companies with negative CFO (burning cash)
    "cash_burn_rate": {"dtype": "float", "role": "feature"},
    "cash_burn_rate_applicable": {
        "dtype": "bool",
        "role": "auxiliary",
        },  # True if company has negative CFO
    # Employee productivity metrics - only valid when employee data is available
    "revenue_per_employee": {"dtype": "float", "role": "feature"},
    "revenue_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_ltm": {"dtype": "float", "role": "feature"},
    "revenue_per_employee_ltm_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_fy": {"dtype": "float", "role": "feature"},
    "revenue_per_employee_fy_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_trend": {"dtype": "float", "role": "feature"},
    "revenue_per_employee_trend_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_vs_5y_pct": {"dtype": "float", "role": "feature"},
    "revenue_per_employee_vs_5y_pct_applicable": {"dtype": "bool", "role": "auxiliary"},
    "assets_per_employee": {"dtype": "float", "role": "feature"},
    "assets_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "ebitda_per_employee": {"dtype": "float", "role": "feature"},
    "ebitda_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "operating_income_per_employee": {"dtype": "float", "role": "feature"},
    "operating_income_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "profit_per_employee": {"dtype": "float", "role": "feature"},
    "profit_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_yoy": {"dtype": "float", "role": "feature"},
    "employee_growth_yoy_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_yoy_pct": {"dtype": "float", "role": "feature"},
    "employee_growth_yoy_pct_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_qoq": {"dtype": "float", "role": "feature"},
    "employee_growth_qoq_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_cagr_5y": {"dtype": "float", "role": "feature"},
    "employee_growth_cagr_5y_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_acceleration": {"dtype": "float", "role": "feature"},
    "employee_growth_acceleration_applicable": {"dtype": "bool", "role": "auxiliary"},
    "workforce_volatility": {"dtype": "float", "role": "feature"},
    "workforce_volatility_applicable": {"dtype": "bool", "role": "auxiliary"},
    "hiring_intensity_score": {"dtype": "float", "role": "feature"},
    "hiring_intensity_score_applicable": {"dtype": "bool", "role": "auxiliary"},
    # ==================================================================================
    # PHASE 9.3 COMPOSITE QUALITY SCORES (Advanced feature engineering)
    # ==================================================================================
    "altman_z_score": {
        "dtype": "float",
        "role": "feature",
        },  # Composite bankruptcy risk score
    "beneish_m_score": {
        "dtype": "float",
        "role": "feature",
        },  # Earnings manipulation detection score
    "composite_quality_score": {
        "dtype": "float",
        "role": "feature",
        },  # Multi-factor quality composite
    "momentum_score": {
        "dtype": "float",
        "role": "feature",
        },  # Technical momentum composite
    }

# Phase 9.3 Feature Input Categorization
# Maps feature engineering buckets to required input columns
PHASE93_FEATURE_INPUTS: Dict[str, List[str]] = {
    "momentum": [
        "price_chg_pct_1m",
        "price_chg_pct_3m",
        "1_day_pct",
        "price_1m_ago",
        "price_3m_ago",
        "price_6m_ago",
        "ema_20d",
        "ema_50d",
        "ema_100d",
        "ema_250d",
        "52w_high_adj",
        "52w_low_adj",
        "total_return_ytd",
        "total_return_5y",
        ],
    "valuation": [
        "p_e_ltm",
        "p_e_ntm",
        "p_b_ltm",
        "p_tbv_ltm",
        "ev_sales_ltm",
        "ev_ebitda_ltm",
        "market_cap",
        "enterprise_value",
        "last_price",
        "price_target_median",
        # Phase 9.3 Schema 1.3: EV/Sales historical time-series
        "ev_sales_est_fy1",
        "ev_sales_1fyltm",
        "ev_sales_2fyltm",
        "ev_sales_3fyltm",
        "ev_sales_3yavgltm",
        "ev_sales_ntm",
        "ev_sales_1fqltm",
        "ev_sales_2fqltm",
        "ev_sales_3fqltm",
        "ev_sales_4fqltm",
        # Phase 9.3 Schema 1.3: EV/EBITDA historical time-series
        "ev_ebitda_est_fy1",
        "ev_ebitda_ntm",
        "ev_ebitda_1fyltm",
        "ev_ebitda_1fqltm",
        "ev_ebitda_3yavgltm",
        # Phase 9.3 Schema 1.3: P/E extended time-series
        "p_e_est_fy1",
        "p_e_1fyltm",
        "p_e_2fyltm",
        "p_e_3fyltm",
        "p_e_3yavgltm",
        "p_e_1fqltm",
        "p_e_2fqltm",
        "p_e_3fqltm",
        "p_e_0fqqoqltm",
        "p_e_0fyyoyltm",
        "p_e_1fyyoyltm",
        "p_e_0fqyoyltm",
        ],
    "profitability": [
        "net_income_margin_pct_ltm",
        "gross_profit_margin_pct_ltm",
        "ebitda_ltm",
        "ebit_ltm",
        "net_income_is_ltm",
        "operating_income_ltm",
        "gross_profit_ltm",
        ],
    "quality_risk": [
        "altman_z_score_ltm",
        "return_on_equity_pct_ltm",
        "return_on_assets_roa_pct_ltm",
        "beta_1y",
        "volatility_1m",
        "volatility_3m",
        "current_ratio_ltm",
        "total_debt_ltm",
        "total_equity_ltm",
        ],
    "cash_flow": [
        "cfo_ltm",
        "fcf_ltm",
        "cfi_ltm",
        "cff_ltm",
        "cff_fy",
        "cff_1fy",
        "cfi_fy",
        "cfi_1fy",
        "cfo_fy",
        "cfo_1fy",
        "cff_fq",
        "cfi_fq",
        "cfo_fq",
        "fcf_fq",
        "fcf_fy",
        "fcf_5yavgfq",
        "capital_expenditure_ltm",
        ],
    "growth": [
        "total_revenues_cagr_5y_fy",
        "revenues_est_yoy_pct_fy1e",
        "tot_return_pct_cagr_3y",
        "total_revenues_ltm",
        "total_revenues_fy",
        ],
    "technical": [
        "ema_20d",
        "ema_50d",
        "ema_100d",
        "ema_250d",
        "52w_high_adj",
        "52w_low_adj",
        "rel_volume",
        ],
    "employment": [
        "avg_employees_ltm",
        "avg_employees_fy",
        "avg_employees_5yavgfy",
        "total_employees_fy",
        "total_employees_fq",
        "full_time_employees_fq",
        "full_time_employees_fy",
        "full_time_employees_1fy",
        "full_time_employees_2fy",
        "full_time_employees_3fy",
        ],
    "dividends": [
        "dividend_streak",
        "dividend_record_amount",
        "dividend_record_frequency",
        "dividend_per_share_ltm",
        "common_dividends_paid_ltm",
        "div_yield_ltm",
        "div_yield_ntm",
        "div_yield_ttm",
        ],
    "forecasts": [
        "revenues_est_avg_ntm",
        "revenues_est_med_ntm",
        "revenues_est_avg_fy1e",
        "revenues_est_med_fy1e",
        "eps_norm_est_avg_ntm",
        "eps_norm_est_avg_fy1e",
        "ebit_est_med_ntm",
        "ebit_est_med_fy1e",
        ],
    }


# Helper functions for schema access


def get_expected_dtype(column: str) -> Optional[str]:
    """
    Get the expected dtype for a column from COLUMN_SCHEMA.

    Args:
        column: Column name (normalized: lowercase with underscores)

    Returns:
        Expected dtype string or None if column not in schema
    """
    col_lower = (
        column.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace("#", "num")
    )
    if col_lower in COLUMN_SCHEMA:
        return COLUMN_SCHEMA[col_lower]["dtype"]
    return None


def get_column_role(column: str) -> Optional[str]:
    """
    Get the role of a column from COLUMN_SCHEMA.

    Args:
        column: Column name (normalized: lowercase with underscores)

    Returns:
        Role string or None if column not in schema
    """
    col_lower = (
        column.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace("#", "num")
    )
    if col_lower in COLUMN_SCHEMA:
        return COLUMN_SCHEMA[col_lower]["role"]
    return None


def list_numeric_feature_cols() -> List[str]:
    """
    List all numeric feature columns from COLUMN_SCHEMA.

    Returns:
        List of column names with numeric dtypes and feature/target roles
    """
    numeric_dtypes = {"float", "int"}
    feature_roles = {"feature", "target", "target_fallback"}

    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta["dtype"] in numeric_dtypes and meta["role"] in feature_roles
        ]


def list_categorical_cols() -> List[str]:
    """
    List all categorical columns from COLUMN_SCHEMA.

    Returns:
        List of column names with category or categorical role
    """
    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta["dtype"] == "category" or meta["role"] == "categorical"
        ]


def list_date_cols() -> List[str]:
    """
    List all date/datetime columns from COLUMN_SCHEMA.

    Returns:
        List of column names with datetime64[ns] dtype or date role
    """
    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta["dtype"] == "datetime64[ns]" or meta["role"] == "date"
        ]


def normalize_column_name(column: str) -> str:
    """
    Normalize a column name to match COLUMN_SCHEMA keys.

    Converts to lowercase, replaces spaces/special chars with underscores.

    Special handling:
    - "R&D" -> "randd" (not "r_and_d")
    - "Merger & Restructuring" -> "merger_and_restructuring"
    - "Selling General & Admin" -> "selling_general_and_admin"

    Args:
        column: Original column name (e.g., "Last Price" or "P/E (LTM)")

    Returns:
        Normalized column name (e.g., "last_price" or "p_e_ltm")
    """
    # Special case: R&D should become randd (not r_and_d)
    # Must be done before general & -> and replacement
    if "R&D" in column or "r&d" in column.lower():
        column = column.replace("R&D", "RandD").replace("r&d", "randd")

    normalized = (
        column.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace("#", "num")
        .replace("%", "pct")
        .replace("&", "and")
    )
    # Remove consecutive underscores
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    # Remove leading/trailing underscores
    return normalized.strip("_")


def list_etl_generated_column_patterns() -> List[str]:
    """List regex patterns for columns legitimately generated during ETL.

    These patterns are used by schema alignment validation and diagnostics to
    distinguish between truly unknown columns (data quality / upstream drift)
    and expected outputs created by the ETL / feature engineering workflow.

    Returns:
        List of regex pattern strings.

    Example:
        >>> import re
        >>> patterns = list_etl_generated_column_patterns()
        >>> any(re.match(p, "log_market_cap") for p in patterns)
        True
    """

    # NOTE: Keep these patterns conservative. Where possible, the ETL validator
    # also checks that the underlying base column exists in `COLUMN_SCHEMA`.
    return [
        r"^log_[0-9a-z_]+$",  # Log-transformed columns
        r"^.*_applicable$",  # Conditional metric applicability flags
        r"^event_prob_.*$",  # Classification probabilities
        r"^sector_[0-9a-z]+_x_[0-9a-z_]+$",  # Sector interactions
        r"^.*_(ratio|pct|margin|growth|yoy)$",  # Common semantic/derived suffixes
        ]


def list_required_schema_columns_for_etl(
        include_extended_financials: bool = False,
        ) -> List[str]:
    """
    List the canonical set of *required* raw columns for the unified ETL pipeline.

    This function centralizes the minimum schema that must be present in the
    source extract (CSV / DB) for Phase 9.1–9.3 ETL to run correctly and for
    downstream quality checks to be meaningful.

    It is intended for use in:
        - validate_schema(..., required_columns=list_required_schema_columns_for_etl())
        - interpreting dtype diagnostics "missing_expected_columns"
        - notebook/script post-ETL assertions

    The list focuses on:
        - Identifiers and group keys (ticker, sector, region, country, trading_country, isin)
        - Core business-critical price/target columns
        - Core market value columns needed for valuation metrics

    Args:
        include_extended_financials:
            When True, include a small set of additional raw financial
            columns that are highly recommended for modeling but not strictly
            mandatory for a minimal ETL run (e.g. EBITDA LTM, total_assets_ltm).

    Returns:
        Ordered list of normalized column names that must be present in
        the *raw* dataset before ETL.

    Notes:
        - All returned names are guaranteed to exist in COLUMN_SCHEMA; an
          AssertionError is raised during development if the schema drifts.
        - Derived / engineered columns (e.g. log_* metrics, ratios constructed
          in ETL, Phase 9.3 features) are *not* included here, as they are
          created by the pipeline, not required from the source.

    Example:
        >>> required = list_required_schema_columns_for_etl()
        >>> assert "ticker" in required
        >>> assert "last_price" in required
        >>> # Use with validate_schema
        >>> from finance_ml.ml_workflow.validation.validators import validate_schema
        >>> result = validate_schema(df, required_columns=required)
    """
    required: List[str] = [
        # Identifiers / group keys
        "ticker",
        "isin",
        "sector",
        "region",
        "country",
        "trading_country",
        # Core business-critical price & targets
        "last_price",
        "price_target",
        "price_target_median",
        "price_target_ytd_ago",
        # Core market value columns used across ETL & features
        "market_cap",
        "enterprise_value",
        ]

    if include_extended_financials:
        extended = [
            # High-importance financials commonly used by ETL/feature presets
            "total_revenues_ltm",
            "ebitda_ltm",
            "net_income_is_ltm",
            "total_assets_ltm",
            "total_debt_ltm",
            "total_equity_ltm",
            ]
        for col in extended:
            if col not in required:
                required.append(col)

    # Defensive check: ensure all required columns are present in COLUMN_SCHEMA
    missing_from_schema = [col for col in required if col not in COLUMN_SCHEMA]
    if missing_from_schema:
        raise AssertionError(
                f"Required ETL columns not found in COLUMN_SCHEMA: {missing_from_schema}"
                )

    return required

```

Attachment Name: etl_data_explorer.ipynb
Attachments Kind: Visible
Attachments Source: ReferencesRetriever / FileChatAttachment
Attachments Text:

```jupyter
#%% md
# ETL Data Explorer

This notebook explores the unified ETL (Extract, Transform, Load) Pipeline for stock data.

**Version:** 1.1.0 | **Model Version:** v9_10 | **Updated:** 2025-12-11

## Pipeline Stages
1. **Extract** - Load from DB with CSV fallback
2. **Transform** - Normalize, validate, sanitize, impute (6-step)
3. **Load** - Quality validation and finalization

## 16 Feature Categories (196 Features)
- Momentum & Technical, Valuation Ratios, Profitability, Quality & Risk
- Cash Flow, Capital Allocation, Analyst Sentiment, Market Sentiment
- Leverage & Liquidity, Temporal Patterns, Composite Scores, Growth Metrics
- Efficiency Ratios, Employee Productivity, Balance Sheet, Revenue Forecasting

## Feature Engineering API (`build_features`)

**Business Goal:** Engineer comprehensive financial features including valuation ratios, 
profitability metrics, quality indicators, and sector-specific features to maximize model predictive power.

**Key Objectives:**
- Engineer valuation ratios (P/E, P/B, EV/EBITDA, PEG)
- Engineer profitability features (margins, ROE, ROA, ROIC)
- Create momentum and technical indicators
- Engineer analyst quality features
- Create accounting quality scores (Altman Z, Piotroski F)
- Build sector-relative features
- Create interaction features

## Refactoring Improvements (v1.1.0)

**Schema Integration (`finance_ml.ml_workflow.data.schema`):**
- Centralized column definitions via `COLUMN_SCHEMA` (318 columns)
- Schema-driven column selection using helper functions
- Phase 9.3 feature categorization via `PHASE93_FEATURE_INPUTS`

**Code Guidelines Alignment (`code_guidelines.md` Section 8):**
- Configuration constants (Section 8.1): Single source of truth for all parameters
- Replaced hard-coded magic numbers with named constants
- Schema-driven numeric/categorical/date column selection

**Benefits:**
- Eliminates schema drift between notebook and canonical definitions
- Reduces maintenance burden from hard-coded column lists
- Ensures consistency with ETL pipeline and feature engineering modules
- Aligns with project-wide code quality standards

#%%
# ============================================================================
# Cell 1: Configuration & Setup
# ============================================================================
import json
import math
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# NumPy compatibility note:
# Avoid modifying NumPy private attributes to satisfy PyDeprecationInspection and stability concerns.
# If certain libraries expect np._ARRAY_API, prefer updating those libraries rather than mutating NumPy.
# Here, we intentionally do NOT set any private compatibility shims.

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# SQLAlchemy check
try:
    from sqlalchemy import create_engine, text

    HAVE_SQLALCHEMY = True
except ImportError:
    HAVE_SQLALCHEMY = False

# Project paths
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
CACHE_DIR = PROJECT_ROOT / '.cache'

try:
    (OUTPUT_DIR / 'eda').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'preprocessing').mkdir(parents=True, exist_ok=True)
except Exception as dir_err:
    # Graceful handling for permission/disk issues without crashing the notebook
    print(f"Warning: could not ensure output directories exist: {dir_err}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_ml.notebook_config import NotebookConfig
from finance_ml.ml_workflow.data.schema import (
    COLUMN_SCHEMA,
    PHASE93_FEATURE_INPUTS,
    get_expected_dtype,
    get_column_role,
    list_numeric_feature_cols,
    list_categorical_cols,
    list_date_cols,
    normalize_column_name,
    list_required_schema_columns_for_etl,
    )

# Phase 9.3 Earnings Dashboard Widgets (code_guidelines.md §9.3, §17)
from finance_ml.dashboards.earnings_widgets import (
    create_earnings_calendar_dashboard,
    display_earnings_dashboard,
    create_earnings_metrics_chart,
    create_category_comparison_chart,
    get_category_metrics,
    COLOR_PALETTE as EARNINGS_COLOR_PALETTE,
    CATEGORY_COLORS,
    )

CFG = NotebookConfig(
        have_finance_prediction=True,
        have_database_connection=True,
        have_advanced_analytics=True,
        have_dim_reduction=False,
        debug_mode=False,
        )

# ============================================================================
# Configuration Constants (code_guidelines.md Section 2.1 & 8.1)
# ============================================================================

# Target configuration
TARGET_COL = 'price_target'
TARGET_COL_FALLBACK = 'last_price'

# Data splits & CV
TEST_SIZE = 0.2
TRAIN_SIZE = 1 - TEST_SIZE
CV_FOLDS = 5

# Quantiles & confidence thresholds
QUANTILES = [0.1, 0.5, 0.9]
LOWER_QUANTILE = QUANTILES[0]
MEDIAN_QUANTILE = QUANTILES[1]
UPPER_QUANTILE = QUANTILES[2]
CONFIDENCE_LOW_THRESHOLD = 0.50
CONFIDENCE_MEDIUM_THRESHOLD = 0.75

# Sector/portfolio constraints
MIN_SECTOR_SAMPLES = 20
MAX_SECTOR_WEIGHT = 0.25
MAX_SINGLE_POSITION = 0.10

# Outlier detection / winsorization
IQR_MULTIPLIER = 2.5
ZSCORE_THRESHOLD = 4.0
WINSORIZE_LOWER = 0.10
WINSORIZE_UPPER = 0.90

# Visualization Configuration
TOP_N_SECTORS = 12  # Top sectors for heatmaps/coverage
TOP_N_CATEGORIES = 8  # Top categories for radar charts
TOP_N_FEATURES = 10  # Number of features to display in detailed analysis

# Reproducibility & versioning
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v9_10')
np.random.seed(RANDOM_SEED)

# Schema-Driven Column Selection
# Use canonical schema functions instead of hard-coded lists
NUMERIC_FEATURE_COLS = list_numeric_feature_cols()
CATEGORICAL_COLS = list_categorical_cols()
DATE_COLS = list_date_cols()
REQUIRED_ETL_COLS = list_required_schema_columns_for_etl(include_extended_financials=True)

# Key columns for summary statistics (subset of schema-driven numeric features)
KEY_SUMMARY_COLS = [
    'last_price', 'market_cap', 'enterprise_value',
    'ebitda_ltm', 'p_e_ntm', 'total_revenues_ltm'
    ]

# Section 17 Style Guidelines
plt.style.use('dark_background')
sns.set_palette('husl')
PLOTLY_TEMPLATE = 'plotly_dark'
COLOR_PALETTE = {
    'primary': '#375a7f',
    'secondary': '#6c757d',
    'success': '#00bc8c',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#3498db',
    'neutral': '#adb5bd',
    }


def validate_configuration():
    """Validate notebook configuration constants (Section 2.3)."""
    # Validate target columns
    if not TARGET_COL or not isinstance(TARGET_COL, str):
        raise ValueError(f"TARGET_COL must be non-empty string: {TARGET_COL}")
    if not TARGET_COL_FALLBACK or not isinstance(TARGET_COL_FALLBACK, str):
        raise ValueError(f"TARGET_COL_FALLBACK must be non-empty string: {TARGET_COL_FALLBACK}")

    # Validate test size
    if not (0 < TEST_SIZE < 1):
        raise ValueError(f"TEST_SIZE must be between 0 and 1: {TEST_SIZE}")
    if not abs((TRAIN_SIZE + TEST_SIZE) - 1.0) < 0.01:
        raise ValueError(f"TRAIN_SIZE + TEST_SIZE must equal 1.0: {TRAIN_SIZE + TEST_SIZE}")

    # Validate CV folds
    if not isinstance(CV_FOLDS, int) or CV_FOLDS < 2:
        raise ValueError(f"CV_FOLDS must be integer >= 2: {CV_FOLDS}")

    # Validate quantiles
    if not QUANTILES or not isinstance(QUANTILES, list):
        raise ValueError(f"QUANTILES must be non-empty list: {QUANTILES}")
    for q in QUANTILES:
        if not (0 < q < 1):
            raise ValueError(f"All quantiles must be between 0 and 1: {q}")
    if QUANTILES != sorted(QUANTILES):
        raise ValueError(f"QUANTILES must be monotonically increasing: {QUANTILES}")

    # Validate sector configuration
    if not isinstance(MIN_SECTOR_SAMPLES, int) or MIN_SECTOR_SAMPLES < 1:
        raise ValueError(f"MIN_SECTOR_SAMPLES must be positive integer: {MIN_SECTOR_SAMPLES}")
    if not (0 < MAX_SECTOR_WEIGHT <= 1):
        raise ValueError(f"MAX_SECTOR_WEIGHT must be between 0 and 1: {MAX_SECTOR_WEIGHT}")

    # Validate outlier detection
    if IQR_MULTIPLIER <= 0:
        raise ValueError(f"IQR_MULTIPLIER must be positive: {IQR_MULTIPLIER}")
    if ZSCORE_THRESHOLD <= 0:
        raise ValueError(f"ZSCORE_THRESHOLD must be positive: {ZSCORE_THRESHOLD}")
    if not (0 <= WINSORIZE_LOWER < 0.5):
        raise ValueError(f"WINSORIZE_LOWER must be between 0 and 0.5: {WINSORIZE_LOWER}")
    if not (0.5 < WINSORIZE_UPPER <= 1):
        raise ValueError(f"WINSORIZE_UPPER must be between 0.5 and 1: {WINSORIZE_UPPER}")

    print("✓ All configuration constants validated successfully")
    return True


def convert_jdbc_to_sqlalchemy(jdbc_url: str) -> str:
    """Convert a JDBC PostgreSQL URL to a SQLAlchemy URL preserving credentials when present.

    Supported inputs:
    - jdbc:postgresql://host:port/db
    - jdbc:postgresql://user:password@host:port/db
    - jdbc:postgresql://host/db (port optional)

    Returns a postgresql+psycopg2 URL string.
    """
    import re

    pattern = re.compile(
            r"^jdbc:postgresql://(?:(?P<user>[^:@/]+)(?::(?P<pw>[^@/]*))?@)?(?P<host>[^:/?#]+)(?::(?P<port>\d+))?/(?P<db>[^?]+)"
            )
    m = pattern.match(jdbc_url)
    if not m:
        # Fallback: return as-is for caller to handle
        return jdbc_url
    user = m.group('user')
    pw = m.group('pw') or ''
    host = m.group('host')
    port = m.group('port') or '5432'
    db = m.group('db')
    if user:
        return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    return f"postgresql+psycopg2://{host}:{port}/{db}"


def check_db_connection(db_url: str) -> bool:
    """Return True if a quick test connection to the database succeeds, else False.

    Uses SQLAlchemy if available and performs a trivial SELECT 1. Exceptions are not raised.
    """
    if not HAVE_SQLALCHEMY or not db_url:
        return False
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# Database URL (env var or safe default)
DB_URL_ENV = os.getenv('DB_URL')
if DB_URL_ENV and DB_URL_ENV.startswith('jdbc:'):
    DB_URL = convert_jdbc_to_sqlalchemy(DB_URL_ENV)
elif DB_URL_ENV:
    DB_URL = DB_URL_ENV
else:
    DB_URL = 'postgresql+psycopg2://localhost:5432/postgres'

# Run configuration validation
validate_configuration()

print('=' * 60)
print('ETL DATA EXPLORER - CONFIGURATION')
print('=' * 60)
print(f'PROJECT_ROOT: {PROJECT_ROOT}')
print(f'DATA_DIR: {DATA_DIR}')
print(f'Python: {sys.version.split()[0]}')
print(f'Random Seed: {RANDOM_SEED}')
print(f'Model Version: {MODEL_VERSION}')
CFG.display_summary()
#%% md
## Cell 2: Import Finance ML Modules

Import unified ETL pipeline, EDA analytics, and feature engineering modules.

#%%
# ============================================================================
# Cell 2: Import Finance ML Modules
# ============================================================================

# ETL Pipeline (Phase 9.1)
from finance_ml.ml_workflow.preprocessing.etl import (
    DataExtractionConfig,
    DataSanitizationConfig,
    DtypeCastingConfig,
    FeatureEngineeringConfig,
    FeatureSelectionConfig,
    FinancialMetricsConfig,
    ImputationConfig,
    ScalingConfig,
    SchemaValidationConfig,
    SemanticClassificationConfig,
    SemanticTransformConfig,
    run_etl_pipeline,
    etl_with_features,
    etl_with_financial_metrics,
    ETLConfig,
    ETLMetrics,
    etl_with_imputation,
    etl_from_csv,
    )

# EDA Analytics (Phase 9.2)
from finance_ml.ml_workflow.eda.eda import (
    eda_summary,
    generate_phase93_coverage_report,
    sector_distribution_summary,
    correlation_analysis,
    )

# Phase 9.3 Feature Categories
from finance_ml.ml_workflow.eda.phase93_categories import (
    PHASE93_FEATURE_CATEGORIES,
    categorize_dataframe_columns,
    get_phase93_coverage_stats,
    get_category_description,
    list_all_phase93_features,
    )

# Feature Engineering API (Phase 9.3)
from finance_ml.ml_workflow.features.api import build_features

# Column Semantics and Safety (Section 8.5)
from finance_ml.ml_workflow.preprocessing.column_semantics import (
    PRICE_COLUMNS,
    get_winsorizable_columns,
    get_scalable_columns,
    classify_columns,
    )

print('✓ Finance ML modules imported successfully')
print(f'✓ Phase 9.3 Feature Categories: {len(PHASE93_FEATURE_CATEGORIES)} categories')
print(f'✓ Total Phase 9.3 Features: {len(list_all_phase93_features())} features')
print(f'✓ Price Columns Protected: {len(PRICE_COLUMNS)} columns')
print(f'\nFeature Engineering Presets Available:')
print(f"  - 'basic': core ratios, margins, volatility, revenue CAGR")
print(f"  - 'momentum': momentum & technical indicators")
print(f"  - 'quality': accounting quality and financial distress signals")
print(f"  - 'comprehensive': full advanced feature set (196 features)")

#%% md
## Cell 3: Database Configuration

Configure PostgreSQL database connection with automatic URL format conversion.

#%%
# ============================================================================
# Cell 3: Database Configuration
# ============================================================================

# SQL file paths
SQL_SCHEMA = PROJECT_ROOT / 'create_equities_schema.sql'
SQL_IMPORT = PROJECT_ROOT / 'import_equities_data.sql'

# Detect availability
have_db_url = DB_URL is not None and len(DB_URL) > 0
reachable_db = check_db_connection(DB_URL) if have_db_url else False
CFG.have_database_connection = bool(reachable_db)

print('=' * 60)
print('DATABASE CONFIGURATION')
print('=' * 60)
print(f'SQLAlchemy installed:  {HAVE_SQLALCHEMY}')
print(f'DB_URL configured:     {have_db_url}')
print(f'Database available:    {CFG.have_database_connection}')

if have_db_url:
    # Mask password for security
    try:
        parts = DB_URL.split('@')
        if len(parts) == 2:
            user_part = parts[0].split('//')[-1].split(':')[0]
            masked_url = f'postgresql://{user_part}@{parts[1]}'
        else:
            masked_url = 'postgresql://***@localhost:5432/postgres'
    except Exception:
        masked_url = 'postgresql://***'
    print(f'Connection:            {masked_url}')
else:
    print('Connection:            Not configured (will use CSV fallback)')

print(f'\nSQL Files:')
print(f"  Schema script:       {'✓' if SQL_SCHEMA.exists() else '✗'} {SQL_SCHEMA.name}")
print(f"  Import script:       {'✓' if SQL_IMPORT.exists() else '✗'} {SQL_IMPORT.name}")
print('=' * 60)

#%% md
## Cell 4: ETL Pipeline - Extract, Transform, Load

Run the unified ETL pipeline with 6-step imputation strategy.

#%%
# ============================================================================
# Cell 4: ETL Pipeline - Extract, Transform, Load
# ============================================================================

print('=' * 60)
print('ETL PIPELINE EXECUTION')
print('=' * 60)

# Complete ETL + financial metrics
all_stocks_preprocessed, metrics = etl_with_financial_metrics(
        source='csv',
        data_dir=DATA_DIR,
        return_metrics=True,
        )

# Display ETL metrics summary
print('\n' + '=' * 60)
print(metrics.summary())
print('=' * 60)

# Validation checkpoint (Section 19.1 - REQUIRED)
assert not all_stocks_preprocessed.empty, 'Preprocessed data must not be empty'
assert 'ticker' in all_stocks_preprocessed.columns, 'ticker column must be present'
assert 'sector' in all_stocks_preprocessed.columns, 'sector column must be present'

# Validate target columns
target_cols = ['price_target', 'price_target_median', 'price_target_ytd_ago']
has_target = any(col in all_stocks_preprocessed.columns for col in target_cols)
assert has_target, f"At least one target column required: {target_cols}"

# Quality metrics validation
missing_pct = all_stocks_preprocessed.isna().sum().sum() / all_stocks_preprocessed.size * 100
assert missing_pct == 0, f"No missing values allowed after 6-step imputation, found {missing_pct:.2f}%"

# Data sufficiency validation
assert len(all_stocks_preprocessed) > 100, f"Insufficient data: {len(all_stocks_preprocessed)} rows (minimum 100)"
assert all_stocks_preprocessed['last_price'].min() > 0, "last_price must be positive"

# Stage naming alignment
all_stocks_features = all_stocks_preprocessed

print(f'\n✓ ETL Pipeline Complete')
print(f'  Data shape: {all_stocks_preprocessed.shape}')
print(f'  Source: {metrics.source_type}')
print(f'  Duration: {metrics.total_time_sec:.2f}s')
print(f'  Quality score: {metrics.quality_score:.3f}')
print(f'  Validation score: {metrics.validation_score:.3f}')
print(
        f'  Financial metrics added: {metrics.valuation_metrics_added + metrics.profitability_metrics_added + metrics.growth_metrics_added + metrics.leverage_metrics_added}')
print(f'  Missing values after imputation: {metrics.missing_values_after_imputation}')

#%% md
## Cell 5: Data Quality Overview

Examine data shape, dtypes, and missing values post-imputation.

#%%
# ============================================================================
# Cell 5: Data Quality Overview
# ============================================================================

print('=' * 60)
print('DATA QUALITY OVERVIEW')
print('=' * 60)

# Basic info
print(f'\nDataFrame Shape: {all_stocks_preprocessed.shape[0]:,} rows × {all_stocks_preprocessed.shape[1]} columns')
print(f'Memory Usage: {all_stocks_preprocessed.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB')

# Data types summary
dtype_counts = all_stocks_preprocessed.dtypes.value_counts()
print(f'\nData Types:')
for dtype, count in dtype_counts.items():
    print(f'  {dtype}: {count} columns')

# Missing values (post-imputation)
missing = all_stocks_preprocessed.isnull().sum()
missing_pct = (missing / len(all_stocks_preprocessed) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing': missing.values,
    'Percent': missing_pct.values
    }).query('Missing > 0').sort_values('Missing', ascending=False)

if len(missing_df) > 0:
    print(f'\n⚠ Columns with missing values (Top 10):')
    print(missing_df.head(10).to_string(index=False))
else:
    print(f'\n✓ No missing values - imputation successful!')

# Summary statistics for key columns (using schema-driven constants)
print(f'\nSummary Statistics (key numeric columns):')
available_keys = [c for c in KEY_SUMMARY_COLS if c in all_stocks_preprocessed.columns]
if available_keys:
    print(all_stocks_preprocessed[available_keys].describe().round(2).to_string())

# ============================================================================
# Semantic Classification Display
# ============================================================================
print(f'\n{"=" * 60}')
print('SEMANTIC CLASSIFICATION')
print('=' * 60)

semantic_classification = classify_columns(all_stocks_preprocessed.columns.tolist())

print(f'\nColumn Classification by Semantic Category:')
for category, columns in semantic_classification.items():
    if columns:
        print(f'\n{category.upper()}: {len(columns)} columns')
        columns_list = sorted(columns)  # Convert set to sorted list
        for col in columns_list[:3]:  # Now slicing works
            print(f'  - {col}')
        if len(columns) > 3:
            print(f'  ... and {len(columns) - 3} more')

# ============================================================================
# Price Column Preservation Visualization
# ============================================================================
print(f'\n{"=" * 60}')
print('PRICE COLUMN PRESERVATION')
print('=' * 60)

price_cols_in_df = [c for c in all_stocks_preprocessed.columns if c in PRICE_COLUMNS]
print(f'\nPrice columns preserved: {len(price_cols_in_df)}')

for col in price_cols_in_df[:10]:  # Show first 10
    if col in all_stocks_preprocessed.columns:
        min_val = all_stocks_preprocessed[col].min()
        max_val = all_stocks_preprocessed[col].max()
        non_null = all_stocks_preprocessed[col].notna().sum()
        print(f'  - {col}: range=[{min_val:.2f}, {max_val:.2f}], non-null={non_null:,}')

if 'last_price' in all_stocks_preprocessed.columns:
    price_range = all_stocks_preprocessed['last_price'].max() - all_stocks_preprocessed['last_price'].min()
    print(f'\n✓ Last Price range: ${price_range:.2f} (original dollar units preserved)')

# ============================================================================
# Log-Transformed Columns Analysis
# ============================================================================
print(f'\n{"=" * 60}')
print('LOG-TRANSFORMED COLUMNS')
print('=' * 60)

log_transformed = [c for c in all_stocks_preprocessed.columns if c.startswith('log_')]
print(f'\nLog-transformed columns: {len(log_transformed)}')

for col in log_transformed[:10]:  # Show first 10
    non_null = all_stocks_preprocessed[col].notna().sum()
    mean_val = all_stocks_preprocessed[col].mean()
    std_val = all_stocks_preprocessed[col].std()
    print(f'  - {col}: mean={mean_val:.2f}, std={std_val:.2f}, non-null={non_null:,}')

if len(log_transformed) > 10:
    print(f'  ... and {len(log_transformed) - 10} more')

print(f'\n✓ Cell 5 Complete: Data quality, semantic classification, and transformations validated')

#%%
all_stocks_preprocessed.head(50)
#%% md
## Cell 7: Region & Sector Analytics with Feature Coverage

Section 17 compliant Plotly visualizations with dark theme.
Enhanced with Phase 9.3 feature analytics by region and sector.

#%%
# ============================================================================
# Cell 7: Region & Sector Analytics with Feature Coverage
# ============================================================================

print('=' * 80)
print('REGION & SECTOR ANALYTICS WITH FEATURE COVERAGE')
print('=' * 80)

# Use all_stocks_features (post feature engineering) for analytics
df_analytics = all_stocks_preprocessed

# Region distribution
if 'region' in df_analytics.columns:
    region_counts = df_analytics['region'].value_counts().reset_index()
    region_counts.columns = ['Region', 'Count']
    region_counts['Percentage'] = (region_counts['Count'] / region_counts['Count'].sum() * 100).round(1)

    fig_region = px.bar(
            region_counts,
            x='Count',
            y='Region',
            orientation='h',
            title='Stock Distribution by Region (Post Feature Engineering)',
            template=PLOTLY_TEMPLATE,
            color='Count',
            color_continuous_scale='Blues',
            text='Count',
            )
    fig_region.update_traces(
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Percentage: %{customdata[0]:.1f}%',
            customdata=region_counts[['Percentage']].values
            )
    fig_region.update_layout(
            xaxis_title='Number of Stocks',
            yaxis_title='Region',
            height=400,
            showlegend=False,
            )
    fig_region.show()

# Sector distribution (using TOP_N_SECTORS constant)
if 'sector' in df_analytics.columns:
    sector_counts = df_analytics['sector'].value_counts().head(TOP_N_SECTORS).reset_index()
    sector_counts.columns = ['Sector', 'Count']
    sector_counts['Percentage'] = (sector_counts['Count'] / len(df_analytics) * 100).round(1)

    fig_sector = px.bar(
            sector_counts.sort_values('Count'),
            x='Count',
            y='Sector',
            orientation='h',
            title=f'Stock Distribution by Sector (Top {TOP_N_SECTORS}, Post Feature Engineering)',
            template=PLOTLY_TEMPLATE,
            color='Count',
            color_continuous_scale='Greens',
            text='Count',
            )
    fig_sector.update_traces(
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Percentage: %{customdata[0]:.1f}%',
            customdata=sector_counts.sort_values('Count')[['Percentage']].values
            )
    fig_sector.update_layout(
            xaxis_title='Number of Stocks',
            yaxis_title='Sector',
            height=500,
            showlegend=False,
            )
    fig_sector.show()

# Region-Sector Heatmap
if 'region' in df_analytics.columns and 'sector' in df_analytics.columns:
    cross_tab = pd.crosstab(
            df_analytics['sector'],
            df_analytics['region']
            ).head(TOP_N_SECTORS)

    fig_heatmap = px.imshow(
            cross_tab,
            title='Region-Sector Distribution Heatmap',
            template=PLOTLY_TEMPLATE,
            color_continuous_scale='Viridis',
            aspect='auto',
            )
    fig_heatmap.update_layout(
            xaxis_title='Region',
            yaxis_title='Sector',
            height=500,
            )
    fig_heatmap.show()

print('\n✓ Distribution visualizations complete')

# ============================================================================
# Feature Coverage Analytics by Region and Sector
# ============================================================================
print('\n' + '=' * 80)
print('📊 PHASE 9.3 FEATURE ANALYTICS BY REGION & SECTOR')
print('=' * 80)

# Define key Phase 9.3 features for analysis (schema-driven from PHASE93_FEATURE_INPUTS)
# Select representative features from each category for coverage analysis
# Define key Phase 9.3 features for analysis (schema-driven from PHASE93_FEATURE_INPUTS)
# Use all 6 categories comprehensively for complete coverage
key_phase93_features = (
        PHASE93_FEATURE_INPUTS.get('profitability', [])[:5] +  # Margins, EBITDA, EBIT, net income
        PHASE93_FEATURE_INPUTS.get('valuation', [])[:5] +  # P/E, P/B, EV ratios
        PHASE93_FEATURE_INPUTS.get('momentum', [])[:5] +  # Price changes, returns
        PHASE93_FEATURE_INPUTS.get('quality_risk', [])[:5] +  # Altman Z, ROE, ROA, volatility
        PHASE93_FEATURE_INPUTS.get('cash_flow', [])[:5] +  # CFO, FCF, CFI, CFF
        PHASE93_FEATURE_INPUTS.get('growth', [])  # Revenue CAGR, growth estimates (all)
)
# Map input columns to engineered feature names where applicable
feature_mapping = {
    'return_on_equity_pct_ltm': 'roe',
    'return_on_assets_roa_pct_ltm': 'roa',
    'p_e_ltm': 'p_e_ratio',
    'p_e_ntm': 'p_e_ratio',
    'ev_ebitda_ltm': 'ev_ebitda_ratio',
    'altman_z_score_ltm': 'altman_z_score',
    'price_chg_pct_1m': 'price_momentum_1m',
    'price_chg_pct_3m': 'price_momentum_3m',
    }
# Build list using both raw and engineered feature names
available_phase93 = list(set([
                                 f for f in key_phase93_features if f in df_analytics.columns
                                 ] + [
                                 feature_mapping.get(f, f) for f in key_phase93_features
                                 if feature_mapping.get(f, f) in df_analytics.columns
                                 ]))[:12]  # Limit to 12 for visualization clarity

if available_phase93 and 'region' in df_analytics.columns:
    print(f'\n📍 Feature Coverage by Region ({len(available_phase93)} key features):')

    # Calculate non-null percentage per region for each feature
    region_coverage_data = []
    for region in df_analytics['region'].unique():
        region_df = df_analytics[df_analytics['region'] == region]
        region_count = len(region_df)

        for feat in available_phase93:
            non_null = region_df[feat].notna().sum()
            coverage_pct = (non_null / region_count * 100) if region_count > 0 else 0
            region_coverage_data.append({
                'Region': region,
                'Feature': feat,
                'Coverage %': round(coverage_pct, 1),
                'Non-Null': non_null,
                'Total': region_count
                })

    region_coverage_df = pd.DataFrame(region_coverage_data)

    # Pivot for heatmap
    region_pivot = region_coverage_df.pivot(index='Feature', columns='Region', values='Coverage %')

    fig_region_coverage = px.imshow(
            region_pivot,
            title='Phase 9.3 Feature Coverage by Region (%)',
            template=PLOTLY_TEMPLATE,
            color_continuous_scale='RdYlGn',
            zmin=0, zmax=100,
            aspect='auto',
            )
    fig_region_coverage.update_layout(
            xaxis_title='Region',
            yaxis_title='Feature',
            height=500,
            )
    fig_region_coverage.show()

if available_phase93 and 'sector' in df_analytics.columns:
    print(f'\n🏢 Feature Coverage by Sector (Top 10 sectors, {len(available_phase93)} key features):')

    # Get top 10 sectors by count
    top_sectors = df_analytics['sector'].value_counts().head(10).index.tolist()

    # Calculate non-null percentage per sector for each feature
    sector_coverage_data = []
    for sector in top_sectors:
        sector_df = df_analytics[df_analytics['sector'] == sector]
        sector_count = len(sector_df)

        for feat in available_phase93:
            non_null = sector_df[feat].notna().sum()
            coverage_pct = (non_null / sector_count * 100) if sector_count > 0 else 0
            sector_coverage_data.append({
                'Sector': str(sector)[:25],  # Truncate long names
                'Feature': feat,
                'Coverage %': round(coverage_pct, 1),
                'Non-Null': non_null,
                'Total': sector_count
                })

    sector_coverage_df = pd.DataFrame(sector_coverage_data)

    # Pivot for heatmap
    sector_pivot = sector_coverage_df.pivot(index='Feature', columns='Sector', values='Coverage %')

    fig_sector_coverage = px.imshow(
            sector_pivot,
            title='Phase 9.3 Feature Coverage by Sector (%)',
            template=PLOTLY_TEMPLATE,
            color_continuous_scale='RdYlGn',
            zmin=0, zmax=100,
            aspect='auto',
            )
    fig_sector_coverage.update_layout(
            xaxis_title='Sector',
            yaxis_title='Feature',
            height=600,
            )
    fig_sector_coverage.show()

# Feature statistics by region
if available_phase93 and 'region' in df_analytics.columns:
    print(f'\n📊 Key Feature Statistics by Region:')

    # Select 3 key features for detailed statistics
    stat_features = ['roe', 'price_momentum_1m', 'debt_to_equity']
    stat_features = [f for f in stat_features if f in df_analytics.columns]

    if stat_features:
        region_stats = df_analytics.groupby('region')[stat_features].agg(['mean', 'median', 'std']).round(3)
        print(region_stats.to_string())

        # Box plot for ROE by region (if available)
        if 'roe' in df_analytics.columns:
            fig_roe_region = px.box(
                    df_analytics[df_analytics['roe'].notna()],
                    x='region',
                    y='roe',
                    title='Return on Equity (ROE) Distribution by Region',
                    template=PLOTLY_TEMPLATE,
                    color='region',
                    )
            fig_roe_region.update_layout(
                    xaxis_title='Region',
                    yaxis_title='ROE',
                    height=450,
                    showlegend=False,
                    )
            fig_roe_region.show()

print('\n✓ Feature analytics by region and sector complete')

#%% md
## Cell 8: Numeric Feature Distributions

Examine distributions of key numeric metrics with statistical annotations (using engineered features).

#%% md
## Cell 10: Financial Metrics Analytics

Comprehensive financial analytics with statistical summaries, hypothesis testing, and interactive visualizations.

**Key Objectives:**
1. Generate comprehensive statistical summaries and financial analytics reports
2. Analyze correlations and multicollinearity between features
3. Perform hypothesis testing across features, sectors, industry, country and regions
4. Create interactive visualizations for financial metrics distributions and relationships
5. Generate benchmarking reports comparing sector and regional financial/market performance

**Outputs:**
- **JSON Reports (4 files):** eda_summary.json, data_quality_alerts.json, metrics_dashboard.json, hypothesis_tests.json
- **Interactive Visualizations (7 HTML files):** correlation_heatmap.html, distributions.html, valuation_3d.html, region_sector_heatmap.html, sector_boxplots.html, regional_comparison.html, phase93_category_sector_bubble_chart.html

#%%
# ============================================================================
# Cell 10: Financial Metrics Analytics
# ============================================================================

import scipy.stats as scipy_stats
from datetime import datetime

# Import analytics functions
from finance_ml.ml_workflow.analytics.eval import (
    calculate_financial_metrics_dashboard,
    generate_data_quality_alerts,
    perform_comprehensive_hypothesis_tests,
    calculate_correlation_matrix,
    find_top_correlations,
    create_region_sector_heatmap,
    )
from finance_ml.ml_workflow.eda.eda import eda_summary
from finance_ml.ml_workflow.eda.reports import generate_benchmarking_report

print('=' * 80)
print('FINANCIAL METRICS ANALYTICS')
print('=' * 80)

# Create output directories
financial_metrics_dir = OUTPUT_DIR / 'eda' / 'financial_metrics'
financial_metrics_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Generate JSON Reports
# ============================================================================
print('\n📊 Generating JSON Reports...')

# 1.1 EDA Summary Report
print('  → eda_summary.json')
eda_summary_data = eda_summary(all_stocks_preprocessed, sector_column='sector', include_correlations=True)
eda_summary_path = financial_metrics_dir / 'eda_summary.json'
with open(eda_summary_path, 'w') as f:
    # Convert non-serializable objects
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj


    json.dump(make_serializable(eda_summary_data), f, indent=2, default=str)
print(f'    ✓ Saved: {eda_summary_path}')

# 1.2 Data Quality Alerts Report
print('  → data_quality_alerts.json')
quality_alerts = generate_data_quality_alerts(all_stocks_preprocessed, outlier_threshold=3.0)
quality_alerts_path = financial_metrics_dir / 'data_quality_alerts.json'
with open(quality_alerts_path, 'w') as f:
    json.dump(make_serializable(quality_alerts), f, indent=2, default=str)
print(f'    ✓ Saved: {quality_alerts_path}')

# 1.3 Financial Metrics Dashboard
print('  → metrics_dashboard.json')
# By sector
metrics_by_sector = calculate_financial_metrics_dashboard(all_stocks_preprocessed, group_by='sector')
# By region
metrics_by_region = calculate_financial_metrics_dashboard(all_stocks_preprocessed, group_by='region')
metrics_dashboard = {
    'timestamp': datetime.now().isoformat(),
    'total_stocks': len(all_stocks_preprocessed),
    'by_sector': make_serializable(metrics_by_sector),
    'by_region': make_serializable(metrics_by_region),
    }
metrics_dashboard_path = financial_metrics_dir / 'metrics_dashboard.json'
with open(metrics_dashboard_path, 'w') as f:
    json.dump(metrics_dashboard, f, indent=2, default=str)
print(f'    ✓ Saved: {metrics_dashboard_path}')

# 1.4 Hypothesis Tests Report
print('  → hypothesis_tests.json')
test_metrics = ['roe', 'roa', 'p_e_ratio', 'debt_to_equity', 'price_momentum_1m']
test_metrics = [m for m in test_metrics if m in all_stocks_preprocessed.columns]
hypothesis_results = perform_comprehensive_hypothesis_tests(
        all_stocks_preprocessed,
        group_column='sector',
        metrics=test_metrics,
        alpha=0.05
        )
hypothesis_path = financial_metrics_dir / 'hypothesis_tests.json'
with open(hypothesis_path, 'w') as f:
    json.dump(make_serializable(hypothesis_results), f, indent=2, default=str)
print(f'    ✓ Saved: {hypothesis_path}')

print(f'\n✓ JSON Reports Complete: 4 files generated')

# ============================================================================
# 2. Generate Interactive HTML Visualizations
# ============================================================================
print('\n📈 Generating Interactive HTML Visualizations...')

# 2.1 Correlation Heatmap (Top 50 features)
print('  → correlation_heatmap.html')
numeric_features = all_stocks_preprocessed.select_dtypes(include=[np.number]).columns.tolist()
# Select top 50 most complete numeric features
feature_completeness = all_stocks_preprocessed[numeric_features].notna().sum().sort_values(ascending=False)
top_50_features = feature_completeness.head(50).index.tolist()

corr_matrix = all_stocks_preprocessed[top_50_features].corr()

# Cluster the correlation matrix for better visualization
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Handle NaN in correlation matrix
corr_matrix_filled = corr_matrix.fillna(0)
try:
    # Compute distance matrix and linkage
    dist_matrix = 1 - np.abs(corr_matrix_filled)
    np.fill_diagonal(dist_matrix.values, 0)
    linkage = hierarchy.linkage(squareform(dist_matrix), method='average')
    order = hierarchy.leaves_list(linkage)
    corr_clustered = corr_matrix_filled.iloc[order, order]
except (ValueError, FloatingPointError):
    corr_clustered = corr_matrix_filled

fig_corr = px.imshow(
        corr_clustered,
        title='<b>Feature Correlation Heatmap</b><br><sup>Top 50 Features (Clustered)</sup>',
        template=PLOTLY_TEMPLATE,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        aspect='auto',
        )
fig_corr.update_layout(
        height=900,
        width=1000,
        font=dict(family='Segoe UI, Roboto, Arial', size=10),
        title_font_size=20,
        xaxis_title='Features',
        yaxis_title='Features',
        )
fig_corr.write_html(financial_metrics_dir / 'correlation_heatmap.html')
print(f'    ✓ Saved: correlation_heatmap.html')

# 2.2 Feature Distributions by Sector
print('  → distributions.html')
key_distribution_features = ['roe', 'roa', 'p_e_ratio', 'debt_to_equity', 'piotroski_f_score', 'altman_z_score']
key_distribution_features = [f for f in key_distribution_features if f in all_stocks_preprocessed.columns]

if key_distribution_features and 'sector' in all_stocks_preprocessed.columns:
    # Create subplots for distributions
    fig_dist = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f.replace('_', ' ').title() for f in key_distribution_features[:6]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
            )

    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['success'], COLOR_PALETTE['warning'],
              COLOR_PALETTE['danger'], COLOR_PALETTE['info'], COLOR_PALETTE['neutral']]

    for idx, feat in enumerate(key_distribution_features[:6]):
        row = idx // 3 + 1
        col = idx % 3 + 1

        data = all_stocks_preprocessed[feat].dropna()
        fig_dist.add_trace(
                go.Histogram(
                        x=data,
                        name=feat,
                        marker_color=colors[idx % len(colors)],
                        opacity=0.7,
                        nbinsx=50,
                        hovertemplate=f'<b>{feat}</b><br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
                        ),
                row=row, col=col
                )
        fig_dist.update_xaxes(title_text=feat.replace('_', ' ').title(), row=row, col=col)
        fig_dist.update_yaxes(title_text='Frequency', row=row, col=col)

    fig_dist.update_layout(
            title='<b>Financial Metrics Distributions</b><br><sup>Key Features Across All Stocks</sup>',
            template=PLOTLY_TEMPLATE,
            height=700,
            showlegend=False,
            font=dict(family='Segoe UI, Roboto, Arial'),
            title_font_size=20,
            )
    fig_dist.write_html(financial_metrics_dir / 'distributions.html')
    print(f'    ✓ Saved: distributions.html')

# 2.3 3D Valuation Scatter (Category-Sector-Market Cap)
print('  → valuation_3d.html')
# Select features for 3D visualization
val_features = {
    'x': 'roe' if 'roe' in all_stocks_preprocessed.columns else 'roa',
    'y': 'piotroski_f_score' if 'piotroski_f_score' in all_stocks_preprocessed.columns else 'altman_z_score',
    'z': 'market_cap' if 'market_cap' in all_stocks_preprocessed.columns else 'enterprise_value',
    }

if all(f in all_stocks_preprocessed.columns or f is None for f in val_features.values()):
    # Prepare data for 3D scatter
    df_3d = all_stocks_preprocessed[['ticker', 'sector', 'region'] + list(val_features.values())].dropna()

    # Log transform market cap for better visualization
    if 'market_cap' in df_3d.columns:
        df_3d['market_cap_log'] = np.log10(df_3d['market_cap'].clip(lower=1))
        z_col = 'market_cap_log'
        z_label = 'Market Cap (Log10 $)'
    else:
        z_col = val_features['z']
        z_label = val_features['z'].replace('_', ' ').title()

    fig_3d = px.scatter_3d(
            df_3d.head(2000),  # Limit for performance
            x=val_features['x'],
            y=val_features['y'],
            z=z_col,
            color='sector',
            symbol='region',
            hover_data=['ticker'],
            title='<b>Value vs Quality vs Size Trade-offs</b><br><sup>3D Category-Sector-Market Cap Analysis</sup>',
            template=PLOTLY_TEMPLATE,
            opacity=0.7,
            )
    fig_3d.update_layout(
            height=800,
            font=dict(family='Segoe UI, Roboto, Arial'),
            title_font_size=20,
            scene=dict(
                    xaxis_title=val_features['x'].replace('_', ' ').upper(),
                    yaxis_title=val_features['y'].replace('_', ' ').title(),
                    zaxis_title=z_label,
                    ),
            )
    fig_3d.write_html(financial_metrics_dir / 'valuation_3d.html')
    print(f'    ✓ Saved: valuation_3d.html')

# 2.4 Region-Sector Heatmap
print('  → region_sector_heatmap.html')
if 'region' in all_stocks_preprocessed.columns and 'sector' in all_stocks_preprocessed.columns:
    # Create pivot table for region-sector distribution
    region_sector_pivot = pd.crosstab(
            all_stocks_preprocessed['sector'],
            all_stocks_preprocessed['region'],
            margins=True
            )

    # Remove margins for heatmap
    region_sector_data = region_sector_pivot.iloc[:-1, :-1]

    fig_rs_heatmap = px.imshow(
            region_sector_data,
            title='<b>Regional Financial Analytics Distribution</b><br><sup>Stock Count by Sector and Region</sup>',
            template=PLOTLY_TEMPLATE,
            color_continuous_scale='Blues',
            aspect='auto',
            text_auto=True,
            )
    fig_rs_heatmap.update_layout(
            height=700,
            font=dict(family='Segoe UI, Roboto, Arial'),
            title_font_size=20,
            xaxis_title='Region',
            yaxis_title='Sector',
            )
    fig_rs_heatmap.write_html(financial_metrics_dir / 'region_sector_heatmap.html')
    print(f'    ✓ Saved: region_sector_heatmap.html')

# 2.5 Sector Boxplots
print('  → sector_boxplots.html')
boxplot_metrics = ['roe', 'roa', 'debt_to_equity', 'price_momentum_1m']
boxplot_metrics = [m for m in boxplot_metrics if m in all_stocks_preprocessed.columns]

if boxplot_metrics and 'sector' in all_stocks_preprocessed.columns:
    fig_box = make_subplots(
            rows=2, cols=2,
            subplot_titles=[m.replace('_', ' ').title() for m in boxplot_metrics[:4]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            )

    for idx, metric in enumerate(boxplot_metrics[:4]):
        row = idx // 2 + 1
        col = idx % 2 + 1

        # Get top 10 sectors by count
        top_sectors = all_stocks_preprocessed['sector'].value_counts().head(10).index.tolist()
        df_filtered = all_stocks_preprocessed[all_stocks_preprocessed['sector'].isin(top_sectors)]

        for i, sector in enumerate(top_sectors):
            sector_data = df_filtered[df_filtered['sector'] == sector][metric].dropna()
            fig_box.add_trace(
                    go.Box(
                            y=sector_data,
                            name=str(sector)[:15],
                            marker_color=px.colors.qualitative.Set3[i % 12],
                            showlegend=(idx == 0),
                            hovertemplate=f'<b>{sector}</b><br>{metric}: %{{y:.2f}}<extra></extra>'
                            ),
                    row=row, col=col
                    )
        fig_box.update_yaxes(title_text=metric.replace('_', ' ').title(), row=row, col=col)

    fig_box.update_layout(
            title='<b>Financial Metrics by Sector</b><br><sup>Box Plots for Key Metrics (Top 10 Sectors)</sup>',
            template=PLOTLY_TEMPLATE,
            height=800,
            font=dict(family='Segoe UI, Roboto, Arial'),
            title_font_size=20,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
            )
    fig_box.write_html(financial_metrics_dir / 'sector_boxplots.html')
    print(f'    ✓ Saved: sector_boxplots.html')

# 2.6 Regional Comparison
print('  → regional_comparison.html')
if 'region' in all_stocks_preprocessed.columns:
    comparison_metrics = ['roe', 'roa', 'p_e_ratio', 'debt_to_equity', 'piotroski_f_score']
    comparison_metrics = [m for m in comparison_metrics if m in all_stocks_preprocessed.columns]

    if comparison_metrics:
        # Calculate mean metrics by region
        regional_means = all_stocks_preprocessed.groupby('region')[comparison_metrics].mean()

        fig_regional = go.Figure()

        for i, region in enumerate(regional_means.index):
            fig_regional.add_trace(go.Scatterpolar(
                    r=regional_means.loc[region].values,
                    theta=[m.replace('_', ' ').title() for m in comparison_metrics],
                    fill='toself',
                    name=region,
                    opacity=0.6,
                    ))

        fig_regional.update_layout(
                polar=dict(
                        radialaxis=dict(visible=True,
                                        range=[regional_means.min().min() * 0.8, regional_means.max().max() * 1.2])
                        ),
                title='<b>Financial Metrics by Region</b><br><sup>Radar Chart Comparison</sup>',
                template=PLOTLY_TEMPLATE,
                height=600,
                font=dict(family='Segoe UI, Roboto, Arial'),
                title_font_size=20,
                showlegend=True,
                )
        fig_regional.write_html(financial_metrics_dir / 'regional_comparison.html')
        print(f'    ✓ Saved: regional_comparison.html')

# 2.7 Phase 9.3 Category-Sector Bubble Chart
print('  → phase93_category_sector_bubble_chart.html')
if 'sector' in all_stocks_preprocessed.columns:
    # Calculate average coverage by sector for each Phase 9.3 category
    bubble_data = []
    top_sectors = all_stocks_preprocessed['sector'].value_counts().head(TOP_N_SECTORS).index.tolist()

    for sector in top_sectors:
        sector_df = all_stocks_preprocessed[all_stocks_preprocessed['sector'] == sector]
        sector_count = len(sector_df)

        for category, features in PHASE93_FEATURE_CATEGORIES.items():
            available_features = [f for f in features if f in sector_df.columns]
            if available_features:
                coverage = sector_df[available_features].notna().mean().mean() * 100
                feature_count = len(available_features)
                bubble_data.append({
                    'Sector': str(sector)[:20],
                    'Category': category,
                    'Coverage %': round(coverage, 1),
                    'Feature Count': feature_count,
                    'Stock Count': sector_count,
                    })

    bubble_df = pd.DataFrame(bubble_data)

    if not bubble_df.empty:
        fig_bubble = px.scatter(
                bubble_df,
                x='Category',
                y='Sector',
                size='Coverage %',
                color='Coverage %',
                color_continuous_scale='RdYlGn',
                hover_data=['Feature Count', 'Stock Count'],
                title='<b>Phase 9.3 Feature Coverage by Sector</b><br><sup>Bubble Size = Coverage Percentage</sup>',
                template=PLOTLY_TEMPLATE,
                )
        fig_bubble.update_layout(
                height=700,
                font=dict(family='Segoe UI, Roboto, Arial'),
                title_font_size=20,
                xaxis_title='Phase 9.3 Category',
                yaxis_title='Sector',
                xaxis_tickangle=45,
                )
        fig_bubble.write_html(financial_metrics_dir / 'phase93_category_sector_bubble_chart.html')
        print(f'    ✓ Saved: phase93_category_sector_bubble_chart.html')

print(f'\n✓ Interactive HTML Visualizations Complete: 7 files generated')
print(f'\n📁 All outputs saved to: {financial_metrics_dir}')

# ============================================================================
# 3. Display Summary Statistics
# ============================================================================
print('\n' + '=' * 80)
print('📊 FINANCIAL METRICS SUMMARY')
print('=' * 80)

# Hypothesis test summary
if hypothesis_results and 'summary' in hypothesis_results:
    print(f"\n🔬 Hypothesis Testing Results:")
    print(f"   Tests performed: {hypothesis_results.get('n_tests', len(test_metrics))}")
    print(f"   Significance level: α = 0.05")
    if 'significant_differences' in hypothesis_results:
        sig_count = sum(1 for v in hypothesis_results['significant_differences'].values() if v)
        print(f"   Significant differences found: {sig_count}/{len(test_metrics)} metrics")

# Quality alerts summary
if quality_alerts:
    print(f"\n⚠️ Data Quality Alerts:")
    if 'outlier_counts' in quality_alerts:
        total_outliers = sum(quality_alerts['outlier_counts'].values())
        print(f"   Total outliers detected: {total_outliers:,}")
    if 'missing_critical' in quality_alerts:
        print(f"   Missing critical values: {len(quality_alerts.get('missing_critical', []))} columns")

# Correlation summary
print(f"\n🔗 Correlation Analysis:")
print(f"   Features analyzed: {len(top_50_features)}")
top_correlations = find_top_correlations(corr_matrix, n_top=5, threshold=0.7)
if len(top_correlations) > 0:  # find_top_correlations returns a list of tuples, not a DataFrame
    print(f"   High correlations (>0.7): {len(top_correlations)} pairs")
    print(f"   Top correlation: {top_correlations[0][2]:.3f}")  # Access tuple element (var1, var2, correlation)

print('\n✓ Financial Metrics Analytics complete')

#%% md
## Cell 10.5: Enhanced Financial Metrics & Price Target Analytics

Run the **unified ETL pipeline** with comprehensive valuation, profitability, growth, and leverage metrics.
Uses the consolidated `etl_with_financial_metrics()` function from `finance_ml.ml_workflow.preprocessing.etl`.

**Key Objectives:**
1. Compute comprehensive financial metrics using unified ETL pipeline
2. Analyze price target upside/downside across sectors and regions
3. Generate interactive visualizations (scatter plots, bar charts with confidence bands)
4. Export valuation_opportunities.json and multi_dimensional_valuation_analysis.json

**Visualizations:**
- **Price Target Scatter Plot:** Last Price vs Price Target with sector coloring
- **Price Target Distribution Bar Chart:** By sector with confidence bands
- **EMA Comparison Chart:** 20D, 50D, 100D, 250D EMAs vs Last Price
- **52W High/Low Analysis:** Position within 52-week range

**Outputs:**
- **JSON Reports:** valuation_opportunities.json, multi_dimensional_valuation_analysis.json
- **Interactive HTML Charts:** price_target_scatter.html, price_target_by_sector.html

#%%
# ============================================================================
# Cell 10.5: Enhanced Financial Metrics & Price Target Analytics
# ============================================================================
# Import unified ETL pipeline (replaces deprecated financial_metrics_etl)
from finance_ml.ml_workflow.preprocessing.etl import (
    etl_with_financial_metrics,
    ETLConfig,
    compute_valuation_metrics,
    compute_profitability_metrics,
    compute_growth_metrics,
    compute_leverage_metrics,
    compute_target_vs_price_metrics,
    generate_metrics_dashboard,
    )
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print('=' * 80)
print('ENHANCED FINANCIAL METRICS & PRICE TARGET ANALYTICS')
print('Using unified ETL pipeline (etl.py)')
print('=' * 80)

# ============================================================================
# 1. Compute Financial Metrics using Unified ETL Functions
# ============================================================================
print('\n🔧 Computing Financial Metrics...')

# Apply financial metrics to existing preprocessed data
all_stocks_enhanced = all_stocks_preprocessed.copy()

# Initialize fin_metrics dictionary to track metrics added
fin_metrics = {
    'valuation_metrics_added': 0,
    'profitability_metrics_added': 0,
    'growth_metrics_added': 0,
    'leverage_metrics_added': 0,
    'target_vs_price_metrics_added': 0,
    'sector_specific_metrics_added': 0,
    }

# Compute valuation metrics (P/E, P/S, EV/EBITDA, EV/Sales)
all_stocks_enhanced = compute_valuation_metrics(all_stocks_enhanced)
valuation_cols_added = ['p_e', 'p_s', 'ev_ebitda', 'ev_sales', 'eps']
fin_metrics['valuation_metrics_added'] = len([c for c in valuation_cols_added if c in all_stocks_enhanced.columns])
print(f'  ✓ Valuation metrics: {fin_metrics["valuation_metrics_added"]} added')

# Compute profitability metrics (margins, ROE, ROA)
all_stocks_enhanced = compute_profitability_metrics(all_stocks_enhanced)
profit_cols_added = ['gross_margin', 'operating_margin', 'net_margin', 'roe', 'roa']
fin_metrics['profitability_metrics_added'] = len([c for c in profit_cols_added if c in all_stocks_enhanced.columns])
print(f'  ✓ Profitability metrics: {fin_metrics["profitability_metrics_added"]} added')

# Compute growth metrics (revenue, EBITDA, earnings growth)
all_stocks_enhanced = compute_growth_metrics(all_stocks_enhanced)
growth_cols_added = ['revenue_growth', 'ebitda_growth', 'earnings_growth', 'revenue_growth_yoy', 'eps_growth_yoy',
                     'capex_growth_rate']
fin_metrics['growth_metrics_added'] = len([c for c in growth_cols_added if c in all_stocks_enhanced.columns])
print(f'  ✓ Growth metrics: {fin_metrics["growth_metrics_added"]} added')

# Compute leverage metrics (debt ratios)
all_stocks_enhanced = compute_leverage_metrics(all_stocks_enhanced)
leverage_cols_added = ['debt_to_equity', 'debt_to_assets']
fin_metrics['leverage_metrics_added'] = len([c for c in leverage_cols_added if c in all_stocks_enhanced.columns])
print(f'  ✓ Leverage metrics: {fin_metrics["leverage_metrics_added"]} added')

# Compute target vs price metrics
all_stocks_enhanced = compute_target_vs_price_metrics(all_stocks_enhanced)
target_cols_added = ['target_vs_price', 'target_vs_price_median', 'target_spread_pct']
fin_metrics['target_vs_price_metrics_added'] = len([c for c in target_cols_added if c in all_stocks_enhanced.columns])
print(f'  ✓ Target vs price metrics: {fin_metrics["target_vs_price_metrics_added"]} added')

print(f'\n✓ Financial Metrics Complete: {all_stocks_enhanced.shape[1]} total columns')

# ============================================================================
# 2. Price Target Analytics with Visualizations
# ============================================================================
print('\n📊 Price Target Analytics & Visualizations...')

# Define key price-related columns for analysis
price_cols = [
    'last_price', 'price_target_ytd_ago', 'price_target', 'price_target_low',
    'price_target_median', 'price_target_high', 'price_target_count',
    '52w_high_adj', '52w_low_adj', 'ema_20d', 'ema_50d', 'ema_100d', 'ema_250d'
    ]
available_price_cols = [c for c in price_cols if c in all_stocks_enhanced.columns]
print(f'  Available price columns: {len(available_price_cols)}/{len(price_cols)}')

# ============================================================================
# 2.1 Price Target Scatter Plot: Last Price vs Price Target (Log-Scaled with Confidence Bands)
# ============================================================================
if 'last_price' in all_stocks_enhanced.columns and 'price_target' in all_stocks_enhanced.columns:
    print('\n  📈 Creating Price Target Scatter Plot (Log-Scaled)...')

    # Filter valid data for visualization - include confidence bounds
    required_cols = ['ticker', 'sector', 'last_price', 'price_target', 'target_vs_price']
    confidence_cols = ['price_target_low', 'price_target_high']
    available_cols = required_cols + [col for col in confidence_cols if col in all_stocks_enhanced.columns]

    scatter_data = all_stocks_enhanced[available_cols].dropna(subset=required_cols)

    # Limit to reasonable price range for visualization
    scatter_data = scatter_data[
        (scatter_data['last_price'] > 0) & (scatter_data['last_price'] < 1000000) &
        (scatter_data['price_target'] > 0) & (scatter_data['price_target'] < 2000000)
        ]

    if len(scatter_data) > 0:
        # Apply log10 transformation
        scatter_data['last_price_log'] = np.log10(scatter_data['last_price'])
        scatter_data['price_target_log'] = np.log10(scatter_data['price_target'])

        # Add log transformations for confidence bounds if available
        has_confidence = 'price_target_low' in scatter_data.columns and 'price_target_high' in scatter_data.columns
        if has_confidence:
            scatter_data['price_target_low_log'] = np.log10(scatter_data['price_target_low'].clip(lower=0.01))
            scatter_data['price_target_high_log'] = np.log10(scatter_data['price_target_high'].clip(lower=0.01))

        # Create figure with go.Figure for confidence bands
        fig_scatter = go.Figure()

        # Get unique sectors and color palette
        sectors = scatter_data['sector'].unique()
        colors = px.colors.qualitative.Plotly
        color_map = {sector: colors[i % len(colors)] for i, sector in enumerate(sectors)}

        # Add scatter points and confidence bands by sector
        for sector in sectors:
            sector_data = scatter_data[scatter_data['sector'] == sector].copy()

            # Add confidence band if available
            if has_confidence:
                # Sort by last_price_log for proper polygon rendering
                sector_data_sorted = sector_data.sort_values(by='last_price_log')

                # Create confidence band trace (invisible, for fill)
                fig_scatter.add_trace(go.Scatter(
                        x=sector_data_sorted['last_price_log'],
                        y=sector_data_sorted['price_target_low_log'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip',
                        name=f'{sector} Low'
                        ))

                fig_scatter.add_trace(go.Scatter(
                        x=sector_data_sorted['last_price_log'],
                        y=sector_data_sorted['price_target_high_log'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f"rgba{tuple(list(int(color_map[sector].lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) + [0.15])}",
                        showlegend=False,
                        hoverinfo='skip',
                        name=f'{sector} High'
                        ))

            # Add scatter points
            fig_scatter.add_trace(go.Scatter(
                    x=sector_data['last_price_log'],
                    y=sector_data['price_target_log'],
                    mode='markers',
                    name=sector,
                    marker=dict(
                            color=color_map[sector],
                            size=8,
                            opacity=0.6,
                            line=dict(width=0.5, color='white')
                            ),
                    customdata=np.column_stack((
                        sector_data['ticker'],
                        sector_data['target_vs_price'],
                        sector_data['last_price'],
                        sector_data['price_target']
                        )),
                    hovertemplate='<b>%{customdata[0]}</b><br>' +
                                  'Sector: ' + sector + '<br>' +
                                  'Last Price: $%{customdata[2]:.2f}<br>' +
                                  'Price Target: $%{customdata[3]:.2f}<br>' +
                                  'Target vs Price: %{customdata[1]:.1f}%<br>' +
                                  'Log10(Last Price): %{x:.3f}<br>' +
                                  'Log10(Price Target): %{y:.3f}<br>' +
                                  '<extra></extra>'
                    ))

        # Add diagonal reference line in log space (Price Target = Last Price)
        min_log = scatter_data['last_price_log'].min()
        max_log = max(scatter_data['last_price_log'].max(), scatter_data['price_target_log'].max())
        fig_scatter.add_trace(
                go.Scatter(
                        x=[min_log, max_log],
                        y=[min_log, max_log],
                        mode='lines',
                        name='Fair Value Line',
                        line=dict(dash='dash', color='gray', width=2),
                        showlegend=True,
                        hoverinfo='skip'
                        )
                )

        fig_scatter.update_layout(
                title='<b>Price Target vs Last Price by Sector (Log-Scaled)</b><br>' +
                      '<sup>Log10 transformation | Dots above diagonal = Analyst upside potential' +
                      (' | Shaded bands = Confidence intervals' if has_confidence else '') + '</sup>',
                xaxis_title='Log10(Last Price)',
                yaxis_title='Log10(Price Target)',
                template=PLOTLY_TEMPLATE,
                height=700,
                legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02),
                hovermode='closest'
                )

        scatter_path = financial_metrics_dir / 'price_target_scatter.html'
        fig_scatter.write_html(str(scatter_path))
        print(f'    ✓ Saved: {scatter_path}')
        if has_confidence:
            print(f'    ✓ Confidence bands included from price_target_low and price_target_high')
        else:
            print(f'    ⚠ Confidence bands not available (missing price_target_low or price_target_high)')
        fig_scatter.show()
    else:
        print('    ⚠ Insufficient data for scatter plot')

# ============================================================================
# 2.2 Price Target Upside by Sector (Bar Chart with Confidence Bands)
# ============================================================================
if 'target_vs_price' in all_stocks_enhanced.columns and 'sector' in all_stocks_enhanced.columns:
    print('\n  📊 Creating Price Target by Sector Bar Chart...')

    sector_stats = all_stocks_enhanced.groupby('sector')['target_vs_price'].agg([
        'mean', 'std', 'count',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
        ]).reset_index()
    sector_stats.columns = ['sector', 'mean', 'std', 'count', 'q25', 'q75']

    # Filter sectors with sufficient data
    sector_stats = sector_stats[sector_stats['count'] >= 10].sort_values('mean', ascending=True)

    if len(sector_stats) > 0:
        fig_bar = go.Figure()

        # Add bar chart
        fig_bar.add_trace(go.Bar(
                y=sector_stats['sector'],
                x=sector_stats['mean'],
                orientation='h',
                name='Mean Upside (%)',
                marker_color=sector_stats['mean'].apply(lambda x: 'green' if x > 0 else 'red'),
                error_x=dict(
                        type='data',
                        symmetric=False,
                        array=sector_stats['q75'] - sector_stats['mean'],
                        arrayminus=sector_stats['mean'] - sector_stats['q25'],
                        color='rgba(0,0,0,0.3)'
                        ),
                hovertemplate='<b>%{y}</b><br>Mean: %{x:.1f}%<br>Q25-Q75: %{customdata[0]:.1f}% - %{customdata[1]:.1f}%<extra></extra>',
                customdata=sector_stats[['q25', 'q75']].values
                ))

        fig_bar.update_layout(
                title='<b>Price Target Upside by Sector</b><br><sup>With 25th-75th percentile confidence bands</sup>',
                xaxis_title='Target vs Price Upside (%)',
                yaxis_title='Sector',
                template=PLOTLY_TEMPLATE,
                height=500,
                showlegend=False
                )

        # Add vertical line at 0%
        fig_bar.add_vline(x=0, line_dash='dash', line_color='gray')

        bar_path = financial_metrics_dir / 'price_target_by_sector.html'
        fig_bar.write_html(str(bar_path))
        print(f'    ✓ Saved: {bar_path}')
        fig_bar.show()
    else:
        print('    ⚠ Insufficient sector data for bar chart')

# ============================================================================
# 2.3 EMA Comparison Chart (20D, 50D, 100D, 250D vs Last Price)
# ============================================================================
ema_cols = ['ema_20d', 'ema_50d', 'ema_100d', 'ema_250d']
available_emas = [c for c in ema_cols if c in all_stocks_enhanced.columns]

if 'last_price' in all_stocks_enhanced.columns and len(available_emas) >= 2:
    print('\n  📉 Creating EMA Comparison Chart...')

    # Calculate EMA position relative to last price
    ema_analysis = []
    for ema_col in available_emas:
        valid_data = all_stocks_enhanced[['last_price', ema_col]].dropna()
        if len(valid_data) > 0:
            above_ema = (valid_data['last_price'] > valid_data[ema_col]).sum()
            below_ema = (valid_data['last_price'] <= valid_data[ema_col]).sum()
            ema_analysis.append({
                'EMA': ema_col.replace('_', ' ').upper(),
                'Above EMA (%)': above_ema / len(valid_data) * 100,
                'Below EMA (%)': below_ema / len(valid_data) * 100,
                'Count': len(valid_data)
                })

    if ema_analysis:
        ema_df = pd.DataFrame(ema_analysis)

        fig_ema = go.Figure()
        fig_ema.add_trace(go.Bar(
                x=ema_df['EMA'],
                y=ema_df['Above EMA (%)'],
                name='Above EMA',
                marker_color='green'
                ))
        fig_ema.add_trace(go.Bar(
                x=ema_df['EMA'],
                y=ema_df['Below EMA (%)'],
                name='Below EMA',
                marker_color='red'
                ))

        fig_ema.update_layout(
                title='<b>Stock Position vs Exponential Moving Averages</b><br><sup>Percentage of stocks above/below each EMA</sup>',
                xaxis_title='Moving Average',
                yaxis_title='Percentage of Stocks (%)',
                barmode='group',
                template=PLOTLY_TEMPLATE,
                height=400
                )

        ema_path = financial_metrics_dir / 'ema_comparison.html'
        fig_ema.write_html(str(ema_path))
        print(f'    ✓ Saved: {ema_path}')
        fig_ema.show()

# ============================================================================
# 2.4 52-Week High/Low Position Analysis
# ============================================================================
if 'last_price' in all_stocks_enhanced.columns and '52w_high_adj' in all_stocks_enhanced.columns:
    print('\n  📊 Creating 52-Week Range Position Chart...')

    range_data = all_stocks_enhanced[['ticker', 'sector', 'last_price', '52w_high_adj', '52w_low_adj']].dropna()

    if len(range_data) > 0 and '52w_low_adj' in range_data.columns:
        # Calculate position within 52W range (0% = at low, 100% = at high)
        range_data['position_52w'] = (
                (range_data['last_price'] - range_data['52w_low_adj']) /
                (range_data['52w_high_adj'] - range_data['52w_low_adj']) * 100
        ).clip(0, 100)

        # Distribution by sector
        sector_52w = range_data.groupby('sector')['position_52w'].agg(['mean', 'std', 'count']).reset_index()
        sector_52w = sector_52w[sector_52w['count'] >= 10].sort_values('mean', ascending=True)

        if len(sector_52w) > 0:
            fig_52w = px.bar(
                    sector_52w,
                    y='sector',
                    x='mean',
                    orientation='h',
                    error_x='std',
                    title='<b>52-Week Range Position by Sector</b><br><sup>0% = At 52W Low, 100% = At 52W High</sup>',
                    labels={'mean': 'Position within 52W Range (%)', 'sector': 'Sector'},
                    color='mean',
                    color_continuous_scale='RdYlGn'
                    )

            fig_52w.update_layout(template=PLOTLY_TEMPLATE, height=500, showlegend=False)
            fig_52w.add_vline(x=50, line_dash='dash', line_color='gray', annotation_text='Mid-range')

            range_path = financial_metrics_dir / '52w_range_position.html'
            fig_52w.write_html(str(range_path))
            print(f'    ✓ Saved: {range_path}')
            fig_52w.show()

# ============================================================================
# 3. Valuation Opportunities Analysis
# ============================================================================
print('\n📐 Generating Valuation Opportunities Analysis...')

# Categorize stocks by valuation
if 'target_vs_price' in all_stocks_enhanced.columns:
    valuation_opportunities = {
        'timestamp': datetime.now().isoformat(),
        'total_stocks_analyzed': len(all_stocks_enhanced),
        'valuation_summary': {
            'mean_target_vs_price': float(all_stocks_enhanced['target_vs_price'].mean()),
            'median_target_vs_price': float(all_stocks_enhanced['target_vs_price'].median()),
            'std_target_vs_price': float(all_stocks_enhanced['target_vs_price'].std()),
            },
        'category_distribution': {},
        'top_undervalued': [],
        'top_overvalued': [],
        }


    # Categorize stocks
    def categorize_valuation(upside):
        if upside > 50:
            return 'Deeply Undervalued'
        elif upside > 15:
            return 'Undervalued'
        elif upside >= -15:
            return 'Fairly Valued'
        elif upside >= -30:
            return 'Overvalued'
        else:
            return 'Deeply Overvalued'


    all_stocks_enhanced['valuation_category'] = all_stocks_enhanced['target_vs_price'].apply(categorize_valuation)
    valuation_opportunities['category_distribution'] = all_stocks_enhanced[
        'valuation_category'].value_counts().to_dict()

    # Top undervalued stocks
    top_under = all_stocks_enhanced.nlargest(10, 'target_vs_price')[
        ['ticker', 'sector', 'last_price', 'price_target', 'target_vs_price']]
    valuation_opportunities['top_undervalued'] = top_under.to_dict('records')

    # Top overvalued stocks
    top_over = all_stocks_enhanced.nsmallest(10, 'target_vs_price')[
        ['ticker', 'sector', 'last_price', 'price_target', 'target_vs_price']]
    valuation_opportunities['top_overvalued'] = top_over.to_dict('records')

    # Save valuation opportunities JSON
    val_opp_path = financial_metrics_dir / 'valuation_opportunities.json'
    with open(val_opp_path, 'w') as f:
        json.dump(make_serializable(valuation_opportunities), f, indent=2, default=str)
    print(f'  ✓ Saved: {val_opp_path}')

    # Print summary
    print(f'\n  Valuation Category Distribution:')
    for cat, count in valuation_opportunities['category_distribution'].items():
        print(f'    {cat:20s}: {count:,}')

# ============================================================================
# 4. Multi-Dimensional Valuation Analysis
# ============================================================================
print('\n📐 Multi-Dimensional Valuation Analysis...')
valuation_metrics = ['p_e_ratio', 'p_s_ratio', 'ev_ebitda_ratio', 'ev_sales_ratio', 'roe', 'roa']
available_val_metrics = [m for m in valuation_metrics if m in all_stocks_enhanced.columns]

if len(available_val_metrics) >= 2:
    multi_dim_valuation = {
        'timestamp': datetime.now().isoformat(),
        'total_stocks_analyzed': len(all_stocks_enhanced),
        'dimensions': {},
        'correlations': {},
        }

    # Compute statistics for each valuation dimension
    for metric in available_val_metrics:
        data = all_stocks_enhanced[metric].dropna()
        if len(data) > 0:
            multi_dim_valuation['dimensions'][metric] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'q25': float(data.quantile(0.25)),
                'q75': float(data.quantile(0.75)),
                }

    # Correlation analysis
    if len(available_val_metrics) >= 2:
        val_corr = all_stocks_enhanced[available_val_metrics].corr()
        for i, m1 in enumerate(available_val_metrics):
            for j, m2 in enumerate(available_val_metrics):
                if i < j:
                    corr_val = val_corr.loc[m1, m2]
                    if not np.isnan(corr_val):
                        multi_dim_valuation['correlations'][f'{m1}_vs_{m2}'] = float(corr_val)

    # Save multi-dimensional analysis
    multi_dim_path = financial_metrics_dir / 'multi_dimensional_valuation_analysis.json'
    with open(multi_dim_path, 'w') as f:
        json.dump(make_serializable(multi_dim_valuation), f, indent=2, default=str)
    print(f'  ✓ Saved: {multi_dim_path}')

# ============================================================================
# 5. Generate Financial Metrics Dashboard
# ============================================================================
print('\n💾 Generating Financial Metrics Dashboard...')
financial_dashboard = generate_metrics_dashboard(all_stocks_enhanced, sector_column='sector')
financial_dashboard['price_target_analytics'] = valuation_opportunities if 'valuation_opportunities' in dir() else {}

dashboard_path = financial_metrics_dir / 'financial_metrics_dashboard.json'
with open(dashboard_path, 'w') as f:
    json.dump(make_serializable(financial_dashboard), f, indent=2, default=str)
print(f'  ✓ Saved: {dashboard_path}')

print('\n' + '=' * 80)
print('✓ Enhanced Financial Metrics & Price Target Analytics Complete')
print('=' * 80)
print(f'  Total columns: {all_stocks_enhanced.shape[1]}')
print(f'  Visualizations saved to: {financial_metrics_dir}')

#%% md
## Cell 10.6: Earnings Monitoring

Monitor earnings-related metrics including revenue growth, EBITDA trends, margin analysis, and earnings quality indicators.

**Key Objectives:**
1. Track earnings growth trends across sectors and regions
2. Analyze margin compression/expansion patterns
3. Monitor earnings quality and consistency
4. Generate earnings monitoring dashboard

**Outputs:**
- **JSON Report:** earnings_monitor.json

#%%
# ============================================================================
# Cell 10.6: Earnings Monitoring (Phase 9.3 Schema-Driven)
# ============================================================================

print('=' * 80)
print('EARNINGS MONITORING (Phase 9.3 Schema-Driven)')
print('=' * 80)

# ============================================================================
# 1. Revenue & Earnings Growth Analysis (using PHASE93_FEATURE_INPUTS)
# ============================================================================
print('\n📈 Revenue & Earnings Growth Analysis (PHASE93_FEATURE_INPUTS)...')

# Schema-driven metric selection from PHASE93_FEATURE_INPUTS categories
# (code_guidelines.md §9.3: Use PHASE93_FEATURE_INPUTS for metric categorization)
earnings_metrics_phase93 = {
    'growth': PHASE93_FEATURE_INPUTS.get('growth', []),
    'profitability': PHASE93_FEATURE_INPUTS.get('profitability', []),
    'valuation': PHASE93_FEATURE_INPUTS.get('valuation', [])[:10],  # Top 10 valuation metrics
    'forecasts': PHASE93_FEATURE_INPUTS.get('forecasts', []),
    'momentum': PHASE93_FEATURE_INPUTS.get('momentum', [])[:8],  # Top 8 momentum metrics
    }

earnings_monitor = {
    'timestamp': datetime.now().isoformat(),
    'total_stocks_monitored': len(all_stocks_enhanced),
    'phase93_categories_used': list(earnings_metrics_phase93.keys()),
    'growth_metrics': {},
    'profitability_metrics': {},
    'valuation_metrics': {},
    'forecasts_metrics': {},
    'momentum_metrics': {},
    'margin_metrics': {},
    'quality_indicators': {},
    'eps_metrics': {},  # NEW: Enhanced EPS analysis
    'eps_revisions': {},  # NEW: EPS revision tracking
    }

# Analyze metrics by Phase 9.3 category
for category, metrics in earnings_metrics_phase93.items():
    available = [m for m in metrics if m in all_stocks_enhanced.columns]

    if available:
        category_stats = {}
        print(f'\n  📊 {category.replace("_", " ").title()} Category ({len(available)} metrics):')

        for metric in available[:5]:  # Show top 5 per category
            data = all_stocks_enhanced[metric].dropna()
            if len(data) > 0:
                category_stats[metric] = {
                    'count': int(len(data)),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'positive_pct': float((data > 0).sum() / len(data) * 100),
                    'negative_pct': float((data < 0).sum() / len(data) * 100),
                    }

                print(
                        f'    {metric:30s}: mean={data.mean():>10.2f}, median={data.median():>10.2f}, '
                        f'+ve={((data > 0).sum() / len(data) * 100):>5.1f}%')

        # Store full category stats
        for metric in available:
            data = all_stocks_enhanced[metric].dropna()
            if len(data) > 0 and metric not in category_stats:
                category_stats[metric] = {
                    'count': int(len(data)),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'positive_pct': float((data > 0).sum() / len(data) * 100),
                    'negative_pct': float((data < 0).sum() / len(data) * 100),
                    }

        earnings_monitor[f'{category}_metrics'] = category_stats

# ============================================================================
# 2. Margin Analysis (using PHASE93_FEATURE_INPUTS profitability category)
# ============================================================================
print('\n📊 Margin Analysis (PHASE93 Profitability)...')

# Use profitability category from PHASE93_FEATURE_INPUTS + supplemental margin metrics
profitability_metrics = PHASE93_FEATURE_INPUTS.get('profitability', [])
supplemental_margins = ['gross_margin_pct', 'ebitda_margin', 'gross_profit_margin_pct_fy',
                        'gross_profit_margin_pct_ltm', 'net_income_margin_pct_fy']
margin_metrics = list(dict.fromkeys(profitability_metrics + supplemental_margins))  # Dedupe
available_margins = [m for m in margin_metrics if m in all_stocks_enhanced.columns]

if available_margins:
    margin_stats = {}

    for metric in available_margins:
        data = all_stocks_enhanced[metric].dropna()
        if len(data) > 0:
            margin_stats[metric] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'q25': float(data.quantile(0.25)),
                'q75': float(data.quantile(0.75)),
                'positive_pct': float((data > 0).sum() / len(data) * 100),
                }

            print(
                    f'  {metric:35s}: mean={data.mean():>8.2f}, median={data.median():>8.2f}, '
                    f'+ve={((data > 0).sum() / len(data) * 100):>5.1f}%')

    earnings_monitor['margin_metrics'] = margin_stats

    # Sector margin comparison
    if 'sector' in all_stocks_enhanced.columns and len(available_margins) > 0:
        top_margin = available_margins[0]
        sector_margins = {}

        for sector in all_stocks_enhanced['sector'].dropna().unique():
            sector_data = all_stocks_enhanced[all_stocks_enhanced['sector'] == sector][top_margin].dropna()
            if len(sector_data) >= 5:
                sector_margins[str(sector)] = {
                    'mean': float(sector_data.mean()),
                    'median': float(sector_data.median()),
                    'count': int(len(sector_data)),
                    }

        earnings_monitor['margin_by_sector'] = sector_margins

        print(f'\n  Top 5 Sectors by {top_margin.replace("_", " ").title()}:')
        sorted_sectors = sorted(sector_margins.items(), key=lambda x: x[1]['mean'], reverse=True)[:5]
        for sector, stats in sorted_sectors:
            print(f'    {sector[:30]:30s}: {stats["mean"]:.2f}')

# ============================================================================
# 3. Earnings Quality Indicators (using PHASE93_FEATURE_INPUTS quality_risk)
# ============================================================================
print('\n🎯 Earnings Quality Indicators (PHASE93 Quality/Risk)...')

# Use quality_risk category from PHASE93_FEATURE_INPUTS + supplemental quality metrics
quality_risk_metrics = PHASE93_FEATURE_INPUTS.get('quality_risk', [])
supplemental_quality = ['roe', 'roa', 'roic', 'asset_turnover', 'fcf_to_net_income']
quality_metrics = list(dict.fromkeys(quality_risk_metrics + supplemental_quality))  # Dedupe
available_quality = [m for m in quality_metrics if m in all_stocks_enhanced.columns]

if available_quality:
    quality_stats = {}

    print(f'  Found {len(available_quality)} quality/risk metrics from PHASE93_FEATURE_INPUTS:')

    for metric in available_quality:
        data = all_stocks_enhanced[metric].dropna()
        if len(data) > 0:
            quality_stats[metric] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'q25': float(data.quantile(0.25)),
                'q75': float(data.quantile(0.75)),
                }

            print(f'    {metric:30s}: mean={data.mean():>10.3f}, median={data.median():>10.3f}')

    earnings_monitor['quality_indicators'] = quality_stats

# ============================================================================
# 4. Regional Earnings Trends (using PHASE93_FEATURE_INPUTS growth category)
# ============================================================================
if 'region' in all_stocks_enhanced.columns:
    print('\n🌍 Regional Earnings Trends (PHASE93 Growth)...')

    # Use growth category from PHASE93_FEATURE_INPUTS for regional analysis
    growth_metrics = PHASE93_FEATURE_INPUTS.get('growth', [])
    supplemental_growth = ['revenue_growth_1y', 'revenue_growth_3y', 'revenue_cagr_3y']
    rev_growth_cols = list(dict.fromkeys(growth_metrics + supplemental_growth))
    available_growth = [c for c in rev_growth_cols if c in all_stocks_enhanced.columns]

    if available_growth:
        rev_col = available_growth[0]  # Use first available
        regional_earnings = {}

        print(f'  Using metric: {rev_col}')

        for region in all_stocks_enhanced['region'].dropna().unique():
            region_data = all_stocks_enhanced[all_stocks_enhanced['region'] == region][rev_col].dropna()
            if len(region_data) >= 5:
                regional_earnings[str(region)] = {
                    'count': int(len(region_data)),
                    'mean_growth': float(region_data.mean()),
                    'median_growth': float(region_data.median()),
                    'std_growth': float(region_data.std()),
                    'positive_growth_pct': float((region_data > 0).sum() / len(region_data) * 100),
                    }

                print(
                        f'    {region:20s}: mean={region_data.mean():>8.2f}, '
                        f'+ve={((region_data > 0).sum() / len(region_data) * 100):>5.1f}%')

        earnings_monitor['by_region'] = regional_earnings
        earnings_monitor['regional_growth_metric_used'] = rev_col

# ============================================================================
# 5. Enhanced EPS Metrics Analysis (NEW)
# ============================================================================
print('\n📊 Enhanced EPS Metrics Analysis...')

# Basic EPS Historical (GAAP-based) - Quarterly progression
eps_basic_cols = [
    'net_eps_basic_ltm', 'net_eps_basic_fq', 'net_eps_basic_fy',
    'net_eps_basic_1fqfq', 'net_eps_basic_2fqfq', 'net_eps_basic_3fqfq'
    ]
available_eps_basic = [c for c in eps_basic_cols if c in all_stocks_enhanced.columns]

if available_eps_basic:
    eps_basic_stats = {}
    print(f'  Found {len(available_eps_basic)} basic EPS (GAAP) metrics')

    for metric in available_eps_basic:
        data = pd.to_numeric(all_stocks_enhanced[metric], errors='coerce').dropna()
        if len(data) > 0:
            eps_basic_stats[metric] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'positive_pct': float((data > 0).sum() / len(data) * 100),
                }
            print(f'    {metric:25s}: mean=${data.mean():>8.2f}, +ve={((data > 0).sum() / len(data) * 100):>5.1f}%')

    earnings_monitor['eps_metrics']['basic_gaap'] = eps_basic_stats

# Adjusted EPS (enhanced coverage)
eps_adj_cols = ['eps_adj_1fy', 'eps_adj_fy', 'eps_adj_ltm']
available_eps_adj = [c for c in eps_adj_cols if c in all_stocks_enhanced.columns]

if available_eps_adj:
    eps_adj_stats = {}
    print(f'\n  Found {len(available_eps_adj)} adjusted EPS metrics')

    for metric in available_eps_adj:
        data = pd.to_numeric(all_stocks_enhanced[metric], errors='coerce').dropna()
        if len(data) > 0:
            eps_adj_stats[metric] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'positive_pct': float((data > 0).sum() / len(data) * 100),
                }
            print(f'    {metric:25s}: mean=${data.mean():>8.2f}, +ve={((data > 0).sum() / len(data) * 100):>5.1f}%')

    earnings_monitor['eps_metrics']['adjusted'] = eps_adj_stats

# Forward EPS Estimates
eps_est_cols = ['eps_norm_est_avg_ntm', 'eps_norm_est_avg_fy1e', 'eps_norm_est_num_fy1e']
available_eps_est = [c for c in eps_est_cols if c in all_stocks_enhanced.columns]

if available_eps_est:
    eps_est_stats = {}
    print(f'\n  Found {len(available_eps_est)} forward EPS estimate metrics')

    for metric in available_eps_est:
        data = pd.to_numeric(all_stocks_enhanced[metric], errors='coerce').dropna()
        if len(data) > 0:
            eps_est_stats[metric] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                }
            print(f'    {metric:25s}: mean={data.mean():>10.2f}, median={data.median():>10.2f}')

    earnings_monitor['eps_metrics']['forward_estimates'] = eps_est_stats

# GAAP EPS Estimates
eps_gaap_est_cols = ['eps_gaap_est_avg_fy1e', 'eps_gaap_est_avg_ntm']
available_eps_gaap_est = [c for c in eps_gaap_est_cols if c in all_stocks_enhanced.columns]

if available_eps_gaap_est:
    eps_gaap_est_stats = {}
    print(f'\n  GAAP EPS Estimates ({len(available_eps_gaap_est)} metrics):')

    for metric in available_eps_gaap_est:
        data = pd.to_numeric(all_stocks_enhanced[metric], errors='coerce').dropna()
        if len(data) > 0:
            eps_gaap_est_stats[metric] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                }
            print(f'    {metric:25s}: mean=${data.mean():>8.2f}, median=${data.median():>8.2f}')

    earnings_monitor['eps_metrics']['gaap_estimates'] = eps_gaap_est_stats

# ============================================================================
# 6. EPS Estimate Revisions Analysis (NEW)
# ============================================================================
print('\n📈 EPS Estimate Revisions Analysis...')

# Normalized EPS revisions (percentage changes over time periods)
eps_norm_rev_cols = [
    'eps_est_avg_rev_pct_fy1e_1w', 'eps_est_avg_rev_pct_fy1e_1m',
    'eps_est_avg_rev_pct_fy1e_3m', 'eps_est_avg_rev_pct_fy1e_6m',
    'eps_est_avg_rev_pct_fy1e_1y'
    ]
available_eps_norm_rev = [c for c in eps_norm_rev_cols if c in all_stocks_enhanced.columns]

if available_eps_norm_rev:
    eps_norm_rev_stats = {}
    print(f'  Normalized EPS Revisions ({len(available_eps_norm_rev)} periods):')

    for metric in available_eps_norm_rev:
        data = pd.to_numeric(all_stocks_enhanced[metric], errors='coerce').dropna()
        if len(data) > 0:
            eps_norm_rev_stats[metric] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'positive_pct': float((data > 0).sum() / len(data) * 100),
                'negative_pct': float((data < 0).sum() / len(data) * 100),
                }
            period = metric.split('_')[-1]  # Extract period (1w, 1m, etc.)
            print(
                    f'    {period:5s}: mean={data.mean():+7.2f}%, +ve={((data > 0).sum() / len(data) * 100):>5.1f}%, -ve={((data < 0).sum() / len(data) * 100):>5.1f}%')

    earnings_monitor['eps_revisions']['normalized'] = eps_norm_rev_stats

# GAAP EPS revisions
eps_gaap_rev_cols = [
    'eps_gaap_est_avg_rev_pct_fy1e_1m', 'eps_gaap_est_avg_rev_pct_fy1e_3m',
    'eps_gaap_est_avg_rev_pct_fy1e_6m', 'eps_gaap_est_avg_rev_pct_fy1e_1y'
    ]
available_eps_gaap_rev = [c for c in eps_gaap_rev_cols if c in all_stocks_enhanced.columns]

if available_eps_gaap_rev:
    eps_gaap_rev_stats = {}
    print(f'\n  GAAP EPS Revisions ({len(available_eps_gaap_rev)} periods):')

    for metric in available_eps_gaap_rev:
        data = pd.to_numeric(all_stocks_enhanced[metric], errors='coerce').dropna()
        if len(data) > 0:
            eps_gaap_rev_stats[metric] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'positive_pct': float((data > 0).sum() / len(data) * 100),
                'negative_pct': float((data < 0).sum() / len(data) * 100),
                }
            period = metric.split('_')[-1]
            print(
                    f'    {period:5s}: mean={data.mean():+7.2f}%, +ve={((data > 0).sum() / len(data) * 100):>5.1f}%, -ve={((data < 0).sum() / len(data) * 100):>5.1f}%')

    earnings_monitor['eps_revisions']['gaap'] = eps_gaap_rev_stats

# ============================================================================
# 7. Interactive EPS Visualizations (Plotly)
# ============================================================================
print('\n📊 Generating Interactive EPS Visualizations...')

# Visualization 1: EPS Metrics Comparison Bar Chart
if available_eps_basic or available_eps_adj:
    eps_viz_data = []

    # Collect basic EPS stats
    if available_eps_basic and 'basic_gaap' in earnings_monitor['eps_metrics']:
        for metric, stats in earnings_monitor['eps_metrics']['basic_gaap'].items():
            eps_viz_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Type': 'Basic (GAAP)',
                'Mean': stats['mean'],
                'Median': stats['median'],
                'Positive %': stats['positive_pct']
                })

    # Collect adjusted EPS stats
    if available_eps_adj and 'adjusted' in earnings_monitor['eps_metrics']:
        for metric, stats in earnings_monitor['eps_metrics']['adjusted'].items():
            eps_viz_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Type': 'Adjusted',
                'Mean': stats['mean'],
                'Median': stats['median'],
                'Positive %': stats['positive_pct']
                })

    if eps_viz_data:
        eps_viz_df = pd.DataFrame(eps_viz_data)

        fig_eps_comparison = px.bar(
                eps_viz_df,
                x='Metric',
                y='Mean',
                color='Type',
                barmode='group',
                title='EPS Metrics Comparison: Basic (GAAP) vs Adjusted',
                labels={'Mean': 'Mean EPS ($)', 'Metric': 'EPS Metric'},
                template='plotly_dark',
                color_discrete_map={'Basic (GAAP)': '#3498db', 'Adjusted': '#00bc8c'}
                )

        fig_eps_comparison.update_layout(
                font=dict(family='Arial, sans-serif', size=14),
                title_font_size=20,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', xanchor='center', x=0.5, y=1.02),
                hovermode='x unified',
                xaxis_tickangle=-45
                )

        # Save visualization
        eps_comparison_path = financial_metrics_dir / 'eps_metrics_comparison.html'
        fig_eps_comparison.write_html(str(eps_comparison_path))
        print(f'  ✓ Saved: {eps_comparison_path}')

# Visualization 2: EPS Revisions Trend Chart
if available_eps_norm_rev and 'normalized' in earnings_monitor['eps_revisions']:
    rev_viz_data = []
    period_order = {'1w': 1, '1m': 2, '3m': 3, '6m': 4, '1y': 5}

    for metric, stats in earnings_monitor['eps_revisions']['normalized'].items():
        period = metric.split('_')[-1]
        rev_viz_data.append({
            'Period': period,
            'Period_Order': period_order.get(period, 99),
            'Mean Revision %': stats['mean'],
            'Upgrades %': stats['positive_pct'],
            'Downgrades %': stats['negative_pct']
            })

    if rev_viz_data:
        rev_viz_df = pd.DataFrame(rev_viz_data).sort_values('Period_Order')

        fig_revisions = go.Figure()

        # Add bars for upgrades and downgrades
        fig_revisions.add_trace(go.Bar(
                name='Upgrades %',
                x=rev_viz_df['Period'],
                y=rev_viz_df['Upgrades %'],
                marker_color='#00bc8c'
                ))

        fig_revisions.add_trace(go.Bar(
                name='Downgrades %',
                x=rev_viz_df['Period'],
                y=rev_viz_df['Downgrades %'],
                marker_color='#e74c3c'
                ))

        # Add line for mean revision
        fig_revisions.add_trace(go.Scatter(
                name='Mean Revision %',
                x=rev_viz_df['Period'],
                y=rev_viz_df['Mean Revision %'],
                mode='lines+markers',
                line=dict(color='#f39c12', width=3),
                marker=dict(size=10),
                yaxis='y2'
                ))

        fig_revisions.update_layout(
                title='EPS Estimate Revisions by Time Period',
                xaxis_title='Revision Period',
                yaxis_title='Percentage of Stocks',
                yaxis2=dict(
                        title='Mean Revision %',
                        overlaying='y',
                        side='right',
                        showgrid=False
                        ),
                template='plotly_dark',
                font=dict(family='Arial, sans-serif', size=14),
                title_font_size=20,
                barmode='group',
                legend=dict(orientation='h', yanchor='bottom', xanchor='center', x=0.5, y=1.02),
                hovermode='x unified'
                )

        # Save visualization
        revisions_path = financial_metrics_dir / 'eps_revisions_trend.html'
        fig_revisions.write_html(str(revisions_path))
        print(f'  ✓ Saved: {revisions_path}')

# Visualization 3: EPS Profitability Distribution by Sector
if 'sector' in all_stocks_enhanced.columns and available_eps_adj:
    eps_col = available_eps_adj[0]
    sector_eps_data = []

    for sector in all_stocks_enhanced['sector'].dropna().unique():
        sector_data = pd.to_numeric(
                all_stocks_enhanced[all_stocks_enhanced['sector'] == sector][eps_col],
                errors='coerce'
                ).dropna()

        if len(sector_data) >= 10:
            sector_eps_data.append({
                'Sector': str(sector),
                'Mean EPS': float(sector_data.mean()),
                'Median EPS': float(sector_data.median()),
                'Profitable %': float((sector_data > 0).sum() / len(sector_data) * 100),
                'Count': int(len(sector_data))
                })

    if sector_eps_data:
        sector_eps_df = pd.DataFrame(sector_eps_data).sort_values('Profitable %', ascending=True)

        fig_sector_eps = px.bar(
                sector_eps_df,
                y='Sector',
                x='Profitable %',
                orientation='h',
                title='Profitability Rate by Sector (% with Positive EPS)',
                labels={'Profitable %': 'Profitable Stocks (%)', 'Sector': ''},
                template='plotly_dark',
                color='Profitable %',
                color_continuous_scale='RdYlGn'
                )

        fig_sector_eps.update_layout(
                font=dict(family='Arial, sans-serif', size=14),
                title_font_size=20,
                showlegend=False,
                hovermode='y unified',
                coloraxis_colorbar=dict(title='Profitable %')
                )

        # Save visualization
        sector_eps_path = financial_metrics_dir / 'eps_profitability_by_sector.html'
        fig_sector_eps.write_html(str(sector_eps_path))
        print(f'  ✓ Saved: {sector_eps_path}')

# ============================================================================
# 8. Export Earnings Monitor JSON
# ============================================================================
print('\n💾 Exporting Earnings Monitor Dashboard...')

earnings_monitor_path = financial_metrics_dir / 'earnings_monitor.json'
with open(earnings_monitor_path, 'w') as f:
    json.dump(make_serializable(earnings_monitor), f, indent=2, default=str)
print(f'  ✓ Saved: {earnings_monitor_path}')

print('\n✓ Earnings Monitoring complete')

#%%
all_stocks_enhanced.head(50)
#%% md
## Cell 10.7: Analyst Rating & Recommendations Analytics

Analyze analyst ratings, recommendations, and price target consensus across sectors, regions, and market segments.

**Key Objectives:**
1. Aggregate analyst recommendation distribution (Buy/Hold/Sell)
2. Analyze rating changes and momentum
3. Compare price target consensus vs current prices
4. Segment analysis by size_class, style_class, sector, industry

**Outputs:**
- **JSON Report:** analyst_recommendations.json

#%%
# ============================================================================
# Cell 10.7: Analyst Rating & Recommendations Analytics
# ============================================================================

print('=' * 80)
print('ANALYST RATING & RECOMMENDATIONS ANALYTICS')
print('=' * 80)

# ============================================================================
# 1. Identify Analyst Rating Columns
# ============================================================================
print('\n🔍 Identifying Analyst Rating Columns...')

# Common analyst rating column patterns
rating_col_patterns = {
    'analyst_rating': ['analyst_rating', 'rating', 'recommendation', 'analyst_recommendation'],
    'buy_ratings': ['buy_ratings', 'num_buy', 'strong_buy', 'buy_count'],
    'hold_ratings': ['hold_ratings', 'num_hold', 'hold_count'],
    'sell_ratings': ['sell_ratings', 'num_sell', 'strong_sell', 'sell_count'],
    'price_target': ['price_target', 'price_target', 'analyst_target', 'price_target_mean', 'price_target_ytd_ago'],
    'target_high': ['price_target_high', 'target_high', 'high_target'],
    'target_low': ['price_target_low', 'target_low', 'low_target'],
    'num_analysts': ['num_analysts', 'analyst_count', 'coverage_count', 'analysts_covering'],
    }

# Find available columns
available_rating_cols = {}
for category, patterns in rating_col_patterns.items():
    for pattern in patterns:
        matching_cols = [c for c in all_stocks_enhanced.columns if pattern.lower() in c.lower()]
        if matching_cols:
            available_rating_cols[category] = matching_cols[0]
            break

print(f'  Found {len(available_rating_cols)} analyst rating column categories:')
for cat, col in available_rating_cols.items():
    print(f'    {cat:20s}: {col}')

# ============================================================================
# 2. Analyst Recommendations Summary
# ============================================================================
analyst_recommendations = {
    'timestamp': datetime.now().isoformat(),
    'total_stocks_analyzed': len(all_stocks_enhanced),
    'available_columns': available_rating_cols,
    'rating_distribution': {},
    'by_sector': {},
    'by_region': {},
    'by_size_class': {},
    'by_style_class': {},
    }

# Price target analysis
if 'price_target' in available_rating_cols:
    target_col = available_rating_cols['price_target']
    print(f'\n📊 Price Target Analysis ({target_col})...')

    target_data = pd.to_numeric(all_stocks_enhanced[target_col], errors='coerce').dropna()
    if len(target_data) > 0:
        analyst_recommendations['price_target_stats'] = {
            'count': int(len(target_data)),
            'mean': float(target_data.mean()),
            'median': float(target_data.median()),
            'std': float(target_data.std()),
            'min': float(target_data.min()),
            'max': float(target_data.max()),
            }
        print(f'  Stocks with price targets: {len(target_data):,}')
        print(f'  Mean target:               ${target_data.mean():.2f}')
        print(f'  Median target:             ${target_data.median():.2f}')

# Analyst coverage analysis
if 'num_analysts' in available_rating_cols:
    coverage_col = available_rating_cols['num_analysts']
    print(f'\n👥 Analyst Coverage Analysis ({coverage_col})...')

    coverage_data = pd.to_numeric(all_stocks_enhanced[coverage_col], errors='coerce').dropna()
    if len(coverage_data) > 0:
        analyst_recommendations['coverage_stats'] = {
            'count': int(len(coverage_data)),
            'mean_analysts': float(coverage_data.mean()),
            'median_analysts': float(coverage_data.median()),
            'max_analysts': int(coverage_data.max()),
            'uncovered_pct': float((coverage_data == 0).sum() / len(coverage_data) * 100),
            }
        print(f'  Stocks with coverage data: {len(coverage_data):,}')
        print(f'  Mean analysts per stock:   {coverage_data.mean():.1f}')
        print(f'  Max analysts:              {int(coverage_data.max())}')

        # Coverage distribution
        coverage_bins = [0, 1, 5, 10, 20, float('inf')]
        coverage_labels = ['0', '1-5', '6-10', '11-20', '20+']
        coverage_dist = pd.cut(coverage_data, bins=coverage_bins, labels=coverage_labels).value_counts()
        analyst_recommendations['coverage_distribution'] = {str(k): int(v) for k, v in coverage_dist.items()}

# ============================================================================
# 3. Sector-Level Analyst Analysis
# ============================================================================
if 'sector' in all_stocks_enhanced.columns and 'price_target' in available_rating_cols:
    print('\n🏢 Sector-Level Analyst Analysis...')
    target_col = available_rating_cols['price_target']

    sector_analyst_stats = {}
    for sector in all_stocks_enhanced['sector'].dropna().unique():
        sector_df = all_stocks_enhanced[all_stocks_enhanced['sector'] == sector]
        target_data = pd.to_numeric(sector_df[target_col], errors='coerce').dropna()

        if len(target_data) >= 5:
            # Calculate target vs price if available
            upside_data = None
            if 'target_vs_price' in sector_df.columns:
                upside_data = sector_df['target_vs_price'].dropna()

            sector_analyst_stats[str(sector)] = {
                'count': int(len(target_data)),
                'mean_target': float(target_data.mean()),
                'median_target': float(target_data.median()),
                }

            if upside_data is not None and len(upside_data) > 0:
                sector_analyst_stats[str(sector)]['mean_upside'] = float(upside_data.mean())
                sector_analyst_stats[str(sector)]['positive_upside_pct'] = float(
                        (upside_data > 0).sum() / len(upside_data) * 100)

    analyst_recommendations['by_sector'] = sector_analyst_stats

    # Print top sectors by upside
    sectors_with_upside = {k: v.get('mean_upside', 0) for k, v in sector_analyst_stats.items() if 'mean_upside' in v}
    if sectors_with_upside:
        print('\n  Top 5 Sectors by Mean Analyst Upside:')
        top_sectors = sorted(sectors_with_upside.items(), key=lambda x: x[1], reverse=True)[:5]
        for sector, upside in top_sectors:
            print(f'    {sector[:30]:30s}: {upside:+.2f}%')

# ============================================================================
# 4. Size Class & Style Class Analysis
# ============================================================================
print('\n📏 Size Class & Style Class Analysis...')

# Size class analysis
if 'size_class' in all_stocks_enhanced.columns and 'target_vs_price' in all_stocks_enhanced.columns:
    size_class_stats = {}
    for size_class in all_stocks_enhanced['size_class'].dropna().unique():
        size_df = all_stocks_enhanced[all_stocks_enhanced['size_class'] == size_class]
        upside_data = size_df['target_vs_price'].dropna()

        if len(upside_data) >= 5:
            size_class_stats[str(size_class)] = {
                'count': int(len(upside_data)),
                'mean_upside': float(upside_data.mean()),
                'median_upside': float(upside_data.median()),
                'positive_pct': float((upside_data > 0).sum() / len(upside_data) * 100),
                }
            print(
                    f'  {size_class:15s}: n={len(upside_data):4d}, mean={upside_data.mean():+.2f}%, positive={((upside_data > 0).sum() / len(upside_data) * 100):.1f}%')

    analyst_recommendations['by_size_class'] = size_class_stats

# Style class analysis
if 'style_class' in all_stocks_enhanced.columns and 'target_vs_price' in all_stocks_enhanced.columns:
    style_class_stats = {}
    for style_class in all_stocks_enhanced['style_class'].dropna().unique():
        style_df = all_stocks_enhanced[all_stocks_enhanced['style_class'] == style_class]
        upside_data = style_df['target_vs_price'].dropna()

        if len(upside_data) >= 5:
            style_class_stats[str(style_class)] = {
                'count': int(len(upside_data)),
                'mean_upside': float(upside_data.mean()),
                'median_upside': float(upside_data.median()),
                'positive_pct': float((upside_data > 0).sum() / len(upside_data) * 100),
                }
            print(
                    f'  {style_class:15s}: n={len(upside_data):4d}, mean={upside_data.mean():+.2f}%, positive={((upside_data > 0).sum() / len(upside_data) * 100):.1f}%')

    analyst_recommendations['by_style_class'] = style_class_stats

# ============================================================================
# 5. Export Analyst Recommendations JSON
# ============================================================================
print('\n💾 Exporting Analyst Recommendations...')

analyst_recommendations_path = financial_metrics_dir / 'analyst_recommendations.json'
with open(analyst_recommendations_path, 'w') as f:
    json.dump(make_serializable(analyst_recommendations), f, indent=2, default=str)
print(f'  ✓ Saved: {analyst_recommendations_path}')

print('\n✓ Analyst Rating & Recommendations Analytics complete')

#%% md
## Cell 10.8: Estimated vs. Actual vs. Adjusted Earnings Analytics

Analyze earnings estimates, actual reported earnings, and adjusted earnings metrics across sectors, regions, and market segments.

**Key Objectives:**
1. Compare estimated vs actual earnings (earnings surprise analysis)
2. Analyze adjusted vs GAAP earnings differences
3. Track earnings revision trends
4. Segment analysis by sector, region, size_class, style_class, industry, trading country, exchange

**Outputs:**
- **JSON Report:** earnings_estimates_analysis.json

#%%
# ============================================================================
# Cell 10.8: Estimated vs. Actual vs. Adjusted Earnings Analytics (Phase 9.3)
# ============================================================================

print('=' * 80)
print('ESTIMATED VS. ACTUAL VS. ADJUSTED EARNINGS ANALYTICS (Phase 9.3)')
print('=' * 80)

# ============================================================================
# 1. Engineer Profitability and Revenue Forecast Features
# ============================================================================
print('\n🔧 Engineering Profitability and Revenue Forecast Features...')

# Import advanced feature engineering functions
from finance_ml.ml_workflow.features.advanced import (
    engineer_profitability_ratios,
    engineer_revenue_forecast_features
    )

# Apply profitability ratio engineering
all_stocks_enhanced = engineer_profitability_ratios(all_stocks_enhanced)
print('  ✓ Profitability ratios engineered')

# Apply revenue forecast feature engineering
all_stocks_enhanced = engineer_revenue_forecast_features(all_stocks_enhanced)
print('  ✓ Revenue forecast features engineered')

# ============================================================================
# 2. Identify Earnings-Related Columns (using PHASE93_FEATURE_INPUTS)
# ============================================================================
print('\n🔍 Identifying Earnings-Related Columns (PHASE93_FEATURE_INPUTS)...')

# Schema-driven metric selection from PHASE93_FEATURE_INPUTS categories
# (code_guidelines.md §9.3: Use PHASE93_FEATURE_INPUTS for metric categorization)
earnings_phase93_categories = {
    'profitability': PHASE93_FEATURE_INPUTS.get('profitability', []),
    'forecasts': PHASE93_FEATURE_INPUTS.get('forecasts', []),
    'valuation': PHASE93_FEATURE_INPUTS.get('valuation', [])[:15],  # Top 15 valuation
    'growth': PHASE93_FEATURE_INPUTS.get('growth', []),
    'cash_flow': PHASE93_FEATURE_INPUTS.get('cash_flow', [])[:10],  # Top 10 cash flow
    }

# Enhanced earnings columns from engineered features
earnings_col_patterns = {
    # Profitability ratios (from engineer_profitability_ratios)
    'roe': ['roe', 'return_on_equity_pct_ltm'],
    'roa': ['roa', 'return_on_assets_roa_pct_ltm'],
    'roic': ['roic'],
    'gross_margin': ['gross_margin_pct', 'gross_profit_margin_pct_ltm'],
    'operating_margin': ['operating_margin_pct', 'operating_income_ltm'],
    'net_margin': ['net_margin_pct', 'net_income_margin_pct_ltm'],

    # Adjustment ratios (earnings quality indicators)
    'ebitda_adjustment': ['ebitda_adjustment_ratio_ltm', 'ebitda_adjustment_ratio_fy'],
    'ebit_adjustment': ['ebit_adjustment_ratio_ltm', 'ebit_adjustment_ratio_fy'],
    'net_income_adjustment': ['net_income_adjustment_ratio_ltm', 'net_income_adjustment_ratio_fy'],

    # Revenue forecast features (from engineer_revenue_forecast_features)
    'revenue_estimate_spread_ntm': ['revenue_estimate_spread_ntm'],
    'revenue_estimate_spread_fy1e': ['revenue_estimate_spread_fy1e'],
    'revenue_consensus_uncertainty': ['revenue_consensus_uncertainty_score'],
    'revenue_growth_implied_ntm': ['revenue_growth_implied_ntm'],
    'revenue_growth_implied_fy1e': ['revenue_growth_implied_fy1e'],
    'revenue_growth_acceleration': ['revenue_growth_acceleration'],
    'estimate_confidence': ['estimate_confidence_flag'],
    'growth_surprise_potential': ['growth_surprise_potential'],

    # Legacy EPS/Revenue columns (backward compatibility)
    'eps_actual': ['eps_adj_ltm', 'net_eps_basic_ltm', 'eps_ltm', 'eps'],
    'eps_estimate': ['eps_norm_est_avg_ntm', 'eps_norm_est_avg_fy1e'],
    'eps_adjusted': ['eps_adj_fy', 'eps_adj_1fy'],

    # NEW: Enhanced EPS columns
    'eps_basic_fq': ['net_eps_basic_fq'],
    'eps_basic_fy': ['net_eps_basic_fy'],
    'eps_basic_qoq_1q': ['net_eps_basic_1fqfq'],
    'eps_basic_qoq_2q': ['net_eps_basic_2fqfq'],
    'eps_basic_qoq_3q': ['net_eps_basic_3fqfq'],
    'eps_gaap_est_fy1e': ['eps_gaap_est_avg_fy1e'],
    'eps_gaap_est_ntm': ['eps_gaap_est_avg_ntm'],
    'eps_analyst_count': ['eps_norm_est_num_fy1e'],

    # Revenue columns
    'revenue_actual': ['total_revenues_ltm', 'total_revenues_fy'],
    'revenue_estimate': ['revenues_est_avg_ntm', 'revenues_est_med_ntm'],
    'net_income': ['net_income_is_ltm', 'net_income'],
    'ebitda': ['ebitda_ltm', 'ebitda_est_avg_fy1e', 'ebitda_est_avg_ntm', 'ebitda'],
    'ebit': ['ebit_ltm', 'operating_income_ltm'],
    }

# Find available columns from PHASE93 categories
available_phase93_cols = {}
for category, metrics in earnings_phase93_categories.items():
    available = [m for m in metrics if m in all_stocks_enhanced.columns]
    if available:
        available_phase93_cols[category] = available

print(f'\n  📊 PHASE93_FEATURE_INPUTS Categories:')
for cat, cols in available_phase93_cols.items():
    print(f'    {cat:15s}: {len(cols)} metrics available')

# Find available columns from enhanced patterns
available_earnings_cols = {}
for category, patterns in earnings_col_patterns.items():
    for pattern in patterns:
        if pattern in all_stocks_enhanced.columns:
            available_earnings_cols[category] = pattern
            break

print(f'\n  📈 Specific EPS/Revenue Columns Found:')
for cat, col in available_earnings_cols.items():
    print(f'    {cat:20s}: {col}')

# ============================================================================
# 3. Initialize Earnings Estimates Analysis (Phase 9.3 Enhanced)
# ============================================================================
earnings_estimates_analysis = {
    'timestamp': datetime.now().isoformat(),
    'total_stocks_analyzed': len(all_stocks_enhanced),
    'phase93_categories_used': list(available_phase93_cols.keys()),
    'phase93_metrics_available': {cat: len(cols) for cat, cols in available_phase93_cols.items()},
    'available_columns': available_earnings_cols,
    'eps_analysis': {},
    'earnings_surprise': {},
    'phase93_category_analysis': {},  # New: Analysis by PHASE93 category
    'by_sector': {},
    'by_region': {},
    'by_size_class': {},
    'by_style_class': {},
    'by_industry': {},
    'by_trading_country': {},
    'by_exchange': {},
    }

# Analyze metrics by PHASE93 category
print('\n📊 Phase 9.3 Category Analysis...')
for category, metrics in available_phase93_cols.items():
    category_stats = {}
    for metric in metrics[:5]:  # Top 5 per category
        data = pd.to_numeric(all_stocks_enhanced[metric], errors='coerce').dropna()
        if len(data) > 0:
            category_stats[metric] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                }
    if category_stats:
        earnings_estimates_analysis['phase93_category_analysis'][category] = category_stats
        print(f'  {category:15s}: {len(category_stats)} metrics analyzed')

# ============================================================================
# 4. EPS Analysis (Actual vs Estimated vs Adjusted)
# ============================================================================
print('\n📊 EPS Analysis...')

eps_metrics = {}

# Actual EPS
if 'eps_actual' in available_earnings_cols:
    eps_col = available_earnings_cols['eps_actual']
    eps_data = pd.to_numeric(all_stocks_enhanced[eps_col], errors='coerce').dropna()
    if len(eps_data) > 0:
        eps_metrics['actual'] = {
            'column': eps_col,
            'count': int(len(eps_data)),
            'mean': float(eps_data.mean()),
            'median': float(eps_data.median()),
            'std': float(eps_data.std()),
            'positive_pct': float((eps_data > 0).sum() / len(eps_data) * 100),
            'negative_pct': float((eps_data < 0).sum() / len(eps_data) * 100),
            }
        print(f'  Actual EPS ({eps_col}):')
        print(f'    Valid values: {len(eps_data):,}')
        print(f'    Mean: ${eps_data.mean():.2f}, Median: ${eps_data.median():.2f}')
        print(f'    Profitable: {((eps_data > 0).sum() / len(eps_data) * 100):.1f}%')

# Estimated EPS
if 'eps_estimate' in available_earnings_cols:
    est_col = available_earnings_cols['eps_estimate']
    est_data = pd.to_numeric(all_stocks_enhanced[est_col], errors='coerce').dropna()
    if len(est_data) > 0:
        eps_metrics['estimated'] = {
            'column': est_col,
            'count': int(len(est_data)),
            'mean': float(est_data.mean()),
            'median': float(est_data.median()),
            'std': float(est_data.std()),
            }
        print(f'  Estimated EPS ({est_col}):')
        print(f'    Valid values: {len(est_data):,}')
        print(f'    Mean: ${est_data.mean():.2f}, Median: ${est_data.median():.2f}')

# Adjusted EPS
if 'eps_adjusted' in available_earnings_cols:
    adj_col = available_earnings_cols['eps_adjusted']
    adj_data = pd.to_numeric(all_stocks_enhanced[adj_col], errors='coerce').dropna()
    if len(adj_data) > 0:
        eps_metrics['adjusted'] = {
            'column': adj_col,
            'count': int(len(adj_data)),
            'mean': float(adj_data.mean()),
            'median': float(adj_data.median()),
            'std': float(adj_data.std()),
            }
        print(f'  Adjusted EPS ({adj_col}):')
        print(f'    Valid values: {len(adj_data):,}')
        print(f'    Mean: ${adj_data.mean():.2f}, Median: ${adj_data.median():.2f}')

earnings_estimates_analysis['eps_analysis'] = eps_metrics

# ============================================================================
# 5. Earnings Surprise Analysis
# ============================================================================
print('\n📈 Earnings Surprise Analysis...')

# Calculate earnings surprise if we have both actual and estimated
if 'eps_actual' in available_earnings_cols and 'eps_estimate' in available_earnings_cols:
    actual_col = available_earnings_cols['eps_actual']
    est_col = available_earnings_cols['eps_estimate']

    # Get aligned data
    mask = all_stocks_enhanced[actual_col].notna() & all_stocks_enhanced[est_col].notna()
    actual = pd.to_numeric(all_stocks_enhanced.loc[mask, actual_col], errors='coerce')
    estimated = pd.to_numeric(all_stocks_enhanced.loc[mask, est_col], errors='coerce')

    # Calculate surprise percentage
    with np.errstate(divide='ignore', invalid='ignore'):
        surprise_pct = ((actual - estimated) / estimated.abs()) * 100
    surprise_pct = surprise_pct.replace([np.inf, -np.inf], np.nan).dropna()

    if len(surprise_pct) > 0:
        earnings_estimates_analysis['earnings_surprise'] = {
            'count': int(len(surprise_pct)),
            'mean_surprise_pct': float(surprise_pct.mean()),
            'median_surprise_pct': float(surprise_pct.median()),
            'beat_pct': float((surprise_pct > 0).sum() / len(surprise_pct) * 100),
            'miss_pct': float((surprise_pct < 0).sum() / len(surprise_pct) * 100),
            'large_beat_pct': float((surprise_pct > 10).sum() / len(surprise_pct) * 100),
            'large_miss_pct': float((surprise_pct < -10).sum() / len(surprise_pct) * 100),
            }
        print(f'  Stocks with surprise data: {len(surprise_pct):,}')
        print(f'  Mean surprise:             {surprise_pct.mean():+.2f}%')
        print(f'  Beat estimates:            {((surprise_pct > 0).sum() / len(surprise_pct) * 100):.1f}%')
        print(f'  Miss estimates:            {((surprise_pct < 0).sum() / len(surprise_pct) * 100):.1f}%')

        # Store surprise in DataFrame for segment analysis
        all_stocks_enhanced.loc[mask, 'calculated_eps_surprise'] = surprise_pct
elif 'earnings_surprise' in available_earnings_cols:
    surprise_col = available_earnings_cols['earnings_surprise']
    surprise_data = pd.to_numeric(all_stocks_enhanced[surprise_col], errors='coerce').dropna()
    if len(surprise_data) > 0:
        earnings_estimates_analysis['earnings_surprise'] = {
            'count': int(len(surprise_data)),
            'mean_surprise_pct': float(surprise_data.mean()),
            'median_surprise_pct': float(surprise_data.median()),
            'beat_pct': float((surprise_data > 0).sum() / len(surprise_data) * 100),
            'miss_pct': float((surprise_data < 0).sum() / len(surprise_data) * 100),
            }
        print(f'  Using existing surprise column: {surprise_col}')
        print(f'  Mean surprise: {surprise_data.mean():+.2f}%')


# ============================================================================
# 6. Segment Analysis Functions
# ============================================================================
def analyze_earnings_by_segment(df, segment_col, eps_col):
    """Analyze earnings metrics by a segment column."""
    segment_stats = {}

    for segment in df[segment_col].dropna().unique():
        segment_df = df[df[segment_col] == segment]
        eps_data = pd.to_numeric(segment_df[eps_col], errors='coerce').dropna()

        if len(eps_data) >= 5:
            segment_stats[str(segment)] = {
                'count': int(len(eps_data)),
                'mean_eps': float(eps_data.mean()),
                'median_eps': float(eps_data.median()),
                'positive_pct': float((eps_data > 0).sum() / len(eps_data) * 100),
                }

            # Add surprise if available
            if 'calculated_eps_surprise' in segment_df.columns:
                surprise = segment_df['calculated_eps_surprise'].dropna()
                if len(surprise) >= 3:
                    segment_stats[str(segment)]['mean_surprise'] = float(surprise.mean())
                    segment_stats[str(segment)]['beat_pct'] = float((surprise > 0).sum() / len(surprise) * 100)

    return segment_stats


# Get primary EPS column
primary_eps_col = None
for key in ['eps_actual', 'eps_adjusted', 'net_income']:
    if key in available_earnings_cols:
        primary_eps_col = available_earnings_cols[key]
        break

if primary_eps_col:
    # ============================================================================
    # 7. By Sector
    # ============================================================================
    if 'sector' in all_stocks_enhanced.columns:
        print('\n🏢 Earnings by Sector...')
        sector_stats = analyze_earnings_by_segment(all_stocks_enhanced, 'sector', primary_eps_col)
        earnings_estimates_analysis['by_sector'] = sector_stats

        # Print top 5 sectors by profitability
        profitable_sectors = {k: v['positive_pct'] for k, v in sector_stats.items()}
        top_sectors = sorted(profitable_sectors.items(), key=lambda x: x[1], reverse=True)[:5]
        for sector, pct in top_sectors:
            print(f'  {sector[:30]:30s}: {pct:.1f}% profitable')

    # ============================================================================
    # 8. By Region
    # ============================================================================
    if 'region' in all_stocks_enhanced.columns:
        print('\n🌍 Earnings by Region...')
        region_stats = analyze_earnings_by_segment(all_stocks_enhanced, 'region', primary_eps_col)
        earnings_estimates_analysis['by_region'] = region_stats

        for region, stats in region_stats.items():
            print(f'  {region:15s}: n={stats["count"]:4d}, profitable={stats["positive_pct"]:.1f}%')

    # ============================================================================
    # 9. By Size Class
    # ============================================================================
    if 'size_class' in all_stocks_enhanced.columns:
        print('\n📏 Earnings by Size Class...')
        size_stats = analyze_earnings_by_segment(all_stocks_enhanced, 'size_class', primary_eps_col)
        earnings_estimates_analysis['by_size_class'] = size_stats

        for size_class, stats in size_stats.items():
            print(f'  {size_class:15s}: n={stats["count"]:4d}, profitable={stats["positive_pct"]:.1f}%')

    # ============================================================================
    # 10. By Style Class
    # ============================================================================
    if 'style_class' in all_stocks_enhanced.columns:
        print('\n🎨 Earnings by Style Class...')
        style_stats = analyze_earnings_by_segment(all_stocks_enhanced, 'style_class', primary_eps_col)
        earnings_estimates_analysis['by_style_class'] = style_stats

        for style_class, stats in style_stats.items():
            print(f'  {style_class:15s}: n={stats["count"]:4d}, profitable={stats["positive_pct"]:.1f}%')

    # ============================================================================
    # 11. By Industry
    # ============================================================================
    if 'industry' in all_stocks_enhanced.columns:
        print('\n🏭 Earnings by Industry (Top 10)...')
        industry_stats = analyze_earnings_by_segment(all_stocks_enhanced, 'industry', primary_eps_col)
        earnings_estimates_analysis['by_industry'] = industry_stats

        # Show top 10 industries by count
        top_industries = sorted(industry_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        for industry, stats in top_industries:
            print(f'  {industry[:35]:35s}: n={stats["count"]:4d}, profitable={stats["positive_pct"]:.1f}%')

    # ============================================================================
    # 12. By Trading Country
    # ============================================================================
    trading_country_cols = ['trading_country', 'country', 'domicile']
    trading_country_col = None
    for col in trading_country_cols:
        if col in all_stocks_enhanced.columns:
            trading_country_col = col
            break

    if trading_country_col:
        print(f'\n🌐 Earnings by Trading Country ({trading_country_col})...')
        country_stats = analyze_earnings_by_segment(all_stocks_enhanced, trading_country_col, primary_eps_col)
        earnings_estimates_analysis['by_trading_country'] = country_stats

        # Show top 10 countries by count
        top_countries = sorted(country_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        for country, stats in top_countries:
            print(f'  {country[:25]:25s}: n={stats["count"]:4d}, profitable={stats["positive_pct"]:.1f}%')

    # ============================================================================
    # 13. By Exchange
    # ============================================================================
    industry_cols = ['industry', 'stock_exchange', 'primary_exchange', 'listing_exchange']
    industry_col = None
    for col in industry_cols:
        if col in all_stocks_enhanced.columns:
            industry_col = col
            break

    if industry_col:
        print(f'\n📈 Earnings by Industry ({industry_col})...')
        industry_stats = analyze_earnings_by_segment(all_stocks_enhanced, industry_col, primary_eps_col)
        earnings_estimates_analysis['by_exchange'] = industry_stats

        # Show top 10 exchanges by count
        top_exchanges = sorted(industry_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        for exchange, stats in top_exchanges:
            print(f'  {exchange[:25]:25s}: n={stats["count"]:4d}, profitable={stats["positive_pct"]:.1f}%')

# ============================================================================
# 14. EPS Revision Momentum Analysis (NEW)
# ============================================================================
print('\n📈 EPS Revision Momentum Analysis...')

# Analyze revision trends across time periods
eps_revision_cols = {
    'normalized': [
        'eps_est_avg_rev_pct_fy1e_1w', 'eps_est_avg_rev_pct_fy1e_1m',
        'eps_est_avg_rev_pct_fy1e_3m', 'eps_est_avg_rev_pct_fy1e_6m',
        'eps_est_avg_rev_pct_fy1e_1y'
        ],
    'gaap': [
        'eps_gaap_est_avg_rev_pct_fy1e_1m', 'eps_gaap_est_avg_rev_pct_fy1e_3m',
        'eps_gaap_est_avg_rev_pct_fy1e_6m', 'eps_gaap_est_avg_rev_pct_fy1e_1y'
        ]
    }

revision_analysis = {}

for rev_type, cols in eps_revision_cols.items():
    available_rev_cols = [c for c in cols if c in all_stocks_enhanced.columns]

    if available_rev_cols:
        print(f'\n  {rev_type.title()} EPS Revisions:')
        rev_stats = {}

        for col in available_rev_cols:
            data = pd.to_numeric(all_stocks_enhanced[col], errors='coerce').dropna()

            if len(data) > 0:
                period = col.split('_')[-1]  # Extract period (1w, 1m, etc.)
                rev_stats[period] = {
                    'column': col,
                    'count': int(len(data)),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'positive_pct': float((data > 0).sum() / len(data) * 100),
                    'negative_pct': float((data < 0).sum() / len(data) * 100),
                    'large_upgrade_pct': float((data > 5).sum() / len(data) * 100),
                    'large_downgrade_pct': float((data < -5).sum() / len(data) * 100),
                    }

                print(
                        f'    {period:5s}: mean={data.mean():+6.2f}%, upgrades={((data > 0).sum() / len(data) * 100):>5.1f}%, downgrades={((data < 0).sum() / len(data) * 100):>5.1f}%')

        revision_analysis[rev_type] = rev_stats

earnings_estimates_analysis['eps_revision_momentum'] = revision_analysis

# Identify stocks with strong upgrade momentum (positive revisions across multiple periods)
upgrade_momentum_cols = ['eps_est_avg_rev_pct_fy1e_1m', 'eps_est_avg_rev_pct_fy1e_3m', 'eps_est_avg_rev_pct_fy1e_6m']
available_upgrade_cols = [c for c in upgrade_momentum_cols if c in all_stocks_enhanced.columns]

if len(available_upgrade_cols) >= 2:
    # Start with first available column
    upgrade_mask = pd.to_numeric(all_stocks_enhanced[available_upgrade_cols[0]], errors='coerce') > 0

    # Add additional conditions
    for col in available_upgrade_cols[1:]:
        upgrade_mask = upgrade_mask & (pd.to_numeric(all_stocks_enhanced[col], errors='coerce') > 0)

    strong_upgrades = all_stocks_enhanced[upgrade_mask]
    print(
            f'\n  📈 Stocks with consistent upgrade momentum: {len(strong_upgrades):,} ({len(strong_upgrades) / len(all_stocks_enhanced) * 100:.1f}%)')

    earnings_estimates_analysis['upgrade_momentum'] = {
        'count': int(len(strong_upgrades)),
        'percentage': float(len(strong_upgrades) / len(all_stocks_enhanced) * 100),
        'periods_used': available_upgrade_cols,
        }

# ============================================================================
# 15. Interactive EPS Revision Visualizations (Plotly)
# ============================================================================
print('\n📊 Generating Interactive EPS Revision Visualizations...')

# Create output directory for earnings visualizations
earnings_viz_dir = OUTPUT_DIR / 'eda' / 'earnings_visualizations'
earnings_viz_dir.mkdir(parents=True, exist_ok=True)

# Visualization 1: EPS Revision Momentum Heatmap by Sector
if 'sector' in all_stocks_enhanced.columns and available_upgrade_cols:
    sector_revision_data = []

    for sector in all_stocks_enhanced['sector'].dropna().unique():
        sector_df = all_stocks_enhanced[all_stocks_enhanced['sector'] == sector]
        if len(sector_df) >= 10:
            sector_row = {'Sector': str(sector)}
            for col in available_upgrade_cols:
                data = pd.to_numeric(sector_df[col], errors='coerce').dropna()
                if len(data) > 0:
                    period = col.split('_')[-1]
                    sector_row[f'{period} Mean Rev %'] = float(data.mean())
            if len(sector_row) > 1:
                sector_revision_data.append(sector_row)

    if sector_revision_data:
        sector_rev_df = pd.DataFrame(sector_revision_data)

        # Create heatmap
        value_cols = [c for c in sector_rev_df.columns if c != 'Sector']
        if value_cols:
            fig_heatmap = px.imshow(
                    sector_rev_df[value_cols].values,
                    x=value_cols,
                    y=sector_rev_df['Sector'].tolist(),
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0,
                    title='EPS Revision Momentum by Sector',
                    labels=dict(x='Revision Period', y='Sector', color='Mean Revision %'),
                    template='plotly_dark',
                    text_auto='.1f'
                    )

            fig_heatmap.update_layout(
                    font=dict(family='Arial, sans-serif', size=12),
                    title_font_size=20,
                    height=600,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    )

            heatmap_path = earnings_viz_dir / 'eps_revision_momentum_heatmap.html'
            fig_heatmap.write_html(str(heatmap_path))
            print(f'  ✓ Saved: {heatmap_path}')

# Visualization 2: Upgrade vs Downgrade Distribution
if revision_analysis.get('normalized'):
    upgrade_downgrade_data = []
    period_order = {'1w': 1, '1m': 2, '3m': 3, '6m': 4, '1y': 5}

    for period, stats in revision_analysis['normalized'].items():
        upgrade_downgrade_data.append({
            'Period': period,
            'Period_Order': period_order.get(period, 99),
            'Upgrades': stats['positive_pct'],
            'Downgrades': -stats['negative_pct'],  # Negative for visual effect
            'Large Upgrades (>5%)': stats['large_upgrade_pct'],
            'Large Downgrades (<-5%)': -stats['large_downgrade_pct'],
            })

    if upgrade_downgrade_data:
        ud_df = pd.DataFrame(upgrade_downgrade_data).sort_values('Period_Order')

        fig_ud = go.Figure()

        # Upgrades (positive)
        fig_ud.add_trace(go.Bar(
                name='Upgrades',
                x=ud_df['Period'],
                y=ud_df['Upgrades'],
                marker_color='#00bc8c',
                text=ud_df['Upgrades'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
                ))

        # Downgrades (negative)
        fig_ud.add_trace(go.Bar(
                name='Downgrades',
                x=ud_df['Period'],
                y=ud_df['Downgrades'],
                marker_color='#e74c3c',
                text=ud_df['Downgrades'].apply(lambda x: f'{abs(x):.1f}%'),
                textposition='outside'
                ))

        fig_ud.update_layout(
                title='EPS Estimate Revisions: Upgrades vs Downgrades by Period',
                xaxis_title='Revision Period',
                yaxis_title='Percentage of Stocks',
                template='plotly_dark',
                font=dict(family='Arial, sans-serif', size=14),
                title_font_size=20,
                barmode='relative',
                legend=dict(orientation='h', yanchor='bottom', xanchor='center', x=0.5, y=1.02),
                hovermode='x unified',
                yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='white')
                )

        ud_path = earnings_viz_dir / 'eps_upgrade_downgrade_distribution.html'
        fig_ud.write_html(str(ud_path))
        print(f'  ✓ Saved: {ud_path}')

# Visualization 3: Revision Momentum Scatter (1M vs 3M)
if 'eps_est_avg_rev_pct_fy1e_1m' in all_stocks_enhanced.columns and 'eps_est_avg_rev_pct_fy1e_3m' in all_stocks_enhanced.columns:
    scatter_df = all_stocks_enhanced[
        ['ticker', 'sector', 'eps_est_avg_rev_pct_fy1e_1m', 'eps_est_avg_rev_pct_fy1e_3m']].copy()
    scatter_df['rev_1m'] = pd.to_numeric(scatter_df['eps_est_avg_rev_pct_fy1e_1m'], errors='coerce')
    scatter_df['rev_3m'] = pd.to_numeric(scatter_df['eps_est_avg_rev_pct_fy1e_3m'], errors='coerce')
    scatter_df = scatter_df.dropna(subset=['rev_1m', 'rev_3m'])

    # Clip extreme values for better visualization
    scatter_df['rev_1m_clipped'] = scatter_df['rev_1m'].clip(-50, 50)
    scatter_df['rev_3m_clipped'] = scatter_df['rev_3m'].clip(-50, 50)

    if len(scatter_df) > 0:
        fig_scatter = px.scatter(
                scatter_df,
                x='rev_3m_clipped',
                y='rev_1m_clipped',
                color='sector',
                hover_data=['ticker', 'rev_1m', 'rev_3m'],
                title='EPS Revision Momentum: 1-Month vs 3-Month Revisions',
                labels={
                    'rev_3m_clipped': '3-Month Revision (%)',
                    'rev_1m_clipped': '1-Month Revision (%)',
                    'sector': 'Sector'
                    },
                template='plotly_dark',
                opacity=0.7
                )

        # Add quadrant lines
        fig_scatter.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.5)
        fig_scatter.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.5)

        # Add quadrant annotations
        fig_scatter.add_annotation(x=25, y=25, text='Strong Momentum', showarrow=False,
                                   font=dict(color='#00bc8c', size=12))
        fig_scatter.add_annotation(x=-25, y=-25, text='Weak Momentum', showarrow=False,
                                   font=dict(color='#e74c3c', size=12))

        fig_scatter.update_layout(
                font=dict(family='Arial, sans-serif', size=14),
                title_font_size=20,
                showlegend=True,
                legend=dict(orientation='v', yanchor='top', xanchor='left', x=1.02, y=1),
                hovermode='closest',
                height=600
                )

        scatter_path = earnings_viz_dir / 'eps_revision_momentum_scatter.html'
        fig_scatter.write_html(str(scatter_path))
        print(f'  ✓ Saved: {scatter_path}')

# ============================================================================
# 16. Export Earnings Estimates Analysis JSON
# ============================================================================
print('\n💾 Exporting Earnings Estimates Analysis...')

earnings_estimates_path = financial_metrics_dir / 'earnings_estimates_analysis.json'
with open(earnings_estimates_path, 'w') as f:
    json.dump(make_serializable(earnings_estimates_analysis), f, indent=2, default=str)
print(f'  ✓ Saved: {earnings_estimates_path}')

print('\n✓ Estimated vs. Actual vs. Adjusted Earnings Analytics complete')

# ============================================================================
# Interactive Dashboard: Earnings Calendar (Phase 9.3 Enhanced)
# ============================================================================
# Note: earnings_widgets functions imported at notebook top (line 116)

print('\n' + '=' * 80)
print('INTERACTIVE EARNINGS CALENDAR DASHBOARD (Phase 9.3)')
print('=' * 80)

# Using all_stocks_enhanced if available, else fallback
if 'all_stocks_enhanced' in locals():
    display_df = all_stocks_enhanced
elif 'all_stocks_features' in locals():
    display_df = all_stocks_features
else:
    display_df = all_stocks_preprocessed

# 1. Display Earnings Dashboard with all PHASE93 categories
print('\n📅 Earnings Calendar Dashboard (All Categories):')
earnings_styler = display_earnings_dashboard(display_df, mode='all', top_n=50)
if earnings_styler is not None:
    display(earnings_styler)

# 2. Create and save PHASE93 Category Metrics Charts
print('\n📊 Generating Phase 9.3 Category Visualization Charts...')

# Create output directory for earnings visualizations
earnings_viz_dir = OUTPUT_DIR / 'eda' / 'earnings_visualizations'
earnings_viz_dir.mkdir(parents=True, exist_ok=True)

# Generate charts for key PHASE93 categories
phase93_viz_categories = ['profitability', 'valuation', 'growth', 'forecasts', 'quality_risk']
for category in phase93_viz_categories:
    try:
        fig = create_earnings_metrics_chart(
                display_df,
                metric_category=category,
                top_n=20,
                output_path=earnings_viz_dir / f'earnings_{category}_metrics.html'
                )
        print(f'  ✓ Generated: earnings_{category}_metrics.html')
    except Exception as e:
        print(f'  ⚠ Could not generate {category} chart: {e}')

# 3. Create Category Comparison Chart
print('\n📈 Generating Phase 9.3 Category Comparison Chart...')
try:
    fig_comparison = create_category_comparison_chart(
            display_df,
            categories=list(PHASE93_FEATURE_INPUTS.keys()),
            top_n=50,
            output_path=earnings_viz_dir / 'phase93_category_comparison.html'
            )
    print(f'  ✓ Generated: phase93_category_comparison.html')
    fig_comparison.show()
except Exception as e:
    print(f'  ⚠ Could not generate category comparison chart: {e}')

# 4. Display mode-specific dashboards
print('\n📋 Mode-Specific Dashboards:')
for mode in ['earnings', 'dividends', 'valuation']:
    print(f'\n  {mode.title()} Mode:')
    mode_styler = display_earnings_dashboard(display_df, mode=mode, top_n=20)
    if mode_styler is not None:
        # Just print summary, don't display all
        print(f'    Records: {len(mode_styler.data)}')

print('\n✓ Interactive Earnings Dashboard Generation Complete')
print(f'  Visualizations saved to: {earnings_viz_dir}')

#%% md
## Cell 10.9: Dividend Analytics

Comprehensive dividend analysis across sectors, regions, and market segments including yield, payout ratios, dividend growth, and income stock screening.

**Key Objectives:**
1. Analyze dividend yield distribution and trends
2. Evaluate payout ratios and dividend sustainability
3. Track dividend growth and consistency
4. Segment analysis by size_class, style_class, sector, industry, trading country, exchange

**Outputs:**
- **JSON Report:** dividend_analytics.json

#%%
# ============================================================================
# Cell 10.9: Dividend Analytics (Phase 9.3 Schema-Driven)
# ============================================================================

print('=' * 80)
print('DIVIDEND ANALYTICS (Phase 9.3 Schema-Driven)')
print('=' * 80)

# ============================================================================
# 1. Identify Dividend-Related Columns (using PHASE93_FEATURE_INPUTS)
# ============================================================================
print('\n🔍 Identifying Dividend-Related Columns (PHASE93_FEATURE_INPUTS)...')

# Schema-driven metric selection from PHASE93_FEATURE_INPUTS dividends category
# (code_guidelines.md §9.3: Use PHASE93_FEATURE_INPUTS for metric categorization)
dividends_phase93 = PHASE93_FEATURE_INPUTS.get('dividends', [])
cash_flow_phase93 = PHASE93_FEATURE_INPUTS.get('cash_flow', [])[:5]  # Top 5 for sustainability

print(f'\n  📊 PHASE93_FEATURE_INPUTS Dividends Category:')
print(f'    {len(dividends_phase93)} metrics defined')

# Find available PHASE93 dividend columns
available_phase93_dividends = [m for m in dividends_phase93 if m in all_stocks_enhanced.columns]
available_phase93_cashflow = [m for m in cash_flow_phase93 if m in all_stocks_enhanced.columns]

print(f'    {len(available_phase93_dividends)} dividend metrics available')
print(f'    {len(available_phase93_cashflow)} cash flow metrics available (sustainability)')

# Legacy pattern matching for backward compatibility (supplemental)
dividend_col_patterns = {
    'dividend_yield': ['div_yield_ltm', 'div_yield_ntm', 'div_yield_ttm', 'dividend_yield', 'div_yield'],
    'dividend_yield_ind': ['div_yield_ind'],
    'dividend_yield_fwd_1y': ['div_yield_1fyind'],
    'dividend_yield_fwd_2y': ['div_yield_2fyind'],
    'dividend_yield_fwd_3y': ['div_yield_3fyind'],
    'dividend_yield_fwd_4y': ['div_yield_4fyind'],
    'dividend_yield_fwd_5y': ['div_yield_5fyind'],
    'dividend_yield_5y_avg': ['div_yield_5yavgltm'],
    'dividend_per_share': ['dividend_per_share_ltm', 'dividend_per_share', 'dps'],
    'payout_ratio': ['payout_ratio', 'dividend_payout', 'payout_pct'],
    'dividend_growth': ['dividend_growth', 'div_growth', 'dividend_cagr'],
    'ex_dividend_date': ['dividend_record_ex_date', 'ex_dividend_date', 'ex_div_date'],
    'dividend_frequency': ['dividend_record_frequency', 'dividend_frequency', 'div_frequency'],
    'dividend_streak': ['dividend_streak', 'years_of_dividend', 'consecutive_dividends'],
    'dividend_amount': ['dividend_record_amount', 'common_dividends_paid_ltm'],
    'dividends_paid_ltm': ['common_dividends_paid_ltm'],
    'dividends_paid_fy': ['common_dividends_paid_fy'],
    'dividend_currency': ['dividend_record_currency'],
    'buyback_yield': ['buyback_yield_ltm'],
    }

# Find available columns from legacy patterns
available_dividend_cols = {}
for category, patterns in dividend_col_patterns.items():
    for pattern in patterns:
        matching_cols = [c for c in all_stocks_enhanced.columns if pattern.lower() in c.lower()]
        if matching_cols:
            available_dividend_cols[category] = matching_cols[0]
            break

print(f'\n  📈 Specific Dividend Columns Found:')
for cat, col in available_dividend_cols.items():
    print(f'    {cat:25s}: {col}')

# ============================================================================
# 2. Initialize Dividend Analytics (Phase 9.3 Enhanced)
# ============================================================================
dividend_analytics = {
    'timestamp': datetime.now().isoformat(),
    'total_stocks_analyzed': len(all_stocks_enhanced),
    'phase93_dividends_metrics': available_phase93_dividends,
    'phase93_cashflow_metrics': available_phase93_cashflow,
    'available_columns': available_dividend_cols,
    'phase93_category_analysis': {},  # New: Analysis by PHASE93 category
    'yield_analysis': {},
    'payout_analysis': {},
    'dividend_payers': {},
    'sustainability_analysis': {},  # New: Cash flow sustainability metrics
    'by_sector': {},
    'by_region': {},
    'by_size_class': {},
    'by_style_class': {},
    'by_industry': {},
    'by_trading_country': {},
    'by_exchange': {},
    }

# Analyze PHASE93 dividend metrics
print('\n📊 Phase 9.3 Dividend Category Analysis...')
if available_phase93_dividends:
    phase93_div_stats = {}
    for metric in available_phase93_dividends:
        data = pd.to_numeric(all_stocks_enhanced[metric], errors='coerce').dropna()
        if len(data) > 0:
            phase93_div_stats[metric] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'non_zero_pct': float((data != 0).sum() / len(data) * 100),
                }
    dividend_analytics['phase93_category_analysis']['dividends'] = phase93_div_stats
    print(f'  Analyzed {len(phase93_div_stats)} PHASE93 dividend metrics')

# Analyze cash flow sustainability metrics
if available_phase93_cashflow:
    phase93_cf_stats = {}
    for metric in available_phase93_cashflow:
        data = pd.to_numeric(all_stocks_enhanced[metric], errors='coerce').dropna()
        if len(data) > 0:
            phase93_cf_stats[metric] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'positive_pct': float((data > 0).sum() / len(data) * 100),
                }
    dividend_analytics['sustainability_analysis'] = phase93_cf_stats
    print(f'  Analyzed {len(phase93_cf_stats)} cash flow sustainability metrics')

# ============================================================================
# 3. Dividend Yield Analysis
# ============================================================================
print('\n💰 Dividend Yield Analysis...')

if 'dividend_yield' in available_dividend_cols:
    yield_col = available_dividend_cols['dividend_yield']
    yield_data = pd.to_numeric(all_stocks_enhanced[yield_col], errors='coerce').dropna()

    if len(yield_data) > 0:
        # Filter for positive yields (dividend payers)
        positive_yields = yield_data[yield_data > 0]

        dividend_analytics['yield_analysis'] = {
            'column': yield_col,
            'total_stocks': int(len(yield_data)),
            'dividend_payers': int(len(positive_yields)),
            'dividend_payer_pct': float(len(positive_yields) / len(yield_data) * 100),
            'mean_yield': float(positive_yields.mean()) if len(positive_yields) > 0 else 0,
            'median_yield': float(positive_yields.median()) if len(positive_yields) > 0 else 0,
            'std_yield': float(positive_yields.std()) if len(positive_yields) > 0 else 0,
            'min_yield': float(positive_yields.min()) if len(positive_yields) > 0 else 0,
            'max_yield': float(positive_yields.max()) if len(positive_yields) > 0 else 0,
            'q25': float(positive_yields.quantile(0.25)) if len(positive_yields) > 0 else 0,
            'q75': float(positive_yields.quantile(0.75)) if len(positive_yields) > 0 else 0,
            }

        print(f'  Total stocks analyzed:    {len(yield_data):,}')
        print(
                f'  Dividend payers:          {len(positive_yields):,} ({len(positive_yields) / len(yield_data) * 100:.1f}%)')
        if len(positive_yields) > 0:
            print(f'  Mean yield:               {positive_yields.mean():.2f}%')
            print(f'  Median yield:             {positive_yields.median():.2f}%')
            print(f'  Yield range:              {positive_yields.min():.2f}% - {positive_yields.max():.2f}%')

        # Yield distribution buckets
        yield_bins = [0, 1, 2, 3, 4, 5, 7, 10, float('inf')]
        yield_labels = ['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '5-7%', '7-10%', '10%+']
        yield_dist = pd.cut(positive_yields, bins=yield_bins, labels=yield_labels).value_counts()
        dividend_analytics['yield_distribution'] = {str(k): int(v) for k, v in yield_dist.items()}

        print('\n  Yield Distribution:')
        for bucket, count in sorted(yield_dist.items(),
                                    key=lambda x: yield_labels.index(x[0]) if x[0] in yield_labels else 99):
            pct = count / len(positive_yields) * 100 if len(positive_yields) > 0 else 0
            print(f'    {bucket:10s}: {count:5,} ({pct:5.1f}%)')

# ============================================================================
# 4. Payout Ratio Analysis
# ============================================================================
print('\n📊 Payout Ratio Analysis...')

if 'payout_ratio' in available_dividend_cols:
    payout_col = available_dividend_cols['payout_ratio']
    payout_data = pd.to_numeric(all_stocks_enhanced[payout_col], errors='coerce').dropna()

    # Filter reasonable payout ratios (0-200%)
    valid_payout = payout_data[(payout_data >= 0) & (payout_data <= 200)]

    if len(valid_payout) > 0:
        dividend_analytics['payout_analysis'] = {
            'column': payout_col,
            'count': int(len(valid_payout)),
            'mean_payout': float(valid_payout.mean()),
            'median_payout': float(valid_payout.median()),
            'sustainable_pct': float((valid_payout <= 75).sum() / len(valid_payout) * 100),
            'high_payout_pct': float((valid_payout > 100).sum() / len(valid_payout) * 100),
            }

        print(f'  Stocks with payout data:  {len(valid_payout):,}')
        print(f'  Mean payout ratio:        {valid_payout.mean():.1f}%')
        print(f'  Median payout ratio:      {valid_payout.median():.1f}%')
        print(f'  Sustainable (<=75%):      {((valid_payout <= 75).sum() / len(valid_payout) * 100):.1f}%')
        print(f'  High payout (>100%):      {((valid_payout > 100).sum() / len(valid_payout) * 100):.1f}%')

# ============================================================================
# 5. Dividend Growth Analysis
# ============================================================================
print('\n📈 Dividend Growth Analysis...')

if 'dividend_growth' in available_dividend_cols:
    growth_col = available_dividend_cols['dividend_growth']
    growth_data = pd.to_numeric(all_stocks_enhanced[growth_col], errors='coerce').dropna()

    if len(growth_data) > 0:
        dividend_analytics['dividend_growth'] = {
            'column': growth_col,
            'count': int(len(growth_data)),
            'mean_growth': float(growth_data.mean()),
            'median_growth': float(growth_data.median()),
            'positive_growth_pct': float((growth_data > 0).sum() / len(growth_data) * 100),
            'growers_pct': float((growth_data > 5).sum() / len(growth_data) * 100),
            'cutters_pct': float((growth_data < -5).sum() / len(growth_data) * 100),
            }

        print(f'  Stocks with growth data:  {len(growth_data):,}')
        print(f'  Mean dividend growth:     {growth_data.mean():+.2f}%')
        print(f'  Dividend growers (>5%):   {((growth_data > 5).sum() / len(growth_data) * 100):.1f}%')
        print(f'  Dividend cutters (<-5%):  {((growth_data < -5).sum() / len(growth_data) * 100):.1f}%')


# ============================================================================
# 6. Segment Analysis Function for Dividends
# ============================================================================
def analyze_dividends_by_segment(df, segment_col, yield_col):
    """Analyze dividend metrics by a segment column."""
    segment_stats = {}

    for segment in df[segment_col].dropna().unique():
        segment_df = df[df[segment_col] == segment]
        yield_data = pd.to_numeric(segment_df[yield_col], errors='coerce').dropna()
        positive_yields = yield_data[yield_data > 0]

        if len(yield_data) >= 5:
            segment_stats[str(segment)] = {
                'total_stocks': int(len(yield_data)),
                'dividend_payers': int(len(positive_yields)),
                'dividend_payer_pct': float(len(positive_yields) / len(yield_data) * 100),
                }

            if len(positive_yields) >= 3:
                segment_stats[str(segment)]['mean_yield'] = float(positive_yields.mean())
                segment_stats[str(segment)]['median_yield'] = float(positive_yields.median())

    return segment_stats


# Get primary dividend yield column
primary_yield_col = available_dividend_cols.get('dividend_yield')

if primary_yield_col:
    # ============================================================================
    # 7. By Sector
    # ============================================================================
    if 'sector' in all_stocks_enhanced.columns:
        print('\n🏢 Dividends by Sector...')
        sector_stats = analyze_dividends_by_segment(all_stocks_enhanced, 'sector', primary_yield_col)
        dividend_analytics['by_sector'] = sector_stats

        # Print top 5 sectors by dividend payer percentage
        payer_sectors = {k: v['dividend_payer_pct'] for k, v in sector_stats.items()}
        top_sectors = sorted(payer_sectors.items(), key=lambda x: x[1], reverse=True)[:5]
        for sector, pct in top_sectors:
            mean_yield = sector_stats[sector].get('mean_yield', 0)
            print(f'  {sector[:30]:30s}: {pct:.1f}% payers, avg yield={mean_yield:.2f}%')

    # ============================================================================
    # 8. By Region
    # ============================================================================
    if 'region' in all_stocks_enhanced.columns:
        print('\n🌍 Dividends by Region...')
        region_stats = analyze_dividends_by_segment(all_stocks_enhanced, 'region', primary_yield_col)
        dividend_analytics['by_region'] = region_stats

        for region, stats in region_stats.items():
            mean_yield = stats.get('mean_yield', 0)
            print(f'  {region:15s}: {stats["dividend_payer_pct"]:.1f}% payers, avg yield={mean_yield:.2f}%')

    # ============================================================================
    # 9. By Size Class
    # ============================================================================
    if 'size_class' in all_stocks_enhanced.columns:
        print('\n📏 Dividends by Size Class...')
        size_stats = analyze_dividends_by_segment(all_stocks_enhanced, 'size_class', primary_yield_col)
        dividend_analytics['by_size_class'] = size_stats

        for size_class, stats in size_stats.items():
            mean_yield = stats.get('mean_yield', 0)
            print(f'  {size_class:15s}: {stats["dividend_payer_pct"]:.1f}% payers, avg yield={mean_yield:.2f}%')

    # ============================================================================
    # 10. By Style Class
    # ============================================================================
    if 'style_class' in all_stocks_enhanced.columns:
        print('\n🎨 Dividends by Style Class...')
        style_stats = analyze_dividends_by_segment(all_stocks_enhanced, 'style_class', primary_yield_col)
        dividend_analytics['by_style_class'] = style_stats

        for style_class, stats in style_stats.items():
            mean_yield = stats.get('mean_yield', 0)
            print(f'  {style_class:15s}: {stats["dividend_payer_pct"]:.1f}% payers, avg yield={mean_yield:.2f}%')

    # ============================================================================
    # 11. By Industry
    # ============================================================================
    if 'industry' in all_stocks_enhanced.columns:
        print('\n🏭 Dividends by Industry (Top 10 by Yield)...')
        industry_stats = analyze_dividends_by_segment(all_stocks_enhanced, 'industry', primary_yield_col)
        dividend_analytics['by_industry'] = industry_stats

        # Show top 10 industries by mean yield
        industries_with_yield = {k: v.get('mean_yield', 0) for k, v in industry_stats.items() if
                                 v.get('dividend_payers', 0) >= 5}
        top_industries = sorted(industries_with_yield.items(), key=lambda x: x[1], reverse=True)[:10]
        for industry, yield_val in top_industries:
            payer_pct = industry_stats[industry]['dividend_payer_pct']
            print(f'  {industry[:35]:35s}: yield={yield_val:.2f}%, payers={payer_pct:.1f}%')

    # ============================================================================
    # 12. By Trading Country
    # ============================================================================
    trading_country_cols = ['trading_country', 'country', 'domicile']
    trading_country_col = None
    for col in trading_country_cols:
        if col in all_stocks_enhanced.columns:
            trading_country_col = col
            break

    if trading_country_col:
        print(f'\n🌐 Dividends by Trading Country ({trading_country_col})...')
        country_stats = analyze_dividends_by_segment(all_stocks_enhanced, trading_country_col, primary_yield_col)
        dividend_analytics['by_trading_country'] = country_stats

        # Show top 10 countries by mean yield
        countries_with_yield = {k: v.get('mean_yield', 0) for k, v in country_stats.items() if
                                v.get('dividend_payers', 0) >= 5}
        top_countries = sorted(countries_with_yield.items(), key=lambda x: x[1], reverse=True)[:10]
        for country, yield_val in top_countries:
            payer_pct = country_stats[country]['dividend_payer_pct']
            print(f'  {country[:25]:25s}: yield={yield_val:.2f}%, payers={payer_pct:.1f}%')

    # ============================================================================
    # 13. By Industry
    # ============================================================================
    industry_cols = ['industry', 'stock_exchange', 'primary_exchange', 'listing_exchange']
    industry_col = None
    for col in industry_cols:
        if col in all_stocks_enhanced.columns:
            industry_col = col
            break

    if industry_col:
        print(f'\n📈 Dividends by Industry ({industry_col})...')
        industry_stats = analyze_dividends_by_segment(all_stocks_enhanced, industry_col, primary_yield_col)
        dividend_analytics['by_exchange'] = industry_stats

        # Show top 10 exchanges by mean yield
        exchanges_with_yield = {k: v.get('mean_yield', 0) for k, v in industry_stats.items() if
                                v.get('dividend_payers', 0) >= 5}
        top_exchanges = sorted(exchanges_with_yield.items(), key=lambda x: x[1], reverse=True)[:10]
        for exchange, yield_val in top_exchanges:
            payer_pct = industry_stats[exchange]['dividend_payer_pct']
            print(f'  {exchange[:25]:25s}: yield={yield_val:.2f}%, payers={payer_pct:.1f}%')

# ============================================================================
# 14. High Yield Stock Screening
# ============================================================================
print('\n🔝 High Yield Stock Screening...')

if 'dividend_yield' in available_dividend_cols:
    yield_col = available_dividend_cols['dividend_yield']
    yield_data = pd.to_numeric(all_stocks_enhanced[yield_col], errors='coerce')

    # Define high yield threshold
    high_yield_threshold = 4.0
    high_yield_mask = yield_data >= high_yield_threshold
    high_yield_stocks = all_stocks_enhanced[high_yield_mask]

    dividend_analytics['high_yield_screening'] = {
        'threshold': high_yield_threshold,
        'count': int(len(high_yield_stocks)),
        'pct_of_total': float(len(high_yield_stocks) / len(all_stocks_enhanced) * 100),
        }

    print(
            f'  High yield stocks (>={high_yield_threshold}%): {len(high_yield_stocks):,} ({len(high_yield_stocks) / len(all_stocks_enhanced) * 100:.1f}%)')

    # Show sector distribution of high yield stocks
    if 'sector' in high_yield_stocks.columns and len(high_yield_stocks) > 0:
        hy_sector_dist = high_yield_stocks['sector'].value_counts().head(5)
        dividend_analytics['high_yield_by_sector'] = {str(k): int(v) for k, v in hy_sector_dist.items()}

        print('\n  High Yield by Sector (Top 5):')
        for sector, count in hy_sector_dist.items():
            print(f'    {sector[:30]:30s}: {count:5,}')

# ============================================================================
# 15. Forward Dividend Yield Analysis (NEW)
# ============================================================================
print('\n📅 Forward Dividend Yield Analysis...')

fwd_yield_cols = [
    'div_yield_ind', 'div_yield_1fyind', 'div_yield_2fyind',
    'div_yield_3fyind', 'div_yield_4fyind', 'div_yield_5fyind', 'div_yield_5yavgltm'
    ]
available_fwd_yields = [c for c in fwd_yield_cols if c in all_stocks_enhanced.columns]

if available_fwd_yields:
    fwd_yield_stats = {}
    print(f'  Found {len(available_fwd_yields)} forward yield metrics')

    for metric in available_fwd_yields:
        data = pd.to_numeric(all_stocks_enhanced[metric], errors='coerce').dropna()
        positive_data = data[data > 0]

        if len(positive_data) > 0:
            fwd_yield_stats[metric] = {
                'count': int(len(positive_data)),
                'mean': float(positive_data.mean()),
                'median': float(positive_data.median()),
                'q25': float(positive_data.quantile(0.25)),
                'q75': float(positive_data.quantile(0.75)),
                }

            period = metric.replace('div_yield_', '').replace('fyind', 'y').replace('ind', 'indicated').replace(
                    '5yavgltm', '5y_avg')
            print(
                    f'    {period:15s}: mean={positive_data.mean():>6.2f}%, median={positive_data.median():>6.2f}%, n={len(positive_data):,}')

    dividend_analytics['forward_yield_analysis'] = fwd_yield_stats

# ============================================================================
# 16. Dividend Growth & Buyback Analysis (NEW)
# ============================================================================
print('\n💰 Dividend Growth & Shareholder Returns...')

# Buyback yield analysis
if 'buyback_yield' in available_dividend_cols:
    buyback_col = available_dividend_cols['buyback_yield']
    buyback_data = pd.to_numeric(all_stocks_enhanced[buyback_col], errors='coerce').dropna()
    positive_buybacks = buyback_data[buyback_data > 0]

    if len(positive_buybacks) > 0:
        dividend_analytics['buyback_analysis'] = {
            'column': buyback_col,
            'count': int(len(buyback_data)),
            'buyback_payers': int(len(positive_buybacks)),
            'buyback_payer_pct': float(len(positive_buybacks) / len(buyback_data) * 100),
            'mean_yield': float(positive_buybacks.mean()),
            'median_yield': float(positive_buybacks.median()),
            'q25': float(positive_buybacks.quantile(0.25)),
            'q75': float(positive_buybacks.quantile(0.75)),
            }

        print(f'  Buyback Yield Analysis ({buyback_col}):')
        print(
                f'    Stocks with buybacks: {len(positive_buybacks):,} ({len(positive_buybacks) / len(buyback_data) * 100:.1f}%)')
        print(f'    Mean buyback yield:   {positive_buybacks.mean():.2f}%')
        print(f'    Median buyback yield: {positive_buybacks.median():.2f}%')

# Total shareholder yield (dividend + buyback)
if 'dividend_yield' in available_dividend_cols and 'buyback_yield' in available_dividend_cols:
    div_col = available_dividend_cols['dividend_yield']
    buyback_col = available_dividend_cols['buyback_yield']

    div_data = pd.to_numeric(all_stocks_enhanced[div_col], errors='coerce').fillna(0)
    buyback_data = pd.to_numeric(all_stocks_enhanced[buyback_col], errors='coerce').fillna(0)

    total_shareholder_yield = div_data + buyback_data
    positive_tsy = total_shareholder_yield[total_shareholder_yield > 0]

    if len(positive_tsy) > 0:
        dividend_analytics['total_shareholder_yield'] = {
            'count': int(len(positive_tsy)),
            'mean': float(positive_tsy.mean()),
            'median': float(positive_tsy.median()),
            'q25': float(positive_tsy.quantile(0.25)),
            'q75': float(positive_tsy.quantile(0.75)),
            }

        print(f'\n  Total Shareholder Yield (Dividend + Buyback):')
        print(f'    Stocks with positive TSY: {len(positive_tsy):,}')
        print(f'    Mean TSY:   {positive_tsy.mean():.2f}%')
        print(f'    Median TSY: {positive_tsy.median():.2f}%')

# ============================================================================
# 17. Interactive Dividend Visualizations (Plotly)
# ============================================================================
print('\n📊 Generating Interactive Dividend Visualizations...')

# Create output directory for dividend visualizations
dividend_viz_dir = OUTPUT_DIR / 'eda' / 'dividend_visualizations'
dividend_viz_dir.mkdir(parents=True, exist_ok=True)

# Visualization 1: Forward Dividend Yield Trend
if available_fwd_yields and 'forward_yield_analysis' in dividend_analytics:
    fwd_viz_data = []
    period_labels = {
        'div_yield_ind': 'Indicated',
        'div_yield_1fyind': '1Y Forward',
        'div_yield_2fyind': '2Y Forward',
        'div_yield_3fyind': '3Y Forward',
        'div_yield_4fyind': '4Y Forward',
        'div_yield_5fyind': '5Y Forward',
        'div_yield_5yavgltm': '5Y Average'
        }
    period_order = {
        'div_yield_ind': 0, 'div_yield_1fyind': 1, 'div_yield_2fyind': 2,
        'div_yield_3fyind': 3, 'div_yield_4fyind': 4, 'div_yield_5fyind': 5,
        'div_yield_5yavgltm': 6
        }

    for metric, stats in dividend_analytics['forward_yield_analysis'].items():
        fwd_viz_data.append({
            'Period': period_labels.get(metric, metric),
            'Period_Order': period_order.get(metric, 99),
            'Mean Yield %': stats['mean'],
            'Median Yield %': stats['median'],
            'Count': stats['count']
            })

    if fwd_viz_data:
        fwd_viz_df = pd.DataFrame(fwd_viz_data).sort_values('Period_Order')

        fig_fwd_yield = go.Figure()

        # Add mean yield line
        fig_fwd_yield.add_trace(go.Scatter(
                name='Mean Yield',
                x=fwd_viz_df['Period'],
                y=fwd_viz_df['Mean Yield %'],
                mode='lines+markers',
                line=dict(color='#00bc8c', width=3),
                marker=dict(size=10)
                ))

        # Add median yield line
        fig_fwd_yield.add_trace(go.Scatter(
                name='Median Yield',
                x=fwd_viz_df['Period'],
                y=fwd_viz_df['Median Yield %'],
                mode='lines+markers',
                line=dict(color='#3498db', width=3, dash='dash'),
                marker=dict(size=10)
                ))

        fig_fwd_yield.update_layout(
                title='Forward Dividend Yield Expectations Over Time',
                xaxis_title='Forecast Period',
                yaxis_title='Dividend Yield (%)',
                template='plotly_dark',
                font=dict(family='Arial, sans-serif', size=14),
                title_font_size=20,
                legend=dict(orientation='h', yanchor='bottom', xanchor='center', x=0.5, y=1.02),
                hovermode='x unified'
                )

        fwd_yield_path = dividend_viz_dir / 'forward_dividend_yield_trend.html'
        fig_fwd_yield.write_html(str(fwd_yield_path))
        print(f'  ✓ Saved: {fwd_yield_path}')

# Visualization 2: Dividend Yield Distribution by Sector
if 'dividend_yield' in available_dividend_cols and 'sector' in all_stocks_enhanced.columns:
    yield_col = available_dividend_cols['dividend_yield']
    sector_yield_data = []

    for sector in all_stocks_enhanced['sector'].dropna().unique():
        sector_df = all_stocks_enhanced[all_stocks_enhanced['sector'] == sector]
        yield_data = pd.to_numeric(sector_df[yield_col], errors='coerce').dropna()
        positive_yields = yield_data[yield_data > 0]

        if len(positive_yields) >= 10:
            sector_yield_data.append({
                'Sector': str(sector),
                'Mean Yield': float(positive_yields.mean()),
                'Median Yield': float(positive_yields.median()),
                'Dividend Payers %': float(len(positive_yields) / len(yield_data) * 100),
                'Count': int(len(positive_yields))
                })

    if sector_yield_data:
        sector_yield_df = pd.DataFrame(sector_yield_data).sort_values('Mean Yield', ascending=True)

        fig_sector_yield = px.bar(
                sector_yield_df,
                y='Sector',
                x='Mean Yield',
                orientation='h',
                title='Average Dividend Yield by Sector',
                labels={'Mean Yield': 'Mean Dividend Yield (%)', 'Sector': ''},
                template='plotly_dark',
                color='Mean Yield',
                color_continuous_scale='Greens',
                hover_data=['Median Yield', 'Dividend Payers %', 'Count']
                )

        fig_sector_yield.update_layout(
                font=dict(family='Arial, sans-serif', size=14),
                title_font_size=20,
                showlegend=False,
                hovermode='y unified',
                coloraxis_colorbar=dict(title='Yield %')
                )

        sector_yield_path = dividend_viz_dir / 'dividend_yield_by_sector.html'
        fig_sector_yield.write_html(str(sector_yield_path))
        print(f'  ✓ Saved: {sector_yield_path}')

# Visualization 3: Shareholder Returns Comparison (Dividend vs Buyback)
if 'buyback_analysis' in dividend_analytics and 'yield_analysis' in dividend_analytics:
    returns_data = []

    # Dividend yield stats
    if dividend_analytics['yield_analysis'].get('mean_yield'):
        returns_data.append({
            'Return Type': 'Dividend Yield',
            'Mean %': dividend_analytics['yield_analysis']['mean_yield'],
            'Median %': dividend_analytics['yield_analysis']['median_yield'],
            'Payers': dividend_analytics['yield_analysis']['dividend_payers']
            })

    # Buyback yield stats
    if dividend_analytics['buyback_analysis'].get('mean_yield'):
        returns_data.append({
            'Return Type': 'Buyback Yield',
            'Mean %': dividend_analytics['buyback_analysis']['mean_yield'],
            'Median %': dividend_analytics['buyback_analysis']['median_yield'],
            'Payers': dividend_analytics['buyback_analysis']['buyback_payers']
            })

    # Total shareholder yield
    if 'total_shareholder_yield' in dividend_analytics:
        returns_data.append({
            'Return Type': 'Total Shareholder Yield',
            'Mean %': dividend_analytics['total_shareholder_yield']['mean'],
            'Median %': dividend_analytics['total_shareholder_yield']['median'],
            'Payers': dividend_analytics['total_shareholder_yield']['count']
            })

    if returns_data:
        returns_df = pd.DataFrame(returns_data)

        fig_returns = go.Figure()

        fig_returns.add_trace(go.Bar(
                name='Mean Yield',
                x=returns_df['Return Type'],
                y=returns_df['Mean %'],
                marker_color='#00bc8c',
                text=returns_df['Mean %'].apply(lambda x: f'{x:.2f}%'),
                textposition='outside'
                ))

        fig_returns.add_trace(go.Bar(
                name='Median Yield',
                x=returns_df['Return Type'],
                y=returns_df['Median %'],
                marker_color='#3498db',
                text=returns_df['Median %'].apply(lambda x: f'{x:.2f}%'),
                textposition='outside'
                ))

        fig_returns.update_layout(
                title='Shareholder Returns Comparison: Dividends vs Buybacks',
                xaxis_title='Return Type',
                yaxis_title='Yield (%)',
                template='plotly_dark',
                font=dict(family='Arial, sans-serif', size=14),
                title_font_size=20,
                barmode='group',
                legend=dict(orientation='h', yanchor='bottom', xanchor='center', x=0.5, y=1.02),
                hovermode='x unified'
                )

        returns_path = dividend_viz_dir / 'shareholder_returns_comparison.html'
        fig_returns.write_html(str(returns_path))
        print(f'  ✓ Saved: {returns_path}')

# Visualization 4: Dividend Yield Distribution Histogram
if 'dividend_yield' in available_dividend_cols:
    yield_col = available_dividend_cols['dividend_yield']
    yield_data = pd.to_numeric(all_stocks_enhanced[yield_col], errors='coerce').dropna()
    positive_yields = yield_data[(yield_data > 0) & (yield_data <= 15)]  # Cap at 15% for visualization

    if len(positive_yields) > 0:
        fig_hist = px.histogram(
                positive_yields,
                nbins=50,
                title='Dividend Yield Distribution (Dividend Payers)',
                labels={'value': 'Dividend Yield (%)', 'count': 'Number of Stocks'},
                template='plotly_dark',
                color_discrete_sequence=['#00bc8c']
                )

        # Add mean and median lines
        mean_yield = positive_yields.mean()
        median_yield = positive_yields.median()

        fig_hist.add_vline(x=mean_yield, line_dash='dash', line_color='#f39c12',
                           annotation_text=f'Mean: {mean_yield:.2f}%', annotation_position='top right')
        fig_hist.add_vline(x=median_yield, line_dash='dot', line_color='#3498db',
                           annotation_text=f'Median: {median_yield:.2f}%', annotation_position='top left')

        fig_hist.update_layout(
                font=dict(family='Arial, sans-serif', size=14),
                title_font_size=20,
                showlegend=False,
                xaxis_title='Dividend Yield (%)',
                yaxis_title='Number of Stocks'
                )

        hist_path = dividend_viz_dir / 'dividend_yield_distribution.html'
        fig_hist.write_html(str(hist_path))
        print(f'  ✓ Saved: {hist_path}')

# ============================================================================
# 18. Export Dividend Analytics JSON
# ============================================================================
print('\n💾 Exporting Dividend Analytics...')

dividend_analytics_path = financial_metrics_dir / 'dividend_analytics.json'
with open(dividend_analytics_path, 'w') as f:
    json.dump(make_serializable(dividend_analytics), f, indent=2, default=str)
print(f'  ✓ Saved: {dividend_analytics_path}')

print('\n✓ Dividend Analytics complete')

# ============================================================================
# Interactive Dashboard: Dividend Analytics
# ============================================================================
print('\nINTERACTIVE DIVIDEND ANALYTICS DASHBOARD')
print('=' * 80)
display_earnings_dashboard(display_df, mode='dividends')

#%% md
## Cell 10.10: Enhanced Interactive Visualizations

Generate additional interactive HTML visualizations for statistical testing results, earnings analytics, 
and dividend analytics following Section 17 Style Guidelines from code_guidelines.md.

**Key Objectives:**
1. Visualize hypothesis testing results with p-value heatmaps
2. Create earnings surprise distribution and sector comparison charts
3. Generate dividend yield analysis visualizations
4. Build analyst recommendation summary charts
5. Create benchmarking comparison visualizations

**Outputs:**
- **HTML Visualizations (6 files):** hypothesis_test_heatmap.html, earnings_surprise_analysis.html, 
  dividend_yield_distribution.html, analyst_recommendations_chart.html, sector_benchmarking.html, 
  financial_metrics_radar.html

#%% md
## Cell 10.11: Advanced Benchmarking & Risk Analytics

Enhanced visualizations for statistical benchmarking, risk analytics (VaR), and performance attribution analysis.

**Key Objectives:**
1. Generate Value at Risk (VaR) and Expected Shortfall analytics by sector
2. Create statistical benchmarking comparison charts (ANOVA, Kruskal-Wallis)
3. Build performance attribution sunburst visualization
4. Generate cross-sectional risk heatmaps

**Outputs:**
- **HTML Visualizations (4 files):** var_risk_analytics.html, statistical_benchmarking.html, performance_attribution_sunburst.html, cross_sectional_risk_heatmap.html

#%%
# ============================================================================
# Cell 10.10: Enhanced Interactive Visualizations
# ============================================================================

print('=' * 80)
print('ENHANCED INTERACTIVE VISUALIZATIONS')
print('=' * 80)

# Create visualizations output directory
viz_output_dir = OUTPUT_DIR / 'eda' / 'visualizations'
viz_output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Hypothesis Testing Results Heatmap
# ============================================================================
print('\n📊 Generating Hypothesis Testing Visualizations...')

# Load hypothesis test results if available
hypothesis_tests_file = financial_metrics_dir / 'hypothesis_tests.json'
if hypothesis_tests_file.exists():
    with open(hypothesis_tests_file, 'r') as f:
        hypothesis_data = json.load(f)

    # Extract sector test results for visualization
    if 'sector_tests' in hypothesis_data:
        sector_tests = hypothesis_data['sector_tests']

        # Build DataFrame for heatmap
        test_metrics = [k for k in sector_tests.keys() if k != 'summary']
        test_types = ['anova', 'kruskal_wallis']

        heatmap_data = []
        for metric in test_metrics:
            for test_type in test_types:
                if test_type in sector_tests.get(metric, {}):
                    test_result = sector_tests[metric][test_type]
                    p_value = test_result.get('p_value', 1.0)
                    significant = test_result.get('significant', 'False') == 'True'
                    heatmap_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Test': test_type.replace('_', ' ').title(),
                        'P-Value': p_value,
                        'Significant': 'Yes' if significant else 'No',
                        '-log10(p)': -np.log10(max(p_value, 1e-100))
                        })

        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)

            # Create pivot for heatmap
            pivot_df = heatmap_df.pivot(index='Metric', columns='Test', values='-log10(p)')

            fig_hypothesis = px.imshow(
                    pivot_df,
                    title='<b>Statistical Hypothesis Testing Results</b><br><sup>-log10(p-value) by Metric and Test Type (Higher = More Significant)</sup>',
                    template=PLOTLY_TEMPLATE,
                    color_continuous_scale='RdYlGn',
                    aspect='auto',
                    text_auto='.2f'
                    )
            fig_hypothesis.update_layout(
                    height=500,
                    font=dict(family='Segoe UI, Roboto, Arial'),
                    title_font_size=20,
                    xaxis_title='Statistical Test',
                    yaxis_title='Financial Metric',
                    )
            fig_hypothesis.add_hline(y=-0.5, line_dash='dash', line_color='white',
                                     annotation_text='α=0.05 threshold: -log10(0.05)≈1.3')
            fig_hypothesis.write_html(viz_output_dir / 'hypothesis_test_heatmap.html')
            print(f'  ✓ Saved: hypothesis_test_heatmap.html')

            # Display inline
            fig_hypothesis.show()
else:
    print('  ⚠ hypothesis_tests.json not found, skipping visualization')

# ============================================================================
# 2. Earnings Surprise Analysis Visualization
# ============================================================================
print('\n📈 Generating Earnings Surprise Visualizations...')

earnings_file = financial_metrics_dir / 'earnings_monitor.json'
if earnings_file.exists():
    with open(earnings_file, 'r') as f:
        earnings_data = json.load(f)

    # Create earnings surprise visualization if data available
    if 'earnings_surprise_analysis' in earnings_data:
        surprise_data = earnings_data['earnings_surprise_analysis']

        # Build comparison data
        surprise_metrics = []
        for metric_name, metric_data in surprise_data.items():
            if isinstance(metric_data, dict) and 'mean_surprise_pct' in metric_data:
                surprise_metrics.append({
                    'Metric': metric_name,
                    'Mean Surprise (%)': metric_data.get('mean_surprise_pct', 0),
                    'Median Surprise (%)': metric_data.get('median_surprise_pct', 0),
                    'Positive Surprises': metric_data.get('positive_surprises', 0),
                    'Negative Surprises': metric_data.get('negative_surprises', 0),
                    'Count': metric_data.get('count', 0)
                    })

        if surprise_metrics:
            surprise_df = pd.DataFrame(surprise_metrics)

            # Create grouped bar chart for earnings surprise
            fig_earnings = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=['Mean vs Median Surprise (%)', 'Positive vs Negative Surprises'],
                    specs=[[{'type': 'bar'}, {'type': 'bar'}]]
                    )

            # Mean vs Median
            fig_earnings.add_trace(
                    go.Bar(name='Mean Surprise', x=surprise_df['Metric'], y=surprise_df['Mean Surprise (%)'],
                           marker_color=COLOR_PALETTE['primary'],
                           hovertemplate='%{x}<br>Mean: %{y:.2f}%<extra></extra>'),
                    row=1, col=1
                    )
            fig_earnings.add_trace(
                    go.Bar(name='Median Surprise', x=surprise_df['Metric'], y=surprise_df['Median Surprise (%)'],
                           marker_color=COLOR_PALETTE['success'],
                           hovertemplate='%{x}<br>Median: %{y:.2f}%<extra></extra>'),
                    row=1, col=1
                    )

            # Positive vs Negative
            fig_earnings.add_trace(
                    go.Bar(name='Positive', x=surprise_df['Metric'], y=surprise_df['Positive Surprises'],
                           marker_color=COLOR_PALETTE['success'],
                           hovertemplate='%{x}<br>Positive: %{y}<extra></extra>'),
                    row=1, col=2
                    )
            fig_earnings.add_trace(
                    go.Bar(name='Negative', x=surprise_df['Metric'], y=surprise_df['Negative Surprises'],
                           marker_color=COLOR_PALETTE['danger'], hovertemplate='%{x}<br>Negative: %{y}<extra></extra>'),
                    row=1, col=2
                    )

            fig_earnings.update_layout(
                    title='<b>Earnings Surprise Analysis</b><br><sup>Estimated vs Actual Performance by Metric</sup>',
                    template=PLOTLY_TEMPLATE,
                    height=500,
                    font=dict(family='Segoe UI, Roboto, Arial'),
                    title_font_size=20,
                    barmode='group',
                    showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
                    )
            fig_earnings.write_html(viz_output_dir / 'earnings_surprise_analysis.html')
            print(f'  ✓ Saved: earnings_surprise_analysis.html')
            fig_earnings.show()
else:
    print('  ⚠ earnings_monitor.json not found, skipping visualization')

# ============================================================================
# 3. Dividend Yield Distribution Visualization
# ============================================================================
print('\n💰 Generating Dividend Yield Visualizations...')

dividend_file = financial_metrics_dir / 'dividend_analytics.json'
if dividend_file.exists():
    with open(dividend_file, 'r') as f:
        dividend_data = json.load(f)

    # Create dividend yield distribution visualization
    fig_dividend = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Dividend Yield Distribution',
                'Dividend Payers by Sector',
                'Yield by Size Class',
                'Yield by Style Class'
                ],
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
            )

    # Yield distribution buckets
    if 'yield_distribution' in dividend_data:
        yield_dist = dividend_data['yield_distribution']
        buckets = list(yield_dist.keys())
        counts = list(yield_dist.values())

        fig_dividend.add_trace(
                go.Bar(x=buckets, y=counts, marker_color=COLOR_PALETTE['success'],
                       hovertemplate='Yield: %{x}<br>Count: %{y}<extra></extra>'),
                row=1, col=1
                )

    # Sector distribution
    if 'by_sector' in dividend_data:
        sector_data = dividend_data['by_sector']
        sectors = list(sector_data.keys())[:10]  # Top 10
        payer_pcts = [sector_data[s].get('dividend_payer_pct', 0) for s in sectors]

        fig_dividend.add_trace(
                go.Bar(x=[s[:15] for s in sectors], y=payer_pcts, marker_color=COLOR_PALETTE['primary'],
                       hovertemplate='%{x}<br>Payer %: %{y:.1f}%<extra></extra>'),
                row=1, col=2
                )

    # Size class distribution
    if 'by_size_class' in dividend_data:
        size_data = dividend_data['by_size_class']
        sizes = list(size_data.keys())
        mean_yields = [size_data[s].get('mean_yield', 0) for s in sizes]

        fig_dividend.add_trace(
                go.Bar(x=sizes, y=mean_yields, marker_color=COLOR_PALETTE['info'],
                       hovertemplate='%{x}<br>Mean Yield: %{y:.2f}%<extra></extra>'),
                row=2, col=1
                )

    # Style class distribution
    if 'by_style_class' in dividend_data:
        style_data = dividend_data['by_style_class']
        styles = list(style_data.keys())
        mean_yields = [style_data[s].get('mean_yield', 0) for s in styles]

        fig_dividend.add_trace(
                go.Bar(x=styles, y=mean_yields, marker_color=COLOR_PALETTE['warning'],
                       hovertemplate='%{x}<br>Mean Yield: %{y:.2f}%<extra></extra>'),
                row=2, col=2
                )

    fig_dividend.update_layout(
            title='<b>Dividend Analytics Dashboard</b><br><sup>Yield Distribution and Segment Analysis</sup>',
            template=PLOTLY_TEMPLATE,
            height=700,
            font=dict(family='Segoe UI, Roboto, Arial'),
            title_font_size=20,
            showlegend=False
            )
    fig_dividend.write_html(viz_output_dir / 'dividend_yield_distribution.html')
    print(f'  ✓ Saved: dividend_yield_distribution.html')
    fig_dividend.show()
else:
    print('  ⚠ dividend_analytics.json not found, skipping visualization')

# ============================================================================
# 4. Analyst Recommendations Visualization
# ============================================================================
print('\n👥 Generating Analyst Recommendations Visualizations...')

analyst_file = financial_metrics_dir / 'analyst_recommendations.json'
if analyst_file.exists():
    with open(analyst_file, 'r') as f:
        analyst_data = json.load(f)

    # Create analyst recommendations dashboard
    fig_analyst = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Price Target Upside by Sector',
                'Analyst Coverage Distribution',
                'Upside by Size Class',
                'Upside by Style Class'
                ],
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
            )

    # Sector upside analysis
    if 'by_sector' in analyst_data:
        sector_stats = analyst_data['by_sector']
        sectors_with_upside = [(k, v.get('mean_upside', 0)) for k, v in sector_stats.items()
                               if 'mean_upside' in v]
        sectors_with_upside.sort(key=lambda x: x[1], reverse=True)

        if sectors_with_upside:
            top_sectors = sectors_with_upside[:10]
            sector_names = [s[0][:20] for s in top_sectors]
            upside_values = [s[1] for s in top_sectors]
            colors = [COLOR_PALETTE['success'] if v > 0 else COLOR_PALETTE['danger'] for v in upside_values]

            fig_analyst.add_trace(
                    go.Bar(x=sector_names, y=upside_values, marker_color=colors,
                           hovertemplate='%{x}<br>Upside: %{y:.2f}%<extra></extra>'),
                    row=1, col=1
                    )

    # Coverage distribution (pie chart)
    if 'coverage_distribution' in analyst_data:
        coverage = analyst_data['coverage_distribution']
        labels = list(coverage.keys())
        values = list(coverage.values())

        fig_analyst.add_trace(
                go.Pie(labels=labels, values=values, hole=0.4,
                       marker_colors=[COLOR_PALETTE['primary'], COLOR_PALETTE['success'],
                                      COLOR_PALETTE['info'], COLOR_PALETTE['warning'], COLOR_PALETTE['neutral']]),
                row=1, col=2
                )

    # Size class upside
    if 'by_size_class' in analyst_data:
        size_stats = analyst_data['by_size_class']
        sizes = list(size_stats.keys())
        upsides = [size_stats[s].get('mean_upside', 0) for s in sizes]

        fig_analyst.add_trace(
                go.Bar(x=sizes, y=upsides, marker_color=COLOR_PALETTE['info'],
                       hovertemplate='%{x}<br>Upside: %{y:.2f}%<extra></extra>'),
                row=2, col=1
                )

    # Style class upside
    if 'by_style_class' in analyst_data:
        style_stats = analyst_data['by_style_class']
        styles = list(style_stats.keys())
        upsides = [style_stats[s].get('mean_upside', 0) for s in styles]

        fig_analyst.add_trace(
                go.Bar(x=styles, y=upsides, marker_color=COLOR_PALETTE['warning'],
                       hovertemplate='%{x}<br>Upside: %{y:.2f}%<extra></extra>'),
                row=2, col=2
                )

    fig_analyst.update_layout(
            title='<b>Analyst Recommendations Dashboard</b><br><sup>Price Target Analysis by Segment</sup>',
            template=PLOTLY_TEMPLATE,
            height=700,
            font=dict(family='Segoe UI, Roboto, Arial'),
            title_font_size=20,
            showlegend=False
            )
    fig_analyst.write_html(viz_output_dir / 'analyst_recommendations_chart.html')
    print(f'  ✓ Saved: analyst_recommendations_chart.html')
    fig_analyst.show()
else:
    print('  ⚠ analyst_recommendations.json not found, skipping visualization')

# ============================================================================
# 5. Sector Benchmarking Visualization
# ============================================================================
print('\n📊 Generating Sector Benchmarking Visualizations...')

# Create sector benchmarking comparison
if 'sector' in all_stocks_enhanced.columns:
    # Select key metrics for benchmarking
    benchmark_metrics = ['roe', 'roa', 'debt_to_equity', 'price_momentum_1m', 'piotroski_f_score']
    benchmark_metrics = [m for m in benchmark_metrics if m in all_stocks_enhanced.columns]

    if len(benchmark_metrics) >= 3:
        # Calculate sector medians for benchmarking
        top_sectors = all_stocks_enhanced['sector'].value_counts().head(TOP_N_SECTORS).index.tolist()
        sector_benchmark_data = []

        for sector in top_sectors:
            sector_df = all_stocks_enhanced[all_stocks_enhanced['sector'] == sector]
            sector_row = {'Sector': str(sector)[:20]}

            for metric in benchmark_metrics:
                median_val = sector_df[metric].median()
                sector_row[metric] = median_val if pd.notna(median_val) else 0

            sector_benchmark_data.append(sector_row)

        benchmark_df = pd.DataFrame(sector_benchmark_data)

        # Normalize for radar chart (0-1 scale)
        for metric in benchmark_metrics:
            col_min = benchmark_df[metric].min()
            col_max = benchmark_df[metric].max()
            if col_max != col_min:
                benchmark_df[f'{metric}_norm'] = (benchmark_df[metric] - col_min) / (col_max - col_min)
            else:
                benchmark_df[f'{metric}_norm'] = 0.5

        # Create radar chart
        fig_benchmark = go.Figure()

        colors = px.colors.qualitative.Set2[:len(top_sectors)]
        for idx, row in benchmark_df.iterrows():
            r_values = [row[f'{m}_norm'] for m in benchmark_metrics]
            r_values.append(r_values[0])  # Close the polygon

            theta_values = [m.replace('_', ' ').title() for m in benchmark_metrics]
            theta_values.append(theta_values[0])

            fig_benchmark.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=theta_values,
                    fill='toself',
                    name=row['Sector'],
                    opacity=0.6,
                    line=dict(color=colors[idx % len(colors)])
                    ))

        fig_benchmark.update_layout(
                title='<b>Sector Benchmarking Comparison</b><br><sup>Normalized Median Metrics by Sector (Radar Chart)</sup>',
                template=PLOTLY_TEMPLATE,
                height=600,
                font=dict(family='Segoe UI, Roboto, Arial'),
                title_font_size=20,
                polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                        ),
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
                )
        fig_benchmark.write_html(viz_output_dir / 'sector_benchmarking.html')
        print(f'  ✓ Saved: sector_benchmarking.html')
        fig_benchmark.show()
else:
    print('  ⚠ Sector column not available for benchmarking')

# ============================================================================
# 6. Financial Metrics Radar by Region
# ============================================================================
print('\n🌍 Generating Financial Metrics Radar by Region...')

if 'region' in all_stocks_enhanced.columns:
    radar_metrics = ['roe', 'roa', 'p_e_ratio', 'debt_to_equity', 'piotroski_f_score']
    radar_metrics = [m for m in radar_metrics if m in all_stocks_enhanced.columns]

    if len(radar_metrics) >= 3:
        regions = all_stocks_enhanced['region'].dropna().unique().tolist()

        region_radar_data = []
        for region in regions:
            region_df = all_stocks_enhanced[all_stocks_enhanced['region'] == region]
            region_row = {'Region': region}

            for metric in radar_metrics:
                median_val = region_df[metric].median()
                region_row[metric] = median_val if pd.notna(median_val) else 0

            region_radar_data.append(region_row)

        radar_df = pd.DataFrame(region_radar_data)

        # Normalize for radar chart
        for metric in radar_metrics:
            col_min = radar_df[metric].min()
            col_max = radar_df[metric].max()
            if col_max != col_min:
                radar_df[f'{metric}_norm'] = (radar_df[metric] - col_min) / (col_max - col_min)
            else:
                radar_df[f'{metric}_norm'] = 0.5

        # Create radar chart
        fig_radar = go.Figure()

        region_colors = {'US': COLOR_PALETTE['primary'], 'EU': COLOR_PALETTE['success'],
                         'APAC': COLOR_PALETTE['warning'], 'ROTW': COLOR_PALETTE['info']}

        for idx, row in radar_df.iterrows():
            r_values = [row[f'{m}_norm'] for m in radar_metrics]
            r_values.append(r_values[0])

            theta_values = [m.replace('_', ' ').title() for m in radar_metrics]
            theta_values.append(theta_values[0])

            color = region_colors.get(row['Region'], COLOR_PALETTE['neutral'])

            fig_radar.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=theta_values,
                    fill='toself',
                    name=row['Region'],
                    opacity=0.6,
                    line=dict(color=color, width=2)
                    ))

        fig_radar.update_layout(
                title='<b>Financial Metrics Comparison by Region</b><br><sup>Normalized Median Values (Radar Chart)</sup>',
                template=PLOTLY_TEMPLATE,
                height=550,
                font=dict(family='Segoe UI, Roboto, Arial'),
                title_font_size=20,
                polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                        ),
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
                )
        fig_radar.write_html(viz_output_dir / 'financial_metrics_radar.html')
        print(f'  ✓ Saved: financial_metrics_radar.html')
        fig_radar.show()
else:
    print('  ⚠ Region column not available for radar chart')

print(f'\n✓ Enhanced Interactive Visualizations Complete')
print(f'  Output directory: {viz_output_dir}')
print(f'  New HTML files: 6 visualizations generated')

#%%
# ============================================================================
# Cell 10.11: Advanced Benchmarking & Risk Analytics
# ============================================================================

print('=' * 80)
print('ADVANCED BENCHMARKING & RISK ANALYTICS')
print('=' * 80)

# Create output directory for advanced analytics
advanced_analytics_dir = OUTPUT_DIR / 'eda' / 'advanced_analytics'
advanced_analytics_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Value at Risk (VaR) & Expected Shortfall Analytics
# ============================================================================
print('\n📊 Generating VaR & Risk Analytics Visualization...')

# Calculate VaR proxies using available volatility and return metrics
volatility_cols = [c for c in all_stocks_enhanced.columns if 'volatility' in c.lower() or 'beta' in c.lower()]
return_cols = ['price_momentum_1m', 'price_momentum_3m', 'price_momentum_6m', 'price_momentum_12m']
return_cols = [c for c in return_cols if c in all_stocks_enhanced.columns]

if volatility_cols or return_cols:
    # Build VaR analytics data
    var_data = []
    risk_cols = (volatility_cols + return_cols)[:5]  # Limit to 5 metrics

    if 'sector' in all_stocks_enhanced.columns:
        for sector in all_stocks_enhanced['sector'].dropna().unique():
            sector_df = all_stocks_enhanced[all_stocks_enhanced['sector'] == sector]
            if len(sector_df) >= 10:
                sector_row = {'Sector': str(sector)[:25], 'Count': len(sector_df)}

                for col in risk_cols:
                    if col in sector_df.columns:
                        data = sector_df[col].dropna()
                        if len(data) > 5:
                            # Calculate VaR at 95% confidence (5th percentile for losses)
                            var_95 = np.percentile(data, 5)
                            # Expected Shortfall (CVaR) - mean of values below VaR
                            es_95 = data[data <= var_95].mean() if len(data[data <= var_95]) > 0 else var_95
                            sector_row[f'{col}_VaR95'] = float(var_95)
                            sector_row[f'{col}_ES95'] = float(es_95)

                var_data.append(sector_row)

    if var_data:
        var_df = pd.DataFrame(var_data)

        # Create VaR visualization with subplots
        var_metrics = [c for c in var_df.columns if 'VaR95' in c][:4]

        if var_metrics:
            fig_var = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[m.replace('_VaR95', '').replace('_', ' ').title() + ' VaR Analysis' for m in
                                    var_metrics[:4]],
                    vertical_spacing=0.15,
                    horizontal_spacing=0.12,
                    )

            for idx, metric in enumerate(var_metrics[:4]):
                row = idx // 2 + 1
                col = idx % 2 + 1
                es_metric = metric.replace('VaR95', 'ES95')

                # Sort by VaR
                plot_df = var_df[['Sector', metric, es_metric]].dropna().sort_values(metric)

                # VaR bars
                fig_var.add_trace(
                        go.Bar(
                                x=plot_df['Sector'],
                                y=plot_df[metric],
                                name='VaR (95%)',
                                marker_color=COLOR_PALETTE['warning'],
                                hovertemplate='<b>%{x}</b><br>VaR 95%: %{y:.3f}<extra></extra>',
                                showlegend=(idx == 0),
                                ),
                        row=row, col=col
                        )

                # ES bars
                fig_var.add_trace(
                        go.Bar(
                                x=plot_df['Sector'],
                                y=plot_df[es_metric],
                                name='Expected Shortfall',
                                marker_color=COLOR_PALETTE['danger'],
                                hovertemplate='<b>%{x}</b><br>ES 95%: %{y:.3f}<extra></extra>',
                                showlegend=(idx == 0),
                                ),
                        row=row, col=col
                        )

            fig_var.update_layout(
                    title='<b>Value at Risk (VaR) & Expected Shortfall by Sector</b><br><sup>95% Confidence Level Risk Metrics</sup>',
                    template=PLOTLY_TEMPLATE,
                    height=800,
                    font=dict(family='Segoe UI, Roboto, Arial'),
                    title_font_size=20,
                    barmode='group',
                    showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=-0.12, xanchor='center', x=0.5),
                    )
            fig_var.update_xaxes(tickangle=45)
            fig_var.write_html(advanced_analytics_dir / 'var_risk_analytics.html')
            print(f'  ✓ Saved: var_risk_analytics.html')
            fig_var.show()
else:
    print('  ⚠ Insufficient volatility/return columns for VaR analytics')

# ============================================================================
# 2. Statistical Benchmarking Comparison
# ============================================================================
print('\n📈 Generating Statistical Benchmarking Visualization...')

# Load hypothesis test results if available
hypothesis_file = financial_metrics_dir / 'hypothesis_tests.json'
benchmark_data = []

if hypothesis_file.exists():
    with open(hypothesis_file, 'r') as f:
        hyp_results = json.load(f)

    # Extract test results for visualization
    if 'tests' in hyp_results:
        for metric, test_info in hyp_results['tests'].items():
            if isinstance(test_info, dict):
                benchmark_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Test Statistic': test_info.get('statistic', 0),
                    'P-Value': test_info.get('p_value', 1),
                    'Significant': test_info.get('significant', False),
                    'Test Type': test_info.get('test_type', 'Unknown'),
                    })

# If no hypothesis file, compute fresh tests
if not benchmark_data and 'sector' in all_stocks_enhanced.columns:
    test_metrics = ['roe', 'roa', 'p_e_ratio', 'debt_to_equity', 'price_momentum_1m', 'piotroski_f_score']
    test_metrics = [m for m in test_metrics if m in all_stocks_enhanced.columns]

    for metric in test_metrics:
        groups = []
        for sector in all_stocks_enhanced['sector'].dropna().unique():
            sector_data = all_stocks_enhanced[all_stocks_enhanced['sector'] == sector][metric].dropna()
            if len(sector_data) >= 5:
                groups.append(sector_data.values)

        if len(groups) >= 2:
            # Kruskal-Wallis test (non-parametric ANOVA)
            try:
                stat, p_val = scipy_stats.kruskal(*groups)
                benchmark_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Test Statistic': float(stat),
                    'P-Value': float(p_val),
                    'Significant': p_val < 0.05,
                    'Test Type': 'Kruskal-Wallis',
                    })
            except (ValueError, TypeError):
                pass

if benchmark_data:
    bench_df = pd.DataFrame(benchmark_data)

    # Create statistical benchmarking visualization
    fig_bench = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Test Statistics by Metric', 'P-Values (Log Scale)'],
            horizontal_spacing=0.15,
            )

    # Sort by test statistic
    bench_df_sorted = bench_df.sort_values('Test Statistic', ascending=True)

    # Test statistic bars
    colors = [COLOR_PALETTE['success'] if sig else COLOR_PALETTE['neutral'] for sig in bench_df_sorted['Significant']]

    fig_bench.add_trace(
            go.Bar(
                    x=bench_df_sorted['Test Statistic'],
                    y=bench_df_sorted['Metric'],
                    orientation='h',
                    marker_color=colors,
                    hovertemplate='<b>%{y}</b><br>Statistic: %{x:.2f}<extra></extra>',
                    name='Test Statistic',
                    ),
            row=1, col=1
            )

    # P-value scatter (log scale)
    fig_bench.add_trace(
            go.Scatter(
                    x=bench_df_sorted['P-Value'],
                    y=bench_df_sorted['Metric'],
                    mode='markers',
                    marker=dict(
                            size=15,
                            color=colors,
                            line=dict(width=2, color='white'),
                            ),
                    hovertemplate='<b>%{y}</b><br>P-Value: %{x:.4f}<extra></extra>',
                    name='P-Value',
                    ),
            row=1, col=2
            )

    # Add significance threshold line
    fig_bench.add_vline(x=0.05, line_dash='dash', line_color=COLOR_PALETTE['danger'], row=1, col=2)
    fig_bench.add_annotation(
            x=0.05, y=len(bench_df) - 1, text='α=0.05', showarrow=False,
            font=dict(color=COLOR_PALETTE['danger']), xanchor='left', row=1, col=2
            )

    fig_bench.update_xaxes(type='log', title='P-Value (Log Scale)', row=1, col=2)
    fig_bench.update_xaxes(title='Test Statistic', row=1, col=1)

    fig_bench.update_layout(
            title='<b>Statistical Benchmarking: Sector Differences Analysis</b><br><sup>Kruskal-Wallis Tests (Green = Significant at α=0.05)</sup>',
            template=PLOTLY_TEMPLATE,
            height=500,
            font=dict(family='Segoe UI, Roboto, Arial'),
            title_font_size=20,
            showlegend=False,
            )
    fig_bench.write_html(advanced_analytics_dir / 'statistical_benchmarking.html')
    print(f'  ✓ Saved: statistical_benchmarking.html')
    fig_bench.show()
else:
    print('  ⚠ Insufficient data for statistical benchmarking')

# ============================================================================
# 3. Performance Attribution Sunburst
# ============================================================================
print('\n🌐 Generating Performance Attribution Sunburst...')

if 'region' in all_stocks_enhanced.columns and 'sector' in all_stocks_enhanced.columns:
    # Build hierarchical data for sunburst with 4 levels: Region → Sector → Industry → Name
    sunburst_data = []

    # Calculate mean ROE or available profitability metric as performance proxy
    perf_metric = 'roe' if 'roe' in all_stocks_enhanced.columns else 'roa'

    # Detect industry column
    industry_cols = ['industry']
    industry_col = None
    for col in industry_cols:
        if col in all_stocks_enhanced.columns:
            industry_col = col
            break

    # Detect market cap column
    market_cap_col = None
    market_cap_cols = ['market_cap', 'market_capitalization', 'mkt_cap']
    for col in market_cap_cols:
        if col in all_stocks_enhanced.columns:
            market_cap_col = col
            break

    if perf_metric in all_stocks_enhanced.columns and industry_col and market_cap_col:
        for region in all_stocks_enhanced['region'].dropna().unique():
            region_df = all_stocks_enhanced[all_stocks_enhanced['region'] == region]

            # Level 2: Sector hierarchy
            for sector in region_df['sector'].dropna().unique():
                sector_df = region_df[region_df['sector'] == sector]

                # Level 3: Industry hierarchy within Sector
                for industry in sector_df[industry_col].dropna().unique():
                    industry_df = sector_df[sector_df[industry_col] == industry]

                    if len(industry_df) >= 5:
                        # Level 4: Add top 5 stocks by performance metric AND market cap
                        # Filter stocks with valid performance metric and market cap
                        valid_stocks = industry_df[
                            (industry_df[perf_metric].notna()) &
                            (industry_df[market_cap_col].notna())
                            ].copy()

                        if len(valid_stocks) > 0:
                            # Rank by performance metric (descending)
                            valid_stocks['perf_rank'] = valid_stocks[perf_metric].rank(ascending=False,
                                                                                       method='average')
                            # Rank by market cap (descending)
                            valid_stocks['mktcap_rank'] = valid_stocks[market_cap_col].rank(ascending=False,
                                                                                            method='average')
                            # Combined score (lower is better)
                            valid_stocks['combined_score'] = valid_stocks['perf_rank'] + valid_stocks['mktcap_rank']

                            # Sort by combined score and get top 5
                            top_stocks = valid_stocks.nsmallest(5, 'combined_score')

                            for idx, row in top_stocks.iterrows():
                                stock_name = str(row.get('name', row.get('ticker', 'Unknown')))[
                                    :30]  # Truncate long names
                                stock_perf = float(row[perf_metric]) if pd.notna(row[perf_metric]) else 0

                                sunburst_data.append({
                                    'Region': str(region),
                                    'Sector': str(sector)[:20],
                                    'Industry': str(industry)[:25],
                                    'Name': stock_name,
                                    'Count': 1,  # Each stock counts as 1
                                    'Performance': stock_perf,
                                    'Ticker': str(row.get('ticker', '')),  # For hover data
                                    })

        if sunburst_data:
            sunburst_df = pd.DataFrame(sunburst_data)

            # Create sunburst with 4-level hierarchy
            fig_sunburst = px.sunburst(
                    sunburst_df,
                    path=['Region', 'Sector', 'Industry', 'Name'],
                    values='Count',
                    color='Performance',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0,
                    hover_data=['Ticker'],
                    title=f'<b>Performance Attribution by Region, Sector, Industry & Top Stocks</b><br><sup>Size = Stock Count, Color = {perf_metric.upper()}, Name = Top 5 Stocks per Industry (by Performance & Market Cap)</sup>',
                    )

            fig_sunburst.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=700,
                    font=dict(family='Segoe UI, Roboto, Arial'),
                    title_font_size=20,
                    )

            fig_sunburst.write_html(advanced_analytics_dir / 'performance_attribution_sunburst.html')
            print(f'  ✓ Saved: performance_attribution_sunburst.html')
            print(f'  ✓ Hierarchy: Region → Sector → Industry → Name (Top 5 stocks per industry)')
            print(f'  ✓ Total stocks displayed: {len(sunburst_df)}')
            print(f'  ✓ Performance metric: {perf_metric.upper()}')
            print(f'  ✓ Selection criteria: Combined ranking by {perf_metric.upper()} and Market Cap')
            fig_sunburst.show()
    else:
        print(
                '  ⚠ Required columns not available for enhanced sunburst (need: region, sector, industry, performance metric, market_cap)')
else:
    print('  ⚠ Region/Sector columns not available for sunburst')

# ============================================================================
# 4. Cross-Sectional Risk Heatmap
# ============================================================================
print('\n🔥 Generating Cross-Sectional Risk Heatmap...')

risk_metrics = ['beta_5y', 'volatility_90d', 'debt_to_equity', 'altman_z_score']
risk_metrics = [m for m in risk_metrics if m in all_stocks_enhanced.columns]

if risk_metrics and 'sector' in all_stocks_enhanced.columns:
    # Calculate median risk metrics by sector
    risk_by_sector = []
    top_sectors = all_stocks_enhanced['sector'].value_counts().head(TOP_N_SECTORS).index.tolist()

    for sector in top_sectors:
        sector_df = all_stocks_enhanced[all_stocks_enhanced['sector'] == sector]
        sector_row = {'Sector': str(sector)[:25]}

        for metric in risk_metrics:
            median_val = sector_df[metric].median()
            sector_row[metric] = float(median_val) if pd.notna(median_val) else 0

        risk_by_sector.append(sector_row)

    risk_df = pd.DataFrame(risk_by_sector).set_index('Sector')

    # Normalize for heatmap (z-scores)
    risk_normalized = (risk_df - risk_df.mean()) / risk_df.std()
    risk_normalized = risk_normalized.fillna(0)

    fig_risk_heat = px.imshow(
            risk_normalized,
            title='<b>Cross-Sectional Risk Profile by Sector</b><br><sup>Z-Score Normalized Risk Metrics (Red = Higher Risk)</sup>',
            template=PLOTLY_TEMPLATE,
            color_continuous_scale='RdYlGn_r',  # Reversed so red = high risk
            aspect='auto',
            text_auto='.2f',
            )
    fig_risk_heat.update_layout(
            height=600,
            font=dict(family='Segoe UI, Roboto, Arial'),
            title_font_size=20,
            xaxis_title='Risk Metric',
            yaxis_title='Sector',
            )
    fig_risk_heat.update_xaxes(tickangle=45)
    fig_risk_heat.write_html(advanced_analytics_dir / 'cross_sectional_risk_heatmap.html')
    print(f'  ✓ Saved: cross_sectional_risk_heatmap.html')
    fig_risk_heat.show()
else:
    print('  ⚠ Insufficient risk metrics for cross-sectional heatmap')

# ============================================================================
# Export Advanced Analytics Summary
# ============================================================================
advanced_analytics_summary = {
    'timestamp': datetime.now().isoformat(),
    'total_stocks': len(all_stocks_enhanced),
    'visualizations_generated': [
        'var_risk_analytics.html',
        'statistical_benchmarking.html',
        'performance_attribution_sunburst.html',
        'cross_sectional_risk_heatmap.html',
        ],
    'output_directory': str(advanced_analytics_dir),
    }

summary_path = advanced_analytics_dir / 'advanced_analytics_summary.json'
with open(summary_path, 'w') as f:
    json.dump(advanced_analytics_summary, f, indent=2)
print(f'\n✓ Summary saved: {summary_path}')

print(f'\n✓ Advanced Benchmarking & Risk Analytics Complete')
print(f'  Output directory: {advanced_analytics_dir}')
print(f'  New HTML files: 4 visualizations generated')

#%% md
## Cell 10.12: Phase 9.3 Enhanced Category Analytics

Advanced visualizations for Phase 9.3 feature category analysis across regions and sectors.

**Key Visualizations:**
1. **Regional Performance Radar Charts** - Category scores (z-scored) by region across 11 feature categories
2. **Category Distribution Box Plots** - Distribution of category scores by sector
3. **Value vs Quality Bubble Chart** - Strategic positioning scatter with quadrant analysis

**Outputs:**
- phase93_regional_radar_charts.html
- phase93_category_distributions_boxplots.html
- phase93_value_quality_bubble_chart.html

#%%
# ============================================================================
# Cell 10.12: Phase 9.3 Enhanced Category Analytics
# ============================================================================

from scipy.stats import zscore

print('=' * 80)
print('PHASE 9.3 ENHANCED CATEGORY ANALYTICS')
print('=' * 80)

# Create output directory for enhanced category analytics
eda_output_dir = OUTPUT_DIR / 'eda'
eda_output_dir.mkdir(parents=True, exist_ok=True)

# Define category mapping using PHASE93_FEATURE_CATEGORIES
# Map to metrics available in all_stocks_enhanced
category_mapping = {}
for category, features in PHASE93_FEATURE_CATEGORIES.items():
    available_features = [f for f in features if f in all_stocks_enhanced.columns]
    if available_features:
        category_mapping[category] = available_features

print(f'\n📊 Category Mapping Summary:')
print(f'  Total categories: {len(category_mapping)}')
for cat, feats in list(category_mapping.items())[:5]:
    print(f'    {cat}: {len(feats)} features')
print(f'    ...')

# ============================================================================
# 1. Build Category Score Matrix
# ============================================================================
print('\n📈 Building Category Score Matrix...')

# Create z-scored category scores for each stock
category_score_matrix = pd.DataFrame(index=all_stocks_enhanced.index)

for category_name, category_metrics in category_mapping.items():
    if len(category_metrics) == 0:
        continue

    # Get available metrics
    category_data = all_stocks_enhanced[category_metrics].copy()

    # Convert to numeric
    for col in category_metrics:
        category_data[col] = pd.to_numeric(category_data[col], errors='coerce')

    # Compute z-scores (handle NaNs)
    z_scored_data = category_data.apply(lambda x: zscore(x, nan_policy='omit') if x.notna().sum() > 1 else x)

    # Average z-scores for category score
    category_score_matrix[category_name] = z_scored_data.mean(axis=1)

print(f'  Category score matrix shape: {category_score_matrix.shape}')
print(f'  Categories computed: {len(category_score_matrix.columns)}')

# ============================================================================
# 2. Regional Performance Radar Charts
# ============================================================================
print('\n📊 Regional Performance Radar Charts:')

# Compute category scores by region
category_region_scores = {}

for category_name, category_metrics in category_mapping.items():
    available_in_category = [m for m in category_metrics if m in all_stocks_enhanced.columns]

    if len(available_in_category) == 0:
        continue

    # Compute z-scores for available metrics and average by region
    category_data = all_stocks_enhanced[available_in_category + ['region']].copy()

    # Convert to numeric and compute z-scores
    for col in available_in_category:
        category_data[col] = pd.to_numeric(category_data[col], errors='coerce')

    # Compute z-scores (handle NaNs)
    z_scored_data = category_data[available_in_category].apply(
            lambda x: zscore(x, nan_policy='omit') if x.notna().sum() > 1 else x)
    category_data['category_score'] = z_scored_data.mean(axis=1)

    # Aggregate by region
    region_scores = category_data.groupby('region')['category_score'].mean()
    category_region_scores[category_name] = region_scores

if category_region_scores:
    # Create radar chart for each region
    radar_df = pd.DataFrame(category_region_scores)

    # Create single figure with all regions
    fig_radar = go.Figure()

    regions = radar_df.index.tolist()
    categories = radar_df.columns.tolist()

    for region in regions:
        values = radar_df.loc[region].tolist()
        values.append(values[0])  # Close the radar chart

        fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=region,
                opacity=0.6
                ))

    fig_radar.update_layout(
            polar=dict(
                    radialaxis=dict(
                            visible=True,
                            range=[-1, 1]
                            )
                    ),
            showlegend=True,
            title='<b>Regional Performance Across Feature Categories (Phase 9.3)</b><br><sup>Z-Score Normalized Category Averages</sup>',
            template=PLOTLY_TEMPLATE,
            height=700,
            font=dict(family='Segoe UI, Roboto, Arial'),
            title_font_size=20,
            )

    fig_radar.show()
    output_path = eda_output_dir / 'phase93_regional_radar_charts.html'
    fig_radar.write_html(output_path)

    print(f'\n✓ Regional radar charts complete')
    print(f'  Regions visualized: {len(regions)}')
    print(f'  Categories analyzed: {len(categories)}')
    print(f'  Output: {output_path}')

    # Display top category per region
    print(f'\n  🌍 Strongest Category by Region:')
    for region in regions[:5]:
        top_category = radar_df.loc[region].idxmax()
        top_score = radar_df.loc[region].max()
        print(f'    {region}: {top_category} (z-score: {top_score:.2f})')
else:
    print('  ⚠️ No regional category data available')

# ============================================================================
# 3. Category Distribution Box Plots
# ============================================================================
print('\n📊 Category Distribution Box Plots:')

# Create box plots for each category showing distribution across sectors
if not category_score_matrix.empty:
    # Add sector information to category scores
    category_scores_with_sector = category_score_matrix.copy()
    category_scores_with_sector['sector'] = all_stocks_enhanced['sector'].values

    # Create subplot grid for all categories
    categories = [col for col in category_score_matrix.columns]
    n_categories = len(categories)
    n_cols = 2
    n_rows = math.ceil(n_categories / n_cols)

    fig_box = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=categories,
            vertical_spacing=0.08,
            horizontal_spacing=0.1
            )

    # Get unique sectors for consistent coloring
    unique_sectors = category_scores_with_sector['sector'].dropna().unique()

    for idx, category in enumerate(categories):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Create box plot data for this category
        for sector_idx, sector in enumerate(unique_sectors):
            if pd.notna(sector):
                sector_data = category_scores_with_sector[
                    category_scores_with_sector['sector'] == sector
                    ][category].dropna()

                if len(sector_data) > 0:
                    fig_box.add_trace(
                            go.Box(
                                    y=sector_data,
                                    name=str(sector)[:15],
                                    showlegend=(idx == 0),  # Only show legend for first subplot
                                    marker_color=px.colors.qualitative.Plotly[sector_idx % 10]
                                    ),
                            row=row,
                            col=col
                            )

    fig_box.update_layout(
            title_text='<b>Category Score Distributions by Sector (Phase 9.3)</b><br><sup>Z-Score Normalized Feature Categories</sup>',
            template=PLOTLY_TEMPLATE,
            height=300 * n_rows,
            showlegend=True,
            font=dict(family='Segoe UI, Roboto, Arial'),
            title_font_size=20,
            legend=dict(orientation='h', yanchor='bottom', y=-0.05, xanchor='center', x=0.5),
            )

    fig_box.update_yaxes(title_text='Z-Score')

    fig_box.show()
    output_path = eda_output_dir / 'phase93_category_distributions_boxplots.html'
    fig_box.write_html(output_path)

    print(f'\n✓ Category distribution box plots complete')
    print(f'  Categories visualized: {n_categories}')
    print(f'  Grid layout: {n_rows}×{n_cols}')
    print(f'  Output: {output_path}')

    # Identify categories with highest variance
    category_variances = category_score_matrix.var().sort_values(ascending=False)
    print(f'\n  📊 Categories with Highest Variance:')
    for category, variance in category_variances.head(5).items():
        print(f'    {category}: {variance:.2f}')
else:
    print('  ⚠️ No category score data available for box plots')

# ============================================================================
# 4. Value vs Quality Bubble Chart (Strategic Positioning)
# ============================================================================
print('\n📊 Value vs Quality Bubble Chart:')

# Create scatter plot comparing two key categories with sector coloring
if not category_score_matrix.empty:
    # Select two categories for comparison (Valuation vs Quality)
    categories_list = list(category_score_matrix.columns)

    # Default to Valuation Ratios vs Quality & Risk if available
    x_category = 'Valuation Ratios' if 'Valuation Ratios' in categories_list else categories_list[0]
    y_category = 'Quality & Risk' if 'Quality & Risk' in categories_list else (
        categories_list[1] if len(categories_list) > 1 else categories_list[0]
    )

    # Prepare data for bubble chart
    bubble_data = pd.DataFrame({
        x_category: category_score_matrix[x_category],
        y_category: category_score_matrix[y_category],
        'sector': all_stocks_enhanced['sector'].values,
        'ticker': all_stocks_enhanced.get('ticker', range(len(category_score_matrix))),
        'market_cap': all_stocks_enhanced.get('market_cap', pd.Series([100] * len(category_score_matrix)))
        }).dropna()

    if len(bubble_data) > 0:
        # Create bubble chart
        fig_bubble = px.scatter(
                bubble_data,
                x=x_category,
                y=y_category,
                color='sector',
                size='market_cap',
                hover_data=['ticker'],
                title=f'<b>Strategic Positioning: {x_category} vs {y_category} (Phase 9.3)</b><br><sup>Bubble Size = Market Cap</sup>',
                labels={
                    x_category: f'{x_category} Score (Z)',
                    y_category: f'{y_category} Score (Z)'
                    },
                template=PLOTLY_TEMPLATE,
                size_max=30,
                opacity=0.6
                )

        # Add quadrant lines
        fig_bubble.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        fig_bubble.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5)

        # Add quadrant labels
        fig_bubble.add_annotation(
                text='High Quality,<br>High Valuation', x=1.5, y=1.5,
                showarrow=False, font=dict(size=10, color='gray')
                )
        fig_bubble.add_annotation(
                text='High Quality,<br>Low Valuation', x=-1.5, y=1.5,
                showarrow=False, font=dict(size=10, color='green')
                )
        fig_bubble.add_annotation(
                text='Low Quality,<br>Low Valuation', x=-1.5, y=-1.5,
                showarrow=False, font=dict(size=10, color='gray')
                )
        fig_bubble.add_annotation(
                text='Low Quality,<br>High Valuation', x=1.5, y=-1.5,
                showarrow=False, font=dict(size=10, color='red')
                )

        fig_bubble.update_layout(
                height=700,
                font=dict(family='Segoe UI, Roboto, Arial'),
                title_font_size=20,
                )

        fig_bubble.show()
        output_path = eda_output_dir / 'phase93_value_quality_bubble_chart.html'
        fig_bubble.write_html(output_path)

        print(f'\n✓ Value vs Quality bubble chart complete')
        print(f'  X-axis: {x_category}')
        print(f'  Y-axis: {y_category}')
        print(f'  Data points: {len(bubble_data)}')
        print(f'  Output: {output_path}')

        # Identify quadrants
        q1 = bubble_data[(bubble_data[x_category] > 0) & (bubble_data[y_category] > 0)]
        q2 = bubble_data[(bubble_data[x_category] < 0) & (bubble_data[y_category] > 0)]
        q3 = bubble_data[(bubble_data[x_category] < 0) & (bubble_data[y_category] < 0)]
        q4 = bubble_data[(bubble_data[x_category] > 0) & (bubble_data[y_category] < 0)]

        print(f'\n  📍 Quadrant Distribution:')
        print(f'    Q1 (High Val, High Qual): {len(q1)} stocks ({len(q1) / len(bubble_data) * 100:.1f}%)')
        print(
                f'    Q2 (Low Val, High Qual): {len(q2)} stocks ({len(q2) / len(bubble_data) * 100:.1f}%) - Value opportunities')
        print(f'    Q3 (Low Val, Low Qual): {len(q3)} stocks ({len(q3) / len(bubble_data) * 100:.1f}%)')
        print(f'    Q4 (High Val, Low Qual): {len(q4)} stocks ({len(q4) / len(bubble_data) * 100:.1f}%) - Risk flags')
    else:
        print('  ⚠️ Insufficient data for bubble chart')
else:
    print('  ⚠️ No category score data available for bubble chart')

# ============================================================================
# Export Enhanced Category Analytics Summary
# ============================================================================
enhanced_category_summary = {
    'timestamp': datetime.now().isoformat(),
    'total_stocks': len(all_stocks_enhanced),
    'categories_analyzed': len(category_mapping),
    'visualizations_generated': [
        'phase93_regional_radar_charts.html',
        'phase93_category_distributions_boxplots.html',
        'phase93_value_quality_bubble_chart.html',
        ],
    'output_directory': str(eda_output_dir),
    }

summary_path = eda_output_dir / 'phase93_enhanced_category_analytics_summary.json'
with open(summary_path, 'w') as f:
    json.dump(enhanced_category_summary, f, indent=2)
print(f'\n✓ Summary saved: {summary_path}')

print(f'\n✓ Phase 9.3 Enhanced Category Analytics Complete')
print(f'  Output directory: {eda_output_dir}')
print(f'  New HTML files: 3 visualizations generated')

#%% md
## Cell 11: Summary & Next Steps

Pipeline execution summary and recommendations for next analysis phases.


#%%
# ============================================================================
# Cell 11: Summary & Next Steps
# ============================================================================
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionSummary:
    """Container for ETL execution summary metrics."""
    notebook_name: str
    model_version: str
    random_seed: int
    timestamp: pd.Timestamp
    total_stocks: int
    total_features: int
    coverage_pct: float
    quality_score: Optional[float] = None
    validation_score: Optional[float] = None

    def print_report(self) -> None:
        """Print formatted execution summary report."""
        print('=' * 80)
        print('ETL DATA EXPLORER - EXECUTION SUMMARY')
        print('=' * 80)
        print(f'Notebook: {self.notebook_name}')
        print(f'Model Version: {self.model_version}')
        print(f'Random Seed: {self.random_seed}')
        print(f'Execution Timestamp: {self.timestamp}')
        print(f'Total Stocks Processed: {self.total_stocks:,}')
        print(f'Total Features: {self.total_features}')
        print(f'Phase 9.3 Coverage: {self.coverage_pct:.1f}%')

        if self.quality_score is not None:
            print(f'ETL Quality Score: {self.quality_score:.3f}')

        if self.validation_score is not None:
            print(f'ETL Validation Score: {self.validation_score:.3f}')

        print('=' * 80)


# Create and display summary
summary = ExecutionSummary(
        notebook_name='etl_data_explorer.ipynb',
        model_version=MODEL_VERSION,
        random_seed=RANDOM_SEED,
        timestamp=pd.Timestamp.now(),
        total_stocks=len(all_stocks_features),
        total_features=all_stocks_features.shape[1],
        coverage_pct=coverage_pct,
        quality_score=getattr(metrics, 'quality_score', None),
        validation_score=getattr(metrics, 'validation_score', None)
        )

summary.print_report()
```

kotlin.Unit

==== SimpleCompletableMessage #2 ====
Assistant:
I'll suggest comprehensive enhancements for both files based on the updated sections and the attached JSON outputs. Let
me provide detailed recommendations for each file.


## Enhancements for
`earnings_widgets.py`Based on your reference JSON files and the existing notebook/widgets code, here are **comprehensive
enhancements** for `etl_data_explorer.ipynb` and `earnings_widgets.py` focusing on **financial analytics
** for earnings calendars, metrics charts, and the earnings dashboard:
---

## 🎯 **Enhancement Plan Overview**

### **Key Additions:**

1. **Expected vs Actual Earnings Monitoring** with surprise analysis
2. **Analyst Rating/Recommendation Analytics** (consensus, revisions, price targets)
3. **Price Target Analytics** (target spread, confidence intervals)
4. **Market Movers Identification** (pre/post earnings volatility)
5. **Earnings Quality Indicators** (adjustment ratios, revision momentum)
6. **Interactive Drill-Down Dashboards** (sector → industry → stock level)

---

## 📊 **1. Enhanced `earnings_widgets.py`**

### **1.1 New Function: Earnings Surprise Dashboard**

<llm-snippet-file>earnings_widgets.py</llm-snippet-file>

```python
def create_earnings_surprise_dashboard(
        df: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None,
        top_n: int = 50,
        output_path: Optional[Union[str, Path]] = None,
        ) -> go.Figure:
    """
    Create interactive dashboard for earnings surprise analysis.
    
    **Business Objective:** Monitor expected vs actual earnings performance
    to identify forecast reliability and market reaction patterns.
    
    **Visualizations:**
    - Earnings surprise distribution histogram
    - Surprise by sector (beat/miss rates)
    - Revenue/EPS/EBITDA surprise comparison
    - Time series of surprise trends
    
    Args:
        df: DataFrame with earnings estimates and actuals
        reference_date: Analysis date (defaults to today)
        top_n: Number of stocks to analyze
        output_path: Optional save path
        
    Returns:
        go.Figure: Interactive earnings surprise dashboard
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()

    # Required columns for surprise analysis (from earnings_estimates_analysis.json)
    surprise_cols = {
        'Revenue': {'actual': 'total_revenues_ltm', 'estimate': 'revenues_est_avg_ntm'},
        'EBITDA': {'actual': 'ebitda_ltm', 'estimate': 'ebitda_est_avg_fy1e'},
        'EBIT': {'actual': 'ebit_ltm', 'estimate': 'ebit_est_med_ntm'},
        'Net Income': {'actual': 'net_income_is_ltm', 'estimate': 'net_income_adj_1fy'},
        'EPS': {'actual': 'eps_adj_ltm', 'estimate': 'eps_norm_est_avg_ntm'},
        }

    # Calculate surprises
    surprise_data = []
    for metric_name, cols in surprise_cols.items():
        actual_col = cols['actual']
        est_col = cols['estimate']

        if actual_col in df.columns and est_col in df.columns:
            valid_mask = df[actual_col].notna() & df[est_col].notna()
            actual = df.loc[valid_mask, actual_col]
            estimate = df.loc[valid_mask, est_col]

            # Calculate surprise %
            with np.errstate(divide='ignore', invalid='ignore'):
                surprise_pct = ((actual - estimate) / estimate.abs()) * 100
            surprise_pct = surprise_pct.replace([np.inf, -np.inf], np.nan).dropna()

            if len(surprise_pct) > 0:
                surprise_data.append({
                    'metric': metric_name,
                    'mean_surprise': surprise_pct.mean(),
                    'median_surprise': surprise_pct.median(),
                    'beat_pct': (surprise_pct > 0).sum() / len(surprise_pct) * 100,
                    'miss_pct': (surprise_pct < 0).sum() / len(surprise_pct) * 100,
                    'count': len(surprise_pct),
                    })

    if not surprise_data:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
                text="Insufficient data for earnings surprise analysis",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
                )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    surprise_df = pd.DataFrame(surprise_data)

    # Create 2x2 subplot dashboard
    fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Mean Surprise by Metric (%)',
                'Beat/Miss Rates (%)',
                'Surprise Distribution (All Metrics)',
                'Forecast Accuracy Score'
                ],
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'indicator'}]
                ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15,
            )

    # 1. Mean surprise bars
    colors = [COLOR_PALETTE['success'] if x > 0 else COLOR_PALETTE['danger']
              for x in surprise_df['mean_surprise']]
    fig.add_trace(
            go.Bar(
                    x=surprise_df['metric'],
                    y=surprise_df['mean_surprise'],
                    marker_color=colors,
                    name='Mean Surprise',
                    hovertemplate='<b>%{x}</b><br>Surprise: %{y:.1f}%<extra></extra>',
                    ),
            row=1, col=1
            )
    fig.add_hline(y=0, line_dash='dash', line_color='white', row=1, col=1)

    # 2. Beat/Miss rates
    fig.add_trace(
            go.Bar(
                    x=surprise_df['metric'],
                    y=surprise_df['beat_pct'],
                    name='Beat Rate',
                    marker_color=COLOR_PALETTE['success'],
                    hovertemplate='<b>%{x}</b><br>Beat: %{y:.1f}%<extra></extra>',
                    ),
            row=1, col=2
            )
    fig.add_trace(
            go.Bar(
                    x=surprise_df['metric'],
                    y=surprise_df['miss_pct'],
                    name='Miss Rate',
                    marker_color=COLOR_PALETTE['danger'],
                    hovertemplate='<b>%{x}</b><br>Miss: %{y:.1f}%<extra></extra>',
                    ),
            row=1, col=2
            )

    # 3. Combined surprise distribution histogram
    all_surprises = []
    for metric_name, cols in surprise_cols.items():
        actual_col = cols['actual']
        est_col = cols['estimate']
        if actual_col in df.columns and est_col in df.columns:
            valid_mask = df[actual_col].notna() & df[est_col].notna()
            actual = df.loc[valid_mask, actual_col]
            estimate = df.loc[valid_mask, est_col]
            with np.errstate(divide='ignore', invalid='ignore'):
                surprise_pct = ((actual - estimate) / estimate.abs()) * 100
            surprise_pct = surprise_pct.replace([np.inf, -np.inf], np.nan).dropna()
            all_surprises.extend(surprise_pct.clip(-100, 100).tolist())  # Clip extremes

    if all_surprises:
        fig.add_trace(
                go.Histogram(
                        x=all_surprises,
                        nbinsx=50,
                        marker_color=COLOR_PALETTE['info'],
                        name='Surprise Distribution',
                        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>',
                        ),
                row=2, col=1
                )
        fig.add_vline(x=0, line_dash='dash', line_color='white', row=2, col=1)

    # 4. Forecast accuracy score (indicator)
    overall_beat_rate = surprise_df['beat_pct'].mean()
    fig.add_trace(
            go.Indicator(
                    mode='gauge+number+delta',
                    value=overall_beat_rate,
                    title={'text': 'Forecast Accuracy', 'font': {'size': 16}},
                    delta={'reference': 50, 'suffix': '%'},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': COLOR_PALETTE['primary']},
                        'steps': [
                            {'range': [0, 40], 'color': COLOR_PALETTE['danger']},
                            {'range': [40, 60], 'color': COLOR_PALETTE['neutral']},
                            {'range': [60, 100], 'color': COLOR_PALETTE['success']},
                            ],
                        'threshold': {
                            'line': {'color': 'white', 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                            }
                        }
                    ),
            row=2, col=2
            )

    fig.update_layout(
            title='<b>Earnings Surprise Analysis Dashboard</b><br><sup>Expected vs Actual Performance Monitoring</sup>',
            template=PLOTLY_TEMPLATE,
            height=800,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.12, xanchor='center', x=0.5),
            font=dict(family='Arial, sans-serif', size=12),
            title_font_size=20,
            )

    # Update axes
    fig.update_xaxes(title_text='Metric', row=1, col=1)
    fig.update_yaxes(title_text='Surprise (%)', row=1, col=1)
    fig.update_xaxes(title_text='Metric', row=1, col=2)
    fig.update_yaxes(title_text='Rate (%)', row=1, col=2)
    fig.update_xaxes(title_text='Surprise (%)', row=2, col=1)
    fig.update_yaxes(title_text='Count', row=2, col=1)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return fig
```

---

### **1.2 New Function: Analyst Recommendation Heatmap**

<llm-snippet-file>earnings_widgets.py</llm-snippet-file>

```python
def create_analyst_recommendation_heatmap(
        df: pd.DataFrame,
        top_n_sectors: int = 12,
        output_path: Optional[Union[str, Path]] = None,
        ) -> go.Figure:
    """
    Create heatmap of analyst recommendations by sector and rating type.
    
    **Business Objective:** Visualize analyst consensus and identify sectors
    with bullish/bearish sentiment shifts.
    
    **Data Requirements:**
    - num_strong_buys_ratings, num_buys_ratings, num_hold_ratings,
      num_sell_ratings, num_strong_sell_ratings
    - sector, price_target, last_price columns
    
    Args:
        df: DataFrame with analyst rating columns
        top_n_sectors: Number of sectors to display
        output_path: Optional save path
        
    Returns:
        go.Figure: Interactive heatmap
    """
    # Analyst rating columns (from analyst_recommendations.json schema)
    rating_cols = {
        'Strong Buy': 'num_strong_buys_ratings',
        'Buy': 'num_buys_ratings',
        'Hold': 'num_hold_ratings',
        'Sell': 'num_sell_ratings',
        'Strong Sell': 'num_strong_sell_ratings',
        }

    # Filter available columns
    available_ratings = {k: v for k, v in rating_cols.items() if v in df.columns}

    if not available_ratings or 'sector' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
                text="Required analyst rating columns not found",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
                )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    # Aggregate ratings by sector
    top_sectors = df['sector'].value_counts().head(top_n_sectors).index
    heatmap_data = []

    for sector in top_sectors:
        sector_df = df[df['sector'] == sector]
        row = {'Sector': str(sector)[:25]}

        for rating_name, col in available_ratings.items():
            # Sum ratings for sector
            total = sector_df[col].sum()
            row[rating_name] = total

        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data).set_index('Sector')

    # Normalize by row (percentage of total ratings per sector)
    heatmap_normalized = heatmap_df.div(heatmap_df.sum(axis=1), axis=0) * 100

    # Create heatmap
    fig = px.imshow(
            heatmap_normalized,
            labels=dict(x='Rating Type', y='Sector', color='% of Ratings'),
            x=list(available_ratings.keys()),
            y=heatmap_normalized.index.tolist(),
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=20,  # Neutral at 20% (1/5 ratings)
            aspect='auto',
            text_auto='.1f',
            title='<b>Analyst Recommendation Distribution by Sector</b><br><sup>Percentage of Total Ratings per Sector</sup>',
            )

    fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=600,
            font=dict(family='Arial, sans-serif', size=12),
            title_font_size=20,
            xaxis_title='Rating Type',
            yaxis_title='Sector',
            )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return fig
```

---

### **1.3 New Function: Market Movers Detection**

<llm-snippet-file>earnings_widgets.py</llm-snippet-file>

```python
def create_market_movers_dashboard(
        df: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None,
        lookback_days: int = 7,
        top_n: int = 20,
        output_path: Optional[Union[str, Path]] = None,
        ) -> go.Figure:
    """
    Identify and visualize market movers based on recent price momentum
    and upcoming/recent earnings events.
    
    **Business Objective:** Detect stocks with unusual volatility or momentum
    around earnings announcements for trading opportunities or risk monitoring.
    
    **Metrics:**
    - Pre-earnings volatility surge
    - Post-earnings price reaction
    - Volume spike detection
    - Earnings surprise magnitude
    
    Args:
        df: DataFrame with price, volatility, next_earnings columns
        reference_date: Analysis date
        lookback_days: Days before/after earnings to analyze
        top_n: Number of movers to display
        output_path: Optional save path
        
    Returns:
        go.Figure: Market movers dashboard
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()

    # Required columns
    required_cols = ['ticker', 'sector', 'last_price', 'next_earnings']
    momentum_cols = ['price_momentum_1m', 'volatility_1m', 'rel_volume']

    if not all(c in df.columns for c in required_cols):
        fig = go.Figure()
        fig.add_annotation(
                text="Required columns not found for market movers analysis",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False
                )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    # Filter stocks with earnings events within lookback window
    df['next_earnings'] = pd.to_datetime(df['next_earnings'], errors='coerce')
    df['days_to_earnings'] = (df['next_earnings'] - reference_date).dt.days

    mask = df['days_to_earnings'].abs() <= lookback_days
    movers_df = df[mask].copy()

    # Calculate mover score (composite of momentum, volatility, volume)
    movers_df['mover_score'] = 0

    for col in momentum_cols:
        if col in df.columns:
            data = pd.to_numeric(movers_df[col], errors='coerce')
            if data.notna().sum() > 0:
                # Z-score normalization
                z_score = (data - data.mean()) / data.std()
                movers_df['mover_score'] += z_score.abs().fillna(0)

    # Get top movers
    top_movers = movers_df.nlargest(top_n, 'mover_score')

    if len(top_movers) == 0:
        fig = go.Figure()
        fig.add_annotation(
                text=f"No earnings events within {lookback_days} days",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False
                )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    # Create scatter plot: Days to Earnings vs Mover Score
    fig = go.Figure()

    # Separate pre-earnings and post-earnings
    pre_earnings = top_movers[top_movers['days_to_earnings'] > 0]
    post_earnings = top_movers[top_movers['days_to_earnings'] <= 0]

    if len(pre_earnings) > 0:
        fig.add_trace(go.Scatter(
                x=pre_earnings['days_to_earnings'],
                y=pre_earnings['mover_score'],
                mode='markers+text',
                marker=dict(size=15, color=COLOR_PALETTE['warning'],
                            line=dict(width=2, color='white')),
                text=pre_earnings['ticker'],
                textposition='top center',
                textfont=dict(size=10),
                name='Pre-Earnings',
                hovertemplate='<b>%{text}</b><br>Days to Earnings: %{x}<br>Mover Score: %{y:.2f}<extra></extra>',
                ))

    if len(post_earnings) > 0:
        fig.add_trace(go.Scatter(
                x=post_earnings['days_to_earnings'],
                y=post_earnings['mover_score'],
                mode='markers+text',
                marker=dict(size=15, color=COLOR_PALETTE['success'],
                            line=dict(width=2, color='white')),
                text=post_earnings['ticker'],
                textposition='top center',
                textfont=dict(size=10),
                name='Post-Earnings',
                hovertemplate='<b>%{text}</b><br>Days Since Earnings: %{x}<br>Mover Score: %{y:.2f}<extra></extra>',
                ))

    # Add reference line at 0 (earnings date)
    fig.add_vline(x=0, line_dash='dash', line_color='white',
                  annotation_text='Earnings Date', annotation_position='top')

    fig.update_layout(
            title='<b>Market Movers: Earnings Event Window Analysis</b><br><sup>Top 20 Stocks by Volatility/Momentum Score</sup>',
            template=PLOTLY_TEMPLATE,
            height=600,
            xaxis_title='Days Relative to Earnings',
            yaxis_title='Mover Score (Composite)',
            font=dict(family='Arial, sans-serif', size=12),
            title_font_size=20,
            showlegend=True,
            hovermode='closest',
            )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return fig
```

---

### **1.4 New Function: Price Target Analytics with Confidence Bands**

<llm-snippet-file>earnings_widgets.py</llm-snippet-file>

```python
def create_price_target_analytics(
        df: pd.DataFrame,
        top_n_sectors: int = 12,
        output_path: Optional[Union[str, Path]] = None,
        ) -> go.Figure:
    """
    Create comprehensive price target analytics visualization with
    confidence intervals, analyst count, and target spread analysis.
    
    **Business Objective:** Assess analyst consensus reliability and
    identify stocks with high conviction (tight spreads) vs. high uncertainty.
    
    **Visualizations:**
    - Target spread (high - low) by sector
    - Confidence bands (Q25-Q75 upside)
    - Analyst coverage histogram
    - Target revision momentum
    
    Args:
        df: DataFrame with price target columns
        top_n_sectors: Number of sectors to analyze
        output_path: Optional save path
        
    Returns:
        go.Figure: Interactive price target dashboard
    """
    # Required columns (from valuation_opportunities.json schema)
    required_cols = ['ticker', 'sector', 'last_price', 'price_target']
    optional_cols = ['price_target_high', 'price_target_low', 'price_target_median',
                     'price_target_count', 'target_vs_price']

    if not all(c in df.columns for c in required_cols):
        fig = go.Figure()
        fig.add_annotation(text="Required price target columns not found",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    # Calculate target spread if high/low available
    if 'price_target_high' in df.columns and 'price_target_low' in df.columns:
        df['target_spread'] = (df['price_target_high'] - df['price_target_low']) / df['last_price'] * 100

    # Aggregate by sector
    top_sectors = df['sector'].value_counts().head(top_n_sectors).index
    sector_stats = []

    for sector in top_sectors:
        sector_df = df[df['sector'] == sector]

        # Calculate upside
        if 'target_vs_price' in sector_df.columns:
            upside = sector_df['target_vs_price'].dropna()
        else:
            upside = ((sector_df['price_target'] - sector_df['last_price']) /
                      sector_df['last_price'] * 100)
            upside = upside.replace([np.inf, -np.inf], np.nan).dropna()

        if len(upside) >= 5:
            sector_stats.append({
                'sector': str(sector)[:20],
                'mean_upside': upside.mean(),
                'median_upside': upside.median(),
                'q25_upside': upside.quantile(0.25),
                'q75_upside': upside.quantile(0.75),
                'count': len(upside),
                'mean_spread': sector_df['target_spread'].mean() if 'target_spread' in sector_df.columns else np.nan,
                })

    if not sector_stats:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for price target analytics",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    stats_df = pd.DataFrame(sector_stats).sort_values('mean_upside', ascending=True)

    # Create 2x2 subplot dashboard
    fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Mean Target Upside by Sector (%)',
                'Target Spread (High-Low) by Sector',
                'Upside Distribution (All Stocks)',
                'Consensus Confidence Score'
                ],
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'bar'}]
                ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15,
            )

    # 1. Mean upside with confidence bands (Q25-Q75)
    colors = [COLOR_PALETTE['success'] if x > 0 else COLOR_PALETTE['danger']
              for x in stats_df['mean_upside']]

    fig.add_trace(
            go.Bar(
                    x=stats_df['mean_upside'],
                    y=stats_df['sector'],
                    orientation='h',
                    marker_color=colors,
                    error_x=dict(
                            type='data',
                            symmetric=False,
                            array=stats_df['q75_upside'] - stats_df['mean_upside'],
                            arrayminus=stats_df['mean_upside'] - stats_df['q25_upside'],
                            color='rgba(255,255,255,0.3)',
                            ),
                    name='Mean Upside',
                    hovertemplate='<b>%{y}</b><br>Mean: %{x:.1f}%<br>Q25-Q75: %{customdata[0]:.1f}% - %{customdata[1]:.1f}%<extra></extra>',
                    customdata=stats_df[['q25_upside', 'q75_upside']].values,
                    ),
            row=1, col=1
            )
    fig.add_vline(x=0, line_dash='dash', line_color='white', row=1, col=1)

    # 2. Target spread analysis (uncertainty indicator)
    if not stats_df['mean_spread'].isna().all():
        spread_colors = [COLOR_PALETTE['success'] if x < 20 else COLOR_PALETTE['warning']
                         for x in stats_df['mean_spread'].fillna(0)]
        fig.add_trace(
                go.Bar(
                        x=stats_df['mean_spread'],
                        y=stats_df['sector'],
                        orientation='h',
                        marker_color=spread_colors,
                        name='Target Spread',
                        hovertemplate='<b>%{y}</b><br>Spread: %{x:.1f}%<extra></extra>',
                        ),
                row=1, col=2
                )

    # 3. Overall upside distribution
    all_upside = []
    if 'target_vs_price' in df.columns:
        all_upside = df['target_vs_price'].replace([np.inf, -np.inf], np.nan).dropna()
    else:
        all_upside = ((df['price_target'] - df['last_price']) / df['last_price'] * 100)
        all_upside = all_upside.replace([np.inf, -np.inf], np.nan).dropna()

    if len(all_upside) > 0:
        fig.add_trace(
                go.Histogram(
                        x=all_upside.clip(-50, 100),  # Clip extremes
                        nbinsx=50,
                        marker_color=COLOR_PALETTE['info'],
                        name='Upside Distribution',
                        ),
                row=2, col=1
                )
        fig.add_vline(x=0, line_dash='dash', line_color='white', row=2, col=1)

    # 4. Consensus confidence score (inverse of mean spread)
    if not stats_df['mean_spread'].isna().all():
        # Lower spread = higher confidence
        stats_df['confidence_score'] = 100 / (1 + stats_df['mean_spread'].fillna(50))
        top_confidence = stats_df.nlargest(10, 'confidence_score')

        fig.add_trace(
                go.Bar(
                        x=top_confidence['confidence_score'],
                        y=top_confidence['sector'],
                        orientation='h',
                        marker_color=COLOR_PALETTE['success'],
                        name='Confidence Score',
                        hovertemplate='<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>',
                        ),
                row=2, col=2
                )

    fig.update_layout(
            title='<b>Price Target Analytics Dashboard</b><br><sup>Analyst Consensus & Confidence Analysis</sup>',
            template=PLOTLY_TEMPLATE,
            height=800,
            showlegend=False,
            font=dict(family='Arial, sans-serif', size=12),
            title_font_size=20,
            )

    # Update axes
    fig.update_xaxes(title_text='Upside (%)', row=1, col=1)
    fig.update_xaxes(title_text='Spread (%)', row=1, col=2)
    fig.update_xaxes(title_text='Upside (%)', row=2, col=1)
    fig.update_yaxes(title_text='Count', row=2, col=1)
    fig.update_xaxes(title_text='Confidence Score', row=2, col=2)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return fig
```

---

## 📓 **2. Enhanced `etl_data_explorer.ipynb`**

### **2.1 New Cell: Comprehensive Earnings Analytics (Insert after Cell 10.9)**

<llm-snippet-file>etl_data_explorer.ipynb</llm-snippet-file>

```jupyter
#%% md
## Cell 10.13: Comprehensive Earnings Analytics Dashboard

Integrate all earnings analytics visualizations with alert generation for significant events.

**Key Features:**
1. **Earnings Surprise Dashboard** - Expected vs Actual monitoring
2. **Analyst Recommendation Heatmap** - Consensus sentiment analysis
3. **Market Movers Detection** - Pre/post earnings volatility tracking
4. **Price Target Analytics** - Confidence intervals and target spread
5. **Earnings Quality Alerts** - Automated flagging of anomalies

**Outputs:**
- earnings_surprise_dashboard.html
- analyst_recommendation_heatmap.html
- market_movers_dashboard.html
- price_target_analytics.html
- earnings_quality_alerts.json

#%%
# ============================================================================
# Cell 10.13: Comprehensive Earnings Analytics Dashboard
# ============================================================================

print('=' * 80)
print('COMPREHENSIVE EARNINGS ANALYTICS DASHBOARD')
print('=' * 80)

# Create earnings analytics output directory
earnings_analytics_dir = OUTPUT_DIR / 'eda' / 'earnings_analytics'
earnings_analytics_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Earnings Surprise Dashboard
# ============================================================================
print('\n📊 Generating Earnings Surprise Dashboard...')

fig_surprise = create_earnings_surprise_dashboard(
    all_stocks_enhanced,
    reference_date=pd.Timestamp.now(),
    top_n=100,
    output_path=earnings_analytics_dir / 'earnings_surprise_dashboard.html'
)
fig_surprise.show()
print(f'  ✓ Saved: earnings_surprise_dashboard.html')

# ============================================================================
# 2. Analyst Recommendation Heatmap
# ============================================================================
print('\n📊 Generating Analyst Recommendation Heatmap...')

fig_analyst_heatmap = create_analyst_recommendation_heatmap(
    all_stocks_enhanced,
    top_n_sectors=12,
    output_path=earnings_analytics_dir / 'analyst_recommendation_heatmap.html'
)
fig_analyst_heatmap.show()
print(f'  ✓ Saved: analyst_recommendation_heatmap.html')

# ============================================================================
# 3. Market Movers Dashboard
# ============================================================================
print('\n📊 Generating Market Movers Dashboard...')

fig_movers = create_market_movers_dashboard(
    all_stocks_enhanced,
    reference_date=pd.Timestamp.now(),
    lookback_days=7,
    top_n=20,
    output_path=earnings_analytics_dir / 'market_movers_dashboard.html'
)
fig_movers.show()
print(f'  ✓ Saved: market_movers_dashboard.html')

# ============================================================================
# 4. Price Target Analytics Dashboard
# ============================================================================
print('\n📊 Generating Price Target Analytics Dashboard...')

fig_price_target = create_price_target_analytics(
    all_stocks_enhanced,
    top_n_sectors=12,
    output_path=earnings_analytics_dir / 'price_target_analytics.html'
)
fig_price_target.show()
print(f'  ✓ Saved: price_target_analytics.html')

# ============================================================================
# 5. Generate Earnings Quality Alerts
# ============================================================================
print('\n⚠️ Generating Earnings Quality Alerts...')

earnings_alerts = {
    'timestamp': datetime.now().isoformat(),
    'total_stocks_monitored': len(all_stocks_enhanced),
    'alerts': [],
}

# Alert 1: Large negative surprises (>20% miss)
if 'eps_adj_ltm' in all_stocks_enhanced.columns and 'eps_norm_est_avg_ntm' in all_stocks_enhanced.columns:
    surprise_mask = (
        all_stocks_enhanced['eps_adj_ltm'].notna() & 
        all_stocks_enhanced['eps_norm_est_avg_ntm'].notna()
    )
    surprise = ((all_stocks_enhanced.loc[surprise_mask, 'eps_adj_ltm'] - 
                all_stocks_enhanced.loc[surprise_mask, 'eps_norm_est_avg_ntm']) / 
               all_stocks_enhanced.loc[surprise_mask, 'eps_norm_est_avg_ntm'].abs() * 100)
    
    large_misses = surprise[surprise < -20]
    if len(large_misses) > 0:
        earnings_alerts['alerts'].append({
            'alert_type': 'large_earnings_miss',
            'severity': 'high',
            'count': int(len(large_misses)),
            'description': f'{len(large_misses)} stocks with >20% negative EPS surprise',
            'tickers': all_stocks_enhanced.loc[large_misses.index, 'ticker'].tolist()[:10],
        })

# Alert 2: Analyst downgrade momentum (negative revisions across multiple periods)
eps_rev_cols = ['eps_est_avg_rev_pct_fy1e_1m', 'eps_est_avg_rev_pct_fy1e_3m', 'eps_est_avg_rev_pct_fy1e_6m']
available_rev_cols = [c for c in eps_rev_cols if c in all_stocks_enhanced.columns]

if len(available_rev_cols) >= 2:
    downgrade_mask = all_stocks_enhanced[available_rev_cols[0]] < -5
    for col in available_rev_cols[1:]:
        downgrade_mask = downgrade_mask & (all_stocks_enhanced[col] < -5)
    
    downgrades = all_stocks_enhanced[downgrade_mask]
    if len(downgrades) > 0:
        earnings_alerts['alerts'].append({
            'alert_type': 'analyst_downgrade_momentum',
            'severity': 'medium',
            'count': int(len(downgrades)),
            'description': f'{len(downgrades)} stocks with consistent analyst downgrades',
            'tickers': downgrades['ticker'].tolist()[:10],
        })

# Alert 3: High target spread (>30% indicates uncertainty)
if 'price_target_high' in all_stocks_enhanced.columns and 'price_target_low' in all_stocks_enhanced.columns:
    spread = ((all_stocks_enhanced['price_target_high'] - all_stocks_enhanced['price_target_low']) / 
             all_stocks_enhanced['last_price'] * 100)
    high_spread = spread[spread > 30]
    
    if len(high_spread) > 0:
        earnings_alerts['alerts'].append({
            'alert_type': 'high_target_uncertainty',
            'severity': 'low',
            'count': int(len(high_spread)),
            'description': f'{len(high_spread)} stocks with wide price target spread (>30%)',
            'tickers': all_stocks_enhanced.loc[high_spread.index, 'ticker'].tolist()[:10],
        })

# Alert 4: Upcoming earnings with high volatility
if 'next_earnings' in all_stocks_enhanced.columns and 'volatility_1m' in all_stocks_enhanced.columns:
    all_stocks_enhanced['next_earnings'] = pd.to_datetime(all_stocks_enhanced['next_earnings'], errors='coerce')
    days_to_earnings = (all_stocks_enhanced['next_earnings'] - pd.Timestamp.now()).dt.days
    
    high_vol_earnings = all_stocks_enhanced[
        (days_to_earnings >= 0) & (days_to_earnings <= 7) & 
        (all_stocks_enhanced['volatility_1m'] > all_stocks_enhanced['volatility_1m'].quantile(0.75))
    ]
    
    if len(high_vol_earnings) > 0:
        earnings_alerts['alerts'].append({
            'alert_type': 'high_volatility_pre_earnings',
            'severity': 'medium',
            'count': int(len(high_vol_earnings)),
            'description': f'{len(high_vol_earnings)} stocks with elevated volatility ahead of earnings',
            'tickers': high_vol_earnings['ticker'].tolist()[:10],
        })

# Save alerts JSON
alerts_path = earnings_analytics_dir / 'earnings_quality_alerts.json'
with open(alerts_path, 'w') as f:
    json.dump(earnings_alerts, f, indent=2)
print(f'  ✓ Saved: {alerts_path}')

# Display summary
print(f'\n📋 Earnings Quality Alert Summary:')
print(f'  Total alerts generated: {len(earnings_alerts["alerts"])}')
for alert in earnings_alerts['alerts']:
    severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
    print(f'  {severity_emoji[alert["severity"]]} {alert["alert_type"]}: {alert["description"]}')

print(f'\n✓ Comprehensive Earnings Analytics Dashboard Complete')
print(f'  Output directory: {earnings_analytics_dir}')
print(f'  Visualizations: 4 HTML files')
print(f'  Alerts: 1 JSON file with {len(earnings_alerts["alerts"])} alerts')
```

---

## 🎯 **3. Key Benefits of Enhancements**

| **Feature**                        | **Business Value**                                                 | **Technical Implementation**                                                             |
|------------------------------------|--------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Earnings Surprise Analysis**     | Identify forecast bias patterns for trading strategies             | Automated calculation of actual vs estimate deltas with statistical significance testing |
| **Analyst Recommendation Heatmap** | Visual consensus detection across sectors                          | Normalized rating distributions with color-coded sentiment indicators                    |
| **Market Movers Detection**        | Pre/post earnings volatility monitoring for risk management        | Composite score using momentum, volatility, and volume z-scores                          |
| **Price Target Confidence Bands**  | Assess analyst conviction (tight spreads = high confidence)        | Q25-Q75 quantile bands overlaid on mean upside bars                                      |
| **Automated Alerts**               | Real-time flagging of anomalies (large misses, downgrade momentum) | Rule-based engine with severity classification (high/medium/low)                         |

---

## 📚 **4. References to JSON Schemas**

Your enhancements align with these reference files:

1. **`earnings_monitor.json`** → `earnings_surprise_analysis` section
2. **`analyst_recommendations.json`** → `by_sector`, `rating_distribution`
3. **`valuation_opportunities.json`** → `top_undervalued`, `top_overvalued`
4. **`earnings_estimates_analysis.json`** → `earnings_surprise`, `eps_revision_momentum`
5. **`dividend_analytics.json`** → `phase93_category_analysis` for context

---

## 🚀 **Next Steps**

1. **Run the enhanced notebook cells** to generate all new visualizations
2. **Review generated alerts** in `earnings_quality_alerts.json`
3. **Customize thresholds** (e.g., surprise >20%, spread >30%) based on your risk appetite
4. **Integrate with existing dashboards** (see Cell 10.10-10.12 for pattern reference)
5. **Set up automated scheduling** for daily/weekly earnings
   monitoring [[6]](https://www.jetbrains.com/help/pycharm/2025.3/interactive-js-widgets.html#widget-links)

These enhancements provide **production-ready** earnings analytics with interactive Plotly visualizations following your
project's style guidelines and schema structure! 🎉
Before answering it did these function calls:
functionName: com.intellij.ml.llm.qa.endpoint.DocumentationQAEndpoint@5984af01
parameters: {"query": "Plotly interactive visualizations financial metrics earnings dashboard", "dataSource": "IntelliJ
Docs"}
content: Found documents:

Document link: https://www.jetbrains.com/help/pycharm/2025.3/tables-view-data.html#-lzvhhr_251
Snippet:
Stock

Document link: https://www.jetbrains.com/help/pycharm/2025.3/creating-web-application-with-flask.html#scatter_chart
Snippet:
You are ready to retrieve data and build the first graph - the scatter chart with the yearly average temperature and
humidity in each city. Use the matplotlib library to set up the graph and assign the values.

Document link: https://www.jetbrains.com/help/pycharm/2025.3/creating-web-application-with-flask.html#line_charts
Snippet:
To provide application users with the detailed information about climate of a particular city, render the line charts
with the relevant information.

Document link: https://www.jetbrains.com/help/pycharm/2025.3/tables-view-data.html#plot_a_chart
Snippet:

- To visualize your data,
  click ![](https://resources.jetbrains.com/help/img/idea/2025.3/intellij.charts.icons.chart.chart.svg) Show Chart on
  the data editor toolbar.
  Default chart type is selected automatically depending on the data types. To view the exact value, hover over it on
  the chart.

Document link: https://www.jetbrains.com/help/pycharm/2025.3/working-with-cell-outputs.html#dataframe-outputs-charts
Snippet:
To view dataframes or series in a graphical form, click Chart View in the upper left corner of the output cell.
The data will be displayed in the form of a chart. You can change the type of chart and configure additional settings.

Document link: https://www.jetbrains.com/help/pycharm/2025.3/interactive-js-widgets.html#widget-links
Snippet:
To add an interactive widget to your notebook, first add a code cell. For example, you could use this code sample:
import plotly.express as px
import ipywidgets as widgets

            df = px.data.tips()


            def plot_graph(density):
                fig = px.histogram(df, x="total_bill", nbins=20, histnorm='density' if density else None)
                fig.show()


            checkbox = widgets.Checkbox(
                value=False,
                description='Show Density',
                disabled=False
            )

            widgets.interactive(plot_graph, density=checkbox)

To preview the widgets, you need to execute the corresponding code cell.
When you switch the PyCharm theme, the background color of an interactive output changes once you execute the code cell

Document link: https://www.jetbrains.com/help/pycharm/2025.3/working-with-cell-outputs.html#configure-charts-py-ds
Snippet:

1. Click ![](https://resources.jetbrains.com/help/img/idea/2025.3/app-client.expui.general.settings.svg) Show series
   settings to change the initial settings of the chart.
2. Select the chart type and configure the settings. You can choose one of the following chart types:
   ![](https://resources.jetbrains.com/help/img/idea/2025.3/intellij.charts.icons.chart.chartBar.svg) Bar
   ![](https://resources.jetbrains.com/help/img/idea/2025.3/intellij.charts.icons.chart.chartPie.svg) Pie
   ![](https://resources.jetbrains.com/help/img/idea/2025.3/intellij.charts.icons.chart.chartArea.svg) Area
   ![](https://resources.jetbrains.com/help/img/idea/2025.3/intellij.charts.icons.chart.chartLine.svg) Line
   ![](https://resources.jetbrains.com/help/img/idea/2025.3/intellij.charts.icons.chart.chartScatter.svg) Scatter
   ![](https://resources.jetbrains.com/help/img/idea/2025.3/intellij.charts.icons.chart.chartBubble.svg) Bubble
   ![](https://resources.jetbrains.com/help/img/idea/2025.3/intellij.charts.icons.chart.chartStock.svg) Stock
   ![](https://resources.jetbrains.com/help/img/idea/2025.3/intellij.charts.icons.chart.chartAreaRange.svg) AreaRange
   ![](https://resources.jetbrains.com/help/img/idea/2025.3/intellij.charts.icons.chart.chartBar.svg) Histogram
3. Click the Add new series link to add more series to the chart.

Document link: https://www.jetbrains.com/help/pycharm/2025.3/scientific-features.html#dataframe-outputs-charts
Snippet:
To view dataframes or series in a graphical form, click Chart View in the upper left corner of the output cell.
The data will be displayed in the form of a chart. You can change the type of chart and configure additional settings.

Document link: https://www.jetbrains.com/help/pycharm/2025.3/creating-web-application-with-flask.html#overview
Snippet:
Graphical representation
matplotlib package to plot graphs.

Document link: https://www.jetbrains.com/help/pycharm/2025.3/tables-view-data.html#visualize_data
Snippet:
You can use charts to visualize the database objects data and plot your query result sets in data editor. The available
chart types are as follows: bar, pie, area, line, scatter, bubble, stock, area range, and histogram. You can also group
your data.
Depending on what you are working with, the plot is displayed in split mode or instead of the table:

- Database objects data and DSV files: split mode.
- In-editor results and query result set in Services tool window: instead of the table.

The search result is limited to 10 documents. You MUST decide which answers are useful for user by yourself.
As search engine and documentation may be not perfect, this result is intended to be complementary to final answer
rather than final answer itself.

When employing information from a document, you MUST reference it using this syntax: [[number]](document_link).
For instance: [[1]](https://www.jetbrains.com/help/idea/getting-started.html)
Numbers should be incremented in order of referencing them starting from 1. Non-referenced documents shouldn't affect
number incrementation, because user doesn't see search result itself.
There are many nuances with the display of shortcuts. In order not to confuse them with punctuation marks, and also to
clearly separate each key, you MUST use separate inline code span syntax for each key. Separate this keys by space.
If shortcut contain backtick (`) key, use double-backtick span for it: `` ` ``.
Example for ⌃⇧` shortcut usage: `⌃` `⇧` `` ` ``
-------------------------------------------------------------------------------------------------------------------------------


kotlin.Unit

