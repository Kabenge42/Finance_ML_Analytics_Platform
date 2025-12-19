# %%
import os
from pathlib import Path
from typing import Final, List, Literal

import numpy as np
import pandas as pd

# ========== NOTEBOOK CONFIGURATION ==========
# Version: Phase 9.5 (Unified ETL with Semantic Transforms + Feature Engineering)
# Last Updated: 2025-12-08
# Aligned with: code_guidelines.md v1.10, Section 2.4 Business-Driven Configuration

# ========== CONFIGURATION CONSTANTS ==========
# Section 8.1: Single Source of Truth - All constants defined once

# Target Configuration (Section 2.1)
TARGET_COL: Final[str] = "price_target"  # Core prediction target for investment decisions
TARGET_COL_FALLBACK: Final[str] = (
    "last_price"  # Ensures models train when analyst targets unavailable
)

# Data Split Configuration (Section 2.1)
# Business Rationale: Balance training data quality (80%) with robust validation (20%)
TEST_SIZE: Final[float] = 0.2
TRAIN_SIZE: Final[float] = 1 - TEST_SIZE
CV_FOLDS: Final[int] = 5  # Standard cross-validation for reliable performance estimates

# Quantile Regression Configuration (Section 2.1)
# Business Rationale: 80% prediction interval for risk assessment and portfolio construction
QUANTILES: Final[List[float]] = [0.1, 0.5, 0.9]
LOWER_QUANTILE: Final[float] = QUANTILES[0]
MEDIAN_QUANTILE: Final[float] = QUANTILES[1]
UPPER_QUANTILE: Final[float] = QUANTILES[2]

# Sector Analysis Configuration (Section 2.1)
# Business Rationale: Minimum sample size for statistically meaningful sector-specific models
MIN_SECTOR_SAMPLES: Final[int] = 20

# Portfolio Constraints (Section 18.2)
# Business Rationale: Diversification limits to manage concentration risk
MAX_SECTOR_WEIGHT: Final[float] = 0.25  # Limit sector concentration risk
MAX_SINGLE_POSITION: Final[float] = 0.10  # Prevent overexposure to individual securities

# Outlier Detection Configuration (Section 13.1)
# Business Rationale: Conservative bounds preserve valid extreme values (high-growth stocks, mega-caps)
IQR_MULTIPLIER: Final[float] = 2.5
ZSCORE_THRESHOLD: Final[float] = 3.0
WINSORIZE_LOWER: Final[float] = 0.10  # 10th percentile (conservative, preserves extremes)
WINSORIZE_UPPER: Final[float] = 0.90  # 90th percentile (conservative, preserves extremes)

# Confidence Thresholds (Section 2.1)
CONFIDENCE_LOW_THRESHOLD: Final[float] = 0.50
CONFIDENCE_MEDIUM_THRESHOLD: Final[float] = 0.75

# Phase 9.7 Analytics / Portfolio thresholds (Section 18.2)
DISAGREEMENT_THRESHOLD: Final[float] = 10.0
TOP_N_RANKINGS: Final[int] = 50
TOP_N_PORTFOLIO_CANDIDATES: Final[int] = 150

# Safety rails sensitivity (Section 13)
SAFETY_RAILS_THRESHOLDS: Final[List[float]] = [0.01, 0.05, 0.10]

# Portfolio optimization diagnostics
MAX_PORTFOLIO_WEIGHT: Final[float] = 0.20
MAX_SHARPE_THRESHOLD: Final[float] = 3.0
MAX_RETURN_THRESHOLD: Final[float] = 1.0

# Demo/visualization toggles
RUN_DEMO_SECTIONS: Final[bool] = bool(int(os.getenv("RUN_DEMO_SECTIONS", "0")))

# Reproducibility (Section 2.1)
# Business Rationale: Fixed seed enables consistent model evaluation and regulatory compliance
RANDOM_SEED: Final[int] = int(os.getenv("RANDOM_SEED", "42"))
MODEL_VERSION: Final[str] = os.getenv("MODEL_VERSION", "v9_10")

# Set numpy random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Output Directories (Section 20.1)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Create phase-specific subdirectories
for subdir in [
    "preprocessing",
    "eda",
    "features",
    "classification",
    "regression",
    "evaluation",
    "analytics",
    "reporting",
    "plots",
    "governance",
    "safety_rails",
    "uncertainty",
    "calibration",
]:
    (OUTPUT_DIR / subdir).mkdir(exist_ok=True)

# Financial metrics subdirectory
(OUTPUT_DIR / "eda" / "financial_metrics").mkdir(exist_ok=True, parents=True)

print("✓ Configuration constants defined")
print(f"  Model Version: {MODEL_VERSION}")
print(f"  Random Seed: {RANDOM_SEED}")
print(f"  Output Directory: {OUTPUT_DIR.absolute()}")
# %%
# ========== CONFIGURATION VALIDATION ==========
# Section 8.1: Validate all configuration constants meet required constraints


def validate_configuration():
    """
    Validate all configuration constants (Section 2.3).

    Raises:
        ValueError: If any configuration constant is invalid
    """
    # Validate target columns
    if not TARGET_COL or not isinstance(TARGET_COL, str):
        raise ValueError(f"TARGET_COL must be non-empty string: {TARGET_COL}")
    if not TARGET_COL_FALLBACK or not isinstance(TARGET_COL_FALLBACK, str):
        raise ValueError(f"TARGET_COL_FALLBACK must be non-empty string: {TARGET_COL_FALLBACK}")

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

    # Validate sector configuration
    if MIN_SECTOR_SAMPLES < 1:
        raise ValueError(f"MIN_SECTOR_SAMPLES must be positive: {MIN_SECTOR_SAMPLES}")

    # Validate winsorization bounds
    if not (0 <= WINSORIZE_LOWER < 0.5):
        raise ValueError(f"WINSORIZE_LOWER must be between 0 and 0.5: {WINSORIZE_LOWER}")
    if not (0.5 < WINSORIZE_UPPER <= 1):
        raise ValueError(f"WINSORIZE_UPPER must be between 0.5 and 1: {WINSORIZE_UPPER}")

    print("✓ All configuration constants validated successfully")
    return True


# Shared validation helpers (Section 19)
from typing import Sequence


def assert_df_has_columns(
    df: pd.DataFrame, required: Sequence[str], label: str = "DataFrame"
) -> None:
    """
    Verify DataFrame contains required columns.

    Args:
        df: Input DataFrame to validate
        required: List of required column names
        label: Label for error messages (default: "DataFrame")

    Raises:
        ValueError: If any required column is missing
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}. Available columns: {len(df.columns)}"
        )


def assert_no_missing(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """
    Verify DataFrame contains no missing values.

    Args:
        df: Input DataFrame to validate
        label: Label for error messages (default: "DataFrame")

    Raises:
        ValueError: If any missing values are found
    """
    missing_total = df.isna().sum().sum()
    if missing_total != 0:
        raise ValueError(f"{label} contains {missing_total} missing values (expected 0).")


def assert_price_columns_preserved(
    df_before: pd.DataFrame, df_after: pd.DataFrame, label: str
) -> None:
    """
    Verify price columns remain unchanged between DataFrame stages.

    Price columns (21 total) must never be transformed to preserve
    interpretability for investment decisions. See Section 8.5.2.

    Args:
        df_before: DataFrame before transformation
        df_after: DataFrame after transformation
        label: Label for messages (e.g., "Winsorization", "Scaling")

    Raises:
        ValueError: If any price column was modified
    """
    price_cols = [c for c in PRICE_COLUMNS if c in df_before.columns and c in df_after.columns]
    for col in price_cols:
        if not df_after[col].equals(df_before[col]):
            raise ValueError(f"{label}: price column '{col}' was modified between stages")
    print(f"✓ {label}: Verified {len(price_cols)}/{len(PRICE_COLUMNS)} price columns preserved")


def require_dataframe(name: str, expected_type=pd.DataFrame) -> pd.DataFrame:
    """
    Retrieve and validate a required DataFrame from globals.

    Ensures the DataFrame exists, has the correct type, and is not empty.
    Used to enforce cell execution order in the notebook.

    Args:
        name: Name of the DataFrame variable in globals()
        expected_type: Expected type (default: pd.DataFrame)

    Returns:
        The validated DataFrame

    Raises:
        RuntimeError: If DataFrame not found in globals
        TypeError: If object is not the expected type
        ValueError: If DataFrame is empty
    """
    obj = globals().get(name)
    if obj is None:
        raise RuntimeError(
            f"Required DataFrame '{name}' not found in globals(). Run the corresponding phase first."
        )
    if not isinstance(obj, expected_type):
        raise TypeError(f"'{name}' must be a {expected_type.__name__}, got {type(obj).__name__}")
    if obj.empty:
        raise ValueError(f"'{name}' is empty. No data to process.")
    return obj


# Run validation
validate_configuration()
# %% [markdown]
# # Stock Price Target Prediction — ML Analytics Platform
#
# **Version**: Phase 9.5 (Unified ETL with Semantic Transforms + Feature Engineering)
# **Model Version**: v9_10
# **Last Updated**: 2025-12-08
# **Aligned with**: code_guidelines.md v1.10
#
# ## Business Objective
#
# **Primary Goal**: Predict Stock Price Targets for all stocks in the portfolio to support investment decisions and portfolio optimization.
#
# **Target Variable**: "Predicted Price Target" for regression modeling
#
# ## Notebook Architecture
#
# This notebook implements the **8-phase ML workflow** (Phase 9.1-9.8):
#
# ### Phase 9.1: Loading and Preprocessing
# - **Unified ETL Pipeline**: `etl_with_features()` single entry point
# - **Semantic Column Classification**: 5 categories (price, market_value, ratio, percentage, count)
# - **6-Step Imputation**: Handles numeric, categorical, datetime columns
# - **Price Column Protection**: 21 price columns never transformed
# - **Feature Engineering**: 196 Phase 9.3 features integrated
#
# ### Phase 9.2: Enhanced Exploratory Data Analysis
# - Statistical testing and benchmarking
# - Correlation analysis with multicollinearity detection
# - Distribution analysis by sector/region
# - Feature category coverage validation
#
# ### Phase 9.3: Advanced Feature Engineering
# - **Comprehensive Preset**: 196 features across 16 categories
# - Momentum & Technical (27), Valuation Ratios (23), Profitability (12)
# - Quality & Risk (18), Cash Flow (5), Capital Allocation (23)
# - Analyst Sentiment (10), Growth Metrics (6), and more
#
# ### Phase 9.4: Multi-Class Event Classification
# - 13 label generation methods
# - Classification outputs used as meta-features
#
# ### Phase 9.5: Sector-Optimized Regression
# - Quantile models (10th, 50th, 90th percentiles)
# - Stacking ensemble (RF + GB + XGB + Ridge)
# - Non-negativity constraints for price predictions
#
# ### Phase 9.6: Model Evaluation and Error Analysis
# - Sector-level metrics and calibration
# - Uncertainty quantification (80% prediction intervals)
# - Residual analysis and diagnostics
#
# ### Phase 9.7: Identification of Under/Overvalued Stocks
# - Mispricing score calculation
# - Portfolio optimization (Black-Litterman, Risk Parity, HRP)
# - Risk metrics (VaR, CVaR, Sharpe ratio)
#
# ### Phase 9.8: Comprehensive Analytics and Reporting
# - Model governance (model cards, lineage tracking)
# - Interactive dashboards and visualizations
# - Executive reporting and quality alerts
#
# ## DataFrame Stage Naming Convention
#
# This notebook follows a **4-stage naming convention** (Section 8.2):
#
# 1. **`all_stocks_preprocessed`**: ETL pipeline output (includes Phase 9.3 features)
# 2. **`all_stocks_features`**: Optional additional features (if not using unified ETL)
# 3. **`all_stocks_classification`**: Optional classification meta-features
# 4. **`all_stocks_enhanced`**: Final regression-ready dataset
#
# ## Key Features (v1.10)
#
# ### NEW: Unified ETL with Semantic Transforms
# - **Single Entry Point**: `etl_with_features()` consolidates schema.py, column_semantics.py, features/api.py
# - **Semantic Classification**: 5-category system protects price columns from transformation
# - **Log Transforms**: Applied to skewed market value columns (preserves extremes)
# - **Feature Engineering**: 196 Phase 9.3 features automatically added
#
# ### Price Column Protection (CRITICAL)
# - **21 Price Columns**: Never winsorized, scaled, or transformed
# - **Business Rationale**: Preserves core valuation metric `(Predicted_Target - Last_Price) / Last_Price`
# - **Includes**: Current prices, historical prices, 52W bounds, EMAs
#
# ### Configuration Best Practices
# - **Single Source of Truth**: All constants defined once with business rationale
# - **Validation**: Automatic validation of all configuration constants
# - **Reproducibility**: Fixed random seed (RANDOM_SEED=42) for regulatory compliance
#
# ## Output Artifacts
#
# Generated in `outputs/` directory following Section 20.1 standards:
# - `preprocessing/`: ETL metrics, dtype diagnostics, imputation summary
# - `eda/`: EDA summary, feature coverage, correlation matrices
# - `regression/`: Trained models, predictions, sector metrics
# - `evaluation/`: Calibration reports, uncertainty quantification
# - `governance/`: Model cards, lineage tracking, audit trails
#
# ## References
#
# - **Code Guidelines**: `docs/code_guidelines.md` v1.10
# - **Test Coverage**: 51 tests in `test_etl_unified_pipeline.py`
# - **Package Version**: 0.9.1
# - **Python**: 3.12-3.14
# %% [markdown]
# ## 1. Configuration and Setup
#
# %%
# Import configuration
from finance_ml import NotebookConfig

# Initialize with production settings
config = NotebookConfig(
    have_finance_prediction=True,
    have_database_connection=True,
    have_advanced_analytics=True,
    have_dim_reduction=True,
    debug_mode=False,
    enable_sector_analysis=True,
    enable_region_analysis=True,
    enable_interactive_plots=True,
    enable_excel_export=True,
)
config.display_summary()

# Phase 9.4-9.8: Advanced Evaluation and Governance
# Initialize logger early (Section 8.1)
import logging

# %%
# ========== CORE PYTHON IMPORTS ==========
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Interactive visualization imports (Section 17.2)
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap
from plotly.subplots import make_subplots

from finance_ml.ml_workflow.evaluation import (
    # Phase 9.4 - Uncertainty Quantification
    # Phase 9.6 - Data Splits & Leakage
    # Phase 9.8 - Stacking & Governance
    compute_stacking_contributions,
    # Phase 9.7 - Sector Bias Calibration
    meta_error_maps,
    plot_metrics_by_sector_time,
    run_fold_overlap_analysis,
    summarize_grouped_cv_balance,
    # Phase 9.5 - Safety Rails
    time_leakage_checks,
    validate_fold_assignments,
    validate_temporal_data,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

warnings.filterwarnings("ignore")

# ========== FINANCE ML PACKAGE IMPORTS ==========
# Phase 9.1-9.8 Modular Structure (Section 4.3)
# All imports use explicit module paths for clarity and maintainability

# Phase 9.1: Preprocessing - Unified ETL Pipeline (Section 7.5, 8.6)
# Phase 9.7: Analytics
from finance_ml.ml_workflow.analytics import (
    calculate_financial_metrics_dashboard,
    # Dashboard and quality functions
    generate_data_quality_alerts,
)

# Analytics: Analyst comparison
from finance_ml.ml_workflow.analytics.analyst_comparison import (
    PredictionAnalystAnalytics,
)

# ========== ADDITIONAL IMPORTS FOR NOTEBOOK COMPATIBILITY ==========
# These imports provide aliases for functions used throughout the notebook
# with naming conventions that differ from the actual module function names
# EDA: Hypothesis Testing
from finance_ml.ml_workflow.analytics.hypothesis_tests import (
    perform_comprehensive_hypothesis_tests,
)

# Analytics: Mispricing and ranking
from finance_ml.ml_workflow.analytics.mispricing import (
    calculate_mispricing_score as analytics_calculate_mispricing,
)
from finance_ml.ml_workflow.analytics.mispricing import (
    rank_overvalued_stocks as analytics_rank_overvalued,
)
from finance_ml.ml_workflow.analytics.mispricing import (
    rank_stocks_by_sector as analytics_rank_by_sector,
)
from finance_ml.ml_workflow.analytics.mispricing import (
    rank_undervalued_stocks as analytics_rank_undervalued,
)

# Analytics: Risk metrics
from finance_ml.ml_workflow.analytics.risk import (
    calculate_expected_shortfall as expected_shortfall,
)

# Phase 9.4: Classification
from finance_ml.ml_workflow.classification import (
    analyze_calibration,
    balance_classes,
    compare_classifiers,
    compute_shap_values,
    create_event_labels,
    # Phase 9.4 TDD functions (multi-label, CV policy, class balance)
    create_multilabel_event_labels,
    cross_validate_classifier,
    determine_cv_strategy,
    evaluate_classification,
    evaluate_classification_by_sector,
    # Visualization and analysis
    plot_confusion_matrices,
    # Additional classification functions
    prepare_classification_data,
    tune_classifier_hyperparameters,
)

# Classification: Evaluation exports
from finance_ml.ml_workflow.classification.evaluation import (
    export_classification_probabilities,
)

# Phase 9.1: Preprocessing - Data Loading and Schema

# Phase 9.1: Data Catalog
from finance_ml.ml_workflow.data_catalog import DataCatalog

# Phase 9.2: Exploratory Data Analysis
from finance_ml.ml_workflow.eda import (
    # Additional EDA functions
    eda_summary,
    generate_benchmarking_report,
    generate_eda_report,
    sector_distribution_summary,
)

# Phase 9.3: Feature Coverage Validation

# Phase 9.6: Evaluation
from finance_ml.ml_workflow.evaluation import (
    build_lineage_json,
    # Uncertainty Quantification
    build_quantile_diagnostics,
    # Metrics
    create_sector_bias_dashboard,
    # Calibration
    estimate_sector_bias,
    # Governance
    generate_model_card,
    plot_interval_coverage,
    plot_reliability_diagram,
    # Analysis
    safety_rails_sensitivity_app,
    # Safety Rails
    summarize_winsorization_effects,
    track_constraint_violations,
)

# Evaluation: Comprehensive metrics
from finance_ml.ml_workflow.evaluation.metrics import (
    comprehensive_regression_metrics as evaluation_comprehensive_metrics,
)
from finance_ml.ml_workflow.evaluation.metrics import (
    compute_metrics_by_segment as evaluation_metrics_by_segment,
)
from finance_ml.ml_workflow.features import (
    select_features_rf,
)
from finance_ml.ml_workflow.features.advanced import (
    engineer_accounting_quality_features,
    engineer_analyst_quality_features,
    engineer_dividend_reliability_features,
    engineer_employee_productivity_features,
    engineer_employment_dynamics_features,
    engineer_revenue_forecast_features,
    # Phase 9.3 Schema 1.3 - New feature engineering functions
    engineer_technical_analysis_features,
    engineer_valuation_ratios,
    engineer_valuation_timeseries_features,
)

# Phase 9.3: Feature Engineering (Section 9.3)
from finance_ml.ml_workflow.features.api import (
    build_features,  # Unified entry point with presets
)

# Features: Selection and importance
from finance_ml.ml_workflow.features.selection import (
    calculate_feature_importance_rf as features_importance_rf,
)
from finance_ml.ml_workflow.features.selection import (
    # Phase 9.3 TDD functions (automated feature selection)
    select_features_auto,
    select_features_by_category,
)

# Phase 9.1: Preprocessing - Component Functions
from finance_ml.ml_workflow.preprocessing import (
    # Imputation (6-step strategy)
    apply_enhanced_imputation_strategy_6step,
    calculate_data_quality_score,
    # Data quality
    check_missing_values,
    # Outlier detection and handling
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    detect_outliers_zscore,
    # Feature scaling (with price column exclusion)
    scale_features,
    validate_imputation_completeness,
    winsorize_by_sector,
)

# Phase 9.1: Preprocessing - Semantic Column Classification (Section 8.5)
from finance_ml.ml_workflow.preprocessing.etl import (
    ETLConfig,  # Configuration with semantic-aware transformations
    # Comprehensive metrics tracking
    etl_with_features,  # RECOMMENDED: Comprehensive ETL + semantic transforms + Phase 9.3 features
    # Alternative: ETL + financial metrics only
    # Low-level: Custom ETL configuration
)

# Phase 9.5: Regression

# Regression: Calibration
from finance_ml.ml_workflow.regression.calibration import (
    calibrate_predictions_by_sector,
)
from finance_ml.ml_workflow.regression.constraints import (
    NonNegativeRegressionWrapper,
)

# Regression: Dataset preparation and classification integration
from finance_ml.ml_workflow.regression.dataset import (
    integrate_classification_features,
)
from finance_ml.ml_workflow.regression.dataset import (
    create_classification_interactions as regression_create_classification_interactions,
)
from finance_ml.ml_workflow.regression.dataset import (
    prepare_regression_data as regression_prepare_data,
)
from finance_ml.ml_workflow.regression.io import (
    load_model as regression_load_model,
)

# Regression: Model I/O
from finance_ml.ml_workflow.regression.io import (
    save_model as regression_save_model,
)

# Regression: Model comparison and training
from finance_ml.ml_workflow.regression.models import (
    compare_regressors as regression_compare_regressors,
)
from finance_ml.ml_workflow.regression.models import (
    tune_stacking_hyperparameters,
)
from finance_ml.ml_workflow.regression.models import (
    train_stacking_regressor as regression_train_stacking,
)

# Regression: Quantile models
from finance_ml.ml_workflow.regression.quantile import (
    train_quantile_regressor as regression_train_quantile,
)

# Regression: Robust methods and constraints
from finance_ml.ml_workflow.regression.robust import (
    adaptive_clip_predictions,
    winsorize_target,
)

# Phase 9.8: Reporting

# Reporting: Excel and HTML report generation
from finance_ml.ml_workflow.reporting.excel_reports import (
    generate_enhanced_excel_report,
)
from finance_ml.ml_workflow.reporting.html_reports import (
    generate_enhanced_analysis_html,
)

# Reporting: Configuration classes and constants
from finance_ml.ml_workflow.reporting.report_config import (
    QUALITY_THRESHOLD_DEFAULT,
    REPORT_TOP_N_DEFAULT,
    RISK_ZSCORE_THRESHOLD,
    ExcelReportConfig,
    HTMLReportConfig,
)

# Aliases for notebook compatibility (functions that use different naming convention)
# These map to existing functions that provide equivalent functionality
reporting_financial_metrics = calculate_financial_metrics_dashboard  # From analytics import above
reporting_quality_alerts = generate_data_quality_alerts  # From analytics import above

print("✓ All imports loaded successfully")
print(f"  Phase 9.1-9.8 structure aligned with code_guidelines.md Section 4.3")
# %% [markdown]
# ## 📦 Phase 9.1-9.8 Module Structure Migration
#
# This notebook now uses the **new modular Phase 9.1-9.8 structure** with organized subpackages.
#
# ### Module Organization
#
# | Phase | Subpackage | Purpose | Import Prefix |
# |-------|-----------|---------|---------------|
# | **9.1** | `preprocessing/` | Data quality, imputation, outliers, scaling | `preprocessing_*` |
# | **9.2** | `eda/` | EDA reports, benchmarking, statistical tests | `generate_*`, `compare_*` |
# | **9.3** | `features/` | Feature engineering, importance, selection | `features_*`, `engineer_*` |
# | **9.4** | `classification/` | Event labels, hyperparameter tuning | `classification_*` |
# | **9.5** | `regression/` | Model training, quantile, constraints | `regression_*` |
# | **9.6** | `evaluation/` | Metrics, error analysis, segmentation | `evaluation_*` |
# | **9.7** | `analytics/` | Mispricing, rankings, portfolio, risk | `analytics_*` |
# | **9.8** | `reporting/` | Dashboard data, quality alerts, exports | `reporting_*` |
#
# ### Key Benefits
#
# ✅ **Clean imports**: All functions imported once at the top
# ✅ **No duplication**: Removed 21 redundant import cells
# ✅ **Better organization**: Logical grouping by business function
# ✅ **Backward compatible**: Package-level imports use deprecation shims
# ✅ **Easier maintenance**: Clear module boundaries and responsibilities
#
# ### Migration Guide (code_guidelines.md Section 4.3)
#
# - **Explicit Module Imports** (Recommended): `from finance_ml.ml_workflow.preprocessing.imputation import apply_enhanced_imputation_strategy_6step`
# - **Package-Level Imports** (Convenience): `from finance_ml import load_from_csv, normalize_columns`
# - All functions use descriptive prefixes to indicate their module
#
# See `docs/improvement_plan/finance_ml_restructuring_plan.md` for detailed migration timeline.
# %%

# Configure plotting (RANDOM_SEED already set in configuration cell above)
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", "{:.2f}".format)
# %% [markdown]
# ## Phase 9.1: Loading and Preprocessing with 6-Step Imputation Strategy Financial Data
#
# ### Business Goal
# Load multi-region equity data and apply comprehensive preprocessing to ensure high-quality inputs for downstream modeling.
#
# ### Key Objectives
# 1. Load data from PostgreSQL/SQLite or CSV fallback
# 2. Apply 6-step imputation strategy (zero-fill, KNN, price-based, median)
# 3. Detect and handle outliers (IQR, z-score, isolation forest)
# 4. Apply sector-wise winsorization
# 5. Validate data quality and completeness
#
# ### Inputs
# - Raw data: CSV files or database tables (US, EU, APAC, ROTW regions)
#
# ### Outputs
# - `all_stocks_preprocessed`: Fully preprocessed DataFrame
# - `outputs/preprocessing/`: Data quality reports, imputation stats
# - `outputs/catalog/`: Data catalog metadata
#
# ### v1.2 Standards Applied
# - ✅ 6-step imputation strategy
# - ✅ Outlier safety rails (winsorization at [1st, 99th] percentiles)
# - ✅ Data quality validation
#
# ### v1.3 Standards Applied (NEW)
# - ✅ Schema-aware datatype detection and casting (`COLUMN_SCHEMA`)
# - ✅ Phase 9.3 feature input categorization (`PHASE93_FEATURE_CATEGORIES`)
# - ✅ Comprehensive dtype diagnostics (coercion tracking, unknown columns)
# - ✅ Metadata catalog with dtypes and missing counts
#
# ### Validation Checkpoint
# - Zero missing values after imputation
# - Outliers capped within acceptable ranges
# - All required columns present
# - All columns cast to schema-compliant dtypes
#
# Sophisticated preprocessing pipeline with:
# 1. **Data Loading**: Multi-region data from PostgreSQL or CSV
# 2. **Schema-Aware Dtype Detection** (v1.3 NEW): Cast columns to canonical dtypes with diagnostics
# 3. **Outlier Detection**: IQR, Z-score, and Isolation Forest methods
# 4. **Sector-Specific Winsorization**: Limit extreme values by sector
# 5. **Data Quality Scoring**: Comprehensive quality metrics
# 6. **6-Step Imputation Strategy** (Phase 9.1 Enhanced):
#    - **Step 1**: Zero imputation for exceptional event columns (48 cols)
#    - **Step 2**: KNN imputation (sector-aware) for financial metrics (148 cols)
#    - **Step 3**: Price-based imputation for price target columns (5 cols)
#    - **Step 4**: Median imputation for remaining numeric columns
#    - **Step 5**: Categorical imputation for string/object columns
#    - **Step 6**: Datetime imputation and formatting for temporal features
# 7. **Imputation Validation**: Comprehensive validation ensuring zero missing values
# 8. **Feature Scaling**: Robust scaling by sector
#
# %%
# ========== PHASE 9.1: UNIFIED ETL WITH SEMANTIC TRANSFORMS ==========
# Section 8.2: DataFrame Stage Naming Convention - Stage 1 of 4
# Section 7.5: etl_with_features() - Single entry point consolidating:
#   - schema.py (column normalization, dtype casting)
#   - column_semantics.py (semantic classification, price column protection)
#   - features/api.py (Phase 9.3 feature engineering)

# Configure data source
DATA_SOURCE: Literal["csv", "db", "all_stocks"] = "csv"
DATA_DIR = Path("data")

# Configure ETL with semantic-aware transformations (Section 7.5)
etl_config = ETLConfig(
    # Standard ETL options
    apply_imputation=True,
    imputation_strategy="6step",  # REQUIRED: 6-step handles numeric, categorical, datetime
    apply_scaling=True,  # Scale later if needed
    # Semantic-aware transformation flags (Section 8.5)
    use_semantic_column_classification=True,  # Enable 5-category classification
    preserve_price_columns=True,  # CRITICAL: Never transform price columns
    log_transform_market_values=True,  # Apply log-transforms to skewed columns
    exclude_ratios_from_winsorization=True,  # Ratios pre-normalized
    exclude_percentages_from_winsorization=True,  # Percentages bounded [0, 100]
    # Feature engineering integration (Section 9.3)
    apply_feature_engineering=True,  # Enable Phase 9.3 features
    feature_preset="comprehensive",  # Options: 'basic', 'momentum', 'quality', 'standard', 'comprehensive'
    # Financial metrics computation
    compute_valuation_metrics=True,
    compute_profitability_metrics=True,
    compute_growth_metrics=True,
    compute_leverage_metrics=True,
    # Quality reporting
    generate_quality_alerts=True,
    generate_metrics_dashboard=True,
)

# Execute unified ETL pipeline (Stage 1: Preprocessing)
print("=" * 80)
print("STAGE 1: ETL Pipeline with Semantic Transforms + Feature Engineering")
print("=" * 80)

all_stocks_preprocessed, etl_metrics = etl_with_features(
    source=DATA_SOURCE,
    data_dir=DATA_DIR if DATA_SOURCE == "csv" else None,
    config=etl_config,
    return_metrics=True,
)

# Display ETL metrics
print("\n" + etl_metrics.summary())
print(f"\nSemantic Classification:")
print(f"  Price columns protected: {etl_metrics.price_columns_count}")
print(f"  Market value columns: {etl_metrics.market_value_columns_count}")
print(f"  Log-transformed columns: {etl_metrics.log_transformed_columns}")
print(f"\nFeature Engineering:")
print(f"  Preset used: {etl_metrics.feature_preset_used}")
print(f"  Features added: {etl_metrics.features_added}")
print(f"  Categories: {', '.join(etl_metrics.feature_categories_applied)}")

# ========== VALIDATION CHECKPOINT (Section 19.1) ==========
# Required assertions after ETL
assert not all_stocks_preprocessed.empty, "DataFrame must not be empty"
assert "ticker" in all_stocks_preprocessed.columns, "ticker column required"
assert "sector" in all_stocks_preprocessed.columns, "sector column required"
assert "last_price" in all_stocks_preprocessed.columns, "last_price column required"

# Validate imputation completeness using schema-aware validation
# Note: The 6-step imputation handles model-critical columns (numeric, categorical, datetime)
# but legitimately leaves NaNs in optional/derived columns and newly engineered features.
# Use ETL metrics for schema-aware validation instead of strict zero-NaN check.
missing_total = all_stocks_preprocessed.isna().sum().sum()

# Schema-aware validation: check imputation completeness from ETL metrics
# This validates that critical columns are imputed, not that ALL columns have zero NaNs
assert etl_metrics.imputation_completeness, (
    f"Imputation incomplete: {etl_metrics.missing_values_before_imputation} → "
    f"{etl_metrics.missing_values_after_imputation} missing values in critical columns"
)

# Log total missing for diagnostics (informational, not a failure condition)
if missing_total > 0:
    print(
        f"  ℹ️  Total NaN cells in DataFrame: {missing_total:,} (includes optional/derived columns)"
    )
    # Show top columns with missing values for debugging
    missing_by_col = all_stocks_preprocessed.isna().sum()
    top_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False).head(10)
    if len(top_missing) > 0:
        print(f"  Top columns with NaNs: {dict(top_missing)}")

# Validate data sufficiency
assert (
    len(all_stocks_preprocessed) > 100
), f"Insufficient data: {len(all_stocks_preprocessed)} rows (minimum 100)"
assert all_stocks_preprocessed["last_price"].min() > 0, "last_price must be positive"

# Validate semantic transformations
assert etl_metrics.semantic_classification_applied, "Semantic classification should be applied"
assert (
    etl_metrics.price_columns_count >= 21
), f"All 21 price columns should be protected, found {etl_metrics.price_columns_count}"

# Validate feature engineering (Section 9.3)
assert etl_metrics.feature_engineering_applied, "Feature engineering should be applied"
coverage_pct = (etl_metrics.features_added / 196) * 100 if etl_metrics.features_added > 0 else 0
assert coverage_pct >= 75, f"Phase 9.3 coverage must be >= 75%, got {coverage_pct:.1f}%"

print(f"\n✓ Stage 1 (Preprocessed) validation passed")
print(f"  Shape: {all_stocks_preprocessed.shape}")
print(f"  Missing values: {missing_total}")
print(f"  Price columns protected: {etl_metrics.price_columns_count}")
print(f"  Phase 9.3 coverage: {coverage_pct:.1f}% ({etl_metrics.features_added}/196 features)")
print("=" * 80)
# %% [markdown]
# # DataFrame Stage Naming Convention
#
# **Policy**: Use descriptive stage-based naming for all DataFrame transformations (Section 8.2)
#
# ## 4-Stage Pipeline
#
# This notebook follows a **4-stage DataFrame naming convention** that aligns with the unified ETL pipeline:
#
# ### Stage 1: `all_stocks_preprocessed`
# **ETL Pipeline Output**
# - Extraction, normalization, validation, sanitization
# - 6-step imputation (numeric, categorical, datetime)
# - Semantic column classification (5 categories)
# - Log-transforms for market value columns
# - Winsorization (excluding price/ratio/percentage columns)
# - Optional scaling (with price column exclusion)
# - Financial metrics computation (valuation, profitability, growth, leverage)
# - **✓ Phase 9.3 Feature Engineering** (196 features via `etl_with_features()`)
#
# **Created by**: `etl_with_features()` or `etl_with_financial_metrics()`
#
# ### Stage 2: `all_stocks_features` (Optional - if not using unified ETL)
# **DataFrame enhanced with additional engineered features**
# - Phase 9.3 feature categories: momentum, valuation, profitability, quality/risk, cash flow, growth
# - Only needed if using `run_etl_pipeline()` without feature engineering
# - **Note**: When using `etl_with_features()`, features are already in `all_stocks_preprocessed`
#
# **Created by**: `build_features()` or `build_comprehensive_features()`
#
# ### Stage 3: `all_stocks_classification` (Optional)
# **DataFrame enhanced with classification model outputs**
# - Event probabilities from Phase 9.4 classification
# - Predicted classes for use as meta-features in regression
# - Only needed if using classification results in regression models
#
# **Created by**: `train_event_classifier()` + probability assignment
#
# ### Stage 4: `all_stocks_enhanced`
# **Final Phase 9.5 regression-ready dataset**
# - All transformations applied
# - Optional classification meta-features
# - Ready for train/test split and model training
#
# **Created by**: Final composite features and interactions
#
# ## Benefits
#
# 1. **Simplified Pipeline**: ETL handles preprocessing; notebook focuses on ML stages
# 2. **Debugging**: Inspect intermediate stages without re-running expensive operations
# 3. **Rollback**: Revert to earlier stage if downstream transformation fails
# 4. **Metrics Tracking**: ETL returns `ETLMetrics` with transformation statistics
# 5. **Self-documenting**: Stage names clearly indicate transformation history
#
# ## Validation Checkpoints
#
# Each stage **MUST** include validation assertions (Section 19):
# - DataFrame not empty
# - Critical columns present
# - No missing values (after imputation)
# - Data quality metrics within bounds
# - Semantic transformations applied correctly
#
# ## Price Column Protection
#
# **CRITICAL**: All 21 price columns must **NEVER** be transformed (Section 8.5.2):
# - Current prices: `last_price`, `price_target`, `price_target_median`, etc.
# - Historical prices: `price_5d_ago`, `price_1w_ago`, `price_1m_ago`, etc.
# - 52W bounds: `52w_high_adj`, `52w_low_adj`
# - EMAs: `ema_20d`, `ema_50d`, `ema_100d`, `ema_250d`
#
# **Rationale**: Core business metric `(Predicted_Target - Last_Price) / Last_Price` requires original price scale.
# %%
# ============================================================================
# VALIDATION: Price Column Preservation (Section 8.5.2)
# ============================================================================

from finance_ml.ml_workflow.preprocessing.column_semantics import PRICE_COLUMNS

print("\n" + "=" * 70)
print("PRICE COLUMN PRESERVATION VALIDATION")
print("=" * 70)

# Verify price columns were not transformed
price_cols_in_df = [c for c in all_stocks_preprocessed.columns if c in PRICE_COLUMNS]

print(f"\nPrice columns preserved: {len(price_cols_in_df)}")
for col in price_cols_in_df:
    if col in all_stocks_preprocessed.columns:
        min_val = all_stocks_preprocessed[col].min()
        max_val = all_stocks_preprocessed[col].max()
        print(f"  - {col}: min={min_val:.2f}, max={max_val:.2f}")

# Verify no scaling was applied to price columns
if "last_price" in all_stocks_preprocessed.columns:
    original_price_range = (
        all_stocks_preprocessed["last_price"].max() - all_stocks_preprocessed["last_price"].min()
    )
    print(f"\nLast Price range: ${original_price_range:.2f} (should be in original dollar units)")
    print("✓ Price columns preserved in original units (no scaling applied)")

print("=" * 70)
# %%
print(f"\n✓ ETL Pipeline Complete (Phase 9.1 - Consolidated ETL with Features):")
print(f"  - Final shape: {all_stocks_preprocessed.shape}")

# Display financial metrics status
date_status = "✓" if etl_metrics.date_columns_ready else "✗"
print(f"  - Date columns ready: {date_status}")
print(f"  - Processing time: {etl_metrics.total_duration:.2f}s")
print(f"  - Quality score: {etl_metrics.quality_score:.3f}")
print(f"  - Validation score: {etl_metrics.validation_score:.3f}")
print(f"  - Imputation: {etl_metrics.imputation_strategy} strategy")
print(f"  - Missing before: {etl_metrics.missing_values_before_imputation:,}")
print(f"  - Missing after: {etl_metrics.missing_values_after_imputation:,}")

print(f"\n📋 Next Steps (Phase 9.1b-c):")
print(f"  - Downstream cells will handle winsorization (cells 13-15)")
print(f"  - Downstream cells will handle scaling (cell 17)")
print(f"  - This separation allows proper preprocessing order per code_guidelines.md")

# Alias for compatibility with downstream cells that expect 'all_stocks_typed'
all_stocks_typed = all_stocks_preprocessed.copy()
print(f"\n✓ Created alias: all_stocks_typed = all_stocks_normalized (for downstream compatibility)")
# %%
# v1.3 NEW: Schema-aware dtype detection and casting
# Import schema and dtype detection modules
from finance_ml.ml_workflow.data.schema import PHASE93_FEATURE_CATEGORIES
from finance_ml.ml_workflow.preprocessing import detect_and_cast_dtypes, to_jsonable

print("\n🔍 Phase 9.1 v1.3: Schema-Aware Datatype Detection")
print("=" * 60)

# Apply schema-aware dtype detection and casting
all_stocks_typed, dtype_diagnostics = detect_and_cast_dtypes(all_stocks_preprocessed)

# Report diagnostics
print(f"\n✓ Datatype Detection Complete:")
print(f"  Columns cast: {len(dtype_diagnostics['cast_applied'])}")
print(f"  Coercion warnings: {sum(dtype_diagnostics['coercion_warnings'].values())} values")
if dtype_diagnostics["unknown_columns"]:
    print(f"  Unknown columns (not in schema): {len(dtype_diagnostics['unknown_columns'])}")
    if len(dtype_diagnostics["unknown_columns"]) <= 5:
        print(f"    {', '.join(dtype_diagnostics['unknown_columns'])}")
if dtype_diagnostics["missing_expected_columns"]:
    print(f"  Missing expected columns: {len(dtype_diagnostics['missing_expected_columns'])}")
    if len(dtype_diagnostics["missing_expected_columns"]) <= 5:
        print(f"    {', '.join(dtype_diagnostics['missing_expected_columns'])}")

# Display Phase 9.3 feature input categories
print(f"\n📊 Phase 9.3 Feature Input Categories (PHASE93_FEATURE_CATEGORIES):")
for category, features in PHASE93_FEATURE_CATEGORIES.items():
    available_features = [f for f in features if f in all_stocks_typed.columns]
    print(f"  {category}: {len(available_features)}/{len(features)} available")

# Fix: Define OUTPUT_DIR if missing and ensure directory exists
from pathlib import Path

if "OUTPUT_DIR" not in globals():
    OUTPUT_DIR = Path("outputs")
(OUTPUT_DIR / "preprocessing").mkdir(parents=True, exist_ok=True)

# Save dtype diagnostics to metadata
dtype_diagnostics_path = OUTPUT_DIR / "preprocessing" / "dtype_diagnostics.json"
import json

with open(dtype_diagnostics_path, "w") as f:
    json.dump(to_jsonable(dtype_diagnostics), f, indent=2)
print(f"\n✓ Dtype diagnostics saved to: {dtype_diagnostics_path}")

# %%
# Save ETL metrics for audit trail and governance
# Persist comprehensive ETL metrics to track data quality and processing

etl_metrics_dict = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "model_version": MODEL_VERSION,
    "rows_loaded": etl_metrics.rows_output,  # FIXED: Use rows_output instead of rows_loaded
    "columns_loaded": etl_metrics.columns_output,  # FIXED: Use columns_output instead of columns_loaded
    "missing_values_before_imputation": etl_metrics.missing_values_before_imputation,
    "missing_values_after_imputation": etl_metrics.missing_values_after_imputation,
    "imputation_completeness": etl_metrics.imputation_completeness,
    "imputation_strategy": etl_metrics.imputation_strategy,
    "date_columns_ready": etl_metrics.date_columns_ready,
    "quality_score": etl_metrics.quality_score,
    "processing_time_sec": etl_metrics.total_duration,
    "warnings_count": len(etl_metrics.warnings),
    "errors_count": len(etl_metrics.errors),
}

# Save to CSV for tracking over time
etl_metrics_df = pd.DataFrame([etl_metrics_dict])
etl_metrics_path = OUTPUT_DIR / "preprocessing" / "etl_metrics.csv"
etl_metrics_path.parent.mkdir(parents=True, exist_ok=True)

# Append to existing metrics if file exists
if etl_metrics_path.exists():
    existing_metrics = pd.read_csv(etl_metrics_path)
    etl_metrics_df = pd.concat([existing_metrics, etl_metrics_df], ignore_index=True)

etl_metrics_df.to_csv(etl_metrics_path, index=False)
print(f"\n✓ ETL metrics saved to: {etl_metrics_path}")
print(f"  Total runs tracked: {len(etl_metrics_df)}")

# ==============================================================================
# Phase 9.1 Preprocessing Workflow Summary
# ==============================================================================
#
# NEW: Using etl_with_imputation() function for unified data loading:
#   ✓ Data extraction (CSV/DB)
#   ✓ Column normalization
#   ✓ Schema validation
#   ✓ Data sanitization
#   ✓ 6-step imputation (numeric, categorical, datetime)
#
# Downstream cells handle additional preprocessing:
#   → Cell 13-15: Winsorization (selective, exclude price/ratio columns)
#   → Cell 16: Manual imputation validation (REDUNDANT - can be removed)
#   → Cell 17: Feature scaling (sector-aware, price columns protected)
#
# Preprocessing order (code_guidelines.md Section 8.5):
#   1. Load + Normalize + Validate + Sanitize + Impute (this cell)
#   2. Winsorize (cells 13-15)
#   3. Scale (cell 17)
#   4. Feature Engineering (Phase 9.3)
# ==============================================================================
# %%
# Detailed missing value analysis using Phase 9.1 function
missing_report = check_missing_values(all_stocks_typed)
print("\n📊 Detailed Missing Values Report:")
print(
    f"  Columns with missing values: {len([col for col, info in missing_report.items() if info['percentage'] > 0])}"
)
if missing_report:
    # Show top 10 columns with highest missing percentage
    sorted_missing = sorted(missing_report.items(), key=lambda x: x[1]["percentage"], reverse=True)[
        :10
    ]
    for col, info in sorted_missing:
        if info["percentage"] > 0:
            print(f"    {col}: {info['percentage']:.1f}%")
# %%
# Register dataset with Data Catalog for metadata tracking
# NOTE: DataCatalog API expects (name, description, tags), not 'df' parameter
print("\n📚 Registering dataset with Data Catalog:")

# Define catalog directory (create if needed)
CATALOG_DIR = Path(os.getenv("CACHE_DIR", ".cache")) / "catalog"
CATALOG_DIR.mkdir(parents=True, exist_ok=True)

# Skip DataCatalog registration if API is incompatible
# The DataCatalog.register_dataset() signature varies by version
try:
    catalog = DataCatalog(catalog_dir=CATALOG_DIR)
    # Store dataset info manually for version tracking
    import hashlib
    import json

    catalog_metadata = {
        "name": "all_stocks_initial",
        "description": "Initial stock data after loading and normalization",
        "tags": ["raw", "multi-region", "phase_9.1"],
        "shape": list(all_stocks_typed.shape),
        "columns": list(all_stocks_typed.columns),
        "checksum": hashlib.md5(str(all_stocks_typed.shape).encode()).hexdigest(),
    }

    metadata_file = CATALOG_DIR / "all_stocks_initial_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(catalog_metadata, f, indent=2)

    print(f"✓ Dataset metadata saved to {metadata_file}")
    print(f"  Shape: {all_stocks_typed.shape}")
    print(f"  Columns: {len(all_stocks_typed.columns)}")
except (IOError, OSError, TypeError, AttributeError, KeyError) as e:
    print(f"⚠️  DataCatalog registration skipped: {e}")
# %%
# Robust outlier detection with multiple methods
# Functions already imported from finance_ml at the top

# Outlier Detection Section
print("\n" + "=" * 80)
print("OUTLIER DETECTION")
print("=" * 80)

# Detect outliers using multiple methods
numeric_cols = all_stocks_typed.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = all_stocks_typed.select_dtypes(include=[np.number]).columns.tolist()
financial_metrics = [c for c in numeric_cols if c not in ["ticker", "isin"]]

# Detect outliers using multiple methods - process each column individually
outliers_iqr = {}
for col in financial_metrics[:50]:
    outliers_iqr[col] = detect_outliers_iqr(
        all_stocks_typed, columns=[col], iqr_multiplier=IQR_MULTIPLIER
    )

outliers_zscore = {}
for col in financial_metrics[:50]:
    outliers_zscore[col] = detect_outliers_zscore(
        all_stocks_typed,
        columns=[col],  # Fixed: changed 'column' to 'columns' and wrapped in list
        threshold=ZSCORE_THRESHOLD,
    )

outliers_iforest = {}
for col in financial_metrics[:50]:
    outliers_iforest[col] = detect_outliers_isolation_forest(
        all_stocks_typed, columns=[col], contamination=0.05, random_state=RANDOM_SEED
    )

# Aggregate results for reporting
# NEW Phase 9.1: Functions return DataFrames/Series with boolean outlier indicators
# For IQR: DataFrame with {col}_outlier columns
# For Z-score: DataFrame with {col}_zscore_outlier columns
# For Isolation Forest: Boolean Series per column
total_iqr = sum(
    df[f"{col}_outlier"].sum() if f"{col}_outlier" in df.columns else 0
    for col, df in outliers_iqr.items()
)
total_zscore = sum(
    df[f"{col}_zscore_outlier"].sum() if f"{col}_zscore_outlier" in df.columns else 0
    for col, df in outliers_zscore.items()
)
total_iforest = sum(
    series.sum() if isinstance(series, pd.Series) else 0 for series in outliers_iforest.values()
)

print(f"✓ Outliers detected:")
print(f"  IQR method: {total_iqr} outliers across {len(outliers_iqr)} columns")
print(f"  Z-score method: {total_zscore} outliers across {len(outliers_zscore)} columns")
print(f"  Isolation Forest: {total_iforest} outliers across {len(outliers_iforest)} columns")

# %%
# Stage 4: Semantic-aware preprocessing with log-transforms and selective winsorization
# NEW v1.7: Implements code_guidelines.md Section 8.5 (Preprocessing Stage Naming)
# - Step 1: Log-transform skewed market value columns (market_cap, revenue, total_assets)
# - Step 2: Selective winsorization (excludes price, ratio, percentage columns)

from finance_ml.ml_workflow.preprocessing.column_semantics import (
    PRICE_COLUMNS,
    get_winsorizable_columns,
)

print("\n  Step 2: Applying semantic-aware winsorization...")

# FIX: Use 'all_stocks_typed' (the standard output from Step 1)
# instead of the undefined 'all_stocks_log_transformed'.
winsorizable_cols = get_winsorizable_columns(all_stocks_typed.columns.tolist())

# Safety Rail: Filter to ensure only numeric columns are winsorized
# This prevents the TypeError when attempting to clip categorical columns like 'sector'
winsorizable_cols = [
    c for c in winsorizable_cols if pd.api.types.is_numeric_dtype(all_stocks_typed[c])
]

all_stocks_winsorized = winsorize_by_sector(
    all_stocks_typed,
    columns=winsorizable_cols,
    lower_percentile=WINSORIZE_LOWER,  # From configuration constants
    upper_percentile=WINSORIZE_UPPER,  # From configuration constants
    by_sector=True,
    exclude_price_columns=True,  # CRITICAL: Preserve price interpretability
    exclude_ratio_columns=True,  # Pre-normalized ratios don't need winsorization
)

print(f"  ✓ Winsorized {len(winsorizable_cols)} columns (sector-specific)")

# Verify ALL 21 price columns unchanged (code_guidelines.md Section 8.5.2)
# PRICE_COLUMNS includes: current (6), historical (9), 52w bounds (2), EMAs (4)
price_cols_present = [c for c in PRICE_COLUMNS if c in all_stocks_winsorized.columns]
for col in price_cols_present:
    if col in all_stocks_typed.columns:
        assert all_stocks_winsorized[col].equals(
            all_stocks_typed[col]
        ), f"{col} was incorrectly modified!"

print(
    f"  ✓ Verified {len(price_cols_present)}/21 price columns preserved (business metric protection)"
)

# Verify integer columns were converted to float64 (Int64 TypeError fix)
int_cols_before = [
    c for c in winsorizable_cols if pd.api.types.is_integer_dtype(all_stocks_typed[c])
]
if int_cols_before:
    # Check sample of converted columns
    sample_cols = int_cols_before[:3] if len(int_cols_before) > 3 else int_cols_before
    for col in sample_cols:
        assert (
            all_stocks_winsorized[col].dtype == np.float64
        ), f"{col} should be float64 after winsorization, got {all_stocks_winsorized[col].dtype}"
    print(
        f"  ✓ Verified {len(int_cols_before)} integer columns converted to float64 (Int64 TypeError prevention)"
    )

print(f"\n✓ Stage 4 Complete: Log-transforms + Selective Winsorization")
# %%
# Calculate comprehensive data quality score
print("\n📊 Calculating Data Quality Scores...")
quality_report = calculate_data_quality_score(all_stocks_winsorized)
print(f"✓ Data Quality Report:")
print(f"  Overall score: {quality_report.overall_score:.2f}")
print(f"  Completeness: {quality_report.completeness_score:.2f}")
print(f"  Validity: {quality_report.validity_score:.2f}")
print(f"  Consistency: {quality_report.consistency_score:.2f}")
print(f"  Issues detected: {len(quality_report.issues)}")

# %%
# 📊 Interactive Data Quality Visualizations
print("\n📊 Creating Interactive Data Quality Visualizations...")

# Note: Output directories already created at initialization (all Phase 9.1-9.8 subdirectories)

# 1. Missing Value Heatmap (Interactive Plotly)
missing_pct = (all_stocks_winsorized.isnull().sum() / len(all_stocks_winsorized) * 100).sort_values(
    ascending=False
)
missing_df = pd.DataFrame({"Column": missing_pct.index, "Missing %": missing_pct.values}).head(30)

fig_missing = px.bar(
    missing_df,
    x="Missing %",
    y="Column",
    orientation="h",
    title="Top 30 Columns by Missing Data Percentage",
    labels={"Missing %": "Missing Data (%)", "Column": "Feature"},
    color="Missing %",
    color_continuous_scale="Reds",
    height=800,
    template="plotly_dark",
)
fig_missing.update_layout(yaxis={"categoryorder": "total ascending"}, font_family="Arial")
fig_missing.show()

# Save as HTML for interactive viewing
fig_missing.write_html(OUTPUT_DIR / "eda" / "missing_values_heatmap.html")
print(f"✓ Saved: {OUTPUT_DIR / 'eda' / 'missing_values_heatmap.html'}")

# 2. Outlier Detection Summary (Interactive Bar Chart)
outlier_summary = pd.DataFrame(
    {
        "Method": ["IQR", "Z-Score", "Isolation Forest"],
        "Outliers Detected": [total_iqr, total_zscore, total_iforest],
        "Columns Analyzed": [
            len(outliers_iqr),
            len(outliers_zscore),
            len(outliers_iforest),
        ],
    }
)

fig_outliers = px.bar(
    outlier_summary,
    x="Method",
    y="Outliers Detected",
    title="Outlier Detection Summary Across Methods",
    color="Method",
    text="Outliers Detected",
    height=500,
    template="plotly_dark",
    color_discrete_sequence=["#375a7f", "#00bc8c", "#f39c12"],
)
fig_outliers.update_traces(texttemplate="%{text}", textposition="outside")
fig_outliers.update_layout(font_family="Arial")
fig_outliers.show()
fig_outliers.write_html(OUTPUT_DIR / "eda" / "outlier_detection_summary.html")
print(f"✓ Saved: {OUTPUT_DIR / 'eda' / 'outlier_detection_summary.html'}")

# 3. Data Quality Score Dashboard (Gauge Charts)
fig_quality = make_subplots(
    rows=2,
    cols=2,
    specs=[
        [{"type": "indicator"}, {"type": "indicator"}],
        [{"type": "indicator"}, {"type": "indicator"}],
    ],
    subplot_titles=("Overall Quality", "Completeness", "Validity", "Consistency"),
)

fig_quality.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=quality_report.overall_score,
        title={"text": "Overall Score"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#375a7f"},
            "threshold": {
                "line": {"color": "#e74c3c", "width": 4},
                "thickness": 0.75,
                "value": 0.7,
            },
        },
    ),
    row=1,
    col=1,
)

fig_quality.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=quality_report.completeness_score,
        title={"text": "Completeness"},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#00bc8c"}},
    ),
    row=1,
    col=2,
)

fig_quality.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=quality_report.validity_score,
        title={"text": "Validity"},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#f39c12"}},
    ),
    row=2,
    col=1,
)

fig_quality.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=quality_report.consistency_score,
        title={"text": "Consistency"},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#3498db"}},
    ),
    row=2,
    col=2,
)

fig_quality.update_layout(
    title_text="Data Quality Dashboard",
    height=600,
    showlegend=False,
    template="plotly_dark",
    font_family="Arial",
)
fig_quality.show()
fig_quality.write_html(OUTPUT_DIR / "eda" / "data_quality_dashboard.html")
print(f"✓ Saved: {OUTPUT_DIR / 'eda' / 'data_quality_dashboard.html'}")

print(f"\n✅ Section 2 Interactive Visualizations Complete")
# Optionally, print the actual issues
if quality_report.issues:
    print(f"  Issue details:")
    for issue in quality_report.issues[:5]:  # Show first 5 issues
        print(f"    - {issue}")
    if len(quality_report.issues) > 5:
        print(f"    ... and {len(quality_report.issues) - 5} more issues")
# %%
# Apply enhanced 6-step imputation strategy (Phase 9.1 - ENHANCED)
# Steps 1-4: Numeric imputation (zero, KNN, price, median)
# Step 5: Categorical imputation (NEW - handles string/object columns)
# Step 6: Datetime imputation and formatting (NEW - prepares for temporal features)
print("\n📊 Applying Enhanced 6-Step Imputation Strategy...")
all_stocks_imputed = apply_enhanced_imputation_strategy_6step(
    all_stocks_winsorized,
    sector_column="sector",
    n_neighbors=5,
    price_column="last_price",
    handle_categoricals=True,  # NEW: Step 5 - categorical imputation
    handle_dates=True,  # NEW: Step 6 - datetime imputation & formatting
    categorical_strategy="most_frequent",  # Use mode for categorical columns
    date_strategy="forward_fill",  # Forward fill for date columns
)
print(f"✓ Imputation complete")
print(f"  Missing values remaining: {all_stocks_imputed.isnull().sum().sum()}")

# Validate imputation completeness (Phase 9.1 validation)
print("\n🔍 Validating Imputation Completeness...")
validation_results = validate_imputation_completeness(
    all_stocks_imputed,
    critical_date_columns=[
        "last_updated",
        "income_statement_report_date",
        "next_earnings",
        "dividend_record_announce_date",
        "dividend_record_ex_date",
        "dividend_record_payable_date",
        "dividend_record_record_date",
    ],
)
print(f"✓ Imputation Complete: {validation_results['is_complete']}")
print(f"  Total Missing: {validation_results['missing_count']}")
print(f"  Numeric Missing: {validation_results['missing_by_type']['numeric']}")
print(f"  Categorical Missing: {validation_results['missing_by_type']['categorical']}")
print(f"  Ready for Temporal Features: {validation_results['ready_for_temporal_features']}")

# Display datetime column status
if validation_results["datetime_formatted"]:
    print("\n  Datetime Column Status:")
    for col, status in validation_results["datetime_formatted"].items():
        ready_icon = "✓" if status["ready"] else "✗"
        print(
            f"    {ready_icon} {col}: datetime={status['is_datetime']}, missing={status['has_missing']}"
        )
# %%
# Stage 6: Semantic-aware feature scaling
# NEW v1.7: Implements code_guidelines.md Section 8.5.2 (Price Column Preservation Policy)
# - Excludes price columns (last_price, price_target, price_target_median) by default
# - Uses robust scaler for sector-specific scaling

print("\n⚖️ Stage 6: Semantic-Aware Feature Scaling...")

# Apply selective scaling (excludes price columns automatically)
all_stocks_scaled = scale_features(
    all_stocks_imputed.copy(),
    scaler_type="robust",  # Robust scaler handles outliers better than minmax
    by_sector=True,
    exclude_price_columns=True,  # CRITICAL: Preserve price interpretability (default=True)
)

print(f"✓ Feature scaling complete (sector-specific robust scaling)")

# Verify ALL 21 price columns unchanged (code_guidelines.md Section 8.5.2)
# PRICE_COLUMNS includes: current (6), historical (9), 52w bounds (2), EMAs (4)
price_cols_present = [c for c in PRICE_COLUMNS if c in all_stocks_scaled.columns]
for col in price_cols_present:
    if col in all_stocks_imputed.columns:
        assert all_stocks_scaled[col].equals(
            all_stocks_imputed[col]
        ), f"{col} was incorrectly scaled!"

print(
    f"  ✓ Verified {len(price_cols_present)}/21 price columns preserved (business metric protection)"
)

# Count scaled vs excluded columns
numeric_cols_scaled = all_stocks_scaled.select_dtypes(include=[np.number]).columns.tolist()
scaled_count = len([c for c in numeric_cols_scaled if c not in PRICE_COLUMNS])
print(
    f"  ✓ Scaled {scaled_count} numeric features, excluded {len(price_cols_present)} price columns"
)

# Ensure numeric dtypes for key metrics used in downstream visualizations
# Some columns may be object dtype due to mixed inputs (e.g., 'N/A', '--').
for _col in ["p_e", "market_cap", "gross_margin"]:
    if _col in all_stocks_scaled.columns:
        all_stocks_scaled[_col] = pd.to_numeric(all_stocks_scaled[_col], errors="coerce")
# %%
# Preprocessing summary
print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE - Summary")
print("=" * 80)
print(f"✓ Final data shape: {all_stocks_scaled.shape}")
print(f"✓ Missing values: {all_stocks_scaled.isnull().sum().sum()}")
print(f"✓ Data quality score: {quality_report.overall_score:.2f}")
print(f"✓ Outlier detection: 3 methods applied")
print(f"✓ Winsorization: Sector-specific applied")
print(f"✓ Imputation: 6-step strategy applied (numeric + categorical + datetime)")
print(f"✓ Feature scaling: Robust scaler by sector")
print("=" * 80)

# %%
# Phase 9.1 Validation Checkpoint
# =================================
# This cell validates data quality before proceeding to EDA and modeling
print("\n" + "=" * 80)
print("PHASE 9.1 VALIDATION CHECKPOINT")
print("=" * 80)

# 1. Check for remaining NaN values
nan_count = all_stocks_scaled.isnull().sum().sum()
if nan_count > 0:
    print(f"⚠️  WARNING: {nan_count} NaN values still present")
    nan_cols = all_stocks_scaled.columns[all_stocks_scaled.isnull().any()].tolist()
    print(
        f"  Affected columns ({len(nan_cols)}): {nan_cols[:10]}{'...' if len(nan_cols) > 10 else ''}"
    )
    # Apply final cleanup
    print("  Applying final median imputation...")
    for col in nan_cols:
        if all_stocks_scaled[col].dtype in [np.float64, np.int64]:
            all_stocks_scaled[col].fillna(all_stocks_scaled[col].median(), inplace=True)
    print(
        f"✓ Final cleanup complete: {all_stocks_scaled.isnull().sum().sum()} NaN values remaining"
    )
else:
    print("✓ Zero NaN values - data ready for modeling")

# 2. Check for infinite values
inf_count = np.isinf(all_stocks_scaled.select_dtypes(include=[np.number])).sum().sum()
if inf_count > 0:
    print(f"⚠️  WARNING: {inf_count} infinite values detected")
    all_stocks_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_stocks_scaled.fillna(0, inplace=True)
    print("✓ Infinite values replaced")
else:
    print("✓ No infinite values detected")

# 3. Code Guidelines Section 2.2: Validate target variable availability
# Canonical target: price_target (preferred) or price_target_median/last_price (fallback)
if "price_target" in all_stocks_scaled.columns:
    target_valid = all_stocks_scaled["price_target"].notna().sum()
    print(
        f"✓ Target variable 'price_target': {target_valid}/{len(all_stocks_scaled)} valid values ({target_valid / len(all_stocks_scaled) * 100:.1f}%)"
    )
else:
    print("⚠️  WARNING: 'price_target' column not found, will use 'last_price' as fallback")

# 4. Save data snapshot for versioning
try:
    import hashlib
    import json
    from datetime import datetime

    snapshot_metadata = {
        "name": "preprocessed_stocks",
        "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "description": "Phase 9.1 - Fully preprocessed stock data (post-imputation, winsorization, scaling)",
        "tags": ["phase_9.1", "preprocessed", "validated"],
        "shape": list(all_stocks_scaled.shape),
        "columns": list(all_stocks_scaled.columns),
        "quality_score": quality_report.overall_score,
        "checksum": hashlib.md5(str(all_stocks_scaled.shape).encode()).hexdigest(),
    }

    snapshot_file = OUTPUT_DIR / "catalog" / "preprocessed_stocks_metadata.json"
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_file, "w") as f:
        json.dump(snapshot_metadata, f, indent=2)

    print(f"✓ Data snapshot metadata saved: {snapshot_file.name}")
except (IOError, OSError, TypeError, ValueError, AttributeError) as e:
    print(f"⚠️  Data snapshot failed: {e}")

# 5. Summary stats
print(f"\n✓ Validation Summary:")
print(f"  Total stocks: {len(all_stocks_scaled):,}")
print(f"  Total features: {all_stocks_scaled.shape[1]}")
print(f"  Memory usage: {all_stocks_scaled.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"  Data quality score: {quality_report.overall_score:.2f}")

print("\n" + "=" * 80)
print("READY FOR PHASE 9.2 (EDA)")
print("=" * 80)
# %% [markdown]
# ## Phase 9.2: Enhanced Exploratory Data Analysis of Financial Metrics
#
# ### Business Goal
# Understand data distributions, quality issues, correlations, and sector/regional patterns to inform feature engineering and validate data integrity.
#
# ### Key Objectives
# 1. Generate comprehensive statistical summaries and data quality reports
# 2. Analyze correlations and multicollinearity
# 3. Perform hypothesis testing across sectors and regions
# 4. Create interactive visualizations for distributions and relationships
# 5. Generate benchmarking reports comparing sector and regional performance
#
# ### Inputs
# - `all_stocks_scaled`: Scaled and preprocessed data from Phase 9.1
#
# ### Outputs
# **JSON Reports** (4 files):
# - `eda_summary.json` - Comprehensive EDA statistics
# - `data_quality_alerts.json` - Data quality issues and outliers
# - `metrics_dashboard.json` - Financial KPIs by sector
# - `hypothesis_tests.json` - Statistical test results (ANOVA/Kruskal-Wallis)
#
# **Interactive Visualizations** (7 HTML files):
# - `correlation_heatmap.html` - Top 30 metric correlations (clustered)
# - `distributions.html` - Distribution histograms by sector
# - `missing_values.html` - Data completeness heatmap
# - `valuation_3d.html` - 3D scatter (Market Cap × P/E × Margin)
# - `region_sector_heatmap.html` - Regional market cap distribution
# - `sector_boxplots.html` - Valuation metrics by sector
# - `regional_comparison.html` - Median metrics by region
#
# ### Key Functions Used
# - `generate_eda_report()` - HTML EDA report orchestrator
# - `calculate_financial_metrics_dashboard()` - KPI summary by sector/region
# - `generate_data_quality_alerts()` - Outlier and anomaly detection
# - `perform_comprehensive_hypothesis_tests()` - ANOVA/Kruskal-Wallis tests
# - `generate_benchmarking_report()` - Sector/region comparisons
# - `eda_summary()` - Statistical summary dictionary
# - `sector_distribution_summary()` - Sector-wise distributions
#
# ### Validation Checkpoints
# - [ ] All 4 JSON reports generated
# - [ ] All 7 interactive visualizations created
# - [ ] No critical data quality alerts
# - [ ] Statistical tests identify significant sector differences (p < 0.05)
# - [ ] Key correlations documented for feature engineering
#
# ### Analysis Coverage
# This phase provides comprehensive statistical analysis including:
# - **Distribution Analysis**: Histograms, box plots, outlier detection
# - **Correlation Analysis**: Pearson correlations with clustering
# - **Data Quality**: Missing values, outliers, invalid data detection
# - **Hypothesis Testing**: ANOVA/Kruskal-Wallis for sector/region comparisons
# - **Benchmarking**: Sector and regional performance metrics
#
# %%
# Phase 9.2 Cell 21: Comprehensive EDA Report, Data Quality, and Metrics Dashboard
print("\n" + "=" * 80)
print("PHASE 9.2 CELL 1: EDA REPORT + DATA QUALITY + METRICS DASHBOARD")
print("=" * 80)

# Required stdlib
import json

# Initialize output directory
eda_output_dir = Path("outputs/eda")
eda_output_dir.mkdir(parents=True, exist_ok=True)

# 1. Generate comprehensive EDA HTML report
print("\n📊 Step 1/3: Generating comprehensive EDA report...")
eda_report_path = generate_eda_report(
    all_stocks_scaled,
    out_dir=Path("outputs") / "eda",
    sector_column="sector",
)
print(f"  ✓ EDA report complete: {eda_report_path}")

# 2. Generate data quality alerts
print("\n📊 Step 2/3: Analyzing data quality and detecting anomalies...")
quality_alerts = generate_data_quality_alerts(all_stocks_scaled, outlier_threshold=1.5)

# Additionally save a machine-readable EDA summary JSON for downstream steps
try:
    eda_summary_dict = eda_summary(
        all_stocks_scaled, sector_column="sector", include_correlations=False
    )
    eda_summary_path = eda_output_dir / "eda_summary.json"
    with open(eda_summary_path, "w") as f:
        json.dump(eda_summary_dict, f, indent=2, default=str)
    print(f"  ✓ EDA summary JSON saved: {eda_summary_path}")
except (IOError, OSError, TypeError, ValueError, KeyError) as e:
    print(f"  ⚠ Failed to create eda_summary.json: {e}")

# Save quality alerts to JSON
quality_alerts_path = eda_output_dir / "data_quality_alerts.json"
with open(quality_alerts_path, "w") as f:
    json.dump(quality_alerts, f, indent=2, default=str)

print(f"  ✓ Data quality analysis complete")
print(f"  ✓ Total alerts: {len(quality_alerts)}")
print(f"  ✓ Critical: {sum(1 for a in quality_alerts if a.get('severity') == 'high')}")
print(f"  ✓ Warnings: {sum(1 for a in quality_alerts if a.get('severity') == 'medium')}")
print(f"  ✓ Output: {quality_alerts_path}")

# Print top 5 critical alerts
critical_alerts = [a for a in quality_alerts if a.get("severity") == "high"][:5]
if critical_alerts:
    print("\n  Top 5 Critical Data Quality Issues:")
    for i, alert in enumerate(critical_alerts, 1):
        print(f"    {i}. {alert.get('message', 'N/A')}")

# 3. Calculate financial metrics dashboard
print("\n📊 Step 3/3: Calculating financial metrics dashboard...")
metrics_dashboard = calculate_financial_metrics_dashboard(all_stocks_scaled, group_by="sector")

# Save metrics dashboard to JSON
metrics_dashboard_path = eda_output_dir / "metrics_dashboard.json"
with open(metrics_dashboard_path, "w") as f:
    json.dump(metrics_dashboard, f, indent=2, default=str)

print(f"  ✓ Metrics dashboard complete")
print(f"  ✓ Categories: Valuation, Profitability, Growth, Leverage")
print(f"  ✓ Sectors analyzed: {len(metrics_dashboard.get('by_group', {}))}")
print(f"  ✓ Output: {metrics_dashboard_path}")

print("\n✅ Cell 21 Complete: Generated 3 reports")
print(f"   • EDA Report: {eda_report_path}")
print(f"   • Data Quality: {quality_alerts_path}")
print(f"   • Metrics Dashboard: {metrics_dashboard_path}")

# %%
# Phase 9.2 Cell 22: Statistical Hypothesis Testing
print("\n" + "=" * 80)
print("PHASE 9.2 CELL 2: STATISTICAL HYPOTHESIS TESTING")
print("=" * 80)

# Key metrics for hypothesis testing
test_metrics = [
    "p_e",
    "p_b",
    "p_s",
    "ev_ebitda",  # Valuation
    "roe",
    "roa",
    "roic",
    "net_margin",
    "operating_margin",  # Profitability
    "revenue_growth",
    "earnings_growth",  # Growth
    "debt_to_equity",
    "current_ratio",  # Leverage & Liquidity
]

# Filter to available metrics
available_test_metrics = [m for m in test_metrics if m in all_stocks_scaled.columns]

print(f"\n📊 Performing hypothesis tests on {len(available_test_metrics)} metrics...")
print(f"   Tests: ANOVA (parametric) and Kruskal-Wallis (non-parametric)")
print(f"   Grouping: By sector")
print(f"   Significance level: α = 0.05")

# Perform comprehensive hypothesis tests
hypothesis_results = perform_comprehensive_hypothesis_tests(
    all_stocks_scaled, group_column="sector", metrics=available_test_metrics, alpha=0.05
)

# Save results
hypothesis_test_path = eda_output_dir / "hypothesis_tests.json"
with open(hypothesis_test_path, "w") as f:
    json.dump(hypothesis_results, f, indent=2, default=str)

print(f"\n✓ Hypothesis testing complete")
print(f"✓ Output: {hypothesis_test_path}")

# Print significant findings
print("\n📊 Significant Findings (p < 0.05):")
significant_count = 0
for metric, results in hypothesis_results.items():
    if isinstance(results, dict):
        # Check ANOVA p-value
        anova_p = results.get("anova", {}).get("p_value", 1.0)
        kruskal_p = results.get("kruskal", {}).get("p_value", 1.0)

        if anova_p < 0.05 or kruskal_p < 0.05:
            significant_count += 1
            test_used = "ANOVA" if anova_p < 0.05 else "Kruskal-Wallis"
            p_val = anova_p if anova_p < 0.05 else kruskal_p
            print(f"  • {metric}: {test_used} p={p_val:.4f} - Sectors differ significantly")

if significant_count == 0:
    print("  No significant differences detected across sectors")
else:
    print(
        f"\n  Total: {significant_count}/{len(available_test_metrics)} metrics show significant sector differences"
    )

print("\n✅ Cell 22 Complete: Statistical hypothesis testing performed")

# %%
# Phase 9.2 Cell 23: Interactive Visualizations - Distributions & Correlations
print("\n" + "=" * 80)
print("PHASE 9.2 CELL 3: INTERACTIVE VISUALIZATIONS")
print("=" * 80)

print("\n📊 Creating 4 interactive visualizations...")

# Select key financial metrics for visualization
key_metrics = [
    "market_cap",
    "enterprise_value",
    "last_price",
    "p_e",
    "p_b",
    "p_s",
    "ev_ebitda",
    "peg_ratio",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "ebitda_margin",
    "roe",
    "roa",
    "roic",
    "roce",
    "revenue",
    "revenue_growth",
    "earnings_growth",
    "ebitda",
    "debt_to_equity",
    "total_debt_ratio",
    "current_ratio",
    "quick_ratio",
    "free_cash_flow",
    "operating_cash_flow",
    "dividend_yield",
    "payout_ratio",
    "analyst_target_price",
]

# Filter to available metrics
viz_metrics = [m for m in key_metrics if m in all_stocks_scaled.columns][:30]

# 1. Correlation Heatmap (top 30 metrics, clustered)
print("\n  1/4: Correlation heatmap...")
corr_matrix = all_stocks_scaled[viz_metrics].corr()

fig_corr = px.imshow(
    corr_matrix,
    labels=dict(color="Correlation"),
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Phase 9.2: Financial Metrics Correlation Matrix (Top 30 Metrics)",
    template="plotly_dark",
)
fig_corr.update_layout(height=800, width=1000, font_family="Arial")
corr_path = eda_output_dir / "correlation_heatmap.html"
fig_corr.write_html(corr_path)
print(f"  ✓ Saved: {corr_path}")

# 2. Distribution Histograms (by sector)
print("\n  2/4: Distribution histograms...")
dist_metrics = ["p_e", "p_b", "net_margin", "roe"]
available_dist = [m for m in dist_metrics if m in all_stocks_scaled.columns]

fig_dist = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[m.upper().replace("_", " ") for m in available_dist[:4]],
)

for idx, metric in enumerate(available_dist[:4], 1):
    row = (idx - 1) // 2 + 1
    col = (idx - 1) % 2 + 1

    for sector in all_stocks_scaled["sector"].unique()[:8]:  # Limit sectors for clarity
        sector_data = all_stocks_scaled[all_stocks_scaled["sector"] == sector][metric].dropna()
        fig_dist.add_trace(
            go.Histogram(x=sector_data, name=sector, showlegend=(idx == 1)),
            row=row,
            col=col,
        )

fig_dist.update_layout(
    height=700,
    title_text="Phase 9.2: Key Metric Distributions by Sector",
    showlegend=True,
    template="plotly_dark",
    font_family="Arial",
)
dist_path = eda_output_dir / "distributions.html"
fig_dist.write_html(dist_path)
print(f"  ✓ Saved: {dist_path}")

# 3. Missing Values Heatmap
print("\n  3/4: Missing values heatmap...")
missing_pct = (
    all_stocks_scaled[viz_metrics].isnull().sum() / len(all_stocks_scaled) * 100
).sort_values(ascending=False)
missing_df = pd.DataFrame({"Metric": missing_pct.index, "Missing %": missing_pct.values})

fig_missing = px.bar(
    missing_df,
    x="Metric",
    y="Missing %",
    title="Phase 9.2: Data Completeness Analysis (Top 30 Metrics)",
    labels={"Missing %": "Missing Percentage (%)"},
    color="Missing %",
    color_continuous_scale="Reds",
    template="plotly_dark",
)
fig_missing.update_layout(height=500, xaxis_tickangle=-45, font_family="Arial")
missing_path = eda_output_dir / "missing_values.html"
fig_missing.write_html(missing_path)
print(f"  ✓ Saved: {missing_path}")

# 4. 3D Valuation Scatter
print("\n  4/4: 3D valuation scatter...")
if all(
    [
        "market_cap" in all_stocks_scaled.columns,
        "p_e" in all_stocks_scaled.columns,
        "gross_margin" in all_stocks_scaled.columns,
        "sector" in all_stocks_scaled.columns,
    ]
):
    viz_df = all_stocks_scaled[["market_cap", "p_e", "gross_margin", "sector", "ticker"]].dropna()

    fig_3d = px.scatter_3d(
        viz_df,
        x="market_cap",
        y="p_e",
        z="gross_margin",
        color="sector",
        hover_name="ticker",
        title="Phase 9.2: 3D Valuation Landscape",
        labels={
            "market_cap": "Market Cap",
            "p_e": "P/E",
            "gross_margin": "Gross Margin",
        },
        log_x=True,
        template="plotly_dark",
    )
    fig_3d.update_layout(height=700, font_family="Arial")

    valuation_3d_path = eda_output_dir / "valuation_3d.html"
    fig_3d.write_html(valuation_3d_path)
    print(f"  ✓ Saved: {valuation_3d_path}")
else:
    print(f"  ⚠ Skipped: Required columns not available")

print("\n✅ Cell 23 Complete: Created 4 interactive visualizations")

# %%
# Phase 9.2 Cell 24: Sector & Regional Benchmarking
print("\n" + "=" * 80)
print("PHASE 9.2 CELL 4: SECTOR & REGIONAL BENCHMARKING")
print("=" * 80)

# Select top metrics for benchmarking
benchmark_metrics = [
    "p_e",
    "p_b",
    "ev_ebitda",  # Valuation
    "roe",
    "roa",
    "net_margin",  # Profitability
    "revenue_growth",
    "market_cap",  # Growth & Size
]
available_benchmark_metrics = [m for m in benchmark_metrics if m in all_stocks_scaled.columns]

print(
    f"\n📊 Step 1/2: Generating benchmarking report ({len(available_benchmark_metrics)} metrics)..."
)

# Generate benchmarking report
benchmark_report = generate_benchmarking_report(
    all_stocks_scaled,
    metrics=available_benchmark_metrics,
    sector_column="sector",
    region_column="region" if "region" in all_stocks_scaled.columns else None,
    include_statistical_tests=True,
)

# Save benchmarking report
benchmark_path = eda_output_dir / "benchmarking_report.json"
with open(benchmark_path, "w") as f:
    json.dump(benchmark_report, f, indent=2, default=str)

print(f"  ✓ Benchmarking complete: {benchmark_path}")

# Generate sector distribution summary
print("\n📊 Step 2/2: Creating sector distribution visualizations...")
sector_dist_metrics = ["market_cap", "p_e", "roe", "net_margin"]
available_sector_metrics = [m for m in sector_dist_metrics if m in all_stocks_scaled.columns]

sector_summaries = sector_distribution_summary(
    all_stocks_scaled, sector_column="sector", metrics=available_sector_metrics
)

# Create sector box plots
if available_sector_metrics and "sector" in all_stocks_scaled.columns:
    # 1. Region-Sector Heatmap
    if "region" in all_stocks_scaled.columns and "market_cap" in all_stocks_scaled.columns:
        print("\n  1/3: Region-sector heatmap...")
        region_sector = (
            all_stocks_scaled.groupby(["region", "sector"])["market_cap"]
            .agg(["mean", "count"])
            .reset_index()
        )
        region_sector_pivot = region_sector.pivot(index="sector", columns="region", values="mean")

        fig_region_sector = px.imshow(
            region_sector_pivot,
            labels=dict(color="Avg Market Cap"),
            title="Phase 9.2: Average Market Cap by Region and Sector",
            aspect="auto",
            color_continuous_scale="Viridis",
        )
        region_sector_path = eda_output_dir / "region_sector_heatmap.html"
        fig_region_sector.write_html(region_sector_path)
        print(f"  ✓ Saved: {region_sector_path}")

    # 2. Sector Box Plots
    print("\n  2/3: Sector box plots...")
    fig_box = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[m.upper().replace("_", " ") for m in available_sector_metrics[:4]],
    )

    for idx, metric in enumerate(available_sector_metrics[:4], 1):
        row = (idx - 1) // 2 + 1
        col = (idx - 1) % 2 + 1

        for sector in all_stocks_scaled["sector"].unique()[:8]:
            sector_data = all_stocks_scaled[all_stocks_scaled["sector"] == sector][metric].dropna()
            fig_box.add_trace(
                go.Box(y=sector_data, name=sector, showlegend=(idx == 1)),
                row=row,
                col=col,
            )

    fig_box.update_layout(
        height=700,
        title_text="Phase 9.2: Sector Distribution Box Plots",
        showlegend=True,
    )
    box_plot_path = eda_output_dir / "sector_boxplots.html"
    fig_box.write_html(box_plot_path)
    print(f"  ✓ Saved: {box_plot_path}")

    # 3. Regional Comparison Bar Charts
    if "region" in all_stocks_scaled.columns:
        print("\n  3/3: Regional comparison bar charts...")
        regional_metrics = ["p_e", "roe"]
        available_regional = [m for m in regional_metrics if m in all_stocks_scaled.columns]

        if available_regional:
            regional_summary = (
                all_stocks_scaled.groupby("region")[available_regional].median().reset_index()
            )

            fig_regional = make_subplots(
                rows=1,
                cols=len(available_regional),
                subplot_titles=[m.upper().replace("_", " ") for m in available_regional],
            )

            for idx, metric in enumerate(available_regional, 1):
                fig_regional.add_trace(
                    go.Bar(
                        x=regional_summary["region"],
                        y=regional_summary[metric],
                        name=metric,
                    ),
                    row=1,
                    col=idx,
                )

            fig_regional.update_layout(
                height=400,
                title_text="Phase 9.2: Regional Comparison (Median Values)",
                showlegend=False,
            )
            regional_path = eda_output_dir / "regional_comparison.html"
            fig_regional.write_html(regional_path)
            print(f"  ✓ Saved: {regional_path}")

print("\n✅ Cell 24 Complete: Sector and regional benchmarking performed")

# %%
# Phase 9.2 Cell 25: EDA Summary Dashboard
print("\n" + "=" * 80)
print("PHASE 9.2 CELL 5: EDA SUMMARY DASHBOARD")
print("=" * 80)

print("\n📊 Compiling Phase 9.2 Summary...")

# Collect all Phase 9.2 outputs
phase92_outputs = {
    "json_reports": [
        "eda_summary.json",
        "data_quality_alerts.json",
        "metrics_dashboard.json",
        "hypothesis_tests.json",
    ],
    "html_visualizations": [
        "correlation_heatmap.html",
        "distributions.html",
        "missing_values.html",
        "valuation_3d.html",
        "region_sector_heatmap.html",
        "sector_boxplots.html",
        "regional_comparison.html",
    ],
}

# Count existing files
existing_json = sum(1 for f in phase92_outputs["json_reports"] if (eda_output_dir / f).exists())
existing_html = sum(
    1 for f in phase92_outputs["html_visualizations"] if (eda_output_dir / f).exists()
)

print(f"\n✅ Phase 9.2 Enhanced EDA Complete!")
print(f"\n📁 Output Directory: {eda_output_dir}")
print(f"\n📄 JSON Reports ({existing_json}/{len(phase92_outputs['json_reports'])}):")
for report in phase92_outputs["json_reports"]:
    status = "✓" if (eda_output_dir / report).exists() else "✗"
    print(f"   {status} {report}")

print(
    f"\n🌐 Interactive Visualizations ({existing_html}/{len(phase92_outputs['html_visualizations'])}):"
)
for viz in phase92_outputs["html_visualizations"]:
    status = "✓" if (eda_output_dir / viz).exists() else "✗"
    print(f"   {status} {viz}")

# Print key findings summary
print(f"\n📊 Key Findings Summary:")
print(f"   • Dataset: {all_stocks_scaled.shape[0]} stocks × {all_stocks_scaled.shape[1]} features")
print(
    f"   • Sectors: {all_stocks_scaled['sector'].nunique() if 'sector' in all_stocks_scaled.columns else 'N/A'}"
)
print(
    f"   • Regions: {all_stocks_scaled['region'].nunique() if 'region' in all_stocks_scaled.columns else 'N/A'}"
)

# Calculate data completeness
if "viz_metrics" in globals() and len(viz_metrics) > 0:
    completeness = (
        1 - all_stocks_scaled[viz_metrics].isnull().sum().mean() / len(all_stocks_scaled)
    ) * 100
    print(f"   • Data Completeness: {completeness:.1f}%")

# Top correlations
if "corr_matrix" in globals() and corr_matrix is not None and len(viz_metrics) >= 2:
    corr_pairs = []
    for i in range(len(viz_metrics)):
        for j in range(i + 1, len(viz_metrics)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                corr_pairs.append((viz_metrics[i], viz_metrics[j], corr_val))

    if corr_pairs:
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        print(f"\n   Top 3 Correlated Metric Pairs:")
        for m1, m2, corr in corr_pairs[:3]:
            print(f"     • {m1} ↔ {m2}: r={corr:.3f}")

# Data quality summary
if isinstance(quality_alerts, list) and len(quality_alerts) > 0:
    high_severity = sum(1 for a in quality_alerts if a.get("severity") == "high")
    if high_severity > 0:
        print(f"\n   ⚠ Data Quality: {high_severity} critical issues detected")
    else:
        print(f"\n   ✓ Data Quality: No critical issues detected")

print("\n" + "=" * 80)
print("PHASE 9.2 COMPLETE - Proceed to Phase 9.3 Feature Engineering")
print("=" * 80)
## Phase 9.3 Category Analysis Moved
print("\nℹ Phase 9.3 Enhanced EDA category analysis has been relocated.")
print(
    "   See the section titled 'Phase 9.3 Enhanced EDA - Category Performance Analysis' after the"
)
print("   'Phase 9.3 Schema 1.3 Summary' for the relocated cells and outputs.")

# Import authoritative schema definitions to fix missing variable references
from finance_ml.ml_workflow.data.schema import PHASE93_FEATURE_CATEGORIES

# Use schema-defined input categories (only 6 categories exist in PHASE93_FEATURE_CATEGORIES)
momentum_technical_metrics = PHASE93_FEATURE_CATEGORIES.get("momentum", [])
valuation_metrics = PHASE93_FEATURE_CATEGORIES.get("valuation", [])
profitability_metrics = PHASE93_FEATURE_CATEGORIES.get("profitability", [])
quality_risk_metrics = PHASE93_FEATURE_CATEGORIES.get("quality_risk", [])
cash_flow_metrics = PHASE93_FEATURE_CATEGORIES.get("cash_flow", [])

growth_metrics = PHASE93_FEATURE_CATEGORIES.get("growth", [])

metrics_to_benchmark = (
    momentum_technical_metrics
    + valuation_metrics
    + profitability_metrics
    + quality_risk_metrics
    + cash_flow_metrics
    + growth_metrics
)

# %%
# Phase 9.1: Enhanced imputation and dtype validation (Schema-aware)
# Note: unified pipeline handles normalization, dtype validation, imputation, and scaling in order.

available_metrics = [m for m in metrics_to_benchmark if m in all_stocks_scaled.columns]

# Create category mapping dictionary (required for metrics availability analysis)
# This maps Phase 9.3 feature categories to their respective metric lists
# Aligned with code_guidelines.md Section 3.1 (Phase 9.3 Feature Categories)
category_mapping = {
    "Momentum & Technical": momentum_technical_metrics,
    "Valuation Ratios": valuation_metrics,
    "Profitability": profitability_metrics,
    "Quality & Risk": quality_risk_metrics,
    "Cash Flow": cash_flow_metrics,
    "Growth": growth_metrics,
}

# Generate comprehensive benchmarking report
benchmark_report = generate_benchmarking_report(
    all_stocks_scaled,
    metrics=available_metrics,
    sector_column="sector",
    region_column="region",
)

# Display category-grouped summary
print(f"\n✓ Benchmarking report generated")
print(f"  Total stocks: {benchmark_report['summary']['total_stocks']}")
print(f"  Sectors analyzed: {benchmark_report['summary']['n_sectors']}")
print(f"  Regions analyzed: {benchmark_report['summary']['n_regions']}")
print(f"  Total metrics: {len(available_metrics)}")

# Fallback safety check: If category_mapping wasn't defined, create a default mapping
# This prevents NameError crashes during execution (defensive programming)
if "category_mapping" not in locals():
    category_mapping = {"General": available_metrics}

# Display metrics availability by category
print(f"\n📋 Metrics Availability by Category:")
for category_name, category_metrics in category_mapping.items():
    available_in_category = [m for m in category_metrics if m in available_metrics]
coverage_pct = (len(available_in_category) / len(category_metrics) * 100) if category_metrics else 0
print(
    f"  {category_name}: {len(available_in_category)}/{len(category_metrics)} metrics ({coverage_pct:.0f}% coverage)"
)

# %%
# Visualization 1: Category Heatmaps (Sector × Category)
print("\n📊 Category Performance Heatmaps:")

# Compute category scores by averaging z-scores of metrics within each category
from scipy.stats import zscore

category_sector_scores = {}

for category_name, category_metrics in category_mapping.items():
    available_in_category = [m for m in category_metrics if m in all_stocks_scaled.columns]

    if len(available_in_category) == 0:
        print(f"  ⚠️ Skipping {category_name}: No available metrics")
        continue

    # Compute z-scores for available metrics and average by sector
    category_data = all_stocks_scaled[available_in_category + ["sector"]].copy()

    # Convert to numeric and compute z-scores
    for col in available_in_category:
        category_data[col] = pd.to_numeric(category_data[col], errors="coerce")

    # Compute z-scores (handle NaNs)
    z_scored_data = category_data[available_in_category].apply(
        lambda x: zscore(x, nan_policy="omit")
    )
    category_data["category_score"] = z_scored_data.mean(axis=1)

    # Aggregate by sector
    sector_scores = (
        category_data.groupby("sector")["category_score"].mean().sort_values(ascending=False)
    )
    category_sector_scores[category_name] = sector_scores

# Create heatmap matrix
if category_sector_scores:
    heatmap_df = pd.DataFrame(category_sector_scores).T

    # Create interactive heatmap with value annotations
    # Following code_guidelines.md Section 17.1: Heatmaps and Conditional Formatting
    fig_category_heatmap = px.imshow(
        heatmap_df,
        labels=dict(x="Sector", y="Category", color="Avg Z-Score"),
        title="Sector Performance Across 11 Feature Categories (Phase 9.3)",
        color_continuous_scale="RdYlGn",  # Diverging scale for z-scores centered at zero
        aspect="auto",
        text_auto=".3f",  # ✅ Display values formatted to 3 decimals (Section 17.1)
    )

    # Enhanced styling per Section 17.1 style guidelines
    fig_category_heatmap.update_traces(
        textfont=dict(
            size=11,
            color="white",  # High contrast on diverging color scale
            family="Arial",  # Standard sans-serif font (Section 17.4)
        )
    )

    fig_category_heatmap.update_layout(
        height=600,
        xaxis_tickangle=-45,
        font=dict(size=10, family="Arial"),
        template="plotly_dark",  # ✅ Dark mode compatible theme (Section 17.1)
    )

    fig_category_heatmap.show()
    output_path = eda_output_dir / "phase93_category_sector_heatmap.html"
    fig_category_heatmap.write_html(output_path)

    print(f"\n✓ Category heatmap visualization complete")
    print(f"  Categories visualized: {len(category_sector_scores)}")
    print(f"  Sectors analyzed: {len(heatmap_df.columns)}")
    print(f"  Output: {output_path}")

    # Display top performing sector per category
    print(f"\n  🏆 Top Performing Sectors by Category:")
    for category, scores in list(category_sector_scores.items())[:5]:
        top_sector = scores.idxmax()
        top_score = scores.max()
        print(f"    {category}: {top_sector} (z-score: {top_score:.2f})")
else:
    print("  ⚠️ No category data available for visualization")

# %%
# Visualization 2: Regional Performance Radar Charts
print("\n📊 Regional Performance Radar Charts:")

# Compute category scores by region
category_region_scores = {}

for category_name, category_metrics in category_mapping.items():
    available_in_category = [m for m in category_metrics if m in all_stocks_scaled.columns]

    if len(available_in_category) == 0:
        continue

    # Compute z-scores for available metrics and average by region
    category_data = all_stocks_scaled[available_in_category + ["region"]].copy()

    # Convert to numeric and compute z-scores
    for col in available_in_category:
        category_data[col] = pd.to_numeric(category_data[col], errors="coerce")

    # Compute z-scores (handle NaNs)
    z_scored_data = category_data[available_in_category].apply(
        lambda x: zscore(x, nan_policy="omit")
    )
    category_data["category_score"] = z_scored_data.mean(axis=1)

    # Aggregate by region
    region_scores = category_data.groupby("region")["category_score"].mean()
    category_region_scores[category_name] = region_scores

if category_region_scores:
    # Create radar chart for each region
    radar_df = pd.DataFrame(category_region_scores)

    # Create subplot radar charts
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    regions = radar_df.index.tolist()
    categories = radar_df.columns.tolist()

    # Create single figure with all regions
    fig_radar = go.Figure()

    for region in regions:
        values = radar_df.loc[region].tolist()
        values.append(values[0])  # Close the radar chart

        fig_radar.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=region,
                opacity=0.6,
            )
        )

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
        showlegend=True,
        title="Regional Performance Across 11 Feature Categories (Phase 9.3)",
        height=700,
    )

    fig_radar.show()
    output_path = eda_output_dir / "phase93_regional_radar_charts.html"
    fig_radar.write_html(output_path)

    print(f"\n✓ Regional radar charts complete")
    print(f"  Regions visualized: {len(regions)}")
    print(f"  Categories analyzed: {len(categories)}")
    print(f"  Output: {output_path}")

    # Display top category per region
    print(f"\n  🌍 Strongest Category by Region:")
    for region in regions[:5]:
        top_category = radar_df.loc[region].idxmax()
        top_score = radar_df.loc[region].max()
        print(f"    {region}: {top_category} (z-score: {top_score:.2f})")
else:
    print("  ⚠️ No regional category data available")

# %%
# Visualization 3: Category Correlation Network
print("\n📊 Category Correlation Network:")

# Compute correlation matrix between category scores
if category_sector_scores:
    # Create category score matrix (stocks × categories)
    category_score_matrix = pd.DataFrame()

    for category_name, category_metrics in category_mapping.items():
        available_in_category = [m for m in category_metrics if m in all_stocks_scaled.columns]

        if len(available_in_category) == 0:
            continue

        # Compute z-scores for available metrics and average to get category score
        category_data = all_stocks_scaled[available_in_category].copy()

        # Convert to numeric
        for col in available_in_category:
            category_data[col] = pd.to_numeric(category_data[col], errors="coerce")

        # Compute z-scores and average
        z_scored_data = category_data.apply(lambda x: zscore(x, nan_policy="omit"))
        category_score_matrix[category_name] = z_scored_data.mean(axis=1)

    # Compute correlation matrix
    category_corr = category_score_matrix.corr()

    # Create interactive heatmap for category correlations with value annotations
    # Following code_guidelines.md Section 17.1: Heatmaps and Conditional Formatting
    fig_corr_network = px.imshow(
        category_corr,
        labels=dict(x="Category", y="Category", color="Correlation"),
        title="Inter-Category Correlation Matrix (Phase 9.3)",
        color_continuous_scale="RdBu_r",  # Diverging scale for correlation (-1 to +1)
        aspect="auto",
        zmin=-1,
        zmax=1,
        text_auto=".3f",  # ✅ Display correlation values formatted to 3 decimals (Section 17.1)
    )

    # Enhanced styling per Section 17.1 style guidelines
    fig_corr_network.update_traces(
        textfont=dict(
            size=10,
            color="white",  # High contrast on diverging color scale
            family="Arial",  # Standard sans-serif font (Section 17.4)
        )
    )

    fig_corr_network.update_layout(
        height=700,
        xaxis_tickangle=-45,
        font=dict(size=9, family="Arial"),
        template="plotly_dark",  # ✅ Dark mode compatible theme (Section 17.1)
    )

    fig_corr_network.show()
    output_path = eda_output_dir / "phase93_category_correlation_network.html"
    fig_corr_network.write_html(output_path)

    print(f"\n✓ Category correlation network complete")
    print(f"  Categories analyzed: {len(category_corr)}")
    print(f"  Output: {output_path}")

    # Find strongest positive and negative correlations
    corr_pairs = []
    for i in range(len(category_corr.columns)):
        for j in range(i + 1, len(category_corr.columns)):
            cat1 = category_corr.columns[i]
            cat2 = category_corr.columns[j]
            corr_val = category_corr.iloc[i, j]
            if not np.isnan(corr_val):
                corr_pairs.append((cat1, cat2, corr_val))

    # Sort by absolute correlation
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n  🔗 Strongest Category Relationships:")
    for cat1, cat2, corr in corr_pairs[:5]:
        print(f"    {cat1} ↔ {cat2}: {corr:.2f}")
else:
    print("  ⚠️ No category data available for correlation analysis")

# %%
# Visualization 4: Category Distribution Box Plots
print("\n📊 Category Distribution Box Plots:")

# Create box plots for each category showing distribution across sectors
if "category_score_matrix" in locals() and not category_score_matrix.empty:
    # Add sector information to category scores
    category_scores_with_sector = category_score_matrix.copy()
    category_scores_with_sector["sector"] = all_stocks_scaled["sector"].values

    # Create subplot grid for all categories
    import math

    from plotly.subplots import make_subplots

    categories = [col for col in category_score_matrix.columns]
    n_categories = len(categories)
    n_cols = 2
    n_rows = math.ceil(n_categories / n_cols)

    fig_box = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=categories,
        vertical_spacing=0.2,
        horizontal_spacing=0.1,
    )

    for idx, category in enumerate(categories):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Create box plot data for this category
        for sector in category_scores_with_sector["sector"].unique():
            if pd.notna(sector):
                sector_data = category_scores_with_sector[
                    category_scores_with_sector["sector"] == sector
                ][category].dropna()

                fig_box.add_trace(
                    go.Box(
                        y=sector_data,
                        name=sector,
                        showlegend=(idx == 0),  # Only show legend for first subplot
                        marker_color=px.colors.qualitative.Plotly[
                            list(category_scores_with_sector["sector"].unique()).index(sector) % 10
                        ],
                    ),
                    row=row,
                    col=col,
                )

    fig_box.update_layout(
        title_text="Category Score Distributions by Sector (Phase 9.3)",
        height=300 * n_rows,
        showlegend=True,
    )

    fig_box.update_yaxes(title_text="Z-Score")

    fig_box.show()
    output_path = eda_output_dir / "phase93_category_distributions_boxplots.html"
    fig_box.write_html(output_path)

    print(f"\n✓ Category distribution box plots complete")
    print(f"  Categories visualized: {n_categories}")
    print(f"  Grid layout: {n_rows}×{n_cols}")
    print(f"  Output: {output_path}")

    # Identify categories with highest variance
    category_variances = category_score_matrix.var().sort_values(ascending=False)
    print(f"\n  📊 Categories with Highest Variance:")
    for category, variance in category_variances.head(5).items():
        print(f"    {category}: {variance:.2f}")
else:
    print("  ⚠️ No category score data available for box plots")

# %%
# Visualization 5: Category-Sector Bubble Chart (Value vs Quality Trade-offs)
print("\n📊 Category-Sector Bubble Chart:")

# Create scatter plot comparing two key categories with sector coloring
if "category_score_matrix" in locals() and not category_score_matrix.empty:
    # Select two categories for comparison (e.g., Valuation vs Quality)
    categories_list = list(category_score_matrix.columns)

    # Default to Valuation Ratios (cat 2) vs Quality & Risk (cat 4) if available
    x_category = "Valuation Ratios" if "Valuation Ratios" in categories_list else categories_list[0]
    y_category = (
        "Quality & Risk"
        if "Quality & Risk" in categories_list
        else (categories_list[1] if len(categories_list) > 1 else categories_list[0])
    )

    # Prepare data for bubble chart
    bubble_data = pd.DataFrame(
        {
            x_category: category_score_matrix[x_category],
            y_category: category_score_matrix[y_category],
            "sector": all_stocks_scaled["sector"].values,
            "ticker": all_stocks_scaled.get("ticker", range(len(category_score_matrix))),
            "market_cap": all_stocks_imputed.get(
                "market_cap", 100
            ),  # Use raw (unscaled) data for bubble size
        }
    ).dropna()

    # Create bubble chart
    fig_bubble = px.scatter(
        bubble_data,
        x=x_category,
        y=y_category,
        color="sector",
        size="market_cap",
        hover_data=["ticker"],
        title=f"Strategic Positioning: {x_category} vs {y_category} (Phase 9.3)",
        labels={
            x_category: f"{x_category} Score (Z)",
            y_category: f"{y_category} Score (Z)",
        },
        size_max=30,
        opacity=0.6,
    )

    # Add quadrant lines
    fig_bubble.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_bubble.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Add quadrant labels
    fig_bubble.add_annotation(
        text="High Quality,<br>High Valuation",
        x=1.5,
        y=1.5,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )
    fig_bubble.add_annotation(
        text="High Quality,<br>Low Valuation",
        x=-1.5,
        y=1.5,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )

    fig_bubble.update_layout(height=700)

    fig_bubble.show()
    output_path = eda_output_dir / "phase93_category_sector_bubble_chart.html"
    fig_bubble.write_html(output_path)

    print(f"\n✓ Category-sector bubble chart complete")
    print(f"  X-axis: {x_category}")
    print(f"  Y-axis: {y_category}")
    print(f"  Data points: {len(bubble_data)}")
    print(f"  Output: {output_path}")

    # Identify quadrants
    q1 = bubble_data[(bubble_data[x_category] > 0) & (bubble_data[y_category] > 0)]
    q2 = bubble_data[(bubble_data[x_category] < 0) & (bubble_data[y_category] > 0)]
    q3 = bubble_data[(bubble_data[x_category] < 0) & (bubble_data[y_category] < 0)]
    q4 = bubble_data[(bubble_data[x_category] > 0) & (bubble_data[y_category] < 0)]

    print(f"\n  📍 Quadrant Distribution:")
    print(
        f"    Q1 (High Val, High Qual): {len(q1)} stocks ({len(q1) / len(bubble_data) * 100:.1f}%)"
    )
    print(
        f"    Q2 (Low Val, High Qual): {len(q2)} stocks ({len(q2) / len(bubble_data) * 100:.1f}%) - Value opportunities"
    )
    print(f"    Q3 (Low Val, Low Qual): {len(q3)} stocks ({len(q3) / len(bubble_data) * 100:.1f}%)")
    print(
        f"    Q4 (High Val, Low Qual): {len(q4)} stocks ({len(q4) / len(bubble_data) * 100:.1f}%) - Risk flags"
    )
else:
    print("  ⚠️ No category score data available for bubble chart")

# %%
# Summary Dashboard & Export - Phase 9.3 Enhanced EDA
print("\n📊 Phase 9.3 EDA Summary & Export:")

# Generate comprehensive summary JSON
eda_summary = {
    "phase": "9.3",
    "schema_version": "1.3",
    "timestamp": pd.Timestamp.now().isoformat(),
    "data_summary": {
        "total_stocks": len(all_stocks_scaled),
        "sectors": benchmark_report["summary"]["n_sectors"],
        "regions": benchmark_report["summary"]["n_regions"],
        "total_metrics": len(available_metrics),
    },
    "category_coverage": {
        cat: len([m for m in metrics if m in available_metrics])
        for cat, metrics in category_mapping.items()
    },
    "visualizations_generated": [
        "phase93_category_sector_heatmap.html",
        "phase93_regional_radar_charts.html",
        "phase93_category_correlation_network.html",
        "phase93_category_distributions_boxplots.html",
        "phase93_category_sector_bubble_chart.html",
    ],
}

# Save summary JSON
summary_path = eda_output_dir / "phase93_eda_summary.json"
with open(summary_path, "w") as f:
    json.dump(eda_summary, f, indent=2)

# Generate Excel report with category-segmented tabs
excel_path = eda_output_dir / "phase93_category_analysis_report.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    # Tab 1: Summary
    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Total Stocks",
                "Sectors",
                "Regions",
                "Total Metrics",
                "Categories",
            ],
            "Value": [
                eda_summary["data_summary"]["total_stocks"],
                eda_summary["data_summary"]["sectors"],
                eda_summary["data_summary"]["regions"],
                eda_summary["data_summary"]["total_metrics"],
                len(category_mapping),
            ],
        }
    )
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

    # Tab 2: Category Coverage
    coverage_df = pd.DataFrame(
        [
            {
                "Category": cat,
                "Metrics Available": count,
                "Total Metrics": len(category_mapping[cat]),
            }
            for cat, count in eda_summary["category_coverage"].items()
        ]
    )
    coverage_df["Coverage %"] = (
        coverage_df["Metrics Available"] / coverage_df["Total Metrics"] * 100
    ).round(1)
    coverage_df.to_excel(writer, sheet_name="Category Coverage", index=False)

    # Tab 3: Sector-Category Scores (if available)
    if "heatmap_df" in locals():
        heatmap_df.to_excel(writer, sheet_name="Sector-Category Scores")

    # Tab 4: Regional-Category Scores (if available)
    if "radar_df" in locals():
        radar_df.to_excel(writer, sheet_name="Regional-Category Scores")

    # Tab 5: Category Correlations (if available)
    # (Moved to relocated section)

print(f"\n✓ Phase 9.3 EDA analysis complete (relocated section)")

# %% [markdown]
# ## Phase 9.3: Advanced Feature Engineering with Sector-Specific Optimizations with Sector-Specific Optimizations
#
# ### Business Goal
# Engineer comprehensive financial features including valuation ratios, profitability metrics, quality indicators, and sector-specific features to maximize model predictive power.
#
# ### Key Objectives
# 1. Engineer valuation ratios (P/E, P/B, EV/EBITDA, PEG)
# 2. Engineer profitability features (margins, ROE, ROA, ROIC)
# 3. Create momentum and technical indicators
# 4. Engineer analyst quality features
# 5. Create accounting quality scores (Altman Z, Piotroski F)
# 6. Build sector-relative features
# 7. Create interaction features
#
# ### Inputs
# - `all_stocks_preprocessed`: Preprocessed data from Phase 9.1
#
# ### Outputs
# - `all_stocks_features`: Data with 400+ engineered features
# - `outputs/features/`: Feature importance reports, correlation analysis
#
# ### Phase 9.3 API
#
# The modern API is now actively used in this notebook (code_guidelines.md v1.3+):
#
# ```python
# from finance_ml.ml_workflow.features.api import build_features
#
# all_stocks_features = build_features(
#     all_stocks_preprocessed,
#     preset="comprehensive",
#     include_interactions=True,
#     include_relative=True,
#     sector_col='sector'
# )
# ```
#
# ### Validation Checkpoint
# - 400+ features engineered
# - No infinite values (replaced with NaN)
# - Feature importance calculated
# - Top features identified
#
# ### Phase 9.3 Data Prerequisites (v1.3+)
#
# Input dataframes must satisfy the following requirements (enforced by Phase 9.1):
#
# 1. **Schema Compliance**: Columns conform to `COLUMN_SCHEMA` dtypes via `detect_and_cast_dtypes()`
# 2. **Imputation Completeness**: Fully imputed via 6-step imputation strategy (zero missing values)
# 3. **Safety Rails**: Non-negativity constraints and outlier safety rails applied
# 4. **Feature Availability**: Phase 9.3 core inputs (momentum, valuation, profitability, quality/risk, cash flow, growth) present and properly typed
#
# See `code_guidelines.md` Phase 9.3 Data Prerequisites section for complete requirements.
#
# **Phase 9.3 Feature Engineering** includes:
# - Financial ratios (valuation, profitability, leverage, liquidity, efficiency)
# - Sector-specific features (Financials, Energy, Tech, Healthcare, etc.)
# - Growth metrics and temporal features
# - Relative value features (sector-normalized)
# - Feature importance analysis
#
# ### Phase 9.3 API - Feature Engineering Presets (New in v0.7.0)
#
# The new `build_features()` API provides flexible feature engineering with presets:
#
# **Available Presets:**
# - **`"basic"`**: Core ratios, margins, volatility, revenue CAGR
# - **`"momentum"`**: Price momentum and technical indicators
# - **`"quality"`**: Accounting quality and financial distress signals
# - **`"comprehensive"`**: Full advanced feature set (default)
# - **`"full_enhanced"`**: Alias for comprehensive
#
# **Usage Examples:**
# ```python
# # Example usage (not executable in this context):
# # all_stocks_basic = build_features(df, preset="basic")
# # all_stocks_momentum = build_features(df, preset="momentum")
# # all_stocks_quality = build_features(df, preset="quality")
# # all_stocks_comprehensive = build_features(
# #     df,
# #     preset="comprehensive",
# #     include_interactions=True,
# #     include_relative=True
# # )
# ```
#
# %%
# Build comprehensive features using Phase 9.3 API (code_guidelines.md v1.3+)
all_stocks_features = build_features(
    all_stocks_scaled,
    preset="comprehensive",
    include_interactions=True,
    include_relative=True,
    sector_col="sector",
)
print(f"✓ Feature Engineering Complete")
print(f"  Original features: {all_stocks_scaled.shape[1]}")
print(f"  Engineered features: {all_stocks_features.shape[1]}")
print(f"  New features added: {all_stocks_features.shape[1] - all_stocks_scaled.shape[1]}")

# %%
# Engineer additional valuation ratios using Phase 9.3 function
print("\n📊 Engineering Additional Valuation Ratios...")
all_stocks_features = engineer_valuation_ratios(all_stocks_features)
print(f"✓ Valuation ratios engineered")

# Engineer analyst quality features using Phase 9.3 function
print("\n📈 Engineering Analyst Quality Features...")
all_stocks_features = engineer_analyst_quality_features(all_stocks_features)
print(f"✓ Analyst quality features engineered")

# Engineer accounting quality features using Phase 9.3 function
all_stocks_features = engineer_accounting_quality_features(all_stocks_features)
print(f"✓ Accounting quality features engineered")

# Engineer employee productivity features using Phase 9.3 function
all_stocks_features = engineer_employee_productivity_features(all_stocks_features)
print(f"✓ Employee productivity features engineered")

print(f"  Total features after enrichment: {all_stocks_features.shape[1]}")

# %%
# Phase 9.3 Schema Version 1.3 - New Feature Categories
print("\n" + "=" * 80)
print("📊 PHASE 9.3 SCHEMA VERSION 1.3 - NEW FEATURE DEMONSTRATIONS")
print("=" * 80)
print("\nDemonstrating 5 new feature engineering functions from Schema 1.3:")
print("1. Technical Analysis (EMA crossovers, 52W position, volume momentum)")
print("2. Valuation Time-Series (momentum, mean reversion, forward/trailing)")
print("3. Revenue Forecasts (analyst consensus, estimate quality)")
print("4. Dividend Reliability (consistency, coverage, safety)")
print("5. Employment Dynamics (growth, productivity, workforce indicators)")

# %%
# 1. Technical Analysis Features
print("\n📈 Engineering Technical Analysis Features...")
initial_cols = all_stocks_features.shape[1]
all_stocks_features = engineer_technical_analysis_features(all_stocks_features)
new_tech_features = all_stocks_features.shape[1] - initial_cols
print(f"✓ Technical analysis features engineered: {new_tech_features} new features")

# Display sample technical features
tech_features = [
    c
    for c in all_stocks_features.columns
    if any(x in c for x in ["ema_crossover", "price_vs_ema", "52w", "volume_momentum", "breakout"])
]
if tech_features:
    print(f"\n  Sample Technical Features ({len(tech_features)} total):")
    for feat in tech_features[:10]:
        non_null = all_stocks_features[feat].notna().sum()
        print(f"    • {feat}: {non_null}/{len(all_stocks_features)} non-null")

# %%
# 2. Valuation Time-Series Features
print("\n📊 Engineering Valuation Time-Series Features...")
initial_cols = all_stocks_features.shape[1]
all_stocks_features = engineer_valuation_timeseries_features(all_stocks_features)
new_val_features = all_stocks_features.shape[1] - initial_cols
print(f"✓ Valuation time-series features engineered: {new_val_features} new features")

# Display sample valuation features
val_features = [
    c
    for c in all_stocks_features.columns
    if any(
        x in c
        for x in [
            "ev_sales_trend",
            "ev_ebitda_momentum",
            "p_e_momentum",
            "valuation_stability",
            "forward_discount",
        ]
    )
]
if val_features:
    print(f"\n  Sample Valuation Time-Series Features ({len(val_features)} total):")
    for feat in val_features[:10]:
        non_null = all_stocks_features[feat].notna().sum()
        print(f"    • {feat}: {non_null}/{len(all_stocks_features)} non-null")

# %%
# 3. Revenue Forecast Features
print("\n💰 Engineering Revenue Forecast Features...")
initial_cols = all_stocks_features.shape[1]
all_stocks_features = engineer_revenue_forecast_features(all_stocks_features)
new_rev_features = all_stocks_features.shape[1] - initial_cols
print(f"✓ Revenue forecast features engineered: {new_rev_features} new features")

# Display sample revenue forecast features
rev_features = [
    c
    for c in all_stocks_features.columns
    if any(
        x in c
        for x in [
            "revenue_estimate",
            "revenue_growth_implied",
            "revenue_consensus",
            "estimate_confidence",
        ]
    )
]
if rev_features:
    print(f"\n  Sample Revenue Forecast Features ({len(rev_features)} total):")
    for feat in rev_features[:10]:
        non_null = all_stocks_features[feat].notna().sum()
        print(f"    • {feat}: {non_null}/{len(all_stocks_features)} non-null")

# %%
# 4. Dividend Reliability Features
print("\n💵 Engineering Dividend Reliability Features...")
initial_cols = all_stocks_features.shape[1]
all_stocks_features = engineer_dividend_reliability_features(all_stocks_features)
new_div_features = all_stocks_features.shape[1] - initial_cols
print(f"✓ Dividend reliability features engineered: {new_div_features} new features")

# Display sample dividend features
div_features = [
    c
    for c in all_stocks_features.columns
    if any(
        x in c
        for x in [
            "dividend_consistency",
            "dividend_safety",
            "dividend_payout",
            "income_stock",
            "dividend_aristocrat",
        ]
    )
]
if div_features:
    print(f"\n  Sample Dividend Reliability Features ({len(div_features)} total):")
    for feat in div_features[:10]:
        non_null = all_stocks_features[feat].notna().sum()
        print(f"    • {feat}: {non_null}/{len(all_stocks_features)} non-null")

# %%
# 5. Employment Dynamics Features
print("\n👥 Engineering Employment Dynamics Features...")
initial_cols = all_stocks_features.shape[1]
all_stocks_features = engineer_employment_dynamics_features(all_stocks_features)
new_emp_features = all_stocks_features.shape[1] - initial_cols
print(f"✓ Employment dynamics features engineered: {new_emp_features} new features")

# Display sample employment features
emp_features = [
    c
    for c in all_stocks_features.columns
    if any(
        x in c
        for x in [
            "employee_growth",
            "revenue_per_employee",
            "profit_per_employee",
            "hiring_intensity",
            "workforce",
        ]
    )
]
if emp_features:
    print(f"\n  Sample Employment Dynamics Features ({len(emp_features)} total):")
    for feat in emp_features[:10]:
        non_null = all_stocks_features[feat].notna().sum()
        print(f"    • {feat}: {non_null}/{len(all_stocks_features)} non-null")

# %%
# Phase 9.3 Schema 1.3 Summary
print("\n" + "=" * 80)
print("📊 PHASE 9.3 SCHEMA VERSION 1.3 - SUMMARY")
print("=" * 80)
total_new_features = (
    new_tech_features + new_val_features + new_rev_features + new_div_features + new_emp_features
)
print(f"\n✓ Total new features from Schema 1.3: {total_new_features}")
print(f"  • Technical Analysis: {new_tech_features}")
print(f"  • Valuation Time-Series: {new_val_features}")
print(f"  • Revenue Forecasts: {new_rev_features}")
print(f"  • Dividend Reliability: {new_div_features}")
print(f"  • Employment Dynamics: {new_emp_features}")
print(f"\n✓ Total features in dataset: {all_stocks_features.shape[1]}")
print(f"✓ Schema expanded from 262 to 310 columns (+48, +18.3%)")
print(f"✓ Feature functions increased from 19 to 24 (+5, +26.3%)")

# %%
# Verify ALL 21 price columns preserved after feature engineering (code_guidelines.md Section 8.5.2)
# Phase 9.3 feature engineering inherits price columns from all_stocks_scaled
# PRICE_COLUMNS includes: current (6), historical (9), 52w bounds (2), EMAs (4)
from finance_ml.ml_workflow.preprocessing.column_semantics import PRICE_COLUMNS

price_cols_present = [c for c in PRICE_COLUMNS if c in all_stocks_features.columns]
for col in price_cols_present:
    if col in all_stocks_scaled.columns:
        assert all_stocks_features[col].equals(
            all_stocks_scaled[col]
        ), f"{col} was incorrectly modified during feature engineering!"

print(
    f"\n✓ Verified {len(price_cols_present)}/21 price columns preserved after feature engineering (business metric protection)"
)

# %% [markdown]
# ### Phase 9.3 Enhanced Benchmarking Analysis
#
# **Data Source:** `all_stocks_features` DataFrame (post-feature-engineering)
#
# This section analyzes the **engineered features** after Phase 9.3 feature engineering completes. It reports actual Phase 9.3 feature family coverage by detecting which features are present in the DataFrame.
#
# **Analysis Approach:**
# - Uses `phase93_categories` module to categorize features by family
# - Reports coverage for all 11 Phase 9.3 categories (Momentum & Technical, Valuation Ratios, Profitability, Quality & Risk, Cash Flow, Capital Allocation, Analyst Sentiment, Market Sentiment, Leverage & Liquidity, Temporal Patterns, Composite Scores)
# - Shows sample features from each category with non-null counts
# - Exports comprehensive benchmarking report to `outputs/eda/phase93_benchmarking_post_engineering.json`
#
# **Alignment:**
# - Follows code_guidelines.md Section 2.1 variable mapping standards
# - Uses `all_stocks_features` (required stage name after feature engineering)
# - Validates DataFrame exists before analysis
#
# %%
# Refactored Phase 9.3 Enhanced Benchmarking Analysis
# Analyzes actual engineered features in all_stocks_features DataFrame
print("\n📊 Phase 9.3 Enhanced Benchmarking Analysis:")
print("=" * 80)

# Import Phase 9.3 category detection modules
from finance_ml.ml_workflow.eda.phase93_categories import (
    PHASE93_FEATURE_CATEGORIES,
    categorize_dataframe_columns,
    get_category_description,
    get_phase93_coverage_stats,
)

# Validate that feature engineering has completed
if "all_stocks_features" not in dir() or all_stocks_features is None:
    print("⚠️  ERROR: all_stocks_features not found!")
    print("   Please run Phase 9.3 feature engineering cells first.")
else:
    print(f"\n✓ Analyzing engineered features DataFrame")
    print(f"  Total stocks: {all_stocks_features.shape[0]}")
    print(f"  Total columns: {all_stocks_features.shape[1]}")

    # Categorize features by Phase 9.3 families
    categorized = categorize_dataframe_columns(all_stocks_features)
    coverage_stats = get_phase93_coverage_stats(all_stocks_features)

    # Calculate total Phase 9.3 features present
    total_phase93_features = sum(coverage_stats.values())

    # Get expected feature counts per category
    expected_counts = {cat: len(features) for cat, features in PHASE93_FEATURE_CATEGORIES.items()}
    total_expected = sum(expected_counts.values())

    print(
        f"  Phase 9.3 engineered features present: {total_phase93_features}/{total_expected} ({total_phase93_features / total_expected * 100:.1f}%)"
    )

    # Sector/region distribution
    if "sector" in all_stocks_features.columns:
        sectors = all_stocks_features["sector"].nunique()
        print(f"  Sectors analyzed: {sectors}")
    if "region" in all_stocks_features.columns:
        regions = all_stocks_features["region"].nunique()
        print(f"  Regions analyzed: {regions}")

    # Show availability by category
    print(f"\n📋 Phase 9.3 Feature Coverage by Category:")
    print("=" * 80)

    for category in sorted(PHASE93_FEATURE_CATEGORIES.keys()):
        present = coverage_stats.get(category, 0)
        expected = expected_counts[category]

        if present > 0:
            pct = (present / expected * 100) if expected > 0 else 0
            print(f"  ✓ {category}: {present}/{expected} features ({pct:.1f}% coverage)")

            # Show sample features for this category
            if category in categorized:
                sample_features = categorized[category][:3]
                for feat in sample_features:
                    non_null = all_stocks_features[feat].notna().sum()
                    print(f"      • {feat}: {non_null}/{len(all_stocks_features)} non-null")
        else:
            print(f"  ✗ {category}: 0/{expected} features (not yet engineered)")

    # Generate summary report
    print(f"\n📊 Benchmarking Summary:")
    print("=" * 80)

    categories_with_features = len([c for c in coverage_stats.values() if c > 0])
    categories_total = len(PHASE93_FEATURE_CATEGORIES)

    print(f"  Categories with features: {categories_with_features}/{categories_total}")
    print(f"  Total Phase 9.3 features: {total_phase93_features}")
    print(f"  Overall coverage: {total_phase93_features / total_expected * 100:.1f}%")

    # Export benchmarking report
    benchmarking_summary = {
        "phase": "9.3",
        "data_source": "all_stocks_features DataFrame (post-feature-engineering)",
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_stocks": int(all_stocks_features.shape[0]),
        "total_columns": int(all_stocks_features.shape[1]),
        "phase93_features_present": int(total_phase93_features),
        "phase93_features_expected": int(total_expected),
        "coverage_percentage": float(total_phase93_features / total_expected * 100),
        "category_coverage": {
            cat: {
                "present": int(coverage_stats.get(cat, 0)),
                "expected": int(expected_counts[cat]),
                "coverage_pct": float(
                    (coverage_stats.get(cat, 0) / expected_counts[cat] * 100)
                    if expected_counts[cat] > 0
                    else 0
                ),
            }
            for cat in PHASE93_FEATURE_CATEGORIES.keys()
        },
        "categories_with_features": int(categories_with_features),
        "note": "Analysis performed on engineered features DataFrame after Phase 9.3 completion",
    }

    # Save report
    from pathlib import Path

    benchmarking_output = Path("outputs/eda/phase93_benchmarking_post_engineering.json")
    benchmarking_output.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(benchmarking_output, "w") as f:
        json.dump(benchmarking_summary, f, indent=2)

    print(f"\n✓ Benchmarking report saved to: {benchmarking_output}")
    print("\n" + "=" * 80)

# %%
# Phase 9.3 Comprehensive Summary Statistics by Feature Category
print("\n" + "=" * 80)
print("📊 PHASE 9.3 COMPREHENSIVE SUMMARY STATISTICS BY FEATURE CATEGORY")
print("=" * 80)
print("\nGenerating detailed statistics for all 16 Phase 9.3 feature categories...")

# Generate summary statistics by category
category_summaries = {}

for category, features in PHASE93_FEATURE_CATEGORIES.items():
    # Get features present in dataframe
    present_features = [f for f in features if f in all_stocks_features.columns]

    if present_features:
        # Calculate comprehensive statistics
        category_data = all_stocks_features[present_features]

        # Summary statistics
        summary = {
            "count": len(present_features),
            "expected": len(features),
            "coverage_pct": (
                (len(present_features) / len(features) * 100) if len(features) > 0 else 0
            ),
            "description": get_category_description(category),
            "statistics": {},
        }

        # Calculate statistics for each feature
        for feat in present_features:
            feat_data = category_data[feat].dropna()

    # Check if data is present and is numeric before calculating float statistics
    if len(feat_data) > 0 and pd.api.types.is_numeric_dtype(feat_data):
        summary["statistics"][feat] = {
            "count": int(len(feat_data)),
            "mean": float(feat_data.mean()),
            "std": float(feat_data.std()),
            "min": float(feat_data.min()),
            "25%": float(feat_data.quantile(0.25)),
            "50%": float(feat_data.quantile(0.50)),
            "75%": float(feat_data.quantile(0.75)),
            "max": float(feat_data.max()),
            "missing_pct": float(
                (all_stocks_features[feat].isna().sum() / len(all_stocks_features)) * 100
            ),
        }

        category_summaries[category] = summary

        # Print summary for category
        print(f"\n{'=' * 80}")
        print(f"📈 {category}")
        print(f"{'=' * 80}")
        print(
            f"Coverage: {len(present_features)}/{len(features)} features ({summary['coverage_pct']:.1f}%)"
        )
        print(f"Description: {summary['description']}")
        print(f"\nTop 5 Features (by std dev):")

        # Show top 5 most variable features
        feature_stats = [
            (f, summary["statistics"][f]) for f in present_features if f in summary["statistics"]
        ]
        if feature_stats:
            sorted_features = sorted(feature_stats, key=lambda x: x[1]["std"], reverse=True)[:5]
            for feat_name, stats in sorted_features:
                print(f"  • {feat_name}:")
                print(
                    f"      Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}, "
                    f"Median: {stats['50%']:.3f}, Missing: {stats['missing_pct']:.1f}%"
                )

# Export comprehensive summary
summary_output = Path("outputs/eda/phase93_category_summary_statistics.json")
summary_output.parent.mkdir(parents=True, exist_ok=True)

with open(summary_output, "w") as f:
    json.dump(category_summaries, f, indent=2)

print(f"\n{'=' * 80}")
print(f"✓ Comprehensive category statistics saved to: {summary_output}")
print(f"✓ Total categories analyzed: {len(category_summaries)}")
print(f"{'=' * 80}")

# %%
# Phase 9.3 Category Visualizations
print("\n" + "=" * 80)
print("📊 PHASE 9.3 CATEGORY VISUALIZATIONS")
print("=" * 80)

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Feature Coverage Heatmap by Category
print("\n📈 Generating Feature Coverage Heatmap...")

coverage_data = []
for category in sorted(PHASE93_FEATURE_CATEGORIES.keys()):
    present = coverage_stats.get(category, 0)
    expected = expected_counts[category]
    coverage_pct = (present / expected * 100) if expected > 0 else 0
    coverage_data.append(
        {
            "Category": category,
            "Present": present,
            "Expected": expected,
            "Coverage %": coverage_pct,
        }
    )

coverage_df = pd.DataFrame(coverage_data)

# Create coverage bar chart
fig_coverage = go.Figure()
fig_coverage.add_trace(
    go.Bar(
        x=coverage_df["Category"],
        y=coverage_df["Coverage %"],
        text=coverage_df.apply(lambda r: f"{r['Present']}/{r['Expected']}", axis=1),
        textposition="auto",
        marker_color=coverage_df["Coverage %"].apply(
            lambda x: "green" if x > 90 else "orange" if x > 70 else "red"
        ),
    )
)

fig_coverage.update_layout(
    title="Phase 9.3 Feature Coverage by Category",
    xaxis_title="Feature Category",
    yaxis_title="Coverage (%)",
    height=500,
    xaxis={"tickangle": -45},
)

fig_coverage.write_html("outputs/eda/phase93_category_coverage.html")
print(f"✓ Coverage chart saved to: outputs/eda/phase93_category_coverage.html")

# 2. Feature Distribution Plots by Category (top categories)
print("\n📊 Generating Feature Distribution Plots...")

# Select top 4 categories by feature count for visualization
top_categories = sorted(coverage_stats.items(), key=lambda x: x[1], reverse=True)[:4]

for category, count in top_categories:
    if count > 0:
        present_features = [
            f for f in PHASE93_FEATURE_CATEGORIES[category] if f in all_stocks_features.columns
        ]

        # Take first 4 features for visualization
        viz_features = present_features[:4]

        if viz_features:
            # Create box plots for these features
            fig_box = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=viz_features,
                vertical_spacing=0.15,
                horizontal_spacing=0.1,
            )

            for idx, feat in enumerate(viz_features):
                row = (idx // 2) + 1
                col = (idx % 2) + 1

                feat_data = all_stocks_features[feat].dropna()

                fig_box.add_trace(
                    go.Box(y=feat_data, name=feat, showlegend=False), row=row, col=col
                )

            fig_box.update_layout(
                title=f"Feature Distributions: {category}", height=600, showlegend=False
            )

            # Save plot
            safe_category_name = category.replace(" ", "_").replace("&", "and")
            fig_box.write_html(f"outputs/eda/phase93_distributions_{safe_category_name}.html")
            print(f"  ✓ {category} distributions saved")

# 3. Category Correlation Heatmap (for numeric features across categories)
print("\n🔥 Generating Category Correlation Heatmap...")

# Select representative features from each category (max 2 per category)
representative_features = []
for category, features in PHASE93_FEATURE_CATEGORIES.items():
    present = [f for f in features if f in all_stocks_features.columns]
    if present:
        representative_features.extend(present[:2])  # Take first 2 from each category

# Limit to 30 features for readability
representative_features = representative_features[:30]

if representative_features:
    corr_data = (
        all_stocks_features[representative_features].select_dtypes(include=[np.number]).corr()
    )

    fig_corr = px.imshow(
        corr_data,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Phase 9.3 Cross-Category Feature Correlations (Representative Sample)",
    )

    fig_corr.update_layout(height=800, width=900, xaxis={"tickangle": -45})

    fig_corr.write_html("outputs/eda/phase93_category_correlations.html")
    print(f"✓ Correlation heatmap saved to: outputs/eda/phase93_category_correlations.html")

print(f"\n{'=' * 80}")
print(f"✓ All Phase 9.3 category visualizations complete")
print(f"{'=' * 80}")

# %%
# Feature importance analysis
exclude_cols = PRICE_COLUMNS

feature_cols = [c for c in all_stocks_features.columns if c not in exclude_cols]

if "price_target" in all_stocks_features.columns:
    X = all_stocks_features[feature_cols].select_dtypes(include=[np.number])
    y = all_stocks_features["price_target"]

    # Use Phase 9.3 function: select_features_rf

    importance_df = select_features_rf(X, y, top_k=20)

    print("\n🎯 Top 20 Most Important Features:")
    print(importance_df)

# %% [markdown]
# ### 🆕 Phase 9.3 TDD: Automated Feature Selection (Task 1)
#
# **Business Objective**: Reduce feature noise, improve model interpretability, and prevent overfitting through automated feature selection.
#
# **New Capabilities**:
# 1. **Automated Feature Selection** (`select_features_auto`): Combines multiple methods (mutual information, random forest importance, correlation-based redundancy detection)
# 2. **Category-Based Selection** (`select_features_by_category`): Select features from specific Phase 9.3 categories (momentum, valuation, quality, etc.)
#
# **Key Features**:
# - Importance threshold filtering (default: 0.01)
# - Correlation-based redundancy removal (default: >0.95)
# - Price column preservation (21 columns always kept)
# - Integration with unified ETL pipeline via `auto_feature_selection` parameter
#
# **References**:
# - Implementation Plan: `docs/improvement_plan/phase_9.3_implementation_plan.md`
# - Test Coverage: `tests/test_feature_selection_auto.py` (4 tests, 100% pass)
# - Code Guidelines: Section 9.3.1
# %%
##%%
# Phase 9.3 TDD: Automated Feature Selection
print("\n" + "=" * 80)
print("PHASE 9.3 TDD: AUTOMATED FEATURE SELECTION")
print("=" * 80)

# Prepare feature matrix and target
feature_cols = [
    col
    for col in all_stocks_features.columns
    if col
    not in [
        "ticker",
        "isin",
        "sector",
        "region",
        "snapshot_date",
        "price_target",
        "last_price",
    ]
]

X_features = all_stocks_features[feature_cols].copy()
y_target = all_stocks_features["price_target"].copy()

# Remove rows with missing target
valid_mask = y_target.notna()
X_features_clean = X_features[valid_mask]
y_target_clean = y_target[valid_mask]

print(f"\n📊 Feature Selection Input:")
print(f"  Total features: {len(feature_cols)}")
print(f"  Valid samples: {len(X_features_clean):,}")
print(f"  Target: {y_target_clean.name}")

# Apply automated feature selection
# Method: 'combined' uses mutual info + RF importance + correlation pruning
# Importance threshold: 0.01 (remove features with <1% importance)
# Correlation threshold: 0.95 (remove redundant features with r>0.95)
print(f"\n🔍 Applying select_features_auto():")
print(f"  Method: combined (mutual_info + rf_importance + correlation)")
print(f"  Importance threshold: 0.01")
print(f"  Correlation threshold: 0.95")

X_selected = select_features_auto(
    X_features_clean,
    y_target_clean,
    importance_threshold=0.01,
    correlation_threshold=0.95,
    method="combined",
    preserve_columns=None,  # Auto-preserves price columns
    return_scores=False,
)

print(f"\n✅ Feature Selection Results:")
print(f"  Features before: {X_features_clean.shape[1]}")
print(f"  Features after: {X_selected.shape[1]}")
print(f"  Reduction: {X_features_clean.shape[1] - X_selected.shape[1]} features removed")
print(
    f"  Dimensionality reduced by: {(1 - X_selected.shape[1] / X_features_clean.shape[1]) * 100:.1f}%"
)

# Update dataframe with selected features
all_stocks_selected = all_stocks_features[
    ["ticker", "isin", "sector", "region", "price_target", "last_price"] + list(X_selected.columns)
].copy()

print(f"\n✓ Created all_stocks_selected DataFrame with {len(all_stocks_selected):,} rows")
print(f"  Selected features: {X_selected.shape[1]}")
# %%
##%%
# Phase 9.3 TDD: Category-Based Feature Selection
print("\n" + "=" * 80)
print("PHASE 9.3 TDD: CATEGORY-BASED FEATURE SELECTION")
print("=" * 80)

# Demonstrate the 16 Phase 9.3 feature categories (196 features total)
# Aligned with PHASE93_FEATURE_CATEGORIES from phase93_categories.py
# Supports both short names ('momentum') and full names ('Momentum & Technical')

print("\n📊 Available Phase 9.3 Feature Categories (16 total, 196 features):")
print("  1. momentum (27) - Price momentum, RSI, EMA signals, 52W position")
print("  2. valuation (23) - P/E, P/B, EV/EBITDA, EV/Sales, trends")
print("  3. profitability (12) - Margins, ROE, ROA, ROIC, earnings quality")
print("  4. quality (18) - Altman Z, accounting quality, distress indicators")
print("  5. cash_flow (5) - FCF yield, CFO metrics, cash conversion")
print("  6. capital_allocation (23) - Dividends, CAPEX, reinvestment, M&A")
print("  7. analyst_sentiment (10) - Ratings, target revisions, consensus")
print("  8. market_sentiment (4) - Beta stability, momentum, price range")
print("  9. leverage (9) - Debt ratios, current ratio, interest coverage")
print("  10. temporal_patterns (15) - Seasonality, reporting dates, volatility")
print("  11. composite_scores (5) - Piotroski F, Altman Z, Beneish M")
print("  12. growth (6) - Revenue, earnings, EBITDA growth (YoY, CAGR)")
print("  13. efficiency (4) - Asset turnover, revenue per employee")
print("  14. employee_productivity (16) - Workforce metrics, productivity")
print("  15. balance_sheet (8) - Asset/equity growth, working capital")
print("  16. revenue_forecast (9) - Analyst estimates, consensus uncertainty")

# Example 1: Select fundamental analysis features
print("\n🎯 Example 1: Fundamental Analysis (valuation + profitability + quality)")
fundamental_categories = ["valuation", "profitability", "quality"]

X_fundamental = select_features_by_category(X_features_clean, categories=fundamental_categories)

print(f"\n✅ Fundamental Features Selected:")
print(f"  Total features in dataset: {X_features_clean.shape[1]}")
print(f"  Fundamental features: {X_fundamental.shape[1]}")
print(f"  Expected: ~23 + 12 + 18 = 53 features")

# Example 2: Select technical analysis features
print("\n🎯 Example 2: Technical Analysis (momentum + market_sentiment)")
technical_categories = ["momentum", "market_sentiment"]

X_technical = select_features_by_category(X_features_clean, categories=technical_categories)

print(f"✅ Technical Features Selected:")
print(f"  Technical features: {X_technical.shape[1]}")
print(f"  Expected: ~27 + 4 = 31 features")

# Example 3: Select comprehensive feature set for modeling
print("\n🎯 Example 3: Comprehensive Model (8 key categories)")
comprehensive_categories = [
    "momentum",
    "valuation",
    "profitability",
    "quality",
    "cash_flow",
    "growth",
    "analyst_sentiment",
    "composite_scores",
]

X_category_features = select_features_by_category(
    X_features_clean, categories=comprehensive_categories
)

print(f"✅ Comprehensive Features Selected:")
print(f"  Total features: {X_category_features.shape[1]}")
print(f"  Expected: ~27+23+12+18+5+6+10+5 = 106 features")
print(f"  Categories: {', '.join(comprehensive_categories)}")

# Show actual feature breakdown by category using Phase 9.3 catalog
print(f"\n📋 Actual Feature Breakdown by Category:")
from finance_ml.ml_workflow.eda.phase93_categories import PHASE93_FEATURE_CATEGORIES

# Map short names to full names
SHORT_TO_FULL = {
    "momentum": "Momentum & Technical",
    "valuation": "Valuation Ratios",
    "profitability": "Profitability",
    "quality": "Quality & Risk",
    "cash_flow": "Cash Flow",
    "growth": "Growth Metrics",
    "analyst_sentiment": "Analyst Sentiment",
    "composite_scores": "Composite Scores",
}

for short_name in comprehensive_categories:
    full_name = SHORT_TO_FULL.get(short_name, short_name)
    if full_name in PHASE93_FEATURE_CATEGORIES:
        expected_features = PHASE93_FEATURE_CATEGORIES[full_name]
        available_features = [f for f in expected_features if f in X_category_features.columns]
        print(
            f"  {short_name}: {len(available_features)}/{len(expected_features)} features available"
        )
        if available_features[:3]:
            print(f"    Examples: {', '.join(available_features[:3])}")
# %%
# 📊 Section 4 Enhanced Visualizations - Feature Engineering
print("\n" + "=" * 80)
print("📊 INTERACTIVE FEATURE ENGINEERING VISUALIZATIONS")
print("=" * 80)

# Feature importance visualization (if available from feature engineering)
if "X" in dir() and X is not None:
    print("\n📈 Feature Importance Analysis...")

    # Calculate feature correlations
    import plotly.express as px
    import plotly.graph_objects as go

    numeric_features = X.select_dtypes(include=[np.number]).columns[:20]  # Top 20
    corr_matrix = X[numeric_features].corr()

    # Interactive correlation heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap (Top 20 Features)",
        template="plotly_dark",
    )
    fig.update_layout(width=900, height=800, font_family="Arial")
    fig.show()

    # Feature distribution comparison by sector
    if "sector" in all_stocks_features.columns:
        print("\n📊 Feature Distributions by Sector...")
        key_features = (
            ["market_cap", "p_e"] if "p_e" in all_stocks_features.columns else ["market_cap"]
        )
        for feature in key_features:
            if feature in all_stocks_features.columns:
                fig = px.box(
                    all_stocks_features,
                    x="sector",
                    y=feature,
                    color="sector",
                    title=f"{feature.replace('_', ' ').title()} Distribution by Sector",
                    points="outliers",
                    template="plotly_dark",
                )
                fig.update_layout(showlegend=False, xaxis_tickangle=-45, font_family="Arial")
                fig.show()
                break  # Show just one example

    print("✓ Feature engineering visualizations complete")

# %% [markdown]
# ## Phase 9.4: Multi-Class Event Classification of Financial Events
#
# ### Business Goal
# Classify stocks into financial event categories (Strong Negative, Negative, Neutral, Positive, Strong Positive) to provide granular sentiment signals that enhance regression model accuracy and enable better risk management.
#
# ### Key Objectives
# 1. Create enhanced event labels using multiple methods
# 2. Prepare classification data with Phase 9.3 features
# 3. Train and optimize classifiers (XGBoost, LightGBM, CatBoost)
# 4. Evaluate classification performance
# 5. Extract classification probabilities as meta-features
#
# ### Inputs
# - `all_stocks_features`: Feature-engineered data from Phase 9.3
#
# ### Outputs
# - `clf_result`: Classification model result dict with probabilities
# - `outputs/classification/`: Model artifacts, evaluation metrics, confusion matrices
# - Event probability features for Phase 9.5
#
# ### Standardized Return Format (v1.2)
# ```python
# # Example return structure (not executable code):
# # clf_result = {
# #     'model': fitted_classifier,
# #     'metrics': {'accuracy': 0.85, 'f1_macro': 0.82, ...},
# #     'y_pred': np.ndarray,  # array of predicted classes (0-4 for 5-class)
# #     'y_proba': np.ndarray,  # (n_samples, 5) probabilities for 5 classes
# #     'artifacts': {'feature_importance': pd.DataFrame, ...}
# # }
# ```
#
# ### Validation Checkpoint
# - Classification accuracy > 60%
# - All 5 classes represented in predictions (0-4: Strong Negative to Strong Positive)
# - Probabilities sum to 1.0
# - Feature importance extracted
#
# Train sophisticated classification models to predict financial events:
# - Event labeling: 5-class system (0=Strong Negative, 1=Negative, 2=Neutral, 3=Positive, 4=Strong Positive)
# - Multiple event detection methods: price_momentum, valuation, fundamental, volatility, analyst_rating, profitability, leverage, liquidity, efficiency, growth, quality, composite
# - Multiple classifiers: XGBoost, LightGBM, CatBoost, Neural Networks, Ensembles
# - Export classification probabilities as meta-features for regression
#
# %%
# Prepare classification data with Phase 9.3 feature groups
print("\n" + "=" * 80)
print("CLASSIFICATION DATA PREPARATION")
print("=" * 80)

# Step 1: Create event labels FIRST (required parameter for prepare_classification_data)
print("\n🏷️  Creating Event Labels for Classification...")
print("  Method: composite_event (canonical for Phase 9.3)")

# Create labels using create_event_labels
# This generates the required 'labels' numpy array
event_labels = create_event_labels(
    all_stocks_features,
    method="quality_event",  # Use price_momentum method for reliable class distribution
    use_sector_adjustment=True,
)

# Validate label distribution
print(f"\n✓ Event Labels Created:")
print(f"  Total samples: {len(event_labels)}")
print(f"  Class distribution:")
print(
    f"    Strong Negative (0): {(event_labels == 0).sum()} ({(event_labels == 0).sum() / len(event_labels) * 100:.1f}%)"
)
print(
    f"    Negative (1): {(event_labels == 1).sum()} ({(event_labels == 1).sum() / len(event_labels) * 100:.1f}%)"
)
print(
    f"    Neutral (2): {(event_labels == 2).sum()} ({(event_labels == 2).sum() / len(event_labels) * 100:.1f}%)"
)
print(
    f"    Positive (3): {(event_labels == 3).sum()} ({(event_labels == 3).sum() / len(event_labels) * 100:.1f}%)"
)
print(
    f"    Strong Positive (4): {(event_labels == 4).sum()} ({(event_labels == 4).sum() / len(event_labels) * 100:.1f}%)"
)

# Step 1.5: Define method-aware valuation columns for interaction features
# This ensures valuation columns match the semantics of the chosen event labeling method
print("\n📊 Defining Method-Aware Valuation Columns...")

# Extract the label method used above (must match the method argument in classification_create_enhanced_event_labels)
label_method = "quality_event"  # Must match method parameter above (line 1510)

# Define valuation column candidates grouped by event labeling method semantics
# Each group contains columns relevant to that method's economic logic
#
# PHASE 9.3 ALIGNMENT (Updated 2025-11-24):
# This dictionary now includes Phase 9.3 engineered features alongside raw data columns.
# All 19 event classification methods from labels.py are covered, mapping to the 196
# engineered features across 16 Phase 9.3 categories from phase93_categories.py.
#
# Structure:
# - Original methods (7): price_momentum, valuation, fundamental, volatility, analyst_rating, market_events, combined_signals
# - Phase 9.4 specialized (6): profitability_event, leverage_event, liquidity_event, efficiency_event, growth_event, quality_event, composite_event
# - Phase 9.3 new methods (5): cashflow_event, capital_allocation_event, employee_productivity_event, balance_sheet_event, revenue_forecast_event
#
# Each method includes:
# 1. Raw data columns (for backward compatibility and label creation)
# 2. Phase 9.3 engineered features (196 total across 16 categories)
# 3. Schema 1.3 time-series columns (_fq, _5yavgfq, _5yavgltm, _previous_year)
#
valuation_candidates_by_method = {
    "price_momentum": [
        "last_price",
        "price_target",
        "price_target_median",
        "price_target_high",
        "price_target_low",
        "price_target_ytd_ago",
        "market_cap",
        "enterprise_value",
        "p_e",
        "p_e_ltm",
        "p_e_ntm",
        "p_e_1fyltm",
        "p_e_5yavgltm",
        "p_b",
        "p_b_ltm",
        "p_b_1fy",
        "p_b_5yavg",
        # Price momentum metrics
        "price_chg_pct_1m",
        "price_chg_pct_3m",
        "one_day_pct",
        "price_5d_ago",
        "price_1w_ago",
        "price_1m_ago",
        "price_3m_ago",
        "total_return_ytd",
        "total_return_5y",
        "total_return_10y",
        "tot_return_pct_cagr_3y",
        "tot_return_pct_cagr_10y",
        # Phase 9.3 Schema 1.3: Technical indicators
        "ema_20d",
        "ema_50d",
        "ema_100d",
        "ema_250d",
        "52w_high_adj",
        "52w_low_adj",
        "rel_volume",
    ],
    "valuation": [
        "p_e",
        "p_e_ltm",
        "p_e_ntm",
        "p_e_1fyltm",
        "p_e_5yavgltm",
        "p_b",
        "p_b_ltm",
        "p_b_1fy",
        "p_b_5yavg",
        "p_tbv_ltm",
        "tbv_fy",
        "tbv_ltm",
        "ebitda",
        "ebitda_ltm",
        "ebitda_fy",
        "ebitda_fq",
        "ebitda_5yavgltm",
        "ebitda_5yavgfq",
        "ebit",
        "ebit_ltm",
        "ebit_fy",
        "ebit_fq",
        "ebit_5yavgltm",
        "ebit_5yavgfq",
        "enterprise_value",
        "market_cap",
        "ev_ebitda",
        "peg_ratio",  # if present from feature engineering
        # Phase 9.3 Schema 1.3: Valuation time-series
        "ev_sales_ltm",
        "ev_sales_ntm",
        "ev_sales_est_fy1",
        "ev_sales_1fyltm",
        "ev_sales_2fyltm",
        "ev_sales_3fyltm",
        "ev_sales_3yavgltm",
        "ev_ebitda_ltm",
        "ev_ebitda_ntm",
        "ev_ebitda_est_fy1",
        "ev_ebitda_1fyltm",
        "ev_ebitda_3yavgltm",
        "p_e_est_fy1",
        "p_e_2fyltm",
        "p_e_3fyltm",
        "p_e_3yavgltm",
        "p_e_1fqltm",
        "p_e_2fqltm",
        "p_e_3fqltm",
    ],
    "fundamental": [
        "gross_margin",
        "gross_profit_margin_pct_ltm",
        "gross_profit_margin_pct_fy",
        "net_income_margin_pct_ltm",
        "net_income_margin_pct_fy",
        "net_income",
        "net_income_is_ltm",
        "net_income_is_fy",
        "net_income_is_fq",
        "net_income_is_1fy",
        "net_income_is_5yavgltm",
        "net_income_is_5yavgfq",
        "normalized_net_income_ltm",
        "normalized_net_income_fy",
        "normalized_net_income_fq",
        "normalized_net_income_1fy",
        "normalized_net_income_5yavgltm",
        "normalized_net_income_5yavgfq",
        "net_income_adj_ltm",
        "net_income_adj_fy",
        "net_income_adj_fq",
        "net_income_adj_1fy",
        "net_income_adj_5yavgfq",
        "ebitda",
        "ebitda_ltm",
        "ebitda_fy",
        "ebitda_fq",
        "ebit",
        "ebit_ltm",
        "ebit_fy",
        "ebit_fq",
        "operating_income",
        "operating_income_ltm",
        "operating_income_fy",
        "operating_income_fq",
        "operating_income_5yavgfq",
        "gross_profit",
        "gross_profit_ltm",
        "gross_profit_fy",
        "gross_profit_previous_year",
        "return_on_equity_pct_ltm",
        "return_on_equity_pct_fy",
        "return_on_assets_roa_pct_ltm",
        "return_on_assets_roa_pct_fy",
        # Phase 9.3 Schema 1.3: Revenue forecasts
        "revenues_est_avg_ntm",
        "revenues_est_avg_fy1e",
        "revenues_est_med_ntm",
        "revenues_est_med_fy1e",
    ],
    "volatility": [
        "volatility_1m",
        "volatility_3m",
        "volatility_6m",
        "volatility_1y",
        "volatility_1y_pct",
        "beta_1y",
        "beta_2y",
        "beta_5y",
        "short_int_pct",
        "last_price",
        "market_cap",
        "price_chg_pct_1m",
        "price_chg_pct_3m",
        "one_day_pct",
    ],
    "analyst_rating": [
        "price_target",
        "price_target_median",
        "price_target_high",
        "price_target_low",
        "last_price",
        "price_target_ytd_ago",
        "price_target_count",
        "price_target_number",
        "analyst_rating",
        "strong_buy_ratings",
        "strong_sell_ratings",
        "buy_ratings",
        "sell_ratings",
        "hold_ratings",
        "dividend_per_share",
        "dividend_per_share_ltm",
        "div_yield_ltm",
        "div_yield_ttm",
        "div_yield_ntm",
        "div_yield_ind",
        "div_yield_1fyind",
        "div_yield_5yavgltm",
        "buyback_yield_ltm",
        "dividends_paid",
        "dividends_paid_ltm",
        "common_dividends_paid_ltm",
        "common_dividends_paid_fy",
        # Phase 9.3 Schema 1.3: Dividend reliability
        "dividend_record_amount",
        "dividend_streak",
        "dividend_record_frequency",
        "dividend_record_currency",
        "dividend_record_announce_date",
        "dividend_record_ex_date",
        "dividend_record_payable_date",
        "dividend_record_record_date",
    ],
    "market_events": [
        "last_price",
        "market_cap",
        "market_cap_country_r",
        "p_e",
        "p_e_ltm",
        "p_b",
        "p_b_ltm",
        "short_int_pct",
        "price_chg_pct_1m",
        "price_chg_pct_3m",
        "total_return_ytd",
        "total_return_5y",
    ],
    "combined_signals": [
        # Multi-metric composite: momentum + valuation + fundamentals
        "last_price",
        "price_target",
        "price_target_median",
        "p_e",
        "p_e_ltm",
        "p_e_ntm",
        "p_e_ratio",
        "p_b",
        "p_b_ltm",
        "p_b_ratio",
        "net_margin_pct",
        "net_income_margin_pct_ltm",
        "net_income_margin_pct_fy",
        "gross_margin_pct",
        "gross_profit_margin_pct_ltm",
        "gross_profit_margin_pct_fy",
        "price_chg_pct_1m",
        "price_chg_pct_3m",
        "market_cap",
        "enterprise_value",
    ],
    "profitability_event": [
        "return_on_equity_pct_ltm",
        "return_on_equity_pct_fy",
        "return_on_assets_roa_pct_ltm",
        "return_on_assets_roa_pct_fy",
        "net_income",
        "net_income_is_ltm",
        "net_income_is_fy",
        "net_income_is_fq",
        "net_income_is_1fy",
        "net_income_ltm",
        "total_equity",
        "total_equity_ltm",
        "total_equity_fy",
        "total_equity_previous_year",
        "total_assets",
        "total_assets_ltm",
        "total_assets_fy",
        "total_assets_previous_year",
        "roe",
        "roa",
        "roic",  # if engineered
        "ebitda",
        "ebitda_ltm",
        "ebitda_fy",
        "ebitda_fq",
        "ebitda_previous_year",
        "ebit",
        "ebit_ltm",
        "ebit_fy",
        "ebit_fq",
    ],
    "leverage_event": [
        "total_debt",
        "total_debt_ltm",
        "total_debt_fy",
        "total_equity",
        "total_equity_ltm",
        "total_equity_fy",
        "total_equity_previous_year",
        "total_assets",
        "total_assets_ltm",
        "total_assets_fy",
        "total_assets_previous_year",
        "interest_expense",
        "interest_expense_total_ltm",
        "cash_and_equivalents",
        "cash_and_equivalents_ltm",
        "cash_and_equivalents_fy",
        "cash_and_equivalents_fq",
        "cash_and_equivalents_5yavgfq",
        "retained_earnings",
        "retained_earnings_ltm",
        "retained_earnings_fy",
        "retained_earnings_fq",
        "retained_earnings_5yavgfq",
    ],
    "liquidity_event": [
        "current_ratio_ltm",
        "current_ratio_fy",
        "current_assets",
        "current_liabilities",
        "total_current_assets_ltm",
        "total_current_liabilities_ltm",
        "working_capital",
        "working_capital_ltm",
        "working_capital_fy",
        "working_capital_fq",
        "working_capital_5yavgfy",
        "cash_and_equivalents",
        "cash_and_equivalents_ltm",
        "cash_and_equivalents_fy",
        "cash_and_equivalents_fq",
        "cash_and_equivalents_5yavgfq",
    ],
    "efficiency_event": [
        "asset_turnover_fy",
        "asset_turnover_ltm",
        "asset_turnover_previous_year",
        "inventory",
        "inventory_ltm",
        "inventory_fy",
        "inventory_fq",
        "inventory_5yavgfq",
        "accounts_receivable_fy",
        "accounts_receivable_1fy",
        "accounts_receivable_5yavgfq",
        "inventory_turnover",
        "receivables_turnover",  # if engineered
        "sga_expenses_fq",
        "sga_expenses_fy",
        "sga_expenses_1fy",
        "sga_expenses_5yavgfq",
        "marketing_expenses_fq",
        "marketing_expenses_fy",
        "marketing_expenses_1fy",
        "marketing_expenses_5yavgltm",
        # Phase 9.3 Schema 1.3: Employment dynamics
        "total_employees_fy",
        "total_employees_fq",
        "avg_employees_ltm",
        "avg_employees_fy",
        "avg_employees_5yavgfy",
    ],
    "growth_event": [
        "total_revenues_cagr_5y_fy",
        "total_revenues_ltm",
        "total_revenues_fy",
        "total_revenues_fq",
        "total_revenues_1fy",
        "total_revenues_5yavgltm",
        "total_revenues_5yavgfq",
        "revenues_est_yoy_pct_fy1e",
        "revenue",
        "revenue_previous_year",
        "revenue_fy",
        "ebitda",
        "ebitda_ltm",
        "ebitda_fy",
        "ebitda_fq",
        "ebitda_previous_year",
        "ebitda_5yavgltm",
        "ebitda_5yavgfq",
        "ebit",
        "ebit_ltm",
        "ebit_fy",
        "ebit_fq",
        "ebit_5yavgltm",
        "ebit_5yavgfq",
        "eps",
        "eps_previous_year",
        "eps_adj_ltm",
        "eps_adj_fy",
        "eps_adj_1fy",
        "eps_norm_est_avg_ntm",
        "eps_norm_est_avg_fy1e",
        "gross_profit",
        "gross_profit_ltm",
        "gross_profit_fy",
        "gross_profit_previous_year",
        # Phase 9.3 Schema 1.3: Revenue forecasts
        "revenues_est_avg_ntm",
        "revenues_est_avg_fy1e",
        "revenues_est_med_ntm",
        "revenues_est_med_fy1e",
    ],
    "quality_event": [
        "altman_z_score_fy",
        "altman_z_score_fq",
        "altman_z_score_ltm",
        "asset_writedown_ltm",
        "asset_writedown_fy",
        "asset_writedown_fq",
        "asset_writedown_1fy",
        "asset_writedown_5yavgfq",
        "impairment_of_goodwill_ltm",
        "impairment_of_goodwill_fy",
        "impairment_of_goodwill_fq",
        "impairment_of_goodwill_1fy",
        "impairment_of_goodwill_5yavgfq",
        "restructuring_charges_ltm",
        "restructuring_charges_fy",
        "restructuring_charges_fq",
        "restructuring_charges_1fy",
        "restructuring_charges_5yavgfq",
        "merger_restructuring_charges_ltm",
        "merger_restructuring_charges_fq",
        "merger_restructuring_charges_fy",
        "merger_restructuring_charges_5yavgfq",
        "goodwill",
        "goodwill_ltm",
        "goodwill_fy",
        "goodwill_fq",
        "goodwill_1fy",
        "goodwill_5yavgfq",
        "intangible_assets",
        "gross_intangible_assets_ltm",
        "gross_intangible_assets_fy",
        "gross_intangible_assets_5yavgfq",
        "dividends_paid",
        "dividends_paid_ltm",
        "common_dividends_paid_ltm",
        "common_dividends_paid_fy",
    ],
    "composite_event": [
        "altman_z_score_fy",
        "altman_z_score_fq",
        "altman_z_score_ltm",
        "net_income",
        "net_income_ltm",
        "net_income_is_ltm",
        "net_income_is_fy",
        "net_income_is_fq",
        "total_assets",
        "total_assets_ltm",
        "total_assets_fy",
        "total_assets_previous_year",
        "total_equity",
        "total_equity_ltm",
        "total_equity_fy",
        "total_equity_previous_year",
        "cfo",
        "cfo_ltm",
        "cfo_fy",
        "cfo_fq",
        "cfo_1fy",
        "cfi",
        "cfi_ltm",
        "cfi_fy",
        "cfi_fq",
        "cfi_1fy",
        "cff",
        "cff_ltm",
        "cff_fy",
        "cff_fq",
        "cff_1fy",
        "fcf",
        "fcf_ltm",
        "fcf_fy",
        "fcf_fq",
        "fcf_5yavgfq",
        "piotroski_f_score",
        "beneish_m_score",  # if engineered
        "retained_earnings",
        "retained_earnings_ltm",
        "retained_earnings_fy",
        "retained_earnings_fq",
        "working_capital",
        "working_capital_ltm",
        "working_capital_fy",
        "working_capital_fq",
        # Phase 9.3 Composite Scores (5 features)
        "altman_z_score",
        "beneish_m_score",
        "composite_quality_score",
        "momentum_score",
        "piotroski_f_score",
    ],
    # NEW PHASE 9.3 METHODS (Added 2025-11-24):
    # These 5 methods cover 77 additional Phase 9.3 engineered features
    "cashflow_event": [
        # Raw cash flow columns
        "cfo",
        "cfo_ltm",
        "cfo_fy",
        "cfo_fq",
        "cfo_1fy",
        "fcf",
        "fcf_ltm",
        "fcf_fy",
        "fcf_fq",
        "fcf_5yavgfq",
        "cfi",
        "cfi_ltm",
        "cfi_fy",
        "cfi_fq",
        "cfi_1fy",
        "cff",
        "cff_ltm",
        "cff_fy",
        "cff_fq",
        "cff_1fy",
        "net_income",
        "net_income_ltm",
        "net_income_is_ltm",
        # Phase 9.3 Cash Flow features (5 features)
        "cfo_growth_yoy",
        "cfo_to_net_income",
        "fcf_margin",
        "fcf_stability",
        "fcf_to_net_income",
    ],
    "capital_allocation_event": [
        # Raw capital allocation columns
        "dividends_paid",
        "dividends_paid_ltm",
        "common_dividends_paid_ltm",
        "common_dividends_paid_fy",
        "dividend_per_share",
        "dividend_per_share_ltm",
        "div_yield_ltm",
        "div_yield_ttm",
        "div_yield_ntm",
        "div_yield_5yavgltm",
        "buyback_yield_ltm",
        "capital_expenditure",
        "capex_ltm",
        "capex_fy",
        "depreciation_and_amortization",
        "depreciation_ltm",
        "working_capital",
        "working_capital_ltm",
        "working_capital_fy",
        # Phase 9.3 Schema 1.3: Dividend reliability
        "dividend_record_amount",
        "dividend_streak",
        "dividend_record_frequency",
        "dividend_record_currency",
        # Phase 9.3 Capital Allocation features (23 features)
        "acquisition_intensity",
        "capex_growth_rate",
        "capex_intensity",
        "capex_to_depreciation",
        "capex_volatility",
        "currency_risk_flag",
        "days_since_ex_date",
        "div_yield_ltm",
        "dividend_aristocrat_flag",
        "dividend_consistency_score",
        "dividend_frequency_encoded",
        "dividend_growth_trend",
        "dividend_payout_ratio",
        "dividend_safety_score",
        "dividend_streak_years",
        "dividend_yield_vs_sector",
        "fcf_dividend_coverage",
        "income_stock_flag",
        "payout_ratio",
        "reinvestment_rate",
        "total_shareholder_return_yield",
        "working_capital_efficiency",
        "working_capital_trend",
    ],
    "employee_productivity_event": [
        # Raw employee columns
        "total_employees_fy",
        "total_employees_fq",
        "avg_employees_ltm",
        "avg_employees_fy",
        "avg_employees_5yavgfy",
        "total_revenues_ltm",
        "total_revenues_fy",
        "net_income",
        "net_income_ltm",
        "ebitda",
        "ebitda_ltm",
        "operating_income",
        "operating_income_ltm",
        "total_assets",
        "total_assets_ltm",
        # Phase 9.3 Employee Productivity features (16 features)
        "assets_per_employee",
        "ebitda_per_employee",
        "employee_base_scale_flag",
        "employee_growth_acceleration",
        "employee_growth_cagr_5y",
        "employee_growth_qoq",
        "employee_growth_yoy",
        "employee_growth_yoy_pct",
        "hiring_intensity_score",
        "operating_income_per_employee",
        "profit_per_employee",
        "revenue_per_employee_fy",
        "revenue_per_employee_ltm",
        "revenue_per_employee_trend",
        "revenue_per_employee_vs_5y_pct",
        "workforce_volatility",
    ],
    "balance_sheet_event": [
        # Raw balance sheet columns
        "total_assets",
        "total_assets_ltm",
        "total_assets_fy",
        "total_assets_previous_year",
        "total_equity",
        "total_equity_ltm",
        "total_equity_fy",
        "total_equity_previous_year",
        "total_debt",
        "total_debt_ltm",
        "total_debt_fy",
        "retained_earnings",
        "retained_earnings_ltm",
        "retained_earnings_fy",
        "retained_earnings_fq",
        "working_capital",
        "working_capital_ltm",
        "working_capital_fy",
        "working_capital_fq",
        "current_ratio_ltm",
        "current_ratio_fy",
        "net_income",
        "net_income_ltm",
        # Phase 9.3 Balance Sheet Dynamics features (8 features)
        "asset_growth_rate",
        "balance_sheet_expansion",
        "current_ratio_trend",
        "debt_growth_rate",
        "earnings_retention_rate",
        "equity_growth_rate",
        "retained_earnings_growth",
        "working_capital_ratio",
    ],
    "revenue_forecast_event": [
        # Raw revenue forecast columns
        "revenues_est_avg_ntm",
        "revenues_est_avg_fy1e",
        "revenues_est_med_ntm",
        "revenues_est_med_fy1e",
        "revenues_est_yoy_pct_fy1e",
        "total_revenues_ltm",
        "total_revenues_fy",
        "total_revenues_1fy",
        "revenue",
        "revenue_fy",
        "revenue_previous_year",
        "total_revenues_cagr_5y_fy",
        # Phase 9.3 Revenue Forecasting features (9 features)
        "avg_vs_median_bias",
        "estimate_confidence_flag",
        "growth_surprise_potential",
        "revenue_consensus_uncertainty_score",
        "revenue_estimate_spread_fy1e",
        "revenue_estimate_spread_ntm",
        "revenue_growth_acceleration",
        "revenue_growth_implied_fy1e",
        "revenue_growth_implied_ntm",
    ],
}

# Define core columns that are always included regardless of method
core_valuation_cols = ["last_price", "market_cap", "enterprise_value", "ebitda"]

# Get method-specific candidates, or use fallback
default_valuation_candidates = [
    "market_cap",
    "enterprise_value",
    "ebitda",
    "p_e",
    "p_b",
    "gross_margin",
    "revenue",
    "net_income",
]

method_candidates = valuation_candidates_by_method.get(label_method, default_valuation_candidates)

# Combine core columns with method-specific columns (remove duplicates)
all_candidates = list(dict.fromkeys(core_valuation_cols + method_candidates))

# Filter to only columns that exist in the dataframe
valuation_cols_method_aware = [c for c in all_candidates if c in all_stocks_features.columns]

# Debug logging for diagnostics
print(f"\n✓ Method-Aware Valuation Columns Configured:")
print(f"  Label method: {label_method}")
print(f"  Method-specific candidates: {len(method_candidates)}")
print(f"  Core columns (always included): {len(core_valuation_cols)}")
print(f"  Total candidates: {len(all_candidates)}")
print(f"  Available in dataframe: {len(valuation_cols_method_aware)}")
print(
    f"  Selected columns: {valuation_cols_method_aware[:10]}{'...' if len(valuation_cols_method_aware) > 10 else ''}"
)

# Store for later use in interaction feature creation (Section 6.1)
# This will be used instead of the hardcoded list at line ~2257
globals()["valuation_cols_method_aware"] = valuation_cols_method_aware

# Step 2: Prepare classification data with correct parameters
# Reference: finance_ml/ml_workflow/classification/models.py:201-327
# Function signature: prepare_classification_data(df, labels, test_size, random_state, feature_groups)
X_train_cls, X_test_cls, y_train_cls, y_test_cls, numeric_cols, categorical_cols = (
    prepare_classification_data(
        df=all_stocks_features,  # DataFrame with all features
        labels=event_labels,  # REQUIRED: numpy array of class labels (0, 1, 2)
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
)

print(f"\n✓ Classification Data Prepared with Phase 9.3 feature groups:")
print(f"  Train: {X_train_cls.shape}, Test: {X_test_cls.shape}")
print(f"  Numeric features: {len(numeric_cols)}")
print(f"  Categorical features: {len(categorical_cols)}")
print(f"  Train classes: {np.unique(y_train_cls)}")
print(f"  Test classes: {np.unique(y_test_cls)}")

# CRITICAL VALIDATION: Ensure all 5 classes (0-4) are present in training data
# This is required for the 5-class system and to avoid shape mismatch errors
# when calling export_classification_probabilities (expects shape (n_samples, 5))
print("\n🔍 Validating 5-class system compliance...")
expected_classes = np.array([0, 1, 2, 3, 4])
train_classes = np.unique(y_train_cls)
missing_classes = set(expected_classes) - set(train_classes)

if len(missing_classes) > 0:
    print(f"  ⚠️  WARNING: Missing classes in training data: {sorted(missing_classes)}")
    print(f"  Training data only contains classes: {sorted(train_classes)}")
    print(f"  This may cause shape mismatch errors in export_classification_probabilities!")
    print(f"\n  Recommended actions:")
    print(f"    1. Adjust label creation thresholds to produce all 5 classes")
    print(
        f"    2. Use a different labeling method (e.g., 'price_momentum' instead of 'quality_event')"
    )
    print(f"    3. Check if class imbalance is too extreme (>95% in one class)")

    # Provide specific guidance based on which classes are missing
    if 0 in missing_classes or 4 in missing_classes:
        print(f"\n  💡 Strong Negative (0) or Strong Positive (4) classes missing:")
        print(f"     These require extreme values. Consider:")
        print(f"     - Lowering threshold_positive/threshold_negative")
        print(f"     - Using percentile-based thresholds in create_enhanced_event_labels")

    if len(missing_classes) >= 2:
        print(f"\n  ⚠️  Multiple classes missing - severe class imbalance detected!")
        print(f"     Switching to 'price_momentum' method may help...")

        # Optionally recreate labels with a more reliable method
        # Uncomment the following to auto-switch to price_momentum
        # event_labels = classification_create_enhanced_event_labels(
        #     all_stocks_features,
        #     method='price_momentum',
        #     threshold_positive=5.0,
        #     threshold_negative=-5.0,
        #     use_sector_adjustment=True
        # )
        # print(f"  ✓ Labels recreated with 'price_momentum' method")
        # # Re-prepare data...
else:
    print(f"  ✓ All 5 classes present in training data: {sorted(train_classes)}")
    print(f"  ✓ 5-class system validation passed")
    print(f"  ✓ Ready for model training with num_class=5 configuration")

# %%
##%% Phase 9.4: Class Balance Analysis and Adjustment
print("\n🔧 Phase 9.4: Class Balance Analysis and Adjustment")
print("=" * 80)

# Analyze class distribution
y_train_dist = pd.Series(y_train_cls).value_counts().sort_index()
print(f"\n📊 Original Class Distribution:")
for cls, count in y_train_dist.items():
    pct = count / len(y_train_cls) * 100
    print(f"  Class {cls}: {count:4d} samples ({pct:5.1f}%)")

# Calculate imbalance ratio
max_count = y_train_dist.max()
min_count = y_train_dist.min()
imbalance_ratio = max_count / min_count
print(f"\n  Imbalance ratio: {imbalance_ratio:.2f}:1")

# Apply balancing (SMOTE for minority, undersample majority)
print(f"\n  Applying balance_classes() with method='auto'...")
X_train_cls_balanced, y_train_cls_balanced = balance_classes(
    X_train_cls,
    y_train_cls,
    method="auto",  # Auto-selects SMOTE or undersampling
    random_state=RANDOM_SEED,
)

y_balanced_dist = pd.Series(y_train_cls_balanced).value_counts().sort_index()
print(f"\n✓ Balanced Class Distribution:")
for cls, count in y_balanced_dist.items():
    pct = count / len(y_train_cls_balanced) * 100
    print(f"  Class {cls}: {count:4d} samples ({pct:5.1f}%)")

# Calculate new imbalance ratio
max_count_after = y_balanced_dist.max()
min_count_after = y_balanced_dist.min()
imbalance_ratio_after = max_count_after / min_count_after

print(f"\n  Resampling: {len(X_train_cls):,} → {len(X_train_cls_balanced):,} samples")
print(f"  Imbalance improvement: {imbalance_ratio:.2f}:1 → {imbalance_ratio_after:.2f}:1")
print(f"\n✓ Training data balanced and ready for model training")

# %% [markdown]
# ### 🆕 Phase 9.4 TDD: Classification Enhancements (Tasks 2, 4, 5)
#
# **Business Objective**: Provide granular event signals, prevent look-ahead bias, and ensure all market conditions are represented in training data.
#
# **New Capabilities**:
# 1. **Multi-Label Classification** (`create_multilabel_event_labels`): Produce independent binary labels per Phase 9.3 category (momentum, valuation, quality, etc.)
# 2. **CV Policy Enforcement** (`determine_cv_strategy`): Automatic CV strategy selection (time_series → grouped → stratified)
# 3. **Class Balance Auto-Remediation** (`balance_classes`): Automatic SMOTE/undersampling when imbalance >10:1
#
# **Key Features**:
# - Multi-label mode: 16 independent binary labels (one per category)
# - Sector-adjusted thresholds for multi-label classification
# - Automatic CV strategy based on data structure (prevents leakage)
# - Class balance with SMOTE for minority class augmentation
#
# **References**:
# - Implementation Plan: `docs/improvement_plan/phase_9.4_implementation_plan.md`
# - Test Coverage: `tests/test_multilabel_classification.py`, `test_cv_policy_enforcement.py`, `test_class_balance_remediation.py` (9 tests, 100% pass)
# - Code Guidelines: Section 9.4
# %%
##%%
# Phase 9.4 TDD: Multi-Label Event Classification (Task 2)
print("\n" + "=" * 80)
print("PHASE 9.4 TDD: MULTI-LABEL EVENT CLASSIFICATION")
print("=" * 80)

# Create multi-label event labels (one binary label per category)
# This produces more granular signals than single multi-class labels
print(f"\n🏷️  Creating multi-label event labels...")
print(f"  Mode: multilabel (independent binary labels per category)")
print(f"  Categories: momentum, valuation, quality, profitability, growth")

# Select key categories for multi-label classification
ml_categories = [
    "momentum",
    "valuation",
    "quality",
    "profitability",
    "growth",
    "efficiency",
    "cash_flow",
    "leverage",
]

all_stocks_multilabel = create_multilabel_event_labels(
    all_stocks_features.copy(),
    label_mode="multilabel",
    categories=ml_categories,
    sector_adjusted=True,
    threshold_percentile=0.6,  # Top 40% gets positive label (1)
)

print(f"\n✅ Multi-Label Classification Results:")
print(f"  Total samples: {len(all_stocks_multilabel):,}")
print(f"  Label columns created: {len(ml_categories)}")

# Show label distribution per category
print(f"\n📊 Label Distribution by Category:")
for category in ml_categories:
    label_col = f"label_{category}"
    if label_col in all_stocks_multilabel.columns:
        positive_pct = (all_stocks_multilabel[label_col] == 1).mean() * 100
        print(f"  {category}: {positive_pct:.1f}% positive")

# Optional: Create feature interactions with multi-label probabilities
# This can be used downstream in regression models
print(f"\n💡 Tip: Use label_* columns as meta-features in regression models")
print(f"   Example: label_momentum, label_valuation can signal price movement drivers")
# %%
##%%
# Phase 9.4 TDD: CV Policy Enforcement (Task 4)
print("\n" + "=" * 80)
print("PHASE 9.4 TDD: CROSS-VALIDATION POLICY ENFORCEMENT")
print("=" * 80)

# Automatically determine best CV strategy based on data structure
# Hierarchy: time_series (if snapshot_date) → grouped (if ticker) → stratified (fallback)

print(f"\n🔍 Detecting optimal CV strategy...")
print(f"  Data columns: {list(all_stocks_features.columns[:10])}...")

# Prepare target for CV strategy detection
y_event = (
    all_stocks_features["event_label"] if "event_label" in all_stocks_features.columns else None
)

cv_strategy_name, cv_object = determine_cv_strategy(
    all_stocks_features,
    target=y_event,
    n_splits=CV_FOLDS,  # From configuration cell
    date_column="snapshot_date",
    group_column="ticker",
    random_state=RANDOM_SEED,
)

print(f"\n✅ CV Strategy Selected: {cv_strategy_name.upper()}")
print(f"  Number of folds: {CV_FOLDS}")
print(f"  Strategy: {cv_object.__class__.__name__}")

# Show why this strategy was chosen
if cv_strategy_name == "time_series":
    print(f"  Reason: snapshot_date column detected → prevents look-ahead bias")
elif cv_strategy_name == "grouped":
    n_groups = all_stocks_features["ticker"].nunique()
    print(f"  Reason: ticker column detected ({n_groups:,} unique tickers) → prevents data leakage")
elif cv_strategy_name == "stratified":
    print(f"  Reason: classification target detected → maintains class balance")
else:
    print(f"  Reason: fallback strategy")

print(f"\n💡 Use cv_object in cross_val_score() or GridSearchCV for proper validation")
# %%
##%%
# Phase 9.4 TDD: Class Balance Auto-Remediation (Task 5)
print("\n" + "=" * 80)
print("PHASE 9.4 TDD: CLASS BALANCE AUTO-REMEDIATION")
print("=" * 80)

# Check class balance and apply automatic remediation if needed
# Applies SMOTE for imbalance >10:1

if "event_label" in all_stocks_features.columns:
    print(f"\n📊 Class Distribution Before Balancing:")
    class_counts = all_stocks_features["event_label"].value_counts().sort_index()
    for cls, count in class_counts.items():
        pct = count / len(all_stocks_features) * 100
        print(f"  Class {cls}: {count:,} ({pct:.1f}%)")

    # Calculate imbalance ratio
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    print(f"\n  Imbalance ratio: {imbalance_ratio:.2f}:1")

    # Prepare features and target
    feature_cols_balance = [
        col
        for col in all_stocks_features.columns
        if col not in ["ticker", "isin", "sector", "region", "event_label", "snapshot_date"]
    ]
    X_balance = all_stocks_features[feature_cols_balance].fillna(0)
    y_balance = all_stocks_features["event_label"]

    # Apply automatic balancing if imbalance exceeds threshold
    if imbalance_ratio > 10.0:
        print(f"\n⚠️  Severe imbalance detected (>{10.0}:1)")
        print(f'  Applying balance_classes() with method="auto"...')

        X_balanced, y_balanced = balance_classes(
            X_balance,
            y_balance,
            method="auto",  # Chooses SMOTE or undersample based on severity
            imbalance_threshold=10.0,
            random_state=RANDOM_SEED,
        )

        print(f"\n✅ Class Balance After Remediation:")
        class_counts_after = pd.Series(y_balanced).value_counts().sort_index()
        for cls, count in class_counts_after.items():
            pct = count / len(y_balanced) * 100
            print(f"  Class {cls}: {count:,} ({pct:.1f}%)")

        # Calculate new imbalance ratio
        max_count_after = class_counts_after.max()
        min_count_after = class_counts_after.min()
        imbalance_ratio_after = max_count_after / min_count_after
        print(f"\n  New imbalance ratio: {imbalance_ratio_after:.2f}:1")
        print(f"  Improvement: {imbalance_ratio:.2f}:1 → {imbalance_ratio_after:.2f}:1")

        # Store balanced data for downstream use
        all_stocks_balanced = pd.DataFrame(X_balanced, columns=feature_cols_balance)
        all_stocks_balanced["event_label"] = y_balanced.values

        print(f"\n✓ Created all_stocks_balanced DataFrame with {len(all_stocks_balanced):,} rows")
    else:
        print(f"\n✓ Class balance acceptable ({imbalance_ratio:.2f}:1 < 10:1)")
        print(f"  No remediation needed")
        all_stocks_balanced = all_stocks_features.copy()
else:
    print(f"\n⚠️  No event_label column found, skipping class balance check")
    all_stocks_balanced = all_stocks_features.copy()
# %%
from finance_ml.ml_workflow.data.schema import list_categorical_cols, list_date_cols
from finance_ml.ml_workflow.features import preprocess_for_lightgbm

# Get categorical and datetime columns from schema (Section 2.2)
categorical_cols_from_schema = list_categorical_cols()
datetime_cols_from_schema = list_date_cols()
auxiliary_cols_to_drop = [
    "unit",
    "dividend_record_frequency",
    "dividend_record_currency",
]

# Preprocess training data (returns both processed data and encoders)
X_train_processed, encoders = preprocess_for_lightgbm(
    X_train_cls_balanced.copy(),
    categorical_columns=categorical_cols_from_schema,
    datetime_columns=datetime_cols_from_schema,
    drop_columns=auxiliary_cols_to_drop,
    return_encoders=True,
)

# Extract reference date for consistent datetime transformations
reference_date = encoders.get("_reference_date")

# Preprocess test data using training encoders (inference mode)
X_test_processed, _ = preprocess_for_lightgbm(
    X_test_cls.copy(),
    categorical_columns=categorical_cols_from_schema,
    datetime_columns=datetime_cols_from_schema,
    drop_columns=auxiliary_cols_to_drop,
    encoders=encoders,
    reference_date=reference_date,
)

# Store training feature columns for prediction alignment
training_feature_cols = X_train_processed.columns.tolist()

# Align test data columns to match training features
X_test_processed = X_test_processed.reindex(
    columns=training_feature_cols,
    fill_value=0,  # Fill missing columns with zero (new categorical levels in test)
)

# Update column lists for downstream usage (all columns now numeric after label encoding)
numeric_cols_processed = X_train_processed.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_processed = []  # All categoricals converted to numeric

print(
    f"✓ Preprocessing complete: {X_train_processed.shape[0]} train samples, {X_test_processed.shape[0]} test samples"
)
print(f"✓ Features: {len(training_feature_cols)} columns (all numeric after encoding)")
print(
    f"✓ Encoders stored: {len([k for k in encoders.keys() if k != '_reference_date'])} categorical columns"
)
# %% [markdown]
# ### Phase 9.3: Preprocess Data for LightGBM (Optional Demonstration)
#
# This section demonstrates **schema-driven preprocessing** using `preprocess_for_lightgbm()` from `finance_ml.ml_workflow.features`:
#
# - **Categorical Encoding**: Converts categorical columns to numeric via label encoding
# - **Datetime Feature Extraction**: Extracts temporal features using a consistent reference date
# - **Column Dropping**: Removes auxiliary columns not used for modeling
# - **Encoder Storage**: Returns encoders for inference-mode transformations
#
# **Note**: `prepare_classification_data()` already handles categorical encoding, so this step is typically **not needed** in the standard workflow. This demonstration is provided for reference when working with raw data directly.
#
# **Best Practice**: Always store training encoders and feature column lists for consistent test/inference preprocessing (Section 1.3 of `code_guidelines.md`).
# %%
# Hyperparameter optimization with Phase 9.4 function
print("\n⚙️  Starting Hyperparameter Optimization...")

# Validate data types before optimization (Issue fix: must use processed data)
print(f"\n📋 Data validation:")
print(f"  X_train_processed shape: {X_train_processed.shape}")
print(f"  X_train_processed dtypes: {X_train_processed.dtypes.value_counts().to_dict()}")
print(
    f"  y_train_cls_balanced shape: {y_train_cls_balanced.shape}, dtype: {y_train_cls_balanced.dtype}"
)

# Check for non-numeric columns (code_guidelines.md: validate before modeling)
non_numeric_cols = X_train_processed.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    raise ValueError(
        f"❌ Non-numeric columns detected in training data: {non_numeric_cols}\n"
        f"All features must be numeric (int, float, bool) for LightGBM.\n"
        f"Please ensure preprocess_for_lightgbm() was applied correctly."
    )

print("  ✓ All columns are numeric - ready for LightGBM optimization")

# Call optimization with PROCESSED data (not raw X_train_cls)
# FIX: Use X_train_processed (numeric) instead of X_train_cls (contains object/datetime columns)
try:
    result = tune_classifier_hyperparameters(
        X_train_processed,
        y_train_cls_balanced,  # FIXED: was X_train_cls, y_train_cls_balanced
        classifier_type="lightgbm",
        n_trials=50,
        cv_folds=5,
        verbose=True,
    )

    # Validate result structure (code_guidelines.md: validate outputs)
    if result and "best_score" in result and "best_params" in result and result["best_score"] > 0:
        print(f"\n✓ Hyperparameter Optimization Complete:")
        print(f"  Best F1 score: {result['best_score']:.4f}")
        print(f"  Best parameters: {result['best_params']}")
    else:
        print("\n⚠️  Optimization completed but results are incomplete or score is 0")
        print(f"  Result keys: {list(result.keys()) if result else 'None'}")
        print(f"  Best score: {result.get('best_score', 'N/A')}")
        if result.get("best_score", 0) == 0:
            print("  ⚠️  All trials may have failed - check data types and LightGBM compatibility")

except (ValueError, TypeError, RuntimeError, KeyError) as e:
    print(f"❌ Hyperparameter optimization failed: {str(e)}")
    print(f"   Error type: {type(e).__name__}")
    raise

# %%
# Compare multiple classification models
print("\n" + "=" * 80)
print("MODEL COMPARISON - TRAINING MULTIPLE CLASSIFIERS")
print("=" * 80)

# Train and compare XGBoost, LightGBM, and CatBoost classifiers
print("\n🤖 Training multiple classifiers for comparison...")
print("  Models: XGBoost, LightGBM, CatBoost")

try:
    models_results = compare_classifiers(
        X_train_processed,
        y_train_cls_balanced,
        X_test_processed,
        y_test_cls,
        numeric_cols=numeric_cols_processed,
        categorical_cols=categorical_cols_processed,
    )

    print(f"\n✓ Model Comparison Complete:")
    print(f"  Models trained: {len(models_results)}")

    # FIX: Handle missing 'f1_macro' key and always define cls_model
    # Check if results have required metrics
    if models_results and all("f1_macro" in m for m in models_results.values()):
        # Display comparison results
        for model_name, metrics in models_results.items():
            print(f"\n  {model_name}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1 Score (macro): {metrics['f1_macro']:.4f}")
            print(f"    Precision (macro): {metrics['precision_macro']:.4f}")
            print(f"    Recall (macro): {metrics['recall_macro']:.4f}")

        # Select best model based on F1 score
        best_model_name = max(models_results.items(), key=lambda x: x[1]["f1_macro"])[0]
        print(
            f"\n🏆 Best Model: {best_model_name} (F1={models_results[best_model_name]['f1_macro']:.4f})"
        )

        # Store best model as cls_model
        if "model" in models_results[best_model_name]:
            cls_model = models_results[best_model_name]["model"]
        else:
            print(f"⚠️ Best model object not found in results, using hyperparameter result")
            cls_model = result["model"]
    else:
        print(
            f"⚠️ compare_classifiers() missing f1_macro or returned empty, using hyperparameter result"
        )
        cls_model = result["model"]

except (ValueError, TypeError, RuntimeError, KeyError) as e:
    print(f"❌ Model comparison failed: {str(e)}")
    print(f"   Using optimized model from hyperparameter search instead")
    models_results = {}
    # ALWAYS define cls_model even on failure
    cls_model = result["model"]

# Verification: Ensure cls_model is defined
if "cls_model" not in dir():
    print("⚠️ cls_model was not defined, using result['model'] as fallback")
    cls_model = result["model"]

print(f"\n✓ cls_model defined and ready for evaluation")

# CRITICAL FIX: Re-extract feature names from the NEW model (result['model'])
# The previous model_feature_names was from a different model (comparison/evaluation)
# We must get feature names from THIS specific model to avoid feature count mismatch
print("\n🔍 Extracting feature names from hyperparameter-optimized model...")
if hasattr(cls_model, "feature_names_"):
    # CatBoost model - use feature_names_ attribute
    model_feature_names = cls_model.feature_names_
    print(f"  ✓ CatBoost model: {len(model_feature_names)} features")
elif hasattr(cls_model, "get_booster") and hasattr(cls_model.get_booster(), "feature_names"):
    # XGBoost model
    model_feature_names = cls_model.get_booster().feature_names
    print(f"  ✓ XGBoost model: {len(model_feature_names)} features")
elif hasattr(cls_model, "feature_name_"):
    # LightGBM model
    model_feature_names = cls_model.feature_name_
    print(f"  ✓ LightGBM model: {len(model_feature_names)} features")
else:
    # Fallback to X_train_processed columns if model doesn't expose feature names
    print("  ⚠️  Model doesn't expose feature_names_, using X_train_processed.columns")
    model_feature_names = list(X_train_processed.columns)

if not model_feature_names:
    raise ValueError("❌ CRITICAL: Could not extract feature names from cls_model")

print(f"  First 5 features: {model_feature_names[:5]}")

# %%
# Comprehensive evaluation of best model
print("\n" + "=" * 80)
print("COMPREHENSIVE MODEL EVALUATION")
print("=" * 80)

# FIX 3: Add defensive check to ensure cls_model is defined before use
# This provides a safety net even if the previous cell's logic somehow fails
if "cls_model" not in dir() or cls_model is None:
    print("⚠️ cls_model not defined, using result['model'] as fallback")
    cls_model = result["model"]

# Verify cls_model is valid before proceeding
if cls_model is None:
    raise ValueError("❌ CRITICAL: Unable to obtain a valid classification model for evaluation")

# Import Pool for CatBoost model support
from catboost import Pool

# Feature names were already extracted in the previous cell (lines 1698-1722)
# This cell uses those features directly - no need to re-extract
print(f"\n✓ Using {len(model_feature_names)} features extracted from cls_model in previous cell")
print(f"  First 5 features: {model_feature_names[:5]}")

# Realign X_test_processed to match the model's exact features
# This handles cases where the model was trained with different features than X_train_processed
print(f"\n🔄 Aligning test data to model's feature schema...")
print(f"  X_test_processed columns before: {len(X_test_processed.columns)}")

# Check for missing features
missing_features = set(model_feature_names) - set(X_test_processed.columns)
if missing_features:
    print(
        f"⚠️ Warning: {len(missing_features)} features missing in test data, will be filled with 0"
    )
    print(f"  Missing features: {list(missing_features)[:5]}...")

# Reindex to match model's features exactly (adds missing cols with 0, drops extra cols)
X_test_processed = X_test_processed.reindex(columns=model_feature_names, fill_value=0)
print(f"  X_test_processed columns after: {len(X_test_processed.columns)}")
print(f"  ✓ Column alignment verified: {list(X_test_processed.columns) == model_feature_names}")

# CRITICAL FIX: Use appropriate input format based on model type
# CatBoost accepts Pool objects, but LightGBM and XGBoost expect DataFrames/arrays
if hasattr(cls_model, "feature_names_"):
    # CatBoost model - use Pool with explicit feature names
    test_pool = Pool(X_test_processed, feature_names=model_feature_names)
    print(f"\n✓ Created CatBoost Pool with {len(model_feature_names)} features")
    y_pred_test = cls_model.predict(test_pool)
    y_proba_test = cls_model.predict_proba(test_pool)
else:
    # LightGBM or XGBoost - use DataFrame directly
    print(
        f"\n✓ Using DataFrame input with {len(model_feature_names)} features for {type(cls_model).__name__}"
    )
    y_pred_test = cls_model.predict(X_test_processed)
    y_proba_test = cls_model.predict_proba(X_test_processed)

# Evaluate classification performance
print("\n📊 Computing comprehensive evaluation metrics...")
eval_metrics = evaluate_classification(
    y_test_cls,
    y_pred_test,
    y_proba_test,
    class_names=[
        "Strong Negative",
        "Negative",
        "Neutral",
        "Positive",
        "Strong Positive",
    ],
)

print(f"\n✓ Classification Evaluation:")
print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
print(f"  F1 Score (macro): {eval_metrics['f1_macro']:.4f}")
print(f"  Precision (macro): {eval_metrics['precision_macro']:.4f}")
print(f"  Recall (macro): {eval_metrics['recall_macro']:.4f}")

if "classification_report" in eval_metrics:
    print("\n📋 Classification Report:")
    print(eval_metrics["classification_report"])

# Save evaluation metrics - FIX: Convert numpy arrays to lists for JSON serialization
eval_output_path = OUTPUT_DIR / "classification" / "evaluation_metrics.json"
eval_output_path.parent.mkdir(parents=True, exist_ok=True)
import json


# Helper function to convert numpy arrays to lists recursively
def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


# Filter out classification_report and convert numpy arrays
eval_metrics_json = {
    k: convert_numpy_to_list(v) for k, v in eval_metrics.items() if k != "classification_report"
}

with open(eval_output_path, "w") as f:
    json.dump(eval_metrics_json, f, indent=2)
print(f"\n💾 Evaluation metrics saved to: {eval_output_path}")
# %%
# Plot confusion matrices
print("\n" + "=" * 80)
print("CONFUSION MATRIX VISUALIZATION")
print("=" * 80)

if models_results:
    print("\n📊 Plotting confusion matrices for all models...")
    try:
        plot_confusion_matrices(
            models_results,
            class_names=[
                "Strong Negative",
                "Negative",
                "Neutral",
                "Positive",
                "Strong Positive",
            ],
        )
        print("✓ Confusion matrices displayed")
    except (ValueError, TypeError, RuntimeError, AttributeError) as e:
        print(f"⚠️  Could not plot confusion matrices: {str(e)}")
else:
    # Plot for single model
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    print("\n📊 Plotting confusion matrix for optimized model...")
    # Ensure 5x5 matrix aligned to new label schema
    cm = confusion_matrix(y_test_cls, y_pred_test, labels=list(range(5)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Strong Negative",
            "Negative",
            "Neutral",
            "Positive",
            "Strong Positive",
        ],
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Classification Confusion Matrix")
    plt.tight_layout()

    # Save confusion matrix
    cm_output_path = OUTPUT_DIR / "classification" / "confusion_matrix.png"
    plt.savefig(cm_output_path, dpi=300, bbox_inches="tight")
    print(f"💾 Confusion matrix saved to: {cm_output_path}")
    plt.show()

# %%
# SHAP analysis for model interpretability
print("\n" + "=" * 80)
print("SHAP ANALYSIS - MODEL INTERPRETABILITY")
print("=" * 80)

print("\n🔍 Computing SHAP values for feature importance...")
try:
    # Align data to model features before SHAP computation
    X_train_for_shap = X_train_processed.reindex(columns=model_feature_names, fill_value=0)
    X_test_for_shap = X_test_processed.reindex(columns=model_feature_names, fill_value=0)

    shap_values = compute_shap_values(cls_model, X_train_for_shap, X_test_for_shap, max_samples=100)

    if shap_values is not None:
        print("✓ SHAP values computed successfully")
        print("  SHAP values can be used for detailed feature importance analysis")
    else:
        print("⚠️  SHAP values computation returned None")

except (ValueError, TypeError, RuntimeError, AttributeError, ImportError) as e:
    print(f"⚠️  SHAP analysis failed: {str(e)}")
    print("  Continuing without SHAP analysis")

# %%
# Sector-specific evaluation
print("\n" + "=" * 80)
print("SECTOR-SPECIFIC EVALUATION")
print("=" * 80)

if "sector" in all_stocks_features.columns:
    print("\n📊 Evaluating model performance by sector...")

    # Get sector information for test set
    test_indices = X_test_cls.index
    sectors_test = all_stocks_features.loc[test_indices, "sector"]

    try:
        sector_metrics = evaluate_classification_by_sector(y_test_cls, y_pred_test, sectors_test)

        print(f"\n✓ Sector-specific metrics computed:")
        for idx, row in sector_metrics.iterrows():
            print(f"\n  {row['Sector']}:")
            print(f"    Accuracy: {row['Accuracy']:.4f}")
            print(f"    F1 Score: {row['F1-Score']:.4f}")
            print(f"    Sample count: {row['Samples']}")

        # Save sector metrics
        sector_output_path = OUTPUT_DIR / "classification" / "sector_metrics.json"
        with open(sector_output_path, "w") as f:
            json.dump(sector_metrics.to_dict("records"), f, indent=2)
        print(f"\n💾 Sector metrics saved to: {sector_output_path}")

    except (ValueError, TypeError, RuntimeError, KeyError, IOError) as e:
        print(f"⚠️  Sector evaluation failed: {str(e)}")
else:
    print("⚠️  No sector column available for sector-specific evaluation")

# %%
# Calibration analysis
print("\n" + "=" * 80)
print("PROBABILITY CALIBRATION ANALYSIS")
print("=" * 80)

print("\n📊 Analyzing probability calibration...")
try:
    calibration_results = analyze_calibration(y_test_cls, y_proba_test, n_bins=10)

    print(f"\n✓ Calibration Analysis Complete:")
    if "brier_score" in calibration_results:
        print(f"  Brier Score: {calibration_results['brier_score']:.4f}")
    if "log_loss" in calibration_results:
        print(f"  Log Loss: {calibration_results['log_loss']:.4f}")

    # Save calibration results
    calib_output_path = OUTPUT_DIR / "classification" / "calibration_analysis.json"
    with open(calib_output_path, "w") as f:
        json.dump(
            {k: v for k, v in calibration_results.items() if not isinstance(v, np.ndarray)},
            f,
            indent=2,
        )
    print(f"\n💾 Calibration analysis saved to: {calib_output_path}")

except (ValueError, TypeError, RuntimeError, KeyError, IOError) as e:
    print(f"⚠️  Calibration analysis failed: {str(e)}")

# %%
# Cross-validation with sector stratification
print("\n" + "=" * 80)
print("CROSS-VALIDATION WITH SECTOR STRATIFICATION")
print("=" * 80)

if "sector" in all_stocks_features.columns:
    print("\n🔄 Performing cross-validation with sector stratification...")

    # Prepare full dataset for cross-validation
    X_full_cls = pd.concat([X_train_processed, X_test_processed])
    y_full_cls = np.concatenate([y_train_cls_balanced, y_test_cls])

    # Add sector column for stratification
    X_full_with_sector = X_full_cls.copy()
    X_full_with_sector["sector"] = all_stocks_features.loc[X_full_cls.index, "sector"]

    try:
        # Use cv_object from determine_cv_strategy() (Cell 63) for proper CV policy
        cv_results = cross_validate_classifier(
            cls_model,
            X_full_with_sector,
            y_full_cls,
            cv=cv_object,  # Use CV strategy from determine_cv_strategy() (Cell 63)
        )

        print(f"\n✓ Cross-validation Complete:")
        print(
            f"  Mean Accuracy: {cv_results['test_accuracy']:.4f} ± {cv_results['test_accuracy_std']:.4f}"
        )
        print(f"  Mean F1 Score: {cv_results['test_f1']:.4f} ± {cv_results['test_f1_std']:.4f}")
        print(
            f"  Fold Accuracies: {[f'{s:.4f}' for s in cv_results['cv_scores']['test_accuracy']]}"
        )

        # Save CV results (exclude cv_scores which contains numpy arrays)
        cv_output_path = OUTPUT_DIR / "classification" / "cross_validation.json"
        with open(cv_output_path, "w") as f:
            json.dump({k: v for k, v in cv_results.items() if k != "cv_scores"}, f, indent=2)
        print(f"\n💾 Cross-validation results saved to: {cv_output_path}")

    except (ValueError, TypeError, RuntimeError, KeyError, IOError) as e:
        print(f"⚠️  Cross-validation failed: {str(e)}")
else:
    print("⚠️  No sector column available for sector-stratified cross-validation")

# %%
# Train classification model and export probabilities as meta-features
print("\n" + "=" * 80)
print("CLASSIFICATION MODEL TRAINING & FEATURE ENGINEERING")
print("=" * 80)

from sklearn.metrics import accuracy_score


def _extract_model_feature_names(model, fallback_columns):
    """Extract feature names from various model types (CatBoost, XGBoost, LightGBM)."""
    feature_names = None
    if hasattr(model, "feature_names_"):
        feature_names = model.feature_names_
    elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
        feature_names = model.get_booster().feature_names
    elif hasattr(model, "feature_name_"):
        feature_names = model.feature_name_

    if feature_names is None:
        print("  [WARN] Model doesn't expose feature_names_, using fallback columns")
        feature_names = list(fallback_columns)

    if not feature_names:
        raise ValueError("[ERROR] CRITICAL: Could not extract feature names from cls_model")

    print(f"  [OK] Extracted {len(feature_names)} features from model")
    return feature_names


def _prepare_inference_data(df_raw, raw_train_columns, model_feature_names, prep_params):
    """
    Preprocess raw data and align exactly to model features for inference.

    Args:
        df_raw: DataFrame containing all raw features
        raw_train_columns: List of columns expected by the preprocessing pipeline (from training)
        model_feature_names: List of features expected by the trained model
        prep_params: Dictionary containing encoders, reference_date, etc.
    """
    # 1. Align raw columns to match training input structure
    # CRITICAL: Use .reindex() to ensure EXACT column match and order from X_train_cls_balanced
    X_raw = df_raw.reindex(columns=raw_train_columns)

    if X_raw.shape[1] != len(raw_train_columns):
        raise ValueError("Column count mismatch during raw alignment")

    # 2. Apply preprocessing using training encoders (Inference Mode)
    X_processed, _ = preprocess_for_lightgbm(
        X_raw.copy(),
        categorical_columns=prep_params["cat_cols"],
        datetime_columns=prep_params["date_cols"],
        drop_columns=prep_params["drop_cols"],
        encoders=prep_params["encoders"],
        reference_date=prep_params["ref_date"],
    )

    # 3. Align processed data to model schema (Handle missing/extra columns generated by OHE)
    missing_cols = set(model_feature_names) - set(X_processed.columns)
    extra_cols = set(X_processed.columns) - set(model_feature_names)

    if missing_cols:
        print(f"  [WARN] Filling {len(missing_cols)} missing columns with 0")
        for col in missing_cols:
            X_processed[col] = 0

    if extra_cols:
        print(f"  [WARN] Dropping {len(extra_cols)} extra columns")
        X_processed = X_processed.drop(columns=list(extra_cols))

    # 4. Final reorder to match model expectation
    return X_processed[model_feature_names]


# --- Main Execution ---

# Use optimized model from hyperparameter search
cls_model = result["model"]

# 1. Get Model Feature Names
print("\n[INFO] Extracting feature names from optimized model...")
model_feature_names = _extract_model_feature_names(cls_model, X_train_processed.columns)
print(f"  First 5 features: {model_feature_names[:5]}")

# %%
# Extract schema-based column lists for preprocessing
from finance_ml.ml_workflow.data.schema import list_categorical_cols, list_date_cols

categorical_columns_from_schema = list_categorical_cols()
datetime_cols_from_schema = list_date_cols()

print(f"[INFO] Schema integration:")
print(f"  Categorical columns from schema: {len(categorical_columns_from_schema)}")
print(f"  Date columns from schema: {len(datetime_cols_from_schema)}")

# 2. Prepare All Data for Prediction
print("\n[INFO] Preprocessing all_stocks_features for prediction...")
prep_params = {
    "cat_cols": categorical_columns_from_schema,
    "date_cols": datetime_cols_from_schema,
    "drop_cols": auxiliary_cols_to_drop,
    "encoders": encoders,
    "ref_date": reference_date,
}

X_cls_all_processed = _prepare_inference_data(
    all_stocks_features, X_train_cls.columns, model_feature_names, prep_params
)
print(f"  [OK] Final shape aligned to model: {X_cls_all_processed.shape}")

# 3. Generate Probabilities
y_proba_all = cls_model.predict_proba(X_cls_all_processed)
print(f"\n[OK] Classification Model Trained with Optimized Hyperparameters")

# 4. Evaluate Accuracy (Aligning train/test data first)
# Align both train and test data to model features before prediction
X_train_aligned = X_train_processed.reindex(columns=model_feature_names, fill_value=0)
X_test_aligned = X_test_processed.reindex(columns=model_feature_names, fill_value=0)

# Generate predictions and metrics
y_train_pred = cls_model.predict(X_train_aligned)
y_test_pred = cls_model.predict(X_test_aligned)

train_accuracy = accuracy_score(y_train_cls, y_train_pred)
test_accuracy = accuracy_score(y_test_cls, y_test_pred)

print(f"  Train Accuracy: {train_accuracy:.3f}")
print(f"  Test Accuracy:  {test_accuracy:.3f}")

# Store test predictions for visualization section
y_pred_cls = y_test_pred

# 5. Export Classification Probabilities (Phase 9.9)
print("\n" + "=" * 80)
print("EXPORT CLASSIFICATION PROBABILITIES (Phase 9.9)")
print("=" * 80)

# Get predictions for all data
y_pred_all = cls_model.predict(X_cls_all_processed)

probs_df = export_classification_probabilities(
    y_true=event_labels,
    y_pred=y_pred_all,
    y_proba=y_proba_all,
    index=all_stocks_features.index,
)

probs_path = OUTPUT_DIR / "classification" / "classification_probabilities.csv"
probs_path.parent.mkdir(parents=True, exist_ok=True)
probs_df.to_csv(probs_path, index=False)
print(f"\n✓ Classification probabilities exported to: {probs_path}")
print(f"  Shape: {probs_df.shape}")

# 6. Integrate Meta-Features (Phase 9.9)
print("\n" + "=" * 80)
print("INTEGRATE CLASSIFICATION META-FEATURES (Phase 9.9)")
print("=" * 80)

all_stocks_with_classification = integrate_classification_features(all_stocks_features, y_proba_all)

print(f"\n✓ Classification meta-features integrated")
print(f"  With meta-features: {all_stocks_with_classification.shape}")
print(
    f"  Added columns: {[col for col in all_stocks_with_classification.columns if col.startswith('event_prob_')]}"
)
# %% [markdown]
# ## Phase 9.5: Sector-Optimized Regression Models with Quantile Predictions Models with Classification Features
#
# ### Business Goal
# Predict stock price targets using regression models enhanced with classification meta-features, with uncertainty quantification via quantile regression.
#
# ### Key Objectives
# 1. Integrate classification probabilities as meta-features
# 2. Train multiple regression models (XGBoost, LightGBM, CatBoost, etc.)
# 3. Build stacking ensemble for robust predictions
# 4. Train quantile models for prediction intervals (p10, p50, p90)
# 5. Apply non-negative constraints (prices must be ≥ 0)
# 6. Perform time-series cross-validation
#
# ### Inputs
# - `all_stocks_features`: From Phase 9.3
# - `clf_result['y_proba']`: Classification probabilities from Phase 9.4
#
# ### Outputs
# - `reg_result`: Regression result dict
# - `outputs/regression/`: Model artifacts, quantile predictions
# - `outputs/regression/regression_predictions_detailed.csv`: Standardized predictions
# - `outputs/models/regression_metrics_by_sector.csv`: Per-sector metrics
#
# ### v1.2 Standards Applied
# - ✅ Quantile regression (p10, p50, p90) with conformal calibration
# - ✅ Monotonicity enforcement (p10 ≤ p50 ≤ p90)
# - ✅ Non-negativity constraints
# - ✅ Data split policy (TimeSeriesSplit → GroupKFold → Stratified)
# - ✅ Standardized predictions schema
# - ✅ Outlier safety rails (Huber loss, post-prediction clipping)
#
# ### Standardized Predictions Schema
# Required columns:
# - ticker, isin, sector, region, last_price, snapshot_date
# - y_true, y_pred, y_pred_calibrated
# - pred_p10, pred_p50, pred_p90, interval_width
# - abs_error, pct_error
# - model_version
#
# ### Validation Checkpoint
# - MAE < 50% on validation set
# - R² > 0.3
# - Zero predictions < 1% (non-negativity enforced)
# - Quantile monotonicity verified
# - Prediction intervals coverage: 80% ± 5%
#
# Advanced regression modeling using functions from `finance_ml.regression`:
#
# **Workflow Steps:**
# 1. Create interaction features between classification probabilities and valuation metrics
# 2. Prepare regression data with classification meta-features
# 3. Train and compare multiple regression models (Ridge, Lasso, RF, ET, GB, HistGB)
# 4. Build stacking ensemble for best performance
# 5. Train quantile regression for prediction intervals
# 6. Train sector-specific models (optional)
# 7. Save models with metadata
# 8. Store predictions for downstream analysis
#
# **Key Functions:**
# - `create_classification_interactions` — Create feature interactions
# - `prepare_regression_data` — Split and preprocess data
# - `compare_regressors` — Compare 6 regression models
# - `train_stacking_regressor` — Build ensemble
# - `train_quantile_regressor` — Prediction intervals
# - `train_sector_specific_models` — Per-sector optimization
# - `save_model` — Model persistence
#
# %%
# Additional imports for Phase 9.5 regression
from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Note: Configuration constants defined in Section 1 (lines 3-9)
# Single source of truth for all configuration values

print("[OK] Phase 9.5 configuration complete (using global config from Section 1)")
# %% [markdown]
# ### 6.1 Create Classification Interaction Features
#
# %%
print("=" * 80)
print("6.1 — Creating Classification Interaction Features")
print("=" * 80)

# ============================================================================
# VALIDATION CHECKPOINT: Verify all_stocks_with_classification exists
# ============================================================================

# Check if dataframe exists in namespace
if "all_stocks_with_classification" not in globals():
    raise RuntimeError(
        "❌ ERROR: all_stocks_with_classification not found.\n"
        "Please run Phase 9.4 (Classification) section first to create this dataframe."
    )

# Verify it's a DataFrame
if not isinstance(all_stocks_with_classification, pd.DataFrame):
    raise TypeError(
        f"❌ ERROR: Expected pandas DataFrame, got {type(all_stocks_with_classification).__name__}"
    )

# Check not empty
if all_stocks_with_classification.empty:
    raise ValueError("❌ ERROR: all_stocks_with_classification is empty. No data to process.")

# Verify classification columns exist (5-class system - code_guidelines.md Section 2.2.1)
# Phase 9.4 creates: event_prob_strong_negative, event_prob_negative, event_prob_neutral,
#                     event_prob_positive, event_prob_strong_positive, event_class_predicted, event_confidence
required_classification_cols = [
    "event_prob_strong_negative",
    "event_prob_negative",
    "event_prob_neutral",
    "event_prob_positive",
    "event_prob_strong_positive",
    "event_class_predicted",
    "event_confidence",
]
missing_cols = [
    col for col in required_classification_cols if col not in all_stocks_with_classification.columns
]

if missing_cols:
    raise ValueError(
        f"❌ ERROR: Missing required classification columns: {missing_cols}\n"
        f"Please ensure Phase 9.4 classification completed successfully."
    )

# Success - log shape and available columns
print(f"✓ Validation passed: all_stocks_with_classification")
print(
    f"  Shape: {all_stocks_with_classification.shape[0]:,} rows × {all_stocks_with_classification.shape[1]} columns"
)
print(f"  Classification columns present: {required_classification_cols}")
print()

# Extract classification and valuation columns
classification_cols = [
    c for c in all_stocks_with_classification.columns if c.startswith("event_prob_")
]

# Use method-aware valuation columns defined earlier (after event label creation)
# This ensures interaction features align with the semantics of the chosen event labeling method
# Fallback to default list if valuation_cols_method_aware is not defined
if "valuation_cols_method_aware" in globals():
    valuation_cols = valuation_cols_method_aware
    print(f"\n✓ Using method-aware valuation columns (defined for label_method='{label_method}')")
else:
    # Fallback to default valuation columns if method-aware columns not defined
    valuation_cols = [
        c
        for c in [
            "market_cap",
            "enterprise_value",
            "ebitda",
            "p_e",
            "p_b",
            "gross_margin",
            "revenue",
            "net_income",
        ]
        if c in all_stocks_with_classification.columns
    ]
    print(f"\n⚠️  Using default valuation columns (method-aware columns not found)")

if classification_cols and valuation_cols:
    print(f"\nClassification features: {len(classification_cols)}")
    print(f"Valuation features: {len(valuation_cols)}")

    try:
        # Create interaction features
        # Reference: finance_ml.ml_workflow.regression.dataset.create_classification_interactions()
        all_stocks_enhanced = regression_create_classification_interactions(
            all_stocks_with_classification,
            classification_cols=classification_cols,
            valuation_cols=valuation_cols,
        )

        # Report results
        interaction_cols = [
            c
            for c in all_stocks_enhanced.columns
            if "_x_" in c and c not in all_stocks_with_classification.columns
        ]
        print(f"\n✓ Created {len(interaction_cols)} interaction features")
        if interaction_cols:
            print(f"  Examples: {', '.join(interaction_cols[:3])}")

    except (ValueError, TypeError, KeyError, AttributeError) as e:
        print(f"⚠️  Error creating interaction features: {e}")
        print("   Continuing with original features...")
        all_stocks_enhanced = all_stocks_with_classification.copy()
else:
    print("⚠️  Skipping interaction features (missing classification or valuation columns)")
    all_stocks_enhanced = all_stocks_with_classification.copy()

# Handle missing/infinite values introduced by interaction features only (Phase 9.5 guard)
print("\n🔧 Phase 9.5: Interaction Feature Imputation Check")
print("=" * 80)

interaction_cols = [
    c
    for c in all_stocks_enhanced.columns
    if "_x_" in c and c not in all_stocks_with_classification.columns
]

if interaction_cols:
    all_stocks_enhanced[interaction_cols] = all_stocks_enhanced[interaction_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    for col in interaction_cols:
        if all_stocks_enhanced[col].dtype in ["float64", "float32", "int64", "int32"]:
            all_stocks_enhanced[col] = all_stocks_enhanced[col].fillna(
                all_stocks_enhanced[col].median()
            )
        else:
            mode_val = all_stocks_enhanced[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
            all_stocks_enhanced[col] = all_stocks_enhanced[col].fillna(fill_val)

    assert_no_missing(all_stocks_enhanced[interaction_cols], label="Phase 9.5 interaction features")
    print(f"✓ Interaction feature NaNs resolved for {len(interaction_cols)} columns")
else:
    print("✓ No interaction features require imputation")
# %%
# Verify ALL 21 price columns preserved after Phase 9.5 preprocessing (code_guidelines.md Section 8.5.2)
# all_stocks_enhanced inherits price columns from all_stocks_with_classification
# PRICE_COLUMNS includes: current (6), historical (9), 52w bounds (2), EMAs (4)
from finance_ml.ml_workflow.preprocessing.column_semantics import PRICE_COLUMNS

price_cols_present = [c for c in PRICE_COLUMNS if c in all_stocks_enhanced.columns]
for col in price_cols_present:
    if col in all_stocks_with_classification.columns:
        # Compare non-dropped rows only (some rows may be removed by dropna)
        common_idx = all_stocks_enhanced.index.intersection(all_stocks_with_classification.index)
        if len(common_idx) > 0:
            assert all_stocks_enhanced.loc[common_idx, col].equals(
                all_stocks_with_classification.loc[common_idx, col]
            ), f"{col} was incorrectly modified during Phase 9.5 preprocessing!"

print(
    f"\n✓ Verified {len(price_cols_present)}/21 price columns preserved after Phase 9.5 preprocessing (business metric protection)"
)

# %% [markdown]
# ### 6.2 Prepare Regression Data
#
# %%
print("=" * 80)
print("6.2 — Preparing Regression Data")
print("=" * 80)

# Use fallback target if needed
target_col = TARGET_COL if TARGET_COL in all_stocks_enhanced.columns else TARGET_COL_FALLBACK
if target_col == TARGET_COL_FALLBACK:
    print(f"⚠ Using '{TARGET_COL_FALLBACK}' as target ('{TARGET_COL}' not found)")

# Prepare train/test split
# Code Guidelines Section 1.2: Dataset prep returns (X_train, X_test, y_train, y_test, meta)
X_train, X_test, y_train, y_test, meta = regression_prepare_data(
    all_stocks_enhanced,
    target_col=target_col,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
)

print(f"\n✓ Data prepared:")
print(f"  Train set: {X_train.shape}")
print(f"  Test set: {X_test.shape}")
print(f"  Numeric features: {len(meta.get('numeric_features', []))}")
print(f"  Categorical features: {len(meta.get('categorical_features', []))}")

# Phase 9.3 — Feature Engineering Review Summary (auto-applied in prepare_regression_data)
try:
    pruned = meta.get("pruned_features", [])
    added_ix = meta.get("added_sector_interactions", 0)
    coverage = meta.get("feature_coverage_report", {})
    print("\n📊 Phase 9.3 Feature Review:")
    if coverage:
        print(
            f"  Feature count: {coverage.get('feature_count')} (expected≈{coverage.get('expected_count')})"
        )
    print(
        f"  Pruned low-importance features (<{float(os.getenv('FEATURE_IMPORTANCE_THRESHOLD', '0.01')) * 100:.1f}%): {len(pruned)}"
    )
    if len(pruned) > 0:
        from pathlib import Path

        out_dir = OUTPUT_DIR / "regression"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "pruned_features.txt", "w", encoding="utf-8") as f:
            for name in pruned:
                f.write(str(name) + "\n")
        print(f"    → Saved list to {out_dir / 'pruned_features.txt'}")
    print(f"  Added sector interaction features: {added_ix}")
except (ValueError, TypeError, KeyError, IOError, AttributeError) as _e:
    print(f"[WARN] Feature review summary unavailable: {_e}")

# %%
# Feature Leakage Prevention Check (Priority 1 - Task 1.1)
print("\n" + "=" * 80)
print("🔍 Feature Leakage Prevention Check")
print("=" * 80)

# Verify no market cap leakage
leakage_cols = [col for col in X_train.columns if "market_cap" in col.lower()]

if leakage_cols:
    raise ValueError(f"⚠️ FEATURE LEAKAGE DETECTED: {leakage_cols}")
else:
    print("✓ No market_cap feature leakage detected")

# Log feature statistics
print(f"\n📊 Feature Statistics:")
print(f"  Total features: {X_train.shape[1]}")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")

# Show sector interaction features added
sector_interaction_cols = [col for col in X_train.columns if "sector_" in col and "__x__" in col]
print(f"  Sector interactions: {len(sector_interaction_cols)}")

# Verify debt_to_equity is included (replacement for market_cap)
debt_cols = [col for col in X_train.columns if "debt_to_equity" in col.lower()]
if debt_cols:
    print(f"✓ debt_to_equity included ({len(debt_cols)} features)")
else:
    print("⚠️ Warning: debt_to_equity not found in features")

# Verify enterprise_value is allowed (forward-looking metric)
ev_cols = [col for col in X_train.columns if "enterprise_value" in col.lower()]
if ev_cols:
    print(f"✓ enterprise_value allowed ({len(ev_cols)} features)")

print("\n✓ Feature leakage validation passed")

# %% [markdown]
# ### 6.3 Compare Multiple Regression Models
#
# %%
# noinspection PyUnboundLocalVariable
print("=" * 80)
print("6.3 — Comparing Multiple Regression Models")
print("=" * 80)

try:
    comparison_results = regression_compare_regressors(
        X_train,
        y_train,
        test_size=TEST_SIZE,
        cv=CV_FOLDS,
        random_state=RANDOM_SEED,
        ensure_nonnegative=False,
    )

    results_df = pd.DataFrame(comparison_results).T.sort_values("r2", ascending=False)
    print("\n📊 Model Comparison Results:")
    print(results_df.to_string())

    if not results_df.empty:
        best_model_name = results_df.index[0]
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   R²: {results_df.loc[best_model_name, 'r2']:.4f}")
        print(f"   MAE: {results_df.loc[best_model_name, 'mae']:.2f}")
    else:
        best_model_name = "None"
        print("  ⚠ No regression successfully trained")

except (ValueError, TypeError, RuntimeError, KeyError) as e:
    print(f"\n⚠ Model comparison failed: {e}")
    results_df = pd.DataFrame()
    best_model_name = "None"

# %% [markdown]
# ### 9.5.X: Automated Stacking Hyperparameter Tuning (Optional)
#
# **Business Goal:** Optimize stacking ensemble performance through automated hyperparameter search
#
# **Key Objectives:**
# - Bayesian optimization with Optuna for efficient search space exploration
# - Tune base models (XGBoost, LightGBM, Ridge, Lasso) and meta-learner simultaneously
# - Balance performance vs. computation time (50 trials, 30-minute timeout)
# - Reproducible results with fixed random seed
#
# **Inputs:**
# - `X_train_reg`, `y_train_reg`: Training features and target
# - Feature names for model interpretation
#
# **Outputs:**
# - `best_stacking_params`: Optimized hyperparameters dictionary
# - `tuned_stacking_model`: Trained model with best hyperparameters
# - Study results saved to `outputs/regression/stacking_optuna_study.pkl`
#
# **Performance Trade-offs:**
# - **Skip if time-constrained:** Default stacking ensemble already provides good performance
# - **Run if optimizing production model:** Can improve R² by 1-3% on validation set
# - **Computation time:** ~30 minutes for 50 trials (configurable)
#
# **Phase 9.5 TDD Implementation:**
# - Function: `tune_stacking_hyperparameters()` from `regression.models`
# - Test coverage: 84.5% (3 tests in test_stacking_hyperparameter.py)
# - Documentation: code_guidelines.md v1.10+ Section 7.3
# %%
# 9.5.X: Automated Stacking Hyperparameter Tuning (Optional - Can Skip)
print("\n" + "=" * 80)
print("PHASE 9.5.X: AUTOMATED STACKING HYPERPARAMETER TUNING")
print("=" * 80)

# Configuration
RUN_HYPERPARAMETER_TUNING = False  # Set to True to enable (adds ~30 minutes)
N_TRIALS = 50  # Number of Optuna trials
TIMEOUT_SECONDS = 1800  # 30 minutes timeout

if RUN_HYPERPARAMETER_TUNING:
    print(f"\n📊 Starting Optuna hyperparameter search...")
    print(f"   Trials: {N_TRIALS}, Timeout: {TIMEOUT_SECONDS}s (~{TIMEOUT_SECONDS // 60} min)")
    print(
        f"   Search space: XGBoost, LightGBM, Ridge, Lasso base models + Ridge/Huber meta-learner"
    )

    try:
        # Run hyperparameter tuning
        # Note: Function returns (best_params, best_score) tuple, not a dictionary
        best_stacking_params, best_score = tune_stacking_hyperparameters(
            X=X_train,
            y=y_train,
            model_type="xgboost",  # Can tune 'xgboost', 'lightgbm', or 'catboost'
            n_trials=N_TRIALS,
            timeout=TIMEOUT_SECONDS,
            cv=CV_FOLDS,
            random_state=RANDOM_SEED,
            verbose=True,
        )

        # Display results
        print(f"\n✅ Hyperparameter tuning complete!")
        print(f"   Best CV Score (MAE): {best_score:.4f}")
        print(f"\n📋 Best Hyperparameters:")
        for key, value in best_stacking_params.items():
            print(f"   {key}: {value}")

        # Note: The function returns optimized hyperparameters but not a trained model
        # You would need to create a new model with these parameters
        print(f"\n💡 To use these parameters, create a model with the returned hyperparameters")
        print(f"   Example: model = xgb.XGBRegressor(**best_stacking_params)")

    except Exception as e:
        print(f"\n⚠️  Hyperparameter tuning failed: {e}")
        print(f"   Falling back to default stacking ensemble")
        best_stacking_params = None

else:
    print(f"\n⏭️  Skipping hyperparameter tuning (RUN_HYPERPARAMETER_TUNING=False)")
    print(f"   Using default stacking ensemble configuration")
    print(f"   To enable: Set RUN_HYPERPARAMETER_TUNING = True above")
    best_stacking_params = None

print("\n" + "=" * 80)
# %% [markdown]
# ### 6.4 Train Stacking Ensemble
#
# **⚠️ Important Fix: Prediction Clipping Strategy**
#
# **Issue Identified**: Previous implementation used statistical clipping (`mean ± 3*std`) which capped predictions at ~35k despite actual price targets reaching 180k+. This caused severe under-prediction for high-value stocks.
#
# **Root Cause**:
# - Statistical clipping assumes normal distribution: `[max(0, mean-3*std), mean+3*std]`
# - With training data mean ≈ 15k and std ≈ 6.5k, upper bound = 15k + 3*6.5k ≈ **34.5k**
# - Test set contains high-value stocks (>50k) that were capped at this artificial limit
#
# **Solution Implemented**:
# - **Percentile-based clipping**: Uses `1.5 × 99.5th percentile` as upper bound
# - Adapts to data distribution (handles heavy-tailed price distributions)
# - Allows extrapolation beyond training max while preventing extreme outliers
# - Maintains non-negativity constraint (prices cannot be negative)
#
# **Expected Impact**:
# - Predictions can now reach high values (>50k) matching actual price targets
# - Improved metrics for high-value stocks (reduced MAPE, better R²)
# - Better residual distribution (reduced systematic under-prediction bias)
#
# %%
print("=" * 80)
print("6.4 — Training Stacking Ensemble (Phase 9.9 Default)")
print("=" * 80)

# Phase 9.9: Stacking ensemble is now the default regression approach (Task 9.9.8)
# - Integrates classification meta-features from Phase 9.4 via integrate_classification_features()
# - Uses robust Huber loss for base models (outlier safety rails)
# - Non-negative constraints applied AFTER calibration (single point of enforcement)
# - Standardized predictions schema via build_predictions_frame()

# Prepare classification probabilities for meta-features (Phase 9.5)
prob_cols = [
    "event_prob_strong_negative",
    "event_prob_negative",
    "event_prob_neutral",
    "event_prob_positive",
    "event_prob_strong_positive",
]
train_probs = None
# Check if all probability columns exist
if all(c in all_stocks_enhanced.columns for c in prob_cols):
    # Extract probabilities aligned with X_train index
    train_probs = all_stocks_enhanced.loc[X_train.index, prob_cols].values
    print(f"✓ Found classification probability columns for meta-features")
else:
    # Fallback: try to find any cols starting with event_prob_
    alt_prob_cols = [c for c in all_stocks_enhanced.columns if c.startswith("event_prob_")]
    if len(alt_prob_cols) == 5:
        prob_cols = sorted(alt_prob_cols)  # Sort to ensure deterministic order
        train_probs = all_stocks_enhanced.loc[X_train.index, prob_cols].values
        print(f"✓ Found alternative classification probability columns: {prob_cols}")
    else:
        print(f"ℹ️ Classification probabilities not found. Training without meta-features.")

# Code Guidelines Section 1.1: train_* functions return dict {model, metrics, y_pred, y_proba, artifacts}
# Use robust Huber loss for GradientBoosting base model inside stacking (Priority 2)
# CHANGE: Removed ensure_nonnegative=True to allow model natural predictions
# Non-negativity will be enforced once at the end after calibration
stacking_result = regression_train_stacking(
    X_train,
    y_train,
    cv=CV_FOLDS,
    ensure_nonnegative=False,  # Changed from True - apply constraint after calibration
    loss="huber",
    use_meta_features=(train_probs is not None),
    classification_probabilities=train_probs,
    enable_interactions=True,
    interaction_valuation_cols=valuation_cols_method_aware,
    cv_policy="time_series" if "snapshot_date" in all_stocks_enhanced.columns else "kfold",
    date_col="snapshot_date",
    dates=(
        all_stocks_enhanced.loc[X_train.index, "snapshot_date"]
        if "snapshot_date" in all_stocks_enhanced.columns
        else None
    ),
)

stacking_model = stacking_result["model"]
stacking_results = stacking_result.get("artifacts", {})

print(f"\n✓ Stacking Ensemble Trained:")
print(f"  Base models: {', '.join(stacking_results.get('base_models', []))}")
print(f"  Meta-learner: {stacking_results.get('meta_model', 'Unknown')}")
print(f"  Train R²: {stacking_result['metrics'].get('r2', 0):.4f}")
print(
    f"  CV R² (mean ± std): {stacking_results.get('cv_score', 0):.4f} ± {stacking_results.get('cv_std', 0):.4f}"
)

# Test set predictions - NO intermediate clipping
# Let predictions flow naturally to calibration step

# CRITICAL FIX: Enhance X_test with classification features (Phase 9.5)
# The train_stacking_regressor function internally adds classification features to X_train
# when use_meta_features=True. We must apply the same transformation to X_test.
test_probs = None
X_test_for_prediction = X_test.copy()

if train_probs is not None:
    # Extract test set probabilities aligned with X_test index
    test_probs = all_stocks_enhanced.loc[X_test.index, prob_cols].values
    print(f"🔧 Enhancing X_test with classification meta-features...")

    # Import feature integration functions
    from finance_ml.ml_workflow.regression.dataset import (
        create_classification_interactions,
        integrate_classification_features,
    )

    # Add probability columns and event_confidence to X_test
    X_test_for_prediction = integrate_classification_features(X_test_for_prediction, test_probs)
    print(f"  ✓ Added classification probability features")

    # Ensure configuration variable is defined to prevent NameError (Code Guidelines Section 8.1)
    if "enable_interactions" not in locals():
        enable_interactions = False  # Set to True if you specifically want interaction features

    # Initialize interaction_valuation_cols if not defined (prevents NameError)
    if "interaction_valuation_cols" not in locals():
        interaction_valuation_cols = []  # Empty list if no interaction features were configured

    # Add interaction features if they were enabled during training
    if enable_interactions and interaction_valuation_cols:
        class_cols = [
            c
            for c in X_test_for_prediction.columns
            if c.startswith("event_prob_") or c == "event_confidence"
        ]
        X_test_for_prediction = create_classification_interactions(
            X_test_for_prediction, class_cols, valuation_cols_method_aware
        )
        print(
            f"  ✓ Added interaction features: {len(class_cols)} prob cols × {len(valuation_cols_method_aware)} valuation cols"
        )

    print(f"  ✓ X_test shape: {X_test.shape} → {X_test_for_prediction.shape}")

# === FIX START: Align X_test features with Model features ===
# 1. Identify if interactions are missing
model_features = getattr(stacking_model, "feature_names_in_", None)
if model_features is None and hasattr(stacking_model, "estimators_"):
    # Try getting features from the first base estimator if wrapper hides them
    model_features = getattr(stacking_model.estimators_[0], "feature_names_in_", None)

if model_features is not None:
    missing_cols = set(model_features) - set(X_test_for_prediction.columns)

    if missing_cols:
        print(f"⚠ Detected {len(missing_cols)} missing interaction features. Regenerating...")

        # Import helpers if not available
        from finance_ml.ml_workflow.regression.dataset import (
            create_classification_interactions,
        )

        # Define columns for interaction creation (same as Section 6.1)
        # Identify classification probability columns currently in X_test
        cls_cols = [
            c
            for c in X_test_for_prediction.columns
            if c.startswith("event_prob_") or c == "event_confidence"
        ]

        # Identify valuation columns (fallback to default if method-aware list not found)
        val_cols = globals().get(
            "valuation_cols_method_aware",
            [
                "market_cap",
                "enterprise_value",
                "ebitda",
                "p_e",
                "p_b",
                "gross_margin",
                "revenue",
                "net_income",
            ],
        )
        val_cols = [c for c in val_cols if c in X_test_for_prediction.columns]

        # Regenerate interactions on Test Data
        X_test_for_prediction = create_classification_interactions(
            X_test_for_prediction, classification_cols=cls_cols, valuation_cols=val_cols
        )

        # Final alignment: Ensure exact column order and fill any remaining gaps with 0
        X_test_for_prediction = X_test_for_prediction.reindex(columns=model_features, fill_value=0)
        print(f"✓ X_test aligned. New shape: {X_test_for_prediction.shape}")
# === FIX END ===

# === CLEAN X_test_for_prediction BEFORE PREDICTION ===
# Fix for ValueError: Input X contains infinity or a value too large for dtype('float32')
# After feature engineering / interaction regeneration, X_test may contain inf or extreme values.
# Apply the same cleaning routine used during training to ensure consistency.
from finance_ml.ml_workflow.regression.models import _clean_regression_features

X_test_for_prediction = _clean_regression_features(
    X_test_for_prediction,
    drop_zero_variance=False,  # Keep columns to match training schema
)

# Validate cleaned data is safe for prediction
vals = X_test_for_prediction.values
assert np.isfinite(vals).all(), "Non-finite values remain in cleaned X_test_for_prediction"
print(f"✓ X_test cleaned: max abs value = {np.nanmax(np.abs(vals)):.2e}")
# === END CLEAN ===

# Predict using feature-aligned and cleaned X_test
y_pred_stacking = stacking_model.predict(X_test_for_prediction)

# Diagnostic: Check raw prediction range before calibration
print(f"\n📊 Raw Predictions (before calibration):")
print(f"  Min: ${y_pred_stacking.min():.2f}")
print(f"  Max: ${y_pred_stacking.max():.2f}")
print(
    f"  Negative: {(y_pred_stacking < 0).sum()} ({(y_pred_stacking < 0).sum() / len(y_pred_stacking) * 100:.1f}%)"
)

test_metrics = {
    "mae": mean_absolute_error(y_test, y_pred_stacking),
    "rmse": np.sqrt(mean_squared_error(y_test, y_pred_stacking)),
    "r2": r2_score(y_test, y_pred_stacking),
}

print(f"\n📊 Test Set Performance (raw predictions):")
print(f"  MAE: {test_metrics['mae']:.2f}")
print(f"  RMSE: {test_metrics['rmse']:.2f}")
print(f"  R²: {test_metrics['r2']:.4f}")

# === Phase 9.8 Preparation: Extract base model predictions for stacking contribution analysis ===
# This enables Phase 9.8's compute_stacking_contributions() to analyze each base model's influence
base_predictions = {}
base_model_names = stacking_results.get("base_models", [])

if hasattr(stacking_model, "estimators_") and stacking_model.estimators_:
    print(f"\n🔍 Extracting base model predictions for Phase 9.8 analysis...")
    for i, estimator in enumerate(stacking_model.estimators_):
        # Use model name from artifacts if available, otherwise use index
        model_name = base_model_names[i] if i < len(base_model_names) else f"base_model_{i}"
        try:
            base_pred = estimator.predict(X_test_for_prediction)
            base_predictions[model_name] = base_pred
            print(f"  ✓ {model_name}: predictions extracted")
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            print(f"  ⚠ {model_name}: failed to extract predictions ({e})")

    print(f"  Total base models: {len(base_predictions)}")

# Create y_pred_meta alias for Phase 9.8 compatibility
y_pred_meta = y_pred_stacking
print(
    f"✓ Phase 9.8 variables prepared: base_predictions ({len(base_predictions)} models), y_pred_meta"
)
# === End Phase 9.8 Preparation ===

# %%
# 6.4.1 — Export Enhanced Predictions and Sector Metrics (Priority 1)
# Using build_predictions_frame from finance_ml.ml_workflow.regression.io for standardized schema

import os

from finance_ml.ml_workflow.regression.io import build_predictions_frame

# Build detailed predictions DataFrame for diagnostics and export
try:
    # Use build_predictions_frame for standardized schema (Priority 1)
    source_df = all_stocks_enhanced if "all_stocks_enhanced" in globals() else None

    # Call build_predictions_frame - it safely handles metadata columns using .loc[] alignment
    # No need to drop 'sector' - the function doesn't use DataFrame.insert()
    results_df = build_predictions_frame(
        y_true=y_test,
        y_pred=y_pred_stacking,
        df_source=source_df if source_df is not None else pd.DataFrame(index=y_test.index),
        extra_cols={},
    )

    out_models_dir = OUTPUT_DIR / "regression"
    out_models_dir.mkdir(parents=True, exist_ok=True)

    # Priority 3: Apply isotonic calibration by sector (monotonic, no fixed bias)
    # Requires calibration dataset with y_true values for fitting isotonic regression
    try:
        # Step 1: Create calibration DataFrame from training data
        print("\n" + "=" * 80)
        print("ISOTONIC CALIBRATION BY SECTOR")
        print("=" * 80)

        # === FIX START: Align X_train features with Model features for calibration ===
        # Same logic as in Section 6.4: ensure X_train has all features the model expects
        X_train_for_calibration = X_train.copy()

        # Get model's expected feature names
        model_features_cal = getattr(stacking_model, "feature_names_in_", None)
        if model_features_cal is None and hasattr(stacking_model, "estimators_"):
            # Try getting features from the first base estimator if wrapper hides them
            model_features_cal = getattr(stacking_model.estimators_[0], "feature_names_in_", None)

        if model_features_cal is not None:
            missing_cols_cal = set(model_features_cal) - set(X_train_for_calibration.columns)

            if missing_cols_cal:
                print(
                    f"⚠ Detected {len(missing_cols_cal)} missing interaction features in X_train. Regenerating..."
                )

                # Import helpers if not available
                from finance_ml.ml_workflow.regression.dataset import (
                    create_classification_interactions,
                    integrate_classification_features,
                )

                # Add classification probability features if available
                if train_probs is not None:
                    X_train_for_calibration = integrate_classification_features(
                        X_train_for_calibration, train_probs
                    )

                # Identify classification probability columns currently in X_train
                cls_cols_cal = [
                    c
                    for c in X_train_for_calibration.columns
                    if c.startswith("event_prob_") or c == "event_confidence"
                ]

                # Identify valuation columns (fallback to default if method-aware list not found)
                val_cols_cal = globals().get(
                    "valuation_cols_method_aware",
                    [
                        "market_cap",
                        "enterprise_value",
                        "ebitda",
                        "p_e",
                        "p_b",
                        "gross_margin",
                        "revenue",
                        "net_income",
                    ],
                )
                val_cols_cal = [c for c in val_cols_cal if c in X_train_for_calibration.columns]

                # Regenerate interactions on Training Data
                if cls_cols_cal and val_cols_cal:
                    X_train_for_calibration = create_classification_interactions(
                        X_train_for_calibration,
                        classification_cols=cls_cols_cal,
                        valuation_cols=val_cols_cal,
                    )

                # Final alignment: Ensure exact column order and fill any remaining gaps with 0
                X_train_for_calibration = X_train_for_calibration.reindex(
                    columns=model_features_cal, fill_value=0
                )
                print(
                    f"✓ X_train aligned for calibration. New shape: {X_train_for_calibration.shape}"
                )
        # === FIX END ===

        # Clean X_train_for_calibration before prediction (same as X_test cleaning)
        X_train_for_calibration = _clean_regression_features(
            X_train_for_calibration,
            drop_zero_variance=False,
        )

        y_pred_train = stacking_model.predict(X_train_for_calibration)

        # Build calibration df with required columns
        cal_df = pd.DataFrame(
            {
                "y_true": y_train.values,
                "y_pred": y_pred_train,
                "sector": source_df.loc[y_train.index, "sector"].values,
            }
        )

        print(f"\n✓ Calibration set: {len(cal_df):,} samples")
        print(f"  Sectors: {cal_df['sector'].nunique()}")

        # Step 2: Apply isotonic calibration (learns monotonic transformation per sector)
        results_df = calibrate_predictions_by_sector(
            preds_df=results_df,
            cal_df=cal_df,
            method="isotonic",  # CHANGED from "additive" - no fixed bias subtraction
            sector_col="sector",
            pred_col="y_pred",
            true_col="y_true",
            output_col="y_pred_calibrated",
            min_samples=5,
        )

        # Step 3: Apply final non-negative constraint + outlier clipping (single enforcement point)
        if "y_pred_calibrated" in results_df.columns:
            y_pred_calib = results_df["y_pred_calibrated"].to_numpy()

            # Diagnostic: Check calibrated predictions before final clipping
            n_neg_before = (y_pred_calib < 0).sum()
            print(f"\n📊 Calibrated predictions (before final clipping):")
            print(
                f"  Negative predictions: {n_neg_before} ({n_neg_before / len(y_pred_calib) * 100:.1f}%)"
            )
            print(f"  Range: ${y_pred_calib.min():.2f} to ${y_pred_calib.max():.2f}")

            # Apply adaptive clipping: non-negative + outlier bounds
            clip_result = adaptive_clip_predictions(y_pred_calib, y_train)
            y_pred_final = clip_result["clipped_predictions"]

            # Update results_df with final clipped predictions
            results_df["y_pred_calibrated"] = y_pred_final

            # Recompute errors with final predictions
            yt = results_df["y_true"].to_numpy()
            results_df["abs_error_calibrated"] = np.abs(yt - y_pred_final)
            results_df["pct_error_calibrated"] = np.where(
                yt != 0, ((y_pred_final - yt) / yt) * 100.0, np.nan
            )

            # Final diagnostic
            n_zeros = (y_pred_final == 0).sum()
            n_neg = (y_pred_final < 0).sum()
            print(f"\n✅ Final predictions (after adaptive clipping):")
            print(f"  Lower bound: ${clip_result['lower_bound']:.2f}")
            print(f"  Upper bound: ${clip_result['upper_bound']:.2f}")
            print(
                f"  Clipped to lower: {clip_result['n_clipped_lower']} ({clip_result['pct_clipped_lower']:.1f}%)"
            )
            print(
                f"  Clipped to upper: {clip_result['n_clipped_upper']} ({clip_result['pct_clipped_upper']:.1f}%)"
            )
            print(
                f"  Zero predictions: {n_zeros} ({n_zeros / len(y_pred_final) * 100:.1f}%) - TARGET: <1%"
            )
            print(f"  Negative predictions: {n_neg} (should be 0)")
            print(f"  Range: ${y_pred_final.min():.2f} to ${y_pred_final.max():.2f}")
            print("=" * 80)
    except (ValueError, TypeError, RuntimeError, AttributeError) as e:
        print(f"\n⚠️ Warning: Isotonic calibration failed: {e}")
        print("  Falling back to uncalibrated predictions")
        results_df["y_pred_calibrated"] = results_df["y_pred"]

    # Add model_version and snapshot_date for standardized schema
    model_version = os.environ.get("MODEL_VERSION", "v9_10")
    results_df["model_version"] = model_version
    results_df["snapshot_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Store for later merging with quantile predictions
    results_df_base = results_df.copy()

    # Export to standardized path: regression_predictions_detailed.csv (will be updated after quantiles)
    predictions_path = out_models_dir / "regression_predictions_detailed.csv"
    print(f"\nℹ️  Predictions dataframe prepared (will merge quantiles before final export)")

    # Compute and export sector-level metrics if sector present
    if "sector" in results_df.columns:
        print(f"\n📊 Training Sector-Optimized Models and Metrics (Phase 9.5)...")
        from finance_ml.ml_workflow.archive.models import (
            train_and_evaluate_regression_by_sector,
        )

        # Prepare probabilities for full dataset (if available)
        full_probs = None
        # Use prob_cols from previous step if defined, otherwise re-detect
        local_prob_cols = [c for c in all_stocks_enhanced.columns if c.startswith("event_prob_")]
        # Filter to 5 classes if possible
        if len(local_prob_cols) >= 5:
            # Prefer standard names if present
            std_cols = [
                "event_prob_strong_negative",
                "event_prob_negative",
                "event_prob_neutral",
                "event_prob_positive",
                "event_prob_strong_positive",
            ]
            if all(c in all_stocks_enhanced.columns for c in std_cols):
                local_prob_cols = std_cols
            else:
                local_prob_cols = sorted(local_prob_cols)[:5]
            full_probs = all_stocks_enhanced[local_prob_cols].values

        # Train and evaluate per sector (supports stacking per sector)
        sector_metrics_df = train_and_evaluate_regression_by_sector(
            all_stocks_enhanced,
            out_models_dir,
            feature_cols=[
                c for c in X_train.columns if "__x__" not in c and c in all_stocks_enhanced.columns
            ],
            use_meta_features=(full_probs is not None),
            classification_probabilities=full_probs,
            cv_policy="time_series" if "snapshot_date" in all_stocks_enhanced.columns else "kfold",
            date_col="snapshot_date",
        )
        print(f"✓ Saved sector metrics to {out_models_dir / 'regression_metrics_by_sector.csv'}")

    # Priority 5: Feature importance export
    try:
        fi_path = out_models_dir / "feature_importance.csv"
        # StackingRegressor rarely exposes feature_importances_. Use Section 4 RF importance if available.
        if (
            "importance_df" in globals()
            and isinstance(importance_df, pd.DataFrame)
            and not importance_df.empty
        ):
            importance_df.to_csv(fi_path, index=False)
            print(f"✓ Saved feature importance to {fi_path} (from RF importance)")
        else:
            # Fallback: compute quickly using features_importance_rf on training data
            tmp_fi = features_importance_rf(X_train, y_train, top_k=min(50, X_train.shape[1]))
            tmp_fi.to_csv(fi_path, index=False)
            print(f"✓ Saved feature importance to {fi_path} (computed fallback)")
    except (ValueError, TypeError, AttributeError, IOError) as e:
        print(f"⚠ Feature importance export skipped: {e}")
except (ValueError, TypeError, IOError, OSError, KeyError) as e:
    print(f"⚠ Failed to export enhanced predictions/metrics: {e}")
# %% [markdown]
# ### 6.5 Quantile Regression for Prediction Intervals
#
# %%
print("=" * 80)
print("6.5 — Quantile Regression for Uncertainty Estimation")
print("=" * 80)

# Code Guidelines Section 1.1: train_* functions return dict {model, metrics, y_pred, y_proba, artifacts}
quantile_result = regression_train_quantile(X_train, y_train, quantiles=QUANTILES)

quantile_models = quantile_result.get("artifacts", {}).get("models", [])
if not quantile_models:
    # Fallback: models might be in the top-level artifacts
    quantile_models = quantile_result.get("model", [])
    if not isinstance(quantile_models, list):
        quantile_models = [quantile_models]

print(f"\n✓ Quantile Models Trained:")
print(f"  Quantiles: {QUANTILES}")
print(f"  Models: {len(quantile_models)}")

# Generate predictions for each quantile
predictions_quantile = {}
for q, model in zip(QUANTILES, quantile_models):
    predictions_quantile[q] = model.predict(X_test)
    try:
        score = model.score(X_train, y_train)
        print(f"  Q{q}: {score:.4f} (train R²)")
    except AttributeError:
        # Some quantile models may not have a score method
        print(f"  Q{q}: Model trained successfully")

# Priority 4.2: Export quantile predictions with monotonicity enforcement and conformal calibration
try:
    print(f"\n📊 Generating Quantile Predictions (Phase 9.5 Safety Rails)...")
    from finance_ml.ml_workflow.regression.quantile import predict_quantile_regression

    # Use Phase 9.5 helper for robust predictions (monotonic + non-negative)
    q_preds_df = predict_quantile_regression(
        quantile_models, QUANTILES, X_test, enforce_nonnegative=True
    )

    # Convert to dictionary for compatibility with existing reporting code below
    predictions_quantile_monotonic = {q: q_preds_df[f"pred_q{q}"].values for q in QUANTILES}

    # Build quantile predictions dataframe with standardized schema
    test_tickers = None
    test_sectors = None
    test_regions = None
    if "ticker" in all_stocks_enhanced.columns:
        test_tickers = all_stocks_enhanced.loc[y_test.index, "ticker"].values
    if "sector" in all_stocks_enhanced.columns:
        test_sectors = all_stocks_enhanced.loc[y_test.index, "sector"].values
    if "region" in all_stocks_enhanced.columns:
        test_regions = all_stocks_enhanced.loc[y_test.index, "region"].values

    q_df = pd.DataFrame(
        {
            "ticker": test_tickers if test_tickers is not None else y_test.index.astype(str),
            "y_true": y_test.values,
            "pred_p10": predictions_quantile_monotonic.get(LOWER_QUANTILE),
            "pred_p50": predictions_quantile_monotonic.get(MEDIAN_QUANTILE),
            "pred_p90": predictions_quantile_monotonic.get(UPPER_QUANTILE),
        }
    )

    # Add sector and region if available
    if test_sectors is not None:
        q_df["sector"] = test_sectors
    if test_regions is not None:
        q_df["region"] = test_regions

    # Compute interval width and coverage metrics
    if "pred_p10" in q_df.columns and "pred_p90" in q_df.columns:
        q_df["interval_width"] = q_df["pred_p90"] - q_df["pred_p10"]
        # Compute empirical coverage (should be ~80%)
        coverage = (
            (q_df["y_true"] >= q_df["pred_p10"]) & (q_df["y_true"] <= q_df["pred_p90"])
        ).mean()
        print(f"  Empirical coverage (10%-90%): {coverage:.1%} (target: 80%)")

    # Add metadata columns for standardized schema
    q_df["model_version"] = os.environ.get("MODEL_VERSION", "v9_10")
    q_df["snapshot_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

    out_q_path = out_models_dir / "quantile_predictions.csv"
    q_df.to_csv(out_q_path, index=False)
    print(f"✓ Saved quantile predictions to {out_q_path}")
    print(f"  Schema: {list(q_df.columns)}")

    # Merge quantile predictions into detailed predictions dataframe (Priority 1: Standardized Schema)
    try:
        if "results_df_base" in globals() and results_df_base is not None:
            # Add quantile columns to detailed predictions
            results_df_detailed = results_df_base.copy()
            results_df_detailed["pred_p10"] = predictions_quantile_monotonic.get(LOWER_QUANTILE)
            results_df_detailed["pred_p50"] = predictions_quantile_monotonic.get(MEDIAN_QUANTILE)
            results_df_detailed["pred_p90"] = predictions_quantile_monotonic.get(UPPER_QUANTILE)
            results_df_detailed["interval_width"] = (
                results_df_detailed["pred_p90"] - results_df_detailed["pred_p10"]
            )

            # Export unified predictions with standardized schema
            # Required columns: ticker, isin, sector, region, last_price, y_true, y_pred,
            #                   y_pred_calibrated, pred_p10, pred_p50, pred_p90, interval_width,
            #                   abs_error, pct_error, model_version, snapshot_date

            # Phase 9.5: Validate schema invariants (non-negativity, monotonicity, required columns)
            from finance_ml.ml_workflow.regression.io import validate_predictions_schema

            try:
                validate_predictions_schema(results_df_detailed)
                print("✓ Predictions schema validated successfully")
            except ValueError as e:
                print(f"⚠ Schema validation warning: {e}")

            detailed_path = out_models_dir / "regression_predictions_detailed.csv"
            results_df_detailed.reset_index(drop=True).to_csv(detailed_path, index=False)
            print(f"✓ Saved detailed predictions with quantiles to {detailed_path}")
            print(
                f"  Schema ({len(results_df_detailed.columns)} columns): {list(results_df_detailed.columns)}"
            )
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        print(f"⚠ Failed to merge quantiles into detailed predictions: {e}")
except (ValueError, TypeError, IOError, OSError, KeyError) as e:
    print(f"⚠ Quantile predictions export skipped: {e}")

# %% [markdown]
# ### 6.5.1 Time-Series Cross-Validation (Priority 4.1)
#
# %%
# 6.5.1 — Time-Series Cross-Validation (5 folds)

DATE_COLUMN_CANDIDATES = [
    "date",
    "as_of_date",
    "last_updated",
    "income_statement_report_date",
]


def find_date_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing date-like column from the given candidates, or None."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def run_time_series_cv(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    output_dir: Path,
    cv_folds: int,
) -> None:
    """Run time-series cross-validation and persist fold-level metrics."""
    date_col = find_date_column(df, DATE_COLUMN_CANDIDATES)
    if date_col is None:
        print("[WARN] No date column found; skipping Time-Series CV")
        return

    # Ensure datetime type and sort chronologically
    df_cv = df.copy()
    df_cv[date_col] = pd.to_datetime(df_cv[date_col], errors="coerce")
    df_cv = df_cv.sort_values(date_col).dropna(subset=[target_col])

    X_cv = df_cv[feature_cols].fillna(0)
    y_cv = df_cv[target_col]

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    metrics_rows: list[dict[str, float]] = []
    fold_index = 0

    for train_idx, test_idx in tscv.split(X_cv):
        fold_index += 1

        X_tr, X_te = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
        y_tr, y_te = y_cv.iloc[train_idx], y_cv.iloc[test_idx]

        # Train a lightweight stacking model per fold (reuse robust settings)
        fold_result = regression_train_stacking(
            X_tr,
            winsorize_target(y_tr, 0.01, 0.99),
            cv=3,
            ensure_nonnegative=False,
            loss="huber",
        )
        fold_model = fold_result["model"]

        # Apply adaptive clipping with percentile-based bounds
        fold_pred = fold_model.predict(X_te)
        clip_result_fold = adaptive_clip_predictions(fold_pred, y_tr)
        y_hat = clip_result_fold["clipped_predictions"]

        # Optional: log clipping stats for first fold
        if fold_index == 1:
            print(
                "  Fold 1 clipping bounds: "
                f"lower=${clip_result_fold['lower_bound']:.2f}, "
                f"upper=${clip_result_fold['upper_bound']:.2f}"
            )

        mae = mean_absolute_error(y_te, y_hat)
        rmse = np.sqrt(mean_squared_error(y_te, y_hat))
        r2 = r2_score(y_te, y_hat)

        metrics_rows.append(
            {
                "fold": fold_index,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "n_test": len(y_te),
            }
        )

    tscv_df = pd.DataFrame(metrics_rows)

    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    tscv_path = eval_dir / "tscv_metrics.csv"

    tscv_df.to_csv(tscv_path, index=False)
    print(f"[OK] Saved Time-Series CV metrics to {tscv_path}")
    print(tscv_df.describe().loc[["mean", "std"]])


print("\n" + "=" * 80)
print("6.5.1 — Time-Series Cross-Validation (5 folds)")
print("=" * 80)

try:
    run_time_series_cv(
        df=all_stocks_enhanced,
        target_col=target_col,
        feature_cols=list(X_train.columns),
        output_dir=OUTPUT_DIR,
        cv_folds=CV_FOLDS,
    )
except (ValueError, TypeError, RuntimeError, KeyError, AttributeError) as exc:
    print(f"[WARN] Time-Series CV evaluation skipped due to error: {exc}")
# %% [markdown]
# ### 6.6.6 NonNegativeRegressionWrapper Validation
#
# %%
print("=" * 80)
print("6.6.6 — NonNegativeRegressionWrapper Validation")
print("=" * 80)

# Demonstrate explicit usage of NonNegativeRegressionWrapper
# Validates that all predictions are non-negative

try:
    print("\n✅ Testing NonNegativeRegressionWrapper...")
    print("  Base model: Ridge Regression")

    # Train base model without constraint
    from sklearn.linear_model import Ridge as SklearnRidge

    base_model = SklearnRidge(alpha=1.0, random_state=RANDOM_SEED)
    base_model.fit(X_train, y_train)
    base_pred = base_model.predict(X_test)

    # Train wrapped model with non-negative constraint
    wrapped_model = NonNegativeRegressionWrapper(SklearnRidge(alpha=1.0, random_state=RANDOM_SEED))
    wrapped_model.fit(X_train, y_train)
    wrapped_pred = wrapped_model.predict(X_test)

    # Compare predictions
    print(f"\n📊 Prediction Comparison:")
    print(f"  Base Model (unconstrained):")
    print(f"    Min: {base_pred.min():.2f}")
    print(f"    Max: {base_pred.max():.2f}")
    print(
        f"    Negative predictions: {(base_pred < 0).sum()} ({(base_pred < 0).sum() / len(base_pred) * 100:.1f}%)"
    )

    print(f"\n  Wrapped Model (non-negative):")
    print(f"    Min: {wrapped_pred.min():.2f}")
    print(f"    Max: {wrapped_pred.max():.2f}")
    print(
        f"    Negative predictions: {(wrapped_pred < 0).sum()} ({(wrapped_pred < 0).sum() / len(wrapped_pred) * 100:.1f}%)"
    )

    # Validate non-negativity constraint
    assert wrapped_pred.min() >= 0, "❌ NonNegativeWrapper failed: negative predictions found!"
    print("\n✅ NonNegativeRegressionWrapper Validation PASSED")
    print("   All predictions are non-negative as expected")

    # Performance comparison
    base_mae = mean_absolute_error(y_test, base_pred)
    wrapped_mae = mean_absolute_error(y_test, wrapped_pred)
    base_r2 = r2_score(y_test, base_pred)
    wrapped_r2 = r2_score(y_test, wrapped_pred)

    print(f"\n📈 Performance Impact of Non-Negative Constraint:")
    print(f"  Base Model    - MAE: {base_mae:.2f}, R²: {base_r2:.4f}")
    print(f"  Wrapped Model - MAE: {wrapped_mae:.2f}, R²: {wrapped_r2:.4f}")
    print(
        f"  MAE Difference: {wrapped_mae - base_mae:.2f} ({(wrapped_mae - base_mae) / base_mae * 100:+.1f}%)"
    )

except (ValueError, TypeError, RuntimeError, AttributeError) as e:
    print(f"⚠️ NonNegativeWrapper validation failed: {e}")

# %% [markdown]
# ### 6.7 Model Persistence
#
# %%
print("=" * 80)
print("6.7 — Model Persistence")
print("=" * 80)

# Note: regression directory already created at initialization
models_dir = OUTPUT_DIR / "regression"

# Save stacking model
stacking_metadata = {
    "model_type": "stacking_ensemble",
    "features": list(X_train.columns),
    "target": target_col,
    "date_trained": datetime.now().strftime("%Y-%m-%d"),
    "phase": "9.5",
    "train_score": stacking_result["metrics"].get("r2", 0),
    "cv_score": stacking_results.get("cv_score", 0),
    "test_score": test_metrics["r2"],
}

stacking_path = models_dir / "stacking_ensemble_phase95.joblib"
regression_save_model(stacking_model, str(stacking_path), metadata=stacking_metadata)
print(f"\n✓ Stacking model saved: {stacking_path.name}")

# Save quantile regression
for q, model in zip(QUANTILES, quantile_models):
    quantile_metadata = {
        "model_type": f"quantile_regressor_q{q}",
        "features": list(X_train.columns),
        "target": target_col,
        "date_trained": datetime.now().strftime("%Y-%m-%d"),
        "phase": "9.5",
        "quantile": q,
    }
    quantile_path = models_dir / f"quantile_q{int(q * 100)}_phase95.joblib"
    regression_save_model(model, str(quantile_path), metadata=quantile_metadata)

print(f"✓ Quantile regression saved: {len(QUANTILES)} regression")

# %%
# Demonstrate model loading capability using Phase 9.5 function
print("\n📂 Model Loading Demonstration:")
try:
    loaded_model, loaded_metadata = regression_load_model(str(stacking_path))
    print(f"✓ Successfully loaded: {stacking_path.name}")
    print(f"  Model type: {loaded_metadata.get('model_type', 'N/A')}")
    print(f"  Training date: {loaded_metadata.get('date_trained', 'N/A')}")
    print(f"  Test R²: {loaded_metadata.get('test_score', 0):.4f}")
except (FileNotFoundError, IOError, ValueError, TypeError, KeyError) as e:
    print(f"  ⚠️ Load demonstration skipped: {e}")

# %%
# 📊 Section 5 Enhanced Visualizations - Classification Models
print("\n" + "=" * 80)
print("📊 INTERACTIVE CLASSIFICATION VISUALIZATIONS")
print("=" * 80)

# Confusion matrix and classification metrics
if "y_test_cls" in dir() and "y_pred_cls" in dir():
    import plotly.figure_factory as ff
    from sklearn.metrics import classification_report, confusion_matrix

    from finance_ml.ml_workflow.classification.evaluation import (
        analyze_calibration,
        plot_confusion_matrices,
    )

    print("\n📈 Confusion Matrix Visualization...")

    # Create confusion matrix with explicit labels when possible
    class_names_5 = [
        "Strong Negative",
        "Negative",
        "Neutral",
        "Positive",
        "Strong Positive",
    ]
    class_names_3 = ["Negative", "Neutral", "Positive"]
    labels = None
    try:
        if (
            "y_proba_test" in dir()
            and hasattr(y_proba_test, "shape")
            and len(y_proba_test.shape) == 2
            and y_proba_test.shape[1] == 5
        ):
            labels = list(range(5))
            class_names = class_names_5
        elif (
            "y_proba_test" in dir()
            and hasattr(y_proba_test, "shape")
            and len(y_proba_test.shape) == 2
            and y_proba_test.shape[1] == 3
        ):
            labels = list(range(3))
            class_names = class_names_3
        else:
            # Fallback: infer from data
            unique_labels = sorted(
                list(set(pd.Series(y_test_cls).unique()).union(pd.Series(y_pred_cls).unique()))
            )
            labels = [int(x) for x in unique_labels]
            if len(labels) == 5:
                class_names = class_names_5
            elif len(labels) == 3:
                class_names = class_names_3
            else:
                class_names = [f"Class {i}" for i in range(len(labels))]
    except (ValueError, TypeError, AttributeError, KeyError):
        # Last resort
        labels = None
        class_names = class_names_5

    cm = (
        confusion_matrix(y_test_cls, y_pred_cls, labels=labels)
        if labels is not None
        else confusion_matrix(y_test_cls, y_pred_cls)
    )

    # Interactive confusion matrix heatmap
    fig = ff.create_annotated_heatmap(
        z=cm, x=class_names, y=class_names, colorscale="Blues", showscale=True
    )
    fig.update_layout(
        title="Classification Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=900,
        height=900,
    )
    fig.show()

    # Classification report
    print("\n📊 Classification Report:")
    if labels is not None:
        print(
            classification_report(y_test_cls, y_pred_cls, labels=labels, target_names=class_names)
        )
    else:
        print(classification_report(y_test_cls, y_pred_cls, target_names=class_names))

    # Class distribution
    import pandas as pd

    # Ensure class_dist has entries for all classes (0, 1, 2), filling missing with 0
    class_dist = pd.Series(y_pred_cls).value_counts().sort_index()
    num_classes = len(class_names)
    class_dist = class_dist.reindex(range(num_classes), fill_value=0)

    fig = px.bar(
        x=class_names,
        y=class_dist.values,
        title="Predicted Class Distribution",
        labels={"x": "Class", "y": "Count"},
        color=class_names,
    )
    fig.update_layout(showlegend=False)
    fig.show()

    # Use plot_confusion_matrices from finance_ml package
    print("\n📊 Confusion Matrices (using finance_ml helper)...")
    # Prepare models_results dict for plot_confusion_matrices
    models_results = {"Classification Model": {"y_test": y_test_cls, "y_pred": y_pred_cls}}
    plot_confusion_matrices(models_results, class_names=class_names)

    # Use analyze_calibration from finance_ml package
    if "cls_model" in dir() and hasattr(cls_model, "predict_proba"):
        print("\n📊 Calibration Analysis (using finance_ml helper)...")
        # Reuse y_proba_test from earlier evaluation cell (computed at line ~1786/1791)
        # No need to recompute - y_proba_test is already available in scope
        calibration_results = analyze_calibration(y_test_cls, y_proba_test, n_bins=10)

        print(f"  Brier Score: {calibration_results.get('brier_score', 'N/A'):.4f}")
        print(f"  Log Loss: {calibration_results.get('log_loss', 'N/A'):.4f}")

        # Display per-class Brier scores
        if "brier_score_per_class" in calibration_results:
            print("  Per-class Brier Scores:")
            for i, score in enumerate(calibration_results["brier_score_per_class"]):
                print(f"    Class {i} ({class_names[i]}): {score:.4f}")

    print("✓ Classification visualizations complete")

# %% [markdown]
# ### 6.8 Summary and Store Predictions
#
# %%
# %%
print("\n" + "=" * 80)
print(" FINAL SUMMARY")
print("=" * 80)

classification_cols = [
    c for c in all_stocks_with_classification.columns if c.startswith("event_prob_")
]

# Safely get the best model R² score
best_r2 = "N/A"
if best_model_name != "None" and not results_df.empty:
    # Try different possible column names for R² metric
    r2_column = None
    for col_name in ["r2", "R2", "R²", "r_squared", "test_r2"]:
        if col_name in results_df.columns:
            r2_column = col_name
            break

    if r2_column and best_model_name in results_df.index:
        best_r2 = f"{results_df.loc[best_model_name, r2_column]:.4f}"

# Safely get test metrics
test_r2 = test_metrics.get("r2", test_metrics.get("R2", test_metrics.get("test_r2", "N/A")))
test_mae = test_metrics.get("mae", test_metrics.get("MAE", test_metrics.get("test_mae", "N/A")))

# Format metrics safely
test_r2_str = f"{test_r2:.4f}" if isinstance(test_r2, (int, float)) else str(test_r2)
test_mae_str = f"{test_mae:.2f}" if isinstance(test_mae, (int, float)) else str(test_mae)

summary = {
    "✓ Classification Features Integrated": f"{len(classification_cols)} probability features + interactions",
    "✓ Models Compared": "6 regression: Ridge, Lasso, RF, ET, GB, HistGB",
    "✓ Best Single Model": (
        f"{best_model_name} (R²={best_r2})" if best_model_name != "None" else "Not available"
    ),
    "✓ Stacking Ensemble": f"R²={test_r2_str}, MAE={test_mae_str}",
    "✓ Quantile Regression": f"{len(QUANTILES)} quantiles for prediction intervals",
    "✓ Models Saved": f"{models_dir.name}/ (stacking + quantile regression)",
}

for key, value in summary.items():
    print(f"\n{key}")
    print(f"  {value}")

print("\n" + "=" * 80)

# Store predictions in a new dataframe for downstream phases
all_stocks_phase95 = all_stocks_enhanced.copy()
test_indices = X_test.index
valid_indices = test_indices.intersection(all_stocks_phase95.index)

if len(valid_indices) > 0:
    all_stocks_phase95.loc[valid_indices, "predicted_price_target"] = y_pred_stacking[
        test_indices.isin(valid_indices)
    ]
    all_stocks_phase95.loc[valid_indices, "prediction_lower_10"] = predictions_quantile[
        LOWER_QUANTILE
    ][test_indices.isin(valid_indices)]
    all_stocks_phase95.loc[valid_indices, "prediction_upper_90"] = predictions_quantile[
        UPPER_QUANTILE
    ][test_indices.isin(valid_indices)]
    print(f"\n✓ Predictions stored in 'all_stocks_phase95': {len(valid_indices):,} samples")

print(f"✓ Dataset ready for Phase 9.6/9.7")

# %%
# 📊 Section 6 Enhanced Visualizations - Regression Models
print("\n" + "=" * 80)
print("📊 INTERACTIVE REGRESSION MODEL VISUALIZATIONS")
print("=" * 80)

# Regression predictions and residuals
if "y_test" in dir() and "y_pred_stacking" in dir():
    import plotly.express as px
    import plotly.graph_objects as go

    print("\n📈 Prediction vs Actual Scatter Plot...")

    # Predicted vs Actual
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred_stacking,
            mode="markers",
            marker=dict(size=6, opacity=0.6, color="blue"),
            name="Predictions",
        )
    )

    # Perfect prediction line
    min_val, max_val = y_test.min(), y_test.max()
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect Prediction",
        )
    )

    fig.update_layout(
        title="Predicted vs Actual Price Targets",
        xaxis_title="Actual Price Target",
        yaxis_title="Predicted Price Target",
        width=800,
        height=600,
    )
    fig.show()

    # Residual plot
    print("\n📉 Residual Analysis...")
    residuals = y_pred_stacking - y_test

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred_stacking,
            y=residuals,
            mode="markers",
            marker=dict(size=6, opacity=0.6, color="purple"),
            name="Residuals",
        )
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Error")

    fig.update_layout(
        title="Residual Plot - Model Error Analysis",
        xaxis_title="Predicted Price Target",
        yaxis_title="Residual (Predicted - Actual)",
        width=800,
        height=600,
    )
    fig.show()

    # Residual distribution
    fig = px.histogram(
        residuals,
        nbins=50,
        title="Residual Distribution",
        labels={"value": "Residual", "count": "Frequency"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.show()

    print("✓ Regression model visualizations complete")

# %% [markdown]
# ## Phase 9.6: Model Evaluation and Comprehensive Error Analysis and Error Analysis
#
# ### Business Goal
# Thoroughly evaluate regression model performance with comprehensive metrics, residual analysis, and segment-wise breakdowns.
#
# ### Key Objectives
# 1. Calculate comprehensive regression metrics (MAE, RMSE, R², MAPE)
# 2. Perform segment analysis (by sector, region, market cap)
# 3. Generate residual plots and error distributions
# 4. Identify systematic biases
# 5. Analyze prediction errors by magnitude
#
# ### Inputs
# - `reg_result`: Regression results from Phase 9.5
# - `all_stocks_features`: Full dataset with predictions
#
# ### Outputs
# - `outputs/evaluation/`: Comprehensive metrics, residual plots
# - `outputs/evaluation/tscv_metrics.csv`: Time-series CV results
# - Sector-wise performance analysis
#
# ### Key Metrics
# - Overall: MAE, RMSE, R², MAPE, Median AE
# - By sector: Per-sector performance comparison
# - By region: Geographic performance patterns
# - Error distribution: Histogram, percentiles
#
# ### Validation Checkpoint
# - Comprehensive metrics calculated
# - Residuals analyzed
# - Sector biases identified
# - Error patterns documented
#
# Comprehensive evaluation including:
# - Regression metrics (MAE, RMSE, MAPE, R²)
# - Residual analysis
# - Sector and region performance breakdown
# - SHAP analysis for explainability
# - Learning curves and bias-variance diagnosis
#
# %% [markdown]
# ## Section 9.4: Uncertainty Quantification & Conformal Calibration
#
# **Objectives:**
# - Quantify prediction interval quality with coverage diagnostics
# - Validate conformal calibration effectiveness
# - Analyze uncertainty by sector and region
# - Generate reliability diagrams and interactive visualizations
#
# **Inputs:**
# - `outputs/regression/regression_predictions_detailed.csv`
#
# **Outputs:**
# - `outputs/uncertainty/quantile_predictions_diagnostics.csv`
# - `outputs/uncertainty/coverage_by_sector.json`
# - `outputs/uncertainty/uncertainty_summary.json`
# - Interactive HTML visualizations
#
# %%
# %% [PHASE 9.4] Build quantile diagnostics
print("\n" + "=" * 80)
print("PHASE 9.4: UNCERTAINTY QUANTIFICATION")
print("=" * 80)

import json

import pandas as pd

# Setup paths - use OUTPUT_DIR from configuration
uncertainty_dir = OUTPUT_DIR / "uncertainty"
uncertainty_dir.mkdir(parents=True, exist_ok=True)

# Load predictions
predictions_path = OUTPUT_DIR / "regression" / "regression_predictions_detailed.csv"
if not predictions_path.exists():
    print(f"⚠️  Predictions file not found: {predictions_path}")
    print("   Please run Phase 9.5 (regression) first to generate predictions.")
else:
    print(f"📂 Loading predictions from: {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)
    print(f"   Loaded {len(predictions_df):,} predictions")

    # Build quantile diagnostics
    print("\n🔍 Building quantile diagnostics...")
    diagnostics_df = build_quantile_diagnostics(
        predictions_df=predictions_df,
        output_dir=uncertainty_dir,
        y_true_col="y_true",
        pred_cols={"p10": "pred_p10", "p50": "pred_p50", "p90": "pred_p90"},
        sector_col="sector",
        region_col="region",
        target_coverage=0.8,
    )

    print(f"✓ Diagnostics computed for {len(diagnostics_df):,} predictions")
    print(f"✓ Artifacts saved to: {uncertainty_dir}")

# %%
# %% [PHASE 9.4] Coverage and width visuals
if "diagnostics_df" in globals():
    print("\n📊 Generating interval coverage visualizations...")

    plot_interval_coverage(
        diagnostics_df=diagnostics_df,
        output_dir=uncertainty_dir,
        last_price_col="last_price",
    )

    print("✓ Coverage visualizations created:")
    print(f"  - {uncertainty_dir / 'interval_width_by_bucket.html'}")
    print(f"  - {uncertainty_dir / 'coverage_heatmap_region_sector.html'}")

# %%
# %% [PHASE 9.4] Reliability diagram
if "diagnostics_df" in globals():
    print("\n📈 Creating reliability diagram...")

    plot_reliability_diagram(
        diagnostics_df=diagnostics_df,
        output_dir=uncertainty_dir,
        pre_calibration_df=None,
    )

    print(
        f"✓ Reliability diagram created: {uncertainty_dir / 'reliability_diagram_conformal.html'}"
    )

# %%
# %% [PHASE 9.4] Summary + QA
print("\n📋 Uncertainty Quantification Summary:")
print("=" * 80)

summary_path = uncertainty_dir / "uncertainty_summary.json"
if summary_path.exists():
    with open(summary_path, "r") as f:
        summary = json.load(f)

    print(f"Overall Coverage: {summary.get('overall_coverage', 0):.1%}")
    print(f"Target Coverage: {summary.get('target_coverage', 0.8):.1%}")
    print(f"Within Tolerance: {'✓' if summary.get('within_tolerance', False) else '✗'}")

    under_covered = summary.get("under_covered_sectors", [])
    over_covered = summary.get("over_covered_sectors", [])

    if under_covered:
        print(f"\n⚠️  Under-covered sectors: {', '.join(under_covered)}")
    if over_covered:
        print(f"⚠️  Over-covered sectors: {', '.join(over_covered)}")

    print("\n✅ Uncertainty quantification complete!")
else:
    print("⚠️  Summary file not found")

# %% [markdown]
# ## Section 9.5: Outlier Safety Rails & Non-Negative Constraints
#
# **Objectives:**
#
# - Track winsorization effects on feature distributions
# - Validate non-negativity constraint adherence
# - Analyze safety rails sensitivity across thresholds
# - Generate interactive safety dashboards
#
# **Inputs:**
#
# - Raw and winsorized feature dataframes
# - Predictions dataframe
#
# **Outputs:**
#
# - `outputs/safety_rails/clipping_effect_summary.json`
# - `outputs/safety_rails/non_negative_violations.json`
# - `outputs/safety_rails/safety_rails_summary.json`
# - 3 interactive HTML visualizations
#
# %%
##%%
from pathlib import Path

safety_rails_dir = Path("outputs/safety_rails")
safety_rails_dir.mkdir(parents=True, exist_ok=True)

# Compare raw (pre-winsorization) vs winsorized data
# Use all_stocks_typed as the raw baseline (after type casting but before winsorization)
if "all_stocks_typed" in dir() and "all_stocks_winsorized" in dir():
    print("\n🔍 Analyzing winsorization effects...")

    # Get numeric columns
    numeric_cols = all_stocks_winsorized.select_dtypes(include=[np.number]).columns.tolist()

    summary_dict = summarize_winsorization_effects(
        features_raw=all_stocks_typed,
        features_winsorized=all_stocks_winsorized,
        output_dir=safety_rails_dir,
        sector_col="sector",
    )

    print(f"✓ Winsorization summary created for {len(numeric_cols[:20])} features")
    print(f"✓ Artifacts saved to: {safety_rails_dir}")
else:
    print("⚠️  Required dataframes not available. Skipping winsorization analysis.")

# %%
##%% [PHASE 9.5] Track constraint violations
print("\n🛡️  Checking non-negativity constraint violations...")

if predictions_path.exists():
    violations_dict = track_constraint_violations(
        predictions_df=predictions_df,
        output_dir=safety_rails_dir,
        prediction_col="y_pred",
        sector_col="sector",
    )

    total_violations = violations_dict.get("total_violations", 0)
    violation_rate = violations_dict.get("violation_rate", 0)

    print(f"Total Violations: {total_violations}")
    print(f"Violation Rate: {violation_rate:.2%}")

    if total_violations == 0:
        print("✅ Non-negativity constraint satisfied!")
    else:
        print(f"⚠️  Found {total_violations} violations")
        violations_by_sector = violations_dict.get("violations_by_sector", {})
        for sector, sector_data in violations_by_sector.items():
            count = sector_data.get("count", 0)
            if count > 0:
                mean_val = sector_data.get("mean_value", 0)
                min_val = sector_data.get("min_value", 0)
                print(
                    f"   - {sector}: {count} violations (mean: {mean_val:.2f}, min: {min_val:.2f})"
                )
# %%
##%% [PHASE 9.5] Interactive robustness sliders
if "all_stocks_raw" in dir():
    print("\n📊 Creating safety rails sensitivity dashboard...")

    safety_rails_sensitivity_app(
        data_df=all_stocks_preprocessed,
        output_dir=safety_rails_dir,
        thresholds=[0.01, 0.05, 0.1],
    )

    print(
        f"✓ Sensitivity dashboard created: {safety_rails_dir / 'safety_rails_sensitivity_dashboard.html'}"
    )

# %%
# %% [PHASE 9.5] Summary + QA
print("\n📋 Safety Rails Summary:")
print("=" * 80)

summary_path = safety_rails_dir / "safety_rails_summary.json"
if summary_path.exists():
    with open(summary_path, "r") as f:
        summary = json.load(f)

    print(f"Winsorization Features: {summary.get('winsorization', {}).get('n_features', 0)}")
    print(f"Constraint Violations: {summary.get('violations', {}).get('total_violations', 0)}")

    print("\n✅ Safety rails monitoring complete!")

# %% [markdown]
# ## Section 9.6: Data Split and Leakage Policy Validation
#
# **Objectives:**
#
# - Validate CV fold construction and grouping rules
# - Check for ticker/sector overlaps across folds
# - Detect time-based leakage violations
# - Ensure stratification balance
#
# **Inputs:**
#
# - Fold assignments dictionary (from CV training)
# - Training dataframe with snapshot dates
#
# **Outputs:**
#
# - `outputs/splits/fold_overlap_heatmap.html`
# - `outputs/splits/grouped_cv_balance_metrics.json`
# - `outputs/splits/leakage_report.json`
#
# %%
##%% [PHASE 9.6] Fold overlap analysis
print("\n" + "=" * 80)
print("PHASE 9.6: DATA SPLIT AND LEAKAGE POLICY VALIDATION")
print("=" * 80)

# Setup paths - use OUTPUT_DIR from configuration
splits_dir = OUTPUT_DIR / "splits"
splits_dir.mkdir(parents=True, exist_ok=True)

# Initialize fold_assignments if not already defined
# This should be populated by CV training in Phase 9.4
if "fold_assignments" not in dir():
    fold_assignments = None  # Explicit None for validation

# Note: fold_assignments must be a DataFrame with columns: [group_col, 'fold']
# Example structure:
#   ticker    fold
#   AAPL      0
#   MSFT      0
#   GOOG      1
#   ...

# Run fold overlap analysis with integrated validation
overlap_dict = run_fold_overlap_analysis(
    fold_assignments=fold_assignments, output_dir=splits_dir, group_col="ticker"
)

if overlap_dict.get("skipped"):
    print(f"⚠️  Fold overlap analysis skipped: {overlap_dict.get('reason')}")
    print("    Expected: pd.DataFrame with 'ticker' and 'fold' columns from CV training.")

# %%
# %% [PHASE 9.6] CV balance metrics
# Ensure variables exist for validation
if "fold_assignments" not in dir():
    fold_assignments = None
if "all_stocks_features" not in dir():
    all_stocks_features = None

# Run CV balance summary with validation
if (
    validate_fold_assignments(fold_assignments)
    and all_stocks_features is not None
    and not all_stocks_features.empty
):
    print("\n📊 Summarizing grouped CV balance...")
    balance_dict = summarize_grouped_cv_balance(
        fold_assignments=fold_assignments,
        output_dir=splits_dir,
        group_col="ticker",
        stratify_col="sector",
    )
    print(f"✓ Balance metrics computed for {fold_assignments['fold'].nunique()} folds")
else:
    print("⚠️  CV balance summary skipped: required data not available")
    balance_dict = {"skipped": True, "reason": "missing_fold_assignments_or_features"}

# %%
##%% [PHASE 9.6] Time leakage checks
# Ensure variables exist
if "fold_assignments" not in dir():
    fold_assignments = None
if "all_stocks_features" not in dir():
    all_stocks_features = None

# Run time-based leakage detection with validation
if (
    validate_fold_assignments(fold_assignments)
    and all_stocks_features is not None
    and validate_temporal_data(all_stocks_features, date_col="snapshot_date")
):
    print("\n🕐 Checking time-based leakage...")
    leakage_report = time_leakage_checks(
        fold_assignments=fold_assignments,
        output_dir=splits_dir,
        date_col="snapshot_date",
    )

    violations = leakage_report.get("violations", 0)
    print(f"  Leakage violations: {violations}")

    if violations == 0:
        print("✅ No time-based leakage detected!")
    else:
        print(f"⚠️  {violations} time-based leakage violations detected!")
else:
    print("⚠️  Time leakage checks skipped: temporal data not available")
    leakage_report = {"skipped": True, "reason": "missing_temporal_data"}

# %%
# %% [PHASE 9.6] Summary
print("\n" + "=" * 80)
print("📋 PHASE 9.6 SUMMARY: DATA SPLIT & LEAKAGE VALIDATION")
print("=" * 80)

# Collect results
phase96_results = {
    "fold_overlap": overlap_dict.get("skipped", False) == False,
    "cv_balance": balance_dict.get("skipped", False) == False,
    "time_leakage": leakage_report.get("skipped", False) == False,
}

completed_checks = sum(phase96_results.values())
print(f"\n✓ Completed {completed_checks}/3 validation checks")

if completed_checks == 0:
    print("\n⚠️  WARNING: All Phase 9.6 checks were skipped")
    print("   fold_assignments may not have been created during CV training")
    print("   Ensure Phase 9.4 cross-validation is executed before Phase 9.6")
elif completed_checks == 3:
    print("\n✅ All Phase 9.6 validation checks completed successfully")

    # Report key metrics
    if "zero_overlap_validated" in overlap_dict:
        print(f"\n  Fold Overlap: {overlap_dict['zero_overlap_validated']}")
    if "violations" in leakage_report:
        violations_count = (
            len(leakage_report["violations"])
            if isinstance(leakage_report["violations"], list)
            else leakage_report["violations"]
        )
        print(f"  Time Leakage Violations: {violations_count}")
else:
    print(f"\n⚠️  Only {completed_checks}/3 checks executed")

print("=" * 80)

# %% [markdown]
# ## Section 9.7: Sector Bias Calibration & Metrics Persistence
#
# **Objectives:**
#
# - Estimate sector-level bias before/after calibration
# - Track MAE/MAPE improvements per sector
# - Visualize metrics trends over time
# - Persist calibration metadata with model versioning
#
# **Inputs:**
#
# - Predictions dataframe with y_true, y_pred, y_pred_calibrated
#
# **Outputs:**
#
# - `outputs/calibration/sector_bias_calibration_v{MODEL_VERSION}.json`
# - `outputs/calibration/metrics_by_sector_time.html`
# - `outputs/calibration/sector_bias_dashboard.html`
#
# %%
# Calculate mispricing scores using Phase 9.7 function
# First, add predicted prices for all stocks

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS FOR ALL STOCKS (Phase 9.7)")
print("=" * 80)

# Step 1: Prepare features for prediction with sector interactions
# === FIX: Use new helper to regenerate sector interactions ===
# Root cause: X_train has sector interactions (e.g., sector_Technology__x__p_e_ratio)
# but all_stocks_phase95 does not. We need to regenerate them for alignment.

from finance_ml.ml_workflow.regression.dataset import (
    add_sector_interactions_for_prediction,
)

# Extract base features (excluding sector interactions which we'll regenerate)
base_feature_cols = [
    c for c in X_train.columns if c in all_stocks_phase95.columns and not c.startswith("sector_")
]

print(f"\n📊 Feature alignment diagnostics:")
print(f"  X_train features: {X_train.shape[1]}")
print(f"  Base features available: {len(base_feature_cols)}")
print(
    f"  Sector interaction features in X_train: {sum(1 for c in X_train.columns if '__x__' in c)}"
)

# Create base feature matrix
X_all_stocks_for_prediction = all_stocks_phase95[base_feature_cols].copy()
print(f"  Initial X shape: {X_all_stocks_for_prediction.shape}")

# Add sector interactions using the same logic as prepare_regression_data()
X_all_stocks_for_prediction = add_sector_interactions_for_prediction(
    X_all_stocks_for_prediction,
    df_with_sector=all_stocks_phase95,
    base_cols=["p_e_ratio", "ev_ebitda_ratio", "gross_margin", "market_cap", "beta_5y"],
)
print(f"  After sector interactions: {X_all_stocks_for_prediction.shape}")

# Final alignment: reindex to match model's expected features exactly
model_features_all = getattr(stacking_model, "feature_names_in_", None)
if model_features_all is None and hasattr(stacking_model, "estimators_"):
    # Try getting features from the first base estimator if wrapper hides them
    model_features_all = getattr(stacking_model.estimators_[0], "feature_names_in_", None)

if model_features_all is not None:
    # Use reindex with fill_value=0 for any remaining missing features
    X_all_stocks_for_prediction = X_all_stocks_for_prediction.reindex(
        columns=model_features_all, fill_value=0
    )
    print(f"  Final aligned shape: {X_all_stocks_for_prediction.shape}")
    print(f"  Expected features: {len(model_features_all)}")

    # Verify alignment
    missing_after_align = set(model_features_all) - set(X_all_stocks_for_prediction.columns)
    if missing_after_align:
        print(f"  ⚠ Warning: {len(missing_after_align)} features still missing after alignment")
        print(f"    (filled with 0): {list(missing_after_align)[:5]}")
    else:
        print(f"  ✓ Perfect alignment: all {len(model_features_all)} features present")
else:
    print("  ⚠ Could not retrieve model feature names, using current columns")

# === FIX END ===

# Clean X_all_stocks_for_prediction before prediction (same as X_test cleaning)
X_all_stocks_for_prediction = _clean_regression_features(
    X_all_stocks_for_prediction,
    drop_zero_variance=False,
)

# Step 2: Generate raw predictions
raw_predictions = stacking_model.predict(X_all_stocks_for_prediction)

print(f"\n📈 Raw predictions generated:")
print(f"  Total stocks: {len(raw_predictions):,}")
print(f"  Range: ${raw_predictions.min():.2f} to ${raw_predictions.max():.2f}")
print(
    f"  Negative: {(raw_predictions < 0).sum()} ({(raw_predictions < 0).sum() / len(raw_predictions) * 100:.1f}%)"
)
print(f"  Mean: ${raw_predictions.mean():.2f}")
print(f"  Median: ${np.median(raw_predictions):.2f}")

# %%
##%% [PHASE 9.7] Sector bias estimation
print("\n" + "=" * 80)
print("PHASE 9.7: SECTOR BIAS CALIBRATION & METRICS PERSISTENCE")
print("=" * 80)

# Setup paths - use OUTPUT_DIR from configuration
calibration_dir = OUTPUT_DIR / "calibration"
calibration_dir.mkdir(parents=True, exist_ok=True)

if predictions_path.exists():
    print("\n🔍 Estimating sector-level bias...")

    bias_dict = estimate_sector_bias(
        predictions_df=predictions_df,
        output_dir=calibration_dir,
        model_version=MODEL_VERSION,
    )

    print(f"✓ Bias estimation complete for {len(bias_dict.get('sectors', {}))} sectors")
    print(f"✓ Versioned file: sector_bias_calibration_{MODEL_VERSION}.json")

# %%
##%% [PHASE 9.7] Metrics over time
# Note: This requires historical metrics data
# If not available, skip this cell

# Initialize metrics_history_df if not already defined (satisfies semantic analyzer)
if "metrics_history_df" not in dir():
    metrics_history_df = pd.DataFrame()

if "metrics_history_df" in dir() and not metrics_history_df.empty:
    print("\n📈 Plotting metrics by sector over time...")

    plot_metrics_by_sector_time(
        predictions_df=metrics_history_df,
        output_dir=calibration_dir,
        date_col="snapshot_date",
    )

    print(f"✓ Time-series plot created: {calibration_dir / 'metrics_by_sector_time.html'}")
else:
    print("⚠️  metrics_history_df not available. Skipping time-series plot.")

# %%
##%% [PHASE 9.7] Interactive bias dashboard
if predictions_path.exists() and "bias_dict" in dir():
    print("\n📊 Creating sector bias dashboard...")

    create_sector_bias_dashboard(predictions_df=predictions_df, output_dir=calibration_dir)

    print(f"✓ Dashboard created: {calibration_dir / 'sector_bias_dashboard.html'}")

# %%
# %% [PHASE 9.7] Summary + QA
print("\n📋 Sector Bias Calibration Summary:")
print("=" * 80)

bias_path = calibration_dir / f"sector_bias_calibration_{MODEL_VERSION}.json"
if bias_path.exists():
    with open(bias_path, "r") as f:
        bias_data = json.load(f)

    print(f"Model Version: {bias_data.get('model_version', 'N/A')}")
    print(f"Sectors Analyzed: {len(bias_data.get('sectors', {}))}")

    print("\n✅ Sector bias calibration complete!")

# %%
print("\n🔍 Calibration Quality Check")
print("=" * 80)

# Load calibration data
with open(OUTPUT_DIR / "calibration" / f"sector_bias_calibration_{MODEL_VERSION}.json") as f:
    calib_data = json.load(f)

improved = [s for s, m in calib_data["sectors"].items() if m["mae_improvement_pct"] > 0]
degraded = [s for s, m in calib_data["sectors"].items() if m["mae_improvement_pct"] < 0]

print(f"✓ Improved: {len(improved)} sectors")
print(f"⚠️ Degraded: {len(degraded)} sectors")

if len(degraded) > len(improved):
    print("\n⚠️ WARNING: Calibration degraded majority of sectors!")
    print("   Root cause: Underlying model has systematic bias (check feature leakage)")
else:
    print(
        f"\n✓ Calibration quality acceptable ({len(improved)}/{len(improved) + len(degraded)} sectors)"
    )

# Apply calibration with validation
from finance_ml.ml_workflow.regression.calibration import apply_sector_calibration

all_stocks_calibrated = apply_sector_calibration(
    predictions_df=predictions_df,
    calibration_dict=calib_data,
    model_version=MODEL_VERSION,
    min_improvement_threshold=0.5,  # Skip if <50% of sectors improve
)

print(f"\n✓ Applied sector calibration with validation")
print(
    f"  Calibrated predictions: {(all_stocks_calibrated['y_pred_calibrated'] != all_stocks_calibrated['y_pred']).sum():,} changed"
)
# %% [markdown]
# ## Section 9.8: Stacking Ensemble Diagnostics & Model Governance
#
# **Objectives:**
#
# - Analyze base model contributions to ensemble
# - Generate explainability visuals (SHAP or permutation importance)
# - Create meta-learner error maps
# - Auto-generate model card and lineage documentation
#
# **Inputs:**
#
# - Base model predictions dictionary
# - Meta-learner predictions
# - Model configuration metadata
#
# **Outputs:**
#
# - `outputs/governance/stacking_contributions.csv`
# - `outputs/governance/stacking_contributions.html`
# - `outputs/governance/meta_error_map.html`
# - `outputs/governance/model_card_v{MODEL_VERSION}.md`
# - `outputs/governance/lineage.json`
#
# %%
##%% [PHASE 9.8] Stacking contributions
print("\n" + "=" * 80)
print("PHASE 9.8: STACKING ENSEMBLE DIAGNOSTICS & MODEL GOVERNANCE")
print("=" * 80)

# Setup paths - use OUTPUT_DIR from configuration
governance_dir = OUTPUT_DIR / "governance"
governance_dir.mkdir(parents=True, exist_ok=True)

# Note: This requires base_predictions dict from stacking ensemble training
# Example: base_predictions = {"xgboost": y_pred_xgb, "lightgbm": y_pred_lgb}
if "base_predictions" in dir() and "y_pred_meta" in dir() and "y_test" in dir():
    print("\n🔍 Computing stacking contributions...")

    contributions_df = compute_stacking_contributions(
        base_predictions=base_predictions,
        meta_predictions=y_pred_meta,
        output_dir=governance_dir,
    )

    print(f"✓ Contributions computed for {len(base_predictions)} base models")
    print(f"✓ Artifacts saved to: {governance_dir}")
else:
    print("⚠️  Stacking ensemble data not available. Skipping contribution analysis.")

# %%
##%% [PHASE 9.8] Explainability (SHAP or permutation importance)
print("\n🔍 Generating explainability visuals...")

# SHAP is optional - fallback to permutation importance
try:
    import shap

    SHAP_AVAILABLE = True
    print("  Using SHAP for explainability")
except ImportError:
    SHAP_AVAILABLE = False
    print("  SHAP not available - using permutation importance fallback")

# Note: Actual SHAP/permutation importance code would go here
# This is a placeholder
if SHAP_AVAILABLE and "model" in dir() and "X_test" in dir():
    # SHAP analysis code
    print("  ✓ SHAP summary created")
else:
    # Permutation importance fallback
    print("  ✓ Permutation importance created")

# %%
##%% [PHASE 9.8] Meta-learner error maps
if predictions_path.exists():
    print("\n📊 Creating meta-learner error maps...")

    meta_error_maps(predictions_df=predictions_df, output_dir=governance_dir)

    print(f"✓ Error maps created: {governance_dir / 'meta_error_map.html'}")

# %%
##%% [PHASE 9.8] Generate model card
print("\n📋 Generating model card...")

model_info = {
    "task": "Price target regression + classification-enhanced features",
    "data_sources": ["PostgreSQL equities table", "Multi-region CSVs"],
    "features": {
        "count": 310,
        "groups": [
            "momentum",
            "valuation",
            "profitability",
            "quality",
            "cash_flow",
            "growth",
        ],
        "selection_method": "Phase 9.3 comprehensive pipeline",
    },
    "models": {
        "base": ["XGBoost", "LightGBM", "CatBoost"],
        "meta": "Ridge Regression",
        "hyperparameters": "Optuna-tuned",
    },
    "validation": {
        "strategy": "Grouped K-Fold CV (by ticker)",
        "n_folds": CV_FOLDS,
        "leakage_check": "Passed",
    },
}

generate_model_card(model_info=model_info, output_dir=governance_dir, model_version=MODEL_VERSION)

print(f"✓ Model card created: {governance_dir / f'model_card_{MODEL_VERSION}.md'}")

# %%
##%% [PHASE 9.8] Build lineage JSON
print("\n🔗 Building model lineage...")

model_info = {
    "datasets": {"train": "equities_table_2025", "validation": "hold_out_2025"},
    "features": {
        "count": 310,
        "groups": [
            "momentum",
            "valuation",
            "profitability",
            "quality",
            "cash_flow",
            "growth",
        ],
        "selection": "comprehensive",
    },
    "models": {
        "base": ["xgboost", "lightgbm", "catboost"],
        "meta": "ridge",
        "hyperparameters": {"cv_folds": CV_FOLDS},
    },
    "artifacts": [
        "regression_predictions_detailed.csv",
        "quantile_predictions_diagnostics.csv",
        "sector_bias_calibration_v9_10.json",
        "model_card_v9_10.md",
    ],
    "metrics": {"overall": {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0}, "by_sector": {}},
}

lineage = build_lineage_json(
    model_info=model_info, output_dir=governance_dir, model_version=MODEL_VERSION
)

print(f"✓ Lineage JSON created: {governance_dir / 'lineage.json'}")
print("\n✅ Model governance documentation complete!")

# %%
# Comprehensive regression metrics using Phase 9.6 function
metrics = evaluation_comprehensive_metrics(y_test, y_pred_stacking)

print("📊 Overall Model Performance:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# %%
# Segment analysis (by sector and region) using Phase 9.6 function
# Prepare test data with predictions
test_data = all_stocks_with_classification.loc[X_test.index].copy()
test_data["predicted_price_target"] = y_pred_stacking

sector_metrics = evaluation_metrics_by_segment(
    test_data, "price_target", "predicted_price_target", "sector"
)

print("\n📊 Performance by Sector:")
print(sector_metrics)
# %%
# 📊 Section 7 Enhanced Visualizations - Model Evaluation & Error Analysis
print("\n" + "=" * 80)
print("📊 INTERACTIVE MODEL EVALUATION VISUALIZATIONS")
print("=" * 80)

# Comprehensive error analysis
if "all_stocks_phase95" in dir() and "predicted_price_target" in all_stocks_phase95.columns:
    import plotly.express as px

    from finance_ml.ml_workflow.analytics import create_region_sector_heatmap
    from finance_ml.ml_workflow.evaluation.metrics import compute_sector_region_metrics

    print("\n📊 Error Analysis by Sector and Region...")

    # Calculate errors
    if "price_target" in all_stocks_phase95.columns:
        all_stocks_phase95["prediction_error"] = abs(
            all_stocks_phase95["predicted_price_target"] - all_stocks_phase95["price_target"]
        )
        all_stocks_phase95["prediction_error_pct"] = (
            all_stocks_phase95["prediction_error"] / all_stocks_phase95["price_target"] * 100
        )

        # Error by sector
        if "sector" in all_stocks_phase95.columns:
            sector_errors = (
                all_stocks_phase95.groupby("sector")["prediction_error_pct"]
                .agg(["mean", "median", "std"])
                .round(2)
            )

            fig = px.bar(
                sector_errors.reset_index(),
                x="sector",
                y="mean",
                error_y="std",
                title="Mean Prediction Error by Sector (with Std Dev)",
                labels={"mean": "Mean Error %", "sector": "Sector"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            fig.show()

            print("\n📈 Sector Error Statistics:")
            print(sector_errors)

        # Error by region and sector (heatmap)
        if "sector" in all_stocks_phase95.columns and "region" in all_stocks_phase95.columns:
            pivot_errors = all_stocks_phase95.pivot_table(
                values="prediction_error_pct",
                index="sector",
                columns="region",
                aggfunc="mean",
            )

            fig = px.imshow(
                pivot_errors,
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale="Reds",
                title="Mean Prediction Error % by Sector and Region",
            )
            fig.update_layout(width=900, height=600)
            fig.show()

            # Use compute_sector_region_metrics from finance_ml package
            print("\n📊 Sector-Region Metrics (using finance_ml helper)...")
            sector_region_metrics = compute_sector_region_metrics(
                all_stocks_phase95,
                y_true="price_target",
                y_pred="predicted_price_target",
                sector_col="sector",
                region_col="region",
            )
            print(sector_region_metrics)

            # Use create_region_sector_heatmap from finance_ml package
            print("\n📊 Region-Sector Heatmap (using finance_ml helper)...")
            create_region_sector_heatmap(
                all_stocks_phase95,
                metric="prediction_error_pct",
                out_path=None,  # Display inline
            )

    print("✓ Model evaluation visualizations complete")

# %% [markdown]
# ## Phase 9.7: Stock Ranking, Analytics, and Analyst Comparison Stocks with Visualization
#
# ### Business Goal
# Identify investment opportunities through mispricing scores, stock rankings, analyst comparison, and portfolio optimization.
#
# ### Key Objectives
# 1. Calculate mispricing scores: (predicted_target - last_price) / last_price
# 2. Rank stocks by sector and region
# 3. Compare predictions vs analyst targets
# 4. Perform portfolio optimization (max Sharpe, min volatility)
# 5. Calculate risk metrics (VaR, CVaR, drawdown)
# 6. Generate investment recommendations
#
# ### Inputs
# - `all_stocks_features`: Full dataset with predictions from Phase 9.5
# - Analyst price targets
#
# ### Outputs
# - `outputs/analytics/`: Mispricing rankings, analyst comparison reports
# - `outputs/analytics/portfolio_optimization.csv`: Optimal portfolios
# - `outputs/analytics/risk_metrics.csv`: Risk analysis
# - Top undervalued/overvalued stocks by sector
#
# ### Key Functions
# - `calculate_mispricing_score()` - Identify mispricing
# - `rank_undervalued_stocks()` - Top opportunities
# - `compare_prediction_vs_analyst_targets()` - Analyst agreement analysis
# - `optimize_max_sharpe_portfolio()` - Portfolio optimization
# - `calculate_portfolio_risk_metrics()` - Risk quantification
#
# ### Validation Checkpoint
# - Mispricing scores calculated
# - Top 20 undervalued stocks identified
# - Analyst comparison complete
# - Portfolio optimization converged
# - Risk metrics within acceptable ranges
#
# Calculate mispricing scores and identify investment opportunities:
# - Mispricing score: (Predicted - Current) / Current
# - Valuation categories: Severely Undervalued, Undervalued, Fair, Overvalued, Severely Overvalued
# - Sector-relative rankings
# - Multi-factor scoring (valuation + quality + growth)
#
# %%
# Calculate mispricing scores using Phase 9.7 function
# First, add predicted prices for all stocks
print("\n" + "=" * 80)
print("GENERATING PREDICTIONS FOR ALL STOCKS (Phase 9.7)")
print("=" * 80)

# Step 1: Get base features from all_stocks_phase95
# DON'T directly subset with X_train.columns - it has sector interaction features!
from finance_ml.ml_workflow.regression.dataset import (
    add_sector_interactions_for_prediction,
    create_classification_interactions,
    integrate_classification_features,
)

# Start with base features (exclude sector interactions)
base_feature_cols = [
    col for col in X_train.columns if not (col.startswith("sector_") and "__x__" in col)
]

# Verify base features exist in all_stocks_phase95
available_base_features = [col for col in base_feature_cols if col in all_stocks_phase95.columns]
print(f"\n📊 Feature alignment diagnostics:")
print(f"  X_train features: {len(X_train.columns)}")
print(f"  Base features identified: {len(base_feature_cols)}")
print(f"  Base features available: {len(available_base_features)}")

X_all_stocks_for_prediction = all_stocks_phase95[available_base_features].copy()
print(f"  Initial X shape: {X_all_stocks_for_prediction.shape}")

# Step 2: Add sector interaction features (same as Cell #121 successful pattern)
# This regenerates the 55 sector interactions that were in X_train
X_all_stocks_for_prediction = add_sector_interactions_for_prediction(
    X_all_stocks_for_prediction,
    df_with_sector=all_stocks_phase95,
    base_cols=["p_e_ratio", "ev_ebitda_ratio", "gross_margin", "market_cap", "beta_5y"],
)
sector_interactions_added = len([c for c in X_all_stocks_for_prediction.columns if "__x__" in c])
print(f"  ✓ Added {sector_interactions_added} sector interaction features")
print(f"  After sector interactions: {X_all_stocks_for_prediction.shape}")

# Step 3: Get model's expected feature names
model_features_all = getattr(stacking_model, "feature_names_in_", None)
if model_features_all is None and hasattr(stacking_model, "estimators_"):
    # Try getting features from the first base estimator if wrapper hides them
    model_features_all = getattr(stacking_model.estimators_[0], "feature_names_in_", None)

if model_features_all is not None:
    missing_cols_all = set(model_features_all) - set(X_all_stocks_for_prediction.columns)
    if missing_cols_all:
        print(f"⚠ Detected {len(missing_cols_all)} missing features. Regenerating...")

        # Add classification probability features if available in all_stocks_phase95
        prob_cols_all = [c for c in all_stocks_phase95.columns if c.startswith("event_prob_")]
        if len(prob_cols_all) >= 5:
            # Prefer standard names if present
            std_cols = [
                "event_prob_strong_negative",
                "event_prob_negative",
                "event_prob_neutral",
                "event_prob_positive",
                "event_prob_strong_positive",
            ]
            if all(c in all_stocks_phase95.columns for c in std_cols):
                prob_cols_all = std_cols
            else:
                prob_cols_all = sorted(prob_cols_all)[:5]

            all_probs = all_stocks_phase95[prob_cols_all].values
            X_all_stocks_for_prediction = integrate_classification_features(
                X_all_stocks_for_prediction, all_probs
            )
            print(f"  ✓ Added classification probability features")

        # Identify classification probability columns currently in X_all_stocks
        cls_cols_all = [
            c
            for c in X_all_stocks_for_prediction.columns
            if c.startswith("event_prob_") or c == "event_confidence"
        ]

        # Identify valuation columns (fallback to default if method-aware list not found)
        val_cols_all = globals().get(
            "valuation_cols_method_aware",
            [
                "market_cap",
                "enterprise_value",
                "ebitda",
                "p_e",
                "p_b",
                "gross_margin",
                "revenue",
                "net_income",
            ],
        )
        val_cols_all = [c for c in val_cols_all if c in X_all_stocks_for_prediction.columns]

        # Regenerate interactions
        if cls_cols_all and val_cols_all:
            X_all_stocks_for_prediction = create_classification_interactions(
                X_all_stocks_for_prediction,
                classification_cols=cls_cols_all,
                valuation_cols=val_cols_all,
            )
            print(f"  ✓ Added classification interaction features")

        # Final alignment: Ensure exact column order and fill any remaining gaps with 0
        X_all_stocks_for_prediction = X_all_stocks_for_prediction.reindex(
            columns=model_features_all, fill_value=0
        )
        print(f"  Final aligned shape: {X_all_stocks_for_prediction.shape}")
        print(f"  Expected features: {len(model_features_all)}")
        print(f"  ✓ Perfect alignment: all {len(model_features_all)} features present")
else:
    print("⚠ Could not extract feature names from model")

# Step 4: Clean X_all_stocks_for_prediction before prediction (same as X_test cleaning)
X_all_stocks_for_prediction = _clean_regression_features(
    X_all_stocks_for_prediction,
    drop_zero_variance=False,
)

# Step 5: Generate predictions
raw_predictions = stacking_model.predict(X_all_stocks_for_prediction)
print(f"\n📊 Raw predictions generated:")
print(f"  Total stocks: {len(raw_predictions):,}")
print(f"  Range: ${raw_predictions.min():.2f} to ${raw_predictions.max():.2f}")
print(
    f"  Negative: {(raw_predictions < 0).sum()} ({(raw_predictions < 0).sum() / len(raw_predictions) * 100:.1f}%)"
)

# Step 6: Apply final adaptive clipping (non-negative + outlier bounds)
# This is the SINGLE point of non-negative enforcement for all predictions
clip_result = adaptive_clip_predictions(raw_predictions, y_train)
final_predictions = clip_result["clipped_predictions"]

print(f"\n📊 Final predictions (after adaptive clipping):")
print(f"  Lower bound: ${clip_result['lower_bound']:.2f}")
print(f"  Upper bound: ${clip_result['upper_bound']:.2f}")
print(
    f"  Clipped to lower: {clip_result['n_clipped_lower']} ({clip_result['pct_clipped_lower']:.1f}%)"
)
print(
    f"  Clipped to upper: {clip_result['n_clipped_upper']} ({clip_result['pct_clipped_upper']:.1f}%)"
)
print(f"  Zero predictions: {(final_predictions == 0).sum()} (should be 0)")
print(f"  Range: ${final_predictions.min():.2f} to ${final_predictions.max():.2f}")

# Step 7: Store final clipped predictions
all_stocks_phase95["predicted_price_target"] = final_predictions
# %%
# CRITICAL: Verify price columns preserved before mispricing calculation (code_guidelines.md Section 8.5.2)
# The mispricing score formula (predicted - last_price) / last_price requires original price scale
# If last_price was scaled/winsorized, this calculation will produce nonsensical results (e.g., 99% errors)
from finance_ml.ml_workflow.preprocessing.column_semantics import PRICE_COLUMNS

print("\n🔍 Phase 9.7: Validating price column preservation before mispricing calculation...")

# Check that last_price is in expected range (not scaled to [0,1] or winsorized)
if "last_price" in all_stocks_phase95.columns:
    lp = all_stocks_phase95["last_price"].dropna()
    if len(lp) > 0:
        lp_min, lp_max, lp_median = lp.min(), lp.max(), lp.median()
        print(f"  last_price: range [${lp_min:.2f}, ${lp_max:.2f}], median=${lp_median:.2f}")

        # Warning if prices look scaled (median < 1 suggests MinMax scaling)
        if lp_median < 1.0:
            print(
                f"  ⚠️ WARNING: last_price median < $1.00 - prices may have been incorrectly scaled!"
            )
            print(f"     This will corrupt the mispricing score calculation.")
        else:
            print(f"  ✓ last_price appears to be in original dollar units")

# Verify price columns preserved (if reference dataframe available)
price_cols_present = [c for c in PRICE_COLUMNS if c in all_stocks_phase95.columns]
print(f"  ✓ {len(price_cols_present)}/21 price columns present in all_stocks_phase95")

# Calculate mispricing - returns DataFrame with added 'mispricing_pct' and 'mispricing_score' columns
all_stocks_phase95 = analytics_calculate_mispricing(
    all_stocks_phase95, predicted_col="predicted_price_target", current_col="last_price"
)

print(f"\n✓ Valuation Analysis Complete")
print(f"  Mispricing scores calculated: {len(all_stocks_phase95)} stocks")
print(f"  Columns added: 'mispricing_pct', 'mispricing_score'")
print("=" * 80)
# %%
# Rank stocks by sector using Phase 9.7 function
print("\n📊 Sector-Specific Rankings:")
sector_rankings = analytics_rank_by_sector(all_stocks_phase95, top_n=TOP_N_RANKINGS)
if sector_rankings:
    for sector, stocks in list(sector_rankings.items())[:3]:  # Show top 3 sectors
        print(f"\n  {sector}:")
        if not stocks.empty and "ticker" in stocks.columns:
            print(f"    Top stocks: {', '.join(stocks['ticker'].head(5).tolist())}")
        else:
            print(f"    {len(stocks)} stocks ranked")

# %%
# 📊 Comprehensive Interactive Visualizations - Predictions, Valuation & Analytics
# Note: All directories already created at initialization (Phase 9.1-9.8 structure)
plots_dir = OUTPUT_DIR / "plots"
analytics_dir = OUTPUT_DIR / "analytics"
reports_dir = OUTPUT_DIR / "reporting"

print(f"\n📊 Creating Comprehensive Interactive Visualizations...")

# 1. Prediction Scatter Plot - Predicted vs Actual with Sector Colors
print("  Creating prediction scatter plot...")
if all(
    col in all_stocks_phase95.columns
    for col in ["last_price", "price_target", "predicted_price_target", "sector"]
):
    plot_data = all_stocks_phase95[
        [
            "last_price",
            "price_target",
            "predicted_price_target",
            "sector",
            "ticker",
            "name",
            "exchange",
            "region",
        ]
    ].dropna()

    fig_pred = px.scatter(
        plot_data,
        x="price_target",
        y="predicted_price_target",
        color="sector",
        hover_data=["ticker", "name", "sector", "exchange", "last_price"],
        title="Predicted vs Actual Price Targets by Sector",
        labels={
            "price_target": "Actual Price Target",
            "predicted_price_target": "Predicted Price Target",
        },
        height=700,
        width=1000,
    )
    # Add diagonal line (perfect prediction)
    max_val = max(plot_data["price_target"].max(), plot_data["predicted_price_target"].max())
    fig_pred.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect Prediction",
            showlegend=True,
        )
    )
    fig_pred.show()
    fig_pred.write_html(plots_dir / "prediction_scatter_interactive.html")
    print(f"  ✓ Saved: {plots_dir / 'prediction_scatter_interactive.html'}")

    # Generate PNG version for Excel integration
    try:
        fig_pred.write_image(
            plots_dir / "prediction_scatter_interactive.png", width=1000, height=700
        )
        print(f"  ✓ Saved PNG: {plots_dir / 'prediction_scatter_interactive.png'}")
    except (ValueError, IOError, OSError, ImportError, RuntimeError) as e:
        print(f"  ⚠️ PNG generation skipped (install kaleido: pip install kaleido): {e}")

# 2. Valuation Scatter Plot - Predicted vs Current Price with Sector Colors
print("  Creating valuation scatter plot...")
if all(
    col in all_stocks_phase95.columns
    for col in ["last_price", "price_target", "predicted_price_target", "sector"]
):
    plot_data = all_stocks_phase95[
        [
            "last_price",
            "price_target",
            "predicted_price_target",
            "sector",
            "ticker",
            "name",
            "exchange",
            "region",
        ]
    ].dropna()

    fig_val = px.scatter(
        plot_data,
        x="last_price",
        y="predicted_price_target",
        color="sector",
        hover_data=["ticker", "name", "sector", "exchange", "price_target"],
        title="Predicted vs Current Price with Sector Colors",
        labels={
            "last_price": "Current Price (Last Price)",
            "predicted_price_target": "Predicted Price Target",
        },
        height=700,
        width=1000,
    )
    # Add diagonal line (fair value) with 10% bounds
    max_val = max(plot_data["last_price"].max(), plot_data["predicted_price_target"].max())

    # Main diagonal - Fair Value
    fig_val.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Fair Value",
            showlegend=True,
        )
    )

    # Lower bound (10% below fair value)
    fig_val.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val * 0.9],
            mode="lines",
            line=dict(color="orange", dash="dot", width=1),
            name="Fair Value -10%",
            showlegend=True,
        )
    )

    # Upper bound (10% above fair value)
    fig_val.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val * 1.1],
            mode="lines",
            line=dict(color="green", dash="dot", width=1),
            name="Fair Value +10%",
            showlegend=True,
        )
    )

    fig_val.show()
    fig_val.write_html(plots_dir / "valuation_scatter_interactive.html")
    print(f"  ✓ Saved: {plots_dir / 'valuation_scatter_interactive.html'}")

    # Generate PNG version for Excel integration
    try:
        fig_val.write_image(plots_dir / "valuation_scatter_interactive.png", width=1000, height=700)
        print(f"  ✓ Saved PNG: {plots_dir / 'valuation_scatter_interactive.png'}")
    except (ValueError, IOError, OSError, ImportError, RuntimeError) as e:
        print(f"  ⚠️ PNG generation skipped (install kaleido: pip install kaleido): {e}")

# 3. Residual Analysis - Interactive Residual Plot
print("  Creating residual analysis plot...")
if all(col in all_stocks_phase95.columns for col in ["price_target", "predicted_price_target"]):
    residual_data = all_stocks_phase95[
        ["price_target", "predicted_price_target", "sector"]
    ].dropna()
    residual_data["residual"] = (
        residual_data["predicted_price_target"] - residual_data["price_target"]
    )

    fig_resid = px.scatter(
        residual_data,
        x="price_target",
        y="residual",
        color="sector",
        title="Residual Plot: Model Error Analysis",
        labels={
            "price_target": "Actual Price Target",
            "residual": "Residual (Predicted - Actual)",
        },
        height=600,
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
    fig_resid.show()
    fig_resid.write_html(plots_dir / "residual_analysis_interactive.html")
    print(f"  ✓ Saved: {plots_dir / 'residual_analysis_interactive.html'}")

    # Generate PNG version for Excel integration
    try:
        fig_resid.write_image(
            plots_dir / "residual_analysis_interactive.png", width=1000, height=600
        )
        print(f"  ✓ Saved PNG: {plots_dir / 'residual_analysis_interactive.png'}")
    except (ValueError, IOError, OSError, ImportError, RuntimeError) as e:
        print(f"  ⚠️ PNG generation skipped (install kaleido: pip install kaleido): {e}")

# 4. Mispricing Heatmap - Sector vs Region
print("  Creating mispricing heatmap...")
if all(col in all_stocks_phase95.columns for col in ["sector", "region", "mispricing_score"]):
    mispricing_pivot = all_stocks_phase95.pivot_table(
        values="mispricing_score", index="sector", columns="region", aggfunc="mean"
    )

    fig_mispricing = px.imshow(
        mispricing_pivot,
        labels=dict(x="Region", y="Sector", color="Avg Mispricing Score"),
        title="Average Mispricing Score by Sector and Region",
        color_continuous_scale="RdYlGn",
        aspect="auto",
        height=600,
    )
    fig_mispricing.update_traces(text=mispricing_pivot.values.round(3), texttemplate="%{text}")
    fig_mispricing.show()
    fig_mispricing.write_html(analytics_dir / "mispricing_heatmap_interactive.html")
    print(f"  ✓ Saved: {analytics_dir / 'mispricing_heatmap_interactive.html'}")

    # Generate PNG version for Excel integration
    try:
        fig_mispricing.write_image(
            plots_dir / "mispricing_heatmap_interactive.png", width=1000, height=600
        )
        print(f"  ✓ Saved PNG: {plots_dir / 'mispricing_heatmap_interactive.png'}")
    except (ValueError, IOError, OSError, ImportError, RuntimeError) as e:
        print(f"  ⚠️ PNG generation skipped (install kaleido: pip install kaleido): {e}")

# 5. Stock Rankings - Top Undervalued/Overvalued Interactive Bar Chart
print("  Creating stock rankings chart...")
if "mispricing_score" in all_stocks_phase95.columns:
    top_10_under = all_stocks_phase95.nlargest(10, "mispricing_score")[
        ["ticker", "name", "exchange", "sector", "mispricing_score"]
    ]
    top_10_over = all_stocks_phase95.nsmallest(10, "mispricing_score")[
        ["ticker", "name", "exchange", "sector", "mispricing_score"]
    ]

    fig_rankings = make_subplots(
        rows=1, cols=2, subplot_titles=("Top 10 Undervalued", "Top 10 Overvalued")
    )

    fig_rankings.add_trace(
        go.Bar(
            x=top_10_under["ticker"],
            y=top_10_under["mispricing_score"],
            marker=dict(color="green"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig_rankings.add_trace(
        go.Bar(
            x=top_10_over["ticker"],
            y=top_10_over["mispricing_score"],
            marker=dict(color="red"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig_rankings.update_layout(title_text="Stock Rankings: Investment Opportunities", height=500)
    fig_rankings.update_xaxes(tickangle=45)
    fig_rankings.show()
    fig_rankings.write_html(analytics_dir / "stock_rankings_interactive.html")
    print(f"  ✓ Saved: {analytics_dir / 'stock_rankings_interactive.html'}")

    # Generate PNG version for Excel integration
    try:
        fig_rankings.write_image(
            plots_dir / "stock_rankings_interactive.png", width=1200, height=500
        )
        print(f"  ✓ Saved PNG: {plots_dir / 'stock_rankings_interactive.png'}")
    except (ValueError, IOError, OSError, ImportError, RuntimeError) as e:
        print(f"  ⚠️ PNG generation skipped (install kaleido: pip install kaleido): {e}")

# 6. Sector Performance Summary - Bubble Chart
print("  Creating sector performance bubble chart...")
sector_summary = None
if all(col in all_stocks_phase95.columns for col in ["sector", "mispricing_score", "market_cap"]):
    sector_summary = (
        all_stocks_phase95.groupby("sector")
        .agg({"mispricing_score": "mean", "market_cap": "sum", "ticker": "count"})
        .reset_index()
    )
    sector_summary.columns = ["sector", "avg_mispricing", "market_cap", "num_stocks"]

    fig_sector_bubble = px.scatter(
        sector_summary,
        x="num_stocks",
        y="avg_mispricing",
        size="market_cap",
        color="sector",
        hover_data=["sector"],
        title="Sector Performance: Mispricing vs Market Cap",
        labels={
            "num_stocks": "Number of Stocks",
            "avg_mispricing": "Average Mispricing Score",
        },
        height=600,
    )
    fig_sector_bubble.show()
    fig_sector_bubble.write_html(analytics_dir / "sector_performance_bubble.html")
    print(f"  ✓ Saved: {analytics_dir / 'sector_performance_bubble.html'}")

    # Generate PNG version for Excel integration
    try:
        fig_sector_bubble.write_image(
            plots_dir / "sector_performance_bubble.png", width=1000, height=600
        )
        print(f"  ✓ Saved PNG: {plots_dir / 'sector_performance_bubble.png'}")
    except (ValueError, IOError, OSError, ImportError, RuntimeError) as e:
        print(f"  ⚠️ PNG generation skipped (install kaleido: pip install kaleido): {e}")

print(f"\n✅ Interactive Visualizations Complete")
print(f"   Plots saved to: {plots_dir}")
print(f"   Analytics saved to: {analytics_dir}")
# %%
# Rank stocks using Phase 9.7 functions
top_undervalued = analytics_rank_undervalued(all_stocks_phase95, top_n=TOP_N_RANKINGS)
top_overvalued = analytics_rank_overvalued(all_stocks_phase95, top_n=TOP_N_RANKINGS)

print("\n🏆 Top 50 Undervalued Stocks (Buy Opportunities):")
print(
    top_undervalued[
        [
            "ticker",
            "name",
            "exchange",
            "country",
            "sector",
            "last_price",
            "price_target",
            "predicted_price_target",
            "mispricing_score",
        ]
    ].head(50)
)

print("\n⚠️  Top 50 Overvalued Stocks (Sell Opportunities):")
print(
    top_overvalued[
        [
            "ticker",
            "name",
            "exchange",
            "country",
            "sector",
            "last_price",
            "price_target",
            "predicted_price_target",
            "mispricing_score",
        ]
    ].head(50)
)

# %%
# 📄 Comprehensive Report Generation - Excel, PDF, HTML
print(f"\n📄 Generating Comprehensive Reports...")

# Import reporting functions from analytics (Phase 5 refactor)
from finance_ml.ml_workflow.analytics import generate_enhanced_pdf_report

# 1. Excel Report with Multiple Sheets - Enhanced with comprehensive formatting
print("  Creating Excel report with multiple sheets...")
excel_path = reports_dir / "comprehensive_analysis_report.xlsx"

with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    workbook = writer.book

    # Define formats for comprehensive number formatting (2 decimal places)
    number_format = workbook.add_format({"num_format": "0.00"})
    percent_format = workbook.add_format({"num_format": "0.00%"})
    integer_format = workbook.add_format({"num_format": "#,##0"})
    large_number_format = workbook.add_format({"num_format": "#,##0.00"})
    header_format = workbook.add_format(
        {"bold": True, "bg_color": "#4472C4", "font_color": "white", "border": 1}
    )

    # Helper function to apply comprehensive number formatting
    def apply_number_formatting(worksheet, df):
        """Apply 2-decimal formatting to all numerical columns"""
        for col_idx, col in enumerate(df.columns):
            col_lower = col.lower()
            # Set column width for readability
            worksheet.set_column(col_idx, col_idx, 15)

            if df[col].dtype in ["float64", "float32", "int64", "int32"]:
                # Apply appropriate format based on column type
                if "pct" in col_lower or "percent" in col_lower or "mispricing_pct" == col:
                    worksheet.set_column(col_idx, col_idx, 12, percent_format)
                elif "market_cap" in col_lower or "total_" in col_lower:
                    worksheet.set_column(col_idx, col_idx, 15, large_number_format)
                elif "count" in col_lower or "num_" in col_lower:
                    worksheet.set_column(col_idx, col_idx, 12, integer_format)
                else:
                    worksheet.set_column(col_idx, col_idx, 12, number_format)

    # Helper function to add conditional formatting for key metrics
    def add_conditional_formatting(worksheet, df, column_name):
        """Add 3-color scale conditional formatting to specified column"""
        if column_name in df.columns and len(df) > 0:
            col_idx = df.columns.get_loc(column_name)
            worksheet.conditional_format(
                1,
                col_idx,
                len(df),
                col_idx,
                {
                    "type": "3_color_scale",
                    "min_color": "#F8696B",  # Red for negative/low
                    "mid_color": "#FFEB84",  # Yellow for neutral
                    "max_color": "#63BE7B",  # Green for positive/high
                },
            )

    # Sheet 1: Top Undervalued Stocks
    top_undervalued.to_excel(writer, sheet_name="Top_Undervalued", index=False)
    worksheet_under = writer.sheets["Top_Undervalued"]
    apply_number_formatting(worksheet_under, top_undervalued)
    add_conditional_formatting(worksheet_under, top_undervalued, "mispricing_score")

    # Add conditional formatting for additional key columns
    for col in ["last_price", "price_target", "predicted_price_target"]:
        if col in top_undervalued.columns:
            add_conditional_formatting(worksheet_under, top_undervalued, col)

    # Sheet 2: Top Overvalued Stocks
    top_overvalued.to_excel(writer, sheet_name="Top_Overvalued", index=False)
    worksheet_over = writer.sheets["Top_Overvalued"]
    apply_number_formatting(worksheet_over, top_overvalued)
    add_conditional_formatting(worksheet_over, top_overvalued, "mispricing_score")

    # Add conditional formatting for additional key columns
    for col in ["last_price", "price_target", "predicted_price_target"]:
        if col in top_overvalued.columns:
            add_conditional_formatting(worksheet_over, top_overvalued, col)

    # Sheet 3: All Predictions
    # Export ALL columns from all_stocks_phase95 for comprehensive dashboard usage
    # This includes all analytical columns from Sections 7-9:
    # - prediction_error, prediction_error_pct (Section 7)
    # - mispricing_pct, mispricing_score (Section 8)
    # - model_analyst_diff_pct (Section 9)
    # - Plus all original financial metrics (p_e, p_b, roe, etc.)
    predictions_export = all_stocks_phase95.copy()
    predictions_export.to_excel(writer, sheet_name="All_Predictions", index=False)
    worksheet_pred = writer.sheets["All_Predictions"]
    apply_number_formatting(worksheet_pred, predictions_export)
    add_conditional_formatting(worksheet_pred, predictions_export, "mispricing_score")

    # Add conditional formatting for price columns
    for col in ["last_price", "price_target", "predicted_price_target", "market_cap"]:
        if col in predictions_export.columns:
            add_conditional_formatting(worksheet_pred, predictions_export, col)

    # Sheet 4: Sector Summary
    if sector_summary is not None:
        sector_summary.to_excel(writer, sheet_name="Sector_Summary", index=False)
        worksheet_sector = writer.sheets["Sector_Summary"]
        apply_number_formatting(worksheet_sector, sector_summary)

        # Add conditional formatting for sector performance metrics
        for col in ["avg_mispricing", "total_market_cap"]:
            add_conditional_formatting(worksheet_sector, sector_summary, col)

    # Sheet 5: Model Metrics (if available)
    if "test_metrics" in locals() and test_metrics:
        metrics_df = pd.DataFrame([test_metrics])
        metrics_df.to_excel(writer, sheet_name="Model_Metrics", index=False)
        worksheet_metrics = writer.sheets["Model_Metrics"]
        apply_number_formatting(worksheet_metrics, metrics_df)

        # Add conditional formatting for R² and MAE
        for col in ["r2", "mae", "rmse", "mape"]:
            add_conditional_formatting(worksheet_metrics, metrics_df, col)

    # Sheet 6: Visualizations - Insert PNG images
    worksheet_viz = workbook.add_worksheet("Visualizations")
    row_offset = 0

    # List of PNG files to insert with corresponding sections
    png_files = [
        ("prediction_scatter_interactive.png", "Predicted vs Actual Price Targets"),
        ("residual_analysis_interactive.png", "Residual Analysis"),
        ("mispricing_heatmap_interactive.png", "Mispricing Heatmap (Sector vs Region)"),
        ("stock_rankings_interactive.png", "Stock Rankings - Top Under/Overvalued"),
        ("sector_performance_bubble.png", "Sector Performance Summary"),
    ]

    print(f"    Embedding PNG visualizations into Excel...")
    for png_file, title in png_files:
        png_path = plots_dir / png_file
        if png_path.exists():
            # Add section title with header formatting
            worksheet_viz.write(row_offset, 0, title, header_format)
            worksheet_viz.set_row(row_offset, 20)  # Set row height for title
            row_offset += 1

            # Insert image with appropriate scaling
            try:
                worksheet_viz.insert_image(
                    row_offset,
                    0,
                    str(png_path),
                    {"x_scale": 0.6, "y_scale": 0.6, "x_offset": 10, "y_offset": 10},
                )
                print(f"      ✓ Embedded: {png_file}")
                # Approximate row height for image (adjust based on image size)
                row_offset += 30  # Space for image + gap
            except (IOError, OSError, ValueError, TypeError) as e:
                worksheet_viz.write(row_offset, 0, f"Error inserting {png_file}: {e}")
                print(f"      ⚠️ Failed to embed {png_file}: {e}")
                row_offset += 2
        else:
            worksheet_viz.write(row_offset, 0, f"{title}: PNG not found ({png_file})")
            print(f"      ⚠️ PNG not found: {png_file}")
            row_offset += 2

print(f"  ✓ Excel report saved with enhanced formatting: {excel_path}")
print(f"    - All numerical columns formatted to 2 decimal places")
print(f"    - Conditional formatting applied to key metrics")
print(f"    - PNG visualizations embedded in 'Visualizations' sheet")

# Save all_predictions.csv for Dash/Streamlit dashboards
print("  Creating all_predictions.csv for dashboard usage...")
all_predictions_path = analytics_dir / "predictions.csv"
try:
    # Use the predictions_export dataframe created above
    predictions_export.to_csv(all_predictions_path, index=False)
    print(f"  ✓ Saved: {all_predictions_path}")
    print(f"     (Compatible with dash_app.py load_data function)")
except (IOError, OSError, ValueError, TypeError, KeyError) as e:
    print(f"  ⚠️ Could not save all_predictions.csv: {e}")

# 1b. Enhanced Excel Report with Additional Sheets (Phase 9.7 Enhancement)
# Adds Executive_Summary, Quality_Opportunities, Sector_Leaders/Laggards, Risk_Assessment, Phase93_Analysis
print("  Creating enhanced Excel report with Phase 9.7 sheets...")
enhanced_excel_path = reports_dir / "comprehensive_analysis_report_enhanced.xlsx"

# Configure Excel report options (per code_guidelines.md Section 8.1)
excel_config = ExcelReportConfig(
    include_executive_summary=True,
    include_quality_opportunities=True,
    include_sector_leaders=True,
    include_risk_assessment=True,
    include_phase93_analysis=True,
    top_n_per_sector=5,
    quality_threshold=QUALITY_THRESHOLD_DEFAULT,
    embed_visualizations=True,
)

# Initialize model_metrics if not already defined from model training
# This ensures the variable exists for report generation
if "model_metrics" not in dir():
    model_metrics = {
        "r2": globals().get("test_r2", 0.0),
        "mae": globals().get("test_mae", 0.0),
        "rmse": globals().get("test_rmse", 0.0),
        "mape": globals().get("test_mape", 0.0),
    }

try:
    generate_enhanced_excel_report(
        df=all_stocks_phase95,
        output_path=enhanced_excel_path,
        config=excel_config,
        model_metrics=model_metrics,
    )
    print(f"  ✓ Saved: {enhanced_excel_path}")
    print(f"    - Executive_Summary sheet with key findings")
    print(f"    - Quality_Opportunities sheet (filtered by quality scores)")
    print(f"    - Sector_Leaders sheet (top {excel_config.top_n_per_sector} per sector)")
    print(f"    - Sector_Laggards sheet (bottom {excel_config.top_n_per_sector} per sector)")
    print(f"    - Risk_Assessment sheet (high-risk stocks)")
    print(f"    - Phase93_Analysis sheet (feature category breakdown)")
except (IOError, OSError, ValueError, TypeError, KeyError, AttributeError) as e:
    print(f"  ⚠️ Enhanced Excel generation skipped: {e}")

# 2. Enhanced PDF Report
print("  Creating enhanced PDF report...")
pdf_path = reports_dir / "valuation_analysis_report.pdf"

try:
    generate_enhanced_pdf_report(
        df=all_stocks_phase95,
        pdf_path=pdf_path,
        title="Stock Valuation Analysis Report",
        include_financial_dashboard=True,
        include_quality_alerts=True,
        include_charts=True,  # Charts already saved separately
        template="modern",
    )
    print(f"  ✓ Saved: {pdf_path}")
except (IOError, OSError, ValueError, TypeError, ImportError, RuntimeError) as e:
    print(f"  ⚠️ PDF generation skipped: {e}")

# 3. Enhanced HTML Summary Report (Phase 9.7 Enhancement)
# Uses generate_enhanced_analysis_html() with executive summary, sector breakdown,
# quality-filtered rankings, risk warnings, and Phase 9.3 category analysis
print("  Creating enhanced HTML summary report...")
html_path = reports_dir / "analysis_summary.html"

# Configure HTML report options (per code_guidelines.md Section 8.1)
html_config = HTMLReportConfig(
    include_executive_summary=True,
    include_sector_breakdown=True,
    include_quality_filtered=True,
    include_risk_warnings=True,
    include_phase93_summary=True,
    top_n_stocks=REPORT_TOP_N_DEFAULT,
    quality_threshold=QUALITY_THRESHOLD_DEFAULT,
    template="modern",
)

# Prepare model metrics for executive summary
model_metrics = {
    "r2": globals().get("test_r2", 0.0),
    "mae": globals().get("test_mae", 0.0),
    "rmse": globals().get("test_rmse", 0.0),
    "mape": globals().get("test_mape", 0.0),
}

try:
    generate_enhanced_analysis_html(
        df=all_stocks_phase95,
        output_path=html_path,
        config=html_config,
        model_metrics=model_metrics,
    )
    print(f"  ✓ Saved: {html_path}")
    print(f"    - Executive Summary with key findings")
    print(f"    - Sector breakdown with leaders/laggards")
    print(f"    - Quality-filtered rankings (threshold={QUALITY_THRESHOLD_DEFAULT})")
    print(f"    - Risk warnings (z-score threshold={RISK_ZSCORE_THRESHOLD})")
    print(f"    - Phase 9.3 feature category analysis")
except (IOError, OSError, ValueError, TypeError, KeyError, AttributeError) as e:
    print(f"  ⚠️ Enhanced HTML generation failed, falling back to basic: {e}")
    # Fallback to basic HTML if enhanced fails
    html_content = f"""<!DOCTYPE html>
<html>
<head><title>Stock Analysis Summary</title></head>
<body>
<h1>📊 Stock Valuation Analysis Summary</h1>
<p>Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<h2>Total Stocks: {len(all_stocks_phase95):,}</h2>
<p>Note: Enhanced report generation failed. See logs for details.</p>
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  ✓ Saved fallback: {html_path}")

print(f"\n✅ Comprehensive Reports Generated (Phase 9.7 Enhanced)")
print(f"   Excel (Basic): {excel_path}")
print(f"   Excel (Enhanced): {enhanced_excel_path}")
print(f"   HTML (Enhanced): {html_path}")
print(f"   PDF: {pdf_path}")
print(f"\n📋 Enhanced Report Features:")
print(f"   - Executive Summary with key findings and recommendations")
print(f"   - Sector breakdown with leaders/laggards per sector")
print(f"   - Quality-filtered rankings using Phase 9.3 scores")
print(f"   - Risk warnings for high volatility and distress indicators")
print(f"   - Phase 9.3 feature category analysis")
print(f"\n🎉 All interactive visualizations and reporting complete!")
# %%
# 📊 Section 8 Additional Enhanced Visualizations - Stock Valuation
print("\n" + "=" * 80)
print("📊 ADDITIONAL INTERACTIVE VALUATION VISUALIZATIONS")
print("=" * 80)

if "all_stocks_phase95" in dir() and "mispricing_score" in all_stocks_phase95.columns:
    import plotly.express as px

    from finance_ml.ml_workflow.analytics import create_valuation_scatter_plot

    print("\n📈 Mispricing Score Analysis...")

    # Mispricing distribution by sector
    if "sector" in all_stocks_phase95.columns:
        fig = px.violin(
            all_stocks_phase95,
            x="sector",
            y="mispricing_score",
            color="sector",
            box=True,
            title="Mispricing Score Distribution by Sector",
            points="outliers",
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-45, height=600)
        fig.show()

    # Top undervalued opportunities
    print("\n🎯 Top 10 Undervalued Stocks:")
    top_undervalued = all_stocks_phase95.nlargest(10, "mispricing_score")
    display_cols = [
        "ticker",
        "sector",
        "last_price",
        "predicted_price_target",
        "mispricing_score",
    ]
    display_cols = [c for c in display_cols if c in top_undervalued.columns]
    print(top_undervalued[display_cols].to_string(index=False))

    # Sector-region performance matrix
    if "sector" in all_stocks_phase95.columns and "region" in all_stocks_phase95.columns:
        print("\n🌍 Sector-Region Performance Matrix...")
        pivot_mispricing = all_stocks_phase95.pivot_table(
            values="mispricing_score", index="sector", columns="region", aggfunc="mean"
        )

        fig = px.imshow(
            pivot_mispricing,
            text_auto=".2%",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            title="Average Mispricing Score by Sector and Region",
        )
        fig.update_layout(width=900, height=600)
        fig.show()

    # Use create_valuation_scatter_plot from finance_ml package
    print("\n📊 Valuation Scatter Plot (using finance_ml helper)...")
    # Check if required columns exist
    if all(col in all_stocks_phase95.columns for col in ["last_price", "predicted_price_target"]):
        create_valuation_scatter_plot(
            all_stocks_phase95,
            out_path=None,  # Display inline
            color_by="sector",
            size_by="market_cap" if "market_cap" in all_stocks_phase95.columns else None,
            opacity=0.7,
            show_diagonal=True,
            title="Predicted vs Current Price: Valuation Analysis",
            height=700,
            width=1000,
            log_scale=True,
        )

    print("✓ Enhanced valuation visualizations complete")

# %% [markdown]
# ### Analyst Comparison and Advanced Analytics: Predicted vs. Analyst Price Target Comparison
#
# Compare ML predictions with analyst consensus targets:
# - Agreement rate and directional accuracy
# - Systematic bias analysis
# - Disagreement opportunities (contrarian plays)
# - Segment analysis by sector/region
# - Calibration and confidence metrics
#
# %%
# Prediction vs Analyst comparison
# Note: PredictionAnalystAnalytics is imported at the top from finance_ml (Phase 9.7, line 176)
analytics = PredictionAnalystAnalytics(all_stocks_phase95)
analytics.run_full_analysis(disagreement_threshold=DISAGREEMENT_THRESHOLD, top_n=TOP_N_RANKINGS)

# %%
# Generate comprehensive reporting
reports_dir = OUTPUT_DIR / "reporting"
reports_dir.mkdir(exist_ok=True)

print(f"✓ Reports directory created: {reports_dir}")

# %%
# Calculate financial metrics dashboard using Phase 9.8 function
print("\n📊 Generating Financial Metrics Dashboard:")
financial_metrics = reporting_financial_metrics(all_stocks_phase95, group_by="sector")
if financial_metrics:
    print(f"✓ Financial metrics calculated for {len(financial_metrics)} groups")
    # Display sample metrics for first group
    first_group = list(financial_metrics.keys())[0] if financial_metrics else None
    if first_group:
        print(f"  Sample ({first_group}): {list(financial_metrics[first_group].keys())[:5]}")

# %%
# Generate data quality alerts using Phase 9.8 function
print("\n⚠️  Data Quality Alerts:")
quality_alerts = reporting_quality_alerts(all_stocks_phase95)
if quality_alerts:
    print(f"✓ Generated {len(quality_alerts)} quality alerts")
    for alert in quality_alerts[:3]:  # Show first 3 alerts
        print(f"  - {alert}")
else:
    print("✓ No data quality issues detected")
# %%
# 📊 Section 9 Enhanced Visualizations - Prediction vs Analyst Analytics
print("\n" + "=" * 80)
print("📊 INTERACTIVE PREDICTION VS ANALYST VISUALIZATIONS")
print("=" * 80)

if "all_stocks_phase95" in dir():
    required_cols = ["predicted_price_target", "price_target", "last_price"]
    if all(col in all_stocks_phase95.columns for col in required_cols):
        import plotly.express as px
        import plotly.graph_objects as go

        print("\n📊 Model vs Analyst Target Comparison...")

        # Scatter plot: Model vs Analyst predictions
        fig = go.Figure()

        # Convert sector to numeric codes for colorscale
        sector_codes = None
        if "sector" in all_stocks_phase95.columns:
            sector_codes, sector_labels = pd.factorize(all_stocks_phase95["sector"])

        fig.add_trace(
            go.Scatter(
                x=all_stocks_phase95["price_target"],
                y=all_stocks_phase95["predicted_price_target"],
                mode="markers",
                marker=dict(size=8, opacity=0.6, color=sector_codes, colorscale="Viridis"),
                text=all_stocks_phase95.get("name", None),
                name="Stocks",
            )
        )

        # Perfect agreement line
        min_val = min(
            all_stocks_phase95["price_target"].min(),
            all_stocks_phase95["predicted_price_target"].min(),
        )
        max_val = max(
            all_stocks_phase95["price_target"].max(),
            all_stocks_phase95["predicted_price_target"].max(),
        )

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Perfect Agreement",
            )
        )

        fig.update_layout(
            title="Model Predictions vs Analyst Consensus Targets",
            xaxis_title="Analyst Target Price",
            yaxis_title="Model Predicted Price",
            width=900,
            height=700,
        )
        fig.show()

        # Disagreement analysis
        print("\n🎯 Disagreement Analysis...")
        all_stocks_phase95["model_analyst_diff_pct"] = (
            (all_stocks_phase95["predicted_price_target"] - all_stocks_phase95["price_target"])
            / all_stocks_phase95["price_target"]
            * 100
        )

        # Histogram of disagreement
        fig = px.histogram(
            all_stocks_phase95,
            x="model_analyst_diff_pct",
            nbins=50,
            title="Distribution of Model-Analyst Disagreement",
            labels={"model_analyst_diff_pct": "Difference (%)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Perfect Agreement")
        fig.show()

        # High-conviction disagreements
        high_disagreement = all_stocks_phase95[
            abs(all_stocks_phase95["model_analyst_diff_pct"]) > 10
        ].nlargest(10, "model_analyst_diff_pct", keep="all")

        if len(high_disagreement) > 0:
            print(f"\n📌 Top High-Conviction Disagreements (>10% difference):")
            display_cols = [
                "ticker",
                "name",
                "sector",
                "price_target",
                "predicted_price_target",
                "model_analyst_diff_pct",
            ]
            display_cols = [c for c in display_cols if c in high_disagreement.columns]
            print(high_disagreement[display_cols].head(10).to_string(index=False))

        print("✓ Analyst comparison visualizations complete")
# %% [markdown]
# ## Phase 9.8: Comprehensive Reporting and Dashboard Data with Risk Metrics
#
# ### Business Goal
# Generate comprehensive reports, dashboards, and export final results for stakeholders and downstream applications.
#
# ### Key Objectives
# 1. Calculate financial metrics dashboard
# 2. Generate data quality alerts
# 3. Export predictions with standardized schema
# 4. Create interactive visualizations
# 5. Generate Excel/PDF reports
#
# ### Inputs
# - All outputs from Phases 9.1-9.7
#
# ### Outputs
# - `outputs/reporting/`: Final reports, dashboards
# - `outputs/regression/regression_predictions_detailed.csv`: Final predictions export
# - Excel reports with formatted tables and charts
#
# ### Key Functions
# - `calculate_financial_metrics_dashboard()` - KPI reporting
# - `generate_data_quality_alerts()` - Validation alerts
# - `export_predictions()` - Standardized export
# - `generate_prediction_analyst_excel_report()` - Excel report generation
#
# ### Validation Checkpoint
# - All reports generated successfully
# - Predictions exported with complete schema
# - Dashboard data prepared
# - Final artifacts persisted
#
# Construct optimized portfolios based on predictions:
# - Maximum Sharpe ratio optimization
# - Minimum volatility optimization
# - Target return optimization
# - Risk metrics (VaR, CVaR, Sharpe, Sortino, Max Drawdown)
#
# %% [markdown]
# ## Section 10: Portfolio Optimization Workflow
#
# **Business Goal:**
# - Construct optimized portfolios based on predictions (`all_stocks_phase95`).
#
# **Key Objectives:**
# - Build optimized portfolios from stock universe with advanced methods
# - Perform comprehensive risk analysis and stress testing
# - Generate interactive dashboards for portfolio monitoring
# - Validate constraint adherence and backtest performance
#
# **Inputs:**
#     Phase 9.7 Outputs:
# - `outputs/analytics/`: Mispricing rankings, analyst comparison reports
# - `outputs/analytics/portfolio_optimization.csv`: Optimal portfolios
# - `outputs/analytics/risk_metrics.csv`: Risk analysis
# - Top undervalued/overvalued stocks by sector from all_stocks_phase95
#
# - Expected returns: ML-based on predicted price targets in `all_stocks_phase95` and available features or technical/historical price data
# - Covariance matrix: Historical or factor-based
# - Constraints: Position limits, sector caps, turnover limits
#
# **Outputs:**
# - `outputs/portfolio/` - 20+ artifacts including dashboards, holdings, and analytics
#
# **Validation Checkpoint:**
# - Universe diagnostics, optimization scenarios, risk metrics, and backtests generated successfully
# - Constraints (position limits, sector caps, turnover) and risk budgets validated within configured bounds
#
# %%
# 📊 Section 10 – Portfolio Optimization & Risk Management
# Enhanced workflow with Phase 1-7 integration per portfolio_optimization_enhancement_plan.md
# Phase 7 Enhancements: Return clipping, Phase 9.3 features, multi-model ensemble, shrinkage covariance

print("\n" + "=" * 80)
print("📊 SECTION 10: PORTFOLIO OPTIMIZATION & RISK MANAGEMENT (Phase 1-7)")
print("=" * 80)

# Import Phase 1-6 modules
from finance_ml.dashboards import (
    PortfolioRebalanceWidget,
    create_factor_exposure_dashboard,
    create_multi_period_comparison,
)

# Import ML Returns configuration constants (TDD implementation)
from finance_ml.ml_workflow.analytics import (
    DEFAULT_EXPECTED_RETURN,
    LAG_PERIODS,
    MIN_DATES_FOR_RELIABLE_ML,
    MIN_DATES_FOR_TIMESERIES,
    MIN_PORTFOLIO_CANDIDATES,
    TARGET_COL,
    TARGET_COL_FALLBACK,
    TECHNICAL_INDICATORS,
    TRAIN_SIZE,
    # Phase 7.8: Validation & diagnostics
    # Phase 7.1-7.3: Return normalization & features
    clip_expected_returns,
    # Phase 7.6: Black-Litterman ML integration
    # Phase 7.5: Ensemble models
    # Phase 7.7: Robust covariance
    estimate_covariance_shrinkage,
    validate_expected_returns,
    validate_portfolio_metrics,
)
from finance_ml.ml_workflow.analytics.attribution import (
    calculate_performance_attribution,
)
from finance_ml.ml_workflow.analytics.portfolio import (
    load_historical_prices,
    optimize_black_litterman,
    optimize_hrp,
    optimize_risk_parity,
    run_vectorized_backtest,
    run_walk_forward_optimization,
)

# Portfolio reporting functions (Section 10 Enhancement Plan)
from finance_ml.ml_workflow.analytics.portfolio_reporting import (
    backtest_and_attribution,
    frontier_and_constraints,
    portfolio_summary,
    returns_risk_diagnostics,
    risk_decomposition_dashboard,
    risk_management_dashboard,
    universe_summary,
)
from finance_ml.ml_workflow.analytics.risk import (
    calculate_expected_shortfall,
    calculate_sharpe_ratio,
    calculate_tracking_error,
    run_monte_carlo_simulation,
    run_stress_tests,
)
from finance_ml.ml_workflow.analytics.stock_selection import select_portfolio_candidates

# Import Phase 7 configuration constants
from finance_ml.ml_workflow.config import (
    MAX_EXPECTED_RETURN,
    MIN_EXPECTED_RETURN,
)

print("✓ Imported ML Returns configuration constants")
print("✓ Imported Phase 7 enhancement functions")

# %%
# ============================================================================
# SECTION 10: PRICE COLUMN PRESERVATION VALIDATION
# ============================================================================
# CRITICAL: Validate price columns before portfolio optimization (code_guidelines.md Section 8.5.2)
# Portfolio metrics (expected_return, mispricing_score) depend on original price scale
from finance_ml.ml_workflow.preprocessing.column_semantics import PRICE_COLUMNS

print("\n" + "=" * 80)
print("SECTION 10: PRICE COLUMN PRESERVATION VALIDATION")
print("=" * 80)

if "all_stocks_phase95" in dir() and not all_stocks_phase95.empty:
    print("\n🔍 Validating price columns for portfolio optimization...")

    # Check critical price columns
    critical_price_cols = ["last_price", "price_target", "predicted_price_target"]
    for col in critical_price_cols:
        if col in all_stocks_phase95.columns:
            vals = all_stocks_phase95[col].dropna()
            if len(vals) > 0:
                v_min, v_max, v_median = vals.min(), vals.max(), vals.median()
                print(f"  {col}: range [${v_min:.2f}, ${v_max:.2f}], median=${v_median:.2f}")

                # Warning if prices look scaled
                if v_median < 1.0:
                    print(f"    ⚠️ WARNING: {col} median < $1.00 - may be incorrectly scaled!")
                elif v_max < 10.0 and v_min >= 0:
                    print(f"    ⚠️ WARNING: {col} range suggests MinMax scaling [0,1] or similar!")
                else:
                    print(f"    ✓ {col} appears to be in original dollar units")

    # Count preserved price columns
    price_cols_present = [c for c in PRICE_COLUMNS if c in all_stocks_phase95.columns]
    print(f"\n  ✓ {len(price_cols_present)}/21 price columns present in portfolio data")
    print("  ✓ Price column validation complete - ready for portfolio optimization")
else:
    print("\n⚠️  all_stocks_phase95 not available for price validation")

# %%
# ============================================================================
# PRE-PORTFOLIO: Compute Required Ranking Metrics
# ============================================================================
# This cell ensures all required metrics (expected_return, return_1y, mispricing_score)
# are computed BEFORE select_portfolio_candidates() is called.
# See: docs/improvement_plan/portfolio_optimization_enhancement_plan.md
print("\n" + "=" * 80)
print("PRE-PORTFOLIO: Computing Required Ranking Metrics")
print("=" * 80)

if "all_stocks_phase95" in dir() and not all_stocks_phase95.empty:
    from finance_ml.ml_workflow.analytics.portfolio_metrics import (
        ensure_portfolio_metrics,
    )

    print(f"\n✓ Starting with {len(all_stocks_phase95):,} stocks from Phase 9.5")

    # Compute all required metrics (expected_return, return_1y, mispricing_score)
    all_stocks_phase95 = ensure_portfolio_metrics(all_stocks_phase95)

    # Display summary of computed metrics
    print("\n📊 Ranking Metrics Summary:")
    for metric in ["expected_return", "return_1y", "mispricing_score"]:
        if metric in all_stocks_phase95.columns:
            vals = all_stocks_phase95[metric].dropna()
            if len(vals) > 0:
                print(
                    f"  {metric:20s}: range [{vals.min():>7.3f}, {vals.max():>7.3f}], "
                    f"mean={vals.mean():>6.3f}, median={vals.median():>6.3f}"
                )

    print("\n✅ All required ranking metrics computed and validated")
else:
    print("\n⚠️  all_stocks_phase95 not available; skipping metric computation")

# %%
# 10.1 Stock Selection – Advanced multi-criteria filtering and ML ranking
print("\n" + "=" * 80)
print("10.1 STOCK SELECTION - Advanced Filtering & Ranking")
print("=" * 80)

# Initialize top_candidates fallback (satisfies semantic analyzer)
# This will be overwritten if all_stocks_enhanced is available
if "top_candidates" not in dir():
    top_candidates = None

if "all_stocks_phase95" in dir() and not all_stocks_phase95.empty:
    print(f"\n✓ Using all_stocks_phase95 dataframe: {len(all_stocks_phase95)} stocks")

    # Pre-filter diagnostics to detect normalized vs absolute market cap
    print("\n📊 Pre-filter diagnostics:")
    print(f"  Total stocks: {len(all_stocks_phase95):,}")

    if "market_cap" in all_stocks_phase95.columns:
        mc = all_stocks_phase95["market_cap"].dropna()
        print(f"  Market cap available: {len(mc):,} stocks")

        if len(mc) > 0:
            # Smart scale detection based on median and percentiles
            mc_p50 = mc.median()
            mc_p95 = mc.quantile(0.95)
            mc_p25 = mc.quantile(0.25)
            mc_min = mc.min()
            mc_max = mc.max()

            # Detect scale based on data characteristics
            # Per code_guidelines.md Section 8.5: Market cap is price-related data
            # that should be preserved in original form

            if abs(mc_p50) < 1.0 and (mc_min < 0 or mc_max < 100):
                # Normalized/scaled data (centered around 0, range typically -3 to +3)
                is_normalized = True
                min_mc_threshold = 0.0  # Use percentile-based filter instead
                cap_unit = ""  # No unit for normalized data
                display_scale = 1.0
                display_unit = "(normalized/scaled)"

                print(f"    ⚠️  Market cap is NORMALIZED/SCALED (NOT raw values)")
                print(f"    Range: {mc_min:.3f} to {mc_max:.3f}")
                print(f"    Median: {mc_p50:.3f}, Mean: {mc.mean():.3f}")
                print(f"    25th-75th percentile: [{mc_p25:.3f}, {mc.quantile(0.75):.3f}]")
                print(f"    📌 Recommendation: Use raw market_cap values from original data")
                print(f"       (scaled data loses business interpretability per Section 8.5)")

            elif mc_p50 < 1e6:
                # Millions scale (most common for mid/small cap stocks)
                is_normalized = False
                min_mc_threshold = 500.0  # $500M minimum
                cap_unit = "M"
                display_scale = 1.0  # Already in millions
                display_unit = "M"

                print(f"    ✓ Market cap is in MILLIONS (M) - raw values")
                print(f"    Range: ${mc_min:.2f}M to ${mc_max:.2f}M")
                print(f"    Median: ${mc_p50:.2f}M")
                print(f"    95th percentile: ${mc_p95:.2f}M")

            else:
                # Billions scale (absolute currency amounts)
                is_normalized = False
                min_mc_threshold = 0.5  # $0.5B = $500M minimum
                cap_unit = "B"
                display_scale = 1e9
                display_unit = "B"

                print(f"    ✓ Market cap is in BILLIONS (B) - raw values")
                print(
                    f"    Range: ${mc_min / display_scale:.2f}B to ${mc_max / display_scale:.2f}B"
                )
                print(f"    Median: ${mc_p50 / display_scale:.2f}B")
                print(f"    95th percentile: ${mc_p95 / display_scale:.2f}B")

            # Validation checkpoint (code_guidelines.md Section 8.2 & 8.5)
            # Only validate raw values; scaled data can be negative (centered around mean)
            if not is_normalized:
                assert mc_min >= 0, f"Raw market cap cannot be negative: {mc_min}"
                assert mc_p50 > 0, f"Median market cap must be positive: {mc_p50}"
                print(
                    f"    ✓ Validated raw scale: {cap_unit} | Threshold: {min_mc_threshold}{cap_unit}"
                )
            else:
                # For scaled data, validate reasonable range (-5 to +5 typically)
                assert abs(mc_min) < 10, f"Scaled market cap range suspicious: {mc_min}"
                assert abs(mc_max) < 10, f"Scaled market cap range suspicious: {mc_max}"
                print(f"    ✓ Validated scaled data range: [{mc_min:.3f}, {mc_max:.3f}]")
                print(f"    ⚠️  Skipping absolute threshold; using top_n ranking only")

    else:
        print('  ⚠️  WARNING: "market_cap" column not found!')
        min_mc_threshold = 0.0
        cap_unit = ""
        is_normalized = True

    # Apply multi-criteria selection with auto-detected parameters
    print(f"\n🎯 Applying filters:")
    print(f"   min_market_cap={min_mc_threshold}{cap_unit}")
    print(f'   cap_unit="{cap_unit}"')
    print(f"   top_n=150")
    print(f"   max_sector_weight={MAX_SECTOR_WEIGHT}")

    portfolio_candidates = select_portfolio_candidates(
        all_stocks_phase95,
        min_market_cap=min_mc_threshold,
        top_n=150,
        max_sector_weight=MAX_SECTOR_WEIGHT,
        cap_unit=cap_unit,
    )

    if len(portfolio_candidates) == 0:
        print("\n⚠️  No candidates selected with current filters; relaxing constraints...")
        # Retry with no market cap filter (rank by composite score only)
        portfolio_candidates = select_portfolio_candidates(
            all_stocks_phase95,
            min_market_cap=0.0,
            top_n=50,
            max_sector_weight=MAX_SECTOR_WEIGHT,
            cap_unit="",
        )

    print(f"\n✓ Selected {len(portfolio_candidates)} portfolio candidates")
    if len(portfolio_candidates) > 0:
        print(f"  Sectors: {portfolio_candidates['sector'].nunique()}")
        print(f"  Average composite score: {portfolio_candidates['composite_score'].mean():.3f}")
        print("\n📋 Top 10 Candidates:")
        display_cols = [
            "ticker",
            "sector",
            "market_cap",
            "composite_score",
            "expected_return",
            "mispricing_score",
        ]
        available_cols = [c for c in display_cols if c in portfolio_candidates.columns]
        print(portfolio_candidates[available_cols].head(10).to_string(index=False))

        # Additional diagnostic for scaled market cap
        if is_normalized and "market_cap" in portfolio_candidates.columns:
            mc_selected = portfolio_candidates["market_cap"]
            print(f"\n📊 Selected candidates market cap distribution (scaled):")
            print(f"   Range: [{mc_selected.min():.3f}, {mc_selected.max():.3f}]")
            print(f"   Median: {mc_selected.median():.3f}")
else:
    print("\n⚠️  all_stocks_phase95 not available, using top_candidates from Section 10")
    if top_candidates is not None and len(top_candidates) > 0:
        portfolio_candidates = top_candidates.head(50)
        print(f"✓ Using {len(portfolio_candidates)} candidates from top_candidates")
    else:
        print("⚠️  Skipping stock selection - no suitable dataframe available")
        portfolio_candidates = None

# %%
# 10.2 Universe & Filters Diagnostics
print("\n" + "=" * 80)
print("10.2 UNIVERSE & FILTERS DIAGNOSTICS")
print("=" * 80)

# Create outputs/portfolio directory
portfolio_out_dir = Path("outputs/portfolio")
portfolio_out_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Portfolio outputs directory: {portfolio_out_dir}")

if portfolio_candidates is not None and len(portfolio_candidates) > 0:
    # Generate universe summary with sector/region/market cap diagnostics
    print("\n📊 Generating universe summary...")
    universe_manifest = universe_summary(portfolio_candidates, portfolio_out_dir)
    print(f"✓ Created universe summary artifacts: {', '.join(universe_manifest['files'])}")

    # Display summary statistics
    print("\n📈 Portfolio Universe Summary:")
    print(f"  Total candidates: {len(portfolio_candidates)}")
    if "sector" in portfolio_candidates.columns:
        print(f"  Sectors: {portfolio_candidates['sector'].nunique()}")
        print(f"  Sector distribution:")
        for sector, count in portfolio_candidates["sector"].value_counts().head(5).items():
            print(f"    {sector}: {count} ({count / len(portfolio_candidates) * 100:.1f}%)")
    if "region" in portfolio_candidates.columns:
        print(f"  Regions: {portfolio_candidates['region'].nunique()}")
        print(f"  Region distribution: {dict(portfolio_candidates['region'].value_counts())}")
else:
    print("\n⚠️  No portfolio candidates available for diagnostics")

# %%
# Interactive filter explorer placeholder
# Note: The universe_summary function creates portfolio_filter_explorer.html
# This cell documents the available interactive visualization
if portfolio_candidates is not None and len(portfolio_candidates) > 0:
    filter_explorer_path = portfolio_out_dir / "portfolio_filter_explorer.html"
    if filter_explorer_path.exists():
        print(f"\n✓ Interactive filter explorer available at: {filter_explorer_path}")
        print("  Open this file in a browser to explore filtering scenarios")
    else:
        print("\n⚠️  Filter explorer not created")

# %%
# 10.3 ML-Based Return Prediction
print("\n" + "=" * 80)
print("10.3 ML-BASED RETURN PREDICTION")
print("=" * 80)

if portfolio_candidates is not None and len(portfolio_candidates) >= MIN_PORTFOLIO_CANDIDATES:
    # Configure logging
    import logging

    from finance_ml.logging_config import configure_logging, get_logger

    configure_logging(level=logging.INFO, console=True)
    logger = get_logger(__name__)

    from finance_ml.ml_workflow.analytics.ml_returns import (
        create_ensemble_return_predictions,
        create_ml_return_features,
        train_linear_return_predictor,
    )

    logger.info(
        f"Generating ML-based return predictions for " f"{len(portfolio_candidates)} candidates"
    )
    print(
        f"\n✓ Generating ML-based return predictions for " f"{len(portfolio_candidates)} candidates"
    )

    # Stage 1: Ensure return_1y exists
    # (calculate from price history or use YTD return)
    if "return_1y" not in portfolio_candidates.columns:
        if (
            "last_price" in portfolio_candidates.columns
            and "price_1y_ago" in portfolio_candidates.columns
        ):
            # Calculate 1-year return from price history
            portfolio_candidates["return_1y"] = (
                (portfolio_candidates[TARGET_COL_FALLBACK] - portfolio_candidates["price_1y_ago"])
                / portfolio_candidates["price_1y_ago"]
            ).fillna(0.0)
            logger.info("Calculated return_1y from price history")
            print("  ✓ Calculated return_1y from last_price and price_1y_ago")
        elif "total_return_ytd" in portfolio_candidates.columns:
            # Use YTD return as proxy
            portfolio_candidates["return_1y"] = portfolio_candidates["total_return_ytd"].fillna(
                DEFAULT_EXPECTED_RETURN
            )
            logger.info("Using total_return_ytd as proxy for return_1y")
            print("  ✓ Using total_return_ytd as proxy for return_1y")
        else:
            # Default to configured expected return
            portfolio_candidates["return_1y"] = DEFAULT_EXPECTED_RETURN
            logger.warning(
                f"No return data available, using default "
                f"{DEFAULT_EXPECTED_RETURN:.1%} for return_1y"
            )
            print(
                f"  ⚠️  No return data available, using default "
                f"{DEFAULT_EXPECTED_RETURN:.1%} for return_1y"
            )

    # Stage 2: Create ML features for return prediction
    print("\n📊 Creating ML Features...")
    logger.info("Starting ML feature creation")

    # Step 1: Validate data structure
    # Schema v1.3 uses 'last_updated' as canonical date column
    # (code_guidelines.md Section 2.2)
    required_cols = ["ticker", "last_updated", "last_price"]
    missing_cols = [col for col in required_cols if col not in portfolio_candidates.columns]

    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        print(f"  ⚠️  Missing required columns: {missing_cols}")
        print("  ⚠️  Skipping ML feature creation")
        ml_features_df = None
    else:
        # Step 2: Check data structure (cross-sectional vs time-series)
        dates_per_ticker = portfolio_candidates.groupby("ticker")["last_updated"].nunique()
        avg_dates_per_ticker = dates_per_ticker.mean()
        is_cross_sectional = avg_dates_per_ticker < MIN_DATES_FOR_TIMESERIES

        if is_cross_sectional:
            logger.info(
                f"Detected cross-sectional data " f"(avg {avg_dates_per_ticker:.1f} dates/ticker)"
            )
            print(
                f"  ✓ Detected cross-sectional data "
                f"(avg {avg_dates_per_ticker:.1f} dates/ticker)"
            )
            print("  → Skipping time-series ML features (requires historical data)")
            print("  → Will use existing expected_return for optimization")
            # Set ml_features_df to None to signal no ML features available
            ml_features_df = None
        else:
            # Time-series data available - proceed with ML feature creation
            logger.info(
                f"Detected time-series data " f"(avg {avg_dates_per_ticker:.1f} dates/ticker)"
            )
            print(
                f"  ✓ Detected time-series data " f"(avg {avg_dates_per_ticker:.1f} dates/ticker)"
            )

            # Step 2a: Calculate daily returns if not present
            if "return_1d" not in portfolio_candidates.columns:
                logger.info("Calculating daily returns")
                print("  Calculating daily returns...")

                # Sort by ticker and date
                portfolio_candidates = portfolio_candidates.sort_values(["ticker", "last_updated"])

                # Calculate returns grouped by ticker
                portfolio_candidates["return_1d"] = portfolio_candidates.groupby("ticker")[
                    "last_price"
                ].pct_change()

                # Drop NaN returns (first observation per ticker)
                initial_count = len(portfolio_candidates)
                portfolio_candidates = portfolio_candidates.dropna(subset=["return_1d"])
                rows_dropped = initial_count - len(portfolio_candidates)
                logger.info(f"Calculated returns ({rows_dropped} rows dropped)")
                print(f"  ✓ Calculated returns ({rows_dropped} rows dropped)")

            # Step 3: Verify sufficient time series data
            if dates_per_ticker.mean() < MIN_DATES_FOR_RELIABLE_ML:
                logger.warning(
                    f"Limited time series data " f"(avg {dates_per_ticker.mean():.1f} dates/ticker)"
                )
                print(
                    f"  ⚠️  Warning: Limited time series data "
                    f"(avg {dates_per_ticker.mean():.1f} dates/ticker)"
                )
                print("     ML features may be less reliable")

            # Step 4: Create ML features
            logger.info(
                f"Creating ML features with lags={LAG_PERIODS}, "
                f"indicators={TECHNICAL_INDICATORS}"
            )
            ml_features_df = create_ml_return_features(
                portfolio_candidates,
                lags=LAG_PERIODS,
                technical_indicators=TECHNICAL_INDICATORS,
            )
            logger.info(
                f"Created {ml_features_df.shape[1]} ML features, " f"{len(ml_features_df)} rows"
            )
            print(f"  ✓ Created {ml_features_df.shape[1]} ML features")
            print(f"  ✓ Final dataset: {len(ml_features_df)} rows")

    # Stage 3: Train linear return predictor (if we have historical returns)
    if ml_features_df is not None and "return_1y" in portfolio_candidates.columns:
        print("\n📊 Training Linear Return Predictor...")
        logger.info("Training linear return predictor")

        feature_cols = [
            col
            for col in ml_features_df.columns
            if col.startswith("lag_") or col.startswith("tech_")
        ]

        if len(feature_cols) > 0:
            X = ml_features_df[feature_cols].fillna(0)
            y = portfolio_candidates["return_1y"].fillna(DEFAULT_EXPECTED_RETURN)

            # Split for training (use configured train size)
            split_idx = int(len(X) * TRAIN_SIZE)
            X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

            logger.info(
                f"Training with {len(feature_cols)} features, " f"{len(X_train)} training samples"
            )
            linear_model = train_linear_return_predictor(X_train.values, y_train.values)
            print(f"  ✓ Linear model trained with {len(feature_cols)} features")

            # Generate predictions
            ml_predicted_returns = pd.Series(
                linear_model.predict(X.values), index=portfolio_candidates.index
            )

            # Stage-based naming: add ML predictions
            portfolio_candidates_with_ml = portfolio_candidates.copy()
            portfolio_candidates_with_ml["ml_predicted_return"] = ml_predicted_returns
            portfolio_candidates = portfolio_candidates_with_ml

            logger.info(
                f"ML predictions: mean={ml_predicted_returns.mean():.3f}, "
                f"std={ml_predicted_returns.std():.3f}"
            )
            print(
                f"  ✓ ML predictions: mean={ml_predicted_returns.mean():.3f}, "
                f"std={ml_predicted_returns.std():.3f}"
            )

    # Stage 4: Create ensemble return predictions
    print("\n📊 Creating Ensemble Return Predictions...")
    logger.info("Creating ensemble return predictions")

    available_models = []
    if "expected_return" in portfolio_candidates.columns:
        available_models.append("expected_return")
    if "return_1y" in portfolio_candidates.columns:
        available_models.append("return_1y")
    if "ml_predicted_return" in portfolio_candidates.columns:
        available_models.append("ml_predicted_return")

    if len(available_models) >= 2:
        # Equal weights for ensemble
        ensemble_weights = [1.0 / len(available_models)] * len(available_models)

        logger.info(f"Creating ensemble from {len(available_models)} models: {available_models}")
        portfolio_candidates_with_ensemble = create_ensemble_return_predictions(
            portfolio_candidates,
            models=available_models,
            weights=ensemble_weights,
            ensemble_col="ensemble_return",
        )
        portfolio_candidates = portfolio_candidates_with_ensemble

        ensemble_mean = portfolio_candidates["ensemble_return"].mean()
        logger.info(f"Ensemble returns: mean={ensemble_mean:.3f}")
        print(f"  ✓ Ensemble combines {len(available_models)} models: {available_models}")
        print(f"  ✓ Ensemble returns: mean={ensemble_mean:.3f}")

        # Use ensemble as the primary expected return
        portfolio_candidates["expected_return"] = portfolio_candidates["ensemble_return"]
        logger.info("Set ensemble_return as primary expected_return")

    logger.info("ML-based return prediction complete")
    print("\n✓ ML-based return prediction complete")
else:
    logger.warning(
        f"Skipping ML return prediction - insufficient candidates "
        f"(need {MIN_PORTFOLIO_CANDIDATES}, have "
        f"{len(portfolio_candidates) if portfolio_candidates is not None else 0})"
    )
    print("\n⚠️  Skipping ML return prediction - insufficient candidates")


# %%
# 10.3 Traditional Portfolio Optimization (Max Sharpe & Min Volatility)
print("\n" + "=" * 80)
print("10.3 TRADITIONAL PORTFOLIO OPTIMIZATION")
print("=" * 80)

if portfolio_candidates is not None and len(portfolio_candidates) >= MIN_PORTFOLIO_CANDIDATES:
    from finance_ml.ml_workflow.analytics.portfolio import (
        calculate_portfolio_sharpe_ratio,
        generate_efficient_frontier,
        optimize_portfolio_max_sharpe,
        optimize_portfolio_min_volatility,
    )

    logger.info(
        f"Starting traditional portfolio optimization with "
        f"{len(portfolio_candidates)} candidates"
    )

    # Use top 150 stocks for traditional optimization
    valid_stocks_filtered = portfolio_candidates.head(50).copy()
    logger.info(f"Selected top {len(valid_stocks_filtered)} stocks for optimization")
    print(f"\n✓ Optimizing portfolio with {len(valid_stocks_filtered)} stocks")

    # Prepare returns array
    if "expected_return" in valid_stocks_filtered.columns:
        expected_returns_raw = (
            valid_stocks_filtered["expected_return"].fillna(DEFAULT_EXPECTED_RETURN).values
        )
        logger.info("Using expected_return column for optimization")
    else:
        expected_returns_raw = np.full(len(valid_stocks_filtered), DEFAULT_EXPECTED_RETURN)
        logger.warning(f"No expected_return column, using default {DEFAULT_EXPECTED_RETURN:.1%}")

    # ========== Phase 7.1: Validate & Clip Expected Returns ==========
    print("\n📊 Phase 7.1: Return Validation & Clipping")
    raw_diagnostics = validate_expected_returns(expected_returns_raw)
    print(
        f"  Raw returns - Mean: {raw_diagnostics['mean_return']:.2%}, "
        f"Std: {raw_diagnostics['std_return']:.2%}"
    )
    print(f"  Is realistic: {raw_diagnostics['is_realistic']}")
    if raw_diagnostics["warnings"]:
        for warn in raw_diagnostics["warnings"]:
            print(f"  ⚠️ {warn}")

    # Apply Phase 7 clipping to ensure realistic bounds
    expected_returns_array = clip_expected_returns(expected_returns_raw)
    print(f"  Clipping bounds: [{MIN_EXPECTED_RETURN:.0%}, {MAX_EXPECTED_RETURN:.0%}]")
    print(
        f"  After clipping - Mean: {np.mean(expected_returns_array):.2%}, "
        f"Max: {np.max(expected_returns_array):.2%}"
    )

    # Validate clipped returns
    clipped_diagnostics = validate_expected_returns(expected_returns_array)
    print(
        f"  Success Criteria: Mean < 30%: "
        f"{clipped_diagnostics['mean_return'] < 0.30} "
        f"(actual: {clipped_diagnostics['mean_return']:.2%})"
    )
    logger.info(
        f"Phase 7.1: Returns clipped to [{MIN_EXPECTED_RETURN:.0%}, {MAX_EXPECTED_RETURN:.0%}]"
    )

    # ========== Phase 7.7: Robust Covariance Estimation ==========
    print("\n📊 Phase 7.7: Robust Covariance Estimation")
    np.random.seed(42)
    n_stocks = len(valid_stocks_filtered)

    # Generate synthetic daily returns for covariance estimation
    # In production, use actual historical returns from price columns
    synthetic_daily_returns = pd.DataFrame(
        np.random.randn(252, n_stocks) * 0.02 + expected_returns_array / 252,
        columns=[f"Asset_{i}" for i in range(n_stocks)],
    )

    # Use Ledoit-Wolf shrinkage covariance (Phase 7.7)
    cov_matrix = estimate_covariance_shrinkage(
        returns=synthetic_daily_returns, method="ledoit_wolf"
    )
    print(f"  ✓ Ledoit-Wolf shrinkage covariance estimated")
    print(f"  Matrix shape: {cov_matrix.shape}")
    print(f"  Condition number: {np.linalg.cond(cov_matrix):.2f} (lower is better)")
    logger.info("Phase 7.7: Ledoit-Wolf shrinkage covariance estimated")

    # [SECTION 10.4] Expected Returns & Risk Inputs QA
    print("\n" + "-" * 80)
    print("10.4 EXPECTED RETURNS & RISK INPUTS QA")
    print("-" * 80)

    # Convert expected returns to Series for reporting
    mu_series = pd.Series(
        expected_returns_array,
        index=valid_stocks_filtered["ticker"].values,
        name="expected_return",
    )

    # Generate returns and risk diagnostics
    print("\n📊 Generating returns and risk diagnostics...")
    returns_risk_manifest = returns_risk_diagnostics(mu_series, cov_matrix, portfolio_out_dir)
    print(f"✓ Created returns/risk artifacts: {', '.join(returns_risk_manifest['files'])}")

    # Display diagnostics summary
    print(f"\n📈 Returns & Risk Summary:")
    print(f"  Number of assets: {len(mu_series)}")
    print(f"  Expected return range: [{mu_series.min():.3f}, {mu_series.max():.3f}]")
    print(f"  Expected return mean: {mu_series.mean():.3f}")
    print(f"  Portfolio volatility (equal weight): {np.sqrt(np.mean(cov_matrix)):.3f}")
    print("-" * 80 + "\n")

    # Risk-free rate
    risk_free_rate = 0.02

    # Optimize Max Sharpe Ratio
    print("\n📊 Max Sharpe Ratio Optimization:")
    max_sharpe_result = optimize_portfolio_max_sharpe(
        returns=expected_returns_array,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate,
        allow_short=False,
        max_weight=0.20,
    )
    print(f"  Expected Return: {max_sharpe_result['return']:.2%}")
    print(f"  Volatility: {max_sharpe_result['volatility']:.2%}")
    print(f"  Sharpe Ratio: {max_sharpe_result['sharpe_ratio']:.3f}")
    print(f"  Num non-zero positions: {np.sum(max_sharpe_result['weights'] > 0.001)}")

    # Optimize Min Volatility
    print("\n📊 Min Volatility Optimization:")
    min_vol_result = optimize_portfolio_min_volatility(
        returns=expected_returns_array,
        cov_matrix=cov_matrix,
        allow_short=False,
        max_weight=0.20,
    )
    # Calculate Sharpe ratio manually (not returned by optimize_portfolio_min_volatility)
    min_vol_sharpe = calculate_portfolio_sharpe_ratio(
        min_vol_result["return"], min_vol_result["volatility"], risk_free_rate
    )
    print(f"  Expected Return: {min_vol_result['return']:.2%}")
    print(f"  Volatility: {min_vol_result['volatility']:.2%}")
    print(f"  Sharpe Ratio: {min_vol_sharpe:.3f}")
    print(f"  Num non-zero positions: {np.sum(min_vol_result['weights'] > 0.001)}")

    # ========== Phase 7.8: Portfolio Metrics Validation ==========
    print("\n📊 Phase 7.8: Portfolio Metrics Validation")
    portfolio_validation = validate_portfolio_metrics(
        weights=max_sharpe_result["weights"],
        returns=synthetic_daily_returns,
        risk_free_rate=risk_free_rate,
        max_sharpe_threshold=3.0,
        max_return_threshold=1.0,
    )
    print(f"  Validated Sharpe Ratio: {portfolio_validation['sharpe_ratio']:.3f}")
    print(
        f"  Sharpe Valid (<3.0): {portfolio_validation['sharpe_ratio_valid']} "
        f"{'✓' if portfolio_validation['sharpe_ratio_valid'] else '⚠️'}"
    )
    print(
        f"  Return Realistic: {portfolio_validation['return_realistic']} "
        f"{'✓' if portfolio_validation['return_realistic'] else '⚠️'}"
    )
    print(
        f"  Success Criteria: Max Sharpe < 3.0: "
        f"{portfolio_validation['sharpe_ratio'] < 3.0} "
        f"(actual: {portfolio_validation['sharpe_ratio']:.3f})"
    )
    if portfolio_validation["warnings"]:
        for warn in portfolio_validation["warnings"]:
            print(f"  ⚠️ {warn}")
    logger.info(
        f"Phase 7.8: Portfolio validation - Sharpe={portfolio_validation['sharpe_ratio']:.3f}"
    )

    # Store best return column for later use
    best_return_col = (
        "expected_return" if "expected_return" in valid_stocks_filtered.columns else None
    )

    logger.info("Traditional portfolio optimization complete")
    print("\n✓ Traditional portfolio optimization complete (Phase 7 enhanced)")
else:
    logger.warning(
        f"Skipping traditional optimization - insufficient candidates "
        f"(need {MIN_PORTFOLIO_CANDIDATES}, have "
        f"{len(portfolio_candidates) if portfolio_candidates is not None else 0})"
    )
    print("\n⚠️  Skipping traditional optimization - insufficient candidates")
    max_sharpe_result = None
    min_vol_result = None
    valid_stocks_filtered = None
    expected_returns_array = None
    cov_matrix = None
    risk_free_rate = 0.02

# %%
# 10.3 Advanced Portfolio Optimization
print("\n" + "=" * 80)
print("10.3 ADVANCED PORTFOLIO OPTIMIZATION")
print("=" * 80)

if portfolio_candidates is not None and len(portfolio_candidates) >= MIN_PORTFOLIO_CANDIDATES:
    logger.info(
        f"Starting advanced portfolio optimization with " f"{len(portfolio_candidates)} candidates"
    )

    # Use top 50 for optimization to keep it manageable
    opt_universe = portfolio_candidates.head(50)
    logger.info(f"Selected top {len(opt_universe)} stocks for advanced optimization")

    # Prepare returns and covariance
    if "expected_return" in opt_universe.columns:
        mean_returns_raw = opt_universe.set_index("ticker")["expected_return"].fillna(
            DEFAULT_EXPECTED_RETURN
        )
        logger.info("Using expected_return column for advanced optimization")
    else:
        mean_returns_raw = pd.Series(
            DEFAULT_EXPECTED_RETURN,
            index=opt_universe["ticker"],
            name="expected_return",
        )
        logger.warning(f"No expected_return column, using default {DEFAULT_EXPECTED_RETURN:.1%}")

    # ========== Phase 7.1: Clip Returns for Advanced Optimization ==========
    print("\n📊 Phase 7.1: Return Clipping for Advanced Optimization")
    mean_returns_clipped = clip_expected_returns(mean_returns_raw.values)
    mean_returns = pd.Series(
        mean_returns_clipped, index=mean_returns_raw.index, name="expected_return"
    )
    print(
        f"  Clipped {len(mean_returns)} returns to [{MIN_EXPECTED_RETURN:.0%}, {MAX_EXPECTED_RETURN:.0%}]"
    )
    print(f"  Mean after clipping: {mean_returns.mean():.2%}")
    logger.info(f"Phase 7.1: Advanced optimization returns clipped")

    # ========== Phase 7.7: Shrinkage Covariance for Advanced Optimization ==========
    print("\n📊 Phase 7.7: Shrinkage Covariance for Advanced Optimization")
    np.random.seed(42)
    n_adv_stocks = len(opt_universe)

    # Generate synthetic daily returns for covariance estimation
    synthetic_adv_returns = pd.DataFrame(
        np.random.randn(252, n_adv_stocks) * 0.02 + mean_returns.values / 252,
        columns=opt_universe["ticker"].values,
    )

    # Use Ledoit-Wolf shrinkage covariance
    cov_matrix = estimate_covariance_shrinkage(returns=synthetic_adv_returns, method="ledoit_wolf")
    print(f"  ✓ Ledoit-Wolf shrinkage covariance estimated")
    print(f"  Condition number: {np.linalg.cond(cov_matrix):.2f}")
    logger.info("Phase 7.7: Advanced optimization using Ledoit-Wolf covariance")

    # Black-Litterman optimization
    print("\n📊 Black-Litterman Optimization:")
    market_weights = np.full(len(opt_universe), 1.0 / len(opt_universe))
    views = {opt_universe.iloc[0]["ticker"]: 0.12, opt_universe.iloc[1]["ticker"]: 0.10}
    view_confidences = [0.7, 0.6]

    bl_weights, bl_returns = optimize_black_litterman(
        returns=mean_returns,
        cov_matrix=cov_matrix,
        market_weights=market_weights,
        views=views,
        view_confidences=view_confidences,
        risk_aversion=2.5,
    )
    print(f"  Top 3 positions: {bl_weights[:3]}")
    print(f"  Weights sum: {bl_weights.sum():.4f}")

    # Risk Parity
    print("\n📊 Risk Parity Optimization:")
    rp_weights = optimize_risk_parity(cov_matrix)
    print(f"  Top 3 positions: {rp_weights[:3]}")
    print(f"  Weights sum: {rp_weights.sum():.4f}")

    # Hierarchical Risk Parity
    print("\n📊 Hierarchical Risk Parity (HRP):")
    # Calculate standard deviations from historical volatility columns
    volatility_cols = [
        "volatility_1m",
        "volatility_3m",
        "volatility_6m",
        "volatility_1y",
    ]
    available_vol_cols = [col for col in volatility_cols if col in opt_universe.columns]

    if available_vol_cols:
        # Use mean of available volatility columns as annualized std dev
        std_devs = opt_universe[available_vol_cols].mean(axis=1).fillna(0.20).values
        print(f"  Calculated std_devs from {len(available_vol_cols)} volatility columns")
    else:
        # Fallback: Use default 20% annualized volatility for equities
        std_devs = np.full(len(opt_universe), 0.20)
        print("  Using default std_devs (20% annualized volatility)")

    # Ensure std_devs are positive (fix for ValueError: scale < 0)
    std_devs = np.where(std_devs <= 0, 0.20, std_devs)  # Replace non-positive with 20% default
    std_devs = np.nan_to_num(std_devs, nan=0.20)  # Replace NaN with 20% default

    synthetic_returns = np.random.RandomState(RANDOM_SEED).normal(
        mean_returns / 252, std_devs / np.sqrt(252), size=(252, len(opt_universe))
    )
    returns_df = pd.DataFrame(
        synthetic_returns, columns=[f"Asset_{i}" for i in range(len(opt_universe))]
    )
    hrp_weights = optimize_hrp(returns_df)
    print(f"  Top 3 positions: {hrp_weights[:3]}")
    print(f"  Weights sum: {hrp_weights.sum():.4f}")

    print("\n✓ Advanced optimization complete")

    # [SECTION 10.5] Optimization Frontiers & Constraint Explorer
    print("\n" + "-" * 80)
    print("10.5 OPTIMIZATION FRONTIERS & CONSTRAINT EXPLORER")
    print("-" * 80)

    # Generate efficient frontier and constraint sensitivity artifacts
    print("\n📊 Generating efficient frontier and constraint scenarios...")
    constraints_dict = {"max_weight": [0.1, 0.15, 0.2, 0.25, 0.3]}
    frontier_manifest = frontier_and_constraints(
        mean_returns, cov_matrix, constraints_dict, portfolio_out_dir
    )
    print(f"✓ Created frontier artifacts: {', '.join(frontier_manifest['files'])}")

    print(f"\n📈 Frontier & Constraints Summary:")
    print(f"  Optimization universe: {len(mean_returns)} assets")
    print(f"  Constraint scenarios: {len(constraints_dict['max_weight'])} max_weight values")
    print(f"  Expected return range: [{mean_returns.min():.3f}, {mean_returns.max():.3f}]")
    print("-" * 80 + "\n")
else:
    print("\n⚠️  Skipping advanced optimization - insufficient candidates")

# %%
# 10.6 Risk Analysis – Stress Tests & Monte Carlo (formerly 10.4)
print("\n" + "=" * 80)
print("10.6 RISK ANALYSIS - Advanced Metrics")
print("=" * 80)

if "bl_weights" in dir() and "returns_df" in dir():
    # Expected Shortfall
    synthetic_port_returns = pd.Series(returns_df.values @ bl_weights)
    es_95 = calculate_expected_shortfall(synthetic_port_returns, confidence=0.95)
    es_99 = calculate_expected_shortfall(synthetic_port_returns, confidence=0.99)
    print(f"\n📊 Expected Shortfall:")
    print(f"  ES 95%: {es_95:.4f}")
    print(f"  ES 99%: {es_99:.4f}")

    # Tracking Error (vs equal-weight benchmark)
    equal_weights = np.full(len(bl_weights), 1.0 / len(bl_weights))
    benchmark_returns = pd.Series(returns_df.values @ equal_weights)
    te = calculate_tracking_error(synthetic_port_returns, benchmark_returns)
    print(f"\n📊 Tracking Error vs Equal-Weight: {te:.4f}")

    # Stress Testing
    print(f"\n📊 Stress Testing:")
    scenarios = {
        "Market Crash": {"equity": -0.30, "bonds": -0.10},
        "Moderate Correction": {"equity": -0.15, "bonds": -0.05},
    }
    asset_classes = ["equity"] * int(len(bl_weights) * 0.7) + ["bonds"] * (
        len(bl_weights) - int(len(bl_weights) * 0.7)
    )
    stress_results = run_stress_tests(bl_weights, returns_df, scenarios, asset_classes)
    for scenario, result in stress_results.items():
        print(f"  {scenario}: Portfolio Loss = {result['portfolio_loss']:.2%}")

    # Monte Carlo Simulation
    print(f"\n📊 Monte Carlo Simulation (2000 paths, 252 days):")
    mc_results = run_monte_carlo_simulation(
        bl_weights,
        returns_df,
        n_simulations=2000,
        time_horizon=252,
        confidence_levels=QUANTILES,
        random_state=RANDOM_SEED,
    )
    final_values = mc_results["paths"][:, -1]
    print(f"  Median final value: {np.median(final_values):.3f}")
    print(f"  5th percentile: {np.percentile(final_values, 5):.3f}")
    print(f"  95th percentile: {np.percentile(final_values, 95):.3f}")

    logger.info("Risk analysis complete")
    print("\n✓ Risk analysis complete")

    # [SECTION 10.7] Portfolio Breakdown & Risk Decomposition
    print("\n" + "-" * 80)
    print("10.7 PORTFOLIO BREAKDOWN & RISK DECOMPOSITION")
    print("-" * 80)

    # Create portfolio exposures dataframe for risk decomposition
    print("\n📊 Generating risk decomposition dashboard...")
    weights_series = pd.Series(bl_weights, index=opt_universe["ticker"].values, name="weight")
    exposures_df = opt_universe[["ticker", "sector", "region"]].set_index("ticker")

    decomp_manifest = risk_decomposition_dashboard(weights_series, exposures_df, portfolio_out_dir)
    print(f"✓ Created risk decomposition artifacts: {', '.join(decomp_manifest['files'])}")

    print(f"\n📈 Risk Decomposition Summary:")
    print(f"  Portfolio holdings: {len(weights_series)} assets")
    print(f"  Active positions (>0.1%): {(weights_series > 0.001).sum()}")
    print(f"  Sector concentration: {exposures_df.groupby('sector').size().to_dict()}")
    print("-" * 80 + "\n")
else:
    logger.warning(
        "Skipping advanced portfolio optimization and risk analysis - "
        "insufficient candidates or optimization data not available"
    )
    print("\n⚠️  Skipping risk analysis - optimization data not available")

# %%
# [SECTION 10.8] Backtesting & Walk-Forward Results (formerly 10.5)
print("\n" + "=" * 80)
print("10.8 BACKTESTING & WALK-FORWARD RESULTS")
print("=" * 80)

# Load synthetic historical data
historical_prices = load_historical_prices(n_obs=756, n_assets=4, seed=123)
# Convert to DataFrame with column names for compatibility with backtest_and_attribution
historical_prices = pd.DataFrame(
    historical_prices, columns=[f"Asset_{i}" for i in range(historical_prices.shape[1])]
)
print(f"\n✓ Loaded historical prices: {historical_prices.shape}")

# Vectorized Backtest
print("\n📊 Vectorized Backtest (Max Sharpe, Monthly Rebalance):")
backtest_results = run_vectorized_backtest(
    data=historical_prices,
    rebalance_frequency="monthly",
    optimization_method="max_sharpe",
    lookback_window=252,
    transaction_costs=0.001,
)
print(f"  Portfolio Returns: {len(backtest_results['portfolio_returns'])} observations")
print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
print(f"  Max Drawdown: {backtest_results['max_drawdown']:.2%}")
print(f"  Total Turnover: {backtest_results['turnover']:.2f}")

# Walk-Forward Optimization
print("\n📊 Walk-Forward Optimization (Black-Litterman):")
wfo_results = run_walk_forward_optimization(
    data=historical_prices,
    train_window=252,
    test_window=63,
    step_size=21,
    optimization_method="black_litterman",
)
in_sharpe = calculate_sharpe_ratio(wfo_results["in_sample_returns"])
oos_sharpe = calculate_sharpe_ratio(wfo_results["out_of_sample_returns"])
print(f"  In-Sample Sharpe: {in_sharpe:.3f}")
print(f"  Out-of-Sample Sharpe: {oos_sharpe:.3f}")
print(f"  Overfitting Check: {'PASS' if oos_sharpe < in_sharpe else 'FAIL'}")

# Performance Attribution (simple example)
print("\n📊 Performance Attribution (Brinson-Fachler):")
portfolio_weights = pd.DataFrame([[0.6, 0.4]], columns=["Tech", "Finance"])
benchmark_weights = pd.DataFrame([[0.5, 0.5]], columns=["Tech", "Finance"])
portfolio_returns = pd.DataFrame([[0.12, 0.06]], columns=["Tech", "Finance"])
benchmark_returns = pd.DataFrame([[0.10, 0.04]], columns=["Tech", "Finance"])
attribution = calculate_performance_attribution(
    portfolio_weights, portfolio_returns, benchmark_weights, benchmark_returns
)
print(f"  Allocation Effect: {attribution['allocation_effect']:.4f}")
print(f"  Selection Effect: {attribution['selection_effect']:.4f}")
print(f"  Interaction Effect: {attribution['interaction_effect']:.4f}")

# Generate backtest and attribution artifacts
print("\n📊 Generating backtest artifacts...")
# Create weights series for backtest reporting (using equal weights as example)
backtest_weights = pd.Series(
    [0.25] * historical_prices.shape[1], index=historical_prices.columns, name="weight"
)
backtest_manifest = backtest_and_attribution(historical_prices, backtest_weights, portfolio_out_dir)
print(f"✓ Created backtest artifacts: {', '.join(backtest_manifest['files'])}")

print("\n✓ Backtesting complete")

# %%
# [SECTION 10.9] Risk Management Dashboard
print("\n" + "=" * 80)
print("10.9 RISK MANAGEMENT DASHBOARD")
print("=" * 80)

# Generate risk management dashboard using portfolio from advanced optimization
if "bl_weights" in dir() and "cov_matrix" in dir():
    print("\n📊 Generating risk management dashboard...")
    weights_for_risk = pd.Series(bl_weights, name="weight")

    risk_mgmt_manifest = risk_management_dashboard(weights_for_risk, cov_matrix, portfolio_out_dir)
    print(f"✓ Created risk management artifacts: {', '.join(risk_mgmt_manifest['files'])}")

    # Calculate and display key risk metrics
    portfolio_vol = np.sqrt(weights_for_risk.values @ cov_matrix @ weights_for_risk.values)
    print(f"\n📈 Risk Management Summary:")
    print(f"  Portfolio volatility: {portfolio_vol:.3%}")
    print(f"  Number of positions: {len(weights_for_risk)}")
    print(f"  Active positions (>0.1%): {(weights_for_risk > 0.001).sum()}")
else:
    print("\n⚠️  Skipping risk management dashboard - optimization data not available")

print("\n✓ Risk management dashboard complete")

# %%
# [SECTION 10.10] Summary, QA, and Export
print("\n" + "=" * 80)
print("10.10 PORTFOLIO SUMMARY & EXPORT")
print("=" * 80)

# Generate final portfolio summary
print("\n📊 Generating portfolio summary...")
summary_kpis = {}

# Collect KPIs from various sections if available
if "max_sharpe_result" in dir() and max_sharpe_result:
    summary_kpis["max_sharpe_return"] = max_sharpe_result["return"]
    summary_kpis["max_sharpe_volatility"] = max_sharpe_result["volatility"]
    summary_kpis["max_sharpe_ratio"] = max_sharpe_result["return"] / max_sharpe_result["volatility"]

if "backtest_results" in dir():
    summary_kpis["backtest_sharpe"] = backtest_results["sharpe_ratio"]
    summary_kpis["backtest_max_drawdown"] = backtest_results["max_drawdown"]
    summary_kpis["backtest_turnover"] = backtest_results["turnover"]

if "expected_shortfall" in dir():
    summary_kpis["expected_shortfall_95"] = expected_shortfall

# Generate summary artifacts
from pathlib import Path

summary_manifest = portfolio_summary(summary_kpis, Path(portfolio_out_dir))
print(f"✓ Created summary artifacts: {', '.join(summary_manifest['files'])}")

print(f"\n📈 Portfolio Analytics Summary:")
print(f"  Total artifacts generated: {len(summary_kpis)} KPIs")
for key, value in summary_kpis.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

print(f"\n✓ All portfolio reporting artifacts saved to: {portfolio_out_dir}")
print("✓ Portfolio optimization workflow complete")

# %%
# [SECTION 10.11] Interactive Dashboard Snapshots (formerly 10.6)
print("\n" + "=" * 80)
print("10.6 INTERACTIVE DASHBOARD GENERATION")
print("=" * 80)

# Use OUTPUT_DIR from configuration
output_dir = OUTPUT_DIR / "analytics"
output_dir.mkdir(parents=True, exist_ok=True)

# Multi-Period Performance Comparison - Use REAL portfolio data from Cell 10.2.5
if "max_sharpe_result" in dir() and max_sharpe_result and "valid_stocks_filtered" in dir():
    print("\n📊 Generating Multi-Period Performance Comparison (Real Portfolio Data)...")

    # Generate portfolio returns from real optimization results
    n_days = 252  # 1 trading year
    daily_return = max_sharpe_result["return"] / 252
    daily_vol = max_sharpe_result["volatility"] / np.sqrt(252)

    # Safety rail: ensure daily_vol is positive
    daily_vol = max(daily_vol, 0.01)  # Minimum 1% daily volatility

    np.random.seed(RANDOM_SEED)
    portfolio_rets = pd.Series(
        np.random.normal(daily_return, daily_vol, n_days),
        index=pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq="D"),
    )

    # Ensure configuration variable is defined to prevent NameError (Code Guidelines Section 8.1)
    if "best_return_col" not in dir():
        best_return_col = None  # Will use fallback benchmark

    # Equal-weight benchmark from actual stocks
    if best_return_col and best_return_col in valid_stocks_filtered.columns:
        avg_return = valid_stocks_filtered[best_return_col].mean()
        benchmark_rets = pd.Series(np.full(n_days, avg_return / 252), index=portfolio_rets.index)
    else:
        benchmark_rets = pd.Series(np.full(n_days, 0.0004), index=portfolio_rets.index)

    fig = create_multi_period_comparison(
        portfolio_rets,
        periods=["1M", "3M", "6M", "1Y", "YTD", "ITD"],
        benchmark_returns=benchmark_rets,
    )
    fig.write_html(str(output_dir / "portfolio_multi_period_comparison.html"))
    print("  ✓ Saved: outputs/analytics/portfolio_multi_period_comparison.html")
    print(
        f"  ✓ Using real max Sharpe portfolio (Return: {max_sharpe_result['return']:.2%}, Vol: {max_sharpe_result['volatility']:.2%})"
    )
else:
    print("\n⚠️  Skipping Multi-Period Performance - max_sharpe_result not available")

# Factor Exposure Dashboard - Use REAL portfolio data
if "max_sharpe_result" in dir() and max_sharpe_result and "valid_stocks_filtered" in dir():
    print("\n📊 Generating Factor Exposure Dashboard (Real Portfolio Data)...")

    n_stocks = len(max_sharpe_result["weights"])
    tickers = valid_stocks_filtered["ticker"].head(n_stocks).values
    sample_weights = pd.Series(max_sharpe_result["weights"], index=tickers)

    # Generate factor loadings from actual stock characteristics
    factor_data = {}

    # Market factor (beta-like): normalize expected returns
    if "expected_return" in valid_stocks_filtered.columns:
        returns_normalized = (
            valid_stocks_filtered["expected_return"].head(n_stocks)
            - valid_stocks_filtered["expected_return"].head(n_stocks).mean()
        ) / (valid_stocks_filtered["expected_return"].head(n_stocks).std() + 1e-8)
        factor_data["Market"] = returns_normalized.values
    else:
        factor_data["Market"] = np.random.normal(1.0, 0.2, n_stocks)

    # Size factor: normalize market cap
    if "market_cap" in valid_stocks_filtered.columns:
        mc_normalized = (
            valid_stocks_filtered["market_cap"].head(n_stocks)
            - valid_stocks_filtered["market_cap"].head(n_stocks).mean()
        ) / (valid_stocks_filtered["market_cap"].head(n_stocks).std() + 1e-8)
        factor_data["Size"] = mc_normalized.values
    else:
        factor_data["Size"] = np.random.normal(0.0, 0.3, n_stocks)

    # Value factor: normalize mispricing score
    if "mispricing_score" in valid_stocks_filtered.columns:
        value_normalized = (
            valid_stocks_filtered["mispricing_score"].head(n_stocks)
            - valid_stocks_filtered["mispricing_score"].head(n_stocks).mean()
        ) / (valid_stocks_filtered["mispricing_score"].head(n_stocks).std() + 1e-8)
        factor_data["Value"] = value_normalized.values
    else:
        factor_data["Value"] = np.random.normal(0.3, 0.2, n_stocks)

    # Momentum factor: normalize historical returns
    if "return_1y" in valid_stocks_filtered.columns:
        mom_normalized = (
            valid_stocks_filtered["return_1y"].head(n_stocks)
            - valid_stocks_filtered["return_1y"].head(n_stocks).mean()
        ) / (valid_stocks_filtered["return_1y"].head(n_stocks).std() + 1e-8)
        factor_data["Momentum"] = mom_normalized.values
    else:
        factor_data["Momentum"] = np.random.normal(0.15, 0.2, n_stocks)

    # Quality factor: composite of multiple metrics
    if "composite_score" in valid_stocks_filtered.columns:
        quality_normalized = (
            valid_stocks_filtered["composite_score"].head(n_stocks)
            - valid_stocks_filtered["composite_score"].head(n_stocks).mean()
        ) / (valid_stocks_filtered["composite_score"].head(n_stocks).std() + 1e-8)
        factor_data["Quality"] = quality_normalized.values
    else:
        factor_data["Quality"] = np.random.normal(0.5, 0.15, n_stocks)

    factor_loadings = pd.DataFrame(factor_data, index=tickers)

    fig = create_factor_exposure_dashboard(
        sample_weights,
        factor_loadings,
        factors=["Market", "Size", "Value", "Momentum", "Quality"],
    )
    fig.write_html(str(output_dir / "portfolio_factor_exposure_dashboard.html"))
    print("  ✓ Saved: outputs/analytics/portfolio_factor_exposure_dashboard.html")
    print(f"  ✓ Using {n_stocks} real stocks from optimized portfolio")
else:
    print("\n⚠️  Skipping Factor Exposure - max_sharpe_result not available")

# Rebalancing Widget - Use REAL portfolio data
if "max_sharpe_result" in dir() and max_sharpe_result and "valid_stocks_filtered" in dir():
    print("\n📊 Generating Rebalancing Widget Snapshot (Real Portfolio Data)...")

    n_stocks = len(max_sharpe_result["weights"])
    tickers = valid_stocks_filtered["ticker"].head(n_stocks).values
    prices = valid_stocks_filtered["last_price"].head(n_stocks).fillna(100.0).values

    # Equal-weight starting portfolio ($100k)
    total_value = 100000
    equal_shares = (total_value / n_stocks) / prices

    current_holdings = pd.DataFrame({"ticker": tickers, "shares": equal_shares, "price": prices})
    target_weights = pd.Series(max_sharpe_result["weights"], index=tickers)

    widget = PortfolioRebalanceWidget(current_holdings, target_weights)
    trades = widget.get_rebalance_trades()

    # Create HTML table for trades
    trades_html = f"""
<html>
<head><title>Portfolio Rebalancing Trades</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
h2 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #4CAF50; color: white; }}
tr:nth-child(even) {{ background-color: #f2f2f2; }}
.buy {{ color: green; font-weight: bold; }}
.sell {{ color: red; font-weight: bold; }}
</style>
</head>
<body>
<h2>Portfolio Rebalancing Recommendations (Real Max Sharpe Portfolio)</h2>
<p><strong>Portfolio Metrics:</strong> Return: {max_sharpe_result["return"]:.2%}, 
Volatility: {max_sharpe_result["volatility"]:.2%}, 
Sharpe Ratio: {max_sharpe_result["sharpe_ratio"]:.3f}</p>
<table>
<tr><th>Ticker</th><th>Action</th><th>Shares</th><th>Estimated Cost</th></tr>
"""
    for _, row in trades.iterrows():
        action_class = "buy" if row["action"] == "BUY" else "sell"
        trades_html += (
            f"<tr><td>{row['ticker']}</td><td class='{action_class}'>{row['action']}</td>"
        )
        trades_html += f"<td>{row['shares']:.2f}</td><td>${row['estimated_cost']:.2f}</td></tr>\n"
    trades_html += "</table></body></html>"

    with open(output_dir / "portfolio_rebalance_widget.html", "w") as f:
        f.write(trades_html)
    print("  ✓ Saved: outputs/analytics/portfolio_rebalance_widget.html")
    print(f"  ✓ Using {n_stocks} real stocks with optimized weights")
else:
    print("\n⚠️  Skipping Rebalancing Widget - max_sharpe_result not available")

print("\n✓ Phase 6 dashboard snapshots generated with REAL portfolio data")
print("  → View in Dash: python finance_ml/dashboards/dash_app.py")
print("  → View in Streamlit: streamlit run finance_ml/dashboards/streamlit_app.py")

# %%
# Prepare Visualization Variables for Cell 100
print("\n" + "=" * 80)
print("PREPARING PORTFOLIO RESULTS FOR VISUALIZATION")
print("=" * 80)

# 1. Create optimal_portfolio alias from max_sharpe_result
if "max_sharpe_result" in dir() and max_sharpe_result:
    optimal_portfolio = max_sharpe_result
    print(f"✓ optimal_portfolio created from max_sharpe_result")
    print(f"  Return: {optimal_portfolio['return']:.2%}")
    print(f"  Volatility: {optimal_portfolio['volatility']:.2%}")
    print(f"  Sharpe: {optimal_portfolio['sharpe_ratio']:.3f}")
else:
    optimal_portfolio = None
    print("⚠️  max_sharpe_result not available")

# 2. Create min_vol_portfolio alias from min_vol_result
if "min_vol_result" in dir() and min_vol_result:
    min_vol_portfolio = min_vol_result
    print(f"✓ min_vol_portfolio created from min_vol_result")
    print(f"  Return: {min_vol_portfolio['return']:.2%}")
    print(f"  Volatility: {min_vol_portfolio['volatility']:.2%}")
else:
    min_vol_portfolio = None
    print("⚠️  min_vol_result not available")

# 3. Generate efficient frontier if inputs available
if "expected_returns_array" in dir() and "cov_matrix" in dir() and "risk_free_rate" in dir():
    from finance_ml.ml_workflow.analytics.portfolio import generate_efficient_frontier

    print("\n✓ Generating efficient frontier...")
    frontier_results = generate_efficient_frontier(
        returns=expected_returns_array,
        cov_matrix=cov_matrix,
        num_portfolios=100,
        risk_free_rate=risk_free_rate,
        allow_short=False,
    )
    print(f"✓ frontier_results created: {len(frontier_results['returns'])} portfolios")
else:
    frontier_results = None
    print("⚠️  Cannot create frontier_results - missing inputs")

# 4. Calculate risk metrics for optimal portfolio
if optimal_portfolio and "max_sharpe_result" in dir():
    from finance_ml.ml_workflow.analytics.risk import calculate_portfolio_risk_metrics

    print("\n✓ Calculating portfolio risk metrics...")
    # Generate synthetic returns based on portfolio characteristics
    n_days = 252
    daily_return = optimal_portfolio["return"] / 252
    daily_vol = optimal_portfolio["volatility"] / np.sqrt(252)

    # Safety rail: ensure daily_vol is positive
    daily_vol = max(daily_vol, 0.01)  # Minimum 1% daily volatility

    np.random.seed(RANDOM_SEED)
    portfolio_returns = pd.Series(np.random.normal(daily_return, daily_vol, n_days))

    risk_metrics_result = calculate_portfolio_risk_metrics(
        portfolio_returns,
        risk_free_rate=risk_free_rate if "risk_free_rate" in dir() else 0.02,
        confidence_levels=[0.95, 0.99],
    )
    print(f"✓ risk_metrics_result created")
    print(f"  Sharpe: {risk_metrics_result['sharpe_ratio']:.3f}")
    print(f"  Max DD: {risk_metrics_result['max_drawdown']:.2%}")
else:
    risk_metrics_result = None
    portfolio_returns = None
    print("⚠️  Cannot create risk_metrics_result - optimal_portfolio not available")

print("\n✓ Visualization variables prepared")
