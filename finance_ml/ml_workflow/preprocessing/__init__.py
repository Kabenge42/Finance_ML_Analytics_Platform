"""
Preprocessing subpackage for finance_ml.ml_workflow.

This package provides comprehensive data preprocessing functionality:
- data: Data loading, normalization, and validation functions
- imputation: 6-step imputation strategy (zero, KNN, price, median)
- outliers: Outlier detection and handling (IQR, Z-score, Isolation Forest)
- scaling: Feature scaling with sector awareness
- quality: Data quality assessment and scoring
- pipeline: High-level preprocessing orchestration

Phase 9.1 refactor: Extracted from advanced_preprocessing.py for better modularity.
"""

# Data loading and validation functions
from finance_ml.ml_workflow.preprocessing.data import (
    setup_logging,
    get_env,
    normalize_columns,
    infer_region_from_filename,
    load_from_csv,
    load_from_db,
    load_from_all_stocks,
    preprocess,
    validate_schema,
    check_missing_values,
    detect_outliers_iqr as detect_outliers_iqr_data,
    validate_numeric_ranges,
    create_sample_financial_dataset,
    validate_financial_data_quality,
    sanitize_dataframe_with_logging,
    perform_early_pipeline_validation,
    _safe_div as _data_safe_div,
    # Phase 9.1: Advanced preprocessing functions from data module
    detect_outliers_iqr_advanced,
    detect_outliers_by_sector,
    detect_outliers_zscore as detect_outliers_zscore_data,
    winsorize_column,
    winsorize_by_sector as winsorize_by_sector_data,
    calculate_completeness_score,
    calculate_consistency_score,
    impute_by_sector,
    safe_divide,
    create_temporal_split,
    create_expanding_windows,
)
from finance_ml.ml_workflow.preprocessing.dtypes import (
    detect_and_cast_dtypes,
    to_jsonable,
    validate_dtypes_against_schema,
)
from finance_ml.ml_workflow.preprocessing.imputation import (
    get_zero_imputation_columns,
    get_knn_imputation_columns,
    impute_missing_values_knn_sector,
    apply_zero_imputation,
    apply_knn_imputation_enhanced,
    apply_price_imputation,
    apply_median_imputation,
    apply_enhanced_imputation_strategy_4step,
    apply_enhanced_imputation_strategy_6step,
    fillna_by_dtype,
)
from finance_ml.ml_workflow.preprocessing.outliers import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    winsorize_by_sector,
)
from finance_ml.ml_workflow.preprocessing.pipeline import (
    prepare_phase91_data,
)
from finance_ml.ml_workflow.preprocessing.quality import (
    DataQualityReport,
    calculate_data_quality_score,
)
from finance_ml.ml_workflow.preprocessing.scaling import (
    create_scaler_pipeline,
    scale_features,
)

__all__ = [
    # Data loading and validation
    "setup_logging",
    "get_env",
    "normalize_columns",
    "infer_region_from_filename",
    "load_from_csv",
    "load_from_db",
    "load_from_all_stocks",
    "preprocess",
    "validate_schema",
    "check_missing_values",
    "validate_numeric_ranges",
    "create_sample_financial_dataset",
    "validate_financial_data_quality",
    "sanitize_dataframe_with_logging",
    "perform_early_pipeline_validation",
    "_data_safe_div",
    # Advanced preprocessing from data module
    "detect_outliers_iqr_advanced",
    "detect_outliers_by_sector",
    "winsorize_column",
    "calculate_completeness_score",
    "calculate_consistency_score",
    "impute_by_sector",
    "safe_divide",
    "create_temporal_split",
    "create_expanding_windows",
    # Imputation column lists
    "get_zero_imputation_columns",
    "get_knn_imputation_columns",
    # Imputation functions
    "impute_missing_values_knn_sector",
    "apply_zero_imputation",
    "apply_knn_imputation_enhanced",
    "apply_price_imputation",
    "apply_median_imputation",
    "apply_enhanced_imputation_strategy_4step",
    "apply_enhanced_imputation_strategy_6step",
    "fillna_by_dtype",
    # Outlier detection
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "detect_outliers_isolation_forest",
    "winsorize_by_sector",
    # Scaling
    "create_scaler_pipeline",
    "scale_features",
    # Quality
    "DataQualityReport",
    "calculate_data_quality_score",
    # Pipeline
    "prepare_phase91_data",
    # Dtypes (Phase 9.9)
    "detect_and_cast_dtypes",
    "to_jsonable",
    "validate_dtypes_against_schema",
]
