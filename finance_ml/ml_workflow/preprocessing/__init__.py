"""
Preprocessing subpackage for finance_ml.ml_workflow.

This package provides comprehensive data preprocessing functionality:
- imputation: 4-step imputation strategy (zero, KNN, price, median)
- outliers: Outlier detection and handling (IQR, Z-score, Isolation Forest)
- scaling: Feature scaling with sector awareness
- quality: Data quality assessment and scoring
- pipeline: High-level preprocessing orchestration

Phase 9.1 refactor: Extracted from advanced_preprocessing.py for better modularity.
"""

from finance_ml.ml_workflow.preprocessing.imputation import (
    get_zero_imputation_columns,
    get_knn_imputation_columns,
    impute_missing_values_knn_sector,
    apply_zero_imputation,
    apply_knn_imputation_enhanced,
    apply_price_imputation,
    apply_median_imputation,
    apply_enhanced_imputation_strategy_4step,
)

from finance_ml.ml_workflow.preprocessing.outliers import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    winsorize_by_sector,
)

from finance_ml.ml_workflow.preprocessing.scaling import (
    create_scaler_pipeline,
    scale_features,
)

from finance_ml.ml_workflow.preprocessing.quality import (
    DataQualityReport,
    calculate_data_quality_score,
)

from finance_ml.ml_workflow.preprocessing.pipeline import (
    prepare_phase91_data,
)

__all__ = [
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
]
