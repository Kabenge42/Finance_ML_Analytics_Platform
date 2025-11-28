"""Deprecation shim for advanced EDA utilities.

This module has been moved under finance_ml.ml_workflow. Import from
`finance_ml.ml_workflow.eda` or `finance_ml.ml_workflow.advanced_eda`.

This shim re-exports commonly used functions for backward compatibility and
emits a DeprecationWarning on import. It also provides lightweight aliases for
older function names used by tests (e.g., find_high_correlations -> wrapper
around find_top_correlations; calculate_distribution_stats ->
calculate_skewness_kurtosis).
"""

from __future__ import annotations

import warnings
from typing import Optional, List

import pandas as pd

# Warn at import time
warnings.warn(
    "DEPRECATION NOTICE: 'finance_ml.advanced_eda' has moved under "
    "'finance_ml.ml_workflow.advanced_eda' and 'finance_ml.ml_workflow.eda'. "
    "Please update imports. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Import the modern implementation
from finance_ml.ml_workflow.advanced_eda import (  # noqa: E402
    CorrelationReport,
    StatisticalTestResult,
    EDAReport,
    calculate_correlation_matrix,
    find_top_correlations,
    test_normality,
    calculate_skewness_kurtosis,
    detect_outliers_statistical,
    calculate_mutual_information,
    calculate_feature_importance_rf,
    perform_pca,
    calculate_optimal_pca_components,
    compare_sector_means,
    compare_two_groups,
    generate_eda_report,
    generate_sector_comparison_report,
)


def find_high_correlations(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compatibility wrapper returning pairs above a threshold.

    Builds a Pearson correlation matrix (unless columns already numeric subset
    is provided), then returns a DataFrame with columns [feature_1, feature_2,
    correlation] filtered by absolute value >= threshold.
    """
    # Use numeric columns by default
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    corr = calculate_correlation_matrix(df, method="pearson", columns=columns)
    # Use the helper to find top correlations then filter by threshold
    top_pos_neg = find_top_correlations(corr, n_top=corr.size)
    # top_pos_neg may be typed as tuple(list, list) in some versions
    if isinstance(top_pos_neg, tuple):
        pairs = list(top_pos_neg[0]) + list(top_pos_neg[1])
    else:
        pairs = list(top_pos_neg)
    records = [
        {"feature_1": a, "feature_2": b, "correlation": float(c)}
        for (a, b, c) in pairs
        if abs(float(c)) >= float(threshold)
    ]
    return pd.DataFrame.from_records(
        records, columns=["feature_1", "feature_2", "correlation"]
    ).reset_index(drop=True)


def calculate_distribution_stats(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compatibility alias for calculate_skewness_kurtosis."""
    return calculate_skewness_kurtosis(df, columns=columns)


__all__ = [
    # Data classes
    "CorrelationReport",
    "StatisticalTestResult",
    "EDAReport",
    # Primary functions
    "calculate_correlation_matrix",
    "find_top_correlations",
    "test_normality",
    "calculate_skewness_kurtosis",
    "detect_outliers_statistical",
    "calculate_mutual_information",
    "calculate_feature_importance_rf",
    "perform_pca",
    "calculate_optimal_pca_components",
    "compare_sector_means",
    "compare_two_groups",
    "generate_eda_report",
    "generate_sector_comparison_report",
    # Compatibility helpers
    "find_high_correlations",
    "calculate_distribution_stats",
]
