"""
Hypothesis testing and statistical diagnostics (facade during eval decomposition).

Re-exports hypothesis/statistical helper functions from analytics.eval to provide
stable import paths under finance_ml.ml_workflow.evaluation.

Public functions:
- test_normality
- calculate_skewness_kurtosis
- detect_outliers_statistical
- compare_two_groups
- compare_sector_means
- perform_comprehensive_hypothesis_tests
"""

from finance_ml.ml_workflow.analytics.eval import (
    test_normality,
    calculate_skewness_kurtosis,
    detect_outliers_statistical,
    compare_two_groups,
    compare_sector_means,
    perform_comprehensive_hypothesis_tests,
)

__all__ = [
    "test_normality",
    "calculate_skewness_kurtosis",
    "detect_outliers_statistical",
    "compare_two_groups",
    "compare_sector_means",
    "perform_comprehensive_hypothesis_tests",
]
