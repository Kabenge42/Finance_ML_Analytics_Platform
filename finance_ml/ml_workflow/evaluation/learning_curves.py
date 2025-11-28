"""
Learning curves and bias-variance utilities (facade during eval decomposition).

Re-exports learning/validation curve helpers and bias-variance diagnostics from
analytics.eval to provide a clean import path under
finance_ml.ml_workflow.evaluation.

Public functions:
- generate_learning_curve
- plot_learning_curve
- generate_validation_curve
- plot_validation_curve
- diagnose_bias_variance
- bias_variance_decomposition
- plot_bias_variance
- identify_optimal_complexity
- create_expanding_window_cv
- create_rolling_window_cv
- create_stratified_sector_cv
- create_grouped_ticker_cv
- evaluate_with_cross_validation
"""

from finance_ml.ml_workflow.analytics.eval import (
    generate_learning_curve,
    plot_learning_curve,
    generate_validation_curve,
    plot_validation_curve,
    diagnose_bias_variance,
    bias_variance_decomposition,
    plot_bias_variance,
    identify_optimal_complexity,
    create_expanding_window_cv,
    create_rolling_window_cv,
    create_stratified_sector_cv,
    create_grouped_ticker_cv,
    evaluate_with_cross_validation,
)

__all__ = [
    "generate_learning_curve",
    "plot_learning_curve",
    "generate_validation_curve",
    "plot_validation_curve",
    "diagnose_bias_variance",
    "bias_variance_decomposition",
    "plot_bias_variance",
    "identify_optimal_complexity",
    "create_expanding_window_cv",
    "create_rolling_window_cv",
    "create_stratified_sector_cv",
    "create_grouped_ticker_cv",
    "evaluate_with_cross_validation",
]
