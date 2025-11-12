"""
Finance ML Analyst Comparison Module

DEPRECATED: This module has been moved to finance_ml.ml_workflow.analytics.analyst_comparison
as part of Phase 9.7 refactoring. Please update your imports:

    from finance_ml.ml_workflow.analytics.analyst_comparison import (
        PredictionAnalystAnalytics,
        compare_prediction_vs_analyst_targets,
        calculate_agreement_rate,
        calculate_directional_accuracy,
        analyze_systematic_bias,
        identify_disagreement_opportunities,
        segment_comparison_by_attribute,
        generate_prediction_analyst_excel_report,
    )

This compatibility wrapper will be removed in a future release.
"""

import warnings

# Re-export all functions and classes from new location
from finance_ml.ml_workflow.analytics.analyst_comparison import (
    PredictionAnalystAnalytics,
    compare_prediction_vs_analyst_targets,
    calculate_agreement_rate,
    calculate_directional_accuracy,
    analyze_systematic_bias,
    identify_disagreement_opportunities,
    segment_comparison_by_attribute,
    generate_prediction_analyst_excel_report,
)

# Issue deprecation warning when module is imported
warnings.warn(
    "finance_ml.ml_workflow.analyst_comparison is deprecated and will be removed in a future release. "
    "Please use finance_ml.ml_workflow.analytics.analyst_comparison instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PredictionAnalystAnalytics",
    "compare_prediction_vs_analyst_targets",
    "calculate_agreement_rate",
    "calculate_directional_accuracy",
    "analyze_systematic_bias",
    "identify_disagreement_opportunities",
    "segment_comparison_by_attribute",
    "generate_prediction_analyst_excel_report",
]
