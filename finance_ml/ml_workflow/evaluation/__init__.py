"""
Evaluation metrics and analysis for regression models.

This package provides comprehensive evaluation tools including:
- Regression metrics (MAE, RMSE, R², MAPE)
- Segment-specific analysis (by sector, region)
- Model performance tracking
- Residual analysis and error diagnostics

Phase 9.6 - Evaluation Refactor

Usage:
    from finance_ml.ml_workflow.evaluation import (
        comprehensive_regression_metrics,
        compute_metrics_by_segment,
        compute_sector_region_metrics,
        residual_analysis,
        error_analysis,
        model_diagnostics,
    )
"""

from finance_ml.ml_workflow.evaluation.metrics import (
    comprehensive_regression_metrics,
    compute_metrics_by_segment,
    compute_sector_region_metrics,
)

from finance_ml.ml_workflow.evaluation.analysis import (
    residual_analysis,
    error_analysis,
    model_diagnostics,
    prediction_intervals,
    cross_validation_analysis,
)

__all__ = [
    # Metrics functions
    "comprehensive_regression_metrics",
    "compute_metrics_by_segment",
    "compute_sector_region_metrics",
    # Analysis functions
    "residual_analysis",
    "error_analysis",
    "model_diagnostics",
    "prediction_intervals",
    "cross_validation_analysis",
]
