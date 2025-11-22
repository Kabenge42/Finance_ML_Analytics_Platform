"""
Evaluation metrics and analysis for regression models.

This package provides comprehensive evaluation tools including:
- Regression metrics (MAE, RMSE, R², MAPE)
- Segment-specific analysis (by sector, region)
- Model performance tracking
- Residual analysis and error diagnostics
- Uncertainty quantification and conformal calibration (Phase 9.4)
- Safety rails monitoring (Phase 9.5)
- Data split and leakage validation (Phase 9.6)
- Sector bias calibration (Phase 9.7)
- Stacking ensemble diagnostics and governance (Phase 9.8)

Phase 9.6 - Evaluation Refactor
Phase 9.4-9.8 - Advanced Reporting and Governance

Usage:
    from finance_ml.ml_workflow.evaluation import (
        comprehensive_regression_metrics,
        compute_metrics_by_segment,
        compute_sector_region_metrics,
        residual_analysis,
        error_analysis,
        model_diagnostics,
        # Phase 9.4 - Uncertainty
        build_quantile_diagnostics,
        plot_interval_coverage,
        plot_reliability_diagram,
        # Phase 9.5 - Safety Rails
        summarize_winsorization_effects,
        track_constraint_violations,
        safety_rails_sensitivity_app,
        # Phase 9.6 - Splits & Leakage
        compute_fold_overlap,
        summarize_grouped_cv_balance,
        time_leakage_checks,
        # Phase 9.7 - Calibration
        estimate_sector_bias,
        plot_metrics_by_sector_time,
        create_sector_bias_dashboard,
        # Phase 9.8 - Stacking & Governance
        compute_stacking_contributions,
        meta_error_maps,
        generate_model_card,
        build_lineage_json,
    )
"""

from finance_ml.ml_workflow.evaluation.analysis import (
    residual_analysis,
    error_analysis,
    model_diagnostics,
    prediction_intervals,
    cross_validation_analysis,
)
from finance_ml.ml_workflow.evaluation.metrics import (
    comprehensive_regression_metrics,
    compute_metrics_by_segment,
    compute_sector_region_metrics,
)

# Phase 9.4 - Uncertainty Quantification & Conformal Calibration
from finance_ml.ml_workflow.evaluation.uncertainty import (
    build_quantile_diagnostics,
    plot_interval_coverage,
    plot_reliability_diagram,
)

# Phase 9.5 - Outlier Safety Rails & Non-Negative Constraints
from finance_ml.ml_workflow.evaluation.safety_rails import (
    summarize_winsorization_effects,
    track_constraint_violations,
    safety_rails_sensitivity_app,
)

# Phase 9.6 - Data Split and Leakage Policy Validation
from finance_ml.ml_workflow.evaluation.splits import (
    compute_fold_overlap,
    summarize_grouped_cv_balance,
    time_leakage_checks,
)

# Phase 9.7 - Sector Bias Calibration & Metrics Persistence
from finance_ml.ml_workflow.evaluation.calibration import (
    estimate_sector_bias,
    plot_metrics_by_sector_time,
    create_sector_bias_dashboard,
)

# Phase 9.8 - Stacking Ensemble Diagnostics & Model Governance
from finance_ml.ml_workflow.evaluation.stacking import (
    compute_stacking_contributions,
    meta_error_maps,
    generate_model_card,
    build_lineage_json,
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
    # Phase 9.4 - Uncertainty Quantification
    "build_quantile_diagnostics",
    "plot_interval_coverage",
    "plot_reliability_diagram",
    # Phase 9.5 - Safety Rails
    "summarize_winsorization_effects",
    "track_constraint_violations",
    "safety_rails_sensitivity_app",
    # Phase 9.6 - Data Splits & Leakage
    "compute_fold_overlap",
    "summarize_grouped_cv_balance",
    "time_leakage_checks",
    # Phase 9.7 - Sector Bias Calibration
    "estimate_sector_bias",
    "plot_metrics_by_sector_time",
    "create_sector_bias_dashboard",
    # Phase 9.8 - Stacking & Governance
    "compute_stacking_contributions",
    "meta_error_maps",
    "generate_model_card",
    "build_lineage_json",
]
