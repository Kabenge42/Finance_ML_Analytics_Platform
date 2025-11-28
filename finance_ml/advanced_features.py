"""Deprecation shim for advanced feature engineering.

This legacy module has been decomposed into the structured features
subpackage. Please import from:

    finance_ml.ml_workflow.features.advanced
    finance_ml.ml_workflow.features.core
    finance_ml.ml_workflow.features.selection

This shim emits a DeprecationWarning and re-exports commonly used helpers.
"""

from __future__ import annotations

import warnings


warnings.warn(
    "DEPRECATION NOTICE: 'finance_ml.advanced_features' has moved to "
    "'finance_ml.ml_workflow.features'. Import from advanced/core/selection "
    "modules directly.",
    DeprecationWarning,
    stacklevel=2,
)

from finance_ml.ml_workflow.features.advanced import (  # noqa: E402
    build_comprehensive_features,
    engineer_valuation_ratios,
    engineer_profitability_ratios,
    engineer_growth_metrics,
    engineer_momentum_features,
    engineer_technical_analysis_features,
    engineer_valuation_timeseries_features,
    engineer_revenue_forecast_features,
    engineer_dividend_reliability_features,
    engineer_employment_dynamics_features,
)
from finance_ml.ml_workflow.features.advanced import (  # noqa: E402
    create_feature_interactions,
    create_relative_value_features,
)
from finance_ml.ml_workflow.features.selection import (  # noqa: E402
    calculate_feature_importance_mutual_info,
    calculate_feature_importance_rf,
    calculate_feature_importance_shap,
)


def select_top_k_features(X, y, method: str = "mutual_info", top_k: int | None = None):
    """Compatibility helper to select top-k features by importance.

    Parameters:
    - method: one of {'mutual_info', 'rf', 'shap'}
    - top_k: optional number of top features to return

    Returns a pandas Series of importances indexed by feature name.
    """
    method = (method or "mutual_info").lower()
    if method in {"mutual_info", "mi"}:
        return calculate_feature_importance_mutual_info(X, y, top_k=top_k)
    if method in {"rf", "random_forest"}:
        return calculate_feature_importance_rf(X, y, top_k=top_k)
    if method == "shap":
        return calculate_feature_importance_shap(X, y, top_k=top_k)
    raise ValueError(f"Unknown feature selection method: {method}")


__all__ = [
    "build_comprehensive_features",
    "engineer_valuation_ratios",
    "engineer_profitability_ratios",
    "engineer_growth_metrics",
    "engineer_momentum_features",
    "engineer_technical_analysis_features",
    "engineer_valuation_timeseries_features",
    "engineer_revenue_forecast_features",
    "engineer_dividend_reliability_features",
    "engineer_employment_dynamics_features",
    "create_feature_interactions",
    "create_relative_value_features",
    "select_top_k_features",
]
