"""
Finance ML Risk Metrics Module

DEPRECATED: This module has been moved to finance_ml.ml_workflow.analytics.risk
as part of Phase 9.7 refactoring. Please update your imports:

    from finance_ml.ml_workflow.analytics.risk import (
        calculate_var_historical,
        calculate_var_parametric,
        calculate_cvar,
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
        calculate_max_drawdown,
        calculate_portfolio_risk_metrics,
    )

This compatibility wrapper will be removed in a future release.
"""

import warnings

# Re-export all functions from new location
from finance_ml.ml_workflow.analytics.risk import *

# Issue deprecation warning when module is imported
warnings.warn(
    "finance_ml.ml_workflow.risk_metrics is deprecated and will be removed in a future release. "
    "Please use finance_ml.ml_workflow.analytics.risk instead.",
    DeprecationWarning,
    stacklevel=2,
)
