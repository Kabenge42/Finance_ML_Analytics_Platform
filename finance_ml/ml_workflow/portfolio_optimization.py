"""
Finance ML Portfolio Optimization Module

DEPRECATED: This module has been moved to finance_ml.ml_workflow.analytics.portfolio
as part of Phase 9.7 refactoring. Please update your imports:

    from finance_ml.ml_workflow.analytics.portfolio import (
        calculate_portfolio_return,
        calculate_portfolio_volatility,
        calculate_portfolio_sharpe_ratio,
        validate_weights,
        generate_efficient_frontier,
        optimize_portfolio_max_sharpe,
        optimize_portfolio_min_volatility,
        optimize_portfolio_target_return,
        rebalance_portfolio,
    )

This compatibility wrapper will be removed in a future release.
"""

import warnings

# Re-export all functions from new location
from finance_ml.ml_workflow.analytics.portfolio import *

# Issue deprecation warning when module is imported
warnings.warn(
    "finance_ml.ml_workflow.portfolio_optimization is deprecated and will be removed in a future release. "
    "Please use finance_ml.ml_workflow.analytics.portfolio instead.",
    DeprecationWarning,
    stacklevel=2,
)
