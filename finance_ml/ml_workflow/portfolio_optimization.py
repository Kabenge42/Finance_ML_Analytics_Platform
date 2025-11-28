"""Compatibility wrapper for portfolio optimization functions.

Historically, portfolio utilities lived in :mod:`finance_ml.portfolio_optimization`
and, during the v9_8 refactor, were moved under
``finance_ml.ml_workflow.analytics.portfolio``.

This module re-exports the public portfolio optimization API so that existing
imports such as::

    from finance_ml.ml_workflow.portfolio_optimization import (
        calculate_portfolio_return,
        optimize_portfolio_max_sharpe,
        ...,
    )

continue to work without modification.

.. deprecated:: v9_8
   This module is kept for backward compatibility only. New code should import
   from :mod:`finance_ml.ml_workflow.analytics.portfolio` instead.

All implementations live in :mod:`finance_ml.ml_workflow.analytics.portfolio`.
"""

from __future__ import annotations

import warnings

from .analytics.portfolio import (
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_portfolio_sharpe_ratio,
    validate_weights,
    generate_efficient_frontier,
    optimize_portfolio_max_sharpe,
    optimize_portfolio_min_volatility,
    optimize_portfolio_target_return,
    rebalance_portfolio,
    optimize_black_litterman,
    optimize_risk_parity,
    optimize_hrp,
    run_vectorized_backtest,
    run_walk_forward_optimization,
)


# Emit deprecation warning on import so callers are guided to the new module
warnings.warn(
    "DEPRECATION NOTICE: 'finance_ml.ml_workflow.portfolio_optimization' has been "
    "consolidated into 'finance_ml.ml_workflow.analytics.portfolio'. Import from "
    "'finance_ml.ml_workflow.analytics.portfolio' instead. This shim will be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "calculate_portfolio_return",
    "calculate_portfolio_volatility",
    "calculate_portfolio_sharpe_ratio",
    "validate_weights",
    "generate_efficient_frontier",
    "optimize_portfolio_max_sharpe",
    "optimize_portfolio_min_volatility",
    "optimize_portfolio_target_return",
    "rebalance_portfolio",
    "optimize_black_litterman",
    "optimize_risk_parity",
    "optimize_hrp",
    "run_vectorized_backtest",
    "run_walk_forward_optimization",
]
