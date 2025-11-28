"""Compatibility wrapper for risk metrics functions.

Historically, risk utilities lived in :mod:`finance_ml.risk_metrics` and,
during the v9_8 refactor, were moved under
``finance_ml.ml_workflow.analytics.risk``.

This module re-exports the public risk metrics API so that existing imports
such as::

    from finance_ml.ml_workflow.risk_metrics import (
        calculate_var_historical,
        calculate_sharpe_ratio,
        ...,
    )

continue to work without modification.

.. deprecated:: v9_8
   This module is kept for backward compatibility only. New code should import
   from :mod:`finance_ml.ml_workflow.analytics.risk` instead.

All implementations live in :mod:`finance_ml.ml_workflow.analytics.risk`.
"""

from __future__ import annotations

import warnings

from .analytics.risk import (
    calculate_var_historical,
    calculate_var_parametric,
    calculate_cvar,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_portfolio_risk_metrics,
    calculate_expected_shortfall,
    calculate_tracking_error,
    run_stress_tests,
    run_monte_carlo_simulation,
)


warnings.warn(
    "DEPRECATION NOTICE: 'finance_ml.ml_workflow.risk_metrics' has been "
    "consolidated into 'finance_ml.ml_workflow.analytics.risk'. Import from "
    "'finance_ml.ml_workflow.analytics.risk' instead. This shim will be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "calculate_var_historical",
    "calculate_var_parametric",
    "calculate_cvar",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_portfolio_risk_metrics",
    "calculate_expected_shortfall",
    "calculate_tracking_error",
    "run_stress_tests",
    "run_monte_carlo_simulation",
]
