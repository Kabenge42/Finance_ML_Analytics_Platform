"""
Compatibility wrapper for risk metrics functions.

Historically, risk utilities lived in `finance_ml.risk_metrics`
and, during the v9_8 refactor, were moved under
`finance_ml.ml_workflow.analytics.risk`.

This module re-exports the public risk metrics API so that
existing imports such as::

    from finance_ml.ml_workflow.risk_metrics import (
        calculate_var_historical,
        calculate_sharpe_ratio,
        ...
    )

continue to work without modification.

All implementations live in `finance_ml.ml_workflow.analytics.risk`.
"""

from __future__ import annotations

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
