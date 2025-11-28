"""Hypothesis testing utilities (facade).

Target-architecture path that exposes hypothesis test helpers. The
implementations are currently hosted in ``analytics.eval`` and will be moved
incrementally. This module re-exports them to stabilize imports.
"""

from __future__ import annotations

from finance_ml.ml_workflow.analytics.eval import (  # noqa: E402
    perform_comprehensive_hypothesis_tests,
    test_market_efficiency_hypothesis,
    perform_time_series_hypothesis_tests,
    perform_multi_factor_anova,
)

__all__ = [
    "perform_comprehensive_hypothesis_tests",
    "test_market_efficiency_hypothesis",
    "perform_time_series_hypothesis_tests",
    "perform_multi_factor_anova",
]
