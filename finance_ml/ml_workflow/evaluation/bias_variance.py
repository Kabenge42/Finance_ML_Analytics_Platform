"""Bias-variance analysis (facade).

Target-architecture module that re-exports bias/variance utilities from
``finance_ml.ml_workflow.analytics.eval``. This avoids behavior changes while
``analytics.eval`` is being decomposed.
"""

from __future__ import annotations

from finance_ml.ml_workflow.analytics.eval import (  # noqa: E402
    diagnose_bias_variance,
    bias_variance_decomposition,
    plot_bias_variance,
)

__all__ = [
    "diagnose_bias_variance",
    "bias_variance_decomposition",
    "plot_bias_variance",
]
