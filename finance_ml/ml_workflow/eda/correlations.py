"""
EDA correlations module (extracted facade).

Temporary re-export shim for correlation utilities while decomposing
analytics.eval into focused modules. Implementations are currently sourced
from finance_ml.ml_workflow.analytics.eval to avoid behavior changes.

Public functions:
- calculate_correlation_matrix
- calculate_distance_correlation
- find_top_correlations
"""

from finance_ml.ml_workflow.analytics.eval import (
    calculate_correlation_matrix,
    calculate_distance_correlation,
    find_top_correlations,
)

__all__ = [
    "calculate_correlation_matrix",
    "calculate_distance_correlation",
    "find_top_correlations",
]
