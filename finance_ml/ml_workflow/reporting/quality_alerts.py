"""Data quality alerts reporting (facade).

Provides target-architecture import path for generating data quality alerts and
related reporting utilities. Re-exports the implementation from
``finance_ml.ml_workflow.analytics.eval``.
"""

from __future__ import annotations

from finance_ml.ml_workflow.analytics.eval import (  # noqa: E402
    generate_data_quality_alerts,
    generate_data_quality_dashboard,
)

__all__ = [
    "generate_data_quality_alerts",
    "generate_data_quality_dashboard",
]
