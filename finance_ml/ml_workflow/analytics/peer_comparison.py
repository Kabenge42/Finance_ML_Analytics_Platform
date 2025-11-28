"""Peer comparison analytics (facade).

Provides target-architecture import path for peer comparison helpers. Currently
re-exports implementations from ``analytics.eval`` to avoid behavior changes.
"""

from __future__ import annotations

from finance_ml.ml_workflow.analytics.eval import (  # noqa: E402
    calculate_peer_comparisons,
)

__all__ = [
    "calculate_peer_comparisons",
]
