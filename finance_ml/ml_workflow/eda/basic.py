"""Basic EDA utilities (thin facade during eval decomposition).

This module stabilizes imports for quick EDA helpers and now also exposes
``basic_describe`` from :mod:`finance_ml.ml_workflow.eda.descriptive` per the
EDA consolidation plan.

Public functions:
- basic_describe
- simple_eda
- create_sector_heatmap
- create_region_sector_heatmap
- create_interactive_prediction_plot
- plot_outlier_boxplots
- plot_outlier_violins
- plot_outlier_scatter
"""

from __future__ import annotations

# Thin wrapper import for descriptive stats
from finance_ml.ml_workflow.eda.descriptive import basic_describe  # noqa: F401

# Re-export quick EDA utilities (implementations currently in analytics.eval)
from finance_ml.ml_workflow.analytics.eval import (  # noqa: E402
    simple_eda,
    create_sector_heatmap,
    create_region_sector_heatmap,
    create_interactive_prediction_plot,
    plot_outlier_boxplots,
    plot_outlier_violins,
    plot_outlier_scatter,
)

__all__ = [
    "basic_describe",
    "simple_eda",
    "create_sector_heatmap",
    "create_region_sector_heatmap",
    "create_interactive_prediction_plot",
    "plot_outlier_boxplots",
    "plot_outlier_violins",
    "plot_outlier_scatter",
]
