"""
Basic EDA utilities (facade during eval decomposition).

Re-exports quick EDA summary and plotting helpers from analytics.eval to provide
clean import paths under finance_ml.ml_workflow.eda.

Public functions:
- simple_eda
- create_sector_heatmap
- create_region_sector_heatmap
- create_interactive_prediction_plot
- plot_outlier_boxplots
- plot_outlier_violins
- plot_outlier_scatter
"""

from finance_ml.ml_workflow.analytics.eval import (
    simple_eda,
    create_sector_heatmap,
    create_region_sector_heatmap,
    create_interactive_prediction_plot,
    plot_outlier_boxplots,
    plot_outlier_violins,
    plot_outlier_scatter,
)

__all__ = [
    "simple_eda",
    "create_sector_heatmap",
    "create_region_sector_heatmap",
    "create_interactive_prediction_plot",
    "plot_outlier_boxplots",
    "plot_outlier_violins",
    "plot_outlier_scatter",
]
