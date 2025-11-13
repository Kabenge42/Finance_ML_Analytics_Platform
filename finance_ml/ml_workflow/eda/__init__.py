"""
Phase 9.2: EDA and Benchmarking Module

This package provides exploratory data analysis and benchmarking utilities:
- eda.py: Quick summaries, distributions, correlations, sector slices
- benchmarking.py: Sector and regional comparisons, peer analysis
- reports.py: HTML/static report generation
"""

from finance_ml.ml_workflow.eda.benchmarking import (
    compare_sector_distributions,
    compare_regional_valuations,
    find_peer_group,
    compare_to_peers,
    analyze_metric_trend,
    generate_benchmarking_report,
)

from finance_ml.ml_workflow.eda.eda import (
    eda_summary,
    sector_distribution_summary,
    correlation_analysis,
    distribution_summary,
)
from finance_ml.ml_workflow.eda.reports import (
    generate_eda_report,
)

__all__ = [
    # Benchmarking functions
    "compare_sector_distributions",
    "compare_regional_valuations",
    "find_peer_group",
    "compare_to_peers",
    "analyze_metric_trend",
    "generate_benchmarking_report",
    # EDA functions
    "eda_summary",
    "sector_distribution_summary",
    "correlation_analysis",
    "distribution_summary",
    # Report functions
    "generate_eda_report",
]
