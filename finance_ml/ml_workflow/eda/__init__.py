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
    # Aliases for code_guidelines.md v1.10 API
    compute_descriptive_stats,
    plot_distributions,
    compute_correlation_matrix,
)
from finance_ml.ml_workflow.eda.reports import (
    generate_eda_report,
)

# Phase 9.2: Hypothesis Testing (re-exported from evaluation for convenience)
from finance_ml.ml_workflow.evaluation.hypothesis import (
    perform_comprehensive_hypothesis_tests,
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
    # Aliases for code_guidelines.md v1.10 API
    "compute_descriptive_stats",
    "plot_distributions",
    "compute_correlation_matrix",
    # Report functions
    "generate_eda_report",
    # Phase 9.2: Hypothesis Testing
    "perform_comprehensive_hypothesis_tests",
]
