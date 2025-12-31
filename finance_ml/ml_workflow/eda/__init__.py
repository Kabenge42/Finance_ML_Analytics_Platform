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

from .analyst_analytics import (
    run_analyst_recommendations_analytics,
    find_rating_columns,
    analyze_sector_analyst_data,
    compute_upside_stats,
    RATING_COL_PATTERNS,
)
from .earnings_analytics import (
    find_available_columns,
    compute_earnings_surprise,
    compute_metric_statistics,
    analyze_segment,
    EARNINGS_COL_PATTERNS,
)
from .visualization_helpers import (
    load_json_file,
    save_and_display_figure,
    create_hypothesis_heatmap,
    create_regional_radar,
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
    # Analyst analytics
    "run_analyst_recommendations_analytics",
    "find_rating_columns",
    "analyze_sector_analyst_data",
    "RATING_COL_PATTERNS",
    # Earnings analytics
    "find_available_columns",
    "compute_earnings_surprise",
    "analyze_segment",
    "EARNINGS_COL_PATTERNS",
    # Visualization helpers
    "load_json_file",
    "save_and_display_figure",
    "create_hypothesis_heatmap",
    "create_regional_radar",
]
