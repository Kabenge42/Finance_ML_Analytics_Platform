"""
Finance ML Benchmarking Module

DEPRECATED: This module has been moved to finance_ml.ml_workflow.eda.benchmarking
as part of Phase 9.2 refactoring. Please update your imports:

    from finance_ml.ml_workflow.eda.benchmarking import (
        compare_sector_distributions,
        compare_regional_valuations,
        find_peer_group,
        compare_to_peers,
        analyze_metric_trend,
        generate_benchmarking_report,
    )

This compatibility wrapper will be removed in a future release.
"""

import warnings

# Re-export all functions from new location
from finance_ml.ml_workflow.eda.benchmarking import (
    compare_sector_distributions,
    compare_regional_valuations,
    find_peer_group,
    compare_to_peers,
    analyze_metric_trend,
    generate_benchmarking_report,
)

# Issue deprecation warning when module is imported
warnings.warn(
    "finance_ml.ml_workflow.benchmarking is deprecated and will be removed in a future release. "
    "Please use finance_ml.ml_workflow.eda.benchmarking instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "compare_sector_distributions",
    "compare_regional_valuations",
    "find_peer_group",
    "compare_to_peers",
    "analyze_metric_trend",
    "generate_benchmarking_report",
]
