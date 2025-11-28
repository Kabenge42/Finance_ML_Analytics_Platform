"""
Reporting utilities (facade during eval decomposition).

Re-exports report generation helpers from analytics.eval to provide a clean
import path under finance_ml.ml_workflow.evaluation.

Public functions:
- generate_sector_comparison_report
- generate_data_quality_dashboard
- export_profiling_report
"""

from finance_ml.ml_workflow.analytics.eval import (
    generate_sector_comparison_report,
    generate_data_quality_dashboard,
    export_profiling_report,
)

__all__ = [
    "generate_sector_comparison_report",
    "generate_data_quality_dashboard",
    "export_profiling_report",
]
