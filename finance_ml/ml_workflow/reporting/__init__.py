"""
Reporting and dashboard data preparation.

This package provides reporting tools including:
- Dashboard data preparation for Streamlit/Dash
- Financial metrics calculation
- Data quality monitoring

Phase 9.8 - Reporting Refactor

Usage:
    from finance_ml.ml_workflow.reporting import (
        calculate_financial_metrics_dashboard,
        generate_data_quality_alerts,
        prepare_plotly_dashboard_data
    )
"""

from finance_ml.ml_workflow.reporting.dashboard_data import (
    calculate_financial_metrics_dashboard,
    generate_data_quality_alerts,
    prepare_plotly_dashboard_data,
)

__all__ = [
    "calculate_financial_metrics_dashboard",
    "generate_data_quality_alerts",
    "prepare_plotly_dashboard_data",
]
