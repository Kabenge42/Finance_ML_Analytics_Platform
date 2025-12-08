"""
Reporting and dashboard data preparation.

This package provides reporting tools including:
- Dashboard data preparation for Streamlit/Dash
- Financial metrics calculation
- Data quality monitoring
- Export functions for predictions and results

Phase 9.8 - Reporting Refactor

Usage:
    from finance_ml.ml_workflow.reporting import (
        calculate_financial_metrics_dashboard,
        generate_data_quality_alerts,
        prepare_plotly_dashboard_data,
        export_predictions,
        export_model_results,
        create_summary_report,
    )
"""

from finance_ml.ml_workflow.reporting.dashboard_data import (
    calculate_financial_metrics_dashboard,
    generate_data_quality_alerts,
    prepare_plotly_dashboard_data,
)

from finance_ml.ml_workflow.reporting.export import (
    export_predictions,
    export_model_results,
    create_summary_report,
)

# ============================================================================
# Aliases for code_guidelines.md v1.10 API compliance
# ============================================================================

# Alias: generate_dashboard_data -> prepare_plotly_dashboard_data
generate_dashboard_data = prepare_plotly_dashboard_data

# Alias: create_quality_alerts -> generate_data_quality_alerts
create_quality_alerts = generate_data_quality_alerts

# ============================================================================
# Prefixed aliases for notebook compatibility (reporting_* pattern)
# ============================================================================
reporting_financial_metrics = calculate_financial_metrics_dashboard
reporting_quality_alerts = generate_data_quality_alerts

__all__ = [
    # Dashboard functions
    "calculate_financial_metrics_dashboard",
    "generate_data_quality_alerts",
    "prepare_plotly_dashboard_data",
    # Aliases for code_guidelines.md v1.10 API
    "generate_dashboard_data",
    "create_quality_alerts",
    # Prefixed aliases for notebook compatibility
    "reporting_financial_metrics",
    "reporting_quality_alerts",
    # Export functions (Phase 9.8)
    "export_predictions",
    "export_model_results",
    "create_summary_report",
]
