"""
DEPRECATED: Use finance_ml.dashboards.widgets instead.
This module is maintained for backward compatibility.
"""

import warnings

from finance_ml.dashboards.widgets import (
    EarningsMode,
    EarningsAlertConfig,
    get_category_metrics,
    create_earnings_calendar_dashboard,
    display_earnings_dashboard,
    create_earnings_metrics_chart,
    create_earnings_surprise_dashboard,
    create_earnings_calendar_analytics,
    analyze_earnings_quality,
    create_gaap_adjusted_comparison_chart,
    generate_earnings_quality_alerts,
    create_analyst_recommendation_heatmap,
    create_market_movers_dashboard,
    create_price_target_scorecard,
    create_dividend_sustainability_scorecard,
    create_employee_productivity_scorecard,
    create_category_comparison_chart,
    create_technical_valuation_dashboard,
    create_category_correlation_network,
)
from finance_ml.dashboards.widgets.base import (
    resolve_reference_date,
    add_formatted_date_columns,
    _write_html_artifact,
)

warnings.warn(
    "finance_ml.dashboards.earnings_widgets is deprecated. "
    "Please import from finance_ml.dashboards.widgets instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "EarningsMode",
    "EarningsAlertConfig",
    "get_category_metrics",
    "create_earnings_calendar_dashboard",
    "display_earnings_dashboard",
    "create_earnings_metrics_chart",
    "create_earnings_surprise_dashboard",
    "create_earnings_calendar_analytics",
    "analyze_earnings_quality",
    "create_gaap_adjusted_comparison_chart",
    "generate_earnings_quality_alerts",
    "create_analyst_recommendation_heatmap",
    "create_market_movers_dashboard",
    "create_price_target_scorecard",
    "create_dividend_sustainability_scorecard",
    "create_employee_productivity_scorecard",
    "create_category_comparison_chart",
    "create_technical_valuation_dashboard",
    "create_category_correlation_network",
    "resolve_reference_date",
    "add_formatted_date_columns",
    "_write_html_artifact",
]
