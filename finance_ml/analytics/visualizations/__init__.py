"""
Visualization modules for feature analytics.

This package provides visualization functions organized by category:
- Interactive dashboards (Plotly)
- Static charts (Matplotlib)
- Specialized analysis visualizations

Modules:
- profitability: Margin and profitability charts (DuPont, waterfall, quadrant)
- technical: Technical analysis charts (momentum ribbon, 52w range, divergence)
- temporal_analysis: Time series analysis (earnings calendar, inventory, FCF, dividends)

Most visualization functions are already available in the parent
feature_analytics module. This package can be extended with additional
specialized visualization functions as needed.
"""

from __future__ import annotations

# Import key visualization functions from parent module
try:
    from ..feature_analytics import (
        create_interactive_momentum_dashboard,
        create_interactive_valuation_heatmap,
        create_leverage_liquidity_quadrant,
        create_summary_dashboard,
        analyze_distress_distribution,
        create_composite_quality_score,
    )

    _parent_exports = [
        "create_interactive_momentum_dashboard",
        "create_interactive_valuation_heatmap",
        "create_leverage_liquidity_quadrant",
        "create_summary_dashboard",
        "analyze_distress_distribution",
        "create_composite_quality_score",
    ]
except ImportError:
    _parent_exports = []

# Import profitability visualization functions
try:
    from .profitability import (
        create_margin_waterfall_chart,
        create_dupont_decomposition_dashboard,
        create_profitability_quadrant,
        create_margin_trend_heatmap,
    )

    _profitability_exports = [
        "create_margin_waterfall_chart",
        "create_dupont_decomposition_dashboard",
        "create_profitability_quadrant",
        "create_margin_trend_heatmap",
    ]
except ImportError:
    _profitability_exports = []

# Import technical analysis visualization functions
try:
    from .technical import (
        create_momentum_ribbon_chart,
        create_52w_range_distribution,
        create_trend_strength_matrix,
        create_momentum_divergence_scatter,
    )

    _technical_exports = [
        "create_momentum_ribbon_chart",
        "create_52w_range_distribution",
        "create_trend_strength_matrix",
        "create_momentum_divergence_scatter",
    ]
except ImportError:
    _technical_exports = []

# Import temporal analysis visualization functions
try:
    from .temporal_analysis import (
        create_earnings_calendar_heatmap,
        create_inventory_cycle_analysis,
        create_fcf_trajectory_chart,
        create_dividend_streak_timeline,
    )

    _temporal_exports = [
        "create_earnings_calendar_heatmap",
        "create_inventory_cycle_analysis",
        "create_fcf_trajectory_chart",
        "create_dividend_streak_timeline",
    ]
except ImportError:
    _temporal_exports = []

# Import category-specific chart functions
try:
    from .category_charts import (
        # Analyst Sentiment
        create_analyst_sentiment_histogram,
        create_analyst_upside_scatter,
        # Earnings Quality
        create_eps_surprise_histogram,
        create_eps_trajectory_scatter,
        # Growth Metrics
        create_growth_correlation_heatmap,
        create_revenue_vs_eps_growth_scatter,
        # Cash Flow
        create_fcf_margin_yield_scatter,
        create_cash_flow_quality_boxplot,
        # Dividend Features
        create_dividend_yield_payout_scatter,
        create_shareholder_yield_histogram,
        # R&D Investment
        create_rnd_intensity_boxplot,
        create_rnd_intensity_growth_scatter,
        create_rnd_per_employee_histogram,
        # Inventory
        create_inventory_days_turnover_scatter,
        # Goodwill & M&A
        create_goodwill_concentration_boxplot,
        create_goodwill_impairment_scatter,
        create_acquisition_activity_histogram,
        # CapEx & Investment
        create_capex_growth_scatter,
        create_investment_efficiency_boxplot,
        create_ma_intensity_histogram,
        # Advanced/Multi-Category
        create_valuation_violin_plot,
        create_quality_risk_radar_chart,
        create_leverage_liquidity_bubble_chart,
    )

    _category_exports = [
        "create_analyst_sentiment_histogram",
        "create_analyst_upside_scatter",
        "create_eps_surprise_histogram",
        "create_eps_trajectory_scatter",
        "create_growth_correlation_heatmap",
        "create_revenue_vs_eps_growth_scatter",
        "create_fcf_margin_yield_scatter",
        "create_cash_flow_quality_boxplot",
        "create_dividend_yield_payout_scatter",
        "create_shareholder_yield_histogram",
        "create_rnd_intensity_boxplot",
        "create_rnd_intensity_growth_scatter",
        "create_rnd_per_employee_histogram",
        "create_inventory_days_turnover_scatter",
        "create_goodwill_concentration_boxplot",
        "create_goodwill_impairment_scatter",
        "create_acquisition_activity_histogram",
        "create_capex_growth_scatter",
        "create_investment_efficiency_boxplot",
        "create_ma_intensity_histogram",
        "create_valuation_violin_plot",
        "create_quality_risk_radar_chart",
        "create_leverage_liquidity_bubble_chart",
    ]
except ImportError:
    _category_exports = []

# Combine all exports
__all__ = (
    _parent_exports
    + _profitability_exports
    + _technical_exports
    + _temporal_exports
    + _category_exports
)
