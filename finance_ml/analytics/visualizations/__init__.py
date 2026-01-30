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

# Combine all exports
__all__ = _parent_exports + _profitability_exports + _technical_exports + _temporal_exports
