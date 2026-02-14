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

# Shared utilities (constants, helpers, column resolver)
from ._shared import (
    PLOTLY_TEMPLATE,
    COLORS,
    MV_COLUMN_ALIASES,
    resolve_column,
    create_no_data_figure,
)

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

# Import valuation visualization functions
try:
    from .valuation import (
        create_valuation_multiples_comparison,
        create_valuation_distribution_dashboard,
        create_relative_valuation_matrix,
        create_valuation_vs_growth_quadrant,
        create_historical_valuation_percentile,
    )

    _valuation_exports = [
        "create_valuation_multiples_comparison",
        "create_valuation_distribution_dashboard",
        "create_relative_valuation_matrix",
        "create_valuation_vs_growth_quadrant",
        "create_historical_valuation_percentile",
    ]
except ImportError:
    _valuation_exports = []

# Import earnings quality visualization functions
try:
    from .earnings_quality import (
        create_earnings_surprise_dashboard,
        create_eps_trajectory_analysis,
        create_earnings_quality_decomposition,
        create_beat_rate_heatmap,
        create_earnings_consistency_matrix,
        create_revision_momentum_chart,
        create_gaap_divergence_plot,
        create_enhanced_beat_probability_dashboard,
    )

    _earnings_quality_exports = [
        "create_earnings_surprise_dashboard",
        "create_eps_trajectory_analysis",
        "create_earnings_quality_decomposition",
        "create_beat_rate_heatmap",
        "create_earnings_consistency_matrix",
        "create_revision_momentum_chart",
        "create_gaap_divergence_plot",
        "create_enhanced_beat_probability_dashboard",
    ]
except ImportError:
    _earnings_quality_exports = []

# Import quality risk visualization functions
try:
    from .quality_risk import (
        create_piotroski_fscore_breakdown,
        create_altman_zscore_distribution,
        create_quality_risk_quadrant,
        create_beneish_mscore_analysis,
        create_risk_tier_sunburst,
        create_distress_early_warning_dashboard,
    )

    _quality_risk_exports = [
        "create_piotroski_fscore_breakdown",
        "create_altman_zscore_distribution",
        "create_quality_risk_quadrant",
        "create_beneish_mscore_analysis",
        "create_risk_tier_sunburst",
        "create_distress_early_warning_dashboard",
    ]
except ImportError:
    _quality_risk_exports = []

# Import growth analysis visualization functions
try:
    from .growth_analysis import (
        create_growth_waterfall_chart,
        create_growth_consistency_matrix,
        create_growth_vs_profitability_quadrant,
        create_growth_acceleration_chart,
        create_sustainable_growth_analysis,
    )

    _growth_analysis_exports = [
        "create_growth_waterfall_chart",
        "create_growth_consistency_matrix",
        "create_growth_vs_profitability_quadrant",
        "create_growth_acceleration_chart",
        "create_sustainable_growth_analysis",
    ]
except ImportError:
    _growth_analysis_exports = []

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

# Import probabilistic ArviZ-backed visualization functions
try:
    from .probability_viz import (
        create_posterior_return_forest,
        create_beat_probability_posterior,
        create_ruin_probability_diagnostic,
        create_mcse_convergence_panel,
        create_bayesian_category_ridge,
        create_tri_model_posterior_comparison,
    )

    _probability_viz_exports = [
        "create_posterior_return_forest",
        "create_beat_probability_posterior",
        "create_ruin_probability_diagnostic",
        "create_mcse_convergence_panel",
        "create_bayesian_category_ridge",
        "create_tri_model_posterior_comparison",
    ]
except ImportError:
    _probability_viz_exports = []

# Shared exports
_shared_exports = [
    "PLOTLY_TEMPLATE",
    "COLORS",
    "MV_COLUMN_ALIASES",
    "resolve_column",
    "create_no_data_figure",
]

# Combine all exports
__all__ = (
    _shared_exports
    + _parent_exports
    + _profitability_exports
    + _technical_exports
    + _temporal_exports
    + _valuation_exports
    + _earnings_quality_exports
    + _quality_risk_exports
    + _growth_analysis_exports
    + _category_exports
    + _probability_viz_exports
)
