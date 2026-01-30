"""
Analytics module for feature engineering analysis.

This module provides interactive visualizations, probabilistic models,
and statistical analytics for financial feature analysis.

Modules:
- feature_analytics: Core visualization dashboards
- data_utils: Data loading and preprocessing
- statistical_analysis: Bayesian, MCMC, and advanced statistics
- screening: Stock screening and filtering
- optimized_ops: Performance-optimized operations
- visualizations: Additional visualization modules
  - profitability: Margin and profitability charts
  - technical: Technical analysis charts
  - temporal_analysis: Time series analysis
"""

from finance_ml.analytics.feature_analytics import (
    PLOTLY_TEMPLATE,
    analyze_distress_distribution,
    bayesian_earnings_beat_model,
    create_composite_quality_score,
    create_interactive_momentum_dashboard,
    create_interactive_valuation_heatmap,
    create_leverage_liquidity_quadrant,
    create_summary_dashboard,
    ensure_subplot_data,
    monte_carlo_price_target_simulation,
    safe_get_column,
)

# Data utilities
from finance_ml.analytics.data_utils import (
    load_feature_data_from_db,
    backfill_feature_columns,
    compute_metric_statistics,
    validate_feature_alignment,
)

# Statistical analysis
from finance_ml.analytics.statistical_analysis import (
    bayesian_category_analysis,
    metropolis_hastings_sampler,
    mcmc_student_t,
    hierarchical_mcmc_by_sector,
    fit_distributions_by_category,
    calculate_ruin_probability,
    calculate_conditional_probabilities,
    # Enhanced methods
    kalman_filter_price_target,
    kalman_momentum_filter,
    fit_gaussian_copula,
    parallel_mcmc_chains,
)

# Screening functions
from finance_ml.analytics.screening import (
    create_enhanced_screener,
    screen_earnings_quality,
    screen_value_opportunities,
    screen_growth_momentum,
    screen_dividend_quality,
    screen_financial_health,
    rank_stocks_by_composite_score,
    create_sector_relative_ranking,
)

# Performance optimizations
from finance_ml.analytics.optimized_ops import (
    dataframe_hash,
    load_feature_data_from_db_cached,
    fast_monte_carlo_simulation,
    fast_ruin_probability,
    vectorized_zscore,
    vectorized_percentile_rank,
    get_optimization_status,
)

__all__ = [
    # Feature analytics
    "PLOTLY_TEMPLATE",
    "create_interactive_momentum_dashboard",
    "create_interactive_valuation_heatmap",
    "create_leverage_liquidity_quadrant",
    "monte_carlo_price_target_simulation",
    "bayesian_earnings_beat_model",
    "analyze_distress_distribution",
    "create_composite_quality_score",
    "create_summary_dashboard",
    "safe_get_column",
    "ensure_subplot_data",
    # Data utilities
    "load_feature_data_from_db",
    "backfill_feature_columns",
    "compute_metric_statistics",
    "validate_feature_alignment",
    # Statistical analysis
    "bayesian_category_analysis",
    "metropolis_hastings_sampler",
    "mcmc_student_t",
    "hierarchical_mcmc_by_sector",
    "fit_distributions_by_category",
    "calculate_ruin_probability",
    "calculate_conditional_probabilities",
    "kalman_filter_price_target",
    "kalman_momentum_filter",
    "fit_gaussian_copula",
    "parallel_mcmc_chains",
    # Screening
    "create_enhanced_screener",
    "screen_earnings_quality",
    "screen_value_opportunities",
    "screen_growth_momentum",
    "screen_dividend_quality",
    "screen_financial_health",
    "rank_stocks_by_composite_score",
    "create_sector_relative_ranking",
    # Optimizations
    "dataframe_hash",
    "load_feature_data_from_db_cached",
    "fast_monte_carlo_simulation",
    "fast_ruin_probability",
    "vectorized_zscore",
    "vectorized_percentile_rank",
    "get_optimization_status",
]
