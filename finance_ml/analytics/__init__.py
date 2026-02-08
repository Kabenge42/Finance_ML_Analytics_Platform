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

# Feature category management (canonical source: data_utils)
from finance_ml.analytics.data_utils import (
    load_feature_data_from_db,
    backfill_feature_columns,
    compute_metric_statistics,
    validate_feature_alignment,
    load_all_feature_views,
    get_view_category_mapping,
    get_view_category_labels,
    get_view_feature_cols,
    export_view_analytics_results,
    VW_FEATURES_VIEWS,
    safe_get_column,
    load_feature_categories_from_db,
    _get_fallback_feature_categories,
    compare_registry_with_local,
    load_identifier_columns,
    get_identifier_cols_set,
)
from finance_ml.analytics.feature_analytics import (
    PLOTLY_TEMPLATE,
    FEATURE_CATEGORIES,
    analyze_distress_distribution,
    bayesian_earnings_beat_model,
    create_composite_quality_score,
    create_interactive_momentum_dashboard,
    create_interactive_valuation_heatmap,
    create_leverage_liquidity_quadrant,
    create_summary_dashboard,
    ensure_subplot_data,
    monte_carlo_price_target_simulation,
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

# Screening functions
from finance_ml.analytics.screening import (
    create_enhanced_screener,
    screen_earnings_quality,
    screen_value_opportunities,
    screen_growth_momentum,
    screen_valuation_reversion_candidates,
    screen_integrity_filtered_growth,
    screen_dividend_quality,
    screen_financial_health,
    rank_stocks_by_composite_score,
    create_sector_relative_ranking,
    screen_garp_opportunities,
    screen_high_yield_safe_dividends,
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
    analyze_employee_productivity_frontier,
    detect_accounting_anomalies,
    analyze_reporting_lag_sentiment,
    run_category_probability_analytics,  # NEW
    run_all_views_probability_analytics,  # NEW
    export_probability_view_results,  # NEW
)

# Probability Analytics (NEW)
try:
    from finance_ml.analytics.probability_analytics import (
        EarningsBeatProbabilityModel,
        CreditRiskProbabilityModel,
        DividendCutProbabilityModel,
        PriceTargetAchievementModel,
        EPSStreakAnalyzer,
        ModelConfidenceEstimator,
        BeatProbabilityResult,
        BeatProbabilityEstimate,
        EPSStreakResult,
        ModelConfidenceResult,
        CreditRiskResult,
        DividendSafetyResult,
        PriceTargetResult,
        PriorParameters,
        ReportedEPSHistory,
        ForwardEstimateSignals,
        create_earnings_probability_dashboard,
        create_confidence_calibration_chart,
        create_eps_streak_analysis_chart,
        export_probability_analytics_results,
    )

    _PROBABILITY_ANALYTICS_AVAILABLE = True
except ImportError:
    _PROBABILITY_ANALYTICS_AVAILABLE = False

__all__ = [
    # Feature analytics
    "PLOTLY_TEMPLATE",
    "FEATURE_CATEGORIES",
    "load_feature_categories_from_db",
    "compare_registry_with_local",
    "_get_fallback_feature_categories",
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
    "load_identifier_columns",
    "get_identifier_cols_set",
    "load_feature_data_from_db",
    "backfill_feature_columns",
    "compute_metric_statistics",
    "validate_feature_alignment",
    "load_all_feature_views",
    "get_view_category_mapping",
    "get_view_category_labels",
    "get_view_feature_cols",
    "export_view_analytics_results",
    "VW_FEATURES_VIEWS",
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
    "run_category_probability_analytics",
    "run_all_views_probability_analytics",
    "export_probability_view_results",
    # Probability Analytics
    "EarningsBeatProbabilityModel",
    "CreditRiskProbabilityModel",
    "DividendCutProbabilityModel",
    "PriceTargetAchievementModel",
    "EPSStreakAnalyzer",
    "ModelConfidenceEstimator",
    "BeatProbabilityResult",
    "BeatProbabilityEstimate",
    "EPSStreakResult",
    "ModelConfidenceResult",
    "CreditRiskResult",
    "DividendSafetyResult",
    "PriceTargetResult",
    "PriorParameters",
    "ReportedEPSHistory",
    "ForwardEstimateSignals",
    "create_earnings_probability_dashboard",
    "create_confidence_calibration_chart",
    "create_eps_streak_analysis_chart",
    "export_probability_analytics_results",
    # Screening
    "create_enhanced_screener",
    "screen_earnings_quality",
    "screen_value_opportunities",
    "screen_growth_momentum",
    "screen_valuation_reversion_candidates",
    "screen_integrity_filtered_growth",
    "screen_dividend_quality",
    "screen_financial_health",
    "rank_stocks_by_composite_score",
    "create_sector_relative_ranking",
    "screen_garp_opportunities",
    "screen_high_yield_safe_dividends",
    # Optimizations
    "dataframe_hash",
    "load_feature_data_from_db_cached",
    "fast_monte_carlo_simulation",
    "fast_ruin_probability",
    "vectorized_zscore",
    "vectorized_percentile_rank",
    "get_optimization_status",
]
