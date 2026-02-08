"""
Market Analytics - Refactored Version
This script demonstrates the refactored modular structure for feature analytics.
All functionality has been organized into logical modules:
- data_utils: Data loading, preprocessing, validation
- statistical_analysis: Bayesian, MCMC, Monte Carlo, distribution fitting
- screening: Multi-factor screening and quality scoring
- feature_analytics: Visualization dashboards (existing)
- visualizations: Additional visualization utilities
Usage:
    python market_analytics.py
"""
from __future__ import annotations

import logging
import os
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text

# --- Data utilities ---
from finance_ml.analytics.data_utils import (
    load_feature_data_from_db,
    backfill_feature_columns,
    compute_metric_statistics,
    validate_feature_alignment,
    export_to_analytics_db,
    load_identifier_columns,
    get_identifier_cols_set,
)

# --- Feature analytics dashboards ---
from finance_ml.analytics.feature_analytics import (
    PLOTLY_TEMPLATE,
    create_interactive_momentum_dashboard,
    create_interactive_valuation_heatmap,
    create_summary_dashboard,
)

# --- Multi-factor screening ---
from finance_ml.analytics.screening import (
    create_enhanced_screener,
    screen_value_opportunities,
    screen_growth_momentum,
    screen_financial_health,
    screen_garp_opportunities,
    screen_high_yield_safe_dividends,
    screen_valuation_reversion_candidates,
    screen_integrity_filtered_growth,
    screen_earnings_quality,
    screen_dividend_quality,
    rank_stocks_by_composite_score,
    create_sector_relative_ranking,
)

# --- Statistical analysis ---
from finance_ml.analytics.statistical_analysis import (
    bayesian_category_analysis,
    calculate_ruin_probability,
    analyze_employee_productivity_frontier,
    detect_accounting_anomalies,
    analyze_reporting_lag_sentiment,
    metropolis_hastings_sampler,
    mcmc_student_t,
    hierarchical_mcmc_by_sector,
    fit_distributions_by_category,
    calculate_conditional_probabilities,
    kalman_filter_price_target,
    kalman_momentum_filter,
    fit_gaussian_copula,
    parallel_mcmc_chains,
    run_category_probability_analytics,
    export_probability_view_results,
)

# --- Performance-optimized operations ---
from finance_ml.analytics.optimized_ops import (
    load_feature_data_from_db_cached,
    fast_monte_carlo_simulation,
    fast_ruin_probability,
    vectorized_zscore,
    vectorized_percentile_rank,
    get_optimization_status,
    dataframe_hash,
)

# --- Probability analytics (optional) ---
try:
    from finance_ml.analytics.probability_analytics import (
        EarningsBeatProbabilityModel,
        CreditRiskProbabilityModel,
        DividendCutProbabilityModel,
        PriceTargetAchievementModel,
        EPSStreakAnalyzer,
        ModelConfidenceEstimator,
        CategoryProbabilityAnalyzer,
        create_earnings_probability_dashboard,
        create_confidence_calibration_chart,
        create_eps_streak_analysis_chart,
        create_view_probability_dashboard,
        export_probability_analytics_results,
    )

    PROBABILITY_ANALYTICS_AVAILABLE = True
except ImportError:
    PROBABILITY_ANALYTICS_AVAILABLE = False
    logging.warning("Probability analytics module not available")

# --- Visualizations: category charts ---
from finance_ml.analytics.visualizations.category_charts import (
    create_analyst_sentiment_histogram,
    create_eps_surprise_histogram,
    create_fcf_margin_yield_scatter,
    create_rnd_intensity_boxplot,
    create_goodwill_concentration_boxplot,
    create_capex_growth_scatter,
    create_valuation_violin_plot,
    create_quality_risk_radar_chart,
    create_leverage_liquidity_bubble_chart,
    create_productivity_quadrant,
    create_accounting_quality_breakdown,
    create_valuation_range_visual,
)

# --- Visualizations: profitability ---
from finance_ml.analytics.visualizations.profitability import (
    create_margin_waterfall_chart,
    create_profitability_quadrant,
)

# --- Visualizations: technical analysis ---
from finance_ml.analytics.visualizations.technical import (
    create_momentum_ribbon_chart,
    create_52w_range_distribution,
)

# --- Visualizations: temporal analysis ---
from finance_ml.analytics.visualizations.temporal_analysis import (
    create_earnings_calendar_heatmap,
    create_dividend_streak_timeline,
)

# --- Visualizations: valuation ---
from finance_ml.analytics.visualizations.valuation import (
    create_valuation_multiples_comparison,
    create_valuation_distribution_dashboard,
    create_relative_valuation_matrix,
    create_valuation_vs_growth_quadrant,
    create_historical_valuation_percentile,
)

# --- Visualizations: earnings quality ---
from finance_ml.analytics.visualizations.earnings_quality import (
    create_earnings_surprise_dashboard,
    create_eps_trajectory_analysis,
    create_earnings_quality_decomposition,
    create_beat_rate_heatmap,
    create_earnings_consistency_matrix,
)

# --- Visualizations: quality & risk ---
from finance_ml.analytics.visualizations.quality_risk import (
    create_piotroski_fscore_breakdown,
    create_altman_zscore_distribution,
    create_quality_risk_quadrant,
    create_beneish_mscore_analysis,
    create_risk_tier_sunburst,
    create_distress_early_warning_dashboard,
)

# --- Visualizations: growth analysis ---
from finance_ml.analytics.visualizations.growth_analysis import (
    create_growth_waterfall_chart,
    create_growth_consistency_matrix,
    create_growth_vs_profitability_quadrant,
    create_growth_acceleration_chart,
    create_sustainable_growth_analysis,
)

# =============================================================================
# Configuration constants
# =============================================================================
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_DB_URL_ENV_VAR = "DB_URL"
_DEFAULT_DB_URL = "postgresql://user:password@localhost:5432/postgres"
_FEATURE_REGISTRY_QUERY = text("""
                               SELECT category, feature_alias
                               FROM public.calculated_features_registry
                               ORDER BY category, feature_alias
                               """)

# =============================================================================
# Global setup
# =============================================================================
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", "{:.2f}".format)
px.defaults.template = PLOTLY_TEMPLATE


# =============================================================================
# Database helpers
# =============================================================================


def _resolve_connection_string(connection_string: str | None) -> str:
    """Return the provided connection string or fall back to the environment / default."""
    if connection_string is not None:
        return connection_string
    return os.environ.get(_DB_URL_ENV_VAR, _DEFAULT_DB_URL)


def _fetch_feature_categories(connection_string: str) -> dict[str, list[str]]:
    """
    Query the feature registry and return a mapping of category → feature aliases.

    Raises on any database error so the caller can decide how to handle it.
    """
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        rows = conn.execute(_FEATURE_REGISTRY_QUERY).fetchall()

    categories: dict[str, list[str]] = defaultdict(list)
    for category, feature_alias in rows:
        categories[category].append(feature_alias)

    logging.info("Loaded %d feature categories from database", len(categories))
    return dict(categories)


def load_feature_categories(connection_string: str | None = None) -> dict[str, list[str]]:
    """
    Load feature categories from the calculated_features_registry table.

    Falls back to hardcoded defaults when the database is unreachable.

    Returns:
        Dictionary mapping category names to lists of feature aliases.
    """
    resolved_url = _resolve_connection_string(connection_string)
    try:
        return _fetch_feature_categories(resolved_url)
    except Exception as exc:
        logging.warning("Could not load categories from database: %s", exc)
        logging.warning("Falling back to hardcoded FEATURE_CATEGORIES")
        return _get_fallback_feature_categories()


def _get_fallback_feature_categories() -> dict[str, list[str]]:
    """Fallback hardcoded categories if database is unavailable."""
    return {
        "Valuation Ratios": [
            "p_e_ratio",
            "p_b_ratio",
            "ev_ebitda_ratio",
            "ev_sales_ratio",
            "dividend_yield",
            "peg_ratio",
            "price_to_tangible_book",
            "tangible_book_value_ltm",
        ],
        "Momentum & Technical": [
            "price_momentum_1m",
            "price_momentum_3m",
            "price_momentum_6m",
            "price_momentum_1y",
            "price_momentum_3y",
            "price_momentum_5y",
            "range_52w_position",
            "long_term_trend_score",
            "secular_trend_flag",
        ],
        "Profitability": [
            "roe",
            "roa",
            "gross_margin_pct",
            "operating_margin_pct",
            "net_margin_pct",
            "ebitda_margin_pct",
            "roic",
            "net_margin_trend_yoy",
        ],
        "Quality & Risk": [
            "piotroski_f_score",
            "distress_risk_score",
            "altman_z_score",
            "accounting_quality_score",
            "earnings_quality_composite",
            "cash_flow_quality_score",
            "beta_stability_score",
        ],
        "Leverage & Liquidity": [
            "debt_to_equity",
            "current_ratio",
            "quick_ratio",
            "interest_coverage_ratio",
            "cash_ratio",
            "working_capital_ratio",
            "debt_deleveraging",
        ],
        "Analyst Sentiment": [
            "analyst_bullish_pct",
            "analyst_neutral_pct",
            "analyst_bearish_pct",
            "upside_potential",
            "analyst_rating_normalized",
            "eps_revision_momentum",
        ],
        "Earnings Quality": [
            "eps_surprise_pct",
            "eps_adjustment_ratio",
            "gaap_adj_eps_gap_pct",
            "eps_trajectory_score",
            "earnings_quality_score",
            "gaap_revision_momentum",
        ],
        "Growth Metrics": [
            "revenue_yoy_growth",
            "ebitda_growth_yoy",
            "eps_yoy_growth",
            "fcf_growth_yoy",
            "revenue_cagr_5y",
        ],
        "Cash Flow": [
            "fcf_positive_years",
            "fcf_margin",
            "fcf_yield",
            "cfo_to_net_income",
            "self_funding_ratio",
            "cash_flow_quality_score",
        ],
        "Dividend Features": [
            "dividend_streak",
            "dividend_yield_ltm",
            "dividend_payout_ratio",
            "fcf_dividend_coverage",
            "total_shareholder_yield",
        ],
        "R&D Investment": [
            "rnd_intensity_ltm",
            "rnd_yoy_growth",
            "rnd_per_employee",
            "high_rnd_intensity_flag",
        ],
        "Inventory Temporal": [
            "inventory_days",
            "inventory_turnover_itf",
            "inventory_yoy_change",
            "inventory_buildup_flag",
        ],
        "Goodwill & M&A": [
            "goodwill_concentration",
            "goodwill_3y_growth",
            "recent_acquisition_flag",
            "impairment_risk_score",
        ],
        "CapEx & Investment": [
            "capex_yoy_growth",
            "capex_vs_5y_avg",
            "acquisitions_ltm_total",
            "ma_intensity_score",
            "investment_efficiency",
        ],
    }


def compare_registry_with_local(
    db_categories: dict[str, list[str]], local_categories: dict[str, list[str]]
) -> dict:
    """
    Compare database registry with local/fallback categories.
    Useful for identifying missing features or new additions.
    """
    report = {
        "categories_only_in_db": [],
        "categories_only_in_local": [],
        "features_only_in_db": {},
        "features_only_in_local": {},
    }

    db_cats = set(db_categories.keys())
    local_cats = set(local_categories.keys())

    report["categories_only_in_db"] = list(db_cats - local_cats)
    report["categories_only_in_local"] = list(local_cats - db_cats)

    for cat in db_cats & local_cats:
        db_features = set(db_categories[cat])
        local_features = set(local_categories[cat])

        only_in_db = db_features - local_features
        only_in_local = local_features - db_features

        if only_in_db:
            report["features_only_in_db"][cat] = list(only_in_db)
        if only_in_local:
            report["features_only_in_local"][cat] = list(only_in_local)

    return report


# Load categories dynamically at module level (with fallback)
FEATURE_CATEGORIES = load_feature_categories()


def main():
    """Main execution function demonstrating refactored modules."""

    print("=" * 80)
    print("MARKET ANALYTICS - REFACTORED VERSION")
    print("=" * 80)
    print()

    # ========================================================================
    # 0. OPTIMIZATION STATUS
    # ========================================================================
    opt_status = get_optimization_status()
    print("⚡ Optimization Status:")
    print(
        f"   Numba JIT:      {'✓ Available' if opt_status['numba_available'] else '✗ Fallback mode'}"
    )
    print(
        f"   Parallel:       {'✓ Available' if opt_status['parallel_available'] else '✗ Sequential'}"
    )
    print(f"   DB Cache:       {opt_status['db_cache_size']} entries")
    print(f"   Stats Cache:    {opt_status['stats_cache_size']} entries")
    print()

    # ========================================================================
    # 1. DATA LOADING AND PREPROCESSING
    # ========================================================================
    print("📊 Step 1: Loading and preprocessing data...")
    print("-" * 80)

    try:
        # Use cached loader for performance (subsequent runs skip DB round-trip)
        df = load_feature_data_from_db_cached()
        print(f"✓ Loaded {len(df):,} stocks with {len(df.columns)} features")
        print(f"   DataFrame hash: {dataframe_hash(df)[:12]}...")

        # Backfill missing columns
        df = backfill_feature_columns(df)
        print(f"✓ Backfilled features, now have {len(df.columns)} columns")

        # Compare registry with fallback to identify drift
        fallback = _get_fallback_feature_categories()
        diff_report = compare_registry_with_local(FEATURE_CATEGORIES, fallback)
        if diff_report["features_only_in_db"]:
            print("   📌 New features in registry (not in fallback):")
            for cat, feats in diff_report["features_only_in_db"].items():
                print(f"      {cat}: {feats}")

    except Exception as e:
        print(f"⚠️  Could not load from database: {e}")
        print("   Using sample data for demonstration...")
        # Create sample data for demonstration
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "name": [
                    "Apple Inc.",
                    "Microsoft Corp.",
                    "Alphabet Inc.",
                    "Amazon.com Inc.",
                    "Tesla Inc.",
                ],
                "industry": [
                    "Technology",
                    "Technology",
                    "Technology",
                    "Consumer",
                    "Automotive",
                ],
                "market_cap": [3000000, 2800000, 1800000, 1600000, 800000],
                "p_e_ratio": [28.5, 32.1, 25.3, 45.2, 65.8],
                "piotroski_f_score": [8, 7, 8, 6, 5],
                "distress_risk_score": [85, 82, 88, 75, 60],
                "eps_trajectory_score": [75, 80, 78, 65, 55],
                "fcf_positive_years": [5, 5, 5, 4, 3],
            }
        )

    print()

    # ========================================================================
    # 2. FEATURE VALIDATION
    # ========================================================================
    print("📋 Step 2: Validating feature coverage...")
    print("-" * 80)

    validation = validate_feature_alignment(df, FEATURE_CATEGORIES)
    low_coverage = {k: v for k, v in validation.items() if v["coverage_pct"] < 80}

    if low_coverage:
        print("⚠️  Categories with <80% feature coverage:")
        for cat, info in low_coverage.items():
            print(f"   {cat}: {info['coverage_pct']:.1f}%")
    else:
        print("✓ All feature categories have ≥80% coverage")

    print()

    # ========================================================================
    # 3. FEATURE VIEW ANALYTICS
    # ========================================================================
    print("📊 Step 3: Loading all feature views...")
    print("-" * 80)

    from finance_ml.analytics.data_utils import (
        load_all_feature_views,
        get_view_category_labels,
        export_view_analytics_results,
    )
    from finance_ml.analytics.statistical_analysis import (
        run_all_views_probability_analytics,
        export_probability_view_results,
    )

    # Load all 17 vw_features views
    try:
        views_dict = load_all_feature_views(return_dict=True)
        print(f"✓ Loaded {len(views_dict)} feature views")

        for view_name, view_df in views_dict.items():
            if not view_df.empty:
                print(f"   {view_name}: {len(view_df):,} rows, {len(view_df.columns)} columns")

        # Run probability analytics on each view
        print("\n📈 Running view-based probability analytics...")
        view_mapping = get_view_category_labels()
        view_analytics = run_all_views_probability_analytics(views_dict, view_mapping)

        # Export results to database
        print("\n💾 Exporting analytics to database...")
        export_counts = export_view_analytics_results(view_analytics)
        for table, count in export_counts.items():
            print(f"   {table}: {count} rows")

        # --- NEW: Export per-feature probability metrics to prob_vw_features_* tables ---
        print("\n📊 Exporting per-feature probability metrics...")
        id_cols = load_identifier_columns()
        id_cols_set = set(id_cols)
        for view_name, view_df in views_dict.items():
            if view_df.empty:
                continue
            feature_cols = [c for c in view_df.columns if c not in id_cols_set]
            rows_exported = export_probability_view_results(
                view_df, view_name, feature_cols, id_cols
            )
            if rows_exported and rows_exported > 0:
                print(f"   ✓ prob_{view_name}: {rows_exported} rows")

        # Generate visualizations for each view
        print("\n📊 Generating view-specific visualizations...")
        output_dir_views = Path("outputs/analytics/views")
        output_dir_views.mkdir(parents=True, exist_ok=True)

        for view_name, view_df in views_dict.items():
            if view_df.empty:
                continue
            category = view_mapping.get(view_name, view_name)

            # Enhanced View Analytics using CategoryProbabilityAnalyzer
            if PROBABILITY_ANALYTICS_AVAILABLE:
                analyzer = CategoryProbabilityAnalyzer(category)
                id_cols_set_viz = get_identifier_cols_set()
                feature_cols = [c for c in view_df.columns if c not in id_cols_set_viz]
                view_prob_results = analyzer.analyze_view(view_df, feature_cols)
                if not view_prob_results.empty:
                    # Export view-specific probability results
                    export_to_analytics_db(view_prob_results, f"prob_{view_name}")

            fig = create_view_probability_dashboard(view_df, view_name, category)
            output_path = output_dir_views / f"{view_name}_probability.html"
            fig.write_html(str(output_path))
            print(f"   ✓ Saved {output_path.name}")
    except Exception as e:
        print(f"⚠️  Error in view-based analytics: {e}")

    print()

    # ========================================================================
    # 4. STATISTICAL ANALYSIS
    # ========================================================================
    print("📈 Step 4: Running statistical analysis...")
    print("-" * 80)

    # Bayesian analysis for profitability features
    if "roe" in df.columns and len(df) > 50:
        print("   Running Bayesian analysis on Profitability features...")
        profitability_features = [
            f for f in FEATURE_CATEGORIES.get("Profitability", []) if f in df.columns
        ][:3]
        bayesian_results = bayesian_category_analysis(df, "Profitability", profitability_features)
        if bayesian_results:
            print(f"   ✓ Analyzed {len(bayesian_results)} profitability metrics")

    # --- NEW: Fast ruin probability (optimized_ops) replaces statistical_analysis ---
    if all(col in df.columns for col in ["market_cap", "distress_risk_score"]):
        print("   Calculating investor's ruin probabilities (Numba-accelerated)...")
        ruin_df = fast_ruin_probability(df, n_simulations=2000, n_days=252)
        high_risk_count = (ruin_df["ruin_probability"] > 0.6).sum()
        print(f"   ✓ Identified {high_risk_count} high-risk stocks ({len(ruin_df)} analyzed)")

        # Also keep analytical ruin for comparison
        ruin_analytical = calculate_ruin_probability(df)
        print(
            f"   ✓ Analytical ruin model: {(ruin_analytical['ruin_probability'] > 0.6).sum()} high-risk"
        )

    # --- NEW: Fast Monte Carlo simulation (optimized_ops) ---
    required_mc_cols = [
        "price_target_low",
        "price_target_median",
        "price_target_high",
        "last_price",
    ]
    if all(col in df.columns for col in required_mc_cols):
        print("   Running Numba-accelerated Monte Carlo price target simulation...")
        mc_results = fast_monte_carlo_simulation(df, n_simulations=10000)
        print(f"   ✓ Monte Carlo: {len(mc_results)} stocks simulated")
        top_mc = mc_results.nlargest(5, "risk_reward_ratio")
        for _, r in top_mc.iterrows():
            print(
                f"      {r['ticker']}: E[upside]={r['expected_upside']:.1f}%, "
                f"P(positive)={r['prob_positive']:.0f}%, RR={r['risk_reward_ratio']:.2f}"
            )

    # --- NEW: Vectorized z-scores and percentile ranks (optimized_ops) ---
    print("   Computing vectorized z-scores and percentile ranks...")
    zscore_cols = [c for c in ["roe", "roa", "p_e_ratio", "debt_to_equity"] if c in df.columns]
    if zscore_cols:
        df = vectorized_zscore(df, zscore_cols, group_col="industry")
        df = vectorized_percentile_rank(df, zscore_cols, group_col="industry")
        print(f"   ✓ Z-scores and percentiles computed for {len(zscore_cols)} metrics by industry")

    # --- NEW: Kalman-filtered price targets (statistical_analysis) ---
    if "last_price" in df.columns and "price_target_median" in df.columns:
        print("   Applying Kalman filter to price targets...")
        kalman_pt = kalman_filter_price_target(
            df, observation_col="last_price", target_col="price_target_median"
        )
        if len(kalman_pt) > 0:
            print(f"   ✓ Kalman-filtered {len(kalman_pt)} price targets")
            high_signal = kalman_pt.nlargest(5, "signal_strength")
            for _, k in high_signal.iterrows():
                print(
                    f"      {k['ticker']}: Filtered upside={k['filtered_upside']:.1f}%, "
                    f"Signal={k['signal_strength']:.1f}"
                )

    # --- NEW: Kalman-filtered momentum (statistical_analysis) ---
    momentum_cols = [
        c
        for c in ["price_momentum_1m", "price_momentum_3m", "price_momentum_6m"]
        if c in df.columns
    ]
    if momentum_cols:
        print("   Applying Kalman filter to momentum indicators...")
        df = kalman_momentum_filter(df, momentum_cols=momentum_cols)
        print(f"   ✓ Smoothed {len(momentum_cols)} momentum columns")

    # --- NEW: Gaussian copula dependency analysis (statistical_analysis) ---
    copula_features = [
        c for c in ["roe", "debt_to_equity", "p_e_ratio", "current_ratio"] if c in df.columns
    ]
    if len(copula_features) >= 2:
        print("   Fitting Gaussian copula for dependency structure...")
        copula_result = fit_gaussian_copula(df, copula_features, n_simulations=5000)
        print(
            f"   ✓ Copula fitted on {len(copula_result['features'])} features "
            f"({copula_result['n_observations']} observations)"
        )

    # --- NEW: Parallel MCMC with convergence diagnostics (statistical_analysis) ---
    if "roe" in df.columns:
        roe_data = df["roe"].dropna().values
        if len(roe_data) > 100:
            print("   Running parallel MCMC chains with Gelman-Rubin diagnostic...")
            mcmc_result = parallel_mcmc_chains(roe_data, n_chains=4, n_samples=5000)
            print(
                f"   ✓ R-hat: {mcmc_result['r_hat']:.4f} "
                f"({'Converged ✓' if mcmc_result['converged'] else 'Not converged ✗'})"
            )
            print(
                f"     Posterior mean ROE: {mcmc_result['posterior_mean']:.2f} "
                f"[{mcmc_result['ci_95'][0]:.2f}, {mcmc_result['ci_95'][1]:.2f}]"
            )

    # --- NEW: Hierarchical MCMC by sector (statistical_analysis) ---
    if "roe" in df.columns and "industry" in df.columns:
        print("   Running hierarchical MCMC by sector...")
        hier_results = hierarchical_mcmc_by_sector(df, "roe", sector_col="industry")
        print(f"   ✓ Hierarchical analysis for {len(hier_results)} sectors")

    # --- NEW: Distribution fitting by category (statistical_analysis) ---
    for cat_name in ["Profitability", "Valuation Ratios"]:
        cat_features = [f for f in FEATURE_CATEGORIES.get(cat_name, []) if f in df.columns][:3]
        if cat_features:
            print(f"   Fitting distributions for {cat_name}...")
            dist_fits = fit_distributions_by_category(df, cat_name, cat_features)
            for feat, info in dist_fits.items():
                print(
                    f"      {feat}: best={info['best_distribution']}, "
                    f"VaR(5%)={info['var_5_pct']:.2f}"
                )

    # --- NEW: Conditional probabilities P(Distress|Feature) (statistical_analysis) ---
    if "distress_risk_score" in df.columns:
        print("   Calculating conditional distress probabilities...")
        cond_probs = calculate_conditional_probabilities(df, FEATURE_CATEGORIES)
        if len(cond_probs) > 0:
            top_predictors = cond_probs.nlargest(5, "separation")
            print(f"   ✓ Top distress predictors:")
            for _, p in top_predictors.iterrows():
                print(f"      {p['feature']} ({p['category']}): separation={p['separation']:.3f}")

    # --- NEW: Per-category probability analytics pipeline (statistical_analysis) ---
    print("   Running per-category probability analytics...")
    for cat_name, cat_feats in list(FEATURE_CATEGORIES.items())[:3]:
        available = [f for f in cat_feats if f in df.columns]
        if available:
            cat_results = run_category_probability_analytics(df, cat_name, available)
            print(f"   ✓ {cat_name}: {cat_results['features_analyzed']} features analyzed")

    # New Statistical Analyses (existing)
    print("   Running Productivity and Accounting analysis...")
    df = analyze_employee_productivity_frontier(df)
    df = detect_accounting_anomalies(df)
    lag_analysis = analyze_reporting_lag_sentiment(df)
    print(f"   ✓ Accounting anomaly detection complete")
    if lag_analysis["sample_size"] > 0:
        print(
            f"   ✓ Reporting lag correlation: {lag_analysis['correlation']:.2f} "
            f"(Hypothesis: {lag_analysis['hypothesis_confirmed']})"
        )

    # ========================================================================
    # 5. PROBABILITY & CONFIDENCE ANALYTICS (ENHANCED)
    # ========================================================================
    if PROBABILITY_ANALYTICS_AVAILABLE:
        print("📊 Step 5: Running probability & confidence analytics...")
        print("-" * 80)

        probability_results = None
        streak_results = None
        confidence_result = None
        credit_results = None
        dividend_results = None
        pt_results = None

        try:
            # Initialize models
            beat_model = EarningsBeatProbabilityModel()
            streak_analyzer = EPSStreakAnalyzer(mean_reversion_weight=0.3)
            confidence_estimator = ModelConfidenceEstimator(n_bins=10)

            # --- Enhanced: Use three-layer evidence fusion when forward data available ---
            print("   Computing Bayesian earnings beat probabilities (enhanced)...")

            # Try enhanced analysis first (uses forward estimates + GAAP quality)
            probability_results = beat_model.analyze_dataframe_enhanced(
                df,
                sector_col="sector" if "sector" in df.columns else "industry",
                ticker_col="ticker" if "ticker" in df.columns else "isin",
                name_col="name",
            )

            # Fallback to basic analysis if enhanced yields no results
            if probability_results is None or len(probability_results) == 0:
                print(
                    "   ⚠️  Enhanced analysis yielded no results, falling back to proxy method..."
                )
                if "eps_trajectory_score" in df.columns:
                    df_analysis = df.copy()
                    # Use eps_positive_years if available, else derive from trajectory score
                    beats_col_name = "eps_beat_count"
                    if "eps_positive_years" in df_analysis.columns:
                        beats_col_name = "eps_positive_years"
                    elif beats_col_name not in df_analysis.columns:
                        df_analysis[beats_col_name] = (
                            df_analysis["eps_trajectory_score"].fillna(50) / 100 * 5
                        ).astype(int)
                    total_col_name = "eps_total_reports"
                    if total_col_name not in df_analysis.columns:
                        df_analysis[total_col_name] = 5

                    probability_results = beat_model.analyze_dataframe(
                        df_analysis,
                        beats_col=beats_col_name,
                        total_col=total_col_name,
                        sector_col="sector" if "sector" in df.columns else "industry",
                        ticker_col="ticker" if "ticker" in df.columns else "isin",
                    )

            if probability_results is not None and len(probability_results) > 0:
                likely_beat_count = (
                    probability_results["beat_classification"] == "likely_beat"
                ).sum()
                print(f"   ✓ Analyzed {len(probability_results)} stocks")
                print(f"   ✓ {likely_beat_count} classified as 'likely beat'")
                print(
                    f"   ✓ Mean posterior beat probability: "
                    f"{probability_results['posterior_beat_prob'].mean():.1%}"
                )
                # Report data source breakdown if enhanced columns present
                if "data_source" in probability_results.columns:
                    source_counts = probability_results["data_source"].value_counts()
                    for src, cnt in source_counts.items():
                        print(f"     Data source '{src}': {cnt} stocks")

            # EPS Streak Analysis
            print("   Analyzing EPS streaks and continuation probabilities...")

            if "eps_trajectory_score" in df.columns:
                streak_results = streak_analyzer.analyze_dataframe(
                    df,
                    trajectory_col="eps_trajectory_score",
                    streak_col=(
                        "eps_positive_streak" if "eps_positive_streak" in df.columns else None
                    ),
                    ticker_col="ticker" if "ticker" in df.columns else "isin",
                )

                if len(streak_results) > 0:
                    beat_streaks = (streak_results["streak_type"] == "beat").sum()
                    miss_streaks = (streak_results["streak_type"] == "miss").sum()
                    print(f"   ✓ Identified {beat_streaks} stocks on beat streaks")
                    print(f"   ✓ Identified {miss_streaks} stocks on miss streaks")
                    print(
                        f"   ✓ Mean continuation probability: "
                        f"{streak_results['continuation_probability'].mean():.1%}"
                    )
                    # Report dynamic total reports
                    if "dynamic_total_reports" in streak_results.columns:
                        avg_total = streak_results["dynamic_total_reports"].mean()
                        print(f"   ✓ Avg dynamic total reports per stock: {avg_total:.1f}")

            # Model Confidence Estimation (using simulated outcomes for demo)
            print("   Estimating model confidence metrics...")

            if probability_results is not None and len(probability_results) > 10:
                # Simulate actual outcomes based on posterior probability
                # In production, this would use actual historical outcomes
                np.random.seed(42)
                simulated_outcomes = (
                    np.random.random(len(probability_results))
                    < probability_results["posterior_beat_prob"].values
                ).astype(float)

                confidence_result = confidence_estimator.compute_confidence_metrics(
                    predicted_probs=probability_results["posterior_beat_prob"].values,
                    actual_outcomes=simulated_outcomes,
                    model_name="Bayesian Earnings Beat Model",
                )

                print(f"   ✓ Brier Score: {confidence_result.brier_score:.4f}")
                print(f"   ✓ Calibration Error: {confidence_result.calibration_error:.4f}")
                print(f"   ✓ AUC-ROC: {confidence_result.discrimination_auc:.3f}")
                print(f"   ✓ Overall Confidence: {confidence_result.overall_confidence:.1f}/100")

            # New Probability Models
            print("   Running Credit Risk, Dividend Cut, and Price Target models...")
            credit_model = CreditRiskProbabilityModel()
            dividend_model = DividendCutProbabilityModel()
            pt_model = PriceTargetAchievementModel()

            credit_results = credit_model.analyze_dataframe(df)
            dividend_results = dividend_model.analyze_dataframe(df)
            pt_results = pt_model.analyze_dataframe(df)

            print(f"   ✓ Credit Risk analysis complete ({len(credit_results)} stocks)")
            print(f"   ✓ Dividend Cut analysis complete ({len(dividend_results)} stocks)")
            print(f"   ✓ Price Target achievement analysis complete ({len(pt_results)} stocks)")

            # --- FIX: Export ALL probability results including credit/dividend/PT ---
            if probability_results is not None and streak_results is not None:
                print("   Exporting probability analytics results...")
                output_dir_prob = Path("outputs/analytics")
                export_paths = export_probability_analytics_results(
                    probability_df=probability_results,
                    streak_df=streak_results,
                    output_dir=output_dir_prob,
                    confidence_result=confidence_result,
                    credit_risk_df=credit_results,
                    dividend_safety_df=dividend_results,
                    price_target_df=pt_results,
                )
                for name, path in export_paths.items():
                    print(f"   ✓ Saved: {path}")

        except Exception as e:
            print(f"   ⚠️  Probability analytics error: {e}")
            import traceback

            traceback.print_exc()

        print()
    else:
        # Ensure variables are defined even when probability analytics unavailable
        probability_results = None
        streak_results = None
        confidence_result = None
        credit_results = None
        dividend_results = None
        pt_results = None

    # ========================================================================
    # 6. STOCK SCREENING
    # ========================================================================
    print("🔍 Step 6: Running stock screens...")
    print("-" * 80)

    # Enhanced quality screener
    if all(col in df.columns for col in ["piotroski_f_score", "distress_risk_score"]):
        quality_stocks = create_enhanced_screener(df, min_fscore=7, min_fcf_positive_years=4)
        print(f"   ✓ Quality screen: {len(quality_stocks)} stocks")

    # Value opportunities
    if "p_e_ratio" in df.columns and "upside_potential" in df.columns:
        value_stocks = screen_value_opportunities(df, max_pe_ratio=100, min_upside_potential=20)
        print(f"   ✓ Value screen: {len(value_stocks)} stocks")

    # Growth momentum
    if "revenue_yoy_growth" in df.columns or "revenue_growth_yoy" in df.columns:
        growth_stocks = screen_growth_momentum(df, min_revenue_growth=5)
        print(f"   ✓ Growth screen: {len(growth_stocks)} stocks")

    # Financial health
    if "distress_risk_score" in df.columns:
        healthy_stocks = screen_financial_health(df, min_distress_score=80)
        print(f"   ✓ Financial health screen: {len(healthy_stocks)} stocks")

    # GARP opportunities
    if "peg_ratio" in df.columns:
        garp_stocks = screen_garp_opportunities(df)
        print(f"   ✓ GARP screen: {len(garp_stocks)} stocks")

    # High-yield safe dividends
    if "dividend_yield" in df.columns or "dividend_yield_ltm" in df.columns:
        safe_div_stocks = screen_high_yield_safe_dividends(df)
        print(f"   ✓ High-yield safe dividend screen: {len(safe_div_stocks)} stocks")

    # New Screens
    reversion_stocks = screen_valuation_reversion_candidates(df)
    growth_integrity_stocks = screen_integrity_filtered_growth(df)
    print(f"   ✓ Valuation reversion screen: {len(reversion_stocks)} stocks")
    print(f"   ✓ Integrity-filtered growth screen: {len(growth_integrity_stocks)} stocks")

    # --- NEW: Earnings quality screening (screening.py) ---
    if "earnings_quality_composite" in df.columns:
        eq_stocks = screen_earnings_quality(df, min_quality_score=65)
        print(f"   ✓ Earnings quality screen: {len(eq_stocks)} stocks")

    # --- NEW: Dividend quality screening (screening.py) ---
    div_yield_col = "dividend_yield_ltm" if "dividend_yield_ltm" in df.columns else "dividend_yield"
    if div_yield_col in df.columns:
        dq_stocks = screen_dividend_quality(df, min_dividend_yield=2.5, min_dividend_streak=3)
        print(f"   ✓ Dividend quality screen: {len(dq_stocks)} stocks")

    # --- NEW: Composite ranking (screening.py) ---
    ranked_df = rank_stocks_by_composite_score(df, export=True)
    print(f"   ✓ Composite ranking: top score = {ranked_df['composite_score'].max():.1f}")

    # --- NEW: Sector-relative rankings (screening.py) ---
    for metric in ["roe", "p_e_ratio"]:
        if metric in df.columns and "industry" in df.columns:
            df = create_sector_relative_ranking(df, metric, sector_col="industry")
            print(f"   ✓ Sector-relative ranking for {metric}")

    print()

    # ========================================================================
    # 7. VISUALIZATIONS
    # ========================================================================
    print("📊 Step 7: Generating visualizations...")
    print("-" * 80)

    output_dir = Path("outputs/analytics")
    output_dir.mkdir(exist_ok=True)

    try:
        # Interactive momentum dashboard
        if all(
            col in df.columns
            for col in [
                "price_momentum_1m",
                "price_momentum_3m",
                "price_momentum_6m",
                "price_momentum_1y",
            ]
        ):
            print("   Creating momentum dashboard...")
            fig = create_interactive_momentum_dashboard(df)
            fig.write_html(output_dir / "momentum_dashboard.html")
            print("   ✓ Saved: outputs/momentum_dashboard.html")

        # Valuation heatmap
        if "p_e_ratio" in df.columns and "industry" in df.columns:
            print("   Creating valuation heatmap...")
            fig = create_interactive_valuation_heatmap(df)
            fig.write_html(output_dir / "valuation_heatmap.html")
            print("   ✓ Saved: outputs/valuation_heatmap.html")

        # Summary dashboard
        print("   Creating summary dashboard...")
        fig = create_summary_dashboard(df)
        fig.write_html(output_dir / "summary_dashboard.html")
        print("   ✓ Saved: outputs/summary_dashboard.html")

        # --- NEW: Probability Analytics Visualizations ---
        if PROBABILITY_ANALYTICS_AVAILABLE:
            if probability_results is not None and len(probability_results) > 0:
                print("   Creating earnings probability dashboard...")
                fig = create_earnings_probability_dashboard(probability_results)
                fig.write_html(output_dir / "earnings_beat_probability_dashboard.html")
                print("   ✓ Saved: outputs/analytics/earnings_beat_probability_dashboard.html")

                if "confidence_result" in locals():
                    print("   Creating model confidence calibration chart...")
                    fig = create_confidence_calibration_chart(confidence_result)
                    fig.write_html(output_dir / "model_confidence_calibration.html")
                    print("   ✓ Saved: outputs/analytics/model_confidence_calibration.html")

            if streak_results is not None and len(streak_results) > 0:
                print("   Creating EPS streak analysis chart...")
                fig = create_eps_streak_analysis_chart(streak_results)
                fig.write_html(output_dir / "eps_streak_analysis.html")
                print("   ✓ Saved: outputs/analytics/eps_streak_analysis.html")

        # --- New Category-Specific Visualizations ---

        # Profitability visualizations
        if all(col in df.columns for col in ["roe", "roa", "industry"]):
            print("   Creating profitability quadrant...")
            fig = create_profitability_quadrant(df)
            fig.write_html(output_dir / "profitability_quadrant.html")
            print("   ✓ Saved: outputs/profitability_quadrant.html")

        if "gross_margin_pct" in df.columns:
            print("   Creating margin waterfall chart...")
            fig = create_margin_waterfall_chart(df)
            fig.write_html(output_dir / "margin_waterfall.html")
            print("   ✓ Saved: outputs/margin_waterfall.html")

        # Technical analysis visualizations
        if "range_52w_position" in df.columns:
            print("   Creating 52-week range distribution...")
            fig = create_52w_range_distribution(df)
            fig.write_html(output_dir / "52w_range_distribution.html")
            print("   ✓ Saved: outputs/52w_range_distribution.html")

        if all(
            col in df.columns
            for col in [
                "price_momentum_1m",
                "price_momentum_3m",
                "price_momentum_6m",
                "price_momentum_1y",
            ]
        ):
            print("   Creating momentum ribbon chart...")
            fig = create_momentum_ribbon_chart(df)
            fig.write_html(output_dir / "momentum_ribbon.html")
            print("   ✓ Saved: outputs/momentum_ribbon.html")

        # Temporal analysis visualizations
        if "next_earnings" in df.columns:
            print("   Creating earnings calendar heatmap...")
            fig = create_earnings_calendar_heatmap(df)
            fig.write_html(output_dir / "earnings_calendar.html")
            print("   ✓ Saved: outputs/earnings_calendar.html")

        if "dividend_streak" in df.columns:
            print("   Creating dividend streak timeline...")
            fig = create_dividend_streak_timeline(df)
            fig.write_html(output_dir / "dividend_streak_timeline.html")
            print("   ✓ Saved: outputs/dividend_streak_timeline.html")

        # New Visualizations
        print("   Creating new enhanced visualizations...")
        fig_prod = create_productivity_quadrant(df)
        fig_prod.write_html(output_dir / "productivity_quadrant.html")

        if len(df) > 0:
            sample_ticker = df["ticker"].iloc[0]
            fig_acc = create_accounting_quality_breakdown(df, sample_ticker)
            fig_acc.write_html(output_dir / f"accounting_quality_{sample_ticker}.html")

            fig_val = create_valuation_range_visual(df, sample_ticker)
            fig_val.write_html(output_dir / f"valuation_range_{sample_ticker}.html")

        print("   ✓ Saved enhanced visualizations to outputs/analytics/")

        # Category-specific charts
        if "analyst_bullish_pct" in df.columns:
            print("   Creating analyst sentiment histogram...")
            fig = create_analyst_sentiment_histogram(df)
            fig.write_html(output_dir / "analyst_sentiment.html")
            print("   ✓ Saved: outputs/analyst_sentiment.html")

        if "eps_surprise_pct" in df.columns:
            print("   Creating EPS surprise histogram...")
            fig = create_eps_surprise_histogram(df)
            fig.write_html(output_dir / "eps_surprise.html")
            print("   ✓ Saved: outputs/eps_surprise.html")

        if all(col in df.columns for col in ["fcf_margin", "fcf_yield"]):
            print("   Creating FCF margin vs yield scatter...")
            fig = create_fcf_margin_yield_scatter(df)
            fig.write_html(output_dir / "fcf_margin_yield.html")
            print("   ✓ Saved: outputs/fcf_margin_yield.html")

        if "rnd_intensity_ltm" in df.columns and "industry" in df.columns:
            print("   Creating R&D intensity boxplot...")
            fig = create_rnd_intensity_boxplot(df)
            fig.write_html(output_dir / "rnd_intensity.html")
            print("   ✓ Saved: outputs/rnd_intensity.html")

        if "goodwill_concentration" in df.columns and "industry" in df.columns:
            print("   Creating goodwill concentration boxplot...")
            fig = create_goodwill_concentration_boxplot(df)
            fig.write_html(output_dir / "goodwill_concentration.html")
            print("   ✓ Saved: outputs/goodwill_concentration.html")

        if all(col in df.columns for col in ["capex_yoy_growth", "capex_vs_5y_avg"]):
            print("   Creating CapEx growth scatter...")
            fig = create_capex_growth_scatter(df)
            fig.write_html(output_dir / "capex_growth.html")
            print("   ✓ Saved: outputs/capex_growth.html")

        # --- New Enhanced Visualizations ---
        if "p_e_ratio" in df.columns:
            print("   Creating valuation violin plot...")
            fig = create_valuation_violin_plot(df)
            fig.write_html(output_dir / "valuation_violin.html")
            print("   ✓ Saved: outputs/valuation_violin.html")

        if all(col in df.columns for col in ["current_ratio", "debt_to_equity"]):
            print("   Creating leverage vs liquidity bubble chart...")
            fig = create_leverage_liquidity_bubble_chart(df)
            fig.write_html(output_dir / "leverage_liquidity_bubble.html")
            print("   ✓ Saved: outputs/leverage_liquidity_bubble.html")

        if len(df) > 0:
            ticker = df.iloc[0]["ticker"]
            print(f"   Creating quality radar chart for {ticker}...")
            fig = create_quality_risk_radar_chart(df, ticker)
            fig.write_html(output_dir / f"quality_radar_{ticker}.html")
            print(f"   ✓ Saved: outputs/quality_radar_{ticker}.html")

        # --- NEW: Valuation Analysis Visualizations ---
        print("   Creating valuation analysis visualizations...")

        fig = create_valuation_distribution_dashboard(df)
        fig.write_html(output_dir / "valuation_distribution_dashboard.html")
        print("   ✓ Saved: valuation_distribution_dashboard.html")

        fig = create_relative_valuation_matrix(df)
        fig.write_html(output_dir / "relative_valuation_matrix.html")
        print("   ✓ Saved: relative_valuation_matrix.html")

        fig = create_valuation_vs_growth_quadrant(df)
        fig.write_html(output_dir / "valuation_vs_growth_quadrant.html")
        print("   ✓ Saved: valuation_vs_growth_quadrant.html")

        fig = create_historical_valuation_percentile(df)
        fig.write_html(output_dir / "historical_valuation_percentile.html")
        print("   ✓ Saved: historical_valuation_percentile.html")

        if len(df) > 0:
            ticker = df.iloc[0]["ticker"]
            fig = create_valuation_multiples_comparison(df, ticker=ticker)
            fig.write_html(output_dir / f"valuation_multiples_{ticker}.html")
            print(f"   ✓ Saved: valuation_multiples_{ticker}.html")

        # --- NEW: Earnings Quality Visualizations ---
        print("   Creating earnings quality visualizations...")

        fig = create_earnings_surprise_dashboard(df)
        fig.write_html(output_dir / "earnings_surprise_dashboard.html")
        print("   ✓ Saved: earnings_surprise_dashboard.html")

        fig = create_eps_trajectory_analysis(df)
        fig.write_html(output_dir / "eps_trajectory_analysis.html")
        print("   ✓ Saved: eps_trajectory_analysis.html")

        fig = create_beat_rate_heatmap(df)
        fig.write_html(output_dir / "beat_rate_heatmap.html")
        print("   ✓ Saved: beat_rate_heatmap.html")

        fig = create_earnings_consistency_matrix(df)
        fig.write_html(output_dir / "earnings_consistency_matrix.html")
        print("   ✓ Saved: earnings_consistency_matrix.html")

        if len(df) > 0:
            ticker = df.iloc[0]["ticker"]
            fig = create_earnings_quality_decomposition(df, ticker=ticker)
            fig.write_html(output_dir / f"earnings_quality_decomposition_{ticker}.html")
            print(f"   ✓ Saved: earnings_quality_decomposition_{ticker}.html")

        # --- NEW: Quality & Risk Visualizations ---
        print("   Creating quality & risk visualizations...")

        fig = create_piotroski_fscore_breakdown(df)
        fig.write_html(output_dir / "piotroski_fscore_breakdown.html")
        print("   ✓ Saved: piotroski_fscore_breakdown.html")

        fig = create_altman_zscore_distribution(df)
        fig.write_html(output_dir / "altman_zscore_distribution.html")
        print("   ✓ Saved: altman_zscore_distribution.html")

        fig = create_quality_risk_quadrant(df)
        fig.write_html(output_dir / "quality_risk_quadrant.html")
        print("   ✓ Saved: quality_risk_quadrant.html")

        fig = create_beneish_mscore_analysis(df)
        fig.write_html(output_dir / "beneish_mscore_analysis.html")
        print("   ✓ Saved: beneish_mscore_analysis.html")

        fig = create_risk_tier_sunburst(df)
        fig.write_html(output_dir / "risk_tier_sunburst.html")
        print("   ✓ Saved: risk_tier_sunburst.html")

        fig = create_distress_early_warning_dashboard(df)
        fig.write_html(output_dir / "distress_early_warning_dashboard.html")
        print("   ✓ Saved: distress_early_warning_dashboard.html")

        # --- NEW: Growth Analysis Visualizations ---
        print("   Creating growth analysis visualizations...")

        fig = create_growth_consistency_matrix(df)
        fig.write_html(output_dir / "growth_consistency_matrix.html")
        print("   ✓ Saved: growth_consistency_matrix.html")

        fig = create_growth_vs_profitability_quadrant(df)
        fig.write_html(output_dir / "growth_vs_profitability_quadrant.html")
        print("   ✓ Saved: growth_vs_profitability_quadrant.html")

        fig = create_growth_acceleration_chart(df)
        fig.write_html(output_dir / "growth_acceleration_chart.html")
        print("   ✓ Saved: growth_acceleration_chart.html")

        fig = create_sustainable_growth_analysis(df)
        fig.write_html(output_dir / "sustainable_growth_analysis.html")
        print("   ✓ Saved: sustainable_growth_analysis.html")

        if len(df) > 0:
            ticker = df.iloc[0]["ticker"]
            fig = create_growth_waterfall_chart(df, ticker=ticker)
            fig.write_html(output_dir / f"growth_waterfall_{ticker}.html")
            print(f"   ✓ Saved: growth_waterfall_{ticker}.html")

    except Exception as e:
        print(f"   ⚠️  Visualization error: {e}")

    print()

    # ========================================================================
    # 8. EXPORT RESULTS
    # ========================================================================
    print("💾 Step 8: Exporting results...")
    print("-" * 80)

    # Export feature statistics
    stats_data = []
    for category, features in FEATURE_CATEGORIES.items():
        for feature in features:
            if feature in df.columns:
                stats = compute_metric_statistics(df[feature])
                if stats:
                    stats["category"] = category
                    stats["feature"] = feature
                    stats_data.append(stats)

    # Helper to reorder DataFrame columns: identifier cols first, then the rest
    identifier_cols = load_identifier_columns()

    def _reorder_with_identifiers(result_df: pd.DataFrame) -> pd.DataFrame:
        id_cols = [c for c in identifier_cols if c in result_df.columns]
        other_cols = [c for c in result_df.columns if c not in id_cols]
        return result_df[id_cols + other_cols]

    if stats_data:
        stats_df = _reorder_with_identifiers(pd.DataFrame(stats_data))
        export_to_analytics_db(stats_df, "feature_statistics")
        print(f"   ✓ Exported {len(stats_df)} features to analytics.feature_statistics")

    # Export screened stocks
    if "quality_stocks" in locals() and len(quality_stocks) > 0:
        export_to_analytics_db(_reorder_with_identifiers(quality_stocks), "quality_stocks")
        print(f"   ✓ Exported {len(quality_stocks)} stocks to analytics.quality_stocks")

    # --- NEW: Export composite scores ---
    if "ranked_df" in locals() and "composite_score" in ranked_df.columns:
        export_cols = ["composite_score"]
        id_cols_present = [c for c in identifier_cols if c in ranked_df.columns]
        available = id_cols_present + [c for c in export_cols if c in ranked_df.columns]
        export_to_analytics_db(ranked_df[available].head(200), "composite_quality_scores")
        print(f"   ✓ Exported top 200 composite scores to analytics.composite_quality_scores")

    # --- NEW: Export Kalman-filtered price targets ---
    if "kalman_pt" in locals() and len(kalman_pt) > 0:
        export_to_analytics_db(
            _reorder_with_identifiers(kalman_pt), "kalman_filtered_price_targets"
        )
        print(f"   ✓ Exported {len(kalman_pt)} Kalman-filtered targets")

    # --- NEW: Export Monte Carlo simulation results ---
    if "mc_results" in locals() and len(mc_results) > 0:
        export_to_analytics_db(_reorder_with_identifiers(mc_results), "monte_carlo_simulation")
        print(
            f"   ✓ Exported {len(mc_results)} Monte Carlo results to analytics.monte_carlo_simulation"
        )

    print()

    # ========================================================================
    # SUMMARY (UPDATED)
    # ========================================================================
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Refactored modules used:")
    print("  • data_utils: Data loading and preprocessing")
    print("  • statistical_analysis: Bayesian, MCMC, Kalman, Copula, Monte Carlo")
    print("  • screening: Multi-factor stock screening (12 screeners)")
    print("  • feature_analytics: Interactive visualizations")
    print("  • optimized_ops: Numba MC, fast ruin, vectorized stats, caching")
    print("  • visualizations.profitability: Margin and DuPont analysis")
    print("  • visualizations.technical: Momentum and range charts")
    print("  • visualizations.temporal_analysis: Earnings and dividend timelines")
    print("  • visualizations.category_charts: Category-specific charts")
    print("  • visualizations.valuation: Valuation ratio analysis")
    print("  • visualizations.earnings_quality: Earnings quality charts")
    print("  • visualizations.quality_risk: Quality & risk assessment")
    print("  • visualizations.growth_analysis: Growth metrics analysis")
    if PROBABILITY_ANALYTICS_AVAILABLE:
        print(
            "  • probability_analytics: Bayesian beat, credit risk, dividend safety, PT achievement"
        )
    print()
    print(f"Total stocks analyzed: {len(df):,}")
    print(f"Total features: {len(df.columns)}")
    print(f"Feature categories: {len(FEATURE_CATEGORIES)}")
    if probability_results is not None:
        print(f"Probability analysis completed: {len(probability_results)} stocks")
    if streak_results is not None:
        print(f"Streak analysis completed: {len(streak_results)} stocks")
    if credit_results is not None:
        print(f"Credit risk analysis completed: {len(credit_results)} stocks")
    if dividend_results is not None:
        print(f"Dividend safety analysis completed: {len(dividend_results)} stocks")
    if pt_results is not None:
        print(f"Price target achievement analysis completed: {len(pt_results)} stocks")
    print()
    print("Check the 'outputs/analytics/' directory for generated files.")
    print()


if __name__ == "__main__":
    main()
