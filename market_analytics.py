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
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# Import refactored modules
from finance_ml.analytics.data_utils import (
    load_feature_data_from_db,
    backfill_feature_columns,
    compute_metric_statistics,
    validate_feature_alignment,
)
from finance_ml.analytics.feature_analytics import (
    PLOTLY_TEMPLATE,
    create_interactive_momentum_dashboard,
    create_interactive_valuation_heatmap,
    create_summary_dashboard,
)
from finance_ml.analytics.screening import (
    create_enhanced_screener,
    screen_value_opportunities,
    screen_growth_momentum,
    screen_financial_health,
)
from finance_ml.analytics.statistical_analysis import (
    bayesian_category_analysis,
    calculate_ruin_probability,
)
# Category-specific chart visualizations
from finance_ml.analytics.visualizations.category_charts import (
    # Analyst Sentiment
    create_analyst_sentiment_histogram,
    # Earnings Quality
    create_eps_surprise_histogram,
    # Growth Metrics
    # Cash Flow
    create_fcf_margin_yield_scatter,
    # Dividend Features
    # R&D Investment
    create_rnd_intensity_boxplot,
    # Inventory
    # Goodwill & M&A
    create_goodwill_concentration_boxplot,
    # CapEx & Investment
    create_capex_growth_scatter,
)
# Profitability visualizations
from finance_ml.analytics.visualizations.profitability import (
    create_margin_waterfall_chart,
    create_profitability_quadrant,
)
# Technical analysis visualizations
from finance_ml.analytics.visualizations.technical import (
    create_momentum_ribbon_chart,
    create_52w_range_distribution,
)
# Temporal analysis visualizations
from finance_ml.analytics.visualizations.temporal_analysis import (
    create_earnings_calendar_heatmap,
    create_dividend_streak_timeline,
)

# Configure logging and warnings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore")

# Configure plotting
plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", "{:.2f}".format)
px.defaults.template = PLOTLY_TEMPLATE


# Feature Category Definitions
FEATURE_CATEGORIES = {
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
        "revenue_growth_yoy",
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
        "inventory_turnover_mv",
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


def main():
    """Main execution function demonstrating refactored modules."""

    print("=" * 80)
    print("MARKET ANALYTICS - REFACTORED VERSION")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. DATA LOADING AND PREPROCESSING
    # ========================================================================
    print("📊 Step 1: Loading and preprocessing data...")
    print("-" * 80)

    try:
        # Load data from database
        df = load_feature_data_from_db()
        print(f"✓ Loaded {len(df):,} stocks with {len(df.columns)} features")

        # Backfill missing columns
        df = backfill_feature_columns(df)
        print(f"✓ Backfilled features, now have {len(df.columns)} columns")

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
    # 3. STATISTICAL ANALYSIS
    # ========================================================================
    print("📈 Step 3: Running statistical analysis...")
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

    # Investor's ruin probability
    if all(col in df.columns for col in ["market_cap", "distress_risk_score"]):
        print("   Calculating investor's ruin probabilities...")
        ruin_df = calculate_ruin_probability(df)
        high_risk_count = (ruin_df["ruin_probability"] > 0.6).sum()
        print(f"   ✓ Identified {high_risk_count} high-risk stocks")

    print()

    # ========================================================================
    # 4. STOCK SCREENING
    # ========================================================================
    print("🔍 Step 4: Running stock screens...")
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
    if "revenue_growth_yoy" in df.columns:
        growth_stocks = screen_growth_momentum(df, min_revenue_growth=5)
        print(f"   ✓ Growth screen: {len(growth_stocks)} stocks")

    # Financial health
    if "distress_risk_score" in df.columns:
        healthy_stocks = screen_financial_health(df, min_distress_score=80)
        print(f"   ✓ Financial health screen: {len(healthy_stocks)} stocks")

    print()

    # ========================================================================
    # 5. VISUALIZATIONS
    # ========================================================================
    print("📊 Step 5: Generating visualizations...")
    print("-" * 80)

    output_dir = Path("outputs")
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
            fig.write_html(output_dir / "momentum_dashboard_refactored.html")
            print("   ✓ Saved: outputs/momentum_dashboard_refactored.html")

        # Valuation heatmap
        if "p_e_ratio" in df.columns and "industry" in df.columns:
            print("   Creating valuation heatmap...")
            fig = create_interactive_valuation_heatmap(df)
            fig.write_html(output_dir / "valuation_heatmap_refactored.html")
            print("   ✓ Saved: outputs/valuation_heatmap_refactored.html")

        # Summary dashboard
        print("   Creating summary dashboard...")
        fig = create_summary_dashboard(df)
        fig.write_html(output_dir / "summary_dashboard_refactored.html")
        print("   ✓ Saved: outputs/summary_dashboard_refactored.html")

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

    except Exception as e:
        print(f"   ⚠️  Visualization error: {e}")

    print()

    # ========================================================================
    # 6. EXPORT RESULTS
    # ========================================================================
    print("💾 Step 6: Exporting results...")
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

    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(output_dir / "feature_statistics_refactored.csv", index=False)
        print(f"   ✓ Saved: outputs/feature_statistics_refactored.csv ({len(stats_df)} features)")

    # Export screened stocks
    if "quality_stocks" in locals() and len(quality_stocks) > 0:
        quality_stocks.to_csv(output_dir / "quality_stocks_refactored.csv", index=False)
        print(f"   ✓ Saved: outputs/quality_stocks_refactored.csv ({len(quality_stocks)} stocks)")

    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Refactored modules used:")
    print("  • data_utils: Data loading and preprocessing")
    print("  • statistical_analysis: Bayesian, MCMC, Monte Carlo")
    print("  • screening: Multi-factor stock screening")
    print("  • feature_analytics: Interactive visualizations")
    print("  • visualizations.profitability: Margin and DuPont analysis")
    print("  • visualizations.technical: Momentum and range charts")
    print("  • visualizations.temporal_analysis: Earnings and dividend timelines")
    print("  • visualizations.category_charts: Category-specific charts")
    print()
    print(f"Total stocks analyzed: {len(df):,}")
    print(f"Total features: {len(df.columns)}")
    print(f"Feature categories: {len(FEATURE_CATEGORIES)}")
    print()
    print("Check the 'outputs/' directory for generated files.")
    print()


if __name__ == "__main__":
    main()
