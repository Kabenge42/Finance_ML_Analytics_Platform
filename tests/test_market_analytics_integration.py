"""
Integration tests for market_analytics refactored modules.

These tests verify that all refactored modules work together correctly
as described in the Market Analytics Refactoring Guide.

Tests cover:
- Full workflow from data loading to screening to analysis
- Cross-module integration
- End-to-end pipeline execution
- FEATURE_CATEGORIES consistency
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# Fixtures for Integration Tests
# =============================================================================


@pytest.fixture
def comprehensive_stock_df() -> pd.DataFrame:
    """Create a comprehensive DataFrame with all columns needed for full workflow."""
    np.random.seed(42)
    n = 100

    industries = ["Technology", "Healthcare", "Financials", "Energy", "Consumer"]

    df = pd.DataFrame(
        {
            # Identifiers
            "ticker": [f"TICK{i:03d}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "industry": np.random.choice(industries, n),
            # Price data
            "last_price": np.random.uniform(10, 500, n).round(2),
            "price_target": np.random.uniform(15, 600, n).round(2),
            "market_cap": np.random.uniform(1e9, 1e12, n),
            # Valuation Ratios
            "p_e_ratio": np.random.uniform(5, 60, n).round(2),
            "p_b_ratio": np.random.uniform(0.5, 10, n).round(2),
            "ev_ebitda_ratio": np.random.uniform(3, 30, n).round(2),
            "ev_sales_ratio": np.random.uniform(0.5, 15, n).round(2),
            "dividend_yield": np.random.uniform(0, 6, n).round(2),
            "peg_ratio": np.random.uniform(0.5, 5, n).round(2),
            "price_to_tangible_book": np.random.uniform(0.5, 8, n).round(2),
            "tangible_book_value_ltm": np.random.uniform(1e6, 1e10, n),
            # Momentum & Technical
            "price_momentum_1m": np.random.uniform(-30, 50, n).round(2),
            "price_momentum_3m": np.random.uniform(-40, 60, n).round(2),
            "price_momentum_6m": np.random.uniform(-50, 80, n).round(2),
            "price_momentum_1y": np.random.uniform(-60, 100, n).round(2),
            "range_52w_position": np.random.uniform(0, 1, n).round(3),
            "long_term_trend_score": np.random.uniform(20, 90, n).round(1),
            "secular_trend_flag": np.random.choice([0, 1], n),
            # Profitability
            "roe": np.random.uniform(-20, 40, n).round(2),
            "roa": np.random.uniform(-10, 25, n).round(2),
            "gross_margin_pct": np.random.uniform(10, 80, n).round(2),
            "operating_margin_pct": np.random.uniform(-10, 40, n).round(2),
            "net_margin_pct": np.random.uniform(-15, 30, n).round(2),
            "ebitda_margin_pct": np.random.uniform(5, 50, n).round(2),
            "roic": np.random.uniform(-5, 30, n).round(2),
            # Quality & Risk
            "piotroski_f_score": np.random.randint(0, 10, n),
            "distress_risk_score": np.random.uniform(10, 95, n).round(1),
            "altman_z_score": np.random.uniform(-2, 10, n).round(2),
            "accounting_quality_score": np.random.uniform(20, 90, n).round(1),
            "earnings_quality_composite_comp": np.random.uniform(20, 90, n).round(1),
            "cash_flow_quality_score": np.random.uniform(20, 90, n).round(1),
            # Leverage & Liquidity
            "debt_to_equity": np.random.uniform(0, 3, n).round(2),
            "current_ratio": np.random.uniform(0.5, 4, n).round(2),
            "quick_ratio": np.random.uniform(0.3, 3, n).round(2),
            "interest_coverage_ratio": np.random.uniform(0, 20, n).round(2),
            "cash_ratio": np.random.uniform(0, 2, n).round(2),
            "working_capital_ratio": np.random.uniform(-0.5, 2, n).round(2),
            "debt_deleveraging": np.random.choice([0, 1], n),
            # Analyst Sentiment
            "analyst_bullish_pct": np.random.uniform(10, 90, n).round(1),
            "analyst_neutral_pct": np.random.uniform(5, 50, n).round(1),
            "analyst_bearish_pct": np.random.uniform(5, 40, n).round(1),
            "upside_potential": np.random.uniform(-30, 80, n).round(2),
            "analyst_rating_normalized": np.random.uniform(1, 5, n).round(2),
            # Earnings Quality
            "eps_surprise_pct": np.random.uniform(-20, 30, n).round(2),
            "eps_adjustment_ratio": np.random.uniform(-0.5, 0.5, n).round(3),
            "eps_trajectory_score": np.random.uniform(20, 90, n).round(1),
            "earnings_quality_score": np.random.uniform(20, 90, n).round(1),
            # Growth Metrics
            "revenue_growth_yoy": np.random.uniform(-30, 60, n).round(2),
            "ebitda_growth_yoy": np.random.uniform(-40, 80, n).round(2),
            "eps_yoy_growth": np.random.uniform(-50, 100, n).round(2),
            "fcf_growth_yoy": np.random.uniform(-60, 100, n).round(2),
            # Cash Flow
            "fcf_positive_years": np.random.randint(0, 6, n),
            "fcf_margin": np.random.uniform(-20, 30, n).round(2),
            "fcf_yield": np.random.uniform(-10, 15, n).round(2),
            "cfo_to_net_income": np.random.uniform(0.5, 3, n).round(2),
            "self_funding_ratio": np.random.uniform(0, 2, n).round(2),
            # Dividend Features
            "dividend_streak": np.random.randint(0, 20, n),
            "dividend_yield_ltm": np.random.uniform(0, 8, n).round(2),
            "dividend_payout_ratio": np.random.uniform(0, 120, n).round(1),
            "fcf_dividend_coverage": np.random.uniform(0, 5, n).round(2),
            "total_shareholder_yield": np.random.uniform(-5, 15, n).round(2),
            # R&D Investment
            "rnd_intensity_ltm": np.random.uniform(0, 25, n).round(2),
            "rnd_yoy_growth": np.random.uniform(-30, 50, n).round(2),
            # Inventory Temporal
            "inventory_days": np.random.uniform(10, 200, n).round(1),
            "inventory_turnover_mv": np.random.uniform(1, 20, n).round(2),
            # Goodwill & M&A
            "goodwill_concentration": np.random.uniform(0, 0.5, n).round(3),
            "goodwill_3y_growth": np.random.uniform(-20, 50, n).round(2),
            # CapEx & Investment
            "capex_yoy_growth": np.random.uniform(-40, 60, n).round(2),
            "capex_vs_5y_avg": np.random.uniform(0.5, 2, n).round(2),
            # Additional columns for statistical analysis
            "cash_burn_rate": np.random.uniform(-1e8, 1e8, n),
            # Use numeric beta_momentum for volatility proxy (not string volatility_regime)
            "beta_momentum": np.random.uniform(0.5, 2.0, n).round(2),
        }
    )

    return df


@pytest.fixture
def feature_categories() -> dict:
    """Feature categories matching market_analytics.py FEATURE_CATEGORIES."""
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
        ],
        "Quality & Risk": [
            "piotroski_f_score",
            "distress_risk_score",
            "altman_z_score",
            "accounting_quality_score",
            "earnings_quality_composite_comp",
            "cash_flow_quality_score",
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
    }


# =============================================================================
# Cross-Module Integration Tests
# =============================================================================


class TestDataUtilsToScreeningWorkflow:
    """Test workflow from data_utils to screening."""

    def test_backfill_then_screen(self, comprehensive_stock_df):
        """Test backfilling data then applying screening."""
        from finance_ml.analytics.data_utils import backfill_feature_columns
        from finance_ml.analytics.screening import create_enhanced_screener

        # Backfill any missing columns
        df = backfill_feature_columns(comprehensive_stock_df)

        # Apply quality screening
        quality_stocks = create_enhanced_screener(df, min_fscore=5)

        assert isinstance(quality_stocks, pd.DataFrame)
        assert len(quality_stocks) <= len(df)

    def test_validate_then_screen(self, comprehensive_stock_df, feature_categories):
        """Test validating features then screening."""
        from finance_ml.analytics.data_utils import validate_feature_alignment
        from finance_ml.analytics.screening import (
            create_enhanced_screener,
            screen_value_opportunities,
        )

        # Validate feature coverage
        validation = validate_feature_alignment(comprehensive_stock_df, feature_categories)

        # Check we have decent coverage for screening
        quality_coverage = validation["Quality & Risk"]["coverage_pct"]
        assert quality_coverage >= 80  # Need quality metrics for screening

        # Apply screenings
        quality = create_enhanced_screener(comprehensive_stock_df, min_fscore=5)
        value = screen_value_opportunities(comprehensive_stock_df, max_pe_ratio=30)

        assert isinstance(quality, pd.DataFrame)
        assert isinstance(value, pd.DataFrame)

    def test_statistics_on_screened_stocks(self, comprehensive_stock_df):
        """Test computing statistics on screened stock subset."""
        from finance_ml.analytics.data_utils import compute_metric_statistics
        from finance_ml.analytics.screening import create_enhanced_screener

        # Screen for quality stocks
        quality_stocks = create_enhanced_screener(comprehensive_stock_df, min_fscore=7)

        if len(quality_stocks) > 0:
            # Compute statistics on screened subset
            pe_stats = compute_metric_statistics(quality_stocks["p_e_ratio"])

            assert pe_stats is not None
            assert "mean" in pe_stats


class TestScreeningToStatisticalWorkflow:
    """Test workflow from screening to statistical analysis."""

    def test_screen_then_bayesian_analysis(self, comprehensive_stock_df):
        """Test screening then Bayesian analysis on results."""
        from finance_ml.analytics.screening import create_enhanced_screener
        from finance_ml.analytics.statistical_analysis import bayesian_category_analysis

        # Screen for quality stocks
        quality_stocks = create_enhanced_screener(comprehensive_stock_df, min_fscore=5)

        if len(quality_stocks) >= 30:
            # Bayesian analysis on quality subset
            bayesian_results = bayesian_category_analysis(
                quality_stocks, "Profitability", ["roe", "roa"]
            )

            assert isinstance(bayesian_results, dict)
            if "roe" in bayesian_results:
                assert "posterior_mean" in bayesian_results["roe"]

    def test_screen_then_ruin_probability(self, comprehensive_stock_df):
        """Test screening then ruin probability calculation."""
        from finance_ml.analytics.screening import screen_financial_health
        from finance_ml.analytics.statistical_analysis import calculate_ruin_probability

        # Screen for financially healthy stocks
        healthy_stocks = screen_financial_health(comprehensive_stock_df, min_distress_score=50)

        if len(healthy_stocks) > 0:
            # Calculate ruin probabilities
            ruin_results = calculate_ruin_probability(healthy_stocks)

            assert isinstance(ruin_results, pd.DataFrame)
            assert "ruin_probability" in ruin_results.columns

    def test_multiple_screens_then_analysis(self, comprehensive_stock_df):
        """Test chaining multiple screens then analysis."""
        from finance_ml.analytics.screening import (
            create_enhanced_screener,
            screen_value_opportunities,
            rank_stocks_by_composite_score,
        )
        from finance_ml.analytics.statistical_analysis import calculate_ruin_probability

        # Quality filter first
        quality = create_enhanced_screener(comprehensive_stock_df, min_fscore=5)

        if len(quality) > 0:
            # Value filter second
            value = screen_value_opportunities(quality, max_pe_ratio=40)

            if len(value) > 0:
                # Rank by composite score
                ranked = rank_stocks_by_composite_score(value)

                # Calculate risk
                ruin_results = calculate_ruin_probability(ranked)

                assert "composite_score" in ranked.columns
                assert "ruin_probability" in ruin_results.columns


class TestFullPipelineIntegration:
    """Test complete pipeline as described in market_analytics.py."""

    def test_complete_workflow(self, comprehensive_stock_df, feature_categories):
        """Test the complete workflow from the refactoring guide."""
        from finance_ml.analytics.data_utils import (
            backfill_feature_columns,
            validate_feature_alignment,
            compute_metric_statistics,
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

        # Step 1: Data preprocessing
        df = backfill_feature_columns(comprehensive_stock_df)

        # Step 2: Feature validation
        validation = validate_feature_alignment(df, feature_categories)
        assert all(v["coverage_pct"] > 0 for v in validation.values())

        # Step 3: Statistical analysis
        bayesian_results = bayesian_category_analysis(df, "Profitability", ["roe", "roa"])
        assert isinstance(bayesian_results, dict)

        ruin_results = calculate_ruin_probability(df)
        assert "ruin_probability" in ruin_results.columns

        # Step 4: Stock screening
        quality_stocks = create_enhanced_screener(df, min_fscore=5)
        value_stocks = screen_value_opportunities(df, max_pe_ratio=35)
        growth_stocks = screen_growth_momentum(df, min_revenue_growth=5)
        healthy_stocks = screen_financial_health(df, min_distress_score=50)

        # Step 5: Feature statistics export
        stats_data = []
        for category, features in feature_categories.items():
            for feature in features:
                if feature in df.columns:
                    stats = compute_metric_statistics(df[feature])
                    if stats:
                        stats["category"] = category
                        stats["feature"] = feature
                        stats_data.append(stats)

        assert len(stats_data) > 0
        stats_df = pd.DataFrame(stats_data)
        assert "mean" in stats_df.columns
        assert "feature" in stats_df.columns

    def test_example_from_guide_quality_stocks(self):
        """Test the exact example from the refactoring guide (lines 336-347)."""
        from finance_ml.analytics.screening import create_enhanced_screener

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "piotroski_f_score": [8, 6],
                "distress_risk_score": [85, 70],
                "eps_trajectory_score": [75, 60],
                "fcf_positive_years": [5, 3],
            }
        )

        result = create_enhanced_screener(df, min_fscore=7)

        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"


# =============================================================================
# FEATURE_CATEGORIES Consistency Tests
# =============================================================================


class TestFeatureCategoriesConsistency:
    """Test that FEATURE_CATEGORIES is consistent across modules."""

    def test_market_analytics_feature_categories_importable(self):
        """FEATURE_CATEGORIES should be accessible from market_analytics."""
        # This imports from market_analytics.py
        import sys

        sys.path.insert(0, ".")

        try:
            from market_analytics import FEATURE_CATEGORIES

            assert isinstance(FEATURE_CATEGORIES, dict)
            assert len(FEATURE_CATEGORIES) >= 10  # Should have many categories

            # Check key categories exist
            assert "Valuation Ratios" in FEATURE_CATEGORIES
            assert "Profitability" in FEATURE_CATEGORIES
            assert "Quality & Risk" in FEATURE_CATEGORIES
        except ImportError:
            pytest.skip("market_analytics.py not in path")

    def test_categories_contain_lists(self, feature_categories):
        """Each category should contain a list of feature names."""
        for category, features in feature_categories.items():
            assert isinstance(features, list), f"{category} should have a list"
            assert len(features) > 0, f"{category} should not be empty"
            for feature in features:
                assert isinstance(feature, str), f"Features in {category} should be strings"


# =============================================================================
# Module Import Tests
# =============================================================================


class TestAllModulesImportable:
    """Test that all refactored modules are importable."""

    def test_data_utils_import(self):
        """data_utils module should be importable."""
        from finance_ml.analytics.data_utils import (
            load_feature_data_from_db,
            backfill_feature_columns,
            compute_metric_statistics,
            validate_feature_alignment,
            safe_get_column,
        )

        assert all(
            [
                callable(load_feature_data_from_db),
                callable(backfill_feature_columns),
                callable(compute_metric_statistics),
                callable(validate_feature_alignment),
                callable(safe_get_column),
            ]
        )

    def test_screening_import(self):
        """screening module should be importable."""
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

        assert all(
            [
                callable(create_enhanced_screener),
                callable(screen_earnings_quality),
                callable(screen_value_opportunities),
                callable(screen_growth_momentum),
                callable(screen_dividend_quality),
                callable(screen_financial_health),
                callable(rank_stocks_by_composite_score),
                callable(create_sector_relative_ranking),
            ]
        )

    def test_statistical_analysis_import(self):
        """statistical_analysis module should be importable."""
        from finance_ml.analytics.statistical_analysis import (
            bayesian_category_analysis,
            metropolis_hastings_sampler,
            hierarchical_mcmc_by_sector,
            fit_distributions_by_category,
            calculate_ruin_probability,
            calculate_conditional_probabilities,
        )

        assert all(
            [
                callable(bayesian_category_analysis),
                callable(metropolis_hastings_sampler),
                callable(hierarchical_mcmc_by_sector),
                callable(fit_distributions_by_category),
                callable(calculate_ruin_probability),
                callable(calculate_conditional_probabilities),
            ]
        )

    def test_feature_analytics_import(self):
        """feature_analytics module should be importable."""
        from finance_ml.analytics.feature_analytics import (
            create_interactive_momentum_dashboard,
            create_interactive_valuation_heatmap,
            create_leverage_liquidity_quadrant,
            monte_carlo_price_target_simulation,
            bayesian_earnings_beat_model,
            analyze_distress_distribution,
            create_composite_quality_score,
            create_summary_dashboard,
        )

        assert all(
            [
                callable(create_interactive_momentum_dashboard),
                callable(create_interactive_valuation_heatmap),
                callable(create_leverage_liquidity_quadrant),
                callable(monte_carlo_price_target_simulation),
                callable(bayesian_earnings_beat_model),
                callable(analyze_distress_distribution),
                callable(create_composite_quality_score),
                callable(create_summary_dashboard),
            ]
        )


# =============================================================================
# Edge Case Integration Tests
# =============================================================================


class TestEdgeCaseHandling:
    """Test edge cases across integrated modules."""

    def test_empty_dataframe_through_pipeline(self):
        """Empty DataFrame should flow through pipeline without errors."""
        from finance_ml.analytics.data_utils import (
            backfill_feature_columns,
            validate_feature_alignment,
        )
        from finance_ml.analytics.screening import create_enhanced_screener

        empty_df = pd.DataFrame()

        # Each step should handle empty gracefully
        backfilled = backfill_feature_columns(empty_df)
        assert isinstance(backfilled, pd.DataFrame)

        validation = validate_feature_alignment(empty_df, {"Test": ["col"]})
        assert isinstance(validation, dict)

        screened = create_enhanced_screener(empty_df)
        assert isinstance(screened, pd.DataFrame)
        assert len(screened) == 0

        # Note: calculate_ruin_probability requires specific columns so we skip it
        # for empty DataFrame test - it's tested separately with proper fixtures

    def test_single_row_dataframe(self):
        """Single-row DataFrame should work through pipeline."""
        from finance_ml.analytics.data_utils import compute_metric_statistics
        from finance_ml.analytics.screening import create_enhanced_screener

        single_row = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "piotroski_f_score": [8],
                "distress_risk_score": [85],
                "eps_trajectory_score": [75],
                "fcf_positive_years": [5],
                "p_e_ratio": [25.0],
            }
        )

        # Screening should work
        result = create_enhanced_screener(single_row, min_fscore=7)
        assert len(result) == 1

        # Statistics might return None for single value
        stats = compute_metric_statistics(single_row["p_e_ratio"])
        assert stats is None or isinstance(stats, dict)

    def test_all_nan_column(self, comprehensive_stock_df):
        """Handle columns that are all NaN."""
        from finance_ml.analytics.data_utils import compute_metric_statistics

        df = comprehensive_stock_df.copy()
        df["all_nan_col"] = np.nan

        stats = compute_metric_statistics(df["all_nan_col"])
        assert stats is None

    def test_extreme_values(self):
        """Handle extreme values in data."""
        from finance_ml.analytics.screening import screen_value_opportunities

        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "p_e_ratio": [0.001, 1000000, 25],
                "upside_potential": [-1000, 5000, 30],
                "fcf_yield": [-100, 200, 5],
                "piotroski_f_score": [0, 9, 5],
            }
        )

        result = screen_value_opportunities(df, max_pe_ratio=50)

        # Should filter extreme values
        assert isinstance(result, pd.DataFrame)
