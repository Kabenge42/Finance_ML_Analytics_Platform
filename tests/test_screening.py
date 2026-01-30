"""
Unit tests for finance_ml.analytics.screening module.

TDD tests for stock screening and filtering utilities following strict TDD approach:
1. Write failing tests first (Red)
2. Implement minimal code to pass (Green)
3. Refactor while keeping tests passing (Refactor)

Tests cover:
- create_enhanced_screener
- screen_earnings_quality
- screen_value_opportunities
- screen_growth_momentum
- screen_dividend_quality
- screen_financial_health
- rank_stocks_by_composite_score
- create_sector_relative_ranking
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# Fixtures for Screening Tests
# =============================================================================


@pytest.fixture
def sample_screening_df() -> pd.DataFrame:
    """Create a sample DataFrame with all columns needed for screening functions."""
    np.random.seed(42)
    n = 50

    industries = ["Technology", "Healthcare", "Financials", "Energy", "Consumer"]

    df = pd.DataFrame(
        {
            # Identifiers
            "ticker": [f"TICK{i:03d}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "industry": np.random.choice(industries, n),
            # Quality metrics for enhanced screener
            "piotroski_f_score": np.random.randint(0, 10, n),
            "distress_risk_score": np.random.uniform(10, 95, n).round(1),
            "eps_trajectory_score": np.random.uniform(20, 90, n).round(1),
            "fcf_positive_years": np.random.randint(0, 6, n),
            "debt_deleveraging": np.random.choice([0, 1], n),
            "secular_trend_flag": np.random.choice([0, 1], n),
            # Earnings quality columns
            "earnings_quality_composite_comp": np.random.uniform(30, 90, n).round(1),
            "eps_adjustment_pct": np.random.uniform(-30, 30, n).round(2),
            "gaap_positive_revision_flag": np.random.choice([0, 1], n),
            "net_income_positive_years": np.random.randint(0, 6, n),
            # Value opportunity columns
            "p_e_ratio": np.random.uniform(5, 60, n).round(2),
            "upside_potential": np.random.uniform(-20, 80, n).round(2),
            "price_to_tangible_book": np.random.uniform(0.5, 5, n).round(2),
            "fcf_yield": np.random.uniform(-5, 15, n).round(2),
            # Growth momentum columns
            "revenue_growth_yoy": np.random.uniform(-20, 50, n).round(2),
            "eps_yoy_growth": np.random.uniform(-30, 60, n).round(2),
            "price_momentum_1y": np.random.uniform(-40, 80, n).round(2),
            "long_term_trend_score": np.random.uniform(20, 90, n).round(1),
            "rnd_intensity_ltm": np.random.uniform(0, 25, n).round(2),
            # Dividend quality columns
            "dividend_yield_ltm": np.random.uniform(0, 8, n).round(2),
            "dividend_streak": np.random.randint(0, 15, n),
            "dividend_payout_ratio": np.random.uniform(0, 120, n).round(1),
            "fcf_dividend_coverage": np.random.uniform(0, 5, n).round(2),
            "dividend_growth_5y_cagr": np.random.uniform(-5, 20, n).round(2),
            # Financial health columns
            "debt_to_equity": np.random.uniform(0, 3, n).round(2),
            "current_ratio": np.random.uniform(0.5, 4, n).round(2),
            "interest_coverage_ratio": np.random.uniform(0, 20, n).round(2),
            "working_capital_ratio": np.random.uniform(-0.5, 2, n).round(2),
            # Market cap for ranking
            "market_cap": np.random.uniform(1e9, 1e12, n),
        }
    )

    return df


@pytest.fixture
def minimal_screening_df() -> pd.DataFrame:
    """Minimal DataFrame with only required columns for create_enhanced_screener."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "piotroski_f_score": [8, 6, 7, 5],
            "distress_risk_score": [85, 70, 80, 65],
            "eps_trajectory_score": [75, 55, 60, 45],
            "fcf_positive_years": [5, 3, 4, 2],
        }
    )


@pytest.fixture
def empty_screening_df() -> pd.DataFrame:
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame()


# =============================================================================
# Tests for create_enhanced_screener
# =============================================================================


class TestCreateEnhancedScreener:
    """Tests for create_enhanced_screener function."""

    def test_returns_dataframe(self, sample_screening_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.screening import create_enhanced_screener

        result = create_enhanced_screener(sample_screening_df)

        assert isinstance(result, pd.DataFrame)

    def test_filters_by_min_fscore(self, minimal_screening_df):
        """Should filter stocks by minimum Piotroski F-Score."""
        from finance_ml.analytics.screening import create_enhanced_screener

        result = create_enhanced_screener(minimal_screening_df, min_fscore=7)

        assert len(result) == 2  # Only AAPL (8) and GOOGL (7) pass
        assert all(result["piotroski_f_score"] >= 7)

    def test_filters_by_distress_risk(self, minimal_screening_df):
        """Should filter stocks by distress risk score threshold."""
        from finance_ml.analytics.screening import create_enhanced_screener

        # max_distress_risk=70 means distress_risk_score >= 30 (100 - 70)
        result = create_enhanced_screener(minimal_screening_df, min_fscore=1, max_distress_risk=30)

        # distress_risk_score >= 70: AAPL(85), MSFT(70), GOOGL(80) = 3 stocks
        assert all(result["distress_risk_score"] >= 70)

    def test_filters_by_fcf_positive_years(self, minimal_screening_df):
        """Should filter by minimum FCF positive years."""
        from finance_ml.analytics.screening import create_enhanced_screener

        result = create_enhanced_screener(
            minimal_screening_df, min_fscore=1, min_fcf_positive_years=4
        )

        assert len(result) == 2  # AAPL (5) and GOOGL (4)
        assert all(result["fcf_positive_years"] >= 4)

    def test_returns_empty_when_missing_required_columns(self, empty_screening_df):
        """Should return empty DataFrame when required columns are missing."""
        from finance_ml.analytics.screening import create_enhanced_screener

        result = create_enhanced_screener(empty_screening_df)

        assert len(result) == 0

    def test_require_deleveraging_filter(self, sample_screening_df):
        """Should filter by debt deleveraging when required."""
        from finance_ml.analytics.screening import create_enhanced_screener

        result = create_enhanced_screener(
            sample_screening_df, min_fscore=1, require_deleveraging=True
        )

        assert all(result["debt_deleveraging"] == 1)

    def test_sector_filter(self, sample_screening_df):
        """Should filter by sector when specified."""
        from finance_ml.analytics.screening import create_enhanced_screener

        result = create_enhanced_screener(
            sample_screening_df, min_fscore=1, sector_filter="Technology"
        )

        if len(result) > 0:
            assert all(result["industry"] == "Technology")

    def test_sorted_by_fscore_descending(self, sample_screening_df):
        """Results should be sorted by piotroski_f_score descending."""
        from finance_ml.analytics.screening import create_enhanced_screener

        result = create_enhanced_screener(sample_screening_df, min_fscore=5)

        if len(result) > 1:
            fscores = result["piotroski_f_score"].tolist()
            assert fscores == sorted(fscores, reverse=True)

    def test_example_from_refactoring_guide(self):
        """Test the exact example from the refactoring guide."""
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
# Tests for screen_earnings_quality
# =============================================================================


class TestScreenEarningsQuality:
    """Tests for screen_earnings_quality function."""

    def test_returns_dataframe(self, sample_screening_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.screening import screen_earnings_quality

        result = screen_earnings_quality(sample_screening_df)

        assert isinstance(result, pd.DataFrame)

    def test_filters_by_quality_score(self, sample_screening_df):
        """Should filter by minimum earnings quality score."""
        from finance_ml.analytics.screening import screen_earnings_quality

        result = screen_earnings_quality(sample_screening_df, min_quality_score=70)

        if len(result) > 0:
            assert all(result["earnings_quality_composite_comp"] >= 70)

    def test_filters_by_adjustment_pct(self, sample_screening_df):
        """Should filter by maximum EPS adjustment percentage."""
        from finance_ml.analytics.screening import screen_earnings_quality

        result = screen_earnings_quality(sample_screening_df, max_adjustment_pct=10)

        if len(result) > 0:
            assert all(result["eps_adjustment_pct"].abs() <= 10)

    def test_require_positive_revisions(self, sample_screening_df):
        """Should filter by positive GAAP revisions when required."""
        from finance_ml.analytics.screening import screen_earnings_quality

        result = screen_earnings_quality(sample_screening_df, require_positive_revisions=True)

        if len(result) > 0:
            assert all(result["gaap_positive_revision_flag"] == 1)

    def test_sorted_by_quality_descending(self, sample_screening_df):
        """Results should be sorted by earnings quality descending."""
        from finance_ml.analytics.screening import screen_earnings_quality

        result = screen_earnings_quality(sample_screening_df, min_quality_score=50)

        if len(result) > 1:
            scores = result["earnings_quality_composite_comp"].tolist()
            assert scores == sorted(scores, reverse=True)


# =============================================================================
# Tests for screen_value_opportunities
# =============================================================================


class TestScreenValueOpportunities:
    """Tests for screen_value_opportunities function."""

    def test_returns_dataframe(self, sample_screening_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.screening import screen_value_opportunities

        result = screen_value_opportunities(sample_screening_df)

        assert isinstance(result, pd.DataFrame)

    def test_filters_by_pe_ratio(self, sample_screening_df):
        """Should filter by maximum P/E ratio."""
        from finance_ml.analytics.screening import screen_value_opportunities

        result = screen_value_opportunities(sample_screening_df, max_pe_ratio=20)

        if len(result) > 0:
            assert all((result["p_e_ratio"] > 0) & (result["p_e_ratio"] <= 20))

    def test_filters_by_upside_potential(self, sample_screening_df):
        """Should filter by minimum upside potential."""
        from finance_ml.analytics.screening import screen_value_opportunities

        result = screen_value_opportunities(sample_screening_df, min_upside_potential=30)

        if len(result) > 0:
            assert all(result["upside_potential"] >= 30)

    def test_require_positive_fcf(self, sample_screening_df):
        """Should filter by positive FCF yield when required."""
        from finance_ml.analytics.screening import screen_value_opportunities

        result = screen_value_opportunities(sample_screening_df, require_positive_fcf=True)

        if len(result) > 0:
            assert all(result["fcf_yield"] > 0)

    def test_sorted_by_upside_descending(self, sample_screening_df):
        """Results should be sorted by upside potential descending."""
        from finance_ml.analytics.screening import screen_value_opportunities

        result = screen_value_opportunities(sample_screening_df, max_pe_ratio=50)

        if len(result) > 1:
            upsides = result["upside_potential"].tolist()
            assert upsides == sorted(upsides, reverse=True)


# =============================================================================
# Tests for screen_growth_momentum
# =============================================================================


class TestScreenGrowthMomentum:
    """Tests for screen_growth_momentum function."""

    def test_returns_dataframe(self, sample_screening_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.screening import screen_growth_momentum

        result = screen_growth_momentum(sample_screening_df)

        assert isinstance(result, pd.DataFrame)

    def test_filters_by_revenue_growth(self, sample_screening_df):
        """Should filter by minimum revenue growth."""
        from finance_ml.analytics.screening import screen_growth_momentum

        result = screen_growth_momentum(sample_screening_df, min_revenue_growth=20)

        if len(result) > 0:
            assert all(result["revenue_growth_yoy"] >= 20)

    def test_filters_by_eps_growth(self, sample_screening_df):
        """Should filter by minimum EPS growth."""
        from finance_ml.analytics.screening import screen_growth_momentum

        result = screen_growth_momentum(sample_screening_df, min_eps_growth=15)

        if len(result) > 0:
            assert all(result["eps_yoy_growth"] >= 15)

    def test_require_rnd_investment(self, sample_screening_df):
        """Should filter by R&D investment when required."""
        from finance_ml.analytics.screening import screen_growth_momentum

        result = screen_growth_momentum(sample_screening_df, require_rnd_investment=True)

        if len(result) > 0:
            assert all(result["rnd_intensity_ltm"] > 0)

    def test_sorted_by_trend_score_descending(self, sample_screening_df):
        """Results should be sorted by long term trend score descending."""
        from finance_ml.analytics.screening import screen_growth_momentum

        result = screen_growth_momentum(sample_screening_df, min_revenue_growth=0)

        if len(result) > 1:
            scores = result["long_term_trend_score"].tolist()
            assert scores == sorted(scores, reverse=True)


# =============================================================================
# Tests for screen_dividend_quality
# =============================================================================


class TestScreenDividendQuality:
    """Tests for screen_dividend_quality function."""

    def test_returns_dataframe(self, sample_screening_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.screening import screen_dividend_quality

        result = screen_dividend_quality(sample_screening_df)

        assert isinstance(result, pd.DataFrame)

    def test_filters_by_dividend_yield(self, sample_screening_df):
        """Should filter by minimum dividend yield."""
        from finance_ml.analytics.screening import screen_dividend_quality

        result = screen_dividend_quality(sample_screening_df, min_dividend_yield=3.0)

        if len(result) > 0:
            assert all(result["dividend_yield_ltm"] >= 3.0)

    def test_filters_by_dividend_streak(self, sample_screening_df):
        """Should filter by minimum dividend streak years."""
        from finance_ml.analytics.screening import screen_dividend_quality

        result = screen_dividend_quality(sample_screening_df, min_dividend_streak=5)

        if len(result) > 0:
            assert all(result["dividend_streak"] >= 5)

    def test_filters_by_payout_ratio(self, sample_screening_df):
        """Should filter by maximum payout ratio."""
        from finance_ml.analytics.screening import screen_dividend_quality

        result = screen_dividend_quality(sample_screening_df, max_payout_ratio=60)

        if len(result) > 0:
            assert all(result["dividend_payout_ratio"] <= 60)


# =============================================================================
# Tests for screen_financial_health
# =============================================================================


class TestScreenFinancialHealth:
    """Tests for screen_financial_health function."""

    def test_returns_dataframe(self, sample_screening_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.screening import screen_financial_health

        result = screen_financial_health(sample_screening_df)

        assert isinstance(result, pd.DataFrame)

    def test_filters_by_distress_score(self, sample_screening_df):
        """Should filter by minimum distress score."""
        from finance_ml.analytics.screening import screen_financial_health

        result = screen_financial_health(sample_screening_df, min_distress_score=80)

        if len(result) > 0:
            assert all(result["distress_risk_score"] >= 80)

    def test_filters_by_debt_to_equity(self, sample_screening_df):
        """Should filter by maximum debt to equity ratio."""
        from finance_ml.analytics.screening import screen_financial_health

        result = screen_financial_health(sample_screening_df, max_debt_to_equity=0.5)

        if len(result) > 0:
            assert all(result["debt_to_equity"] <= 0.5)

    def test_filters_by_current_ratio(self, sample_screening_df):
        """Should filter by minimum current ratio."""
        from finance_ml.analytics.screening import screen_financial_health

        result = screen_financial_health(sample_screening_df, min_current_ratio=2.0)

        if len(result) > 0:
            assert all(result["current_ratio"] >= 2.0)


# =============================================================================
# Tests for rank_stocks_by_composite_score
# =============================================================================


class TestRankStocksByCompositeScore:
    """Tests for rank_stocks_by_composite_score function."""

    def test_returns_dataframe(self, sample_screening_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.screening import rank_stocks_by_composite_score

        result = rank_stocks_by_composite_score(sample_screening_df)

        assert isinstance(result, pd.DataFrame)

    def test_adds_composite_score_column(self, sample_screening_df):
        """Should add a composite_score column."""
        from finance_ml.analytics.screening import rank_stocks_by_composite_score

        result = rank_stocks_by_composite_score(sample_screening_df)

        assert "composite_score" in result.columns

    def test_sorted_by_composite_score_descending(self, sample_screening_df):
        """Results should be sorted by composite score descending."""
        from finance_ml.analytics.screening import rank_stocks_by_composite_score

        result = rank_stocks_by_composite_score(sample_screening_df)

        if len(result) > 1:
            scores = result["composite_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_custom_weights(self, sample_screening_df):
        """Should accept custom weights for scoring."""
        from finance_ml.analytics.screening import rank_stocks_by_composite_score

        custom_weights = {"piotroski_f_score": 0.5, "distress_risk_score": 0.5}

        result = rank_stocks_by_composite_score(sample_screening_df, weights=custom_weights)

        assert "composite_score" in result.columns


# =============================================================================
# Tests for create_sector_relative_ranking
# =============================================================================


class TestCreateSectorRelativeRanking:
    """Tests for create_sector_relative_ranking function."""

    def test_returns_dataframe(self, sample_screening_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.screening import create_sector_relative_ranking

        result = create_sector_relative_ranking(sample_screening_df, metric="piotroski_f_score")

        assert isinstance(result, pd.DataFrame)

    def test_adds_sector_rank_column(self, sample_screening_df):
        """Should add a sector rank column."""
        from finance_ml.analytics.screening import create_sector_relative_ranking

        result = create_sector_relative_ranking(sample_screening_df, metric="piotroski_f_score")

        assert "sector_rank" in result.columns or "industry_rank" in result.columns

    def test_ranking_within_sectors(self, sample_screening_df):
        """Ranking should be relative within each sector."""
        from finance_ml.analytics.screening import create_sector_relative_ranking

        result = create_sector_relative_ranking(sample_screening_df, metric="piotroski_f_score")

        # Each sector should have its own ranking starting from 1
        rank_col = "sector_rank" if "sector_rank" in result.columns else "industry_rank"
        for industry in result["industry"].unique():
            sector_ranks = result[result["industry"] == industry][rank_col]
            if len(sector_ranks) > 0:
                assert sector_ranks.min() == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestScreeningIntegration:
    """Integration tests for screening module."""

    def test_all_functions_importable(self):
        """All screening functions should be importable."""
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

        assert callable(create_enhanced_screener)
        assert callable(screen_earnings_quality)
        assert callable(screen_value_opportunities)
        assert callable(screen_growth_momentum)
        assert callable(screen_dividend_quality)
        assert callable(screen_financial_health)
        assert callable(rank_stocks_by_composite_score)
        assert callable(create_sector_relative_ranking)

    def test_chained_screening_workflow(self, sample_screening_df):
        """Test chaining multiple screening functions."""
        from finance_ml.analytics.screening import (
            create_enhanced_screener,
            screen_financial_health,
        )

        # First pass: quality screening
        quality = create_enhanced_screener(sample_screening_df, min_fscore=5)

        # Second pass: financial health
        if len(quality) > 0:
            healthy = screen_financial_health(quality, min_distress_score=50)
            assert isinstance(healthy, pd.DataFrame)

    def test_empty_dataframe_handling(self, empty_screening_df):
        """All functions should handle empty DataFrames gracefully."""
        from finance_ml.analytics.screening import (
            create_enhanced_screener,
            screen_earnings_quality,
            screen_value_opportunities,
            screen_growth_momentum,
        )

        assert len(create_enhanced_screener(empty_screening_df)) == 0
        assert len(screen_earnings_quality(empty_screening_df)) == 0
        assert len(screen_value_opportunities(empty_screening_df)) == 0
        assert len(screen_growth_momentum(empty_screening_df)) == 0
