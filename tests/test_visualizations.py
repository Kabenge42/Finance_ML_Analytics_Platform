"""
Tests for visualization modules.

Tests cover:
- visualizations/profitability.py
- visualizations/technical.py
- visualizations/temporal_analysis.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from plotly.graph_objs import Figure

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_profitability_df() -> pd.DataFrame:
    """Create sample DataFrame with profitability metrics."""
    np.random.seed(42)
    n = 50

    return pd.DataFrame(
        {
            "ticker": [f"TICK{i:03d}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "industry": np.random.choice(["Technology", "Healthcare", "Financials", "Energy"], n),
            "roe": np.random.uniform(-10, 40, n),
            "roa": np.random.uniform(-5, 20, n),
            "roic": np.random.uniform(-5, 30, n),
            "gross_margin_pct": np.random.uniform(20, 80, n),
            "operating_margin_pct": np.random.uniform(5, 40, n),
            "ebitda_margin_pct": np.random.uniform(10, 50, n),
            "net_margin_pct": np.random.uniform(-5, 30, n),
            "net_margin_trend_yoy": np.random.uniform(-20, 20, n),
            "debt_to_equity": np.random.uniform(0, 3, n),
        }
    )


@pytest.fixture
def sample_technical_df() -> pd.DataFrame:
    """Create sample DataFrame with technical/momentum metrics."""
    np.random.seed(42)
    n = 50

    return pd.DataFrame(
        {
            "ticker": [f"TICK{i:03d}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "industry": np.random.choice(["Technology", "Healthcare", "Financials", "Energy"], n),
            "price_momentum_1m": np.random.uniform(-30, 30, n),
            "price_momentum_3m": np.random.uniform(-40, 40, n),
            "price_momentum_6m": np.random.uniform(-50, 50, n),
            "price_momentum_1y": np.random.uniform(-60, 80, n),
            "price_momentum_3y": np.random.uniform(-50, 150, n),
            "price_momentum_5y": np.random.uniform(-30, 200, n),
            "range_52w_position": np.random.uniform(0, 1, n),
            "long_term_trend_score": np.random.uniform(20, 80, n),
            "secular_trend_flag": np.random.choice([0, 1], n),
        }
    )


@pytest.fixture
def sample_temporal_df() -> pd.DataFrame:
    """Create sample DataFrame with temporal/time series metrics."""
    np.random.seed(42)
    n = 50

    # Generate dates for next 3 months
    base_date = pd.Timestamp("2026-02-01")
    dates = [base_date + pd.Timedelta(days=np.random.randint(0, 90)) for _ in range(n)]

    return pd.DataFrame(
        {
            "ticker": [f"TICK{i:03d}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "industry": np.random.choice(["Technology", "Healthcare", "Financials", "Energy"], n),
            "next_earnings": dates,
            "earnings_quality_composite_comp": np.random.uniform(30, 90, n),
            "inventory_days": np.random.uniform(20, 200, n),
            "inventory_turnover_mv": np.random.uniform(2, 20, n),
            "inventory_yoy_change": np.random.uniform(-30, 50, n),
            "inventory_buildup_flag": np.random.choice([0, 1], n),
            "fcf_positive_years": np.random.randint(0, 6, n),
            "fcf_margin": np.random.uniform(-20, 30, n),
            "fcf_yield": np.random.uniform(-10, 15, n),
            "fcf_growth_yoy": np.random.uniform(-50, 100, n),
            "dividend_streak": np.random.randint(0, 25, n),
            "dividend_yield_ltm": np.random.uniform(0, 8, n),
            "dividend_payout_ratio": np.random.uniform(0, 100, n),
        }
    )


# =============================================================================
# Profitability Visualization Tests
# =============================================================================


class TestMarginWaterfallChart:
    """Tests for create_margin_waterfall_chart function."""

    def test_returns_plotly_figure(self, sample_profitability_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.profitability import create_margin_waterfall_chart

        fig = create_margin_waterfall_chart(sample_profitability_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self):
        """Function should handle missing margin columns gracefully."""
        from finance_ml.analytics.visualizations.profitability import create_margin_waterfall_chart

        df = pd.DataFrame({"ticker": ["A", "B"], "other_col": [1, 2]})
        fig = create_margin_waterfall_chart(df)
        assert isinstance(fig, Figure)

    def test_specific_ticker(self, sample_profitability_df):
        """Function should work with specific ticker."""
        from finance_ml.analytics.visualizations.profitability import create_margin_waterfall_chart

        fig = create_margin_waterfall_chart(sample_profitability_df, ticker="TICK001")
        assert isinstance(fig, Figure)


class TestDupontDecompositionDashboard:
    """Tests for create_dupont_decomposition_dashboard function."""

    def test_returns_plotly_figure(self, sample_profitability_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.profitability import (
            create_dupont_decomposition_dashboard,
        )

        fig = create_dupont_decomposition_dashboard(sample_profitability_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.profitability import (
            create_dupont_decomposition_dashboard,
        )

        df = pd.DataFrame({"ticker": ["A", "B"], "other_col": [1, 2]})
        fig = create_dupont_decomposition_dashboard(df)
        assert isinstance(fig, Figure)

    def test_top_n_parameter(self, sample_profitability_df):
        """Function should respect top_n parameter."""
        from finance_ml.analytics.visualizations.profitability import (
            create_dupont_decomposition_dashboard,
        )

        fig = create_dupont_decomposition_dashboard(sample_profitability_df, top_n=10)
        assert isinstance(fig, Figure)


class TestProfitabilityQuadrant:
    """Tests for create_profitability_quadrant function."""

    def test_returns_plotly_figure(self, sample_profitability_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.profitability import create_profitability_quadrant

        fig = create_profitability_quadrant(sample_profitability_df)
        assert isinstance(fig, Figure)

    def test_custom_metrics(self, sample_profitability_df):
        """Function should work with custom metrics."""
        from finance_ml.analytics.visualizations.profitability import create_profitability_quadrant

        fig = create_profitability_quadrant(
            sample_profitability_df, x_metric="roa", y_metric="roe", size_metric="gross_margin_pct"
        )
        assert isinstance(fig, Figure)


class TestMarginTrendHeatmap:
    """Tests for create_margin_trend_heatmap function."""

    def test_returns_plotly_figure(self, sample_profitability_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.profitability import create_margin_trend_heatmap

        fig = create_margin_trend_heatmap(sample_profitability_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.profitability import create_margin_trend_heatmap

        df = pd.DataFrame({"ticker": ["A", "B"], "other_col": [1, 2]})
        fig = create_margin_trend_heatmap(df)
        assert isinstance(fig, Figure)


# =============================================================================
# Technical Visualization Tests
# =============================================================================


class TestMomentumRibbonChart:
    """Tests for create_momentum_ribbon_chart function."""

    def test_returns_plotly_figure(self, sample_technical_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.technical import create_momentum_ribbon_chart

        fig = create_momentum_ribbon_chart(sample_technical_df)
        assert isinstance(fig, Figure)

    def test_top_n_parameter(self, sample_technical_df):
        """Function should respect top_n parameter."""
        from finance_ml.analytics.visualizations.technical import create_momentum_ribbon_chart

        fig = create_momentum_ribbon_chart(sample_technical_df, top_n=10)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.technical import create_momentum_ribbon_chart

        df = pd.DataFrame({"ticker": ["A", "B"], "other_col": [1, 2]})
        fig = create_momentum_ribbon_chart(df)
        assert isinstance(fig, Figure)


class Test52wRangeDistribution:
    """Tests for create_52w_range_distribution function."""

    def test_returns_plotly_figure(self, sample_technical_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.technical import create_52w_range_distribution

        fig = create_52w_range_distribution(sample_technical_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.technical import create_52w_range_distribution

        df = pd.DataFrame({"ticker": ["A", "B"], "other_col": [1, 2]})
        fig = create_52w_range_distribution(df)
        assert isinstance(fig, Figure)


class TestTrendStrengthMatrix:
    """Tests for create_trend_strength_matrix function."""

    def test_returns_plotly_figure(self, sample_technical_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.technical import create_trend_strength_matrix

        fig = create_trend_strength_matrix(sample_technical_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.technical import create_trend_strength_matrix

        df = pd.DataFrame({"ticker": ["A", "B"], "other_col": [1, 2]})
        fig = create_trend_strength_matrix(df)
        assert isinstance(fig, Figure)


class TestMomentumDivergenceScatter:
    """Tests for create_momentum_divergence_scatter function."""

    def test_returns_plotly_figure(self, sample_technical_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.technical import create_momentum_divergence_scatter

        fig = create_momentum_divergence_scatter(sample_technical_df)
        assert isinstance(fig, Figure)

    def test_custom_columns(self, sample_technical_df):
        """Function should work with custom column names."""
        from finance_ml.analytics.visualizations.technical import create_momentum_divergence_scatter

        fig = create_momentum_divergence_scatter(
            sample_technical_df,
            short_term_col="price_momentum_3m",
            long_term_col="price_momentum_1y",
        )
        assert isinstance(fig, Figure)


# =============================================================================
# Temporal Analysis Visualization Tests
# =============================================================================


class TestEarningsCalendarHeatmap:
    """Tests for create_earnings_calendar_heatmap function."""

    def test_returns_plotly_figure(self, sample_temporal_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.temporal_analysis import (
            create_earnings_calendar_heatmap,
        )

        fig = create_earnings_calendar_heatmap(sample_temporal_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.temporal_analysis import (
            create_earnings_calendar_heatmap,
        )

        df = pd.DataFrame({"ticker": ["A", "B"], "other_col": [1, 2]})
        fig = create_earnings_calendar_heatmap(df)
        assert isinstance(fig, Figure)


class TestInventoryCycleAnalysis:
    """Tests for create_inventory_cycle_analysis function."""

    def test_returns_plotly_figure(self, sample_temporal_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.temporal_analysis import (
            create_inventory_cycle_analysis,
        )

        fig = create_inventory_cycle_analysis(sample_temporal_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.temporal_analysis import (
            create_inventory_cycle_analysis,
        )

        df = pd.DataFrame({"ticker": ["A", "B"], "other_col": [1, 2]})
        fig = create_inventory_cycle_analysis(df)
        assert isinstance(fig, Figure)


class TestFcfTrajectoryChart:
    """Tests for create_fcf_trajectory_chart function."""

    def test_returns_plotly_figure(self, sample_temporal_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.temporal_analysis import (
            create_fcf_trajectory_chart,
        )

        fig = create_fcf_trajectory_chart(sample_temporal_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.temporal_analysis import (
            create_fcf_trajectory_chart,
        )

        df = pd.DataFrame({"ticker": ["A", "B"], "other_col": [1, 2]})
        fig = create_fcf_trajectory_chart(df)
        assert isinstance(fig, Figure)


class TestDividendStreakTimeline:
    """Tests for create_dividend_streak_timeline function."""

    def test_returns_plotly_figure(self, sample_temporal_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.temporal_analysis import (
            create_dividend_streak_timeline,
        )

        fig = create_dividend_streak_timeline(sample_temporal_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.temporal_analysis import (
            create_dividend_streak_timeline,
        )

        df = pd.DataFrame({"ticker": ["A", "B"], "other_col": [1, 2]})
        fig = create_dividend_streak_timeline(df)
        assert isinstance(fig, Figure)


# =============================================================================
# Module Import Tests
# =============================================================================


class TestVisualizationModuleImports:
    """Tests for visualization module imports."""

    def test_profitability_module_imports(self):
        """Profitability module should import successfully."""
        from finance_ml.analytics.visualizations import profitability

        assert hasattr(profitability, "create_margin_waterfall_chart")
        assert hasattr(profitability, "create_dupont_decomposition_dashboard")
        assert hasattr(profitability, "create_profitability_quadrant")
        assert hasattr(profitability, "create_margin_trend_heatmap")

    def test_technical_module_imports(self):
        """Technical module should import successfully."""
        from finance_ml.analytics.visualizations import technical

        assert hasattr(technical, "create_momentum_ribbon_chart")
        assert hasattr(technical, "create_52w_range_distribution")
        assert hasattr(technical, "create_trend_strength_matrix")
        assert hasattr(technical, "create_momentum_divergence_scatter")

    def test_temporal_analysis_module_imports(self):
        """Temporal analysis module should import successfully."""
        from finance_ml.analytics.visualizations import temporal_analysis

        assert hasattr(temporal_analysis, "create_earnings_calendar_heatmap")
        assert hasattr(temporal_analysis, "create_inventory_cycle_analysis")
        assert hasattr(temporal_analysis, "create_fcf_trajectory_chart")
        assert hasattr(temporal_analysis, "create_dividend_streak_timeline")

    def test_visualizations_package_exports(self):
        """Visualizations package should export all functions."""
        from finance_ml.analytics import visualizations

        # Check that __all__ contains expected functions
        assert "create_margin_waterfall_chart" in visualizations.__all__
        assert "create_momentum_ribbon_chart" in visualizations.__all__
        assert "create_earnings_calendar_heatmap" in visualizations.__all__
