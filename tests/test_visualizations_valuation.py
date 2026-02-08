"""
Tests for valuation visualization module.

Tests cover:
- visualizations/valuation.py

Following TDD approach: tests written first, then implementation.
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
def sample_valuation_df() -> pd.DataFrame:
    """Create sample DataFrame with valuation metrics."""
    np.random.seed(42)
    n = 50

    return pd.DataFrame(
        {
            "ticker": [f"TICK{i:03d}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "industry": np.random.choice(
                ["Technology", "Healthcare", "Financials", "Energy", "Consumer"], n
            ),
            "sector": np.random.choice(["Tech", "Health", "Finance", "Energy", "Consumer"], n),
            # Valuation ratios
            "p_e_ratio": np.random.uniform(5, 50, n),
            "p_b_ratio": np.random.uniform(0.5, 10, n),
            "ev_ebitda_ratio": np.random.uniform(3, 25, n),
            "ev_sales_ratio": np.random.uniform(0.5, 15, n),
            "peg_ratio": np.random.uniform(0.5, 3, n),
            "dividend_yield": np.random.uniform(0, 8, n),
            "forward_pe": np.random.uniform(5, 40, n),
            "trailing_pe": np.random.uniform(5, 50, n),
            "price_to_sales": np.random.uniform(0.5, 20, n),
            "price_to_fcf": np.random.uniform(5, 50, n),
            # Growth metrics for PEG analysis
            "eps_growth_yoy": np.random.uniform(-20, 50, n),
            "revenue_growth_yoy": np.random.uniform(-10, 40, n),
            # Market data
            "market_cap": np.random.uniform(1e9, 1e12, n),
            "last_price": np.random.uniform(10, 500, n),
        }
    )


@pytest.fixture
def sample_valuation_df_minimal() -> pd.DataFrame:
    """Create minimal DataFrame for edge case testing."""
    return pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "name": ["Company A", "Company B"],
            "industry": ["Tech", "Health"],
        }
    )


# =============================================================================
# Valuation Multiples Comparison Tests
# =============================================================================


class TestValuationMultiplesComparison:
    """Tests for create_valuation_multiples_comparison function."""

    def test_returns_plotly_figure(self, sample_valuation_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.valuation import (
            create_valuation_multiples_comparison,
        )

        fig = create_valuation_multiples_comparison(sample_valuation_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_valuation_df_minimal):
        """Function should handle missing valuation columns gracefully."""
        from finance_ml.analytics.visualizations.valuation import (
            create_valuation_multiples_comparison,
        )

        fig = create_valuation_multiples_comparison(sample_valuation_df_minimal)
        assert isinstance(fig, Figure)

    def test_specific_ticker(self, sample_valuation_df):
        """Function should work with specific ticker."""
        from finance_ml.analytics.visualizations.valuation import (
            create_valuation_multiples_comparison,
        )

        fig = create_valuation_multiples_comparison(sample_valuation_df, ticker="TICK001")
        assert isinstance(fig, Figure)

    def test_figure_has_data(self, sample_valuation_df):
        """Figure should contain data traces."""
        from finance_ml.analytics.visualizations.valuation import (
            create_valuation_multiples_comparison,
        )

        fig = create_valuation_multiples_comparison(sample_valuation_df)
        assert len(fig.data) > 0


# =============================================================================
# Valuation Distribution Dashboard Tests
# =============================================================================


class TestValuationDistributionDashboard:
    """Tests for create_valuation_distribution_dashboard function."""

    def test_returns_plotly_figure(self, sample_valuation_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.valuation import (
            create_valuation_distribution_dashboard,
        )

        fig = create_valuation_distribution_dashboard(sample_valuation_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_valuation_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.valuation import (
            create_valuation_distribution_dashboard,
        )

        fig = create_valuation_distribution_dashboard(sample_valuation_df_minimal)
        assert isinstance(fig, Figure)

    def test_custom_group_col(self, sample_valuation_df):
        """Function should work with custom group column."""
        from finance_ml.analytics.visualizations.valuation import (
            create_valuation_distribution_dashboard,
        )

        fig = create_valuation_distribution_dashboard(sample_valuation_df, group_col="sector")
        assert isinstance(fig, Figure)


# =============================================================================
# Relative Valuation Matrix Tests
# =============================================================================


class TestRelativeValuationMatrix:
    """Tests for create_relative_valuation_matrix function."""

    def test_returns_plotly_figure(self, sample_valuation_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.valuation import (
            create_relative_valuation_matrix,
        )

        fig = create_relative_valuation_matrix(sample_valuation_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_valuation_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.valuation import (
            create_relative_valuation_matrix,
        )

        fig = create_relative_valuation_matrix(sample_valuation_df_minimal)
        assert isinstance(fig, Figure)

    def test_custom_group_col(self, sample_valuation_df):
        """Function should work with custom group column."""
        from finance_ml.analytics.visualizations.valuation import (
            create_relative_valuation_matrix,
        )

        fig = create_relative_valuation_matrix(sample_valuation_df, group_col="sector")
        assert isinstance(fig, Figure)


# =============================================================================
# Valuation vs Growth Quadrant Tests
# =============================================================================


class TestValuationVsGrowthQuadrant:
    """Tests for create_valuation_vs_growth_quadrant function."""

    def test_returns_plotly_figure(self, sample_valuation_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.valuation import (
            create_valuation_vs_growth_quadrant,
        )

        fig = create_valuation_vs_growth_quadrant(sample_valuation_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_valuation_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.valuation import (
            create_valuation_vs_growth_quadrant,
        )

        fig = create_valuation_vs_growth_quadrant(sample_valuation_df_minimal)
        assert isinstance(fig, Figure)

    def test_custom_metrics(self, sample_valuation_df):
        """Function should work with custom valuation and growth metrics."""
        from finance_ml.analytics.visualizations.valuation import (
            create_valuation_vs_growth_quadrant,
        )

        fig = create_valuation_vs_growth_quadrant(
            sample_valuation_df,
            valuation_metric="ev_ebitda_ratio",
            growth_metric="revenue_growth_yoy",
        )
        assert isinstance(fig, Figure)

    def test_figure_has_quadrant_annotations(self, sample_valuation_df):
        """Figure should have quadrant annotations."""
        from finance_ml.analytics.visualizations.valuation import (
            create_valuation_vs_growth_quadrant,
        )

        fig = create_valuation_vs_growth_quadrant(sample_valuation_df)
        # Should have layout with annotations or shapes for quadrants
        assert isinstance(fig, Figure)


# =============================================================================
# Historical Valuation Percentile Tests
# =============================================================================


class TestHistoricalValuationPercentile:
    """Tests for create_historical_valuation_percentile function."""

    def test_returns_plotly_figure(self, sample_valuation_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.valuation import (
            create_historical_valuation_percentile,
        )

        fig = create_historical_valuation_percentile(sample_valuation_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_valuation_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.valuation import (
            create_historical_valuation_percentile,
        )

        fig = create_historical_valuation_percentile(sample_valuation_df_minimal)
        assert isinstance(fig, Figure)

    def test_custom_metric(self, sample_valuation_df):
        """Function should work with custom valuation metric."""
        from finance_ml.analytics.visualizations.valuation import (
            create_historical_valuation_percentile,
        )

        fig = create_historical_valuation_percentile(sample_valuation_df, metric="ev_ebitda_ratio")
        assert isinstance(fig, Figure)


# =============================================================================
# Module Import Tests
# =============================================================================


class TestValuationModuleImports:
    """Tests for valuation module imports."""

    def test_valuation_module_imports(self):
        """All valuation functions should be importable."""
        from finance_ml.analytics.visualizations.valuation import (
            create_valuation_multiples_comparison,
            create_valuation_distribution_dashboard,
            create_relative_valuation_matrix,
            create_valuation_vs_growth_quadrant,
            create_historical_valuation_percentile,
        )

        assert callable(create_valuation_multiples_comparison)
        assert callable(create_valuation_distribution_dashboard)
        assert callable(create_relative_valuation_matrix)
        assert callable(create_valuation_vs_growth_quadrant)
        assert callable(create_historical_valuation_percentile)

    def test_valuation_exports_in_package(self):
        """Valuation functions should be exported from visualizations package."""
        from finance_ml.analytics.visualizations import (
            create_valuation_multiples_comparison,
            create_valuation_distribution_dashboard,
            create_relative_valuation_matrix,
            create_valuation_vs_growth_quadrant,
            create_historical_valuation_percentile,
        )

        assert callable(create_valuation_multiples_comparison)
        assert callable(create_valuation_distribution_dashboard)
        assert callable(create_relative_valuation_matrix)
        assert callable(create_valuation_vs_growth_quadrant)
        assert callable(create_historical_valuation_percentile)
