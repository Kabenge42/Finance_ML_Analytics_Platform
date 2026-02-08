"""
Tests for growth analysis visualization module.

Tests cover:
- visualizations/growth_analysis.py

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
def sample_growth_df() -> pd.DataFrame:
    """Create sample DataFrame with growth metrics."""
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
            # Growth metrics
            "revenue_growth_yoy": np.random.uniform(-20, 50, n),
            "revenue_growth_3y_cagr": np.random.uniform(-10, 30, n),
            "revenue_growth_5y_cagr": np.random.uniform(-5, 25, n),
            "ebitda_growth_yoy": np.random.uniform(-30, 60, n),
            "eps_growth_yoy": np.random.uniform(-40, 80, n),
            "eps_growth_3y_cagr": np.random.uniform(-20, 40, n),
            "operating_income_growth": np.random.uniform(-30, 50, n),
            "net_income_growth": np.random.uniform(-40, 60, n),
            # Profitability for quadrant
            "roe": np.random.uniform(-10, 40, n),
            "net_margin_pct": np.random.uniform(-5, 25, n),
            "roic": np.random.uniform(-5, 30, n),
            # Additional
            "market_cap": np.random.uniform(1e9, 1e12, n),
        }
    )


@pytest.fixture
def sample_growth_df_minimal() -> pd.DataFrame:
    """Create minimal DataFrame for edge case testing."""
    return pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "name": ["Company A", "Company B"],
            "industry": ["Tech", "Health"],
        }
    )


# =============================================================================
# Growth Waterfall Chart Tests
# =============================================================================


class TestGrowthWaterfallChart:
    """Tests for create_growth_waterfall_chart function."""

    def test_returns_plotly_figure(self, sample_growth_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_growth_waterfall_chart,
        )

        fig = create_growth_waterfall_chart(sample_growth_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_growth_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_growth_waterfall_chart,
        )

        fig = create_growth_waterfall_chart(sample_growth_df_minimal)
        assert isinstance(fig, Figure)

    def test_specific_ticker(self, sample_growth_df):
        """Function should work with specific ticker."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_growth_waterfall_chart,
        )

        fig = create_growth_waterfall_chart(sample_growth_df, ticker="TICK001")
        assert isinstance(fig, Figure)


# =============================================================================
# Growth Consistency Matrix Tests
# =============================================================================


class TestGrowthConsistencyMatrix:
    """Tests for create_growth_consistency_matrix function."""

    def test_returns_plotly_figure(self, sample_growth_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_growth_consistency_matrix,
        )

        fig = create_growth_consistency_matrix(sample_growth_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_growth_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_growth_consistency_matrix,
        )

        fig = create_growth_consistency_matrix(sample_growth_df_minimal)
        assert isinstance(fig, Figure)

    def test_custom_group_col(self, sample_growth_df):
        """Function should work with custom group column."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_growth_consistency_matrix,
        )

        fig = create_growth_consistency_matrix(sample_growth_df, group_col="sector")
        assert isinstance(fig, Figure)


# =============================================================================
# Growth vs Profitability Quadrant Tests
# =============================================================================


class TestGrowthVsProfitabilityQuadrant:
    """Tests for create_growth_vs_profitability_quadrant function."""

    def test_returns_plotly_figure(self, sample_growth_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_growth_vs_profitability_quadrant,
        )

        fig = create_growth_vs_profitability_quadrant(sample_growth_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_growth_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_growth_vs_profitability_quadrant,
        )

        fig = create_growth_vs_profitability_quadrant(sample_growth_df_minimal)
        assert isinstance(fig, Figure)


# =============================================================================
# Growth Acceleration Chart Tests
# =============================================================================


class TestGrowthAccelerationChart:
    """Tests for create_growth_acceleration_chart function."""

    def test_returns_plotly_figure(self, sample_growth_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_growth_acceleration_chart,
        )

        fig = create_growth_acceleration_chart(sample_growth_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_growth_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_growth_acceleration_chart,
        )

        fig = create_growth_acceleration_chart(sample_growth_df_minimal)
        assert isinstance(fig, Figure)

    def test_top_n_parameter(self, sample_growth_df):
        """Function should respect top_n parameter."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_growth_acceleration_chart,
        )

        fig = create_growth_acceleration_chart(sample_growth_df, top_n=15)
        assert isinstance(fig, Figure)


# =============================================================================
# Sustainable Growth Analysis Tests
# =============================================================================


class TestSustainableGrowthAnalysis:
    """Tests for create_sustainable_growth_analysis function."""

    def test_returns_plotly_figure(self, sample_growth_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_sustainable_growth_analysis,
        )

        fig = create_sustainable_growth_analysis(sample_growth_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_growth_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_sustainable_growth_analysis,
        )

        fig = create_sustainable_growth_analysis(sample_growth_df_minimal)
        assert isinstance(fig, Figure)


# =============================================================================
# Module Import Tests
# =============================================================================


class TestGrowthAnalysisModuleImports:
    """Tests for growth analysis module imports."""

    def test_growth_analysis_module_imports(self):
        """All growth analysis functions should be importable."""
        from finance_ml.analytics.visualizations.growth_analysis import (
            create_growth_waterfall_chart,
            create_growth_consistency_matrix,
            create_growth_vs_profitability_quadrant,
            create_growth_acceleration_chart,
            create_sustainable_growth_analysis,
        )

        assert callable(create_growth_waterfall_chart)
        assert callable(create_growth_consistency_matrix)
        assert callable(create_growth_vs_profitability_quadrant)
        assert callable(create_growth_acceleration_chart)
        assert callable(create_sustainable_growth_analysis)

    def test_growth_analysis_exports_in_package(self):
        """Growth analysis functions should be exported from visualizations package."""
        from finance_ml.analytics.visualizations import (
            create_growth_waterfall_chart,
            create_growth_consistency_matrix,
            create_growth_vs_profitability_quadrant,
            create_growth_acceleration_chart,
            create_sustainable_growth_analysis,
        )

        assert callable(create_growth_waterfall_chart)
        assert callable(create_growth_consistency_matrix)
        assert callable(create_growth_vs_profitability_quadrant)
        assert callable(create_growth_acceleration_chart)
        assert callable(create_sustainable_growth_analysis)
