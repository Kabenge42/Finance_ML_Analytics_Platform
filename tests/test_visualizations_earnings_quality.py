"""
Tests for earnings quality visualization module.

Tests cover:
- visualizations/earnings_quality.py

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
def sample_earnings_df() -> pd.DataFrame:
    """Create sample DataFrame with earnings quality metrics."""
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
            # Earnings metrics
            "eps_surprise_pct": np.random.uniform(-20, 30, n),
            "eps_beat_count": np.random.randint(0, 8, n),
            "eps_total_reports": np.random.randint(4, 12, n),
            "eps_trajectory_score": np.random.uniform(20, 90, n),
            "eps_positive_streak": np.random.randint(0, 10, n),
            "eps_improvement_count": np.random.randint(0, 8, n),
            "earnings_quality_composite": np.random.uniform(30, 90, n),
            # Quality metrics
            "accruals_ratio": np.random.uniform(-0.2, 0.3, n),
            "cash_earnings_ratio": np.random.uniform(0.5, 1.5, n),
            "earnings_persistence": np.random.uniform(0.3, 0.9, n),
            # Additional
            "market_cap": np.random.uniform(1e9, 1e12, n),
            "last_price": np.random.uniform(10, 500, n),
        }
    )


@pytest.fixture
def sample_earnings_df_minimal() -> pd.DataFrame:
    """Create minimal DataFrame for edge case testing."""
    return pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "name": ["Company A", "Company B"],
            "industry": ["Tech", "Health"],
        }
    )


# =============================================================================
# Earnings Surprise Dashboard Tests
# =============================================================================


class TestEarningsSurpriseDashboard:
    """Tests for create_earnings_surprise_dashboard function."""

    def test_returns_plotly_figure(self, sample_earnings_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_earnings_surprise_dashboard,
        )

        fig = create_earnings_surprise_dashboard(sample_earnings_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_earnings_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_earnings_surprise_dashboard,
        )

        fig = create_earnings_surprise_dashboard(sample_earnings_df_minimal)
        assert isinstance(fig, Figure)

    def test_figure_has_data(self, sample_earnings_df):
        """Figure should contain data traces."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_earnings_surprise_dashboard,
        )

        fig = create_earnings_surprise_dashboard(sample_earnings_df)
        assert len(fig.data) > 0


# =============================================================================
# EPS Trajectory Analysis Tests
# =============================================================================


class TestEpsTrajectoryAnalysis:
    """Tests for create_eps_trajectory_analysis function."""

    def test_returns_plotly_figure(self, sample_earnings_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_eps_trajectory_analysis,
        )

        fig = create_eps_trajectory_analysis(sample_earnings_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_earnings_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_eps_trajectory_analysis,
        )

        fig = create_eps_trajectory_analysis(sample_earnings_df_minimal)
        assert isinstance(fig, Figure)

    def test_top_n_parameter(self, sample_earnings_df):
        """Function should respect top_n parameter."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_eps_trajectory_analysis,
        )

        fig = create_eps_trajectory_analysis(sample_earnings_df, top_n=15)
        assert isinstance(fig, Figure)


# =============================================================================
# Earnings Quality Decomposition Tests
# =============================================================================


class TestEarningsQualityDecomposition:
    """Tests for create_earnings_quality_decomposition function."""

    def test_returns_plotly_figure(self, sample_earnings_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_earnings_quality_decomposition,
        )

        fig = create_earnings_quality_decomposition(sample_earnings_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_earnings_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_earnings_quality_decomposition,
        )

        fig = create_earnings_quality_decomposition(sample_earnings_df_minimal)
        assert isinstance(fig, Figure)

    def test_specific_ticker(self, sample_earnings_df):
        """Function should work with specific ticker."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_earnings_quality_decomposition,
        )

        fig = create_earnings_quality_decomposition(sample_earnings_df, ticker="TICK001")
        assert isinstance(fig, Figure)


# =============================================================================
# Beat Rate Heatmap Tests
# =============================================================================


class TestBeatRateHeatmap:
    """Tests for create_beat_rate_heatmap function."""

    def test_returns_plotly_figure(self, sample_earnings_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_beat_rate_heatmap,
        )

        fig = create_beat_rate_heatmap(sample_earnings_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_earnings_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_beat_rate_heatmap,
        )

        fig = create_beat_rate_heatmap(sample_earnings_df_minimal)
        assert isinstance(fig, Figure)

    def test_custom_group_col(self, sample_earnings_df):
        """Function should work with custom group column."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_beat_rate_heatmap,
        )

        fig = create_beat_rate_heatmap(sample_earnings_df, group_col="sector")
        assert isinstance(fig, Figure)


# =============================================================================
# Earnings Consistency Matrix Tests
# =============================================================================


class TestEarningsConsistencyMatrix:
    """Tests for create_earnings_consistency_matrix function."""

    def test_returns_plotly_figure(self, sample_earnings_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_earnings_consistency_matrix,
        )

        fig = create_earnings_consistency_matrix(sample_earnings_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_earnings_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_earnings_consistency_matrix,
        )

        fig = create_earnings_consistency_matrix(sample_earnings_df_minimal)
        assert isinstance(fig, Figure)

    def test_custom_group_col(self, sample_earnings_df):
        """Function should work with custom group column."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_earnings_consistency_matrix,
        )

        fig = create_earnings_consistency_matrix(sample_earnings_df, group_col="sector")
        assert isinstance(fig, Figure)


# =============================================================================
# Module Import Tests
# =============================================================================


class TestEarningsQualityModuleImports:
    """Tests for earnings quality module imports."""

    def test_earnings_quality_module_imports(self):
        """All earnings quality functions should be importable."""
        from finance_ml.analytics.visualizations.earnings_quality import (
            create_earnings_surprise_dashboard,
            create_eps_trajectory_analysis,
            create_earnings_quality_decomposition,
            create_beat_rate_heatmap,
            create_earnings_consistency_matrix,
        )

        assert callable(create_earnings_surprise_dashboard)
        assert callable(create_eps_trajectory_analysis)
        assert callable(create_earnings_quality_decomposition)
        assert callable(create_beat_rate_heatmap)
        assert callable(create_earnings_consistency_matrix)

    def test_earnings_quality_exports_in_package(self):
        """Earnings quality functions should be exported from visualizations package."""
        from finance_ml.analytics.visualizations import (
            create_earnings_surprise_dashboard,
            create_eps_trajectory_analysis,
            create_earnings_quality_decomposition,
            create_beat_rate_heatmap,
            create_earnings_consistency_matrix,
        )

        assert callable(create_earnings_surprise_dashboard)
        assert callable(create_eps_trajectory_analysis)
        assert callable(create_earnings_quality_decomposition)
        assert callable(create_beat_rate_heatmap)
        assert callable(create_earnings_consistency_matrix)
