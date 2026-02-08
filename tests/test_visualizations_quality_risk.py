"""
Tests for quality and risk visualization module.

Tests cover:
- visualizations/quality_risk.py

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
def sample_quality_risk_df() -> pd.DataFrame:
    """Create sample DataFrame with quality and risk metrics."""
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
            # Quality scores
            "piotroski_f_score": np.random.randint(0, 10, n),
            "altman_z_score": np.random.uniform(0.5, 5, n),
            "beneish_m_score": np.random.uniform(-3, 0, n),
            # Risk metrics
            "distress_risk_score": np.random.uniform(10, 90, n),
            "quality_composite_score": np.random.uniform(30, 90, n),
            "accounting_quality_flag": np.random.choice([0, 1], n),
            "manipulation_probability": np.random.uniform(0, 0.5, n),
            # Additional
            "market_cap": np.random.uniform(1e9, 1e12, n),
            "last_price": np.random.uniform(10, 500, n),
            "price_momentum_1y": np.random.uniform(-30, 50, n),
        }
    )


@pytest.fixture
def sample_quality_risk_df_minimal() -> pd.DataFrame:
    """Create minimal DataFrame for edge case testing."""
    return pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "name": ["Company A", "Company B"],
            "industry": ["Tech", "Health"],
        }
    )


# =============================================================================
# Piotroski F-Score Breakdown Tests
# =============================================================================


class TestPiotrostkiFscoreBreakdown:
    """Tests for create_piotroski_fscore_breakdown function."""

    def test_returns_plotly_figure(self, sample_quality_risk_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_piotroski_fscore_breakdown,
        )

        fig = create_piotroski_fscore_breakdown(sample_quality_risk_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_quality_risk_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_piotroski_fscore_breakdown,
        )

        fig = create_piotroski_fscore_breakdown(sample_quality_risk_df_minimal)
        assert isinstance(fig, Figure)

    def test_specific_ticker(self, sample_quality_risk_df):
        """Function should work with specific ticker."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_piotroski_fscore_breakdown,
        )

        fig = create_piotroski_fscore_breakdown(sample_quality_risk_df, ticker="TICK001")
        assert isinstance(fig, Figure)


# =============================================================================
# Altman Z-Score Distribution Tests
# =============================================================================


class TestAltmanZscoreDistribution:
    """Tests for create_altman_zscore_distribution function."""

    def test_returns_plotly_figure(self, sample_quality_risk_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_altman_zscore_distribution,
        )

        fig = create_altman_zscore_distribution(sample_quality_risk_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_quality_risk_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_altman_zscore_distribution,
        )

        fig = create_altman_zscore_distribution(sample_quality_risk_df_minimal)
        assert isinstance(fig, Figure)

    def test_custom_group_col(self, sample_quality_risk_df):
        """Function should work with custom group column."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_altman_zscore_distribution,
        )

        fig = create_altman_zscore_distribution(sample_quality_risk_df, group_col="sector")
        assert isinstance(fig, Figure)


# =============================================================================
# Quality Risk Quadrant Tests
# =============================================================================


class TestQualityRiskQuadrant:
    """Tests for create_quality_risk_quadrant function."""

    def test_returns_plotly_figure(self, sample_quality_risk_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_quality_risk_quadrant,
        )

        fig = create_quality_risk_quadrant(sample_quality_risk_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_quality_risk_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_quality_risk_quadrant,
        )

        fig = create_quality_risk_quadrant(sample_quality_risk_df_minimal)
        assert isinstance(fig, Figure)

    def test_figure_has_data(self, sample_quality_risk_df):
        """Figure should contain data traces."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_quality_risk_quadrant,
        )

        fig = create_quality_risk_quadrant(sample_quality_risk_df)
        assert len(fig.data) > 0


# =============================================================================
# Beneish M-Score Analysis Tests
# =============================================================================


class TestBeneishMscoreAnalysis:
    """Tests for create_beneish_mscore_analysis function."""

    def test_returns_plotly_figure(self, sample_quality_risk_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_beneish_mscore_analysis,
        )

        fig = create_beneish_mscore_analysis(sample_quality_risk_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_quality_risk_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_beneish_mscore_analysis,
        )

        fig = create_beneish_mscore_analysis(sample_quality_risk_df_minimal)
        assert isinstance(fig, Figure)


# =============================================================================
# Risk Tier Sunburst Tests
# =============================================================================


class TestRiskTierSunburst:
    """Tests for create_risk_tier_sunburst function."""

    def test_returns_plotly_figure(self, sample_quality_risk_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_risk_tier_sunburst,
        )

        fig = create_risk_tier_sunburst(sample_quality_risk_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_quality_risk_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_risk_tier_sunburst,
        )

        fig = create_risk_tier_sunburst(sample_quality_risk_df_minimal)
        assert isinstance(fig, Figure)


# =============================================================================
# Distress Early Warning Dashboard Tests
# =============================================================================


class TestDistressEarlyWarningDashboard:
    """Tests for create_distress_early_warning_dashboard function."""

    def test_returns_plotly_figure(self, sample_quality_risk_df):
        """Function should return a Plotly Figure."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_distress_early_warning_dashboard,
        )

        fig = create_distress_early_warning_dashboard(sample_quality_risk_df)
        assert isinstance(fig, Figure)

    def test_handles_missing_columns(self, sample_quality_risk_df_minimal):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_distress_early_warning_dashboard,
        )

        fig = create_distress_early_warning_dashboard(sample_quality_risk_df_minimal)
        assert isinstance(fig, Figure)


# =============================================================================
# Module Import Tests
# =============================================================================


class TestQualityRiskModuleImports:
    """Tests for quality risk module imports."""

    def test_quality_risk_module_imports(self):
        """All quality risk functions should be importable."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_piotroski_fscore_breakdown,
            create_altman_zscore_distribution,
            create_quality_risk_quadrant,
            create_beneish_mscore_analysis,
            create_risk_tier_sunburst,
            create_distress_early_warning_dashboard,
        )

        assert callable(create_piotroski_fscore_breakdown)
        assert callable(create_altman_zscore_distribution)
        assert callable(create_quality_risk_quadrant)
        assert callable(create_beneish_mscore_analysis)
        assert callable(create_risk_tier_sunburst)
        assert callable(create_distress_early_warning_dashboard)

    def test_quality_risk_exports_in_package(self):
        """Quality risk functions should be exported from visualizations package."""
        from finance_ml.analytics.visualizations import (
            create_piotroski_fscore_breakdown,
            create_altman_zscore_distribution,
            create_quality_risk_quadrant,
            create_beneish_mscore_analysis,
            create_risk_tier_sunburst,
            create_distress_early_warning_dashboard,
        )

        assert callable(create_piotroski_fscore_breakdown)
        assert callable(create_altman_zscore_distribution)
        assert callable(create_quality_risk_quadrant)
        assert callable(create_beneish_mscore_analysis)
        assert callable(create_risk_tier_sunburst)
        assert callable(create_distress_early_warning_dashboard)
