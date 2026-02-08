"""
Tests for earnings quality visualization functions, including the new
enhanced earnings beat probability visualizations.

Covers:
1. Existing functions (create_earnings_surprise_dashboard, etc.) — smoke tests
2. New: create_revision_momentum_chart
3. New: create_gaap_divergence_plot
4. New: create_enhanced_beat_probability_dashboard
5. Updated: create_earnings_probability_dashboard (enhanced columns support)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from finance_ml.analytics.visualizations.earnings_quality import (
    create_beat_rate_heatmap,
    create_earnings_consistency_matrix,
    create_earnings_quality_decomposition,
    create_earnings_surprise_dashboard,
    create_enhanced_beat_probability_dashboard,
    create_eps_trajectory_analysis,
    create_gaap_divergence_plot,
    create_revision_momentum_chart,
)
from finance_ml.analytics.probability_analytics import (
    create_earnings_probability_dashboard,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def enhanced_df() -> pd.DataFrame:
    """DataFrame simulating output from analyze_dataframe_enhanced."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame(
        {
            "ticker": [f"T{i:03d}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "sector": np.random.choice(["Technology", "Energy", "Financials", "Healthcare"], n),
            "posterior_beat_prob": np.random.uniform(0.2, 0.9, n),
            "posterior_std": np.random.uniform(0.05, 0.15, n),
            "historical_beat_rate": np.random.uniform(0.3, 1.0, n),
            "historical_beats": np.random.randint(1, 6, n),
            "total_reports": np.random.randint(3, 6, n),
            "confidence_score": np.random.uniform(0.3, 0.9, n),
            "prior_influence_pct": np.random.uniform(10, 50, n),
            "effective_sample_size": np.random.uniform(3, 10, n),
            "ci_90_lower": np.random.uniform(0.1, 0.4, n),
            "ci_90_upper": np.random.uniform(0.6, 0.95, n),
            "ci_95_lower": np.random.uniform(0.05, 0.35, n),
            "ci_95_upper": np.random.uniform(0.65, 0.98, n),
            "classification_confidence": np.random.choice(["High", "Medium", "Low"], n),
            "beat_classification": np.random.choice(["likely_beat", "uncertain"], n),
            "revision_momentum_score": np.random.uniform(10, 95, n),
            "gaap_norm_spread": np.random.uniform(-40, 5, n),
            "revision_trend_short": np.random.uniform(-3, 3, n),
            "revision_trend_medium": np.random.uniform(-2, 2, n),
            "eps_norm_est_fy1e": np.random.uniform(1, 10, n),
            "quarterly_beat_streak": np.random.randint(0, 6, n),
            "data_source": ["forward_enhanced"] * n,
        }
    )


@pytest.fixture
def legacy_probability_df() -> pd.DataFrame:
    """DataFrame simulating output from analyze_dataframe (legacy, no enhanced cols)."""
    np.random.seed(99)
    n = 10
    return pd.DataFrame(
        {
            "ticker": [f"L{i:02d}" for i in range(n)],
            "name": [f"Legacy Co {i}" for i in range(n)],
            "sector": np.random.choice(["Tech", "Energy"], n),
            "posterior_beat_prob": np.random.uniform(0.3, 0.8, n),
            "historical_beat_rate": np.random.uniform(0.4, 0.9, n),
            "confidence_score": np.random.uniform(0.2, 0.8, n),
            "beat_classification": np.random.choice(["likely_beat", "uncertain"], n),
        }
    )


@pytest.fixture
def earnings_df() -> pd.DataFrame:
    """DataFrame with traditional earnings quality columns."""
    np.random.seed(7)
    n = 15
    return pd.DataFrame(
        {
            "ticker": [f"E{i:02d}" for i in range(n)],
            "sector": np.random.choice(["Tech", "Energy", "Health"], n),
            "industry": np.random.choice(["Software", "Oil", "Pharma"], n),
            "eps_surprise_pct": np.random.uniform(-5, 10, n),
            "eps_beat_count": np.random.randint(0, 8, n),
            "eps_total_reports": np.random.randint(4, 8, n),
            "eps_trajectory_score": np.random.uniform(20, 90, n),
            "eps_positive_streak": np.random.randint(0, 6, n),
            "eps_improvement_count": np.random.randint(0, 5, n),
            "earnings_quality_composite": np.random.uniform(0, 1, n),
            "accruals_ratio": np.random.uniform(0, 0.3, n),
            "cash_earnings_ratio": np.random.uniform(0.3, 1.2, n),
            "earnings_persistence": np.random.uniform(0.2, 0.9, n),
        }
    )


# =============================================================================
# EXISTING FUNCTIONS — SMOKE TESTS
# =============================================================================


class TestExistingEarningsQualityViz:
    """Smoke tests for pre-existing visualization functions."""

    def test_earnings_surprise_dashboard_returns_figure(self, earnings_df):
        fig = create_earnings_surprise_dashboard(earnings_df)
        assert isinstance(fig, go.Figure)

    def test_eps_trajectory_analysis_returns_figure(self, earnings_df):
        fig = create_eps_trajectory_analysis(earnings_df, top_n=10)
        assert isinstance(fig, go.Figure)

    def test_earnings_quality_decomposition_returns_figure(self, earnings_df):
        fig = create_earnings_quality_decomposition(earnings_df)
        assert isinstance(fig, go.Figure)

    def test_beat_rate_heatmap_returns_figure(self, earnings_df):
        fig = create_beat_rate_heatmap(earnings_df, group_col="sector")
        assert isinstance(fig, go.Figure)

    def test_earnings_consistency_matrix_returns_figure(self, earnings_df):
        fig = create_earnings_consistency_matrix(earnings_df, group_col="sector")
        assert isinstance(fig, go.Figure)


# =============================================================================
# NEW: create_revision_momentum_chart
# =============================================================================


class TestRevisionMomentumChart:
    """Tests for the revision momentum visualization."""

    def test_returns_figure(self, enhanced_df):
        fig = create_revision_momentum_chart(enhanced_df)
        assert isinstance(fig, go.Figure)

    def test_no_data_returns_placeholder(self):
        empty = pd.DataFrame({"ticker": ["A"], "sector": ["Tech"]})
        fig = create_revision_momentum_chart(empty)
        assert isinstance(fig, go.Figure)
        # Should contain "No Data" annotation
        assert any(
            "no data" in str(a.text).lower()
            for a in fig.layout.annotations
            if hasattr(a, "text") and a.text
        )

    def test_has_multiple_traces(self, enhanced_df):
        fig = create_revision_momentum_chart(enhanced_df)
        assert len(fig.data) > 0

    def test_with_trends_adds_extra_panel(self, enhanced_df):
        fig = create_revision_momentum_chart(enhanced_df)
        # Should have 3 rows when trends are present
        # Check subplot layout has 3 rows
        assert fig.layout.yaxis3 is not None

    def test_without_trends_two_panels(self, enhanced_df):
        df = enhanced_df.drop(columns=["revision_trend_short", "revision_trend_medium"])
        fig = create_revision_momentum_chart(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_without_sector_still_works(self, enhanced_df):
        df = enhanced_df.drop(columns=["sector"])
        fig = create_revision_momentum_chart(df)
        assert isinstance(fig, go.Figure)

    def test_custom_top_n(self, enhanced_df):
        fig = create_revision_momentum_chart(enhanced_df, top_n=5)
        assert isinstance(fig, go.Figure)


# =============================================================================
# NEW: create_gaap_divergence_plot
# =============================================================================


class TestGaapDivergencePlot:
    """Tests for the GAAP divergence visualization."""

    def test_returns_figure(self, enhanced_df):
        fig = create_gaap_divergence_plot(enhanced_df)
        assert isinstance(fig, go.Figure)

    def test_no_data_returns_placeholder(self):
        empty = pd.DataFrame({"ticker": ["A"]})
        fig = create_gaap_divergence_plot(empty)
        assert isinstance(fig, go.Figure)
        assert any(
            "no data" in str(a.text).lower()
            for a in fig.layout.annotations
            if hasattr(a, "text") and a.text
        )

    def test_has_traces(self, enhanced_df):
        fig = create_gaap_divergence_plot(enhanced_df)
        assert len(fig.data) > 0

    def test_with_sector_shows_box_plots(self, enhanced_df):
        fig = create_gaap_divergence_plot(enhanced_df, group_col="sector")
        assert isinstance(fig, go.Figure)
        # Should have Box traces for sector panel
        box_traces = [t for t in fig.data if isinstance(t, go.Box)]
        assert len(box_traces) > 0

    def test_without_posterior_uses_fallback(self):
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "gaap_norm_spread": [-30, -10, -5],
            }
        )
        fig = create_gaap_divergence_plot(df)
        assert isinstance(fig, go.Figure)

    def test_without_momentum_uses_fallback(self):
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "gaap_norm_spread": [-30, -10, -5],
                "posterior_beat_prob": [0.4, 0.6, 0.7],
            }
        )
        fig = create_gaap_divergence_plot(df)
        assert isinstance(fig, go.Figure)


# =============================================================================
# NEW: create_enhanced_beat_probability_dashboard
# =============================================================================


class TestEnhancedBeatProbabilityDashboard:
    """Tests for the comprehensive enhanced beat probability dashboard."""

    def test_returns_figure(self, enhanced_df):
        fig = create_enhanced_beat_probability_dashboard(enhanced_df)
        assert isinstance(fig, go.Figure)

    def test_no_data_returns_placeholder(self):
        empty = pd.DataFrame({"ticker": ["A"]})
        fig = create_enhanced_beat_probability_dashboard(empty)
        assert isinstance(fig, go.Figure)
        assert any(
            "no data" in str(a.text).lower()
            for a in fig.layout.annotations
            if hasattr(a, "text") and a.text
        )

    def test_has_six_subplot_titles(self, enhanced_df):
        fig = create_enhanced_beat_probability_dashboard(enhanced_df)
        # 6 subplot titles + possible annotation extras
        subplot_titles = [
            a.text
            for a in fig.layout.annotations
            if hasattr(a, "text") and a.text and "No Data" not in str(a.text)
        ]
        assert len(subplot_titles) >= 6

    def test_height_is_tall(self, enhanced_df):
        fig = create_enhanced_beat_probability_dashboard(enhanced_df)
        assert fig.layout.height >= 1000

    def test_custom_title(self, enhanced_df):
        fig = create_enhanced_beat_probability_dashboard(enhanced_df, title="Custom Title")
        assert fig.layout.title.text == "Custom Title"

    def test_minimal_data_still_works(self):
        df = pd.DataFrame(
            {
                "posterior_beat_prob": [0.6, 0.4, 0.7],
            }
        )
        fig = create_enhanced_beat_probability_dashboard(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_has_pie_trace_for_confidence(self, enhanced_df):
        fig = create_enhanced_beat_probability_dashboard(enhanced_df)
        pie_traces = [t for t in fig.data if isinstance(t, go.Pie)]
        assert len(pie_traces) >= 1


# =============================================================================
# UPDATED: create_earnings_probability_dashboard (enhanced support)
# =============================================================================


class TestEarningsProbabilityDashboardEnhanced:
    """Tests for the updated probability dashboard with enhanced column support."""

    def test_legacy_df_returns_2x2_figure(self, legacy_probability_df):
        fig = create_earnings_probability_dashboard(legacy_probability_df)
        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 700

    def test_enhanced_df_returns_3x2_figure(self, enhanced_df):
        fig = create_earnings_probability_dashboard(enhanced_df)
        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 1000

    def test_enhanced_has_more_traces_than_legacy(self, enhanced_df, legacy_probability_df):
        fig_enhanced = create_earnings_probability_dashboard(enhanced_df)
        fig_legacy = create_earnings_probability_dashboard(legacy_probability_df)
        assert len(fig_enhanced.data) > len(fig_legacy.data)

    def test_enhanced_has_momentum_panel(self, enhanced_df):
        fig = create_earnings_probability_dashboard(enhanced_df)
        # Check subplot annotations include momentum title
        titles = [a.text for a in fig.layout.annotations if hasattr(a, "text") and a.text]
        assert any("Momentum" in t for t in titles)

    def test_enhanced_has_gaap_panel(self, enhanced_df):
        fig = create_earnings_probability_dashboard(enhanced_df)
        titles = [a.text for a in fig.layout.annotations if hasattr(a, "text") and a.text]
        assert any("GAAP" in t for t in titles)

    def test_legacy_no_enhanced_panels(self, legacy_probability_df):
        fig = create_earnings_probability_dashboard(legacy_probability_df)
        titles = [a.text for a in fig.layout.annotations if hasattr(a, "text") and a.text]
        assert not any("Momentum" in t for t in titles)
        assert not any("GAAP" in t for t in titles)


# =============================================================================
# EXPORTS: verify __init__.py exports
# =============================================================================


class TestVisualizationExports:
    """Verify the new functions are exported from the visualizations package."""

    def test_revision_momentum_chart_importable(self):
        from finance_ml.analytics.visualizations import create_revision_momentum_chart

        assert callable(create_revision_momentum_chart)

    def test_gaap_divergence_plot_importable(self):
        from finance_ml.analytics.visualizations import create_gaap_divergence_plot

        assert callable(create_gaap_divergence_plot)

    def test_enhanced_beat_dashboard_importable(self):
        from finance_ml.analytics.visualizations import (
            create_enhanced_beat_probability_dashboard,
        )

        assert callable(create_enhanced_beat_probability_dashboard)

    def test_all_exports_include_new_functions(self):
        from finance_ml.analytics.visualizations import __all__

        assert "create_revision_momentum_chart" in __all__
        assert "create_gaap_divergence_plot" in __all__
        assert "create_enhanced_beat_probability_dashboard" in __all__
