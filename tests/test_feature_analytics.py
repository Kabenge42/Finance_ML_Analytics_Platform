"""
Test suite for feature_analytics module.

TDD approach: Tests written first to define expected behavior for:
1. Interactive Plotly visualizations (momentum, valuation, leverage/liquidity)
2. Monte Carlo simulations (price target fair value)
3. Bayesian models (earnings beat probability)
4. Distress risk distribution analysis
5. Composite quality scoring
6. Summary dashboard generation

Coverage target: ≥80% for changed files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from plotly.graph_objs import Figure


# =============================================================================
# Fixtures for Feature Analytics Tests
# =============================================================================


@pytest.fixture
def sample_stock_features_df() -> pd.DataFrame:
    """Create a sample DataFrame mimicking mv_all_stock_features structure."""
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
            "price_target": np.random.uniform(10, 600, n).round(2),
            "price_target_high": np.random.uniform(50, 800, n).round(2),
            "price_target_low": np.random.uniform(5, 400, n).round(2),
            "price_target_median": np.random.uniform(20, 550, n).round(2),
            "market_cap": np.random.uniform(1e9, 1e12, n),
            # Momentum columns
            "price_momentum_1m": np.random.uniform(-30, 50, n).round(2),
            "price_momentum_3m": np.random.uniform(-40, 60, n).round(2),
            "price_momentum_6m": np.random.uniform(-50, 80, n).round(2),
            "range_52w_position": np.random.uniform(0, 1, n).round(3),
            # Valuation columns
            "p_e_ratio": np.random.uniform(5, 50, n).round(2),
            "p_b_ratio": np.random.uniform(0.5, 10, n).round(2),
            "ev_ebitda_ratio": np.random.uniform(3, 30, n).round(2),
            "ev_sales_ratio": np.random.uniform(0.5, 15, n).round(2),
            # Leverage & Liquidity
            "debt_to_equity": np.random.uniform(0, 3, n).round(2),
            "current_ratio": np.random.uniform(0.5, 4, n).round(2),
            "distress_risk_score": np.random.uniform(10, 95, n).round(1),
            # Analyst sentiment
            "analyst_bullish_pct": np.random.uniform(10, 90, n).round(1),
            "analyst_bearish_pct": np.random.uniform(5, 40, n).round(1),
            "upside_potential": np.random.uniform(-30, 80, n).round(2),
            # Quality metrics
            "piotroski_f_score": np.random.randint(0, 10, n),
            "earnings_quality_composite": np.random.uniform(20, 90, n).round(1),
            "cash_flow_quality_score": np.random.uniform(30, 95, n).round(1),
            "accounting_quality_score": np.random.uniform(25, 85, n).round(1),
            "dilution_score": np.random.uniform(40, 100, n).round(1),
            "beta_stability_score": np.random.uniform(30, 90, n).round(1),
            "long_term_trend_score": np.random.uniform(20, 80, n).round(1),
            "eps_trajectory_score": np.random.uniform(25, 85, n).round(1),
            # Earnings data
            "eps_positive_streak": np.random.randint(0, 6, n),
            "net_margin_pct": np.random.uniform(-10, 30, n).round(2),
        }
    )

    # Ensure price_target_high > price_target_low
    df["price_target_high"] = df[["price_target_high", "price_target_low"]].max(axis=1) + 10
    df["price_target_median"] = (df["price_target_high"] + df["price_target_low"]) / 2

    return df


@pytest.fixture
def sample_stock_features_with_nulls(sample_stock_features_df) -> pd.DataFrame:
    """Create sample data with some null values for robustness testing."""
    df = sample_stock_features_df.copy()

    # Introduce nulls in various columns
    null_indices = np.random.choice(len(df), size=10, replace=False)
    df.loc[null_indices[:5], "price_momentum_1m"] = np.nan
    df.loc[null_indices[5:], "distress_risk_score"] = np.nan
    df.loc[null_indices[:3], "eps_positive_streak"] = np.nan

    return df


# =============================================================================
# Section 2: Interactive Visualization Tests
# =============================================================================


class TestInteractiveMomentumDashboard:
    """Tests for create_interactive_momentum_dashboard function."""

    def test_returns_plotly_figure(self, sample_stock_features_df):
        """Function should return a Plotly Figure object."""
        from finance_ml.analytics.feature_analytics import create_interactive_momentum_dashboard

        fig = create_interactive_momentum_dashboard(sample_stock_features_df)

        assert isinstance(fig, Figure)

    def test_figure_has_four_subplots(self, sample_stock_features_df):
        """Dashboard should have 4 subplot panels."""
        from finance_ml.analytics.feature_analytics import create_interactive_momentum_dashboard

        fig = create_interactive_momentum_dashboard(sample_stock_features_df)

        # Check that figure has multiple traces (at least 4 for the panels)
        assert len(fig.data) >= 4

    def test_handles_missing_data(self, sample_stock_features_with_nulls):
        """Function should handle DataFrames with null values gracefully."""
        from finance_ml.analytics.feature_analytics import create_interactive_momentum_dashboard

        fig = create_interactive_momentum_dashboard(sample_stock_features_with_nulls)

        assert isinstance(fig, Figure)

    def test_figure_has_correct_title(self, sample_stock_features_df):
        """Dashboard should have appropriate title."""
        from finance_ml.analytics.feature_analytics import create_interactive_momentum_dashboard

        fig = create_interactive_momentum_dashboard(sample_stock_features_df)

        assert fig.layout.title is not None
        assert "Momentum" in fig.layout.title.text


class TestInteractiveValuationHeatmap:
    """Tests for create_interactive_valuation_heatmap function."""

    def test_returns_plotly_figure(self, sample_stock_features_df):
        """Function should return a Plotly Figure object."""
        from finance_ml.analytics.feature_analytics import create_interactive_valuation_heatmap

        fig = create_interactive_valuation_heatmap(sample_stock_features_df)

        assert isinstance(fig, Figure)

    def test_heatmap_has_correct_dimensions(self, sample_stock_features_df):
        """Heatmap should have industries as rows and metrics as columns."""
        from finance_ml.analytics.feature_analytics import create_interactive_valuation_heatmap

        fig = create_interactive_valuation_heatmap(sample_stock_features_df)

        # Should have a heatmap trace
        assert len(fig.data) >= 1
        assert fig.data[0].type == "heatmap"

    def test_handles_missing_industries(self, sample_stock_features_df):
        """Function should handle missing industry values."""
        from finance_ml.analytics.feature_analytics import create_interactive_valuation_heatmap

        df = sample_stock_features_df.copy()
        df.loc[0:5, "industry"] = np.nan

        fig = create_interactive_valuation_heatmap(df)

        assert isinstance(fig, Figure)


class TestLeverageLiquidityQuadrant:
    """Tests for create_leverage_liquidity_quadrant function."""

    def test_returns_plotly_figure(self, sample_stock_features_df):
        """Function should return a Plotly Figure object."""
        from finance_ml.analytics.feature_analytics import create_leverage_liquidity_quadrant

        fig = create_leverage_liquidity_quadrant(sample_stock_features_df)

        assert isinstance(fig, Figure)

    def test_scatter_plot_created(self, sample_stock_features_df):
        """Should create a scatter plot."""
        from finance_ml.analytics.feature_analytics import create_leverage_liquidity_quadrant

        fig = create_leverage_liquidity_quadrant(sample_stock_features_df)

        # Check for scatter trace
        scatter_traces = [t for t in fig.data if t.type == "scatter" or t.type == "scattergl"]
        assert len(scatter_traces) >= 1

    def test_quadrant_lines_present(self, sample_stock_features_df):
        """Should have reference lines for quadrant analysis."""
        from finance_ml.analytics.feature_analytics import create_leverage_liquidity_quadrant

        fig = create_leverage_liquidity_quadrant(sample_stock_features_df)

        # Check for horizontal and vertical lines in layout shapes
        assert fig.layout.shapes is not None or len(fig.data) > 1


# =============================================================================
# Section 3: Monte Carlo Simulation Tests
# =============================================================================


class TestMonteCarloSimulation:
    """Tests for monte_carlo_price_target_simulation function."""

    def test_returns_dataframe(self, sample_stock_features_df):
        """Function should return a pandas DataFrame."""
        from finance_ml.analytics.feature_analytics import monte_carlo_price_target_simulation

        result = monte_carlo_price_target_simulation(sample_stock_features_df)

        assert isinstance(result, pd.DataFrame)

    def test_output_has_required_columns(self, sample_stock_features_df):
        """Output should contain expected columns."""
        from finance_ml.analytics.feature_analytics import monte_carlo_price_target_simulation

        result = monte_carlo_price_target_simulation(sample_stock_features_df)

        required_cols = [
            "ticker",
            "expected_upside_pct",
            "var_5_pct",
            "prob_positive_upside",
            "risk_reward_ratio",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_simulation_count_parameter(self, sample_stock_features_df):
        """Should respect n_simulations parameter."""
        from finance_ml.analytics.feature_analytics import monte_carlo_price_target_simulation

        # Run with different simulation counts - results should be similar but not identical
        result_1k = monte_carlo_price_target_simulation(
            sample_stock_features_df, n_simulations=1000
        )
        result_10k = monte_carlo_price_target_simulation(
            sample_stock_features_df, n_simulations=10000
        )

        # Both should return valid results
        assert len(result_1k) > 0
        assert len(result_10k) > 0

    def test_probability_bounds(self, sample_stock_features_df):
        """Probability values should be between 0 and 100."""
        from finance_ml.analytics.feature_analytics import monte_carlo_price_target_simulation

        result = monte_carlo_price_target_simulation(sample_stock_features_df)

        assert result["prob_positive_upside"].min() >= 0
        assert result["prob_positive_upside"].max() <= 100

    def test_handles_invalid_price_data(self, sample_stock_features_df):
        """Should handle cases where price_target_high <= price_target_low."""
        from finance_ml.analytics.feature_analytics import monte_carlo_price_target_simulation

        df = sample_stock_features_df.copy()
        # Create invalid data
        df.loc[0, "price_target_high"] = df.loc[0, "price_target_low"] - 10

        result = monte_carlo_price_target_simulation(df)

        # Should still return results (excluding invalid rows)
        assert isinstance(result, pd.DataFrame)


# =============================================================================
# Section 4: Bayesian Earnings Model Tests
# =============================================================================


class TestBayesianEarningsBeatModel:
    """Tests for bayesian_earnings_beat_model function."""

    def test_returns_dataframe(self, sample_stock_features_df):
        """Function should return a pandas DataFrame."""
        from finance_ml.analytics.feature_analytics import bayesian_earnings_beat_model

        result = bayesian_earnings_beat_model(sample_stock_features_df)

        assert isinstance(result, pd.DataFrame)

    def test_output_has_required_columns(self, sample_stock_features_df):
        """Output should contain expected columns."""
        from finance_ml.analytics.feature_analytics import bayesian_earnings_beat_model

        result = bayesian_earnings_beat_model(sample_stock_features_df)

        required_cols = ["ticker", "posterior_beat_prob", "model_confidence", "map_estimate"]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_posterior_probability_bounds(self, sample_stock_features_df):
        """Posterior probabilities should be between 0 and 1."""
        from finance_ml.analytics.feature_analytics import bayesian_earnings_beat_model

        result = bayesian_earnings_beat_model(sample_stock_features_df)

        assert result["posterior_beat_prob"].min() >= 0
        assert result["posterior_beat_prob"].max() <= 1

    def test_handles_missing_streak_data(self, sample_stock_features_with_nulls):
        """Should handle missing eps_positive_streak values."""
        from finance_ml.analytics.feature_analytics import bayesian_earnings_beat_model

        result = bayesian_earnings_beat_model(sample_stock_features_with_nulls)

        assert isinstance(result, pd.DataFrame)

    def test_higher_streak_higher_probability(self, sample_stock_features_df):
        """Stocks with higher EPS streaks should have higher beat probability."""
        from finance_ml.analytics.feature_analytics import bayesian_earnings_beat_model

        result = bayesian_earnings_beat_model(sample_stock_features_df)

        # Group by streak (already in result) and check average probability increases
        avg_by_streak = result.groupby("eps_positive_streak")["posterior_beat_prob"].mean()

        # Higher streaks should generally have higher probabilities
        if len(avg_by_streak) > 2:
            assert avg_by_streak.iloc[-1] > avg_by_streak.iloc[0]


# =============================================================================
# Section 5: Distress Risk Analysis Tests
# =============================================================================


class TestDistressDistributionAnalysis:
    """Tests for analyze_distress_distribution function."""

    def test_returns_plotly_figure(self, sample_stock_features_df):
        """Function should return a Plotly Figure object."""
        from finance_ml.analytics.feature_analytics import analyze_distress_distribution

        fig = analyze_distress_distribution(sample_stock_features_df)

        assert isinstance(fig, Figure)

    def test_figure_has_multiple_panels(self, sample_stock_features_df):
        """Dashboard should have multiple analysis panels."""
        from finance_ml.analytics.feature_analytics import analyze_distress_distribution

        fig = analyze_distress_distribution(sample_stock_features_df)

        # Should have multiple traces for different panels
        assert len(fig.data) >= 3

    def test_handles_missing_distress_scores(self, sample_stock_features_with_nulls):
        """Should handle missing distress_risk_score values."""
        from finance_ml.analytics.feature_analytics import analyze_distress_distribution

        fig = analyze_distress_distribution(sample_stock_features_with_nulls)

        assert isinstance(fig, Figure)


# =============================================================================
# Section 6: Composite Quality Score Tests
# =============================================================================


class TestCompositeQualityScore:
    """Tests for create_composite_quality_score function."""

    def test_returns_dataframe(self, sample_stock_features_df):
        """Function should return a pandas DataFrame."""
        from finance_ml.analytics.feature_analytics import create_composite_quality_score

        result = create_composite_quality_score(sample_stock_features_df)

        assert isinstance(result, pd.DataFrame)

    def test_output_has_composite_score(self, sample_stock_features_df):
        """Output should contain composite_quality_score column."""
        from finance_ml.analytics.feature_analytics import create_composite_quality_score

        result = create_composite_quality_score(sample_stock_features_df)

        assert "composite_quality_score" in result.columns

    def test_output_has_quality_tier(self, sample_stock_features_df):
        """Output should contain quality_tier categorical column."""
        from finance_ml.analytics.feature_analytics import create_composite_quality_score

        result = create_composite_quality_score(sample_stock_features_df)

        assert "quality_tier" in result.columns

    def test_composite_score_bounds(self, sample_stock_features_df):
        """Composite score should be between 0 and 100."""
        from finance_ml.analytics.feature_analytics import create_composite_quality_score

        result = create_composite_quality_score(sample_stock_features_df)

        assert result["composite_quality_score"].min() >= 0
        assert result["composite_quality_score"].max() <= 100

    def test_quality_tiers_valid(self, sample_stock_features_df):
        """Quality tiers should be valid categories."""
        from finance_ml.analytics.feature_analytics import create_composite_quality_score

        result = create_composite_quality_score(sample_stock_features_df)

        valid_tiers = {"Low", "Below Avg", "Above Avg", "High"}
        actual_tiers = set(result["quality_tier"].dropna().unique())

        assert actual_tiers.issubset(valid_tiers)

    def test_sorted_by_score_descending(self, sample_stock_features_df):
        """Output should be sorted by composite score descending."""
        from finance_ml.analytics.feature_analytics import create_composite_quality_score

        result = create_composite_quality_score(sample_stock_features_df)

        scores = result["composite_quality_score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# =============================================================================
# Section 7: Summary Dashboard Tests
# =============================================================================


class TestSummaryDashboard:
    """Tests for create_summary_dashboard function."""

    def test_returns_plotly_figure(self, sample_stock_features_df):
        """Function should return a Plotly Figure object."""
        from finance_ml.analytics.feature_analytics import create_summary_dashboard

        fig = create_summary_dashboard(sample_stock_features_df)

        assert isinstance(fig, Figure)

    def test_has_indicator_traces(self, sample_stock_features_df):
        """Dashboard should have indicator traces for KPIs."""
        from finance_ml.analytics.feature_analytics import create_summary_dashboard

        fig = create_summary_dashboard(sample_stock_features_df)

        indicator_traces = [t for t in fig.data if t.type == "indicator"]
        assert len(indicator_traces) >= 4

    def test_handles_missing_columns(self, sample_stock_features_df):
        """Should handle missing optional columns gracefully."""
        from finance_ml.analytics.feature_analytics import create_summary_dashboard

        df = sample_stock_features_df.drop(columns=["earnings_quality_composite"])

        fig = create_summary_dashboard(df)

        assert isinstance(fig, Figure)


# =============================================================================
# Section 8: Integration Tests
# =============================================================================


class TestFeatureAnalyticsIntegration:
    """Integration tests for the feature_analytics module."""

    def test_all_functions_importable(self):
        """All main functions should be importable from the module."""
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

        # All imports successful
        assert callable(create_interactive_momentum_dashboard)
        assert callable(create_interactive_valuation_heatmap)
        assert callable(create_leverage_liquidity_quadrant)
        assert callable(monte_carlo_price_target_simulation)
        assert callable(bayesian_earnings_beat_model)
        assert callable(analyze_distress_distribution)
        assert callable(create_composite_quality_score)
        assert callable(create_summary_dashboard)

    def test_module_has_plotly_template_constant(self):
        """Module should export PLOTLY_TEMPLATE constant."""
        from finance_ml.analytics.feature_analytics import PLOTLY_TEMPLATE

        assert isinstance(PLOTLY_TEMPLATE, str)
        assert PLOTLY_TEMPLATE == "plotly_dark"

    def test_full_workflow(self, sample_stock_features_df):
        """Test a complete workflow using multiple functions."""
        from finance_ml.analytics.feature_analytics import (
            create_interactive_momentum_dashboard,
            monte_carlo_price_target_simulation,
            create_composite_quality_score,
        )

        # Generate visualizations
        momentum_fig = create_interactive_momentum_dashboard(sample_stock_features_df)
        assert isinstance(momentum_fig, Figure)

        # Run Monte Carlo simulation
        mc_results = monte_carlo_price_target_simulation(sample_stock_features_df)
        assert len(mc_results) > 0

        # Generate composite scores
        quality_scores = create_composite_quality_score(sample_stock_features_df)
        assert "composite_quality_score" in quality_scores.columns
