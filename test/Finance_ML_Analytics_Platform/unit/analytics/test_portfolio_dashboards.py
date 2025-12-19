"""Tests for Phase 6 interactive portfolio dashboards.

The test cases are derived from
docs/improvement_plan/portfolio_optimization_enhancement_plan.md and
cover:

- Real-time portfolio rebalancing widget (PortfolioRebalanceWidget)
- Multi-period performance comparison visualization
- Factor exposure radar / spider chart
"""

import unittest

import numpy as np
import pandas as pd

from finance_ml.dashboards import (
    PortfolioRebalanceWidget,
    create_multi_period_comparison,
    create_factor_exposure_dashboard,
)


def create_sample_holdings() -> pd.DataFrame:
    """Create a small deterministic holdings DataFrame for tests.

    Columns:
        ticker, shares, price
    """

    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "shares": [100.0, 100.0, 100.0],
            "price": [10.0, 20.0, 30.0],  # portfolio value = 1000 + 2000 + 3000 = 6000
        }
    )


def create_target_weights() -> pd.Series:
    """Target weights for the rebalance widget tests.

    The targets intentionally differ from the current equal-weight
    allocation so that non-trivial trades are required.
    """

    return pd.Series(
        data=[0.5, 0.3, 0.2],
        index=["AAA", "BBB", "CCC"],
        name="target_weight",
    )


def create_sample_returns(n: int = 252) -> pd.Series:
    """Synthetic daily portfolio returns series for plotting tests."""

    rng = np.random.RandomState(42)
    return pd.Series(rng.normal(0.0005, 0.01, size=n))


def create_benchmark_returns(n: int = 252) -> pd.Series:
    """Synthetic benchmark returns that roughly track the portfolio."""

    rng = np.random.RandomState(7)
    return pd.Series(rng.normal(0.0004, 0.009, size=n))


def create_sample_weights() -> pd.Series:
    """Sample portfolio weights for factor exposure tests."""

    return pd.Series([0.4, 0.35, 0.25], index=["AAA", "BBB", "CCC"], name="weight")


def load_factor_loadings() -> pd.DataFrame:
    """Create deterministic factor loadings for a few assets.

    Columns correspond to common equity factors such as Market, Size,
    Value, Momentum, and Quality.
    """

    return pd.DataFrame(
        {
            "Market": [1.1, 0.9, 0.8],
            "Size": [0.2, -0.1, 0.05],
            "Value": [0.3, 0.4, 0.2],
            "Momentum": [0.1, 0.2, 0.15],
            "Quality": [0.5, 0.6, 0.4],
        },
        index=["AAA", "BBB", "CCC"],
    )


class TestInteractiveRebalanceWidget(unittest.TestCase):
    """Phase 6.1.1 – interactive portfolio rebalancing widget."""

    def test_interactive_rebalance_widget(self):
        """Test interactive portfolio rebalancing widget.

        The widget should propose trades that move the portfolio from the
        current holdings to the desired target weights.
        """

        widget = PortfolioRebalanceWidget(
            current_holdings=create_sample_holdings(),
            target_weights=create_target_weights(),
        )

        trades = widget.get_rebalance_trades()

        # Required columns
        self.assertIn("ticker", trades.columns)
        self.assertIn("action", trades.columns)  # BUY/SELL
        self.assertIn("shares", trades.columns)
        self.assertIn("estimated_cost", trades.columns)

        # Verify that applying trades leads to target weights (within tolerance)
        holdings = create_sample_holdings().set_index("ticker")
        merged = trades.set_index("ticker").reindex(holdings.index).fillna(0.0)

        new_shares = holdings["shares"] + merged["shares"]
        prices = holdings["price"]
        new_values = new_shares * prices
        total_value = float(new_values.sum())
        new_weights = new_values / total_value

        target_weights = create_target_weights()
        # Align indices and compare
        target_aligned = target_weights.reindex(new_weights.index)
        self.assertTrue(np.allclose(new_weights.values, target_aligned.values, atol=1e-4))


class TestMultiPeriodComparisonPlot(unittest.TestCase):
    """Phase 6.2.1 – multi-period performance comparison visualization."""

    def test_multi_period_comparison_plot(self):
        """Test multi-period performance comparison visualization."""

        portfolio_returns = create_sample_returns()
        benchmark_returns = create_benchmark_returns(len(portfolio_returns))

        periods = ["1M", "3M", "6M", "1Y", "YTD", "ITD"]

        fig = create_multi_period_comparison(
            portfolio_returns,
            periods=periods,
            benchmark_returns=benchmark_returns,
        )

        # Verify plot has correct structure
        self.assertGreaterEqual(len(fig.data), 2)  # Portfolio + Benchmark
        self.assertIn("Period", fig.layout.xaxis.title.text)
        self.assertIn("Return", fig.layout.yaxis.title.text)


class TestFactorExposureDashboard(unittest.TestCase):
    """Phase 6.3.1 – factor exposure analysis dashboard."""

    def test_factor_exposure_visualization(self):
        """Test factor exposure analysis dashboard (radar chart)."""

        portfolio_weights = create_sample_weights()
        factor_loadings = load_factor_loadings()

        factors = ["Market", "Size", "Value", "Momentum", "Quality"]

        fig = create_factor_exposure_dashboard(
            portfolio_weights,
            factor_loadings,
            factors=factors,
        )

        # Verify spider/radar chart structure
        self.assertIsNotNone(fig.layout.polar)
        self.assertGreaterEqual(len(fig.data), 1)


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    unittest.main()
