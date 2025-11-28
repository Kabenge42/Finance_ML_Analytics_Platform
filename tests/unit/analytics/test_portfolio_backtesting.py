"""Tests for Phase 5 backtesting framework (vectorised backtests & attribution).

The test cases are derived from
docs/improvement_plan/portfolio_optimization_enhancement_plan.md and
focus on:

- Vectorised backtest engine (run_vectorized_backtest)
- Walk‑forward optimisation backtest (run_walk_forward_optimization)
- Brinson‑Fachler performance attribution
"""

import unittest

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.analytics.portfolio import (
    load_historical_prices,
    run_vectorized_backtest,
    run_walk_forward_optimization,
)
from finance_ml.ml_workflow.analytics.risk import calculate_sharpe_ratio
from finance_ml.ml_workflow.analytics.attribution import (
    calculate_performance_attribution,
)


class TestVectorizedBacktest(unittest.TestCase):
    """Phase 5.1.1 – vectorised portfolio backtest engine."""

    def test_vectorized_backtest(self):
        """Test vectorised portfolio backtest contract and basic outputs."""

        historical_data = load_historical_prices()

        backtest_results = run_vectorized_backtest(
            data=historical_data,
            rebalance_frequency="monthly",
            optimization_method="max_sharpe",
            lookback_window=252,
            transaction_costs=0.001,
        )

        # Required keys
        self.assertIn("portfolio_returns", backtest_results)
        self.assertIn("turnover", backtest_results)
        self.assertIn("sharpe_ratio", backtest_results)
        self.assertIn("max_drawdown", backtest_results)

        # Basic type/shape checks
        portfolio_returns = backtest_results["portfolio_returns"]
        self.assertIsInstance(portfolio_returns, pd.Series)
        self.assertGreater(len(portfolio_returns), 0)

        self.assertIsInstance(backtest_results["turnover"], float)
        self.assertIsInstance(backtest_results["sharpe_ratio"], float)
        self.assertIsInstance(backtest_results["max_drawdown"], float)


class TestWalkForwardOptimization(unittest.TestCase):
    """Phase 5.1.2 – walk‑forward optimisation backtest."""

    def test_walk_forward_optimization(self):
        """Test walk‑forward optimisation backtest behaviour.

        Out‑of‑sample Sharpe is expected to be lower than in‑sample Sharpe,
        mimicking a typical over‑fitting diagnostic.
        """

        historical_data = load_historical_prices()

        wfo_results = run_walk_forward_optimization(
            data=historical_data,
            train_window=252,
            test_window=63,
            step_size=21,
            optimization_method="black_litterman",
        )

        self.assertIn("out_of_sample_returns", wfo_results)
        self.assertIn("in_sample_returns", wfo_results)

        oos = wfo_results["out_of_sample_returns"]
        ins = wfo_results["in_sample_returns"]

        self.assertIsInstance(oos, pd.Series)
        self.assertIsInstance(ins, pd.Series)
        self.assertGreater(len(oos), 0)
        self.assertGreater(len(ins), 0)

        # Out‑of‑sample should be less smooth (lower Sharpe) than in‑sample.
        oos_sharpe = calculate_sharpe_ratio(oos)
        is_sharpe = calculate_sharpe_ratio(ins)
        self.assertLess(oos_sharpe, is_sharpe)


class TestPerformanceAttribution(unittest.TestCase):
    """Phase 5.2.1 – Brinson‑Fachler performance attribution."""

    def test_performance_attribution(self):
        """Test Brinson‑Fachler allocation, selection, interaction effects.

        We construct a simple two‑sector, single‑period example with known
        weights and returns to validate that the sum of allocation,
        selection and interaction effects equals the total excess return.
        """

        # Single‑period sector weights and returns
        portfolio_weights = pd.DataFrame([[0.6, 0.4]], columns=["Tech", "Finance"])
        benchmark_weights = pd.DataFrame([[0.5, 0.5]], columns=["Tech", "Finance"])

        # Sector returns (e.g., over one month)
        portfolio_returns = pd.DataFrame([[0.12, 0.06]], columns=["Tech", "Finance"])
        benchmark_returns = pd.DataFrame([[0.10, 0.04]], columns=["Tech", "Finance"])

        attribution = calculate_performance_attribution(
            portfolio_weights,
            portfolio_returns,
            benchmark_weights,
            benchmark_returns,
        )

        self.assertIn("allocation_effect", attribution)
        self.assertIn("selection_effect", attribution)
        self.assertIn("interaction_effect", attribution)

        total_attr = (
            attribution["allocation_effect"]
            + attribution["selection_effect"]
            + attribution["interaction_effect"]
        )

        # Total excess return of the portfolio vs benchmark
        port_total = float((portfolio_weights.iloc[0] * portfolio_returns.iloc[0]).sum())
        bench_total = float((benchmark_weights.iloc[0] * benchmark_returns.iloc[0]).sum())
        excess_return = port_total - bench_total

        self.assertTrue(np.isclose(total_attr, excess_return))


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    unittest.main()
