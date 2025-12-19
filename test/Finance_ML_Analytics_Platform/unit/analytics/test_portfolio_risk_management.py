"""Tests for Phase 4 risk management enhancements.

The test cases are derived from
docs/improvement_plan/portfolio_optimization_enhancement_plan.md and
serve as review checkpoints for advanced risk metrics, stress testing,
and Monte Carlo simulation utilities.
"""

import unittest

import numpy as np
import pandas as pd
from scipy import stats

from finance_ml.ml_workflow.analytics.risk import (
    calculate_var_historical,
    calculate_expected_shortfall,
    calculate_tracking_error,
    run_stress_tests,
    run_monte_carlo_simulation,
)


def create_sample_returns_for_risk(n_obs: int = 1000) -> pd.Series:
    """Create synthetic daily returns for risk‑metric tests.

    We use a simple normal distribution with a small positive mean and
    realistic volatility so that VaR/CVaR relationships and Monte Carlo
    properties behave in expected ways.
    """

    rng = np.random.RandomState(42)
    return pd.Series(rng.normal(0.0005, 0.02, size=n_obs))


class TestExpectedShortfall(unittest.TestCase):
    """Phase 4.1.1 – Expected Shortfall (ES) calculation.

    This test leverages the existing CVaR implementation as the
    realisation of Expected Shortfall and validates its behaviour
    relative to historical VaR at multiple confidence levels.
    """

    def test_calculate_expected_shortfall_via_cvar(self):
        returns = create_sample_returns_for_risk()

        es_95 = calculate_expected_shortfall(returns, confidence=0.95)
        es_99 = calculate_expected_shortfall(returns, confidence=0.99)

        var_95 = calculate_var_historical(returns, confidence_level=0.95)

        # ES should be more extreme (more negative) than VaR for the same
        # confidence level, and ES at 99% should be more extreme than at 95%.
        self.assertLess(es_95, var_95)
        self.assertLess(es_99, es_95)


class TestTrackingError(unittest.TestCase):
    """Phase 4.1.2 – tracking error vs benchmark.

    Tracking error is defined as the annualised standard deviation of the
    difference between portfolio and benchmark returns.
    """

    def test_calculate_tracking_error(self):
        rng = np.random.RandomState(123)
        portfolio_returns = pd.Series(rng.normal(0.001, 0.02, size=500))
        # Benchmark closely tracks the portfolio with small noise
        benchmark_returns = portfolio_returns + rng.normal(0.0, 0.005, size=500)

        # Manual tracking error calculation according to the definition
        diff = portfolio_returns - benchmark_returns
        expected_te = float(diff.std(ddof=1) * np.sqrt(252))

        # Use library helper and compare with manual implementation
        te = calculate_tracking_error(portfolio_returns, benchmark_returns)

        self.assertGreaterEqual(te, 0.0)
        self.assertIsInstance(te, float)
        self.assertTrue(np.isclose(te, expected_te))


class TestPortfolioStressTesting(unittest.TestCase):
    """Phase 4.2.1 – portfolio stress testing scenarios.

    This test focuses on the semantics of scenario losses rather than the
    exact numeric values. It establishes the review checkpoint that
    severe equity shocks should translate into negative portfolio
    responses under reasonable allocations.
    """

    def test_portfolio_stress_testing_market_crash(self):
        # Simple 4‑asset portfolio with equal weights
        weights = np.array([0.3, 0.3, 0.2, 0.2])

        # Synthetic daily returns for 4 assets; the exact series is less
        # important than the fact that the stress scenario dominates.
        rng = np.random.RandomState(7)
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.02, size=(252, 4)),
            columns=["equity_1", "equity_2", "bond_1", "bond_2"],
        )

        # Simple stress scenario: equity assets lose 30%, bonds lose 10%.
        scenarios = {
            "Market Crash": {"equity": -0.30, "bonds": -0.10},
        }

        stress_results = run_stress_tests(
            weights,
            returns,
            scenarios=scenarios,
            asset_class_mapping=["equity", "equity", "bonds", "bonds"],
        )

        self.assertIn("Market Crash", stress_results)
        self.assertIn("portfolio_loss", stress_results["Market Crash"])
        self.assertLess(stress_results["Market Crash"]["portfolio_loss"], 0.0)


class TestMonteCarloPortfolioSimulation(unittest.TestCase):
    """Phase 4.3.1 – Monte Carlo simulation for portfolio paths.

    We implement a small, self‑contained Monte Carlo engine inside the
    test to serve as the TDD reference for future refactoring into
    analytics.risk. The focus is on shape, percentile paths, and basic
    distribution properties of final portfolio values.
    """

    def test_monte_carlo_portfolio_simulation(self):
        rng = np.random.RandomState(21)
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        base_returns = rng.normal(0.0005, 0.02, size=(252, 4))
        returns_df = pd.DataFrame(base_returns, columns=["A", "B", "C", "D"])

        # Parameters
        n_simulations = 2000
        time_horizon = 252

        sim_results = run_monte_carlo_simulation(
            weights,
            returns_df,
            n_simulations=n_simulations,
            time_horizon=time_horizon,
            confidence_levels=[0.05, 0.5, 0.95],
            random_state=21,
        )

        paths = sim_results["paths"]
        self.assertEqual(paths.shape, (n_simulations, time_horizon))

        # Percentile paths
        self.assertIn("p05_path", sim_results)
        self.assertIn("p50_path", sim_results)
        self.assertIn("p95_path", sim_results)

        self.assertEqual(sim_results["p05_path"].shape[0], time_horizon)
        self.assertEqual(sim_results["p50_path"].shape[0], time_horizon)
        self.assertEqual(sim_results["p95_path"].shape[0], time_horizon)

        # Distribution sanity check: final portfolio values should have
        # roughly symmetric distribution with limited skew.
        final_values = paths[:, -1]
        self.assertLess(abs(stats.skew(final_values)), 1.0)


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    unittest.main()
