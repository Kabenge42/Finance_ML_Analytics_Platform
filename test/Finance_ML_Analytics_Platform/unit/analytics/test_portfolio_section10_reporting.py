"""Section 10 — Portfolio Optimization Workflow (Enhancement Plan)

Lightweight TDD tests that validate the presence and minimal schema
of portfolio reporting artifacts under outputs/portfolio/.

These tests purposefully avoid heavy optimization or plotting. They
assert that the reporting wrapper creates the expected files with
non-empty content and minimal required keys/columns.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.analytics.portfolio_reporting import (
    universe_summary,
    returns_risk_diagnostics,
    frontier_and_constraints,
    risk_decomposition_dashboard,
    backtest_and_attribution,
    risk_management_dashboard,
    portfolio_summary,
)


def _make_universe_df(n: int = 8) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    sectors = ["Tech", "Health"]
    regions = ["US", "EU"]
    df = pd.DataFrame(
        {
            "ticker": [f"T{i:02d}" for i in range(n)],
            "sector": [sectors[i % len(sectors)] for i in range(n)],
            "region": [regions[i % len(regions)] for i in range(n)],
            "market_cap": rng.lognormal(mean=10, sigma=0.3, size=n),
        }
    )
    return df


def _make_expected_returns(n: int = 6) -> pd.Series:
    rng = np.random.RandomState(1)
    tickers = [f"A{i}" for i in range(n)]
    mu = pd.Series(rng.uniform(0.05, 0.15, size=n), index=tickers, name="mu")
    return mu


def _make_cov(n: int = 6) -> np.ndarray:
    # Positive definite covariance
    rng = np.random.RandomState(2)
    A = rng.randn(n, n)
    cov = A.T @ A / n
    return cov


def _make_weights(n: int = 5) -> pd.Series:
    w = np.ones(n) / n
    tickers = [f"W{i}" for i in range(n)]
    return pd.Series(w, index=tickers, name="weight")


def _make_exposures(weights: pd.Series) -> pd.DataFrame:
    # Minimal exposures mapping for sector/region
    df = pd.DataFrame(
        {
            "ticker": weights.index,
            "sector": ["Tech" if i % 2 == 0 else "Health" for i in range(len(weights))],
            "region": ["US" if i % 2 == 0 else "EU" for i in range(len(weights))],
        }
    ).set_index("ticker")
    return df


def _make_prices(n_assets: int = 4, n_days: int = 60) -> pd.DataFrame:
    rng = np.random.RandomState(3)
    rets = rng.normal(0.0005, 0.01, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = [f"P{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


class TestPortfolioSection10Reporting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp.name) / "portfolio"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_10_2_universe_and_filters_diagnostics(self):
        df = _make_universe_df()
        manifest = universe_summary(df, self.out_dir)
        json_path = self.out_dir / "portfolio_universe_summary.json"
        html_path = self.out_dir / "portfolio_universe_summary.html"
        self.assertTrue(json_path.exists(), "universe summary JSON missing")
        self.assertTrue(html_path.exists(), "universe summary HTML missing")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Minimal schema keys
        self.assertIn("by_sector", data)
        self.assertIn("by_region", data)
        self.assertIn("market_cap_buckets", data)
        # Manifest should list created files
        self.assertIn("portfolio_universe_summary.json", "|".join(manifest.get("files", [])))

    def test_10_3_expected_returns_and_risk_inputs(self):
        mu = _make_expected_returns(6)
        cov = _make_cov(6)
        manifest = returns_risk_diagnostics(mu, cov, self.out_dir)
        for fname in [
            "expected_returns_diagnostics.json",
            "expected_returns_distribution.html",
            "risk_correlation_heatmap.html",
            "risk_drift_dashboard.html",
        ]:
            path = self.out_dir / fname
            self.assertTrue(path.exists(), f"missing artifact: {fname}")
            self.assertGreater(path.stat().st_size, 0)
        # Sanity on JSON content
        with open(self.out_dir / "expected_returns_diagnostics.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        self.assertGreaterEqual(d.get("num_assets", 0), 1)
        self.assertIn("mean_return", d)

    def test_10_4_frontier_and_constraints(self):
        mu = _make_expected_returns(5)
        cov = _make_cov(5)
        constraints = {"max_weight": [0.1, 0.2]}
        manifest = frontier_and_constraints(mu, cov, constraints, self.out_dir)
        for fname in [
            "efficient_frontier.html",
            "constraints_sensitivity.html",
            "constraints_scenarios.csv",
            "transaction_cost_impact.html",
            "transaction_cost_summary.json",
        ]:
            self.assertTrue((self.out_dir / fname).exists(), f"missing {fname}")
        # CSV should have header and at least 1 row
        df = pd.read_csv(self.out_dir / "constraints_scenarios.csv")
        self.assertIn("max_weight", df.columns)
        self.assertGreaterEqual(len(df), 1)

    def test_10_5_breakdown_and_risk_decomposition(self):
        weights = _make_weights(5)
        exposures = _make_exposures(weights)
        manifest = risk_decomposition_dashboard(weights, exposures, self.out_dir)
        for fname in [
            "portfolio_holdings_detailed.csv",
            "portfolio_exposures.html",
            "risk_decomposition.html",
            "stress_tests_dashboard.html",
        ]:
            self.assertTrue((self.out_dir / fname).exists(), f"missing {fname}")
        # Holdings CSV should include ticker and weight
        df = pd.read_csv(self.out_dir / "portfolio_holdings_detailed.csv")
        self.assertIn("ticker", df.columns)
        self.assertIn("weight", df.columns)

    def test_10_6_backtesting_and_attribution(self):
        prices = _make_prices(4, 40)
        weights = _make_weights(4)
        manifest = backtest_and_attribution(prices, weights, self.out_dir)
        for fname in [
            "backtest_performance.html",
            "performance_attribution.html",
            "attribution_breakdown.csv",
        ]:
            self.assertTrue((self.out_dir / fname).exists(), f"missing {fname}")
        df = pd.read_csv(self.out_dir / "attribution_breakdown.csv")
        self.assertIn("ticker", df.columns)
        self.assertIn("contribution", df.columns)

    def test_10_7_risk_management_dashboard(self):
        cov = _make_cov(4)
        weights = _make_weights(4)
        manifest = risk_management_dashboard(weights, cov, self.out_dir)
        for fname in [
            "risk_management_dashboard.html",
            "portfolio_rebalancing_widget.html",
        ]:
            self.assertTrue((self.out_dir / fname).exists(), f"missing {fname}")

    def test_10_8_summary_and_export(self):
        kpis = {"sharpe": 1.2, "volatility": 0.15}
        manifest = portfolio_summary(kpis, self.out_dir)
        json_path = self.out_dir / "portfolio_summary.json"
        self.assertTrue(json_path.exists())
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.assertIn("kpis", d)
        # Ensure placeholder comparison HTML exists
        self.assertTrue((self.out_dir / "portfolio_multi_period_comparison.html").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
