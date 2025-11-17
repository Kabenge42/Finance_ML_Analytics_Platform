"""Tests for advanced portfolio optimization methods (Phase 3).

Concrete TDD tests derived from
docs/improvement_plan/portfolio_optimization_enhancement_plan.md.
"""

import unittest

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.analytics.portfolio import (
    optimize_black_litterman,
    optimize_risk_parity,
    optimize_hrp,
    optimize_portfolio_min_volatility,
    calculate_portfolio_return,
    calculate_portfolio_volatility,
)


def create_sample_returns(n_obs: int = 252) -> pd.DataFrame:
    """Create small synthetic daily return sample for a few assets.

    The assets are named to match the Black-Litterman example in the
    enhancement plan and have modest correlations suitable for testing.
    """

    rng = np.random.RandomState(123)

    # Base uncorrelated shocks
    aapl = rng.normal(0.0005, 0.02, size=n_obs)
    msft = rng.normal(0.0004, 0.018, size=n_obs)
    goog = rng.normal(0.0003, 0.022, size=n_obs)

    # Introduce mild correlation between AAPL and MSFT
    common = rng.normal(0.0, 0.01, size=n_obs)
    aapl = aapl + 0.5 * common
    msft = msft + 0.4 * common

    return pd.DataFrame({"AAPL": aapl, "MSFT": msft, "GOOG": goog})


class TestBlackLittermanOptimization(unittest.TestCase):
    """Phase 3.1 – Black-Litterman optimization tests."""

    def test_black_litterman_optimization(self):
        returns_df = create_sample_returns()
        cov_matrix = returns_df.cov()

        # Market equilibrium returns (annualised mean returns)
        mean_returns = returns_df.mean() * 252
        market_weights = np.array([1 / len(mean_returns)] * len(mean_returns))

        # Investor views with confidences
        views = {"AAPL": 0.15, "MSFT": 0.12}
        view_confidences = [0.8, 0.7]

        bl_weights, bl_returns = optimize_black_litterman(
            returns=mean_returns,
            cov_matrix=cov_matrix * 252,
            market_weights=market_weights,
            views=views,
            view_confidences=view_confidences,
            risk_aversion=2.5,
        )

        self.assertEqual(len(bl_weights), len(mean_returns))
        self.assertTrue(np.isclose(bl_weights.sum(), 1.0))
        self.assertTrue((bl_weights >= 0).all())  # Long-only
        self.assertEqual(len(bl_returns), len(mean_returns))


class TestRiskParityOptimization(unittest.TestCase):
    """Phase 3.2 – risk parity portfolio tests."""

    def test_risk_parity_optimization(self):
        returns_df = create_sample_returns()
        cov_matrix = returns_df.cov().values

        rp_weights = optimize_risk_parity(cov_matrix)

        self.assertEqual(len(rp_weights), cov_matrix.shape[0])
        self.assertTrue(np.isclose(rp_weights.sum(), 1.0))
        self.assertTrue((rp_weights >= 0).all())

        # Verify approximately equal risk contribution
        portfolio_vol = np.sqrt(rp_weights @ cov_matrix @ rp_weights)
        marginal_contrib = cov_matrix @ rp_weights
        risk_contrib = rp_weights * marginal_contrib / portfolio_vol

        self.assertLess(np.std(risk_contrib), 0.01)


class TestHierarchicalRiskParity(unittest.TestCase):
    """Phase 3.3 – Hierarchical Risk Parity (HRP) tests."""

    def test_hierarchical_risk_parity(self):
        returns_df = create_sample_returns()

        hrp_weights = optimize_hrp(returns_df)

        self.assertEqual(len(hrp_weights), returns_df.shape[1])
        self.assertTrue(np.isclose(hrp_weights.sum(), 1.0))
        self.assertTrue((hrp_weights >= 0).all())


class TestAdvancedOptimizersVsMPTBaseline(unittest.TestCase):
    """Phase 3 review checkpoint – compare advanced optimizers vs MPT.

    These tests do not enforce a specific optimal allocation but ensure that
    advanced methods (risk parity, HRP, Black–Litterman) produce sensible
    risk/return characteristics relative to a standard MPT min‑volatility
    portfolio and an equal‑weight benchmark.
    """

    def test_advanced_optimizers_behave_sensibly_vs_mpt(self):
        returns_df = create_sample_returns()
        cov_matrix = returns_df.cov().values * 252
        mean_returns = returns_df.mean().values * 252

        n_assets = returns_df.shape[1]

        # Equal‑weight benchmark portfolio
        w_equal = np.full(n_assets, 1.0 / n_assets)
        r_equal = calculate_portfolio_return(w_equal, mean_returns)
        vol_equal = calculate_portfolio_volatility(w_equal, cov_matrix)

        # MPT min‑volatility portfolio
        mpt_result = optimize_portfolio_min_volatility(mean_returns, cov_matrix)
        w_mpt = mpt_result["weights"]
        r_mpt = mpt_result["return"]
        vol_mpt = mpt_result["volatility"]

        # Risk parity and HRP portfolios
        w_rp = optimize_risk_parity(cov_matrix)
        w_hrp = optimize_hrp(returns_df)

        vol_rp = calculate_portfolio_volatility(w_rp, cov_matrix)
        vol_hrp = calculate_portfolio_volatility(w_hrp, cov_matrix)

        # Sanity checks on weight constraints
        self.assertTrue(np.isclose(w_equal.sum(), 1.0))
        self.assertTrue(np.isclose(w_mpt.sum(), 1.0))
        self.assertTrue(np.isclose(w_rp.sum(), 1.0))
        self.assertTrue(np.isclose(w_hrp.sum(), 1.0))
        self.assertTrue((w_mpt >= 0).all())
        self.assertTrue((w_rp >= 0).all())
        self.assertTrue((w_hrp >= 0).all())

        # Volatility ordering: min‑vol should be the lowest risk; advanced
        # methods should sit between min‑vol and equal‑weight.
        self.assertLessEqual(vol_mpt, vol_equal)
        self.assertLessEqual(vol_rp, vol_equal * 1.1)
        self.assertLessEqual(vol_hrp, vol_equal * 1.1)
        self.assertGreaterEqual(vol_rp, vol_mpt * 0.9)
        self.assertGreaterEqual(vol_hrp, vol_mpt * 0.9)

        # Black–Litterman portfolio Sharpe ratio should not be materially
        # worse than the equal‑weight benchmark on this synthetic data.
        market_weights = w_equal
        views = {"AAPL": 0.15, "MSFT": 0.12}
        view_confidences = [0.8, 0.7]
        bl_weights, bl_returns = optimize_black_litterman(
            returns=returns_df.mean() * 252,
            cov_matrix=returns_df.cov() * 252,
            market_weights=market_weights,
            views=views,
            view_confidences=view_confidences,
        )

        vol_bl = calculate_portfolio_volatility(bl_weights, cov_matrix)
        r_bl = calculate_portfolio_return(bl_weights, mean_returns)

        sharpe_equal = r_equal / vol_equal
        sharpe_bl = r_bl / vol_bl

        self.assertGreaterEqual(sharpe_bl, sharpe_equal - 0.1)


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    unittest.main()
