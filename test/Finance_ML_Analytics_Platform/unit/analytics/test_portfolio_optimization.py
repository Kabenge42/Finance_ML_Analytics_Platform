"""
Test suite for portfolio optimization module.

Tests Modern Portfolio Theory (MPT) implementation including:
- Portfolio metrics calculation (return, volatility, Sharpe ratio)
- Efficient frontier generation
- Portfolio optimization (max Sharpe, min volatility, target return)
- Weight constraints and validation
- Edge cases and error handling

Following strict TDD methodology: write tests first, then implement.
"""

import unittest

import numpy as np

from finance_ml.portfolio_optimization import (
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_portfolio_sharpe_ratio,
    generate_efficient_frontier,
    optimize_portfolio_max_sharpe,
    optimize_portfolio_min_volatility,
    optimize_portfolio_target_return,
    validate_weights,
    rebalance_portfolio,
)


class TestPortfolioMetrics(unittest.TestCase):
    """Test portfolio metrics calculations."""

    def setUp(self):
        """Set up test data."""
        # Simple 3-asset portfolio
        self.weights = np.array([0.4, 0.3, 0.3])
        self.returns = np.array([0.10, 0.12, 0.08])  # Expected returns
        self.cov_matrix = np.array(
            [[0.04, 0.01, 0.005], [0.01, 0.09, 0.015], [0.005, 0.015, 0.016]]
        )
        self.risk_free_rate = 0.02

    def test_calculate_portfolio_return(self):
        """Test portfolio expected return calculation."""
        expected_return = calculate_portfolio_return(self.weights, self.returns)

        # Manual calculation: 0.4*0.10 + 0.3*0.12 + 0.3*0.08 = 0.1
        self.assertAlmostEqual(expected_return, 0.10, places=6)

    def test_calculate_portfolio_return_zero_weights(self):
        """Test portfolio return with zero weights."""
        weights = np.array([0.0, 0.0, 1.0])
        expected_return = calculate_portfolio_return(weights, self.returns)
        self.assertAlmostEqual(expected_return, 0.08, places=6)

    def test_calculate_portfolio_volatility(self):
        """Test portfolio volatility calculation."""
        volatility = calculate_portfolio_volatility(self.weights, self.cov_matrix)

        # Should be positive
        self.assertGreater(volatility, 0)
        # Volatility should be reasonable (annualized)
        self.assertLess(volatility, 1.0)

    def test_calculate_portfolio_volatility_single_asset(self):
        """Test volatility with 100% allocation to single asset."""
        weights = np.array([1.0, 0.0, 0.0])
        volatility = calculate_portfolio_volatility(weights, self.cov_matrix)

        # Should equal sqrt of variance for asset 0
        expected = np.sqrt(self.cov_matrix[0, 0])
        self.assertAlmostEqual(volatility, expected, places=6)

    def test_calculate_portfolio_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        portfolio_return = 0.10
        portfolio_volatility = 0.15

        sharpe = calculate_portfolio_sharpe_ratio(
            portfolio_return, portfolio_volatility, self.risk_free_rate
        )

        # (0.10 - 0.02) / 0.15 = 0.533...
        expected = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        self.assertAlmostEqual(sharpe, expected, places=6)

    def test_calculate_portfolio_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility raises error."""
        with self.assertRaises(ValueError):
            calculate_portfolio_sharpe_ratio(0.10, 0.0, self.risk_free_rate)

    def test_calculate_portfolio_sharpe_ratio_negative_excess_return(self):
        """Test Sharpe ratio with negative excess return."""
        sharpe = calculate_portfolio_sharpe_ratio(0.01, 0.15, self.risk_free_rate)

        # (0.01 - 0.02) / 0.15 = -0.0667
        self.assertLess(sharpe, 0)


class TestWeightValidation(unittest.TestCase):
    """Test portfolio weight validation."""

    def test_validate_weights_valid(self):
        """Test validation of valid weights."""
        weights = np.array([0.3, 0.3, 0.4])
        is_valid, message = validate_weights(weights)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Valid")

    def test_validate_weights_sum_not_one(self):
        """Test validation fails when weights don't sum to 1."""
        weights = np.array([0.3, 0.3, 0.3])  # Sum = 0.9
        is_valid, message = validate_weights(weights)
        self.assertFalse(is_valid)
        self.assertIn("sum to 1", message.lower())

    def test_validate_weights_negative(self):
        """Test validation fails with negative weights."""
        weights = np.array([0.5, 0.6, -0.1])
        is_valid, message = validate_weights(weights, allow_short=False)
        self.assertFalse(is_valid)
        self.assertIn("negative", message.lower())

    def test_validate_weights_negative_allowed(self):
        """Test validation passes with negative weights when shorting allowed."""
        weights = np.array([0.5, 0.6, -0.1])
        is_valid, message = validate_weights(weights, allow_short=True)
        self.assertTrue(is_valid)

    def test_validate_weights_empty(self):
        """Test validation fails with empty weights."""
        weights = np.array([])
        is_valid, message = validate_weights(weights)
        self.assertFalse(is_valid)
        self.assertIn("empty", message.lower())

    def test_validate_weights_with_tolerance(self):
        """Test validation with numerical tolerance."""
        weights = np.array([0.333, 0.333, 0.334])  # Sum = 1.0 within tolerance
        is_valid, message = validate_weights(weights, tolerance=1e-3)
        self.assertTrue(is_valid)


class TestEfficientFrontier(unittest.TestCase):
    """Test efficient frontier generation."""

    def setUp(self):
        """Set up test data."""
        self.returns = np.array([0.08, 0.10, 0.12, 0.15])
        self.cov_matrix = np.array(
            [
                [0.01, 0.002, 0.001, 0.003],
                [0.002, 0.04, 0.005, 0.008],
                [0.001, 0.005, 0.09, 0.012],
                [0.003, 0.008, 0.012, 0.16],
            ]
        )
        self.risk_free_rate = 0.02

    def test_generate_efficient_frontier(self):
        """Test efficient frontier generation."""
        result = generate_efficient_frontier(self.returns, self.cov_matrix, num_portfolios=50)

        self.assertIn("returns", result)
        self.assertIn("volatilities", result)
        self.assertIn("sharpe_ratios", result)
        self.assertIn("weights", result)

        # Check dimensions
        self.assertEqual(len(result["returns"]), 50)
        self.assertEqual(len(result["volatilities"]), 50)
        self.assertEqual(len(result["sharpe_ratios"]), 50)
        self.assertEqual(result["weights"].shape, (50, 4))

    def test_efficient_frontier_return_range(self):
        """Test efficient frontier returns are within expected range."""
        result = generate_efficient_frontier(self.returns, self.cov_matrix, num_portfolios=100)

        # Returns should be between min and max asset returns
        self.assertGreaterEqual(result["returns"].min(), self.returns.min())
        self.assertLessEqual(result["returns"].max(), self.returns.max())

    def test_efficient_frontier_weights_valid(self):
        """Test all efficient frontier weights are valid."""
        result = generate_efficient_frontier(self.returns, self.cov_matrix, num_portfolios=30)

        for weights in result["weights"]:
            # Weights should sum to 1
            self.assertAlmostEqual(weights.sum(), 1.0, places=4)
            # No negative weights (no short selling by default)
            self.assertTrue(np.all(weights >= -1e-6))


class TestPortfolioOptimization(unittest.TestCase):
    """Test portfolio optimization algorithms."""

    def setUp(self):
        """Set up test data."""
        self.returns = np.array([0.08, 0.10, 0.12, 0.15])
        self.cov_matrix = np.array(
            [
                [0.01, 0.002, 0.001, 0.003],
                [0.002, 0.04, 0.005, 0.008],
                [0.001, 0.005, 0.09, 0.012],
                [0.003, 0.008, 0.012, 0.16],
            ]
        )
        self.risk_free_rate = 0.02

    def test_optimize_max_sharpe_ratio(self):
        """Test max Sharpe ratio optimization."""
        result = optimize_portfolio_max_sharpe(self.returns, self.cov_matrix, self.risk_free_rate)

        self.assertIn("weights", result)
        self.assertIn("return", result)
        self.assertIn("volatility", result)
        self.assertIn("sharpe_ratio", result)

        # Weights should sum to 1
        self.assertAlmostEqual(result["weights"].sum(), 1.0, places=4)
        # Sharpe ratio should be positive
        self.assertGreater(result["sharpe_ratio"], 0)

    def test_optimize_min_volatility(self):
        """Test minimum volatility optimization."""
        result = optimize_portfolio_min_volatility(self.returns, self.cov_matrix)

        self.assertIn("weights", result)
        self.assertIn("return", result)
        self.assertIn("volatility", result)

        # Weights should sum to 1
        self.assertAlmostEqual(result["weights"].sum(), 1.0, places=4)
        # Volatility should be positive and relatively low
        self.assertGreater(result["volatility"], 0)
        self.assertLess(result["volatility"], 0.5)

    def test_optimize_min_volatility_is_minimal(self):
        """Test that min volatility portfolio has lower volatility than random portfolios."""
        result = optimize_portfolio_min_volatility(self.returns, self.cov_matrix)

        # Generate random portfolios and compare
        np.random.seed(42)
        for _ in range(10):
            random_weights = np.random.dirichlet(np.ones(len(self.returns)))
            random_vol = calculate_portfolio_volatility(random_weights, self.cov_matrix)
            # Optimized should be <= random (with small tolerance for numerical issues)
            self.assertLessEqual(result["volatility"], random_vol + 1e-4)

    def test_optimize_target_return(self):
        """Test target return optimization."""
        target_return = 0.11
        result = optimize_portfolio_target_return(self.returns, self.cov_matrix, target_return)

        self.assertIn("weights", result)
        self.assertIn("return", result)
        self.assertIn("volatility", result)

        # Return should match target (within tolerance)
        self.assertAlmostEqual(result["return"], target_return, places=4)
        # Weights should sum to 1
        self.assertAlmostEqual(result["weights"].sum(), 1.0, places=4)

    def test_optimize_target_return_infeasible(self):
        """Test target return optimization with infeasible target."""
        target_return = 0.20  # Higher than max asset return

        with self.assertRaises(ValueError):
            optimize_portfolio_target_return(self.returns, self.cov_matrix, target_return)

    def test_optimize_with_constraints(self):
        """Test optimization with weight constraints."""
        # Max 40% in any single asset
        result = optimize_portfolio_max_sharpe(
            self.returns, self.cov_matrix, self.risk_free_rate, max_weight=0.4
        )

        # No weight should exceed 40%
        self.assertTrue(np.all(result["weights"] <= 0.4 + 1e-6))


class TestPortfolioRebalancing(unittest.TestCase):
    """Test portfolio rebalancing utilities."""

    def test_rebalance_portfolio_basic(self):
        """Test basic portfolio rebalancing."""
        current_weights = np.array([0.5, 0.3, 0.2])
        target_weights = np.array([0.4, 0.3, 0.3])

        result = rebalance_portfolio(current_weights, target_weights)

        self.assertIn("trades", result)
        self.assertIn("total_turnover", result)

        # Trades should bring portfolio to target
        trades = result["trades"]
        np.testing.assert_array_almost_equal(current_weights + trades, target_weights, decimal=6)

    def test_rebalance_portfolio_no_change(self):
        """Test rebalancing when no change needed."""
        weights = np.array([0.4, 0.3, 0.3])
        result = rebalance_portfolio(weights, weights)

        # No trades needed
        self.assertEqual(result["total_turnover"], 0.0)
        np.testing.assert_array_almost_equal(result["trades"], np.zeros(3))

    def test_rebalance_portfolio_with_threshold(self):
        """Test rebalancing with trade threshold."""
        current_weights = np.array([0.35, 0.33, 0.32])
        target_weights = np.array([0.34, 0.33, 0.33])

        # Small differences, high threshold should result in no trades
        result = rebalance_portfolio(current_weights, target_weights, trade_threshold=0.02)

        # Should skip small trades below threshold
        self.assertLess(result["total_turnover"], 0.02)

    def test_rebalance_portfolio_turnover_calculation(self):
        """Test turnover calculation is correct."""
        current_weights = np.array([0.6, 0.2, 0.2])
        target_weights = np.array([0.3, 0.4, 0.3])

        result = rebalance_portfolio(current_weights, target_weights)

        # Turnover = sum of absolute trades / 2
        expected_turnover = np.abs(result["trades"]).sum() / 2
        self.assertAlmostEqual(result["total_turnover"], expected_turnover, places=6)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_single_asset_portfolio(self):
        """Test optimization with single asset."""
        returns = np.array([0.10])
        cov_matrix = np.array([[0.04]])

        result = optimize_portfolio_min_volatility(returns, cov_matrix)

        # Should allocate 100% to the only asset
        np.testing.assert_array_almost_equal(result["weights"], np.array([1.0]))

    def test_mismatched_dimensions(self):
        """Test error handling for mismatched dimensions."""
        returns = np.array([0.08, 0.10, 0.12])
        cov_matrix = np.array([[0.01, 0.002], [0.002, 0.04]])  # Wrong size

        with self.assertRaises(ValueError):
            optimize_portfolio_min_volatility(returns, cov_matrix)

    def test_negative_covariance_matrix_diagonal(self):
        """Test error handling for invalid covariance matrix."""
        returns = np.array([0.08, 0.10])
        cov_matrix = np.array([[-0.01, 0.002], [0.002, 0.04]])  # Negative variance

        with self.assertRaises(ValueError):
            calculate_portfolio_volatility(np.array([0.5, 0.5]), cov_matrix)

    def test_all_zero_returns(self):
        """Test optimization with all zero returns."""
        returns = np.array([0.0, 0.0, 0.0])
        cov_matrix = np.array([[0.01, 0.002, 0.001], [0.002, 0.04, 0.005], [0.001, 0.005, 0.09]])

        # Should still be able to optimize for min volatility
        result = optimize_portfolio_min_volatility(returns, cov_matrix)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
