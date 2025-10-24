"""
Tests for risk metrics module (TDD implementation).

This test module follows strict TDD methodology:
1. Write failing tests first (RED)
2. Implement minimal code to pass (GREEN)
3. Refactor and optimize (REFACTOR)

Risk metrics to implement:
- Value at Risk (VaR): Historical and Parametric methods
- Conditional Value at Risk (CVaR): Expected shortfall
- Sharpe Ratio: Risk-adjusted returns
- Sortino Ratio: Downside risk-adjusted returns
- Maximum Drawdown: Peak-to-trough decline
"""
import unittest
import numpy as np
import pandas as pd
from finance_ml.risk_metrics import (
    calculate_var_historical,
    calculate_var_parametric,
    calculate_cvar,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_portfolio_risk_metrics,
)


class TestValueAtRisk(unittest.TestCase):
    """Test Value at Risk (VaR) calculations."""

    def setUp(self):
        """Set up test data."""
        # Generate sample returns
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        
    def test_historical_var_at_95_confidence(self):
        """Test historical VaR calculation at 95% confidence level."""
        var_95 = calculate_var_historical(self.returns, confidence_level=0.95)
        self.assertIsInstance(var_95, float)
        self.assertLess(var_95, 0)  # VaR should be negative (loss)
        
    def test_historical_var_at_99_confidence(self):
        """Test historical VaR calculation at 99% confidence level."""
        var_99 = calculate_var_historical(self.returns, confidence_level=0.99)
        self.assertIsInstance(var_99, float)
        self.assertLess(var_99, 0)  # VaR should be negative (loss)
        # 99% VaR should be more extreme than 95% VaR
        var_95 = calculate_var_historical(self.returns, confidence_level=0.95)
        self.assertLess(var_99, var_95)
        
    def test_parametric_var_calculation(self):
        """Test parametric VaR using normal distribution assumption."""
        var_param = calculate_var_parametric(self.returns, confidence_level=0.95)
        self.assertIsInstance(var_param, float)
        self.assertLess(var_param, 0)  # VaR should be negative (loss)
        
    def test_var_with_empty_returns_raises_error(self):
        """Test that VaR calculation raises error with empty returns."""
        empty_returns = pd.Series([])
        with self.assertRaises(ValueError):
            calculate_var_historical(empty_returns)
            
    def test_var_with_invalid_confidence_raises_error(self):
        """Test that VaR calculation raises error with invalid confidence level."""
        with self.assertRaises(ValueError):
            calculate_var_historical(self.returns, confidence_level=1.5)


class TestConditionalValueAtRisk(unittest.TestCase):
    """Test Conditional Value at Risk (CVaR) calculations."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        
    def test_cvar_calculation_at_95_confidence(self):
        """Test CVaR calculation at 95% confidence level."""
        cvar_95 = calculate_cvar(self.returns, confidence_level=0.95)
        self.assertIsInstance(cvar_95, float)
        self.assertLess(cvar_95, 0)  # CVaR should be negative (loss)
        
    def test_cvar_is_more_extreme_than_var(self):
        """Test that CVaR is more extreme (larger loss) than VaR."""
        var_95 = calculate_var_historical(self.returns, confidence_level=0.95)
        cvar_95 = calculate_cvar(self.returns, confidence_level=0.95)
        self.assertLess(cvar_95, var_95)  # CVaR should be worse than VaR
        
    def test_cvar_with_empty_returns_raises_error(self):
        """Test that CVaR calculation raises error with empty returns."""
        empty_returns = pd.Series([])
        with self.assertRaises(ValueError):
            calculate_cvar(empty_returns)


class TestSharpeRatio(unittest.TestCase):
    """Test Sharpe ratio calculations."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Positive returns with some volatility
        self.positive_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        # Negative returns
        self.negative_returns = pd.Series(np.random.normal(-0.001, 0.02, 252))
        
    def test_sharpe_ratio_with_positive_returns(self):
        """Test Sharpe ratio with positive returns."""
        sharpe = calculate_sharpe_ratio(self.positive_returns, risk_free_rate=0.0)
        self.assertIsInstance(sharpe, float)
        self.assertGreater(sharpe, 0)  # Should be positive with positive returns
        
    def test_sharpe_ratio_with_negative_returns(self):
        """Test Sharpe ratio with negative returns."""
        sharpe = calculate_sharpe_ratio(self.negative_returns, risk_free_rate=0.0)
        self.assertIsInstance(sharpe, float)
        self.assertLess(sharpe, 0)  # Should be negative with negative returns
        
    def test_sharpe_ratio_with_risk_free_rate(self):
        """Test Sharpe ratio with non-zero risk-free rate."""
        sharpe_no_rf = calculate_sharpe_ratio(self.positive_returns, risk_free_rate=0.0)
        sharpe_with_rf = calculate_sharpe_ratio(self.positive_returns, risk_free_rate=0.02)
        self.assertLess(sharpe_with_rf, sharpe_no_rf)  # Higher RF should lower Sharpe
        
    def test_sharpe_ratio_annualization(self):
        """Test that Sharpe ratio is properly annualized."""
        sharpe_daily = calculate_sharpe_ratio(self.positive_returns, periods_per_year=252)
        sharpe_monthly = calculate_sharpe_ratio(self.positive_returns, periods_per_year=12)
        # Different annualization should produce different results
        self.assertNotEqual(sharpe_daily, sharpe_monthly)
        
    def test_sharpe_ratio_with_zero_volatility_raises_error(self):
        """Test that Sharpe ratio raises error with zero volatility."""
        constant_returns = pd.Series([0.01] * 100)
        with self.assertRaises(ValueError):
            calculate_sharpe_ratio(constant_returns)


class TestSortinoRatio(unittest.TestCase):
    """Test Sortino ratio calculations."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        sortino = calculate_sortino_ratio(self.returns, risk_free_rate=0.0)
        self.assertIsInstance(sortino, float)
        
    def test_sortino_ratio_higher_than_sharpe_with_positive_skew(self):
        """Test that Sortino is typically higher than Sharpe for positive skew."""
        # Create positively skewed returns (more upside than downside)
        positive_skew_returns = pd.Series(np.concatenate([
            np.random.normal(0.01, 0.01, 200),  # More positive
            np.random.normal(-0.005, 0.01, 50)   # Fewer negative
        ]))
        sharpe = calculate_sharpe_ratio(positive_skew_returns, risk_free_rate=0.0)
        sortino = calculate_sortino_ratio(positive_skew_returns, risk_free_rate=0.0)
        self.assertGreater(sortino, sharpe)  # Sortino should be higher
        
    def test_sortino_ratio_with_target_return(self):
        """Test Sortino ratio with custom target return."""
        sortino_zero = calculate_sortino_ratio(self.returns, target_return=0.0)
        sortino_higher = calculate_sortino_ratio(self.returns, target_return=0.001)
        self.assertNotEqual(sortino_zero, sortino_higher)


class TestMaximumDrawdown(unittest.TestCase):
    """Test maximum drawdown calculations."""
    
    def test_max_drawdown_with_declining_prices(self):
        """Test max drawdown with steadily declining prices."""
        prices = pd.Series([100, 90, 80, 70, 60])
        max_dd = calculate_max_drawdown(prices)
        self.assertIsInstance(max_dd, float)
        self.assertLess(max_dd, 0)  # Drawdown should be negative
        self.assertAlmostEqual(max_dd, -0.4, places=2)  # 40% drawdown
        
    def test_max_drawdown_with_recovery(self):
        """Test max drawdown with price recovery."""
        prices = pd.Series([100, 120, 80, 90, 110])
        max_dd = calculate_max_drawdown(prices)
        self.assertLess(max_dd, 0)  # Should capture the drawdown
        
    def test_max_drawdown_with_flat_prices(self):
        """Test max drawdown with flat prices."""
        prices = pd.Series([100] * 10)
        max_dd = calculate_max_drawdown(prices)
        self.assertEqual(max_dd, 0.0)  # No drawdown
        
    def test_max_drawdown_with_only_gains(self):
        """Test max drawdown with only gains."""
        prices = pd.Series([100, 110, 120, 130, 140])
        max_dd = calculate_max_drawdown(prices)
        self.assertEqual(max_dd, 0.0)  # No drawdown
        
    def test_max_drawdown_returns_dict_with_details(self):
        """Test that max drawdown can return detailed metrics."""
        prices = pd.Series([100, 120, 80, 90, 110])
        result = calculate_max_drawdown(prices, return_details=True)
        self.assertIsInstance(result, dict)
        self.assertIn('max_drawdown', result)
        self.assertIn('peak_date', result)
        self.assertIn('trough_date', result)


class TestPortfolioRiskMetrics(unittest.TestCase):
    """Test comprehensive portfolio risk metrics calculation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        self.prices = (1 + self.returns).cumprod() * 100
        
    def test_portfolio_risk_metrics_returns_dict(self):
        """Test that portfolio risk metrics returns a dictionary."""
        metrics = calculate_portfolio_risk_metrics(self.returns, self.prices)
        self.assertIsInstance(metrics, dict)
        
    def test_portfolio_risk_metrics_contains_all_metrics(self):
        """Test that all risk metrics are included in the result."""
        metrics = calculate_portfolio_risk_metrics(self.returns, self.prices)
        expected_keys = [
            'var_95_historical',
            'var_99_historical',
            'var_95_parametric',
            'cvar_95',
            'cvar_99',
            'sharpe_ratio',
            'sortino_ratio',
            'max_drawdown',
            'volatility',
            'mean_return'
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)
            
    def test_portfolio_risk_metrics_with_dataframe_input(self):
        """Test portfolio risk metrics with DataFrame input (multiple assets)."""
        # Create multi-asset returns
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns_df = pd.DataFrame({
            'Asset1': np.random.normal(0.001, 0.02, 252),
            'Asset2': np.random.normal(0.0005, 0.015, 252),
        }, index=dates)
        
        # Should work with DataFrame
        metrics = calculate_portfolio_risk_metrics(returns_df['Asset1'])
        self.assertIsInstance(metrics, dict)


if __name__ == '__main__':
    unittest.main()
