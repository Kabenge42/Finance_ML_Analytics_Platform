"""
Quick test to verify the efficient frontier optimization fix.
"""
import sys
sys.path.insert(0, '/')

import numpy as np
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.WARNING)  # Only show warnings

# Direct import to avoid full package initialization
from finance_ml.ml_workflow.analytics.portfolio import (
    generate_efficient_frontier,
    optimize_portfolio_target_return,
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_portfolio_sharpe_ratio
)

# Test with a realistic scenario: 150 stocks (similar to notebook)
np.random.seed(42)
n_stocks = 150

# Generate expected returns (annualized)
expected_returns = np.random.uniform(0.05, 0.50, n_stocks)

# Generate covariance matrix (similar to notebook setup)
corr_matrix = np.eye(n_stocks) * 0.8 + np.random.rand(n_stocks, n_stocks) * 0.2
corr_matrix = (corr_matrix + corr_matrix.T) / 2
np.fill_diagonal(corr_matrix, 1.0)
std_devs = np.full(n_stocks, 0.2)
cov_matrix = np.outer(std_devs, std_devs) * corr_matrix

print(f"Testing efficient frontier with {n_stocks} stocks")
print(f"Expected returns range: {expected_returns.min():.2%} to {expected_returns.max():.2%}")

# Generate efficient frontier
frontier_results = generate_efficient_frontier(
    returns=expected_returns,
    cov_matrix=cov_matrix,
    num_portfolios=100,
    risk_free_rate=0.02,
    allow_short=False
)

# Check results
n_portfolios = len(frontier_results['returns'])
print(f"\n✓ Generated {n_portfolios} efficient frontier portfolios")

if n_portfolios > 0:
    print(f"  Return range: {frontier_results['returns'].min():.2%} to {frontier_results['returns'].max():.2%}")
    print(f"  Volatility range: {frontier_results['volatilities'].min():.2%} to {frontier_results['volatilities'].max():.2%}")
    print(f"  Sharpe ratio range: {frontier_results['sharpe_ratios'].min():.3f} to {frontier_results['sharpe_ratios'].max():.3f}")
    print("\n✅ TEST PASSED: Efficient frontier generated successfully!")
else:
    print("\n❌ TEST FAILED: No portfolios generated")
    exit(1)
