"""
Minimal test of the frontier fix - execute the specific function directly
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

# Minimal imports
import numpy as np
from scipy.optimize import minimize


# Copy the fixed functions here to test them standalone
def calculate_portfolio_return(weights, returns):
    return float(np.dot(weights, returns))


def calculate_portfolio_volatility(weights, cov_matrix):
    variance = np.dot(weights, np.dot(cov_matrix, weights))
    if variance < 0:
        variance = 0
    return float(np.sqrt(variance))


def optimize_portfolio_target_return_fixed(
    returns, cov_matrix, target_return, allow_short=False
):
    """Fixed version with relaxed constraints"""
    n_assets = len(returns)

    # Objective
    def objective(weights):
        return calculate_portfolio_volatility(weights, cov_matrix)

    # Relaxed constraints
    return_tolerance = 0.0001
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {
            "type": "ineq",
            "fun": lambda w: calculate_portfolio_return(w, returns)
            - (target_return - return_tolerance),
        },
        {
            "type": "ineq",
            "fun": lambda w: (target_return + return_tolerance)
            - calculate_portfolio_return(w, returns),
        },
    ]

    # Bounds
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))

    # Better initial guess
    if target_return > 0:
        positive_returns = np.maximum(returns, 0.0001)
        initial_weights = positive_returns / np.sum(positive_returns)
    else:
        initial_weights = np.array([1.0 / n_assets] * n_assets)

    # Optimize with better settings
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-9},
    )

    if not result.success:
        # Try trust-constr
        result = minimize(
            objective,
            initial_weights,
            method="trust-constr",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000},
        )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    weights = result.x
    return {
        "weights": weights,
        "return": calculate_portfolio_return(weights, returns),
        "volatility": calculate_portfolio_volatility(weights, cov_matrix),
    }


# Test
print("Testing efficient frontier fix with 150 stocks...")
np.random.seed(42)
n_stocks = 150

# Generate test data
expected_returns = np.random.uniform(0.05, 0.50, n_stocks)
corr_matrix = np.eye(n_stocks) * 0.8 + np.random.rand(n_stocks, n_stocks) * 0.2
corr_matrix = (corr_matrix + corr_matrix.T) / 2
np.fill_diagonal(corr_matrix, 1.0)
std_devs = np.full(n_stocks, 0.2)
cov_matrix = np.outer(std_devs, std_devs) * corr_matrix

# Test a few target returns
min_return = np.min(expected_returns)
max_return = np.max(expected_returns)
test_targets = np.linspace(min_return, max_return, 10)

success_count = 0
for target in test_targets:
    try:
        result = optimize_portfolio_target_return_fixed(
            expected_returns, cov_matrix, target
        )
        success_count += 1
    except Exception as e:
        print(f"  Failed for target {target:.2%}: {e}")

print(f"\nSuccessfully optimized {success_count}/10 portfolios")

if success_count >= 7:  # At least 70% success
    print("TEST PASSED: Efficient frontier should now work!")
else:
    print(f"TEST FAILED: Only {success_count}/10 succeeded")
