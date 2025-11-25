"""Validation script to test portfolio_reporting visualizations."""
import numpy as np
import pandas as pd
from pathlib import Path
from finance_ml.ml_workflow.analytics.portfolio_reporting import (
    universe_summary,
    returns_risk_diagnostics,
    frontier_and_constraints,
    risk_decomposition_dashboard,
    backtest_and_attribution,
)

# Create output directory
out_dir = Path("outputs/portfolio")
out_dir.mkdir(parents=True, exist_ok=True)

print("Generating portfolio visualization artifacts...")

# Test data
n = 50  # 50 assets to test dynamic pie chart sizing

# 1. Universe summary with sector data
df_universe = pd.DataFrame({
    'ticker': [f'TICK{i}' for i in range(n)],
    'sector': np.random.choice(['Tech', 'Healthcare', 'Finance', 'Energy', 'Consumer'], n),
    'region': np.random.choice(['US', 'EU', 'Asia'], n),
    'market_cap': np.random.lognormal(10, 2, n)
})
df_universe = df_universe.set_index('ticker')

print("1. Generating universe_summary...")
universe_summary(df_universe, out_dir)

# 2. Returns and risk diagnostics
mu = pd.Series(np.random.normal(0.1, 0.3, n), index=df_universe.index)
cov = np.random.rand(n, n)
cov = (cov + cov.T) / 2 + np.eye(n) * 0.1  # Make symmetric positive definite

print("2. Generating returns_risk_diagnostics...")
returns_risk_diagnostics(mu, cov, out_dir)

# 3. Efficient frontier
constraints = {"max_weight": [0.1, 0.15, 0.2, 0.25, 0.3]}

print("3. Generating frontier_and_constraints...")
frontier_and_constraints(mu, cov, constraints, out_dir)

# 4. Risk decomposition with actual weights
weights = pd.Series(np.random.dirichlet(np.ones(n)), index=df_universe.index)
exposures = df_universe.copy()

print("4. Generating risk_decomposition_dashboard...")
risk_decomposition_dashboard(weights, exposures, out_dir)

# 5. Backtest and attribution
dates = pd.date_range('2023-01-01', periods=100, freq='D')
prices = pd.DataFrame(
    np.random.lognormal(0, 0.02, (100, n)),
    index=dates,
    columns=df_universe.index
).cumprod()

print("5. Generating backtest_and_attribution...")
backtest_and_attribution(prices, weights, out_dir)

print("\nValidation complete!")
print(f"Artifacts generated in: {out_dir.absolute()}")
print("\nGenerated files:")
for file in sorted(out_dir.glob("*")):
    size = file.stat().st_size
    print(f"  {file.name:50s} {size:>10,} bytes")
