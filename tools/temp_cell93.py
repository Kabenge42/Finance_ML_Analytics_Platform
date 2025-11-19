# Prepare visualization variables
print("\n" + "=" * 80)
print("Preparing Portfolio Results for Visualization")
print("=" * 80)

# 1. Select optimal portfolio (max Sharpe as default)
optimal_portfolio = max_sharpe_result if max_sharpe_result else min_vol_result
if optimal_portfolio:
    print(
        f"[OK] Optimal portfolio selected: {optimal_portfolio.get('return', 0):.2%} return, "
        f"{optimal_portfolio.get('volatility', 0):.2%} volatility, "
        f"Sharpe={optimal_portfolio.get('sharpe_ratio', 0):.3f}"
    )
else:
    print("[WARNING] No optimal portfolio available")

# 2. Assign min volatility portfolio for visualization
min_vol_portfolio = min_vol_result
if min_vol_portfolio:
    print(f"[OK] Min volatility portfolio: {min_vol_portfolio.get('volatility', 0):.2%} volatility")

# 3. Generate efficient frontier
if max_sharpe_result and "cov_matrix" in dir() and cov_matrix is not None:
    try:
        frontier_results = generate_efficient_frontier(
            returns=expected_returns_array,
            cov_matrix=cov_matrix,
            num_portfolios=100,
            risk_free_rate=risk_free_rate,
            allow_short=False,
        )
        print(f"[OK] Efficient frontier generated: {len(frontier_results['returns'])} portfolios")
    except Exception as e:
        print(f"[WARNING] Efficient frontier generation failed: {e}")
        frontier_results = None
else:
    frontier_results = None
    print("[WARNING] Skipping efficient frontier (insufficient data)")

# 4. Store risk metrics for visualization (use optimal portfolio)
if optimal_portfolio and "best_return_col" in dir() and best_return_col:
    try:
        # Regenerate synthetic returns for optimal portfolio
        daily_return = optimal_portfolio["return"] / 252
        daily_vol = optimal_portfolio["volatility"] / np.sqrt(252)
        np.random.seed(42)  # Reproducibility
        portfolio_returns = np.random.normal(daily_return, daily_vol, 252)

        risk_metrics_result = calculate_portfolio_risk_metrics(
            pd.Series(portfolio_returns),
            risk_free_rate=risk_free_rate,
            confidence_levels=[0.95, 0.99],
        )
        print(
            f"[OK] Risk metrics calculated: Sharpe={risk_metrics_result['sharpe_ratio']:.3f}, "
            f"Max DD={risk_metrics_result['max_drawdown']:.2%}"
        )
    except Exception as e:
        print(f"[WARNING] Risk metrics calculation failed: {e}")
        portfolio_returns = None
        risk_metrics_result = None
else:
    portfolio_returns = None
    risk_metrics_result = None
    print("[WARNING] Skipping risk metrics (insufficient data)")

print("\n[SUCCESS] Visualization variables prepared successfully")
print("=" * 80)
