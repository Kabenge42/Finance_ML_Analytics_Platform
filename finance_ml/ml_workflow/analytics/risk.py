"""
Risk Metrics Module for Finance ML Analytics Platform.

This module provides comprehensive risk metrics calculations for portfolio analysis:
- Value at Risk (VaR): Historical and Parametric methods
- Conditional Value at Risk (CVaR): Expected shortfall
- Sharpe Ratio: Risk-adjusted returns
- Sortino Ratio: Downside risk-adjusted returns
- Maximum Drawdown: Peak-to-trough decline

Implemented using strict TDD methodology (Test-Driven Development).
"""

from __future__ import annotations

from typing import Union, Dict, Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def calculate_var_historical(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using historical simulation method.

    VaR represents the maximum expected loss at a given confidence level.
    Historical VaR uses the empirical distribution of returns.

    Args:
        returns: Series of returns (e.g., daily returns)
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)

    Returns:
        VaR value (negative number representing loss)

    Raises:
        ValueError: If returns is empty or confidence_level is invalid

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
        >>> var_95 = calculate_var_historical(returns, confidence_level=0.95)
        >>> print(f"95% VaR: {var_95:.4f}")
    """
    if len(returns) == 0:
        raise ValueError("Returns series cannot be empty")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")

    # Calculate the percentile corresponding to the confidence level
    # For 95% confidence, we look at the 5th percentile (worst 5% of returns)
    alpha = 1 - confidence_level
    var = np.percentile(returns, alpha * 100)

    return float(var)


def calculate_var_parametric(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using parametric method (normal distribution).

    Assumes returns follow a normal distribution and uses mean and standard deviation.

    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)

    Returns:
        VaR value (negative number representing loss)

    Raises:
        ValueError: If returns is empty or confidence_level is invalid

    Example:
        >>> returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        >>> var_95 = calculate_var_parametric(returns, confidence_level=0.95)
    """
    if len(returns) == 0:
        raise ValueError("Returns series cannot be empty")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")

    # Calculate mean and standard deviation
    mean = returns.mean()
    std = returns.std()

    # Calculate z-score for the confidence level
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(alpha)

    # Parametric VaR
    var = mean + z_score * std

    return float(var)


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.

    CVaR represents the expected loss given that the loss exceeds VaR.
    It's the average of all losses beyond the VaR threshold.

    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)

    Returns:
        CVaR value (negative number representing expected loss beyond VaR)

    Raises:
        ValueError: If returns is empty or confidence_level is invalid

    Example:
        >>> returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        >>> cvar_95 = calculate_cvar(returns, confidence_level=0.95)
    """
    if len(returns) == 0:
        raise ValueError("Returns series cannot be empty")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")

    # First, calculate VaR
    var = calculate_var_historical(returns, confidence_level)

    # CVaR is the mean of all returns that are below VaR
    cvar = returns[returns <= var].mean()

    return float(cvar)


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio for risk-adjusted returns.

    Sharpe ratio measures excess return per unit of total risk (volatility).
    Higher Sharpe ratio indicates better risk-adjusted performance.

    Args:
        returns: Series of returns (e.g., daily returns)
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year for annualization (default: 252 for daily)

    Returns:
        Annualized Sharpe ratio

    Raises:
        ValueError: If volatility is zero (no variation in returns)

    Example:
        >>> returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        >>> sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    """
    if len(returns) == 0:
        raise ValueError("Returns series cannot be empty")

    # Calculate mean and standard deviation
    mean_return = returns.mean()
    std_return = returns.std()

    # Check for zero or near-zero volatility (with small epsilon for floating point precision)
    if std_return < 1e-10:
        raise ValueError("Standard deviation is zero; cannot calculate Sharpe ratio")

    # Convert annual risk-free rate to period rate
    rf_period = risk_free_rate / periods_per_year

    # Calculate Sharpe ratio and annualize
    sharpe = (mean_return - rf_period) / std_return
    sharpe_annualized = sharpe * np.sqrt(periods_per_year)

    return float(sharpe_annualized)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio for downside risk-adjusted returns.

    Similar to Sharpe ratio but only considers downside volatility (negative returns).
    Better reflects risk for investors who care more about downside than upside volatility.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        target_return: Target return threshold (default: 0.0)
        periods_per_year: Number of periods per year for annualization (default: 252)

    Returns:
        Annualized Sortino ratio

    Raises:
        ValueError: If downside deviation is zero

    Example:
        >>> returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        >>> sortino = calculate_sortino_ratio(returns, target_return=0.0)
    """
    if len(returns) == 0:
        raise ValueError("Returns series cannot be empty")

    # Calculate mean return
    mean_return = returns.mean()

    # Calculate downside deviation (only negative returns below target)
    downside_returns = returns[returns < target_return]
    downside_deviation = downside_returns.std()

    # Check for zero or near-zero downside deviation (with small epsilon for floating point precision)
    if downside_deviation < 1e-10 or len(downside_returns) == 0:
        raise ValueError("Downside deviation is zero; cannot calculate Sortino ratio")

    # Convert annual risk-free rate to period rate
    rf_period = risk_free_rate / periods_per_year

    # Calculate Sortino ratio and annualize
    sortino = (mean_return - rf_period) / downside_deviation
    sortino_annualized = sortino * np.sqrt(periods_per_year)

    return float(sortino_annualized)


def calculate_max_drawdown(
    prices: pd.Series, return_details: bool = False
) -> Union[float, Dict[str, Any]]:
    """
    Calculate maximum drawdown from peak to trough.

    Maximum drawdown represents the largest peak-to-trough decline in value.
    It's a key measure of downside risk.

    Args:
        prices: Series of prices or cumulative returns
        return_details: If True, return dict with detailed metrics

    Returns:
        Maximum drawdown as float (negative percentage), or dict with details

    Example:
        >>> prices = pd.Series([100, 120, 80, 90, 110])
        >>> max_dd = calculate_max_drawdown(prices)
        >>> print(f"Max Drawdown: {max_dd:.2%}")
    """
    if len(prices) == 0:
        raise ValueError("Prices series cannot be empty")

    # Calculate running maximum (peak)
    running_max = prices.expanding().max()

    # Calculate drawdown from peak
    drawdown = (prices - running_max) / running_max

    # Find maximum drawdown
    max_dd = drawdown.min()

    if return_details:
        # Find indices of peak and trough
        max_dd_idx = drawdown.idxmin()
        peak_idx = running_max[:max_dd_idx].idxmax() if max_dd_idx is not None else None

        return {
            "max_drawdown": float(max_dd),
            "peak_date": peak_idx,
            "trough_date": max_dd_idx,
            "peak_value": float(prices[peak_idx]) if peak_idx is not None else None,
            "trough_value": float(prices[max_dd_idx]) if max_dd_idx is not None else None,
        }

    return float(max_dd)


def calculate_portfolio_risk_metrics(
    returns: pd.Series,
    prices: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    confidence_levels: list = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio risk metrics.

    Combines multiple risk measures into a single comprehensive report.

    Args:
        returns: Series of returns
        prices: Optional series of prices for drawdown calculation
        risk_free_rate: Annual risk-free rate (default: 0.0)
        confidence_levels: List of confidence levels for VaR/CVaR (default: [0.95, 0.99])

    Returns:
        Dictionary containing all risk metrics

    Example:
        >>> returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        >>> prices = (1 + returns).cumprod() * 100
        >>> metrics = calculate_portfolio_risk_metrics(returns, prices)
        >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    metrics = {"mean_return": float(returns.mean()), "volatility": float(returns.std())}

    # Basic statistics

    # VaR metrics
    for conf in confidence_levels:
        conf_pct = int(conf * 100)
        metrics[f"var_{conf_pct}_historical"] = calculate_var_historical(returns, conf)
        metrics[f"var_{conf_pct}_parametric"] = calculate_var_parametric(returns, conf)
        metrics[f"cvar_{conf_pct}"] = calculate_cvar(returns, conf)

    # Risk-adjusted return metrics
    try:
        metrics["sharpe_ratio"] = calculate_sharpe_ratio(returns, risk_free_rate)
    except ValueError:
        metrics["sharpe_ratio"] = np.nan

    try:
        metrics["sortino_ratio"] = calculate_sortino_ratio(returns, risk_free_rate)
    except ValueError:
        metrics["sortino_ratio"] = np.nan

    # Drawdown metrics (if prices provided)
    if prices is not None:
        metrics["max_drawdown"] = calculate_max_drawdown(prices)
    else:
        # Calculate from cumulative returns
        cum_returns = (1 + returns).cumprod()
        metrics["max_drawdown"] = calculate_max_drawdown(cum_returns)

    return metrics


def calculate_expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Alias for CVaR/Expected Shortfall used in the enhancement plan.

    This thin wrapper exists mainly for semantic clarity in the
    portfolio optimisation roadmap: Expected Shortfall is implemented as
    historical CVaR via :func:`calculate_cvar`.
    """

    return calculate_cvar(returns, confidence_level=confidence)


def calculate_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualised tracking error vs a benchmark.

    Tracking error is defined as the standard deviation of the
    portfolio-minus-benchmark return series, annualised by
    ``sqrt(periods_per_year)``.
    """

    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        raise ValueError("Return series cannot be empty")

    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("portfolio_returns and benchmark_returns must have the same length")

    diff = portfolio_returns.astype(float) - benchmark_returns.astype(float)
    te = diff.std(ddof=1) * np.sqrt(periods_per_year)
    return float(te)


def run_stress_tests(
    weights: np.ndarray,
    returns: pd.DataFrame,
    scenarios: Dict[str, Dict[str, float]],
    asset_class_mapping: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Run simple portfolio stress tests for predefined scenarios.

    Each scenario specifies shocks at the **asset-class** level (for
    example ``{"equity": -0.30, "bonds": -0.10}``). Assets are mapped
    to classes via ``asset_class_mapping`` which must be the same length
    as the number of columns in ``returns``. If no mapping is provided,
    all assets are assumed to belong to the ``"equity"`` class.

    The function returns a dictionary keyed by scenario name containing
    at least the scenario ``portfolio_loss`` (a negative number
    indicates loss).
    """

    if returns.shape[1] != len(weights):
        raise ValueError("weights length must match number of assets in returns")

    n_assets = returns.shape[1]
    if asset_class_mapping is None:
        asset_class_mapping = ["equity"] * n_assets

    if len(asset_class_mapping) != n_assets:
        raise ValueError("asset_class_mapping length must match number of assets")

    results: Dict[str, Dict[str, float]] = {}

    for name, shocks in scenarios.items():
        loss = 0.0
        for w, asset_class in zip(weights, asset_class_mapping):
            shock = shocks.get(asset_class, 0.0)
            loss += w * shock

        results[name] = {"portfolio_loss": float(loss)}

    return results


def run_monte_carlo_simulation(
    weights: np.ndarray,
    returns: pd.DataFrame,
    n_simulations: int = 10_000,
    time_horizon: int = 252,
    confidence_levels: Sequence[float] | None = None,
    random_state: Optional[int] = 21,
) -> Dict[str, Any]:
    """Run a Monte Carlo simulation for portfolio value paths.

    The simulation uses a multivariate normal approximation calibrated on
    the provided ``returns`` DataFrame. Outputs include the full matrix
    of simulated paths and percentile summary paths suitable for
    visualisation or risk analysis.
    """

    if confidence_levels is None:
        confidence_levels = [0.05, 0.5, 0.95]

    if returns.shape[1] != len(weights):
        raise ValueError("weights length must match number of assets in returns")

    rng = np.random.RandomState(random_state)

    base = returns.to_numpy(dtype=float)
    mean_vec = base.mean(axis=0)
    cov_matrix = np.cov(base, rowvar=False)

    paths = np.zeros((n_simulations, time_horizon), dtype=float)

    for i in range(n_simulations):
        simulated = rng.multivariate_normal(mean_vec, cov_matrix, size=time_horizon)
        port_rets = simulated @ weights
        paths[i] = (1 + port_rets).cumprod()

    out: Dict[str, Any] = {"paths": paths}

    # Add percentile paths keyed by pXX_path for each confidence level
    for cl in confidence_levels:
        pct = int(round(cl * 100))
        key = f"p{pct:02d}_path"
        out[key] = np.percentile(paths, pct, axis=0)

    return out
