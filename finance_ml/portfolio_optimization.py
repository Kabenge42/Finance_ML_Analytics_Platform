"""
Portfolio Optimization Module

Modern Portfolio Theory (MPT) implementation including:
- Portfolio metrics calculation (return, volatility, Sharpe ratio)
- Efficient frontier generation
- Portfolio optimization (max Sharpe, min volatility, target return)
- Weight constraints and validation
- Portfolio rebalancing utilities

Based on Harry Markowitz's Modern Portfolio Theory.
"""

import logging
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def calculate_portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    """
    Calculate expected portfolio return.

    Args:
        weights: Portfolio weights (must sum to 1)
        returns: Expected returns for each asset

    Returns:
        Expected portfolio return

    Example:
        >>> weights = np.array([0.4, 0.6])
        >>> returns = np.array([0.10, 0.12])
        >>> calculate_portfolio_return(weights, returns)
        0.112
    """
    return float(np.dot(weights, returns))


def calculate_portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio volatility (standard deviation).

    Args:
        weights: Portfolio weights (must sum to 1)
        cov_matrix: Covariance matrix of asset returns

    Returns:
        Portfolio volatility (standard deviation)

    Raises:
        ValueError: If covariance matrix has negative diagonal elements

    Example:
        >>> weights = np.array([0.5, 0.5])
        >>> cov_matrix = np.array([[0.04, 0.01], [0.01, 0.09]])
        >>> calculate_portfolio_volatility(weights, cov_matrix)
        0.1581...
    """
    # Check for negative variances
    if np.any(np.diag(cov_matrix) < 0):
        raise ValueError(
            "Covariance matrix has negative variance (diagonal elements must be non-negative)"
        )

    variance = np.dot(weights, np.dot(cov_matrix, weights))

    # Handle numerical issues
    if variance < 0:
        variance = 0

    return float(np.sqrt(variance))


def calculate_portfolio_sharpe_ratio(
    portfolio_return: float, portfolio_volatility: float, risk_free_rate: float = 0.02
) -> float:
    """
    Calculate portfolio Sharpe ratio.

    Args:
        portfolio_return: Expected portfolio return
        portfolio_volatility: Portfolio volatility (standard deviation)
        risk_free_rate: Risk-free rate (default: 0.02)

    Returns:
        Sharpe ratio (excess return per unit of risk)

    Raises:
        ValueError: If volatility is zero

    Example:
        >>> calculate_portfolio_sharpe_ratio(0.12, 0.15, 0.02)
        0.6666...
    """
    if portfolio_volatility == 0:
        raise ValueError("Portfolio volatility cannot be zero when calculating Sharpe ratio")

    return (portfolio_return - risk_free_rate) / portfolio_volatility


def validate_weights(
    weights: np.ndarray, tolerance: float = 1e-6, allow_short: bool = False
) -> Tuple[bool, str]:
    """
    Validate portfolio weights.

    Args:
        weights: Portfolio weights
        tolerance: Numerical tolerance for sum constraint (default: 1e-6)
        allow_short: Whether to allow short positions (negative weights)

    Returns:
        Tuple of (is_valid, message)

    Example:
        >>> validate_weights(np.array([0.3, 0.3, 0.4]))
        (True, 'Valid')
        >>> validate_weights(np.array([0.3, 0.3, 0.3]))
        (False, 'Weights must sum to 1.0 (sum=0.9)')
    """
    if len(weights) == 0:
        return False, "Weights array is empty"

    weights_sum = np.sum(weights)
    if not np.isclose(weights_sum, 1.0, atol=tolerance):
        return False, f"Weights must sum to 1.0 (sum={weights_sum:.6f})"

    if not allow_short and np.any(weights < 0):
        return False, "Negative weights not allowed (short selling disabled)"

    return True, "Valid"


def generate_efficient_frontier(
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    num_portfolios: int = 100,
    risk_free_rate: float = 0.02,
    allow_short: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate efficient frontier portfolios.

    Args:
        returns: Expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
        num_portfolios: Number of portfolios to generate
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        allow_short: Whether to allow short positions

    Returns:
        Dictionary with keys:
        - 'returns': Portfolio returns
        - 'volatilities': Portfolio volatilities
        - 'sharpe_ratios': Portfolio Sharpe ratios
        - 'weights': Portfolio weights (num_portfolios x num_assets)

    Example:
        >>> returns = np.array([0.08, 0.10, 0.12])
        >>> cov_matrix = np.eye(3) * 0.04
        >>> result = generate_efficient_frontier(returns, cov_matrix, num_portfolios=10)
        >>> result['returns'].shape
        (10,)
    """
    n_assets = len(returns)

    # Generate target returns from min to max
    min_return = np.min(returns)
    max_return = np.max(returns)
    target_returns = np.linspace(min_return, max_return, num_portfolios)

    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_sharpe_ratios = []
    portfolio_weights = []

    for target_return in target_returns:
        try:
            result = optimize_portfolio_target_return(
                returns, cov_matrix, target_return, allow_short=allow_short
            )

            portfolio_returns.append(result["return"])
            portfolio_volatilities.append(result["volatility"])

            # Calculate Sharpe ratio
            sharpe = calculate_portfolio_sharpe_ratio(
                result["return"], result["volatility"], risk_free_rate
            )
            portfolio_sharpe_ratios.append(sharpe)
            portfolio_weights.append(result["weights"])

        except (ValueError, RuntimeError):
            # Skip infeasible portfolios
            continue

    return {
        "returns": np.array(portfolio_returns),
        "volatilities": np.array(portfolio_volatilities),
        "sharpe_ratios": np.array(portfolio_sharpe_ratios),
        "weights": np.array(portfolio_weights),
    }


def optimize_portfolio_max_sharpe(
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
) -> Dict[str, any]:
    """
    Optimize portfolio for maximum Sharpe ratio.

    Args:
        returns: Expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
        risk_free_rate: Risk-free rate
        allow_short: Whether to allow short positions
        max_weight: Maximum weight per asset (optional)

    Returns:
        Dictionary with keys:
        - 'weights': Optimal portfolio weights
        - 'return': Expected portfolio return
        - 'volatility': Portfolio volatility
        - 'sharpe_ratio': Portfolio Sharpe ratio

    Example:
        >>> returns = np.array([0.08, 0.10, 0.12])
        >>> cov_matrix = np.eye(3) * 0.04
        >>> result = optimize_portfolio_max_sharpe(returns, cov_matrix)
        >>> result['sharpe_ratio'] > 0
        True
    """
    n_assets = len(returns)

    # Validate dimensions
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError(
            f"Covariance matrix shape {cov_matrix.shape} doesn't match returns length {n_assets}"
        )

    # Objective: minimize negative Sharpe ratio
    def objective(weights):
        """Objective function: minimize negative Sharpe ratio.

        Args:
            weights: Portfolio weights

        Returns:
            Negative Sharpe ratio (for minimization)
        """
        port_return = calculate_portfolio_return(weights, returns)
        port_volatility = calculate_portfolio_volatility(weights, cov_matrix)

        if port_volatility == 0:
            return np.inf

        sharpe = (port_return - risk_free_rate) / port_volatility
        return -sharpe  # Minimize negative Sharpe = maximize Sharpe

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds
    if allow_short:
        bounds = tuple((-1.0, 2.0) for _ in range(n_assets))
    else:
        bounds = tuple((0.0, max_weight if max_weight else 1.0) for _ in range(n_assets))

    # Initial guess: equal weights
    initial_weights = np.array([1.0 / n_assets] * n_assets)

    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    if not result.success:
        logger.warning(f"Optimization did not converge: {result.message}")

    weights = result.x
    port_return = calculate_portfolio_return(weights, returns)
    port_volatility = calculate_portfolio_volatility(weights, cov_matrix)
    sharpe = calculate_portfolio_sharpe_ratio(port_return, port_volatility, risk_free_rate)

    return {
        "weights": weights,
        "return": port_return,
        "volatility": port_volatility,
        "sharpe_ratio": sharpe,
    }


def optimize_portfolio_min_volatility(
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
) -> Dict[str, any]:
    """
    Optimize portfolio for minimum volatility.

    Args:
        returns: Expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
        allow_short: Whether to allow short positions
        max_weight: Maximum weight per asset (optional)

    Returns:
        Dictionary with keys:
        - 'weights': Optimal portfolio weights
        - 'return': Expected portfolio return
        - 'volatility': Portfolio volatility

    Example:
        >>> returns = np.array([0.08, 0.10, 0.12])
        >>> cov_matrix = np.eye(3) * 0.04
        >>> result = optimize_portfolio_min_volatility(returns, cov_matrix)
        >>> result['volatility'] > 0
        True
    """
    n_assets = len(returns)

    # Validate dimensions
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError(
            f"Covariance matrix shape {cov_matrix.shape} doesn't match returns length {n_assets}"
        )

    # Objective: minimize volatility
    def objective(weights):
        """Objective function: minimize portfolio volatility.

        Args:
            weights: Portfolio weights

        Returns:
            Portfolio volatility
        """
        return calculate_portfolio_volatility(weights, cov_matrix)

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds
    if allow_short:
        bounds = tuple((-1.0, 2.0) for _ in range(n_assets))
    else:
        bounds = tuple((0.0, max_weight if max_weight else 1.0) for _ in range(n_assets))

    # Initial guess: equal weights
    initial_weights = np.array([1.0 / n_assets] * n_assets)

    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    if not result.success:
        logger.warning(f"Optimization did not converge: {result.message}")

    weights = result.x
    port_return = calculate_portfolio_return(weights, returns)
    port_volatility = calculate_portfolio_volatility(weights, cov_matrix)

    return {"weights": weights, "return": port_return, "volatility": port_volatility}


def optimize_portfolio_target_return(
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_return: float,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
) -> Dict[str, any]:
    """
    Optimize portfolio for minimum volatility given target return.

    Args:
        returns: Expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
        target_return: Target expected return
        allow_short: Whether to allow short positions
        max_weight: Maximum weight per asset (optional)

    Returns:
        Dictionary with keys:
        - 'weights': Optimal portfolio weights
        - 'return': Expected portfolio return (should match target)
        - 'volatility': Portfolio volatility

    Raises:
        ValueError: If target return is infeasible

    Example:
        >>> returns = np.array([0.08, 0.10, 0.12])
        >>> cov_matrix = np.eye(3) * 0.04
        >>> result = optimize_portfolio_target_return(returns, cov_matrix, 0.10)
        >>> abs(result['return'] - 0.10) < 0.01
        True
    """
    n_assets = len(returns)

    # Validate dimensions
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError(
            f"Covariance matrix shape {cov_matrix.shape} doesn't match returns length {n_assets}"
        )

    # Check if target return is feasible
    min_possible = np.min(returns)
    max_possible = np.max(returns)

    if target_return < min_possible or target_return > max_possible:
        raise ValueError(
            f"Target return {target_return:.4f} is outside feasible range "
            f"[{min_possible:.4f}, {max_possible:.4f}]"
        )

    # Objective: minimize volatility
    def objective(weights):
        """Objective function: minimize portfolio volatility subject to target return.

        Args:
            weights: Portfolio weights

        Returns:
            Portfolio volatility
        """
        return calculate_portfolio_volatility(weights, cov_matrix)

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: calculate_portfolio_return(w, returns) - target_return},
    ]

    # Bounds
    if allow_short:
        bounds = tuple((-1.0, 2.0) for _ in range(n_assets))
    else:
        bounds = tuple((0.0, max_weight if max_weight else 1.0) for _ in range(n_assets))

    # Initial guess: equal weights
    initial_weights = np.array([1.0 / n_assets] * n_assets)

    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    weights = result.x
    port_return = calculate_portfolio_return(weights, returns)
    port_volatility = calculate_portfolio_volatility(weights, cov_matrix)

    return {"weights": weights, "return": port_return, "volatility": port_volatility}


def rebalance_portfolio(
    current_weights: np.ndarray, target_weights: np.ndarray, trade_threshold: float = 0.0
) -> Dict[str, any]:
    """
    Calculate trades needed to rebalance portfolio.

    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        trade_threshold: Minimum trade size (skip trades below this threshold)

    Returns:
        Dictionary with keys:
        - 'trades': Trade amounts (positive = buy, negative = sell)
        - 'total_turnover': Total turnover (sum of absolute trades / 2)

    Example:
        >>> current = np.array([0.5, 0.3, 0.2])
        >>> target = np.array([0.4, 0.3, 0.3])
        >>> result = rebalance_portfolio(current, target)
        >>> result['trades'][0]  # Should sell asset 0
        -0.1
    """
    if len(current_weights) != len(target_weights):
        raise ValueError("Current and target weights must have same length")

    # Calculate raw trades
    raw_trades = target_weights - current_weights

    # Apply threshold
    trades = np.where(np.abs(raw_trades) >= trade_threshold, raw_trades, 0.0)

    # Calculate turnover (sum of absolute trades / 2)
    # Divide by 2 because each trade involves both buying and selling
    total_turnover = np.sum(np.abs(trades)) / 2

    return {"trades": trades, "total_turnover": float(total_turnover)}
