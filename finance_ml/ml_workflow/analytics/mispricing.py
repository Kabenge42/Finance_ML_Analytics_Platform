"""
Stock mispricing analysis and ranking.

This module provides functions to calculate mispricing scores (difference between
predicted and current prices) and rank stocks by valuation opportunity.

Phase 9.7 - Analytics Refactor
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_mispricing_score(
    df: pd.DataFrame, predicted_col: str = "predicted_price_target", current_col: str = "last_price"
) -> pd.DataFrame:
    """Calculate mispricing score for each stock.

    Formula: (predicted_price_target - last_price) / last_price

    Positive score = undervalued (predicted > current)
    Negative score = overvalued (predicted < current)

    Args:
        df: DataFrame with stock data
        predicted_col: Name of predicted price column (default: "predicted_price_target")
        current_col: Name of current price column (default: "last_price")

    Returns:
        DataFrame with added 'mispricing_pct' and 'mispricing_score' columns

    Raises:
        ValueError: If required columns are missing

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['AAPL', 'MSFT'],
        ...     'last_price': [150.0, 300.0],
        ...     'predicted_price_target': [180.0, 270.0]
        ... })
        >>> result = calculate_mispricing_score(df)
        >>> result['mispricing_pct'].iloc[0]  # AAPL: 20% undervalued
        20.0
    """
    required_columns = [predicted_col, current_col]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    result_df = df.copy()
    mispricing = (df[predicted_col] - df[current_col]) / df[current_col]
    result_df["mispricing_pct"] = mispricing * 100
    result_df["mispricing_score"] = (
        mispricing  # Alias for backward compatibility with rank functions
    )
    return result_df


def calculate_mispricing_from_predictions_schema(
    df: pd.DataFrame,
    prediction_col: str = "y_pred",
    calibrated_col: str = "y_pred_calibrated",
    current_price_col: str = "last_price",
    use_calibrated: bool = False,
) -> pd.DataFrame:
    """Calculate mispricing using standardized predictions schema.

    This helper is Phase 9.3–aware and is designed for DataFrames that
    follow the standardized predictions schema (y_true, y_pred,
    y_pred_calibrated, pred_p10/50/90, last_price, etc.).

    It delegates to :func:`calculate_mispricing_score` while selecting the
    appropriate prediction column.

    Args:
        df: Input dataframe with standardized prediction columns.
        prediction_col: Name of base prediction column (default ``"y_pred"``).
        calibrated_col: Name of calibrated prediction column
            (default ``"y_pred_calibrated"``).
        current_price_col: Name of current price column
            (default ``"last_price"``).
        use_calibrated: If True and ``calibrated_col`` exists, use it
            instead of ``prediction_col``.

    Returns:
        DataFrame with added ``mispricing_pct`` and ``mispricing_score``
        columns, preserving all original columns.
    """

    if use_calibrated and calibrated_col in df.columns:
        predicted_col = calibrated_col
    else:
        predicted_col = prediction_col

    return calculate_mispricing_score(
        df, predicted_col=predicted_col, current_col=current_price_col
    )


def calculate_risk_adjusted_mispricing(
    df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    use_confidence_interval: bool = False,
    default_volatility: float = 0.20,
) -> pd.Series:
    """Calculate risk-adjusted mispricing score.

    Formula: (Expected_Return - Risk_Free_Rate) / Volatility

    This adjusts the mispricing score by the stock's volatility to account for risk.
    Higher risk-adjusted scores indicate better risk-reward opportunities.

    Args:
        df: DataFrame with 'predicted_price_target', 'last_price', and optionally 'volatility' columns
        risk_free_rate: Risk-free rate to subtract from expected return (default 0.0)
        use_confidence_interval: If True and confidence intervals available, adjust for uncertainty
        default_volatility: Default volatility to use if column missing (default 0.20)

    Returns:
        Series with risk-adjusted mispricing scores

    Example:
        >>> df = pd.DataFrame({
        ...     'predicted_price_target': [120, 90],
        ...     'last_price': [100, 100],
        ...     'volatility': [0.20, 0.30]
        ... })
        >>> scores = calculate_risk_adjusted_mispricing(df, risk_free_rate=0.05)
        >>> scores.iloc[0] > 0  # Undervalued with positive risk-adjusted return
        True
    """
    # Calculate expected return
    expected_return = (df["predicted_price_target"] - df["last_price"]) / df["last_price"]

    # Use volatility column if available, otherwise use default
    if "volatility" in df.columns:
        volatility = df["volatility"].copy()
    else:
        logger.warning(f"Volatility column not found; using default {default_volatility}")
        volatility = pd.Series(default_volatility, index=df.index)

    # Replace zero or negative volatility with a small value to avoid division by zero
    volatility = volatility.clip(lower=0.01)

    # Adjust for confidence interval width if requested
    if (
        use_confidence_interval
        and "confidence_lower" in df.columns
        and "confidence_upper" in df.columns
    ):
        # Wider confidence intervals indicate more uncertainty
        ci_width = (df["confidence_upper"] - df["confidence_lower"]) / df["last_price"]
        # Penalize by confidence interval width (wider = more uncertain = lower score)
        uncertainty_penalty = 1.0 / (1.0 + ci_width)
        risk_adjusted = ((expected_return - risk_free_rate) / volatility) * uncertainty_penalty
    else:
        # Standard risk-adjusted calculation
        risk_adjusted = (expected_return - risk_free_rate) / volatility

    return risk_adjusted


def calculate_risk_adjusted_mispricing_from_predictions_schema(
    df: pd.DataFrame,
    prediction_col: str = "y_pred",
    current_price_col: str = "last_price",
    risk_free_rate: float = 0.0,
    use_quantile_interval: bool = False,
    default_volatility: float = 0.20,
    pred_p10_col: str = "pred_p10",
    pred_p90_col: str = "pred_p90",
    volatility_col: str = "volatility",
) -> pd.Series:
    """Risk-adjusted mispricing for standardized predictions schema.

    This helper wires the standardized prediction columns into
    :func:`calculate_risk_adjusted_mispricing` while optionally
    using the prediction interval width as a volatility proxy when
    ``use_quantile_interval`` is True and explicit volatility is
    not available.

    Args:
        df: DataFrame with standardized prediction columns.
        prediction_col: Column containing point predictions (default
            ``"y_pred"``).
        current_price_col: Column containing current prices (default
            ``"last_price"``).
        risk_free_rate: Risk-free rate to subtract from expected return.
        use_quantile_interval: If True and ``pred_p10_col`` / ``pred_p90_col``
            are present, use the interval width as a volatility proxy when
            an explicit ``volatility_col`` is missing.
        default_volatility: Fallback volatility when no other information
            is available.
        pred_p10_col: Lower prediction quantile column name.
        pred_p90_col: Upper prediction quantile column name.
        volatility_col: Column name for explicit volatility, if present.

    Returns:
        Series with risk-adjusted mispricing scores aligned to ``df.index``.
    """

    tmp = pd.DataFrame(index=df.index)
    tmp["predicted_price_target"] = df[prediction_col]
    tmp["last_price"] = df[current_price_col]

    # Determine volatility: explicit column, quantile-based proxy, or default.
    if volatility_col in df.columns:
        tmp["volatility"] = df[volatility_col]
        use_ci = False
    elif use_quantile_interval and pred_p10_col in df.columns and pred_p90_col in df.columns:
        # Use prediction interval width as a volatility proxy.
        width = (df[pred_p90_col] - df[pred_p10_col]).abs()
        # Normalize by current price to keep scale comparable and avoid zeros.
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_proxy = width / df[current_price_col].replace(0, np.nan)
        vol_proxy = vol_proxy.fillna(default_volatility)
        tmp["volatility"] = vol_proxy
        use_ci = False
    else:
        tmp["volatility"] = default_volatility
        use_ci = False

    return calculate_risk_adjusted_mispricing(
        tmp,
        risk_free_rate=risk_free_rate,
        use_confidence_interval=use_ci,
        default_volatility=default_volatility,
    )


def rank_undervalued_stocks(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank and return top N most undervalued stocks.

    Args:
        df: DataFrame with 'mispricing_score' column
        top_n: Number of top stocks to return

    Returns:
        DataFrame sorted by mispricing_score descending (most undervalued first)

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        ...     'mispricing_score': [0.20, -0.10, 0.15]
        ... })
        >>> top = rank_undervalued_stocks(df, top_n=2)
        >>> top['ticker'].iloc[0]
        'AAPL'
    """
    sorted_df = df.sort_values("mispricing_score", ascending=False)
    return sorted_df.head(top_n)


def rank_overvalued_stocks(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank and return top N most overvalued stocks.

    Args:
        df: DataFrame with 'mispricing_score' column
        top_n: Number of top stocks to return

    Returns:
        DataFrame sorted by mispricing_score ascending (most overvalued first)

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        ...     'mispricing_score': [0.20, -0.10, 0.15]
        ... })
        >>> top = rank_overvalued_stocks(df, top_n=2)
        >>> top['ticker'].iloc[0]
        'MSFT'
    """
    sorted_df = df.sort_values("mispricing_score", ascending=True)
    return sorted_df.head(top_n)


def rank_stocks_by_sector(
    df: pd.DataFrame, top_n: int = 5, order: str = "undervalued"
) -> Dict[str, pd.DataFrame]:
    """Rank stocks within each sector.

    Args:
        df: DataFrame with 'sector' and 'mispricing_score' columns
        top_n: Number of top stocks per sector
        order: 'undervalued' (descending score) or 'overvalued' (ascending score)

    Returns:
        Dict with sector names as keys and ranked DataFrames as values

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['AAPL', 'MSFT', 'JPM', 'BAC'],
        ...     'sector': ['Tech', 'Tech', 'Finance', 'Finance'],
        ...     'mispricing_score': [0.20, 0.15, -0.05, -0.10]
        ... })
        >>> rankings = rank_stocks_by_sector(df, top_n=2, order='undervalued')
        >>> rankings['Tech']['ticker'].iloc[0]
        'AAPL'
    """
    result = {}
    ascending = order == "overvalued"

    for sector, group in df.groupby("sector"):
        sorted_group = group.sort_values("mispricing_score", ascending=ascending)
        result[sector] = sorted_group.head(top_n)

    return result
