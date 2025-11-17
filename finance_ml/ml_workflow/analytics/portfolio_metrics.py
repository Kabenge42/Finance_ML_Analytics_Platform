"""Portfolio ranking metrics computation utilities.

This module implements Phase 1 metric computation from
docs/improvement_plan/portfolio_optimization_enhancement_plan.md:

- compute_return_1y: Calculate realized 1-year returns from historical prices
- compute_expected_return: Calculate forward-looking returns from predictions
- ensure_portfolio_metrics: Wrapper that ensures all required metrics exist

These functions are designed to be called BEFORE select_portfolio_candidates()
to ensure the required columns (expected_return, return_1y, mispricing_score)
are present in the DataFrame.

Developed using strict TDD methodology per code_guidelines.md v1.2.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_return_1y(
    df: pd.DataFrame,
    price_col: str = "last_price",
    price_1y_ago_col: str = "price_1y_ago",
    output_col: str = "return_1y",
    overwrite: bool = True,
) -> pd.DataFrame:
    """Compute realized 1-year return from historical price data.

    Parameters
    ----------
    df:
        Input DataFrame containing price columns.
    price_col:
        Column name for current price (default: "last_price").
    price_1y_ago_col:
        Column name for price 1 year ago (default: "price_1y_ago").
    output_col:
        Column name for output (default: "return_1y").
    overwrite:
        If False and output_col exists, preserve existing values.

    Returns
    -------
    DataFrame with return_1y column added (returns a copy).

    Notes
    -----
    - If source columns are missing, sets return_1y to 0.0 (fallback).
    - Formula: (current_price / price_1y_ago) - 1.0
    - NaN values in source columns result in NaN or 0.0 in output.
    """
    result = df.copy()

    # If column exists and overwrite=False, preserve existing values
    if output_col in result.columns and not overwrite:
        logger.debug(f"{output_col} already exists and overwrite=False; preserving")
        return result

    # Check if source columns exist
    if price_col not in result.columns or price_1y_ago_col not in result.columns:
        logger.warning(
            f"Missing columns for return_1y computation: "
            f"{price_col}={price_col in result.columns}, "
            f"{price_1y_ago_col}={price_1y_ago_col in result.columns}; "
            f"setting {output_col} to 0.0"
        )
        result[output_col] = 0.0
        return result

    # Compute return: (current / historical) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = (result[price_col] / result[price_1y_ago_col]) - 1.0

    # Replace inf/-inf/NaN with 0.0 for robustness
    result[output_col] = returns.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    logger.debug(
        f"Computed {output_col}: "
        f"range [{result[output_col].min():.3f}, {result[output_col].max():.3f}], "
        f"mean={result[output_col].mean():.3f}"
    )

    return result


def compute_expected_return(
    df: pd.DataFrame,
    price_col: str = "last_price",
    predicted_col: str = "predicted_price_target",
    mispricing_col: str = "mispricing_score",
    output_col: str = "expected_return",
    overwrite: bool = True,
) -> pd.DataFrame:
    """Compute expected return from model predictions or mispricing score.

    Parameters
    ----------
    df:
        Input DataFrame containing prediction or mispricing columns.
    price_col:
        Column name for current price (default: "last_price").
    predicted_col:
        Column name for predicted price target (default: "predicted_price_target").
    mispricing_col:
        Column name for mispricing score fallback (default: "mispricing_score").
    output_col:
        Column name for output (default: "expected_return").
    overwrite:
        If False and output_col exists, preserve existing values.

    Returns
    -------
    DataFrame with expected_return column added (returns a copy).

    Notes
    -----
    - Primary: Uses (predicted_price_target / last_price) - 1.0
    - Fallback 1: If predicted_col missing, uses mispricing_score directly
    - Fallback 2: If both missing, sets expected_return to 0.0
    - NaN values result in 0.0 for robustness.
    """
    result = df.copy()

    # If column exists and overwrite=False, preserve existing values
    if output_col in result.columns and not overwrite:
        logger.debug(f"{output_col} already exists and overwrite=False; preserving")
        return result

    # Strategy 1: Compute from predicted price target
    if {price_col, predicted_col}.issubset(result.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = (result[predicted_col] / result[price_col]) - 1.0
        result[output_col] = expected.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        logger.debug(f"Computed {output_col} from {predicted_col}")

    # Strategy 2: Fallback to mispricing_score if available
    elif mispricing_col in result.columns:
        result[output_col] = result[mispricing_col].fillna(0.0)
        logger.info(f"Using {mispricing_col} as {output_col} (fallback)")

    # Strategy 3: No data available, use 0.0
    else:
        logger.warning(
            f"No prediction or mispricing columns available; " f"setting {output_col} to 0.0"
        )
        result[output_col] = 0.0

    logger.debug(
        f"{output_col}: "
        f"range [{result[output_col].min():.3f}, {result[output_col].max():.3f}], "
        f"mean={result[output_col].mean():.3f}"
    )

    return result


def ensure_portfolio_metrics(
    df: pd.DataFrame,
    required_metrics: Optional[list] = None,
) -> pd.DataFrame:
    """Ensure all required portfolio ranking metrics are present.

    This is the main entry point used by notebooks. It computes any missing
    metrics from available source columns, using sensible fallbacks when
    data is unavailable.

    Parameters
    ----------
    df:
        Input DataFrame (typically output from Phase 9.5 analytics).
    required_metrics:
        List of required metric columns. Defaults to:
        ["expected_return", "return_1y", "mispricing_score"]

    Returns
    -------
    DataFrame with all required metrics (returns a copy).

    Notes
    -----
    This function is idempotent: calling it multiple times produces the same
    result. It does not modify the input DataFrame.

    Required for select_portfolio_candidates() from stock_selection.py.
    """
    if required_metrics is None:
        required_metrics = ["expected_return", "return_1y", "mispricing_score"]

    # Handle empty DataFrame
    if df.empty:
        logger.warning("Empty DataFrame passed to ensure_portfolio_metrics")
        result = df.copy()
        for metric in required_metrics:
            if metric not in result.columns:
                result[metric] = pd.Series(dtype=float)
        return result

    result = df.copy()

    # Compute return_1y if missing
    if "return_1y" in required_metrics and "return_1y" not in result.columns:
        result = compute_return_1y(result, overwrite=False)

    # Compute expected_return if missing
    if "expected_return" in required_metrics and "expected_return" not in result.columns:
        result = compute_expected_return(result, overwrite=False)

    # Ensure mispricing_score exists (may already be from Phase 9.5)
    if "mispricing_score" in required_metrics and "mispricing_score" not in result.columns:
        # Try fallback strategies
        if "mispricing_pct" in result.columns:
            result["mispricing_score"] = result["mispricing_pct"] / 100.0
            logger.debug("Computed mispricing_score from mispricing_pct")
        elif "expected_return" in result.columns:
            result["mispricing_score"] = result["expected_return"]
            logger.info("Using expected_return as mispricing_score (fallback)")
        else:
            result["mispricing_score"] = 0.0
            logger.warning("Set mispricing_score to 0.0 (no source data)")

    # Verify all required metrics are present
    missing = [m for m in required_metrics if m not in result.columns]
    if missing:
        logger.error(f"Failed to compute required metrics: {missing}")
        raise ValueError(f"Could not compute required metrics: {missing}")

    logger.info(f"✓ All required portfolio metrics present: " f"{', '.join(required_metrics)}")

    return result
