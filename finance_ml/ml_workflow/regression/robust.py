"""
Robust training and prediction safety helpers (Priority 2: Extreme Outliers)

Provides small utilities to reduce the impact of catastrophic outliers:
- winsorize_target: cap the target distribution tails at given percentiles
- clip_predictions: clip predictions to a reasonable range based on training target

These helpers are intentionally lightweight and dependency‑minimal to be used
in notebooks and scripts without heavy refactoring.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# Note: We intentionally use percentile-based clipping for deterministic behavior
_HAS_SCIPY = False  # kept for backward compat; percentile clipping used by default


def winsorize_target(y: Iterable[float], lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """
    Winsorize target values by capping the distribution tails at given percentiles.

    Args:
        y: Target values (array-like)
        lower: Lower tail proportion to cap (e.g., 0.01 for 1st percentile)
        upper: Upper percentile (0.99 for 99th percentile)

    Returns:
        Winsorized target as numpy array (float)
    """
    y_arr = np.asarray(y, dtype=float)
    if y_arr.size == 0:
        return y_arr

    if not (0 <= lower < 0.5) or not (0.5 < upper <= 1.0) or lower >= upper:
        raise ValueError("Invalid winsorization limits: ensure 0<=lower<0.5<upper<=1 and lower<upper")

    # Percentile-based clipping for consistent small-sample behavior
    lo = np.nanpercentile(y_arr, lower * 100.0)
    hi = np.nanpercentile(y_arr, upper * 100.0)
    return np.clip(y_arr, lo, hi)


def clip_predictions(preds: Iterable[float], y_train: Iterable[float], n_std: float = 3.0) -> np.ndarray:
    """
    Clip predictions to within mean ± n_std * std of training target distribution.

    Ensures predictions stay within a reasonable range (and non-negative lower bound).

    Args:
        preds: Model predictions (array-like)
        y_train: Training targets used to derive clipping bounds
        n_std: Number of standard deviations from the mean for bounds (default: 3)

    Returns:
        Clipped predictions as numpy array (float)
    """
    preds_arr = np.asarray(preds, dtype=float)
    y_arr = np.asarray(y_train, dtype=float)
    if preds_arr.size == 0:
        return preds_arr
    if y_arr.size == 0 or not np.isfinite(y_arr).any():
        # Nothing to learn bounds from; just clip at zero to avoid negatives
        return np.maximum(preds_arr, 0.0)

    mean = float(np.nanmean(y_arr))
    std = float(np.nanstd(y_arr))
    # Protect against zero std
    if not np.isfinite(std) or std == 0.0:
        lower = max(0.0, mean)
        upper = max(lower, mean)
    else:
        lower = max(0.0, mean - n_std * std)
        upper = mean + n_std * std

    return np.clip(preds_arr, lower, upper)


def adaptive_clip_predictions(
    preds: Iterable[float], y_train: Iterable[float], min_lower_bound: float = 0.10
) -> dict:
    """
    Clip predictions using percentile-based adaptive bounds to eliminate zero predictions.

    This function implements the fix for the zero predictions issue where 24.75% of
    predictions were being clipped to exactly $0.00, destroying low-value stock predictions.

    Strategy:
    - Lower bound: 0.5 × p0.5 (0.5th percentile), minimum $0.10
    - Upper bound: 1.5 × p99.5 (99.5th percentile)
    - Adaptive: bounds scale with training data distribution
    - Zero elimination: minimum threshold ensures no exact zeros

    Args:
        preds: Model predictions (array-like)
        y_train: Training targets used to derive clipping bounds
        min_lower_bound: Minimum lower bound to prevent exact zeros (default: $0.10)

    Returns:
        Dictionary containing:
        - clipped_predictions: Clipped predictions as numpy array (float)
        - lower_bound: Calculated lower bound value
        - upper_bound: Calculated upper bound value
        - n_clipped_lower: Count of predictions clipped to lower bound
        - n_clipped_upper: Count of predictions clipped to upper bound
        - pct_clipped_lower: Percentage of predictions clipped to lower bound
        - pct_clipped_upper: Percentage of predictions clipped to upper bound

    Example:
        >>> y_train = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
        >>> preds = np.array([-5.0, 1.0, 10.0, 100.0])
        >>> result = adaptive_clip_predictions(preds, y_train)
        >>> result['clipped_predictions']
        array([0.25, 1.0, 10.0, 75.0])  # No zeros!
        >>> result['n_clipped_lower']
        1

    References:
        - ZERO_PREDICTIONS_FIX.md: Issue resolution documentation
        - code_guidelines.md v1.2: Outlier Safety Rails Policy
    """
    preds_arr = np.asarray(preds, dtype=float)
    y_arr = np.asarray(y_train, dtype=float)

    # Handle empty predictions
    if preds_arr.size == 0:
        return {
            "clipped_predictions": preds_arr,
            "lower_bound": min_lower_bound,
            "upper_bound": min_lower_bound,
            "n_clipped_lower": 0,
            "n_clipped_upper": 0,
            "pct_clipped_lower": 0.0,
            "pct_clipped_upper": 0.0,
        }

    # Handle empty or invalid training data - use safe fallback
    if y_arr.size == 0 or not np.isfinite(y_arr).any():
        # Fallback: clip at minimum bound to avoid negatives/zeros
        clipped = np.maximum(preds_arr, min_lower_bound)
        n_clipped_low = int(np.sum(preds_arr < min_lower_bound))
        return {
            "clipped_predictions": clipped,
            "lower_bound": min_lower_bound,
            "upper_bound": float(np.max(clipped)) if clipped.size > 0 else min_lower_bound,
            "n_clipped_lower": n_clipped_low,
            "n_clipped_upper": 0,
            "pct_clipped_lower": 100.0 * n_clipped_low / len(preds_arr),
            "pct_clipped_upper": 0.0,
        }

    # Calculate percentile-based adaptive bounds
    train_p0_5 = float(np.nanpercentile(y_arr, 0.5))  # 0.5th percentile
    train_p99_5 = float(np.nanpercentile(y_arr, 99.5))  # 99.5th percentile

    # Lower bound: 0.5 × p0.5, with minimum threshold to prevent zeros
    lower_bound = max(min_lower_bound, train_p0_5 * 0.5)

    # Upper bound: 1.5 × p99.5 (allows extrapolation beyond training range)
    upper_bound = train_p99_5 * 1.5

    # Ensure upper >= lower (handle edge cases)
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        lower_bound = min_lower_bound
        upper_bound = max(min_lower_bound, float(np.nanmax(y_arr)))
    if upper_bound < lower_bound:
        upper_bound = lower_bound

    # Apply clipping
    clipped_preds = np.clip(preds_arr, lower_bound, upper_bound)

    # Calculate diagnostic statistics
    n_clipped_low = int(np.sum(clipped_preds == lower_bound))
    n_clipped_high = int(np.sum(clipped_preds == upper_bound))

    n_total = len(preds_arr)
    pct_clipped_low = 100.0 * n_clipped_low / n_total if n_total > 0 else 0.0
    pct_clipped_high = 100.0 * n_clipped_high / n_total if n_total > 0 else 0.0

    return {
        "clipped_predictions": clipped_preds,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "n_clipped_lower": n_clipped_low,
        "n_clipped_upper": n_clipped_high,
        "pct_clipped_lower": pct_clipped_low,
        "pct_clipped_upper": pct_clipped_high,
    }
