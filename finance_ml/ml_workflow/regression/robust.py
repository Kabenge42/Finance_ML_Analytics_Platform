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
