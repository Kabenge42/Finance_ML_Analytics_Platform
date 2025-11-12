"""
Uncertainty quantification helpers with simple, distribution-free conformal prediction.

Implements functions to calibrate prediction intervals to a desired coverage level
using absolute residuals on a calibration set. This directly addresses Priority 0
from Model Optimization Recommendations: severely miscalibrated prediction intervals.

Public API:
- conformal_prediction_intervals(y_cal, y_cal_pred, y_test_pred, alpha=0.2, clip_lower_at_zero=True)
- compute_interval_coverage(y_true, lower, upper)

Notes:
- alpha = 0.2 corresponds to 80% prediction intervals.
- Uses absolute residual quantile on calibration set to build symmetric intervals.
- Optionally clips lower bound at zero to satisfy non-negative price targets.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _validate_inputs(y_cal: np.ndarray, y_cal_pred: np.ndarray, y_test_pred: np.ndarray, alpha: float) -> None:
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if y_cal is None or y_cal_pred is None or y_test_pred is None:
        raise ValueError("Inputs must not be None")
    y_cal = np.asarray(y_cal)
    y_cal_pred = np.asarray(y_cal_pred)
    y_test_pred = np.asarray(y_test_pred)
    if y_cal.size == 0 or y_cal_pred.size == 0 or y_test_pred.size == 0:
        raise ValueError("Input arrays must be non-empty")
    if y_cal.shape[0] != y_cal_pred.shape[0]:
        raise ValueError("y_cal and y_cal_pred must have the same length")


def conformal_prediction_intervals(
    y_cal: np.ndarray,
    y_cal_pred: np.ndarray,
    y_test_pred: np.ndarray,
    alpha: float = 0.2,
    clip_lower_at_zero: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute symmetric conformal prediction intervals with target coverage 1 - alpha.

    Parameters
    ----------
    y_cal : array-like, shape (n_cal,)
        Calibration true targets.
    y_cal_pred : array-like, shape (n_cal,)
        Calibration predictions (from a model not trained on the calibration subset).
    y_test_pred : array-like, shape (n_test,)
        Predictions for which to build intervals (e.g., validation/test set).
    alpha : float, default=0.2
        Miscoverage rate; 0.2 gives 80% intervals.
    clip_lower_at_zero : bool, default=True
        Clip lower bound at zero to enforce non-negative price targets.

    Returns
    -------
    lower : np.ndarray
        Lower bounds for prediction intervals.
    upper : np.ndarray
        Upper bounds for prediction intervals.
    """
    _validate_inputs(y_cal, y_cal_pred, y_test_pred, alpha)

    y_cal = np.asarray(y_cal).astype(float)
    y_cal_pred = np.asarray(y_cal_pred).astype(float)
    y_test_pred = np.asarray(y_test_pred).astype(float)

    # Absolute residuals on calibration set
    cal_residuals = np.abs(y_cal - y_cal_pred)

    # Conformal quantile (1 - alpha) of residuals
    q = np.quantile(cal_residuals, 1 - alpha)

    lower = y_test_pred - q
    upper = y_test_pred + q

    if clip_lower_at_zero:
        lower = np.maximum(lower, 0.0)

    return lower, upper


def compute_interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Compute empirical coverage of intervals: P(lower <= y_true <= upper).

    Parameters
    ----------
    y_true : array-like
    lower : array-like
    upper : array-like

    Returns
    -------
    coverage : float
        Fraction in [0,1] of targets covered by intervals.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if y_true.size == 0 or lower.size == 0 or upper.size == 0:
        raise ValueError("Inputs must be non-empty arrays")
    if not (y_true.shape[0] == lower.shape[0] == upper.shape[0]):
        raise ValueError("y_true, lower, upper must have the same length")
    return np.mean((y_true >= lower) & (y_true <= upper))
