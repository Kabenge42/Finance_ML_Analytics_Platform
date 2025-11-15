"""
Phase 9.5.1: Quantile Regression for Uncertainty Estimation

This module provides quantile regression capabilities for estimating prediction
uncertainty intervals. Unlike traditional point predictions, quantile regression
predicts multiple quantiles (e.g., 10th, 50th, 90th percentiles) to capture the
full distribution of possible outcomes.

Key Features:
- Multiple quantile prediction (default: 0.1, 0.5, 0.9)
- Uses HistGradientBoostingRegressor with quantile loss
- Returns separate models for each quantile
- Suitable for risk assessment and confidence intervals

Use Cases:
- Price target prediction with confidence bands
- Risk-adjusted valuation (worst-case, expected, best-case scenarios)
- Portfolio optimization with uncertainty quantification
- Stress testing and scenario analysis

Integration with Phase 9.5:
- Compatible with prepare_regression_data from regression.dataset
- Can be trained on same features as point prediction models
- Outputs can be combined with classification meta-features

Example:
    >>> from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor
    >>> from finance_ml.ml_workflow.regression.dataset import prepare_regression_data
    >>>
    >>> # Prepare data
    >>> X_train, X_test, y_train, y_test, _ = prepare_regression_data(df)
    >>>
    >>> # Train quantile models for 10th, 50th, 90th percentiles
    >>> models, results = train_quantile_regressor(
    ...     X_train, y_train,
    ...     quantiles=[0.1, 0.5, 0.9]
    ... )
    >>>
    >>> # Make predictions for each quantile
    >>> pred_10th = models[0].predict(X_test)  # Lower bound (10th percentile)
    >>> pred_50th = models[1].predict(X_test)  # Median prediction
    >>> pred_90th = models[2].predict(X_test)  # Upper bound (90th percentile)
    >>>
    >>> # Compute prediction interval width
    >>> interval_width = pred_90th - pred_10th
    >>> print(f"Average 80% prediction interval: ${interval_width.mean():.2f}")

Reference:
- Quantile Regression: Koenker & Bassett (1978)
- HistGradientBoosting: sklearn.ensemble.HistGradientBoostingRegressor
"""

import logging
from typing import List, Optional, Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# Re-use conformal logic from the dedicated uncertainty module to avoid
# duplication and keep all coverage maths in a single place.
from .uncertainty import conformal_prediction_intervals

# Configure logger
logger = logging.getLogger(__name__)


def train_quantile_regressor(
    X: pd.DataFrame, y: pd.Series, quantiles: Optional[List[float]] = None, random_state: int = 42
) -> Dict[str, Any]:
    """
    Train quantile regression for uncertainty estimation.

    This function trains separate HistGradientBoostingRegressor models for each
    requested quantile using the quantile loss function. Each model predicts a
    different percentile of the target distribution, enabling uncertainty estimation.

    Args:
        X: Feature matrix (pandas DataFrame with numeric features)
        y: Target vector (pandas Series with numeric target values)
        quantiles: List of quantiles to predict (default: [0.1, 0.5, 0.9])
                  Each value should be in range (0, 1)
                  Common choices:
                  - [0.1, 0.5, 0.9] for 80% prediction interval
                  - [0.05, 0.5, 0.95] for 90% prediction interval
                  - [0.25, 0.5, 0.75] for interquartile range
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary with standardized keys per code_guidelines.md Section 1.1:
        {
            'model': List of trained HistGradientBoostingRegressor models, one per quantile,
                    ordered by quantile value (e.g., [model_0.1, model_0.5, model_0.9]),
            'metrics': Dict[str, float] with 'mean_r2' (average R² across all quantiles),
            'y_pred': None (not computed during training; predictions made per quantile model),
            'artifacts': Dict with:
                - 'models': List of models (for backward compatibility),
                - 'quantiles': List of quantile values trained,
                - 'n_models': Number of models (length of quantiles list),
                - 'quantile_results': List of per-quantile dicts with:
                    - 'quantile': float (e.g., 0.1)
                    - 'train_score': R² score on training data
                    - 'model_type': 'quantile_regression'
        }

    Raises:
        ValueError: If quantiles list contains values outside (0, 1)

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Create sample data
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randn(100) * 10 + 50)
        >>>
        >>> # Train quantile models
        >>> result = train_quantile_regressor(X, y, quantiles=[0.1, 0.5, 0.9])
        >>>
        >>> # Check results
        >>> models = result['model']
        >>> print(f"Trained {len(models)} quantile models")
        >>> for qr in result['artifacts']['quantile_results']:
        ...     print(f"  Q{qr['quantile']}: R² = {qr['train_score']:.3f}")
        >>>
        >>> # Make predictions
        >>> X_new = pd.DataFrame(np.random.randn(10, 5), columns=[f'feat_{i}' for i in range(5)])
        >>> pred_low = models[0].predict(X_new)   # 10th percentile
        >>> pred_mid = models[1].predict(X_new)   # 50th percentile (median)
        >>> pred_high = models[2].predict(X_new)  # 90th percentile

    Notes:
        - The 50th percentile (median) is often more robust to outliers than mean prediction
        - Quantile models should satisfy: pred_q1 <= pred_q2 for q1 < q2, but this is
          not strictly enforced and may be violated due to independent model training
        - For price target prediction, quantile intervals provide natural confidence bands
        - Training time scales linearly with number of quantiles
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    # Validate quantiles
    for q in quantiles:
        if not (0 < q < 1):
            raise ValueError(f"Quantile {q} is outside valid range (0, 1)")

    models = []
    quantile_results = []  # Store results per quantile

    logger.info(f"Training {len(quantiles)} quantile regression models: {quantiles}")

    for q in quantiles:
        # Use HistGradientBoosting with quantile loss
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=100, random_state=random_state
        )
        model.fit(X, y)
        models.append(model)

        # Store per-quantile results
        train_score = model.score(X, y)
        quantile_results.append(
            {"quantile": q, "train_score": train_score, "model_type": "quantile_regression"}
        )
        logger.debug(f"  Trained quantile {q:.2f}: R² = {train_score:.3f}")

    # Calculate mean R² across all quantiles for metrics
    mean_r2 = np.mean([qr["train_score"] for qr in quantile_results])

    logger.info(
        f"✓ Quantile regression training complete: {len(models)} models, "
        f"mean R² = {mean_r2:.3f}"
    )

    # Return standardized dict format per code_guidelines.md Section 1.1
    return {
        "model": models,  # List of quantile models
        "metrics": {"mean_r2": mean_r2},
        "y_pred": None,  # Not computed during training
        "artifacts": {
            "models": models,  # Keep models in artifacts for backward compatibility
            "quantiles": quantiles,
            "n_models": len(models),
            "quantile_results": quantile_results,
        },
    }


def enforce_monotonic_quantiles(quantile_preds: dict) -> dict:
    """
    Enforce monotonicity constraint on quantile predictions.

    Ensures that for any sample i and quantiles q1 < q2:
        pred_q1[i] <= pred_q2[i]

    Uses averaging approach to resolve violations while minimizing changes.

    Args:
        quantile_preds: Dict mapping quantile values to prediction arrays.
                       E.g., {0.1: array([...]), 0.5: array([...]), 0.9: array([...])}

    Returns:
        Dict with same structure but monotonic predictions.

    Example:
        >>> preds = {0.1: np.array([50, 40]), 0.5: np.array([60, 35]), 0.9: np.array([70, 30])}
        >>> monotonic = enforce_monotonic_quantiles(preds)
        >>> # Index 1 had violations (40 > 35 > 30), now fixed
    """
    if not quantile_preds:
        return quantile_preds

    # Sort quantiles
    sorted_quantiles = sorted(quantile_preds.keys())

    # Convert to numpy arrays and stack
    n_samples = len(quantile_preds[sorted_quantiles[0]])
    n_quantiles = len(sorted_quantiles)

    # Stack predictions: shape (n_samples, n_quantiles)
    pred_matrix = np.column_stack([quantile_preds[q] for q in sorted_quantiles])

    # Apply monotonic constraint row by row
    for i in range(n_samples):
        row = pred_matrix[i, :]
        # Sort the row to enforce monotonicity
        pred_matrix[i, :] = np.sort(row)

    # Convert back to dict
    result = {}
    for j, q in enumerate(sorted_quantiles):
        result[q] = pred_matrix[:, j]

    return result


def conformal_calibrate_intervals(
    y_cal: np.ndarray,
    y_cal_pred: np.ndarray,
    y_test_pred: np.ndarray,
    alpha: float = 0.2,
    clip_lower_at_zero: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Thin wrapper around :func:`conformal_prediction_intervals`.

    This helper exists to match the Phase 9.9 API described in
    ``code_guidelines.md`` and the implementation plan.  It simply
    delegates to :func:`conformal_prediction_intervals`, which already
    implements the finite-sample conformal coverage logic and optional
    non-negativity clipping.

    The function is placed in ``quantile.py`` so that quantile-centric
    workflows can import it alongside quantile training utilities while
    keeping the actual maths in :mod:`uncertainty`.
    """

    lower, upper = conformal_prediction_intervals(
        y_cal=y_cal,
        y_cal_pred=y_cal_pred,
        y_test_pred=y_test_pred,
        alpha=alpha,
        clip_lower_at_zero=clip_lower_at_zero,
    )
    return lower, upper


def clip_negative_intervals(
    lower: np.ndarray,
    upper: np.ndarray,
    zero_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clip interval lower bounds at a non-negative threshold.

    This utility is a small post-processing step for price prediction
    intervals to ensure that all lower bounds are at least
    ``zero_threshold`` (default ``0.0``) while preserving the upper
    bounds and the ordering ``lower <= upper``.

    It is intentionally lightweight and operates element-wise:

    - ``lower_clipped = max(lower, zero_threshold)``
    - ``upper_clipped = max(upper, lower_clipped)``

    The behaviour is validated by tests in
    ``tests/test_uncertainty_calibration.py``.
    """

    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)

    # Raise early if shapes are incompatible
    if lower_arr.shape != upper_arr.shape:
        raise ValueError(
            f"lower and upper must have the same shape; got {lower_arr.shape} and {upper_arr.shape}"
        )

    # Enforce non-negative lower bounds
    lower_clipped = np.maximum(lower_arr, float(zero_threshold))

    # Ensure upper is at least as large as the (possibly raised) lower
    upper_clipped = np.maximum(upper_arr, lower_clipped)

    return lower_clipped, upper_clipped
