"""Helpers for ML-based return prediction (Phase 2 of the portfolio plan).

This module provides lightweight utilities for:

- Feature engineering for return prediction (``create_ml_return_features``)
- Training a compact linear return predictor
  (``train_linear_return_predictor``)
- Combining multiple prediction sources into an ensemble
  (``create_ensemble_return_predictions``)

The implementations are intentionally simple and fast so that unit tests can
run quickly while still exercising realistic workflows derived from
``05_machine_learning.ipynb`` and ``07_dense_networks.ipynb``.
"""

from __future__ import annotations

from typing import Sequence, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def create_ml_return_features(
    df: pd.DataFrame,
    lags: Sequence[int] | None = None,
    technical_indicators: Sequence[str] | None = None,
    return_col: str | None = None,
    price_col: str | None = None,
) -> pd.DataFrame:
    """Create features for ML-based return prediction.

    Parameters
    ----------
    df:
        Input DataFrame that must contain at least a return column and
        price column. If not specified via ``return_col`` and ``price_col``,
        will auto-detect from common schema patterns.
    lags:
        Collection of integer lags (in days) for which lagged returns will be
        created. Defaults to ``[5, 10, 20]`` when ``None``.
    technical_indicators:
        Collection specifying which technical indicators to include. Supported
        values are ``"sma"``, ``"momentum"`` and ``"volatility"``. When
        ``None``, all three are used.
    return_col:
        Name of the daily return column. If ``None``, will attempt to detect
        from common patterns: ``return_1d``, ``1_day_pct``, ``1_day_%``.
    price_col:
        Name of the price column. If ``None``, will attempt to detect
        from common patterns: ``last_price``, ``price``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the original columns plus engineered features.
        Leading rows that do not have full feature coverage are dropped so the
        result contains no missing values.
    """

    # Auto-detect return column if not specified
    if return_col is None:
        for candidate in ["return_1d", "1_day_pct", "1_day_%", "return_1d_pct"]:
            if candidate in df.columns:
                return_col = candidate
                break
        if return_col is None:
            raise KeyError(
                "Could not find daily return column. Expected one of: "
                "return_1d, 1_day_pct, 1_day_%, return_1d_pct. "
                "Specify explicitly via return_col parameter."
            )

    # Auto-detect price column if not specified
    if price_col is None:
        for candidate in ["last_price", "price", "close", "close_price"]:
            if candidate in df.columns:
                price_col = candidate
                break
        if price_col is None:
            raise KeyError(
                "Could not find price column. Expected one of: "
                "last_price, price, close, close_price. "
                "Specify explicitly via price_col parameter."
            )

    # Verify columns exist
    if return_col not in df.columns:
        raise KeyError(f"Specified return column '{return_col}' not found in DataFrame")
    if price_col not in df.columns:
        raise KeyError(f"Specified price column '{price_col}' not found in DataFrame")

    if lags is None:
        lags = [5, 10, 20]

    if technical_indicators is None:
        technical_indicators = ["sma", "momentum", "volatility"]

    features = df.copy()

    # Lagged returns - use detected return column
    for lag in lags:
        features[f"return_lag_{lag}"] = features[return_col].shift(lag)

    # Technical indicators – window sizes chosen to mirror the examples in
    # the enhancement plan while remaining lightweight.
    tech_set = {t.lower() for t in technical_indicators}

    if "sma" in tech_set:
        features["sma_20"] = features[price_col].rolling(20, min_periods=20).mean()

    if "momentum" in tech_set:
        features["momentum_10"] = features[return_col].rolling(10, min_periods=10).mean()

    if "volatility" in tech_set:
        features["volatility_20"] = features[return_col].rolling(20, min_periods=20).std(ddof=0)

    # Drop rows with any NaNs so downstream models receive clean data.
    features = features.dropna(axis=0, how="any")

    return features


def train_linear_return_predictor(X_train: np.ndarray, y_train: np.ndarray) -> Ridge:
    """Train a compact linear regressor for return prediction.

    A small ``Ridge`` model is used instead of a deep neural network to keep
    dependency weight and training time low, while still providing a flexible
    regression baseline suitable for unit tests and simple pipelines.

    Parameters
    ----------
    X_train:
        2D NumPy array of input features.
    y_train:
        1D NumPy array of target returns.

    Returns
    -------
    sklearn.linear_model.Ridge
        Fitted Ridge regression model.
    """

    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array")
    if y_train.ndim != 1:
        raise ValueError("y_train must be a 1D array")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of rows")

    model = Ridge(alpha=1e-2, random_state=42)
    model.fit(X_train, y_train)
    return model


def create_ensemble_return_predictions(
    df: pd.DataFrame,
    models: Sequence[str],
    weights: Sequence[float],
    ensemble_col: str = "ensemble_return",
) -> pd.DataFrame:
    """Combine multiple prediction columns into a weighted ensemble.

    Parameters
    ----------
    df:
        DataFrame containing the individual model prediction columns.
    models:
        Column names whose predictions will be combined.
    weights:
        Non-negative weights associated with each model. They are normalized
        to sum to 1.
    ensemble_col:
        Name of the output column that will contain the ensemble prediction.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with an additional ``ensemble_col``.
    """

    if not models:
        raise ValueError("models list must not be empty")

    if len(models) != len(weights):
        raise ValueError("models and weights must have the same length")

    missing = [m for m in models if m not in df.columns]
    if missing:
        raise KeyError(f"Missing prediction columns: {missing}")

    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")

    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")

    w = w / total

    preds = df[models].to_numpy(dtype=float)
    ensemble = preds @ w

    out = df.copy()
    out[ensemble_col] = ensemble
    return out


def evaluate_return_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> Dict[str, Any]:
    """Compute basic evaluation metrics for return predictions.

    This helper underpins the Phase 2 review checkpoint by providing a
    consistent way to summarise model performance (correlation and error
    metrics) for y_true vs. y_pred.

    Parameters
    ----------
    y_true:
        Array-like of realised returns.
    y_pred:
        Array-like of predicted returns with the same length as ``y_true``.

    Returns
    -------
    dict
        Dictionary containing at least the following keys:

        - ``correlation``: Pearson correlation coefficient between
          ``y_true`` and ``y_pred`` (NaN if undefined).
        - ``mae``: Mean absolute error.
        - ``rmse``: Root mean squared error.
    """

    true_arr = np.asarray(y_true, dtype=float).ravel()
    pred_arr = np.asarray(y_pred, dtype=float).ravel()

    if true_arr.shape != pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if true_arr.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")

    # Error metrics
    diff = pred_arr - true_arr
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    # Correlation – guard against zero-variance edge cases
    if np.allclose(true_arr, true_arr[0]) or np.allclose(pred_arr, pred_arr[0]):
        corr = float("nan")
    else:
        corr_matrix = np.corrcoef(pred_arr, true_arr)
        corr = float(corr_matrix[0, 1])

    return {"correlation": corr, "mae": mae, "rmse": rmse}
