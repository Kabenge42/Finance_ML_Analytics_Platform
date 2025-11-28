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

import logging
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


def create_ml_return_features(
    df: pd.DataFrame,
    lags: Sequence[int] | None = None,
    technical_indicators: Sequence[str] | None = None,
    return_col: str | None = None,
    price_col: str | None = None,
    require_time_series: bool = False,
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
    require_time_series:
        If ``False`` (default), the function will detect cross-sectional data
        (insufficient time-series observations) and return the input DataFrame
        unchanged with a warning. If ``True``, will raise an error when
        insufficient time-series data is available.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the original columns plus engineered features.
        Leading rows that do not have full feature coverage are dropped so the
        result contains no missing values.

        For cross-sectional data (when ``require_time_series=False``), returns
        the input DataFrame unchanged if insufficient observations are available.

    Raises
    ------
    ValueError
        When ``require_time_series=True`` and data is cross-sectional.
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

    # Detect if data is cross-sectional (insufficient time-series observations)
    # Required minimum observations = max of (max lag, max window size)
    max_lag = max(lags) if lags else 0
    tech_set = {t.lower() for t in technical_indicators}
    max_window = (
        20
        if "sma" in tech_set or "volatility" in tech_set
        else (10 if "momentum" in tech_set else 0)
    )
    min_required_obs = max(max_lag, max_window)

    # Check if we have sufficient observations
    if len(df) < min_required_obs:
        # Cross-sectional data detected
        if require_time_series:
            raise ValueError(
                f"Insufficient time-series data: {len(df)} rows < {min_required_obs} required. "
                f"Cannot create ML features with lags={lags} and technical_indicators={list(technical_indicators)}. "
                f"Data appears to be cross-sectional (single snapshot). "
                f"Either provide historical time-series data or set require_time_series=False."
            )
        else:
            # Return input unchanged with a warning in a way that's visible to the caller
            # The caller should check if the output == input to detect this case
            import warnings

            warnings.warn(
                f"Cross-sectional data detected ({len(df)} rows < {min_required_obs} required for time-series features). "
                f"Returning input DataFrame unchanged. ML features not created.",
                UserWarning,
                stacklevel=2,
            )
            return df.copy()

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


# ========== Phase 7 Enhancement Functions ==========


def clip_expected_returns(
    returns: np.ndarray | pd.Series,
    min_return: float | None = None,
    max_return: float | None = None,
) -> np.ndarray:
    """Clip expected returns to realistic bounds.

    This function addresses the return calculation issue where expected returns
    can be unrealistically high (e.g., 95.6% mean) leading to inflated Sharpe
    ratios (e.g., 42.4).

    Parameters
    ----------
    returns:
        Array-like of expected returns to clip.
    min_return:
        Minimum allowed return. Defaults to MIN_EXPECTED_RETURN from config.
    max_return:
        Maximum allowed return. Defaults to MAX_EXPECTED_RETURN from config.

    Returns
    -------
    np.ndarray
        Clipped returns within [min_return, max_return] bounds.

    Examples
    --------
    >>> returns = np.array([0.10, 0.956, 1.5, -0.50, -1.5])
    >>> clipped = clip_expected_returns(returns)
    >>> np.max(clipped) <= 1.0  # MAX_EXPECTED_RETURN
    True
    """
    from finance_ml.ml_workflow.config.ml_returns_config import (
        MAX_EXPECTED_RETURN,
        MIN_EXPECTED_RETURN,
    )

    if min_return is None:
        min_return = MIN_EXPECTED_RETURN
    if max_return is None:
        max_return = MAX_EXPECTED_RETURN

    arr = np.asarray(returns, dtype=float)

    # Safety check for empty array
    if arr.size == 0:
        logger.warning("clip_expected_returns received empty array. Returning empty array.")
        return arr

    return np.clip(arr, min_return, max_return)


def calculate_historical_returns(
    df: pd.DataFrame,
    current_price_col: str = "last_price",
) -> pd.DataFrame:
    """Calculate historical returns from price columns.

    Uses PRICE_COLUMNS registry to identify historical price columns and
    calculates returns as: (current_price - historical_price) / historical_price.

    Parameters
    ----------
    df:
        DataFrame containing price columns.
    current_price_col:
        Name of the current price column. Defaults to 'last_price'.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with additional return columns:
        return_1w, return_1m, return_3m, return_6m, return_1y, etc.

    Examples
    --------
    >>> df = pd.DataFrame({'last_price': [100], 'price_1m_ago': [90]})
    >>> result = calculate_historical_returns(df)
    >>> 'return_1m' in result.columns
    True
    """
    from finance_ml.ml_workflow.config.ml_returns_config import PRICE_COLUMNS

    result = df.copy()

    if current_price_col not in result.columns:
        return result  # Cannot calculate returns without current price

    current_price = result[current_price_col]

    # Mapping from historical column names to return column names
    historical_mapping = {
        "price_5d_ago": "return_5d",
        "price_1w_ago": "return_1w",
        "price_1m_ago": "return_1m",
        "price_3m_ago": "return_3m",
        "price_6m_ago": "return_6m",
        "price_1y_ago": "return_1y",
        "price_2y_ago": "return_2y",
        "price_3y_ago": "return_3y",
        "price_5y_ago": "return_5y",
    }

    historical_cols = PRICE_COLUMNS.get("historical", [])

    for hist_col in historical_cols:
        if hist_col in result.columns and hist_col in historical_mapping:
            return_col = historical_mapping[hist_col]
            historical_price = pd.to_numeric(result[hist_col], errors="coerce")
            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                returns = (current_price - historical_price) / historical_price
                returns = returns.replace([np.inf, -np.inf], np.nan)
            result[return_col] = returns

    return result


def get_phase93_return_features() -> Dict[str, list]:
    """Get Phase 9.3 feature categories relevant for return prediction.

    Returns high-relevance feature categories from the 196 Phase 9.3
    engineered features, prioritized by their predictive value for
    expected returns.

    Returns
    -------
    Dict[str, list]
        Dictionary mapping category names to lists of feature column names.

    Examples
    --------
    >>> categories = get_phase93_return_features()
    >>> 'Momentum & Technical' in categories
    True
    >>> len(categories) >= 4
    True
    """
    from finance_ml.ml_workflow.config.ml_returns_config import (
        PHASE93_RETURN_FEATURE_CATEGORIES,
    )

    try:
        from finance_ml.ml_workflow.eda.phase93_categories import (
            PHASE93_FEATURE_CATEGORIES,
        )

        # Filter to only return-relevant categories
        return {
            cat: features
            for cat, features in PHASE93_FEATURE_CATEGORIES.items()
            if cat in PHASE93_RETURN_FEATURE_CATEGORIES
        }
    except ImportError:
        # Fallback if phase93_categories is not available
        return {
            "Momentum & Technical": [
                "price_momentum_1m",
                "price_momentum_3m",
                "price_momentum_6m",
                "price_momentum_1y",
                "rsi_14d",
                "sharpe_proxy",
            ],
            "Valuation Ratios": [
                "p_e_ratio",
                "p_b_ratio",
                "ev_ebitda_ratio",
                "peg_ratio",
            ],
            "Growth Metrics": [
                "revenue_growth_yoy",
                "earnings_growth_yoy",
            ],
            "Analyst Sentiment": [
                "analyst_rating_avg",
                "price_target_upside",
            ],
        }


def create_ml_return_features_enhanced(
    df: pd.DataFrame,
    include_phase93: bool = True,
    include_historical_returns: bool = True,
) -> pd.DataFrame:
    """Create enhanced ML features for return prediction using Phase 9.3 features.

    This function extends the basic create_ml_return_features by integrating
    Phase 9.3 engineered features and historical return calculations, addressing
    the feature underutilization issue (6 features vs 196 available).

    Parameters
    ----------
    df:
        Input DataFrame with financial metrics.
    include_phase93:
        If True, include available Phase 9.3 features. Defaults to True.
    include_historical_returns:
        If True, calculate historical returns from price columns. Defaults to True.

    Returns
    -------
    pd.DataFrame
        DataFrame with enhanced features for ML return prediction.

    Examples
    --------
    >>> df = create_sample_data_with_phase93_features()
    >>> result = create_ml_return_features_enhanced(df)
    >>> len(result.columns) > len(df.columns)
    True
    """
    result = df.copy()

    # Add historical returns if requested
    if include_historical_returns:
        result = calculate_historical_returns(result)

    # Include Phase 9.3 features if requested
    if include_phase93:
        phase93_categories = get_phase93_return_features()

        # Collect all Phase 9.3 feature names
        all_phase93_features = []
        for features in phase93_categories.values():
            all_phase93_features.extend(features)

        # Keep Phase 9.3 features that exist in the input DataFrame
        # (they are already included, just ensure they are preserved)
        available_phase93 = [f for f in all_phase93_features if f in result.columns]

        # Log the number of Phase 9.3 features found
        if available_phase93:
            logger.debug(f"Found {len(available_phase93)} Phase 9.3 features in input data")

    return result


def validate_expected_returns(
    returns: np.ndarray | pd.Series,
    mean_threshold: float | None = None,
) -> Dict[str, Any]:
    """Validate expected returns for realism and safely handle empty arrays.

    This diagnostic function checks if expected returns are realistic and
    flags potential issues like the 95.6% mean return problem. It safely
    handles empty arrays without crashing.

    Parameters
    ----------
    returns:
        Array-like of expected returns to validate.
    mean_threshold:
        Maximum acceptable mean return. Defaults to REALISTIC_RETURN_MEAN_THRESHOLD.

    Returns
    -------
    dict
        Dictionary containing:
        - is_realistic: bool - True if returns pass validation
        - mean: float - Mean of the returns (np.nan if empty)
        - mean_return: float - Mean of the returns (alias for mean, np.nan if empty)
        - std_return: float - Standard deviation of the returns (np.nan if empty)
        - max: float - Maximum return (np.nan if empty)
        - min: float - Minimum return (np.nan if empty)
        - n_samples: int - Number of samples
        - reason: str - Reason for unrealistic returns (if applicable)
        - num_extreme_high: int - Count of returns > MAX_EXPECTED_RETURN (optional)
        - num_extreme_low: int - Count of returns < MIN_EXPECTED_RETURN (optional)
        - warnings: list - List of warning messages (optional)

    Examples
    --------
    >>> returns = np.array([0.956] * 100)  # Unrealistic 95.6% mean
    >>> diagnostics = validate_expected_returns(returns)
    >>> diagnostics['is_realistic']
    False
    >>> empty_returns = np.array([])
    >>> diagnostics = validate_expected_returns(empty_returns)
    >>> diagnostics['is_realistic']
    False
    >>> diagnostics['reason']
    'Empty returns array'
    """
    from finance_ml.ml_workflow.config.ml_returns_config import (
        MAX_EXPECTED_RETURN,
        MIN_EXPECTED_RETURN,
        REALISTIC_RETURN_MEAN_THRESHOLD,
    )

    if mean_threshold is None:
        mean_threshold = REALISTIC_RETURN_MEAN_THRESHOLD

    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]  # Remove NaN values

    # Handle empty input safely
    if arr.size == 0:
        return {
            "is_realistic": False,
            "reason": "Empty returns array",
            "mean": np.nan,
            "mean_return": np.nan,
            "std_return": np.nan,
            "max": np.nan,
            "min": np.nan,
            "n_samples": 0,
            "warnings": [],  # Consistent schema: always include warnings
        }

    mean_return = float(np.mean(arr))
    std_return = float(np.std(arr))
    max_return = float(np.max(arr))
    min_return = float(np.min(arr))

    num_extreme_high = int(np.sum(arr > MAX_EXPECTED_RETURN))
    num_extreme_low = int(np.sum(arr < MIN_EXPECTED_RETURN))

    warnings = []
    is_realistic = True

    # Validation Criteria:
    # 1. Mean return should be < 50% (0.50) for realistic equity portfolios
    # 2. Bounds should not be egregiously high (checked against config limits)
    if abs(mean_return) >= 0.50:
        is_realistic = False
        warnings.append(f"Mean return {mean_return:.2%} exceeds 50% threshold")
    elif abs(mean_return) > mean_threshold:
        warnings.append(f"Mean return {mean_return:.2%} exceeds threshold {mean_threshold:.0%}")

    if max_return > MAX_EXPECTED_RETURN * 2.0:
        is_realistic = False
        warnings.append(f"Max return {max_return:.2%} exceeds safety threshold")

    # Check for extreme values
    if num_extreme_high > 0:
        pct_extreme = num_extreme_high / arr.size * 100
        warnings.append(
            f"{num_extreme_high} returns ({pct_extreme:.1f}%) exceed MAX {MAX_EXPECTED_RETURN:.0%}"
        )
        if pct_extreme > 10:
            is_realistic = False

    if num_extreme_low > 0:
        pct_extreme = num_extreme_low / arr.size * 100
        warnings.append(
            f"{num_extreme_low} returns ({pct_extreme:.1f}%) below MIN {MIN_EXPECTED_RETURN:.0%}"
        )
        if pct_extreme > 10:
            is_realistic = False

    return {
        "is_realistic": is_realistic,
        "mean": mean_return,
        "mean_return": mean_return,
        "std_return": std_return,
        "max": max_return,
        "min": min_return,
        "n_samples": len(arr),
        "num_extreme_high": num_extreme_high,
        "num_extreme_low": num_extreme_low,
        "warnings": warnings,
    }


# ========== Phase 7.4: Dense Neural Network Implementation ==========

# Check TensorFlow availability at module level
try:
    import tensorflow as tf
    from tensorflow import keras

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None
    keras = None


def build_dnn_return_predictor(
    input_dim: int,
    hidden_layers: Sequence[int] | None = None,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    output_activation: str = "linear",
) -> Any:
    """Build a Dense Neural Network for return prediction.

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_layers:
        List of hidden layer sizes. Defaults to [64, 32, 16].
    dropout_rate:
        Dropout rate for regularization. Defaults to 0.3.
    l2_reg:
        L2 regularization factor. Defaults to 1e-4.
    output_activation:
        Activation function for output layer. Defaults to 'linear'.

    Returns
    -------
    keras.Model
        Compiled Keras model for return prediction.

    Raises
    ------
    ImportError
        If TensorFlow is not installed.
    """
    if not HAS_TENSORFLOW:
        raise ImportError(
            "TensorFlow is required for DNN models. " "Install with: pip install tensorflow"
        )

    if hidden_layers is None:
        hidden_layers = [64, 32, 16]

    from tensorflow.keras import layers, models, regularizers

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
            )
        )
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation=output_activation))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    return model


def train_dnn_return_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    hidden_layers: Sequence[int] | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    early_stopping_patience: int = 10,
    verbose: int = 1,
) -> tuple:
    """Train a DNN return predictor.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training targets.
    X_val:
        Validation features (optional).
    y_val:
        Validation targets (optional).
    hidden_layers:
        List of hidden layer sizes.
    epochs:
        Maximum number of training epochs.
    batch_size:
        Training batch size.
    early_stopping_patience:
        Patience for early stopping.
    verbose:
        Verbosity level (0=silent, 1=progress bar, 2=one line per epoch).

    Returns
    -------
    tuple
        (trained_model, history_dict) where history_dict contains 'loss'
        and optionally 'val_loss' lists.
    """
    if not HAS_TENSORFLOW:
        raise ImportError(
            "TensorFlow is required for DNN models. " "Install with: pip install tensorflow"
        )

    model = build_dnn_return_predictor(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
    )

    callbacks = []
    validation_data = None

    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
        from tensorflow.keras.callbacks import EarlyStopping

        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
    )

    history_dict = {
        "loss": history.history["loss"],
    }
    if "val_loss" in history.history:
        history_dict["val_loss"] = history.history["val_loss"]

    return model, history_dict


def train_dnn_quantile_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    quantile: float = 0.5,
    hidden_layers: Sequence[int] | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: int = 1,
) -> Any:
    """Train a DNN for quantile regression.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training targets.
    quantile:
        Quantile to predict (0.1, 0.5, 0.9, etc.).
    hidden_layers:
        List of hidden layer sizes.
    epochs:
        Maximum number of training epochs.
    batch_size:
        Training batch size.
    verbose:
        Verbosity level.

    Returns
    -------
    keras.Model
        Trained quantile regression model.
    """
    if not HAS_TENSORFLOW:
        raise ImportError(
            "TensorFlow is required for DNN models. " "Install with: pip install tensorflow"
        )

    if hidden_layers is None:
        hidden_layers = [64, 32, 16]

    from tensorflow.keras import layers, models, regularizers, backend as K

    # Quantile loss function
    def quantile_loss(q):
        def loss(y_true, y_pred):
            error = y_true - y_pred
            return K.mean(K.maximum(q * error, (q - 1) * error), axis=-1)

        return loss

    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1, activation="linear"))

    model.compile(
        optimizer="adam",
        loss=quantile_loss(quantile),
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    return model


# ========== Phase 7.5: Ensemble Model Enhancement ==========


class ReturnEnsemble:
    """Ensemble of multiple models for return prediction.

    This class combines predictions from multiple model types using
    configurable weighting schemes.
    """

    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] | None = None):
        """Initialize ensemble with trained models.

        Parameters
        ----------
        models:
            Dictionary mapping model names to trained model objects.
        weights:
            Dictionary mapping model names to weights. If None, uses equal weights.
        """
        self.models = models
        if weights is None:
            n_models = len(models)
            self.weights = {name: 1.0 / n_models for name in models}
        else:
            # Normalize weights to sum to 1
            total = sum(weights.values())
            self.weights = {name: w / total for name, w in weights.items()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions.

        Parameters
        ----------
        X:
            Input features.

        Returns
        -------
        np.ndarray
            Weighted ensemble predictions.
        """
        predictions = np.zeros(X.shape[0])
        for name, model in self.models.items():
            pred = model.predict(X)
            if hasattr(pred, "flatten"):
                pred = pred.flatten()
            predictions += self.weights[name] * pred
        return predictions

    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights.copy()

    def get_individual_mses(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Calculate MSE for each individual model.

        Parameters
        ----------
        X:
            Input features.
        y_true:
            True target values.

        Returns
        -------
        dict
            Dictionary mapping model names to their MSE values.
        """
        mses = {}
        for name, model in self.models.items():
            pred = model.predict(X)
            if hasattr(pred, "flatten"):
                pred = pred.flatten()
            mses[name] = float(np.mean((pred - y_true) ** 2))
        return mses


def create_return_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: Sequence[str] | None = None,
    cv_folds: int = 5,
) -> ReturnEnsemble:
    """Create an ensemble of return prediction models.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training targets.
    models:
        List of model types to include. Options: 'ridge', 'random_forest',
        'gradient_boosting', 'dnn'. Defaults to ['ridge', 'random_forest'].
    cv_folds:
        Number of cross-validation folds for model selection (not currently used).

    Returns
    -------
    ReturnEnsemble
        Trained ensemble object.
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    if models is None:
        models = ["ridge", "random_forest"]

    trained_models = {}

    for model_name in models:
        if model_name == "ridge":
            model = Ridge(alpha=1e-2, random_state=42)
            model.fit(X_train, y_train)
            trained_models["ridge"] = model

        elif model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            trained_models["random_forest"] = model

        elif model_name == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=42,
            )
            model.fit(X_train, y_train)
            trained_models["gradient_boosting"] = model

        elif model_name == "dnn" and HAS_TENSORFLOW:
            model, _ = train_dnn_return_predictor(
                X_train,
                y_train,
                hidden_layers=[32, 16],
                epochs=30,
                verbose=0,
            )
            trained_models["dnn"] = model

    return ReturnEnsemble(trained_models)


def create_dynamic_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: Sequence[str] | None = None,
    weighting_method: str = "inverse_mse",
    validation_data: tuple | None = None,
) -> ReturnEnsemble:
    """Create an ensemble with dynamic weighting based on validation performance.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training targets.
    models:
        List of model types to include.
    weighting_method:
        Method for computing weights: 'inverse_mse', 'softmax', or 'equal'.
    validation_data:
        Tuple of (X_val, y_val) for computing weights.

    Returns
    -------
    ReturnEnsemble
        Ensemble with dynamically computed weights.
    """
    # First create ensemble with equal weights
    ensemble = create_return_ensemble(X_train, y_train, models)

    if validation_data is None or weighting_method == "equal":
        return ensemble

    X_val, y_val = validation_data

    # Compute MSEs on validation set
    mses = ensemble.get_individual_mses(X_val, y_val)

    # Compute weights based on method
    if weighting_method == "inverse_mse":
        # Inverse MSE weighting (lower MSE = higher weight)
        inv_mses = {name: 1.0 / (mse + 1e-8) for name, mse in mses.items()}
        total = sum(inv_mses.values())
        weights = {name: v / total for name, v in inv_mses.items()}

    elif weighting_method == "softmax":
        # Softmax weighting on negative MSE
        neg_mses = np.array([-mse for mse in mses.values()])
        exp_vals = np.exp(neg_mses - neg_mses.max())  # Numerically stable
        softmax = exp_vals / exp_vals.sum()
        weights = {name: float(softmax[i]) for i, name in enumerate(mses.keys())}

    else:
        weights = None

    # Update ensemble weights
    ensemble.weights = weights
    return ensemble


# ========== Phase 7.6: Black-Litterman ML Integration ==========


def create_bl_views_from_ml(
    ml_predictions: pd.Series | np.ndarray,
    tickers: Sequence[str] | None = None,
    confidence_method: str = "uniform",
    min_confidence: float = 0.3,
    max_confidence: float = 0.9,
) -> tuple:
    """Create Black-Litterman views from ML predictions.

    Parameters
    ----------
    ml_predictions:
        ML-predicted expected returns (Series with ticker index or array).
    tickers:
        List of ticker symbols. Required if ml_predictions is an array.
    confidence_method:
        Method for computing confidences: 'uniform', 'prediction_interval',
        or 'model_r2'. Currently only 'uniform' is implemented.
    min_confidence:
        Minimum confidence level for views.
    max_confidence:
        Maximum confidence level for views.

    Returns
    -------
    tuple
        (views_dict, confidences_list) where views_dict maps tickers to
        expected returns and confidences_list contains confidence values.
    """
    if isinstance(ml_predictions, pd.Series):
        if tickers is None:
            tickers = ml_predictions.index.tolist()
        predictions = ml_predictions.values
    else:
        predictions = np.asarray(ml_predictions)
        if tickers is None:
            raise ValueError("tickers must be provided when ml_predictions is an array")

    # Create views dictionary
    views = {ticker: float(pred) for ticker, pred in zip(tickers, predictions)}

    # Compute confidences
    if confidence_method == "uniform":
        # Use middle of confidence range
        mid_confidence = (min_confidence + max_confidence) / 2
        confidences = [mid_confidence] * len(tickers)

    elif confidence_method == "prediction_interval":
        # Scale confidence by prediction magnitude (higher predictions = higher confidence)
        pred_abs = np.abs(predictions)
        pred_scaled = (pred_abs - pred_abs.min()) / (pred_abs.max() - pred_abs.min() + 1e-8)
        confidences = [min_confidence + (max_confidence - min_confidence) * p for p in pred_scaled]

    else:
        # Default to uniform
        mid_confidence = (min_confidence + max_confidence) / 2
        confidences = [mid_confidence] * len(tickers)

    return views, confidences


def detect_market_regime(
    returns: pd.DataFrame,
    method: str = "volatility",
    thresholds: Dict[str, float] | None = None,
) -> str:
    """Detect current market regime based on return characteristics.

    Parameters
    ----------
    returns:
        DataFrame of asset returns.
    method:
        Detection method: 'volatility', 'momentum', or 'hmm'.
        Currently only 'volatility' is implemented.
    thresholds:
        Dictionary with threshold values for regime classification.
        For 'volatility': {'low_vol': 0.10, 'high_vol': 0.25}.

    Returns
    -------
    str
        Regime classification: 'low_volatility', 'normal', or 'high_volatility'.
    """
    if thresholds is None:
        thresholds = {"low_vol": 0.10, "high_vol": 0.25}

    if method == "volatility":
        # Calculate annualized portfolio volatility (equal-weighted)
        portfolio_returns = returns.mean(axis=1)
        annualized_vol = portfolio_returns.std() * np.sqrt(252)

        if annualized_vol < thresholds["low_vol"]:
            return "low_volatility"
        elif annualized_vol > thresholds["high_vol"]:
            return "high_volatility"
        else:
            return "normal"

    elif method == "momentum":
        # Calculate average momentum
        cumulative_return = (1 + returns.mean(axis=1)).prod() - 1
        if cumulative_return > 0.10:
            return "low_volatility"  # Bull market
        elif cumulative_return < -0.10:
            return "high_volatility"  # Bear market
        else:
            return "normal"

    else:
        return "normal"


# ========== Phase 7.7: Robust Covariance Estimation ==========


def estimate_covariance_shrinkage(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
) -> np.ndarray:
    """Estimate covariance matrix using shrinkage methods.

    Parameters
    ----------
    returns:
        DataFrame of asset returns.
    method:
        Shrinkage method: 'ledoit_wolf', 'oracle_approx', or 'empirical_bayes'.

    Returns
    -------
    np.ndarray
        Shrunk covariance matrix (positive definite).
    """
    from sklearn.covariance import LedoitWolf, OAS

    returns_array = returns.values

    if method == "ledoit_wolf":
        lw = LedoitWolf()
        lw.fit(returns_array)
        return lw.covariance_

    elif method == "oracle_approx" or method == "oas":
        oas = OAS()
        oas.fit(returns_array)
        return oas.covariance_

    else:
        # Default to Ledoit-Wolf
        lw = LedoitWolf()
        lw.fit(returns_array)
        return lw.covariance_


def estimate_covariance_ewm(
    returns: pd.DataFrame,
    halflife: int = 60,
    min_periods: int = 30,
) -> np.ndarray:
    """Estimate covariance matrix using exponentially weighted method.

    Parameters
    ----------
    returns:
        DataFrame of asset returns.
    halflife:
        Halflife in days for exponential weighting.
    min_periods:
        Minimum number of observations required.

    Returns
    -------
    np.ndarray
        Exponentially weighted covariance matrix.
    """
    # Calculate EWM covariance using pandas
    ewm_cov = returns.ewm(halflife=halflife, min_periods=min_periods).cov()

    # Get the last covariance matrix
    last_date = returns.index[-1]
    cov_matrix = ewm_cov.loc[last_date].values

    # Ensure symmetry
    cov_matrix = (cov_matrix + cov_matrix.T) / 2

    return cov_matrix


# ========== Phase 7.8: Model Validation & Diagnostics ==========


def calculate_return_prediction_diagnostics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    include_distribution_tests: bool = False,
    include_autocorrelation: bool = False,
) -> Dict[str, Any]:
    """Calculate comprehensive diagnostics for return predictions.

    Parameters
    ----------
    y_true:
        Actual return values.
    y_pred:
        Predicted return values.
    include_distribution_tests:
        If True, include distribution tests for residuals.
    include_autocorrelation:
        If True, include autocorrelation analysis.

    Returns
    -------
    dict
        Dictionary containing diagnostic metrics:
        - mse: Mean squared error
        - mae: Mean absolute error
        - r2: R-squared
        - ic: Information coefficient (correlation)
        - residual_* metrics if distribution tests enabled
    """
    true_arr = np.asarray(y_true, dtype=float).ravel()
    pred_arr = np.asarray(y_pred, dtype=float).ravel()

    # Basic error metrics
    residuals = pred_arr - true_arr
    mse = float(np.mean(residuals**2))
    mae = float(np.mean(np.abs(residuals)))

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Information Coefficient (correlation)
    if np.std(true_arr) > 0 and np.std(pred_arr) > 0:
        ic = float(np.corrcoef(true_arr, pred_arr)[0, 1])
    else:
        ic = 0.0

    diagnostics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "ic": ic,
    }

    if include_distribution_tests:
        from scipy import stats

        # Normality test
        if len(residuals) >= 8:
            _, normality_pvalue = stats.normaltest(residuals)
            diagnostics["residual_normality_pvalue"] = float(normality_pvalue)
        diagnostics["residual_skewness"] = float(stats.skew(residuals))
        diagnostics["residual_kurtosis"] = float(stats.kurtosis(residuals))

    if include_autocorrelation:
        # Lag-1 autocorrelation of residuals
        if len(residuals) > 1:
            acf_lag1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
            diagnostics["residual_acf_lag1"] = acf_lag1

    return diagnostics


def validate_portfolio_metrics(
    weights: np.ndarray,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.03,
    max_sharpe_threshold: float = 5.0,
    max_return_threshold: float = 1.0,
) -> Dict[str, Any]:
    """Validate portfolio metrics for realism.

    Parameters
    ----------
    weights:
        Portfolio weights.
    returns:
        DataFrame of asset returns.
    risk_free_rate:
        Annual risk-free rate for Sharpe calculation.
    max_sharpe_threshold:
        Maximum acceptable Sharpe ratio (above this is flagged).
    max_return_threshold:
        Maximum acceptable annual return (above this is flagged).

    Returns
    -------
    dict
        Dictionary containing:
        - sharpe_ratio: Calculated Sharpe ratio
        - sharpe_ratio_valid: True if Sharpe is realistic
        - portfolio_return: Annualized portfolio return
        - return_realistic: True if return is realistic
        - portfolio_volatility: Annualized portfolio volatility
        - warnings: List of warning messages
    """
    weights = np.asarray(weights).flatten()
    returns_array = returns.values

    # Calculate portfolio returns
    portfolio_returns = returns_array @ weights

    # Annualize
    annual_return = float(np.mean(portfolio_returns) * 252)
    annual_vol = float(np.std(portfolio_returns) * np.sqrt(252))

    # Sharpe ratio
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0

    warnings = []
    sharpe_valid = True
    return_realistic = True

    if abs(sharpe) > max_sharpe_threshold:
        sharpe_valid = False
        warnings.append(f"Sharpe ratio {sharpe:.2f} exceeds threshold {max_sharpe_threshold}")

    if abs(annual_return) > max_return_threshold:
        return_realistic = False
        warnings.append(
            f"Annual return {annual_return:.2%} exceeds threshold {max_return_threshold:.0%}"
        )

    return {
        "sharpe_ratio": sharpe,
        "sharpe_ratio_valid": sharpe_valid,
        "portfolio_return": annual_return,
        "return_realistic": return_realistic,
        "portfolio_volatility": annual_vol,
        "warnings": warnings,
    }
