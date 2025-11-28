"""
Phase 9.5.1: Model Persistence and Serialization

This module provides utilities for saving and loading trained regression models
along with their metadata. It uses joblib for efficient serialization of
scikit-learn compatible models and handles model versioning metadata.

Key Features:
- Save trained models with optional metadata (hyperparameters, metrics, version)
- Load models with metadata retrieval
- Automatic directory creation for save paths
- File existence validation on load
- Support for any scikit-learn compatible model

Metadata Best Practices:
- Model type and version (e.g., 'xgboost', 'v1.0')
- Training date and data version
- Hyperparameters used
- Performance metrics (R², MAE, RMSE)
- Feature names and engineering steps
- Preprocessing transformations applied

Integration with Phase 9.5:
- Compatible with all models from regression.models
- Can save results from compare_regressors
- Integrates with model versioning systems
- Supports ML pipeline checkpointing

Example:
    >>> from finance_ml.ml_workflow.regression.models import train_xgboost_regressor
    >>> from finance_ml.ml_workflow.regression.io import save_model, load_model
    >>> from finance_ml.ml_workflow.regression.dataset import prepare_regression_data
    >>>
    >>> # Train model
    >>> X_train, X_test, y_train, y_test, _ = prepare_regression_data(df)
    >>> results = train_xgboost_regressor(X_train, y_train, random_state=42)
    >>> model = results['model']
    >>>
    >>> # Save with metadata
    >>> metadata = {
    ...     'model_type': 'xgboost',
    ...     'version': '1.0',
    ...     'train_r2': results['train_score'],
    ...     'n_features': X_train.shape[1],
    ...     'training_date': '2025-01-08',
    ...     'hyperparameters': {'n_estimators': 100, 'max_depth': 5}
    ... }
    >>> save_model(model, 'models/xgboost_v1.pkl', metadata=metadata)
    >>>
    >>> # Load later
    >>> loaded_model, loaded_metadata = load_model('models/xgboost_v1.pkl')
    >>> print(f"Loaded {loaded_metadata['model_type']} model")
    >>> print(f"Training R²: {loaded_metadata['train_r2']:.3f}")
    >>> predictions = loaded_model.predict(X_test)

Performance Notes:
- joblib is optimized for scikit-learn models and numpy arrays
- Compression is applied by default (protocol 4)
- Large models (>100MB) may take several seconds to save/load
- Consider using model.compress=3 for additional compression

Reference:
- joblib documentation: https://joblib.readthedocs.io/
- Model serialization best practices: https://scikit-learn.org/stable/model_persistence.html
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib

# Configure logger
logger = logging.getLogger(__name__)


def save_model(
    model: Any, filepath: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save trained model with optional metadata to disk.

    This function serializes a trained regression model and its associated
    metadata to a file using joblib. The model and metadata are stored together
    in a single pickle file for easy retrieval.

    Args:
        model: Trained regression model (any scikit-learn compatible estimator)
               Examples: Ridge, XGBRegressor, Pipeline, NonNegativeRegressionWrapper
        filepath: Path where model will be saved (str or Path object)
                 Directory will be created automatically if it doesn't exist
                 Recommended extensions: .pkl, .joblib
                 Example: 'models/xgboost_sector_tech_v1.pkl'
        metadata: Optional dictionary with model metadata (default: None)
                 Common keys:
                 - model_type: str (e.g., 'xgboost', 'ridge', 'stacking')
                 - version: str (e.g., 'v1.0', 'prod_2025_01')
                 - train_r2: float (training R² score)
                 - cv_r2: float (cross-validation R² score)
                 - n_features: int (number of features)
                 - feature_names: List[str] (feature column names)
                 - hyperparameters: Dict[str, Any] (model hyperparameters)
                 - training_date: str (date model was trained)
                 - data_version: str (version of training data)
                 - sector: str (if sector-specific model)

    Returns:
        None (writes to disk)

    Raises:
        OSError: If filepath cannot be written (permissions, disk space)
        TypeError: If model cannot be pickled

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Train model
        >>> X = pd.DataFrame(np.random.randn(100, 5))
        >>> y = pd.Series(np.random.randn(100))
        >>> model = RandomForestRegressor(n_estimators=50, random_state=42)
        >>> model.fit(X, y)
        >>>
        >>> # Save with comprehensive metadata
        >>> metadata = {
        ...     'model_type': 'random_forest',
        ...     'version': 'v2.1',
        ...     'train_r2': model.score(X, y),
        ...     'n_features': X.shape[1],
        ...     'feature_names': list(X.columns),
        ...     'hyperparameters': model.get_params(),
        ...     'training_date': '2025-01-08',
        ...     'notes': 'Trained on Q4 2024 data'
        ... }
        >>> save_model(model, 'models/rf_v2_1.pkl', metadata=metadata)
        >>> print("Model saved successfully")

    Notes:
        - The function creates parent directories automatically
        - Existing files are overwritten without warning
        - joblib uses pickle protocol 4 by default (Python 3.4+)
        - Models with custom classes may require the class definition at load time
        - For production, consider versioning the filepath (e.g., model_v1.pkl)
    """
    filepath = Path(filepath)

    # Create parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {filepath}")

    # Package model and metadata together
    save_dict = {"model": model, "metadata": metadata or {}}

    try:
        joblib.dump(save_dict, filepath)
        logger.info(
            f"✓ Model saved successfully: {filepath} ({filepath.stat().st_size / 1024:.1f} KB)"
        )

        # Log metadata summary
        if metadata:
            logger.debug(f"  Metadata keys: {list(metadata.keys())}")
            if "model_type" in metadata:
                logger.debug(f"  Model type: {metadata['model_type']}")
            if "version" in metadata:
                logger.debug(f"  Version: {metadata['version']}")
    except Exception as e:
        logger.error(f"Failed to save model to {filepath}: {e}")
        raise


def load_model(filepath: Union[str, Path]) -> Tuple[Any, Dict[str, Any]]:
    """
    Load trained model with metadata from disk.

    This function deserializes a model and its metadata from a file created by
    save_model(). It performs validation to ensure the file exists and can be
    loaded properly.

    Args:
        filepath: Path to saved model file (str or Path object)
                 Must be a file created by save_model()
                 Example: 'models/xgboost_sector_tech_v1.pkl'

    Returns:
        Tuple of (model, metadata):
        - model: Loaded regression model (ready for prediction)
        - metadata: Dictionary with model metadata (empty dict if none was saved)

    Raises:
        FileNotFoundError: If filepath does not exist
        EOFError: If file is corrupted or incomplete
        pickle.UnpicklingError: If file format is incompatible

    Example:
        >>> from finance_ml.ml_workflow.regression.io import load_model
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Load model
        >>> model, metadata = load_model('models/xgboost_v1.pkl')
        >>>
        >>> # Inspect metadata
        >>> print(f"Model type: {metadata.get('model_type', 'unknown')}")
        >>> print(f"Version: {metadata.get('version', 'unknown')}")
        >>> print(f"Training R²: {metadata.get('train_r2', 'N/A')}")
        >>> print(f"Features: {metadata.get('n_features', 'N/A')}")
        >>>
        >>> # Use for prediction
        >>> X_new = pd.DataFrame(np.random.randn(10, metadata['n_features']))
        >>> predictions = model.predict(X_new)
        >>> print(f"Generated {len(predictions)} predictions")

    Example with error handling:
        >>> try:
        ...     model, metadata = load_model('models/missing_model.pkl')
        ... except FileNotFoundError:
        ...     print("Model file not found, training new model...")
        ...     # Train new model here

    Notes:
        - The loaded model is ready for immediate use (no retraining needed)
        - Ensure the same library versions used for training are available at load time
        - For models with custom classes, the class definition must be importable
        - Metadata access is safe: metadata.get('key', default) won't raise KeyError
        - Large models (>100MB) may take several seconds to load
    """
    filepath = Path(filepath)

    # Validate file exists
    if not filepath.exists():
        raise FileNotFoundError(
            f"Model file not found: {filepath}. "
            f"Please check the path or train a new model with save_model()."
        )

    logger.info(f"Loading model from {filepath}")

    try:
        save_dict = joblib.load(filepath)
        model = save_dict["model"]
        metadata = save_dict.get("metadata", {})

        logger.info(
            f"✓ Model loaded successfully: {filepath} ({filepath.stat().st_size / 1024:.1f} KB)"
        )

        # Log metadata summary
        if metadata:
            logger.debug(f"  Metadata keys: {list(metadata.keys())}")
            if "model_type" in metadata:
                logger.debug(f"  Model type: {metadata['model_type']}")
            if "version" in metadata:
                logger.debug(f"  Version: {metadata['version']}")
        else:
            logger.debug("  No metadata found in saved model")

        return model, metadata
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {e}")
        raise


def build_predictions_frame(
    y_true: "pd.Series",
    y_pred: "np.ndarray",
    df_source: "pd.DataFrame",
    extra_cols: Optional[Dict[str, "np.ndarray"]] = None,
) -> "pd.DataFrame":
    """
    Build standardized predictions DataFrame with required columns.

    Creates a DataFrame combining predictions, errors, and metadata columns
    (ticker, sector, region, last_price) from the source dataframe.

    Addresses Priority 1: Missing sector/ticker in regression_predictions.csv

    Args:
        y_true: True target values (pandas Series with index)
        y_pred: Model predictions (numpy array, same length as y_true)
        df_source: Source dataframe containing metadata columns
                   Should be indexed compatibly with y_true
        extra_cols: Optional dict of additional columns to include
                   E.g., {'pred_p10': array, 'pred_p50': array, 'pred_p90': array}

    Returns:
        DataFrame with standardized schema:
        - y_true, y_pred (predictions)
        - abs_error, pct_error (error metrics)
        - ticker, sector, region, last_price (metadata, if available)
        - Any columns from extra_cols

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> y_true = pd.Series([100, 200], index=[0, 1])
        >>> y_pred = np.array([95, 210])
        >>> df_source = pd.DataFrame({
        ...     'ticker': ['AAPL', 'MSFT'],
        ...     'sector': ['Tech', 'Tech'],
        ...     'last_price': [150.0, 250.0]
        ... }, index=[0, 1])
        >>> result = build_predictions_frame(y_true, y_pred, df_source)
        >>> result.columns
        Index(['y_true', 'y_pred', 'abs_error', 'pct_error', 'ticker', 'sector', 'last_price'])
    """
    import pandas as pd
    import numpy as np

    # Build base predictions dataframe
    result = pd.DataFrame(
        {
            "y_true": y_true.values,
            "y_pred": y_pred,
        },
        index=y_true.index,
    )

    # Phase 9.5 Safety Rails: Enforce non-negativity on predictions before error calc
    # This minimizes downstream schema violations and aligns with code_guidelines.md v1.4
    with np.errstate(invalid="ignore"):
        neg_mask = pd.Series(np.isfinite(result["y_pred"])) & (result["y_pred"] < 0)
    if neg_mask.any():
        n_neg = int(neg_mask.sum())
        try:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                f"Clamping {n_neg} negative predictions to 0.0 (Phase 9.5 safety rail)"
            )
        except Exception:
            pass
        result.loc[neg_mask, "y_pred"] = 0.0

    # Compute error metrics
    result["abs_error"] = np.abs(result["y_true"] - result["y_pred"])

    # Compute percentage error (handle divide-by-zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["pct_error"] = ((result["y_pred"] - result["y_true"]) / result["y_true"]) * 100
        result["pct_error"] = result["pct_error"].replace([np.inf, -np.inf], np.nan)

    # Add metadata columns from source dataframe if available
    metadata_cols = ["ticker", "isin", "sector", "region", "last_price", "market_cap"]
    for col in metadata_cols:
        if col in df_source.columns:
            result[col] = df_source.loc[y_true.index, col]

    # Add any extra columns (e.g., quantile predictions)
    if extra_cols:
        for col_name, col_values in extra_cols.items():
            result[col_name] = col_values

    # If quantiles are present, enforce Phase 9.5 requirements and add interval_width
    q_cols = [c for c in ["pred_p10", "pred_p50", "pred_p90"] if c in result.columns]
    if {"pred_p10", "pred_p90"}.issubset(result.columns):
        # Safety: clip quantiles to be non-negative
        for q in ["pred_p10", "pred_p50", "pred_p90"]:
            if q in result.columns:
                with np.errstate(invalid="ignore"):
                    neg_q_mask = pd.Series(np.isfinite(result[q])) & (result[q] < 0)
                if neg_q_mask.any():
                    result.loc[neg_q_mask, q] = 0.0

        # Ensure interval_width exists
        if "interval_width" not in result.columns:
            result["interval_width"] = result["pred_p90"] - result["pred_p10"]

    return result


def validate_predictions_schema(df: "pd.DataFrame") -> "pd.DataFrame":
    """Validate standardized regression predictions schema.

    This lightweight helper enforces the core schema contract described in
    ``code_guidelines.md`` Section 2.4 while remaining backward compatible
    with existing artifacts.

    The function focuses on core responsibilities aligned with code_guidelines.md v1.4+:

    1. Ensure **core prediction columns** are present::

           ["y_true", "y_pred", "abs_error", "pct_error"]

       If any are missing, a :class:`ValueError` is raised listing the
       missing columns.

    2. If lower/upper quantile columns (``pred_p10``/``pred_p50``/``pred_p90``)
       are present:
       - **interval_width** must be present (Phase 9.5 P0.4).
       - Values must be non-negative (>=0).
       - Monotonicity per-row (p10 ≤ p50 ≤ p90) must be satisfied.
       - If violations are found, a :class:`ValueError` is raised.

    3. Enforce simple invariants on non-negative price-like columns. When
       a ``last_price`` column is present, all finite values must be
       non-negative; otherwise a :class:`ValueError` is raised.
       Any negative ``y_pred`` values raise a :class:`ValueError` (Phase 9.5 P0.4).

    Parameters
    ----------
    df : pandas.DataFrame
        Predictions dataframe to validate.

    Returns
    -------
    pandas.DataFrame
        The validated dataframe. A shallow copy is returned if modifications
        are needed (though strict mode prefers raising errors).
    """

    import numpy as np
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("validate_predictions_schema expects a pandas DataFrame")

    # 1. Core required prediction columns
    required_core = ["y_true", "y_pred", "abs_error", "pct_error"]
    missing = [col for col in required_core if col not in df.columns]
    if missing:
        raise ValueError(
            f"Predictions schema validation failed: missing required columns: {missing}"
        )

    result = df

    # 2. Quantile handling: non-negativity, monotonicity and interval width
    has_p10 = "pred_p10" in df.columns
    has_p50 = "pred_p50" in df.columns
    has_p90 = "pred_p90" in df.columns
    has_interval = "interval_width" in df.columns

    # If lower and upper quantiles exist, auto-add interval_width to be flexible
    if has_p10 and has_p90 and not has_interval:
        try:
            result = result.copy()
            result["interval_width"] = result["pred_p90"] - result["pred_p10"]
            has_interval = True
        except Exception:
            # Fallback: keep as-is; downstream code may handle
            pass

    # Non-negativity checks for any present quantiles
    for qcol in (c for c in ("pred_p10", "pred_p50", "pred_p90") if c in result.columns):
        neg_count = (result[qcol] < 0).sum(skipna=True)
        if neg_count > 0:
            raise ValueError(
                f"Predictions schema validation failed: {qcol} contains {neg_count} negative values. "
                "Models must enforce non-negativity before validation."
            )

    # Monotonicity only when all three quantiles are present
    if has_p10 and has_p50 and has_p90:
        monotonic_violation = (
            (result["pred_p10"] > result["pred_p50"]) | (result["pred_p50"] > result["pred_p90"])
        ).sum(skipna=True)
        if monotonic_violation > 0:
            raise ValueError(
                f"Predictions schema validation failed: {monotonic_violation} rows violate "
                "quantile monotonicity (p10 <= p50 <= p90)."
            )

    # Strict check for negative y_pred
    if "y_pred" in df.columns:
        neg_pred = (result["y_pred"] < 0).sum(skipna=True)
        if neg_pred > 0:
            raise ValueError(
                f"Predictions schema validation failed: y_pred contains {neg_pred} negative values. "
                "Models must enforce non-negativity before validation."
            )

    # 3. Non-negative last_price invariant (if column exists)
    if "last_price" in result.columns:
        finite_mask = np.isfinite(result["last_price"])  # type: ignore[arg-type]
        if (result.loc[finite_mask, "last_price"] < 0).any():
            raise ValueError(
                "Predictions schema validation failed: last_price contains negative values, "
                "which violates the standardized predictions schema."
            )

    return result
