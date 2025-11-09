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
