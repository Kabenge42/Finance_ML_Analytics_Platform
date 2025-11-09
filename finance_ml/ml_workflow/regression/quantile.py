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

# Configure logger
logger = logging.getLogger(__name__)


def train_quantile_regressor(
    X: pd.DataFrame, y: pd.Series, quantiles: Optional[List[float]] = None, random_state: int = 42
) -> Tuple[List[Any], Dict[str, Any]]:
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
        Tuple of (models, results):
        - models: List of trained HistGradientBoostingRegressor models, one per quantile
                 Ordered by quantile value (e.g., [model_0.1, model_0.5, model_0.9])
        - results: Dictionary with aggregated metrics:
          {
              'quantiles': List of quantile values trained,
              'n_models': Number of models (length of quantiles list),
              'model_type': 'quantile_regression',
              'quantile_results': List of per-quantile dicts with:
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
        >>> models, results = train_quantile_regressor(X, y, quantiles=[0.1, 0.5, 0.9])
        >>>
        >>> # Check results
        >>> print(f"Trained {results['n_models']} quantile models")
        >>> for qr in results['quantile_results']:
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

    results = {
        "quantiles": quantiles,
        "n_models": len(models),
        "model_type": "quantile_regression",
        "quantile_results": quantile_results,  # Add per-quantile results
    }

    logger.info(
        f"✓ Quantile regression training complete: {len(models)} models, "
        f"mean R² = {np.mean([qr['train_score'] for qr in quantile_results]):.3f}"
    )

    return models, results
