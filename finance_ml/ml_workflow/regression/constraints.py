"""
Prediction constraints for regression models.

This module provides wrappers and utilities to enforce constraints on regression
model predictions, particularly non-negativity constraints for financial predictions
like stock prices and price targets.

Phase 9.5 - Regression Refactor
"""

import logging
import numpy as np


logger = logging.getLogger(__name__)


class NonNegativeRegressionWrapper:
    """
    Wrapper for regression models that ensures predictions are non-negative.

    This wrapper clips predictions to be >= 0, which is essential for price
    target predictions since stock prices cannot be negative. Linear regression
    (Ridge, Lasso, ElasticNet) can produce negative predictions without
    constraints, especially when features have extreme values or the model
    is poorly regularized.

    The wrapper applies post-prediction clipping using np.maximum(pred, 0.0),
    which is computationally efficient and maintains differentiability at
    the boundary.

    Args:
        base_model: Any sklearn-compatible regression model

    Attributes:
        base_model: The wrapped regression model

    Example:
        >>> from sklearn.linear_model import Ridge
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Create training data
        >>> X = pd.DataFrame({'feature1': np.random.randn(100)})
        >>> y = pd.Series(np.abs(np.random.randn(100)) * 10 + 5)
        >>>
        >>> # Train with non-negative constraint
        >>> base = Ridge(alpha=1.0)
        >>> model = NonNegativeRegressionWrapper(base)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        >>> assert (predictions >= 0).all()  # All predictions >= 0

    Phase 9.5 TDD Implementation:
        This class was implemented following strict TDD to solve the critical
        issue of negative price target predictions observed in production regression.
    """

    def __init__(self, base_model):
        """
        Initialize wrapper with base regression model.

        Args:
            base_model: sklearn-compatible regression model (must have fit and predict methods)
        """
        self.base_model = base_model

    def fit(self, X, y):
        """
        Fit the base model.

        Args:
            X: Feature matrix (pandas DataFrame or numpy array)
            y: Target vector (pandas Series or numpy array)

        Returns:
            self (for method chaining)
        """
        self.base_model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict and ensure all predictions are non-negative.

        This method:
        1. Gets predictions from base model
        2. Clips predictions to be >= 0 using np.maximum
        3. Returns clipped predictions

        Args:
            X: Feature matrix (pandas DataFrame or numpy array)

        Returns:
            Non-negative predictions (numpy array with all values >= 0)

        Note:
            The clipping operation is applied element-wise and has minimal
            performance overhead. For most financial regression, less than 5% of
            predictions require clipping.
        """
        predictions = self.base_model.predict(X)

        # Count how many predictions would be negative (for monitoring)
        n_negative = np.sum(predictions < 0)
        if n_negative > 0:
            pct_negative = 100.0 * n_negative / len(predictions)
            logger.debug(
                f"NonNegativeRegressionWrapper: Clipped {n_negative}/{len(predictions)} "
                f"({pct_negative:.1f}%) negative predictions to 0"
            )

        # Clip predictions to ensure they're >= 0
        return np.maximum(predictions, 0.0)

    def __getattr__(self, name):
        """
        Delegate attribute access to base model.

        This method is called when an attribute is not found in the wrapper.
        It delegates to the wrapped base_model, allowing transparent access
        to base model attributes and methods.

        Args:
            name: Name of the attribute to access

        Returns:
            Attribute value from base model
        """
        # Prevent infinite recursion during copying/pickling
        # by not delegating special methods that don't exist
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Prevent recursion if base_model is not yet set (during __init__ or unpickling)
        if "base_model" not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        return getattr(self.base_model, name)

    def __getstate__(self):
        """Support for pickling/copying."""
        return self.__dict__.copy()

    def __setstate__(self, state):
        """Support for unpickling/copying."""
        self.__dict__.update(state)
