"""
Phase 9.5: Advanced Regression Models with Classification Features

This module implements sector-optimized regression regression enhanced with classification
meta-features, including:
- Diverse model architectures (linear, gradient boosting, neural networks)
- Hyperparameter optimization with Optuna
- Advanced ensemble methods (stacking, voting)
- Quantile regression for uncertainty estimation
- Model persistence and versioning
- Sector-specific optimization

Reference notebooks:
- 04_training_linear_models.ipynb
- 07_ensemble_learning_and_random_forests.ipynb
- 10_neural_nets_with_keras.ipynb
- 11_training_deep_neural_networks.ipynb

## ⚠️ PHASE 9.5 REFACTOR NOTICE ⚠️

Parts of this module have been refactored into the new regression subpackage:
    finance_ml.ml_workflow.regression/

### Already Migrated (Phase 9.5.0):
    - NonNegativeRegressionWrapper → regression.constraints
    - extract_classification_features → regression.dataset
    - integrate_classification_features_into_dataframe → regression.dataset
    - create_classification_interactions → regression.dataset
    - prepare_regression_data → regression.dataset
    - validate_training_data → regression.dataset
    - prepare_features_for_training → regression.dataset
    - extract_numeric_feature_columns → regression.dataset
    - train_sector_specific_models → regression.dataset

For new code, import from the regression subpackage:
    from finance_ml.ml_workflow.regression.constraints import NonNegativeRegressionWrapper
    from finance_ml.ml_workflow.regression.dataset import (
        extract_classification_features,
        prepare_regression_data,
        validate_training_data,
        train_sector_specific_models,
    )

### Migrated in Phase 9.5.1 (✅ COMPLETED):
    - All 14 train_*_regressor functions → regression.models
      (ridge, lasso, elastic_net, bayesian_ridge, polynomial, xgboost, lightgbm,
       catboost, histgb, random_forest, extra_trees, neural_network, voting, stacking)
    - compare_regressors → regression.models
    - train_quantile_regressor → regression.quantile
    - optimize_hyperparameters_optuna → regression.tuning
    - save_model, load_model → regression.io

For new code, import from the regression subpackage:
    from finance_ml.ml_workflow.regression.models import train_xgboost_regressor
    from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor
    from finance_ml.ml_workflow.regression.tuning import optimize_hyperparameters_optuna
    from finance_ml.ml_workflow.regression.io import save_model, load_model

This file remains for backward compatibility. All functions are now available
from both locations (here and regression/) during the transition period.
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import joblib
import numpy as np
import pandas as pd

# Configure logger for this module
logger = logging.getLogger(__name__)
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Optional dependencies with graceful fallback
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    xgb = None  # type: ignore
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    lgb = None  # type: ignore
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    CatBoostRegressor = None  # type: ignore
    HAS_CATBOOST = False

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore
    HAS_OPTUNA = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    HAS_TENSORFLOW = True
except ImportError:
    tf = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore
    HAS_TENSORFLOW = False

# Suppress warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# Non-Negative Prediction Constraint
# ==============================================================================


class NonNegativeRegressionWrapper:
    """
    Wrapper for regression regression that ensures predictions are non-negative.

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
            import logging

            pct_negative = 100.0 * n_negative / len(predictions)
            logging.debug(
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


# ==============================================================================
# Category 1: Feature Integration
# ==============================================================================


def extract_classification_features(probabilities: np.ndarray) -> pd.DataFrame:
    """
    Extract classification features from predicted probabilities.

    This function converts raw classifier probabilities into structured features
    that can be used as inputs for regression regression. The classification features
    provide meta-information about market sentiment and event likelihood.

    Creates DataFrame with 5 columns:
    - event_prob_neutral: Probability of neutral class (class 0, -10% to +10% price change)
    - event_prob_positive: Probability of positive class (class 1, >= +10% upside)
    - event_prob_negative: Probability of negative class (class 2, >= -10% downside)
    - event_class_predicted: Predicted class (0, 1, or 2 based on argmax)
    - event_confidence: Confidence score (max probability across classes)

    Args:
        probabilities: Array of shape (n_samples, 3) with class probabilities
                      from a trained 3-class event classifier

    Returns:
        DataFrame with classification features (n_samples rows, 5 columns)

    Raises:
        ValueError: If probabilities array doesn't have exactly 3 classes

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import numpy as np
        >>>
        >>> # Train event classifier
        >>> classifier = RandomForestClassifier()
        >>> classifier.fit(X_train, y_train)
        >>>
        >>> # Extract classification features for regression
        >>> probs = classifier.predict_proba(X_test)
        >>> features = extract_classification_features(probs)
        >>>
        >>> # Use in regression
        >>> X_regression = pd.concat([X_test, features], axis=1)

    Phase 9.5 Implementation:
        This function enables integration of classification meta-features into
        regression regression, as specified in the Phase 9.5 requirements for
        sector-optimized regression with classification feature enhancement.
    """
    import logging

    if probabilities.shape[1] != 3:
        raise ValueError(f"Expected 3 classes, got {probabilities.shape[1]}")

    n_samples = probabilities.shape[0]
    logging.debug(f"Extracting classification features for {n_samples} samples")

    features = pd.DataFrame(
        {
            "event_prob_neutral": probabilities[:, 0],
            "event_prob_positive": probabilities[:, 1],
            "event_prob_negative": probabilities[:, 2],
            "event_class_predicted": probabilities.argmax(axis=1),
            "event_confidence": probabilities.max(axis=1),
        }
    )

    # Log summary statistics
    avg_confidence = features["event_confidence"].mean()
    class_distribution = features["event_class_predicted"].value_counts()
    logging.debug(
        f"Average classification confidence: {avg_confidence:.3f}, "
        f"Class distribution: {dict(class_distribution)}"
    )

    return features


def integrate_classification_features_into_dataframe(
    df: pd.DataFrame, classification_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Integrate classification features into main DataFrame.

    This function combines the original stock data DataFrame with the
    classification meta-features, creating a unified dataset suitable for
    training regression regression with classification feature enhancement.

    The function:
    1. Resets indices on both DataFrames to ensure proper row alignment
    2. Concatenates horizontally (axis=1)
    3. Returns combined DataFrame with all columns

    Args:
        df: Original DataFrame with stock data (ticker, sector, price_target, etc.)
        classification_features: DataFrame with classification features from
                                extract_classification_features()

    Returns:
        Combined DataFrame with both original and classification features.
        Row count equals len(df), column count equals len(df.columns) + 5

    Raises:
        ValueError: If DataFrames have different row counts (implicit via concat)

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Original stock data
        >>> df = pd.DataFrame({
        ...     'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        ...     'sector': ['Tech', 'Tech', 'Tech'],
        ...     'last_price': [150.0, 300.0, 2500.0],
        ...     'price_target': [180.0, 350.0, 2800.0]
        ... })
        >>>
        >>> # Classification features from trained classifier
        >>> probs = np.array([[0.2, 0.7, 0.1], [0.3, 0.5, 0.2], [0.1, 0.8, 0.1]])
        >>> class_features = extract_classification_features(probs)
        >>>
        >>> # Combine for regression
        >>> df_enhanced = integrate_classification_features_into_dataframe(df, class_features)
        >>> print(df_enhanced.columns)
        # ['ticker', 'sector', 'last_price', 'price_target',
        #  'event_prob_neutral', 'event_prob_positive', 'event_prob_negative',
        #  'event_class_predicted', 'event_confidence']

    Phase 9.5 Integration:
        This function is part of the classification feature enhancement pipeline,
        enabling sector-optimized regression regression to leverage event classifier
        outputs as meta-features for improved price target prediction.

    Note:
        Both DataFrames must have the same number of rows. The function resets
        indices to avoid alignment issues, so original index values are not preserved.
    """
    import logging

    # Validate input
    if len(df) != len(classification_features):
        raise ValueError(
            f"DataFrame length mismatch: df has {len(df)} rows, "
            f"classification_features has {len(classification_features)} rows"
        )

    logging.debug(
        f"Integrating {len(classification_features.columns)} classification features "
        f"into DataFrame with {len(df.columns)} original columns"
    )

    # Reset indices to ensure proper alignment
    df_reset = df.reset_index(drop=True)
    features_reset = classification_features.reset_index(drop=True)

    # Concatenate horizontally
    result = pd.concat([df_reset, features_reset], axis=1)

    logging.debug(f"Integration complete: {len(result)} rows, {len(result.columns)} total columns")

    return result


def prepare_regression_data(
    df: pd.DataFrame,
    target_col: str = "price_target",
    exclude_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, List[str]]]:
    """
    Prepare regression data with classification meta-features.

    Args:
        df: Input DataFrame with features and target
        target_col: Target column name
        exclude_cols: Columns to exclude from features
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test, feature_info
    """
    if exclude_cols is None:
        exclude_cols = [target_col, "last_price"]

    # Identify feature types
    classification_features = [
        col
        for col in df.columns
        if col.startswith("event_prob_") or col in ["event_class_predicted", "event_confidence"]
    ]

    # Get all feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Separate numeric and categorical
    numeric_features = []
    categorical_features = []

    for col in feature_cols:
        if col in classification_features:
            numeric_features.append(col)
        elif df[col].dtype in ["int64", "float64"]:
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    # Prepare X and y (only numeric features for now)
    X = df[numeric_features].copy()
    y = df[target_col].copy()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Feature info - CRITICAL FIX: 'all_features' should only contain numeric features
    # to prevent passing non-numeric columns to model training
    feature_info = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classification_features": classification_features,
        "all_features": numeric_features,  # ✓ Only numeric features for training
    }

    return X_train, X_test, y_train, y_test, feature_info


def create_classification_interactions(
    df: pd.DataFrame, classification_cols: List[str], valuation_cols: List[str]
) -> pd.DataFrame:
    """
    Create interaction features between classification probabilities and valuation metrics.

    Args:
        df: Input DataFrame
        classification_cols: Classification feature columns (e.g., event probabilities)
        valuation_cols: Valuation metric columns (e.g., P/E, P/B ratios)

    Returns:
        DataFrame with additional interaction features
    """
    df_enhanced = df.copy()

    # Create pairwise interactions
    for class_col in classification_cols:
        for val_col in valuation_cols:
            interaction_name = f"{class_col}_x_{val_col}"
            df_enhanced[interaction_name] = df[class_col] * df[val_col]

    return df_enhanced


# ==============================================================================
# Category 2: Linear Models
# ==============================================================================


def train_ridge_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 1.0,
    cv: int = 5,
    random_state: int = 42,
    ensure_nonnegative: bool = False,
    positive: bool = False,
) -> Dict[str, Any]:
    """
    Train Ridge regression with L2 regularization.

    Args:
        X: Feature matrix
        y: Target vector
        alpha: Regularization strength
        cv: Cross-validation folds
        random_state: Random seed
        ensure_nonnegative: If True, wrap model to ensure predictions >= 0 (post-prediction clipping)
        positive: If True, constrain coefficients to be positive (sklearn's built-in constraint)

    Returns:
        Dictionary with 'model' and metrics

    Note:
        - positive=True constrains coefficients during training (sklearn native)
        - ensure_nonnegative=True clips predictions after training (wrapper approach)
        - Both can be used together for maximum constraint
    """
    # Grid search for best alpha
    alphas = np.logspace(-2, 3, 20)
    param_grid = {"alpha": alphas}

    ridge = Ridge(random_state=random_state, positive=positive)
    grid_search = GridSearchCV(ridge, param_grid, cv=cv, scoring="r2", n_jobs=-1)
    grid_search.fit(X, y)

    # Train final model with best alpha
    best_model = Ridge(
        alpha=grid_search.best_params_["alpha"], random_state=random_state, positive=positive
    )
    best_model.fit(X, y)

    # Wrap with non-negative constraint if requested
    if ensure_nonnegative:
        best_model = NonNegativeRegressionWrapper(best_model)

    # Cross-validation scores
    cv_scores = cross_val_score(
        Ridge(alpha=grid_search.best_params_["alpha"], random_state=random_state),
        X,
        y,
        cv=cv,
        scoring="r2",
    )

    results = {
        "model": best_model,
        "train_score": best_model.predict(X) if ensure_nonnegative else best_model.score(X, y),
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "best_alpha": grid_search.best_params_["alpha"],
        "model_type": "ridge",
        "nonnegative_constraint": ensure_nonnegative,
        "positive_coefficients": positive,
    }

    # Fix train_score calculation
    if ensure_nonnegative:
        from sklearn.metrics import r2_score

        y_pred = best_model.predict(X)
        results["train_score"] = r2_score(y, y_pred)

    return results


def train_lasso_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.1,
    cv: int = 5,
    random_state: int = 42,
    ensure_nonnegative: bool = False,
    positive: bool = False,
) -> Dict[str, Any]:
    """
    Train Lasso regression with L1 regularization for feature selection.

    Args:
        X: Feature matrix
        y: Target vector
        alpha: Regularization strength
        cv: Cross-validation folds
        random_state: Random seed
        ensure_nonnegative: If True, wrap model to ensure predictions >= 0 (post-prediction clipping)
        positive: If True, constrain coefficients to be positive (sklearn's built-in constraint)

    Returns:
        Dictionary with 'model' and metrics

    Note:
        - positive=True constrains coefficients during training (sklearn native)
        - ensure_nonnegative=True clips predictions after training (wrapper approach)
        - Both can be used together for maximum constraint
    """
    # Grid search for best alpha (wider range for sparse solutions)
    alphas = np.logspace(-3, 2, 20)
    param_grid = {"alpha": alphas}

    lasso = Lasso(random_state=random_state, max_iter=10000, positive=positive)
    grid_search = GridSearchCV(lasso, param_grid, cv=cv, scoring="r2", n_jobs=-1)
    grid_search.fit(X, y)

    # Train final model
    best_model = Lasso(
        alpha=grid_search.best_params_["alpha"],
        random_state=random_state,
        max_iter=10000,
        positive=positive,
    )
    best_model.fit(X, y)

    # Count non-zero coefficients (sparse solution)
    n_nonzero = np.sum(best_model.coef_ != 0)

    # Store train score before wrapping
    train_score = best_model.score(X, y)

    # Wrap with non-negative constraint if requested
    if ensure_nonnegative:
        best_model = NonNegativeRegressionWrapper(best_model)
        # Recalculate train score with wrapped model
        from sklearn.metrics import r2_score

        y_pred = best_model.predict(X)
        train_score = r2_score(y, y_pred)

    results = {
        "model": best_model,
        "train_score": train_score,
        "best_alpha": grid_search.best_params_["alpha"],
        "n_nonzero_coefs": n_nonzero,
        "n_zero_coefs": len(best_model.base_model.coef_ if ensure_nonnegative else best_model.coef_)
        - n_nonzero,
        "model_type": "lasso",
        "nonnegative_constraint": ensure_nonnegative,
        "positive_coefficients": positive,
    }

    return results


def train_elastic_net_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    cv: int = 5,
    random_state: int = 42,
    ensure_nonnegative: bool = False,
    positive: bool = False,
) -> Dict[str, Any]:
    """
    Train Elastic Net combining L1 and L2 regularization.

    Args:
        X: Feature matrix
        y: Target vector
        alpha: Overall regularization strength
        l1_ratio: Mix of L1 and L2 (0=Ridge, 1=Lasso)
        cv: Cross-validation folds
        random_state: Random seed
        ensure_nonnegative: If True, wrap model to ensure predictions >= 0 (post-prediction clipping)
        positive: If True, constrain coefficients to be positive (sklearn's built-in constraint)

    Returns:
        Dictionary with 'model' and metrics

    Note:
        - positive=True constrains coefficients during training (sklearn native)
        - ensure_nonnegative=True clips predictions after training (wrapper approach)
        - Both can be used together for maximum constraint
    """
    # Grid search for best alpha and l1_ratio
    param_grid = {"alpha": np.logspace(-4, 1, 10), "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}

    elastic = ElasticNet(random_state=random_state, max_iter=10000, positive=positive)
    grid_search = GridSearchCV(elastic, param_grid, cv=cv, scoring="r2", n_jobs=-1)
    grid_search.fit(X, y)

    # Train final model
    best_model = ElasticNet(
        alpha=grid_search.best_params_["alpha"],
        l1_ratio=grid_search.best_params_["l1_ratio"],
        random_state=random_state,
        max_iter=10000,
        positive=positive,
    )
    best_model.fit(X, y)

    # Store metrics before wrapping
    train_score = best_model.score(X, y)
    n_nonzero = np.sum(best_model.coef_ != 0)

    # Wrap with non-negative constraint if requested
    if ensure_nonnegative:
        best_model = NonNegativeRegressionWrapper(best_model)
        # Recalculate train score with wrapped model
        from sklearn.metrics import r2_score

        y_pred = best_model.predict(X)
        train_score = r2_score(y, y_pred)

    results = {
        "model": best_model,
        "train_score": train_score,
        "best_alpha": grid_search.best_params_["alpha"],
        "best_l1_ratio": grid_search.best_params_["l1_ratio"],
        "n_nonzero_coefs": n_nonzero,
        "model_type": "elastic_net",
        "nonnegative_constraint": ensure_nonnegative,
        "positive_coefficients": positive,
    }

    return results


def train_bayesian_ridge_regressor(
    X: pd.DataFrame, y: pd.Series, n_iter: int = 300, random_state: int = 42
) -> Tuple[BayesianRidge, Dict[str, Any]]:
    """
    Train Bayesian Ridge for uncertainty estimation.

    Args:
        X: Feature matrix
        y: Target vector
        n_iter: Number of iterations
        random_state: Random seed

    Returns:
        Trained model and results dictionary
    """
    model = BayesianRidge(max_iter=n_iter, compute_score=True)
    model.fit(X, y)

    # Get predictions with uncertainty
    y_pred, y_std = model.predict(X, return_std=True)

    results = {
        "train_score": model.score(X, y),
        "mean_uncertainty": y_std.mean(),
        "model_type": "bayesian_ridge",
    }

    return model, results


def train_polynomial_regressor(
    X: pd.DataFrame, y: pd.Series, degree: int = 2, alpha: float = 1.0, random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train polynomial regression with regularization.

    Args:
        X: Feature matrix
        y: Target vector
        degree: Polynomial degree
        alpha: Regularization strength
        random_state: Random seed

    Returns:
        Trained pipeline and results dictionary
    """
    # Create pipeline with polynomial features and Ridge
    pipeline = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("ridge", Ridge(alpha=alpha, random_state=random_state)),
        ]
    )

    pipeline.fit(X, y)

    results = {
        "train_score": pipeline.score(X, y),
        "degree": degree,
        "alpha": alpha,
        "model_type": "polynomial",
    }

    return pipeline, results


# ==============================================================================
# Category 3: Gradient Boosting Models
# ==============================================================================


def train_xgboost_regressor(
    X: pd.DataFrame, y: pd.Series, params: Optional[Dict[str, Any]] = None, random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train XGBoost regressor.

    Args:
        X: Feature matrix
        y: Target vector
        params: Model parameters
        random_state: Random seed

    Returns:
        Trained model and results dictionary
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")

    if params is None:
        params = {
            "max_depth": 5,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    model = xgb.XGBRegressor(**params, random_state=random_state, n_jobs=-1)
    model.fit(X, y)

    results = {"train_score": model.score(X, y), "params": params, "model_type": "xgboost"}

    return model, results


def train_lightgbm_regressor(
    X: pd.DataFrame, y: pd.Series, params: Optional[Dict[str, Any]] = None, random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train LightGBM regressor.

    Args:
        X: Feature matrix
        y: Target vector
        params: Model parameters
        random_state: Random seed

    Returns:
        Trained model and results dictionary
    """
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

    if params is None:
        params = {
            "num_leaves": 31,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    model = lgb.LGBMRegressor(**params, random_state=random_state, n_jobs=-1, verbose=-1)
    model.fit(X, y)

    results = {"train_score": model.score(X, y), "params": params, "model_type": "lightgbm"}

    return model, results


def train_catboost_regressor(
    X: pd.DataFrame, y: pd.Series, params: Optional[Dict[str, Any]] = None, random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train CatBoost regressor.

    Args:
        X: Feature matrix
        y: Target vector
        params: Model parameters
        random_state: Random seed

    Returns:
        Trained model and results dictionary
    """
    if not HAS_CATBOOST:
        raise ImportError("CatBoost not installed. Install with: pip install catboost")

    if params is None:
        params = {"depth": 6, "iterations": 100, "learning_rate": 0.1}

    model = CatBoostRegressor(**params, random_state=random_state, verbose=False)
    model.fit(X, y)

    results = {"train_score": model.score(X, y), "params": params, "model_type": "catboost"}

    return model, results


def train_histgb_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    max_iter: int = 100,
    max_depth: int = None,
    random_state: int = 42,
) -> Tuple[HistGradientBoostingRegressor, Dict[str, Any]]:
    """
    Train sklearn HistGradientBoosting regressor (fast, native).

    Args:
        X: Feature matrix
        y: Target vector
        max_iter: Maximum iterations
        max_depth: Maximum tree depth
        random_state: Random seed

    Returns:
        Trained model and results dictionary
    """
    model = HistGradientBoostingRegressor(
        max_iter=max_iter, max_depth=max_depth, random_state=random_state
    )
    model.fit(X, y)

    results = {
        "train_score": model.score(X, y),
        "max_iter": max_iter,
        "max_depth": max_depth,
        "model_type": "hist_gradient_boosting",
    }

    return model, results


# ==============================================================================
# Category 4: Tree Ensemble and Neural Models
# ==============================================================================


def train_random_forest_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """
    Train Random Forest regressor.

    Args:
        X: Feature matrix
        y: Target vector
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed

    Returns:
        Trained model and results dictionary

    Raises:
        ValueError: If X or y contain NaN/Inf values
    """
    # ============================================================================
    # VALIDATE INPUT DATA - NO NaN/Inf ALLOWED
    # ============================================================================
    if X.isnull().any().any():
        nan_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(
            f"Feature matrix X contains NaN values in columns: {nan_cols[:5]}... "
            f"({len(nan_cols)} total). Please impute missing values before training."
        )

    if y.isnull().any():
        nan_count = y.isnull().sum()
        raise ValueError(
            f"Target variable y contains {nan_count} NaN values. "
            f"Please remove or impute these before training."
        )

    if np.isinf(X.values).any():
        raise ValueError(
            "Feature matrix X contains infinite values. "
            "Please handle with replace([np.inf, -np.inf], np.nan) then impute."
        )

    if np.isinf(y.values).any():
        raise ValueError(
            "Target variable y contains infinite values. "
            "Please clip or replace with appropriate bounds."
        )

    # ============================================================================
    # TRAIN MODEL
    # ============================================================================
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
    )
    model.fit(X, y)

    # Feature importance
    feature_importance = model.feature_importances_

    results = {
        "train_score": model.score(X, y),
        "feature_importance": feature_importance,
        "n_estimators": n_estimators,
        "model_type": "random_forest",
    }

    return model, results


def train_extra_trees_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[ExtraTreesRegressor, Dict[str, Any]]:
    """
    Train Extra Trees regressor (more randomness than RF).

    Args:
        X: Feature matrix
        y: Target vector
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed

    Returns:
        Trained model and results dictionary
    """
    model = ExtraTreesRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
    )
    model.fit(X, y)

    results = {
        "train_score": model.score(X, y),
        "n_estimators": n_estimators,
        "model_type": "extra_trees",
    }

    return model, results


def train_neural_network_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    hidden_layers: Optional[List[int]] = None,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train neural network regressor with Keras.

    Args:
        X: Feature matrix
        y: Target vector
        hidden_layers: List of hidden layer sizes (default: [128, 64])
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate
        epochs: Training epochs
        batch_size: Batch size
        validation_split: Validation split ratio
        random_state: Random seed

    Returns:
        Trained model and results dictionary
    """
    if hidden_layers is None:
        hidden_layers = [128, 64]

    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")

    # Set seeds
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build model
    model = keras.Sequential()
    model.add(layers.Input(shape=(X.shape[1],)))

    for hidden_size in hidden_layers:
        model.add(layers.Dense(hidden_size, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation="linear"))

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    # Train
    history = model.fit(
        X_scaled,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0,
    )

    # Store scaler with model for predictions
    model.scaler = scaler

    results = {
        "train_loss": history.history["loss"][-1],
        "val_loss": history.history["val_loss"][-1],
        "train_mae": history.history["mae"][-1],
        "val_mae": history.history["val_mae"][-1],
        "hidden_layers": hidden_layers,
        "model_type": "neural_network",
    }

    return model, results


# ==============================================================================
# Category 5: Ensemble Methods and Quantile Regression
# ==============================================================================


def train_voting_regressor(
    X: pd.DataFrame, y: pd.Series, weights: Optional[List[float]] = None, random_state: int = 42
) -> Tuple[VotingRegressor, Dict[str, Any]]:
    """
    Train voting ensemble regressor.

    Args:
        X: Feature matrix
        y: Target vector
        weights: Model weights (None for equal weights)
        random_state: Random seed

    Returns:
        Trained ensemble and results dictionary
    """
    # Define base regression
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)),
        ("et", ExtraTreesRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=50, random_state=random_state)),
    ]

    # Create voting ensemble
    model = VotingRegressor(estimators=estimators, weights=weights, n_jobs=-1)
    model.fit(X, y)

    results = {
        "train_score": model.score(X, y),
        "base_models": [name for name, _ in estimators],
        "weights": weights,
        "model_type": "voting",
    }

    return model, results


def train_stacking_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
    ensure_nonnegative: bool = False,
    loss: str = "squared_error",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train stacking ensemble with meta-learner.

    Args:
        X: Feature matrix
        y: Target vector
        cv: Cross-validation folds for out-of-fold predictions
        random_state: Random seed
        ensure_nonnegative: If True, wrap model with NonNegativeRegressionWrapper
                           to ensure predictions >= 0
        loss: Loss function for GradientBoosting base estimator ('squared_error', 'huber', 'absolute_error')
              If 'huber', uses robust loss for outlier handling (Model Optimization Priority 2.1)

    Returns:
        Trained ensemble (wrapped if ensure_nonnegative=True) and results dictionary
    """
    # Define base regression with robust loss support
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)),
        ("et", ExtraTreesRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)),
        (
            "gb",
            GradientBoostingRegressor(
                loss=loss,
                alpha=0.9 if loss == "huber" else 0.9,  # Quantile for Huber transition
                n_estimators=50,
                random_state=random_state,
            ),
        ),
    ]

    # Meta-learner
    meta_model = Ridge(alpha=1.0)

    # Create stacking ensemble
    base_model = StackingRegressor(
        estimators=estimators, final_estimator=meta_model, cv=cv, n_jobs=-1
    )
    base_model.fit(X, y)

    # Wrap with NonNegativeRegressionWrapper if requested
    if ensure_nonnegative:
        model = NonNegativeRegressionWrapper(base_model)
    else:
        model = base_model

    # Cross-validation score (using base_model for CV to avoid wrapper issues)
    cv_scores = cross_val_score(base_model, X, y, cv=cv, scoring="r2")

    results = {
        "train_score": base_model.score(X, y),
        "cv_score": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "base_models": [name for name, _ in estimators],
        "meta_model": "Ridge",
        "model_type": "stacking",
        "ensure_nonnegative": ensure_nonnegative,
    }

    return model, results


def train_quantile_regressor(
    X: pd.DataFrame, y: pd.Series, quantiles: Optional[List[float]] = None, random_state: int = 42
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Train quantile regression for uncertainty estimation.

    Args:
        X: Feature matrix
        y: Target vector
        quantiles: Quantiles to predict (default: [0.1, 0.5, 0.9])
        random_state: Random seed

    Returns:
        List of trained regression (one per quantile) and results dictionary.
        The results dictionary includes 'quantile_results' key with a list of
        per-quantile metrics (quantile, train_score, model_type).
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    models = []
    quantile_results = []  # Store results per quantile

    for q in quantiles:
        # Use HistGradientBoosting with quantile loss
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=100, random_state=random_state
        )
        model.fit(X, y)
        models.append(model)

        # Store per-quantile results
        quantile_results.append(
            {"quantile": q, "train_score": model.score(X, y), "model_type": "quantile_regression"}
        )

    results = {
        "quantiles": quantiles,
        "n_models": len(models),
        "model_type": "quantile_regression",
        "quantile_results": quantile_results,  # Add per-quantile results
    }

    return models, results


def optimize_hyperparameters_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest",
    n_trials: int = 50,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Any]:
    """
    Optimize hyperparameters using Optuna.

    Args:
        X: Feature matrix
        y: Target vector
        model_type: Model type to optimize
        n_trials: Number of optimization trials
        cv: Cross-validation folds
        random_state: Random seed

    Returns:
        Best parameters and study object
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna not installed. Install with: pip install optuna")

    def objective(trial):
        """Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Mean cross-validation R² score
        """
        if model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
            model = RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        return scores.mean()

    # Run optimization
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study


# ==============================================================================
# Category 6: Utilities and Model Comparison
# ==============================================================================


def validate_training_data(X: pd.DataFrame, y: pd.Series, strict: bool = True) -> Dict[str, Any]:
    """
    Validate training data before model fitting.

    This function implements Priority 1 from ML Workflow Improvement Plan:
    comprehensive validation gates to prevent NaN/Inf values from reaching model training.

    Args:
        X: Feature matrix
        y: Target vector
        strict: If True, raise exceptions on validation failures

    Returns:
        Dictionary with validation results:
        - valid: bool indicating if data passed all checks
        - nan_features: count of NaN values in features
        - nan_target: count of NaN values in target
        - inf_features: count of infinite values in features
        - inf_target: count of infinite values in target
        - zero_var_columns: list of zero-variance column names
        - issues: list of issue descriptions

    Raises:
        ValueError: If validation fails and strict=True

    Example:
        >>> X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> y_train = pd.Series([10, 20, 30])
        >>> result = validate_training_data(X_train, y_train, strict=True)
        >>> assert result['valid'] == True
    """
    issues = []

    # Check for empty data
    if len(X) == 0 or len(y) == 0:
        msg = "Feature matrix X or target vector y is empty"
        if strict:
            raise ValueError(f"{msg}. Cannot train on empty data.")
        issues.append(msg)

    # Check for NaN in features
    nan_count_X = X.isnull().sum().sum()
    if nan_count_X > 0:
        msg = f"Feature matrix X contains {nan_count_X} NaN values"
        if strict:
            raise ValueError(
                f"{msg}. Apply imputation before training. "
                f"Use finance_ml.ml_workflow.preprocessing.imputation.apply_enhanced_imputation_strategy_6step()"
            )
        issues.append(msg)

    # Check for NaN in target
    nan_count_y = y.isnull().sum()
    if nan_count_y > 0:
        msg = f"Target vector y contains {nan_count_y} NaN values"
        if strict:
            raise ValueError(f"{msg}. Remove or impute target NaN before training.")
        issues.append(msg)

    # Check for infinite values in features
    inf_count_X = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    if inf_count_X > 0:
        msg = f"Feature matrix X contains {inf_count_X} infinite values"
        if strict:
            raise ValueError(f"{msg}. Replace infinite values before training.")
        issues.append(msg)

    # Check for infinite values in target
    inf_count_y = np.isinf(y).sum()
    if inf_count_y > 0:
        msg = f"Target vector y contains {inf_count_y} infinite values"
        if strict:
            raise ValueError(f"{msg}. Replace infinite values in target.")
        issues.append(msg)

    # Check for zero-variance columns (warning, not blocker)
    zero_var_cols = X.columns[X.var() == 0].tolist()
    if len(zero_var_cols) > 0:
        msg = f"Feature matrix X contains {len(zero_var_cols)} zero-variance columns: {zero_var_cols[:5]}"
        issues.append(msg)

    return {
        "valid": len(issues) == 0 or (len(issues) == 1 and len(zero_var_cols) > 0),
        "nan_features": nan_count_X,
        "nan_target": nan_count_y,
        "inf_features": inf_count_X,
        "inf_target": inf_count_y,
        "zero_var_columns": zero_var_cols,
        "issues": issues,
    }


def prepare_features_for_training(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    apply_imputation: bool = True,
    sector_column: str = "sector",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for model training with final imputation checkpoint.

    This function implements Priority 3 from ML Workflow Improvement Plan:
    pre-model training imputation checkpoint to ensure zero NaN values.

    Args:
        df: Input DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        apply_imputation: If True, apply 6-step imputation before extraction
        sector_column: Sector column for KNN imputation

    Returns:
        Tuple of (X, y) ready for model training with zero NaN

    Example:
        >>> df = pd.DataFrame({
        ...     'sector': ['Tech', 'Finance'],
        ...     'market_cap': [1e9, np.nan],
        ...     'last_price': [100, 150],
        ...     'price_target': [110, 160]
        ... })
        >>> X, y = prepare_features_for_training(
        ...     df, ['market_cap'], 'price_target',
        ...     apply_imputation=True, sector_column='sector'
        ... )
        >>> assert X.isnull().sum().sum() == 0
    """
    from finance_ml.ml_workflow.preprocessing.imputation import (
        apply_enhanced_imputation_strategy_6step,
    )

    # Extract target BEFORE imputation to preserve NaN for removal
    y = df[target_col].copy()

    # Drop rows with NaN in target
    valid_mask = ~y.isnull()
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        logger.warning(f"Dropping {n_dropped} rows with NaN target values")
        df = df[valid_mask].copy()
        y = y[valid_mask]

    # Apply final imputation if requested (only on features, target already extracted)
    if apply_imputation:
        logger.info("Applying final imputation before feature extraction...")
        df = apply_enhanced_imputation_strategy_6step(
            df,
            sector_column=sector_column,
            n_neighbors=5,
            price_column="last_price" if "last_price" in df.columns else None,
        )

    # Extract features after imputation
    X = df[feature_cols].copy()

    # Final validation - handle any residual NaN/Inf
    nan_X = X.isnull().sum().sum()
    nan_y = y.isnull().sum()
    inf_X = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    inf_y = np.isinf(y).sum()

    if nan_X > 0 or nan_y > 0 or inf_X > 0 or inf_y > 0:
        logger.error(
            f"Features have {nan_X} NaN, {inf_X} Inf; target has {nan_y} NaN, {inf_y} Inf after preparation"
        )
        # Emergency fallback: fill with 0
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan)
        y = y.fillna(y.median() if pd.notna(y.median()) else 0)
        logger.warning("Applied emergency fillna(0) to ensure training can proceed")

    logger.info(f"✓ Features prepared: {X.shape}, target: {y.shape}, zero NaN confirmed")

    return X, y


# Note: validate_training_data is defined above at line 1288


def compare_regressors(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    cv: int = 5,
    random_state: int = 42,
    ensure_nonnegative: bool = False,
    loss: str = "squared_error",
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple regression regression.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Test set proportion
        cv: Cross-validation folds
        random_state: Random seed
        ensure_nonnegative: If True, wrap regression with NonNegativeRegressionWrapper
                           to ensure predictions >= 0
        loss: Loss function for GradientBoosting ('squared_error', 'huber', 'absolute_error')
              If 'huber', uses robust loss for outlier handling (Model Optimization Priority 2.1)

    Returns:
        Dictionary of model results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Validate training data (ML Workflow Improvement Plan Priority 1)
    try:
        validation_result = validate_training_data(X_train, y_train, strict=False)
        if not validation_result["valid"]:
            logger.warning(
                f"Training data validation issues detected: {validation_result['issues']}"
            )
            # Log specific counts
            if validation_result["nan_features"] > 0:
                logger.warning(f"  NaN in features: {validation_result['nan_features']}")
            if validation_result["inf_features"] > 0:
                logger.warning(f"  Inf in features: {validation_result['inf_features']}")

            # Apply emergency imputation
            from sklearn.impute import SimpleImputer

            logger.info("Applying emergency median imputation to X_train and X_test...")
            imputer = SimpleImputer(strategy="median")
            X_train = pd.DataFrame(
                imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
            )
            X_test = pd.DataFrame(
                imputer.transform(X_test), columns=X_test.columns, index=X_test.index
            )
            logger.info("Emergency imputation completed")
    except Exception as e:
        logger.error(f"Validation check failed: {e}")
        # Continue anyway but log the issue
        pass

    results = {}

    # Define regression to compare
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.1, random_state=random_state, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=50, random_state=random_state, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=50, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            loss=loss,
            alpha=0.9 if loss == "huber" else 0.9,  # Quantile for Huber transition
            n_estimators=50,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=50, random_state=random_state
        ),
    }

    # Wrap regression with NonNegativeRegressionWrapper if requested
    if ensure_nonnegative:
        models = {name: NonNegativeRegressionWrapper(model) for name, model in models.items()}

    # Train and evaluate each model with graceful error handling
    # (ML Workflow Improvement Plan Priority 2: Graceful Model Fallback)
    for name, model in models.items():
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Metrics
            results[name] = {
                "mae": mean_absolute_error(y_test, y_pred_test),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "r2": r2_score(y_test, y_pred_test),
                "train_r2": r2_score(y_train, y_pred_train),
                "train_time": train_time,
                "status": "success",
            }
            logger.info(f"✓ {name} trained successfully (MAE: {results[name]['mae']:.2f})")

        except ValueError as e:
            # Handle NaN-related errors gracefully
            error_msg = str(e)
            if "NaN" in error_msg or "missing values" in error_msg or "Input contains" in error_msg:
                logger.warning(f"Model {name} failed due to data quality issue: {error_msg}")
                results[name] = {
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "train_r2": np.nan,
                    "train_time": 0,
                    "status": "failed_data_quality",
                    "error": error_msg[:200],  # Truncate long error messages
                }
            else:
                # Re-raise unexpected ValueError
                logger.error(f"Model {name} failed with unexpected ValueError: {e}")
                raise

        except Exception as e:
            # Log and continue with other regression
            logger.error(f"Model {name} failed with unexpected error: {e}")
            results[name] = {
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "train_r2": np.nan,
                "train_time": 0,
                "status": "failed_other",
                "error": str(e)[:200],
            }

    # Check if at least one model succeeded
    successful_models = {k: v for k, v in results.items() if v.get("status") == "success"}

    if len(successful_models) == 0:
        logger.error("All regression failed to train. Check data quality and preprocessing.")
        raise RuntimeError(
            "All regression regression failed. Data validation and imputation required. "
            f"Failed regression: {list(results.keys())}"
        )

    if len(successful_models) < len(models):
        failed_models = [k for k, v in results.items() if v.get("status") != "success"]
        logger.warning(
            f"{len(successful_models)}/{len(models)} regression trained successfully. "
            f"Failed: {failed_models}"
        )
    else:
        logger.info(f"✓ All {len(models)} regression trained successfully")

    return results


def extract_numeric_feature_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """
    Extract numeric feature columns from DataFrame, excluding targets and metadata.

    This utility function identifies all numeric columns in a DataFrame and filters
    out common non-feature columns like identifiers, targets, and event labels.

    Args:
        df: Input DataFrame
        exclude_cols: Explicit list of column names to exclude (default: None)
        exclude_patterns: List of substring patterns to match for exclusion
            (default: ['event_proba_', 'event_label'])

    Returns:
        List of numeric column names suitable for model training

    Default Exclusions:
        - Identifier columns: 'ticker', 'isin', 'name', 'description'
        - Categorical columns: 'sector', 'region', 'industry', 'country'
        - Target columns: 'price_target', 'analyst_target_price'
        - Event-related columns: 'event_label', 'event_proba_*'
        - Any custom columns in exclude_cols parameter

    Examples:
        >>> df = pd.DataFrame({
        ...     'ticker': ['AAPL', 'MSFT'],
        ...     'sector': ['Tech', 'Tech'],
        ...     'last_price': [150.0, 300.0],
        ...     'market_cap': [2.5e12, 2.3e12],
        ...     'price_target': [180.0, 350.0]
        ... })
        >>> features = extract_numeric_feature_columns(df)
        >>> # Returns: ['last_price', 'market_cap']
        >>> # (excludes ticker, sector, price_target)

        >>> # Custom exclusions
        >>> features = extract_numeric_feature_columns(
        ...     df, exclude_cols=['last_price', 'price_target']
        ... )
        >>> # Returns: ['market_cap']
    """
    if df.empty:
        logger.info("DataFrame is empty, returning empty feature list")
        return []

    # Default exclusion set
    default_exclude = {
        # Identifiers
        "ticker",
        "isin",
        "name",
        "description",
        # Categorical grouping columns (even if accidentally numeric)
        "sector",
        "region",
        "industry",
        "country",
        "trading_country",
        # Common target columns
        "price_target",
        "analyst_target_price",
        "price_target_median",
        # Event classification outputs
        "event_label",
    }

    # Combine with user-provided exclusions
    if exclude_cols:
        default_exclude.update(exclude_cols)

    # Default patterns to exclude
    if exclude_patterns is None:
        exclude_patterns = ["event_proba_"]

    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    logger.info(f"DataFrame analysis: {len(df.columns)} total columns, {len(numeric_cols)} numeric")

    # Filter out excluded columns and patterns
    feature_cols = []
    for col in numeric_cols:
        # Check explicit exclusions
        if col in default_exclude:
            continue

        # Check pattern exclusions
        if any(pattern in col for pattern in exclude_patterns):
            continue

        feature_cols.append(col)

    logger.info(
        f"Extracted {len(feature_cols)} numeric feature columns "
        f"(excluded {len(numeric_cols) - len(feature_cols)} columns)"
    )

    if len(feature_cols) == 0:
        logger.warning("No numeric feature columns found after exclusions")
    else:
        logger.debug(
            f"Feature columns: {feature_cols[:10]}"
            + (f" ... and {len(feature_cols) - 10} more" if len(feature_cols) > 10 else "")
        )

    return feature_cols


def train_sector_specific_models(
    df: pd.DataFrame,
    feature_cols: Union[List[str], Dict[str, List[str]]],
    target_col: str,
    sector_col: str = "sector",
    model_type: str = "random_forest",
    random_state: int = 42,
    min_samples: int = 20,
    ensure_nonnegative: bool = False,
    auto_extract_fallback: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train separate regression for each sector.

    Args:
        df: Input DataFrame
        feature_cols: Feature column names. Accepts a list of column names or a dict
            with keys like 'all_features', 'numeric_features', 'categorical_features',
            and 'classification_features'. If a dict is provided, the function will
            try 'all_features' first, otherwise combine available groups.
        target_col: Target column name
        sector_col: Sector column name
        model_type: Model type to train
        random_state: Random seed
        min_samples: Minimum samples required per sector (default: 20)
        ensure_nonnegative: If True, wrap regression with NonNegativeRegressionWrapper
                           to ensure predictions >= 0
        auto_extract_fallback: If True, automatically extract numeric features from
                              DataFrame when provided feature_cols are invalid or missing.
                              Uses extract_numeric_feature_columns() to identify suitable
                              features (default: False)

    Returns:
        Tuple of (sector_models, results):
        - sector_models: Dictionary mapping sector names to trained regression
        - results: Dictionary with metrics and metadata

    Raises:
        ValueError: If no valid features remain after validation against df and
                   auto_extract_fallback is False
    """
    # DataFrame structure diagnostics
    logger.info("=" * 60)
    logger.info("TRAIN SECTOR-SPECIFIC MODELS - DataFrame Diagnostics")
    logger.info("=" * 60)
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Total columns: {len(df.columns)}")

    # Analyze column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    logger.info(f"  Numeric columns: {len(numeric_cols)}")
    logger.info(f"  Object columns: {len(object_cols)}")

    # Check for target and sector columns
    if target_col in df.columns:
        logger.info(f"  ✓ Target column '{target_col}' present")
    else:
        logger.warning(f"  ⚠ Target column '{target_col}' NOT FOUND")

    if sector_col in df.columns:
        n_sectors = df[sector_col].nunique()
        logger.info(f"  ✓ Sector column '{sector_col}' present ({n_sectors} unique sectors)")
    else:
        logger.warning(f"  ⚠ Sector column '{sector_col}' NOT FOUND")

    logger.info("=" * 60)

    # Smart handling of feature_cols
    actual_feature_cols: List[str]
    if isinstance(feature_cols, dict):
        logger.info("feature_cols is a dict, extracting feature list...")
        all_key = feature_cols.get("all_features")
        if all_key:
            actual_feature_cols = list(all_key)
            logger.info(f"  Using 'all_features' key: {len(actual_feature_cols)} features")
        else:
            combined: List[str] = []
            for key in ["numeric_features", "categorical_features", "classification_features"]:
                vals = feature_cols.get(key, [])
                if vals:
                    combined.extend(list(vals))
            actual_feature_cols = combined
            logger.info(f"  Combined feature types: {len(actual_feature_cols)} features")
        # Deduplicate while preserving order
        before = len(actual_feature_cols)
        actual_feature_cols = list(dict.fromkeys(actual_feature_cols))
        if len(actual_feature_cols) != before:
            logger.info(f"  After deduplication: {len(actual_feature_cols)} features")
        else:
            logger.info(f"  After deduplication: {len(actual_feature_cols)} features")
    elif isinstance(feature_cols, list):
        actual_feature_cols = feature_cols
        logger.info(f"feature_cols is already a list: {len(actual_feature_cols)} features")
    else:
        # Attempt a graceful conversion (e.g., pandas Index or numpy array)
        try:
            actual_feature_cols = list(feature_cols)  # type: ignore[arg-type]
            logger.info(
                f"feature_cols provided as {type(feature_cols).__name__}; converted to list with "
                f"{len(actual_feature_cols)} features"
            )
        except Exception as e:
            raise TypeError(
                f"feature_cols must be a list or dict of lists; got {type(feature_cols).__name__}"
            ) from e

    # Basic empty check
    if len(actual_feature_cols) == 0:
        raise ValueError("feature_cols cannot be empty")

    # Validate that feature columns exist in the DataFrame; skip missing with warning
    available_features = [c for c in actual_feature_cols if c in df.columns]
    missing_features = [c for c in actual_feature_cols if c not in df.columns]

    if missing_features:
        msg = (
            f"⚠ Warning: {len(missing_features)} features not in DataFrame (will be skipped). "
            f"Missing: {missing_features[:5]}..."
            if len(missing_features) > 5
            else f"⚠ Warning: {len(missing_features)} features not in DataFrame (will be skipped): {missing_features}"
        )
        logger.warning(msg)

    actual_feature_cols = available_features

    if len(actual_feature_cols) == 0:
        # Try auto-extraction fallback if enabled
        if auto_extract_fallback:
            logger.warning("No valid features from input, attempting auto-extraction...")
            actual_feature_cols = extract_numeric_feature_columns(
                df, exclude_cols=[target_col, sector_col]
            )

            if len(actual_feature_cols) > 0:
                logger.info(
                    f"✓ Auto-extracted {len(actual_feature_cols)} numeric features from DataFrame"
                )
                logger.info(f"  First 10 features: {actual_feature_cols[:10]}")
            else:
                # Still no features after auto-extraction
                error_msg = (
                    "❌ No valid feature columns found even after auto-extraction.\n"
                    f"  DataFrame has {len(df.columns)} columns total:\n"
                    f"    - {len(numeric_cols)} numeric columns\n"
                    f"    - {len(object_cols)} object/categorical columns\n"
                    f"  Tried to exclude: {target_col}, {sector_col}\n"
                    f"  Available columns: {list(df.columns)[:20]}"
                    + ("..." if len(df.columns) > 20 else "")
                )
                raise ValueError(error_msg)
        else:
            # Auto-extraction not enabled, provide detailed error
            sample_cols = list(df.columns)[:20]
            error_msg = (
                "❌ No valid feature columns remain after validation against DataFrame.\n"
                f"  Requested features: {len(actual_feature_cols + missing_features)} "
                f"(0 valid, {len(missing_features)} missing)\n"
                f"  DataFrame columns ({len(df.columns)} total): {sample_cols}"
                + ("..." if len(df.columns) > 20 else "")
                + f"\n  Missing features: {missing_features[:10]}"
                + ("..." if len(missing_features) > 10 else "")
                + "\n\n💡 Tip: Set auto_extract_fallback=True to automatically extract "
                "numeric features from the DataFrame."
            )
            raise ValueError(error_msg)

    logger.info(f"✓ Final feature count for sector regression: {len(actual_feature_cols)}")

    # ============================================================================
    # VALIDATE AND CLEAN TARGET COLUMN
    # ============================================================================
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Drop rows with NaN in target before training
    nan_target_count = df[target_col].isna().sum()
    if nan_target_count > 0:
        logger.warning(
            f"⚠ Target column '{target_col}' contains {nan_target_count} NaN values. "
            f"Dropping these rows before training."
        )
        df = df[df[target_col].notna()].copy()
        logger.info(f"✓ After dropping NaN targets: {len(df)} rows remaining")

    sector_models: Dict[str, Any] = {}
    sector_metrics: Dict[str, Any] = {}

    sectors = df[sector_col].unique()

    for sector in sectors:
        # Filter data for sector
        sector_df = df[df[sector_col] == sector]

        if len(sector_df) < min_samples:  # Skip sectors with too few samples
            continue

        # Apply preprocessing with imputation to handle NaN values
        # This ensures clean data before training sector-specific regression
        try:
            X_sector, y_sector = prepare_features_for_training(
                df=sector_df,
                feature_cols=actual_feature_cols,
                target_col=target_col,
                apply_imputation=True,
                sector_column=sector_col,
            )
        except Exception as e:
            logger.warning(
                f"⚠ Failed to prepare features for sector '{sector}': {e}. Skipping this sector."
            )
            continue

        # Train model
        if model_type == "random_forest":
            model, metrics = train_random_forest_regressor(
                X_sector, y_sector, n_estimators=50, random_state=random_state
            )
        else:
            # For ridge, pass ensure_nonnegative if supported
            results_dict = train_ridge_regressor(
                X_sector, y_sector, random_state=random_state, ensure_nonnegative=ensure_nonnegative
            )
            model = results_dict["model"]
            metrics = results_dict

        # Wrap with NonNegativeRegressionWrapper if requested and not already wrapped
        if ensure_nonnegative and model_type == "random_forest":
            model = NonNegativeRegressionWrapper(model)

        sector_models[sector] = model
        sector_metrics[sector] = metrics

    results = {
        "sector_metrics": sector_metrics,
        "n_sectors": len(sector_models),
        "ensure_nonnegative": ensure_nonnegative,
    }

    return sector_models, results


def save_model(
    model: Any, filepath: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save trained model with metadata.

    Args:
        model: Trained model
        filepath: Save path
        metadata: Optional metadata dictionary
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {"model": model, "metadata": metadata or {}}

    joblib.dump(save_dict, filepath)


def load_model(filepath: Union[str, Path]) -> Tuple[Any, Dict[str, Any]]:
    """
    Load trained model with metadata.

    Args:
        filepath: Model file path

    Returns:
        Loaded model and metadata
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    save_dict = joblib.load(filepath)

    return save_dict["model"], save_dict.get("metadata", {})


def standardize_comparison_results(results: "pd.DataFrame | dict | None") -> pd.DataFrame:
    """
    Convert comparison results (dict-of-dicts or DataFrame) into a standardized
    DataFrame with a 'Model' column and normalized metric names.

    - Accepts dict like {"Ridge": {"mae": 1.2, "rmse": 2.3, "r2": 0.8, ...}, ...}
    - Accepts a DataFrame with columns in any case (e.g., 'model', 'mae', ...)
    - Returns a DataFrame with at least ['Model', 'MAE', 'RMSE', 'R2'] columns
    - Sorts by 'MAE' ascending if available (lower is better)

    This utility is designed to eliminate KeyError: 'Model' by providing a consistent
    tabular structure regardless of the original output shape from model comparison.
    """
    import pandas as pd  # local import to avoid issues if pandas unavailable at import time

    if results is None:
        return pd.DataFrame(columns=["Model", "MAE", "RMSE", "R2"])

    # Case 1: dict-of-dicts
    if isinstance(results, dict):
        rows = []
        for model_name, metrics in results.items():
            row = {"Model": model_name}
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    key = str(k).strip().lower()
                    normalized = {
                        "mae": "MAE",
                        "rmse": "RMSE",
                        "r2": "R2",
                        "train_r2": "Train_R2",
                        "train_time": "Train_Time",
                        "status": "Status",
                        "error": "Error",
                        "model": "Model",
                    }.get(key, k)
                    row[normalized] = v
            rows.append(row)
        df = pd.DataFrame(rows)

    # Case 2: already a DataFrame
    elif isinstance(results, pd.DataFrame):
        df = results.copy()
        # Normalize model column name (case-insensitive)
        model_col = next((c for c in df.columns if str(c).lower() == "model"), None)
        if model_col and model_col != "Model":
            df = df.rename(columns={model_col: "Model"})
        # Normalize metric column names
        rename_map = {}
        for c in list(df.columns):
            lc = str(c).lower()
            if lc in ("mae", "rmse", "r2", "train_r2", "train_time"):
                rename_map[c] = {
                    "mae": "MAE",
                    "rmse": "RMSE",
                    "r2": "R2",
                    "train_r2": "Train_R2",
                    "train_time": "Train_Time",
                }[lc]
        if rename_map:
            df = df.rename(columns=rename_map)
    else:
        # Fallback: try constructing DataFrame and then normalize
        df = pd.DataFrame(results)

    # Ensure expected columns exist
    for col in ["Model", "MAE", "RMSE", "R2"]:
        if col not in df.columns:
            df[col] = np.nan

    # Sort by MAE if available
    try:
        if "MAE" in df.columns:
            df = df.sort_values("MAE", ascending=True, kind="mergesort").reset_index(drop=True)
    except Exception:
        # Do not fail sorting
        pass

    return df
