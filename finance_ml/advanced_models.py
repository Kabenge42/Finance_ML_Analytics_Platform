"""
Phase 9.5: Advanced Regression Models with Classification Features

This module implements sector-optimized regression models enhanced with classification
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
"""

import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import joblib
import numpy as np
import pandas as pd
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
    Wrapper for regression models that ensures predictions are non-negative.
    
    This wrapper clips predictions to be >= 0, which is essential for price
    target predictions since stock prices cannot be negative. Linear models
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
        issue of negative price target predictions observed in production models.
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
            performance overhead. For most financial models, less than 5% of
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
        
        This allows accessing base model attributes like coef_, intercept_, etc.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute value from base model
        """
        return getattr(self.base_model, name)


# ==============================================================================
# Category 1: Feature Integration
# ==============================================================================


def extract_classification_features(probabilities: np.ndarray) -> pd.DataFrame:
    """
    Extract classification features from predicted probabilities.
    
    This function converts raw classifier probabilities into structured features
    that can be used as inputs for regression models. The classification features
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
        regression models, as specified in the Phase 9.5 requirements for
        sector-optimized regression with classification feature enhancement.
    """
    import logging
    
    if probabilities.shape[1] != 3:
        raise ValueError(f"Expected 3 classes, got {probabilities.shape[1]}")
    
    n_samples = probabilities.shape[0]
    logging.debug(
        f"Extracting classification features for {n_samples} samples"
    )
    
    features = pd.DataFrame({
        'event_prob_neutral': probabilities[:, 0],
        'event_prob_positive': probabilities[:, 1],
        'event_prob_negative': probabilities[:, 2],
        'event_class_predicted': probabilities.argmax(axis=1),
        'event_confidence': probabilities.max(axis=1),
    })
    
    # Log summary statistics
    avg_confidence = features['event_confidence'].mean()
    class_distribution = features['event_class_predicted'].value_counts()
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
    training regression models with classification feature enhancement.
    
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
        enabling sector-optimized regression models to leverage event classifier
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
    
    logging.debug(
        f"Integration complete: {len(result)} rows, {len(result.columns)} total columns"
    )
    
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

    # Feature info
    feature_info = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classification_features": classification_features,
        "all_features": feature_cols,
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
        "n_zero_coefs": len(best_model.base_model.coef_ if ensure_nonnegative else best_model.coef_) - n_nonzero,
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
    """
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
    # Define base models
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

    Returns:
        Trained ensemble (wrapped if ensure_nonnegative=True) and results dictionary
    """
    # Define base models
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)),
        ("et", ExtraTreesRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=50, random_state=random_state)),
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
        List of trained models (one per quantile) and results dictionary.
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


def compare_regressors(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    cv: int = 5,
    random_state: int = 42,
    ensure_nonnegative: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple regression models.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Test set proportion
        cv: Cross-validation folds
        random_state: Random seed
        ensure_nonnegative: If True, wrap models with NonNegativeRegressionWrapper
                           to ensure predictions >= 0

    Returns:
        Dictionary of model results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = {}

    # Define models to compare
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.1, random_state=random_state, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=50, random_state=random_state, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=50, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=50, random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=50, random_state=random_state
        ),
    }

    # Wrap models with NonNegativeRegressionWrapper if requested
    if ensure_nonnegative:
        models = {name: NonNegativeRegressionWrapper(model) for name, model in models.items()}

    # Train and evaluate each model
    for name, model in models.items():
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
        }

    return results


def train_sector_specific_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sector_col: str = "sector",
    model_type: str = "random_forest",
    random_state: int = 42,
    min_samples: int = 20,
    ensure_nonnegative: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train separate models for each sector.

    Args:
        df: Input DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        sector_col: Sector column name
        model_type: Model type to train
        random_state: Random seed
        min_samples: Minimum samples required per sector (default: 20)
        ensure_nonnegative: If True, wrap models with NonNegativeRegressionWrapper
                           to ensure predictions >= 0

    Returns:
        Dictionary of sector models and results
    """
    sector_models = {}
    sector_metrics = {}

    sectors = df[sector_col].unique()

    for sector in sectors:
        # Filter data for sector
        sector_df = df[df[sector_col] == sector]

        if len(sector_df) < min_samples:  # Skip sectors with too few samples
            continue

        X_sector = sector_df[feature_cols]
        y_sector = sector_df[target_col]

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
            model = results_dict['model']
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
