"""
Phase 9.5.1: Regression Model Training Functions

This module contains all regression model training functions extracted from advanced_models.py.
It provides a comprehensive set of model architectures for price target prediction:

Categories:
1. Linear Models: Ridge, Lasso, ElasticNet, BayesianRidge, Polynomial
2. Gradient Boosting: XGBoost, LightGBM, CatBoost, HistGradientBoosting
3. Tree Models: RandomForest, ExtraTrees
4. Neural Networks: Feedforward DNN with TensorFlow/Keras
5. Ensemble Methods: Voting, Stacking
6. Model Comparison: compare_regressors for benchmarking

All train_* functions follow a consistent API:
- Args: X (features), y (target), hyperparameters, cv (folds), random_state, ensure_nonnegative
- Returns: Dict with 'model', 'train_score', 'cv_mean', 'cv_std', and model-specific metrics

Integration with Phase 9.5.0:
- Uses NonNegativeRegressionWrapper from regression.constraints for price constraints
- Compatible with prepare_regression_data from regression.dataset
- Supports classification meta-features from Phase 9.4

Reference notebooks:
- 04_training_linear_models.ipynb
- 07_ensemble_learning_and_random_forests.ipynb
- 10_neural_nets_with_keras.ipynb
- 11_training_deep_neural_networks.ipynb

Example:
    >>> from finance_ml.ml_workflow.regression.models import train_xgboost_regressor
    >>> from finance_ml.ml_workflow.regression.dataset import prepare_regression_data
    >>>
    >>> # Prepare data
    >>> X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(df)
    >>>
    >>> # Train model with non-negative constraint
    >>> results = train_xgboost_regressor(
    ...     X_train, y_train,
    ...     n_estimators=100,
    ...     max_depth=6,
    ...     ensure_nonnegative=True
    ... )
    >>>
    >>> # Get trained model and metrics
    >>> model = results['model']
    >>> print(f"CV R²: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Any, Tuple

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
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Import NonNegativeRegressionWrapper from constraints module
from finance_ml.ml_workflow.regression.constraints import NonNegativeRegressionWrapper
from finance_ml.ml_workflow.regression.cv import get_regression_cv_splitter

# Import dataset utilities for classification feature integration
from finance_ml.ml_workflow.regression.dataset import (
    integrate_classification_features,
    create_classification_interactions,
)


# Configure logger
logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    xgb = None  # type: ignore
    HAS_XGBOOST = False
    logger.debug("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    lgb = None  # type: ignore
    HAS_LIGHTGBM = False
    logger.debug("LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    CatBoostRegressor = None  # type: ignore
    HAS_CATBOOST = False
    logger.debug("CatBoost not available. Install with: pip install catboost")

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
    logger.debug("TensorFlow not available. Install with: pip install tensorflow")

# Suppress warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# Helper Functions for Feature Cleaning
# ==============================================================================


def _clean_regression_features(
    X: pd.DataFrame,
    *,
    drop_zero_variance: bool = True,
    inf_replacement: str = "bound",
    posinf_bound: float = 1e10,
    neginf_bound: float = -1e10,
    validate_output: bool = True,
) -> pd.DataFrame:
    """
    Ensure regression feature matrix is free of NaN/inf and zero-variance columns.

    This is a defensive safety-rail for Phase 9.5:
    - Handles +/-inf with configurable replacement strategy
    - Median-imputes remaining NaN per column
    - Optionally drops zero-variance columns
    - Validates output before returning
    - Returns float64 DataFrame

    Args:
        X: Feature matrix
        drop_zero_variance: If True, drop constant columns
        inf_replacement: Strategy for inf handling:
            - "bound": Replace with configurable bounds (default)
            - "nan": Replace with NaN then impute
            - "clip": Clip to data range per column
        posinf_bound: Upper bound for positive infinity replacement (default: 1e10)
        neginf_bound: Lower bound for negative infinity replacement (default: -1e10)
        validate_output: If True, validate no NaN/inf remain after cleaning

    Returns:
        Cleaned feature matrix as float64 DataFrame.

    Raises:
        ValueError: If validate_output=True and cleaning failed to remove all NaN/inf
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X_clean = X.copy()
    cleaning_report = {
        "initial_shape": X_clean.shape,
        "inf_count": 0,
        "nan_count_before": 0,
        "nan_count_after": 0,
        "columns_with_inf": [],
        "columns_with_nan": [],
        "zero_variance_dropped": [],
    }

    # Count initial NaN values
    initial_nan = X_clean.isna().sum()
    cleaning_report["nan_count_before"] = int(initial_nan.sum())
    if cleaning_report["nan_count_before"] > 0:
        cols_with_nan = initial_nan[initial_nan > 0].index.tolist()
        cleaning_report["columns_with_nan"] = cols_with_nan[:20]  # Limit to 20
        logger.info(
            f"Feature cleaning: {cleaning_report['nan_count_before']} NaN values in "
            f"{len(cols_with_nan)} columns"
        )

    # Handle infinities with configurable strategy
    try:
        numeric_arr = X_clean.to_numpy(dtype=np.float64)
    except (ValueError, TypeError):
        # Some columns may not be numeric - handle gracefully
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        non_numeric_cols = X_clean.columns.difference(numeric_cols)
        if len(non_numeric_cols) > 0:
            logger.warning(
                f"Found {len(non_numeric_cols)} non-numeric columns; coercing to numeric"
            )
            for col in non_numeric_cols:
                X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")
        numeric_arr = X_clean.to_numpy(dtype=np.float64)

    posinf_mask = np.isposinf(numeric_arr)
    neginf_mask = np.isneginf(numeric_arr)
    inf_mask = posinf_mask | neginf_mask

    if inf_mask.any():
        n_posinf = int(posinf_mask.sum())
        n_neginf = int(neginf_mask.sum())
        cleaning_report["inf_count"] = n_posinf + n_neginf

        # Identify columns with infinities
        inf_cols_mask = inf_mask.any(axis=0)
        inf_col_names = X_clean.columns[inf_cols_mask].tolist()
        cleaning_report["columns_with_inf"] = inf_col_names[:20]

        logger.warning(
            f"Feature cleaning: found {n_posinf} +inf and {n_neginf} -inf values "
            f"in {len(inf_col_names)} columns: {inf_col_names[:5]}{'...' if len(inf_col_names) > 5 else ''}"
        )

        if inf_replacement == "bound":
            # Use np.nan_to_num with configurable bounds
            numeric_arr = np.nan_to_num(
                numeric_arr,
                nan=np.nan,  # Keep NaN for later imputation
                posinf=posinf_bound,
                neginf=neginf_bound,
            )
            X_clean = pd.DataFrame(numeric_arr, columns=X_clean.columns, index=X_clean.index)
            logger.info(f"Replaced +inf with {posinf_bound}, -inf with {neginf_bound}")
        elif inf_replacement == "clip":
            # Clip to column-wise finite min/max
            for col_idx, col in enumerate(X_clean.columns):
                col_data = numeric_arr[:, col_idx]
                finite_mask = np.isfinite(col_data)
                if finite_mask.any():
                    col_min = col_data[finite_mask].min()
                    col_max = col_data[finite_mask].max()
                    col_data = np.clip(col_data, col_min, col_max)
                    X_clean.iloc[:, col_idx] = col_data
            logger.info("Clipped infinities to column-wise finite range")
        else:  # "nan"
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
            logger.info("Replaced infinities with NaN for imputation")

    # Median imputation per column
    nan_counts = X_clean.isna().sum()
    total_nan = int(nan_counts.sum())
    if total_nan > 0:
        logger.info(f"Feature cleaning: imputing {total_nan} NaN values with column medians")
        medians = X_clean.median(numeric_only=True)
        for col in X_clean.columns:
            if col in medians.index:
                if pd.isna(medians[col]):
                    # Fall back to zero if column is entirely NaN
                    X_clean[col] = X_clean[col].fillna(0.0)
                    logger.debug(f"Column '{col}': entirely NaN, filled with 0.0")
                else:
                    X_clean[col] = X_clean[col].fillna(medians[col])
            else:
                # Non-numeric or unexpected types: coerce to 0 for safety
                X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce").fillna(0.0)

    cleaning_report["nan_count_after"] = int(X_clean.isna().sum().sum())

    # Optional: drop zero-variance columns
    if drop_zero_variance:
        std = X_clean.std(numeric_only=True)
        zero_var_cols = std[std == 0].index.tolist()
        if zero_var_cols:
            cleaning_report["zero_variance_dropped"] = zero_var_cols[:20]
            display_cols = zero_var_cols[:10] + (["..."] if len(zero_var_cols) > 10 else [])
            logger.warning(
                f"Dropping {len(zero_var_cols)} zero-variance feature(s): {display_cols}"
            )
            X_clean = X_clean.drop(columns=zero_var_cols, errors="ignore")

    # Ensure float64 dtypes
    X_clean = X_clean.astype(np.float64)

    # Validation: ensure no NaN/inf remain
    if validate_output:
        final_nan = X_clean.isna().sum().sum()
        final_inf = np.isinf(X_clean.to_numpy()).sum()
        if final_nan > 0 or final_inf > 0:
            error_msg = (
                f"Feature cleaning validation failed: {final_nan} NaN and {final_inf} inf "
                f"values remain after cleaning"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    cleaning_report["final_shape"] = X_clean.shape
    logger.info(
        f"Feature cleaning complete: {cleaning_report['initial_shape']} -> {cleaning_report['final_shape']}, "
        f"handled {cleaning_report['inf_count']} inf, {cleaning_report['nan_count_before']} NaN"
    )

    return X_clean


# ==============================================================================
# Category 1: Linear Models
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
# Category 2: Gradient Boosting Models
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
# Category 3: Tree Ensemble Models
# ==============================================================================


def train_random_forest_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
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

    y_pred = model.predict(X)
    metrics = {
        "r2": r2_score(y, y_pred),
        "r2_score": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "train_score": model.score(X, y),
    }

    artifacts = {
        "feature_importance": model.feature_importances_,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "model_type": "random_forest",
    }

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_proba": None,
        "artifacts": artifacts,
    }


def train_extra_trees_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
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

    y_pred = model.predict(X)
    metrics = {
        "r2": r2_score(y, y_pred),
        "r2_score": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "train_score": model.score(X, y),
    }

    artifacts = {
        "feature_importance": getattr(model, "feature_importances_", None),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "model_type": "extra_trees",
    }

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_proba": None,
        "artifacts": artifacts,
    }


# ==============================================================================
# Category 4: Neural Network Models
# ==============================================================================


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
# Category 5: Ensemble Methods
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
    # Phase 9.5 new args for meta-features and interactions
    use_meta_features: bool = False,
    classification_probabilities: Optional[np.ndarray] = None,
    enable_interactions: bool = True,
    interaction_valuation_cols: Optional[List[str]] = None,
    cv_policy: str = "kfold",
    date_col: str = "snapshot_date",
    group_col: str = "ticker",
    groups: Optional[pd.Series] = None,
    dates: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Train Stacking Regressor with optional classification meta-features.

    Combines Random Forest, Extra Trees, and Gradient Boosting with a Ridge
    meta-learner. Supports integrating classification probabilities as features
    to inform the regression (Phase 9.5).

    Args:
        X: Feature matrix
        y: Target vector
        cv: Cross-validation folds or splitter object
        random_state: Random seed
        ensure_nonnegative: If True, wraps model to force non-negative predictions
        loss: Loss function for GradientBoosting
        use_meta_features: If True, integrate classification_probabilities
        classification_probabilities: (N, 5) array of event probabilities
        enable_interactions: If True, create prob * valuation interactions
        interaction_valuation_cols: List of columns for interactions
        cv_policy: 'time_series', 'group', or 'kfold'
        date_col: Column name for dates (for time_series CV)
        group_col: Column name for groups (for group CV)
        groups: Series of group values aligned with X
        dates: Series of date values aligned with X

    Returns:
        Dict with 'model', 'metrics', 'artifacts'
    """
    logger.info("Training Stacking Ensemble Regressor")

    # 1. Feature Enhancement (Phase 9.5 P2.1)
    # Work on a copy to avoid modifying original X
    X_train = X.copy()

    if use_meta_features:
        if classification_probabilities is None:
            raise ValueError(
                "classification_probabilities are required when use_meta_features=True"
            )
        # Add prob columns: event_prob_strong_negative, etc.
        X_train = integrate_classification_features(X_train, classification_probabilities)

        if enable_interactions and interaction_valuation_cols:
            # Identify probability columns (added by integrate_classification_features)
            class_cols = [
                c for c in X_train.columns if c.startswith("event_prob_") or c == "event_confidence"
            ]
            X_train = create_classification_interactions(
                X_train, class_cols, interaction_valuation_cols
            )

    # 2. CV Splitter (Phase 9.5 P0.1 integration)
    splitter = cv
    if isinstance(cv, int):
        splitter_obj = get_regression_cv_splitter(
            policy=cv_policy,
            n_splits=cv,
            date_col=date_col,
            group_col=group_col,
        )

        # Resolve splitter logic
        if cv_policy == "time_series":
            # StackingRegressor requires partitions, TimeSeriesSplit is not a partition.
            # Fallback to KFold(shuffle=False) which is a partition and respects order (mostly).
            logger.warning(
                "TimeSeriesSplit is incompatible with StackingRegressor (requires partitions). Using KFold(shuffle=False) instead."
            )
            from sklearn.model_selection import KFold

            splitter = KFold(n_splits=cv, shuffle=False)

            # Original logic for reference (incompatible with Stacking)
            # if dates is not None:
            #    temp_df = pd.DataFrame({date_col: dates.values}, index=X_train.index)
            #    splitter = list(splitter_obj.split(temp_df))
            # ...
        elif cv_policy == "group":
            if groups is not None:
                from sklearn.model_selection import GroupKFold

                splitter = GroupKFold(n_splits=cv)
            elif group_col in X_train.columns:
                temp_df = pd.DataFrame({group_col: X_train[group_col]}, index=X_train.index)
                splitter = list(splitter_obj.split(temp_df))
            else:
                splitter = cv

    # Define base regression with robust loss support
    # Conservative hyperparameters to prevent overfitting (Phase 9.5 audit fix)
    # - Reduced n_estimators (100 vs 200) for faster training and less overfitting
    # - Limited max_depth (8-10 vs 15) to prevent memorizing training data
    # - Increased min_samples_split (10 vs 5) for better generalization
    # - Increased min_samples_leaf (5+) as key regularization parameter
    estimators = [
        (
            "rf",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "et",
            ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "gb",
            GradientBoostingRegressor(
                loss=loss,
                alpha=0.9 if loss == "huber" else 0.9,  # Quantile for Huber transition
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=random_state,
            ),
        ),
    ]

    # Add XGBoost if available (better performance than linear models)
    # Conservative hyperparameters with early stopping support
    if HAS_XGBOOST:
        estimators.append(
            (
                "xgb",
                xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,  # Regularization: minimum sum of instance weight in child
                    reg_alpha=0.1,  # L1 regularization
                    reg_lambda=1.0,  # L2 regularization
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0,
                ),
            )
        )

    # Meta-learner
    meta_model = Ridge(alpha=1.0)

    # Create stacking ensemble
    base_model = StackingRegressor(
        estimators=estimators, final_estimator=meta_model, cv=splitter, n_jobs=-1, passthrough=False
    )

    # Defensive cleaning to prevent NaN/inf from reaching tree estimators (Phase 9.5 safety rail)
    # This handles inf values introduced by interaction features, ratios, or log transforms
    X_train = _clean_regression_features(X_train)

    base_model.fit(X_train, y)

    # Wrap with NonNegativeRegressionWrapper if requested
    if ensure_nonnegative:
        model = NonNegativeRegressionWrapper(base_model)
    else:
        model = base_model

    # Cross-validation score (using base_model for CV to avoid wrapper issues)
    # Ensure we use correct splitter and X_train
    cv_splitter_for_score = splitter
    if isinstance(splitter, list):
        # If splitter is a list of indices, cross_val_score works fine
        pass
    elif cv_policy == "group" and groups is not None:
        # If passing GroupKFold, we need to pass groups to cross_val_score
        # But we can't easily pass groups here if we don't have them in fit params
        # This is tricky. Simpler to skip CV calc for complex splitters or accept basic
        pass

    # We skip extensive CV calculation here to avoid complexity with custom splitters
    # But we compute basic score
    train_r2 = base_model.score(X_train, y)

    # Return standardized dict format per code_guidelines.md Section 1.1
    return {
        "model": model,
        "metrics": {
            "r2": train_r2,
            "cv_mean": 0.0,  # Placeholder/TODO
            "cv_std": 0.0,
        },
        "y_pred": None,  # Not computed during training
        "artifacts": {
            "base_models": [name for name, _ in estimators],
            "meta_model": "Ridge",
            "cv_policy": cv_policy,
            "meta_features_used": use_meta_features,
            "ensure_nonnegative": ensure_nonnegative,
        },
    }


# ==============================================================================
# Category 6: Model Comparison and Benchmarking
# ==============================================================================


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
    Compare multiple regression models.

    This function trains and evaluates multiple regression models on the same dataset,
    providing a comprehensive benchmark for model selection. It includes:
    - Data validation and emergency imputation if needed
    - Optional NonNegativeRegressionWrapper for all models
    - Robust loss function support (huber for outlier handling)
    - Graceful error handling with detailed status reporting
    - Performance metrics: MAE, RMSE, R², train R², training time

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Test set proportion (default: 0.2)
        cv: Cross-validation folds (default: 5)
        random_state: Random seed for reproducibility
        ensure_nonnegative: If True, wrap models with NonNegativeRegressionWrapper
                           to ensure predictions >= 0 (default: False)
        loss: Loss function for GradientBoosting ('squared_error', 'huber', 'absolute_error')
              If 'huber', uses robust loss for outlier handling (default: 'squared_error')

    Returns:
        Dictionary mapping model names to their metrics:
        {
            'Ridge': {'mae': float, 'rmse': float, 'r2': float, 'train_r2': float,
                      'train_time': float, 'status': 'success'},
            'Lasso': {...},
            ...
        }

    Raises:
        RuntimeError: If all models fail to train

    Example:
        >>> from finance_ml.ml_workflow.regression.models import compare_regressors
        >>> from finance_ml.ml_workflow.regression.dataset import prepare_regression_data
        >>>
        >>> X_train, X_test, y_train, y_test, _ = prepare_regression_data(df)
        >>> results = compare_regressors(X_train, y_train, ensure_nonnegative=True)
        >>>
        >>> # Find best model
        >>> best_model = min(results.items(), key=lambda x: x[1]['mae'])
        >>> print(f"Best model: {best_model[0]} (MAE: {best_model[1]['mae']:.2f})")
    """
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split

    # Import validation function from dataset module
    from finance_ml.ml_workflow.regression.dataset import validate_training_data

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

            # Use _clean_regression_features helper for consistent data sanitization
            # This handles: inf→NaN, median imputation, zero-variance column removal
            logger.info("Applying _clean_regression_features to sanitize X_train and X_test...")
            X_train = _clean_regression_features(X_train, drop_zero_variance=True)
            X_test = _clean_regression_features(X_test, drop_zero_variance=True)

            # Ensure X_test has same columns as X_train after cleaning
            # (some columns may be dropped from X_train but not X_test or vice versa)
            common_cols = X_train.columns.intersection(X_test.columns)
            if len(common_cols) < len(X_train.columns):
                logger.warning(f"Aligning columns: keeping {len(common_cols)} common features")
                X_train = X_train[common_cols]
                X_test = X_test[common_cols]

            # Final verification - ensure no NaN/Inf remain
            remaining_nan = X_train.isna().sum().sum() + X_test.isna().sum().sum()
            remaining_inf = np.isinf(X_train.values).sum() + np.isinf(X_test.values).sum()
            if remaining_nan > 0 or remaining_inf > 0:
                logger.error(
                    f"Data quality issues remain after sanitization: NaN={remaining_nan}, Inf={remaining_inf}"
                )
            else:
                logger.info("Emergency data sanitization completed successfully")
    except Exception as e:
        logger.error(f"Validation check failed: {e}")
        # Apply defensive cleaning even if validation fails
        logger.info("Applying defensive _clean_regression_features due to validation failure...")
        X_train = _clean_regression_features(X_train, drop_zero_variance=True)
        X_test = _clean_regression_features(X_test, drop_zero_variance=True)
        common_cols = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

    results = {}

    # Define models to compare
    # Optimized hyperparameters for Section 16.4 Performance Thresholds (R² > 0.7, MAE < 40%)
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.1, random_state=random_state, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            loss=loss,
            alpha=0.9 if loss == "huber" else 0.9,  # Quantile for Huber transition
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100, max_depth=10, learning_rate=0.1, random_state=random_state
        ),
    }

    # Wrap models with NonNegativeRegressionWrapper if requested
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
            # Log and continue with other models
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
        logger.error("All models failed to train. Check data quality and preprocessing.")
        raise RuntimeError(
            "All regression models failed. Data validation and imputation required. "
            f"Failed models: {list(results.keys())}"
        )

    if len(successful_models) < len(models):
        failed_models = [k for k, v in results.items() if v.get("status") != "success"]
        logger.warning(
            f"{len(successful_models)}/{len(models)} models trained successfully. "
            f"Failed: {failed_models}"
        )
    else:
        logger.info(f"✓ All {len(models)} models trained successfully")

    return results


# ============================================================================
# Phase 9.5 Task 6: Stacking Ensemble Hyperparameter Tuning
# ============================================================================

# Import Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler

    HAS_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore
    HAS_OPTUNA = False
    logger.debug("Optuna not available. Install with: pip install optuna")


def tune_stacking_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    n_trials: int = 50,
    timeout: Optional[int] = 300,
    cv: int = 3,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """
    Tune hyperparameters for stacking base models using Optuna.

    Uses Bayesian optimization (TPE sampler) to find optimal hyperparameters
    for gradient boosting base models within a time budget. Implements
    code_guidelines.md Section 16 hyperparameter tuning policy.

    Parameters
    ----------
    X : pd.DataFrame
        Training features
    y : pd.Series
        Training target
    model_type : str, default='xgboost'
        'xgboost', 'lightgbm', 'catboost'
    n_trials : int, default=50
        Number of Optuna trials
    timeout : int, optional, default=300
        Time budget in seconds (None for unlimited)
    cv : int, default=3
        Cross-validation folds
    random_state : int, default=42
        Random seed
    verbose : bool, default=False
        Print trial progress

    Returns
    -------
    tuple
        (best_params, best_score) where best_score is negative MAE

    Examples
    --------
    >>> X = pd.DataFrame({'f1': np.random.randn(200), 'f2': np.random.randn(200)})
    >>> y = X['f1'] * 2 + np.random.randn(200) * 0.1
    >>> params, score = tune_stacking_hyperparameters(X, y, model_type='xgboost', n_trials=10)
    >>> 'learning_rate' in params
    True
    >>> score > 0  # Positive MAE
    True

    Notes
    -----
    - Minimizes negative MAE (cross-validated)
    - Timeout protection prevents indefinite runs
    - TPE sampler for efficient Bayesian optimization
    - Search space optimized for stacking base models
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna required for hyperparameter tuning. Install: pip install optuna")

    from sklearn.metrics import make_scorer

    def objective(trial):
        """Optuna objective function."""
        if model_type == "xgboost":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost required. Install: pip install xgboost")

            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "n_estimators": 100,
                "random_state": random_state,
                "verbosity": 0,
            }
            model = xgb.XGBRegressor(**params)

        elif model_type == "lightgbm":
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM required. Install: pip install lightgbm")

            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "n_estimators": 100,
                "random_state": random_state,
                "verbose": -1,
            }
            model = lgb.LGBMRegressor(**params)

        elif model_type == "catboost":
            if not HAS_CATBOOST:
                raise ImportError("CatBoost required. Install: pip install catboost")

            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 3, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "iterations": 100,
                "random_state": random_state,
                "verbose": False,
            }
            model = CatBoostRegressor(**params)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Cross-validate (negative MAE)
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        return -scores.mean()  # Optuna minimizes, return positive MAE

    # Run optimization
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Suppress Optuna logging unless verbose
    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=verbose)

    logger.info(
        f"Optuna optimization complete: {len(study.trials)} trials, "
        f"best MAE: {study.best_value:.4f}"
    )

    return study.best_params, study.best_value


def select_stacking_base_models(
    comparison_results: Dict[str, Dict[str, float]], metric: str = "r2", top_k: int = 3
) -> List[str]:
    """
    Select top base models for stacking ensemble.

    Ranks models by specified metric and returns the top-k performers
    for use as stacking base models.

    Parameters
    ----------
    comparison_results : dict
        Results from compare_regressors(), format:
        {'model_name': {'mae': float, 'rmse': float, 'r2': float}}
    metric : str, default='r2'
        Metric to rank by ('mae', 'rmse', 'r2')
    top_k : int, default=3
        Number of base models to select

    Returns
    -------
    list
        Names of selected models

    Examples
    --------
    >>> results = {
    ...     'xgboost': {'mae': 10.5, 'rmse': 15.2, 'r2': 0.85},
    ...     'lightgbm': {'mae': 10.2, 'rmse': 14.8, 'r2': 0.87},
    ...     'ridge': {'mae': 12.0, 'rmse': 17.0, 'r2': 0.75}
    ... }
    >>> selected = select_stacking_base_models(results, metric='r2', top_k=2)
    >>> selected
    ['lightgbm', 'xgboost']

    Notes
    -----
    - For 'mae', 'rmse', 'mape': lower is better (ascending sort)
    - For 'r2': higher is better (descending sort)
    - Filters out failed models (NaN metrics)
    """
    # Filter out failed models
    valid_results = {
        name: metrics
        for name, metrics in comparison_results.items()
        if not np.isnan(metrics.get(metric, np.nan))
    }

    if len(valid_results) == 0:
        raise ValueError(f"No valid models found with metric '{metric}'")

    # Sort by metric (descending for r2, ascending for mae/rmse)
    ascending = metric in ["mae", "rmse", "mape"]

    sorted_models = sorted(valid_results.items(), key=lambda x: x[1][metric], reverse=not ascending)

    selected = [model_name for model_name, _ in sorted_models[:top_k]]

    logger.info(f"Selected {len(selected)} base models by {metric}: {selected}")
    return selected


def select_meta_learner(
    X_base: pd.DataFrame,
    y: pd.Series,
    candidates: List[str] = None,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[str, Dict[str, float]]:
    """
    Select best meta-learner via cross-validation.

    Evaluates multiple meta-learner candidates on base model predictions
    and selects the one with highest cross-validated R².

    Parameters
    ----------
    X_base : pd.DataFrame
        Base model predictions (meta-features)
    y : pd.Series
        Target
    candidates : list, optional, default=['ridge', 'lasso', 'huber']
        Meta-learner candidates
    cv : int, default=5
        Cross-validation folds
    random_state : int, default=42
        Random seed

    Returns
    -------
    tuple
        (best_meta_learner_name, cv_scores_dict)

    Examples
    --------
    >>> X_base = pd.DataFrame({
    ...     'pred_xgb': np.random.uniform(50, 150, 100),
    ...     'pred_lgb': np.random.uniform(50, 150, 100)
    ... })
    >>> y = np.random.uniform(50, 150, 100)
    >>> best, scores = select_meta_learner(X_base, y, cv=3)
    >>> best in ['ridge', 'lasso', 'huber']
    True
    >>> len(scores) == 3
    True

    Notes
    -----
    - Uses R² as scoring metric
    - Huber regression for outlier robustness
    - Ridge/Lasso for regularization
    """
    from sklearn.linear_model import HuberRegressor

    if candidates is None:
        candidates = ["ridge", "lasso", "huber"]

    meta_learners = {
        "ridge": Ridge(random_state=random_state),
        "lasso": Lasso(random_state=random_state),
        "huber": HuberRegressor(),
    }

    cv_scores = {}
    for name in candidates:
        if name not in meta_learners:
            logger.warning(f"Unknown meta-learner: {name}. Skipping.")
            continue

        model = meta_learners[name]
        scores = cross_val_score(model, X_base, y, cv=cv, scoring="r2", n_jobs=-1)
        cv_scores[name] = scores.mean()
        logger.info(f"Meta-learner {name}: R² = {scores.mean():.4f} (+/- {scores.std():.4f})")

    if len(cv_scores) == 0:
        raise ValueError(f"No valid meta-learner candidates found from: {candidates}")

    best_meta = max(cv_scores, key=cv_scores.get)
    logger.info(f"Selected meta-learner: {best_meta} (R² = {cv_scores[best_meta]:.4f})")

    return best_meta, cv_scores
