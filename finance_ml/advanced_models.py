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
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# Suppress warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# Category 1: Feature Integration
# ==============================================================================


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
    X: pd.DataFrame, y: pd.Series, alpha: float = 1.0, cv: int = 5, random_state: int = 42
) -> Tuple[Ridge, Dict[str, Any]]:
    """
    Train Ridge regression with L2 regularization.

    Args:
        X: Feature matrix
        y: Target vector
        alpha: Regularization strength
        cv: Cross-validation folds
        random_state: Random seed

    Returns:
        Trained model and results dictionary
    """
    # Grid search for best alpha
    alphas = np.logspace(-2, 3, 20)
    param_grid = {"alpha": alphas}

    ridge = Ridge(random_state=random_state)
    grid_search = GridSearchCV(ridge, param_grid, cv=cv, scoring="r2", n_jobs=-1)
    grid_search.fit(X, y)

    # Train final model with best alpha
    best_model = Ridge(alpha=grid_search.best_params_["alpha"], random_state=random_state)
    best_model.fit(X, y)

    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="r2")

    results = {
        "train_score": best_model.score(X, y),
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "best_alpha": grid_search.best_params_["alpha"],
        "model_type": "ridge",
    }

    return best_model, results


def train_lasso_regressor(
    X: pd.DataFrame, y: pd.Series, alpha: float = 0.1, cv: int = 5, random_state: int = 42
) -> Tuple[Lasso, Dict[str, Any]]:
    """
    Train Lasso regression with L1 regularization for feature selection.

    Args:
        X: Feature matrix
        y: Target vector
        alpha: Regularization strength
        cv: Cross-validation folds
        random_state: Random seed

    Returns:
        Trained model and results dictionary
    """
    # Grid search for best alpha (wider range for sparse solutions)
    alphas = np.logspace(-3, 2, 20)
    param_grid = {"alpha": alphas}

    lasso = Lasso(random_state=random_state, max_iter=10000)
    grid_search = GridSearchCV(lasso, param_grid, cv=cv, scoring="r2", n_jobs=-1)
    grid_search.fit(X, y)

    # Train final model
    best_model = Lasso(
        alpha=grid_search.best_params_["alpha"], random_state=random_state, max_iter=10000
    )
    best_model.fit(X, y)

    # Count non-zero coefficients (sparse solution)
    n_nonzero = np.sum(best_model.coef_ != 0)

    results = {
        "train_score": best_model.score(X, y),
        "best_alpha": grid_search.best_params_["alpha"],
        "n_nonzero_coefs": n_nonzero,
        "n_zero_coefs": len(best_model.coef_) - n_nonzero,
        "model_type": "lasso",
    }

    return best_model, results


def train_elastic_net_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[ElasticNet, Dict[str, Any]]:
    """
    Train Elastic Net combining L1 and L2 regularization.

    Args:
        X: Feature matrix
        y: Target vector
        alpha: Overall regularization strength
        l1_ratio: Mix of L1 and L2 (0=Ridge, 1=Lasso)
        cv: Cross-validation folds
        random_state: Random seed

    Returns:
        Trained model and results dictionary
    """
    # Grid search for best alpha and l1_ratio
    param_grid = {"alpha": np.logspace(-4, 1, 10), "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}

    elastic = ElasticNet(random_state=random_state, max_iter=10000)
    grid_search = GridSearchCV(elastic, param_grid, cv=cv, scoring="r2", n_jobs=-1)
    grid_search.fit(X, y)

    # Train final model
    best_model = ElasticNet(
        alpha=grid_search.best_params_["alpha"],
        l1_ratio=grid_search.best_params_["l1_ratio"],
        random_state=random_state,
        max_iter=10000,
    )
    best_model.fit(X, y)

    results = {
        "train_score": best_model.score(X, y),
        "best_alpha": grid_search.best_params_["alpha"],
        "best_l1_ratio": grid_search.best_params_["l1_ratio"],
        "n_nonzero_coefs": np.sum(best_model.coef_ != 0),
        "model_type": "elastic_net",
    }

    return best_model, results


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
    hidden_layers: List[int] = [128, 64],
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
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate
        epochs: Training epochs
        batch_size: Batch size
        validation_split: Validation split ratio
        random_state: Random seed

    Returns:
        Trained model and results dictionary
    """
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
    X: pd.DataFrame, y: pd.Series, cv: int = 5, random_state: int = 42
) -> Tuple[StackingRegressor, Dict[str, Any]]:
    """
    Train stacking ensemble with meta-learner.

    Args:
        X: Feature matrix
        y: Target vector
        cv: Cross-validation folds for out-of-fold predictions
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

    # Meta-learner
    meta_model = Ridge(alpha=1.0)

    # Create stacking ensemble
    model = StackingRegressor(estimators=estimators, final_estimator=meta_model, cv=cv, n_jobs=-1)
    model.fit(X, y)

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    results = {
        "train_score": model.score(X, y),
        "cv_score": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "base_models": [name for name, _ in estimators],
        "meta_model": "Ridge",
        "model_type": "stacking",
    }

    return model, results


def train_quantile_regressor(
    X: pd.DataFrame, y: pd.Series, quantiles: List[float] = [0.1, 0.5, 0.9], random_state: int = 42
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Train quantile regression for uncertainty estimation.

    Args:
        X: Feature matrix
        y: Target vector
        quantiles: Quantiles to predict
        random_state: Random seed

    Returns:
        List of trained models (one per quantile) and results dictionary.
        The results dictionary includes 'quantile_results' key with a list of
        per-quantile metrics (quantile, train_score, model_type).
    """
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
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, cv: int = 5, random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple regression models.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Test set proportion
        cv: Cross-validation folds
        random_state: Random seed

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
            model, metrics = train_ridge_regressor(X_sector, y_sector, random_state=random_state)

        sector_models[sector] = model
        sector_metrics[sector] = metrics

    results = {"sector_metrics": sector_metrics, "n_sectors": len(sector_models)}

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
