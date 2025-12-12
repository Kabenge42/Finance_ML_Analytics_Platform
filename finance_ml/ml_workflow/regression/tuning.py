"""
Phase 9.5.1: Hyperparameter Optimization with Optuna

This module provides hyperparameter tuning capabilities using Optuna's Bayesian
optimization framework. Optuna uses Tree-structured Parzen Estimator (TPE) to
intelligently search the hyperparameter space, converging faster than grid search
or random search.

Key Features:
- Bayesian optimization with TPE sampler
- Configurable number of trials
- Cross-validation based objective function
- Support for multiple model types
- Early stopping for efficient search
- Deterministic results with random_state

Supported Model Types:
- random_forest: RandomForestRegressor with n_estimators, max_depth, min_samples_split, min_samples_leaf
- (Future: xgboost, lightgbm, catboost, neural_network)

Integration with Phase 9.5:
- Compatible with any train_* function from regression.models
- Results can be passed directly to model training functions
- Integrates with prepare_regression_data from regression.dataset

Example:
    >>> from finance_ml.ml_workflow.regression.tuning import optimize_hyperparameters_optuna
    >>> from finance_ml.ml_workflow.regression.dataset import prepare_regression_data
    >>> from finance_ml.ml_workflow.regression.models import train_random_forest_regressor
    >>>
    >>> # Prepare data
    >>> X_train, X_test, y_train, y_test, _ = prepare_regression_data(df)
    >>>
    >>> # Optimize hyperparameters
    >>> best_params, study = optimize_hyperparameters_optuna(
    ...     X_train, y_train,
    ...     model_type='random_forest',
    ...     n_trials=50,
    ...     cv=5
    ... )
    >>>
    >>> # Train final model with best parameters
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    >>> model.fit(X_train, y_train)
    >>>
    >>> print(f"Best parameters: {best_params}")
    >>> print(f"Best CV R²: {study.best_value:.3f}")

Performance Tips:
- Start with n_trials=20-50 for quick exploration
- Use n_trials=100-200 for thorough optimization
- Monitor study progress with study.trials_dataframe()
- Use study.best_params to get optimal hyperparameters

Reference:
- Optuna: https://optuna.readthedocs.io/
- TPE: Bergstra et al. (2011) - Algorithms for Hyper-Parameter Optimization
"""

import logging
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Configure logger
logger = logging.getLogger(__name__)

# Optional Optuna dependency
try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore
    HAS_OPTUNA = False
    logger.warning("Optuna not available. Install with: pip install optuna")


def optimize_hyperparameters_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest",
    n_trials: int = 50,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Any]:
    """
    Optimize hyperparameters using Optuna Bayesian optimization.

    This function uses Optuna's Tree-structured Parzen Estimator (TPE) to
    intelligently search the hyperparameter space. It evaluates candidate
    hyperparameters using cross-validation R² score and returns the best
    configuration found.

    Args:
        X: Feature matrix (pandas DataFrame with numeric features)
        y: Target vector (pandas Series with numeric target values)
        model_type: Model type to optimize (default: 'random_forest')
                   Currently supported: 'random_forest'
                   Future: 'xgboost', 'lightgbm', 'catboost', 'neural_network'
        n_trials: Number of optimization trials (default: 50)
                 More trials = better optimization but longer runtime
                 Recommended ranges:
                 - Quick: 20-50 trials
                 - Thorough: 100-200 trials
                 - Exhaustive: 200+ trials
        cv: Cross-validation folds (default: 5)
           Higher values = more robust but slower
        random_state: Random seed for reproducibility (default: 42)
                     Ensures deterministic optimization runs

    Returns:
        Tuple of (best_params, study):
        - best_params: Dictionary with optimal hyperparameters
          Example for random_forest:
          {
              'n_estimators': 150,
              'max_depth': 12,
              'min_samples_split': 5,
              'min_samples_leaf': 2
          }
        - study: Optuna Study object with full optimization history
          Access with:
          - study.best_value: Best cross-validation R² score
          - study.best_trial: Best trial object
          - study.trials_dataframe(): DataFrame of all trials

    Raises:
        ImportError: If Optuna is not installed
        ValueError: If model_type is not supported

    Hyperparameter Search Spaces:
        random_forest:
        - n_estimators: 50-100 (integer, capped to prevent overfitting)
        - max_depth: 3-10 (integer, limited to prevent overfitting)
        - min_samples_split: 5-30 (integer, higher minimum for regularization)
        - min_samples_leaf: 5-50 (integer, higher values prevent overfitting)

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from sklearn.datasets import make_regression
        >>>
        >>> # Create sample data
        >>> X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)
        >>> X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
        >>> y_series = pd.Series(y)
        >>>
        >>> # Optimize hyperparameters
        >>> best_params, study = optimize_hyperparameters_optuna(
        ...     X_df, y_series,
        ...     model_type='random_forest',
        ...     n_trials=30,
        ...     cv=3
        ... )
        >>>
        >>> # Review results
        >>> print(f"Best R² (CV): {study.best_value:.3f}")
        >>> print(f"Best parameters: {best_params}")
        >>>
        >>> # Visualize optimization (requires plotly)
        >>> # import optuna.visualization as vis
        >>> # vis.plot_optimization_history(study).show()

    Notes:
        - Optimization is CPU-intensive; consider using fewer trials for large datasets
        - TPE sampler learns from previous trials to focus search on promising regions
        - Set optuna.logging.set_verbosity(optuna.logging.WARNING) to reduce output
        - Results are deterministic when random_state is fixed
        - For production, save study object: joblib.dump(study, 'study.pkl')
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna not installed. Install with: pip install optuna")

    logger.info(
        f"Starting Optuna hyperparameter optimization: model={model_type}, "
        f"n_trials={n_trials}, cv={cv}"
    )

    def objective(trial):
        """
        Optuna objective function for hyperparameter optimization.

        This function is called by Optuna for each trial. It suggests hyperparameters,
        trains the model with cross-validation, and returns the mean CV R² score.

        Args:
            trial: Optuna trial object

        Returns:
            Mean cross-validation R² score (float)
        """
        if model_type == "random_forest":
            # Suggest hyperparameters with conservative ranges to prevent overfitting
            # - Lower n_estimators cap (50-100) for faster training
            # - Limited max_depth (3-10) to prevent memorizing training data
            # - Higher min_samples_split (5-30) for better generalization
            # - Higher min_samples_leaf (5-50) as key regularization parameter
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 5, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            }
            model = RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. " f"Currently supported: 'random_forest'"
            )

        # Cross-validation with R² scoring
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        mean_score = scores.mean()

        # Log progress
        logger.debug(
            f"Trial {trial.number}: params={params}, CV R²={mean_score:.4f} "
            f"(std={scores.std():.4f})"
        )

        return mean_score

    # Create study with TPE sampler
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )

    # Run optimization
    logger.info(f"Running {n_trials} optimization trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Extract best parameters
    best_params = study.best_params
    best_score = study.best_value

    logger.info(
        f"✓ Optimization complete: Best CV R²={best_score:.4f}, " f"Parameters={best_params}"
    )
    logger.info(f"  Total trials: {len(study.trials)}, Best trial: {study.best_trial.number}")

    return best_params, study
