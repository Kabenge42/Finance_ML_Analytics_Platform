"""
finance_ml.ml_workflow.classification.tuning - Hyperparameter tuning and cross-validation

This module provides advanced tuning capabilities for classification models:
- Optuna-based Bayesian hyperparameter optimization
- Sector-stratified cross-validation for financial data
- Support for XGBoost, LightGBM, CatBoost, and Random Forest

Phase 9.4 refactor: Extracted from classification_enhanced.py for better modularity.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Any, Optional, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

# Optional imports with fallback handling
try:
    import xgboost as xgb

    HAVE_XGBOOST = True
except ImportError:
    xgb = None
    HAVE_XGBOOST = False

try:
    import lightgbm as lgb

    HAVE_LIGHTGBM = True
except ImportError:
    lgb = None
    HAVE_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier

    HAVE_CATBOOST = True
except ImportError:
    CatBoostClassifier = None
    HAVE_CATBOOST = False

logger = logging.getLogger(__name__)

__all__ = [
    "optimize_classifier_hyperparameters",
    "cross_validate_with_sector_stratification",
]


def optimize_classifier_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    classifier_type: Literal["xgboost", "lightgbm", "catboost", "random_forest"] = "xgboost",
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Optimize classifier hyperparameters using Optuna.

    This function performs Bayesian optimization to find the best hyperparameters
    for the specified classifier. It uses F1-macro score as the optimization metric.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training labels (0, 1, 2 for multi-class)
    classifier_type : str
        Type of classifier to optimize:
        - 'xgboost': XGBoost classifier
        - 'lightgbm': LightGBM classifier
        - 'catboost': CatBoost classifier
        - 'random_forest': Random Forest classifier
    n_trials : int, default=50
        Number of optimization trials
    cv_folds : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to show progress bar

    Returns:
    --------
    dict : Contains the following keys:
        - 'best_params': Dict of best hyperparameters
        - 'best_score': Float, best F1-macro score achieved
        - 'study': Optuna study object (for further analysis)
        - 'model': Trained model with best parameters

    Example:
    --------
    >>> from finance_ml.ml_workflow.classification.tuning import optimize_classifier_hyperparameters
    >>> result = optimize_classifier_hyperparameters(
    ...     X_train, y_train,
    ...     classifier_type='xgboost',
    ...     n_trials=100
    ... )
    >>> print(f"Best F1 score: {result['best_score']:.4f}")
    >>> print(f"Best parameters: {result['best_params']}")
    >>> best_model = result['model']

    Notes:
    ------
    - Requires optuna: pip install optuna
    - Uses F1-macro for multi-class classification
    - Automatically handles class imbalance through class weights
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        logger.error("Optuna not installed. Install with: pip install optuna")
        return {"best_params": {}, "best_score": 0.0, "study": None, "model": None}

    def objective(trial):
        """Optuna objective function for hyperparameter optimization."""
        try:
            if classifier_type == "xgboost" and HAVE_XGBOOST:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                    "random_state": random_state,
                    "eval_metric": "mlogloss",
                    "use_label_encoder": False,
                    "verbosity": 0,
                }
                model = xgb.XGBClassifier(**params)

            elif classifier_type == "lightgbm" and HAVE_LIGHTGBM:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "random_state": random_state,
                    "verbose": -1,
                }
                model = lgb.LGBMClassifier(**params)

            elif classifier_type == "catboost" and HAVE_CATBOOST:
                params = {
                    "iterations": trial.suggest_int("iterations", 50, 500),
                    "depth": trial.suggest_int("depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                    "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
                    "random_seed": random_state,
                    "verbose": False,
                }
                model = CatBoostClassifier(**params)

            elif classifier_type == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical(
                        "max_features", ["sqrt", "log2", None]
                    ),
                    "random_state": random_state,
                }
                model = RandomForestClassifier(**params)

            else:
                raise ValueError(f"Unsupported classifier type: {classifier_type}")

            # Cross-validation with F1 macro score
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
            return scores.mean()

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0

    # Run optimization
    try:
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))

        if verbose:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Train final model with best parameters
        best_params = study.best_params
        best_params["random_state"] = random_state

        if classifier_type == "xgboost" and HAVE_XGBOOST:
            best_params["eval_metric"] = "mlogloss"
            best_params["use_label_encoder"] = False
            best_params["verbosity"] = 0
            best_model = xgb.XGBClassifier(**best_params)
        elif classifier_type == "lightgbm" and HAVE_LIGHTGBM:
            best_params["verbose"] = -1
            best_model = lgb.LGBMClassifier(**best_params)
        elif classifier_type == "catboost" and HAVE_CATBOOST:
            best_params["verbose"] = False
            best_model = CatBoostClassifier(**best_params)
        else:
            best_model = RandomForestClassifier(**best_params)

        best_model.fit(X_train, y_train)

        logger.info(f"Best {classifier_type} F1 score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
            "model": best_model,
        }

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return {"best_params": {}, "best_score": 0.0, "study": None, "model": None}


def cross_validate_with_sector_stratification(
    X: pd.DataFrame,
    y: np.ndarray,
    model: Any,
    sector_col: str = "sector",
    cv_folds: int = 5,
    scoring: str = "f1_macro",
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Perform cross-validation with sector stratification.

    Ensures that each fold has proportional representation from each sector,
    which is important for financial data where sectors have different characteristics.

    Parameters:
    -----------
    X : pd.DataFrame
        Features (must include sector column)
    y : np.ndarray
        Labels
    model : sklearn-compatible model
        Model to evaluate
    sector_col : str, default='sector'
        Name of sector column in X
    cv_folds : int, default=5
        Number of cross-validation folds
    scoring : str, default='f1_macro'
        Scoring metric ('f1_macro' or 'accuracy')
    random_state : int, default=42
        Random seed

    Returns:
    --------
    dict : CV metrics including mean, std, and fold scores

    Example:
    --------
    >>> from finance_ml.ml_workflow.classification.tuning import cross_validate_with_sector_stratification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier(random_state=42)
    >>> cv_results = cross_validate_with_sector_stratification(
    ...     X_train, y_train, model, sector_col='sector'
    ... )
    >>> print(f"Mean F1: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
    """
    from sklearn.model_selection import StratifiedGroupKFold

    if sector_col not in X.columns:
        logger.warning(f"Sector column '{sector_col}' not found. Using standard stratified CV.")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    else:
        # Use sector as groups for stratification
        groups = X[sector_col].values
        cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Remove sector column for model training
        X_no_sector = X.drop(columns=[sector_col])

        scores = []
        for train_idx, test_idx in cv.split(X_no_sector, y, groups):
            X_train_fold = X_no_sector.iloc[train_idx]
            X_test_fold = X_no_sector.iloc[test_idx]
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_test_fold)

            if scoring == "f1_macro":
                score = f1_score(y_test_fold, y_pred_fold, average="macro")
            elif scoring == "accuracy":
                score = accuracy_score(y_test_fold, y_pred_fold)
            else:
                score = 0.0

            scores.append(score)

        scores = np.array(scores)

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "fold_scores": scores.tolist(),
    }
