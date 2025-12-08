"""
Phase 9.5: Regression Models and Predictions

This module provides regression model training, quantile regression, constraints,
and robust prediction helpers for price target prediction.

Submodules:
    - dataset: Dataset preparation and feature integration
    - models: XGBoost, LightGBM, CatBoost, stacking ensemble
    - quantile: Quantile regression for uncertainty intervals
    - constraints: Non-negativity constraints for price predictions
    - robust: Winsorization and adaptive clipping helpers
    - io: Model persistence (save/load)

Architecture:
    Aligned with code_guidelines.md Section 4 (Phase 9.5 workflow)
    and finance_ml_restructuring_plan.md (consolidated regression module)
"""

# Dataset preparation
from finance_ml.ml_workflow.regression.dataset import (
    prepare_regression_data,
    integrate_classification_features,
)

# Models
from finance_ml.ml_workflow.regression.models import (
    train_xgboost_regressor,
    train_lightgbm_regressor,
    train_catboost_regressor,
    train_stacking_regressor,
    compare_regressors,
)

# Quantile regression
from finance_ml.ml_workflow.regression.quantile import (
    train_quantile_regressor,
)

# Constraints
from finance_ml.ml_workflow.regression.constraints import (
    NonNegativeRegressionWrapper,
)

# Robust helpers
from finance_ml.ml_workflow.regression.robust import (
    winsorize_target,
    clip_predictions,
    adaptive_clip_predictions,
    enforce_non_negative,
)

# I/O
from finance_ml.ml_workflow.regression.io import (
    save_regression_model,
    load_regression_model,
)

# ============================================================================
# Aliases for code_guidelines.md v1.10 API compliance
# ============================================================================

# Alias: train_sector_models -> compare_regressors (trains and compares sector models)
train_sector_models = compare_regressors

# Alias: apply_nonnegative_constraint -> enforce_non_negative
apply_nonnegative_constraint = enforce_non_negative

# ============================================================================
# Prefixed aliases for notebook compatibility (regression_* pattern)
# ============================================================================
regression_prepare_data = prepare_regression_data
regression_compare_regressors = compare_regressors
regression_train_stacking = train_stacking_regressor
regression_train_quantile = train_quantile_regressor
regression_save_model = save_regression_model
regression_load_model = load_regression_model
regression_create_classification_interactions = integrate_classification_features

__all__ = [
    # Dataset
    "prepare_regression_data",
    "integrate_classification_features",
    # Models
    "train_xgboost_regressor",
    "train_lightgbm_regressor",
    "train_catboost_regressor",
    "train_stacking_regressor",
    "compare_regressors",
    # Aliases for code_guidelines.md v1.10 API
    "train_sector_models",
    "apply_nonnegative_constraint",
    # Quantile
    "train_quantile_regressor",
    # Constraints
    "NonNegativeRegressionWrapper",
    # Robust
    "winsorize_target",
    "clip_predictions",
    "adaptive_clip_predictions",
    "enforce_non_negative",
    # I/O
    "save_regression_model",
    "load_regression_model",
    # Prefixed aliases for notebook compatibility
    "regression_prepare_data",
    "regression_compare_regressors",
    "regression_train_stacking",
    "regression_train_quantile",
    "regression_save_model",
    "regression_load_model",
    "regression_create_classification_interactions",
]
