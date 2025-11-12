"""
Regression models and utilities for financial prediction.

This package provides sector-optimized regression models with classification
feature integration, non-negative prediction constraints, quantile regression,
and hyperparameter tuning.

Phase 9.5 - Regression Refactor

## Implementation Status

### ✅ Phase 9.5.0 (Completed):
- constraints.py: NonNegativeRegressionWrapper for non-negative predictions
- dataset.py: Data preparation, validation, classification integration, sector training

### ✅ Phase 9.5.1 (Completed):
- models.py: 15 train_* functions + compare_regressors (1,059 lines)
  * Linear Models: Ridge, Lasso, ElasticNet, BayesianRidge, Polynomial
  * Gradient Boosting: XGBoost, LightGBM, CatBoost, HistGradientBoosting
  * Tree Models: RandomForest, ExtraTrees
  * Neural Networks: Feedforward DNN with TensorFlow/Keras
  * Ensemble Methods: Voting, Stacking
  * Benchmarking: compare_regressors for model selection
- quantile.py: train_quantile_regressor for uncertainty estimation (173 lines)
- tuning.py: optimize_hyperparameters_optuna for Bayesian optimization (244 lines)
- io.py: save_model and load_model for model persistence (259 lines)

**Total: 1,735 lines of new regression code extracted and organized**

Backward compatibility maintained via advanced_models.py shims.

## Usage Examples

### Non-Negative Predictions:
```python
from finance_ml.ml_workflow.regression.constraints import NonNegativeRegressionWrapper
from sklearn.linear_model import Ridge

base_model = Ridge(alpha=1.0)
model = NonNegativeRegressionWrapper(base_model)
model.fit(X_train, y_train)
predictions = model.predict(X_test)  # All predictions >= 0
```

### Classification Feature Integration:
```python
from finance_ml.ml_workflow.regression.dataset import (
    extract_classification_features,
    integrate_classification_features_into_dataframe
)

# From trained classifier
probs = classifier.predict_proba(X)
class_features = extract_classification_features(probs)

# Integrate into main dataframe
df_enhanced = integrate_classification_features_into_dataframe(df, class_features)
```

### Data Preparation and Validation:
```python
from finance_ml.ml_workflow.regression.dataset import (
    prepare_regression_data,
    validate_training_data,
    prepare_features_for_training
)

# Prepare train/test split with feature info
X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(
    df, target_col="price_target", test_size=0.2
)

# Validate before training
validation = validate_training_data(X_train, y_train, strict=True)

# Prepare with imputation
X, y = prepare_features_for_training(
    df, feature_cols=features, target_col="price_target",
    apply_imputation=True, sector_column="sector"
)
```

### Sector-Specific Training:
```python
from finance_ml.ml_workflow.regression.dataset import train_sector_specific_models

sector_models, results = train_sector_specific_models(
    df=data,
    feature_cols=feature_list,
    target_col="price_target",
    sector_col="sector",
    model_type="random_forest",
    min_samples=20,
    ensure_nonnegative=True,
    auto_extract_fallback=True
)

print(f"Trained {results['n_sectors_trained']} sector models")
```

## Public API
"""

# Constraints
from finance_ml.ml_workflow.regression.constraints import (
    NonNegativeRegressionWrapper,
    )
# Dataset preparation and validation
from finance_ml.ml_workflow.regression.dataset import (
    # Classification feature integration
    extract_classification_features,
    integrate_classification_features_into_dataframe,
    create_classification_interactions,
    # Data preparation
    prepare_regression_data,
    # Validation
    validate_training_data,
    prepare_features_for_training,
    extract_numeric_feature_columns,
    # Sector-specific training
    train_sector_specific_models,
    )
# Phase 9.5.1: Model persistence
from finance_ml.ml_workflow.regression.io import (
    save_model,
    load_model,
    )
# Phase 9.5.1: Model training functions
from finance_ml.ml_workflow.regression.models import (
    # Linear models
    train_ridge_regressor,
    train_lasso_regressor,
    train_elastic_net_regressor,
    train_bayesian_ridge_regressor,
    train_polynomial_regressor,
    # Gradient boosting models
    train_xgboost_regressor,
    train_lightgbm_regressor,
    train_catboost_regressor,
    train_histgb_regressor,
    # Tree models
    train_random_forest_regressor,
    train_extra_trees_regressor,
    # Neural network
    train_neural_network_regressor,
    # Ensemble methods
    train_voting_regressor,
    train_stacking_regressor,
    # Model comparison
    compare_regressors,
    )
# Phase 9.5.1: Quantile regression
from finance_ml.ml_workflow.regression.quantile import (
    train_quantile_regressor,
    )
# Phase 9.5.1: Hyperparameter tuning
from finance_ml.ml_workflow.regression.tuning import (
    optimize_hyperparameters_optuna,
    )
# Phase 9.5.1: Uncertainty & conformal prediction
from finance_ml.ml_workflow.regression.uncertainty import (
    conformal_prediction_intervals,
    compute_interval_coverage,
    )
from finance_ml.ml_workflow.regression.calibration import (
    calibrate_predictions_by_sector,
    DEFAULT_SECTOR_BIAS,
)

__all__ = [
    # Constraints (Phase 9.5.0)
    "NonNegativeRegressionWrapper",
    # Classification feature integration (Phase 9.5.0)
    "extract_classification_features",
    "integrate_classification_features_into_dataframe",
    "create_classification_interactions",
    # Data preparation (Phase 9.5.0)
    "prepare_regression_data",
    # Validation (Phase 9.5.0)
    "validate_training_data",
    "prepare_features_for_training",
    "extract_numeric_feature_columns",
    # Sector-specific training (Phase 9.5.0)
    "train_sector_specific_models",
    # Linear models (Phase 9.5.1)
    "train_ridge_regressor",
    "train_lasso_regressor",
    "train_elastic_net_regressor",
    "train_bayesian_ridge_regressor",
    "train_polynomial_regressor",
    # Gradient boosting models (Phase 9.5.1)
    "train_xgboost_regressor",
    "train_lightgbm_regressor",
    "train_catboost_regressor",
    "train_histgb_regressor",
    # Tree models (Phase 9.5.1)
    "train_random_forest_regressor",
    "train_extra_trees_regressor",
    # Neural network (Phase 9.5.1)
    "train_neural_network_regressor",
    # Ensemble methods (Phase 9.5.1)
    "train_voting_regressor",
    "train_stacking_regressor",
    # Model comparison (Phase 9.5.1)
    "compare_regressors",
    # Quantile regression (Phase 9.5.1)
    "train_quantile_regressor",
    # Hyperparameter tuning (Phase 9.5.1)
    "optimize_hyperparameters_optuna",
    # Model persistence (Phase 9.5.1)
    "save_model",
    "load_model",
    # Uncertainty & conformal prediction (Phase 9.5.1)
    "conformal_prediction_intervals",
    "compute_interval_coverage",
    # Sector calibration (Priority 3)
    "calibrate_predictions_by_sector",
    "DEFAULT_SECTOR_BIAS",
]
