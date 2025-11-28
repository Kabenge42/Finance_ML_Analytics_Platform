"""
Phase 9.4: Multi-Class Classification Module - DEPRECATION SHIM

.. deprecated:: v9_8
    This module is maintained for backward compatibility only.
    New code should import from the classification subpackage:

    - :mod:`finance_ml.ml_workflow.classification.labels` for event label creation
    - :mod:`finance_ml.ml_workflow.classification.models` for model training
    - :mod:`finance_ml.ml_workflow.classification.tuning` for hyperparameter optimization
    - :mod:`finance_ml.ml_workflow.classification.evaluation` for model evaluation

    Example migration::

        # Old (deprecated):
        from finance_ml.ml_workflow.classification import train_xgboost_classifier

        # New (recommended):
        from finance_ml.ml_workflow.classification.models import train_xgboost_classifier
        # Or use the subpackage __init__:
        from finance_ml.ml_workflow.classification import train_xgboost_classifier

DEPRECATION NOTICE:
-------------------
This module has been refactored into the classification/ subpackage structure:

- classification/labels.py: Event label creation (create_enhanced_event_labels)
- classification/models.py: Model training, data preparation, sampling
- classification/tuning.py: Hyperparameter optimization, cross-validation
- classification/evaluation.py: Model evaluation, SHAP, visualization

All functions are re-exported here for backward compatibility.
Import directly from submodules for better performance and clarity.

Original module description:
----------------------------
Sophisticated event classification with multiple model architectures:
- Gradient Boosting: XGBoost, LightGBM, CatBoost
- Neural Networks: Feedforward DNN with batch normalization
- Ensemble Methods: Voting and Stacking classifiers
- Class Imbalance Handling: SMOTE, ADASYN, class weights
- Model Interpretation: SHAP values, feature importance

Classes:
- 0: Neutral (price_target within ±10% of last_price)
- 1: Positive Catalyst (price_target > last_price by >=10%)
- 2: Negative Catalyst (price_target < last_price by >=10%)
"""

from __future__ import annotations

import warnings

# Emit deprecation warning when module is imported directly
warnings.warn(
    "The classification module (finance_ml.ml_workflow.classification) is deprecated as of v9_8. "
    "Import from the classification subpackage instead:\n"
    "  from finance_ml.ml_workflow.classification import <function_name>\n"
    "Or import from specific submodules:\n"
    "  from finance_ml.ml_workflow.classification.models import train_xgboost_classifier\n"
    "  from finance_ml.ml_workflow.classification.labels import create_enhanced_event_labels",
    DeprecationWarning,
    stacklevel=2,
)

# =============================================================================
# Re-export all public functions from classification subpackage
# =============================================================================

# Labels
from finance_ml.ml_workflow.classification.labels import (
    create_enhanced_event_labels,
)

# Models - Data preparation
from finance_ml.ml_workflow.classification.models import (
    _prepare_categorical_features,
    prepare_classification_data,
    export_classification_features,
    clean_extreme_values,
    validate_data_quality,
)

# Models - Sampling
from finance_ml.ml_workflow.classification.models import (
    apply_smote,
    apply_adasyn,
    apply_undersampling,
    apply_combined_sampling,
)

# Models - Training
from finance_ml.ml_workflow.classification.models import (
    train_xgboost_classifier,
    train_lightgbm_classifier,
    train_catboost_classifier,
    train_svm_classifier,
    train_neural_network_classifier,
    train_voting_classifier,
    train_stacking_classifier,
)

# Models - Comparison and orchestration
from finance_ml.ml_workflow.classification.models import (
    compare_classifiers,
    fit_classifier,
)

# Tuning
from finance_ml.ml_workflow.classification.tuning import (
    optimize_classifier_hyperparameters,
    cross_validate_with_sector_stratification,
)

# Evaluation - Metrics
from finance_ml.ml_workflow.classification.evaluation import (
    evaluate_classification,
    evaluate_classification_by_sector,
)

# Evaluation - Interpretation
from finance_ml.ml_workflow.classification.evaluation import (
    compute_shap_values,
    analyze_per_class_feature_importance,
)

# Evaluation - Cross-validation
from finance_ml.ml_workflow.classification.evaluation import (
    cross_validate_classifier,
)

# Evaluation - Visualization
from finance_ml.ml_workflow.classification.evaluation import (
    plot_confusion_matrices,
    plot_learning_curves,
)

# Evaluation - Feature analysis
from finance_ml.ml_workflow.classification.evaluation import (
    compare_feature_importance,
)

# Evaluation - Calibration
from finance_ml.ml_workflow.classification.evaluation import (
    analyze_calibration,
)

# Phase 9.4.3 - Enhanced feature importance analysis
from finance_ml.ml_workflow.classification.evaluation import (
    analyze_feature_importance_by_groups,
    analyze_feature_importance_by_sector,
    analyze_shap_by_feature_groups,
)

# =============================================================================
# __all__ for explicit public API
# =============================================================================

__all__ = [
    # Labels
    "create_enhanced_event_labels",
    # Data preparation
    "prepare_classification_data",
    "_prepare_categorical_features",
    "export_classification_features",
    "clean_extreme_values",
    "validate_data_quality",
    # Sampling
    "apply_smote",
    "apply_adasyn",
    "apply_undersampling",
    "apply_combined_sampling",
    # Model training
    "train_xgboost_classifier",
    "train_lightgbm_classifier",
    "train_catboost_classifier",
    "train_svm_classifier",
    "train_neural_network_classifier",
    "train_voting_classifier",
    "train_stacking_classifier",
    # Comparison and orchestration
    "compare_classifiers",
    "fit_classifier",
    # Tuning
    "optimize_classifier_hyperparameters",
    "cross_validate_with_sector_stratification",
    # Evaluation - Metrics
    "evaluate_classification",
    "evaluate_classification_by_sector",
    # Evaluation - Interpretation
    "compute_shap_values",
    "analyze_per_class_feature_importance",
    # Evaluation - Cross-validation
    "cross_validate_classifier",
    # Evaluation - Visualization
    "plot_confusion_matrices",
    "plot_learning_curves",
    # Evaluation - Feature analysis
    "compare_feature_importance",
    # Evaluation - Calibration
    "analyze_calibration",
    # Phase 9.4.3 - Enhanced feature importance analysis
    "analyze_feature_importance_by_groups",
    "analyze_feature_importance_by_sector",
    "analyze_shap_by_feature_groups",
]
