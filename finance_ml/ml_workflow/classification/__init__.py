"""
finance_ml.ml_workflow.classification - Classification subpackage

This package provides comprehensive classification capabilities for financial event detection:
- labels: Event label creation (price momentum, valuation, fundamental, etc.)
- tuning: Hyperparameter optimization and sector-stratified cross-validation
- models: Model training, data preparation, sampling, and orchestration (Phase 9.4.1)
- evaluation: Model evaluation, metrics, SHAP, calibration analysis (Phase 9.4.2)

Phase 9.4.1 refactor: Extracted models module from classification.py with comprehensive
model training functions, Phase 9.3 feature integration, and fit_classifier orchestrator.

Phase 9.4.2 refactor: Extracted evaluation module from classification.py and
classification_enhanced.py with 9 evaluation functions for comprehensive model assessment.

Public API:
-----------
From labels module:
    - create_enhanced_event_labels

From tuning module:
    - optimize_classifier_hyperparameters
    - cross_validate_with_sector_stratification

From models module:
    - Data preparation: prepare_classification_data, _prepare_categorical_features
    - Utilities: export_classification_features, clean_extreme_values, validate_data_quality
    - Sampling: apply_smote, apply_adasyn, apply_undersampling, apply_combined_sampling
    - Model training: train_xgboost_classifier, train_lightgbm_classifier, train_catboost_classifier,
      train_svm_classifier, train_neural_network_classifier, train_voting_classifier, train_stacking_classifier
    - Comparison and orchestration: compare_classifiers, fit_classifier (high-level API)

From evaluation module:
    - Metrics: evaluate_classification, evaluate_classification_by_sector
    - Interpretation: compute_shap_values, analyze_per_class_feature_importance
    - Cross-validation: cross_validate_classifier
    - Visualization: plot_confusion_matrices, plot_learning_curves
    - Feature analysis: compare_feature_importance
    - Calibration: analyze_calibration
"""

from __future__ import annotations

# Import from new submodules
from finance_ml.ml_workflow.classification.labels import (
    create_enhanced_event_labels,
)

from finance_ml.ml_workflow.classification.tuning import (
    optimize_classifier_hyperparameters,
    cross_validate_with_sector_stratification,
)

from finance_ml.ml_workflow.classification.models import (
    # Data preparation
    prepare_classification_data,
    _prepare_categorical_features,
    # Utilities
    export_classification_features,
    clean_extreme_values,
    validate_data_quality,
    # Sampling
    apply_smote,
    apply_adasyn,
    apply_undersampling,
    apply_combined_sampling,
    # Model training
    train_xgboost_classifier,
    train_lightgbm_classifier,
    train_catboost_classifier,
    train_svm_classifier,
    train_neural_network_classifier,
    train_voting_classifier,
    train_stacking_classifier,
    # Comparison and orchestration
    compare_classifiers,
    fit_classifier,
)

from finance_ml.ml_workflow.classification.evaluation import (
    # Metrics
    evaluate_classification,
    evaluate_classification_by_sector,
    # Interpretation
    compute_shap_values,
    analyze_per_class_feature_importance,
    # Cross-validation
    cross_validate_classifier,
    # Visualization
    plot_confusion_matrices,
    plot_learning_curves,
    # Feature analysis
    compare_feature_importance,
    # Calibration
    analyze_calibration,
    # Phase 9.4.3 - Enhanced feature importance analysis
    analyze_feature_importance_by_groups,
    analyze_feature_importance_by_sector,
    analyze_shap_by_feature_groups,
)

__all__ = [
    # Labels
    "create_enhanced_event_labels",
    # Tuning
    "optimize_classifier_hyperparameters",
    "cross_validate_with_sector_stratification",
    # Data preparation
    "prepare_classification_data",
    "_prepare_categorical_features",
    # Utilities
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
