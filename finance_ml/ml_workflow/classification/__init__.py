"""
finance_ml.ml_workflow.classification - Classification subpackage

This package provides comprehensive classification capabilities for financial event detection:
- labels: Event label creation (price momentum, valuation, fundamental, etc.)
- tuning: Hyperparameter optimization and sector-stratified cross-validation
- models: Multiple classifier implementations (will be extracted in future phase)
- evaluation: Model evaluation, calibration analysis (will be extracted in future phase)

Phase 9.4 refactor: Extracted labels and tuning modules from classification.py and
classification_enhanced.py for better modularity.

Public API:
-----------
From labels module:
    - create_enhanced_event_labels

From tuning module:
    - optimize_classifier_hyperparameters
    - cross_validate_with_sector_stratification
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

__all__ = [
    # Labels
    "create_enhanced_event_labels",
    # Tuning
    "optimize_classifier_hyperparameters",
    "cross_validate_with_sector_stratification",
]
