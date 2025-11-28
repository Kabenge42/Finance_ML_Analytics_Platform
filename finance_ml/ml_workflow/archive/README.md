# Archive Directory - Deprecated Modules

**Status:** DEPRECATED  
**Created:** 2025-11-28  
**Removal Target:** Version 1.0.0

## Overview

This directory contains **archived modules** that have been superseded by the
reorganized package structure. These modules are maintained for backward
compatibility but will be removed in a future major release.

## Deprecated Modules

| Module                       | Replacement                                            | Notes                                 |
|------------------------------|--------------------------------------------------------|---------------------------------------|
| `classification.py`          | `finance_ml.ml_workflow.classification` (subpackage)   | Full classification API in subpackage |
| `models.py`                  | `finance_ml.ml_workflow.regression` + `classification` | Split into focused subpackages        |
| `advanced_models.py`         | `finance_ml.ml_workflow.regression.models`             | Regression training functions         |
| `advanced_preprocessing.py`  | `finance_ml.ml_workflow.preprocessing` (subpackage)    | Imputation, outliers, scaling         |
| `advanced_features.py`       | `finance_ml.ml_workflow.features.advanced`             | Advanced feature engineering          |
| `advanced_eda.py`            | `finance_ml.ml_workflow.eda` (subpackage)              | EDA functions split by category       |
| `classification_enhanced.py` | `finance_ml.ml_workflow.classification.tuning`         | Hyperparameter tuning                 |

## Migration Guide

### Before (Deprecated)

```python
# Old imports - still work but emit deprecation warnings
from finance_ml.ml_workflow.classification import train_xgboost_classifier
from finance_ml.ml_workflow.models import train_ridge_regressor
from finance_ml.ml_workflow.advanced_preprocessing import apply_enhanced_imputation_strategy
```

### After (Recommended)

```python
# New imports - use the organized subpackages
from finance_ml.ml_workflow.classification.models import train_xgboost_classifier
from finance_ml.ml_workflow.regression.models import train_ridge_regressor
from finance_ml.ml_workflow.preprocessing.imputation import apply_enhanced_imputation_strategy

# Or use the clean public API facade
from finance_ml.api import train_event_classifier, train_sector_specific_models
```

## Deprecation Timeline

1. **v0.8.x** (Current): Deprecation warnings issued on import
2. **v0.9.x**: Deprecation warnings become more prominent
3. **v1.0.0**: Archived modules removed entirely

## Do Not Import Directly

These archived modules should **not** be imported directly:

```python
# DON'T DO THIS
from finance_ml.ml_workflow.archive.classification import

...  # Wrong!

# DO THIS INSTEAD
from finance_ml.ml_workflow.classification import

...  # Deprecation shim
from finance_ml.ml_workflow.classification.models import

...  # Direct new location
```

The deprecation shims in the parent `ml_workflow` directory handle the transition
automatically and emit appropriate warnings.

## Questions?

See `docs/improvement_plan/finance_ml_restructuring_plan.md` for the full
restructuring plan and rationale.
