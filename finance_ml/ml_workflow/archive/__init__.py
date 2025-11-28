"""
Archive package for deprecated Finance ML modules.

WARNING: This package contains archived/deprecated modules that will be
removed in a future major release (v1.0.0). Do not import directly from
this package - use the deprecation shims in the parent ml_workflow
directory instead, which will emit appropriate warnings.

See README.md in this directory for migration guidance.
"""

import warnings

warnings.warn(
    "The finance_ml.ml_workflow.archive package contains deprecated modules. "
    "Import from finance_ml.ml_workflow.* instead for deprecation shims, "
    "or migrate to the new subpackage structure. "
    "See docs/improvement_plan/finance_ml_restructuring_plan.md for details.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "classification",
    "models",
    "advanced_models",
    "advanced_preprocessing",
    "advanced_features",
    "advanced_eda",
    "classification_enhanced",
]
