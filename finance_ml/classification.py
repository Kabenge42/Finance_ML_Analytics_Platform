"""Deprecation shim for classification utilities.

This root-level module is deprecated. Please import from the structured
subpackage instead:

    finance_ml.ml_workflow.classification

For example:
    from finance_ml.ml_workflow.classification import labels, models, evaluation

This shim re-exports the subpackage modules for backward compatibility and
emits a DeprecationWarning on import.
"""

from __future__ import annotations

import warnings


warnings.warn(
    "DEPRECATION NOTICE: 'finance_ml.classification' has moved to "
    "'finance_ml.ml_workflow.classification'. Import from the subpackage "
    "going forward. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the subpackage so existing imports like
# `from finance_ml import classification` continue to work.
from finance_ml.ml_workflow import classification as _classification  # noqa: F401

# Optionally surface common modules directly for legacy callers
from finance_ml.ml_workflow.classification import (  # noqa: F401,E402
    labels,
    models,
    tuning,
    evaluation,
)

__all__ = [
    "labels",
    "models",
    "tuning",
    "evaluation",
]
