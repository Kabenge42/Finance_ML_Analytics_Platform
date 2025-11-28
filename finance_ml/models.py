"""Deprecation shim for legacy top-level models module.

New code should import from:
  - finance_ml.ml_workflow.regression.models
  - finance_ml.ml_workflow.regression.quantile
  - finance_ml.ml_workflow.regression.constraints
  - finance_ml.ml_workflow.classification.models

This shim re-exports a minimal set of commonly used symbols and emits a
DeprecationWarning on import.
"""

from __future__ import annotations

import warnings


warnings.warn(
    "DEPRECATION NOTICE: 'finance_ml.models' is deprecated. Use the modules "
    "under 'finance_ml.ml_workflow.regression' and '...classification' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Regression training helpers
from finance_ml.ml_workflow.regression.models import (  # noqa: E402
    train_stacking_regressor,
)
from finance_ml.ml_workflow.regression.quantile import (  # noqa: E402
    train_quantile_regressor,
)
from finance_ml.ml_workflow.regression.constraints import (  # noqa: E402
    NonNegativeRegressionWrapper,
)


def train_stacking_ensemble(*args, **kwargs):  # pragma: no cover - trivial shim
    """Backward-compatible alias for train_stacking_regressor(X, y, ...)."""
    return train_stacking_regressor(*args, **kwargs)


__all__ = [
    "train_stacking_regressor",
    "train_quantile_regressor",
    "NonNegativeRegressionWrapper",
    # Legacy alias
    "train_stacking_ensemble",
]
