"""Backward-compatible shim for advanced regression models.

This top-level module previously contained advanced regression functionality
before the refactor to the :mod:`finance_ml.ml_workflow.regression` subpackage.

Phase 3 (Restructuring Plan):
---------------------------------
- ``finance_ml.advanced_models`` is now a thin compatibility layer.
- New code should import from ``finance_ml.ml_workflow.regression.*``.
- This shim re-exports a minimal set of functions that are still imported
  directly by tests and notebooks:

    - :func:`train_sector_specific_models`
    - :func:`train_quantile_regressor`

The heavy implementations live in ``finance_ml.ml_workflow.advanced_models``
and the dedicated regression submodules; this shim only forwards calls and
emits a deprecation warning.
"""

from __future__ import annotations

import logging
import warnings

from finance_ml.ml_workflow import advanced_models as _advanced_models


# Expose a logger with the expected name so existing tests that assert on
# "finance_ml.advanced_models" log records continue to work.
logger = logging.getLogger(__name__)

# Ensure the underlying module uses the same logger instance so that all
# log records appear under the top-level shim's logger name.
try:  # pragma: no cover - defensive; underlying module always defines logger
    _advanced_models.logger = logger
except Exception:  # pragma: no cover
    pass


warnings.warn(
    "DEPRECATION NOTICE: 'finance_ml.advanced_models' has been consolidated "
    "into the 'finance_ml.ml_workflow.regression' subpackage. Import from "
    "'finance_ml.ml_workflow.regression.*' instead. This shim will be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)


# Re-export only the functions that are still imported from this namespace.
from finance_ml.ml_workflow.advanced_models import (  # noqa: E402
    train_sector_specific_models,
    train_quantile_regressor,
)


__all__ = [
    "train_sector_specific_models",
    "train_quantile_regressor",
]
