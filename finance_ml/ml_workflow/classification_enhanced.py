"""Deprecation shim for enhanced classification utilities (archived).

This module has been archived under:
    finance_ml.ml_workflow.archive.classification_enhanced

Please migrate to the structured classification subpackage:
    - finance_ml.ml_workflow.classification.tuning
    - finance_ml.ml_workflow.classification.evaluation
"""

from __future__ import annotations

import warnings


warnings.warn(
    "DEPRECATION NOTICE: 'finance_ml.ml_workflow.classification_enhanced' is archived. "
    "Use 'finance_ml.ml_workflow.classification.tuning' and '...classification.evaluation' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the archived implementation to preserve behavior
from .archive.classification_enhanced import *  # noqa: F401,F403,E402

try:  # pragma: no cover
    from .archive.classification_enhanced import __all__ as _ALL  # type: ignore

    __all__ = list(_ALL)
except Exception:  # pragma: no cover
    __all__ = [name for name in globals().keys() if not name.startswith("_")]
