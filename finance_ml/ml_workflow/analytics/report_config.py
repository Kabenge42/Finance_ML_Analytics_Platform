"""Shim module.

This analytics module has been relocated under reporting/.
Import from:
    finance_ml.ml_workflow.reporting.report_config

This file re-exports the reporting implementation and will be removed in a
future release.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "DEPRECATION: analytics.report_config moved to reporting.report_config. "
    "Import from finance_ml.ml_workflow.reporting.report_config",
    DeprecationWarning,
    stacklevel=2,
)

from finance_ml.ml_workflow.reporting.report_config import *  # noqa: F401,F403,E402

try:  # pragma: no cover
    from finance_ml.ml_workflow.reporting.report_config import __all__ as _ALL  # type: ignore

    __all__ = list(_ALL)
except Exception:  # pragma: no cover
    __all__ = [name for name in globals().keys() if not name.startswith("_")]
