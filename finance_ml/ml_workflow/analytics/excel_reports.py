"""Shim module.

This analytics module has been relocated under reporting/.
Import from:
    finance_ml.ml_workflow.reporting.excel_reports

This file re-exports the reporting implementation and will be removed in a
future release.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "DEPRECATION: analytics.excel_reports moved to reporting.excel_reports. "
    "Import from finance_ml.ml_workflow.reporting.excel_reports",
    DeprecationWarning,
    stacklevel=2,
)

from finance_ml.ml_workflow.reporting.excel_reports import *  # noqa: F401,F403,E402

try:  # pragma: no cover
    from finance_ml.ml_workflow.reporting.excel_reports import __all__ as _ALL  # type: ignore

    __all__ = list(_ALL)
except Exception:  # pragma: no cover
    __all__ = [name for name in globals().keys() if not name.startswith("_")]
