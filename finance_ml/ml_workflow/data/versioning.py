"""Data versioning module (facade).

Target-architecture path that re-exports the current implementation from
``finance_ml.ml_workflow.data_versioning`` to preserve behavior during the
decomposition of the package.
"""

from __future__ import annotations

from finance_ml.ml_workflow.data_versioning import *  # noqa: F401,F403

try:  # pragma: no cover
    from finance_ml.ml_workflow.data_versioning import __all__ as _ALL  # type: ignore

    __all__ = list(_ALL)
except Exception:  # pragma: no cover
    __all__ = [name for name in globals().keys() if not name.startswith("_")]
