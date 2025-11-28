"""Data catalog module (facade).

This module aligns with the target architecture by exposing the data catalog
APIs under ``finance_ml.ml_workflow.data``. It temporarily re-exports the
existing implementation from ``finance_ml.ml_workflow.data_catalog`` to avoid
behavior changes during the restructuring.
"""

from __future__ import annotations

# Re-export the current implementation
from finance_ml.ml_workflow.data_catalog import *  # noqa: F401,F403

# Best-effort __all__ for static analyzers; ignore if source doesn't expose it
try:  # pragma: no cover - simple passthrough
    from finance_ml.ml_workflow.data_catalog import __all__ as _ALL  # type: ignore

    __all__ = list(_ALL)  # type: ignore[name-defined]
except Exception:  # pragma: no cover - fallback
    __all__ = [name for name in globals().keys() if not name.startswith("_")]
