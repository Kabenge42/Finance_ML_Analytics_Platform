"""safety_rails.py
Thin wrapper module for outlier safety rails utilities (Phase 9.9, Gap 4).

This module exists primarily to match the naming in the Phase 9.9
implementation plan. It re-exports the core robust helpers defined in
``finance_ml.ml_workflow.regression.robust`` so that higher-level code and
documentation can depend on a single, semantically clear entry point:

- :func:`winsorize_target` – pre-training target winsorization.
- :func:`clip_predictions` – post-prediction clipping based on the training
  target distribution.
- :func:`enforce_non_negative` – non-negativity (or minimum value) guard.

The underlying implementations remain in :mod:`robust` to avoid duplicating
logic; this file simply provides a Phase 9.9-friendly facade.
"""

from __future__ import annotations

from .robust import clip_predictions, enforce_non_negative, winsorize_target

__all__ = [
    "winsorize_target",
    "clip_predictions",
    "enforce_non_negative",
]
