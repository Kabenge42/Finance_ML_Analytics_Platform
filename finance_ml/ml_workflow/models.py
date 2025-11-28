"""Deprecation shim for legacy models module.

Use structured subpackages instead:
  - finance_ml.ml_workflow.regression.models
  - finance_ml.ml_workflow.regression.quantile
  - finance_ml.ml_workflow.regression.constraints
  - finance_ml.ml_workflow.classification.models

This shim re-exports commonly used symbols from the structured modules and
emits a DeprecationWarning. It intentionally avoids importing archived legacy
implementations to prevent outdated dependencies (e.g., regression.cv).
"""

from __future__ import annotations

import warnings

warnings.warn(
    "DEPRECATION NOTICE: 'finance_ml.ml_workflow.models' is deprecated. "
    "Import from 'finance_ml.ml_workflow.regression.*' and '...classification.*' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Minimal, stable surface re-exported from structured modules
from finance_ml.ml_workflow.regression.models import (  # noqa: E402,F401
    train_stacking_regressor,
)
from finance_ml.ml_workflow.regression.quantile import (  # noqa: E402,F401
    train_quantile_regressor,
)
from finance_ml.ml_workflow.regression.constraints import (  # noqa: E402,F401
    NonNegativeRegressionWrapper,
)
from finance_ml.ml_workflow.classification.models import (  # noqa: E402,F401
    compare_classifiers,
    fit_classifier,
)

__all__ = [
    "train_stacking_regressor",
    "train_quantile_regressor",
    "NonNegativeRegressionWrapper",
    "compare_classifiers",
    "fit_classifier",
]


# Backward-compatible aliases expected by finance_ml.api
def create_event_labels(df, use_volatility: bool = False):  # pragma: no cover - thin wrapper
    """Deprecated: use classification.labels.create_enhanced_event_labels.

    Provided for backward compatibility to satisfy imports in finance_ml.api.
    """
    from finance_ml.ml_workflow.classification.labels import (
        create_enhanced_event_labels as _create_labels,
    )

    return _create_labels(df)


def train_event_classifier(df, labels, **kwargs):  # pragma: no cover - thin wrapper
    """Deprecated: use classification.models.fit_classifier.

    Provided for backward compatibility to satisfy imports in finance_ml.api.
    Delegates to fit_classifier with defaults.
    """
    return fit_classifier(df, labels, **kwargs)


__all__ += [
    "create_event_labels",
    "train_event_classifier",
]
