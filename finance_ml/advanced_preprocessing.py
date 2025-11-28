"""Deprecation shim for advanced preprocessing utilities.

Moved under finance_ml.ml_workflow.preprocessing with clearer module
boundaries (imputation, outliers, scaling, dtypes, pipeline, transforms).

Import from these submodules directly going forward. This shim re-exports
commonly used entry points and emits a DeprecationWarning on import.
"""

from __future__ import annotations

import warnings


warnings.warn(
    "DEPRECATION NOTICE: 'finance_ml.advanced_preprocessing' has moved to "
    "'finance_ml.ml_workflow.preprocessing'. Import specific modules like "
    "imputation, outliers, scaling, dtypes, pipeline, transforms.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export a small set of commonly used helpers from the new locations
from finance_ml.ml_workflow.preprocessing.data import normalize_columns  # noqa: E402
from finance_ml.ml_workflow.preprocessing.imputation import (  # noqa: E402
    apply_enhanced_imputation_strategy_6step,
    apply_enhanced_imputation_strategy_4step,
)
from finance_ml.ml_workflow.preprocessing.outliers import (  # noqa: E402
    winsorize_by_sector,
)
from finance_ml.ml_workflow.preprocessing.scaling import scale_features  # noqa: E402
from finance_ml.ml_workflow.preprocessing.pipeline import (  # noqa: E402
    prepare_phase91_data,
)

__all__ = [
    "normalize_columns",
    "apply_enhanced_imputation_strategy_6step",
    "apply_enhanced_imputation_strategy_4step",
    "winsorize_by_sector",
    "scale_features",
    "prepare_phase91_data",
]
