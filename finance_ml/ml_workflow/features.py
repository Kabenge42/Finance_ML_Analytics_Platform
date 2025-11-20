"""
finance_ml.features - Feature engineering functions

DEPRECATION NOTICE (Phase 9.3 refactor):
Core functions from this module have been moved to finance_ml.ml_workflow.features.core:
- _safe_div, engineer_basic_ratios, engineer_margin_features
- engineer_volatility_features, engineer_revenue_cagr, build_features_and_target

This module provides backward compatibility shims. Please update imports to use the new structure:
    from finance_ml.ml_workflow.features.core import engineer_basic_ratios

Advanced feature engineering functions remain in advanced_features.py and will be
refactored in a follow-up phase.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple, List

import pandas as pd

# Phase 9.3: Import from new location for backward compatibility
from finance_ml.ml_workflow.features.core import (
    _safe_div as _new_safe_div,
    engineer_basic_ratios as _new_engineer_basic_ratios,
    engineer_margin_features as _new_engineer_margin_features,
    engineer_volatility_features as _new_engineer_volatility_features,
    engineer_revenue_cagr as _new_engineer_revenue_cagr,
    build_features_and_target as _new_build_features_and_target,
)

logger = logging.getLogger(__name__)


# Deprecation wrappers for Phase 9.3 refactor
def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Safely divide two Series, replacing inf with NaN.

    .. deprecated:: Phase 9.3
        Use :func:`finance_ml.ml_workflow.features.core._safe_div` instead.

    Args:
        numer: Numerator Series
        denom: Denominator Series

    Returns:
        Result Series with inf values replaced by NaN
    """
    warnings.warn(
        "_safe_div from features is deprecated. "
        "Use finance_ml.ml_workflow.features.core._safe_div instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_safe_div(numer, denom)


def engineer_basic_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add a minimal set of engineered ratio features if source columns exist.

    .. deprecated:: Phase 9.3
        Use :func:`finance_ml.ml_workflow.features.core.engineer_basic_ratios` instead.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with ratio features added (preserves original columns)
    """
    warnings.warn(
        "engineer_basic_ratios from features is deprecated. "
        "Use finance_ml.ml_workflow.features.core.engineer_basic_ratios instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_engineer_basic_ratios(df)


def engineer_margin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add margin features if source columns exist.

    .. deprecated:: Phase 9.3
        Use :func:`finance_ml.ml_workflow.features.core.engineer_margin_features` instead.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with margin features added
    """
    warnings.warn(
        "engineer_margin_features from features is deprecated. "
        "Use finance_ml.ml_workflow.features.core.engineer_margin_features instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_engineer_margin_features(df)


def engineer_volatility_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Calculate or aggregate volatility features.

    .. deprecated:: Phase 9.3
        Use :func:`finance_ml.ml_workflow.features.core.engineer_volatility_features` instead.

    Args:
        df: Input DataFrame
        window: Rolling window size for volatility calculation (default: 30)

    Returns:
        DataFrame with volatility features added
    """
    warnings.warn(
        "engineer_volatility_features from features is deprecated. "
        "Use finance_ml.ml_workflow.features.core.engineer_volatility_features instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_engineer_volatility_features(df, window)


def engineer_revenue_cagr(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate revenue CAGR (Compound Annual Growth Rate).

    .. deprecated:: Phase 9.3
        Use :func:`finance_ml.ml_workflow.features.core.engineer_revenue_cagr` instead.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with revenue CAGR features added
    """
    warnings.warn(
        "engineer_revenue_cagr from features is deprecated. "
        "Use finance_ml.ml_workflow.features.core.engineer_revenue_cagr instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_engineer_revenue_cagr(df)


def build_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str], List[str]]:
    """Build feature matrix and target variable from DataFrame.

    .. deprecated:: Phase 9.3
        Use :func:`finance_ml.ml_workflow.features.core.build_features_and_target` instead.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (X, y, numeric_features, categorical_features)
    """
    warnings.warn(
        "build_features_and_target from features is deprecated. "
        "Use finance_ml.ml_workflow.features.core.build_features_and_target instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_build_features_and_target(df)


# Module exports
__all__ = [
    # Phase 9.3 refactor: Core feature engineering functions (now in features.core)
    "_safe_div",
    "engineer_basic_ratios",
    "engineer_margin_features",
    "engineer_volatility_features",
    "engineer_revenue_cagr",
    "build_features_and_target",
    # Note: Advanced feature engineering functions remain in advanced_features.py
    # and will be refactored in a future phase.
]
