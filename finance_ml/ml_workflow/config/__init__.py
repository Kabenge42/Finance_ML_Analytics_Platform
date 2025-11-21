"""Configuration constants for ML workflow modules.

This package provides centralized configuration constants to ensure
consistency across the ML pipeline and avoid magic numbers scattered
throughout the codebase.

Modules:
    ml_returns_config: Configuration for ML-based return prediction (Phase 2)
"""

from .ml_returns_config import (
    MIN_DATES_FOR_TIMESERIES,
    MIN_DATES_FOR_RELIABLE_ML,
    MIN_PORTFOLIO_CANDIDATES,
    DEFAULT_EXPECTED_RETURN,
    TRAIN_SIZE,
    TARGET_COL,
    TARGET_COL_FALLBACK,
    LAG_PERIODS,
    TECHNICAL_INDICATORS,
)

__all__ = [
    "MIN_DATES_FOR_TIMESERIES",
    "MIN_DATES_FOR_RELIABLE_ML",
    "MIN_PORTFOLIO_CANDIDATES",
    "DEFAULT_EXPECTED_RETURN",
    "TRAIN_SIZE",
    "TARGET_COL",
    "TARGET_COL_FALLBACK",
    "LAG_PERIODS",
    "TECHNICAL_INDICATORS",
]
