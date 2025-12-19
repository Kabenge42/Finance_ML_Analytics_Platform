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
    TEST_SIZE,
    CV_FOLDS,
    QUANTILES,
    RANDOM_SEED,
    MODEL_VERSION,
    MIN_SECTOR_SAMPLES,
    WINSORIZE_LOWER,
    WINSORIZE_UPPER,
    TARGET_COL,
    TARGET_COL_FALLBACK,
    LAG_PERIODS,
    TECHNICAL_INDICATORS,
    # Phase 7 Enhancement constants
    MAX_EXPECTED_RETURN,
    MIN_EXPECTED_RETURN,
    REALISTIC_RETURN_MEAN_THRESHOLD,
    PRICE_COLUMNS,
    PHASE93_RETURN_FEATURE_CATEGORIES,
)

__all__ = [
    "MIN_DATES_FOR_TIMESERIES",
    "MIN_DATES_FOR_RELIABLE_ML",
    "MIN_PORTFOLIO_CANDIDATES",
    "DEFAULT_EXPECTED_RETURN",
    "TRAIN_SIZE",
    "TEST_SIZE",
    "CV_FOLDS",
    "QUANTILES",
    "RANDOM_SEED",
    "MODEL_VERSION",
    "MIN_SECTOR_SAMPLES",
    "WINSORIZE_LOWER",
    "WINSORIZE_UPPER",
    "TARGET_COL",
    "TARGET_COL_FALLBACK",
    "LAG_PERIODS",
    "TECHNICAL_INDICATORS",
    # Phase 7 Enhancement constants
    "MAX_EXPECTED_RETURN",
    "MIN_EXPECTED_RETURN",
    "REALISTIC_RETURN_MEAN_THRESHOLD",
    "PRICE_COLUMNS",
    "PHASE93_RETURN_FEATURE_CATEGORIES",
]
