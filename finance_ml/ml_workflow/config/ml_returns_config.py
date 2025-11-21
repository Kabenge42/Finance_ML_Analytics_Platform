"""Configuration constants for ML-based return prediction.

This module provides centralized configuration constants for the ML return
prediction workflow (Phase 2 of portfolio optimization enhancement plan).
All constants are designed to eliminate magic numbers and ensure consistency
across the codebase.

Compliance:
    - Code Guidelines Section 8.1: Configuration Constants
    - Code Guidelines Section 2.2: Schema Compliance

Usage:
    from finance_ml.ml_workflow.config import (
        MIN_DATES_FOR_TIMESERIES,
        DEFAULT_EXPECTED_RETURN,
        TRAIN_SIZE,
    )
"""

from typing import List

# ========== DATA VALIDATION THRESHOLDS ==========

MIN_DATES_FOR_TIMESERIES: float = 2.0
"""Minimum average dates per ticker to consider data as time-series.

If the average number of dates per ticker is below this threshold, the data
is considered cross-sectional (single snapshot) rather than time-series.

Type: float
Default: 2.0
Used by: create_ml_return_features (cross-sectional detection)
"""

MIN_DATES_FOR_RELIABLE_ML: int = 20
"""Minimum average dates per ticker for reliable ML features.

While 2 dates may technically allow time-series features, at least 20 dates
are recommended for stable lagged features and technical indicators.

Type: int
Default: 20
Used by: Notebook Section 10.2 (ML feature reliability check)
"""

MIN_PORTFOLIO_CANDIDATES: int = 3
"""Minimum number of portfolio candidates required for ML prediction.

If the number of candidates falls below this threshold, ML-based return
prediction is skipped.

Type: int
Default: 3
Used by: Notebook Section 10.2 (early exit check)
"""

# ========== DEFAULT VALUES ==========

DEFAULT_EXPECTED_RETURN: float = 0.08
"""Default expected annual return (8%) when historical data is unavailable.

This is a conservative baseline return assumption used as a fallback when
return_1y cannot be calculated from price history or other proxies.

Type: float
Default: 0.08 (8% annual return)
Used by: Notebook Section 10.2 (return_1y fallback)
"""

# ========== TRAIN/TEST SPLIT ==========

TRAIN_SIZE: float = 0.80
"""Training set proportion for train/test split.

Specifies the fraction of data used for training in time-series splits.
The remaining (1 - TRAIN_SIZE) is used for validation/testing.

Type: float
Default: 0.80 (80% train, 20% test)
Used by: Notebook Section 10.2 (linear model training)
"""

# ========== SCHEMA COMPLIANCE (Section 2.2) ==========

TARGET_COL: str = "price_target"
"""Canonical column name for price target predictions.

Primary target column for regression models following the standardized
schema naming convention.

Type: str
Default: 'price_target'
Used by: Notebook Section 10.2, schema-compliant pipelines
"""

TARGET_COL_FALLBACK: str = "last_price"
"""Fallback column name when price_target is unavailable.

Used as a proxy for current valuation when calculating returns or when
the primary target column is missing.

Type: str
Default: 'last_price'
Used by: Notebook Section 10.2 (return calculation fallback)
"""

# ========== FEATURE ENGINEERING ==========

LAG_PERIODS: List[int] = [1, 3, 6, 12]
"""Lag periods (in days) for creating lagged return features.

These lags create temporal features that capture short-term, medium-term,
and longer-term momentum signals.

Type: List[int]
Default: [1, 3, 6, 12]
Used by: Notebook Section 10.2, create_ml_return_features
"""

TECHNICAL_INDICATORS: List[str] = ["momentum", "volatility"]
"""Technical indicators to include in ML feature engineering.

Supported values:
    - "momentum": Rolling average of returns
    - "volatility": Rolling standard deviation of returns
    - "sma": Simple moving average of prices (optional)

Type: List[str]
Default: ['momentum', 'volatility']
Used by: Notebook Section 10.2, create_ml_return_features
"""
