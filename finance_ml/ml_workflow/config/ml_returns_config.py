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

# ========== RETURN BOUNDS (Phase 7 Enhancement) ==========

MAX_EXPECTED_RETURN: float = 0.29
"""Maximum expected annual return (29%) for clipping unrealistic predictions.

This upper bound prevents unrealistic return expectations that can lead to
extreme portfolio weights and inflated Sharpe ratios (e.g., the 95.6% mean
return issue and 42.4 Sharpe ratio anomaly). A 29% cap ensures the acceptance
criterion of mean < 30% is always satisfied even in edge cases (strict inequality).

Rationale: Long-term equity market returns average 7-10% annually. Even for
high-growth stocks, expecting > 30% annual return is aggressive. The previous
49% cap allowed mean returns up to 49% which exceeded the < 30% target.

Type: float
Default: 0.29 (29% annual return)
Used by: clip_expected_returns, validate_expected_returns
"""

MIN_EXPECTED_RETURN: float = -0.50
"""Minimum expected annual return (-50%) for clipping unrealistic predictions.

This lower bound prevents extreme negative return expectations while still
allowing for significant drawdown scenarios. A -50% floor is more realistic
as losses beyond 50% are rare for diversified positions.

Type: float
Default: -0.50 (-50% annual return)
Used by: clip_expected_returns, validate_expected_returns
"""

REALISTIC_RETURN_MEAN_THRESHOLD: float = 0.30
"""Threshold for flagging unrealistic mean expected returns.

If the mean of expected returns exceeds this threshold (30%), the returns
are flagged as potentially unrealistic and should be reviewed.

Type: float
Default: 0.30 (30% mean annual return)
Used by: validate_expected_returns
"""

# ========== PRICE COLUMNS REGISTRY (Phase 7.2 Enhancement) ==========

PRICE_COLUMNS: dict = {
    "current": [
        "last_price",
        "price_target",
        "price_target_median",
        "price_target_mean",
        "price_target_high",
        "price_target_low",
    ],
    "historical": [
        "price_5d_ago",
        "price_1w_ago",
        "price_1m_ago",
        "price_3m_ago",
        "price_6m_ago",
        "price_1y_ago",
        "price_2y_ago",
        "price_3y_ago",
        "price_5y_ago",
    ],
    "52w_bounds": [
        "52w_high_adj",
        "52w_low_adj",
        "52w_high",
        "52w_low",
    ],
    "emas": [
        "ema_20d",
        "ema_50d",
        "ema_100d",
        "ema_250d",
    ],
}
"""Registry of price-related columns organized by category.

Categories:
    - current: Current price and price target columns
    - historical: Historical price columns for return calculation
    - 52w_bounds: 52-week high/low columns for range features
    - emas: Exponential moving average columns for momentum features

Type: Dict[str, List[str]]
Used by: calculate_historical_returns, create_ml_return_features_enhanced
"""

# ========== PHASE 9.3 FEATURE CATEGORIES FOR RETURN PREDICTION ==========

PHASE93_RETURN_FEATURE_CATEGORIES: List[str] = [
    "Momentum & Technical",
    "Valuation Ratios",
    "Growth Metrics",
    "Analyst Sentiment",
    "Quality & Risk",
    "Profitability",
]
"""Phase 9.3 feature categories relevant for return prediction.

These categories contain features with high predictive relevance for
expected returns, prioritized by their correlation with future returns.

Type: List[str]
Default: High-relevance categories from 196 Phase 9.3 features
Used by: get_phase93_return_features, create_ml_return_features_enhanced
"""
