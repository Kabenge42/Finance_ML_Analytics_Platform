"""
Core constants for the Finance ML Analytics Platform.
"""

import pandas as pd

# Styling
PLOTLY_TEMPLATE = "plotly_dark"

COLOR_PALETTE = {
    "primary": "#375a7f",
    "secondary": "#6c757d",
    "success": "#00bc8c",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#3498db",
    "neutral": "#adb5bd",
}

# Date formats
DATE_DISPLAY_FORMAT = "%d %b %Y"
DEFAULT_REFERENCE_DATE = pd.Timestamp("2025-12-21")

# Feature category colors (unified from earnings_widgets.py and schema.py)
CATEGORY_COLORS = {
    "Momentum & Technical": "#3498db",
    "Valuation Ratios": "#375a7f",
    "Profitability": "#00bc8c",
    "Quality & Risk": "#e74c3c",
    "Cash Flow": "#f39c12",
    "Capital Allocation": "#9b59b6",
    "Analyst Sentiment": "#1abc9c",
    "Market Sentiment": "#34495e",
    "Leverage & Liquidity": "#2980b9",
    "Temporal Patterns": "#8e44ad",
    "Composite Scores": "#16a085",
    "Growth Metrics": "#27ae60",
    "Efficiency Ratios": "#d35400",
    "Employee Productivity": "#7f8c8d",
    "Balance Sheet Dynamics": "#2c3e50",
    "Revenue Forecasting": "#c0392b",
    "Earnings Quality": "#e67e22",
    "Technical Analysis": "#1abc9c",
    "Valuation Timeseries": "#3498db",
    "Dividend Reliability": "#27ae60",
    "Employment Dynamics": "#34495e",
}

# ========== ML Workflow Constants ==========

# Target Configuration
TARGET_COL = "price_target"
TARGET_COL_FALLBACK = "last_price"

# Data Split Configuration
TRAIN_SIZE = 0.80
TEST_SIZE = 0.20
CV_FOLDS = 5

# Model Configuration
QUANTILES = [0.1, 0.5, 0.9]
RANDOM_SEED = 42
MODEL_VERSION = "v9_10"

# Preprocessing Constraints
MIN_SECTOR_SAMPLES = 20
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99

# Sector/portfolio constraints
MAX_SECTOR_WEIGHT = 0.25
MAX_SINGLE_POSITION = 0.10

# Outlier detection
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0

# Confidence Thresholds
CONFIDENCE_LEVEL = 0.80
ALPHA = 1 - CONFIDENCE_LEVEL

# ========== Portfolio Optimization Constants ==========
MAX_EXPECTED_RETURN = 0.29
MIN_EXPECTED_RETURN = -0.50
REALISTIC_RETURN_MEAN_THRESHOLD = 0.30
