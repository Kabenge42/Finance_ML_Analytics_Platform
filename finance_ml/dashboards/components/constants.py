"""Shared constants for dashboard components."""

# Standard color palette (aligned with code_guidelines.md Section 17.1)
COLOR_PALETTE = {
    "primary": "#375a7f",
    "secondary": "#6c757d",
    "success": "#00bc8c",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#3498db",
    "neutral": "#adb5bd",
}

# Plotly template (aligned with code_guidelines.md Section 17.2)
PLOTLY_TEMPLATE = "plotly_dark"

# Font configuration (aligned with code_guidelines.md Section 17.4)
FONT_FAMILY = "Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
FONT_SIZES = {
    "h1": 32,  # 2rem
    "h2": 24,  # 1.5rem
    "h3": 20,  # 1.25rem
    "body": 16,  # 1rem
    "caption": 14,  # 0.875rem
}

# Standard Plotly layout configuration
PLOTLY_LAYOUT_DEFAULTS = {
    "font": {"family": FONT_FAMILY, "size": FONT_SIZES["caption"]},
    "title_font_size": FONT_SIZES["h3"],
    "showlegend": True,
    "legend": {
        "orientation": "v",
        "yanchor": "top",
        "xanchor": "right",
        "x": 1.02,
        "y": 1,
    },
    "hovermode": "closest",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
}

# Standard DataTable styles (aligned with code_guidelines.md Section 17.3)
TABLE_STYLE_CELL = {
    "backgroundColor": "#111",
    "color": "#ffffff",
    "border": f"1px solid {COLOR_PALETTE['secondary']}",
    "fontFamily": FONT_FAMILY,
    "fontSize": f"{FONT_SIZES['caption']}px",
    "padding": "8px",
    "whiteSpace": "normal",
    "height": "auto",
    "minWidth": "80px",
}

TABLE_STYLE_HEADER = {
    "backgroundColor": COLOR_PALETTE["primary"],
    "fontWeight": "bold",
    "color": "#ffffff",
}

TABLE_STYLE_TABLE = {
    "overflowX": "auto",
    "maxHeight": "500px",
    "overflowY": "auto",
}

# =============================================================================
# Dashboard Configuration Constants
# =============================================================================

# DataTable pagination defaults
DEFAULT_PAGE_SIZE_CALENDAR = 15
DEFAULT_PAGE_SIZE_ALERTS = 20
DEFAULT_PAGE_SIZE_EXPLORER = 20

# Earnings calendar defaults
DEFAULT_EARNINGS_DAYS_WINDOW = 10
MIN_EARNINGS_DAYS_WINDOW = 3
MAX_EARNINGS_DAYS_WINDOW = 30
EARNINGS_DAYS_MARKS = {3: "3", 7: "7", 10: "10", 14: "14", 21: "21", 30: "30"}

# Top N defaults
DEFAULT_TOP_N = 50
MIN_TOP_N = 10
MAX_TOP_N = 200
TOP_N_STEP = 10

# Alert thresholds (default values for UI inputs)
DEFAULT_EPS_MISS_THRESHOLD = 20.0
DEFAULT_DOWNGRADE_THRESHOLD = 5.0
DEFAULT_MIN_DOWNGRADE_PERIODS = 2
DEFAULT_TARGET_SPREAD_THRESHOLD = 30.0
DEFAULT_PRE_EARNINGS_WINDOW_DAYS = 7
DEFAULT_VOLATILITY_QUANTILE = 0.75
DEFAULT_MAX_TICKERS_PER_ALERT = 10

# Explorer defaults
DEFAULT_EXPLORER_ROW_LIMIT = 200
EXPLORER_ROW_LIMIT_STEP = 50
MIN_EXPLORER_ROW_LIMIT = 10
