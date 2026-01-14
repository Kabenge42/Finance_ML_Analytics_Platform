"""
Earnings Dashboard Widgets.

This module provides visualization and analysis tools for earnings-related data,
including earnings calendars, quality analysis, surprise dashboards, and alerts.

Key Functions:
    - display_earnings_dashboard: Pandas Styler-based earnings calendar display
    - create_earnings_calendar_dashboard: Core dashboard data preparation
    - analyze_earnings_quality: GAAP vs Adjusted earnings analysis
    - create_earnings_surprise_dashboard: Plotly earnings surprise visualization
    - generate_earnings_quality_alerts: Alert generation for earnings events

Schema Alignment:
    Uses COLUMN_SCHEMA and PHASE93_FEATURE_CATEGORIES from finance_ml.core.schema
    for consistent column references and formatting.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, TypedDict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from finance_ml.core.constants import PLOTLY_TEMPLATE, COLOR_PALETTE
from finance_ml.core.schema import (
    COLUMN_SCHEMA,
    PHASE93_FEATURE_CATEGORIES,
    list_date_cols,
)
from .base import (
    EarningsMode,
    resolve_reference_date,
    _write_html_artifact,
    _build_format_dict,
    _ensure_schema_dtypes,
    EarningsAlertConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Category Metrics Resolution
# =============================================================================
class CategoryMetricsResolver:
    """Resolves category names to metrics from PHASE93_FEATURE_CATEGORIES."""

    _SHORT_NAME_ALIASES: Dict[str, str] = {
        "profitability": "Profitability",
        "valuation": "Valuation Ratios",
        "growth": "Growth Metrics",
        "momentum": "Momentum & Technical",
        "quality_risk": "Quality & Risk",
        "cash_flow": "Cash Flow",
        "capital_allocation": "Capital Allocation",
        "analyst_sentiment": "Analyst Sentiment",
        "market_sentiment": "Market Sentiment",
        "leverage_liquidity": "Leverage & Liquidity",
        "temporal": "Temporal Patterns",
        "composite": "Composite Scores",
        "growth_metrics": "Growth Metrics",
        "efficiency": "Efficiency Ratios",
        "productivity": "Employee Productivity",
        "balance_sheet": "Balance Sheet Dynamics",
        "forecasts": "Revenue Forecasting",
        "earnings_quality": "Earnings Quality",
        "dividends": "Dividend Reliability",
        # NEW: Add missing aliases for v1.15 categories
        "technical": "Technical Analysis",
        "valuation_ts": "Valuation Timeseries",
        "employment": "Employment Dynamics",
    }

    # Cross-category feature supplements (features that span multiple domains)
    _CROSS_CATEGORY_SUPPLEMENTS: Dict[str, List[str]] = {
        "Profitability": [
            "ebitda_vs_5y_avg",
            "ebitda_stability_score",
            "ebit_vs_5y_avg",
            "operating_leverage_ratio",
            "gross_margin_consistency",
        ],
        "Earnings Quality": [
            "normalized_vs_gaap_spread",
            "normalized_vs_gaap_ratio",
            "forward_eps_gaap_adjusted_spread",
            "earnings_stability_score",
        ],
    }

    @classmethod
    def resolve_name(cls, category: str) -> str:
        """Resolve short category name to full PHASE93_FEATURE_CATEGORIES key."""
        return cls._SHORT_NAME_ALIASES.get(category.lower(), category)

    @classmethod
    def get_metrics(
        cls,
        categories: List[str],
        include_supplemental: bool = True,
    ) -> Dict[str, List[str]]:
        """Get metrics from specified PHASE93_FEATURE_CATEGORIES categories.

        Args:
            categories: List of category names or short aliases.
            include_supplemental: Whether to include cross-category supplements.

        Returns:
            Dict mapping category name to list of metric column names.
        """
        result: Dict[str, List[str]] = {}

        for cat in categories:
            full_cat = cls.resolve_name(cat)
            # Pull directly from schema - single source of truth
            metrics = PHASE93_FEATURE_CATEGORIES.get(full_cat, []).copy()
            result[full_cat] = metrics

        if include_supplemental:
            # Add cross-category supplements for features not yet in schema or needed across domains
            for category, supplements in cls._CROSS_CATEGORY_SUPPLEMENTS.items():
                if category in result:
                    existing = set(result[category])
                    result[category].extend(
                        [s for s in supplements if s not in existing]
                    )

        return result


# Public API functions (maintain backward compatibility)
def _resolve_category_name(category: str) -> str:
    """Resolve short category name to full PHASE93_FEATURE_CATEGORIES key."""
    return CategoryMetricsResolver.resolve_name(category)


def get_category_metrics(
    categories: List[str],
    include_supplemental: bool = True,
) -> Dict[str, List[str]]:
    """Get metrics from specified PHASE93_FEATURE_CATEGORIES categories.

    See CategoryMetricsResolver.get_metrics for full documentation.
    """
    return CategoryMetricsResolver.get_metrics(categories, include_supplemental)


def create_earnings_calendar_dashboard(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 100,
    mode: EarningsMode = "all",
    categories: Optional[List[str]] = None,
    days_window: Optional[int] = 10,
) -> pd.DataFrame:
    """
    Creates a dashboard (styled DataFrame) for Earnings and Dividend Analytics.
    Filters for companies with upcoming or recent earnings.

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Uses PHASE93_FEATURE_CATEGORIES categories for metric selection.

    Args:
        df: Input DataFrame containing stock data.
        reference_date: Date to compare next_earnings against. Defaults to today.
        top_n: Number of top companies (by Market Cap) to include.
        mode: Display mode - 'all', 'earnings', 'dividends', 'earnings_quality',
            or specific category name.
        categories: Optional list of specific PHASE93 categories to include.
            Overrides mode if provided.
        days_window: Optional filter for next_earnings within +/- N days.
            Defaults to 10-day window; pass None to disable temporal filtering.

    Returns:
        pd.DataFrame: Filtered DataFrame with selected metrics.
    """
    critical_columns = ["ticker", "sector"]
    missing_critical = [col for col in critical_columns if col not in df.columns]
    if missing_critical:
        logger.warning("Missing critical columns: %s", missing_critical)
        return pd.DataFrame()

    reference_date = resolve_reference_date(df, reference_date)

    # Schema-aligned date columns for earnings calendar (ordered by preference)
    # These columns are defined in COLUMN_SCHEMA with role="date"
    earnings_date_candidates = [
        "next_earnings",
        "fy_end_date",
        "next_fy_end_date",
        "income_statement_report_date",
        "dividend_record_ex_date",
        "dividend_record_payable_date",
        "dividend_record_announce_date",
        "dividend_record_record_date",
    ]

    # Ensure date columns are datetime
    for col in earnings_date_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Determine the best available earnings date column
    anchor_date_col = None
    for col in earnings_date_candidates:
        if col in df.columns and df[col].notna().any():
            anchor_date_col = col
            logger.info(f"Using '{col}' as anchor date column for earnings calendar")
            break

    filtered_df = df.copy()

    # Apply temporal filtering only if we have a valid anchor column and days_window
    if anchor_date_col is not None and days_window is not None:
        temporal_window = timedelta(days=days_window)
        # Calculate days difference, handling NaT values
        date_diff = (filtered_df[anchor_date_col] - reference_date).abs()
        mask = date_diff <= temporal_window
        filtered_df = filtered_df[mask]

        logger.info(
            f"Temporal filter applied: {mask.sum()} rows within ±{days_window} days "
            f"(anchor: {anchor_date_col})"
        )
    elif anchor_date_col is None:
        # No valid date column found - log warning but continue without temporal filter
        logger.warning(
            "No valid earnings date column found with non-null values. "
            f"Checked columns: {earnings_date_candidates}. "
            "Proceeding without temporal filtering."
        )

    if filtered_df.empty:
        logger.warning(
            f"No companies found after filtering. Reference date: {reference_date}, "
            f"Days window: ±{days_window}, Anchor column: {anchor_date_col}"
        )
        return pd.DataFrame()

    # Sort by next_earnings date in ascending order (soonest first)
    if anchor_date_col and anchor_date_col in filtered_df.columns:
        filtered_df = filtered_df.sort_values(
            by=anchor_date_col, ascending=True, na_position="last"
        )

    filtered_df = filtered_df.head(top_n)

    # Get market cap column for display purposes (not for sorting)
    mcap_col = _get_market_cap_column(filtered_df)

    # Define identity columns + temporal enrichments when available
    display_cols = [
        "isin",
        "ticker",
        "name",
        "exchange",
        "sector",
        "country",
        "trading_country",
        "industry",
        "region",
        "income_statement_report_date",
        "next_earnings",
        "days_to_earnings",
        "dividend_record_announce_date",
        "dividend_record_ex_date",
        "fy_end_date",
        "next_fy_end_date",
        "current_fiscal_quarter",
        "next_fiscal_quarter",
    ]
    display_cols = [
        c
        for c in display_cols
        if c in df.columns
        or c == "next_earnings"
        or c == "days_to_earnings"
        or c == "fy_end_date"
        or c == "next_fy_end_date"
        or c == "current_fiscal_quarter"
        or c == "next_fiscal_quarter"
    ]
    if mcap_col and mcap_col not in display_cols:
        display_cols.append(mcap_col)

    temporal_enrichments = [
        col
        for col in ["fiscal_quarter_inferred", "fy_end_vs_isrd_days"]
        if col in filtered_df.columns
    ]

    # Determine which categories to include based on mode or explicit categories
    if categories is not None:
        selected_categories = categories
    elif mode == "all":
        # Include all major categories for comprehensive view
        selected_categories = [
            "Profitability",
            "Valuation Ratios",
            "Growth Metrics",
            "Momentum & Technical",
            "Quality & Risk",
            "Cash Flow",
            "Dividend Reliability",
            "Revenue Forecasting",
            "Earnings Quality",
        ]
    elif mode == "earnings":
        # Earnings-focused categories
        selected_categories = [
            "Profitability",
            "Valuation Ratios",
            "Growth Metrics",
            "Momentum & Technical",
            "Revenue Forecasting",
            "Earnings Quality",
        ]
    elif mode == "dividends":
        # Dividend-focused categories
        selected_categories = ["Dividend Reliability", "Cash Flow"]
    elif mode in PHASE93_FEATURE_CATEGORIES:
        # Single category mode
        selected_categories = [mode]
    else:
        # Default to earnings mode
        selected_categories = [
            "Profitability",
            "Growth Metrics",
            "Momentum & Technical",
        ]

    # Get metrics from selected categories
    category_metrics = get_category_metrics(
        selected_categories, include_supplemental=True
    )

    # Build final columns list
    final_cols = display_cols.copy()
    for cat, metrics in category_metrics.items():
        existing_metrics = [c for c in metrics if c in df.columns]
        final_cols.extend(existing_metrics)

    final_cols.extend(temporal_enrichments)

    # Remove duplicates while preserving order
    final_cols = list(dict.fromkeys(final_cols))

    # Filter to only columns that exist
    final_cols = [c for c in final_cols if c in filtered_df.columns]

    dashboard_df = filtered_df[final_cols].copy()

    # Add computed columns - Use the best available date column for days_to_earnings
    date_col_for_days = anchor_date_col or "next_earnings"
    if date_col_for_days in dashboard_df.columns:
        dashboard_df["days_to_earnings"] = (
            pd.to_datetime(dashboard_df[date_col_for_days], errors="coerce")
            - reference_date
        ).dt.days

        # Reorder: Put days_to_earnings near the anchor date column
        cols = list(dashboard_df.columns)
        if "days_to_earnings" in cols:
            cols.remove("days_to_earnings")
            if date_col_for_days in cols:
                idx = cols.index(date_col_for_days) + 1
                cols.insert(idx, "days_to_earnings")
            elif "next_earnings" in cols:
                idx = cols.index("next_earnings") + 1
                cols.insert(idx, "days_to_earnings")
            dashboard_df = dashboard_df[cols]

    return dashboard_df


def display_earnings_dashboard(
    df: pd.DataFrame,
    mode: EarningsMode = "all",
    categories: Optional[List[str]] = None,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 100,
    date_columns: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional["pd.io.formats.style.Styler"]:
    """
    Displays the earnings dashboard using Pandas Styler with enhanced formatting.

    **ETL Pipeline Compatibility:**
    - Expects DataFrame from etl.py Stage 11 (post-validation) for best results
    - Respects semantic column classifications (code_guidelines.md §8.5)
    - Preserves PRICE_COLUMNS formatting per preservation policy

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Supports all PHASE93_FEATURE_CATEGORIES categories with appropriate formatting:
    - Currency formatting for financial metrics
    - Percentage formatting for yields, margins, returns
    - Ratio formatting for valuation multiples
    - Color-coded days_to_earnings indicator

    **Style Guide Alignment (code_guidelines.md §17.1):**
    - Color palette: danger (red) for past, warning (yellow) for today,
      success (green) for future earnings
    - Consistent number formatting with appropriate precision
    - NA values displayed as "-"

    Args:
        df: Input DataFrame containing stock data.
        mode: Display mode - 'all', 'earnings', 'dividends', or specific category.
        categories: Optional list of specific PHASE93 categories to include.
        reference_date: Date for earnings comparison. Defaults to today.
        top_n: Number of top companies to include.
        date_columns: Optional list of date columns to format (auto-detected if None).
        output_path: Optional path to save HTML output.

    Returns:
        pd.io.formats.style.Styler: Styled DataFrame for display, or None if empty.
    """
    # Create the base dashboard DataFrame using existing logic
    dashboard_df = create_earnings_calendar_dashboard(
        df,
        reference_date=reference_date,
        top_n=top_n,
        mode=mode,
        categories=categories,
    )

    if dashboard_df.empty:
        logger.warning("No companies found with earnings within the specified window.")
        return None

    # Default date columns for earnings dashboard if not provided
    if date_columns is None:
        date_columns = [
            "next_earnings",
            "income_statement_report_date",
            "fy_end_date",
            "next_fy_end_date",
            "fy_end",
            "_reference_date",
            "reference_date",
            "dividend_record_ex_date",
            "dividend_record_payable_date",
            "dividend_record_announce_date",
            "dividend_record_record_date",
        ]

    # Filter to columns that exist in dashboard_df
    date_columns = [c for c in date_columns if c in dashboard_df.columns]

    # Ensure schema-compliant dtypes before styling
    df_styled = _ensure_schema_dtypes(dashboard_df, date_columns=date_columns)

    # Build schema-aware format dictionary with safe callables
    format_dict = _build_format_dict(
        df_styled,
        date_columns=date_columns,
    )

    # Apply styling with safe formatters
    styler = df_styled.style.format(format_dict, na_rep="—")

    # Color-code days_to_earnings (aligned with code_guidelines.md §17.1 colors)
    def color_days(val):
        if pd.isna(val):
            return ""
        try:
            val_float = float(val)
            if val_float < 0:
                return f"color: {COLOR_PALETTE['danger']}"  # Past
            if val_float == 0:
                return f"background-color: {COLOR_PALETTE['warning']}; color: black"  # Today
            if val_float > 0:
                return f"color: {COLOR_PALETTE['success']}"  # Future
        except (ValueError, TypeError):
            pass
        return ""

    if "days_to_earnings" in df_styled.columns:
        styler = styler.map(color_days, subset=["days_to_earnings"])

    # --- Conditional Formatting for Metric/Indicator Columns (§17.3) ---
    numeric_cols = df_styled.select_dtypes(include=[np.number]).columns.tolist()

    # Quality/Score metrics: higher is better (green)
    quality_score_cols = [
        c
        for c in [
            "earnings_quality_score",
            "piotroski_f_score",
            "altman_z_score",
            "accounting_quality_score",
            "dividend_reliability_score",
        ]
        if c in numeric_cols
    ]

    # Profitability metrics: higher is better (green)
    profitability_cols = [
        c
        for c in [
            "roe",
            "roa",
            "roic",
            "gross_margin_pct",
            "operating_margin_pct",
            "net_margin_pct",
            "ebitda_margin_pct",
        ]
        if c in numeric_cols
    ]

    # Growth metrics: higher is better (green)
    growth_cols = [
        c
        for c in [
            "revenue_growth_yoy",
            "ebitda_growth_yoy",
            "eps_growth_yoy",
            "revenue_cagr_3y",
            "eps_cagr_3y",
        ]
        if c in numeric_cols
    ]

    # Momentum metrics: can be positive or negative
    momentum_cols = [
        c
        for c in [
            "price_momentum_1m",
            "price_momentum_3m",
            "price_momentum_6m",
            "eps_surprise_pct",
            "revenue_surprise_pct",
        ]
        if c in numeric_cols
    ]

    # Valuation metrics: lower is typically better (reverse gradient)
    valuation_cols = [
        c
        for c in [
            "p_e_ltm",
            "p_e_ntm",
            "ev_ebitda_ltm",
            "p_b_ratio",
            "p_s_ratio",
        ]
        if c in numeric_cols
    ]

    # Dividend metrics: higher yield is better
    dividend_cols = [
        c
        for c in [
            "div_yield_ltm",
            "div_yield_ntm",
            "dividend_payout_ratio",
        ]
        if c in numeric_cols
    ]

    # Apply background gradients for identified metric columns
    if quality_score_cols:
        styler = styler.background_gradient(
            subset=quality_score_cols,
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
        )

    if profitability_cols:
        styler = styler.background_gradient(
            subset=profitability_cols,
            cmap="RdYlGn",
            vmin=-50,
            vmax=50,
        )

    if growth_cols:
        styler = styler.background_gradient(
            subset=growth_cols,
            cmap="RdYlGn",
            vmin=-50,
            vmax=50,
        )

    if momentum_cols:
        styler = styler.background_gradient(
            subset=momentum_cols,
            cmap="RdYlGn",
            vmin=-30,
            vmax=30,
        )

    if valuation_cols:
        # Reverse gradient: lower valuation = greener
        styler = styler.background_gradient(
            subset=valuation_cols,
            cmap="RdYlGn_r",
            vmin=0,
            vmax=50,
        )

    if dividend_cols:
        styler = styler.background_gradient(
            subset=dividend_cols,
            cmap="RdYlGn",
            vmin=0,
            vmax=10,
        )


# =============================================================================
# Schema-Driven Column Configuration
# =============================================================================


def _get_schema_columns_by_role(role: str) -> List[str]:
    """Get columns from COLUMN_SCHEMA matching a specific role."""
    return [col for col, meta in COLUMN_SCHEMA.items() if meta.get("role") == role]


def _get_market_value_columns() -> List[str]:
    """Get market value columns from schema (replaces _MARKET_CAP_COLUMNS)."""
    market_cols = [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta.get("role") == "market" and "market_cap" in col
    ]
    # Preserve preference order
    preferred_order = ["market_cap", "market_cap_usd", "market_cap_country_r"]
    return [c for c in preferred_order if c in market_cols] + [
        c for c in market_cols if c not in preferred_order
    ]


# Schema-driven earnings date columns (role="date" with earnings semantics)
_EARNINGS_DATE_COLUMNS: List[str] = [
    col
    for col in list_date_cols()
    if any(kw in col for kw in ["earnings", "fy_end", "income_statement", "dividend"])
]

# GAAP vs Adjusted EPS column pairs from schema (role="ratio" with eps semantics)
_GAAP_ADJUSTED_PAIRS: List[tuple[str, str]] = [
    ("eps_gaap_est_avg_fy1e", "eps_norm_est_avg_fy1e"),
    ("eps_gaap_est_avg_ntm", "eps_norm_est_avg_ntm"),
    ("net_eps_basic_ltm", "eps_adj_ltm"),
    ("net_eps_basic_fy", "eps_adj_fy"),
    ("net_eps_basic_1fy", "eps_adj_1fy"),
]

# Earnings surprise column pairs: (actual_column, estimate_column)
# =============================================================================
# SURPRISE PAIRS & ANALYTICS HELPER
# =============================================================================


def _get_earnings_surprise_pairs() -> Dict[str, tuple[str, str]]:
    """Build earnings surprise column pairs from COLUMN_SCHEMA.

    Returns actual/estimate column pairs for surprise calculation,
    validated against schema definitions.
    """
    # Define pairs with schema-validated column names
    pairs = {
        "Revenue": ("total_revenues_ltm", "revenues_est_avg_ntm"),
        "EBITDA": ("ebitda_ltm", "ebitda_est_avg_fy1e"),
        "EBIT": ("ebit_ltm", "ebit_est_med_ntm"),
        "Net Income": ("net_income_is_ltm", "net_income_adj_1fy"),
        "EPS": ("eps_adj_ltm", "eps_norm_est_avg_ntm"),
    }

    # Validate all columns exist in schema
    validated_pairs = {}
    for metric, (actual, estimate) in pairs.items():
        if actual in COLUMN_SCHEMA and estimate in COLUMN_SCHEMA:
            validated_pairs[metric] = (actual, estimate)
        else:
            logger.debug(f"Surprise pair for {metric} not fully in schema")

    return validated_pairs


# Use at module level
_EARNINGS_SURPRISE_PAIRS = _get_earnings_surprise_pairs()


# =============================================================================
# Gradient Configuration (Schema-Aligned)
# =============================================================================


class GradientConfig(TypedDict):
    """Configuration for background gradient styling."""

    cmap: str
    vmin: float
    vmax: float


# Schema-aligned metric column patterns for conditional formatting
_METRIC_GRADIENT_CONFIG: Dict[str, tuple[List[str], GradientConfig]] = {
    "quality_score": (
        [
            "earnings_quality_score",
            "piotroski_f_score",
            "altman_z_score",
            "accounting_quality_score",
            "dividend_reliability_score",
        ],
        {"cmap": "RdYlGn", "vmin": 0, "vmax": 100},
    ),
    "profitability": (
        [
            "roe",
            "roa",
            "roic",
            "gross_margin_pct",
            "operating_margin_pct",
            "net_margin_pct",
            "ebitda_margin_pct",
        ],
        {"cmap": "RdYlGn", "vmin": -50, "vmax": 50},
    ),
    "growth": (
        [
            "revenue_growth_yoy",
            "ebitda_growth_yoy",
            "eps_growth_yoy",
            "revenue_cagr_3y",
            "eps_cagr_3y",
            "revenue_growth",
            "ebitda_growth",
        ],
        {"cmap": "RdYlGn", "vmin": -50, "vmax": 50},
    ),
    "momentum": (
        [
            "price_momentum_1m",
            "price_momentum_3m",
            "price_momentum_6m",
            "eps_surprise_pct",
            "revenue_surprise_pct",
        ],
        {"cmap": "RdYlGn", "vmin": -30, "vmax": 30},
    ),
    "valuation": (
        ["p_e_ltm", "p_e_ntm", "ev_ebitda_ltm", "p_b_ratio", "p_s_ratio"],
        {"cmap": "RdYlGn_r", "vmin": 0, "vmax": 50},  # Reverse: lower is better
    ),
    "dividend": (
        ["div_yield_ltm", "div_yield_ntm", "dividend_payout_ratio"],
        {"cmap": "RdYlGn", "vmin": 0, "vmax": 10},
    ),
}


# =============================================================================
# Styling Helper Functions (Extracted for Testability)
# =============================================================================


def _color_days_to_earnings(val) -> str:
    """Color-code days_to_earnings values per code_guidelines.md §17.1."""
    if pd.isna(val):
        return ""
    try:
        val_float = float(val)
        if val_float < 0:
            return f"color: {COLOR_PALETTE['danger']}"  # Past
        if val_float == 0:
            return (
                f"background-color: {COLOR_PALETTE['warning']}; color: black"  # Today
            )
        if val_float > 0:
            return f"color: {COLOR_PALETTE['success']}"  # Future
    except (ValueError, TypeError):
        pass
    return ""


def _get_default_date_columns() -> List[str]:
    """Get default date columns for earnings dashboard from schema."""
    schema_dates = set(list_date_cols())
    preferred = [
        "next_earnings",
        "income_statement_report_date",
        "fy_end_date",
        "next_fy_end_date",
        "fy_end",
        "_reference_date",
        "reference_date",
        "dividend_record_ex_date",
        "dividend_record_payable_date",
        "dividend_record_announce_date",
        "dividend_record_record_date",
    ]
    return [c for c in preferred if c in schema_dates or c.startswith("_")]


def _apply_metric_gradients(
    styler: "pd.io.formats.style.Styler",
    numeric_cols: List[str],
) -> "pd.io.formats.style.Styler":
    """Apply schema-aligned background gradients to metric columns."""
    for category, (column_list, config) in _METRIC_GRADIENT_CONFIG.items():
        available_cols = [c for c in column_list if c in numeric_cols]
        if available_cols:
            styler = styler.background_gradient(subset=available_cols, **config)
    return styler


def _get_table_styles() -> List[Dict]:
    """Get standard table styles per code_guidelines.md §17.3."""
    return [
        {
            "selector": "table",
            "props": [
                ("width", "100%"),
                ("table-layout", "auto"),
                ("border-collapse", "collapse"),
            ],
        },
        {
            "selector": "th",
            "props": [
                ("white-space", "nowrap"),
                ("font-weight", "bold"),
                ("text-align", "center"),
                ("padding", "8px 12px"),
                ("background-color", "#2c3e50"),
                ("color", "#ecf0f1"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("white-space", "nowrap"),
                ("padding", "6px 10px"),
                ("text-align", "right"),
            ],
        },
        {
            "selector": "td:nth-child(-n+10)",
            "props": [("text-align", "left")],
        },
    ]


# =============================================================================
# Market Cap and Validation Utilities
# =============================================================================


def _get_market_cap_column(df: pd.DataFrame) -> Optional[str]:
    """Find the best available market cap column using schema metadata."""
    for col in _get_market_value_columns():
        if col in df.columns:
            return col
    return None


def _validate_surprise_columns(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate availability of earnings surprise columns against schema."""
    return {
        metric: (actual in df.columns and estimate in df.columns)
        for metric, (actual, estimate) in _EARNINGS_SURPRISE_PAIRS.items()
    }


# ... existing code for CategoryMetricsResolver ...


# =============================================================================
# Main Dashboard Functions (Refactored)
# =============================================================================


def display_earnings_dashboard(
    df: pd.DataFrame,
    mode: EarningsMode = "all",
    categories: Optional[List[str]] = None,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 100,
    date_columns: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional["pd.io.formats.style.Styler"]:
    """
    Displays the earnings dashboard using Pandas Styler with enhanced formatting.

    Schema Alignment: Uses COLUMN_SCHEMA for column classification and
    PHASE93_FEATURE_CATEGORIES for metric grouping.
    """
    dashboard_df = create_earnings_calendar_dashboard(
        df,
        reference_date=reference_date,
        top_n=top_n,
        mode=mode,
        categories=categories,
    )

    if dashboard_df.empty:
        logger.warning("No companies found with earnings within the specified window.")
        return None

    # Use schema-driven date columns
    if date_columns is None:
        date_columns = _get_default_date_columns()
    date_columns = [c for c in date_columns if c in dashboard_df.columns]

    df_styled = _ensure_schema_dtypes(dashboard_df, date_columns=date_columns)
    format_dict = _build_format_dict(df_styled, date_columns=date_columns)
    styler = df_styled.style.format(format_dict, na_rep="—")

    # Apply days_to_earnings coloring
    if "days_to_earnings" in df_styled.columns:
        styler = styler.map(_color_days_to_earnings, subset=["days_to_earnings"])

    # Apply schema-aligned metric gradients
    numeric_cols = df_styled.select_dtypes(include=[np.number]).columns.tolist()
    styler = _apply_metric_gradients(styler, numeric_cols)

    # Caption and table styles
    mode_display = mode.replace("_", " ").title()
    styler = styler.set_caption(
        f"Earnings Calendar Dashboard - Mode: {mode_display} "
        f"(Top {len(df_styled)} by Upcoming Earnings)"
    )
    styler = styler.set_table_styles(_get_table_styles())

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        styler.to_html(output_path)
        logger.info("Saved earnings dashboard to %s", output_path)

    return styler


def analyze_earnings_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze discrepancies between GAAP and Adjusted earnings.

    Schema Alignment: Uses _GAAP_ADJUSTED_PAIRS derived from COLUMN_SCHEMA
    for consistent column references.
    """
    base_cols = [
        c
        for c in [
            "ticker",
            "name",
            "exchange",
            "sector",
            "region",
            "fy_end_date",
            "next_fiscal_quarter",
        ]
        if c in df.columns
    ]
    earnings_quality = df[base_cols].copy()

    for gaap_col, adj_col in _GAAP_ADJUSTED_PAIRS:
        if gaap_col in df.columns and adj_col in df.columns:
            gaap_data = pd.to_numeric(df[gaap_col], errors="coerce")
            adj_data = pd.to_numeric(df[adj_col], errors="coerce")

            with np.errstate(divide="ignore", invalid="ignore"):
                adjustment = (adj_data - gaap_data) / gaap_data.abs()
            adjustment = adjustment.replace([np.inf, -np.inf], np.nan)

            suffix = gaap_col.split("_")[-1]
            earnings_quality[f"adj_magnitude_{suffix}"] = adjustment * 100
            earnings_quality[f"large_adj_flag_{suffix}"] = (
                earnings_quality[f"adj_magnitude_{suffix}"].abs() > 35
            )

    adj_cols = [c for c in earnings_quality.columns if "adj_magnitude" in c]
    if adj_cols:
        earnings_quality["earnings_quality_score"] = 100 - earnings_quality[
            adj_cols
        ].abs().mean(axis=1).clip(0, 100)

    return earnings_quality


def create_earnings_metrics_chart(
    df: pd.DataFrame,
    metric_category: str = "PHASE93_FEATURE_CATEGORIES",
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 20,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """
    Creates an interactive Plotly chart showing metrics for upcoming earnings.

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Visualizes metrics from specified PHASE93_FEATURE_CATEGORIES category.

    **Style Guide Alignment (code_guidelines.md §17.2):**
    - Uses PLOTLY_TEMPLATE ('plotly_dark')
    - Standard color palette from COLOR_PALETTE
    - Hover data includes ticker, sector, region
    - Labeled axes with units

    Args:
        df: Input DataFrame containing stock data.
        metric_category: PHASE93_FEATURE_CATEGORIES category or short name to visualize.
        reference_date: Date for earnings comparison. Defaults to today.
        top_n: Number of companies to include.
        output_path: Optional path to save HTML output.

    Returns:
        go.Figure: Plotly figure object.
    """
    # Standardize category name
    metric_category = _resolve_category_name(metric_category)

    reference_date = resolve_reference_date(df, reference_date)

    # Get dashboard data for the specific category
    dashboard_df = create_earnings_calendar_dashboard(
        df,
        reference_date=reference_date,
        top_n=top_n,
        mode=(
            metric_category
            if metric_category in PHASE93_FEATURE_CATEGORIES
            else "earnings"
        ),
        categories=(
            [metric_category] if metric_category in PHASE93_FEATURE_CATEGORIES else None
        ),
    )

    if dashboard_df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No companies found with earnings within +/- 10 days",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    # Get metrics for the category
    category_metrics = PHASE93_FEATURE_CATEGORIES.get(metric_category, [])
    available_metrics = [c for c in category_metrics if c in dashboard_df.columns][:5]

    if not available_metrics:
        # Fallback to any numeric columns
        numeric_cols = dashboard_df.select_dtypes(include=["float64", "int64"]).columns
        available_metrics = [
            c
            for c in numeric_cols
            if c not in ["days_to_earnings"] and "date" not in c.lower()
        ][:5]

    if not available_metrics:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No metrics available for category: {metric_category}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    # Create subplots for multiple metrics
    n_metrics = len(available_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + 1) // 2

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[m.replace("_", " ").title() for m in available_metrics],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Add bar charts for each metric
    for i, metric in enumerate(available_metrics):
        row = i // 2 + 1
        col = i % 2 + 1

        # Sort by metric value
        plot_df = dashboard_df.dropna(subset=[metric]).head(15)
        if plot_df.empty:
            continue

        plot_df = plot_df.sort_values(by=metric, ascending=True)

        # Determine color based on value sign
        colors = [
            COLOR_PALETTE["success"] if v >= 0 else COLOR_PALETTE["danger"]
            for v in plot_df[metric]
        ]

        fig.add_trace(
            go.Bar(
                x=plot_df[metric],
                y=plot_df["ticker"] if "ticker" in plot_df.columns else plot_df.index,
                orientation="h",
                marker_color=colors,
                name=metric.replace("_", " ").title(),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + f"{metric}: "
                    + "%{x:.2f}<br>"
                    + "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    # Update layout
    category_display = metric_category.replace("_", " ").title()
    fig.update_layout(
        title=dict(
            text=f"Earnings Calendar: {category_display} Metrics (Top {top_n})",
            font=dict(size=20),
        ),
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=12),
        height=300 * n_rows,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    _write_html_artifact(fig, output_path)

    return fig


def create_earnings_surprise_dashboard(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 50,
    output_path: Optional[Union[str, Path]] = None,
    validate_quality: bool = False,
    use_precomputed: bool = True,
) -> go.Figure:
    """Create an interactive dashboard for earnings surprise analysis.

    Business objective: monitor expected vs actual earnings performance to
    identify forecast reliability and potential market reaction patterns.

    **ETL Pipeline Integration:**
    - Best paired with data from etl_with_financial_metrics() output (semantic typing)
    - Optional quality validation mirrors ETL Stage 9 checks
    - Supports pre-computed surprise columns from ETL pipeline

    **Schema Alignment:**
    - Uses COLUMN_SCHEMA-defined columns for market cap detection (market_value role)
    - Surprise mappings reference schema-validated actual/estimate column pairs
    - Pre-computed columns (eps_surprise_pct, etc.) defined in COLUMN_SCHEMA

    Args:
        df: DataFrame with earnings estimates and actuals.
        reference_date: Analysis date (defaults to now).
        top_n: Number of rows to analyze (prefers market cap ordering when available).
        output_path: Optional path to save an HTML dashboard.
        validate_quality: Run ETL-style quality checks before processing.
        use_precomputed: Use schema-defined surprise columns if available
            (eps_surprise_pct, revenue_surprise_pct, ebitda_surprise_pct).

    Returns:
        go.Figure: Plotly figure.
    """
    reference_date = resolve_reference_date(df, reference_date)

    # Check for pre-computed surprise columns from ETL (COLUMN_SCHEMA defines these)
    precomputed_surprise_cols = {
        "EPS": "eps_surprise_pct",
        "Revenue": "revenue_surprise_pct",
        "EBITDA": "ebitda_surprise_pct",
    }

    has_precomputed = use_precomputed and any(
        col in df.columns for col in precomputed_surprise_cols.values()
    )

    if validate_quality:
        # Use schema-defined market_value columns for quality checks
        required_cols = [
            col
            for col, meta in COLUMN_SCHEMA.items()
            if meta.get("role") == "market_value"
            and col in ["total_revenues_ltm", "ebitda_ltm", "eps_adj_ltm"]
        ]
        # Fallback if schema filtering returns empty (eps_adj_ltm is ratio, not market_value)
        if not required_cols:
            required_cols = ["total_revenues_ltm", "ebitda_ltm", "eps_adj_ltm"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.warning("Quality check: Missing recommended columns: %s", missing)

        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            logger.warning(
                "Quality check: %s missing values detected (expected 0 after ETL imputation)",
                missing_count,
            )

        # Log column availability status using helper
        col_availability = _validate_surprise_columns(df)
        unavailable = [m for m, avail in col_availability.items() if not avail]
        if unavailable:
            logger.info("Surprise columns unavailable for: %s", unavailable)

    df_local = df.copy()

    # Prefer analyzing the most liquid/large names when possible.
    mcap_col = _get_market_cap_column(df_local)
    if mcap_col is not None:
        df_local[mcap_col] = pd.to_numeric(df_local[mcap_col], errors="coerce")
        df_local = df_local.sort_values(by=mcap_col, ascending=False)
    df_local = df_local.head(int(top_n))

    if df_local.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for surprise analysis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    surprise_data: List[Dict[str, float]] = []
    all_surprises: List[float] = []

    # Check for Column Availability and Provide Better Error Reporting
    col_availability = _validate_surprise_columns(df_local)
    if not any(col_availability.values()) and not has_precomputed:
        fig = go.Figure()
        fig.add_annotation(
            text="Missing required Actual/Estimate columns for surprise calculation.<br>"
            "Ensure ETL Phase 8 and 9 enrichments are complete.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=True,
            font=dict(size=14),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    for metric_name, (actual_col, est_col) in _EARNINGS_SURPRISE_PAIRS.items():
        # Check if precomputed surprise column is available for this metric
        precomputed_col = precomputed_surprise_cols.get(metric_name)
        if has_precomputed and precomputed_col and precomputed_col in df_local.columns:
            # Use precomputed surprise values from ETL pipeline
            surprise_pct = pd.to_numeric(df_local[precomputed_col], errors="coerce")
            surprise_pct = surprise_pct.replace([np.inf, -np.inf], np.nan).dropna()
        else:
            # Calculate surprise from actual/estimate columns

            if actual_col not in df_local.columns or est_col not in df_local.columns:
                continue

            actual = pd.to_numeric(df_local[actual_col], errors="coerce")
            estimate = pd.to_numeric(df_local[est_col], errors="coerce")
            valid_mask = actual.notna() & estimate.notna() & (estimate.abs() > 0)

            if valid_mask.sum() == 0:
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                surprise_pct = (
                    (actual[valid_mask] - estimate[valid_mask])
                    / estimate[valid_mask].abs()
                ) * 100
            surprise_pct = surprise_pct.replace([np.inf, -np.inf], np.nan).dropna()

        if len(surprise_pct) == 0:
            continue

        surprise_data.append(
            {
                "metric": metric_name,
                "mean_surprise": float(surprise_pct.mean()),
                "median_surprise": float(surprise_pct.median()),
                "beat_pct": float((surprise_pct > 0).sum() / len(surprise_pct) * 100),
                "miss_pct": float((surprise_pct < 0).sum() / len(surprise_pct) * 100),
                "count": float(len(surprise_pct)),
            }
        )

        all_surprises.extend(surprise_pct.clip(-100, 100).tolist())

    if not surprise_data:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for earnings surprise analysis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=True,
            font=dict(size=16),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    surprise_df = pd.DataFrame(surprise_data)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Mean Surprise by Metric (%)",
            "Beat/Miss Rates (%)",
            "Surprise Distribution (All Metrics)",
            "Forecast Accuracy Score",
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "indicator"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
    )

    colors = [
        COLOR_PALETTE["success"] if x > 0 else COLOR_PALETTE["danger"]
        for x in surprise_df["mean_surprise"]
    ]
    fig.add_trace(
        go.Bar(
            x=surprise_df["metric"],
            y=surprise_df["mean_surprise"],
            marker_color=colors,
            name="Mean Surprise",
            hovertemplate="<b>%{x}</b><br>Surprise: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=1)

    fig.add_trace(
        go.Bar(
            x=surprise_df["metric"],
            y=surprise_df["beat_pct"],
            name="Beat Rate",
            marker_color=COLOR_PALETTE["success"],
            hovertemplate="<b>%{x}</b><br>Beat: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=surprise_df["metric"],
            y=surprise_df["miss_pct"],
            name="Miss Rate",
            marker_color=COLOR_PALETTE["danger"],
            hovertemplate="<b>%{x}</b><br>Miss: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )

    if all_surprises:
        fig.add_trace(
            go.Histogram(
                x=all_surprises,
                nbinsx=50,
                marker_color=COLOR_PALETTE["info"],
                name="Surprise Distribution",
                hovertemplate="Surprise: %{x:.1f}%<br>Count: %{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white", row=2, col=1)

    overall_beat_rate = float(surprise_df["beat_pct"].mean())
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=overall_beat_rate,
            title={"text": "Forecast Accuracy", "font": {"size": 16}},
            delta={"reference": 50, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLOR_PALETTE["primary"]},
                "steps": [
                    {"range": [0, 40], "color": COLOR_PALETTE["danger"]},
                    {"range": [40, 60], "color": COLOR_PALETTE["neutral"]},
                    {"range": [60, 100], "color": COLOR_PALETTE["success"]},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="<b>Earnings Surprise Analysis Dashboard</b><br><sup>Expected vs Actual Performance Monitoring</sup>",
        template=PLOTLY_TEMPLATE,
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5
        ),
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=20,
    )

    fig.update_xaxes(title_text="Metric", row=1, col=1)
    fig.update_yaxes(title_text="Surprise (%)", row=1, col=1)
    fig.update_xaxes(title_text="Metric", row=1, col=2)
    fig.update_yaxes(title_text="Rate (%)", row=1, col=2)
    fig.update_xaxes(title_text="Surprise (%)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    _write_html_artifact(fig, output_path)

    return fig


def create_earnings_calendar_analytics(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    reference_date: Optional[pd.Timestamp] = None,
    days_window: int = 30,
) -> Dict[str, object]:
    """Build interactive earnings calendar analytics with timeline and density views."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_date = resolve_reference_date(df, reference_date)

    # Schema-aligned earnings date columns (from COLUMN_SCHEMA with role="date")
    earnings_date_cols = [
        "next_earnings",
        "fy_end_date",
        "next_fy_end_date",
        "income_statement_report_date",
        "last_earnings_date",
        "next_earnings_date",
        "earnings_announcement_date",
    ]
    available_earnings_dates = [c for c in earnings_date_cols if c in df.columns]

    # Check which columns have actual non-null data
    cols_with_data = [
        c for c in available_earnings_dates if c in df.columns and df[c].notna().any()
    ]

    if not cols_with_data:
        logger.warning(
            f"No earnings date columns with data found. "
            f"Checked: {available_earnings_dates}. "
            f"Available columns: {[c for c in df.columns if 'date' in c.lower() or 'earnings' in c.lower()]}"
        )
        return _engineer_earnings_events_from_fiscal_data(
            df, output_dir, reference_date
        )

    identity_cols = [
        c
        for c in [
            "ticker",
            "name",
            "exchange",
            "sector",
            "industry",
            "region",
            "trading_country",
        ]
        if c in df.columns
    ]
    earnings_df = df[identity_cols + cols_with_data].copy()

    for col in cols_with_data:
        earnings_df[col] = pd.to_datetime(earnings_df[col], errors="coerce")

    # Use the first available column with data as anchor
    anchor_col = cols_with_data[0]
    logger.info(f"Using '{anchor_col}' as anchor for earnings analytics")

    earnings_df["days_to_earnings"] = (earnings_df[anchor_col] - reference_date).dt.days

    if days_window is not None:
        earnings_df = earnings_df[
            earnings_df["days_to_earnings"].between(-days_window, days_window)
        ]

    earnings_df = earnings_df.dropna(subset=["days_to_earnings"])

    if earnings_df.empty:
        logger.warning(
            f"No earnings events within ±{days_window} days of {reference_date}. "
            f"Consider increasing days_window or checking data quality."
        )
        # Return informative empty state instead of generic placeholder
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"No earnings events within ±{days_window} days<br>"
            f"<sub>Reference date: {reference_date.strftime('%Y-%m-%d')}<br>"
            f"Anchor column: {anchor_col}</sub>",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=True,
            font=dict(size=16),
        )
        empty_fig.update_layout(template=PLOTLY_TEMPLATE)

        _write_html_artifact(empty_fig, output_dir / "earnings_calendar.html")
        _write_html_artifact(empty_fig, output_dir / "earnings_density_heatmap.html")

        return {
            "earnings_df": pd.DataFrame(),
            "timeline_fig": empty_fig,
            "heatmap_fig": empty_fig,
            "reference_date": reference_date,
            "anchor_column": anchor_col,
        }

    timeline_fig = _create_earnings_timeline_plotly(earnings_df, reference_date)
    heatmap_fig = _create_earnings_density_heatmap(earnings_df, reference_date)

    _write_html_artifact(timeline_fig, output_dir / "earnings_calendar.html")
    _write_html_artifact(heatmap_fig, output_dir / "earnings_density_heatmap.html")

    return {
        "earnings_df": earnings_df,
        "timeline_fig": timeline_fig,
        "heatmap_fig": heatmap_fig,
        "reference_date": reference_date,
        "anchor_column": anchor_col,
    }


def _create_earnings_timeline_plotly(
    earnings_df: pd.DataFrame, reference_date: pd.Timestamp
) -> go.Figure:
    """Create interactive earnings timeline with exchange coloring."""

    valid_df = earnings_df[earnings_df["days_to_earnings"].notna()].copy()

    if valid_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid earnings dates found in data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    valid_df = valid_df[
        (valid_df["days_to_earnings"] >= -60) & (valid_df["days_to_earnings"] <= 60)
    ]

    y_dim = (
        "exchange"
        if "exchange" in valid_df.columns
        else "region" if "region" in valid_df.columns else "ticker"
    )
    color_dim = "exchange" if "exchange" in valid_df.columns else None

    fig = px.scatter(
        valid_df,
        x="days_to_earnings",
        y=y_dim,
        color=color_dim,
        hover_data=[
            c
            for c in [
                "ticker",
                "name",
                "sector",
                "exchange",
                "region",
                "fy_end_date",
                "next_fy_end_date",
                "next_earnings_when",
                "next_fiscal_quarter",
                "next_earnings_report",
                "one_day_pct",
                "last_price",
                "price_target",
                "price_chg_pct_1m",
                "price_chg_pct_3m",
                "eps_adj_1fy",
                "eps_adj_fy",
                "eps_adj_ltm",
                "net_eps_basic_ltm",
                "net_eps_basic_fq",
                "net_eps_basic_fy",
            ]
            if c in valid_df.columns
        ],
        title="<b>Earnings Events Timeline</b><br><sup>Days relative to today</sup>",
        template=PLOTLY_TEMPLATE,
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="white",
        annotation_text="Today",
    )

    fig.add_vrect(
        x0=-7,
        x1=7,
        fillcolor="yellow",
        opacity=0.1,
        annotation_text="±7 days",
    )

    fig.update_layout(height=400)

    return fig


def _create_earnings_density_heatmap(
    earnings_df: pd.DataFrame, reference_date: pd.Timestamp
) -> go.Figure:
    """Create earnings density heatmap by exchange and week."""

    if earnings_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No earnings events within selected window",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    heatmap_dim = (
        "exchange"
        if "exchange" in earnings_df.columns
        else "region" if "region" in earnings_df.columns else "ticker"
    )

    heatmap_df = earnings_df.copy()
    heatmap_df["event_date"] = reference_date + pd.to_timedelta(
        heatmap_df["days_to_earnings"], unit="D"
    )
    heatmap_df["event_week"] = (
        heatmap_df["event_date"].dt.to_period("W").apply(lambda r: r.start_time)
    )

    fig = px.density_heatmap(
        heatmap_df,
        x="event_week",
        y=heatmap_dim,
        histfunc="count",
        color_continuous_scale="Viridis",
        title="<b>Earnings Event Density</b><br><sup>Counts by sector and week</sup>",
        template=PLOTLY_TEMPLATE,
    )

    fig.update_xaxes(title_text="Week of Event")
    fig.update_yaxes(title_text=heatmap_dim.title())
    fig.update_layout(height=400)

    return fig


def _engineer_earnings_events_from_fiscal_data(
    df: pd.DataFrame, output_dir: Path, reference_date: pd.Timestamp
) -> Dict[str, object]:
    """Fallback earnings event engineering using fiscal cadence when explicit dates are missing."""

    placeholder_fig = go.Figure()
    placeholder_fig.add_annotation(
        text="No earnings date columns found; unable to engineer events",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14),
    )
    placeholder_fig.update_layout(template=PLOTLY_TEMPLATE)

    _write_html_artifact(placeholder_fig, output_dir / "earnings_calendar.html")
    _write_html_artifact(placeholder_fig, output_dir / "earnings_density_heatmap.html")

    return {
        "earnings_df": pd.DataFrame(),
        "timeline_fig": placeholder_fig,
        "heatmap_fig": placeholder_fig,
        "reference_date": reference_date,
    }


def create_gaap_adjusted_comparison_chart(
    df: pd.DataFrame, output_path: Path
) -> go.Figure:
    """Create sector-level GAAP vs Adjusted comparison visualization."""

    earnings_quality = analyze_earnings_quality(df)

    if (
        "earnings_quality_score" not in earnings_quality.columns
        or "sector" not in earnings_quality.columns
    ):
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient GAAP/Adjusted EPS data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    sector_quality = (
        earnings_quality.groupby("sector", dropna=True)
        .agg({"earnings_quality_score": ["mean", "std", "count"]})
        .round(2)
    )
    if sector_quality.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No sector-level earnings quality data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig
    sector_quality.columns = ["mean_quality", "std_quality", "count"]
    sector_quality = sector_quality.reset_index().sort_values("mean_quality")

    fig = px.bar(
        sector_quality,
        y="sector",
        x="mean_quality",
        error_x="std_quality",
        orientation="h",
        color="mean_quality",
        color_continuous_scale="RdYlGn",
        title="<b>Earnings Quality Score by Sector</b><br><sup>Higher = Less GAAP-to-Adjusted discrepancy</sup>",
        template=PLOTLY_TEMPLATE,
    )

    _write_html_artifact(fig, output_path)
    return fig


def generate_earnings_quality_alerts(
    df: pd.DataFrame,
    config: Optional[EarningsAlertConfig] = None,
    reference_date: Optional[pd.Timestamp] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """Generate rule-based earnings quality alerts from an enriched dataset.

    Designed for notebook + scheduled monitoring runs.

    Args:
        df: Enriched stocks DataFrame.
        config: Threshold configuration.
        reference_date: Reference timestamp for earnings window logic (defaults to now).
        output_path: Optional JSON path to write the alert payload.

    Returns:
        Dict payload with metadata + a list of alerts.
    """
    if config is None:
        config = EarningsAlertConfig()

    reference_date = resolve_reference_date(df, reference_date)

    alerts: List[Dict[str, object]] = []

    payload: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "reference_date": str(reference_date),
        "total_stocks_monitored": int(len(df)),
        "alerts": alerts,
        "thresholds": {
            "eps_surprise_miss_threshold_pct": config.eps_surprise_miss_threshold_pct,
            "analyst_downgrade_threshold_pct": config.analyst_downgrade_threshold_pct,
            "analyst_downgrade_min_periods": config.analyst_downgrade_min_periods,
            "target_spread_threshold_pct": config.target_spread_threshold_pct,
            "pre_earnings_window_days": config.pre_earnings_window_days,
            "pre_earnings_volatility_quantile": config.pre_earnings_volatility_quantile,
        },
    }

    ticker_col = "ticker" if "ticker" in df.columns else None

    # ---------------------------------------------------------------------
    # Alert 1: EPS surprise misses
    # ---------------------------------------------------------------------
    eps_actual_col, eps_est_col = _EARNINGS_SURPRISE_PAIRS["EPS"]
    if eps_actual_col in df.columns and eps_est_col in df.columns:
        eps_actual = pd.to_numeric(df[eps_actual_col], errors="coerce")
        eps_est = pd.to_numeric(df[eps_est_col], errors="coerce")
        valid = eps_actual.notna() & eps_est.notna() & (eps_est.abs() > 0)

        with np.errstate(divide="ignore", invalid="ignore"):
            surprise = (
                (eps_actual[valid] - eps_est[valid]) / eps_est[valid].abs()
            ) * 100
        surprise = surprise.replace([np.inf, -np.inf], np.nan).dropna()

        misses = surprise[surprise < -float(config.eps_surprise_miss_threshold_pct)]
        if len(misses) > 0:
            alerts.append(
                {
                    "alert_type": "large_earnings_miss",
                    "severity": "high",
                    "count": int(len(misses)),
                    "description": (
                        f"{len(misses)} stocks with EPS surprise < -{config.eps_surprise_miss_threshold_pct:.0f}%"
                    ),
                    "tickers": (
                        df.loc[misses.index, ticker_col]
                        .astype(str)
                        .tolist()[: int(config.max_tickers_per_alert)]
                        if ticker_col is not None
                        else []
                    ),
                }
            )

    # ---------------------------------------------------------------------
    # Alert 2: Analyst downgrade momentum (negative revisions across periods)
    # ---------------------------------------------------------------------
    default_rev_cols = [
        "eps_est_avg_rev_pct_fy1e_1w",
        "eps_est_avg_rev_pct_fy1e_1m",
        "eps_est_avg_rev_pct_fy1e_3m",
        "eps_est_avg_rev_pct_fy1e_6m",
        "eps_est_avg_rev_pct_fy1e_1y",
        "eps_gaap_est_avg_rev_pct_fy1e_1m",
        "eps_gaap_est_avg_rev_pct_fy1e_1y",
        "eps_gaap_est_avg_rev_pct_fy1e_3m",
        "eps_gaap_est_avg_rev_pct_fy1e_6m",
    ]
    available_rev_cols = [c for c in default_rev_cols if c in df.columns]
    # Generates alert for stocks with analyst downgrade momentum
    if len(available_rev_cols) >= int(config.analyst_downgrade_min_periods):
        downgrade_mask = pd.Series(True, index=df.index)
        for col in available_rev_cols:
            downgrade_mask &= pd.to_numeric(df[col], errors="coerce") < -float(
                config.analyst_downgrade_threshold_pct
            )
        downgrades = df[downgrade_mask]
        if len(downgrades) > 0:
            alerts.append(
                {
                    "alert_type": "analyst_downgrade_momentum",
                    "severity": "medium",
                    "count": int(len(downgrades)),
                    "description": (
                        f"{len(downgrades)} stocks with analyst EPS revisions < -{config.analyst_downgrade_threshold_pct:.0f}%"
                        f" across {len(available_rev_cols)} periods"
                    ),
                    "tickers": (
                        downgrades[ticker_col]
                        .astype(str)
                        .tolist()[: int(config.max_tickers_per_alert)]
                        if ticker_col is not None
                        else []
                    ),
                }
            )

    # ---------------------------------------------------------------------
    # Alert 3: High target spread uncertainty
    # ---------------------------------------------------------------------
    if (
        "price_target_high" in df.columns
        and "price_target_low" in df.columns
        and "last_price" in df.columns
    ):
        high = pd.to_numeric(df["price_target_high"], errors="coerce").astype("Float64")
        low = pd.to_numeric(df["price_target_low"], errors="coerce").astype("Float64")
        last_price = (
            pd.to_numeric(df["last_price"], errors="coerce")
            .replace(0, pd.NA)
            .astype("Float64")
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            spread_pct = ((high - low) / last_price) * 100
        spread_pct = spread_pct.replace([np.inf, -np.inf], pd.NA).dropna()

        high_spread = spread_pct[spread_pct > float(config.target_spread_threshold_pct)]
        if len(high_spread) > 0:
            alerts.append(
                {
                    "alert_type": "high_target_uncertainty",
                    "severity": "low",
                    "count": int(len(high_spread)),
                    "description": (
                        f"{len(high_spread)} stocks with price target spread > {config.target_spread_threshold_pct:.0f}%"
                    ),
                    "tickers": (
                        df.loc[high_spread.index, ticker_col]
                        .astype(str)
                        .tolist()[: int(config.max_tickers_per_alert)]
                        if ticker_col is not None
                        else []
                    ),
                }
            )

    # ---------------------------------------------------------------------
    # Alert 4: Upcoming earnings with high volatility
    # ---------------------------------------------------------------------
    if "next_earnings" in df.columns and "volatility_1m" in df.columns:
        next_earnings = pd.to_datetime(df["next_earnings"], errors="coerce")
        days_to_earnings = (next_earnings - reference_date).dt.days
        vol = pd.to_numeric(df["volatility_1m"], errors="coerce")
        vol_threshold = float(
            vol.quantile(float(config.pre_earnings_volatility_quantile))
        )

        high_vol = df[
            (days_to_earnings >= 0)
            & (days_to_earnings <= int(config.pre_earnings_window_days))
            & (vol > vol_threshold)
        ]

        if len(high_vol) > 0:
            alerts.append(
                {
                    "alert_type": "high_volatility_pre_earnings",
                    "severity": "medium",
                    "count": int(len(high_vol)),
                    "description": (
                        f"{len(high_vol)} stocks with elevated volatility (> q{config.pre_earnings_volatility_quantile:.2f})"
                        f" within {config.pre_earnings_window_days} days of earnings"
                    ),
                    "tickers": (
                        high_vol[ticker_col]
                        .astype(str)
                        .tolist()[: int(config.max_tickers_per_alert)]
                        if ticker_col is not None
                        else []
                    ),
                }
            )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    return payload


# =============================================================================
# New v1.14 Dashboard Functions
# =============================================================================


def create_analyst_sentiment_dashboard(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 50,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create comprehensive analyst sentiment dashboard using v1.15 schema features.

    Leverages the expanded Analyst Sentiment category (65+ features) including:
    - Price target momentum across multiple timeframes
    - Analyst coverage trajectory
    - EPS revision patterns
    - Consensus convergence metrics

    Args:
        df: DataFrame with analyst sentiment columns.
        reference_date: Analysis reference date.
        top_n: Number of stocks to analyze.
        output_path: Optional path to save HTML output.

    Returns:
        go.Figure: Plotly figure with analyst sentiment visualization.
    """
    reference_date = resolve_reference_date(df, reference_date)

    # Get all available Analyst Sentiment features from schema
    analyst_features = PHASE93_FEATURE_CATEGORIES.get("Analyst Sentiment", [])
    available_features = [f for f in analyst_features if f in df.columns]

    if len(available_features) < 5:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient analyst sentiment features available.<br>"
            f"Found {len(available_features)} of 65+ expected features.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    df_local = df.copy()
    mcap_col = _get_market_cap_column(df_local)
    if mcap_col:
        df_local = df_local.sort_values(by=mcap_col, ascending=False)
    df_local = df_local.head(top_n)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "PT Momentum Heatmap",
            "Coverage Trajectory",
            "EPS Revision Momentum",
            "Consensus Convergence",
            "Rating Distribution",
            "Sentiment Composite",
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "scatter"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "bar"}, {"type": "indicator"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Panel 1: PT Momentum Heatmap (timeframes as columns)
    pt_momentum_cols = [
        ("1W", "pt_momentum_1w"),
        ("1M", "pt_momentum_1m"),
        ("3M", "pt_momentum_3m"),
        ("6M", "pt_momentum_6m"),
        ("1Y", "pt_momentum_1y"),
    ]
    available_pt = [
        (label, col) for label, col in pt_momentum_cols if col in df_local.columns
    ]

    if available_pt and "sector" in df_local.columns:
        heatmap_data = []
        sectors = df_local["sector"].dropna().unique()[:10]
        for sector in sectors:
            sector_df = df_local[df_local["sector"] == sector]
            row_data = {"Sector": str(sector)[:20]}
            for label, col in available_pt:
                row_data[label] = sector_df[col].mean()
            heatmap_data.append(row_data)

        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data).set_index("Sector")
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_df.values,
                    x=[label for label, _ in available_pt],
                    y=heatmap_df.index.tolist(),
                    colorscale="RdYlGn",
                    zmid=0,
                    hovertemplate="Sector: %{y}<br>Period: %{x}<br>Momentum: %{z:.2f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # Panel 2: Coverage Trajectory
    coverage_cols = [
        "analyst_coverage_change_1m",
        "analyst_coverage_change_3m",
        "analyst_coverage_change_1y",
    ]
    available_cov = [c for c in coverage_cols if c in df_local.columns]
    if available_cov and "analyst_coverage_acceleration" in df_local.columns:
        valid_cov = df_local.dropna(
            subset=[available_cov[0], "analyst_coverage_acceleration"]
        )
        if not valid_cov.empty:
            fig.add_trace(
                go.Scatter(
                    x=valid_cov[available_cov[0]],
                    y=valid_cov["analyst_coverage_acceleration"],
                    mode="markers",
                    marker=dict(
                        color=valid_cov.get("analyst_interest_score", 50),
                        colorscale="Viridis",
                        size=8,
                        showscale=True,
                        colorbar=dict(title="Interest", x=0.65),
                    ),
                    text=valid_cov.get("ticker", valid_cov.index),
                    hovertemplate="<b>%{text}</b><br>Coverage Chg: %{x:.1f}<br>Accel: %{y:.2f}<extra></extra>",
                    name="Coverage",
                ),
                row=1,
                col=2,
            )

    # Panel 3: EPS Revision Momentum
    if "eps_revision_momentum" in df_local.columns:
        revision_means = {}
        if "sector" in df_local.columns:
            revision_means = (
                df_local.groupby("sector")["eps_revision_momentum"].mean().sort_values()
            )
        if revision_means is not None and len(revision_means) > 0:
            colors = [
                COLOR_PALETTE["success"] if v > 0 else COLOR_PALETTE["danger"]
                for v in revision_means.values
            ]
            fig.add_trace(
                go.Bar(
                    x=revision_means.values,
                    y=revision_means.index,
                    orientation="h",
                    marker_color=colors,
                    name="EPS Rev",
                    hovertemplate="<b>%{y}</b><br>Revision Mom: %{x:.2f}<extra></extra>",
                ),
                row=1,
                col=3,
            )

    # Panel 4: Consensus Convergence Distribution
    if "pt_consensus_convergence" in df_local.columns:
        convergence = df_local["pt_consensus_convergence"].dropna()
        if len(convergence) > 0:
            fig.add_trace(
                go.Histogram(
                    x=convergence,
                    nbinsx=25,
                    marker_color=COLOR_PALETTE["info"],
                    name="Convergence",
                ),
                row=2,
                col=1,
            )

    # Panel 5: Rating Distribution
    if "analyst_rating_normalized" in df_local.columns:
        rating_bins = pd.cut(
            df_local["analyst_rating_normalized"].dropna(),
            bins=[0, 20, 40, 60, 80, 100],
            labels=["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"],
        )
        rating_counts = rating_bins.value_counts().sort_index()
        if not rating_counts.empty:
            fig.add_trace(
                go.Bar(
                    x=rating_counts.index.astype(str),
                    y=rating_counts.values,
                    marker_color=[
                        COLOR_PALETTE["danger"],
                        COLOR_PALETTE["warning"],
                        COLOR_PALETTE["neutral"],
                        COLOR_PALETTE["info"],
                        COLOR_PALETTE["success"],
                    ],
                    name="Ratings",
                ),
                row=2,
                col=2,
            )

    # Panel 6: Composite Sentiment Gauge
    sentiment_score = 50.0  # Default neutral
    if "analyst_rating_normalized" in df_local.columns:
        sentiment_score = df_local["analyst_rating_normalized"].mean()
    elif "pt_momentum_1m" in df_local.columns:
        # Derive from momentum if rating not available
        sentiment_score = 50 + df_local["pt_momentum_1m"].clip(-50, 50).mean()

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            title={"text": "Sentiment Score", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLOR_PALETTE["primary"]},
                "steps": [
                    {"range": [0, 30], "color": COLOR_PALETTE["danger"]},
                    {"range": [30, 70], "color": COLOR_PALETTE["neutral"]},
                    {"range": [70, 100], "color": COLOR_PALETTE["success"]},
                ],
            },
        ),
        row=2,
        col=3,
    )

    fig.update_layout(
        title="<b>Analyst Sentiment Dashboard</b><br><sup>Phase 9.3 v1.15 - 65+ Sentiment Features</sup>",
        template=PLOTLY_TEMPLATE,
        height=700,
        showlegend=False,
    )

    _write_html_artifact(fig, output_path)
    return fig


def create_price_target_dynamics_dashboard(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 50,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create interactive dashboard for price target momentum and dynamics.

    Visualizes analyst price target revisions, momentum across timeframes,
    and consensus convergence patterns.

    Args:
        df: DataFrame with price target columns (pt_momentum_*, price_target_*).
        reference_date: Analysis reference date (defaults to today).
        top_n: Number of stocks to analyze.
        output_path: Optional path to save HTML output.

    Returns:
        go.Figure: Plotly figure with price target dynamics visualization.
    """
    reference_date = resolve_reference_date(df, reference_date)

    # Price target momentum columns from schema
    pt_momentum_cols = [
        "pt_momentum_1w",
        "pt_momentum_1m",
        "pt_momentum_3m",
        "pt_momentum_6m",
        "pt_momentum_1y",
    ]
    available_momentum = [c for c in pt_momentum_cols if c in df.columns]

    # Additional PT analytics columns
    pt_analytics_cols = [
        "pt_acceleration_short",
        "pt_acceleration_long",
        "pt_consensus_convergence",
        "pt_vs_price_momentum",
        "pt_skew_trend",
        "pt_high_low_spread_trend",
    ]
    available_analytics = [c for c in pt_analytics_cols if c in df.columns]

    if not available_momentum and not available_analytics:
        fig = go.Figure()
        fig.add_annotation(
            text="No price target dynamics columns available.<br>"
            "Required: pt_momentum_*, pt_acceleration_*, pt_consensus_*",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    # Sort by market cap if available
    df_local = df.copy()
    mcap_col = _get_market_cap_column(df_local)
    if mcap_col:
        df_local = df_local.sort_values(by=mcap_col, ascending=False)
    df_local = df_local.head(top_n)

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Price Target Momentum by Timeframe",
            "PT Acceleration (Short vs Long)",
            "Consensus Convergence Distribution",
            "PT vs Price Momentum Scatter",
        ],
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Plot 1: Momentum by timeframe (box plot style as bars showing mean)
    if available_momentum:
        momentum_means = {
            col.replace("pt_momentum_", ""): df_local[col].mean()
            for col in available_momentum
            if df_local[col].notna().any()
        }
        if momentum_means:
            colors = [
                COLOR_PALETTE["success"] if v > 0 else COLOR_PALETTE["danger"]
                for v in momentum_means.values()
            ]
            fig.add_trace(
                go.Bar(
                    x=list(momentum_means.keys()),
                    y=list(momentum_means.values()),
                    marker_color=colors,
                    name="Avg PT Momentum",
                    hovertemplate="<b>%{x}</b><br>Avg Momentum: %{y:.2f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # Plot 2: Acceleration scatter
    if (
        "pt_acceleration_short" in df_local.columns
        and "pt_acceleration_long" in df_local.columns
    ):
        valid_accel = df_local.dropna(
            subset=["pt_acceleration_short", "pt_acceleration_long"]
        )
        if not valid_accel.empty:
            fig.add_trace(
                go.Scatter(
                    x=valid_accel["pt_acceleration_short"],
                    y=valid_accel["pt_acceleration_long"],
                    mode="markers",
                    marker=dict(
                        color=valid_accel.get("pt_momentum_1m", 0),
                        colorscale="RdYlGn",
                        size=8,
                        showscale=True,
                        colorbar=dict(title="1M Mom", x=0.45),
                    ),
                    text=valid_accel.get("ticker", valid_accel.index),
                    hovertemplate="<b>%{text}</b><br>Short Accel: %{x:.2f}<br>Long Accel: %{y:.2f}<extra></extra>",
                    name="Acceleration",
                ),
                row=1,
                col=2,
            )

    # Plot 3: Consensus convergence histogram
    if "pt_consensus_convergence" in df_local.columns:
        convergence = df_local["pt_consensus_convergence"].dropna()
        if len(convergence) > 0:
            fig.add_trace(
                go.Histogram(
                    x=convergence,
                    nbinsx=30,
                    marker_color=COLOR_PALETTE["info"],
                    name="Convergence",
                    hovertemplate="Convergence: %{x:.2f}<br>Count: %{y}<extra></extra>",
                ),
                row=2,
                col=1,
            )

    # Plot 4: PT vs Price momentum
    if (
        "pt_vs_price_momentum" in df_local.columns
        and "price_momentum_1m" in df_local.columns
    ):
        valid_vs = df_local.dropna(subset=["pt_vs_price_momentum", "price_momentum_1m"])
        if not valid_vs.empty:
            fig.add_trace(
                go.Scatter(
                    x=valid_vs["price_momentum_1m"],
                    y=valid_vs["pt_vs_price_momentum"],
                    mode="markers",
                    marker=dict(color=COLOR_PALETTE["primary"], size=6),
                    text=valid_vs.get("ticker", valid_vs.index),
                    hovertemplate="<b>%{text}</b><br>Price Mom: %{x:.1f}%<br>PT vs Price: %{y:.2f}<extra></extra>",
                    name="PT vs Price",
                ),
                row=2,
                col=2,
            )
            # Add diagonal reference line
            fig.add_trace(
                go.Scatter(
                    x=[-50, 50],
                    y=[-50, 50],
                    mode="lines",
                    line=dict(dash="dash", color="white", width=1),
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

    fig.update_layout(
        title="<b>Price Target Dynamics Dashboard</b><br><sup>Analyst Revision Momentum & Consensus Patterns</sup>",
        template=PLOTLY_TEMPLATE,
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5
        ),
    )

    fig.update_xaxes(title_text="Timeframe", row=1, col=1)
    fig.update_yaxes(title_text="Avg Momentum (%)", row=1, col=1)
    fig.update_xaxes(title_text="Short-term Acceleration", row=1, col=2)
    fig.update_yaxes(title_text="Long-term Acceleration", row=1, col=2)
    fig.update_xaxes(title_text="Convergence Score", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Price Momentum (%)", row=2, col=2)
    fig.update_yaxes(title_text="PT vs Price Momentum", row=2, col=2)

    _write_html_artifact(fig, output_path)
    return fig


def create_eps_trajectory_dashboard(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 50,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create interactive dashboard for EPS trajectory and trend analysis.

    Visualizes EPS growth patterns, quarterly trends, positive streaks,
    and year-over-year comparisons.

    Args:
        df: DataFrame with EPS trajectory columns.
        reference_date: Analysis reference date.
        top_n: Number of stocks to analyze.
        output_path: Optional path to save HTML output.

    Returns:
        go.Figure: Plotly figure with EPS trajectory visualization.
    """
    reference_date = resolve_reference_date(df, reference_date)

    # EPS trajectory columns from CategoryMetricsResolver._SUPPLEMENTAL_EARNINGS_QUALITY
    eps_trajectory_cols = [
        "eps_quarterly_trend",
        "eps_quarterly_volatility",
        "eps_yoy_quarterly_growth",
        "eps_qoq_growth",
        "eps_positive_streak",
        "eps_cagr_5y",
        "eps_cagr_3y",
        "eps_annual_trend",
        "eps_vs_5y_avg",
        "eps_growth_acceleration",
    ]
    available_eps = [c for c in eps_trajectory_cols if c in df.columns]

    if not available_eps:
        fig = go.Figure()
        fig.add_annotation(
            text="No EPS trajectory columns available.<br>"
            "Required: eps_quarterly_trend, eps_cagr_*, eps_positive_streak",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    df_local = df.copy()
    mcap_col = _get_market_cap_column(df_local)
    if mcap_col:
        df_local = df_local.sort_values(by=mcap_col, ascending=False)
    df_local = df_local.head(top_n)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "EPS CAGR Distribution (3Y vs 5Y)",
            "EPS Positive Streak by Sector",
            "Quarterly Trend vs Volatility",
            "EPS Growth Acceleration",
        ],
        specs=[
            [{"type": "box"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "histogram"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Plot 1: EPS CAGR box plots
    cagr_cols = [c for c in ["eps_cagr_3y", "eps_cagr_5y"] if c in df_local.columns]
    for i, col in enumerate(cagr_cols):
        valid_data = df_local[col].dropna()
        if len(valid_data) > 0:
            # Clip extreme outliers for visualization
            clipped = valid_data.clip(-100, 200)
            fig.add_trace(
                go.Box(
                    y=clipped,
                    name=col.replace("eps_cagr_", "").upper(),
                    marker_color=(
                        COLOR_PALETTE["success"]
                        if "3y" in col
                        else COLOR_PALETTE["info"]
                    ),
                    boxpoints="outliers",
                ),
                row=1,
                col=1,
            )

    # Plot 2: EPS positive streak by sector
    if "eps_positive_streak" in df_local.columns and "sector" in df_local.columns:
        sector_streak = (
            df_local.groupby("sector")["eps_positive_streak"]
            .mean()
            .sort_values(ascending=True)
        )
        if not sector_streak.empty:
            fig.add_trace(
                go.Bar(
                    x=sector_streak.values,
                    y=sector_streak.index,
                    orientation="h",
                    marker_color=COLOR_PALETTE["success"],
                    name="Avg Streak",
                    hovertemplate="<b>%{y}</b><br>Avg Streak: %{x:.1f} quarters<extra></extra>",
                ),
                row=1,
                col=2,
            )

    # Plot 3: Quarterly trend vs volatility scatter
    if (
        "eps_quarterly_trend" in df_local.columns
        and "eps_quarterly_volatility" in df_local.columns
    ):
        valid_scatter = df_local.dropna(
            subset=["eps_quarterly_trend", "eps_quarterly_volatility"]
        )
        if not valid_scatter.empty:
            fig.add_trace(
                go.Scatter(
                    x=valid_scatter["eps_quarterly_volatility"],
                    y=valid_scatter["eps_quarterly_trend"],
                    mode="markers",
                    marker=dict(
                        color=valid_scatter.get("eps_cagr_3y", 0),
                        colorscale="RdYlGn",
                        size=8,
                        showscale=True,
                        colorbar=dict(title="3Y CAGR", x=1.02),
                    ),
                    text=valid_scatter.get("ticker", valid_scatter.index),
                    hovertemplate="<b>%{text}</b><br>Volatility: %{x:.2f}<br>Trend: %{y:.2f}<extra></extra>",
                    name="Trend vs Vol",
                ),
                row=2,
                col=1,
            )

    # Plot 4: Growth acceleration histogram
    if "eps_growth_acceleration" in df_local.columns:
        accel = df_local["eps_growth_acceleration"].dropna().clip(-100, 100)
        if len(accel) > 0:
            fig.add_trace(
                go.Histogram(
                    x=accel,
                    nbinsx=40,
                    marker_color=COLOR_PALETTE["warning"],
                    name="Acceleration",
                    hovertemplate="Acceleration: %{x:.1f}<br>Count: %{y}<extra></extra>",
                ),
                row=2,
                col=2,
            )
            fig.add_vline(x=0, line_dash="dash", line_color="white", row=2, col=2)

    fig.update_layout(
        title="<b>EPS Trajectory Dashboard</b><br><sup>Earnings Growth Patterns & Momentum Analysis</sup>",
        template=PLOTLY_TEMPLATE,
        height=700,
        showlegend=False,
    )

    fig.update_yaxes(title_text="CAGR (%)", row=1, col=1)
    fig.update_xaxes(title_text="Avg Positive Quarters", row=1, col=2)
    fig.update_xaxes(title_text="Quarterly Volatility", row=2, col=1)
    fig.update_yaxes(title_text="Quarterly Trend", row=2, col=1)
    fig.update_xaxes(title_text="Growth Acceleration", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    _write_html_artifact(fig, output_path)
    return fig


def create_cashflow_temporal_dashboard(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 50,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create interactive dashboard for cash flow temporal patterns.

    Visualizes CFO/FCF trends, stability scores, cash conversion efficiency,
    and temporal cash flow dynamics.

    Args:
        df: DataFrame with cash flow columns.
        reference_date: Analysis reference date.
        top_n: Number of stocks to analyze.
        output_path: Optional path to save HTML output.

    Returns:
        go.Figure: Plotly figure with cash flow temporal visualization.
    """
    reference_date = resolve_reference_date(df, reference_date)

    # Cash flow columns from CategoryMetricsResolver._SUPPLEMENTAL_CASH_FLOW
    cashflow_cols = [
        "fcf_to_cfo_ratio",
        "cfo_stability_score",
        "fcf_stability_score",
        "capex_to_revenue_trend",
        "fcf_yield_momentum",
        "cash_conversion_efficiency",
        "operating_cash_flow_growth",
        "fcf_cagr_3y",
        "acquisition_intensity",
        "dividend_cash_flow_coverage",
        "cash_burn_rate",
        "net_cash_flow_trend",
        "cfo_5y_trend",
        "fcf_quarterly_trend",
        "cfo_quarterly_trend",
    ]
    available_cf = [c for c in cashflow_cols if c in df.columns]

    if not available_cf:
        fig = go.Figure()
        fig.add_annotation(
            text="No cash flow temporal columns available.<br>"
            "Required: cfo_stability_score, fcf_*, cash_conversion_*",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    df_local = df.copy()
    mcap_col = _get_market_cap_column(df_local)
    if mcap_col:
        df_local = df_local.sort_values(by=mcap_col, ascending=False)
    df_local = df_local.head(top_n)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "CFO vs FCF Stability Scores",
            "Cash Conversion Efficiency by Sector",
            "FCF Yield Momentum Distribution",
            "Cash Flow Trend Indicators",
        ],
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "bar"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Plot 1: CFO vs FCF stability scatter
    if (
        "cfo_stability_score" in df_local.columns
        and "fcf_stability_score" in df_local.columns
    ):
        valid = df_local.dropna(subset=["cfo_stability_score", "fcf_stability_score"])
        if not valid.empty:
            fig.add_trace(
                go.Scatter(
                    x=valid["cfo_stability_score"],
                    y=valid["fcf_stability_score"],
                    mode="markers",
                    marker=dict(
                        color=valid.get("fcf_cagr_3y", 0),
                        colorscale="RdYlGn",
                        size=8,
                        showscale=True,
                        colorbar=dict(title="FCF CAGR", x=0.45),
                    ),
                    text=valid.get("ticker", valid.index),
                    hovertemplate="<b>%{text}</b><br>CFO Stability: %{x:.2f}<br>FCF Stability: %{y:.2f}<extra></extra>",
                    name="Stability",
                ),
                row=1,
                col=1,
            )

    # Plot 2: Cash conversion by sector
    if (
        "cash_conversion_efficiency" in df_local.columns
        and "sector" in df_local.columns
    ):
        sector_conv = (
            df_local.groupby("sector")["cash_conversion_efficiency"]
            .mean()
            .sort_values(ascending=True)
        )
        if not sector_conv.empty:
            colors = [
                COLOR_PALETTE["success"] if v > 0.5 else COLOR_PALETTE["warning"]
                for v in sector_conv.values
            ]
            fig.add_trace(
                go.Bar(
                    x=sector_conv.values,
                    y=sector_conv.index,
                    orientation="h",
                    marker_color=colors,
                    name="Avg Conversion",
                    hovertemplate="<b>%{y}</b><br>Conversion: %{x:.2f}<extra></extra>",
                ),
                row=1,
                col=2,
            )

    # Plot 3: FCF yield momentum histogram
    if "fcf_yield_momentum" in df_local.columns:
        momentum = df_local["fcf_yield_momentum"].dropna().clip(-50, 50)
        if len(momentum) > 0:
            fig.add_trace(
                go.Histogram(
                    x=momentum,
                    nbinsx=30,
                    marker_color=COLOR_PALETTE["info"],
                    name="FCF Yield Mom",
                    hovertemplate="Momentum: %{x:.1f}<br>Count: %{y}<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig.add_vline(x=0, line_dash="dash", line_color="white", row=2, col=1)

    # Plot 4: Trend indicators summary
    trend_cols = {
        "CFO 5Y": "cfo_5y_trend",
        "FCF Qtr": "fcf_quarterly_trend",
        "CFO Qtr": "cfo_quarterly_trend",
        "Net CF": "net_cash_flow_trend",
    }
    trend_means = {}
    for label, col in trend_cols.items():
        if col in df_local.columns:
            mean_val = df_local[col].mean()
            if pd.notna(mean_val):
                trend_means[label] = mean_val

    if trend_means:
        colors = [
            COLOR_PALETTE["success"] if v > 0 else COLOR_PALETTE["danger"]
            for v in trend_means.values()
        ]
        fig.add_trace(
            go.Bar(
                x=list(trend_means.keys()),
                y=list(trend_means.values()),
                marker_color=colors,
                name="Avg Trend",
                hovertemplate="<b>%{x}</b><br>Avg Trend: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white", row=2, col=2)

    fig.update_layout(
        title="<b>Cash Flow Temporal Dashboard</b><br><sup>CFO/FCF Stability & Trend Patterns</sup>",
        template=PLOTLY_TEMPLATE,
        height=700,
        showlegend=False,
    )

    fig.update_xaxes(title_text="CFO Stability Score", row=1, col=1)
    fig.update_yaxes(title_text="FCF Stability Score", row=1, col=1)
    fig.update_xaxes(title_text="Avg Conversion Efficiency", row=1, col=2)
    fig.update_xaxes(title_text="FCF Yield Momentum", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Trend Indicator", row=2, col=2)
    fig.update_yaxes(title_text="Avg Trend Score", row=2, col=2)

    _write_html_artifact(fig, output_path)
    return fig


def create_fiscal_calendar_dashboard(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 100,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create interactive dashboard for fiscal calendar and reporting patterns.

    Visualizes fiscal year progress, reporting lag patterns, quarter-end timing,
    and earnings announcement clustering.

    Args:
        df: DataFrame with fiscal calendar columns.
        reference_date: Analysis reference date.
        top_n: Number of stocks to analyze.
        output_path: Optional path to save HTML output.

    Returns:
        go.Figure: Plotly figure with fiscal calendar visualization.
    """
    reference_date = resolve_reference_date(df, reference_date)

    # Temporal columns from CategoryMetricsResolver._SUPPLEMENTAL_TEMPORAL
    temporal_cols = [
        "fiscal_year_progress",
        "days_to_quarter_end",
        "fiscal_half",
        "reporting_lag_zscore",
        "late_reporter_flag",
        "days_since_fy_end",
        "days_to_next_fy_end",
        "earnings_imminent",
        "pre_earnings_window",
        "fiscal_quarter",
        "current_fiscal_quarter",
        "next_fiscal_quarter",
    ]
    available_temporal = [c for c in temporal_cols if c in df.columns]

    if not available_temporal:
        fig = go.Figure()
        fig.add_annotation(
            text="No fiscal calendar columns available.<br>"
            "Required: fiscal_year_progress, days_to_quarter_end, fiscal_quarter",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    df_local = df.copy()
    mcap_col = _get_market_cap_column(df_local)
    if mcap_col:
        df_local = df_local.sort_values(by=mcap_col, ascending=False)
    df_local = df_local.head(top_n)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Fiscal Year Progress Distribution",
            "Reporting Lag by Sector",
            "Days to Quarter End",
            "Fiscal Quarter Distribution",
        ],
        specs=[
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "pie"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Plot 1: Fiscal year progress histogram
    if "fiscal_year_progress" in df_local.columns:
        progress = df_local["fiscal_year_progress"].dropna()
        if len(progress) > 0:
            fig.add_trace(
                go.Histogram(
                    x=progress,
                    nbinsx=20,
                    marker_color=COLOR_PALETTE["primary"],
                    name="FY Progress",
                    hovertemplate="Progress: %{x:.1%}<br>Count: %{y}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # Plot 2: Reporting lag by sector
    if "reporting_lag_zscore" in df_local.columns and "sector" in df_local.columns:
        sector_lag = (
            df_local.groupby("sector")["reporting_lag_zscore"]
            .mean()
            .sort_values(ascending=True)
        )
        if not sector_lag.empty:
            colors = [
                COLOR_PALETTE["danger"] if v > 1 else COLOR_PALETTE["success"]
                for v in sector_lag.values
            ]
            fig.add_trace(
                go.Bar(
                    x=sector_lag.values,
                    y=sector_lag.index,
                    orientation="h",
                    marker_color=colors,
                    name="Avg Lag Z-Score",
                    hovertemplate="<b>%{y}</b><br>Lag Z-Score: %{x:.2f}<extra></extra>",
                ),
                row=1,
                col=2,
            )

    # Plot 3: Days to quarter end
    if "days_to_quarter_end" in df_local.columns:
        days_qe = df_local["days_to_quarter_end"].dropna()
        if len(days_qe) > 0:
            fig.add_trace(
                go.Histogram(
                    x=days_qe,
                    nbinsx=30,
                    marker_color=COLOR_PALETTE["warning"],
                    name="Days to QE",
                    hovertemplate="Days: %{x}<br>Count: %{y}<extra></extra>",
                ),
                row=2,
                col=1,
            )

    # Plot 4: Fiscal quarter pie chart
    quarter_col = next(
        (
            c
            for c in ["fiscal_quarter", "current_fiscal_quarter"]
            if c in df_local.columns
        ),
        None,
    )
    if quarter_col:
        quarter_counts = df_local[quarter_col].value_counts()
        if not quarter_counts.empty:
            fig.add_trace(
                go.Pie(
                    labels=quarter_counts.index.astype(str),
                    values=quarter_counts.values,
                    marker=dict(
                        colors=[
                            COLOR_PALETTE["primary"],
                            COLOR_PALETTE["success"],
                            COLOR_PALETTE["warning"],
                            COLOR_PALETTE["info"],
                        ]
                    ),
                    name="Quarter Dist",
                    hovertemplate="<b>Q%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
                ),
                row=2,
                col=2,
            )

    fig.update_layout(
        title="<b>Fiscal Calendar Dashboard</b><br><sup>Reporting Patterns & Fiscal Year Timing</sup>",
        template=PLOTLY_TEMPLATE,
        height=700,
        showlegend=False,
    )

    fig.update_xaxes(title_text="Fiscal Year Progress (%)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Reporting Lag Z-Score", row=1, col=2)
    fig.update_xaxes(title_text="Days to Quarter End", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    _write_html_artifact(fig, output_path)
    return fig
