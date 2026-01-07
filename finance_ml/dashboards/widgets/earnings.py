"""earnings.py - Dashboard widgets."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from finance_ml.core.constants import PLOTLY_TEMPLATE, COLOR_PALETTE
from finance_ml.core.schema import COLUMN_SCHEMA, PHASE93_FEATURE_CATEGORIES
from .base import (
    EarningsMode,
    resolve_reference_date,
    add_formatted_date_columns,
    _write_html_artifact,
    _build_format_dict,
    _ensure_schema_dtypes,
    EarningsAlertConfig,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Market Cap Configuration
# =============================================================================
# Columns with role="market_value" in COLUMN_SCHEMA, ordered by preference
_MARKET_CAP_COLUMNS: List[str] = [
    "market_cap",
    "market_cap_usd",
    "market_cap_country_r",
]


# =============================================================================
# Earnings Surprise Configuration
# =============================================================================
# Column pairs: (actual_column, estimate_column)
_EARNINGS_SURPRISE_PAIRS: Dict[str, tuple[str, str]] = {
    "Revenue": ("total_revenues_ltm", "revenues_est_avg_ntm"),
    "EBITDA": ("ebitda_ltm", "ebitda_est_avg_fy1e"),
    "EBIT": ("ebit_ltm", "ebit_est_med_ntm"),
    "Net Income": ("net_income_is_ltm", "net_income_adj_1fy"),
    "EPS": ("eps_adj_ltm", "eps_norm_est_avg_ntm"),
}


# =============================================================================
# Category Metrics Resolution
# =============================================================================
class CategoryMetricsResolver:
    """Resolves category names to metrics from PHASE93_FEATURE_CATEGORIES.

    Handles short name aliases and supplemental domain-specific metrics.
    """

    # Short name aliases mapping to PHASE93_FEATURE_CATEGORIES keys
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
    }

    # Supplemental metrics by domain
    _SUPPLEMENTAL_EARNINGS: List[str] = [
        "net_income_adj_1fy",
        "ebitda_adj_fy",
        "ebitda_adj_1fy",
        "ebit_adj_1fy",
        "ebit_adj_fy",
        "net_income_adj_fy",
        "net_income_adj_fq",
        "net_income_adj_5yavgfq",
        "eps_adj_1fy",
        "eps_adj_fy",
        "eps_adj_ltm",
    ]

    _SUPPLEMENTAL_DIVIDENDS: List[str] = [
        "dividend_record_announce_date",
        "dividend_record_ex_date",
        "dividend_record_payable_date",
        "dividend_record_record_date",
        "dividend_record_frequency",
        "dividend_record_currency",
        "div_yield_ltm",
        "div_yield_ntm",
        "div_yield_ind",
        "div_yield_1fyind",
        "div_yield_5yavgltm",
        "dividend_per_share",
        "common_dividends_paid_fy",
        "dividends_paid",
        "dividends_paid_ltm",
    ]

    _SUPPLEMENTAL_EARNINGS_QUALITY: List[str] = [
        "eps_surprise_pct",
        "earnings_beat_indicator",
        "eps_surprise_magnitude",
        "revenue_surprise_pct",
        "revenue_beat_indicator",
        "ebitda_surprise_pct",
        "surprise_momentum_score",
        "positive_revision_momentum",
        "consensus_uncertainty_score",
        "estimate_revision_acceleration",
        "accelerating_upgrades_flag",
        "eps_adjustment_ratio_ltm",
        "eps_adjustment_pct_ltm",
        "eps_quality_flag_ltm",
        "net_income_adjustment_ratio_ltm",
        "ebitda_adjustment_pct_ltm",
        "adjustment_consistency_score",
        "earnings_quality_warning_flag",
        "earnings_quality_score",
        "exceptional_items_impact_ratio",
    ]

    # Categories that receive earnings supplemental metrics
    _EARNINGS_CATEGORIES: Set[str] = {"Profitability", "Growth Metrics"}

    # Categories for dividend supplemental metrics (first match wins)
    _DIVIDEND_CATEGORIES: List[str] = ["Dividend Reliability", "Capital Allocation"]

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

        **ETL Pipeline Note:** Some metrics may have companion `*_applicable` flags
        emitted by the ETL pipeline (Stage 8g conditional metrics handling).

        Args:
            categories: List of category names or short aliases.
            include_supplemental: Whether to include domain-specific metrics.

        Returns:
            Dict mapping category name to list of metric column names.
        """
        result: Dict[str, List[str]] = {}

        for cat in categories:
            full_cat = cls.resolve_name(cat)
            metrics = PHASE93_FEATURE_CATEGORIES.get(full_cat, []).copy()
            result[full_cat] = metrics

        if include_supplemental:
            cls._add_supplemental_metrics(result)

        return result

    @classmethod
    def _add_supplemental_metrics(cls, result: Dict[str, List[str]]) -> None:
        """Add supplemental domain-specific metrics to category results in-place."""
        result_keys = set(result.keys())

        # Add earnings metrics to applicable categories
        for category in cls._EARNINGS_CATEGORIES & result_keys:
            result[category].extend(cls._SUPPLEMENTAL_EARNINGS)

        # Add dividend metrics to first matching category
        for category in cls._DIVIDEND_CATEGORIES:
            if category in result_keys:
                result[category].extend(cls._SUPPLEMENTAL_DIVIDENDS)
                break

        # Add earnings quality metrics
        if "Earnings Quality" in result_keys:
            result["Earnings Quality"].extend(cls._SUPPLEMENTAL_EARNINGS_QUALITY)


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


# =============================================================================
# Market Cap and Validation Utilities
# =============================================================================
def _get_market_cap_column(df: pd.DataFrame) -> Optional[str]:
    """Find the best available market cap column using schema metadata.

    Prefers columns in order of liquidity/standardization.
    All columns have role="market_value" in COLUMN_SCHEMA.
    """
    for col in _MARKET_CAP_COLUMNS:
        if col in df.columns:
            return col
    return None


def _validate_surprise_columns(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate availability of earnings surprise columns against schema.

    Checks for actual/estimate column pairs needed for surprise calculations.

    Returns:
        Dict mapping metric name to boolean indicating pair availability.
    """
    return {
        metric: (actual in df.columns and estimate in df.columns)
        for metric, (actual, estimate) in _EARNINGS_SURPRISE_PAIRS.items()
    }


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

    # Ensure date columns are datetime
    date_cols = [
        "next_earnings",
        "dividend_record_ex_date",
        "dividend_record_payable_date",
        "dividend_record_announce_date",
        "dividend_record_record_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Filter logic: next_earnings within +/- N days if days_window provided
    filtered_df = df.copy()
    if days_window is None:
        temporal_window = None
    else:
        temporal_window = timedelta(days=days_window)

    if "next_earnings" in df.columns and temporal_window is not None:
        mask = (filtered_df["next_earnings"] - reference_date).abs() <= temporal_window
        filtered_df = filtered_df[mask]

    if filtered_df.empty:
        return pd.DataFrame()

    # Sort by Market Cap using schema-aware helper
    mcap_col = _get_market_cap_column(df)

    if mcap_col:
        filtered_df = filtered_df.sort_values(by=mcap_col, ascending=False)

    filtered_df = filtered_df.head(top_n)

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
        selected_categories = ["Profitability", "Growth Metrics", "Momentum & Technical"]

    # Get metrics from selected categories
    category_metrics = get_category_metrics(selected_categories, include_supplemental=True)

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

    # Add computed columns - Use reference_date for temporal consistency
    if "next_earnings" in dashboard_df.columns:
        dashboard_df["days_to_earnings"] = (
            pd.to_datetime(dashboard_df["next_earnings"], errors="coerce") - reference_date
        ).dt.days

        # Reorder: Put days_to_earnings near next_earnings
        cols = list(dashboard_df.columns)
        if "days_to_earnings" in cols:
            cols.remove("days_to_earnings")
            if "next_earnings" in cols:
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
            "fy_end_date" "next_fy_end_date",
            "fy_end",
            "_reference_date",
            "reference_date",
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

    # Apply additional styling (background gradients for key metrics)
    numeric_cols = df_styled.select_dtypes(include=[np.number]).columns.tolist()
    gradient_cols = [c for c in ["earnings_quality_score", "roe", "roa"] if c in numeric_cols]
    if gradient_cols:
        styler = styler.background_gradient(
            subset=gradient_cols,
            cmap="RdYlGn",
            vmin=-50,
            vmax=50,
        )

    # Add caption with mode info
    mode_display = mode.replace("_", " ").title()
    styler = styler.set_caption(
        f"Earnings Calendar Dashboard - Mode: {mode_display} "
        f"(Top {len(df_styled)} by Market Cap)"
    )

    # Save to HTML if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        styler.to_html(output_path)
        logger.info("Saved earnings dashboard to %s", output_path)

    return styler


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
        mode=metric_category if metric_category in PHASE93_FEATURE_CATEGORIES else "earnings",
        categories=[metric_category] if metric_category in PHASE93_FEATURE_CATEGORIES else None,
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
            c for c in numeric_cols if c not in ["days_to_earnings"] and "date" not in c.lower()
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
            COLOR_PALETTE["success"] if v >= 0 else COLOR_PALETTE["danger"] for v in plot_df[metric]
        ]

        fig.add_trace(
            go.Bar(
                x=plot_df[metric],
                y=plot_df["ticker"] if "ticker" in plot_df.columns else plot_df.index,
                orientation="h",
                marker_color=colors,
                name=metric.replace("_", " ").title(),
                hovertemplate=(
                    "<b>%{y}</b><br>" + f"{metric}: " + "%{x:.2f}<br>" + "<extra></extra>"
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
            showarrow=False,
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
                    (actual[valid_mask] - estimate[valid_mask]) / estimate[valid_mask].abs()
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
            showarrow=False,
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
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
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

    earnings_date_cols = [
        "next_earnings",
        "last_earnings_date",
        "income_statement_report_date",
        "next_earnings_date",
        "earnings_announcement_date",
        "fy_end_date",
        "next_fy_end_date",
    ]
    available_earnings_dates = [c for c in earnings_date_cols if c in df.columns]

    if not available_earnings_dates:
        logger.warning("No earnings date columns found. Falling back to engineered events.")
        return _engineer_earnings_events_from_fiscal_data(df, output_dir, reference_date)

    identity_cols = [
        c
        for c in ["ticker", "name", "exchange", "sector", "industry", "region", "trading_country"]
        if c in df.columns
    ]
    earnings_df = df[identity_cols + available_earnings_dates].copy()

    for col in available_earnings_dates:
        earnings_df[col] = pd.to_datetime(earnings_df[col], errors="coerce")

    anchor_col = (
        "next_earnings" if "next_earnings" in earnings_df.columns else available_earnings_dates[0]
    )
    earnings_df["days_to_earnings"] = (earnings_df[anchor_col] - reference_date).dt.days

    if days_window is not None:
        earnings_df = earnings_df[
            earnings_df["days_to_earnings"].between(-days_window, days_window)
        ]

    earnings_df = earnings_df.dropna(subset=["days_to_earnings"])

    timeline_fig = _create_earnings_timeline_plotly(earnings_df, reference_date)
    heatmap_fig = _create_earnings_density_heatmap(earnings_df, reference_date)

    _write_html_artifact(timeline_fig, output_dir / "earnings_calendar.html")
    _write_html_artifact(heatmap_fig, output_dir / "earnings_density_heatmap.html")

    return {
        "earnings_df": earnings_df,
        "timeline_fig": timeline_fig,
        "heatmap_fig": heatmap_fig,
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


def analyze_earnings_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze discrepancies between GAAP and Adjusted earnings."""

    gaap_adj_pairs = [
        ("eps_gaap_est_avg_fy1e", "eps_norm_est_avg_fy1e"),
        ("eps_gaap_est_avg_ntm", "eps_norm_est_avg_ntm"),
        ("net_eps_basic_ltm", "eps_adj_ltm"),
        ("net_eps_basic_fy", "eps_adj_fy"),  # Separate pair for FY
        ("net_eps_basic_1fy", "eps_adj_1fy"),  # Separate pair for 1FY
    ]

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

    for gaap_col, adj_col in gaap_adj_pairs:
        if gaap_col in df.columns and adj_col in df.columns:
            gaap_data = pd.to_numeric(df[gaap_col], errors="coerce")
            adj_data = pd.to_numeric(df[adj_col], errors="coerce")

            with np.errstate(divide="ignore", invalid="ignore"):
                adjustment = (adj_data - gaap_data) / gaap_data.abs()

            adjustment = adjustment.replace([np.inf, -np.inf], np.nan)

            suffix = gaap_col.split("_")[-1]
            adjustment_col = f"adj_magnitude_{suffix}"
            earnings_quality[adjustment_col] = adjustment * 100

            flag_col = f"large_adj_flag_{suffix}"
            earnings_quality[flag_col] = earnings_quality[adjustment_col].abs() > 35

    adj_cols = [c for c in earnings_quality.columns if "adj_magnitude" in c]
    if adj_cols:
        earnings_quality["earnings_quality_score"] = 100 - earnings_quality[adj_cols].abs().mean(
            axis=1
        ).clip(0, 100)

    return earnings_quality


def create_gaap_adjusted_comparison_chart(df: pd.DataFrame, output_path: Path) -> go.Figure:
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
            surprise = ((eps_actual[valid] - eps_est[valid]) / eps_est[valid].abs()) * 100
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
            pd.to_numeric(df["last_price"], errors="coerce").replace(0, pd.NA).astype("Float64")
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
        vol_threshold = float(vol.quantile(float(config.pre_earnings_volatility_quantile)))

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
