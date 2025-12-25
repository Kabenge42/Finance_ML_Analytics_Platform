"""
Earnings and Dividend Dashboard Widgets.

Enhanced with Phase 9.3 PHASE93_FEATURE_CATEGORIES categories for comprehensive
metric selection across earnings, dividends, valuation, quality/risk, and
technical analysis domains.

Aligned with code_guidelines.md v1.4 Section 17 (Style Guides for Visual Elements).
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Schema-driven Phase 9.3 feature categorization (code_guidelines.md §9.3)
from finance_ml.ml_workflow.data.schema import (
    COLUMN_SCHEMA,
    PHASE93_FEATURE_CATEGORIES,
)
from finance_ml.ml_workflow.preprocessing.dtypes import detect_and_cast_dtypes

logger = logging.getLogger(__name__)

# =============================================================================
# Style Constants (aligned with code_guidelines.md §17.1, §17.2)
# =============================================================================

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

DATE_DISPLAY_FORMAT = "%d %b %Y"
DEFAULT_REFERENCE_DATE = pd.Timestamp("2025-12-21")

# Category colors for visualization
CATEGORY_COLORS = {
    "momentum": "#3498db",  # Info blue
    "valuation": "#375a7f",  # Primary blue
    "profitability": "#00bc8c",  # Success green
    "quality_risk": "#e74c3c",  # Danger red
    "cash_flow": "#f39c12",  # Warning orange
    "growth": "#9b59b6",  # Purple
    "technical": "#1abc9c",  # Teal
    "employment": "#34495e",  # Dark gray
    "dividends": "#27ae60",  # Green
    "forecasts": "#2980b9",  # Blue
    "earnings_quality": "#e67e22",  # Orange - GAAP vs Adjusted, surprise analytics
}


def _resolve_reference_date(
    df: Optional[pd.DataFrame], reference_date: Optional[pd.Timestamp]
) -> pd.Timestamp:
    """Resolve reference_date using dataset provenance with a stable fallback.

    Preference order:
    1) Explicit reference_date argument
    2) Dataset-provided `_reference_date` or `reference_date` columns (max value)
    3) DEFAULT_REFERENCE_DATE (aligned with dtype casting & temporal features)
    """

    if reference_date is not None:
        return pd.Timestamp(reference_date).normalize()

    if df is not None:
        for candidate in ["_reference_date", "reference_date"]:
            if candidate in df.columns:
                ref_series = pd.to_datetime(df[candidate], errors="coerce")
                if ref_series.notna().any():
                    return ref_series.dropna().max().normalize()

    return DEFAULT_REFERENCE_DATE


def _add_formatted_date_columns(df: pd.DataFrame, date_columns: List[str]) -> List[str]:
    """Add `*_formatted` companions using the canonical date display format."""

    formatted_cols: List[str] = []

    for col in date_columns:
        if col not in df.columns:
            continue

        series = pd.to_datetime(df[col], errors="coerce")
        formatted_col = f"{col}_formatted"

        if formatted_col in df.columns:
            # Respect existing column but ensure the canonical format
            df[formatted_col] = series.dt.strftime(DATE_DISPLAY_FORMAT).where(series.notna(), pd.NA)
        else:
            df[formatted_col] = series.dt.strftime(DATE_DISPLAY_FORMAT).where(series.notna(), pd.NA)

        formatted_cols.append(formatted_col)

    return formatted_cols


# Valid mode options
EarningsMode = Literal[
    "all",
    "earnings",
    "dividends",
    "valuation",
    "quality_risk",
    "technical",
    "forecasts",
    "momentum",
    "profitability",
    "growth",
    "cash_flow",
    "employment",
    "earnings_quality",
]


def _write_html_artifact(fig: go.Figure, output_path: Optional[Union[str, Path]]) -> None:
    """Write a Plotly figure to HTML when an output path is provided.

    Several dashboard builders return early with an 'empty' figure (e.g., missing
    columns or no events in window). For artifact generation workflows we still
    want a deterministic HTML file to be produced.
    """

    if not output_path:
        return
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(p))


def get_category_metrics(
    categories: List[str],
    include_supplemental: bool = True,
) -> Dict[str, List[str]]:
    """
    Get metrics from specified PHASE93_FEATURE_CATEGORIES categories.

    **ETL Pipeline Note:** Some metrics may have companion `*_applicable` flags
    emitted by the ETL pipeline (Stage 8g conditional metrics handling). These
    flags indicate row-level applicability for the metric computations.

    Args:
        categories: List of category names from PHASE93_FEATURE_CATEGORIES or short names.
        include_supplemental: Whether to include supplemental domain-specific metrics.

    Returns:
        Dict mapping category name to list of metric column names.
    """
    # Mapping of short names to PHASE93_FEATURE_CATEGORIES keys
    category_mapping = {
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

    result = {}
    for cat in categories:
        # Resolve to full category name if short name provided
        full_cat = category_mapping.get(cat.lower(), cat)
        metrics = PHASE93_FEATURE_CATEGORIES.get(full_cat, []).copy()
        result[full_cat] = metrics

    # Add supplemental metrics for specific categories (using normalized keys)
    if include_supplemental:
        # Earnings-related supplemental metrics
        if "Profitability" in result or "Growth Metrics" in result:
            supplemental_earnings = [
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
            if "Profitability" in result:
                result["Profitability"].extend(supplemental_earnings)
            if "Growth Metrics" in result:
                result["Growth Metrics"].extend(supplemental_earnings)

        # Dividend-related supplemental metrics
        div_key = next((k for k in ["Dividend Reliability", "Capital Allocation"] if k in result), None)
        if div_key:
            supplemental_dividends = [
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
            result[div_key].extend(supplemental_dividends)

        # Earnings Quality supplemental metrics (Phase 9.3 output features)
        if "Earnings Quality" in result:
            supplemental_earnings_quality = [
                # Estimated vs. Actual analytics outputs
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
                # GAAP vs. Adjusted analytics outputs
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
            result["Earnings Quality"].extend(supplemental_earnings_quality)

    return result


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

    reference_date = _resolve_reference_date(df, reference_date)

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

    # Sort by Market Cap
    mcap_col = None
    for col in ["market_cap", "market_cap_usd", "market_cap_curr"]:
        if col in df.columns:
            mcap_col = col
            break

    if mcap_col:
        filtered_df = filtered_df.sort_values(by=mcap_col, ascending=False)

    filtered_df = filtered_df.head(top_n)

    # Define identity columns + temporal enrichments when available
    display_cols = ["ticker", "sector", "region", "next_earnings"]
    display_cols = [c for c in display_cols if c in df.columns or c == "next_earnings"]
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

    # Add standardized formatted date companions for display/export
    date_columns = [
        col
        for col in dashboard_df.columns
        if pd.api.types.is_datetime64_any_dtype(dashboard_df[col])
    ]

    formatted_cols = _add_formatted_date_columns(dashboard_df, date_columns)

    if formatted_cols:
        reordered: List[str] = []
        for col in dashboard_df.columns:
            reordered.append(col)
            formatted_col = f"{col}_formatted"
            if formatted_col in formatted_cols:
                reordered.append(formatted_col)
        dashboard_df = dashboard_df.loc[:, list(dict.fromkeys(reordered))]

    return dashboard_df


from typing import Any


def _build_format_dict(
    df: pd.DataFrame,
    date_columns: list[str] | None = None,
    date_format: str = DATE_DISPLAY_FORMAT,
) -> dict[str, Any]:
    """Build a format dictionary for Styler, safely handling column types.
    
    Only applies date formatting to columns that are actually datetime dtype.
    """
    format_dict = {}
    
    if date_columns is None:
        date_columns = []
    
    for col in df.columns:
        if col in date_columns:
            # Only apply date format if the column is actually datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                format_dict[col] = lambda x, fmt=date_format: (
                    x.strftime(fmt) if pd.notna(x) else ""
                )
            # If it's not datetime, skip date formatting (leave as default)
        elif pd.api.types.is_float_dtype(df[col]):
            format_dict[col] = "{:.2f}"
        elif pd.api.types.is_integer_dtype(df[col]):
            format_dict[col] = "{:,}"
    
    return format_dict


def _ensure_schema_dtypes(df: pd.DataFrame, date_columns: list[str] | None = None) -> pd.DataFrame:
    """Ensure DataFrame columns have schema-compliant dtypes before styling.

    Uses detect_and_cast_dtypes from dtypes.py for schema-aware casting.
    Specifically ensures date columns are datetime64[ns] to prevent
    strftime errors on float values.

    Args:
        df: DataFrame to cast
        date_columns: Specific date columns to ensure are datetime

    Returns:
        DataFrame with schema-compliant dtypes
    """
    df = df.copy()

    # First pass: use schema-aware dtype casting
    df_cast, diagnostics = detect_and_cast_dtypes(df, schema=COLUMN_SCHEMA)

    # Second pass: explicit datetime conversion for known date columns
    if date_columns:
        for col in date_columns:
            if col in df_cast.columns:
                if not pd.api.types.is_datetime64_any_dtype(df_cast[col]):
                    df_cast[col] = pd.to_datetime(df_cast[col], errors="coerce")

    # Log any coercion warnings for debugging
    if diagnostics.get("coercion_warnings"):
        for col, count in diagnostics["coercion_warnings"].items():
            if count > 0:
                logger.debug("Column '%s': %s values coerced during dtype casting", col, count)

    return df_cast


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
            "dividend_record_ex_date",
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
    metric_category: str = "Profitability",
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
    category_mapping = {
        "profitability": "Profitability",
        "valuation": "Valuation Ratios",
        "growth": "Growth Metrics",
        "momentum": "Momentum & Technical",
        "quality_risk": "Quality & Risk",
        "cash_flow": "Cash Flow",
        "forecasts": "Revenue Forecasting",
        "earnings_quality": "Earnings Quality",
        "dividends": "Dividend Reliability",
    }
    metric_category = category_mapping.get(metric_category.lower(), metric_category)

    reference_date = _resolve_reference_date(df, reference_date)

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


def create_category_comparison_chart(
    df: pd.DataFrame,
    categories: Optional[List[str]] = None,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """
    Creates an interactive chart comparing metrics across PHASE93 categories.

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Provides visual comparison of metric availability and values across
    all PHASE93_FEATURE_CATEGORIES categories for earnings calendar companies.

    Args:
        df: Input DataFrame containing stock data.
        categories: List of categories to compare. Defaults to all.
        reference_date: Date for earnings comparison.
        top_n: Number of companies to include.
        output_path: Optional path to save HTML output.

    Returns:
        go.Figure: Plotly figure with category comparison.
    """
    if categories is None:
        categories = list(PHASE93_FEATURE_CATEGORIES.keys())

    reference_date = _resolve_reference_date(df, reference_date)

    # Get base dashboard data
    dashboard_df = create_earnings_calendar_dashboard(
        df,
        reference_date=reference_date,
        top_n=top_n,
        mode="all",
    )

    if dashboard_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No companies found with earnings within +/- 10 days",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    # Calculate coverage statistics per category
    coverage_data = []
    for cat in categories:
        metrics = PHASE93_FEATURE_CATEGORIES.get(cat, [])
        available = [m for m in metrics if m in df.columns]
        non_null_counts = [
            dashboard_df[m].notna().sum() for m in available if m in dashboard_df.columns
        ]

        coverage_data.append(
            {
                "category": cat.replace("_", " ").title(),
                "total_metrics": len(metrics),
                "available_metrics": len(available),
                "coverage_pct": len(available) / len(metrics) * 100 if metrics else 0,
                "avg_non_null": (
                    sum(non_null_counts) / len(non_null_counts) if non_null_counts else 0
                ),
                "color": CATEGORY_COLORS.get(cat, COLOR_PALETTE["neutral"]),
            }
        )

    coverage_df = pd.DataFrame(coverage_data)

    # Create figure with two subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Metric Coverage by Category",
            "Data Availability (Avg Non-Null)",
        ],
        horizontal_spacing=0.15,
    )

    # Coverage bar chart
    fig.add_trace(
        go.Bar(
            x=coverage_df["category"],
            y=coverage_df["coverage_pct"],
            marker_color=coverage_df["color"],
            name="Coverage %",
            hovertemplate=("<b>%{x}</b><br>Coverage: %{y:.1f}%<br><extra></extra>"),
        ),
        row=1,
        col=1,
    )

    # Data availability bar chart
    fig.add_trace(
        go.Bar(
            x=coverage_df["category"],
            y=coverage_df["avg_non_null"],
            marker_color=coverage_df["color"],
            name="Avg Non-Null",
            hovertemplate=("<b>%{x}</b><br>Avg Records: %{y:.0f}<br><extra></extra>"),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text="Phase 9.3 Category Metrics: Coverage & Availability Analysis",
            font=dict(size=18),
        ),
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=12),
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_yaxes(title_text="Coverage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Avg Non-Null Records", row=1, col=2)
    fig.update_xaxes(tickangle=45)

    _write_html_artifact(fig, output_path)

    return fig


def create_earnings_surprise_dashboard(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 50,
    output_path: Optional[Union[str, Path]] = None,
    validate_quality: bool = False,
) -> go.Figure:
    """Create an interactive dashboard for earnings surprise analysis.

    Business objective: monitor expected vs actual earnings performance to
    identify forecast reliability and potential market reaction patterns.

    **ETL Pipeline Integration:**
    - Best paired with data from etl_with_financial_metrics() output (semantic typing)
    - Optional quality validation mirrors ETL Stage 9 checks

    Args:
        df: DataFrame with earnings estimates and actuals.
        reference_date: Analysis date (defaults to now).
        top_n: Number of rows to analyze (prefers market cap ordering when available).
        output_path: Optional path to save an HTML dashboard.
        validate_quality: Run ETL-style quality checks before processing.

    Returns:
        go.Figure: Plotly figure.
    """
    reference_date = _resolve_reference_date(df, reference_date)

    if validate_quality:
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

    df_local = df.copy()

    # Prefer analyzing the most liquid/large names when possible.
    mcap_col = None
    for col in ["market_cap", "market_cap_usd", "market_cap_curr"]:
        if col in df_local.columns:
            mcap_col = col
            break
    if mcap_col is not None:
        df_local[mcap_col] = pd.to_numeric(df_local[mcap_col], errors="coerce")
        df_local = df_local.sort_values(by=mcap_col, ascending=False)
    df_local = df_local.head(int(top_n))

    surprise_cols: Dict[str, Dict[str, str]] = {
        "Revenue": {"actual": "total_revenues_ltm", "estimate": "revenues_est_avg_ntm"},
        "EBITDA": {"actual": "ebitda_ltm", "estimate": "ebitda_est_avg_fy1e"},
        "EBIT": {"actual": "ebit_ltm", "estimate": "ebit_est_med_ntm"},
        "Net Income": {"actual": "net_income_is_ltm", "estimate": "net_income_adj_1fy"},
        "EPS": {"actual": "eps_adj_ltm", "estimate": "eps_norm_est_avg_ntm"},
    }

    surprise_data: List[Dict[str, float]] = []
    all_surprises: List[float] = []

    for metric_name, cols in surprise_cols.items():
        actual_col = cols["actual"]
        est_col = cols["estimate"]

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


def create_analyst_recommendation_heatmap(
    df: pd.DataFrame,
    top_n_sectors: int = 12,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create a heatmap of analyst recommendations by sector.

    Args:
        df: DataFrame containing analyst rating count columns.
        top_n_sectors: Number of sectors to display.
        output_path: Optional path to save HTML.

    Returns:
        go.Figure: Plotly figure.
    """
    rating_cols = {
        "Strong Buy": "num_strong_buys_ratings",
        "Buy": "num_buys_ratings",
        "Hold": "num_hold_ratings",
        "Sell": "num_sell_ratings",
        "Strong Sell": "num_strong_sell_ratings",
    }

    available_ratings = {k: v for k, v in rating_cols.items() if v in df.columns}
    if not available_ratings or "sector" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Required analyst rating columns not found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    df_local = df.copy()
    top_sectors = df_local["sector"].value_counts().head(int(top_n_sectors)).index

    heatmap_data: List[Dict[str, float]] = []
    for sector in top_sectors:
        sector_df = df_local[df_local["sector"] == sector]
        row: Dict[str, float] = {"Sector": str(sector)[:25]}
        for rating_name, col in available_ratings.items():
            row[rating_name] = float(pd.to_numeric(sector_df[col], errors="coerce").sum())
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data).set_index("Sector")
    row_sums = heatmap_df.sum(axis=1).replace(0, np.nan)
    heatmap_normalized = heatmap_df.div(row_sums, axis=0) * 100
    heatmap_normalized = heatmap_normalized.fillna(0)

    fig = px.imshow(
        heatmap_normalized,
        labels=dict(x="Rating Type", y="Sector", color="% of Ratings"),
        x=list(available_ratings.keys()),
        y=heatmap_normalized.index.tolist(),
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=20,
        aspect="auto",
        text_auto=".1f",
        title="<b>Analyst Recommendation Distribution by Sector</b><br><sup>Percentage of Total Ratings per Sector</sup>",
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=20,
        xaxis_title="Rating Type",
        yaxis_title="Sector",
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return fig


def create_market_movers_dashboard(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 7,
    top_n: int = 20,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Identify and visualize market movers around earnings events.

    Uses a composite mover score derived from z-scored momentum/volatility/volume
    signals when columns are available.

    Args:
        df: DataFrame with at least ticker/sector/last_price/next_earnings.
        reference_date: Analysis date.
        lookback_days: Event window around earnings.
        top_n: Number of movers to display.
        output_path: Optional path to save HTML.

    Returns:
        go.Figure: Plotly figure.
    """
    reference_date = _resolve_reference_date(df, reference_date)

    required_cols = ["ticker", "sector", "last_price", "next_earnings"]
    if not all(c in df.columns for c in required_cols):
        fig = go.Figure()
        fig.add_annotation(
            text="Required columns not found for market movers analysis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    df_local = df.copy()
    df_local["next_earnings"] = pd.to_datetime(df_local["next_earnings"], errors="coerce")
    df_local["days_to_earnings"] = (df_local["next_earnings"] - reference_date).dt.days

    mask = df_local["days_to_earnings"].abs() <= int(lookback_days)
    movers_df = df_local[mask].copy()

    if movers_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No earnings events within {lookback_days} days",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    momentum_cols = ["price_momentum_1m", "volatility_1m", "rel_volume"]
    movers_df["mover_score"] = 0.0

    for col in momentum_cols:
        if col not in movers_df.columns:
            continue
        data = pd.to_numeric(movers_df[col], errors="coerce")
        if data.notna().sum() == 0:
            continue
        std = float(data.std())
        if std == 0 or np.isnan(std):
            continue
        z_score = (data - float(data.mean())) / std
        movers_df["mover_score"] += z_score.abs().fillna(0.0)

    top_movers = movers_df.sort_values(by="mover_score", ascending=False).head(int(top_n))
    if top_movers.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient momentum/volatility data to compute mover scores",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    fig = go.Figure()

    pre_earnings = top_movers[top_movers["days_to_earnings"] > 0]
    post_earnings = top_movers[top_movers["days_to_earnings"] <= 0]

    if not pre_earnings.empty:
        fig.add_trace(
            go.Scatter(
                x=pre_earnings["days_to_earnings"],
                y=pre_earnings["mover_score"],
                mode="markers+text",
                marker=dict(
                    size=15,
                    color=COLOR_PALETTE["warning"],
                    line=dict(width=2, color="white"),
                ),
                text=pre_earnings["ticker"],
                textposition="top center",
                textfont=dict(size=10),
                name="Pre-Earnings",
                hovertemplate=(
                    "<b>%{text}</b><br>Days to Earnings: %{x}<br>Mover Score: %{y:.2f}<extra></extra>"
                ),
            )
        )

    if not post_earnings.empty:
        fig.add_trace(
            go.Scatter(
                x=post_earnings["days_to_earnings"],
                y=post_earnings["mover_score"],
                mode="markers+text",
                marker=dict(
                    size=15,
                    color=COLOR_PALETTE["success"],
                    line=dict(width=2, color="white"),
                ),
                text=post_earnings["ticker"],
                textposition="top center",
                textfont=dict(size=10),
                name="Post-Earnings",
                hovertemplate=(
                    "<b>%{text}</b><br>Days Since Earnings: %{x}<br>Mover Score: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="white",
        annotation_text="Earnings Date",
        annotation_position="top",
    )

    fig.update_layout(
        title="<b>Market Movers: Earnings Event Window Analysis</b><br><sup>Top Stocks by Volatility/Momentum Score</sup>",
        template=PLOTLY_TEMPLATE,
        height=600,
        xaxis_title="Days Relative to Earnings",
        yaxis_title="Mover Score (Composite)",
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=20,
        showlegend=True,
        hovermode="closest",
    )

    _write_html_artifact(fig, output_path)

    return fig


def create_price_target_analytics(
    df: pd.DataFrame,
    top_n_sectors: int = 12,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create price target analytics with confidence bands and spread analysis."""
    required_cols = ["ticker", "sector", "last_price", "price_target"]
    if not all(c in df.columns for c in required_cols):
        fig = go.Figure()
        fig.add_annotation(
            text="Required price target columns not found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    df_local = df.copy()

    for col in [
        "last_price",
        "price_target",
        "price_target_high",
        "price_target_low",
        "target_vs_price",
    ]:
        if col in df_local.columns:
            df_local[col] = pd.to_numeric(df_local[col], errors="coerce")

    if "price_target_high" in df_local.columns and "price_target_low" in df_local.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df_local["target_spread"] = (
                (df_local["price_target_high"] - df_local["price_target_low"])
                / df_local["last_price"]
            ) * 100
        df_local["target_spread"] = df_local["target_spread"].replace([np.inf, -np.inf], np.nan)

    top_sectors = df_local["sector"].value_counts().head(int(top_n_sectors)).index
    sector_stats: List[Dict[str, float]] = []

    for sector in top_sectors:
        sector_df = df_local[df_local["sector"] == sector]

        if "target_vs_price" in sector_df.columns:
            upside = sector_df["target_vs_price"].replace([np.inf, -np.inf], np.nan).dropna()
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                upside = (
                    (sector_df["price_target"] - sector_df["last_price"]) / sector_df["last_price"]
                ) * 100
            upside = upside.replace([np.inf, -np.inf], np.nan).dropna()

        if len(upside) < 5:
            continue

        sector_stats.append(
            {
                "sector": str(sector)[:20],
                "mean_upside": float(upside.mean()),
                "median_upside": float(upside.median()),
                "q25_upside": float(upside.quantile(0.25)),
                "q75_upside": float(upside.quantile(0.75)),
                "count": float(len(upside)),
                "mean_spread": (
                    float(sector_df["target_spread"].mean())
                    if "target_spread" in sector_df.columns
                    else np.nan
                ),
            }
        )

    if not sector_stats:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for price target analytics",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    stats_df = pd.DataFrame(sector_stats).sort_values("mean_upside", ascending=True)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Mean Target Upside by Sector (%)",
            "Target Spread (High-Low) by Sector",
            "Upside Distribution (All Stocks)",
            "Consensus Confidence Score",
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "bar"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
    )

    colors = [
        COLOR_PALETTE["success"] if x > 0 else COLOR_PALETTE["danger"]
        for x in stats_df["mean_upside"]
    ]
    fig.add_trace(
        go.Bar(
            x=stats_df["mean_upside"],
            y=stats_df["sector"],
            orientation="h",
            marker_color=colors,
            error_x=dict(
                type="data",
                symmetric=False,
                array=(stats_df["q75_upside"] - stats_df["mean_upside"]).clip(lower=0),
                arrayminus=(stats_df["mean_upside"] - stats_df["q25_upside"]).clip(lower=0),
                color="rgba(255,255,255,0.3)",
            ),
            name="Mean Upside",
            hovertemplate=(
                "<b>%{y}</b><br>Mean: %{x:.1f}%<br>Q25-Q75: %{customdata[0]:.1f}% - %{customdata[1]:.1f}%<extra></extra>"
            ),
            customdata=stats_df[["q25_upside", "q75_upside"]].values,
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="white", row=1, col=1)

    if not stats_df["mean_spread"].isna().all():
        spread_colors = [
            COLOR_PALETTE["success"] if x < 20 else COLOR_PALETTE["warning"]
            for x in stats_df["mean_spread"].fillna(0)
        ]
        fig.add_trace(
            go.Bar(
                x=stats_df["mean_spread"],
                y=stats_df["sector"],
                orientation="h",
                marker_color=spread_colors,
                name="Target Spread",
                hovertemplate="<b>%{y}</b><br>Spread: %{x:.1f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

    if "target_vs_price" in df_local.columns:
        all_upside = df_local["target_vs_price"].replace([np.inf, -np.inf], np.nan).dropna()
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            all_upside = (
                (df_local["price_target"] - df_local["last_price"]) / df_local["last_price"]
            ) * 100
        all_upside = all_upside.replace([np.inf, -np.inf], np.nan).dropna()

    if len(all_upside) > 0:
        fig.add_trace(
            go.Histogram(
                x=all_upside.clip(-50, 100),
                nbinsx=50,
                marker_color=COLOR_PALETTE["info"],
                name="Upside Distribution",
            ),
            row=2,
            col=1,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white", row=2, col=1)

    if not stats_df["mean_spread"].isna().all():
        stats_df["confidence_score"] = 100 / (1 + stats_df["mean_spread"].fillna(50))
        top_confidence = stats_df.nlargest(10, "confidence_score")
        fig.add_trace(
            go.Bar(
                x=top_confidence["confidence_score"],
                y=top_confidence["sector"],
                orientation="h",
                marker_color=COLOR_PALETTE["success"],
                name="Confidence Score",
                hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title="<b>Price Target Analytics Dashboard</b><br><sup>Analyst Consensus & Confidence Analysis</sup>",
        template=PLOTLY_TEMPLATE,
        height=800,
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=20,
    )

    fig.update_xaxes(title_text="Upside (%)", row=1, col=1)
    fig.update_xaxes(title_text="Spread (%)", row=1, col=2)
    fig.update_xaxes(title_text="Upside (%)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Confidence Score", row=2, col=2)

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

    reference_date = _resolve_reference_date(df, reference_date)

    earnings_date_cols = [
        "next_earnings",
        "last_earnings_date",
        "income_statement_report_date",
        "next_earnings_date",
        "earnings_announcement_date",
    ]
    available_earnings_dates = [c for c in earnings_date_cols if c in df.columns]

    if not available_earnings_dates:
        logger.warning("No earnings date columns found. Falling back to engineered events.")
        return _engineer_earnings_events_from_fiscal_data(df, output_dir, reference_date)

    identity_cols = [c for c in ["ticker", "sector", "region"] if c in df.columns]
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
    """Create interactive earnings timeline with sector coloring."""

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
        "sector"
        if "sector" in valid_df.columns
        else "region" if "region" in valid_df.columns else "ticker"
    )
    color_dim = "sector" if "sector" in valid_df.columns else None

    fig = px.scatter(
        valid_df,
        x="days_to_earnings",
        y=y_dim,
        color=color_dim,
        hover_data=[c for c in ["ticker", "region"] if c in valid_df.columns],
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
    """Create earnings density heatmap by sector and week."""

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
        "sector"
        if "sector" in earnings_df.columns
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
    ]

    base_cols = [c for c in ["ticker", "sector", "region"] if c in df.columns]
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
            earnings_quality[flag_col] = earnings_quality[adjustment_col].abs() > 20

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


def create_technical_valuation_dashboard(df: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    """Create technical analysis dashboard overlaying valuation metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    ema_cols = ["ema_20d", "ema_50d", "ema_100d", "ema_250d"]
    momentum_cols = ["price_momentum_1m", "price_momentum_3m", "price_momentum_6m"]
    valuation_cols = ["p_e_ratio", "ev_ebitda_ratio", "p_s_ratio", "p_e_ntm", "ev_ebitda_ltm"]

    if "ema_20d" in df.columns and "ema_50d" in df.columns:
        df["ema_crossover_bullish"] = df["ema_20d"] > df["ema_50d"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["ema_crossover_score"] = ((df["ema_20d"] - df["ema_50d"]) / df["ema_50d"]) * 100

    if "52w_high_adj" in df.columns and "52w_low_adj" in df.columns and "last_price" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["price_52w_position"] = (
                (df["last_price"] - df["52w_low_adj"]) / (df["52w_high_adj"] - df["52w_low_adj"])
            ) * 100
        df["price_52w_position"] = df["price_52w_position"].clip(0, 100)

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "EMA Crossover Distribution",
            "Momentum Scatter (1M vs 3M)",
            "52W Position by Sector",
            "Valuation vs Momentum",
            "Technical Score Distribution",
            "Bullish/Bearish Ratio by Sector",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    if "ema_crossover_score" in df.columns:
        fig.add_trace(
            go.Histogram(
                x=df["ema_crossover_score"].dropna().clip(-50, 50),
                nbinsx=50,
                marker_color=COLOR_PALETTE["success"],
                name="EMA Crossover %",
            ),
            row=1,
            col=1,
        )

    available_momentum = [c for c in momentum_cols if c in df.columns]
    if len(available_momentum) >= 2:
        fig.add_trace(
            go.Scatter(
                x=df[available_momentum[0]],
                y=df[available_momentum[1]],
                mode="markers",
                marker=dict(
                    size=6,
                    color=df.get("price_52w_position", 50),
                    colorscale="RdYlGn",
                    opacity=0.7,
                ),
                name="Momentum",
            ),
            row=1,
            col=2,
        )

    if "price_52w_position" in df.columns and "sector" in df.columns:
        sector_position = df.groupby("sector")["price_52w_position"].median().dropna()
        if not sector_position.empty:
            fig.add_trace(
                go.Bar(
                    y=sector_position.index,
                    x=sector_position.values,
                    orientation="h",
                    marker_color=COLOR_PALETTE["info"],
                    name="52W Position",
                ),
                row=2,
                col=1,
            )

    if available_momentum and valuation_cols:
        valuation_col = next((c for c in valuation_cols if c in df.columns), None)
        if valuation_col and available_momentum:
            fig.add_trace(
                go.Scatter(
                    x=df[valuation_col],
                    y=df[available_momentum[0]],
                    mode="markers",
                    marker=dict(color=df.get("ema_crossover_score", 0), colorscale="Bluered"),
                    name="Valuation vs Momentum",
                    text=df.get("ticker"),
                ),
                row=2,
                col=2,
            )

    if "ema_crossover_score" in df.columns:
        fig.add_trace(
            go.Box(
                x=df["ema_crossover_score"].dropna(),
                marker_color=COLOR_PALETTE["secondary"],
                name="Technical Score",
                boxmean=True,
            ),
            row=3,
            col=1,
        )

    if "ema_crossover_bullish" in df.columns and "sector" in df.columns:
        bullish_ratio = df.groupby("sector")["ema_crossover_bullish"].mean().dropna()
        if not bullish_ratio.empty:
            fig.add_trace(
                go.Bar(
                    x=bullish_ratio.values * 100,
                    y=bullish_ratio.index,
                    orientation="h",
                    marker_color=COLOR_PALETTE["success"],
                    name="Bullish %",
                ),
                row=3,
                col=2,
            )

    fig.update_layout(
        title="<b>Technical Analysis + Valuation Dashboard</b>",
        template=PLOTLY_TEMPLATE,
        height=900,
        showlegend=True,
    )

    output_path = output_dir / "technical_valuation_dashboard.html"
    _write_html_artifact(fig, output_path)

    return {"figure": fig, "output_path": output_path}


def create_dividend_sustainability_scorecard(
    df: pd.DataFrame, output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Create comprehensive dividend sustainability scoring."""

    scorecard = df[[c for c in ["ticker", "sector", "region"] if c in df.columns]].copy()

    if "payout_ratio" in df.columns:
        payout = pd.to_numeric(df["payout_ratio"], errors="coerce")
        scorecard["payout_score"] = np.where(
            payout <= 50,
            25,
            np.where(payout <= 75, 20, np.where(payout <= 100, 10, 0)),
        )
    else:
        scorecard["payout_score"] = 0

    fcf_col = next((c for c in df.columns if "fcf" in c.lower() and "yield" not in c.lower()), None)
    div_paid_col = next(
        (c for c in df.columns if "dividend" in c.lower() and "paid" in c.lower()), None
    )

    if fcf_col and div_paid_col:
        fcf = pd.to_numeric(df[fcf_col], errors="coerce")
        div_paid = pd.to_numeric(df[div_paid_col], errors="coerce").abs()
        coverage = fcf / div_paid.replace(0, np.nan)

        scorecard["fcf_coverage_score"] = np.where(
            coverage >= 2.0,
            25,
            np.where(
                coverage >= 1.5, 20, np.where(coverage >= 1.0, 15, np.where(coverage >= 0.5, 5, 0))
            ),
        )
    else:
        scorecard["fcf_coverage_score"] = 0

    growth_cols = [c for c in df.columns if "div" in c.lower() and "growth" in c.lower()]
    if growth_cols:
        div_growth = pd.to_numeric(df[growth_cols[0]], errors="coerce")
        scorecard["div_growth_score"] = np.where(
            div_growth >= 10,
            25,
            np.where(
                div_growth >= 5, 20, np.where(div_growth >= 0, 15, np.where(div_growth >= -5, 5, 0))
            ),
        )
    else:
        scorecard["div_growth_score"] = 0

    if "debt_to_equity" in df.columns:
        dte = pd.to_numeric(df["debt_to_equity"], errors="coerce")
        scorecard["balance_sheet_score"] = np.where(
            dte <= 0.5,
            25,
            np.where(dte <= 1.0, 20, np.where(dte <= 2.0, 10, 0)),
        )
    else:
        scorecard["balance_sheet_score"] = 0

    score_cols = [
        "payout_score",
        "fcf_coverage_score",
        "div_growth_score",
        "balance_sheet_score",
    ]
    scorecard["dividend_sustainability_score"] = scorecard[score_cols].sum(axis=1)

    scorecard["sustainability_grade"] = pd.cut(
        scorecard["dividend_sustainability_score"],
        bins=[0, 40, 60, 75, 90, 100],
        labels=["F", "D", "C", "B", "A"],
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scorecard.to_csv(output_path, index=False)

    return scorecard


def create_employee_productivity_dashboard(df: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    """Analyze employee productivity metrics and trends."""

    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    sector_categories = df["sector"].astype("category") if "sector" in df.columns else None

    if "total_revenues_ltm" in df.columns and "full_time_employees_fq" in df.columns:
        revenue = pd.to_numeric(df["total_revenues_ltm"], errors="coerce")
        employees = pd.to_numeric(df["full_time_employees_fq"], errors="coerce")
        df["revenue_per_employee"] = (revenue / employees.replace(0, np.nan)) / 1000

    if "total_assets_ltm" in df.columns and "full_time_employees_fq" in df.columns:
        assets = pd.to_numeric(df["total_assets_ltm"], errors="coerce")
        employees = pd.to_numeric(df["full_time_employees_fq"], errors="coerce")
        df["assets_per_employee"] = (assets / employees.replace(0, np.nan)) / 1000

    emp_fq = "full_time_employees_fq"
    emp_1fy = "full_time_employees_1fy"
    if emp_fq in df.columns and emp_1fy in df.columns:
        current = pd.to_numeric(df[emp_fq], errors="coerce")
        prior = pd.to_numeric(df[emp_1fy], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["employee_growth_yoy"] = ((current - prior) / prior * 100).replace(
                [np.inf, -np.inf], np.nan
            )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Revenue per Employee by Sector",
            "Employee Growth Distribution",
            "Productivity vs Profitability",
            "Employee Count vs Revenue (Log Scale)",
        ],
    )

    if "revenue_per_employee" in df.columns and "ebitda_ltm" in df.columns:
        # Map sectors to numeric values for color scale to avoid invalid color strings
        sector_codes = sector_categories.cat.codes if sector_categories is not None else None
        colorbar = None
        if sector_categories is not None:
            unique_codes = sorted(sector_categories.cat.codes.unique())
            colorbar = dict(
                title="Sector",
                tickmode="array",
                tickvals=unique_codes,
                ticktext=[sector_categories.cat.categories[code] for code in unique_codes],
            )

        fig.add_trace(
            go.Scatter(
                x=df["revenue_per_employee"],
                y=df["ebitda_ltm"],
                mode="markers",
                marker=dict(
                    color=sector_codes,
                    colorscale="Viridis",
                    showscale=sector_categories is not None,
                    colorbar=colorbar,
                ),
                text=df.get("ticker"),
                name="Productivity vs Profitability",
            ),
            row=2,
            col=1,
        )

    if "employee_growth_yoy" in df.columns:
        fig.add_trace(
            go.Histogram(
                x=df["employee_growth_yoy"].dropna(),
                marker_color=COLOR_PALETTE["warning"],
                name="Employee Growth YoY",
            ),
            row=1,
            col=2,
        )

    if "full_time_employees_fq" in df.columns and "total_revenues_ltm" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["full_time_employees_fq"],
                y=df["total_revenues_ltm"],
                mode="markers",
                marker=dict(
                    color=sector_categories.cat.codes if sector_categories is not None else None,
                    colorscale="Viridis",
                    showscale=sector_categories is not None,
                    colorbar=colorbar,
                ),
                text=df.get("ticker"),
                name="Employees vs Revenue",
            ),
            row=2,
            col=2,
        )
        fig.update_xaxes(type="log", row=2, col=2)
        fig.update_yaxes(type="log", row=2, col=2)

    fig.update_layout(
        title="<b>Employee Productivity Analytics</b>", template=PLOTLY_TEMPLATE, height=800
    )

    output_path = output_dir / "employee_productivity.html"
    _write_html_artifact(fig, output_path)

    metrics = {
        k: df[k]
        for k in ["revenue_per_employee", "assets_per_employee", "employee_growth_yoy"]
        if k in df.columns
    }

    return {"figure": fig, "metrics": metrics, "output_path": output_path}


def create_category_correlation_network(
    df: pd.DataFrame,
    category_mapping: Optional[Dict[str, List[str]]],
    output_dir: Path,
) -> Optional[go.Figure]:
    """Create interactive network visualization showing correlations between feature categories."""

    if category_mapping is None:
        category_mapping = PHASE93_FEATURE_CATEGORIES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_scores: Dict[str, pd.Series] = {}
    for cat_name, features in category_mapping.items():
        available = [f for f in features if f in df.columns]
        if available:
            cat_data = df[available].apply(pd.to_numeric, errors="coerce")
            cat_data = cat_data.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)
            category_scores[cat_name] = cat_data.mean(axis=1)

    if len(category_scores) < 2:
        return None

    category_df = pd.DataFrame(category_scores)
    corr_matrix = category_df.corr()

    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed; skipping category correlation network generation")
        return None

    G = nx.Graph()
    for cat in corr_matrix.columns:
        G.add_node(cat)

    threshold = 0.3
    for i, cat1 in enumerate(corr_matrix.columns):
        for j, cat2 in enumerate(corr_matrix.columns):
            if i < j:
                corr = corr_matrix.loc[cat1, cat2]
                if abs(corr) > threshold:
                    G.add_edge(cat1, cat2, weight=abs(corr), correlation=corr)

    pos = nx.spring_layout(G, k=2, iterations=50)

    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        corr = edge[2]["correlation"]
        color = "#00bc8c" if corr > 0 else "#e74c3c"

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=abs(corr) * 5, color=color),
                hoverinfo="none",
            )
        )

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(size=20, color="#3498db"),
        hoverinfo="text",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="<b>Feature Category Correlation Network</b><br><sup>Green=Positive, Red=Negative correlation</sup>",
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
    )

    output_path = output_dir / "category_correlation_network.html"
    _write_html_artifact(fig, output_path)

    return fig


@dataclass(frozen=True)
class EarningsAlertConfig:
    """Configuration for rule-based earnings monitoring alerts."""

    # Alert 1: EPS surprise miss threshold (negative surprise beyond this magnitude)
    eps_surprise_miss_threshold_pct: float = 20.0

    # Alert 2: Analyst downgrade momentum thresholds
    analyst_downgrade_threshold_pct: float = 5.0
    analyst_downgrade_min_periods: int = 2

    # Alert 3: Price target uncertainty via spread (high-low) relative to price
    target_spread_threshold_pct: float = 30.0

    # Alert 4: Pre-earnings elevated volatility
    pre_earnings_window_days: int = 7
    pre_earnings_volatility_quantile: float = 0.75

    # Output controls
    max_tickers_per_alert: int = 10


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

    reference_date = _resolve_reference_date(df, reference_date)

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
    eps_actual_col = "eps_adj_ltm"
    eps_est_col = "eps_norm_est_avg_ntm"
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
        "eps_est_avg_rev_pct_fy1e_1m",
        "eps_est_avg_rev_pct_fy1e_3m",
        "eps_est_avg_rev_pct_fy1e_6m",
    ]
    available_rev_cols = [c for c in default_rev_cols if c in df.columns]
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
        high = pd.to_numeric(df["price_target_high"], errors="coerce")
        low = pd.to_numeric(df["price_target_low"], errors="coerce")
        last_price = pd.to_numeric(df["last_price"], errors="coerce").replace(0, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            spread_pct = ((high - low) / last_price) * 100
        spread_pct = spread_pct.replace([np.inf, -np.inf], np.nan).dropna()

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
