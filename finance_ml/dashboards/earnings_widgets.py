"""
Earnings and Dividend Dashboard Widgets.

Enhanced with Phase 9.3 PHASE93_FEATURE_INPUTS categories for comprehensive
metric selection across earnings, dividends, valuation, quality/risk, and
technical analysis domains.

Aligned with code_guidelines.md v1.4 Section 17 (Style Guides for Visual Elements).
"""

import json
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
from finance_ml.ml_workflow.data.schema import PHASE93_FEATURE_INPUTS

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
    Get metrics from specified PHASE93_FEATURE_INPUTS categories.

    Args:
        categories: List of category names from PHASE93_FEATURE_INPUTS.
        include_supplemental: Whether to include supplemental domain-specific metrics.

    Returns:
        Dict mapping category name to list of metric column names.
    """
    result = {}
    for cat in categories:
        metrics = PHASE93_FEATURE_INPUTS.get(cat, []).copy()
        result[cat] = metrics

    # Add supplemental metrics for specific categories
    if include_supplemental:
        # Earnings-related supplemental metrics
        if "profitability" in result or "growth" in result:
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
            if "profitability" in result:
                result["profitability"].extend(supplemental_earnings)

        # Dividend-related supplemental metrics
        if "dividends" in result:
            supplemental_dividends = [
                "dividend_record_announce_date",
                "dividend_record_ex_date",
                "dividend_record_payable_date",
                "dividend_record_record_date",
                "dividend_record_frequency",
                "dividend_record_currency",
                "div_yield_ind",
                "div_yield_1fyind",
                "div_yield_5yavgltm",
                "dividend_per_share",
                "common_dividends_paid_fy",
                "dividends_paid",
                "dividends_paid_ltm",
            ]
            result["dividends"].extend(supplemental_dividends)

        # Earnings Quality supplemental metrics (Phase 9.3 output features)
        if "earnings_quality" in result:
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
            result["earnings_quality"].extend(supplemental_earnings_quality)

    return result


def create_earnings_calendar_dashboard(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 100,
    mode: EarningsMode = "all",
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Creates a dashboard (styled DataFrame) for Earnings and Dividend Analytics.
    Filters for companies with upcoming or recent earnings (t +/- 10 days).

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Uses PHASE93_FEATURE_INPUTS categories for metric selection:
    - **momentum**: Price changes, returns, EMAs (market reaction context)
    - **valuation**: P/E, P/B, EV/EBITDA multiples (valuation impact)
    - **profitability**: Margins, EBITDA, EBIT, net income (earnings quality)
    - **quality_risk**: Altman Z-score, ROE, ROA, volatility (risk assessment)
    - **cash_flow**: CFO, FCF (dividend sustainability, earnings quality)
    - **growth**: Revenue CAGR, growth estimates (growth trajectory)
    - **technical**: EMAs, 52W high/low (price context)
    - **employment**: Workforce metrics (operational health)
    - **dividends**: Dividend yields, streaks, payments (income metrics)
    - **forecasts**: Analyst estimates (earnings expectations)
    - **earnings_quality**: EPS/revenue surprises, GAAP vs. Adjusted analytics,
      earnings quality scores, adjustment flags (earnings integrity assessment)

    Args:
        df: Input DataFrame containing stock data.
        reference_date: Date to compare next_earnings against. Defaults to today.
        top_n: Number of top companies (by Market Cap) to include.
        mode: Display mode - 'all', 'earnings', 'dividends', 'earnings_quality',
            or specific category name.
        categories: Optional list of specific PHASE93 categories to include.
            Overrides mode if provided.

    Returns:
        pd.DataFrame: Filtered DataFrame with selected metrics.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()

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

    # Filter logic: next_earnings within +/- 10 days
    if "next_earnings" not in df.columns:
        print("Warning: 'next_earnings' column not found. Returning empty dataframe.")
        return pd.DataFrame()

    mask = (df["next_earnings"] - reference_date).abs() <= timedelta(days=10)
    filtered_df = df[mask].copy()

    # Sort by Market Cap
    mcap_col = None
    for col in ["market_cap", "market_cap_usd", "market_cap_curr"]:
        if col in df.columns:
            mcap_col = col
            break

    if mcap_col:
        filtered_df = filtered_df.sort_values(by=mcap_col, ascending=False)

    filtered_df = filtered_df.head(top_n)

    # Define identity columns
    display_cols = ["ticker", "sector", "region", "next_earnings"]
    display_cols = [c for c in display_cols if c in df.columns or c == "next_earnings"]
    if mcap_col and mcap_col not in display_cols:
        display_cols.append(mcap_col)

    # Determine which categories to include based on mode or explicit categories
    if categories is not None:
        selected_categories = categories
    elif mode == "all":
        # Include all major categories for comprehensive view
        selected_categories = [
            "profitability",
            "valuation",
            "growth",
            "momentum",
            "quality_risk",
            "cash_flow",
            "dividends",
            "forecasts",
            "earnings_quality",
        ]
    elif mode == "earnings":
        # Earnings-focused categories
        selected_categories = [
            "profitability",
            "valuation",
            "growth",
            "momentum",
            "forecasts",
            "earnings_quality",
        ]
    elif mode == "dividends":
        # Dividend-focused categories
        selected_categories = ["dividends", "cash_flow"]
    elif mode in PHASE93_FEATURE_INPUTS:
        # Single category mode
        selected_categories = [mode]
    else:
        # Default to earnings mode
        selected_categories = ["profitability", "growth", "momentum"]

    # Get metrics from selected categories
    category_metrics = get_category_metrics(
        selected_categories, include_supplemental=True
    )

    # Build final columns list
    final_cols = display_cols.copy()
    for cat, metrics in category_metrics.items():
        existing_metrics = [c for c in metrics if c in df.columns]
        final_cols.extend(existing_metrics)

    # Remove duplicates while preserving order
    final_cols = list(dict.fromkeys(final_cols))

    # Filter to only columns that exist
    final_cols = [c for c in final_cols if c in filtered_df.columns]

    dashboard_df = filtered_df[final_cols].copy()

    # Add computed columns
    if "next_earnings" in dashboard_df.columns:
        dashboard_df["days_to_earnings"] = (
            dashboard_df["next_earnings"] - reference_date
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


def _build_format_dict(columns: List[str]) -> Dict[str, str]:
    """
    Build format dictionary for DataFrame styling based on column names.

    Args:
        columns: List of column names to format.

    Returns:
        Dict mapping column names to format strings.
    """
    format_dict = {}

    for col in columns:
        col_lower = col.lower()

        # Date columns
        if any(
            x in col_lower
            for x in ["date", "next_earnings", "last_updated", "record_date"]
        ):
            format_dict[col] = "{:%Y-%m-%d}"

        # Currency/Price columns
        elif any(
            x in col_lower
            for x in [
                "market_cap",
                "enterprise_value",
                "price",
                "ebitda",
                "ebit",
                "revenue",
                "income",
                "fcf",
                "cfo",
                "cfi",
                "cff",
                "capex",
                "capital_expenditure",
                "eps",
                "dividend_per_share",
                "dividends_paid",
                "gross_profit",
            ]
        ):
            if "pct" in col_lower or "yield" in col_lower or "margin" in col_lower:
                format_dict[col] = "{:.2%}"
            elif "eps" in col_lower or "per_share" in col_lower:
                format_dict[col] = "${:.2f}"
            else:
                format_dict[col] = "${:,.0f}"

        # Percentage columns
        elif any(
            x in col_lower
            for x in [
                "pct",
                "yield",
                "margin",
                "return",
                "roe",
                "roa",
                "cagr",
                "volatility",
            ]
        ):
            format_dict[col] = "{:.2%}"

        # Ratio columns
        elif any(
            x in col_lower
            for x in [
                "p_e",
                "p_b",
                "p_tbv",
                "ev_sales",
                "ev_ebitda",
                "current_ratio",
                "beta",
                "altman",
                "z_score",
            ]
        ):
            format_dict[col] = "{:.2f}"

        # Integer columns
        elif any(x in col_lower for x in ["employees", "streak", "count", "num"]):
            format_dict[col] = "{:,.0f}"

        # Days column
        elif "days" in col_lower:
            format_dict[col] = "{:+.0f}"

    return format_dict


def display_earnings_dashboard(
    df: pd.DataFrame,
    mode: EarningsMode = "all",
    categories: Optional[List[str]] = None,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 100,
) -> Optional["pd.io.formats.style.Styler"]:
    """
    Displays the earnings dashboard using Pandas Styler with enhanced formatting.

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Supports all PHASE93_FEATURE_INPUTS categories with appropriate formatting:
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

    Returns:
        pd.io.formats.style.Styler: Styled DataFrame for display, or None if empty.
    """
    dashboard_df = create_earnings_calendar_dashboard(
        df,
        reference_date=reference_date,
        top_n=top_n,
        mode=mode,
        categories=categories,
    )

    if dashboard_df.empty:
        print("No companies found with earnings within +/- 10 days.")
        return None

    # Build comprehensive format dictionary
    format_dict = _build_format_dict(list(dashboard_df.columns))

    # Apply styling
    styler = dashboard_df.style.format(format_dict, na_rep="-")

    # Color-code days_to_earnings (aligned with code_guidelines.md §17.1 colors)
    def color_days(val):
        if pd.isna(val):
            return ""
        if val < 0:
            return f"color: {COLOR_PALETTE['danger']}"  # Past
        if val == 0:
            return (
                f"background-color: {COLOR_PALETTE['warning']}; color: black"  # Today
            )
        if val > 0:
            return f"color: {COLOR_PALETTE['success']}"  # Future
        return ""

    if "days_to_earnings" in dashboard_df.columns:
        styler = styler.map(color_days, subset=["days_to_earnings"])

    # Add caption with mode info
    mode_display = mode.replace("_", " ").title()
    styler = styler.set_caption(
        f"Earnings Calendar Dashboard - Mode: {mode_display} "
        f"(Top {len(dashboard_df)} by Market Cap)"
    )

    return styler


def create_earnings_metrics_chart(
    df: pd.DataFrame,
    metric_category: str = "profitability",
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 20,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """
    Creates an interactive Plotly chart showing metrics for upcoming earnings.

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Visualizes metrics from specified PHASE93_FEATURE_INPUTS category.

    **Style Guide Alignment (code_guidelines.md §17.2):**
    - Uses PLOTLY_TEMPLATE ('plotly_dark')
    - Standard color palette from COLOR_PALETTE
    - Hover data includes ticker, sector, region
    - Labeled axes with units

    Args:
        df: Input DataFrame containing stock data.
        metric_category: PHASE93_FEATURE_INPUTS category to visualize.
        reference_date: Date for earnings comparison. Defaults to today.
        top_n: Number of companies to include.
        output_path: Optional path to save HTML output.

    Returns:
        go.Figure: Plotly figure object.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()

    # Get dashboard data for the specific category
    dashboard_df = create_earnings_calendar_dashboard(
        df,
        reference_date=reference_date,
        top_n=top_n,
        mode=metric_category
        if metric_category in PHASE93_FEATURE_INPUTS
        else "earnings",
        categories=[metric_category]
        if metric_category in PHASE93_FEATURE_INPUTS
        else None,
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
    category_metrics = PHASE93_FEATURE_INPUTS.get(metric_category, [])
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
    all PHASE93_FEATURE_INPUTS categories for earnings calendar companies.

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
        categories = list(PHASE93_FEATURE_INPUTS.keys())

    if reference_date is None:
        reference_date = pd.Timestamp.now()

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
        metrics = PHASE93_FEATURE_INPUTS.get(cat, [])
        available = [m for m in metrics if m in df.columns]
        non_null_counts = [
            dashboard_df[m].notna().sum()
            for m in available
            if m in dashboard_df.columns
        ]

        coverage_data.append(
            {
                "category": cat.replace("_", " ").title(),
                "total_metrics": len(metrics),
                "available_metrics": len(available),
                "coverage_pct": len(available) / len(metrics) * 100 if metrics else 0,
                "avg_non_null": (
                    sum(non_null_counts) / len(non_null_counts)
                    if non_null_counts
                    else 0
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
) -> go.Figure:
    """Create an interactive dashboard for earnings surprise analysis.

    Business objective: monitor expected vs actual earnings performance to
    identify forecast reliability and potential market reaction patterns.

    Args:
        df: DataFrame with earnings estimates and actuals.
        reference_date: Analysis date (defaults to now).
        top_n: Number of rows to analyze (prefers market cap ordering when available).
        output_path: Optional path to save an HTML dashboard.

    Returns:
        go.Figure: Plotly figure.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()

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
            row[rating_name] = float(
                pd.to_numeric(sector_df[col], errors="coerce").sum()
            )
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
    if reference_date is None:
        reference_date = pd.Timestamp.now()

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
    df_local["next_earnings"] = pd.to_datetime(
        df_local["next_earnings"], errors="coerce"
    )
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

    top_movers = movers_df.sort_values(by="mover_score", ascending=False).head(
        int(top_n)
    )
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

    if (
        "price_target_high" in df_local.columns
        and "price_target_low" in df_local.columns
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            df_local["target_spread"] = (
                (df_local["price_target_high"] - df_local["price_target_low"])
                / df_local["last_price"]
            ) * 100
        df_local["target_spread"] = df_local["target_spread"].replace(
            [np.inf, -np.inf], np.nan
        )

    top_sectors = df_local["sector"].value_counts().head(int(top_n_sectors)).index
    sector_stats: List[Dict[str, float]] = []

    for sector in top_sectors:
        sector_df = df_local[df_local["sector"] == sector]

        if "target_vs_price" in sector_df.columns:
            upside = (
                sector_df["target_vs_price"].replace([np.inf, -np.inf], np.nan).dropna()
            )
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                upside = (
                    (sector_df["price_target"] - sector_df["last_price"])
                    / sector_df["last_price"]
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
                arrayminus=(stats_df["mean_upside"] - stats_df["q25_upside"]).clip(
                    lower=0
                ),
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
        all_upside = (
            df_local["target_vs_price"].replace([np.inf, -np.inf], np.nan).dropna()
        )
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            all_upside = (
                (df_local["price_target"] - df_local["last_price"])
                / df_local["last_price"]
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

    if reference_date is None:
        reference_date = pd.Timestamp.now()

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
