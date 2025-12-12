"""
Earnings and Dividend Dashboard Widgets.

Enhanced with Phase 9.3 PHASE93_FEATURE_INPUTS categories for comprehensive
metric selection across earnings, dividends, valuation, quality/risk, and
technical analysis domains.

Aligned with code_guidelines.md v1.4 Section 17 (Style Guides for Visual Elements).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Literal, Union
from datetime import timedelta
from pathlib import Path

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
]


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

    Args:
        df: Input DataFrame containing stock data.
        reference_date: Date to compare next_earnings against. Defaults to today.
        top_n: Number of top companies (by Market Cap) to include.
        mode: Display mode - 'all', 'earnings', 'dividends', or specific category name.
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
        ]
    elif mode == "earnings":
        # Earnings-focused categories
        selected_categories = [
            "profitability",
            "valuation",
            "growth",
            "momentum",
            "forecasts",
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

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

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

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return fig
