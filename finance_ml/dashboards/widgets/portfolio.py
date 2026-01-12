"""portfolio.py - Dashboard widgets for portfolio analytics.

All dashboard creation functions follow a consistent pattern:
- Accept `output_path` as Optional[Union[str, Path]]
- Caller can pass either a directory OR a complete file path
- Each function has a well-defined default filename
- Uses `_write_html_artifact()` with default_filename for resolution

See Section 20 of code_guidelines.md for output artifact standards.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from finance_ml.core.constants import PLOTLY_TEMPLATE, COLOR_PALETTE
from finance_ml.core.schema import PHASE93_FEATURE_CATEGORIES
from .base import _write_html_artifact

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS (Section 2 Guidelines)
# =============================================================================

# Default filenames for each dashboard type (Section 20 - Output Artifact Standards)
DEFAULT_FILENAMES = {
    "dividend_reliability": "dividend_reliability_dashboard.html",
    "dividend_sustainability": "dividend_sustainability_scorecard.html",
    "analyst_recommendation": "analyst_recommendation_heatmap.html",
    "analyst_consensus": "analyst_consensus_dashboard.html",
    "leverage_liquidity": "leverage_liquidity_heatmap.html",
    "market_movers": "market_movers_dashboard.html",
    "price_target": "price_target_analytics.html",
    "employee_productivity": "employee_productivity_dashboard.html",
    "earnings_quality": "earnings_quality_dashboard.html",
    "revenue_forecast": "revenue_forecast_momentum.html",
}

DIVIDEND_SUSTAINABILITY_THRESHOLDS = {
    "excellent": 90,
    "good": 75,
    "moderate": 60,
    "poor": 40,
}

# Schema-driven column mappings derived from PHASE93_FEATURE_CATEGORIES
DIVIDEND_RELIABILITY_COLS = {
    "reliability": "dividend_reliability_score",
    "streak": "dividend_streak",
    "yield_stability": "dividend_yield_stability",
    "fcf_coverage": "fcf_dividend_coverage",
    "payout_consistency": "payout_consistency_score",
    "sustainable_flag": "sustainable_dividend_flag",
}

LEVERAGE_LIQUIDITY_COLS = {
    "Current Ratio": "current_ratio",
    "Quick Ratio": "quick_ratio",
    "Cash Ratio": "cash_ratio",
    "Debt/Assets": "debt_to_assets",
    "Debt/Equity": "debt_to_equity",
    "Interest Coverage": "interest_coverage",
}

MARKET_MOVERS_COLS = {
    "1D Change": "one_day_pct",
    "1M Change": "price_chg_pct_1m",
    "3M Change": "price_chg_pct_3m",
}

ANALYST_RATING_COLS = {
    "Strong Buy": "num_strong_buys_ratings",
    "Buy": "num_buys_ratings",
    "Hold": "num_hold_ratings",
    "Sell": "num_sell_ratings",
    "Strong Sell": "num_strong_sell_ratings",
}


# =============================================================================
# PRIVATE HELPER FUNCTIONS
# =============================================================================


def _get_available_columns(
    df: pd.DataFrame,
    column_mapping: Dict[str, str],
) -> Dict[str, str]:
    """Filter column mapping to only include columns present in DataFrame.

    Args:
        df: DataFrame to check for column presence.
        column_mapping: Dict mapping display names to column names.

    Returns:
        Filtered dict with only available columns.
    """
    return {k: v for k, v in column_mapping.items() if v in df.columns}


def _get_phase93_category_columns(
    df: pd.DataFrame,
    category: str,
) -> List[str]:
    """Get available columns from a Phase 9.3 feature category.

    Args:
        df: DataFrame to check for column presence.
        category: Category name from PHASE93_FEATURE_CATEGORIES.

    Returns:
        List of column names present in both category and DataFrame.
    """
    category_cols = PHASE93_FEATURE_CATEGORIES.get(category, [])
    return [col for col in category_cols if col in df.columns]


def _create_empty_figure_with_message(
    message: str,
    output_path: Optional[Union[str, Path]] = None,
    default_filename: str = "empty_dashboard.html",
) -> go.Figure:
    """Create an empty figure with a centered message annotation.

    Args:
        message: Message to display.
        output_path: Optional path to save HTML (directory or file).
        default_filename: Filename to use if output_path is a directory.

    Returns:
        Empty figure with annotation.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    _write_html_artifact(fig, output_path, default_filename=default_filename)
    return fig


def _prepare_numeric_columns(
    df: pd.DataFrame,
    columns: Dict[str, str],
) -> pd.DataFrame:
    """Copy DataFrame and convert specified columns to numeric.

    Args:
        df: Source DataFrame.
        columns: Dict mapping keys to column names to convert.

    Returns:
        Copy of DataFrame with numeric columns.
    """
    df_local = df.copy()
    for col in columns.values():
        if col in df_local.columns:
            df_local[col] = pd.to_numeric(df_local[col], errors="coerce")
    return df_local


def _get_top_sectors(df: pd.DataFrame, top_n: int) -> pd.Index:
    """Get top N sectors by count from DataFrame.

    Args:
        df: DataFrame with 'sector' column.
        top_n: Number of top sectors to return.

    Returns:
        Index of top sector names.
    """
    return df["sector"].value_counts().head(int(top_n)).index


# =============================================================================
# PUBLIC DASHBOARD FUNCTIONS
# =============================================================================


def create_dividend_reliability_dashboard(
    df: pd.DataFrame,
    top_n_sectors: int = 12,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create comprehensive dividend reliability visualization.

    Leverages dividend reliability metrics from PHASE93_FEATURE_CATEGORIES:
    - dividend_reliability_score, dividend_streak (correlation: 0.94)
    - dividend_yield_stability, fcf_dividend_coverage
    - payout_consistency_score, sustainable_dividend_flag

    Args:
        df: DataFrame with dividend metrics.
        top_n_sectors: Number of sectors to display.
        output_path: Optional path to save HTML. Can be:
            - Directory path: filename auto-appended
            - Full file path: used as-is
            - None: no file output

    Returns:
        go.Figure: Multi-panel dividend reliability dashboard.
    """
    default_filename = DEFAULT_FILENAMES["dividend_reliability"]
    available = _get_available_columns(df, DIVIDEND_RELIABILITY_COLS)

    if len(available) < 2 or "sector" not in df.columns:
        return _create_empty_figure_with_message(
            "Insufficient dividend reliability columns",
            output_path,
            default_filename,
        )

    df_local = _prepare_numeric_columns(df, available)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Dividend Reliability Score Distribution",
            "Reliability vs Streak (High Correlation)",
            "Sector Dividend Sustainability",
            "FCF Coverage Distribution",
        ],
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "histogram"}],
        ],
    )

    # Panel 1: Reliability score distribution
    if "reliability" in available:
        fig.add_trace(
            go.Histogram(
                x=df_local[available["reliability"]].dropna(),
                nbinsx=30,
                marker_color=COLOR_PALETTE["info"],
                name="Reliability Score",
            ),
            row=1,
            col=1,
        )

    # Panel 2: Reliability vs Streak scatter (exploits 0.94 correlation)
    if "reliability" in available and "streak" in available:
        fig.add_trace(
            go.Scatter(
                x=df_local[available["streak"]],
                y=df_local[available["reliability"]],
                mode="markers",
                marker=dict(
                    size=6,
                    color=df_local.get("sustainable_dividend_flag", 0),
                    colorscale="RdYlGn",
                    opacity=0.6,
                ),
                text=df_local.get("ticker"),
                name="Reliability vs Streak",
                hovertemplate="<b>%{text}</b><br>Streak: %{x}<br>Score: %{y:.1f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    # Panel 3: Sector sustainability summary
    top_sectors = _get_top_sectors(df_local, top_n_sectors)
    if "sustainable_flag" in available:
        sector_pct = (
            df_local[df_local["sector"].isin(top_sectors)]
            .groupby("sector")[available["sustainable_flag"]]
            .mean()
            * 100
        ).sort_values(ascending=True)

        colors = [
            COLOR_PALETTE["success"] if x >= 50 else COLOR_PALETTE["danger"] for x in sector_pct
        ]
        fig.add_trace(
            go.Bar(
                x=sector_pct.values,
                y=sector_pct.index,
                orientation="h",
                marker_color=colors,
                name="% Sustainable",
            ),
            row=2,
            col=1,
        )

    # Panel 4: FCF coverage distribution
    if "fcf_coverage" in available:
        coverage = df_local[available["fcf_coverage"]].dropna()
        coverage_clipped = coverage.clip(-10, 20)
        fig.add_trace(
            go.Histogram(
                x=coverage_clipped,
                nbinsx=40,
                marker_color=COLOR_PALETTE["warning"],
                name="FCF Coverage",
            ),
            row=2,
            col=2,
        )
        fig.add_vline(x=1.5, line_dash="dash", line_color="white", row=2, col=2)

    dividend_feature_count = len(PHASE93_FEATURE_CATEGORIES.get("Dividend Reliability", []))

    fig.update_layout(
        title=f"<b>Dividend Reliability Dashboard</b><br><sup>Based on {dividend_feature_count} reliability metrics</sup>",
        template=PLOTLY_TEMPLATE,
        height=800,
        showlegend=False,
    )

    _write_html_artifact(fig, output_path, default_filename=default_filename)
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
        output_path: Optional path to save HTML (directory or file).

    Returns:
        go.Figure: Plotly figure.
    """
    default_filename = DEFAULT_FILENAMES["analyst_recommendation"]
    available_ratings = _get_available_columns(df, ANALYST_RATING_COLS)

    if not available_ratings or "sector" not in df.columns:
        return _create_empty_figure_with_message(
            "Required analyst rating columns not found",
            output_path,
            default_filename,
        )

    df_local = df.copy()
    top_sectors = _get_top_sectors(df_local, top_n_sectors)

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

    _write_html_artifact(fig, output_path, default_filename=default_filename)
    return fig


def create_leverage_liquidity_heatmap(
    df: pd.DataFrame,
    top_n_sectors: int = 12,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create sector-level leverage and liquidity risk heatmap.

    Uses columns from PHASE93_FEATURE_CATEGORIES["Leverage & Liquidity"]:
    - cash_ratio, current_ratio, quick_ratio
    - debt_to_assets, debt_to_equity, interest_coverage

    Args:
        df: DataFrame with leverage/liquidity metrics.
        top_n_sectors: Number of sectors to display.
        output_path: Optional path to save HTML (directory or file).

    Returns:
        go.Figure: Heatmap of leverage/liquidity metrics by sector.
    """
    default_filename = DEFAULT_FILENAMES["leverage_liquidity"]
    available = _get_available_columns(df, LEVERAGE_LIQUIDITY_COLS)

    if len(available) < 2 or "sector" not in df.columns:
        return _create_empty_figure_with_message(
            "Insufficient leverage/liquidity columns",
            output_path,
            default_filename,
        )

    df_local = _prepare_numeric_columns(df, available)
    top_sectors = _get_top_sectors(df_local, top_n_sectors)

    heatmap_data: List[Dict[str, float]] = []
    for sector in top_sectors:
        sector_df = df_local[df_local["sector"] == sector]
        row: Dict[str, float] = {"Sector": str(sector)[:20]}
        for metric_name, col in available.items():
            median_val = sector_df[col].median()
            global_median = df_local[col].median()
            global_std = df_local[col].std()
            if global_std and global_std > 0:
                row[metric_name] = (median_val - global_median) / global_std
            else:
                row[metric_name] = 0.0
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data).set_index("Sector").fillna(0)

    fig = px.imshow(
        heatmap_df,
        labels=dict(x="Metric", y="Sector", color="Z-Score vs Global"),
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        aspect="auto",
        text_auto=".2f",
        title="<b>Leverage & Liquidity Risk Heatmap</b><br><sup>Sector medians vs global (z-score)</sup>",
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        font=dict(family="Arial, sans-serif", size=12),
    )

    _write_html_artifact(fig, output_path, default_filename=default_filename)
    return fig


def create_market_movers_dashboard(
    df: pd.DataFrame,
    top_n: int = 10,
    time_period: str = "1D",
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> go.Figure:
    """Create dashboard showing top market movers (gainers and losers).

    Uses price change columns from COLUMN_SCHEMA with role='percentage':
    - one_day_pct (1-Day %)
    - price_chg_pct_1m (Price Chg. % 1M)
    - price_chg_pct_3m (Price Chg. % 3M)

    Args:
        df: DataFrame with price change metrics and ticker/sector columns.
        top_n: Number of top gainers/losers to display per panel.
        time_period: Time period for ranking - "1D", "1M", or "3M".
        output_path: Optional path to save HTML (directory or file).
        **kwargs: Additional arguments for backward compatibility.

    Returns:
        go.Figure: Two-panel dashboard with gainers (left) and losers (right).
    """
    default_filename = DEFAULT_FILENAMES["market_movers"]
    period_col_map = {
        "1D": "one_day_pct",
        "1M": "price_chg_pct_1m",
        "3M": "price_chg_pct_3m",
    }

    change_col = period_col_map.get(time_period.upper(), "one_day_pct")

    if change_col not in df.columns:
        if time_period.upper() == "1D" and "1_day_pct" in df.columns:
            change_col = "1_day_pct"
        else:
            return _create_empty_figure_with_message(
                f"Required column '{change_col}' not found for {time_period} movers",
                output_path,
                default_filename,
            )

    ticker_col = "ticker" if "ticker" in df.columns else None
    if ticker_col is None:
        return _create_empty_figure_with_message(
            "Required 'ticker' column not found",
            output_path,
            default_filename,
        )

    df_local = df.copy()
    df_local[change_col] = pd.to_numeric(df_local[change_col], errors="coerce")
    df_valid = df_local.dropna(subset=[change_col, ticker_col])

    if len(df_valid) < 2:
        return _create_empty_figure_with_message(
            "Insufficient data for market movers",
            output_path,
            default_filename,
        )

    df_sorted = df_valid.sort_values(change_col, ascending=False)
    top_gainers = df_sorted.head(top_n).copy()
    top_losers = df_sorted.tail(top_n).sort_values(change_col, ascending=True).copy()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Top {top_n} Gainers ({time_period})",
            f"Top {top_n} Losers ({time_period})",
        ],
        horizontal_spacing=0.15,
    )

    fig.add_trace(
        go.Bar(
            x=top_gainers[change_col],
            y=top_gainers[ticker_col],
            orientation="h",
            marker_color=COLOR_PALETTE["success"],
            text=[f"{x:+.2f}%" for x in top_gainers[change_col]],
            textposition="outside",
            name="Gainers",
            hovertemplate="<b>%{y}</b><br>Change: %{x:+.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=top_losers[change_col],
            y=top_losers[ticker_col],
            orientation="h",
            marker_color=COLOR_PALETTE["danger"],
            text=[f"{x:+.2f}%" for x in top_losers[change_col]],
            textposition="outside",
            name="Losers",
            hovertemplate="<b>%{y}</b><br>Change: %{x:+.2f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )

    period_labels = {"1D": "1-Day", "1M": "1-Month", "3M": "3-Month"}
    period_label = period_labels.get(time_period.upper(), time_period)

    fig.update_layout(
        title=f"<b>Market Movers Dashboard</b><br><sup>{period_label} Price Changes</sup>",
        template=PLOTLY_TEMPLATE,
        height=max(400, top_n * 35 + 100),
        showlegend=False,
    )

    fig.update_xaxes(title_text="Change %", row=1, col=1)
    fig.update_xaxes(title_text="Change %", row=1, col=2)
    fig.update_yaxes(categoryorder="total ascending", row=1, col=1)
    fig.update_yaxes(categoryorder="total descending", row=1, col=2)

    _write_html_artifact(fig, output_path, default_filename=default_filename)
    return fig


def create_price_target_analytics(
    df: pd.DataFrame,
    top_n_sectors: int = 12,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create comprehensive price target analytics visualization.

    Business Objective: Assess analyst consensus reliability and
    identify stocks with high conviction (tight spreads) vs. high uncertainty.

    Args:
        df: DataFrame with price target columns.
        top_n_sectors: Number of sectors to analyze.
        output_path: Optional path to save HTML (directory or file).

    Returns:
        go.Figure: Interactive price target dashboard.
    """
    default_filename = DEFAULT_FILENAMES["price_target"]
    required_cols = ["ticker", "sector", "last_price", "price_target"]

    if not all(c in df.columns for c in required_cols):
        return _create_empty_figure_with_message(
            "Required price target columns not found",
            output_path,
            default_filename,
        )

    df_local = df.copy()
    if "price_target_high" in df_local.columns and "price_target_low" in df_local.columns:
        df_local["target_spread"] = (
            (df_local["price_target_high"] - df_local["price_target_low"])
            / df_local["last_price"]
            * 100
        )

    top_sectors = df_local["sector"].value_counts().head(top_n_sectors).index
    sector_stats = []

    for sector in top_sectors:
        sector_df = df_local[df_local["sector"] == sector]

        if "target_vs_price" in sector_df.columns:
            upside = sector_df["target_vs_price"].dropna()
        else:
            upside = (
                (sector_df["price_target"] - sector_df["last_price"])
                / sector_df["last_price"]
                * 100
            )
            upside = upside.replace([np.inf, -np.inf], np.nan).dropna()

        if len(upside) >= 1:
            sector_stats.append(
                {
                    "sector": str(sector)[:20],
                    "mean_upside": upside.mean(),
                    "median_upside": upside.median(),
                    "q25_upside": upside.quantile(0.25),
                    "q75_upside": upside.quantile(0.75),
                    "count": len(upside),
                    "mean_spread": (
                        sector_df["target_spread"].mean()
                        if "target_spread" in sector_df.columns
                        else np.nan
                    ),
                }
            )

    if not sector_stats:
        return _create_empty_figure_with_message(
            "Insufficient data for price target analytics",
            output_path,
            default_filename,
        )

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
                array=stats_df["q75_upside"] - stats_df["mean_upside"],
                arrayminus=stats_df["mean_upside"] - stats_df["q25_upside"],
                color="rgba(255,255,255,0.3)",
            ),
            name="Mean Upside",
            hovertemplate="<b>%{y}</b><br>Mean: %{x:.1f}%<br>Q25-Q75: %{customdata[0]:.1f}% - %{customdata[1]:.1f}%<extra></extra>",
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
        all_upside = (
            (df_local["price_target"] - df_local["last_price"]) / df_local["last_price"] * 100
        )
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
    )

    fig.update_xaxes(title_text="Upside (%)", row=1, col=1)
    fig.update_xaxes(title_text="Spread (%)", row=1, col=2)
    fig.update_xaxes(title_text="Upside (%)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Confidence Score", row=2, col=2)

    _write_html_artifact(fig, output_path, default_filename=default_filename)
    return fig


def create_dividend_sustainability_scorecard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create dividend sustainability scorecard.

    Args:
        df: DataFrame with dividend metrics.
        output_path: Optional path to save HTML (directory or file).

    Returns:
        go.Figure: Dividend sustainability scorecard.
    """
    default_filename = DEFAULT_FILENAMES["dividend_sustainability"]
    return _create_empty_figure_with_message(
        "Dividend sustainability scorecard not implemented",
        output_path,
        default_filename,
    )


def create_employee_productivity_dashboard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create employee productivity dashboard.

    Args:
        df: DataFrame with employee productivity metrics.
        output_path: Optional path to save HTML (directory or file).

    Returns:
        go.Figure: Employee productivity dashboard.
    """
    default_filename = DEFAULT_FILENAMES["employee_productivity"]
    return _create_empty_figure_with_message(
        "Employee productivity dashboard not implemented",
        output_path,
        default_filename,
    )


def create_analyst_consensus_dashboard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create analyst consensus dashboard.

    Args:
        df: DataFrame with analyst consensus data.
        output_path: Optional path to save HTML (directory or file).

    Returns:
        go.Figure: Analyst consensus dashboard.
    """
    default_filename = DEFAULT_FILENAMES["analyst_consensus"]
    return _create_empty_figure_with_message(
        "Analyst consensus dashboard not implemented",
        output_path,
        default_filename,
    )


def create_earnings_quality_dashboard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create earnings quality dashboard.

    Args:
        df: DataFrame with earnings quality metrics.
        output_path: Optional path to save HTML (directory or file).

    Returns:
        go.Figure: Earnings quality dashboard.
    """
    default_filename = DEFAULT_FILENAMES["earnings_quality"]
    return _create_empty_figure_with_message(
        "Earnings quality dashboard not implemented",
        output_path,
        default_filename,
    )


def create_revenue_forecast_momentum_chart(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create revenue forecast momentum chart.

    Args:
        df: DataFrame with revenue forecast data.
        output_path: Optional path to save HTML (directory or file).

    Returns:
        go.Figure: Revenue forecast momentum chart.
    """
    default_filename = DEFAULT_FILENAMES["revenue_forecast"]
    return _create_empty_figure_with_message(
        "Revenue forecast momentum chart not implemented",
        output_path,
        default_filename,
    )
