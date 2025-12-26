"""portfolio.py - Dashboard widgets."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from finance_ml.core.constants import PLOTLY_TEMPLATE, COLOR_PALETTE
from .base import (
    resolve_reference_date, _write_html_artifact
)

logger = logging.getLogger(__name__)

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
    reference_date = resolve_reference_date(df, reference_date)

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

    momentum_cols = ["momentum_20d","price_acceleration_3m","price_momentum_1m", "volatility_1m", "rel_volume"]
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

