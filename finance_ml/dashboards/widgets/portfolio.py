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

# Add import for schema-driven column lists
from finance_ml.core.schema import COLUMN_SCHEMA, list_price_cols

logger = logging.getLogger(__name__)

# Add new constants per Section 2 guidelines
DIVIDEND_SUSTAINABILITY_THRESHOLDS = {
    "excellent": 90,
    "good": 75,
    "moderate": 60,
    "poor": 40,
}


def create_dividend_reliability_dashboard(
    df: pd.DataFrame,
    top_n_sectors: int = 12,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create comprehensive dividend reliability visualization.

    Leverages dividend reliability metrics from analytics:
    - dividend_reliability_score, dividend_streak (correlation: 0.94)
    - dividend_yield_stability, fcf_dividend_coverage
    - payout_consistency_score, sustainable_dividend_flag

    Args:
        df: DataFrame with dividend metrics.
        top_n_sectors: Number of sectors to display.
        output_path: Optional path to save HTML.

    Returns:
        go.Figure: Multi-panel dividend reliability dashboard.
    """
    # Schema-driven column detection
    dividend_cols = {
        "reliability": "dividend_reliability_score",
        "streak": "dividend_streak",
        "yield_stability": "dividend_yield_stability",
        "fcf_coverage": "fcf_dividend_coverage",
        "payout_consistency": "payout_consistency_score",
        "sustainable_flag": "sustainable_dividend_flag",
    }

    available = {k: v for k, v in dividend_cols.items() if v in df.columns}
    if len(available) < 2 or "sector" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient dividend reliability columns",
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
    for col in available.values():
        df_local[col] = pd.to_numeric(df_local[col], errors="coerce")

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
    top_sectors = df_local["sector"].value_counts().head(int(top_n_sectors)).index
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
        coverage_clipped = coverage.clip(-10, 20)  # Handle extremes
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

    fig.update_layout(
        title="<b>Dividend Reliability Dashboard</b><br><sup>Based on 9 reliability metrics</sup>",
        template=PLOTLY_TEMPLATE,
        height=800,
        showlegend=False,
    )

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
            showarrow=True,
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
            showarrow=True,
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
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate dividend sustainability scorecard with improved grade distribution.

    Uses percentile-based scoring for better granularity across all grades (A-F).
    Incorporates additional dividend-related metrics from schema.py.

    Args:
        df: DataFrame with dividend-paying stocks
        output_path: Optional path to save CSV output

    Returns:
        DataFrame with sustainability scores and grades
    """
    import pandas as pd
    import numpy as np

    scorecard = df[["ticker", "sector", "region"]].copy()

    # =========================================================================
    # 1. PAYOUT SCORE (0-25 points) - Percentile-based
    # =========================================================================
    # Optimal payout: 20-60% range scores highest, extremes penalized
    payout_col = "dividend_payout_ratio"
    if payout_col in df.columns:
        payout = pd.to_numeric(df[payout_col], errors="coerce").clip(-1, 5)  # Winsorize outliers

        # Score based on proximity to optimal range (30-50%)
        optimal_center = 0.40
        payout_distance = np.abs(payout - optimal_center)

        # Normalize: closer to optimal = higher score
        max_distance = payout_distance.quantile(0.95)
        payout_score = (1 - (payout_distance / max_distance).clip(0, 1)) * 25

        # Penalize negative payout (losses) and extreme high payout (>100%)
        payout_score = payout_score.where(payout >= 0, 0)
        payout_score = payout_score.where(payout <= 1.0, payout_score * 0.5)

        scorecard["payout_score"] = payout_score.fillna(0)
    else:
        scorecard["payout_score"] = 0

    # =========================================================================
    # 2. FCF COVERAGE SCORE (0-20 points) - Tiered thresholds
    # =========================================================================
    fcf_col = "fcf_dividend_coverage"
    if fcf_col in df.columns:
        fcf_cov = pd.to_numeric(df[fcf_col], errors="coerce").clip(-10, 50)

        # Tiered scoring: higher coverage = better sustainability
        fcf_score = pd.cut(
            fcf_cov,
            bins=[-np.inf, 0, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf],
            labels=[0, 4, 8, 12, 16, 18, 20],
        ).astype(float)

        scorecard["fcf_coverage_score"] = fcf_score.fillna(0)
    else:
        scorecard["fcf_coverage_score"] = 0

    # =========================================================================
    # 3. DIVIDEND GROWTH SCORE (0-15 points) - NEW: Uses growth metrics
    # =========================================================================
    growth_score = pd.Series(0.0, index=df.index)

    # 3-year dividend growth
    if "dividend_growth_3y" in df.columns:
        growth_3y = pd.to_numeric(df["dividend_growth_3y"], errors="coerce").clip(-50, 100)
        growth_score += (growth_3y.rank(pct=True) * 7.5).fillna(0)

    # 5-year dividend growth
    if "dividend_growth_5y" in df.columns:
        growth_5y = pd.to_numeric(df["dividend_growth_5y"], errors="coerce").clip(-50, 100)
        growth_score += (growth_5y.rank(pct=True) * 7.5).fillna(0)
    elif "div_yield_5yavgltm" in df.columns:
        # Fallback: use yield stability as proxy
        yield_5y = pd.to_numeric(df["div_yield_5yavgltm"], errors="coerce")
        yield_current = pd.to_numeric(
            df.get("div_yield_ltm", pd.Series(dtype=float)), errors="coerce"
        )

        # Positive yield trend = growth proxy
        yield_trend = (yield_current - yield_5y).clip(-0.1, 0.1)
        growth_score += (yield_trend.rank(pct=True) * 7.5).fillna(0)

    scorecard["div_growth_score"] = growth_score

    # =========================================================================
    # 4. DIVIDEND STREAK SCORE (0-15 points) - Years of consecutive dividends
    # =========================================================================
    streak_col = "dividend_streak"
    if streak_col in df.columns:
        streak = pd.to_numeric(df[streak_col], errors="coerce").fillna(0)

        # Tiered: 0-2 yrs = 0-3, 3-5 = 4-7, 6-10 = 8-11, 11-20 = 12-14, 20+ = 15
        streak_score = pd.cut(
            streak, bins=[-np.inf, 0, 2, 5, 10, 20, np.inf], labels=[0, 3, 7, 11, 14, 15]
        ).astype(float)

        scorecard["dividend_streak_score"] = streak_score.fillna(0)
    else:
        scorecard["dividend_streak_score"] = 0

    # =========================================================================
    # 5. BALANCE SHEET HEALTH SCORE (0-15 points) - NEW: Leverage metrics
    # =========================================================================
    balance_score = pd.Series(0.0, index=df.index)

    # Debt-to-Equity: lower is better for dividend safety
    if "debt_to_equity" in df.columns:
        dte = pd.to_numeric(df["debt_to_equity"], errors="coerce").clip(0, 10)
        # Invert: lower D/E = higher score
        dte_score = ((1 - dte.rank(pct=True)) * 5).fillna(2.5)
        balance_score += dte_score

    # Interest Coverage: higher is better
    if "interest_coverage" in df.columns:
        int_cov = pd.to_numeric(df["interest_coverage"], errors="coerce").clip(-5, 50)
        int_score = (int_cov.rank(pct=True) * 5).fillna(2.5)
        balance_score += int_score
    elif "ebit_ltm" in df.columns and "interest_expense_total_ltm" in df.columns:
        # Calculate if not present
        ebit = pd.to_numeric(df["ebit_ltm"], errors="coerce")
        interest = pd.to_numeric(df["interest_expense_total_ltm"], errors="coerce").abs()
        int_cov = (ebit / interest.replace(0, np.nan)).clip(-5, 50)
        int_score = (int_cov.rank(pct=True) * 5).fillna(2.5)
        balance_score += int_score

    # Current Ratio: healthy liquidity
    if "current_ratio" in df.columns:
        curr_ratio = pd.to_numeric(df["current_ratio"], errors="coerce").clip(0, 5)
        # Optimal around 1.5-2.0
        cr_optimal = np.abs(curr_ratio - 1.75)
        cr_score = ((1 - cr_optimal.rank(pct=True)) * 5).fillna(2.5)
        balance_score += cr_score

    scorecard["balance_sheet_score"] = balance_score

    # =========================================================================
    # 6. YIELD QUALITY SCORE (0-10 points) - NEW: Yield sustainability signals
    # =========================================================================
    yield_score = pd.Series(0.0, index=df.index)

    # Dividend yield stability
    if "dividend_yield_stability" in df.columns:
        stability = pd.to_numeric(df["dividend_yield_stability"], errors="coerce")
        yield_score += (stability.rank(pct=True) * 5).fillna(0)

    # Payout consistency score
    if "payout_consistency_score" in df.columns:
        consistency = pd.to_numeric(df["payout_consistency_score"], errors="coerce").clip(0, 2)
        yield_score += (consistency.rank(pct=True) * 5).fillna(0)
    elif "dividend_reliability_score" in df.columns:
        # Fallback to reliability score
        reliability = pd.to_numeric(df["dividend_reliability_score"], errors="coerce")
        yield_score += (reliability.rank(pct=True) * 5).fillna(0)

    scorecard["yield_quality_score"] = yield_score

    # =========================================================================
    # 7. TOTAL SHAREHOLDER RETURN BONUS (0-5 points) - NEW
    # =========================================================================
    tsr_score = pd.Series(0.0, index=df.index)

    # Include buyback yield for total shareholder return
    if "buyback_yield_ltm" in df.columns:
        buyback = pd.to_numeric(df["buyback_yield_ltm"], errors="coerce").clip(-0.1, 0.2)
        # Positive buybacks = bonus points
        tsr_score += (buyback.clip(0, None).rank(pct=True) * 2.5).fillna(0)

    # Total shareholder yield if available
    if "total_shareholder_yield" in df.columns:
        tsy = pd.to_numeric(df["total_shareholder_yield"], errors="coerce").clip(-0.1, 0.3)
        tsr_score += (tsy.rank(pct=True) * 2.5).fillna(0)

    scorecard["shareholder_return_score"] = tsr_score

    # =========================================================================
    # COMPOSITE SCORE CALCULATION (0-100)
    # =========================================================================
    score_columns = [
        "payout_score",  # 0-25
        "fcf_coverage_score",  # 0-20
        "div_growth_score",  # 0-15
        "dividend_streak_score",  # 0-15
        "balance_sheet_score",  # 0-15
        "yield_quality_score",  # 0-10 (NEW)
        "shareholder_return_score",  # 0-5 (NEW, total = 105, will normalize)
    ]

    # Sum all component scores
    total_score = scorecard[score_columns].sum(axis=1)

    # Normalize to 0-100 scale (max possible is 105)
    scorecard["dividend_sustainability_score"] = (total_score / 1.05).round(1)

    # =========================================================================
    # GRADE ASSIGNMENT - Percentile-based for better distribution
    # =========================================================================
    # Use percentile thresholds to ensure distribution across grades
    score_percentiles = scorecard["dividend_sustainability_score"].quantile(
        [0.1, 0.3, 0.5, 0.75, 0.9]
    )

    def assign_grade(score):
        if pd.isna(score) or score < score_percentiles[0.1]:
            return "F"
        elif score < score_percentiles[0.3]:
            return "D"
        elif score < score_percentiles[0.5]:
            return "C"
        elif score < score_percentiles[0.75]:
            return "B"
        else:
            return "A"

    # Alternative: Fixed thresholds tuned for typical distributions
    # Uncomment below if you prefer absolute thresholds over percentiles
    # def assign_grade(score):
    #     if pd.isna(score) or score < 25:
    #         return 'F'
    #     elif score < 40:
    #         return 'D'
    #     elif score < 55:
    #         return 'C'
    #     elif score < 70:
    #         return 'B'
    #     else:
    #         return 'A'

    scorecard["sustainability_grade"] = scorecard["dividend_sustainability_score"].apply(
        assign_grade
    )

    # =========================================================================
    # OUTPUT
    # =========================================================================
    if output_path:
        scorecard.to_csv(output_path, index=False)

    return scorecard


def create_employee_productivity_dashboard(df: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    """Analyze employee productivity metrics and trends."""

    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    sector_categories = df["sector"].astype("category") if "sector" in df.columns else None

    if "total_revenues_ltm" in df.columns and "full_time_employees_fq" in df.columns:
        revenue = pd.to_numeric(df["total_revenues_ltm"], errors="coerce").astype("Float64")
        employees = pd.to_numeric(df["full_time_employees_fq"], errors="coerce").astype("Float64")
        df["revenue_per_employee"] = (revenue / employees.replace(0, pd.NA)) / 1000

    if "total_assets_ltm" in df.columns and "full_time_employees_fq" in df.columns:
        assets = pd.to_numeric(df["total_assets_ltm"], errors="coerce").astype("Float64")
        employees = pd.to_numeric(df["full_time_employees_fq"], errors="coerce").astype("Float64")
        df["assets_per_employee"] = (assets / employees.replace(0, pd.NA)) / 1000

    emp_fq = "full_time_employees_fq"
    emp_1fy = "full_time_employees_1fy"
    if emp_fq in df.columns and emp_1fy in df.columns:
        current = pd.to_numeric(df[emp_fq], errors="coerce").astype("Float64")
        prior = pd.to_numeric(df[emp_1fy], errors="coerce").astype("Float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["employee_growth_yoy"] = (
                ((current - prior) / prior.replace(0, pd.NA) * 100)
                .replace([np.inf, -np.inf], pd.NA)
                .astype("Float64")
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


def create_leverage_liquidity_heatmap(
    df: pd.DataFrame,
    top_n_sectors: int = 12,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create sector-level leverage and liquidity risk heatmap.

    Uses highly correlated liquidity ratios (correlation ~0.9999):
    - cash_ratio, current_ratio, quick_ratio
    - debt_to_assets, debt_to_equity, equity_ratio

    Args:
        df: DataFrame with leverage/liquidity metrics.
        top_n_sectors: Number of sectors to display.
        output_path: Optional path to save HTML.

    Returns:
        go.Figure: Heatmap of leverage/liquidity metrics by sector.
    """
    leverage_cols = {
        "Current Ratio": "current_ratio",
        "Quick Ratio": "quick_ratio",
        "Cash Ratio": "cash_ratio",
        "Debt/Assets": "debt_to_assets",
        "Debt/Equity": "debt_to_equity",
        "Interest Coverage": "interest_coverage",
    }

    available = {k: v for k, v in leverage_cols.items() if v in df.columns}
    if len(available) < 2 or "sector" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient leverage/liquidity columns",
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
    for col in available.values():
        df_local[col] = pd.to_numeric(df_local[col], errors="coerce")

    top_sectors = df_local["sector"].value_counts().head(int(top_n_sectors)).index

    # Build sector median matrix
    heatmap_data: List[Dict[str, float]] = []
    for sector in top_sectors:
        sector_df = df_local[df_local["sector"] == sector]
        row: Dict[str, float] = {"Sector": str(sector)[:20]}
        for metric_name, col in available.items():
            median_val = sector_df[col].median()
            # Normalize to z-score for comparability
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

    _write_html_artifact(fig, output_path)
    return fig


def create_analyst_consensus_dashboard(
    df: pd.DataFrame,
    top_n: int = 30,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create analyst consensus strength visualization.

    Addresses redundant columns from analytics (correlation=1.0):
    - price_target_range ≡ price_target_spread_pct (use one)
    - target_price_upside_pct ≡ upside_potential (use one)

    Visualizes: consensus_strength vs analyst_conviction vs upside.

    Args:
        df: DataFrame with analyst sentiment metrics.
        top_n: Number of stocks to highlight.
        output_path: Optional path to save HTML.

    Returns:
        go.Figure: Analyst consensus dashboard.
    """
    # Use canonical names, avoid duplicates per analytics
    sentiment_cols = {
        "consensus": "consensus_strength",
        "conviction": "analyst_conviction",
        "upside": "target_price_upside_pct",  # NOT upside_potential (duplicate)
        "bullish": "analyst_bullish_pct",
        "spread": "price_target_spread_pct",  # NOT price_target_range (duplicate)
    }

    available = {k: v for k, v in sentiment_cols.items() if v in df.columns}
    if len(available) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient analyst sentiment columns",
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
    for col in available.values():
        df_local[col] = pd.to_numeric(df_local[col], errors="coerce")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Consensus Strength vs Conviction",
            "High-Conviction Bullish Stocks",
            "Upside vs Spread (Uncertainty)",
            "Consensus Distribution by Sector",
        ],
    )

    # Panel 1: Consensus vs Conviction scatter
    if "consensus" in available and "conviction" in available:
        fig.add_trace(
            go.Scatter(
                x=df_local[available["consensus"]],
                y=df_local[available["conviction"]],
                mode="markers",
                marker=dict(
                    size=6,
                    color=df_local.get(available.get("upside", ""), 0),
                    colorscale="RdYlGn",
                    cmin=-20,
                    cmax=50,
                    opacity=0.6,
                ),
                text=df_local.get("ticker"),
                hovertemplate="<b>%{text}</b><br>Consensus: %{x:.1f}<br>Conviction: %{y:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Panel 2: Top high-conviction bullish stocks
    if "conviction" in available and "bullish" in available:
        high_conviction = df_local[
            (df_local[available["conviction"]] > 70) & (df_local[available["bullish"]] > 70)
        ].nlargest(int(top_n), available["bullish"])

        if not high_conviction.empty:
            fig.add_trace(
                go.Bar(
                    x=high_conviction[available["bullish"]],
                    y=high_conviction["ticker"],
                    orientation="h",
                    marker_color=COLOR_PALETTE["success"],
                    name="Bullish %",
                ),
                row=1,
                col=2,
            )

    # Panel 3: Upside vs Spread (risk/reward)
    if "upside" in available and "spread" in available:
        fig.add_trace(
            go.Scatter(
                x=df_local[available["spread"]].clip(0, 100),
                y=df_local[available["upside"]].clip(-50, 100),
                mode="markers",
                marker=dict(size=5, color=COLOR_PALETTE["info"], opacity=0.5),
                text=df_local.get("ticker"),
                hovertemplate="<b>%{text}</b><br>Spread: %{x:.1f}%<br>Upside: %{y:.1f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Panel 4: Sector consensus box plot
    if "consensus" in available and "sector" in df_local.columns:
        for sector in df_local["sector"].value_counts().head(8).index:
            sector_data = df_local[df_local["sector"] == sector][available["consensus"]].dropna()
            fig.add_trace(
                go.Box(y=sector_data, name=str(sector)[:15], boxmean=True),
                row=2,
                col=2,
            )

    fig.update_layout(
        title="<b>Analyst Consensus Dashboard</b><br><sup>Sentiment metrics (deduplicated)</sup>",
        template=PLOTLY_TEMPLATE,
        height=800,
        showlegend=False,
    )

    _write_html_artifact(fig, output_path)
    return fig


def create_earnings_quality_dashboard(
    df: pd.DataFrame,
    top_n_concerns: int = 25,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create earnings quality warning dashboard.

    Based on analytics showing:
    - 50% of stocks have earnings_quality_warning_flag=True
    - High correlation (-0.999) between revision acceleration and surprise momentum

    Args:
        df: DataFrame with earnings quality metrics.
        top_n_concerns: Number of concerning stocks to highlight.
        output_path: Optional path to save HTML.

    Returns:
        go.Figure: Earnings quality dashboard.
    """
    quality_cols = {
        "warning_flag": "earnings_quality_warning_flag",
        "eps_surprise": "eps_surprise_pct",
        "revenue_surprise": "revenue_surprise_pct",
        "revision_accel": "estimate_revision_acceleration",
        "eps_quality_flag": "eps_quality_flag_ltm",
        "ebitda_adj_ratio": "ebitda_adjustment_ratio_ltm",
    }

    available = {k: v for k, v in quality_cols.items() if v in df.columns}
    if len(available) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient earnings quality columns",
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
    for col in available.values():
        if col != available.get("warning_flag") and col != available.get("eps_quality_flag"):
            df_local[col] = pd.to_numeric(df_local[col], errors="coerce")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Earnings Quality Warning Distribution",
            "EPS Surprise % Distribution",
            "Top Stocks with Quality Concerns",
            "EBITDA Adjustment Ratio by Sector",
        ],
    )

    # Panel 1: Warning flag bar chart
    if "warning_flag" in available:
        warning_counts = df_local[available["warning_flag"]].value_counts()
        no_warning = warning_counts.get(False, warning_counts.get(0, 0))
        warning = warning_counts.get(True, warning_counts.get(1, 0))
        fig.add_trace(
            go.Bar(
                x=["No Warning", "Warning"],
                y=[no_warning, warning],
                marker_color=[COLOR_PALETTE["success"], COLOR_PALETTE["danger"]],
                text=[no_warning, warning],
                textposition="auto",
            ),
            row=1,
            col=1,
        )

    # Panel 2: EPS surprise distribution
    if "eps_surprise" in available:
        surprise = df_local[available["eps_surprise"]].dropna().clip(-200, 200)
        fig.add_trace(
            go.Histogram(
                x=surprise,
                nbinsx=50,
                marker_color=COLOR_PALETTE["info"],
                name="EPS Surprise %",
            ),
            row=1,
            col=2,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white", row=1, col=2)

    # Panel 3: Top concerning stocks
    if "warning_flag" in available and "ebitda_adj_ratio" in available:
        concerning = df_local[df_local[available["warning_flag"]] == True].copy()
        if not concerning.empty:
            concerning["concern_score"] = concerning[available["ebitda_adj_ratio"]].abs()
            top_concerns = concerning.nlargest(int(top_n_concerns), "concern_score")

            fig.add_trace(
                go.Bar(
                    x=top_concerns["concern_score"].clip(0, 5),
                    y=top_concerns["ticker"],
                    orientation="h",
                    marker_color=COLOR_PALETTE["danger"],
                    name="Concern Score",
                ),
                row=2,
                col=1,
            )

    # Panel 4: Sector adjustment ratios
    if "ebitda_adj_ratio" in available and "sector" in df_local.columns:
        sector_adj = (
            df_local.groupby("sector")[available["ebitda_adj_ratio"]]
            .median()
            .sort_values()
            .tail(10)
        )
        colors = [
            COLOR_PALETTE["warning"] if abs(x - 1) > 0.2 else COLOR_PALETTE["success"]
            for x in sector_adj
        ]
        fig.add_trace(
            go.Bar(
                x=sector_adj.values,
                y=sector_adj.index,
                orientation="h",
                marker_color=colors,
            ),
            row=2,
            col=2,
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color="white", row=2, col=2)

    fig.update_layout(
        title="<b>Earnings Quality Dashboard</b><br><sup>~50% of stocks flagged for quality concerns</sup>",
        template=PLOTLY_TEMPLATE,
        height=800,
        showlegend=False,
    )

    _write_html_artifact(fig, output_path)
    return fig


def create_revenue_forecast_momentum_chart(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create revenue forecasting momentum visualization.

    Focuses on non-redundant metrics from analytics:
    - eps_est_avg_rev_pct_fy1e_1m (short-term momentum)
    - revenues_est_yoy_pct_fy1e (growth expectations)
    - Avoids 3m/6m columns due to 0.995 correlation

    Args:
        df: DataFrame with revenue forecast metrics.
        output_path: Optional path to save HTML.

    Returns:
        go.Figure: Revenue forecast momentum chart.
    """
    # Use non-redundant columns per analytics correlation analysis
    forecast_cols = {
        "rev_1w": "eps_est_avg_rev_pct_fy1e_1w",
        "rev_1m": "eps_est_avg_rev_pct_fy1e_1m",
        # Skip 3m/6m due to 0.995 correlation with each other
        "rev_1y": "eps_est_avg_rev_pct_fy1e_1y",
        "yoy_growth": "revenues_est_yoy_pct_fy1e",
        # Use only fy1e estimate (0.997 corr with ntm)
        "revenue_est": "revenues_est_avg_fy1e",
    }

    available = {k: v for k, v in forecast_cols.items() if v in df.columns}
    if len(available) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient revenue forecast columns",
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
    for col in available.values():
        df_local[col] = pd.to_numeric(df_local[col], errors="coerce")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "EPS Revision Momentum (1W vs 1M)",
            "Revenue Growth Expectations by Sector",
        ],
    )

    # Panel 1: Short-term revision momentum scatter
    if "rev_1w" in available and "rev_1m" in available:
        # Identify stocks with accelerating upgrades
        df_local["revision_acceleration"] = (
            df_local[available["rev_1m"]] - df_local[available["rev_1w"]]
        )

        fig.add_trace(
            go.Scatter(
                x=df_local[available["rev_1w"]] * 100,
                y=df_local[available["rev_1m"]] * 100,
                mode="markers",
                marker=dict(
                    size=6,
                    color=df_local["revision_acceleration"],
                    colorscale="RdYlGn",
                    cmin=-0.05,
                    cmax=0.05,
                    opacity=0.6,
                    colorbar=dict(title="Acceleration"),
                ),
                text=df_local.get("ticker"),
                hovertemplate="<b>%{text}</b><br>1W: %{x:.2f}%<br>1M: %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )
        # Add 45-degree reference line
        fig.add_trace(
            go.Scatter(
                x=[-5, 5],
                y=[-5, 5],
                mode="lines",
                line=dict(dash="dash", color="white", width=1),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Panel 2: Sector revenue growth expectations
    if "yoy_growth" in available and "sector" in df_local.columns:
        sector_growth = (
            df_local.groupby("sector")[available["yoy_growth"]].median().sort_values().tail(12)
        )
        colors = [
            (
                COLOR_PALETTE["success"]
                if x > 0.1
                else COLOR_PALETTE["warning"] if x > 0 else COLOR_PALETTE["danger"]
            )
            for x in sector_growth
        ]
        fig.add_trace(
            go.Bar(
                x=sector_growth.values * 100,
                y=sector_growth.index,
                orientation="h",
                marker_color=colors,
            ),
            row=1,
            col=2,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white", row=1, col=2)

    fig.update_layout(
        title="<b>Revenue Forecast Momentum</b><br><sup>Non-redundant revision metrics</sup>",
        template=PLOTLY_TEMPLATE,
        height=500,
        showlegend=False,
    )
    fig.update_xaxes(title_text="1W Revision %", row=1, col=1)
    fig.update_yaxes(title_text="1M Revision %", row=1, col=1)
    fig.update_xaxes(title_text="YoY Growth %", row=1, col=2)

    _write_html_artifact(fig, output_path)
    return fig
