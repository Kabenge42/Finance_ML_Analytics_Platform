from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .constants import COLOR_PALETTE, FONT_FAMILY, FONT_SIZES
from .utils import create_empty_state_figure


def _target_vs_price_scatter(
    df: pd.DataFrame,
    use_log_scale: bool = True,
    plotly_layout_defaults: Optional[dict] = None,
) -> go.Figure:
    """Create scatter plot of price target vs last price with optional log scale.

    Styling aligned with code_guidelines.md Section 17.1-17.2.
    """
    if df is None or df.empty:
        return create_empty_state_figure(
            "Target vs Price", "No data available", plotly_layout_defaults
        )

    if "last_price" not in df.columns or "price_target" not in df.columns:
        return create_empty_state_figure(
            "Target vs Price",
            "Missing required columns: last_price, price_target",
            plotly_layout_defaults,
        )

    # Filter valid data
    plot_df = df[
        (df["last_price"].notna())
        & (df["price_target"].notna())
        & (df["last_price"] > 0)
        & (df["price_target"] > 0)
    ].copy()

    if plot_df.empty:
        return create_empty_state_figure(
            "Target vs Price",
            "No valid price data after filtering",
            plotly_layout_defaults,
        )

    # Include detailed hover information (code_guidelines.md Section 17.1)
    hover_cols = [
        c
        for c in [
            "ticker",
            "name",
            "sector",
            "region",
            "country",
            "industry",
            "exchange",
            "market_cap",
        ]
        if c in plot_df.columns
    ]

    # Use log scale for better visibility across price ranges
    title = "Price Target vs Last Price" + (" (Log Scale)" if use_log_scale else "")

    fig = px.scatter(
        plot_df,
        x="last_price",
        y="price_target",
        color="sector" if "sector" in plot_df.columns else None,
        hover_name="ticker" if "ticker" in plot_df.columns else None,
        hover_data=hover_cols,
        log_x=use_log_scale,
        log_y=use_log_scale,
        title=title,
        template="plotly_dark",
    )

    # Add 45-degree line (Price = Target)
    min_val = min(plot_df["last_price"].min(), plot_df["price_target"].min())
    max_val = max(plot_df["last_price"].max(), plot_df["price_target"].max())
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color=COLOR_PALETTE["neutral"], width=1, dash="dash"),
    )

    fig.update_layout(
        **(plotly_layout_defaults or {}),
        xaxis_title="Last Price",
        yaxis_title="Price Target",
    )
    return fig


def _market_cap_distribution(
    df: pd.DataFrame, plotly_layout_defaults: Optional[dict] = None
) -> go.Figure:
    """Create market cap distribution histogram by sector.

    Styling aligned with code_guidelines.md Section 17.1-17.2.
    """
    if df is None or df.empty or "market_cap" not in df.columns:
        return create_empty_state_figure(
            "Market Cap Distribution", "No data available", plotly_layout_defaults
        )

    plot_df = df[df["market_cap"].notna() & (df["market_cap"] > 0)].copy()
    if plot_df.empty:
        return create_empty_state_figure(
            "Market Cap Distribution", "No valid data", plotly_layout_defaults
        )

    fig = px.histogram(
        plot_df,
        x="market_cap",
        color="sector" if "sector" in plot_df.columns else None,
        marginal="box",
        title="Market Cap Distribution by Sector",
        log_x=True,
        template="plotly_dark",
        labels={"market_cap": "Market Cap ($M)"},
    )

    fig.update_layout(
        **(plotly_layout_defaults or {}),
        xaxis_title="Market Cap ($M) - Log Scale",
        yaxis_title="Count",
        bargap=0.1,
    )
    return fig
