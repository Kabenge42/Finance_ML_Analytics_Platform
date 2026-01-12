from __future__ import annotations

import pandas as pd
import plotly.express as px

from .constants import COLOR_PALETTE, PLOTLY_TEMPLATE, PLOTLY_LAYOUT_DEFAULTS
from .utils import create_empty_state_figure


def create_earnings_events_chart(df: pd.DataFrame, days_window: int = 30):
    """Create dynamic earnings events timeline chart.

    Styling aligned with code_guidelines.md Section 17.1-17.2.

    Args:
        df: DataFrame with next_earnings column
        days_window: Number of days before/after today to include

    Returns:
        Plotly figure
    """
    if df is None or df.empty or "next_earnings" not in df.columns:
        return create_empty_state_figure(
            "Earnings Events Timeline", "Earnings data not available"
        )

    # Filter data - Use reference_date for temporal consistency
    ref_date = pd.Timestamp.now()
    df_work = df.copy()
    df_work["next_earnings"] = pd.to_datetime(df_work["next_earnings"], errors="coerce")
    df_work["days_to_earnings"] = (df_work["next_earnings"] - ref_date).dt.days

    # Filter to window
    mask = df_work["days_to_earnings"].notna() & (
        df_work["days_to_earnings"].abs() <= days_window
    )
    events_df = df_work[mask].copy()

    if events_df.empty:
        return create_empty_state_figure(
            "Earnings Events Timeline", f"No earnings events within ±{days_window} days"
        )

    # Create timeline chart
    events_df = events_df.sort_values("days_to_earnings")

    # Color by sector if available
    if "sector" in events_df.columns:
        color = events_df["sector"]
    else:
        color = None

    fig = px.scatter(
        events_df,
        x="days_to_earnings",
        y="ticker" if "ticker" in events_df.columns else events_df.index,
        color=color,
        hover_data=[
            c
            for c in ["ticker", "name", "sector", "next_earnings"]
            if c in events_df.columns
        ],
        title=f"Earnings Events Timeline (±{days_window} days)",
        template=PLOTLY_TEMPLATE,
        height=max(400, len(events_df) * 15),
        labels={
            "days_to_earnings": "Days to Earnings",
            "ticker": "Ticker",
            "sector": "Sector",
        },
    )

    # Add vertical line at today using COLOR_PALETTE
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color=COLOR_PALETTE["neutral"],
        annotation_text="Today",
        annotation_position="top",
    )

    # Apply standard layout configuration
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        xaxis_title="Days to Earnings (negative = past)",
        yaxis_title="Ticker",
    )

    return fig
