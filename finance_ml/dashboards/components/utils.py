from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html

from .constants import COLOR_PALETTE, FONT_FAMILY, FONT_SIZES


def compute_surprise(
    actual: pd.Series,
    estimate: pd.Series,
    mode: Literal["pct", "abs"] = "pct",
    clip_bounds: Tuple[float, float] = (-100, 100),
) -> pd.Series:
    """Compute surprise (pct or abs) between actual and estimate.

    Args:
        actual: Actual values Series
        estimate: Estimate values Series
        mode: surprise mode ('pct' for percentage surprise, 'abs' for absolute)
        clip_bounds: Optional bounds to clip surprise values (default: -100% to 100%)

    Returns:
        Series of surprise values (uses Float64 nullable type)
    """
    actual_num = pd.to_numeric(actual, errors="coerce").astype("Float64")
    estimate_num = pd.to_numeric(estimate, errors="coerce").astype("Float64")

    if mode == "pct":
        # Use absolute estimate as denominator to avoid sign issues
        denom = estimate_num.abs().replace(0, pd.NA)
        surprise = ((actual_num - estimate_num) / denom) * 100
    else:
        surprise = actual_num - estimate_num

    # Replace inf with NA and clip
    surprise = surprise.replace([np.inf, -np.inf], pd.NA)
    if clip_bounds:
        surprise = surprise.clip(lower=clip_bounds[0], upper=clip_bounds[1])

    return surprise.astype("Float64")


def create_empty_state_figure(
    title: str,
    message: str = "No data available",
    plotly_layout_defaults: Optional[dict] = None,
) -> go.Figure:
    """Create standardized empty state figure per code_guidelines.md Section 17.2."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(
            family=FONT_FAMILY,
            size=FONT_SIZES["body"],
            color=COLOR_PALETTE["neutral"],
        ),
    )
    layout = plotly_layout_defaults.copy() if plotly_layout_defaults else {}
    layout["title"] = title
    fig.update_layout(**layout)
    return fig


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    context_name: str,
) -> Tuple[List[str], List[str]]:
    """Validate required columns exist in DataFrame."""
    if df is None or df.empty:
        return [], required_cols

    present = [c for c in required_cols if c in df.columns]
    missing = [c for c in required_cols if c not in df.columns]

    return present, missing


def create_missing_columns_warning(
    missing_cols: List[str],
    context_name: str,
) -> dbc.Alert:
    """Create standardized alert for missing columns."""
    return dbc.Alert(
        [
            html.H4("Missing Data", className="alert-heading"),
            html.P(
                f"The following columns required for {context_name} are missing from the data source:"
            ),
            html.Ul([html.Li(c) for c in missing_cols]),
            html.Hr(),
            html.P(
                "Please run the full ETL pipeline or check your data source to enable this view.",
                className="mb-0",
            ),
        ],
        color="warning",
        dismissable=True,
    )


def _coerce_list(value: Any) -> List[str]:
    """Coerce any input to a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _severity_style(severity: str) -> Dict[str, str]:
    """Return conditional style dict for alert severity.

    Uses COLOR_PALETTE colors for consistency (code_guidelines.md Section 17.1).
    """
    sev = str(severity).lower().strip()
    if sev == "high":
        return {"backgroundColor": COLOR_PALETTE["danger"], "color": "#ffffff"}
    if sev == "medium":
        return {"backgroundColor": COLOR_PALETTE["warning"], "color": "#000000"}
    if sev == "low":
        return {"backgroundColor": COLOR_PALETTE["info"], "color": "#ffffff"}
    return {}


def _alerts_to_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert alerts payload to DataTable rows."""
    alerts = payload.get("alerts", [])
    if not isinstance(alerts, list):
        return []

    rows: List[Dict[str, Any]] = []
    for a in alerts:
        if not isinstance(a, dict):
            continue
        rows.append(
            {
                "severity": a.get("severity", ""),
                "alert_type": a.get("alert_type", ""),
                "count": a.get("count", ""),
                "description": a.get("description", ""),
                "tickers": ", ".join([str(t) for t in (a.get("tickers") or [])]),
            }
        )
    return rows
