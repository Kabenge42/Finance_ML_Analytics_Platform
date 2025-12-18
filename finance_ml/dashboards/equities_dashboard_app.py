"""Equities Dashboard (Plotly Dash)

Run:
    python finance_ml/dashboards/equities_dashboard_app.py

Design goals:
- No heavy work at import time (use create_app()).
- Safe fallbacks when data sources / artifacts are missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
from flask import send_from_directory

from finance_ml.dashboards.earnings_widgets import (
    EarningsAlertConfig,
    create_analyst_recommendation_heatmap,
    create_category_comparison_chart,
    create_earnings_calendar_dashboard,
    create_earnings_metrics_chart,
    create_earnings_surprise_dashboard,
    create_market_movers_dashboard,
    create_price_target_analytics,
    generate_earnings_quality_alerts,
    get_category_metrics,
)
from finance_ml.ml_workflow.data.schema import PHASE93_FEATURE_INPUTS
from finance_ml.ml_workflow.preprocessing.etl import etl_with_features

DataSource = Literal["auto", "csv", "db"]

# Standard color palette (aligned with code_guidelines.md Section 17.1)
COLOR_PALETTE = {
    "primary": "#375a7f",
    "secondary": "#6c757d",
    "success": "#00bc8c",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#3498db",
    "neutral": "#adb5bd",
}

# Plotly template (aligned with code_guidelines.md Section 17.2)
PLOTLY_TEMPLATE = "plotly_dark"

# Apply template globally to all Plotly figures
px.defaults.template = PLOTLY_TEMPLATE

# Font configuration (aligned with code_guidelines.md Section 17.4)
FONT_FAMILY = "Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
FONT_SIZES = {
    "h1": 32,  # 2rem
    "h2": 24,  # 1.5rem
    "h3": 20,  # 1.25rem
    "body": 16,  # 1rem
    "caption": 14,  # 0.875rem
}

# Standard Plotly layout configuration
PLOTLY_LAYOUT_DEFAULTS = {
    "font": {"family": FONT_FAMILY, "size": FONT_SIZES["caption"]},
    "title_font_size": FONT_SIZES["h3"],
    "showlegend": True,
    "legend": {
        "orientation": "v",
        "yanchor": "top",
        "xanchor": "right",
        "x": 1.02,
        "y": 1,
    },
    "hovermode": "closest",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
}

# Standard DataTable styles (aligned with code_guidelines.md Section 17.3)
TABLE_STYLE_CELL = {
    "backgroundColor": "#111",
    "color": "#ffffff",
    "border": f"1px solid {COLOR_PALETTE['secondary']}",
    "fontFamily": FONT_FAMILY,
    "fontSize": f"{FONT_SIZES['caption']}px",
    "padding": "8px",
    "whiteSpace": "normal",
    "height": "auto",
    "minWidth": "80px",
}

TABLE_STYLE_HEADER = {
    "backgroundColor": COLOR_PALETTE["primary"],
    "fontWeight": "bold",
    "color": "#ffffff",
}

TABLE_STYLE_TABLE = {
    "overflowX": "auto",
    "maxHeight": "500px",
    "overflowY": "auto",
}

# Earnings calendar mode options
EARNINGS_MODE_OPTIONS = [
    {"label": "All Categories", "value": "all"},
    {"label": "Earnings Focus", "value": "earnings"},
    {"label": "Dividends Focus", "value": "dividends"},
    {"label": "Valuation", "value": "valuation"},
    {"label": "Profitability", "value": "profitability"},
    {"label": "Growth", "value": "growth"},
    {"label": "Momentum", "value": "momentum"},
    {"label": "Quality & Risk", "value": "quality_risk"},
]

# Default columns for the Data Explorer tab - always included in initial view
# (code_guidelines.md Section 8.1: Single Source of Truth for configuration constants)
DEFAULT_EXPLORER_COLUMNS = [
    "ticker",
    "name",
    "sector",
    "region",
    "last_price",
    "price_target",
    "market_cap",
]

PROJECT_ROOT = Path(__file__).parent.parent.parent
DASHBOARD_ROOT = PROJECT_ROOT / "outputs" / "dashboards" / "equities_dashboard"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV_EXPORT_PATH = DASHBOARD_ROOT / "equities_dash_df.csv"
DEFAULT_METADATA_PATH = DASHBOARD_ROOT / "metadata.json"
ARTIFACTS_DIR = DASHBOARD_ROOT / "artifacts"
ARTIFACTS_METADATA_PATH = DASHBOARD_ROOT / "artifacts_metadata.json"
DEFAULT_ALERTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "eda"
    / "earnings_analytics"
    / "earnings_quality_alerts.json"
)

# Logging setup
import logging

logger = logging.getLogger(__name__)


def _validate_explorer_columns(df: pd.DataFrame, source_label: str) -> None:
    """Log warnings for missing DEFAULT_EXPLORER_COLUMNS in loaded data.

    Args:
        df: Loaded DataFrame to validate
        source_label: Data source label for logging context
    """
    if df is None or df.empty:
        return

    missing_cols = [c for c in DEFAULT_EXPLORER_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning(
            f"Data source '{source_label}' is missing explorer columns: {missing_cols}. "
            "Some Data Explorer features may be limited."
        )
    # Specifically warn about 'name' column as it's important for display
    if "name" not in df.columns:
        logger.warning(
            f"Data source '{source_label}' is missing 'name' column. "
            "Stock names will not be displayed in the Data Explorer."
        )


def load_data_csv_first(
    *,
    data_dir: Optional[Path] = None,
    db_url: Optional[str] = None,
    feature_preset: str = "comprehensive",
    force_etl: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Load data preferring CSV export, falling back to ETL.

    Returns:
        Tuple of (DataFrame, source_label) where source_label is one of:
        'csv_export', 'etl_csv', 'etl_db'
    """
    resolved_data_dir = data_dir or DEFAULT_DATA_DIR
    resolved_db_url = db_url or os.getenv("DB_URL")

    # Fast path: load from exported CSV if it exists and is recent
    if not force_etl and DEFAULT_CSV_EXPORT_PATH.exists():
        try:
            df = pd.read_csv(DEFAULT_CSV_EXPORT_PATH)
            _validate_explorer_columns(df, "csv_export")
            return df, "csv_export"
        except Exception as e:
            logger.warning(f"Failed to load CSV export: {e}, falling back to ETL")

    # Slow path: run ETL pipeline
    try:
        source: Literal["csv", "db"] = "db" if resolved_db_url else "csv"
        # Phase 9.1-9.3: Unified ETL Pipeline (STANDARD Pattern)
        df, metrics = etl_with_features(
            source=source,
            data_dir=resolved_data_dir,
            db_url=resolved_db_url,
            feature_preset=feature_preset,
            return_metrics=True,
        )

        # Export to CSV for next time
        if not df.empty:
            try:
                export_equities_data(df)
            except Exception:
                pass  # Non-critical

        # Return the metrics summary as the source label for the status bar
        source_label = metrics.summary()
        _validate_explorer_columns(df, f"etl_{source}")
        return df, source_label
    except Exception as e:
        logger.error(f"ETL Pipeline failed: {e}")
        return pd.DataFrame(), "failed"


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    context_name: str,
) -> Tuple[List[str], List[str]]:
    """Validate required columns exist in DataFrame.

    Args:
        df: DataFrame to check
        required_cols: List of required column names
        context_name: Name of the context (for logging)

    Returns:
        Tuple of (present_cols, missing_cols)
    """
    if df is None or df.empty:
        return [], required_cols

    present = [c for c in required_cols if c in df.columns]
    missing = [c for c in required_cols if c not in df.columns]

    return present, missing


def create_missing_columns_warning(
    missing_cols: List[str],
    context_name: str,
) -> html.Div:
    """Create a warning panel for missing columns."""
    if not missing_cols:
        return html.Div()

    return html.Div(
        [
            html.H5(f"⚠️ Missing Columns for {context_name}", className="text-warning"),
            html.P(
                f"The following columns are unavailable: {', '.join(missing_cols[:10])}"
            ),
            html.P(
                "Some charts may be limited or unavailable.", className="text-muted"
            ),
        ],
        style={
            "padding": "10px",
            "backgroundColor": "#2d2d2d",
            "borderRadius": "5px",
            "marginBottom": "10px",
        },
    )


def compute_surprise(
    actual: pd.Series,
    estimate: pd.Series,
    mode: Literal["pct", "abs"] = "pct",
    clip_bounds: Tuple[float, float] = (-100, 100),
) -> pd.Series:
    """Compute earnings surprise with safe handling.

    Args:
        actual: Actual values
        estimate: Estimated values
        mode: 'pct' for percentage, 'abs' for absolute
        clip_bounds: Bounds to clip extreme values

    Returns:
        Series of surprise values
    """
    actual_num = pd.to_numeric(actual, errors="coerce")
    estimate_num = pd.to_numeric(estimate, errors="coerce")

    if mode == "pct":
        # Use absolute estimate as denominator to avoid sign issues
        denom = estimate_num.abs().replace(0, np.nan)
        surprise = ((actual_num - estimate_num) / denom) * 100
    else:
        surprise = actual_num - estimate_num

    # Replace inf with NaN and clip
    surprise = surprise.replace([np.inf, -np.inf], np.nan)
    if clip_bounds:
        surprise = surprise.clip(lower=clip_bounds[0], upper=clip_bounds[1])

    return surprise


def create_empty_state_figure(
    title: str,
    message: str = "No data available",
) -> go.Figure:
    """Create standardized empty state figure per code_guidelines.md Section 17.2.

    Args:
        title: Figure title
        message: Message to display in empty state

    Returns:
        Plotly Figure with empty state annotation
    """
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
    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, title=title)
    return fig


# Metric mappings for Est vs Actual tab
EST_ACTUAL_METRICS = {
    "EPS": {
        "actual": "eps_adj_ltm",
        "estimate": "eps_norm_est_avg_ntm",
        "adjusted": "eps_adj_ltm",
        "gaap": "net_eps_basic_ltm",
    },
    "Revenue": {
        "actual": "total_revenues_ltm",
        "estimate": "revenues_est_avg_ntm",
        "adjusted": None,
        "gaap": "total_revenues_ltm",
    },
    "EBITDA": {
        "actual": "ebitda_ltm",
        "estimate": "ebitda_est_avg_ntm",
        "adjusted": "ebitda_adj_ltm",
        "gaap": "ebitda_ltm",
    },
}


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def export_equities_data(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Export equities data to CSV with metadata.

    Args:
        df: DataFrame to export
        output_path: Path for CSV file (defaults to DEFAULT_CSV_EXPORT_PATH)
        metadata_path: Path for metadata JSON (defaults to DEFAULT_METADATA_PATH)

    Returns:
        Dict with export metadata
    """
    if output_path is None:
        output_path = DEFAULT_CSV_EXPORT_PATH
    if metadata_path is None:
        metadata_path = DEFAULT_METADATA_PATH

    # Ensure directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Export CSV
    df.to_csv(output_path, index=False)

    # Generate metadata
    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "file_path": str(output_path),
        "file_size_mb": output_path.stat().st_size / (1024 * 1024)
        if output_path.exists()
        else 0,
    }

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def generate_dashboard_artifacts(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate all dashboard artifacts using earnings_widgets.

    Args:
        df: Source DataFrame (equities_dash_df)
        output_dir: Directory for artifacts (defaults to ARTIFACTS_DIR)
        metadata_path: Path for artifacts metadata JSON

    Returns:
        Dict with artifact generation metadata
    """
    if output_dir is None:
        output_dir = ARTIFACTS_DIR
    if metadata_path is None:
        metadata_path = ARTIFACTS_METADATA_PATH

    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    timestamp = pd.Timestamp.now().isoformat()

    try:
        # Generate main dashboard widgets
        artifacts["earnings_surprise"] = {
            "file": "earnings_surprise_dashboard.html",
            "title": "Earnings Surprise Analysis",
            "section": "earnings",
        }
        create_earnings_surprise_dashboard(
            df, output_path=output_dir / artifacts["earnings_surprise"]["file"]
        )

        artifacts["analyst_heatmap"] = {
            "file": "analyst_recommendation_heatmap.html",
            "title": "Analyst Recommendations by Sector",
            "section": "earnings",
        }
        create_analyst_recommendation_heatmap(
            df, output_path=output_dir / artifacts["analyst_heatmap"]["file"]
        )

        artifacts["market_movers"] = {
            "file": "market_movers_dashboard.html",
            "title": "Market Movers Around Earnings",
            "section": "earnings",
        }
        create_market_movers_dashboard(
            df, output_path=output_dir / artifacts["market_movers"]["file"]
        )

        artifacts["price_target_analytics"] = {
            "file": "price_target_analytics.html",
            "title": "Price Target Analytics",
            "section": "analytics",
        }
        create_price_target_analytics(
            df, output_path=output_dir / artifacts["price_target_analytics"]["file"]
        )

        # Generate Phase 9.3 category charts
        phase93_categories = [
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

        for category in phase93_categories:
            key = f"earnings_metrics_{category}"
            artifacts[key] = {
                "file": f"earnings_metrics_{category}.html",
                "title": f"Earnings Metrics: {category.replace('_', ' ').title()}",
                "section": "phase93",
            }
            create_earnings_metrics_chart(
                df,
                metric_category=category,
                output_path=output_dir / artifacts[key]["file"],
            )

        # Generate category comparison chart
        artifacts["category_comparison"] = {
            "file": "phase93_category_comparison.html",
            "title": "Phase 9.3 Category Comparison",
            "section": "phase93",
        }
        create_category_comparison_chart(
            df, output_path=output_dir / artifacts["category_comparison"]["file"]
        )

    except Exception as e:
        print(f"Warning: Error generating some artifacts: {e}")

    # Create metadata
    metadata = {
        "timestamp": timestamp,
        "total_stocks": len(df),
        "artifacts_dir": str(output_dir),
        "artifacts": artifacts,
        "generation_status": "completed",
    }

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_data(
    *,
    data_source: DataSource = "auto",
    data_dir: Optional[str | Path] = None,
    db_url: Optional[str] = None,
    feature_preset: str = "comprehensive",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load equities data using the unified ETL + features pipeline.

    - auto: try DB if DB_URL is provided, otherwise fall back to CSV.
    - csv: load from CSV region files under data_dir.
    - db: load from database (requires db_url or DB_URL env var).

    Returns an empty DataFrame on failures.
    """

    resolved_data_dir: Path = (
        Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    )
    resolved_db_url = db_url or os.getenv("DB_URL")

    def _etl(source: Literal["csv", "db"]) -> pd.DataFrame:
        result = etl_with_features(
            source=source,
            data_dir=resolved_data_dir,
            db_url=resolved_db_url,
            feature_preset=feature_preset,
            return_metrics=False,
        )
        return result

    try:
        if data_source == "db":
            if not resolved_db_url:
                return pd.DataFrame()
            df = _etl("db")
        elif data_source == "csv":
            df = _etl("csv")
        else:
            if resolved_db_url:
                try:
                    df = _etl("db")
                except Exception:
                    df = _etl("csv")
            else:
                df = _etl("csv")

        if limit is not None and limit > 0:
            return df.head(int(limit)).copy()
        return df
    except Exception:
        return pd.DataFrame()


def apply_filters(
    df: pd.DataFrame,
    *,
    sectors: Optional[Iterable[str]] = None,
    regions: Optional[Iterable[str]] = None,
    countries: Optional[Iterable[str]] = None,
    trading_countries: Optional[Iterable[str]] = None,
    industries: Optional[Iterable[str]] = None,
    exchanges: Optional[Iterable[str]] = None,
    style_classes: Optional[Iterable[str]] = None,
    size_classes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Filter helper with graceful missing-column behavior."""

    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])

    filtered = df
    filters: List[Tuple[str, Optional[Iterable[str]]]] = [
        ("sector", sectors),
        ("region", regions),
        ("country", countries),
        ("trading_country", trading_countries),
        ("industry", industries),
        ("exchange", exchanges),
        ("style_class", style_classes),
        ("size_class", size_classes),
    ]

    for col, values in filters:
        values_list = list(values) if values is not None else []
        if not values_list:
            continue
        if col not in filtered.columns:
            continue
        filtered = filtered[filtered[col].isin(values_list)]

    return filtered


def load_alerts_payload(path: str | Path = DEFAULT_ALERTS_PATH) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _alerts_to_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def _monitoring_kpi_cards(df: pd.DataFrame) -> List[Any]:
    """Generate monitoring KPI cards.

    Styling aligned with code_guidelines.md Section 17.4.
    """
    cards = []

    def card(title: str, value: str, color: str = "primary") -> dbc.Card:
        return dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        title,
                        className="kpi-title",
                        style={
                            "fontSize": f"{FONT_SIZES['caption']}px",
                            "fontFamily": FONT_FAMILY,
                        },
                    ),
                    html.Div(
                        value,
                        className="kpi-value",
                        style={
                            "fontSize": f"{FONT_SIZES['h3']}px",
                            "fontWeight": "bold",
                            "fontFamily": FONT_FAMILY,
                        },
                    ),
                ]
            ),
            color=color,
            inverse=True,
            className="kpi-card",
            style={"minWidth": "150px"},
        )

    # 1. % Positive Revenue Growth
    if "total_revenues_cagr_5y_fy" in df.columns:
        growth = pd.to_numeric(df["total_revenues_cagr_5y_fy"], errors="coerce")
        pct_positive = (growth > 0).sum() / len(growth) * 100 if len(growth) > 0 else 0
        cards.append(
            card(
                "% Positive Rev Growth",
                f"{pct_positive:.1f}%",
                "success" if pct_positive > 50 else "warning",
            )
        )

    # 2. Median Net Margin
    if "net_income_margin_pct_ltm" in df.columns:
        margin = pd.to_numeric(df["net_income_margin_pct_ltm"], errors="coerce")
        median_margin = margin.median() if margin.notna().any() else 0
        cards.append(card("Median Net Margin", f"{median_margin:.1f}%", "info"))

    # 3. % Flagged by Alerts
    payload = load_alerts_payload(DEFAULT_ALERTS_PATH)
    alert_tickers = set()
    for a in payload.get("alerts", []):
        alert_tickers.update(a.get("tickers", []))
    if "ticker" in df.columns and len(df) > 0:
        pct_flagged = len(alert_tickers & set(df["ticker"])) / len(df) * 100
        cards.append(
            card(
                "% With Alerts",
                f"{pct_flagged:.1f}%",
                "danger" if pct_flagged > 20 else "secondary",
            )
        )

    # 4. Median EPS Revision (if available)
    rev_cols = [c for c in df.columns if "eps_est_avg_rev_pct" in c.lower()]
    if rev_cols:
        rev = pd.to_numeric(df[rev_cols[0]], errors="coerce")
        median_rev = rev.median() if rev.notna().any() else 0
        cards.append(
            card(
                "Median EPS Revision",
                f"{median_rev:+.1f}%",
                "success" if median_rev > 0 else "danger",
            )
        )

    return cards


def _safe_options(df: pd.DataFrame, col: str) -> List[Dict[str, str]]:
    if df is None or df.empty or col not in df.columns:
        return []
    values = sorted([v for v in df[col].dropna().astype(str).unique().tolist()])
    return [{"label": v, "value": v} for v in values]


def _kpi_cards(df: pd.DataFrame) -> List[Any]:
    """Generate overview KPI cards.

    Styling aligned with code_guidelines.md Section 17.4.
    """

    def _num(series: pd.Series) -> float:
        return float(pd.to_numeric(series, errors="coerce").dropna().mean())

    total = int(len(df))
    tickers = int(df["ticker"].nunique()) if "ticker" in df.columns else 0
    mean_upside = None
    if "price_target" in df.columns and "last_price" in df.columns:
        pt = pd.to_numeric(df["price_target"], errors="coerce")
        lp = pd.to_numeric(df["last_price"], errors="coerce")
        valid = pt.notna() & lp.notna() & (lp > 0)
        if valid.any():
            mean_upside = float((((pt[valid] - lp[valid]) / lp[valid]) * 100).mean())

    market_cap_mean = _num(df["market_cap"]) if "market_cap" in df.columns else None

    def card(title: str, value: str) -> dbc.Card:
        return dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        title,
                        className="kpi-title",
                        style={
                            "fontSize": f"{FONT_SIZES['caption']}px",
                            "fontFamily": FONT_FAMILY,
                        },
                    ),
                    html.Div(
                        value,
                        className="kpi-value",
                        style={
                            "fontSize": f"{FONT_SIZES['h3']}px",
                            "fontWeight": "bold",
                            "fontFamily": FONT_FAMILY,
                        },
                    ),
                ]
            ),
            className="kpi-card",
        )

    cards = [
        card("Rows", f"{total:,}"),
        card("Tickers", f"{tickers:,}"),
    ]
    if mean_upside is not None:
        cards.append(card("Mean Upside", f"{mean_upside:,.1f}%"))
    if market_cap_mean is not None and market_cap_mean == market_cap_mean:
        cards.append(card("Mean Market Cap", f"${market_cap_mean:,.0f}"))
    return cards


def _target_vs_price_scatter(df: pd.DataFrame, use_log_scale: bool = True):
    """Create scatter plot of price target vs last price with optional log scale.

    Styling aligned with code_guidelines.md Section 17.1-17.2.
    """
    if df is None or df.empty:
        return create_empty_state_figure("Target vs Price", "No data available")

    if "last_price" not in df.columns or "price_target" not in df.columns:
        return create_empty_state_figure(
            "Target vs Price", "Missing required columns: last_price, price_target"
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
            "Target vs Price", "No valid price data after filtering"
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
        hover_data=hover_cols,
        title=title,
        template=PLOTLY_TEMPLATE,
        log_x=use_log_scale,
        log_y=use_log_scale,
        labels={
            "last_price": "Last Price ($)",
            "price_target": "Price Target ($)",
            "sector": "Sector",
        },
    )

    # Add diagonal reference line (y=x) using COLOR_PALETTE
    if use_log_scale:
        min_val = min(plot_df["last_price"].min(), plot_df["price_target"].min())
        max_val = max(plot_df["last_price"].max(), plot_df["price_target"].max())
        fig.add_scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color=COLOR_PALETTE["neutral"], dash="dash", width=1),
            name="Current Price",
            showlegend=True,
        )

    # Apply standard layout configuration
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        xaxis_title="Last Price ($)" + (" - Log Scale" if use_log_scale else ""),
        yaxis_title="Price Target ($)" + (" - Log Scale" if use_log_scale else ""),
    )
    return fig


def _market_cap_distribution(df: pd.DataFrame):
    """Create market cap distribution with log scale.

    Styling aligned with code_guidelines.md Section 17.1-17.2.
    """
    if df is None or df.empty or "market_cap" not in df.columns:
        return create_empty_state_figure(
            "Market Cap Distribution", "Market cap data not available"
        )

    valid_df = df[df["market_cap"].notna() & (df["market_cap"] > 0)].copy()

    if valid_df.empty:
        return create_empty_state_figure(
            "Market Cap Distribution", "No valid market cap data"
        )

    # Use log10 for market cap
    valid_df["log_market_cap"] = np.log10(valid_df["market_cap"])

    fig = px.histogram(
        valid_df,
        x="log_market_cap",
        nbins=50,
        title="Market Cap Distribution (Log Scale)",
        template=PLOTLY_TEMPLATE,
        color="sector" if "sector" in valid_df.columns else None,
        labels={
            "log_market_cap": "Market Cap (Log₁₀ $)",
            "sector": "Sector",
        },
    )

    # Apply standard layout configuration, merging defaults with custom values
    layout_config = {
        **PLOTLY_LAYOUT_DEFAULTS,
        "xaxis_title": "Market Cap (Log₁₀ $)",
        "yaxis_title": "Count",
        "showlegend": "sector" in valid_df.columns,
    }
    fig.update_layout(**layout_config)

    return fig


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

    # Filter data
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


def _list_artifacts() -> List[Dict[str, str]]:
    """List all available artifacts from earnings_analytics and dashboard artifacts dirs."""
    items: List[Dict[str, str]] = []

    # Include artifacts from earnings_analytics directory
    base1 = PROJECT_ROOT / "outputs" / "eda" / "earnings_analytics"
    if base1.exists():
        for p in sorted(base1.glob("*")):
            if p.suffix.lower() not in {".html", ".json"}:
                continue
            items.append({"label": f"[Earnings] {p.name}", "value": str(p)})

    # Include artifacts from dashboard artifacts directory
    if ARTIFACTS_DIR.exists():
        for p in sorted(ARTIFACTS_DIR.glob("*")):
            if p.suffix.lower() not in {".html", ".json"}:
                continue
            items.append({"label": f"[Dashboard] {p.name}", "value": str(p)})

    return items


def _render_artifact(path_str: str) -> Any:
    """Render artifact content for the Artifacts tab.

    Styling aligned with code_guidelines.md Section 17.
    """
    if not path_str:
        return html.Div(
            "Select an artifact",
            style={"padding": "10px", "color": COLOR_PALETTE["neutral"]},
        )

    p = Path(path_str)
    if not p.exists():
        return html.Div(
            "Artifact not found",
            style={"padding": "10px", "color": COLOR_PALETTE["warning"]},
        )

    if p.suffix.lower() == ".html":
        # Serve via /app_assets route so iframe can load it.
        rel = (
            p.relative_to(PROJECT_ROOT / "outputs")
            if str(p).startswith(str(PROJECT_ROOT / "outputs"))
            else None
        )
        if rel is not None:
            src = f"/app_assets/{rel.as_posix()}"
            return html.Iframe(
                src=src,
                style={
                    "width": "100%",
                    "height": "650px",
                    "border": f"1px solid {COLOR_PALETTE['secondary']}",
                },
            )
        # Fallback: show simple message
        return html.Div(
            "HTML artifact is outside outputs/ and cannot be embedded.",
            style={"padding": "10px"},
        )

    if p.suffix.lower() == ".json":
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            pretty = json.dumps(payload, indent=2, sort_keys=True)
        except Exception:
            pretty = p.read_text(encoding="utf-8", errors="replace")
        return html.Pre(
            pretty,
            style={
                "maxHeight": "650px",
                "overflowY": "auto",
                "backgroundColor": "#111",
                "color": "#ffffff",
                "padding": "10px",
                "fontFamily": FONT_FAMILY,
            },
        )

    return html.Div("Unsupported artifact type", style={"padding": "10px"})


def create_app(
    *,
    data_source: DataSource = "auto",
    data_dir: Optional[str | Path] = None,
    db_url: Optional[str] = None,
    load_on_start: bool = True,
) -> dash.Dash:
    """Create Dash app instance.

    Set load_on_start=True when running interactively.
    Keep it False in tests to avoid running ETL.
    """

    initial_df = (
        load_data(data_source=data_source, data_dir=data_dir, db_url=db_url)
        if load_on_start
        else pd.DataFrame()
    )

    app = dash.Dash(
        __name__,
        title="Equities Dashboard",
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )
    server = app.server

    @server.route("/app_assets/<path:filename>")
    def serve_outputs(filename: str):
        return send_from_directory(PROJECT_ROOT / "outputs", filename)

    # Layout
    app.layout = html.Div(
        [
            html.H1("📈 Equities Analytics Dashboard", style={"textAlign": "center"}),
            dcc.Store(
                id="equities-data-store",
                data=initial_df.to_json(orient="split")
                if not initial_df.empty
                else None,
            ),
            html.Div(
                id="kpi-cards",
                style={
                    "display": "flex",
                    "justifyContent": "space-around",
                    "margin": "20px",
                },
            ),
            html.Div(
                [
                    html.H4(
                        "Filters", style={"marginBottom": "10px", "color": "white"}
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Sector", className="filter-label"),
                                    dcc.Dropdown(
                                        id="sector-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "sector"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Region", className="filter-label"),
                                    dcc.Dropdown(
                                        id="region-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "region"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Country", className="filter-label"),
                                    dcc.Dropdown(
                                        id="country-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "country"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Trading Country", className="filter-label"
                                    ),
                                    dcc.Dropdown(
                                        id="trading-country-dropdown",
                                        multi=True,
                                        options=_safe_options(
                                            initial_df, "trading_country"
                                        ),
                                    ),
                                ],
                                className="filter-item",
                            ),
                        ],
                        className="filter-row",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Industry", className="filter-label"),
                                    dcc.Dropdown(
                                        id="industry-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "industry"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Exchange", className="filter-label"),
                                    dcc.Dropdown(
                                        id="exchange-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "exchange"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Style Class", className="filter-label"),
                                    dcc.Dropdown(
                                        id="style-class-dropdown",
                                        multi=True,
                                        options=_safe_options(
                                            initial_df, "style_class"
                                        ),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Size Class", className="filter-label"),
                                    dcc.Dropdown(
                                        id="size-class-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "size_class"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                        ],
                        className="filter-row",
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Load / Refresh Data",
                                id="refresh-data-btn",
                                color="primary",
                                style={"marginRight": "10px"},
                            ),
                            dbc.Button(
                                "Reset Filters",
                                id="reset-filters-btn",
                                color="secondary",
                                style={"marginRight": "10px"},
                            ),
                            dbc.Button(
                                "Generate Artifacts",
                                id="generate-artifacts-btn",
                                color="success",
                            ),
                            html.Span(
                                id="data-status",
                                style={
                                    "marginLeft": "10px",
                                    "color": COLOR_PALETTE["neutral"],
                                    "fontFamily": FONT_FAMILY,
                                },
                            ),
                        ],
                        style={"margin": "10px 0"},
                    ),
                ],
                style={"padding": "10px"},
            ),
            dcc.Tabs(
                id="tabs",
                value="overview",
                children=[
                    dcc.Tab(
                        label="📋 Overview",
                        value="overview",
                        children=[
                            html.Div(
                                [
                                    dcc.Graph(id="target-vs-price-scatter"),
                                    dcc.Graph(id="market-cap-distribution"),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="📅 Earnings Analytics Dashboard",
                        value="earnings",
                        children=[
                            html.Div(
                                [
                                    # Alert summary panel (Task 3)
                                    html.Div(id="earnings-alert-summary"),
                                    # Alert filter dropdown (Task 4)
                                    html.Div(
                                        [
                                            html.Label(
                                                "Filter by alerts:",
                                                className="filter-label",
                                            ),
                                            dcc.Dropdown(
                                                id="earnings-alert-filter-dropdown",
                                                options=[
                                                    {
                                                        "label": "All tickers",
                                                        "value": "all",
                                                    },
                                                    {
                                                        "label": "Only tickers with alerts",
                                                        "value": "alerts_only",
                                                    },
                                                ],
                                                value="all",
                                                clearable=False,
                                                style={"width": "250px"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "10px",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    # Generate artifacts button (Task 5)
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Generate Earnings Analytics Artifacts",
                                                id="generate-earnings-artifacts-btn",
                                                color="info",
                                            ),
                                            html.Span(
                                                id="earnings-artifacts-status",
                                                style={
                                                    "marginLeft": "10px",
                                                    "color": COLOR_PALETTE["neutral"],
                                                    "fontFamily": FONT_FAMILY,
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "15px"},
                                    ),
                                    # Interactive Earnings Calendar Dashboard
                                    html.Div(
                                        [
                                            html.H5(
                                                "📅 Interactive Earnings Calendar",
                                                style={
                                                    "marginBottom": "10px",
                                                    "color": COLOR_PALETTE["info"],
                                                },
                                            ),
                                            # Calendar controls row
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Calendar Mode:",
                                                                className="filter-label",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="earnings-calendar-mode",
                                                                options=EARNINGS_MODE_OPTIONS,
                                                                value="all",
                                                                clearable=False,
                                                                style={
                                                                    "width": "180px"
                                                                },
                                                            ),
                                                        ],
                                                        className="filter-item",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Days Window (±):",
                                                                className="filter-label",
                                                            ),
                                                            dcc.Slider(
                                                                id="earnings-calendar-days",
                                                                min=3,
                                                                max=30,
                                                                step=1,
                                                                value=10,
                                                                marks={
                                                                    3: "3",
                                                                    7: "7",
                                                                    10: "10",
                                                                    14: "14",
                                                                    21: "21",
                                                                    30: "30",
                                                                },
                                                                tooltip={
                                                                    "placement": "bottom",
                                                                    "always_visible": False,
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "width": "200px",
                                                            "marginLeft": "20px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Top N:",
                                                                className="filter-label",
                                                            ),
                                                            dcc.Input(
                                                                id="earnings-calendar-top-n",
                                                                type="number",
                                                                value=50,
                                                                min=10,
                                                                max=200,
                                                                step=10,
                                                                style={"width": "80px"},
                                                            ),
                                                        ],
                                                        style={"marginLeft": "20px"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            dbc.Checkbox(
                                                                id="earnings-calendar-apply-filters",
                                                                label="Apply Global Filters",
                                                                value=True,
                                                            ),
                                                        ],
                                                        style={
                                                            "marginLeft": "20px",
                                                            "display": "flex",
                                                            "alignItems": "center",
                                                        },
                                                    ),
                                                ],
                                                className="filter-row",
                                                style={"marginBottom": "10px"},
                                            ),
                                            # Calendar status
                                            html.Div(
                                                id="earnings-calendar-status",
                                                style={
                                                    "color": COLOR_PALETTE["neutral"],
                                                    "fontSize": f"{FONT_SIZES['caption']}px",
                                                    "fontFamily": FONT_FAMILY,
                                                    "marginBottom": "5px",
                                                },
                                            ),
                                            # Earnings Calendar DataTable (code_guidelines.md Section 17.3)
                                            dash_table.DataTable(
                                                id="earnings-calendar-table",
                                                data=[],
                                                columns=[],
                                                style_table={
                                                    **TABLE_STYLE_TABLE,
                                                    "maxHeight": "400px",
                                                    "minWidth": "100%",
                                                },
                                                style_cell={
                                                    **TABLE_STYLE_CELL,
                                                    "width": "auto",
                                                    "maxWidth": "200px",
                                                    "textOverflow": "ellipsis",
                                                },
                                                style_header={
                                                    **TABLE_STYLE_HEADER,
                                                    "textTransform": "capitalize",
                                                    "fontWeight": "bold",
                                                },
                                                style_data_conditional=[
                                                    # Past earnings (red)
                                                    {
                                                        "if": {
                                                            "filter_query": "{days_to_earnings} < 0",
                                                            "column_id": "days_to_earnings",
                                                        },
                                                        "color": COLOR_PALETTE[
                                                            "danger"
                                                        ],
                                                    },
                                                    # Today (warning background)
                                                    {
                                                        "if": {
                                                            "filter_query": "{days_to_earnings} = 0",
                                                            "column_id": "days_to_earnings",
                                                        },
                                                        "backgroundColor": COLOR_PALETTE[
                                                            "warning"
                                                        ],
                                                        "color": "#000000",
                                                    },
                                                    # Future earnings (green)
                                                    {
                                                        "if": {
                                                            "filter_query": "{days_to_earnings} > 0",
                                                            "column_id": "days_to_earnings",
                                                        },
                                                        "color": COLOR_PALETTE[
                                                            "success"
                                                        ],
                                                    },
                                                ],
                                                sort_action="native",
                                                filter_action="native",
                                                page_action="native",
                                                page_size=15,
                                                row_selectable="multi",
                                                selected_rows=[],
                                            ),
                                        ],
                                        style={
                                            "padding": "15px",
                                            "backgroundColor": "#1a1a1a",
                                            "borderRadius": "8px",
                                            "marginBottom": "20px",
                                            "border": f"1px solid {COLOR_PALETTE['secondary']}",
                                        },
                                    ),
                                    # Existing charts
                                    dcc.Graph(id="earnings-events-timeline"),
                                    dcc.Graph(id="earnings-surprise-fig"),
                                    dcc.Graph(id="analyst-heatmap-fig"),
                                    dcc.Graph(id="market-movers-fig"),
                                    dcc.Graph(id="price-target-analytics-fig"),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="🚨 Alerts",
                        value="alerts",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H4(
                                                "Earnings Quality Alerts",
                                                style={"marginTop": "10px"},
                                            ),
                                            html.Div(
                                                id="alerts-meta",
                                                style={
                                                    "color": COLOR_PALETTE["neutral"],
                                                    "fontFamily": FONT_FAMILY,
                                                },
                                            ),
                                        ]
                                    ),
                                    html.H5("Regenerate"),
                                    html.Div(
                                        [
                                            html.Label("EPS miss threshold (%)"),
                                            dcc.Input(
                                                id="cfg-eps-miss",
                                                type="number",
                                                value=20.0,
                                                step=1,
                                            ),
                                            html.Label(
                                                "Downgrade threshold (%)",
                                                style={"marginLeft": "10px"},
                                            ),
                                            dcc.Input(
                                                id="cfg-downgrade",
                                                type="number",
                                                value=5.0,
                                                step=0.5,
                                            ),
                                            html.Label(
                                                "Downgrade min periods",
                                                style={"marginLeft": "10px"},
                                            ),
                                            dcc.Input(
                                                id="cfg-min-periods",
                                                type="number",
                                                value=2,
                                                step=1,
                                            ),
                                            html.Br(),
                                            html.Label("Target spread threshold (%)"),
                                            dcc.Input(
                                                id="cfg-target-spread",
                                                type="number",
                                                value=30.0,
                                                step=1,
                                            ),
                                            html.Label(
                                                "Pre-earnings window (days)",
                                                style={"marginLeft": "10px"},
                                            ),
                                            dcc.Input(
                                                id="cfg-window-days",
                                                type="number",
                                                value=7,
                                                step=1,
                                            ),
                                            html.Label(
                                                "Volatility quantile",
                                                style={"marginLeft": "10px"},
                                            ),
                                            dcc.Input(
                                                id="cfg-vol-quantile",
                                                type="number",
                                                value=0.75,
                                                step=0.05,
                                                min=0,
                                                max=1,
                                            ),
                                            html.Label(
                                                "Max tickers per alert",
                                                style={"marginLeft": "10px"},
                                            ),
                                            dcc.Input(
                                                id="cfg-max-tickers",
                                                type="number",
                                                value=10,
                                                step=1,
                                            ),
                                            html.Br(),
                                            dbc.Button(
                                                "Generate Alerts",
                                                id="generate-alerts-btn",
                                                color="warning",
                                            ),
                                            html.Span(
                                                id="generate-alerts-status",
                                                style={
                                                    "marginLeft": "10px",
                                                    "color": COLOR_PALETTE["neutral"],
                                                    "fontFamily": FONT_FAMILY,
                                                },
                                            ),
                                        ],
                                        style={
                                            "padding": "10px",
                                            "border": f"1px solid {COLOR_PALETTE['secondary']}",
                                        },
                                    ),
                                    # Alerts DataTable (code_guidelines.md Section 17.3)
                                    dash_table.DataTable(
                                        id="alerts-table",
                                        columns=[
                                            {"name": "Severity", "id": "severity"},
                                            {"name": "Type", "id": "alert_type"},
                                            {"name": "Count", "id": "count"},
                                            {
                                                "name": "Description",
                                                "id": "description",
                                            },
                                            {"name": "Tickers", "id": "tickers"},
                                        ],
                                        data=[],
                                        style_table=TABLE_STYLE_TABLE,
                                        style_cell=TABLE_STYLE_CELL,
                                        style_header=TABLE_STYLE_HEADER,
                                        sort_action="native",
                                        filter_action="native",
                                        page_action="native",
                                        page_size=20,
                                        style_data_conditional=[
                                            {
                                                "if": {
                                                    "filter_query": '{severity} = "high"'
                                                },
                                                **_severity_style("high"),
                                            },
                                            {
                                                "if": {
                                                    "filter_query": '{severity} = "medium"'
                                                },
                                                **_severity_style("medium"),
                                            },
                                            {
                                                "if": {
                                                    "filter_query": '{severity} = "low"'
                                                },
                                                **_severity_style("low"),
                                            },
                                        ],
                                    ),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="🔎 Data Explorer",
                        value="explorer",
                        children=[
                            html.Div(
                                [
                                    html.Label("Feature category"),
                                    dcc.Dropdown(
                                        id="feature-category-dropdown",
                                        options=[
                                            {"label": k, "value": k}
                                            for k in sorted(
                                                PHASE93_FEATURE_INPUTS.keys()
                                            )
                                        ],
                                        multi=True,
                                        value=["profitability"],
                                    ),
                                    html.Label("Columns"),
                                    dcc.Dropdown(
                                        id="explorer-columns-dropdown", multi=True
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Row limit"),
                                            dcc.Input(
                                                id="explorer-row-limit",
                                                type="number",
                                                value=200,
                                                step=50,
                                                min=10,
                                            ),
                                        ],
                                        style={"marginTop": "10px"},
                                    ),
                                    dbc.Button(
                                        "Update Table",
                                        id="explorer-update-btn",
                                        color="secondary",
                                        style={"marginTop": "10px"},
                                    ),
                                ],
                                style={
                                    "padding": "10px",
                                    "border": f"1px solid {COLOR_PALETTE['secondary']}",
                                },
                            ),
                            # Data Explorer DataTable (code_guidelines.md Section 17.3)
                            dash_table.DataTable(
                                id="explorer-table",
                                data=[],
                                columns=[],
                                style_table=TABLE_STYLE_TABLE,
                                style_cell=TABLE_STYLE_CELL,
                                style_header=TABLE_STYLE_HEADER,
                                sort_action="native",
                                filter_action="native",
                                page_action="native",
                                page_size=20,
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="🗂️ Artifacts",
                        value="artifacts",
                        children=[
                            html.Div(
                                [
                                    dcc.Dropdown(id="artifact-dropdown"),
                                    html.Div(
                                        id="artifact-viewer",
                                        style={"marginTop": "10px"},
                                    ),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                    # Task 6-7: Est. vs Actual vs Adjusted Tab
                    dcc.Tab(
                        label="📊 Est. vs Actual vs Adjusted",
                        value="est-actual",
                        children=[
                            html.Div(
                                [
                                    # Missing columns warning
                                    html.Div(id="est-actual-missing-cols-warning"),
                                    # Controls row
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Metric",
                                                        className="filter-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="est-actual-metric-selector",
                                                        options=[
                                                            {"label": k, "value": k}
                                                            for k in EST_ACTUAL_METRICS.keys()
                                                        ],
                                                        value="EPS",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="filter-item",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Surprise Calculation",
                                                        className="filter-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="est-actual-surprise-method",
                                                        options=[
                                                            {
                                                                "label": "Percentage",
                                                                "value": "pct",
                                                            },
                                                            {
                                                                "label": "Absolute",
                                                                "value": "abs",
                                                            },
                                                        ],
                                                        value="pct",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="filter-item",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Segment By",
                                                        className="filter-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="est-actual-segment-by",
                                                        options=[
                                                            {
                                                                "label": "Sector",
                                                                "value": "sector",
                                                            },
                                                            {
                                                                "label": "Region",
                                                                "value": "region",
                                                            },
                                                            {
                                                                "label": "Size Class",
                                                                "value": "size_class",
                                                            },
                                                            {
                                                                "label": "Style Class",
                                                                "value": "style_class",
                                                            },
                                                            {
                                                                "label": "Industry",
                                                                "value": "industry",
                                                            },
                                                            {
                                                                "label": "Trading Country",
                                                                "value": "trading_country",
                                                            },
                                                            {
                                                                "label": "Exchange",
                                                                "value": "exchange",
                                                            },
                                                        ],
                                                        value="sector",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="filter-item",
                                            ),
                                        ],
                                        className="filter-row",
                                    ),
                                    # Charts
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id="est-actual-scatter-fig",
                                                style={"height": "400px"},
                                            ),
                                            dcc.Graph(
                                                id="est-actual-distribution-fig",
                                                style={"height": "400px"},
                                            ),
                                        ],
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "1fr 1fr",
                                            "gap": "10px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id="est-actual-adjusted-fig",
                                                style={"height": "400px"},
                                            ),
                                            dcc.Graph(
                                                id="est-actual-revision-fig",
                                                style={"height": "400px"},
                                            ),
                                        ],
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "1fr 1fr",
                                            "gap": "10px",
                                        },
                                    ),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                    # Task 8: Earnings Monitoring Tab
                    dcc.Tab(
                        label="📈 Earnings Monitoring",
                        value="monitoring",
                        children=[
                            html.Div(
                                [
                                    # KPI cards row
                                    html.Div(
                                        id="monitoring-kpi-row", className="kpi-row"
                                    ),
                                    # Controls
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Segment By",
                                                        className="filter-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="monitoring-segment-by",
                                                        options=[
                                                            {
                                                                "label": "Sector",
                                                                "value": "sector",
                                                            },
                                                            {
                                                                "label": "Region",
                                                                "value": "region",
                                                            },
                                                            {
                                                                "label": "Size Class",
                                                                "value": "size_class",
                                                            },
                                                        ],
                                                        value="sector",
                                                        clearable=False,
                                                        style={"width": "200px"},
                                                    ),
                                                ],
                                                style={"marginRight": "20px"},
                                            ),
                                            dbc.Button(
                                                "Generate Monitoring Report",
                                                id="generate-monitoring-report-btn",
                                                color="success",
                                            ),
                                            html.Span(
                                                id="monitoring-report-status",
                                                style={
                                                    "marginLeft": "10px",
                                                    "color": COLOR_PALETTE["neutral"],
                                                    "fontFamily": FONT_FAMILY,
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "marginBottom": "15px",
                                        },
                                    ),
                                    # Charts
                                    dcc.Graph(id="monitoring-growth-fig"),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id="monitoring-margin-fig",
                                                style={"flex": "1"},
                                            ),
                                            dcc.Graph(
                                                id="monitoring-quality-fig",
                                                style={"flex": "1"},
                                            ),
                                        ],
                                        style={"display": "flex", "gap": "10px"},
                                    ),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                ],
            ),
        ]
    )

    # ---------------------- Callbacks ----------------------

    @app.callback(
        Output("equities-data-store", "data"),
        Output("data-status", "children"),
        Output("sector-dropdown", "options"),
        Output("region-dropdown", "options"),
        Output("country-dropdown", "options"),
        Output("trading-country-dropdown", "options"),
        Output("industry-dropdown", "options"),
        Output("exchange-dropdown", "options"),
        Output("style-class-dropdown", "options"),
        Output("size-class-dropdown", "options"),
        Input("refresh-data-btn", "n_clicks"),
        prevent_initial_call=not load_on_start,
    )
    def _refresh_data(_n_clicks):
        df, status_summary = load_data_csv_first(
            data_dir=data_dir,
            db_url=db_url,
        )

        if not df.empty:
            # status_summary now contains the detailed metrics.summary()
            status = f"Rows: {len(df):,} | {status_summary}"
        else:
            status = "No data loaded or ETL failed"

        # Update filter dropdown options with loaded data
        return (
            df.to_json(orient="split"),
            status,
            _safe_options(df, "sector"),
            _safe_options(df, "region"),
            _safe_options(df, "country"),
            _safe_options(df, "trading_country"),
            _safe_options(df, "industry"),
            _safe_options(df, "exchange"),
            _safe_options(df, "style_class"),
            _safe_options(df, "size_class"),
        )

    @app.callback(
        Output("kpi-cards", "children"),
        Output("target-vs-price-scatter", "figure"),
        Output("market-cap-distribution", "figure"),
        Input("equities-data-store", "data"),
        Input("sector-dropdown", "value"),
        Input("region-dropdown", "value"),
        Input("country-dropdown", "value"),
        Input("trading-country-dropdown", "value"),
        Input("industry-dropdown", "value"),
        Input("exchange-dropdown", "value"),
        Input("style-class-dropdown", "value"),
        Input("size-class-dropdown", "value"),
        prevent_initial_call=False,
    )
    def _update_overview(
        data_json,
        sectors,
        regions,
        countries,
        trading_countries,
        industries,
        exchanges,
        style_classes,
        size_classes,
    ):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        filtered = apply_filters(
            df,
            sectors=_coerce_list(sectors),
            regions=_coerce_list(regions),
            countries=_coerce_list(countries),
            trading_countries=_coerce_list(trading_countries),
            industries=_coerce_list(industries),
            exchanges=_coerce_list(exchanges),
            style_classes=_coerce_list(style_classes),
            size_classes=_coerce_list(size_classes),
        )

        return (
            _kpi_cards(filtered),
            _target_vs_price_scatter(filtered, use_log_scale=True),
            _market_cap_distribution(filtered),
        )

    # Task 3: Alert summary callback
    @app.callback(
        Output("earnings-alert-summary", "children"),
        Input("equities-data-store", "data"),
    )
    def _update_alert_summary(data_json):
        """Render compact alert summary panel."""
        payload = load_alerts_payload(DEFAULT_ALERTS_PATH)
        alerts = payload.get("alerts", [])

        if not alerts:
            return html.Div(
                "No alerts available. Click 'Generate Alerts' in the Alerts tab.",
                style={
                    "color": COLOR_PALETTE["neutral"],
                    "padding": "10px",
                    "fontFamily": FONT_FAMILY,
                },
            )

        # Build summary cards
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for a in alerts:
            sev = a.get("severity", "low").lower()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        cards = []
        for sev, count in severity_counts.items():
            if count > 0:
                color = {"high": "danger", "medium": "warning", "low": "info"}.get(
                    sev, "secondary"
                )
                cards.append(
                    dbc.Badge(f"{sev.upper()}: {count}", color=color, className="me-2")
                )

        return html.Div(
            [html.Span("Alerts: ", style={"fontWeight": "bold"})] + cards,
            style={
                "padding": "10px",
                "backgroundColor": "#1a1a1a",
                "borderRadius": "5px",
                "marginBottom": "10px",
            },
        )

    # Task 4: Updated earnings figures callback with alert filter and global filter integration
    @app.callback(
        Output("earnings-events-timeline", "figure"),
        Output("earnings-surprise-fig", "figure"),
        Output("analyst-heatmap-fig", "figure"),
        Output("market-movers-fig", "figure"),
        Output("price-target-analytics-fig", "figure"),
        Input("equities-data-store", "data"),
        Input("earnings-alert-filter-dropdown", "value"),
        Input("sector-dropdown", "value"),
        Input("region-dropdown", "value"),
        Input("country-dropdown", "value"),
        Input("trading-country-dropdown", "value"),
        Input("industry-dropdown", "value"),
        Input("exchange-dropdown", "value"),
        Input("style-class-dropdown", "value"),
        Input("size-class-dropdown", "value"),
    )
    def _update_earnings_figs(
        data_json,
        alert_filter,
        sectors,
        regions,
        countries,
        trading_countries,
        industries,
        exchanges,
        style_classes,
        size_classes,
    ):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            empty = create_empty_state_figure("Earnings Analytics", "No data loaded")
            return empty, empty, empty, empty, empty

        # Apply global filters for cross-tab synchronization
        df = apply_filters(
            df,
            sectors=_coerce_list(sectors),
            regions=_coerce_list(regions),
            countries=_coerce_list(countries),
            trading_countries=_coerce_list(trading_countries),
            industries=_coerce_list(industries),
            exchanges=_coerce_list(exchanges),
            style_classes=_coerce_list(style_classes),
            size_classes=_coerce_list(size_classes),
        )

        if df.empty:
            empty = create_empty_state_figure(
                "Earnings Analytics", "No data matching filters"
            )
            return empty, empty, empty, empty, empty

        # Apply alert filter (Task 4)
        if alert_filter == "alerts_only":
            payload = load_alerts_payload(DEFAULT_ALERTS_PATH)
            alert_tickers = set()
            for a in payload.get("alerts", []):
                alert_tickers.update(a.get("tickers", []))
            if alert_tickers and "ticker" in df.columns:
                df = df[df["ticker"].isin(alert_tickers)]
                if df.empty:
                    empty = create_empty_state_figure(
                        "Earnings Analytics", "No tickers with active alerts"
                    )
                    return empty, empty, empty, empty, empty

        # These functions are designed to be robust to missing columns.
        return (
            create_earnings_events_chart(df),
            create_earnings_surprise_dashboard(df),
            create_analyst_recommendation_heatmap(df),
            create_market_movers_dashboard(df),
            create_price_target_analytics(df),
        )

    # Task 5: Generate earnings artifacts callback
    @app.callback(
        Output("earnings-artifacts-status", "children"),
        Input("generate-earnings-artifacts-btn", "n_clicks"),
        State("equities-data-store", "data"),
        prevent_initial_call=True,
    )
    def _generate_earnings_artifacts(_n, data_json):
        """Generate earnings analytics artifacts (Task 5)."""
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            return "No data available"

        try:
            # Generate the 4 main earnings artifacts
            artifacts_generated = []

            for name, func in [
                (
                    "earnings_surprise_dashboard.html",
                    create_earnings_surprise_dashboard,
                ),
                (
                    "analyst_recommendation_heatmap.html",
                    create_analyst_recommendation_heatmap,
                ),
                ("market_movers_dashboard.html", create_market_movers_dashboard),
                ("price_target_analytics.html", create_price_target_analytics),
            ]:
                output_path = ARTIFACTS_DIR / name
                func(df, output_path=output_path)
                artifacts_generated.append(name)

            return f"✓ Generated {len(artifacts_generated)} artifacts"
        except Exception as e:
            return f"Error: {e}"

    # Interactive Earnings Calendar callback with global filter integration
    @app.callback(
        Output("earnings-calendar-table", "columns"),
        Output("earnings-calendar-table", "data"),
        Output("earnings-calendar-status", "children"),
        Input("equities-data-store", "data"),
        Input("earnings-calendar-mode", "value"),
        Input("earnings-calendar-days", "value"),
        Input("earnings-calendar-top-n", "value"),
        Input("earnings-calendar-apply-filters", "value"),
        Input("sector-dropdown", "value"),
        Input("region-dropdown", "value"),
        Input("country-dropdown", "value"),
        Input("trading-country-dropdown", "value"),
        Input("industry-dropdown", "value"),
        Input("exchange-dropdown", "value"),
        Input("style-class-dropdown", "value"),
        Input("size-class-dropdown", "value"),
        prevent_initial_call=False,
    )
    def _update_earnings_calendar(
        data_json,
        mode,
        days_window,
        top_n,
        should_apply_filters,
        sectors,
        regions,
        countries,
        trading_countries,
        industries,
        exchanges,
        style_classes,
        size_classes,
    ):
        """Update the interactive earnings calendar DataTable."""
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            return [], [], "No data available"

        # Apply global filters if checkbox is checked
        if should_apply_filters:
            df = apply_filters(
                df,
                sectors=_coerce_list(sectors),
                regions=_coerce_list(regions),
                countries=_coerce_list(countries),
                trading_countries=_coerce_list(trading_countries),
                industries=_coerce_list(industries),
                exchanges=_coerce_list(exchanges),
                style_classes=_coerce_list(style_classes),
                size_classes=_coerce_list(size_classes),
            )

        if df.empty:
            return [], [], "No data after applying filters"

        # Ensure next_earnings column exists (graceful fallback)
        if "next_earnings" not in df.columns:
            logger.warning(
                "next_earnings column missing, creating empty column for calendar"
            )
            df = df.copy()
            df["next_earnings"] = pd.NaT
        else:
            df = df.copy()

        # Parse dates and calculate days to earnings
        reference_date = pd.Timestamp.now()
        df["next_earnings"] = pd.to_datetime(df["next_earnings"], errors="coerce")

        # Filter by days window
        days_window = int(days_window) if days_window else 10
        df["days_to_earnings"] = (df["next_earnings"] - reference_date).dt.days
        mask = df["days_to_earnings"].notna() & (
            df["days_to_earnings"].abs() <= days_window
        )
        filtered_df = df[mask].copy()

        if filtered_df.empty:
            return [], [], f"No earnings within ±{days_window} days"

        # Use create_earnings_calendar_dashboard for consistent column selection
        try:
            calendar_df = create_earnings_calendar_dashboard(
                filtered_df,
                reference_date=reference_date,
                top_n=int(top_n) if top_n else 50,
                mode=mode or "all",
            )
        except Exception as e:
            logger.warning(f"Calendar dashboard creation failed: {e}")
            # Fallback to basic columns
            basic_cols = [
                "ticker",
                "sector",
                "region",
                "next_earnings",
                "days_to_earnings",
            ]
            available_cols = [c for c in basic_cols if c in filtered_df.columns]
            if "market_cap" in filtered_df.columns:
                available_cols.append("market_cap")
            calendar_df = filtered_df[available_cols].head(int(top_n) if top_n else 50)

        if calendar_df.empty:
            return [], [], "No data to display"

        # Ensure days_to_earnings is in the output
        if (
            "days_to_earnings" not in calendar_df.columns
            and "next_earnings" in calendar_df.columns
        ):
            calendar_df["days_to_earnings"] = (
                pd.to_datetime(calendar_df["next_earnings"], errors="coerce")
                - reference_date
            ).dt.days

        # Format columns for DataTable (code_guidelines.md Section 17.3)
        columns = []
        for col in calendar_df.columns:
            col_name = col.replace("_", " ").strip().capitalize()
            # Headers: Bold, sentence case (handled via css/DataTable props)
            col_def = {"name": col_name, "id": col, "selectable": True}

            # Apply numeric formatting based on column role/type
            if any(
                x in col
                for x in [
                    "price",
                    "market_cap",
                    "enterprise_value",
                    "ebitda",
                    "ebit",
                    "income",
                    "revenue",
                ]
            ):
                col_def.update({"type": "numeric", "format": {"specifier": "$,.2f"}})
            elif "pct" in col or "margin" in col or "growth" in col or "yield" in col:
                col_def.update({"type": "numeric", "format": {"specifier": ".2%"}})
            elif col in calendar_df.columns and calendar_df[col].dtype in [
                np.float64,
                np.float32,
            ]:
                col_def.update({"type": "numeric", "format": {"specifier": ".2f"}})

            columns.append(col_def)

        # Convert to records, handling dates and rounding
        display_df = calendar_df.copy()
        for col in display_df.columns:
            if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                display_df[col] = display_df[col].dt.strftime("%Y-%m-%d")
            elif "days_to_earnings" in col:
                # Special handling for days display as seen in images (+0, -1)
                display_df[col] = display_df[col].apply(
                    lambda x: f"{int(x):+d}" if pd.notnull(x) else ""
                )

        data = display_df.to_dict("records")

        # Status message
        mode_display = (mode or "all").replace("_", " ").title()
        status = f"Showing {len(data)} companies | Mode: {mode_display} | Window: ±{days_window} days"
        if should_apply_filters:
            status += " | Global filters applied"

        return columns, data, status

    @app.callback(
        Output("alerts-table", "data"),
        Output("alerts-meta", "children"),
        Output("generate-alerts-status", "children"),
        Input("generate-alerts-btn", "n_clicks"),
        State("equities-data-store", "data"),
        State("cfg-eps-miss", "value"),
        State("cfg-downgrade", "value"),
        State("cfg-min-periods", "value"),
        State("cfg-target-spread", "value"),
        State("cfg-window-days", "value"),
        State("cfg-vol-quantile", "value"),
        State("cfg-max-tickers", "value"),
        prevent_initial_call=True,
    )
    def _generate_alerts(
        _n,
        data_json,
        eps_miss,
        downgrade,
        min_periods,
        target_spread,
        window_days,
        vol_quantile,
        max_tickers,
    ):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            payload = load_alerts_payload(DEFAULT_ALERTS_PATH)
            rows = _alerts_to_rows(payload)
            meta = (
                f"Loaded {len(rows)} alerts from disk"
                if rows
                else "No alerts available"
            )
            return rows, meta, ""

        cfg = EarningsAlertConfig(
            eps_surprise_miss_threshold_pct=float(eps_miss)
            if eps_miss is not None
            else 20.0,
            analyst_downgrade_threshold_pct=float(downgrade)
            if downgrade is not None
            else 5.0,
            analyst_downgrade_min_periods=int(min_periods)
            if min_periods is not None
            else 2,
            target_spread_threshold_pct=float(target_spread)
            if target_spread is not None
            else 30.0,
            pre_earnings_window_days=int(window_days) if window_days is not None else 7,
            pre_earnings_volatility_quantile=(
                float(vol_quantile) if vol_quantile is not None else 0.75
            ),
            max_tickers_per_alert=int(max_tickers) if max_tickers is not None else 10,
        )
        payload = generate_earnings_quality_alerts(
            df,
            config=cfg,
            output_path=DEFAULT_ALERTS_PATH,
        )
        rows = _alerts_to_rows(payload)
        meta = f"Generated {len(rows)} alerts (monitored: {payload.get('total_stocks_monitored', '')})"
        status = (
            f"Wrote {DEFAULT_ALERTS_PATH.name}"
            if DEFAULT_ALERTS_PATH.parent.exists()
            else ""
        )
        return rows, meta, status

    @app.callback(
        Output("explorer-columns-dropdown", "options"),
        Output("explorer-columns-dropdown", "value"),
        Input("feature-category-dropdown", "value"),
        Input("equities-data-store", "data"),
    )
    def _update_explorer_columns(categories, data_json):
        """Update explorer column options based on selected categories.

        Uses DEFAULT_EXPLORER_COLUMNS as base selection, then adds category-specific
        columns up to 10 total. Ensures 'name' and other key columns are always
        included when available.
        """
        categories_list = _coerce_list(categories)
        metrics = get_category_metrics(categories_list)
        cols = sorted({c for values in metrics.values() for c in values})
        # Only show columns that exist
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is not None and not df.empty:
            cols = [c for c in cols if c in df.columns]
            # Use DEFAULT_EXPLORER_COLUMNS as base, filtering to available columns
            default = [c for c in DEFAULT_EXPLORER_COLUMNS if c in df.columns]
        else:
            default = []

        # Add additional columns from category selection up to 10 total
        for c in cols:
            if c not in default and len(default) < 10:
                default.append(c)

        return ([{"label": c, "value": c} for c in cols], default)

    @app.callback(
        Output("explorer-table", "columns"),
        Output("explorer-table", "data"),
        Input("explorer-update-btn", "n_clicks"),
        State("equities-data-store", "data"),
        State("explorer-columns-dropdown", "value"),
        State("explorer-row-limit", "value"),
        prevent_initial_call=True,
    )
    def _update_explorer_table(_n, data_json, columns, row_limit):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        cols = _coerce_list(columns)
        limit = int(row_limit) if row_limit is not None else 200
        if df is None or df.empty or not cols:
            return [], []
        existing_cols = [c for c in cols if c in df.columns]
        view = df[existing_cols].head(max(10, limit)).copy()
        return ([{"name": c, "id": c} for c in existing_cols], view.to_dict("records"))

    @app.callback(
        Output("artifact-dropdown", "options"),
        Input("tabs", "value"),
    )
    def _populate_artifact_dropdown(tab_value):
        if tab_value != "artifacts":
            return []
        return _list_artifacts()

    @app.callback(
        Output("artifact-viewer", "children"),
        Input("artifact-dropdown", "value"),
    )
    def _show_artifact(path_str):
        return _render_artifact(path_str or "")

    @app.callback(
        Output("sector-dropdown", "value"),
        Output("region-dropdown", "value"),
        Output("country-dropdown", "value"),
        Output("trading-country-dropdown", "value"),
        Output("industry-dropdown", "value"),
        Output("exchange-dropdown", "value"),
        Output("style-class-dropdown", "value"),
        Output("size-class-dropdown", "value"),
        Input("reset-filters-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_filters(_n):
        """Reset all filter dropdowns to empty."""
        return None, None, None, None, None, None, None, None

    @app.callback(
        Output("data-status", "children", allow_duplicate=True),
        Input("generate-artifacts-btn", "n_clicks"),
        State("equities-data-store", "data"),
        prevent_initial_call=True,
    )
    def _generate_artifacts(_n, data_json):
        """Generate dashboard artifacts from current data."""
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            return "No data available for artifact generation"

        try:
            metadata = generate_dashboard_artifacts(df)
            total_artifacts = len(metadata.get("artifacts", {}))
            return f"Generated {total_artifacts} artifacts in {ARTIFACTS_DIR.name}"
        except Exception as e:
            return f"Artifact generation failed: {e}"

    # Task 6-7: Est vs Actual tab callback
    @app.callback(
        Output("est-actual-missing-cols-warning", "children"),
        Output("est-actual-scatter-fig", "figure"),
        Output("est-actual-distribution-fig", "figure"),
        Output("est-actual-adjusted-fig", "figure"),
        Output("est-actual-revision-fig", "figure"),
        Input("equities-data-store", "data"),
        Input("est-actual-metric-selector", "value"),
        Input("est-actual-surprise-method", "value"),
        Input("est-actual-segment-by", "value"),
    )
    def _update_est_actual_tab(data_json, metric, surprise_method, segment_by):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            empty_fig = create_empty_state_figure(
                "Estimated vs Actual", "No data available"
            )
            return html.Div(), empty_fig, empty_fig, empty_fig, empty_fig

        # Get metric columns
        metric_config = EST_ACTUAL_METRICS.get(metric, EST_ACTUAL_METRICS["EPS"])
        actual_col = metric_config["actual"]
        estimate_col = metric_config["estimate"]
        adjusted_col = metric_config.get("adjusted")
        gaap_col = metric_config.get("gaap")

        # Check for missing columns
        required = [actual_col, estimate_col]
        _, missing = validate_required_columns(df, required, f"{metric} Analysis")
        warning = create_missing_columns_warning(missing, f"{metric} Analysis")

        # 1. Scatter: Estimated vs Actual
        scatter_fig = create_empty_state_figure(
            f"{metric}: Estimated vs Actual", "Required columns not available"
        )
        if actual_col in df.columns and estimate_col in df.columns:
            plot_df = df[[actual_col, estimate_col]].dropna()
            if segment_by in df.columns:
                plot_df[segment_by] = df.loc[plot_df.index, segment_by]

            if not plot_df.empty:
                scatter_fig = px.scatter(
                    plot_df,
                    x=estimate_col,
                    y=actual_col,
                    color=segment_by if segment_by in plot_df.columns else None,
                    title=f"{metric}: Estimated vs Actual",
                    template="plotly_dark",
                    hover_data=[segment_by] if segment_by in plot_df.columns else None,
                )
                # Add diagonal line
                min_val = min(plot_df[estimate_col].min(), plot_df[actual_col].min())
                max_val = max(plot_df[estimate_col].max(), plot_df[actual_col].max())
                scatter_fig.add_scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(dash="dash", color="white"),
                    name="Perfect Forecast",
                    showlegend=True,
                )

        # 2. Distribution: Surprise histogram
        dist_fig = create_empty_state_figure(
            f"{metric} Surprise Distribution", "Data not available"
        )
        if actual_col in df.columns and estimate_col in df.columns:
            surprise = compute_surprise(
                df[actual_col], df[estimate_col], mode=surprise_method
            )
            surprise_df = pd.DataFrame({"surprise": surprise})
            if segment_by in df.columns:
                surprise_df[segment_by] = df[segment_by]
            surprise_df = surprise_df.dropna(subset=["surprise"])

            if not surprise_df.empty:
                dist_fig = px.histogram(
                    surprise_df,
                    x="surprise",
                    color=segment_by if segment_by in surprise_df.columns else None,
                    nbins=50,
                    title=f"{metric} Surprise Distribution ({'%' if surprise_method == 'pct' else 'Absolute'})",
                    template="plotly_dark",
                )
                dist_fig.add_vline(x=0, line_dash="dash", line_color="white")

        # 3. Adjusted vs GAAP delta
        adjusted_fig = create_empty_state_figure(
            f"{metric}: Adjusted vs GAAP", "Data not available"
        )
        if (
            adjusted_col
            and gaap_col
            and adjusted_col in df.columns
            and gaap_col in df.columns
        ):
            adj_num = pd.to_numeric(df[adjusted_col], errors="coerce")
            gaap_num = pd.to_numeric(df[gaap_col], errors="coerce")
            delta = adj_num - gaap_num
            delta_df = pd.DataFrame({"delta": delta})
            if segment_by in df.columns:
                delta_df[segment_by] = df[segment_by]
            delta_df = delta_df.dropna(subset=["delta"])

            if not delta_df.empty:
                adjusted_fig = px.box(
                    delta_df,
                    x=segment_by if segment_by in delta_df.columns else None,
                    y="delta",
                    title=f"{metric}: Adjusted vs GAAP Delta by {segment_by.replace('_', ' ').title()}",
                    template="plotly_dark",
                )
        else:
            adjusted_fig.add_annotation(
                text="Adjusted vs GAAP comparison not available for this metric",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # 4. Revision trend (if revision columns exist)
        revision_fig = create_empty_state_figure(
            "Estimate Revision Trend", "Revision data not available"
        )
        revision_cols = [c for c in df.columns if "rev_pct" in c.lower()]
        if revision_cols:
            # Use first available revision column
            rev_col = revision_cols[0]
            rev_df = pd.DataFrame(
                {rev_col: pd.to_numeric(df[rev_col], errors="coerce")}
            )
            if segment_by in df.columns:
                rev_df[segment_by] = df[segment_by]
            rev_df = rev_df.dropna(subset=[rev_col])

            if not rev_df.empty:
                revision_fig = px.box(
                    rev_df,
                    x=segment_by if segment_by in rev_df.columns else None,
                    y=rev_col,
                    title=f"Estimate Revision Trend ({rev_col})",
                    template="plotly_dark",
                )

        return warning, scatter_fig, dist_fig, adjusted_fig, revision_fig

    # Task 8: Monitoring tab callback
    @app.callback(
        Output("monitoring-kpi-row", "children"),
        Output("monitoring-growth-fig", "figure"),
        Output("monitoring-margin-fig", "figure"),
        Output("monitoring-quality-fig", "figure"),
        Input("equities-data-store", "data"),
        Input("monitoring-segment-by", "value"),
    )
    def _update_monitoring_tab(data_json, segment_by):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            empty_fig = create_empty_state_figure("Monitoring", "No data available")
            return [], empty_fig, empty_fig, empty_fig

        # KPI cards
        kpi_cards = _monitoring_kpi_cards(df)

        # 1. Growth trends by segment
        growth_fig = create_empty_state_figure("Revenue Growth", "Data not available")
        growth_cols = ["total_revenues_cagr_5y_fy", "revenues_est_yoy_pct_fy1e"]
        available_growth = [c for c in growth_cols if c in df.columns]
        if available_growth and segment_by in df.columns:
            growth_col = available_growth[0]
            growth_df = df[[segment_by, growth_col]].copy()
            growth_df[growth_col] = pd.to_numeric(
                growth_df[growth_col], errors="coerce"
            )
            growth_df = growth_df.dropna()

            if not growth_df.empty:
                growth_fig = px.box(
                    growth_df,
                    x=segment_by,
                    y=growth_col,
                    title=f"Revenue Growth by {segment_by.replace('_', ' ').title()}",
                    template="plotly_dark",
                )

        # 2. Margin distribution
        margin_fig = create_empty_state_figure("Margin Analysis", "Data not available")
        margin_cols = ["gross_profit_margin_pct_ltm", "net_income_margin_pct_ltm"]
        available_margin = [c for c in margin_cols if c in df.columns]
        if available_margin and segment_by in df.columns:
            margin_col = available_margin[0]
            margin_df = df[[segment_by, margin_col]].copy()
            margin_df[margin_col] = pd.to_numeric(
                margin_df[margin_col], errors="coerce"
            )
            margin_df = margin_df.dropna()

            if not margin_df.empty:
                margin_fig = px.box(
                    margin_df,
                    x=segment_by,
                    y=margin_col,
                    title=f"Margin Distribution by {segment_by.replace('_', ' ').title()}",
                    template="plotly_dark",
                )

        # 3. Quality metrics (ROE, Altman Z, etc.)
        quality_fig = create_empty_state_figure("Quality Metrics", "Data not available")
        quality_cols = [
            "return_on_equity_pct_ltm",
            "altman_z_score_ltm",
            "return_on_assets_roa_pct_ltm",
        ]
        available_quality = [c for c in quality_cols if c in df.columns]
        if available_quality and segment_by in df.columns:
            quality_col = available_quality[0]
            quality_df = df[[segment_by, quality_col]].copy()
            quality_df[quality_col] = pd.to_numeric(
                quality_df[quality_col], errors="coerce"
            )
            quality_df = quality_df.dropna()

            if not quality_df.empty:
                quality_fig = px.box(
                    quality_df,
                    x=segment_by,
                    y=quality_col,
                    title=f"Quality Metrics ({quality_col.replace('_', ' ').title()}) by {segment_by.replace('_', ' ').title()}",
                    template="plotly_dark",
                )

        return kpi_cards, growth_fig, margin_fig, quality_fig

    # Task 8: Generate monitoring report callback
    @app.callback(
        Output("monitoring-report-status", "children"),
        Input("generate-monitoring-report-btn", "n_clicks"),
        State("equities-data-store", "data"),
        prevent_initial_call=True,
    )
    def _generate_monitoring_report(_n, data_json):
        """Generate monitoring JSON report."""
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            return "No data available"

        try:
            report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "total_stocks": len(df),
                "kpis": {},
            }

            # Collect KPI values
            if "total_revenues_cagr_5y_fy" in df.columns:
                growth = pd.to_numeric(df["total_revenues_cagr_5y_fy"], errors="coerce")
                report["kpis"]["pct_positive_revenue_growth"] = float(
                    (growth > 0).mean() * 100
                )
                report["kpis"]["median_revenue_growth"] = float(growth.median())

            if "net_income_margin_pct_ltm" in df.columns:
                margin = pd.to_numeric(df["net_income_margin_pct_ltm"], errors="coerce")
                report["kpis"]["median_net_margin"] = float(margin.median())

            if "return_on_equity_pct_ltm" in df.columns:
                roe = pd.to_numeric(df["return_on_equity_pct_ltm"], errors="coerce")
                report["kpis"]["median_roe"] = float(roe.median())

            # Save report
            output_path = ARTIFACTS_DIR / "monitoring_report.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            return f"✓ Report saved to {output_path.name}"
        except Exception as e:
            return f"Error: {e}"

    return app


def main() -> None:
    app = create_app(load_on_start=True)
    app.run(debug=False)


if __name__ == "__main__":
    main()
