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
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, dash_table, dcc, html
from flask import send_from_directory

from finance_ml.dashboards.earnings_widgets import (
    EarningsAlertConfig,
    create_analyst_recommendation_heatmap,
    create_earnings_surprise_dashboard,
    create_market_movers_dashboard,
    create_price_target_analytics,
    generate_earnings_quality_alerts,
    get_category_metrics,
)
from finance_ml.ml_workflow.data.schema import PHASE93_FEATURE_INPUTS
from finance_ml.ml_workflow.preprocessing.etl import etl_with_features

DataSource = Literal["auto", "csv", "db"]


PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ALERTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "eda"
    / "earnings_analytics"
    / "earnings_quality_alerts.json"
)


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


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
    sev = str(severity).lower().strip()
    if sev == "high":
        return {"backgroundColor": "#4d0000", "color": "white"}
    if sev == "medium":
        return {"backgroundColor": "#4d3300", "color": "white"}
    if sev == "low":
        return {"backgroundColor": "#00334d", "color": "white"}
    return {}


def _safe_options(df: pd.DataFrame, col: str) -> List[Dict[str, str]]:
    if df is None or df.empty or col not in df.columns:
        return []
    values = sorted([v for v in df[col].dropna().astype(str).unique().tolist()])
    return [{"label": v, "value": v} for v in values]


def _kpi_cards(df: pd.DataFrame) -> List[Any]:
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
                    html.Div(title, className="kpi-title"),
                    html.Div(value, className="kpi-value"),
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
        cards.append(card("Mean Market Cap", f"{market_cap_mean:,.0f}"))
    return cards


def _target_vs_price_scatter(df: pd.DataFrame):
    if df is None or df.empty:
        return px.scatter(title="Target vs Price (no data)")

    if "last_price" not in df.columns or "price_target" not in df.columns:
        return px.scatter(title="Target vs Price (missing columns)")

    hover_cols = [
        c
        for c in [
            "ticker",
            "sector",
            "region",
            "country",
            "industry",
            "exchange",
            "market_cap",
        ]
        if c in df.columns
    ]
    fig = px.scatter(
        df,
        x="last_price",
        y="price_target",
        color="sector" if "sector" in df.columns else None,
        hover_data=hover_cols,
        title="Price Target vs Last Price",
        template="plotly_dark",
    )
    fig.update_layout(xaxis_title="Last Price", yaxis_title="Price Target")
    return fig


def _list_artifacts() -> List[Dict[str, str]]:
    base = PROJECT_ROOT / "outputs" / "eda" / "earnings_analytics"
    if not base.exists():
        return []
    items: List[Dict[str, str]] = []
    for p in sorted(base.glob("*")):
        if p.suffix.lower() not in {".html", ".json"}:
            continue
        items.append({"label": p.name, "value": str(p)})
    return items


def _render_artifact(path_str: str) -> Any:
    if not path_str:
        return html.Div(
            "Select an artifact", style={"padding": "10px", "color": "#aaa"}
        )

    p = Path(path_str)
    if not p.exists():
        return html.Div(
            "Artifact not found", style={"padding": "10px", "color": "orange"}
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
                style={"width": "100%", "height": "650px", "border": "1px solid #333"},
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
                "padding": "10px",
            },
        )

    return html.Div("Unsupported artifact type", style={"padding": "10px"})


def create_app(
    *,
    data_source: DataSource = "auto",
    data_dir: Optional[str | Path] = None,
    db_url: Optional[str] = None,
    load_on_start: bool = False,
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
            dcc.Store(id="equities-data-store"),
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
                            ),
                            html.Span(
                                id="data-status",
                                style={"marginLeft": "10px", "color": "#aaa"},
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
                            dcc.Graph(id="target-vs-price-scatter"),
                        ],
                    ),
                    dcc.Tab(
                        label="📅 Earnings Analytics",
                        value="earnings",
                        children=[
                            dcc.Graph(id="earnings-surprise-fig"),
                            dcc.Graph(id="analyst-heatmap-fig"),
                            dcc.Graph(id="market-movers-fig"),
                            dcc.Graph(id="price-target-analytics-fig"),
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
                                                style={"color": "#aaa"},
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
                                                    "color": "#aaa",
                                                },
                                            ),
                                        ],
                                        style={
                                            "padding": "10px",
                                            "border": "1px solid #333",
                                        },
                                    ),
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
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "backgroundColor": "#111",
                                            "color": "white",
                                            "border": "1px solid #333",
                                            "fontFamily": "Segoe UI, Roboto, Arial",
                                            "fontSize": "14px",
                                            "padding": "6px",
                                            "whiteSpace": "normal",
                                            "height": "auto",
                                        },
                                        style_header={
                                            "backgroundColor": "#222",
                                            "fontWeight": "bold",
                                        },
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
                                style={"padding": "10px", "border": "1px solid #333"},
                            ),
                            dash_table.DataTable(
                                id="explorer-table",
                                data=[],
                                columns=[],
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "backgroundColor": "#111",
                                    "color": "white",
                                    "border": "1px solid #333",
                                    "fontFamily": "Segoe UI, Roboto, Arial",
                                    "fontSize": "14px",
                                    "padding": "6px",
                                },
                                style_header={
                                    "backgroundColor": "#222",
                                    "fontWeight": "bold",
                                },
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
                ],
            ),
        ]
    )

    # ---------------------- Callbacks ----------------------

    @app.callback(
        Output("equities-data-store", "data"),
        Output("data-status", "children"),
        Input("refresh-data-btn", "n_clicks"),
        prevent_initial_call=not load_on_start,
    )
    def _refresh_data(_n_clicks):
        df = load_data(data_source=data_source, data_dir=data_dir, db_url=db_url)
        status = f"Loaded {len(df):,} rows" if not df.empty else "No data loaded"
        return df.to_json(orient="split"), status

    @app.callback(
        Output("kpi-cards", "children"),
        Output("target-vs-price-scatter", "figure"),
        Input("equities-data-store", "data"),
        Input("sector-dropdown", "value"),
        Input("region-dropdown", "value"),
        Input("country-dropdown", "value"),
        Input("trading-country-dropdown", "value"),
        Input("industry-dropdown", "value"),
        Input("exchange-dropdown", "value"),
        Input("style-class-dropdown", "value"),
        Input("size-class-dropdown", "value"),
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

        return _kpi_cards(filtered), _target_vs_price_scatter(filtered)

    @app.callback(
        Output("earnings-surprise-fig", "figure"),
        Output("analyst-heatmap-fig", "figure"),
        Output("market-movers-fig", "figure"),
        Output("price-target-analytics-fig", "figure"),
        Input("equities-data-store", "data"),
    )
    def _update_earnings_figs(data_json):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            empty = px.scatter(title="No data")
            return empty, empty, empty, empty

        # These functions are designed to be robust to missing columns.
        return (
            create_earnings_surprise_dashboard(df),
            create_analyst_recommendation_heatmap(df),
            create_market_movers_dashboard(df),
            create_price_target_analytics(df),
        )

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
        default = cols[:10]
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

    return app


def main() -> None:
    app = create_app(load_on_start=True)
    app.run(debug=False)


if __name__ == "__main__":
    main()
