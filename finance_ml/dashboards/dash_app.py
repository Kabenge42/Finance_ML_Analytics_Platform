"""
Plotly Dash Dashboard for Finance ML Analytics
Run: python finance_ml/dashboards/dash_app.py
"""

from pathlib import Path
import os
from datetime import datetime

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, dash_table
from flask import send_from_directory

try:
    from finance_ml.dashboards.artifact_registry import ARTIFACTS
except ImportError:
    # Fallback for when running as script
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from finance_ml.dashboards.artifact_registry import ARTIFACTS

# Project root path for consistent path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_file_age(filepath: Path) -> str:
    """Get formatted age of a file."""
    if filepath.exists():
        mtime = os.path.getmtime(filepath)
        age = datetime.now() - datetime.fromtimestamp(mtime)
        if age.days == 0:
            return f"Updated today ({age.seconds//3600}h ago)"
        return f"Updated {age.days} days ago"
    return "Not generated"


def get_artifact_path(category: str, filename: str) -> str | None:
    """Resolve artifact path and check existence.

    Returns the source URL path (/app_assets/category/filename) if the file exists in outputs.
    """
    # Check if source exists in outputs
    source_path = PROJECT_ROOT / "outputs" / category / filename
    if source_path.exists():
        return f"/app_assets/{category}/{filename}"
    return None


def render_artifact_or_placeholder(category: str, key: str, height: int = 550):
    """Render artifact HTML or show placeholder with instructions."""
    if category not in ARTIFACTS or key not in ARTIFACTS[category]:
        return html.Div(f"Configuration missing for {category}/{key}", style={"color": "red"})

    item = ARTIFACTS[category][key]
    filename = item["file"]
    title = item["title"]
    section = item["section"]

    src = get_artifact_path(category, filename)
    file_path = PROJECT_ROOT / "outputs" / category / filename

    content = []
    content.append(html.H3(title, style={"textAlign": "center"}))

    if src:
        age = get_file_age(file_path)
        content.append(
            html.Div(
                f"🕒 {age}",
                style={
                    "textAlign": "right",
                    "fontSize": "0.8em",
                    "color": "#888",
                    "marginBottom": "5px",
                },
            )
        )
        content.append(
            html.Iframe(
                src=src,
                style={"width": "100%", "height": f"{height}px", "border": "1px solid #ddd"},
            )
        )
    else:
        content.append(
            html.Div(
                [
                    html.P(
                        f"⚠️ Artifact '{filename}' not available",
                        style={"textAlign": "center", "color": "orange"},
                    ),
                    html.P(
                        f"Run notebook Section {section} to generate",
                        style={"textAlign": "center", "fontStyle": "italic", "color": "#666"},
                    ),
                ],
                style={"padding": "50px", "border": "1px dashed #ccc", "margin": "10px"},
            )
        )

    return html.Div(content, style={"padding": "20px"})


def render_model_card():
    """Render model card markdown."""
    governance_dir = PROJECT_ROOT / "outputs" / "governance"
    if governance_dir.exists():
        model_cards = list(governance_dir.glob("model_card_*.md"))
        if model_cards:
            # Pick the latest or first
            latest_card = sorted(model_cards)[-1]
            with open(latest_card, "r", encoding="utf-8") as f:
                content = f.read()
            return html.Div(
                [
                    html.H3("Model Card", style={"textAlign": "center"}),
                    dcc.Markdown(
                        content,
                        style={
                            "padding": "20px",
                            "border": "1px solid #ddd",
                            "backgroundColor": "#f9f9f9",
                            "overflowY": "auto",
                            "maxHeight": "600px",
                        },
                    ),
                ],
                style={"padding": "20px"},
            )

    return html.Div(
        [
            html.P("⚠️ Model Card not available", style={"textAlign": "center", "color": "orange"}),
            html.P(
                "Run notebook Section 9.8 to generate",
                style={"textAlign": "center", "fontStyle": "italic", "color": "#666"},
            ),
        ],
        style={"padding": "50px", "border": "1px dashed #ccc", "margin": "10px"},
    )


def get_status_indicators():
    """Generate status indicators for header."""
    uncertainty_exists = (
        PROJECT_ROOT / "outputs" / "uncertainty" / "interval_width_by_bucket.html"
    ).exists()
    governance_exists = (PROJECT_ROOT / "outputs" / "governance" / "meta_error_map.html").exists()
    portfolio_exists = (PROJECT_ROOT / "outputs" / "portfolio" / "efficient_frontier.html").exists()

    return html.Div(
        id="status-indicators",
        children=[
            html.Span("✅ Basic Analysis", style={"margin": "0 10px", "color": "green"}),
            html.Span(
                "✅ Portfolio Optimization" if portfolio_exists else "⚠️ Portfolio Optimization",
                style={"margin": "0 10px", "color": "green" if portfolio_exists else "orange"},
            ),
            html.Span(
                "✅ Uncertainty Analysis" if uncertainty_exists else "⚠️ Uncertainty Analysis",
                style={"margin": "0 10px", "color": "green" if uncertainty_exists else "orange"},
            ),
            html.Span(
                "✅ Governance" if governance_exists else "⚠️ Governance",
                style={"margin": "0 10px", "color": "green" if governance_exists else "orange"},
            ),
        ],
        style={"textAlign": "center", "padding": "10px"},
    )


# Sample data loading (replace with actual data source)
def load_data():
    """Load prediction data from CSV file or return empty DataFrame.

    Returns:
        pd.DataFrame: Prediction data with all columns from predictions.csv
    """
    # Load from outputs or database
    # Use project root path for consistent resolution
    csv_path = PROJECT_ROOT / "outputs" / "analytics" / "predictions.csv"

    if csv_path.exists():
        all_stocks_phase95 = pd.read_csv(csv_path, low_memory=False)
        # Convert numeric columns to proper dtypes
        numeric_columns = [
            "market_cap",
            "last_price",
            "price_target",
            "predicted_price_target",
            "mispricing_score",
            "mispricing_pct",
            "prediction_error",
            "prediction_error_pct",
            "model_analyst_diff_pct",
            "p_e",
            "p_b",
            "roe",
            "roa",
            "ev_ebitda",
            "operating_margin",
            "net_margin",
            "debt_to_equity",
            "current_ratio",
        ]
        for col in numeric_columns:
            if col in all_stocks_phase95.columns:
                all_stocks_phase95[col] = pd.to_numeric(all_stocks_phase95[col], errors="coerce")
        return all_stocks_phase95
    # Return empty DataFrame with expected columns if file doesn't exist
    return pd.DataFrame(
        columns=[
            "ticker",
            "sector",
            "region",
            "country",
            "trading_country",
            "exchange",
            "unit",
            "industry",
            "style_class",
            "next_earnings_status",
            "size_class",
            "market_cap",
            "last_price",
            "price_target",
            "predicted_price_target",
            "mispricing_score",
            "mispricing_pct",
        ]
    )


df = load_data()

app = dash.Dash(__name__, title="Finance ML Analytics", external_stylesheets=[dbc.themes.DARKLY])
server = app.server


@server.route("/app_assets/<category>/<path:filename>")
def serve_artifacts(category, filename):
    return send_from_directory(PROJECT_ROOT / "outputs" / category, filename)


app.layout = html.Div(
    [
        html.H1("📊 Finance ML Analytics Dashboard", style={"textAlign": "center"}),
        get_status_indicators(),
        # KPI Summary Cards
        html.Div(
            id="kpi-cards",
            style={"display": "flex", "justifyContent": "space-around", "margin": "20px"},
        ),
        # Filters
        html.Div(
            [
                html.H4("Filters", style={"marginBottom": "10px", "color": "white"}),
                html.Div(
                    [
                        # Row 1
                        html.Div(
                            [
                                html.Label("Sector", className="filter-label"),
                                dcc.Dropdown(
                                    id="sector-dropdown",
                                    multi=True,
                                    options=(
                                        [
                                            {"label": i, "value": i}
                                            for i in sorted(df["sector"].dropna().unique())
                                        ]
                                        if "sector" in df.columns
                                        else []
                                    ),
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
                                    options=(
                                        [
                                            {"label": i, "value": i}
                                            for i in sorted(df["region"].dropna().unique())
                                        ]
                                        if "region" in df.columns
                                        else []
                                    ),
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
                                    options=(
                                        [
                                            {"label": i, "value": i}
                                            for i in sorted(df["country"].dropna().unique())
                                        ]
                                        if "country" in df.columns
                                        else []
                                    ),
                                ),
                            ],
                            className="filter-item",
                        ),
                        html.Div(
                            [
                                html.Label("Trading Country", className="filter-label"),
                                dcc.Dropdown(
                                    id="trading-country-dropdown",
                                    multi=True,
                                    options=(
                                        [
                                            {"label": i, "value": i}
                                            for i in sorted(df["trading_country"].dropna().unique())
                                        ]
                                        if "trading_country" in df.columns
                                        else []
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
                        # Row 2
                        html.Div(
                            [
                                html.Label("Industry", className="filter-label"),
                                dcc.Dropdown(
                                    id="industry-dropdown",
                                    multi=True,
                                    options=(
                                        [
                                            {"label": i, "value": i}
                                            for i in sorted(df["industry"].dropna().unique())
                                        ]
                                        if "industry" in df.columns
                                        else []
                                    ),
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
                                    options=(
                                        [
                                            {"label": i, "value": i}
                                            for i in sorted(df["style_class"].dropna().unique())
                                        ]
                                        if "style_class" in df.columns
                                        else []
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
                                    options=(
                                        [
                                            {"label": i, "value": i}
                                            for i in sorted(df["size_class"].dropna().unique())
                                        ]
                                        if "size_class" in df.columns
                                        else []
                                    ),
                                ),
                            ],
                            className="filter-item",
                        ),
                        html.Div(
                            [
                                html.Label("Next Earnings", className="filter-label"),
                                dcc.Dropdown(
                                    id="earnings-status-dropdown",
                                    multi=True,
                                    options=(
                                        [
                                            {"label": i, "value": i}
                                            for i in sorted(
                                                df["next_earnings_status"].dropna().unique()
                                            )
                                        ]
                                        if "next_earnings_status" in df.columns
                                        else []
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
                        # Row 3
                        html.Div(
                            [
                                html.Label("Exchange", className="filter-label"),
                                dcc.Dropdown(
                                    id="exchange-dropdown",
                                    multi=True,
                                    options=(
                                        [
                                            {"label": i, "value": i}
                                            for i in sorted(df["exchange"].dropna().unique())
                                        ]
                                        if "exchange" in df.columns
                                        else []
                                    ),
                                ),
                            ],
                            className="filter-item",
                        ),
                        html.Div(
                            [
                                html.Label("Unit", className="filter-label"),
                                dcc.Dropdown(
                                    id="unit-dropdown",
                                    multi=True,
                                    options=(
                                        [
                                            {"label": i, "value": i}
                                            for i in sorted(df["unit"].dropna().unique())
                                        ]
                                        if "unit" in df.columns
                                        else []
                                    ),
                                ),
                            ],
                            className="filter-item",
                        ),
                        html.Div(
                            [
                                html.Label("Model Version", className="filter-label"),
                                dcc.Dropdown(
                                    id="model-version-dropdown",
                                    options=[{"label": "v9_10", "value": "v9_10"}],
                                    value="v9_10",
                                ),
                            ],
                            className="filter-item",
                        ),
                    ],
                    className="filter-row",
                ),
            ],
            className="filter-container",
        ),
        # Tabs for different views
        dcc.Tabs(
            [
                dcc.Tab(
                    label="📈 Overview",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(id="scatter-plot"),
                                dcc.Graph(id="heatmap-plot"),
                                html.Hr(),
                                html.H3("Phase 9.3 Enhanced EDA", style={"textAlign": "center"}),
                                render_artifact_or_placeholder("eda", "correlation"),
                                render_artifact_or_placeholder("eda", "regional_radar"),
                                render_artifact_or_placeholder("eda", "sector_bubbles"),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="🎯 Prediction Analysis",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(id="prediction-error-plot"),
                                dcc.Graph(id="model-analyst-comparison"),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="📊 Stock Rankings",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3("Top Undervalued Stocks"),
                                        dash_table.DataTable(
                                            id="undervalued-table",
                                            page_size=10,
                                            style_table={"overflowX": "auto"},
                                            style_header={
                                                "backgroundColor": "rgb(30, 30, 30)",
                                                "color": "white",
                                                "fontWeight": "bold",
                                                "border": "1px solid #444",
                                            },
                                            style_data={
                                                "backgroundColor": "rgb(50, 50, 50)",
                                                "color": "white",
                                                "border": "1px solid #444",
                                            },
                                            style_data_conditional=[
                                                {
                                                    "if": {"row_index": "odd"},
                                                    "backgroundColor": "rgb(40, 40, 40)",
                                                }
                                            ],
                                        ),
                                    ],
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "padding": "10px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.H3("Top Overvalued Stocks"),
                                        dash_table.DataTable(
                                            id="overvalued-table",
                                            page_size=10,
                                            style_table={"overflowX": "auto"},
                                            style_header={
                                                "backgroundColor": "rgb(30, 30, 30)",
                                                "color": "white",
                                                "fontWeight": "bold",
                                                "border": "1px solid #444",
                                            },
                                            style_data={
                                                "backgroundColor": "rgb(50, 50, 50)",
                                                "color": "white",
                                                "border": "1px solid #444",
                                            },
                                            style_data_conditional=[
                                                {
                                                    "if": {"row_index": "odd"},
                                                    "backgroundColor": "rgb(40, 40, 40)",
                                                }
                                            ],
                                        ),
                                    ],
                                    style={
                                        "width": "48%",
                                        "float": "right",
                                        "display": "inline-block",
                                        "padding": "10px",
                                    },
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="🔬 Uncertainty & Calibration",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "Uncertainty Quantification & Conformal Calibration",
                                    style={"textAlign": "center", "padding": "20px"},
                                ),
                                render_artifact_or_placeholder("uncertainty", "interval_width"),
                                render_artifact_or_placeholder("uncertainty", "coverage_heatmap"),
                                render_artifact_or_placeholder(
                                    "uncertainty", "reliability_diagram"
                                ),
                                html.Hr(),
                                render_artifact_or_placeholder("calibration", "sector_bias"),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="🛡️ Safety Rails & Data Quality",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "Safety Rails & Data Quality Monitoring",
                                    style={"textAlign": "center", "padding": "20px"},
                                ),
                                render_artifact_or_placeholder("safety_rails", "winsorization"),
                                render_artifact_or_placeholder("safety_rails", "violations"),
                                render_artifact_or_placeholder("safety_rails", "sensitivity"),
                                html.Hr(),
                                render_artifact_or_placeholder("eda", "data_quality"),
                                render_artifact_or_placeholder("eda", "outliers"),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="🏛️ Model Governance",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "Model Governance & Lineage",
                                    style={"textAlign": "center", "padding": "20px"},
                                ),
                                render_artifact_or_placeholder("governance", "error_map"),
                                render_model_card(),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="💼 Portfolio & Risk Metrics",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "Portfolio Optimization & Risk Analysis",
                                    style={"textAlign": "center", "padding": "20px"},
                                ),
                                # Efficient Frontier
                                html.Div(
                                    [
                                        html.H3(
                                            "Efficient Frontier", style={"textAlign": "center"}
                                        ),
                                        (
                                            html.Iframe(
                                                id="efficient-frontier-iframe",
                                                src=(
                                                    "/assets/efficient_frontier_interactive.html"
                                                    if (
                                                        PROJECT_ROOT
                                                        / "outputs"
                                                        / "analytics"
                                                        / "efficient_frontier_interactive.html"
                                                    ).exists()
                                                    else ""
                                                ),
                                                style={
                                                    "width": "100%",
                                                    "height": "650px",
                                                    "border": "1px solid #ddd",
                                                },
                                            )
                                            if (
                                                PROJECT_ROOT
                                                / "outputs"
                                                / "analytics"
                                                / "efficient_frontier_interactive.html"
                                            ).exists()
                                            else html.Div(
                                                "⚠️ Run notebook Section 10 to generate portfolio optimization visualizations",
                                                style={
                                                    "textAlign": "center",
                                                    "padding": "50px",
                                                    "color": "orange",
                                                },
                                            )
                                        ),
                                    ],
                                    style={"padding": "20px"},
                                ),
                                # Risk Metrics Dashboard
                                html.Div(
                                    [
                                        html.H3(
                                            "Risk Metrics Dashboard", style={"textAlign": "center"}
                                        ),
                                        (
                                            html.Iframe(
                                                id="risk-metrics-iframe",
                                                src=(
                                                    "/assets/risk_metrics_dashboard.html"
                                                    if (
                                                        PROJECT_ROOT
                                                        / "outputs"
                                                        / "analytics"
                                                        / "risk_metrics_dashboard.html"
                                                    ).exists()
                                                    else ""
                                                ),
                                                style={
                                                    "width": "100%",
                                                    "height": "850px",
                                                    "border": "1px solid #ddd",
                                                },
                                            )
                                            if (
                                                PROJECT_ROOT
                                                / "outputs"
                                                / "analytics"
                                                / "risk_metrics_dashboard.html"
                                            ).exists()
                                            else html.Div(
                                                "⚠️ Risk metrics dashboard not available",
                                                style={
                                                    "textAlign": "center",
                                                    "padding": "50px",
                                                    "color": "orange",
                                                },
                                            )
                                        ),
                                    ],
                                    style={"padding": "20px"},
                                ),
                                # Drawdown Analysis
                                html.Div(
                                    [
                                        html.H3(
                                            "Portfolio Drawdown Analysis",
                                            style={"textAlign": "center"},
                                        ),
                                        (
                                            html.Iframe(
                                                id="drawdown-iframe",
                                                src=(
                                                    "/assets/portfolio_drawdown_analysis.html"
                                                    if (
                                                        PROJECT_ROOT
                                                        / "outputs"
                                                        / "analytics"
                                                        / "portfolio_drawdown_analysis.html"
                                                    ).exists()
                                                    else ""
                                                ),
                                                style={
                                                    "width": "100%",
                                                    "height": "750px",
                                                    "border": "1px solid #ddd",
                                                },
                                            )
                                            if (
                                                PROJECT_ROOT
                                                / "outputs"
                                                / "analytics"
                                                / "portfolio_drawdown_analysis.html"
                                            ).exists()
                                            else html.Div(
                                                "⚠️ Drawdown analysis not available",
                                                style={
                                                    "textAlign": "center",
                                                    "padding": "50px",
                                                    "color": "orange",
                                                },
                                            )
                                        ),
                                    ],
                                    style={"padding": "20px"},
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            "📝 Note: These visualizations are generated from Section 10 of the notebook.",
                                            style={
                                                "textAlign": "center",
                                                "fontStyle": "italic",
                                                "color": "#666",
                                            },
                                        ),
                                        html.P(
                                            "To update, run ml_finance_model_main.ipynb Section 10.",
                                            style={
                                                "textAlign": "center",
                                                "fontStyle": "italic",
                                                "color": "#666",
                                            },
                                        ),
                                    ],
                                    style={"padding": "20px"},
                                ),
                                # Phase 6 – additional interactive analytics (optional)
                                html.Hr(),
                                html.Div(
                                    [
                                        html.H3(
                                            "Phase 6 – Interactive Portfolio Analytics",
                                            style={"textAlign": "center"},
                                        ),
                                        html.P(
                                            "The following optional views are generated by the Phase 6 cells in Section 10 "
                                            + "(10.6 Interactive Dashboard). If the files are not present yet, run the corresponding "
                                            "notebook cells to create them.",
                                            style={
                                                "textAlign": "center",
                                                "fontStyle": "italic",
                                                "color": "#666",
                                            },
                                        ),
                                    ],
                                    style={"padding": "10px 20px 0 20px"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Multi-Period Performance Comparison",
                                            style={"textAlign": "center"},
                                        ),
                                        (
                                            html.Iframe(
                                                id="multi-period-comparison-iframe",
                                                src=(
                                                    "/assets/portfolio_multi_period_comparison.html"
                                                    if (
                                                        PROJECT_ROOT
                                                        / "outputs"
                                                        / "analytics"
                                                        / "portfolio_multi_period_comparison.html"
                                                    ).exists()
                                                    else ""
                                                ),
                                                style={
                                                    "width": "100%",
                                                    "height": "500px",
                                                    "border": "1px solid #ddd",
                                                },
                                            )
                                            if (
                                                PROJECT_ROOT
                                                / "outputs"
                                                / "analytics"
                                                / "portfolio_multi_period_comparison.html"
                                            ).exists()
                                            else html.Div(
                                                "⚠️ Multi-period comparison visualization not available. "
                                                "Run Section 10.6 in the notebook to generate it.",
                                                style={
                                                    "textAlign": "center",
                                                    "padding": "30px",
                                                    "color": "orange",
                                                },
                                            )
                                        ),
                                    ],
                                    style={"padding": "10px 20px"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Factor Exposure Dashboard",
                                            style={"textAlign": "center"},
                                        ),
                                        (
                                            html.Iframe(
                                                id="factor-exposure-iframe",
                                                src=(
                                                    "/assets/portfolio_factor_exposure_dashboard.html"
                                                    if (
                                                        PROJECT_ROOT
                                                        / "outputs"
                                                        / "analytics"
                                                        / "portfolio_factor_exposure_dashboard.html"
                                                    ).exists()
                                                    else ""
                                                ),
                                                style={
                                                    "width": "100%",
                                                    "height": "500px",
                                                    "border": "1px solid #ddd",
                                                },
                                            )
                                            if (
                                                PROJECT_ROOT
                                                / "outputs"
                                                / "analytics"
                                                / "portfolio_factor_exposure_dashboard.html"
                                            ).exists()
                                            else html.Div(
                                                "⚠️ Factor exposure dashboard not available. "
                                                "Run Section 10.6 in the notebook to generate it.",
                                                style={
                                                    "textAlign": "center",
                                                    "padding": "30px",
                                                    "color": "orange",
                                                },
                                            )
                                        ),
                                    ],
                                    style={"padding": "10px 20px"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Rebalancing Suggestions (Static Example)",
                                            style={"textAlign": "center"},
                                        ),
                                        (
                                            html.Iframe(
                                                id="rebalance-widget-iframe",
                                                src=(
                                                    "/assets/portfolio_rebalance_widget.html"
                                                    if (
                                                        PROJECT_ROOT
                                                        / "outputs"
                                                        / "analytics"
                                                        / "portfolio_rebalance_widget.html"
                                                    ).exists()
                                                    else ""
                                                ),
                                                style={
                                                    "width": "100%",
                                                    "height": "500px",
                                                    "border": "1px solid #ddd",
                                                },
                                            )
                                            if (
                                                PROJECT_ROOT
                                                / "outputs"
                                                / "analytics"
                                                / "portfolio_rebalance_widget.html"
                                            ).exists()
                                            else html.Div(
                                                "⚠️ Rebalance widget snapshot not available. "
                                                "Run Section 10.6 in the notebook to generate it.",
                                                style={
                                                    "textAlign": "center",
                                                    "padding": "30px",
                                                    "color": "orange",
                                                },
                                            )
                                        ),
                                    ],
                                    style={"padding": "10px 20px 30px 20px"},
                                ),
                            ]
                        )
                    ],
                ),
            ]
        ),
    ]
)


@app.callback(
    [
        Output("kpi-cards", "children"),
        Output("scatter-plot", "figure"),
        Output("heatmap-plot", "figure"),
        Output("prediction-error-plot", "figure"),
        Output("model-analyst-comparison", "figure"),
        Output("undervalued-table", "data"),
        Output("overvalued-table", "data"),
    ],
    [
        Input("sector-dropdown", "value"),
        Input("region-dropdown", "value"),
        Input("country-dropdown", "value"),
        Input("trading-country-dropdown", "value"),
        Input("industry-dropdown", "value"),
        Input("style-class-dropdown", "value"),
        Input("size-class-dropdown", "value"),
        Input("earnings-status-dropdown", "value"),
        Input("exchange-dropdown", "value"),
        Input("unit-dropdown", "value"),
    ],
)
def update_dashboard(
    sectors,
    regions,
    countries,
    trading_countries,
    industries,
    style_classes,
    size_classes,
    earnings_statuses,
    exchanges,
    units,
):
    """Update dashboard visualizations based on selected filters."""
    filtered_df = df.copy()

    if sectors:
        filtered_df = filtered_df[filtered_df["sector"].isin(sectors)]
    if regions:
        filtered_df = filtered_df[filtered_df["region"].isin(regions)]
    if countries:
        filtered_df = filtered_df[filtered_df["country"].isin(countries)]
    if trading_countries:
        filtered_df = filtered_df[filtered_df["trading_country"].isin(trading_countries)]
    if industries:
        filtered_df = filtered_df[filtered_df["industry"].isin(industries)]
    if style_classes:
        filtered_df = filtered_df[filtered_df["style_class"].isin(style_classes)]
    if size_classes:
        filtered_df = filtered_df[filtered_df["size_class"].isin(size_classes)]
    if earnings_statuses:
        filtered_df = filtered_df[filtered_df["next_earnings_status"].isin(earnings_statuses)]
    if exchanges:
        filtered_df = filtered_df[filtered_df["exchange"].isin(exchanges)]
    if units:
        filtered_df = filtered_df[filtered_df["unit"].isin(units)]

    # KPI Cards
    kpi_cards = []
    if not filtered_df.empty:
        total_stocks = len(filtered_df)
        avg_mispricing = (
            filtered_df["mispricing_score"].mean()
            if "mispricing_score" in filtered_df.columns
            else 0
        )
        sectors_count = filtered_df["sector"].nunique() if "sector" in filtered_df.columns else 0
        regions_count = filtered_df["region"].nunique() if "region" in filtered_df.columns else 0

        kpi_cards = [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Total Stocks", className="card-title"),
                        html.H2(f"{total_stocks:,}", className="card-text"),
                    ]
                ),
                color="primary",
                inverse=True,
                style={"width": "23%", "textAlign": "center"},
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Avg Mispricing", className="card-title"),
                        html.H2(f"{avg_mispricing:.2%}", className="card-text"),
                    ]
                ),
                color="success" if avg_mispricing > 0 else "danger",
                inverse=True,
                style={"width": "23%", "textAlign": "center"},
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Sectors", className="card-title"),
                        html.H2(f"{sectors_count}", className="card-text"),
                    ]
                ),
                color="info",
                inverse=True,
                style={"width": "23%", "textAlign": "center"},
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Regions", className="card-title"),
                        html.H2(f"{regions_count}", className="card-text"),
                    ]
                ),
                color="info",
                inverse=True,
                style={"width": "23%", "textAlign": "center"},
            ),
        ]

    # Scatter plot - Mispricing vs Market Cap
    if "market_cap" in filtered_df.columns and "mispricing_score" in filtered_df.columns:
        scatter_fig = px.scatter(
            filtered_df,
            x="market_cap",
            y="mispricing_score",
            color="sector" if "sector" in filtered_df.columns else None,
            size="last_price" if "last_price" in filtered_df.columns else None,
            hover_data=(
                ["ticker", "name"]
                if all(c in filtered_df.columns for c in ["ticker", "name"])
                else None
            ),
            title="Mispricing Score vs Market Cap",
            labels={"market_cap": "Market Cap", "mispricing_score": "Mispricing Score"},
            template="plotly_dark",
        )
        scatter_fig.update_layout(font_family="Arial")
    else:
        scatter_fig = {}

    # Heatmap - Sector-Region Performance
    if all(c in filtered_df.columns for c in ["sector", "region", "mispricing_score"]):
        pivot = filtered_df.pivot_table(
            values="mispricing_score", index="sector", columns="region", aggfunc="mean"
        )
        heatmap_fig = px.imshow(
            pivot,
            text_auto=".2f",
            title="Sector-Region Performance Heatmap",
            labels={"color": "Avg Mispricing Score"},
            color_continuous_scale="RdYlGn",
            template="plotly_dark",
        )
        heatmap_fig.update_layout(font_family="Arial")
    else:
        heatmap_fig = {}

    # Prediction Error Plot
    if "prediction_error_pct" in filtered_df.columns:
        error_fig = px.histogram(
            filtered_df,
            x="prediction_error_pct",
            nbins=50,
            title="Distribution of Prediction Errors (%)",
            labels={"prediction_error_pct": "Prediction Error (%)"},
            template="plotly_dark",
            color_discrete_sequence=["#375a7f"],  # Primary color
        )
        error_fig.add_vline(
            x=0, line_dash="dash", line_color="#e74c3c", annotation_text="Zero Error"
        )  # Danger color
        error_fig.update_layout(font_family="Arial")
    else:
        error_fig = {}

    # Model vs Analyst Comparison
    if all(c in filtered_df.columns for c in ["price_target", "predicted_price_target"]):
        comparison_fig = px.scatter(
            filtered_df,
            x="price_target",
            y="predicted_price_target",
            color="sector" if "sector" in filtered_df.columns else None,
            hover_data=(
                ["ticker", "name"]
                if all(c in filtered_df.columns for c in ["ticker", "name"])
                else None
            ),
            title="Model Predictions vs Analyst Targets",
            labels={"price_target": "Analyst Target", "predicted_price_target": "Model Prediction"},
            template="plotly_dark",
        )
        comparison_fig.update_layout(font_family="Arial")
        # Add perfect agreement line
        if not filtered_df.empty:
            min_val = min(
                filtered_df["price_target"].min(), filtered_df["predicted_price_target"].min()
            )
            max_val = max(
                filtered_df["price_target"].max(), filtered_df["predicted_price_target"].max()
            )
            comparison_fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color="#e74c3c", dash="dash"),  # Danger color
                    name="Perfect Agreement",
                    showlegend=True,
                )
            )
    else:
        comparison_fig = {}

    # Top undervalued stocks
    undervalued_stocks = []
    if "mispricing_score" in filtered_df.columns and not filtered_df.empty:
        valid_df = filtered_df.dropna(subset=["mispricing_score"])
        if not valid_df.empty:
            cols_to_show = [
                "ticker",
                "sector",
                "mispricing_score",
                "last_price",
                "predicted_price_target",
            ]
            cols_available = [c for c in cols_to_show if c in valid_df.columns]
            undervalued_stocks = valid_df.nlargest(10, "mispricing_score")[cols_available].to_dict(
                "records"
            )

    # Top overvalued stocks
    overvalued_stocks = []
    if "mispricing_score" in filtered_df.columns and not filtered_df.empty:
        valid_df = filtered_df.dropna(subset=["mispricing_score"])
        if not valid_df.empty:
            cols_to_show = [
                "ticker",
                "sector",
                "mispricing_score",
                "last_price",
                "predicted_price_target",
            ]
            cols_available = [c for c in cols_to_show if c in valid_df.columns]
            overvalued_stocks = valid_df.nsmallest(10, "mispricing_score")[cols_available].to_dict(
                "records"
            )

    return (
        kpi_cards,
        scatter_fig,
        heatmap_fig,
        error_fig,
        comparison_fig,
        undervalued_stocks,
        overvalued_stocks,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
