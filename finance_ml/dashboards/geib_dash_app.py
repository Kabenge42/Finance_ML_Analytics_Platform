"""
Global Equity Investment Board (GEIB) Dashboard
Loads data from analytics.expected_returns_summary table (import from postgres.analytics.expected_returns_summary)
Run: python finance_ml/dashboards/geib_dash_app.py

Environment Variable Required: GEIB_DASHBOARD=true
"""

import os
from datetime import datetime
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import dcc, html, Input, Output, State, dash_table
from dash.dash_table.Format import Format, Scheme, Symbol
from flask import send_from_directory
from sqlalchemy import create_engine

# Import probabilistic visualizations
try:
    from finance_ml.analytics.visualizations import probability_viz
    from finance_ml.analytics.statistical_analysis import bayesian_category_analysis
    PROB_VIZ_AVAILABLE = True
except ImportError:
    PROB_VIZ_AVAILABLE = False
    print("⚠️ Probabilistic visualizations not available")

# Project root path for consistent path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent

# =============================================================================
# Table Formatting Configuration
# =============================================================================

TABLE_STYLE_HEADER = {
    "backgroundColor": "#375a7f",  # Primary Blue
    "color": "#ffffff",
    "fontWeight": "bold",
    "border": "1px solid #444",
    "textAlign": "center",
}

TABLE_STYLE_CELL = {
    "backgroundColor": "#222",
    "color": "#ffffff",
    "border": "1px solid #444",
    "padding": "8px",
    "fontFamily": "Arial",
    "fontSize": "13px",
}

TABLE_STYLE_DATA_CONDITIONAL = [
    {"if": {"row_index": "odd"}, "backgroundColor": "#2c2c2c"},
    # Conditional formatting for upside/return columns (0-100 scale)
    {
        "if": {"column_id": "expected_upside_pct", "filter_query": "{expected_upside_pct} > 15"},
        "color": "#00bc8c", "fontWeight": "bold"  # Success Green
    },
    {
        "if": {"column_id": "expected_upside_pct", "filter_query": "{expected_upside_pct} < 0"},
        "color": "#e74c3c"  # Danger Red
    },
    {
        "if": {"column_id": "filtered_upside", "filter_query": "{filtered_upside} > 15"},
        "color": "#00bc8c", "fontWeight": "bold"
    },
    {
        "if": {"column_id": "filtered_upside", "filter_query": "{filtered_upside} < 0"},
        "color": "#e74c3c"
    },
    {
        "if": {"column_id": "expected_return_prob_weighted", "filter_query": "{expected_return_prob_weighted} > 10"},
        "color": "#00bc8c", "fontWeight": "bold"
    },
    # Conditional formatting for probability (0-1 scale)
    {
        "if": {"column_id": "achievement_probability", "filter_query": "{achievement_probability} > 0.7"},
        "color": "#00bc8c", "fontWeight": "bold"
    },
    {
        "if": {"column_id": "achievement_probability", "filter_query": "{achievement_probability} < 0.4"},
        "color": "#f39c12"  # Warning Orange
    },
]


def get_formatted_columns(cols_list):
    """Return column definitions with specific formatting rules."""
    formatted = []
    for col in cols_list:
        spec = {"id": col, "name": col.replace("_", " ").capitalize()}

        # Currency formatting
        if col in ["last_price", "price_target", "market_cap", "enterprise_value"]:
            spec.update({
                "type": "numeric",
                "format": Format(precision=2, scheme=Scheme.fixed, group=True).symbol(Symbol.yes).symbol_prefix("$")
            })

        # Percentage formatting (for columns already in 0-100 scale)
        elif col in ["expected_upside_pct", "filtered_upside", "expected_return_prob_weighted"]:
            spec.update({
                "type": "numeric",
                "format": Format(precision=2, scheme=Scheme.fixed).symbol(Symbol.yes).symbol_suffix("%")
            })

        # Percentage formatting (for probability columns in 0-1 scale)
        elif col in ["achievement_probability", "posterior_beat_prob", "confidence_score"]:
            spec.update({
                "type": "numeric",
                "format": Format(precision=2, scheme=Scheme.percentage)
            })

        formatted.append(spec)
    return formatted


# =============================================================================
# Filter Configuration (Single Source of Truth)
# =============================================================================

FILTER_CONFIG = [
    {"label": "Region",              "id": "region-dropdown",              "column": "region",              "width": "23%"},
    {"label": "Country",             "id": "country-dropdown",             "column": "country",             "width": "23%"},
    {"label": "Exchange",            "id": "exchange-dropdown",            "column": "exchange",            "width": "23%"},
    {"label": "Sector",              "id": "sector-dropdown",              "column": "sector",              "width": "23%"},
    {"label": "Industry",            "id": "industry-dropdown",            "column": "industry",            "width": "23%"},
    {"label": "Signal",              "id": "signal-dropdown",              "column": "signal",              "width": "23%"},
    {"label": "Trading Country",     "id": "trading-country-dropdown",     "column": "trading_country",     "width": "18%"},
    {"label": "Style Class",         "id": "style-class-dropdown",         "column": "style_class",         "width": "18%"},
    {"label": "Size Class",          "id": "size-class-dropdown",          "column": "size_class",          "width": "18%"},
    {"label": "Unit",                "id": "unit-dropdown",                "column": "unit",                "width": "18%"},
    {"label": "Beat Classification", "id": "beat-classification-dropdown", "column": "beat_classification", "width": "18%"},
    {"label": "Confidence Level",    "id": "confidence-dropdown",          "column": "confidence_level",    "width": "23%"},
]

# All filter dropdown IDs (for callbacks)
ALL_FILTER_IDS = [f["id"] for f in FILTER_CONFIG]
ALL_FILTER_COLUMNS = [f["column"] for f in FILTER_CONFIG]


def build_filter_options(dataframe: pd.DataFrame, column: str) -> list:
    """Build sorted dropdown options from a DataFrame column."""
    if column in dataframe.columns and len(dataframe) > 0:
        return [{"label": v, "value": v} for v in sorted(dataframe[column].dropna().unique())]
    return []


def build_filter_panel(dataframe: pd.DataFrame) -> html.Div:
    """Build the entire filter panel from FILTER_CONFIG."""
    dropdowns = []
    for f in FILTER_CONFIG:
        dropdowns.append(
            html.Div(
                [
                    html.Label(f["label"], className="filter-label"),
                    dcc.Dropdown(
                        id=f["id"],
                        multi=True,
                        options=build_filter_options(dataframe, f["column"]),
                        style={
                            "backgroundColor": "#333",
                            "color": "white",
                        },
                        className="custom-dropdown",
                    ),
                ],
                className="filter-item",
                style={"width": f["width"], "display": "inline-block", "margin": "5px"},
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.H4(
                        "Filters",
                        style={"marginBottom": "10px", "color": "white", "display": "inline-block"},
                    ),
                    html.Button(
                        "Reset Filters",
                        id="reset-filters-btn",
                        n_clicks=0,
                        style={
                            "marginLeft": "20px",
                            "backgroundColor": "#e74c3c",
                            "color": "white",
                            "border": "none",
                            "padding": "5px 15px",
                            "borderRadius": "3px",
                            "cursor": "pointer",
                        },
                    ),
                ]
            ),
            html.Div(
                dropdowns,
                style={"display": "flex", "flexWrap": "wrap", "justifyContent": "space-around"},
            ),
        ],
        style={
            "padding": "20px",
            "backgroundColor": "#222",
            "margin": "10px",
            "borderRadius": "5px",
        },
    )


def apply_global_filters(dataframe: pd.DataFrame, filter_values: dict) -> pd.DataFrame:
    """
    Apply all global filters to a DataFrame consistently.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to filter.
    filter_values : dict
        Mapping of column name -> selected values (list or None).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    filtered = dataframe.copy()
    for column, values in filter_values.items():
        if values and column in filtered.columns:
            filtered = filtered[filtered[column].isin(values)]
    return filtered


def collect_filter_values(*args) -> dict:
    """
    Zip filter arguments (in FILTER_CONFIG order) into a {column: values} dict.

    Usage inside a callback that receives all 12 filter inputs first:
        filter_values = collect_filter_values(*args[:12])
    """
    return {
        cfg["column"]: val
        for cfg, val in zip(FILTER_CONFIG, args)
    }

# =============================================================================
# Kelly Criterion Position Sizing
# =============================================================================

# Kelly-specific dropdown option definitions
KELLY_FRACTION_OPTIONS = [
    {"label": "Full Kelly (1.0)", "value": 1.0},
    {"label": "Half Kelly (0.5)", "value": 0.5},
    {"label": "Quarter Kelly (0.25)", "value": 0.25},
    {"label": "Eighth Kelly (0.125)", "value": 0.125},
]

MAX_POSITION_OPTIONS = [
    {"label": "5%", "value": 0.05},
    {"label": "10%", "value": 0.10},
    {"label": "15%", "value": 0.15},
    {"label": "20%", "value": 0.20},
    {"label": "No cap", "value": "no_cap"},
]

KELLY_ADJUSTMENT_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "Confidence-weighted", "value": "confidence"},
    {"label": "Achievement-weighted", "value": "achievement"},
    {"label": "Both", "value": "both"},
]

KELLY_MIN_CONFIDENCE_OPTIONS = [
    {"label": "0.15", "value": 0.15},
    {"label": "0.25", "value": 0.25},
    {"label": "0.35", "value": 0.35},
    {"label": "0.45", "value": 0.45},
]

KELLY_BAR_COLOR_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "Sector", "value": "sector"},
    {"label": "Confidence Level", "value": "confidence_level"},
]

KELLY_SCATTER_COLOR_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "Confidence Level", "value": "confidence_level"},
]

KELLY_SCATTER_SIZE_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "Achievement Probability", "value": "achievement_probability"},
]


def calculate_kelly_metrics(
        dataframe: pd.DataFrame,
        kelly_fraction: float = 0.25,
        max_position: str | float = 0.10,
        adjustment_method: str = "both",
) -> pd.DataFrame:
    """
    Calculate Kelly Criterion metrics for each position.

    Uses columns from analytics.expected_returns_summary:
      - prob_positive_upside  (0-100 scale)
      - filtered_upside       (percentage)
      - confidence_score      (0-1 scale)
      - achievement_probability (0-1 scale)

    Returns the DataFrame with added columns:
      kelly_raw, kelly_fractional, kelly_adjusted, kelly_pct
    """
    result = dataframe.copy()

    for col in ["prob_positive_upside", "filtered_upside", "confidence_score", "achievement_probability"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    # Kelly formula: f* = (p*b - q) / b  where p=win prob, q=1-p, b=win/loss ratio
    p = result["prob_positive_upside"] / 100.0
    q = 1.0 - p
    b = result["filtered_upside"] / 100.0

    result["kelly_raw"] = np.where(b != 0, (p * b - q) / b, 0)
    result["kelly_raw"] = result["kelly_raw"].clip(lower=0)

    # Apply fractional Kelly
    result["kelly_fractional"] = result["kelly_raw"] * kelly_fraction

    # Apply adjustment method
    if adjustment_method == "confidence":
        result["kelly_adjusted"] = result["kelly_fractional"] * result["confidence_score"]
    elif adjustment_method == "achievement":
        result["kelly_adjusted"] = result["kelly_fractional"] * result["achievement_probability"]
    elif adjustment_method == "both":
        result["kelly_adjusted"] = (
                result["kelly_fractional"] * result["confidence_score"] * result["achievement_probability"]
        )
    else:
        result["kelly_adjusted"] = result["kelly_fractional"]

    # Apply max position cap
    if max_position != "no_cap":
        result["kelly_adjusted"] = result["kelly_adjusted"].clip(upper=float(max_position))

    # Normalize to portfolio percentage
    total_kelly = result["kelly_adjusted"].sum()
    result["kelly_pct"] = (result["kelly_adjusted"] / total_kelly * 100.0) if total_kelly > 0 else 0

    return result


# =============================================================================
# Artifact Helper Functions
# =============================================================================


def get_artifact_path(artifact_name: str, artifact_type: str = "html") -> Path:
    """
    Get the path to an artifact file in the outputs/analytics directory.

    Parameters
    ----------
    artifact_name : str
        Name of the artifact file (without extension).
    artifact_type : str, default "html"
        File extension/type of the artifact.

    Returns
    -------
    Path
        Full path to the artifact file.
    """
    artifacts_dir = PROJECT_ROOT / "outputs" / "analytics"
    return artifacts_dir / f"{artifact_name}.{artifact_type}"


def render_artifact_or_placeholder(
    artifact_name: str, title: str = "Artifact", artifact_type: str = "html"
) -> html.Div:
    """
    Render an artifact if it exists, otherwise return a placeholder.

    Parameters
    ----------
    artifact_name : str
        Name of the artifact file (without extension).
    title : str, default "Artifact"
        Title to display above the artifact or in the placeholder.
    artifact_type : str, default "html"
        File extension/type of the artifact.

    Returns
    -------
    html.Div
        Dash HTML component containing either the artifact iframe or a placeholder.
    """
    artifact_path = get_artifact_path(artifact_name, artifact_type)

    if artifact_path.exists():
        # Return an iframe to display the HTML artifact
        relative_path = f"/artifacts/{artifact_name}.{artifact_type}"
        return html.Div(
            [
                html.H4(title, style={"textAlign": "center", "marginBottom": "10px"}),
                html.Iframe(
                    src=relative_path,
                    style={
                        "width": "100%",
                        "height": "600px",
                        "border": "1px solid #444",
                        "borderRadius": "5px",
                    },
                ),
            ]
        )
    else:
        # Return a placeholder indicating the artifact is not available
        return html.Div(
            [
                html.H4(title, style={"textAlign": "center", "marginBottom": "10px"}),
                dbc.Alert(
                    [
                        html.I(className="fas fa-info-circle", style={"marginRight": "10px"}),
                        f"Artifact '{artifact_name}.{artifact_type}' not found. ",
                        "Run the analytics pipeline to generate this artifact.",
                    ],
                    color="warning",
                    style={"textAlign": "center"},
                ),
            ],
            style={"padding": "20px"},
        )


def load_geib_data():
    """Load all necessary data for the GEIB dashboard.
    
    Returns:
        dict: Dictionary containing DataFrames for different components
    """
    data = {
        "summary": pd.DataFrame(),
        "tri_model": pd.DataFrame(),
        "earnings": pd.DataFrame(),
        "credit": pd.DataFrame(),
        "model_confidence": pd.DataFrame(),
    }

    # Check if GEIB_DASHBOARD environment variable is set
    if not os.environ.get("GEIB_DASHBOARD", "").lower() == "true":
        print("⚠️ GEIB_DASHBOARD environment variable not set to 'true'")
        return data

    db_url = os.environ.get("DB_URL")
    if not db_url:
        print("⚠️ DB_URL environment variable not set")
        return data

    try:
        engine = create_engine(db_url)

        # 1. Summary Data
        query_summary = """
        SELECT *
        FROM analytics.expected_returns_summary
        WHERE expected_upside_pct IS NOT NULL AND volume_shrs IS NOT NULL 
        ORDER BY expected_return_prob_weighted DESC
        """
        data['summary'] = pd.read_sql(query_summary, engine)

        # Convert numeric columns for summary
        numeric_cols = [
            'prob_positive_upside', 'last_price', 'expected_upside_pct',
            'filtered_upside', 'expected_return_prob_weighted',
            'achievement_probability', 'posterior_beat_prob', 
            'confidence_score', 'agreement_score'
        ]
        for col in numeric_cols:
            if col in data['summary'].columns:
                data['summary'][col] = pd.to_numeric(data['summary'][col], errors='coerce')

        # 2. Tri-Model Data (subset of columns needed for viz)
        query_tri = "SELECT * FROM analytics.expected_returns_summary "
        try:
            data['tri_model'] = pd.read_sql(query_tri, engine)
        except Exception:
            print("⚠️ analytics.expected_returns_summary not found, using summary fallback")
            data['tri_model'] = data['summary']

        # 3. Earnings Probability Data
        query_earnings = "SELECT * FROM analytics.earnings_probability_analysis "
        try:
            data['earnings'] = pd.read_sql(query_earnings, engine)
        except Exception:
            print("⚠️ analytics.earnings_probability_analysis not found")

        # 4. Credit Risk Data
        query_credit = "SELECT * FROM analytics.credit_risk_analysis "
        try:
            data['credit'] = pd.read_sql(query_credit, engine)
        except Exception:
            print("⚠️ analytics.credit_risk_analysis not found")

        # 5. Model Confidence Metrics (for governance tab)
        query_confidence = "SELECT * FROM analytics.model_confidence_metrics"
        try:
            data["model_confidence"] = pd.read_sql(query_confidence, engine)
        except Exception:
            print("⚠️ analytics.model_confidence_metrics not found")

        print(f"✓ Loaded GEIB data successfully")
        return data

    except Exception as e:
        print(f"❌ Error loading GEIB data: {e}")
        return data


# Load all data
all_data = load_geib_data()
df = all_data['summary']
df_tri = all_data['tri_model']
df_earnings = all_data['earnings']
df_credit = all_data['credit']
df_confidence = all_data["model_confidence"]

# Initialize Dash app
app = dash.Dash(
    __name__,
    title="Global Equity Analytics Dashboard",
    external_stylesheets=[dbc.themes.DARKLY]
)
server = app.server

# Add custom CSS for dropdown selection highlighting
app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                /* Highlight selected values in multi-dropdown */
                .Select-value {
                    background-color: #111 !important;
                    border: 1px solid #444 !important;
                    color: white !important;
                }
                .Select-value-icon {
                    border-right: 1px solid #444 !important;
                    color: #e74c3c !important;
                }
                .Select-value-icon:hover {
                    background-color: #e74c3c !important;
                    color: white !important;
                }
                .filter-label {
                    color: #adb5bd;
                    font-weight: bold;
                    font-size: 0.9rem;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

# Layout
app.layout = html.Div(
    [
        html.H1("🌍 Global Equity Analytics Dashboard (GEIB)", style={"textAlign": "center"}),
        html.P(
            "Expected Returns Analysis from Tri-Model Consensus (Monte Carlo, Kalman Filter, Price Target Achievement)",
            style={"textAlign": "center", "fontStyle": "italic", "color": "#999"},
        ),
        # Status indicator
        html.Div(
            id="status-indicator",
            children=[
                html.Span(
                    f"✅ Data Loaded: {len(df):,} stocks" if len(df) > 0 else "⚠️ No data loaded",
                    style={"margin": "0 10px", "color": "green" if len(df) > 0 else "orange"},
                )
            ],
            style={"textAlign": "center", "padding": "10px"},
        ),
        # KPI Summary Cards
        html.Div(
            id="kpi-cards",
            style={"display": "flex", "justifyContent": "space-around", "margin": "20px"},
        ),
        # Filters — generated from FILTER_CONFIG
        build_filter_panel(df),
        # Tabs for different views
        dcc.Tabs(
            [
                dcc.Tab(
                    label="📊 Expected Returns Overview",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(id="returns-scatter"),
                                # Price Target vs Current Price Scatter Controls
                                html.Div(
                                    [
                                        html.H4("Price Target vs Current Price", style={"textAlign": "center", "marginBottom": "10px"}),
                                        html.P(
                                            "Scatter plot showing analyst price targets versus current stock prices. Points above the diagonal line indicate upside potential.",
                                            style={"textAlign": "center", "fontStyle": "italic", "color": "#999", "marginBottom": "15px"},
                                        ),
                                        html.Div(
                                            style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px", "justifyContent": "center"},
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Label("Size Encoding:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block", "color": "white"}),
                                                        dcc.Dropdown(
                                                            id="pt-scatter-size-control",
                                                            options=[
                                                                {"label": "Expected Upside %", "value": "expected_upside_pct"},
                                                                {"label": "Market Cap", "value": "market_cap"},
                                                                {"label": "Volume", "value": "volume_shrs"},
                                                                {"label": "None", "value": "none"},
                                                            ],
                                                            value="expected_upside_pct",
                                                            style={"minWidth": "200px", "color": "black"},
                                                            searchable=False,
                                                        ),
                                                    ],
                                                    style={"display": "flex", "flexDirection": "column", "marginRight": "15px"},
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Label("Color Encoding:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block", "color": "white"}),
                                                        dcc.Dropdown(
                                                            id="pt-scatter-color-control",
                                                            options=[
                                                                {"label": "Sector", "value": "sector"},
                                                                {"label": "Confidence Level", "value": "confidence_level"},
                                                                {"label": "Beat Classification", "value": "beat_classification"},
                                                                {"label": "None", "value": "none"},
                                                            ],
                                                            value="sector",
                                                            style={"minWidth": "200px", "color": "black"},
                                                            searchable=False,
                                                        ),
                                                    ],
                                                    style={"display": "flex", "flexDirection": "column", "marginRight": "15px"},
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Label("Last Price Range:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block", "color": "white"}),
                                                        dcc.RangeSlider(
                                                            id="pt-scatter-last-price-slider",
                                                            min=0,
                                                            max=1000,
                                                            step=10,
                                                            value=[0, 1000],
                                                            marks={0: "$0", 250: "$250", 500: "$500", 750: "$750", 1000: "$1000+"},
                                                        ),
                                                    ],
                                                    style={"display": "flex", "flexDirection": "column", "marginRight": "15px", "minWidth": "250px"},
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Label("Price Target Range:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block", "color": "white"}),
                                                        dcc.RangeSlider(
                                                            id="pt-scatter-price-target-slider",
                                                            min=0,
                                                            max=1000,
                                                            step=10,
                                                            value=[0, 1000],
                                                            marks={0: "$0", 250: "$250", 500: "$500", 750: "$750", 1000: "$1000+"},
                                                        ),
                                                    ],
                                                    style={"display": "flex", "flexDirection": "column", "marginRight": "15px", "minWidth": "250px"},
                                                ),
                                            ],
                                        ),
                                    ],
                                    style={"padding": "15px", "backgroundColor": "#333", "borderRadius": "5px", "margin": "10px"},
                                ),
                                dcc.Loading(
                                    type="circle",
                                    children=[dcc.Graph(id="price_target_vs_current_scatter", style={"minHeight": "550px"})],
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="🎯 Model Consensus",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(id="model-signals-plot"),
                                dcc.Graph(id="confidence-distribution"),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="🏆 Top Opportunities",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "High Conviction Opportunities", style={"textAlign": "center", "margin": "20px"}
                                ),
                                dash_table.DataTable(
                                    id="top-opportunities-table",
                                    columns=get_formatted_columns([
                                        "ticker", "name", "sector", "last_price", "price_target",
                                        "expected_upside_pct", "filtered_upside",
                                        "expected_return_prob_weighted", "achievement_probability",
                                        "confidence_level", "signal"
                                    ]),
                                    page_size=15,
                                    sort_action="native",
                                    filter_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_header=TABLE_STYLE_HEADER,
                                    style_cell=TABLE_STYLE_CELL,
                                    style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="📈 Signal Analysis",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(id="signal-breakdown"),
                                dcc.Graph(id="regional-performance"),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="🏆 Risk-Adjusted Ranking",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Expected Value Risk-Adjusted Ranking",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Rank stocks by their risk-adjusted expected value, combining upside potential, probability of success, and confidence levels.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # Ranking Filters
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Scoring Method", style={"color": "white"}
                                                ),
                                                dcc.Dropdown(
                                                    id="scoring-method-dropdown",
                                                    options=[
                                                        {"label": "Base EV", "value": "base_ev"},
                                                        {
                                                            "label": "Probability-weighted",
                                                            "value": "prob_weighted",
                                                        },
                                                        {
                                                            "label": "Confidence-adjusted",
                                                            "value": "confidence_adj",
                                                        },
                                                        {
                                                            "label": "Achievement-adjusted",
                                                            "value": "achievement_adj",
                                                        },
                                                        {"label": "Combined", "value": "combined"},
                                                    ],
                                                    value="combined",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "18%",
                                                "display": "inline-block",
                                                "margin": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Min Agreement", style={"color": "white"}
                                                ),
                                                dcc.Dropdown(
                                                    id="min-agreement-dropdown",
                                                    options=[
                                                        {"label": str(i), "value": i}
                                                        for i in [0, 1, 2, 3, 4]
                                                    ],
                                                    value=2,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "18%",
                                                "display": "inline-block",
                                                "margin": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Min Confidence", style={"color": "white"}
                                                ),
                                                dcc.Dropdown(
                                                    id="min-confidence-dropdown",
                                                    options=[
                                                        {"label": str(i), "value": i}
                                                        for i in [0.15, 0.25, 0.35, 0.45]
                                                    ],
                                                    value=0.25,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "18%",
                                                "display": "inline-block",
                                                "margin": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Risk-Free Rate", style={"color": "white"}
                                                ),
                                                dcc.Dropdown(
                                                    id="risk-free-rate-dropdown",
                                                    options=[
                                                        {"label": f"{i}%", "value": float(i)}
                                                        for i in [0, 2, 3, 4, 5]
                                                    ],
                                                    value=3.0,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "18%",
                                                "display": "inline-block",
                                                "margin": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Scatter Color By", style={"color": "white"}
                                                ),
                                                dcc.Dropdown(
                                                    id="scatter-color-dropdown",
                                                    options=[
                                                        {"label": "None", "value": "none"},
                                                        {"label": "Signal", "value": "signal"},
                                                    ],
                                                    value="signal",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "18%",
                                                "display": "inline-block",
                                                "margin": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Select Tickers (Probabilistic)",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="prob-ticker-dropdown",
                                                    multi=True,
                                                    options=(
                                                        [
                                                            {"label": i, "value": i}
                                                            for i in sorted(
                                                                df["ticker"].dropna().unique()
                                                            )
                                                        ]
                                                        if "ticker" in df.columns and len(df) > 0
                                                        else []
                                                    ),
                                                    placeholder="Select tickers for detailed analysis...",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "97%",
                                                "display": "inline-block",
                                                "margin": "10px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#333",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "margin": "10px",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col([dcc.Graph(id="ranking-bar-chart")], width=6),
                                        dbc.Col([dcc.Graph(id="risk-reward-scatter")], width=6),
                                    ]
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="🔮 Probabilistic Analysis",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Bayesian Probabilistic Analysis",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "ArviZ-enhanced visualizations for posterior returns, beat probabilities, and ruin diagnostics.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col([dcc.Graph(id="posterior-forest-plot")], width=6),
                                        dbc.Col([dcc.Graph(id="tri-model-posterior")], width=6),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col([dcc.Graph(id="beat-prob-posterior")], width=6),
                                        dbc.Col([dcc.Graph(id="ruin-diagnostic")], width=6),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col([dcc.Graph(id="category-ridge-plot")], width=12),
                                    ]
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
                                html.H3(
                                    "Model Uncertainty & Calibration Analysis",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Assess model confidence calibration, prediction intervals, and uncertainty quantification.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col([dcc.Graph(id="calibration-curve")], width=6),
                                        dbc.Col(
                                            [dcc.Graph(id="prediction-interval-coverage")], width=6
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="uncertainty-distribution")], width=6
                                        ),
                                        dbc.Col([dcc.Graph(id="model-agreement-heatmap")], width=6),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [html.Div(id="calibration-metrics-display")], width=12
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="🛡️ Safety Rails & Data Quality",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Data Quality & Safety Rails",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Monitor data quality metrics, missing values, outliers, and safety thresholds.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col([dcc.Graph(id="data-completeness-chart")], width=6),
                                        dbc.Col([dcc.Graph(id="outlier-detection-chart")], width=6),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col([dcc.Graph(id="data-freshness-chart")], width=6),
                                        dbc.Col([dcc.Graph(id="safety-threshold-chart")], width=6),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col([html.Div(id="data-quality-summary")], width=12),
                                    ]
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="🏛️ Model Governance",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Model Governance & Audit Trail",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Track model versions, performance metrics over time, and governance documentation.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col([dcc.Graph(id="model-performance-trend")], width=6),
                                        dbc.Col([dcc.Graph(id="model-drift-chart")], width=6),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        html.H4(
                                                            "Model Registry",
                                                            style={"marginTop": "20px"},
                                                        ),
                                                        dash_table.DataTable(
                                                            id="model-registry-table",
                                                            page_size=10,
                                                            style_table={"overflowX": "auto"},
                                                            style_header={
                                                                "backgroundColor": "rgb(30, 30, 30)",
                                                                "color": "white",
                                                                "fontWeight": "bold",
                                                            },
                                                            style_data={
                                                                "backgroundColor": "rgb(50, 50, 50)",
                                                                "color": "white",
                                                            },
                                                        ),
                                                    ]
                                                )
                                            ],
                                            width=12,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [html.Div(id="governance-metrics-display")], width=12
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="🎲 Monte Carlo Simulator",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Monte Carlo Portfolio Outcome Simulator",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Simulate thousands of possible portfolio outcomes based on expected returns and probabilities. See the range of potential results and the likelihood of achieving your target return.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # Monte Carlo Filters
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Simulations:", style={"color": "white"}
                                                ),
                                                dcc.Dropdown(
                                                    id="mc-num-simulations",
                                                    options=[
                                                        {"label": "1,000", "value": 1000},
                                                        {"label": "5,000", "value": 5000},
                                                        {"label": "10,000", "value": 10000},
                                                        {"label": "50,000", "value": 50000},
                                                    ],
                                                    value=10000,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Loss Ratio:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="mc-loss-ratio",
                                                    options=[
                                                        {"label": "0.25 (25%)", "value": 0.25},
                                                        {"label": "0.5 (50%)", "value": 0.5},
                                                        {"label": "0.75 (75%)", "value": 0.75},
                                                        {"label": "1.0 (100%)", "value": 1.0},
                                                    ],
                                                    value=0.5,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Weighting:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="mc-weighting",
                                                    options=[
                                                        {
                                                            "label": "Equal-weighted",
                                                            "value": "equal",
                                                        },
                                                        {
                                                            "label": "Kelly-weighted",
                                                            "value": "kelly",
                                                        },
                                                        {
                                                            "label": "Market cap proxy",
                                                            "value": "market_cap",
                                                        },
                                                    ],
                                                    value="equal",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Target Return:", style={"color": "white"}
                                                ),
                                                dcc.Slider(
                                                    id="mc-target-return",
                                                    min=0.0,
                                                    max=20.0,
                                                    step=0.5,
                                                    value=0.0,
                                                    marks={i: f"{i}%" for i in range(0, 25, 5)},
                                                    tooltip={"placement": "bottom", "always_visible": True},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Signal Filter:", style={"color": "white"}
                                                ),
                                                dcc.Dropdown(
                                                    id="mc-signal-filter",
                                                    options=[
                                                        {"label": "Strong Bullish (4/4)","value": "Strong Bullish (4/4)",},
                                                        {"label": "Bullish (3/4)", "value": "Bullish (3/4)",},
                                                        {"label": "Neutral (2/4)", "value": "Neutral (2/4)",},
                                                        {"label": "Bearish (1/4)", "value": "Bearish (1/4)",},
                                                        {"label": "Strong Bearish (0/4)","value": "Strong Bearish (0/4)",},
                                                    ],
                                                    value=["Strong Bullish (4/4)", "Bullish (3/4)"],
                                                    multi=True,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "30%",
                                                "display": "inline-block",
                                                "margin": "10px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#333",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "margin": "10px",
                                    },
                                ),
                                # Stats Display
                                html.Div(
                                    id="mc-stats-display",
                                    style={
                                        "backgroundColor": "#f5f5f5",
                                        "padding": "15px",
                                        "margin": "10px",
                                        "borderRadius": "5px",
                                        "color": "black",
                                    },
                                ),
                                # Charts
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H4(
                                                    "Percentile Distribution",
                                                    style={"textAlign": "center"},
                                                ),
                                                dcc.Graph(id="mc-percentile-chart"),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.H4(
                                                    "Return Distribution",
                                                    style={"textAlign": "center"},
                                                ),
                                                dcc.Graph(id="mc-distribution-chart"),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="📉 Beta & CAPM Analysis",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Beta & CAPM: Systematic Risk and Expected Return",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Analyze stock sensitivity to market movements (beta) and expected returns using CAPM. "
                                    "Positive alpha indicates outperformance vs. CAPM prediction.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # CAPM-specific controls
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("Risk-Free Rate:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="capm-risk-free-rate",
                                                    options=[
                                                        {"label": "0%", "value": 0.0},
                                                        {"label": "2%", "value": 2.0},
                                                        {"label": "3%", "value": 3.0},
                                                        {"label": "4%", "value": 4.0},
                                                        {"label": "5%", "value": 5.0},
                                                    ],
                                                    value=3.0,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={"width": "15%", "display": "inline-block", "margin": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Market Return:", style={"color": "white"}),
                                                dcc.Slider(
                                                    id="capm-market-return",
                                                    min=0.0,
                                                    max=20.0,
                                                    step=0.5,
                                                    value=10.0,
                                                    marks={i: f"{i}%" for i in range(0, 25, 5)},
                                                    tooltip={"placement": "bottom", "always_visible": True},
                                                ),
                                            ],
                                            style={"width": "15%", "display": "inline-block", "margin": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Size Encoding:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="capm-size-encoding",
                                                    options=[
                                                        {"label": "Market Cap", "value": "market_cap"},
                                                        {"label": "None", "value": "none"},
                                                    ],
                                                    value="market_cap",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={"width": "15%", "display": "inline-block", "margin": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Confidence Level:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="capm-confidence-level",
                                                    options=[
                                                        {"label": "High Only", "value": "high_only"},
                                                        {"label": "High or Medium", "value": "high_medium"},
                                                        {"label": "All", "value": "all"},
                                                    ],
                                                    value="high_medium",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={"width": "15%", "display": "inline-block", "margin": "10px"},
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#333",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "margin": "10px",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H4("Beta vs Expected Return", style={"textAlign": "center"}),
                                                dcc.Loading(
                                                    type="circle",
                                                    children=[dcc.Graph(id="capm-scatter-graph")],
                                                ),
                                                html.Pre(id="capm-scatter-error", style={"color": "red", "fontSize": "12px"}),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.H4("Alpha (Excess Return)", style={"textAlign": "center"}),
                                                dcc.Loading(
                                                    type="circle",
                                                    children=[dcc.Graph(id="capm-bar-graph")],
                                                ),
                                                html.Pre(id="capm-bar-error", style={"color": "red", "fontSize": "12px"}),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="🎰 Kelly Criterion Position Sizer",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Kelly Criterion Optimal Position Sizing",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Calculate optimal position sizing based on expected returns and win probabilities "
                                    "using the Kelly Criterion formula. Adjust for confidence and achievement probability.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # Kelly-specific Controls
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("Kelly Fraction:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="kelly-fraction-dropdown",
                                                    options=KELLY_FRACTION_OPTIONS,
                                                    value=0.25,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={"width": "15%", "display": "inline-block", "margin": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Max Position Size:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="kelly-max-position-dropdown",
                                                    options=MAX_POSITION_OPTIONS,
                                                    value=0.10,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={"width": "15%", "display": "inline-block", "margin": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Min Confidence Score:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="kelly-min-confidence-dropdown",
                                                    options=KELLY_MIN_CONFIDENCE_OPTIONS,
                                                    value=0.35,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={"width": "15%", "display": "inline-block", "margin": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Adjustment Method:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="kelly-adjustment-dropdown",
                                                    options=KELLY_ADJUSTMENT_OPTIONS,
                                                    value="both",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={"width": "15%", "display": "inline-block", "margin": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Bar Color By:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="kelly-bar-color-dropdown",
                                                    options=KELLY_BAR_COLOR_OPTIONS,
                                                    value="none",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={"width": "15%", "display": "inline-block", "margin": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Scatter Color By:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="kelly-scatter-color-dropdown",
                                                    options=KELLY_SCATTER_COLOR_OPTIONS,
                                                    value="none",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={"width": "15%", "display": "inline-block", "margin": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Scatter Size By:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="kelly-scatter-size-dropdown",
                                                    options=KELLY_SCATTER_SIZE_OPTIONS,
                                                    value="none",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={"width": "15%", "display": "inline-block", "margin": "10px"},
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#333",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "margin": "10px",
                                    },
                                ),
                                # Kelly KPI summary row
                                html.Div(id="kelly-kpi-summary", style={"margin": "10px"}),
                                # Charts side-by-side
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H4("Top 30 Positions by Kelly %", style={"textAlign": "center"}),
                                                dcc.Loading(
                                                    type="circle",
                                                    children=[dcc.Graph(id="kelly-bar-chart", style={"minHeight": "550px"})],
                                                ),
                                                html.Pre(id="kelly-bar-error", style={"color": "red", "fontSize": "12px"}),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.H4("Kelly % vs Expected Upside", style={"textAlign": "center"}),
                                                dcc.Loading(
                                                    type="circle",
                                                    children=[dcc.Graph(id="kelly-scatter-chart", style={"minHeight": "550px"})],
                                                ),
                                                html.Pre(id="kelly-scatter-error", style={"color": "red", "fontSize": "12px"}),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                                # Top opportunities table
                                html.Div(
                                    [
                                        html.H4("Kelly-Weighted Top Positions", style={"textAlign": "center", "marginTop": "20px"}),
                                        dash_table.DataTable(
                                            id="kelly-positions-table",
                                            columns=get_formatted_columns([
                                                "ticker", "sector", "last_price", "price_target",
                                                "filtered_upside", "achievement_probability",
                                                "confidence_level", "kelly_pct"
                                            ]),
                                            page_size=15,
                                            sort_action="native",
                                            style_table={"overflowX": "auto"},
                                            style_header=TABLE_STYLE_HEADER,
                                            style_cell=TABLE_STYLE_CELL,
                                            style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
                                        ),
                                    ],
                                    style={"margin": "10px"},
                                ),
                            ]
                        )
                    ],
                ),
            ]
        ),
    ]
)


# Monte Carlo Simulation Functions
def run_monte_carlo_simulation(sim_df, num_simulations, loss_ratio, weighting, target_return):
    """Run Monte Carlo simulation and return results."""
    if sim_df.empty:
        return np.array([]), {}

    sim_df = sim_df.copy()
    sim_df['prob_positive_upside'] = pd.to_numeric(sim_df['prob_positive_upside'], errors='coerce')
    sim_df['filtered_upside'] = pd.to_numeric(sim_df['filtered_upside'], errors='coerce')
    sim_df['achievement_probability'] = pd.to_numeric(sim_df['achievement_probability'], errors='coerce')
    sim_df = sim_df.dropna(subset=['prob_positive_upside', 'filtered_upside', 'achievement_probability'])

    if sim_df.empty:
        return np.array([]), {}

    num_stocks = len(sim_df)

    # Calculate weights
    if weighting == "equal":
        weights = np.ones(num_stocks) / num_stocks
    elif weighting == "kelly":
        kelly_fractions = []
        for _, row in sim_df.iterrows():
            p = row['prob_positive_upside'] / 100.0
            b = row['filtered_upside'] / 100.0
            if b > 0 and p > 0 and p < 1:
                kelly = (p * b - (1 - p) * loss_ratio * b) / (b * b) if b != 0 else 0
                kelly = max(0, min(kelly, 0.25))
            else:
                kelly = 0
            kelly_fractions.append(kelly)
        kelly_fractions = np.array(kelly_fractions)
        total = kelly_fractions.sum()
        weights = kelly_fractions / total if total > 0 else np.ones(num_stocks) / num_stocks
    else:  # market_cap
        weights = np.ones(num_stocks) / num_stocks

    # Calculate probabilities and returns
    prob_wins = (sim_df['prob_positive_upside'].values / 100.0) * sim_df['achievement_probability'].values
    prob_wins = np.clip(prob_wins, 0, 1.0)
    upside_returns = sim_df['filtered_upside'].values / 100.0

    # Run simulations
    portfolio_returns = np.zeros(num_simulations)
    np.random.seed(42)
    # Simulates portfolio returns based on stock outcomes
    for sim in range(num_simulations):
        outcomes = np.random.random(num_stocks) < prob_wins
        stock_returns = np.where(outcomes, upside_returns, -upside_returns * loss_ratio)
        portfolio_returns[sim] = np.dot(weights, stock_returns) * 100

    # Calculate statistics
    percentiles = np.percentile(portfolio_returns, [5, 25, 50, 75, 95])
    var_5 = percentiles[0]
    below_var = portfolio_returns[portfolio_returns <= var_5]
    cvar_5 = below_var.mean() if len(below_var) > 0 else var_5
    prob_positive = (portfolio_returns > 0).sum() / num_simulations * 100
    prob_target = (portfolio_returns > target_return).sum() / num_simulations * 100

    stats = {
        "num_simulations": num_simulations,
        "num_stocks": num_stocks,
        "var_5": var_5,
        "cvar_5": cvar_5,
        "median": percentiles[2],
        "prob_positive": prob_positive,
        "prob_target": prob_target,
        "target_return": target_return,
        "p5": percentiles[0],
        "p25": percentiles[1],
        "p50": percentiles[2],
        "p75": percentiles[3],
        "p95": percentiles[4]
    }

    return portfolio_returns, stats


# =============================================================================
# Reset Filters Callback
# =============================================================================


@app.callback(
    [Output(f["id"], "value") for f in FILTER_CONFIG],
    Input("reset-filters-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _reset_filters(_n):
    """Reset all filter dropdowns to None (no selection)."""
    return [None] * len(FILTER_CONFIG)


@app.callback(
    [
        Output("kpi-cards", "children"),
        Output("returns-scatter", "figure"),
        Output("model-signals-plot", "figure"),
        Output("confidence-distribution", "figure"),
        Output("top-opportunities-table", "data"),
        Output("signal-breakdown", "figure"),
        Output("regional-performance", "figure"),
        Output("ranking-bar-chart", "figure"),
        Output("risk-reward-scatter", "figure"),
        Output("posterior-forest-plot", "figure"),
        Output("tri-model-posterior", "figure"),
        Output("beat-prob-posterior", "figure"),
        Output("ruin-diagnostic", "figure"),
        Output("category-ridge-plot", "figure"),
    ],
    [Input(f["id"], "value") for f in FILTER_CONFIG]
    + [
        Input("scoring-method-dropdown", "value"),
        Input("min-agreement-dropdown", "value"),
        Input("min-confidence-dropdown", "value"),
        Input("risk-free-rate-dropdown", "value"),
        Input("scatter-color-dropdown", "value"),
        Input("prob-ticker-dropdown", "value"),
    ],
)
def update_dashboard(*args):
    """Update dashboard visualizations based on selected filters."""
    # Unpack: first 12 args are global filters, rest are tab-specific
    num_filters = len(FILTER_CONFIG)
    filter_values = collect_filter_values(*args[:num_filters])

    scoring_method, min_agreement, min_confidence, risk_free_rate, scatter_color, prob_tickers = args[num_filters:]

    # Apply all global filters consistently
    filtered_df = apply_global_filters(df, filter_values)

    # 2. Apply Numerical Threshold Filters
    if min_agreement is not None and "agreement_score" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["agreement_score"] >= min_agreement]
    if min_confidence is not None and "confidence_score" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["confidence_score"] >= min_confidence]

    # 3. Handle Empty States
    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(title="No data matching selected filters")
        return [[]] + [empty_fig] * 13

    # ---------------------------------------------------------
    # Visualization Logic
    # ---------------------------------------------------------
    fig_scatter = px.scatter(
        filtered_df,
        x="expected_upside_pct",
        y="prob_positive_upside",
        color=scatter_color if scatter_color != "none" else None,
        hover_name="ticker",
        title="Upside vs. Probability",
        template="plotly_dark"
    )

    # ---------------------------------------------------------
    # Risk-Adjusted Ranking Calculations
    # ---------------------------------------------------------
    ranking_df = filtered_df.copy()
    if not ranking_df.empty:
        # Scoring methods
        ev_base = (ranking_df['filtered_upside'] / 100) * (ranking_df['prob_positive_upside'] / 100)
        ev_prob = ranking_df['expected_return_prob_weighted'] / 100
        ev_conf = ev_base * ranking_df['confidence_score']
        ev_achieve = ev_base * ranking_df['achievement_probability']
        ev_final = ev_base * ranking_df['confidence_score'] * ranking_df['achievement_probability'] * (1 + ranking_df['posterior_beat_prob'])

        if scoring_method == "base_ev":
            ranking_df['ev_score'] = ev_base
        elif scoring_method == "prob_weighted":
            ranking_df['ev_score'] = ev_prob
        elif scoring_method == "confidence_adj":
            ranking_df['ev_score'] = ev_conf
        elif scoring_method == "achievement_adj":
            ranking_df['ev_score'] = ev_achieve
        else:
            ranking_df['ev_score'] = ev_final

        # Risk & Reward scores
        uncertainty_penalty = 1 - (ranking_df['prob_positive_upside'] / 100) * ranking_df['confidence_score']
        disagreement_penalty = 1 - (ranking_df['agreement_score'] / 4)
        ranking_df['risk_score'] = uncertainty_penalty * (1 + disagreement_penalty)

        upside_potential = (ranking_df['filtered_upside'] / 100) * ranking_df['achievement_probability']
        beat_probability_bonus = 1 + ranking_df['posterior_beat_prob']
        ranking_df['reward_score'] = upside_potential * beat_probability_bonus

        # Risk-adjusted return & Sharpe-like ratio
        ranking_df['risk_adjusted_return'] = ranking_df['reward_score'] / (ranking_df['risk_score'] + 1e-6)
        risk_free_rate_decimal = (risk_free_rate or 0) / 100
        ranking_df['sharpe_like_ratio'] = (ranking_df['expected_return_prob_weighted'] / 100 - risk_free_rate_decimal) / (ranking_df['risk_score'] + 1e-6)

        # Apply ranking-specific filters
        if min_agreement is not None:
            ranking_df = ranking_df[ranking_df['agreement_score'] >= min_agreement]
        if min_confidence is not None:
            ranking_df = ranking_df[ranking_df['confidence_score'] >= min_confidence]

    # KPI Cards
    kpi_cards = []
    if not filtered_df.empty:
        total_stocks = len(filtered_df)
        avg_expected_return = filtered_df["expected_return_prob_weighted"].mean()
        high_confidence = len(filtered_df[filtered_df["confidence_level"] == "high"]) if "confidence_level" in filtered_df.columns else 0
        strong_buy = len(filtered_df[filtered_df["signal"] == "Strong Bullish (4/4)"]) if "signal" in filtered_df.columns else 0

        kpi_cards = [
            dbc.Card(
                dbc.CardBody([
                    html.H4("Total Stocks", className="card-title"),
                    html.H2(f"{total_stocks:,}", className="card-text"),
                ]),
                color="primary",
                inverse=True,
                style={"width": "23%", "textAlign": "center"},
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H4("Avg Expected Return", className="card-title"),
                    html.H2(f"{avg_expected_return:.1f}%", className="card-text"),
                ]),
                color="success" if avg_expected_return > 0 else "danger",
                inverse=True,
                style={"width": "23%", "textAlign": "center"},
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H4("High Confidence", className="card-title"),
                    html.H2(f"{high_confidence}", className="card-text"),
                ]),
                color="info",
                inverse=True,
                style={"width": "23%", "textAlign": "center"},
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H4("Strong Buy Signals", className="card-title"),
                    html.H2(f"{strong_buy}", className="card-text"),
                ]),
                color="warning",
                inverse=True,
                style={"width": "23%", "textAlign": "center"},
            ),
        ]

    # Expected Returns Scatter Plot
    returns_scatter = {}
    if not filtered_df.empty and all(col in filtered_df.columns for col in ["expected_return_prob_weighted", "achievement_probability"]):
        returns_scatter = px.scatter(
            filtered_df,
            x="achievement_probability",
            y="expected_return_prob_weighted",
            color="signal" if "signal" in filtered_df.columns else None,
            size="confidence_score" if "confidence_score" in filtered_df.columns else None,
            hover_data=["ticker", "name","country", "sector","industry","exchange"] if all(c in filtered_df.columns for c in ["ticker", "name","country", "sector","industry","exchange"]) else None,
            title="Expected Return vs Achievement Probability",
            labels={
                "achievement_probability": "Achievement Probability",
                "expected_return_prob_weighted": "Expected Return (%)"
            },
            template="plotly_dark",
        )

    # Model Signals Plot
    model_signals = {}
    if not filtered_df.empty:
        signals_data = []
        for model in ["mc_bullish", "kal_bullish", "pt_bullish", "earn_bullish"]:
            if model in filtered_df.columns:
                signals_data.append({
                    "Model": model.replace("_bullish", "").upper(),
                    "Bullish Count": filtered_df[model].sum(),
                    "Bearish Count": (~filtered_df[model]).sum()
                })

        if signals_data:
            signals_df = pd.DataFrame(signals_data)
            model_signals = px.bar(
                signals_df,
                x="Model",
                y=["Bullish Count", "Bearish Count"],
                title="Model Signal Breakdown",
                barmode="group",
                template="plotly_dark",
            )

    # Confidence Distribution
    confidence_distribution = {}
    if not filtered_df.empty and "confidence_score" in filtered_df.columns:
        confidence_distribution = px.histogram(
            filtered_df,
            x="confidence_score",
            nbins=30,
            title="Confidence Score Distribution",
            labels={"confidence_score": "Confidence Score"},
            template="plotly_dark",
            color_discrete_sequence=["#375a7f"],
        )

    # Top Opportunities Table
    top_opportunities_data = []
    if not filtered_df.empty:
        cols_to_show = [
            "ticker",
            "name",
            "country",
            "trading_country",
            "exchange",
            "sector",
            "industry",
            "next_earnings",
            "market_cap",
            "last_price",
            "price_target",
            "expected_upside_pct",
            "filtered_upside",
            "expected_return_prob_weighted",
            "achievement_probability",
            "confidence_level",
            "signal",
            "agreement_score",
        ]
        cols_available = [c for c in cols_to_show if c in filtered_df.columns]
        sort_col = "expected_return_prob_weighted" if "expected_return_prob_weighted" in filtered_df.columns else filtered_df.columns[0]
        top_opportunities_data = (
            filtered_df
            .nlargest(20, sort_col)
            [cols_available]
            .to_dict("records")
        )

    # Signal Breakdown by Sector
    signal_breakdown = {}
    if not filtered_df.empty and "signal" in filtered_df.columns and "sector" in filtered_df.columns:
        signal_counts = filtered_df.groupby(["sector", "signal"]).size().reset_index(name="count")
        signal_breakdown = px.bar(
            signal_counts,
            x="sector",
            y="count",
            color="signal",
            title="Signal Distribution by Sector",
            barmode="stack",
            template="plotly_dark",
        )

    # Regional Performance
    regional_perf = {}
    if not filtered_df.empty and "region" in filtered_df.columns and "expected_return_prob_weighted" in filtered_df.columns:
        regional_stats = (
            filtered_df
            .groupby("region")
            ["expected_return_prob_weighted"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        regional_perf = px.bar(
            regional_stats,
            x="region",
            y="mean",
            title="Average Expected Return by Region",
            labels={"mean": "Avg Expected Return (%)", "region": "Region"},
            template="plotly_dark",
            text="count",
        )
        regional_perf.update_traces(texttemplate='n=%{text}', textposition='outside')

    # Ranking Bar Chart (Top 50)
    ranking_bar = {}
    if not ranking_df.empty and 'ev_score' in ranking_df.columns:
        top_50 = ranking_df.nlargest(50, 'ev_score').sort_values('ev_score', ascending=True)
        ranking_bar = px.bar(
            top_50,
            x='ev_score',
            y='ticker',
            orientation='h',
            color='confidence_level',
            title=f"Top 50 Stocks by {scoring_method.replace('_', ' ').title()} Score",
            labels={'ev_score': 'Expected Value Score', 'ticker': 'Ticker', 'confidence_level': 'Confidence Level'},
            template="plotly_dark",
            height=800
        )
        ranking_bar.update_layout(yaxis={'categoryorder': 'total ascending'})

    # Risk vs Reward Scatter
    risk_reward_scatter = {}
    if not ranking_df.empty and 'risk_score' in ranking_df.columns and 'reward_score' in ranking_df.columns:
        risk_reward_scatter = px.scatter(
            ranking_df,
            x='risk_score',
            y='reward_score',
            color='signal' if scatter_color == 'signal' else None,
            size='ev_score' if ranking_df['ev_score'].min() >= 0 else None,
            hover_data=['ticker', 'name', 'sector', 'ev_score', 'risk_adjusted_return', 'sharpe_like_ratio'],
            title="Risk vs Reward Analysis",
            labels={
                'risk_score': 'Risk Score (Uncertainty & Disagreement)',
                'reward_score': 'Reward Score (Upside & Beat Prob)',
                'ev_score': 'Expected Value Score',
                'risk_adjusted_return': 'Risk-Adj Return',
                'sharpe_like_ratio': 'Sharpe-like Ratio'
            },
            template="plotly_dark"
        )
        if scatter_color == 'signal':
            risk_reward_scatter.update_traces(marker=dict(sizemin=5))

    # Probabilistic Visualizations
    posterior_forest = {}
    tri_model_post = {}
    beat_prob_post = {}
    ruin_diag = {}
    ridge_plot = {}

    if PROB_VIZ_AVAILABLE:
        # 1. Forest Plot (uses summary)
        if not filtered_df.empty:
            posterior_forest = probability_viz.create_posterior_return_forest(
                filtered_df, top_n=20, title="Expected Upside Forest Plot"
            )

            # Apply same filters to probabilistic data sources CONSISTENTLY
            filtered_tri = apply_global_filters(df_tri, filter_values)
            filtered_earnings = apply_global_filters(df_earnings, filter_values)
            filtered_credit = apply_global_filters(df_credit, filter_values)

            # 2. Tri-Model Comparison (now filtered)
            if not filtered_tri.empty:
                tri_model_post = probability_viz.create_tri_model_posterior_comparison(
                    filtered_tri, tickers=prob_tickers, top_n=8
                )

            # 3. Beat Probability Posterior (now filtered)
            if not filtered_earnings.empty:
                beat_prob_post = probability_viz.create_beat_probability_posterior(
                    filtered_earnings, tickers=prob_tickers, top_n=10
                )

            # 4. Ruin Probability Diagnostic (now filtered)
            if not filtered_credit.empty:
                ruin_diag = probability_viz.create_ruin_probability_diagnostic(
                    filtered_credit, top_n=15
                )

            # 5. Bayesian Ridge Plot (dynamic analysis)
        if not filtered_df.empty:
            try:
                # Use Profitability features as example
                prof_features = ['roe', 'roa', 'roic', 'operating_margin', 'net_margin']
                # Check which are available
                available = [f for f in prof_features if f in filtered_df.columns]
                if available:
                    # Run on-the-fly analysis for the Ridge plot
                    # Use a sample for speed if many stocks
                    sample_df = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)
                    analysis_results = bayesian_category_analysis(
                        sample_df, "Profitability", available
                    )
                    ridge_plot = probability_viz.create_bayesian_category_ridge(
                        analysis_results, category_name="Profitability"
                    )
            except Exception as e:
                print(f"Error generating ridge plot: {e}")

    return (
        kpi_cards,
        returns_scatter,
        model_signals,
        confidence_distribution,
        top_opportunities_data,
        signal_breakdown,
        regional_perf,
        ranking_bar,
        risk_reward_scatter,
        posterior_forest,
        tri_model_post,
        beat_prob_post,
        ruin_diag,
        ridge_plot,
    )


# =============================================================================
# Price Target vs Current Price Scatter Callback
# =============================================================================


@app.callback(
    Output("price_target_vs_current_scatter", "figure"),
    [Input(f["id"], "value") for f in FILTER_CONFIG]
    + [
        Input("pt-scatter-size-control", "value"),
        Input("pt-scatter-color-control", "value"),
        Input("pt-scatter-last-price-slider", "value"),
        Input("pt-scatter-price-target-slider", "value"),
    ],
)
def update_price_target_scatter(*args):
    """Update the Price Target vs Current Price scatter plot."""
    import traceback as tb

    num_filters = len(FILTER_CONFIG)
    filter_values = collect_filter_values(*args[:num_filters])

    size_encoding, color_encoding, last_price_range, price_target_range = args[num_filters:]

    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No data available",
        template="plotly_dark",
        annotations=[{"text": "No data available to display", "showarrow": False, "font": {"size": 20}}],
    )

    try:
        # Apply ALL global filters consistently
        filtered_df = apply_global_filters(df, filter_values)

        if filtered_df.empty:
            return empty_fig

        # Select required columns
        required_cols = ["last_price", "price_target", "expected_upside_pct", "market_cap",
                         "volume_shrs", "sector", "confidence_level", "beat_classification", "ticker", "name"]
        available_cols = [c for c in required_cols if c in filtered_df.columns]
        plot_df = filtered_df[available_cols].copy()

        # Ensure numeric types
        for col in ["last_price", "price_target", "expected_upside_pct", "market_cap", "volume_shrs"]:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

        plot_df = plot_df.dropna(subset=["last_price", "price_target"])

        if plot_df.empty:
            return empty_fig

        # Apply price range filters
        size_encoding = size_encoding or "expected_upside_pct"
        color_encoding = color_encoding or "sector"
        last_price_range = last_price_range or [0, 1000]
        price_target_range = price_target_range or [0, 1000]

        plot_df = plot_df[
            (plot_df["last_price"] >= last_price_range[0]) & (plot_df["last_price"] <= last_price_range[1])
        ]
        plot_df = plot_df[
            (plot_df["price_target"] >= price_target_range[0]) & (plot_df["price_target"] <= price_target_range[1])
        ]

        if plot_df.empty:
            empty_range = go.Figure()
            empty_range.update_layout(
                title="No data available for selected price range",
                template="plotly_dark",
                annotations=[{"text": "Adjust the price range sliders", "showarrow": False, "font": {"size": 20}}],
            )
            return empty_range

        # Normalize size column
        size_col = None if size_encoding == "none" else size_encoding
        if size_col and size_col in plot_df.columns:
            plot_df["size_normalized"] = plot_df[size_col].clip(lower=0) + 1
            size_col = "size_normalized"
        elif size_col and size_col not in plot_df.columns:
            size_col = None

        color_col = None if color_encoding == "none" else color_encoding
        if color_col and color_col not in plot_df.columns:
            color_col = None

        # Build scatter plot
        scatter_kwargs = dict(
            data_frame=plot_df,
            x="last_price",
            y="price_target",
            size=size_col,
            hover_data={
                "ticker": True,
                "name": True,
                "last_price": ":.2f",
                "price_target": ":.2f",
                "expected_upside_pct": ":.2f",
            },
            labels={
                "last_price": "Current Price ($)",
                "price_target": "Price Target ($)",
                "expected_upside_pct": "Expected Upside (%)",
                "market_cap": "Market Cap ($M)",
                "volume_shrs": "Volume (Shares)",
                "sector": "Sector",
                "confidence_level": "Confidence Level",
                "beat_classification": "Beat Classification",
            },
            template="plotly_dark",
        )
        if color_col:
            scatter_kwargs["color"] = color_col

        fig = px.scatter(**scatter_kwargs)

        if size_col:
            fig.update_traces(marker=dict(sizemin=6))

        # Add diagonal fair-value line (y = x)
        x_min = plot_df["last_price"].min()
        x_max = plot_df["last_price"].max()
        y_min = plot_df["price_target"].min()
        y_max = plot_df["price_target"].max()
        axis_min = min(x_min, y_min) * 0.9
        axis_max = max(x_max, y_max) * 1.1

        fig.add_shape(
            type="line",
            x0=axis_min, y0=axis_min,
            x1=axis_max, y1=axis_max,
            line=dict(color="gray", dash="dash", width=2),
        )

        fig.update_layout(
            title="Price Target vs Current Price",
            xaxis_title="Current Price ($)",
            yaxis_title="Price Target ($)",
            hovermode="closest",
            xaxis=dict(type="log"),
            yaxis=dict(type="log"),
        )

        if color_col:
            fig.update_layout(legend_title_text=color_col.replace("_", " ").title())

        return fig

    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(
            title="Error in chart",
            template="plotly_dark",
            annotations=[{"text": f"Error: {str(e)}", "showarrow": False, "font": {"size": 14}}],
        )
        return error_fig


@app.callback(
    [
        Output("mc-percentile-chart", "figure"),
        Output("mc-distribution-chart", "figure"),
        Output("mc-stats-display", "children"),
    ],
    [Input(f["id"], "value") for f in FILTER_CONFIG]
    + [
        Input("mc-num-simulations", "value"),
        Input("mc-loss-ratio", "value"),
        Input("mc-weighting", "value"),
        Input("mc-target-return", "value"),
        Input("mc-signal-filter", "value"),
    ],
)
def update_monte_carlo(*args):
    """Update Monte Carlo simulation visualizations."""
    num_filters = len(FILTER_CONFIG)
    filter_values = collect_filter_values(*args[:num_filters])

    num_simulations, loss_ratio, weighting, target_return, signal_filter = args[num_filters:]

    # Apply ALL global filters consistently
    mc_df = apply_global_filters(df, filter_values)

    # Apply MC-specific signal filter
    if signal_filter:
        mc_df = mc_df[mc_df["signal"].isin(signal_filter)]

    # Set defaults
    num_simulations = num_simulations or 10000
    loss_ratio = loss_ratio or 0.5
    weighting = weighting or "equal"
    target_return = target_return or 10.0

    # Run simulation
    portfolio_returns, stats = run_monte_carlo_simulation(mc_df, num_simulations, loss_ratio, weighting, target_return)

    # Create empty figures if no data
    if len(portfolio_returns) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            template="plotly_dark",
            annotations=[{"text": "No valid data for simulation", "showarrow": False, "font": {"size": 16}}]
        )
        return empty_fig, empty_fig, html.Div("No data available for simulation", style={"color": "red"})

    # Percentile Distribution Chart
    percentiles = np.arange(0, 101, 1)
    percentile_values = np.percentile(portfolio_returns, percentiles)

    fig_percentile = go.Figure()
    fig_percentile.add_trace(go.Scatter(
        x=percentiles,
        y=percentile_values,
        mode="lines",
        name="Portfolio Return",
        line=dict(width=2, color="#00bc8c")
    ))
    fig_percentile.update_layout(
        xaxis_title="Percentile",
        yaxis_title="Simulated Portfolio Return (%)",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=20, b=60),
        template="plotly_dark"
    )

    # Return Distribution Chart
    min_return = portfolio_returns.min()
    max_return = portfolio_returns.max()
    bucket_width = 10
    bucket_edges = np.arange(
        np.floor(min_return / bucket_width) * bucket_width,
        np.ceil(max_return / bucket_width) * bucket_width + bucket_width,
        bucket_width
    )
    counts, _ = np.histogram(portfolio_returns, bins=bucket_edges)
    bucket_labels = [f"{bucket_edges[i]:.0f}% to {bucket_edges[i+1]:.0f}%" for i in range(len(bucket_edges) - 1)]

    fig_distribution = go.Figure()
    fig_distribution.add_trace(go.Bar(
        x=bucket_labels,
        y=counts,
        name="Frequency",
        marker_color="#375a7f",
        marker_line_width=0
    ))
    fig_distribution.update_layout(
        xaxis_title="Return Bucket",
        yaxis_title="Frequency (Number of Simulations)",
        hovermode="x",
        margin=dict(l=60, r=20, t=20, b=80),
        xaxis=dict(tickangle=-45),
        template="plotly_dark"
    )

    # Stats Display
    stats_content = html.Div([
        html.B(f"Simulation Results ({stats['num_simulations']:,} runs, {stats['num_stocks']} stocks)"),
        html.Br(),
        html.Span(f"Value at Risk (5th percentile): {stats['var_5']:.2f}%"),
        html.Br(),
        html.Span(f"Conditional VaR (avg below 5th): {stats['cvar_5']:.2f}%"),
        html.Br(),
        html.Span(f"Median Return: {stats['median']:.2f}%"),
        html.Br(),
        html.Span(f"Probability of Positive Return: {stats['prob_positive']:.1f}%"),
        html.Br(),
        html.Span(f"Probability of Beating {stats['target_return']:.0f}% Target: {stats['prob_target']:.1f}%"),
        html.Br(),
        html.Span(f"Percentiles: 5th: {stats['p5']:.2f}% | 25th: {stats['p25']:.2f}% | 50th: {stats['p50']:.2f}% | 75th: {stats['p75']:.2f}% | 95th: {stats['p95']:.2f}%"),
    ])

    return fig_percentile, fig_distribution, stats_content


# =============================================================================
# Uncertainty & Calibration Tab Callback
# =============================================================================


@app.callback(
    [
        Output("calibration-curve", "figure"),
        Output("prediction-interval-coverage", "figure"),
        Output("uncertainty-distribution", "figure"),
        Output("model-agreement-heatmap", "figure"),
        Output("calibration-metrics-display", "children"),
    ],
    [Input(f["id"], "value") for f in FILTER_CONFIG],
)
def update_uncertainty_calibration(*args):
    """Update Uncertainty & Calibration tab visualizations."""
    filter_values = collect_filter_values(*args)
    filtered_df = apply_global_filters(df, filter_values)

    empty_fig = go.Figure().update_layout(title="No data available", template="plotly_dark")

    if filtered_df.empty:
        return empty_fig, empty_fig, empty_fig, empty_fig, html.Div("No data")

    # 1. Calibration Curve (predicted probability vs observed frequency)
    calibration_fig = go.Figure()
    if "posterior_beat_prob" in filtered_df.columns:
        bins = np.linspace(0, 1, 11)
        filtered_df["prob_bin"] = pd.cut(filtered_df["posterior_beat_prob"], bins=bins)

        if "agreement_score" in filtered_df.columns:
            bin_stats = (
                filtered_df.groupby("prob_bin", observed=True)
                .agg({"posterior_beat_prob": "mean", "agreement_score": lambda x: (x >= 3).mean()})
                .dropna()
            )

            if not bin_stats.empty:
                calibration_fig = go.Figure()
                calibration_fig.add_trace(
                    go.Scatter(
                        x=bin_stats["posterior_beat_prob"],
                        y=bin_stats["agreement_score"],
                        mode="markers+lines",
                        name="Model Calibration",
                        marker=dict(size=10, color="#00bc8c"),
                    )
                )
                calibration_fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Perfect Calibration",
                        line=dict(dash="dash", color="gray"),
                    )
                )
                calibration_fig.update_layout(
                    title="Calibration Curve: Predicted vs Observed",
                    xaxis_title="Mean Predicted Probability",
                    yaxis_title="Observed Frequency",
                    template="plotly_dark",
                )

    # 2. Prediction Interval Coverage
    coverage_fig = go.Figure()
    if all(col in filtered_df.columns for col in ["expected_upside_pct", "filtered_upside"]):
        spread = filtered_df["expected_upside_pct"].std() if len(filtered_df) > 1 else 10
        lower = filtered_df["expected_upside_pct"] - 1.96 * spread
        upper = filtered_df["expected_upside_pct"] + 1.96 * spread
        in_interval = (
            (filtered_df["filtered_upside"] >= lower) & (filtered_df["filtered_upside"] <= upper)
        ).mean() * 100

        coverage_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=in_interval,
                title={"text": "95% Prediction Interval Coverage"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00bc8c" if in_interval >= 90 else "#e74c3c"},
                    "steps": [
                        {"range": [0, 80], "color": "#e74c3c"},
                        {"range": [80, 90], "color": "#f39c12"},
                        {"range": [90, 100], "color": "#00bc8c"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": 95,
                    },
                },
            )
        )
        coverage_fig.update_layout(template="plotly_dark", height=300)

    # 3. Uncertainty Distribution
    uncertainty_fig = go.Figure()
    if "confidence_score" in filtered_df.columns:
        filtered_df["uncertainty"] = 1 - filtered_df["confidence_score"]
        uncertainty_fig = px.histogram(
            filtered_df,
            x="uncertainty",
            nbins=30,
            title="Uncertainty Score Distribution",
            labels={"uncertainty": "Uncertainty (1 - Confidence Score)"},
            template="plotly_dark",
            color_discrete_sequence=["#375a7f"],
        )
        uncertainty_fig.add_vline(
            x=filtered_df["uncertainty"].median(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {filtered_df['uncertainty'].median():.2f}",
        )

    # 4. Model Agreement Heatmap
    agreement_heatmap = go.Figure()
    if all(
        col in filtered_df.columns
        for col in ["mc_bullish", "kal_bullish", "pt_bullish", "earn_bullish"]
    ):
        model_cols = ["mc_bullish", "kal_bullish", "pt_bullish", "earn_bullish"]
        model_df = filtered_df[model_cols].astype(float)
        corr_matrix = model_df.corr()

        agreement_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            title="Model Signal Agreement Matrix",
            labels={"color": "Correlation"},
            color_continuous_scale="RdYlGn",
            template="plotly_dark",
        )

    # 5. Calibration Metrics Display
    metrics_display = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Mean Confidence Score"),
                                            html.H3(
                                                f"{filtered_df['confidence_score'].mean():.3f}"
                                                if "confidence_score" in filtered_df.columns
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color="info",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Model Agreement Rate"),
                                            html.H3(
                                                f"{(filtered_df['agreement_score'] >= 3).mean() * 100:.1f}%"
                                                if "agreement_score" in filtered_df.columns
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color="success",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("High Confidence %"),
                                            html.H3(
                                                f"{(filtered_df['confidence_level'] == 'high').mean() * 100:.1f}%"
                                                if "confidence_level" in filtered_df.columns
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color="warning",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Likely Beat %"),
                                            html.H3(
                                                f"{(filtered_df['beat_classification'] == 'likely_beat').mean() * 100:.1f}%"
                                                if "beat_classification" in filtered_df.columns
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color="primary",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                ]
            )
        ],
        style={"marginTop": "20px"},
    )

    return calibration_fig, coverage_fig, uncertainty_fig, agreement_heatmap, metrics_display


# =============================================================================
# Safety Rails & Data Quality Tab Callback
# =============================================================================


@app.callback(
    [
        Output("data-completeness-chart", "figure"),
        Output("outlier-detection-chart", "figure"),
        Output("data-freshness-chart", "figure"),
        Output("safety-threshold-chart", "figure"),
        Output("data-quality-summary", "children"),
    ],
    [Input(f["id"], "value") for f in FILTER_CONFIG],
)
def update_safety_rails(*args):
    """Update Safety Rails & Data Quality tab visualizations."""
    filter_values = collect_filter_values(*args)
    filtered_df = apply_global_filters(df, filter_values)

    empty_fig = go.Figure().update_layout(title="No data available", template="plotly_dark")

    if filtered_df.empty:
        return empty_fig, empty_fig, empty_fig, empty_fig, html.Div("No data")

    # 1. Data Completeness Chart
    completeness_data = []
    key_columns = [
        "expected_upside_pct",
        "filtered_upside",
        "expected_return_prob_weighted",
        "achievement_probability",
        "posterior_beat_prob",
        "confidence_score",
        "agreement_score",
        "signal",
        "confidence_level",
        "beat_classification",
    ]
    for col in key_columns:
        if col in filtered_df.columns:
            completeness = (1 - filtered_df[col].isna().mean()) * 100
            completeness_data.append({"Column": col, "Completeness": completeness})

    completeness_fig = go.Figure()
    if completeness_data:
        comp_df = pd.DataFrame(completeness_data).sort_values("Completeness")
        completeness_fig = px.bar(
            comp_df,
            x="Completeness",
            y="Column",
            orientation="h",
            title="Data Completeness by Column",
            labels={"Completeness": "Completeness (%)", "Column": ""},
            template="plotly_dark",
            color="Completeness",
            color_continuous_scale="RdYlGn",
        )
        completeness_fig.add_vline(
            x=95, line_dash="dash", line_color="white", annotation_text="95% threshold"
        )

    # 2. Outlier Detection Chart
    outlier_fig = go.Figure()
    numeric_cols = ["expected_upside_pct", "filtered_upside", "prob_positive_upside"]
    outlier_data = []
    for col in numeric_cols:
        if col in filtered_df.columns:
            q1 = filtered_df[col].quantile(0.25)
            q3 = filtered_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((filtered_df[col] < lower) | (filtered_df[col] > upper)).sum()
            outlier_data.append(
                {"Column": col, "Outliers": outliers, "Pct": outliers / len(filtered_df) * 100}
            )

    if outlier_data:
        outlier_df = pd.DataFrame(outlier_data)
        outlier_fig = px.bar(
            outlier_df,
            x="Column",
            y="Pct",
            title="Outlier Percentage by Column (IQR Method)",
            labels={"Pct": "Outlier %", "Column": ""},
            template="plotly_dark",
            color="Pct",
            color_continuous_scale="Reds",
        )

    # 3. Data Freshness Chart (placeholder)
    freshness_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=100,
            title={"text": "Data Freshness Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00bc8c"},
                "steps": [
                    {"range": [0, 50], "color": "#e74c3c"},
                    {"range": [50, 80], "color": "#f39c12"},
                    {"range": [80, 100], "color": "#00bc8c"},
                ],
            },
        )
    )
    freshness_fig.update_layout(template="plotly_dark", height=300)

    # 4. Safety Threshold Chart
    safety_fig = go.Figure()
    safety_checks = []

    if "expected_upside_pct" in filtered_df.columns:
        extreme_upside = (filtered_df["expected_upside_pct"].abs() > 200).mean() * 100
        safety_checks.append({"Check": "Extreme Upside (>200%)", "Violation %": extreme_upside})

    if "agreement_score" in filtered_df.columns:
        no_agreement = (filtered_df["agreement_score"] == 0).mean() * 100
        safety_checks.append({"Check": "Zero Model Agreement", "Violation %": no_agreement})

    if "confidence_score" in filtered_df.columns:
        low_confidence = (filtered_df["confidence_score"] < 0.2).mean() * 100
        safety_checks.append({"Check": "Very Low Confidence (<0.2)", "Violation %": low_confidence})

    if safety_checks:
        safety_df = pd.DataFrame(safety_checks)
        safety_fig = px.bar(
            safety_df,
            x="Check",
            y="Violation %",
            title="Safety Threshold Violations",
            template="plotly_dark",
            color="Violation %",
            color_continuous_scale="Reds",
        )

    # 5. Data Quality Summary
    total_rows = len(filtered_df)
    complete_rows = filtered_df.dropna(
        subset=key_columns[:5] if len(key_columns) >= 5 else key_columns
    ).shape[0]

    summary_display = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [html.H5("Total Records"), html.H3(f"{total_rows:,}")]
                                    )
                                ],
                                color="info",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Complete Records"),
                                            html.H3(
                                                f"{complete_rows:,} ({complete_rows/total_rows*100:.1f}%)"
                                                if total_rows > 0
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color=(
                                    "success"
                                    if total_rows > 0 and complete_rows / total_rows > 0.9
                                    else "warning"
                                ),
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Avg Completeness"),
                                            html.H3(
                                                f"{np.mean([c['Completeness'] for c in completeness_data]):.1f}%"
                                                if completeness_data
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color="primary",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Data Quality Score"),
                                            html.H3(
                                                f"{min(100, np.mean([c['Completeness'] for c in completeness_data]) - sum(s['Violation %'] for s in safety_checks)/10):.0f}/100"
                                                if completeness_data
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color="success",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                ]
            )
        ],
        style={"marginTop": "20px"},
    )

    return completeness_fig, outlier_fig, freshness_fig, safety_fig, summary_display


# =============================================================================
# Model Governance Tab Callback
# =============================================================================


@app.callback(
    [
        Output("model-performance-trend", "figure"),
        Output("model-drift-chart", "figure"),
        Output("model-registry-table", "data"),
        Output("governance-metrics-display", "children"),
    ],
    [Input(f["id"], "value") for f in FILTER_CONFIG],
)
def update_model_governance(*args):
    """Update Model Governance tab visualizations."""
    filter_values = collect_filter_values(*args)
    filtered_df = apply_global_filters(df, filter_values)

    empty_fig = go.Figure().update_layout(title="No data available", template="plotly_dark")

    # 1. Model Performance Trend
    performance_fig = go.Figure()

    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    models = ["Monte Carlo", "Kalman Filter", "PT Achievement", "Earnings Beat"]

    # Populates model performance trend chart with sample data
    for i, model in enumerate(models):
        np.random.seed(i)
        base_accuracy = 0.65 + i * 0.05
        accuracies = base_accuracy + np.random.normal(0, 0.02, len(dates))
        performance_fig.add_trace(
            go.Scatter(x=dates, y=accuracies, mode="lines+markers", name=model, line=dict(width=2))
        )

    performance_fig.update_layout(
        title="Model Accuracy Trend (30 Days)",
        xaxis_title="Date",
        yaxis_title="Accuracy",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # 2. Model Drift Chart
    drift_fig = go.Figure()
    drift_data = {
        "Model": models,
        "Feature Drift": [0.02, 0.015, 0.025, 0.01],
        "Prediction Drift": [0.03, 0.02, 0.018, 0.022],
    }
    drift_df = pd.DataFrame(drift_data)

    drift_fig = px.bar(
        drift_df,
        x="Model",
        y=["Feature Drift", "Prediction Drift"],
        title="Model Drift Metrics",
        barmode="group",
        template="plotly_dark",
    )
    drift_fig.add_hline(
        y=0.05, line_dash="dash", line_color="red", annotation_text="Drift Threshold (5%)"
    )

    # 3. Model Registry Table
    registry_data = [
        {
            "Model": "Monte Carlo Simulation",
            "Version": "2.4.0",
            "Last Updated": "2026-02-15",
            "Status": "Active",
            "Accuracy": "72.3%",
        },
        {
            "Model": "Kalman Filter",
            "Version": "2.4.0",
            "Last Updated": "2026-02-15",
            "Status": "Active",
            "Accuracy": "68.5%",
        },
        {
            "Model": "Price Target Achievement",
            "Version": "2.4.0",
            "Last Updated": "2026-02-15",
            "Status": "Active",
            "Accuracy": "74.1%",
        },
        {
            "Model": "Earnings Beat Probability",
            "Version": "2.4.0",
            "Last Updated": "2026-02-15",
            "Status": "Active",
            "Accuracy": "69.8%",
        },
    ]

    # 4. Governance Metrics Display
    governance_display = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [dbc.CardBody([html.H5("Active Models"), html.H3("4")])],
                                color="info",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [dbc.CardBody([html.H5("Avg Model Accuracy"), html.H3("71.2%")])],
                                color="success",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [dbc.CardBody([html.H5("Models in Drift"), html.H3("0")])],
                                color="success",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [dbc.CardBody([html.H5("Last Audit"), html.H3("2026-02-15")])],
                                color="primary",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                ]
            )
        ],
        style={"marginTop": "20px"},
    )

    return performance_fig, drift_fig, registry_data, governance_display


# =============================================================================
# Beta & CAPM Analysis Functions
# =============================================================================


def _calculate_sector_beta(sector: str) -> float:
    """Calculate beta based on sector classification."""
    sector_betas = {
        "Information Technology": 1.3,
        "Health Care": 1.3,
        "Utilities": 0.7,
        "Consumer Staples": 0.7,
        "Financials": 1.0,
        "Industrials": 1.0,
    }
    return sector_betas.get(sector, 1.1)


def _calculate_capm_expected_return(beta: float, rf: float, rm: float) -> float:
    """Calculate expected return using CAPM formula: E(R) = Rf + β(Rm - Rf)."""
    return rf + beta * (rm - rf)


# =============================================================================
# Beta & CAPM Tab Callback
# =============================================================================


@app.callback(
    [
        Output("capm-scatter-graph", "figure"),
        Output("capm-scatter-error", "children"),
        Output("capm-bar-graph", "figure"),
        Output("capm-bar-error", "children"),
    ],
    [Input(f["id"], "value") for f in FILTER_CONFIG]
    + [
        Input("capm-risk-free-rate", "value"),
        Input("capm-market-return", "value"),
        Input("capm-size-encoding", "value"),
        Input("capm-confidence-level", "value"),
    ],
)
def update_beta_capm(*args):
    """Update Beta & CAPM scatter and alpha bar charts."""
    import traceback

    num_filters = len(FILTER_CONFIG)
    filter_values = collect_filter_values(*args[:num_filters])

    risk_free_rate, market_return, size_encoding, capm_confidence_level = args[num_filters:]

    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No data available",
        template="plotly_dark",
        annotations=[{"text": "No data matches the selected filters", "showarrow": False, "font": {"size": 16}}],
    )

    scatter_fig = empty_fig
    scatter_error = ""
    bar_fig = empty_fig
    bar_error = ""

    try:
        # Apply ALL global filters consistently
        filtered_df = apply_global_filters(df, filter_values)

        # Apply CAPM-specific confidence level filter
        if capm_confidence_level == "high_only":
            filtered_df = filtered_df[filtered_df["confidence_level"] == "High"]
        elif capm_confidence_level == "high_medium":
            filtered_df = filtered_df[filtered_df["confidence_level"].isin(["High", "Medium"])]

        if filtered_df.empty:
            return empty_fig, "", empty_fig, ""

        # CAPM parameters
        rf = float(risk_free_rate) if risk_free_rate is not None else 3.0
        rm = float(market_return) if market_return is not None else 10.0

        # Calculate beta and CAPM return
        filtered_df = filtered_df.copy()
        filtered_df["beta"] = filtered_df["sector"].apply(_calculate_sector_beta)
        filtered_df["capm_return"] = filtered_df["beta"].apply(
            lambda b: _calculate_capm_expected_return(b, rf, rm)
        )
        filtered_df["alpha"] = filtered_df["expected_upside_pct"] - filtered_df["capm_return"]

        # --- Scatter Chart: Beta vs Expected Return ---
        scatter_kwargs = dict(
            data_frame=filtered_df,
            x="beta",
            y="expected_upside_pct",
            color="sector",
            hover_data={
                "name": True,
                "ticker": True,
                "beta": ":.2f",
                "expected_upside_pct": ":.2f",
                "sector": True,
            },
            labels={
                "beta": "Beta (Systematic Risk)",
                "expected_upside_pct": "Expected Return (%)",
                "sector": "Sector",
            },
            template="plotly_dark",
        )

        if size_encoding == "market_cap" and "market_cap" in filtered_df.columns:
            scatter_kwargs["size"] = "market_cap"
            scatter_kwargs["hover_data"]["market_cap"] = ":.0f"

        scatter_fig = px.scatter(**scatter_kwargs)

        if size_encoding == "market_cap":
            scatter_fig.update_traces(marker=dict(sizemin=6))

        scatter_fig.update_layout(
            title="Beta vs Expected Return (CAPM)",
            xaxis_title="Beta (Systematic Risk)",
            yaxis_title="Expected Return (%)",
            legend_title_text="Sector",
            hovermode="closest",
        )

        # Add Security Market Line (SML)
        beta_range = np.linspace(0, max(2.0, filtered_df["beta"].max() + 0.2), 50)
        sml_returns = [_calculate_capm_expected_return(b, rf, rm) for b in beta_range]
        scatter_fig.add_trace(
            go.Scatter(
                x=beta_range,
                y=sml_returns,
                mode="lines",
                name=f"SML (Rf={rf}%, Rm={rm}%)",
                line=dict(dash="dash", color="rgba(255,255,255,0.5)", width=2),
            )
        )

        # --- Bar Chart: Alpha (Excess Return) ---
        df_sorted = filtered_df.sort_values("alpha", ascending=False)
        top_n = 15
        top_stocks = df_sorted.head(top_n)
        bottom_stocks = df_sorted.tail(top_n)
        df_display = pd.concat([top_stocks, bottom_stocks]).drop_duplicates(subset=["ticker"]).sort_values("alpha", ascending=True)

        df_display = df_display.copy()
        df_display["alpha_color"] = df_display["alpha"].apply(lambda x: "Positive" if x > 0 else "Negative")

        bar_fig = px.bar(
            df_display,
            x="alpha",
            y="name",
            color="alpha_color",
            orientation="h",
            hover_data={
                "ticker": True,
                "alpha": ":.2f",
                "expected_upside_pct": ":.2f",
                "capm_return": ":.2f",
                "alpha_color": False,
            },
            labels={
                "alpha": "Alpha (Excess Return %)",
                "name": "Company",
                "alpha_color": "Alpha Sign",
            },
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
            template="plotly_dark",
        )
        bar_fig.update_layout(
            title="Alpha: Top & Bottom Stocks vs CAPM",
            xaxis_title="Alpha (Excess Return %)",
            yaxis_title="Company",
            legend_title_text="Alpha Sign",
            height=max(400, len(df_display) * 22),
            hovermode="closest",
        )

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        scatter_error = error_msg
        bar_error = error_msg

    return scatter_fig, scatter_error, bar_fig, bar_error

# =============================================================================
# Kelly Criterion Position Sizer Callback
# =============================================================================


@app.callback(
    [
        Output("kelly-bar-chart", "figure"),
        Output("kelly-bar-error", "children"),
        Output("kelly-scatter-chart", "figure"),
        Output("kelly-scatter-error", "children"),
        Output("kelly-kpi-summary", "children"),
        Output("kelly-positions-table", "data"),
    ],
    [Input(f["id"], "value") for f in FILTER_CONFIG]
    + [
        Input("kelly-fraction-dropdown", "value"),
        Input("kelly-max-position-dropdown", "value"),
        Input("kelly-min-confidence-dropdown", "value"),
        Input("kelly-adjustment-dropdown", "value"),
        Input("kelly-bar-color-dropdown", "value"),
        Input("kelly-scatter-color-dropdown", "value"),
        Input("kelly-scatter-size-dropdown", "value"),
    ],
    )
def update_kelly_criterion(*args):
    """Update Kelly Criterion Position Sizer tab visualizations."""
    import traceback as tb

    num_filters = len(FILTER_CONFIG)
    filter_values = collect_filter_values(*args[:num_filters])

    (
        kelly_fraction,
        max_position,
        min_confidence,
        adjustment_method,
        bar_color_by,
        scatter_color_by,
        scatter_size_by,
    ) = args[num_filters:]

    # Defaults
    kelly_fraction = kelly_fraction if kelly_fraction is not None else 0.25
    max_position = max_position if max_position is not None else 0.10
    min_confidence = min_confidence if min_confidence is not None else 0.35
    adjustment_method = adjustment_method if adjustment_method is not None else "both"
    bar_color_by = bar_color_by if bar_color_by is not None else "none"
    scatter_color_by = scatter_color_by if scatter_color_by is not None else "none"
    scatter_size_by = scatter_size_by if scatter_size_by is not None else "none"

    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No data available",
        template="plotly_dark",
        annotations=[{"text": "No data matches the selected filters", "showarrow": False, "font": {"size": 16}}],
    )

    bar_fig = empty_fig
    bar_error = ""
    scatter_fig = empty_fig
    scatter_error = ""
    kpi_summary = html.Div()
    table_data = []

    try:
        # Apply all global filters
        filtered_df = apply_global_filters(df, filter_values)

        if filtered_df.empty:
            return empty_fig, "", empty_fig, "", html.Div("No data"), []

        # Apply Kelly-specific confidence filter
        if "confidence_score" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["confidence_score"] >= min_confidence]

        if filtered_df.empty:
            return empty_fig, "", empty_fig, "", html.Div("No data after confidence filter"), []

        # Calculate Kelly metrics
        kelly_df = calculate_kelly_metrics(
            filtered_df,
            kelly_fraction=kelly_fraction,
            max_position=max_position,
            adjustment_method=adjustment_method,
        )

        # Drop rows where Kelly calculation is invalid
        kelly_df = kelly_df.dropna(subset=["kelly_pct"])
        kelly_df = kelly_df[kelly_df["kelly_pct"] > 0]

        if kelly_df.empty:
            return empty_fig, "", empty_fig, "", html.Div("No positions with positive Kelly %"), []

        # ----- KPI Summary Cards -----
        total_positions = len(kelly_df)
        mean_kelly = kelly_df["kelly_pct"].mean()
        max_kelly = kelly_df["kelly_pct"].max()
        top_ticker = kelly_df.nlargest(1, "kelly_pct")["ticker"].values[0] if total_positions > 0 else "N/A"
        total_kelly_raw_sum = kelly_df["kelly_raw"].sum()

        kpi_summary = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([html.H5("Eligible Positions"), html.H3(f"{total_positions:,}")]),
                        color="primary", inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([html.H5("Mean Kelly %"), html.H3(f"{mean_kelly:.2f}%")]),
                        color="info", inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([html.H5("Max Kelly %"), html.H3(f"{max_kelly:.2f}%")]),
                        color="success", inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([html.H5("Top Position"), html.H3(f"{top_ticker}")]),
                        color="warning", inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([html.H5("Sum Raw Kelly"), html.H3(f"{total_kelly_raw_sum:.3f}")]),
                        color="secondary", inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Kelly Fraction"),
                            html.H3(f"{kelly_fraction}× | {adjustment_method.title()}"),
                        ]),
                        color="dark", inverse=True,
                    ),
                    width=2,
                ),
            ],
            style={"marginBottom": "10px"},
        )

        # ----- Bar Chart: Top 30 Positions -----
        top30 = kelly_df.nlargest(30, "kelly_pct")

        bar_kwargs = dict(
            data_frame=top30,
            x="ticker",
            y="kelly_pct",
            labels={"kelly_pct": "Kelly % (Position Size)", "ticker": "Ticker"},
            template="plotly_dark",
        )

        if bar_color_by == "sector" and "sector" in top30.columns:
            bar_kwargs["color"] = "sector"
            bar_kwargs["hover_data"] = {"sector": True, "kelly_pct": ":.2f", "confidence_level": True}
        elif bar_color_by == "confidence_level" and "confidence_level" in top30.columns:
            bar_kwargs["color"] = "confidence_level"
            bar_kwargs["hover_data"] = {"confidence_level": True, "kelly_pct": ":.2f"}
        else:
            bar_kwargs["hover_data"] = {"kelly_pct": ":.2f", "sector": True, "confidence_level": True}

        bar_fig = px.bar(**bar_kwargs)
        bar_fig.update_xaxes(tickangle=-45)
        bar_fig.update_layout(
            xaxis_title="Ticker",
            yaxis_title="Kelly % (Position Size)",
            hovermode="x unified",
            legend_title_text=bar_color_by.replace("_", " ").title() if bar_color_by != "none" else "",
        )

        # ----- Scatter Chart: Kelly % vs Expected Upside -----
        scatter_kwargs = dict(
            data_frame=kelly_df,
            x="filtered_upside",
            y="kelly_pct",
            labels={"filtered_upside": "Expected Upside (%)", "kelly_pct": "Kelly % (Position Size)"},
            template="plotly_dark",
        )

        hover_cols = {"ticker": True, "filtered_upside": ":.2f", "kelly_pct": ":.2f"}

        if scatter_color_by == "confidence_level" and "confidence_level" in kelly_df.columns:
            scatter_kwargs["color"] = "confidence_level"
            hover_cols["confidence_level"] = True

        if scatter_size_by == "achievement_probability" and "achievement_probability" in kelly_df.columns:
            scatter_kwargs["size"] = "achievement_probability"
            hover_cols["achievement_probability"] = ":.2f"

        scatter_kwargs["hover_data"] = hover_cols

        scatter_fig = px.scatter(**scatter_kwargs)
        scatter_fig.update_traces(marker=dict(sizemin=6))
        scatter_fig.update_layout(
            xaxis_title="Expected Upside (%)",
            yaxis_title="Kelly % (Position Size)",
            hovermode="closest",
            legend_title_text=scatter_color_by.replace("_", " ").title() if scatter_color_by != "none" else "",
        )

        # ----- Positions Table -----
        table_cols = [
            "ticker", "name", "sector", "industry", "exchange",
            "last_price", "price_target", "filtered_upside",
            "prob_positive_upside", "confidence_score", "achievement_probability",
            "confidence_level", "signal", "kelly_pct",
        ]
        available_cols = [c for c in table_cols if c in kelly_df.columns]
        table_df = kelly_df.nlargest(50, "kelly_pct")[available_cols].copy()

        # Round numeric columns for display
        for col in ["filtered_upside", "prob_positive_upside", "confidence_score", "achievement_probability", "kelly_pct", "last_price", "price_target"]:
            if col in table_df.columns:
                table_df[col] = table_df[col].round(3)

        table_data = table_df.to_dict("records")

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{tb.format_exc()}"
        bar_error = error_msg
        scatter_error = error_msg

    return bar_fig, bar_error, scatter_fig, scatter_error, kpi_summary, table_data


if __name__ == "__main__":
    if len(df) == 0:
        print("\n" + "="*60)
        print("⚠️  No data loaded - Dashboard cannot start")
        print("="*60)
        print("\nTo use this dashboard:")
        print("1. Set environment variable: GEIB_DASHBOARD=true")
        print("2. Ensure DB_URL is configured")
        print("3. Verify analytics.expected_returns_summary table exists")
        print("\nExample:")
        print("  $env:GEIB_DASHBOARD='true'  # PowerShell")
        print("  export GEIB_DASHBOARD=true  # Bash")
        print("  python finance_ml/dashboards/geib_dash_app.py")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("🚀 Starting Global Equity Investment Board Dashboard")
        print("="*60)
        print(f"   Loaded: {len(df):,} stocks")
        print(f"   URL: http://127.0.0.1:8051")
        print("="*60 + "\n")
        app.run(debug=True, port=8051)
