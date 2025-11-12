"""
Plotly Dash Dashboard for Finance ML Analytics
Run: python finance_ml/dashboards/dash_app.py
"""

from pathlib import Path

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, dash_table

# Project root path for consistent path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent


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
            "market_cap",
            "last_price",
            "price_target",
            "predicted_price_target",
            "mispricing_score",
            "mispricing_pct",
        ]
    )


df = load_data()

app = dash.Dash(__name__, title="Finance ML Analytics")

app.layout = html.Div(
    [
        html.H1("📊 Finance ML Analytics Dashboard", style={"textAlign": "center"}),
        # KPI Summary Cards
        html.Div(
            id="kpi-cards",
            style={"display": "flex", "justifyContent": "space-around", "margin": "20px"},
        ),
        # Filters
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Select Sector:"),
                        dcc.Dropdown(
                            id="sector-dropdown",
                            options=(
                                [{"label": s, "value": s} for s in df["sector"].unique()]
                                if "sector" in df.columns and not df.empty
                                else []
                            ),
                            value=None,
                            multi=True,
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Label("Select Region:"),
                        dcc.Dropdown(
                            id="region-dropdown",
                            options=(
                                [{"label": r, "value": r} for r in df["region"].unique()]
                                if "region" in df.columns and not df.empty
                                else []
                            ),
                            value=None,
                            multi=True,
                        ),
                    ],
                    style={"width": "48%", "float": "right", "display": "inline-block"},
                ),
            ]
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
                                        dash_table.DataTable(id="undervalued-table", page_size=10),
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
                                        dash_table.DataTable(id="overvalued-table", page_size=10),
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
    [Input("sector-dropdown", "value"), Input("region-dropdown", "value")],
)
def update_dashboard(sectors, regions):
    """Update dashboard visualizations based on selected filters.

    Args:
        sectors: List of selected sector values from dropdown (or None).
        regions: List of selected region values from dropdown (or None).

    Returns:
        tuple: (kpi_cards, scatter_figure, heatmap_figure, error_figure,
                comparison_figure, undervalued_data, overvalued_data)
    """
    filtered_df = df.copy()

    if sectors:
        filtered_df = filtered_df[filtered_df["sector"].isin(sectors)]
    if regions:
        filtered_df = filtered_df[filtered_df["region"].isin(regions)]

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
            html.Div(
                [html.H4("Total Stocks"), html.H2(f"{total_stocks:,}")],
                style={
                    "textAlign": "center",
                    "padding": "20px",
                    "backgroundColor": "#f0f0f0",
                    "borderRadius": "5px",
                },
            ),
            html.Div(
                [html.H4("Avg Mispricing"), html.H2(f"{avg_mispricing:.2%}")],
                style={
                    "textAlign": "center",
                    "padding": "20px",
                    "backgroundColor": "#f0f0f0",
                    "borderRadius": "5px",
                },
            ),
            html.Div(
                [html.H4("Sectors"), html.H2(f"{sectors_count}")],
                style={
                    "textAlign": "center",
                    "padding": "20px",
                    "backgroundColor": "#f0f0f0",
                    "borderRadius": "5px",
                },
            ),
            html.Div(
                [html.H4("Regions"), html.H2(f"{regions_count}")],
                style={
                    "textAlign": "center",
                    "padding": "20px",
                    "backgroundColor": "#f0f0f0",
                    "borderRadius": "5px",
                },
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
        )
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
        )
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
        )
        error_fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
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
        )
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
                    line=dict(color="red", dash="dash"),
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
    app.run_server(debug=True, port=8050)
