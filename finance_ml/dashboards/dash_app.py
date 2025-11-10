"""
Plotly Dash Dashboard for Finance ML Analytics
Run: python finance_ml/dashboards/dash_app.py
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
from pathlib import Path


# Sample data loading (replace with actual data source)
def load_data():
    """Load prediction data from CSV file or return empty DataFrame.

    Returns:
        pd.DataFrame: Prediction data with columns for ticker, sector, region,
                     market_cap, last_price, and mispricing_score.
    """
    # Load from outputs or database
    csv_path = Path("outputs/analytics/predictions.csv")
    if csv_path.exists():
        all_stocks_phase95 = pd.read_csv(csv_path)
        # Convert numeric columns to proper dtypes
        numeric_columns = ["mispricing_score", "last_price", "market_cap"]
        for col in numeric_columns:
            if col in all_stocks_phase95.columns:
                all_stocks_phase95[col] = pd.to_numeric(all_stocks_phase95[col], errors="coerce")
        return all_stocks_phase95
    # Return empty DataFrame with expected columns if file doesn't exist
    return pd.DataFrame(
        columns=["ticker", "sector", "region", "market_cap", "last_price", "mispricing_score"]
    )


df = load_data()

app = dash.Dash(__name__, title="Finance ML Analytics")

app.layout = html.Div(
    [
        html.H1("📊 Finance ML Analytics Dashboard", style={"textAlign": "center"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Select Sector:"),
                        dcc.Dropdown(
                            id="sector-dropdown",
                            options=(
                                [{"label": s, "value": s} for s in df["sector"].unique()]
                                if "sector" in df.columns
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
                                if "region" in df.columns
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
        html.Div(
            [
                dcc.Graph(id="scatter-plot"),
                dcc.Graph(id="heatmap-plot"),
            ]
        ),
        html.Div(
            [
                html.H3("Top Undervalued Stocks"),
                dash_table.DataTable(id="undervalued-table", page_size=10),
            ]
        ),
    ]
)


@app.callback(
    [
        Output("scatter-plot", "figure"),
        Output("heatmap-plot", "figure"),
        Output("undervalued-table", "data"),
    ],
    [Input("sector-dropdown", "value"), Input("region-dropdown", "value")],
)
def update_dashboard(sectors, regions):
    """Update dashboard visualizations based on selected filters.

    Args:
        sectors: List of selected sector values from dropdown (or None).
        regions: List of selected region values from dropdown (or None).

    Returns:
        tuple: (scatter_figure, heatmap_figure, undervalued_stocks_data)
    """
    filtered_df = df.copy()

    if sectors:
        filtered_df = filtered_df[filtered_df["sector"].isin(sectors)]
    if regions:
        filtered_df = filtered_df[filtered_df["region"].isin(regions)]

    # Scatter plot
    if "market_cap" in filtered_df.columns and "mispricing_score" in filtered_df.columns:
        scatter_fig = px.scatter(
            filtered_df,
            x="market_cap",
            y="mispricing_score",
            color="sector" if "sector" in filtered_df.columns else None,
            size="last_price" if "last_price" in filtered_df.columns else None,
            hover_data=["ticker"] if "ticker" in filtered_df.columns else None,
            title="Mispricing Score vs Market Cap",
        )
    else:
        scatter_fig = {}

    # Heatmap
    if (
        "sector" in filtered_df.columns
        and "region" in filtered_df.columns
        and "mispricing_score" in filtered_df.columns
    ):
        pivot = filtered_df.pivot_table(
            values="mispricing_score", index="sector", columns="region", aggfunc="mean"
        )
        heatmap_fig = px.imshow(pivot, text_auto=".2f", title="Sector-Region Performance")
    else:
        heatmap_fig = {}

    # Top undervalued
    if "mispricing_score" in filtered_df.columns and not filtered_df.empty:
        # Filter out NaN values before sorting
        valid_df = filtered_df.dropna(subset=["mispricing_score"])
        if not valid_df.empty:
            top_stocks = valid_df.nlargest(10, "mispricing_score")[
                ["ticker", "sector", "mispricing_score", "last_price"]
            ].to_dict("records")
        else:
            top_stocks = []
    else:
        top_stocks = []

    return scatter_fig, heatmap_fig, top_stocks


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
