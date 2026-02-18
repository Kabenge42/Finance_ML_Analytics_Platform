import sys
import os
import traceback
from typing import TypedDict, Any, Tuple
from datetime import datetime, timedelta

from dash import callback, html, dcc, Output, Input
import dash_design_kit as ddk
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from data import get_data
from components.filter_component import filter_data, FILTER_CALLBACK_INPUTS
from logger import logger, schema, tbl

class TestInput(TypedDict):
    options: list[Any]
    default: Any

class ComponentResponse(TypedDict):
    layout: ddk.Card
    test_inputs: dict[str, TestInput]

component_id = "price_target_vs_current_scatter"

size_control_id = f"{component_id}_size_control"
size_options = [
    {"label": "Expected Upside %", "value": "expected_upside_pct"},
    {"label": "Market Cap", "value": "market_cap"},
    {"label": "Volume", "value": "volume_shrs"},
    {"label": "None", "value": "none"}
]
size_default = "expected_upside_pct"

color_control_id = f"{component_id}_color_control"
color_options = [
    {"label": "Sector", "value": "sector"},
    {"label": "Confidence Level", "value": "confidence_level"},
    {"label": "Beat Classification", "value": "beat_classification"},
    {"label": "None", "value": "none"}
]
color_default = "sector"

last_price_slider_id = f"{component_id}_last_price_slider"
last_price_slider_default = [0, 1000]

price_target_slider_id = f"{component_id}_price_target_slider"
price_target_slider_default = [0, 1000]

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    title = "Price Target vs Current Price"
    description = "Scatter plot showing analyst price targets versus current stock prices. Points above the diagonal line indicate upside potential, while points below indicate downside risk."

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Size Encoding:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=size_control_id,
                                options=size_options,
                                value=size_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Color Encoding:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=color_control_id,
                                options=color_options,
                                value=color_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Last Price Range:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.RangeSlider(
                                id=last_price_slider_id,
                                min=0,
                                max=1000,
                                step=10,
                                value=last_price_slider_default,
                                marks={0: "$0", 250: "$250", 500: "$500", 750: "$750", 1000: "$1000+"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px", "minWidth": "250px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Price Target Range:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.RangeSlider(
                                id=price_target_slider_id,
                                min=0,
                                max=1000,
                                step=10,
                                value=price_target_slider_default,
                                marks={0: "$0", 250: "$250", 500: "$500", 750: "$750", 1000: "$1000+"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px", "minWidth": "250px"}
                    ),
                ],
            ),
            dcc.Loading(
                id=loading_id,
                type="circle",
                children=[
                    ddk.Graph(id=graph_id, style={"minHeight": "550px", "height": "calc(100vh - 600px)"}),
                ]
            ),
            html.Pre(id=error_id, style={"color": "red", "margin": "10px 0"}),
            ddk.CardFooter(title=description)
        ],
        width=100
    )

    test_inputs: dict[str, TestInput] = {
        size_control_id: {
            "options": [option["value"] for option in size_options],
            "default": size_default
        },
        color_control_id: {
            "options": [option["value"] for option in color_options],
            "default": color_default
        },
        last_price_slider_id: {
            "options": None,
            "default": last_price_slider_default
        },
        price_target_slider_id: {
            "options": None,
            "default": price_target_slider_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _update_logic(**kwargs) -> go.Figure:
    """Core chart update logic without error handling."""
    logger.debug("Updating chart with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{
                "text": "No data is available to display",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig

    logger.debug("Selecting columns for analysis...")
    df = df[['last_price', 'price_target_median', 'expected_upside_pct', 'market_cap', 'volume_shrs', 'sector', 'confidence_level', 'beat_classification', 'ticker', 'name']].copy()
    logger.debug(schema(df))
    logger.debug(tbl(df))

    size_encoding = kwargs.get(size_control_id, size_default)
    if size_encoding is None:
        size_encoding = size_default

    color_encoding = kwargs.get(color_control_id, color_default)
    if color_encoding is None:
        color_encoding = color_default

    last_price_range = kwargs.get(last_price_slider_id, last_price_slider_default)
    if last_price_range is None:
        last_price_range = last_price_slider_default

    price_target_range = kwargs.get(price_target_slider_id, price_target_slider_default)
    if price_target_range is None:
        price_target_range = price_target_slider_default

    logger.debug("Applying last_price range filter: %s...", last_price_range)
    df = df[(df['last_price'] >= last_price_range[0]) & (df['last_price'] <= last_price_range[1])]
    logger.debug(tbl(df))

    logger.debug("Applying price_target range filter: %s...", price_target_range)
    df = df[(df['price_target_median'] >= price_target_range[0]) & (df['price_target_median'] <= price_target_range[1])]
    logger.debug(tbl(df))

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available for selected price range",
            annotations=[{
                "text": "No data is available for the selected price range",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig

    logger.debug("Normalizing size column if needed...")
    size_col = None if size_encoding == "none" else size_encoding

    if size_col == "expected_upside_pct":
        df['size_normalized'] = df['expected_upside_pct'].clip(lower=0) + 1
        size_col = 'size_normalized'
    elif size_col == "market_cap":
        df['size_normalized'] = df['market_cap'].clip(lower=0) + 1
        size_col = 'size_normalized'
    elif size_col == "volume_shrs":
        df['size_normalized'] = df['volume_shrs'].clip(lower=0) + 1
        size_col = 'size_normalized'

    color_col = None if color_encoding == "none" else color_encoding

    logger.debug("Creating scatter plot...")

    if color_col is not None:
        fig = px.scatter(
            df,
            x='last_price',
            y='price_target_median',
            size=size_col,
            color=color_col,
            hover_data={'ticker': True, 'name': True, 'last_price': ':.2f', 'price_target_median': ':.2f', 'expected_upside_pct': ':.2f'},
            labels={
                'last_price': 'Current Price ($)',
                'price_target_median': 'Price Target Median ($)',
                'expected_upside_pct': 'Expected Upside (%)',
                'market_cap': 'Market Cap ($M)',
                'volume_shrs': 'Volume (Shares)',
                'sector': 'Sector',
                'confidence_level': 'Confidence Level',
                'beat_classification': 'Beat Classification'
            }
        )
    else:
        fig = px.scatter(
            df,
            x='last_price',
            y='price_target_median',
            size=size_col,
            hover_data={'ticker': True, 'name': True, 'last_price': ':.2f', 'price_target_median': ':.2f', 'expected_upside_pct': ':.2f'},
            labels={
                'last_price': 'Current Price ($)',
                'price_target_median': 'Price Target Median ($)',
                'expected_upside_pct': 'Expected Upside (%)',
                'market_cap': 'Market Cap ($M)',
                'volume_shrs': 'Volume (Shares)'
            }
        )

    if size_col:
        fig.update_traces(marker=dict(sizemin=6))

    x_min = df['last_price'].min()
    x_max = df['last_price'].max()
    y_min = df['price_target_median'].min()
    y_max = df['price_target_median'].max()

    axis_min = min(x_min, y_min) * 0.9
    axis_max = max(x_max, y_max) * 1.1

    fig.add_shape(
        type="line",
        x0=axis_min,
        y0=axis_min,
        x1=axis_max,
        y1=axis_max,
        line=dict(color="gray", dash="dash", width=2),
        name="Fair Value (y=x)"
    )

    fig.update_layout(
        xaxis_title="Current Price ($)",
        yaxis_title="Price Target Median ($)",
        hovermode="closest",
        xaxis=dict(type="log"),
        yaxis=dict(type="log")
    )

    if color_col:
        fig.update_layout(legend_title_text=color_col.replace('_', ' ').title())

    logger.debug("Done")
    return fig

@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        size_control_id: Input(size_control_id, "value"),
        color_control_id: Input(color_control_id, "value"),
        last_price_slider_id: Input(last_price_slider_id, "value"),
        price_target_slider_id: Input(price_target_slider_id, "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        figure = _update_logic(**kwargs)
        return figure, ""

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg