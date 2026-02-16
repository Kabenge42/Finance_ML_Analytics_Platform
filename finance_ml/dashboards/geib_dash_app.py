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


def load_geib_data():
    """Load all necessary data for the GEIB dashboard.
    
    Returns:
        dict: Dictionary containing DataFrames for different components
    """
    data = {
        'summary': pd.DataFrame(),
        'tri_model': pd.DataFrame(),
        'earnings': pd.DataFrame(),
        'credit': pd.DataFrame()
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
        SELECT 
            ticker, name, region, country, exchange, sector, industry,
            prob_positive_upside, last_price, expected_upside_pct,
            filtered_upside, expected_return_prob_weighted,
            achievement_probability, confidence_level,
            posterior_beat_prob, confidence_score, beat_classification,
            mc_bullish, kal_bullish, pt_bullish, earn_bullish,
            agreement_score, signal
        FROM analytics.expected_returns_summary
        WHERE expected_upside_pct IS NOT NULL
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

# Initialize Dash app
app = dash.Dash(
    __name__, 
    title="Global Equity Investment Board", 
    external_stylesheets=[dbc.themes.DARKLY]
)
server = app.server


# Layout
app.layout = html.Div(
    [
        html.H1("🌍 Global Equity Investment Board (GEIB)", style={"textAlign": "center"}),
        html.P(
            "Expected Returns Analysis from Tri-Model Consensus (Monte Carlo, Kalman Filter, Price Target Achievement)",
            style={"textAlign": "center", "fontStyle": "italic", "color": "#999"}
        ),
        
        # Status indicator
        html.Div(
            id="status-indicator",
            children=[
                html.Span(
                    f"✅ Data Loaded: {len(df):,} stocks" if len(df) > 0 else "⚠️ No data loaded",
                    style={
                        "margin": "0 10px", 
                        "color": "green" if len(df) > 0 else "orange"
                    }
                )
            ],
            style={"textAlign": "center", "padding": "10px"}
        ),
        
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
                        html.Div(
                            [
                                html.Label("Region", className="filter-label"),
                                dcc.Dropdown(
                                    id="region-dropdown",
                                    multi=True,
                                    options=(
                                        [{"label": i, "value": i} for i in sorted(df["region"].dropna().unique())]
                                        if "region" in df.columns and len(df) > 0
                                        else []
                                    ),
                                ),
                            ],
                            className="filter-item",
                            style={"width": "23%", "display": "inline-block", "margin": "5px"}
                        ),
                        html.Div(
                            [
                                html.Label("Country", className="filter-label"),
                                dcc.Dropdown(
                                    id="country-dropdown",
                                    multi=True,
                                    options=(
                                        [{"label": i, "value": i} for i in sorted(df["country"].dropna().unique())]
                                        if "country" in df.columns and len(df) > 0
                                        else []
                                    ),
                                ),
                            ],
                            className="filter-item",
                            style={"width": "23%", "display": "inline-block", "margin": "5px"}
                        ),
                        html.Div(
                            [
                                html.Label("Exchange", className="filter-label"),
                                dcc.Dropdown(
                                    id="exchange-dropdown",
                                    multi=True,
                                    options=(
                                        [{"label": i, "value": i} for i in sorted(df["exchange"].dropna().unique())]
                                        if "exchange" in df.columns and len(df) > 0
                                        else []
                                    ),
                                ),
                            ],
                            className="filter-item",
                            style={"width": "23%", "display": "inline-block", "margin": "5px"}
                        ),
                        html.Div(
                            [
                                html.Label("Sector", className="filter-label"),
                                dcc.Dropdown(
                                    id="sector-dropdown",
                                    multi=True,
                                    options=(
                                        [{"label": i, "value": i} for i in sorted(df["sector"].dropna().unique())]
                                        if "sector" in df.columns and len(df) > 0
                                        else []
                                    ),
                                ),
                            ],
                            className="filter-item",
                            style={"width": "23%", "display": "inline-block", "margin": "5px"}
                        ),
                        html.Div(
                            [
                                html.Label("Industry", className="filter-label"),
                                dcc.Dropdown(
                                    id="industry-dropdown",
                                    multi=True,
                                    options=(
                                        [{"label": i, "value": i} for i in sorted(df["industry"].dropna().unique())]
                                        if "industry" in df.columns and len(df) > 0
                                        else []
                                    ),
                                ),
                            ],
                            className="filter-item",
                            style={"width": "23%", "display": "inline-block", "margin": "5px"}
                        ),
                        html.Div(
                            [
                                html.Label("Signal", className="filter-label"),
                                dcc.Dropdown(
                                    id="signal-dropdown",
                                    multi=True,
                                    options=(
                                        [{"label": i, "value": i} for i in sorted(df["signal"].dropna().unique())]
                                        if "signal" in df.columns and len(df) > 0
                                        else []
                                    ),
                                ),
                            ],
                            className="filter-item",
                            style={"width": "23%", "display": "inline-block", "margin": "5px"}
                        ),
                        html.Div(
                            [
                                html.Label("Confidence Level", className="filter-label"),
                                dcc.Dropdown(
                                    id="confidence-dropdown",
                                    multi=True,
                                    options=(
                                        [{"label": i, "value": i} for i in sorted(df["confidence_level"].dropna().unique())]
                                        if "confidence_level" in df.columns and len(df) > 0
                                        else []
                                    ),
                                ),
                            ],
                            className="filter-item",
                            style={"width": "23%", "display": "inline-block", "margin": "5px"}
                        ),
                    ],
                    style={"display": "flex", "flexWrap": "wrap", "justifyContent": "space-around"}
                ),
            ],
            style={"padding": "20px", "backgroundColor": "#222", "margin": "10px", "borderRadius": "5px"}
        ),
        
        # Tabs for different views
        dcc.Tabs(
            [
                dcc.Tab(
                    label="📊 Expected Returns Overview",
                    children=[
                        html.Div([
                            dcc.Graph(id="returns-scatter"),
                            dcc.Graph(id="agreement-heatmap"),
                        ])
                    ]
                ),
                dcc.Tab(
                    label="🎯 Model Consensus",
                    children=[
                        html.Div([
                            dcc.Graph(id="model-signals-plot"),
                            dcc.Graph(id="confidence-distribution"),
                        ])
                    ]
                ),
                dcc.Tab(
                    label="🏆 Top Opportunities",
                    children=[
                        html.Div([
                            html.H3("High Conviction Opportunities", style={"textAlign": "center"}),
                            dash_table.DataTable(
                                id="top-opportunities-table",
                                page_size=20,
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
                        ])
                    ]
                ),
                dcc.Tab(
                    label="📈 Signal Analysis",
                    children=[
                        html.Div([
                            dcc.Graph(id="signal-breakdown"),
                            dcc.Graph(id="regional-performance"),
                        ])
                    ]
                ),
                dcc.Tab(
                    label="🏆 Risk-Adjusted Ranking",
                    children=[
                        html.Div([
                            html.H3("Expected Value Risk-Adjusted Ranking", style={"textAlign": "center", "marginTop": "20px"}),
                            html.P("Rank stocks by their risk-adjusted expected value, combining upside potential, probability of success, and confidence levels.", 
                                   style={"textAlign": "center", "fontStyle": "italic", "color": "#999"}),
                            
                            # Ranking Filters
                            html.Div([
                                html.Div([
                                    html.Label("Scoring Method", style={"color": "white"}),
                                    dcc.Dropdown(
                                        id="scoring-method-dropdown",
                                        options=[
                                            {"label": "Base EV", "value": "base_ev"},
                                            {"label": "Probability-weighted", "value": "prob_weighted"},
                                            {"label": "Confidence-adjusted", "value": "confidence_adj"},
                                            {"label": "Achievement-adjusted", "value": "achievement_adj"},
                                            {"label": "Combined", "value": "combined"}
                                        ],
                                        value="combined",
                                        style={"color": "black"}
                                    ),
                                ], style={"width": "18%", "display": "inline-block", "margin": "10px"}),
                                
                                html.Div([
                                    html.Label("Min Agreement", style={"color": "white"}),
                                    dcc.Dropdown(
                                        id="min-agreement-dropdown",
                                        options=[{"label": str(i), "value": i} for i in [0, 1, 2, 3, 4]],
                                        value=2,
                                        style={"color": "black"}
                                    ),
                                ], style={"width": "18%", "display": "inline-block", "margin": "10px"}),
                                
                                html.Div([
                                    html.Label("Min Confidence", style={"color": "white"}),
                                    dcc.Dropdown(
                                        id="min-confidence-dropdown",
                                        options=[{"label": str(i), "value": i} for i in [0.15, 0.25, 0.35, 0.45]],
                                        value=0.25,
                                        style={"color": "black"}
                                    ),
                                ], style={"width": "18%", "display": "inline-block", "margin": "10px"}),
                                
                                html.Div([
                                    html.Label("Risk-Free Rate", style={"color": "white"}),
                                    dcc.Dropdown(
                                        id="risk-free-rate-dropdown",
                                        options=[{"label": f"{i}%", "value": float(i)} for i in [0, 2, 3, 4, 5]],
                                        value=3.0,
                                        style={"color": "black"}
                                    ),
                                ], style={"width": "18%", "display": "inline-block", "margin": "10px"}),
                                
                                html.Div([
                                    html.Label("Scatter Color By", style={"color": "white"}),
                                    dcc.Dropdown(
                                        id="scatter-color-dropdown",
                                        options=[
                                            {"label": "None", "value": "none"},
                                            {"label": "Signal", "value": "signal"}
                                        ],
                                        value="signal",
                                        style={"color": "black"}
                                    ),
                                ], style={"width": "18%", "display": "inline-block", "margin": "10px"}),
                                
                                html.Div([
                                    html.Label("Select Tickers (Probabilistic)", style={"color": "white"}),
                                    dcc.Dropdown(
                                        id="prob-ticker-dropdown",
                                        multi=True,
                                        options=(
                                            [{"label": i, "value": i} for i in sorted(df["ticker"].dropna().unique())]
                                            if "ticker" in df.columns and len(df) > 0
                                            else []
                                        ),
                                        placeholder="Select tickers for detailed analysis...",
                                        style={"color": "black"}
                                    ),
                                ], style={"width": "97%", "display": "inline-block", "margin": "10px"}),
                            ], style={"backgroundColor": "#333", "padding": "10px", "borderRadius": "5px", "margin": "10px"}),
                            
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="ranking-bar-chart")
                                ], width=6),
                                dbc.Col([
                                    dcc.Graph(id="risk-reward-scatter")
                                ], width=6),
                            ])
                        ])
                    ]
                ),
                dcc.Tab(
                    label="🔮 Probabilistic Analysis",
                    children=[
                        html.Div([
                            html.H3("Bayesian Probabilistic Analysis", style={"textAlign": "center", "marginTop": "20px"}),
                            html.P("ArviZ-enhanced visualizations for posterior returns, beat probabilities, and ruin diagnostics.", 
                                   style={"textAlign": "center", "fontStyle": "italic", "color": "#999"}),
                            
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="posterior-forest-plot")
                                ], width=6),
                                dbc.Col([
                                    dcc.Graph(id="tri-model-posterior")
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="beat-prob-posterior")
                                ], width=6),
                                dbc.Col([
                                    dcc.Graph(id="ruin-diagnostic")
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="category-ridge-plot")
                                ], width=12),
                            ]),
                        ])
                    ]
                ),
                dcc.Tab(
                    label="🎲 Monte Carlo Simulator",
                    children=[
                        html.Div([
                            html.H3("Monte Carlo Portfolio Outcome Simulator", style={"textAlign": "center", "marginTop": "20px"}),
                            html.P("Simulate thousands of possible portfolio outcomes based on expected returns and probabilities. See the range of potential results and the likelihood of achieving your target return.",
                                   style={"textAlign": "center", "fontStyle": "italic", "color": "#999"}),
                            
                            # Monte Carlo Filters
                            html.Div([
                                html.Div([
                                    html.Label("Simulations:", style={"color": "white"}),
                                    dcc.Dropdown(
                                        id="mc-num-simulations",
                                        options=[
                                            {"label": "1,000", "value": 1000},
                                            {"label": "5,000", "value": 5000},
                                            {"label": "10,000", "value": 10000},
                                            {"label": "50,000", "value": 50000}
                                        ],
                                        value=10000,
                                        style={"color": "black"}
                                    ),
                                ], style={"width": "15%", "display": "inline-block", "margin": "10px"}),
                                
                                html.Div([
                                    html.Label("Loss Ratio:", style={"color": "white"}),
                                    dcc.Dropdown(
                                        id="mc-loss-ratio",
                                        options=[
                                            {"label": "0.25 (25%)", "value": 0.25},
                                            {"label": "0.5 (50%)", "value": 0.5},
                                            {"label": "0.75 (75%)", "value": 0.75},
                                            {"label": "1.0 (100%)", "value": 1.0}
                                        ],
                                        value=0.5,
                                        style={"color": "black"}
                                    ),
                                ], style={"width": "15%", "display": "inline-block", "margin": "10px"}),
                                
                                html.Div([
                                    html.Label("Weighting:", style={"color": "white"}),
                                    dcc.Dropdown(
                                        id="mc-weighting",
                                        options=[
                                            {"label": "Equal-weighted", "value": "equal"},
                                            {"label": "Kelly-weighted", "value": "kelly"},
                                            {"label": "Market cap proxy", "value": "market_cap"}
                                        ],
                                        value="equal",
                                        style={"color": "black"}
                                    ),
                                ], style={"width": "15%", "display": "inline-block", "margin": "10px"}),
                                
                                html.Div([
                                    html.Label("Target Return:", style={"color": "white"}),
                                    dcc.Dropdown(
                                        id="mc-target-return",
                                        options=[
                                            {"label": "0%", "value": 0.0},
                                            {"label": "5%", "value": 5.0},
                                            {"label": "10%", "value": 10.0},
                                            {"label": "15%", "value": 15.0},
                                            {"label": "20%", "value": 20.0}
                                        ],
                                        value=10.0,
                                        style={"color": "black"}
                                    ),
                                ], style={"width": "15%", "display": "inline-block", "margin": "10px"}),
                                
                                html.Div([
                                    html.Label("Signal Filter:", style={"color": "white"}),
                                    dcc.Dropdown(
                                        id="mc-signal-filter",
                                        options=[
                                            {"label": "Strong Bullish (4/4)", "value": "strong_buy"},
                                            {"label": "Bullish (3/4)", "value": "buy"},
                                            {"label": "Neutral (2/4)", "value": "hold"},
                                            {"label": "Bearish (1/4)", "value": "sell"},
                                            {"label": "Strong Bearish (0/4)", "value": "strong_sell"}
                                        ],
                                        value=["strong_buy", "buy"],
                                        multi=True,
                                        style={"color": "black"}
                                    ),
                                ], style={"width": "30%", "display": "inline-block", "margin": "10px"}),
                            ], style={"backgroundColor": "#333", "padding": "10px", "borderRadius": "5px", "margin": "10px"}),
                            
                            # Stats Display
                            html.Div(
                                id="mc-stats-display",
                                style={"backgroundColor": "#f5f5f5", "padding": "15px", "margin": "10px", "borderRadius": "5px", "color": "black"}
                            ),
                            
                            # Charts
                            dbc.Row([
                                dbc.Col([
                                    html.H4("Percentile Distribution", style={"textAlign": "center"}),
                                    dcc.Graph(id="mc-percentile-chart")
                                ], width=6),
                                dbc.Col([
                                    html.H4("Return Distribution", style={"textAlign": "center"}),
                                    dcc.Graph(id="mc-distribution-chart")
                                ], width=6),
                            ]),
                        ])
                    ]
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


@app.callback(
    [
        Output("kpi-cards", "children"),
        Output("returns-scatter", "figure"),
        Output("agreement-heatmap", "figure"),
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
    [
        Input("region-dropdown", "value"),
        Input("country-dropdown", "value"),
        Input("exchange-dropdown", "value"),
        Input("sector-dropdown", "value"),
        Input("industry-dropdown", "value"),
        Input("signal-dropdown", "value"),
        Input("confidence-dropdown", "value"),
        Input("scoring-method-dropdown", "value"),
        Input("min-agreement-dropdown", "value"),
        Input("min-confidence-dropdown", "value"),
        Input("risk-free-rate-dropdown", "value"),
        Input("scatter-color-dropdown", "value"),
        Input("prob-ticker-dropdown", "value"),
    ],
)
def update_dashboard(regions, countries, exchanges, sectors, industries, signals, confidence_levels, scoring_method, min_agreement, min_confidence, risk_free_rate, scatter_color, prob_tickers):
    """Update dashboard visualizations based on selected filters."""
    filtered_df = df.copy()
    
    # 1. Apply Categorical Filters
    if regions:
        filtered_df = filtered_df[filtered_df["region"].isin(regions)]
    if countries:
        filtered_df = filtered_df[filtered_df["country"].isin(countries)]
    if exchanges:
        filtered_df = filtered_df[filtered_df["exchange"].isin(exchanges)]
    if sectors:
        filtered_df = filtered_df[filtered_df["sector"].isin(sectors)]
    if industries:
        filtered_df = filtered_df[filtered_df["industry"].isin(industries)]
    if signals:
        filtered_df = filtered_df[filtered_df["signal"].isin(signals)]
    if confidence_levels:
        filtered_df = filtered_df[filtered_df["confidence_level"].isin(confidence_levels)]

    # 2. Apply Numerical Threshold Filters
    if min_agreement is not None:
        filtered_df = filtered_df[filtered_df["agreement_score"] >= min_agreement]
    
    if min_confidence is not None:
        filtered_df = filtered_df[filtered_df["confidence_score"] >= min_confidence]

    # 3. Handle Empty States
    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(title="No data matching selected filters")
        return [[]] + [empty_fig] * 14

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
        risk_free_rate_decimal = risk_free_rate / 100
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
        strong_buy = len(filtered_df[filtered_df["signal"] == "strong_buy"]) if "signal" in filtered_df.columns else 0

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

    # Expected Return Heatmap
    agreement_heatmap = {}
    if not filtered_df.empty and all(col in filtered_df.columns for col in ["industry", "exchange", "expected_return_prob_weighted"]):
        pivot = filtered_df.pivot_table(
            values="expected_return_prob_weighted",
            index="industry",
            columns="exchange",
            aggfunc="mean"
        )
        agreement_heatmap = px.imshow(
            pivot,
            text_auto=".1f",
            title="Expected Return Score by Industry & Exchange",
            labels={"color": "Avg Expected Return (%)"},
            color_continuous_scale="RdYlGn",
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
            "ticker", "name", "sector", "region", "expected_return_prob_weighted",
            "achievement_probability", "confidence_level", "signal", "agreement_score"
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

        # 2. Tri-Model Comparison
        if not df_tri.empty:
            tri_model_post = probability_viz.create_tri_model_posterior_comparison(
                df_tri, tickers=prob_tickers, top_n=8
            )

        # 3. Beat Probability Posterior
        if not df_earnings.empty:
            beat_prob_post = probability_viz.create_beat_probability_posterior(
                df_earnings, tickers=prob_tickers, top_n=10
            )

        # 4. Ruin Probability Diagnostic
        if not df_credit.empty:
            ruin_diag = probability_viz.create_ruin_probability_diagnostic(
                df_credit, top_n=15
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
        agreement_heatmap,
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


@app.callback(
    [
        Output("mc-percentile-chart", "figure"),
        Output("mc-distribution-chart", "figure"),
        Output("mc-stats-display", "children"),
    ],
    [
        Input("region-dropdown", "value"),
        Input("sector-dropdown", "value"),
        Input("mc-num-simulations", "value"),
        Input("mc-loss-ratio", "value"),
        Input("mc-weighting", "value"),
        Input("mc-target-return", "value"),
        Input("mc-signal-filter", "value"),
    ],
)
def update_monte_carlo(regions, sectors, num_simulations, loss_ratio, weighting, target_return, signal_filter):
    """Update Monte Carlo simulation visualizations."""
    # Filter data
    mc_df = df.copy()
    if regions:
        mc_df = mc_df[mc_df["region"].isin(regions)]
    if sectors:
        mc_df = mc_df[mc_df["sector"].isin(sectors)]
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
