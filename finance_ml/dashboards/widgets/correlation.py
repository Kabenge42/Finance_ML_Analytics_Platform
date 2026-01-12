"""correlation.py - Dashboard widgets."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from finance_ml.core.constants import CATEGORY_COLORS, PLOTLY_TEMPLATE, COLOR_PALETTE
from finance_ml.core.schema import PHASE93_FEATURE_CATEGORIES
from .base import resolve_reference_date, _write_html_artifact
from .earnings import create_earnings_calendar_dashboard, CategoryMetricsResolver

logger = logging.getLogger(__name__)


def create_category_comparison_chart(
    df: pd.DataFrame,
    categories: Optional[List[str]] = None,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """
    Creates an interactive chart comparing metrics across PHASE93 categories.

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Provides visual comparison of metric availability and values across
    all PHASE93_FEATURE_CATEGORIES categories for earnings calendar companies.

    Args:
        df: Input DataFrame containing stock data.
        categories: List of categories to compare. Defaults to all.
        reference_date: Date for earnings comparison.
        top_n: Number of companies to include.
        output_path: Optional path to save HTML output.

    Returns:
        go.Figure: Plotly figure with category comparison.
    """
    if categories is None:
        categories = list(PHASE93_FEATURE_CATEGORIES.keys())

    reference_date = resolve_reference_date(df, reference_date)

    # Get base dashboard data
    dashboard_df = create_earnings_calendar_dashboard(
        df,
        reference_date=reference_date,
        top_n=top_n,
        mode="all",
    )

    if dashboard_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No companies found with earnings within +/- 10 days",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path)
        return fig

    # Calculate coverage statistics per category
    category_metrics_map = CategoryMetricsResolver.get_metrics(categories)

    coverage_data = []
    for cat, metrics in category_metrics_map.items():
        available = [m for m in metrics if m in df.columns]
        non_null_counts = [
            dashboard_df[m].notna().sum() for m in available if m in dashboard_df.columns
        ]

        coverage_data.append(
            {
                "category": cat,
                "total_metrics": len(metrics),
                "available_metrics": len(available),
                "coverage_pct": len(available) / len(metrics) * 100 if metrics else 0,
                "avg_non_null": (
                    sum(non_null_counts) / len(non_null_counts) if non_null_counts else 0
                ),
                "color": CATEGORY_COLORS.get(cat, COLOR_PALETTE["neutral"]),
            }
        )

    coverage_df = pd.DataFrame(coverage_data)

    # Create figure with two subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Metric Coverage by Category",
            "Data Availability (Avg Non-Null)",
        ],
        horizontal_spacing=0.15,
    )

    # Coverage bar chart
    fig.add_trace(
        go.Bar(
            x=coverage_df["category"],
            y=coverage_df["coverage_pct"],
            marker_color=coverage_df["color"],
            name="Coverage %",
            hovertemplate=("<b>%{x}</b><br>Coverage: %{y:.1f}%<br><extra></extra>"),
        ),
        row=1,
        col=1,
    )

    # Data availability bar chart
    fig.add_trace(
        go.Bar(
            x=coverage_df["category"],
            y=coverage_df["avg_non_null"],
            marker_color=coverage_df["color"],
            name="Avg Non-Null",
            hovertemplate=("<b>%{x}</b><br>Avg Records: %{y:.0f}<br><extra></extra>"),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text="Phase 9.3 Category Metrics: Coverage & Availability Analysis",
            font=dict(size=18),
        ),
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=12),
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_yaxes(title_text="Coverage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Avg Non-Null Records", row=1, col=2)
    fig.update_xaxes(tickangle=45)

    _write_html_artifact(fig, output_path)

    return fig


def create_technical_valuation_dashboard(df: pd.DataFrame, output_dir: Path) -> go.Figure:
    """Create technical analysis dashboard overlaying valuation metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    ema_cols = ["ema_20d", "ema_50d", "ema_100d", "ema_250d"]
    momentum_cols = ["price_momentum_1m", "price_momentum_3m", "price_momentum_6m"]
    valuation_cols = ["p_e_ratio", "ev_ebitda_ratio", "p_s_ratio", "p_e_ntm", "ev_ebitda_ltm"]

    if "ema_20d" in df.columns and "ema_50d" in df.columns:
        df["ema_crossover_bullish"] = df["ema_20d"] > df["ema_50d"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["ema_crossover_score"] = (
                (df["ema_20d"].astype("Float64") - df["ema_50d"].astype("Float64"))
                / df["ema_50d"].replace(0, pd.NA).astype("Float64")
            ) * 100

    if "52w_high_adj" in df.columns and "52w_low_adj" in df.columns and "last_price" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = (
                df["52w_high_adj"].astype("Float64") - df["52w_low_adj"].astype("Float64")
            ).replace(0, pd.NA)
            df["price_52w_position"] = (
                (df["last_price"].astype("Float64") - df["52w_low_adj"].astype("Float64")) / denom
            ) * 100
        df["price_52w_position"] = df["price_52w_position"].clip(0, 100)

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "EMA Crossover Distribution",
            "Momentum Scatter (1M vs 3M)",
            "52W Position by Sector",
            "Valuation vs Momentum",
            "Technical Score Distribution",
            "Bullish/Bearish Ratio by Sector",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    if "ema_crossover_score" in df.columns:
        fig.add_trace(
            go.Histogram(
                x=df["ema_crossover_score"].dropna().clip(-50, 50),
                nbinsx=50,
                marker_color=COLOR_PALETTE["success"],
                name="EMA Crossover %",
            ),
            row=1,
            col=1,
        )

    available_momentum = [c for c in momentum_cols if c in df.columns]
    if len(available_momentum) >= 2:
        fig.add_trace(
            go.Scatter(
                x=df[available_momentum[0]],
                y=df[available_momentum[1]],
                mode="markers",
                marker=dict(
                    size=6,
                    color=df.get("price_52w_position", 50),
                    colorscale="RdYlGn",
                    opacity=0.7,
                ),
                name="Momentum",
            ),
            row=1,
            col=2,
        )

    if "price_52w_position" in df.columns and "sector" in df.columns:
        sector_position = df.groupby("sector")["price_52w_position"].median().dropna()
        if not sector_position.empty:
            fig.add_trace(
                go.Bar(
                    y=sector_position.index,
                    x=sector_position.values,
                    orientation="h",
                    marker_color=COLOR_PALETTE["info"],
                    name="52W Position",
                ),
                row=2,
                col=1,
            )

    if available_momentum and valuation_cols:
        valuation_col = next((c for c in valuation_cols if c in df.columns), None)
        if valuation_col and available_momentum:
            fig.add_trace(
                go.Scatter(
                    x=df[valuation_col],
                    y=df[available_momentum[0]],
                    mode="markers",
                    marker=dict(color=df.get("ema_crossover_score", 0), colorscale="Bluered"),
                    name="Valuation vs Momentum",
                    text=df.get("ticker"),
                ),
                row=2,
                col=2,
            )

    if "ema_crossover_score" in df.columns:
        fig.add_trace(
            go.Box(
                x=df["ema_crossover_score"].dropna(),
                marker_color=COLOR_PALETTE["secondary"],
                name="Technical Score",
                boxmean=True,
            ),
            row=3,
            col=1,
        )

    if "ema_crossover_bullish" in df.columns and "sector" in df.columns:
        bullish_ratio = df.groupby("sector")["ema_crossover_bullish"].mean().dropna()
        if not bullish_ratio.empty:
            fig.add_trace(
                go.Bar(
                    x=bullish_ratio.values * 100,
                    y=bullish_ratio.index,
                    orientation="h",
                    marker_color=COLOR_PALETTE["success"],
                    name="Bullish %",
                ),
                row=3,
                col=2,
            )

    fig.update_layout(
        title="<b>Technical Analysis + Valuation Dashboard</b>",
        template=PLOTLY_TEMPLATE,
        height=900,
        showlegend=True,
    )

    output_path = output_dir / "technical_valuation_dashboard.html"
    _write_html_artifact(fig, output_path)
    return fig


CORRELATION_THRESHOLD = 0.3


def _compute_category_scores(
    df: pd.DataFrame,
    category_mapping: Dict[str, List[str]],
) -> Dict[str, pd.Series]:
    """Compute normalized aggregate scores for each feature category."""
    category_scores: Dict[str, pd.Series] = {}
    for cat_name, features in category_mapping.items():
        available = [f for f in features if f in df.columns]
        if available:
            cat_data = df[available].apply(pd.to_numeric, errors="coerce")
            cat_data = cat_data.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)
            category_scores[cat_name] = cat_data.mean(axis=1)
    return category_scores


def _build_correlation_graph(
    corr_matrix: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
):
    """Build a NetworkX graph from correlation matrix with edges above threshold."""
    import networkx as nx

    G = nx.Graph()
    for cat in corr_matrix.columns:
        G.add_node(cat)

    for i, cat1 in enumerate(corr_matrix.columns):
        for j, cat2 in enumerate(corr_matrix.columns):
            if i < j:
                corr = corr_matrix.loc[cat1, cat2]
                if abs(corr) > threshold:
                    G.add_edge(cat1, cat2, weight=abs(corr), correlation=corr)
    return G


def _create_network_traces(G, pos: dict) -> tuple[list, go.Scatter]:
    """Create Plotly traces for network edges and nodes."""
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        corr = edge[2]["correlation"]
        color = "#00bc8c" if corr > 0 else "#e74c3c"
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=abs(corr) * 5, color=color),
                hoverinfo="none",
            )
        )

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(size=20, color="#3498db"),
        hoverinfo="text",
    )
    return edge_traces, node_trace


def create_category_correlation_network(
    df: pd.DataFrame,
    category_mapping: Optional[Dict[str, List[str]]],
    output_dir: Path,
) -> Optional[go.Figure]:
    """Create interactive network visualization showing correlations between feature categories."""
    if category_mapping is None:
        category_mapping = PHASE93_FEATURE_CATEGORIES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_scores = _compute_category_scores(df, category_mapping)

    if len(category_scores) < 2:
        return None

    category_df = pd.DataFrame(category_scores)
    corr_matrix = category_df.corr()

    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed; skipping category correlation network generation")
        return None

    G = _build_correlation_graph(corr_matrix, threshold=CORRELATION_THRESHOLD)
    pos = nx.spring_layout(G, k=2, iterations=50)

    edge_traces, node_trace = _create_network_traces(G, pos)

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="<b>Feature Category Correlation Network</b><br><sup>Green=Positive, Red=Negative correlation</sup>",
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
    )

    output_path = output_dir / "category_correlation_network.html"
    _write_html_artifact(fig, output_path)
    return fig
