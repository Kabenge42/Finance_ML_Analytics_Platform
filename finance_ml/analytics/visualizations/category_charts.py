"""
Category-specific visualization functions for feature analytics.

This module provides reusable chart functions for various feature categories
including analyst sentiment, earnings quality, growth metrics, cash flow,
dividend features, R&D investment, inventory, goodwill & M&A, and CapEx.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Dark theme for Plotly (consistent with feature_analytics.py)
PLOTLY_TEMPLATE = "plotly_dark"


# =============================================================================
# Analyst Sentiment Visualizations
# =============================================================================


def create_analyst_sentiment_histogram(
    df: pd.DataFrame,
    color_by: str = "industry",
    nbins: int = 30,
) -> go.Figure:
    """
    Create histogram of analyst bullish percentage distribution by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with analyst sentiment features
    color_by : str, default "industry"
        Column to use for color grouping
    nbins : int, default 30
        Number of histogram bins

    Returns
    -------
    go.Figure
        Plotly figure with histogram and marginal box plot
    """
    fig = px.histogram(
        df,
        x="analyst_bullish_pct",
        color=color_by,
        title="Analyst Bullish Percentage Distribution by Industry",
        nbins=nbins,
        marginal="box",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def create_analyst_upside_scatter(
    df: pd.DataFrame,
    color_by: str = "industry",
    size_col: str = "market_cap",
    max_upside: float = 1000,
) -> go.Figure:
    """
    Create scatter plot of analyst rating vs upside potential.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with analyst sentiment features
    color_by : str, default "industry"
        Column to use for color grouping
    size_col : str, default "market_cap"
        Column to use for marker size
    max_upside : float, default 1000
        Maximum y-axis value for upside potential

    Returns
    -------
    go.Figure
        Plotly scatter figure
    """
    fig = px.scatter(
        df,
        x="analyst_rating_normalized",
        y="upside_potential",
        color=color_by,
        size=size_col,
        hover_data=["ticker", "name"],
        title="Analyst Rating vs Upside Potential",
    )
    fig.update_layout(
        width=1200,
        height=700,
        yaxis=dict(range=[None, max_upside]),
        template=PLOTLY_TEMPLATE,
    )
    return fig


# =============================================================================
# Earnings Quality Visualizations
# =============================================================================


def create_eps_surprise_histogram(
    df: pd.DataFrame,
    color_by: str = "industry",
    nbins: int = 40,
) -> go.Figure:
    """
    Create histogram of EPS surprise distribution by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with earnings quality features
    color_by : str, default "industry"
        Column to use for color grouping
    nbins : int, default 40
        Number of histogram bins

    Returns
    -------
    go.Figure
        Plotly figure with histogram and marginal violin plot
    """
    fig = px.histogram(
        df,
        x="eps_surprise_pct",
        color=color_by,
        title="EPS Surprise Distribution by Industry",
        nbins=nbins,
        marginal="violin",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def create_eps_trajectory_scatter(
    df: pd.DataFrame,
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of EPS trajectory vs GAAP adjustment gap.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with earnings quality features
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure with color scale
    """
    fig = px.scatter(
        df,
        x="eps_trajectory_score",
        y="gaap_adj_eps_gap_pct",
        color="earnings_quality_score",
        size=size_col,
        hover_data=["ticker", "name"],
        title="EPS Trajectory vs GAAP Adjustment Gap",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


# =============================================================================
# Growth Metrics Visualizations
# =============================================================================


def create_growth_correlation_heatmap(
    df: pd.DataFrame,
    growth_cols: Optional[list] = None,
) -> go.Figure:
    """
    Create correlation heatmap for growth metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with growth metrics
    growth_cols : list, optional
        List of growth metric columns. If None, uses default growth columns.

    Returns
    -------
    go.Figure
        Plotly heatmap figure
    """
    if growth_cols is None:
        growth_cols = [
            "revenue_growth_yoy",
            "ebitda_growth_yoy",
            "eps_yoy_growth",
            "fcf_growth_yoy",
            "revenue_cagr_5y",
        ]

    available_cols = [col for col in growth_cols if col in df.columns]
    if not available_cols:
        fig = go.Figure()
        fig.add_annotation(
            text="No growth columns available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    growth_corr = df[available_cols].corr()
    fig = px.imshow(
        growth_corr,
        text_auto=".2f",
        title="Growth Metrics Correlation Matrix",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def create_revenue_vs_eps_growth_scatter(
    df: pd.DataFrame,
    color_by: str = "industry",
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of revenue growth vs EPS growth.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with growth metrics
    color_by : str, default "industry"
        Column to use for color grouping
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure with reference lines
    """
    fig = px.scatter(
        df,
        x="revenue_growth_yoy",
        y="eps_yoy_growth",
        color=color_by,
        size=size_col,
        hover_data=["ticker", "name", "revenue_cagr_5y"],
        title="Revenue Growth vs EPS Growth (YoY)",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


# =============================================================================
# Cash Flow Visualizations
# =============================================================================


def create_fcf_margin_yield_scatter(
    df: pd.DataFrame,
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of FCF margin vs FCF yield.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with cash flow features
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure with color scale
    """
    fig = px.scatter(
        df,
        x="fcf_margin",
        y="fcf_yield",
        color="fcf_positive_years",
        size=size_col,
        hover_data=["ticker", "name", "self_funding_ratio"],
        title="FCF Margin vs FCF Yield (colored by FCF Positive Years)",
        color_continuous_scale="Greens",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def create_cash_flow_quality_boxplot(
    df: pd.DataFrame,
    group_by: str = "industry",
) -> go.Figure:
    """
    Create box plot of cash flow quality score by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with cash flow features
    group_by : str, default "industry"
        Column to use for grouping

    Returns
    -------
    go.Figure
        Plotly box plot figure
    """
    fig = px.box(
        df,
        x=group_by,
        y="cash_flow_quality_score",
        title="Cash Flow Quality Score by Industry",
        color=group_by,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


# =============================================================================
# Dividend Features Visualizations
# =============================================================================


def create_dividend_yield_payout_scatter(
    df: pd.DataFrame,
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of dividend yield vs payout ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with dividend features
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure with color scale
    """
    fig = px.scatter(
        df,
        x="dividend_payout_ratio",
        y="dividend_yield_ltm",
        color="dividend_streak",
        size=size_col,
        hover_data=["ticker", "name", "fcf_dividend_coverage"],
        title="Dividend Yield vs Payout Ratio (colored by Dividend Streak)",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=500,
        title_font_size=16,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=60, r=40, t=80, b=60),
        xaxis_title="Dividend Payout Ratio",
        yaxis_title="Dividend Yield (LTM)",
    )
    return fig


def create_shareholder_yield_histogram(
    df: pd.DataFrame,
    color_by: str = "industry",
    nbins: int = 40,
) -> go.Figure:
    """
    Create histogram of total shareholder yield distribution by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with dividend features
    color_by : str, default "industry"
        Column to use for color grouping
    nbins : int, default 40
        Number of histogram bins

    Returns
    -------
    go.Figure
        Plotly figure with histogram and marginal box plot
    """
    fig = px.histogram(
        df,
        x="total_shareholder_yield",
        color=color_by,
        title="Total Shareholder Yield Distribution by Industry",
        nbins=nbins,
        marginal="box",
    )
    fig.update_layout(
        height=550,
        title_font_size=16,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=60, r=40, t=80, b=60),
        xaxis_title="Total Shareholder Yield (%)",
        yaxis_title="Count",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


# =============================================================================
# R&D Investment Visualizations
# =============================================================================


def create_rnd_intensity_boxplot(
    df: pd.DataFrame,
    group_by: str = "industry",
) -> go.Figure:
    """
    Create box plot of R&D intensity by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with R&D features
    group_by : str, default "industry"
        Column to use for grouping

    Returns
    -------
    go.Figure
        Plotly box plot figure
    """
    fig = px.box(
        df,
        x=group_by,
        y="rnd_intensity_ltm",
        title="R&D Intensity by Industry",
        color=group_by,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def create_rnd_intensity_growth_scatter(
    df: pd.DataFrame,
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of R&D intensity vs YoY R&D growth.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with R&D features
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure
    """
    fig = px.scatter(
        df,
        x="rnd_intensity_ltm",
        y="rnd_yoy_growth",
        color="high_rnd_intensity_flag",
        size=size_col,
        hover_data=["ticker", "name", "rnd_per_employee"],
        title="R&D Intensity vs YoY R&D Growth",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def create_rnd_per_employee_histogram(
    df: pd.DataFrame,
    color_by: str = "industry",
    nbins: int = 30,
) -> go.Figure:
    """
    Create histogram of R&D per employee distribution by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with R&D features
    color_by : str, default "industry"
        Column to use for color grouping
    nbins : int, default 30
        Number of histogram bins

    Returns
    -------
    go.Figure
        Plotly histogram figure
    """
    df_filtered = df[df["rnd_per_employee"].notna()]
    fig = px.histogram(
        df_filtered,
        x="rnd_per_employee",
        color=color_by,
        nbins=nbins,
        title="R&D per Employee Distribution by Industry",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


# =============================================================================
# Inventory Visualizations
# =============================================================================


def create_inventory_days_turnover_scatter(
    df: pd.DataFrame,
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of inventory days vs turnover.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with inventory features
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure
    """
    fig = px.scatter(
        df,
        x="inventory_days",
        y="inventory_turnover",
        color="inventory_buildup_flag",
        size=size_col,
        hover_data=["ticker", "name", "inventory_yoy_change"],
        title="Inventory Days vs Turnover (flagged for buildup)",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


# =============================================================================
# Goodwill & M&A Visualizations
# =============================================================================


def create_goodwill_concentration_boxplot(
    df: pd.DataFrame,
    group_by: str = "industry",
) -> go.Figure:
    """
    Create box plot of goodwill concentration by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with goodwill features
    group_by : str, default "industry"
        Column to use for grouping

    Returns
    -------
    go.Figure
        Plotly box plot figure
    """
    fig = px.box(
        df,
        x=group_by,
        y="goodwill_concentration",
        title="Goodwill Concentration by Industry",
        color=group_by,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def create_goodwill_impairment_scatter(
    df: pd.DataFrame,
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of goodwill 3Y growth vs impairment risk score.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with goodwill features
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure
    """
    fig = px.scatter(
        df,
        x="goodwill_3y_growth",
        y="impairment_risk_score",
        color="recent_acquisition_flag",
        size=size_col,
        hover_data=["ticker", "name", "goodwill_concentration"],
        title="Goodwill 3Y Growth vs Impairment Risk Score",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def create_acquisition_activity_histogram(
    df: pd.DataFrame,
    color_by: str = "industry",
) -> go.Figure:
    """
    Create histogram of recent acquisition activity by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with M&A features
    color_by : str, default "industry"
        Column to use for color grouping

    Returns
    -------
    go.Figure
        Plotly histogram figure
    """
    fig = px.histogram(
        df,
        x="recent_acquisition_flag",
        color=color_by,
        title="Recent Acquisition Activity by Industry",
        barmode="group",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


# =============================================================================
# CapEx & Investment Visualizations
# =============================================================================


def create_capex_growth_scatter(
    df: pd.DataFrame,
    color_by: str = "industry",
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of CapEx YoY growth vs 5Y average comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with CapEx features
    color_by : str, default "industry"
        Column to use for color grouping
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure with reference lines
    """
    fig = px.scatter(
        df,
        x="capex_yoy_growth",
        y="capex_vs_5y_avg",
        color=color_by,
        size=size_col,
        hover_data=["ticker", "name", "investment_efficiency"],
        title="CapEx YoY Growth vs 5Y Average Comparison",
    )
    fig.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="5Y Avg")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def create_investment_efficiency_boxplot(
    df: pd.DataFrame,
    group_by: str = "industry",
) -> go.Figure:
    """
    Create box plot of investment efficiency by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with investment features
    group_by : str, default "industry"
        Column to use for grouping

    Returns
    -------
    go.Figure
        Plotly box plot figure
    """
    fig = px.box(
        df,
        x=group_by,
        y="investment_efficiency",
        title="Investment Efficiency by Industry",
        color=group_by,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def create_ma_intensity_histogram(
    df: pd.DataFrame,
    color_by: str = "industry",
    nbins: int = 30,
) -> go.Figure:
    """
    Create histogram of M&A intensity score distribution by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with M&A features
    color_by : str, default "industry"
        Column to use for color grouping
    nbins : int, default 30
        Number of histogram bins

    Returns
    -------
    go.Figure
        Plotly figure with histogram and marginal box plot
    """
    fig = px.histogram(
        df,
        x="ma_intensity_score",
        color=color_by,
        title="M&A Intensity Score Distribution by Industry",
        nbins=nbins,
        marginal="box",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


# =============================================================================
# Multi-Category & Advanced Visualizations
# =============================================================================


def create_valuation_violin_plot(
    df: pd.DataFrame,
    metric: str = "p_e_ratio",
    group_by: str = "industry",
    max_val: Optional[float] = 100.0,
) -> go.Figure:
    """
    Create violin plot for valuation metrics across different groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    metric : str, default "p_e_ratio"
        Valuation metric to plot
    group_by : str, default "industry"
        Grouping column
    max_val : float, optional
        Maximum value to show (to filter outliers)

    Returns
    -------
    go.Figure
        Plotly violin plot figure
    """
    df_plot = df.copy()
    if max_val is not None and metric in df_plot.columns:
        df_plot = df_plot[df_plot[metric] <= max_val]

    fig = px.violin(
        df_plot,
        x=group_by,
        y=metric,
        color=group_by,
        box=True,
        points="all",
        hover_data=["ticker", "name"],
        title=f"{metric.replace('_', ' ').title()} Distribution by {group_by.title()}",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False)
    fig.update_xaxes(tickangle=45)
    return fig


def create_quality_risk_radar_chart(
    df: pd.DataFrame,
    ticker: str,
    metrics: Optional[list[str]] = None,
) -> go.Figure:
    """
    Create radar chart for a specific stock's quality and risk metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    ticker : str
        Ticker symbol to highlight
    metrics : list, optional
        List of metrics for radar axes. Defaults to key quality scores.

    Returns
    -------
    go.Figure
        Plotly radar chart figure
    """
    if metrics is None:
        metrics = [
            "piotroski_f_score",
            "distress_risk_score",
            "eps_trajectory_score",
            "earnings_quality_score",
            "cash_flow_quality_score",
        ]

    # Filter for the specific ticker
    stock_data = df[df["ticker"] == ticker]
    if stock_data.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"Ticker {ticker} not found", showarrow=False)
        return fig

    # Prepare values (normalized to 0-100 where needed)
    values = []
    for m in metrics:
        val = stock_data[m].iloc[0] if m in stock_data.columns else 0
        if m == "piotroski_f_score":
            val = (val / 9) * 100
        values.append(val)

    # Close the radar loop
    metrics_label = [m.replace("_", " ").title() for m in metrics]
    metrics_label.append(metrics_label[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=metrics_label,
            fill="toself",
            name=ticker,
            line_color="cyan",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        showlegend=True,
        title=f"Quality & Risk Radar: {ticker}",
        template=PLOTLY_TEMPLATE,
    )

    return fig


def create_leverage_liquidity_bubble_chart(
    df: pd.DataFrame,
    size_col: str = "market_cap",
    color_by: str = "industry",
) -> go.Figure:
    """
    Create bubble chart of leverage vs liquidity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    size_col : str, default "market_cap"
        Column for bubble size
    color_by : str, default "industry"
        Column for bubble color

    Returns
    -------
    go.Figure
        Plotly bubble chart
    """
    fig = px.scatter(
        df,
        x="current_ratio",
        y="debt_to_equity",
        size=size_col,
        color=color_by,
        hover_data=["ticker", "name"],
        title="Leverage (D/E) vs Liquidity (Current Ratio)",
        labels={
            "current_ratio": "Current Ratio (Liquidity)",
            "debt_to_equity": "Debt to Equity (Leverage)",
        },
    )

    # Add reference lines for healthy levels
    fig.add_vline(x=1.5, line_dash="dash", line_color="green", annotation_text="Healthy Liquidity")
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="High Leverage")

    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig
