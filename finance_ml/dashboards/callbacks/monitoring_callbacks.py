from __future__ import annotations

import pandas as pd
import plotly.express as px
from dash import Input, Output, State

from finance_ml.dashboards.components import (
    _monitoring_kpi_cards,
    _coerce_list,
    apply_filters,
    create_empty_state_figure,
)
from finance_ml.dashboards.components.constants import PLOTLY_TEMPLATE


def register_monitoring_callbacks(app, initial_df, generate_dashboard_artifacts):

    @app.callback(
        Output("monitoring-kpi-row", "children"),
        Output("monitoring-growth-fig", "figure"),
        Output("monitoring-margin-fig", "figure"),
        Output("monitoring-quality-fig", "figure"),
        Input("equities-data-store", "data"),
        Input("monitoring-segment-by", "value"),
        Input("sector-dropdown", "value"),
        Input("region-dropdown", "value"),
        Input("country-dropdown", "value"),
        Input("trading-country-dropdown", "value"),
        Input("industry-dropdown", "value"),
        Input("exchange-dropdown", "value"),
        Input("style-class-dropdown", "value"),
        Input("size-class-dropdown", "value"),
    )
    def _update_monitoring_tab(
        data_json,
        segment_by,
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
            empty = create_empty_state_figure("Monitoring", "No data")
            return [], empty, empty, empty

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

        if filtered.empty:
            empty = create_empty_state_figure("Monitoring", "No data matching filters")
            return [], empty, empty, empty

        return (
            _monitoring_kpi_cards(filtered),
            _monitoring_growth_fig(filtered, segment_by),
            _monitoring_margin_fig(filtered, segment_by),
            _monitoring_quality_fig(filtered, segment_by),
        )

    @app.callback(
        Output("monitoring-report-status", "children"),
        Input("generate-monitoring-report-btn", "n_clicks"),
        State("equities-data-store", "data"),
        prevent_initial_call=True,
    )
    def _generate_monitoring_report(_n, data_json):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            return "No data available"

        try:
            generate_dashboard_artifacts(df)
            return "✓ Monitoring report generated successfully"
        except Exception as e:
            return f"Error: {e}"


def _monitoring_growth_fig(df: pd.DataFrame, segment_by: str):
    """Create growth monitoring chart."""
    col = "total_revenues_cagr_5y_fy"
    if col not in df.columns:
        return create_empty_state_figure("Growth Monitoring", f"Column {col} missing")

    plot_df = df[df[col].notna()].copy()
    if plot_df.empty:
        return create_empty_state_figure("Growth Monitoring", "No growth data available")

    # Group by segment
    agg_df = plot_df.groupby(segment_by)[col].median().reset_index()
    agg_df = agg_df.sort_values(col, ascending=False)

    fig = px.bar(
        agg_df,
        x=segment_by,
        y=col,
        title=f"Median 5Y Revenue CAGR by {segment_by.title()}",
        template=PLOTLY_TEMPLATE,
        color=col,
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(yaxis_title="Median CAGR (%)")
    return fig


def _monitoring_margin_fig(df: pd.DataFrame, segment_by: str):
    """Create margin monitoring chart."""
    col = "net_income_margin_pct_ltm"
    if col not in df.columns:
        return create_empty_state_figure("Margin Monitoring", f"Column {col} missing")

    plot_df = df[df[col].notna()].copy()
    if plot_df.empty:
        return create_empty_state_figure("Margin Monitoring", "No margin data available")

    fig = px.box(
        plot_df,
        x=segment_by,
        y=col,
        title=f"Net Margin Distribution by {segment_by.title()}",
        template=PLOTLY_TEMPLATE,
        color=segment_by,
    )
    fig.update_layout(yaxis_title="Net Margin (%)", showlegend=False)
    return fig


def _monitoring_quality_fig(df: pd.DataFrame, segment_by: str):
    """Create quality monitoring chart."""
    col = "earnings_quality_score"
    if col not in df.columns:
        # Try to compute it if possible or show missing
        from finance_ml.dashboards.widgets.earnings import analyze_earnings_quality

        df_quality = analyze_earnings_quality(df)
        if col in df_quality.columns:
            plot_df = df_quality
        else:
            return create_empty_state_figure("Quality Monitoring", "Earnings Quality Score missing")
    else:
        plot_df = df.copy()

    plot_df = plot_df[plot_df[col].notna()].copy()
    if plot_df.empty:
        return create_empty_state_figure("Quality Monitoring", "No quality data available")

    agg_df = plot_df.groupby(segment_by)[col].mean().reset_index()
    agg_df = agg_df.sort_values(col, ascending=False)

    fig = px.bar(
        agg_df,
        x=segment_by,
        y=col,
        title=f"Average Earnings Quality Score by {segment_by.title()}",
        template=PLOTLY_TEMPLATE,
        color=col,
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
    )
    fig.update_layout(yaxis_title="Quality Score (0-100)")
    return fig
