from __future__ import annotations
import pandas as pd
from dash import Input, Output, State, html
from finance_ml.dashboards.components import (
    _monitoring_kpi_cards,
    _coerce_list,
    apply_filters,
)
from finance_ml.dashboards.earnings_widgets import (
    create_earnings_metrics_chart,
)


def register_monitoring_callbacks(app, initial_df, generate_dashboard_artifacts):
    @app.callback(
        Output("monitoring-kpi-cards", "children"),
        Output("monitoring-metrics-fig", "figure"),
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
            from finance_ml.dashboards.components import create_empty_state_figure

            empty = create_empty_state_figure("Monitoring", "No data")
            return [], empty

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

        return (
            _monitoring_kpi_cards(filtered),
            create_earnings_metrics_chart(filtered, segment_by=segment_by),
        )

    @app.callback(
        Output("generate-monitoring-report-status", "children"),
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
