from __future__ import annotations
import pandas as pd
from dash import Input, Output, State, html, callback, ctx
from dash.exceptions import PreventUpdate
from finance_ml.dashboards.components import (
    _safe_options,
    apply_filters,
    _kpi_cards,
    _target_vs_price_scatter,
    _market_cap_distribution,
    _coerce_list,
)


def register_data_callbacks(
    app, data_dir, db_url, load_on_start, initial_df, load_data_csv_first
):

    @app.callback(
        Output("equities-data-store", "data"),
        Output("data-status", "children"),
        Output("sector-dropdown", "options"),
        Output("region-dropdown", "options"),
        Output("country-dropdown", "options"),
        Output("trading-country-dropdown", "options"),
        Output("industry-dropdown", "options"),
        Output("exchange-dropdown", "options"),
        Output("style-class-dropdown", "options"),
        Output("size-class-dropdown", "options"),
        Output("fiscal-quarter-dropdown", "options"),
        Output("fiscal-year-dropdown", "options"),
        Output("earnings-status-dropdown", "options"),
        Output("earnings-report-dropdown", "options"),
        Input("refresh-data-btn", "n_clicks"),
        prevent_initial_call=True,  # Only trigger on explicit button click
    )
    def _refresh_data(_n_clicks):
        df, status_summary = load_data_csv_first(
            data_dir=data_dir,
            db_url=db_url,
        )

        if not df.empty:
            status = f"Rows: {len(df):,} | {status_summary}"
        else:
            status = "No data loaded or ETL failed"

        return (
            df.to_json(orient="split"),
            status,
            _safe_options(df, "sector"),
            _safe_options(df, "region"),
            _safe_options(df, "country"),
            _safe_options(df, "trading_country"),
            _safe_options(df, "industry"),
            _safe_options(df, "exchange"),
            _safe_options(df, "style_class"),
            _safe_options(df, "size_class"),
            _safe_options(df, "fiscal_quarter"),
            _safe_options(df, "fiscal_year"),
            _safe_options(df, "next_earnings_status"),
            _safe_options(df, "next_earnings_report"),
        )

    @app.callback(
        Output("kpi-cards", "children"),
        Output("target-vs-price-scatter", "figure"),
        Output("market-cap-distribution", "figure"),
        Input("equities-data-store", "data"),
        Input("sector-dropdown", "value"),
        Input("region-dropdown", "value"),
        Input("country-dropdown", "value"),
        Input("trading-country-dropdown", "value"),
        Input("industry-dropdown", "value"),
        Input("exchange-dropdown", "value"),
        Input("style-class-dropdown", "value"),
        Input("size-class-dropdown", "value"),
        Input("fiscal-quarter-dropdown", "value"),
        Input("fiscal-year-dropdown", "value"),
        Input("earnings-status-dropdown", "value"),
        Input("earnings-report-dropdown", "value"),
        prevent_initial_call=False,
    )
    def _update_overview(
        data_json,
        sectors,
        regions,
        countries,
        trading_countries,
        industries,
        exchanges,
        style_classes,
        size_classes,
        fiscal_quarters,
        fiscal_years,
        earnings_statuses,
        earnings_reports,
    ):
        # Parse data from store, with proper fallback to initial_df
        df = pd.DataFrame()

        if data_json:
            try:
                df = pd.read_json(data_json, orient="split")
            except Exception:
                pass

        # Fallback to initial_df if store is empty but initial_df has data
        if df.empty and initial_df is not None and not initial_df.empty:
            df = initial_df

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
            fiscal_quarters=_coerce_list(fiscal_quarters),
            fiscal_years=_coerce_list(fiscal_years),
            earnings_statuses=_coerce_list(earnings_statuses),
            earnings_reports=_coerce_list(earnings_reports),
        )

        return (
            _kpi_cards(filtered),
            _target_vs_price_scatter(filtered, use_log_scale=True),
            _market_cap_distribution(filtered),
        )
