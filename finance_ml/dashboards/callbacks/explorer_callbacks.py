from __future__ import annotations

import pandas as pd
from dash import Input, Output, State

from finance_ml.dashboards.components import (
    build_explorer_column_options,
    _coerce_list,
)
from finance_ml.dashboards.components.data_utils import (
    DEFAULT_EXPLORER_COLUMNS,
)


def register_explorer_callbacks(app, initial_df):
    @app.callback(
        Output("explorer-columns-dropdown", "options"),
        Output("explorer-columns-dropdown", "value"),
        Input("feature-category-dropdown", "value"),
        Input("equities-data-store", "data"),
    )
    def _update_explorer_columns(categories, data_json):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        options, defaults = build_explorer_column_options(
            df,
            categories=_coerce_list(categories),
            base_columns=DEFAULT_EXPLORER_COLUMNS,
        )
        return options, defaults

    @app.callback(
        Output("explorer-table", "columns"),
        Output("explorer-table", "data"),
        Input("explorer-update-btn", "n_clicks"),
        State("equities-data-store", "data"),
        State("explorer-columns-dropdown", "value"),
        State("explorer-row-limit", "value"),
        prevent_initial_call=True,
    )
    def _update_explorer_table(_n, data_json, columns, row_limit):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            return [], []

        selected_cols = columns if columns else DEFAULT_EXPLORER_COLUMNS
        # Filter columns to only those present in df
        selected_cols = [c for c in selected_cols if c in df.columns]

        display_df = df[selected_cols].head(row_limit if row_limit else 100)

        cols = [{"name": i.replace("_", " ").title(), "id": i} for i in selected_cols]
        data = display_df.to_dict("records")
        return cols, data
