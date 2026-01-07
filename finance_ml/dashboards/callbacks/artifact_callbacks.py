from __future__ import annotations
import pandas as pd
from dash import Input, Output, State
from finance_ml.dashboards.components import (
    _list_artifacts,
    _render_artifact,
)


def register_artifact_callbacks(app, initial_df, generate_dashboard_artifacts):

    @app.callback(
        Output("artifact-dropdown", "options"),
        Input("tabs", "value"),
    )
    def _populate_artifact_dropdown(tab_value):
        if tab_value == "artifacts":
            return _list_artifacts()
        return []

    @app.callback(
        Output("artifact-viewer", "children"),
        Input("artifact-dropdown", "value"),
    )
    def _show_artifact(path_str):
        return _render_artifact(path_str)

    @app.callback(
        Output("generate-artifacts-status", "children"),
        Input("generate-artifacts-btn", "n_clicks"),
        State("equities-data-store", "data"),
        prevent_initial_call=True,
    )
    def _generate_artifacts(_n, data_json):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            return "No data available"

        try:
            generate_dashboard_artifacts(df)
            return "✓ Artifacts generated successfully"
        except Exception as e:
            return f"Error: {e}"
