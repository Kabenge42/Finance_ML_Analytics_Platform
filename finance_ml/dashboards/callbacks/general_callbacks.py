from __future__ import annotations
import pandas as pd
from dash import Input, Output, State, html
import plotly.express as px
from finance_ml.dashboards.components import (
    _coerce_list,
    apply_filters,
    compute_surprise,
    create_empty_state_figure,
    validate_required_columns,
    create_missing_columns_warning,
)
from finance_ml.dashboards.earnings_widgets import (
    get_category_metrics,
    create_category_comparison_chart,
)

# Standard metrics for Est vs Actual tab
EST_ACTUAL_METRICS = {
    "EPS": {
        "actual": "eps_actual_ltm_fy",
        "estimate": "eps_est_avg_ltm_fy",
        "adjusted": "eps_actual_ltm_fy",
        "gaap": "eps_actual_gaap_ltm_fy",
        "revisions": "eps_est_avg_rev_pct_1m",
    },
    "Revenue": {
        "actual": "total_revenues_actual_ltm_fy",
        "estimate": "total_revenues_est_avg_ltm_fy",
        "revisions": "total_revenues_est_avg_rev_pct_1m",
    },
    "EBITDA": {
        "actual": "ebitda_actual_ltm_fy",
        "estimate": "ebitda_est_avg_ltm_fy",
        "revisions": "ebitda_est_avg_rev_pct_1m",
    },
}


def register_general_callbacks(app, initial_df):
    @app.callback(
        Output("sector-dropdown", "value"),
        Output("region-dropdown", "value"),
        Output("country-dropdown", "value"),
        Output("trading-country-dropdown", "value"),
        Output("industry-dropdown", "value"),
        Output("exchange-dropdown", "value"),
        Output("style-class-dropdown", "value"),
        Output("size-class-dropdown", "value"),
        Input("reset-filters-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_filters(_n):
        return None, None, None, None, None, None, None, None

    @app.callback(
        Output("est-actual-missing-cols-warning", "children"),
        Output("est-actual-scatter-fig", "figure"),
        Output("est-actual-distribution-fig", "figure"),
        Output("est-actual-adjusted-fig", "figure"),
        Output("est-actual-revision-fig", "figure"),
        Input("equities-data-store", "data"),
        Input("est-actual-metric-selector", "value"),
        Input("est-actual-surprise-method", "value"),
        Input("est-actual-segment-by", "value"),
    )
    def _update_est_actual_tab(data_json, metric, surprise_method, segment_by):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            empty_fig = create_empty_state_figure(
                "Estimated vs Actual", "No data available"
            )
            return html.Div(), empty_fig, empty_fig, empty_fig, empty_fig

        # Get metric columns
        metric_config = EST_ACTUAL_METRICS.get(metric, EST_ACTUAL_METRICS["EPS"])
        actual_col = metric_config["actual"]
        estimate_col = metric_config["estimate"]
        adjusted_col = metric_config.get("adjusted")
        gaap_col = metric_config.get("gaap")

        # Check for missing columns
        required = [actual_col, estimate_col]
        _, missing = validate_required_columns(df, required, f"{metric} Analysis")
        warning = create_missing_columns_warning(missing, f"{metric} Analysis")

        # 1. Scatter: Estimated vs Actual
        scatter_fig = create_empty_state_figure(
            f"{metric}: Estimated vs Actual", "Required columns not available"
        )
        if actual_col in df.columns and estimate_col in df.columns:
            plot_df = df[[actual_col, estimate_col]].dropna()
            if segment_by in df.columns:
                plot_df[segment_by] = df.loc[plot_df.index, segment_by]

            if not plot_df.empty:
                scatter_fig = px.scatter(
                    plot_df,
                    x=estimate_col,
                    y=actual_col,
                    color=segment_by if segment_by in plot_df.columns else None,
                    title=f"{metric}: Estimated vs Actual",
                    template="plotly_dark",
                    hover_data=[segment_by] if segment_by in plot_df.columns else None,
                )
                # Add diagonal line
                min_val = min(plot_df[estimate_col].min(), plot_df[actual_col].min())
                max_val = max(plot_df[estimate_col].max(), plot_df[actual_col].max())
                scatter_fig.add_scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(dash="dash", color="white"),
                    name="Perfect Forecast",
                    showlegend=True,
                )

        # 2. Distribution: Surprise histogram
        dist_fig = create_empty_state_figure(
            f"{metric} Surprise Distribution", "Data not available"
        )
        if actual_col in df.columns and estimate_col in df.columns:
            surprise = compute_surprise(
                df[actual_col], df[estimate_col], mode=surprise_method
            )
            surprise_df = pd.DataFrame({"surprise": surprise})
            if segment_by in df.columns:
                surprise_df[segment_by] = df[segment_by]
            surprise_df = surprise_df.dropna(subset=["surprise"])

            if not surprise_df.empty:
                dist_fig = px.histogram(
                    surprise_df,
                    x="surprise",
                    color=segment_by if segment_by in surprise_df.columns else None,
                    nbins=50,
                    title=f"{metric} Surprise Distribution ({'%' if surprise_method == 'pct' else 'Absolute'})",
                    template="plotly_dark",
                )
                dist_fig.add_vline(x=0, line_dash="dash", line_color="white")

        # 3. Adjusted vs GAAP delta
        adjusted_fig = create_empty_state_figure(
            f"{metric}: Adjusted vs GAAP", "Data not available"
        )
        if (
            adjusted_col
            and gaap_col
            and adjusted_col in df.columns
            and gaap_col in df.columns
        ):
            adj_num = pd.to_numeric(df[adjusted_col], errors="coerce")
            gaap_num = pd.to_numeric(df[gaap_col], errors="coerce")
            delta = adj_num - gaap_num
            delta_df = pd.DataFrame({"delta": delta})
            if segment_by in df.columns:
                delta_df[segment_by] = df[segment_by]
            delta_df = delta_df.dropna(subset=["delta"])

            if not delta_df.empty:
                adjusted_fig = px.box(
                    delta_df,
                    x=segment_by if segment_by in delta_df.columns else None,
                    y="delta",
                    title=f"{metric}: Adjusted vs GAAP Delta",
                    template="plotly_dark",
                )

        # 4. Analyst Revisions
        revision_fig = create_empty_state_figure(
            f"{metric} Revision Trend", "Data not available"
        )
        rev_col = metric_config.get("revisions")
        if rev_col and rev_col in df.columns:
            rev_df = df[[rev_col]].dropna()
            if segment_by in df.columns:
                rev_df[segment_by] = df.loc[rev_df.index, segment_by]

            if not rev_df.empty:
                revision_fig = px.violin(
                    rev_df,
                    y=rev_col,
                    x=segment_by if segment_by in rev_df.columns else None,
                    color=segment_by if segment_by in rev_df.columns else None,
                    box=True,
                    points="all",
                    title=f"{metric} Analyst Revisions (1M)",
                    template="plotly_dark",
                )

        return warning, scatter_fig, dist_fig, adjusted_fig, revision_fig
