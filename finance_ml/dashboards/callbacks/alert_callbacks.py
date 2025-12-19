from __future__ import annotations
import pandas as pd
from dash import Input, Output, State
from finance_ml.dashboards.components import (
    load_alerts_payload,
)
from finance_ml.dashboards.components.data_utils import (
    DEFAULT_ALERTS_PATH,
)
from finance_ml.dashboards.earnings_widgets import (
    EarningsAlertConfig,
    generate_earnings_quality_alerts,
)


def _alerts_to_rows(payload):
    # This matches the local function in equities_dashboard_app.py
    rows = []
    for a in payload.get("alerts", []):
        rows.append(
            {
                "alert_type": a.get("alert_type"),
                "severity": a.get("severity"),
                "tickers": ", ".join(a.get("tickers", [])),
                "description": a.get("description"),
                "count": a.get("count"),
            }
        )
    return rows


def register_alert_callbacks(app, initial_df):
    @app.callback(
        Output("alerts-table", "data"),
        Output("alerts-meta", "children"),
        Output("generate-alerts-status", "children"),
        Input("generate-alerts-btn", "n_clicks"),
        State("equities-data-store", "data"),
        State("cfg-eps-miss", "value"),
        State("cfg-downgrade", "value"),
        State("cfg-min-periods", "value"),
        State("cfg-target-spread", "value"),
        State("cfg-window-days", "value"),
        State("cfg-vol-quantile", "value"),
        State("cfg-max-tickers", "value"),
        prevent_initial_call=True,
    )
    def _generate_alerts(
        _n,
        data_json,
        eps_miss,
        downgrade,
        min_periods,
        target_spread,
        window_days,
        vol_quantile,
        max_tickers,
    ):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            payload = load_alerts_payload(DEFAULT_ALERTS_PATH)
            rows = _alerts_to_rows(payload)
            meta = (
                f"Loaded {len(rows)} alerts from disk"
                if rows
                else "No alerts available"
            )
            return rows, meta, ""

        cfg = EarningsAlertConfig(
            eps_surprise_miss_threshold_pct=float(eps_miss)
            if eps_miss is not None
            else 20.0,
            analyst_downgrade_threshold_pct=float(downgrade)
            if downgrade is not None
            else 5.0,
            analyst_downgrade_min_periods=int(min_periods)
            if min_periods is not None
            else 2,
            target_spread_threshold_pct=float(target_spread)
            if target_spread is not None
            else 30.0,
            pre_earnings_window_days=int(window_days) if window_days is not None else 7,
            pre_earnings_volatility_quantile=(
                float(vol_quantile) if vol_quantile is not None else 0.75
            ),
            max_tickers_per_alert=int(max_tickers) if max_tickers is not None else 10,
        )
        payload = generate_earnings_quality_alerts(
            df,
            config=cfg,
            output_path=DEFAULT_ALERTS_PATH,
        )
        rows = _alerts_to_rows(payload)
        meta = f"Generated {len(rows)} alerts (monitored: {payload.get('total_stocks_monitored', '')})"
        status = (
            f"Wrote {DEFAULT_ALERTS_PATH.name}"
            if DEFAULT_ALERTS_PATH.parent.exists()
            else ""
        )
        return rows, meta, status
