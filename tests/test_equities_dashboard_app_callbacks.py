import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
from dash import html


def _sample_df() -> pd.DataFrame:
    base_date = pd.Timestamp("2025-12-01")
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "name": [
                "Company A",
                "Company B",
                "Company C",
                "Company D",
                "Company E",
                "Company F",
            ],
            "sector": ["Tech", "Tech", "Banks", "Banks", "Energy", "Energy"],
            "region": ["US", "US", "EU", "EU", "US", "EU"],
            "market_cap": [3000, 2500, 800, 600, 500, 400],
            "last_price": [200.0, 350.0, 250.0, 50.0, 10.0, 20.0],
            "price_target": [220.0, 360.0, 230.0, 60.0, 15.0, 25.0],
            "price_target_high": [260.0, 420.0, 300.0, 90.0, 25.0, 40.0],
            "price_target_low": [180.0, 300.0, 200.0, 40.0, 8.0, 15.0],
            "next_earnings": [
                base_date + timedelta(days=2),
                base_date - timedelta(days=1),
                base_date + timedelta(days=5),
                pd.NaT,
                base_date + timedelta(days=40),
                base_date - timedelta(days=40),
            ],
            # Alert-related / earnings widgets
            "eps_adj_ltm": [5.0, 4.0, 1.0, 2.0, 0.5, 0.1],
            "eps_norm_est_avg_ntm": [4.5, 4.2, 1.1, 1.8, 0.6, 0.1],
            "total_revenues_ltm": [100.0, 120.0, 50.0, 80.0, 30.0, 25.0],
            "revenues_est_avg_ntm": [95.0, 125.0, 55.0, 75.0, 35.0, 20.0],
            "ebitda_ltm": [30.0, 35.0, 10.0, 15.0, 5.0, 4.0],
            "ebitda_est_avg_fy1e": [28.0, 36.0, 11.0, 14.0, 6.0, 3.0],
            "num_strong_buys_ratings": [10, 8, 4, 1, 2, 0],
            "num_buys_ratings": [15, 12, 8, 2, 1, 0],
            "num_hold_ratings": [5, 7, 5, 5, 2, 1],
            "num_sell_ratings": [1, 1, 2, 2, 1, 0],
            "num_strong_sell_ratings": [0, 0, 1, 0, 0, 0],
            "price_momentum_1m": [0.10, -0.05, -0.12, 0.08, 0.02, -0.01],
            "volatility_1m": [0.25, 0.18, 0.40, 0.22, 0.30, 0.15],
            "rel_volume": [1.2, 0.9, 2.0, 1.1, 1.5, 0.8],
        }
    )


def _get_unwrapped_callback(app, output_key_substr: str):
    for key, entry in app.callback_map.items():
        if output_key_substr in key:
            cb = entry["callback"]
            while hasattr(cb, "__wrapped__"):
                cb = cb.__wrapped__
            return cb
    raise KeyError(f"Callback not found for output key containing: {output_key_substr}")


class TestEquitiesDashboardAppCallbacks(unittest.TestCase):
    def test_helpers_and_artifact_rendering(self):
        from finance_ml.dashboards.equities_dashboard_app import (
            _coerce_list,
            _list_artifacts,
            _render_artifact,
            PROJECT_ROOT,
        )

        self.assertEqual(_coerce_list(None), [])
        self.assertEqual(_coerce_list("A"), ["A"])
        self.assertEqual(_coerce_list(["A", 1]), ["A", "1"])

        # Create small artifacts inside outputs/ so iframe embedding is allowed
        outputs = PROJECT_ROOT / "outputs"
        html_path = outputs / "eda" / "earnings_analytics" / "_tmp_test_artifact.html"
        json_path = (
            outputs / "dashboards" / "equities_dashboard" / "artifacts" / "_tmp_test_artifact.json"
        )

        try:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.parent.mkdir(parents=True, exist_ok=True)

            html_path.write_text("<html><body>ok</body></html>", encoding="utf-8")
            json_path.write_text(json.dumps({"a": 1}), encoding="utf-8")

            items = _list_artifacts()
            values = {i["value"] for i in items}
            self.assertIn(str(html_path), values)
            self.assertIn(str(json_path), values)

            rendered_html = _render_artifact(str(html_path))
            self.assertIsInstance(rendered_html, html.Iframe)
            self.assertIn("/app_assets/", rendered_html.src)

            rendered_json = _render_artifact(str(json_path))
            self.assertIsInstance(rendered_json, html.Pre)
        finally:
            if html_path.exists():
                html_path.unlink()
            if json_path.exists():
                json_path.unlink()

    def test_load_data_branching_without_external_services(self):
        from finance_ml.dashboards import equities_dashboard_app as mod

        df = _sample_df()
        calls = []

        def _fake_etl_with_features(*, source, **_kwargs):
            calls.append(source)
            return df

        with patch.object(mod, "etl_with_features", side_effect=_fake_etl_with_features):
            out = mod.load_data(data_source="csv", limit=2)
            self.assertEqual(len(out), 2)
            self.assertEqual(calls[-1], "csv")

            # auto falls back to csv when no DB_URL is provided
            out2 = mod.load_data(data_source="auto", db_url=None, limit=1)
            self.assertEqual(len(out2), 1)
            self.assertEqual(calls[-1], "csv")

            # db without a URL should return empty
            out3 = mod.load_data(data_source="db", db_url=None)
            self.assertTrue(out3.empty)

    def test_callbacks_can_be_invoked_directly_without_etl(self):
        from finance_ml.dashboards import equities_dashboard_app as mod

        df = _sample_df()
        data_json = df.to_json(orient="split")
        app = mod.create_app(load_on_start=False)

        # Overview callback
        cb_overview = _get_unwrapped_callback(app, "kpi-cards.children")
        cards, scatter, mcap = cb_overview(
            data_json,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.assertIsNotNone(cards)
        self.assertIsInstance(scatter, go.Figure)
        self.assertIsInstance(mcap, go.Figure)

        # Earnings figs callback
        cb_earnings = _get_unwrapped_callback(app, "earnings-events-timeline.figure")
        figs = cb_earnings(data_json)
        self.assertEqual(len(figs), 5)
        for fig in figs:
            self.assertIsInstance(fig, go.Figure)

        # Refresh callback: patch load_data and export to avoid ETL and filesystem writes
        cb_refresh = _get_unwrapped_callback(app, "equities-data-store.data")
        with (
            patch.object(mod, "load_data", return_value=df),
            patch.object(mod, "export_equities_data", return_value={"row_count": len(df)}),
        ):
            stored_json, status = cb_refresh(1)
            self.assertIn("Loaded", status)
            self.assertIsInstance(stored_json, str)

    def test_additional_callbacks_cover_alerts_explorer_and_artifacts(self):
        from finance_ml.dashboards import equities_dashboard_app as mod

        df = _sample_df()
        data_json = df.to_json(orient="split")
        app = mod.create_app(load_on_start=False)

        # Alerts callback (generation path)
        cb_alerts = _get_unwrapped_callback(app, "alerts-table.data")

        fake_payload = {
            "timestamp": "2025-12-16T00:00:00",
            "total_stocks_monitored": len(df),
            "alerts": [
                {
                    "alert_type": "large_earnings_miss",
                    "severity": "high",
                    "count": 2,
                    "description": "example",
                    "tickers": ["AAA", "BBB"],
                }
            ],
        }
        with patch.object(mod, "generate_earnings_quality_alerts", return_value=fake_payload):
            rows, meta, status = cb_alerts(
                1,
                data_json,
                20.0,
                5.0,
                2,
                30.0,
                7,
                0.75,
                10,
            )
            self.assertIsInstance(rows, list)
            self.assertTrue(meta)
            self.assertIsInstance(status, str)

        # Explorer columns callback
        cb_cols = _get_unwrapped_callback(app, "explorer-columns-dropdown.options")
        options, default = cb_cols(["profitability"], data_json)
        self.assertIsInstance(options, list)
        self.assertIsInstance(default, list)

        # Explorer table callback
        cb_table = _get_unwrapped_callback(app, "explorer-table.columns")
        cols, rows = cb_table(1, data_json, ["ticker", "sector"], 3)
        self.assertEqual([c["id"] for c in cols], ["ticker", "sector"])
        # The UI enforces a minimum table preview size for usability.
        self.assertLessEqual(len(rows), max(10, 3))

        # Artifacts dropdown + viewer callbacks
        cb_artifacts_opts = _get_unwrapped_callback(app, "artifact-dropdown.options")
        opts = cb_artifacts_opts("artifacts")
        self.assertIsInstance(opts, list)

        cb_show = _get_unwrapped_callback(app, "artifact-viewer.children")
        # Create a JSON artifact inside outputs/ and ensure it renders.
        tmp_json = (
            mod.PROJECT_ROOT
            / "outputs"
            / "dashboards"
            / "equities_dashboard"
            / "artifacts"
            / "_tmp_cb.json"
        )
        try:
            tmp_json.parent.mkdir(parents=True, exist_ok=True)
            tmp_json.write_text(json.dumps({"ok": True}), encoding="utf-8")
            rendered = cb_show(str(tmp_json))
            self.assertIsInstance(rendered, html.Pre)
        finally:
            if tmp_json.exists():
                tmp_json.unlink()

        # Reset filters
        cb_reset = _get_unwrapped_callback(app, "sector-dropdown.value")
        reset_values = cb_reset(1)
        self.assertEqual(reset_values, (None, None, None, None, None, None, None, None))

        # Generate artifacts callback (mock generation)
        cb_gen = None
        for key, entry in app.callback_map.items():
            if "data-status.children" not in key:
                continue
            cb = entry["callback"]
            while hasattr(cb, "__wrapped__"):
                cb = cb.__wrapped__
            # _generate_artifacts(_n, data_json) takes 2 args, while _refresh_data(_n_clicks) takes 1.
            if cb.__code__.co_argcount == 2:
                cb_gen = cb
                break

        self.assertIsNotNone(cb_gen)
        with patch.object(
            mod, "generate_dashboard_artifacts", return_value={"artifacts": {"a": {}, "b": {}}}
        ):
            msg = cb_gen(1, data_json)
            self.assertIn("Generated", msg)


if __name__ == "__main__":
    unittest.main()
