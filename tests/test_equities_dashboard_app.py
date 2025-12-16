import json
import unittest
from pathlib import Path

import pandas as pd


class TestEquitiesDashboardAppSmoke(unittest.TestCase):
    def test_import_and_create_app_no_load(self):
        """create_app() should be available and should not run ETL when load_on_start=False."""
        from finance_ml.dashboards import equities_dashboard_app

        app = equities_dashboard_app.create_app(load_on_start=False)
        self.assertIsNotNone(app)
        self.assertTrue(hasattr(app, "layout"))

    def test_apply_filters_graceful_missing_columns(self):
        from finance_ml.dashboards.equities_dashboard_app import apply_filters

        df = pd.DataFrame({"ticker": ["A", "B"], "sector": ["Tech", "Banks"]})
        # Filter on a missing column should not crash and should return unmodified by that filter
        out = apply_filters(df, regions=["US"], sectors=["Tech"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["ticker"], "A")

    def test_load_alerts_payload_missing_file(self):
        from finance_ml.dashboards.equities_dashboard_app import load_alerts_payload

        payload = load_alerts_payload(path=Path("_definitely_missing_file_.json"))
        self.assertEqual(payload, {})

    def test_load_alerts_payload_valid_json(self):
        from finance_ml.dashboards.equities_dashboard_app import load_alerts_payload

        tmp = Path("tests") / "_tmp_alerts_payload.json"
        try:
            tmp.write_text(
                json.dumps(
                    {
                        "timestamp": "2025-01-01T00:00:00",
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
                ),
                encoding="utf-8",
            )
            payload = load_alerts_payload(path=tmp)
            self.assertIn("alerts", payload)
            self.assertEqual(len(payload["alerts"]), 1)
        finally:
            if tmp.exists():
                tmp.unlink()


if __name__ == "__main__":
    unittest.main()
