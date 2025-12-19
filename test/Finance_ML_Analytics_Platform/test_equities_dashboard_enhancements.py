import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd


def _sample_equities_df() -> pd.DataFrame:
    base_date = pd.Timestamp("2025-12-01")
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "name": ["Company A", "Company B", "Company C"],
            "sector": ["Tech", "Tech", "Banks"],
            "region": ["US", "US", "EU"],
            "market_cap": [3000, 2500, 800],
            "next_earnings": [
                base_date + timedelta(days=2),
                base_date - timedelta(days=1),
                base_date + timedelta(days=5),
            ],
            # Surprise dashboards
            "eps_adj_ltm": [5.0, 4.0, 1.0],
            "eps_norm_est_avg_ntm": [4.5, 4.2, 1.1],
            "total_revenues_ltm": [100.0, 120.0, 50.0],
            "revenues_est_avg_ntm": [95.0, 125.0, 55.0],
            "ebitda_ltm": [30.0, 35.0, 10.0],
            "ebitda_est_avg_fy1e": [28.0, 36.0, 11.0],
            # Analyst ratings
            "num_strong_buys_ratings": [10, 8, 4],
            "num_buys_ratings": [15, 12, 8],
            "num_hold_ratings": [5, 7, 5],
            "num_sell_ratings": [1, 1, 2],
            "num_strong_sell_ratings": [0, 0, 1],
            # Price target analytics
            "last_price": [200.0, 350.0, 250.0],
            "price_target": [220.0, 360.0, 230.0],
            "price_target_high": [260.0, 420.0, 300.0],
            "price_target_low": [180.0, 300.0, 200.0],
            # Market movers
            "price_momentum_1m": [0.10, -0.05, -0.12],
            "volatility_1m": [0.25, 0.18, 0.40],
            "rel_volume": [1.2, 0.9, 2.0],
        }
    )


class TestEquitiesDashboardEnhancements(unittest.TestCase):
    def test_export_equities_data_creates_output_and_metadata_dirs(self):
        from finance_ml.dashboards.equities_dashboard_app import export_equities_data

        df = _sample_equities_df()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_path = root / "exports" / "equities_dash_df.csv"
            metadata_path = root / "metadata" / "metadata.json"

            meta = export_equities_data(df, output_path=output_path, metadata_path=metadata_path)

            self.assertTrue(output_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(meta["row_count"], len(df))
            self.assertEqual(meta["column_count"], len(df.columns))
            self.assertIn("timestamp", meta)

            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["row_count"], len(df))

    def test_generate_dashboard_artifacts_creates_metadata_dir(self):
        from finance_ml.dashboards.equities_dashboard_app import generate_dashboard_artifacts

        df = _sample_equities_df()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_dir = root / "artifacts"
            metadata_path = root / "registry" / "artifacts_metadata.json"

            meta = generate_dashboard_artifacts(
                df, output_dir=output_dir, metadata_path=metadata_path
            )

            self.assertTrue(metadata_path.exists())
            self.assertEqual(meta.get("generation_status"), "completed")
            self.assertIn("artifacts", meta)
            self.assertGreaterEqual(len(meta["artifacts"]), 5)

            for artifact in meta["artifacts"].values():
                artifact_path = output_dir / artifact["file"]
                self.assertTrue(
                    artifact_path.exists(),
                    msg=f"Missing artifact file: {artifact_path}",
                )

    def test_create_earnings_events_chart_non_empty_path(self):
        from finance_ml.dashboards.equities_dashboard_app import create_earnings_events_chart

        df = _sample_equities_df().copy()
        # Force at least one event within the default window.
        df.loc[0, "next_earnings"] = pd.Timestamp.now() + timedelta(days=1)
        fig = create_earnings_events_chart(df, days_window=30)
        self.assertGreater(len(fig.data), 0)


if __name__ == "__main__":
    unittest.main()
