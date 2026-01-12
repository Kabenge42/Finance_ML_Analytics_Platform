import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from finance_ml.dashboards.widgets import (
    create_earnings_calendar_dashboard,
    display_earnings_dashboard,
    create_earnings_surprise_dashboard,
    create_analyst_recommendation_heatmap,
    create_market_movers_dashboard,
    create_price_target_analytics,
    EarningsAlertConfig,
    generate_earnings_quality_alerts,
)


class TestEarningsWidgets(unittest.TestCase):
    def setUp(self):
        self.base_date = pd.Timestamp("2025-12-01")

        # Create mock dataframe
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "sector": ["Tech", "Tech", "Tech", "Retail", "Auto"],
                "market_cap": [3000, 2500, 2000, 1500, 800],
                "next_earnings": [
                    self.base_date + timedelta(days=2),  # +2 days (In range)
                    self.base_date - timedelta(days=5),  # -5 days (In range)
                    self.base_date + timedelta(days=15),  # +15 days (Out of range)
                    self.base_date,  # Today (In range)
                    pd.NaT,  # Missing
                ],
                "dividend_record_ex_date": [
                    self.base_date + timedelta(days=5),
                    self.base_date - timedelta(days=2),
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                ],
                "eps_adj_ltm": [5.0, 4.0, 3.0, 2.0, 1.0],
                "eps_norm_est_avg_ntm": [4.5, 4.2, 3.1, 2.2, 1.1],
                "total_revenues_ltm": [100.0, 120.0, 80.0, 200.0, 50.0],
                "revenues_est_avg_ntm": [95.0, 125.0, 85.0, 190.0, 55.0],
                "ebitda_ltm": [30.0, 35.0, 25.0, 40.0, 10.0],
                "ebitda_est_avg_fy1e": [28.0, 36.0, 26.0, 38.0, 11.0],
                "num_strong_buys_ratings": [10, 8, 6, 5, 4],
                "num_buys_ratings": [15, 12, 10, 9, 8],
                "num_hold_ratings": [5, 7, 8, 6, 5],
                "num_sell_ratings": [1, 1, 2, 1, 2],
                "num_strong_sell_ratings": [0, 0, 0, 0, 1],
                "last_price": [200.0, 350.0, 150.0, 180.0, 250.0],
                "price_target": [220.0, 360.0, 155.0, 210.0, 230.0],
                "price_target_high": [260.0, 420.0, 200.0, 260.0, 300.0],
                "price_target_low": [180.0, 300.0, 120.0, 160.0, 200.0],
                "price_momentum_1m": [0.10, -0.05, 0.02, 0.08, -0.12],
                "volatility_1m": [0.25, 0.18, 0.30, 0.22, 0.40],
                "rel_volume": [1.2, 0.9, 1.5, 1.1, 2.0],
                "one_day_pct": [1.5, -2.0, 0.5, -1.0, 3.0],
                "price_chg_pct_1m": [5.0, -3.0, 2.0, -1.5, 8.0],
                "price_chg_pct_3m": [10.0, -5.0, 4.0, -2.0, 15.0],
                "div_yield_ltm": [0.01, 0.02, 0.0, 0.0, 0.0],
            }
        )

    def test_filtering_logic(self):
        """Test that companies outside the +/- 10 day window are filtered out."""
        dashboard_df = create_earnings_calendar_dashboard(
            self.df, reference_date=self.base_date, top_n=100
        )

        tickers = dashboard_df["ticker"].tolist()
        self.assertIn("AAPL", tickers)
        self.assertIn("MSFT", tickers)
        self.assertIn("AMZN", tickers)
        self.assertNotIn("GOOGL", tickers)  # +15 days
        self.assertNotIn("TSLA", tickers)  # NaT

    def test_sorting_and_top_n(self):
        """Test that results are sorted by next_earnings ascending and limited by top_n."""
        # Add more data to exceed top_n
        df_large = pd.concat([self.df] * 5, ignore_index=True)
        df_large["market_cap"] = range(len(df_large))  # Unique mcap

        dashboard_df = create_earnings_calendar_dashboard(
            df_large, reference_date=self.base_date, top_n=3
        )
        self.assertEqual(len(dashboard_df), 3)
        # Should be sorted by next_earnings ascending (soonest first)
        earnings_dates = pd.to_datetime(dashboard_df["next_earnings"])
        self.assertTrue(earnings_dates.is_monotonic_increasing)

    def test_mode_selection(self):
        """Test column selection based on mode."""
        # Earnings mode
        earnings_df = create_earnings_calendar_dashboard(
            self.df, reference_date=self.base_date, mode="earnings"
        )
        self.assertIn("eps_adj_ltm", earnings_df.columns)
        self.assertNotIn("div_yield_ltm", earnings_df.columns)

        # Dividends mode
        dividends_df = create_earnings_calendar_dashboard(
            self.df, reference_date=self.base_date, mode="dividends"
        )
        self.assertNotIn("eps_adj_ltm", dividends_df.columns)
        self.assertIn("div_yield_ltm", dividends_df.columns)

        # All mode
        all_df = create_earnings_calendar_dashboard(
            self.df, reference_date=self.base_date, mode="all"
        )
        self.assertIn("eps_adj_ltm", all_df.columns)
        self.assertIn("div_yield_ltm", all_df.columns)

    def test_missing_columns_handling(self):
        """Test that missing columns don't crash the function."""
        df_missing = self.df.drop(columns=["eps_adj_ltm", "div_yield_ltm"])
        result = create_earnings_calendar_dashboard(df_missing, reference_date=self.base_date)
        self.assertIn("ticker", result.columns)
        self.assertNotIn("eps_adj_ltm", result.columns)

    def test_days_to_earnings_calculation(self):
        """Test correct calculation of days_to_earnings."""
        result = create_earnings_calendar_dashboard(self.df, reference_date=self.base_date)

        # AAPL: +2 days
        aapl_row = result[result["ticker"] == "AAPL"].iloc[0]
        self.assertEqual(aapl_row["days_to_earnings"], 2)

        # MSFT: -5 days
        msft_row = result[result["ticker"] == "MSFT"].iloc[0]
        self.assertEqual(msft_row["days_to_earnings"], -5)

    def test_display_smoke(self):
        """Smoke test for display function (ensure no crash)."""
        try:
            # We need to ensure some data is in the 10-day window for it to return a Styler
            # AAPL is already +2 days from base_date
            styler = display_earnings_dashboard(self.df, reference_date=self.base_date)
            self.assertIsNotNone(styler)
        except Exception as e:
            self.fail(f"display_earnings_dashboard raised exception: {e}")

    def test_earnings_surprise_dashboard_smoke(self):
        fig = create_earnings_surprise_dashboard(self.df, reference_date=self.base_date, top_n=5)
        self.assertIsInstance(fig, go.Figure)

    def test_analyst_recommendation_heatmap_smoke(self):
        fig = create_analyst_recommendation_heatmap(self.df, top_n_sectors=5)
        self.assertIsInstance(fig, go.Figure)

    def test_market_movers_dashboard_smoke(self):
        fig = create_market_movers_dashboard(
            self.df, reference_date=self.base_date, lookback_days=10, top_n=10
        )
        self.assertIsInstance(fig, go.Figure)

    def test_price_target_analytics_smoke(self):
        fig = create_price_target_analytics(self.df, top_n_sectors=5)
        self.assertIsInstance(fig, go.Figure)

    def test_generate_earnings_quality_alerts_threshold_customization(self):
        df_alert = pd.DataFrame(
            {
                "ticker": ["MISS", "OK"],
                "eps_adj_ltm": [1.0, 5.0],
                "eps_norm_est_avg_ntm": [2.0, 5.0],
            }
        )

        config = EarningsAlertConfig(eps_surprise_miss_threshold_pct=20.0)
        payload = generate_earnings_quality_alerts(
            df_alert,
            config=config,
            reference_date=self.base_date,
        )

        alert_types = [a["alert_type"] for a in payload["alerts"]]
        self.assertIn("large_earnings_miss", alert_types)

        # Higher threshold should suppress the miss alert
        config_high = EarningsAlertConfig(eps_surprise_miss_threshold_pct=60.0)
        payload_high = generate_earnings_quality_alerts(
            df_alert,
            config=config_high,
            reference_date=self.base_date,
        )
        alert_types_high = [a["alert_type"] for a in payload_high["alerts"]]
        self.assertNotIn("large_earnings_miss", alert_types_high)

    def test_generate_earnings_quality_alerts_missing_columns(self):
        df_min = pd.DataFrame({"ticker": ["A", "B"]})
        payload = generate_earnings_quality_alerts(df_min, reference_date=self.base_date)
        self.assertIn("alerts", payload)
        self.assertIsInstance(payload["alerts"], list)

    def test_generate_earnings_quality_alerts_writes_json(self):
        df_alert = pd.DataFrame(
            {
                "ticker": ["MISS"],
                "eps_adj_ltm": [1.0],
                "eps_norm_est_avg_ntm": [2.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "earnings_quality_alerts.json"
            payload = generate_earnings_quality_alerts(
                df_alert,
                reference_date=self.base_date,
                output_path=out_path,
            )
            self.assertTrue(out_path.exists())
            with open(out_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["total_stocks_monitored"], payload["total_stocks_monitored"])
            self.assertIn("alerts", loaded)


if __name__ == "__main__":
    unittest.main()
