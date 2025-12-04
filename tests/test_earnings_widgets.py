import unittest
import pandas as pd
import numpy as np
from datetime import timedelta
from finance_ml.dashboards.earnings_widgets import (
    create_earnings_calendar_dashboard,
    display_earnings_dashboard,
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
        """Test that results are sorted by market cap and limited by top_n."""
        # Add more data to exceed top_n
        df_large = pd.concat([self.df] * 5, ignore_index=True)
        df_large["market_cap"] = range(len(df_large))  # Unique mcap

        dashboard_df = create_earnings_calendar_dashboard(
            df_large, reference_date=self.base_date, top_n=3
        )
        self.assertEqual(len(dashboard_df), 3)
        # Should be sorted desc
        self.assertTrue(dashboard_df["market_cap"].is_monotonic_decreasing)

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
            styler = display_earnings_dashboard(self.df)
            self.assertIsNotNone(styler)
        except Exception as e:
            self.fail(f"display_earnings_dashboard raised exception: {e}")


if __name__ == "__main__":
    unittest.main()
