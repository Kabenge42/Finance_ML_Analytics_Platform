import unittest
import pandas as pd
from finance_ml.dashboards.components.earnings import create_earnings_events_chart


class TestDashboardEarnings(unittest.TestCase):
    def test_create_earnings_events_chart_no_data(self):
        fig = create_earnings_events_chart(None)
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.title.text, "Earnings Events Timeline")

    def test_create_earnings_events_chart_valid(self):
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "next_earnings": [
                    (pd.Timestamp.now() + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                    (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                ],
                "sector": ["Tech", "Tech"],
            }
        )
        fig = create_earnings_events_chart(df, days_window=10)
        self.assertIsNotNone(fig)
        self.assertIn("Earnings Events Timeline", fig.layout.title.text)


if __name__ == "__main__":
    unittest.main()
