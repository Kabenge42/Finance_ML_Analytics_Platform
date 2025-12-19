import unittest
import pandas as pd
import plotly.graph_objects as go
from finance_ml.dashboards.components.charts import (
    _target_vs_price_scatter,
    _market_cap_distribution,
)


class TestDashboardCharts(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "last_price": [100.0, 200.0, 300.0],
                "price_target": [110.0, 210.0, 310.0],
                "market_cap": [1000, 2000, 3000],
                "sector": ["Tech", "Energy", "Banks"],
            }
        )

    def test_target_vs_price_scatter(self):
        fig = _target_vs_price_scatter(self.df)
        self.assertIsInstance(fig, go.Figure)

    def test_market_cap_distribution(self):
        fig = _market_cap_distribution(self.df)
        self.assertIsInstance(fig, go.Figure)


if __name__ == "__main__":
    unittest.main()
