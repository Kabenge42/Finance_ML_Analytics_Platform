import unittest

import pandas as pd

from finance_ml.dashboards.components.explorer import build_explorer_column_options


class TestDashboardColumnSemantics(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "sector": ["Tech", "Banks"],
                "region": ["US", "EU"],
                "last_price": [100.0, 50.0],
                "price_target": [120.0, 60.0],
                "market_cap": [1_000_000_000, 500_000_000],
                "ev_to_ebitda": [12.0, 9.5],
                "profit_margin": [0.25, 0.18],
                "employees": [1200, 800],
            }
        )

    def test_build_explorer_options_uses_column_semantics(self):
        options, defaults = build_explorer_column_options(
            self.df, categories=["price", "ratio"]
        )

        values = [opt["value"] for opt in options]

        # Price and ratio classifications should surface in options
        self.assertIn("last_price", values)
        self.assertIn("price_target", values)
        self.assertIn("ev_to_ebitda", values)

        # Defaults should be subset of the available options and columns
        self.assertTrue(all(val in values for val in defaults))
        self.assertTrue(all(val in self.df.columns for val in values))


if __name__ == "__main__":
    unittest.main()
