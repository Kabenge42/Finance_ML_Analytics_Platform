import unittest
import pandas as pd
from finance_ml.dashboards.components.filters import _safe_options, apply_filters


class TestDashboardFilters(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Tech", "Energy"],
                "region": ["US", "EU", "US"],
                "market_cap": [1000, 2000, 3000],
            }
        )

    def test_safe_options(self):
        options = _safe_options(self.df, "sector")
        self.assertEqual(len(options), 2)
        values = {opt["value"] for opt in options}
        self.assertEqual(values, {"Tech", "Energy"})

    def test_safe_options_missing_col(self):
        options = _safe_options(self.df, "missing")
        self.assertEqual(options, [])

    def test_apply_filters_none(self):
        out = apply_filters(self.df)
        self.assertEqual(len(out), 3)

    def test_apply_filters_sector(self):
        out = apply_filters(self.df, sectors=["Tech"])
        self.assertEqual(len(out), 2)
        self.assertTrue(all(out["sector"] == "Tech"))

    def test_apply_filters_multiple(self):
        out = apply_filters(self.df, sectors=["Tech"], regions=["EU"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["ticker"], "B")

    def test_apply_filters_missing_col_graceful(self):
        # Filter on a missing column should not crash and should return unmodified by that filter
        out = apply_filters(self.df, industries=["Some Industry"])
        self.assertEqual(len(out), 3)


if __name__ == "__main__":
    unittest.main()
