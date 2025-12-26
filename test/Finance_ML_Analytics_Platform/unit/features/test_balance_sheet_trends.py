"""
Phase 9.3 (Week 7) — Balance Sheet Trends (TDD)

Covers:
- Debt/Equity/Asset growth rates and composite expansion
- Liquidity trends (current_ratio_trend, cash_ratio)
- Retained earnings patterns (retained_earnings_growth, earnings_retention_rate)
"""

from __future__ import annotations

import unittest

import pandas as pd

try:  # pragma: no cover - import shim
    from finance_ml.features.advanced import engineer_balance_sheet_trends
except Exception:  # pragma: no cover
    from finance_ml.advanced_features import engineer_balance_sheet_trends  # type: ignore


class TestBalanceSheetTrends(unittest.TestCase):
    def test_growth_and_liquidity_and_retained_earnings(self):
        df = pd.DataFrame(
            {
                # Growth bases
                "total_debt_fy": [100.0],
                "total_debt_ltm": [120.0],
                "total_equity_fy": [200.0],
                "total_equity_ltm": [220.0],
                "total_assets_fy": [500.0],
                "total_assets_ltm": [550.0],
                # Liquidity
                "current_ratio_fy": [1.5],
                "current_ratio_ltm": [1.8],
                "cash_and_equivalents": [50.0],
                "current_liabilities": [100.0],
                # Working capital
                "working_capital_ltm": [80.0],
                # Retained earnings
                "retained_earnings_fy": [300.0],
                "retained_earnings_ltm": [330.0],
                "net_income_ltm": [40.0],
            }
        )
        res = engineer_balance_sheet_trends(df)
        # Growth rates
        self.assertAlmostEqual(
            float(res.loc[0, "debt_growth_rate"]), (120.0 - 100.0) / 100.0, places=6
        )
        self.assertAlmostEqual(
            float(res.loc[0, "equity_growth_rate"]), (220.0 - 200.0) / 200.0, places=6
        )
        self.assertAlmostEqual(
            float(res.loc[0, "asset_growth_rate"]), (550.0 - 500.0) / 500.0, places=6
        )
        # Composite expansion ~ mean of three
        exp_mean = ((20.0 / 100.0) + (20.0 / 200.0) + (50.0 / 500.0)) / 3.0
        self.assertAlmostEqual(float(res.loc[0, "balance_sheet_expansion"]), exp_mean, places=6)
        # Liquidity
        self.assertAlmostEqual(float(res.loc[0, "current_ratio_trend"]), 1.8 - 1.5, places=6)
        self.assertAlmostEqual(float(res.loc[0, "cash_ratio"]), 50.0 / 100.0, places=6)
        # Working capital ratio
        self.assertAlmostEqual(float(res.loc[0, "working_capital_ratio"]), 80.0 / 550.0, places=6)
        # Retained earnings patterns
        self.assertIn("retained_earnings_growth", res.columns)
        self.assertIn("earnings_retention_rate", res.columns)
        self.assertAlmostEqual(
            float(res.loc[0, "retained_earnings_growth"]), (330.0 - 300.0) / 220.0, places=6
        )
        self.assertAlmostEqual(
            float(res.loc[0, "earnings_retention_rate"]), (330.0 - 300.0) / 40.0, places=6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
