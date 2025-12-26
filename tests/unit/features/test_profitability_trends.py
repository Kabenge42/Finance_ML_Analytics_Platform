"""
Phase 9.3 (Week 6) — Profitability Trends (TDD)

Covers:
- Margin evolution (EBITDA and Gross margin trends)
- Operating leverage (percent change EBIT / percent change Revenue)
- Earnings quality scoring (simple composite using adjustment ratios)
"""

from __future__ import annotations

import unittest

import pandas as pd

try:  # pragma: no cover - import shim
    from finance_ml.ml_workflow.features.advanced import (
        engineer_profitability_ratios,
        engineer_margin_trends,
    )
except Exception:  # pragma: no cover
    from finance_ml.advanced_features import engineer_profitability_ratios, engineer_margin_trends  # type: ignore


class TestProfitabilityTrends(unittest.TestCase):
    def test_margin_trends_and_operating_leverage(self):
        df = pd.DataFrame(
            {
                # LTM
                "ebitda_ltm": [200.0],
                "total_revenues_ltm": [1000.0],
                "gross_profit_margin_pct_ltm": [40.0],  # 400/1000 * 100
                "ebit_ltm": [150.0],
                # -1FY / FY
                "ebitda_1fy": [150.0],
                "total_revenues_1fy": [900.0],
                "gross_profit_margin_pct_fy": [38.888889],  # 350/900 * 100
                "ebit_1fy": [120.0],
            }
        )
        res = engineer_margin_trends(df)
        
        # EBITDA margin trend = (cur_margin - prev_margin) / prev_margin
        cur_ebitda_margin = 200.0 / 1000.0 * 100
        prev_ebitda_margin = 150.0 / 900.0 * 100
        exp_ebitda_trend = (cur_ebitda_margin - prev_ebitda_margin) / prev_ebitda_margin
        self.assertAlmostEqual(float(res.loc[0, "ebitda_margin_trend"]), exp_ebitda_trend, places=6)
        
        # Gross margin trend = (cur_margin - prev_margin) / prev_margin
        exp_gross_trend = (40.0 - 38.888889) / 38.888889
        self.assertAlmostEqual(float(res.loc[0, "gross_margin_trend"]), exp_gross_trend, places=6)
        
        # Operating leverage = %ΔEBIT / %ΔRevenue
        pct_ebit = (150.0 - 120.0) / 120.0
        pct_rev = (1000.0 - 900.0) / 900.0
        exp_ol = pct_ebit / pct_rev
        self.assertAlmostEqual(float(res.loc[0, "operating_leverage"]), exp_ol, places=6)

    def test_adjustment_ratios_and_earnings_quality(self):
        df = pd.DataFrame(
            {
                "ebitda_ltm": [200.0],
                "ebitda_adj_ltm": [20.0],  # 10%
                "ebit_ltm": [150.0],
                "ebit_adj_ltm": [15.0],  # 10%
            }
        )
        res = engineer_profitability_ratios(df)
        self.assertIn("ebitda_adjustment_ratio_ltm", res.columns)
        self.assertIn("ebit_adjustment_ratio_ltm", res.columns)
        self.assertAlmostEqual(float(res.loc[0, "ebitda_adjustment_ratio_ltm"]), 0.1, places=6)
        self.assertAlmostEqual(float(res.loc[0, "ebit_adjustment_ratio_ltm"]), 0.1, places=6)

        # Earnings quality score via engineer_margin_trends when ratios available
        # engineer_margin_trends will now use the *_ltm versions created by engineer_profitability_ratios
        res2 = engineer_margin_trends(res)
        # Expected simple composite: 100 - 50*a - 30*b (clipped)
        expected_score = max(0.0, min(100.0, 100.0 - 50.0 * 0.1 - 30.0 * 0.1))
        self.assertIn("earnings_quality_score", res2.columns)
        self.assertAlmostEqual(
            float(res2.loc[0, "earnings_quality_score"]), expected_score, places=6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
