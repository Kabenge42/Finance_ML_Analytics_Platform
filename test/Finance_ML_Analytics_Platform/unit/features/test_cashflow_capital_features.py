"""
Tests for Cash Flow Quality and Capital Allocation features (Phase 9.3 — Phase 4 Week 4, TDD)

Covers:
- Cash flow quality ratios (CFO/NI, FCF/NI, FCF margin, YoY growth, stability)
- Capital allocation metrics (CapEx intensity, CapEx/Depreciation, growth, volatility)
- Shareholder yield & payout, reinvestment rate, acquisition intensity
- Working capital efficiency and trend
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

# Prefer new module path with refactor; fallback to legacy if needed
try:
    from finance_ml.features.advanced import (
        engineer_cash_flow_quality_features,
        engineer_capital_allocation_features,
    )
except Exception:  # pragma: no cover
    from finance_ml.advanced_features import (  # type: ignore
        engineer_cash_flow_quality_features,
        engineer_capital_allocation_features,
    )


class TestCashFlowQuality(unittest.TestCase):
    def test_cash_flow_quality_core_ratios(self):
        df = pd.DataFrame(
            {
                "cfo_ltm": [120.0, 50.0],
                "net_income_ltm": [100.0, 100.0],
                "fcf_ltm": [80.0, 10.0],
                "total_revenues_ltm": [400.0, 200.0],
                "cfo_1fy": [100.0, 50.0],
                # Include extra periods to test stability calc (std across periods)
                "fcf_fy": [70.0, 12.0],
                "fcf_1fy": [60.0, 8.0],
            }
        )
        res = engineer_cash_flow_quality_features(df)
        # CFO/NI: 1.2 and 0.5
        self.assertIn("cfo_to_net_income", res.columns)
        np.testing.assert_allclose(res["cfo_to_net_income"].values, [1.2, 0.5])
        # FCF/NI: 0.8 and 0.1
        self.assertIn("fcf_to_net_income", res.columns)
        np.testing.assert_allclose(res["fcf_to_net_income"].values, [0.8, 0.1])
        # FCF margin: fcf / revenue
        self.assertIn("fcf_margin", res.columns)
        np.testing.assert_allclose(res["fcf_margin"].values, [0.2, 0.05])
        # CFO growth YoY: (120-100)/100 = 0.2; (50-50)/50 = 0
        self.assertIn("cfo_growth_yoy", res.columns)
        np.testing.assert_allclose(res["cfo_growth_yoy"].values, [0.2, 0.0])
        # Stability should be finite (std of available periods)
        self.assertIn("fcf_stability", res.columns)
        self.assertTrue(np.isfinite(res["fcf_stability"].to_numpy(dtype=float)).all())


class TestCapitalAllocation(unittest.TestCase):
    def test_capital_allocation_core_metrics(self):
        df = pd.DataFrame(
            {
                "capital_expenditure_ltm": [50.0, 20.0],
                "capital_expenditure_1fy": [40.0, 25.0],
                "capital_expenditure_fy": [45.0, 22.0],
                "depreciation_amortization_ltm": [25.0, 10.0],
                "total_revenues_ltm": [500.0, 200.0],
                "div_yield_ltm": [2.0, 1.0],  # percent
                "buyback_yield_ltm": [3.0, 0.0],  # percent
                "dividends_paid_ltm": [8.0, 2.0],
                "share_repurchases_ltm": [12.0, 0.0],
                "net_income_ltm": [100.0, 10.0],
                "cfo_ltm": [120.0, 5.0],
                "cash_acquisitions_ltm": [15.0, 0.0],
                "total_assets_ltm": [1000.0, 300.0],
                "working_capital_ltm": [100.0, 50.0],
                "working_capital_1fy": [80.0, 60.0],
            }
        )
        res = engineer_capital_allocation_features(df)
        # CapEx intensity: capex/revenue
        self.assertIn("capex_intensity", res.columns)
        np.testing.assert_allclose(res["capex_intensity"].values, [0.1, 0.1])
        # CapEx/Depreciation
        self.assertIn("capex_to_depreciation", res.columns)
        np.testing.assert_allclose(res["capex_to_depreciation"].values, [2.0, 2.0])
        # CapEx growth rate: (50-40)/40=0.25 ; (20-25)/25=-0.2
        self.assertIn("capex_growth_rate", res.columns)
        np.testing.assert_allclose(res["capex_growth_rate"].values, [0.25, -0.2])
        # Shareholder yield sum
        self.assertIn("total_shareholder_return_yield", res.columns)
        np.testing.assert_allclose(res["total_shareholder_return_yield"].values, [5.0, 1.0])
        # Payout ratio = (dividends + buybacks) / net income
        self.assertIn("payout_ratio", res.columns)
        np.testing.assert_allclose(res["payout_ratio"].values, [0.2, 0.2])
        # Reinvestment rate = (CapEx + M&A) / CFO
        self.assertIn("reinvestment_rate", res.columns)
        np.testing.assert_allclose(res["reinvestment_rate"].values, [65.0 / 120.0, 20.0 / 5.0])
        # Acquisition intensity = acquisitions / assets
        self.assertIn("acquisition_intensity", res.columns)
        np.testing.assert_allclose(res["acquisition_intensity"].values, [0.015, 0.0])
        # Working capital efficiency and trend
        self.assertIn("working_capital_efficiency", res.columns)
        np.testing.assert_allclose(res["working_capital_efficiency"].values, [5.0, 4.0])
        self.assertIn("working_capital_trend", res.columns)
        # (100-80)/500 = 0.04 ; (50-60)/200 = -0.05
        np.testing.assert_allclose(res["working_capital_trend"].values, [0.04, -0.05])


if __name__ == "__main__":
    unittest.main(verbosity=2)
