"""
Phase 9.3 (Week 3) — Quality & Risk Signals

Tests cover:
- Altman Z-Score trends and volatility
- Exceptional items aggregation and scaling ratios
- Accounting quality composite scoring
- Financial sector edge-case handling (graceful NaNs)
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

# Prefer consolidated module path; fall back to legacy for BC
try:  # pragma: no cover - import resolution shim
    from finance_ml.ml_workflow.features.advanced import (
        engineer_accounting_quality_features,
        engineer_financial_distress_features,
        build_comprehensive_features,
    )
except Exception:  # pragma: no cover - legacy import path
    from finance_ml.advanced_features import (  # type: ignore
        engineer_accounting_quality_features,
        engineer_financial_distress_features,
        build_comprehensive_features,
    )


class TestFinancialDistressAltman(unittest.TestCase):
    def test_altman_z_trend_and_volatility(self):
        df = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "sector": ["Tech", "Financials"],
                "altman_z_score_fy": [3.0, 1.5],
                "altman_z_score_1fy": [2.5, 2.0],
                "altman_z_score_ltm": [3.2, 1.2],
                "altman_z_score_fq": [2.8, 1.4],
            }
        )
        res = engineer_financial_distress_features(df)
        # Non-financial: trend = fy - 1fy = 0.5
        self.assertIn("altman_z_trend", res.columns)
        self.assertAlmostEqual(float(res.loc[0, "altman_z_trend"]), 0.5, places=6)
        # Financials: expect NaNs for distress features
        self.assertTrue(np.isnan(res.loc[1, "altman_z_trend"]))
        self.assertIn("z_score_volatility", res.columns)
        self.assertTrue(np.isnan(res.loc[1, "z_score_volatility"]))
        # Distress risk score in [0, 100] for non-financials
        self.assertIn("distress_risk_score", res.columns)
        val = (
            float(res.loc[0, "distress_risk_score"])
            if pd.notna(res.loc[0, "distress_risk_score"])
            else -1
        )
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 100.0)


class TestAccountingQualityAggregation(unittest.TestCase):
    def test_exceptional_items_aggregation_and_ratios(self):
        df = pd.DataFrame(
            {
                "sector": ["Tech", "Energy"],
                # Exceptional items LTM and -1FY
                "impairment_of_goodwill_ltm": [10.0, 0.0],
                "asset_writedown_ltm": [5.0, 2.0],
                "restructuring_charges_ltm": [3.0, 1.0],
                "impairment_of_goodwill_1fy": [4.0, 0.0],
                "asset_writedown_1fy": [2.0, 2.0],
                "restructuring_charges_1fy": [1.0, 0.0],
                # Scalers
                "ebitda_ltm": [20.0, 10.0],
                "total_assets_ltm": [200.0, 100.0],
                # Goodwill
                "goodwill_ltm": [50.0, 10.0],
                "goodwill_1fy": [40.0, 8.0],
                # Intangibles
                "intangible_assets": [60.0, 5.0],
                # Net income (for backward compatible feature)
                "net_income_ltm": [100.0, 50.0],
            }
        )
        res = engineer_accounting_quality_features(df)
        self.assertIn("total_exceptional_items_ltm", res.columns)
        self.assertIn("exceptional_items_to_ebitda", res.columns)
        self.assertIn("restructuring_intensity", res.columns)
        # Aggregation
        self.assertAlmostEqual(
            float(res.loc[0, "total_exceptional_items_ltm"]), 10.0 + 5.0 + 3.0, places=6
        )
        # Ratio to EBITDA (not percent)
        expected_ratio = (10.0 + 5.0 + 3.0) / 20.0
        self.assertAlmostEqual(
            float(res.loc[0, "exceptional_items_to_ebitda"]), expected_ratio, places=6
        )
        # Restructuring intensity: restructuring / total assets
        self.assertAlmostEqual(float(res.loc[0, "restructuring_intensity"]), 3.0 / 200.0, places=6)
        # Goodwill change rate
        self.assertIn("goodwill_change_rate", res.columns)
        self.assertAlmostEqual(
            float(res.loc[0, "goodwill_change_rate"]), (50.0 - 40.0) / 40.0, places=6
        )
        # Aliases present for compatibility
        self.assertIn("goodwill_impairment_flag", res.columns)
        self.assertIn("goodwill_to_assets", res.columns)
        self.assertIn("intangible_intensity", res.columns)

    def test_accounting_quality_composite_score(self):
        # Construct penalties: goodwill_to_assets_pct > 20, impairment present, restructuring present
        df = pd.DataFrame(
            {
                "impairment_of_goodwill_ltm": [1.0],
                "asset_writedown_ltm": [0.0],
                "restructuring_charges_ltm": [1.0],
                "goodwill_ltm": [25.0],
                "total_assets_ltm": [100.0],
            }
        )
        res = engineer_accounting_quality_features(df)
        # Penalties: 30 (impairment) + 15 (restructuring) + 20 (goodwill>20%) = 65 => score 35
        self.assertIn("accounting_quality_score", res.columns)
        self.assertAlmostEqual(float(res.loc[0, "accounting_quality_score"]), 35.0, places=6)
        # Flag alias matches underlying flag
        self.assertEqual(
            int(res.loc[0, "goodwill_impairment_flag"]), int(res.loc[0, "has_goodwill_impairment"])
        )


class TestIntegrationQualityRisk(unittest.TestCase):
    def test_integration_with_build_comprehensive_features(self):
        df = pd.DataFrame(
            {
                "ticker": ["T1", "T2"],
                "sector": ["Tech", "Financials"],
                "last_price": [100.0, 50.0],
                # Altman inputs
                "altman_z_score_fy": [3.0, 2.0],
                "altman_z_score_1fy": [2.0, 2.5],
                "altman_z_score_ltm": [3.5, 1.8],
                # Exceptional items + scalers
                "impairment_of_goodwill_ltm": [0.0, 0.0],
                "asset_writedown_ltm": [0.0, 0.0],
                "restructuring_charges_ltm": [0.0, 0.0],
                "ebitda_ltm": [10.0, 5.0],
                "total_assets_ltm": [100.0, 80.0],
                # Goodwill
                "goodwill_ltm": [5.0, 10.0],
                "goodwill_1fy": [4.0, 8.0],
                # Net income for backward compatible feature
                "net_income_ltm": [20.0, 10.0],
            }
        )
        res = build_comprehensive_features(
            df, include_interactions=False, include_relative_values=False
        )
        # Presence of key new columns
        for col in [
            "altman_z_trend",
            "distress_risk_score",
            "z_score_volatility",
            "total_exceptional_items_ltm",
            "exceptional_items_to_ebitda",
            "accounting_quality_score",
        ]:
            self.assertIn(col, res.columns)
        # Financials sector handled gracefully (NaNs for distress)
        self.assertTrue(np.isnan(res.loc[1, "distress_risk_score"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
