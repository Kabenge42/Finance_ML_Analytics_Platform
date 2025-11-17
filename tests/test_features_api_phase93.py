from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from tests.utils.feature_test_helpers import (
    assert_no_inf,
    assert_within_range,
    time_block,
)


class TestFeaturesApiPhase93(unittest.TestCase):
    def test_api_momentum_preset_basic(self):
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Energy", "Financials"],
                "last_price": [110.0, 50.0, 25.0],
                "price_1m_ago": [100.0, 55.0, 20.0],
                "price_3m_ago": [90.0, 60.0, 30.0],
            }
        )

        # Import here to avoid hard dependency if refactor moves paths
        from finance_ml.ml_workflow.features.api import build_features

        with time_block(0.5):
            result = build_features(df, preset="momentum")

        self.assertEqual(len(result), len(df))
        # Presence of momentum columns
        self.assertIn("price_momentum_1m", result.columns)
        self.assertIn("price_momentum_3m", result.columns)
        # Numerical hygiene
        assert_no_inf(result)

    def test_api_quality_preset_core_signals(self):
        df = pd.DataFrame(
            {
                "ticker": ["Q1", "Q2"],
                "sector": ["Tech", "Energy"],
                # Exceptional items and scaling bases
                "impairment_of_goodwill_ltm": [0.0, 10.0],
                "asset_writedown_ltm": [0.0, 5.0],
                "restructuring_charges_ltm": [0.0, 2.0],
                "ebitda_ltm": [100.0, 100.0],
                "net_income_ltm": [50.0, 40.0],
                "total_assets_ltm": [1000.0, 500.0],
                # Altman Z (for distress composite)
                "altman_z_score_fy": [2.5, 3.2],
                "altman_z_score_1fy": [2.0, 3.1],
            }
        )

        from finance_ml.ml_workflow.features.api import build_features

        with time_block(0.5):
            result = build_features(df, preset="quality")

        self.assertEqual(len(result), len(df))
        # Exceptional items aggregation present
        self.assertIn("total_exceptional_items_ltm", result.columns)
        self.assertIn("exceptional_items_to_ebitda", result.columns)
        # Accounting quality score in 0..100
        if "accounting_quality_score" in result.columns:
            assert_within_range(
                result, column="accounting_quality_score", min_value=0.0, max_value=100.0
            )
        # Distress features
        self.assertIn("altman_z_trend", result.columns)
        self.assertIn("distress_risk_score", result.columns)
        assert_no_inf(result)

    def test_api_cashflow_preset_uses_cashflow_groups(self):
        df = pd.DataFrame(
            {
                "cfo_ltm": [120.0],
                "net_income_ltm": [100.0],
                "fcf_ltm": [80.0],
                "total_revenues_ltm": [400.0],
                "cfo_1fy": [100.0],
                "capital_expenditure_ltm": [50.0],
                "capital_expenditure_1fy": [40.0],
                "depreciation_amortization_ltm": [25.0],
                "cash_acquisitions_ltm": [15.0],
                "total_assets_ltm": [1000.0],
                "working_capital_ltm": [100.0],
                "working_capital_1fy": [80.0],
                "div_yield_ltm": [2.0],
                "buyback_yield_ltm": [3.0],
                "dividends_paid_ltm": [8.0],
                "share_repurchases_ltm": [12.0],
            }
        )

        from finance_ml.ml_workflow.features.api import build_features

        with time_block(0.5):
            result = build_features(df, preset="cashflow")

        # Core cash flow and capital allocation outputs should be present
        for col in [
            "cfo_to_net_income",
            "fcf_to_net_income",
            "fcf_margin",
            "capex_intensity",
            "capex_to_depreciation",
            "total_shareholder_return_yield",
            "payout_ratio",
        ]:
            self.assertIn(col, result.columns)

        assert_no_inf(result)

    def test_api_full_enhanced_delegates_to_advanced(self):
        # Minimal sample sufficient for valuation ratios
        df = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "sector": ["Tech", "Energy"],
                "last_price": [100.0, 50.0],
                "eps": [5.0, 2.5],
                "book_value_per_share": [20.0, 10.0],
                "revenue": [1000.0, 500.0],
                "shares_outstanding": [100.0, 200.0],
                "enterprise_value": [1000.0, 500.0],
                "ebitda": [100.0, 50.0],
            }
        )
        from finance_ml.ml_workflow.features.api import build_features
        from finance_ml.ml_workflow.features.advanced import build_comprehensive_features

        res_api = build_features(
            df, preset="full_enhanced", include_interactions=False, include_relative=False
        )
        res_adv = build_comprehensive_features(
            df, include_interactions=False, include_relative_values=False
        )

        # Both paths should yield the same core valuation ratio
        self.assertIn("p_e_ratio", res_api.columns)
        self.assertIn("p_e_ratio", res_adv.columns)
        pd.testing.assert_series_equal(
            res_api["p_e_ratio"], res_adv["p_e_ratio"], check_names=False
        )

    def test_api_comprehensive_v2_alias_matches_comprehensive(self):
        """ "comprehensive_v2" preset should behave as an alias of "comprehensive".

        This ensures the public API stays aligned with the Phase 9.3
        implementation plan, which references a "comprehensive_v2" preset
        while keeping backwards compatibility with existing callers that use
        "comprehensive".
        """

        df = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "sector": ["Tech", "Energy"],
                "last_price": [100.0, 50.0],
                "eps": [5.0, 2.5],
                "book_value_per_share": [20.0, 10.0],
            }
        )

        from finance_ml.ml_workflow.features.api import build_features

        res_comprehensive = build_features(
            df,
            preset="comprehensive",
            include_interactions=False,
            include_relative=False,
        )
        res_comprehensive_v2 = build_features(
            df,
            preset="comprehensive_v2",
            include_interactions=False,
            include_relative=False,
        )

        # The two presets should yield identical columns and values
        self.assertEqual(set(res_comprehensive.columns), set(res_comprehensive_v2.columns))
        pd.testing.assert_frame_equal(
            res_comprehensive.sort_index(axis=1),
            res_comprehensive_v2.sort_index(axis=1),
            check_names=False,
        )

    def test_advanced_build_comprehensive_features_supports_presets(self):
        df = pd.DataFrame(
            {
                "ticker": ["A"],
                "sector": ["Tech"],
                "last_price": [100.0],
                "price_1m_ago": [90.0],
            }
        )
        from finance_ml.ml_workflow.features.advanced import build_comprehensive_features

        out = build_comprehensive_features(
            df, preset="momentum", include_interactions=False, include_relative_values=False
        )
        self.assertIn("price_momentum_1m", out.columns)
        # Sanity: should not be excessively slow
        assert_no_inf(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
