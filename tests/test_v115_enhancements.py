"""
TDD Tests for Phase 9.3 Feature Enhancements (v1.15)
Covers all 10 modules: leverage, earnings, profitability, employment, quality,
momentum, valuation, dividends, revenue, growth.
"""

import unittest

import pandas as pd

from finance_ml.features.advanced.dividends import (
    engineer_dividend_reliability_features,
)
from finance_ml.features.advanced.earnings import engineer_eps_trajectory_features
from finance_ml.features.advanced.employment import (
    engineer_employee_productivity_features,
)
from finance_ml.features.advanced.growth import engineer_growth_metrics
from finance_ml.features.advanced.leverage import engineer_balance_sheet_trends
from finance_ml.features.advanced.momentum import (
    engineer_momentum_features,
    engineer_technical_analysis_features,
)
from finance_ml.features.advanced.profitability import engineer_profitability_ratios
from finance_ml.features.advanced.quality import engineer_accounting_quality_features
from finance_ml.features.advanced.revenue import engineer_revenue_forecast_features
from finance_ml.features.advanced.valuation import (
    engineer_valuation_timeseries_features,
)


class TestPhase93EnhancementsV115(unittest.TestCase):
    def test_leverage_enhancements(self):
        df = pd.DataFrame(
            {
                "working_capital_ltm": [120, 80],
                "working_capital_5yavgfy": [100, 100],
                "cash_and_equivalents_ltm": [50, 60],
                "cash_and_equivalents_5yavgfq": [40, 60],
                "inventory_ltm": [30, 45],
                "inventory_5yavgfq": [25, 45],
                "goodwill_ltm": [10, 20],
                "goodwill_5yavgfq": [10, 25],
            }
        )
        result = engineer_balance_sheet_trends(df)
        self.assertAlmostEqual(result.loc[0, "working_capital_vs_5y_avg"], 1.2)
        self.assertAlmostEqual(result.loc[0, "cash_stability_ratio"], 1.25)
        self.assertAlmostEqual(result.loc[0, "inventory_vs_5y_avg"], 1.2)
        self.assertAlmostEqual(result.loc[0, "goodwill_stability"], 1.0)

    def test_earnings_enhancements(self):
        df = pd.DataFrame(
            {
                "net_eps_basic_fq": [1.0, 0.8],
                "net_eps_basic_1fqfq": [0.9, 0.7],
                "net_eps_basic_2fqfq": [0.8, 0.6],
                "net_eps_basic_3fqfq": [0.7, 0.5],
                "net_eps_basic_4fqfq": [0.6, 0.4],
                "normalized_net_income_ltm": [110, 90],
                "net_income_is_ltm": [100, 100],
                "eps_gaap_est_avg_fy1e": [1.2, 0.9],
                "eps_norm_est_avg_fy1e": [1.3, 1.0],
                "normalized_net_income_5yavgltm": [100, 80],
            }
        )
        result = engineer_eps_trajectory_features(df)
        self.assertAlmostEqual(result.loc[0, "normalized_vs_gaap_spread"], 10.0)
        self.assertAlmostEqual(result.loc[0, "normalized_vs_gaap_ratio"], 1.1)
        self.assertAlmostEqual(result.loc[0, "forward_eps_gaap_adjusted_spread"], 0.1)
        self.assertAlmostEqual(result.loc[0, "earnings_stability_score"], 1.1)

    def test_profitability_enhancements(self):
        df = pd.DataFrame(
            {
                "ebitda_ltm": [120, 80],
                "ebitda_5yavgltm": [100, 100],
                "ebit_ltm": [110, 70],
                "ebit_5yavgltm": [100, 100],
                "operating_income_ltm": [110, 130],
                "operating_income_fy": [100, 100],
                "total_revenues_ltm": [1100, 1200],
                "total_revenues_fy": [1000, 1000],
                "gross_profit_margin_pct_fy": [40, 35],
                "gross_profit_margin_pct_ltm": [42, 33],
            }
        )
        result = engineer_profitability_ratios(df)
        self.assertAlmostEqual(result.loc[0, "ebitda_vs_5y_avg"], 1.2)
        self.assertAlmostEqual(result.loc[0, "ebitda_stability_score"], 0.8)
        self.assertAlmostEqual(result.loc[0, "ebit_vs_5y_avg"], 1.1)
        # oi_growth = (110-100)/100 = 0.1; rev_growth = (1100-1000)/1000 = 0.1; ratio = 1.0
        self.assertAlmostEqual(result.loc[0, "operating_leverage_ratio"], 1.0)
        self.assertAlmostEqual(result.loc[0, "gross_margin_consistency"], 2.0)

    def test_employment_enhancements(self):
        df = pd.DataFrame(
            {
                "full_time_employees_fy": [110, 90],
                "full_time_employees_1fy": [100, 100],
                "full_time_employees_2fy": [90, 110],
                "full_time_employees_3fy": [80, 120],
                "avg_employees_5yavgfy": [100, 100],
            }
        )
        result = engineer_employee_productivity_features(df)
        self.assertAlmostEqual(result.loc[0, "fte_growth_1y_pct"], 10.0)
        self.assertAlmostEqual(result.loc[0, "fte_growth_2y_pct"], 22.222222, places=5)
        self.assertAlmostEqual(result.loc[0, "fte_cagr_3y_pct"], 11.199, places=3)
        self.assertAlmostEqual(result.loc[0, "fte_vs_5y_avg"], 1.1)
        self.assertAlmostEqual(result.loc[0, "workforce_stability_score"], 0.9)

    def test_quality_enhancements(self):
        df = pd.DataFrame(
            {
                "impairment_of_goodwill_5yavgfq": [10, 0],
                "impairment_of_goodwill_fq": [5, 0],
                "asset_writedown_5yavgfq": [20, 0],
                "asset_writedown_fq": [30, 0],
                "restructuring_charges_5yavgfq": [15, 0],
                "restructuring_charges_fq": [15, 0],
                "merger_and_restructuring_charges_5yavgfq": [50, 0],
                "merger_and_restructuring_charges_fq": [25, 0],
                "other_unusual_items_total_ltm": [10, 5],
                "ebitda_ltm": [100, 100],
            }
        )
        result = engineer_accounting_quality_features(df)
        self.assertAlmostEqual(result.loc[0, "impairment_of_goodwill_vs_5y_avg"], 0.5)
        self.assertAlmostEqual(result.loc[0, "asset_writedown_vs_5y_avg"], 1.5)
        self.assertAlmostEqual(result.loc[0, "other_unusual_to_ebitda"], 0.1)
        self.assertEqual(result.loc[0, "exceptional_items_frequency"], 3)

    def test_momentum_enhancements(self):
        df = pd.DataFrame(
            {
                "last_price": [105, 95],
                "price_5d_ago": [100, 100],
                "ema_100d": [100, 100],
                "volatility_1m": [20, 30],
                "volatility_1y": [25, 25],
                "volatility_3m": [22, 28],
                "volatility_6m": [24, 26],
                "rel_volume": [2.0, 0.4],
                "tot_return_pct_cagr_3y": [15, 10],
                "tot_return_pct_cagr_10y": [12, 12],
            }
        )
        result = engineer_momentum_features(df)
        result = engineer_technical_analysis_features(result)
        self.assertAlmostEqual(result.loc[0, "price_momentum_5d"], 5.0)
        self.assertAlmostEqual(result.loc[0, "price_vs_ema_100d"], 5.0)
        self.assertAlmostEqual(result.loc[0, "volatility_regime"], 0.8)
        self.assertAlmostEqual(result.loc[0, "volatility_compression"], 5.0)
        self.assertAlmostEqual(result.loc[0, "volatility_term_structure"], -2.0)
        self.assertEqual(result.loc[0, "high_volume_flag"], 1)
        self.assertEqual(result.loc[0, "low_volume_flag"], 0)
        self.assertAlmostEqual(result.loc[0, "return_acceleration"], 3.0)

    def test_valuation_enhancements(self):
        df = pd.DataFrame(
            {
                "ev_sales_ltm": [2.2, 1.8],
                "ev_sales_1fqltm": [2.1, 1.9],
                "ev_sales_2fqltm": [2.0, 2.0],
                "p_e_0fqqoqltm": [0.05, -0.02],
                "p_e_0fyyoyltm": [0.15, -0.05],
                "p_b_ltm": [3.5, 1.5],
                "p_b_5yavg": [3.0, 2.0],
            }
        )
        result = engineer_valuation_timeseries_features(df)
        self.assertIn("ev_sales_quarterly_volatility", result.columns)
        self.assertEqual(result.loc[0, "ev_sales_trend_consistency"], 1)
        self.assertAlmostEqual(result.loc[0, "p_e_qoq_momentum"], 0.05)
        self.assertAlmostEqual(result.loc[0, "p_b_vs_5y_avg"], 3.5 / 3.0)
        self.assertEqual(
            result.loc[0, "p_b_mean_reversion_signal"], 0
        )  # 3.5 < 3.0 * 1.2 = 3.6

    def test_dividend_enhancements(self):
        df = pd.DataFrame(
            {
                "div_yield_ind": [3.5, 2.0],
                "div_yield_1fyind": [3.2, 2.1],
                "div_yield_2fyind": [3.0, 2.2],
                "div_yield_3fyind": [2.8, 2.3],
                "div_yield_4fyind": [2.5, 2.4],
                "div_yield_5fyind": [2.2, 2.5],
                "div_yield_ltm": [3.4, 1.9],
                "div_yield_5yavgltm": [3.0, 2.2],
                "common_dividends_paid_ltm": [110, 95],
                "common_dividends_paid_fy": [100, 100],
            }
        )
        result = engineer_dividend_reliability_features(df)
        self.assertIn("dividend_yield_volatility", result.columns)
        self.assertAlmostEqual(result.loc[0, "dividend_yield_trend"], 0.25)
        self.assertAlmostEqual(result.loc[0, "dividend_yield_vs_5y_avg"], 3.4 / 3.0)
        self.assertAlmostEqual(result.loc[0, "dividend_payout_growth"], 0.1)
        self.assertEqual(result.loc[0, "dividend_consistency_years"], 6)
        self.assertAlmostEqual(result.loc[0, "dividend_yield_cagr_5y"], 9.731, places=3)

    def test_revenue_enhancements(self):
        df = pd.DataFrame(
            {
                "revenues_est_avg_fy1e": [1100, 900],
                "revenues_est_med_fy1e": [1050, 950],
                "ebitda_est_avg_fy1e": [220, 180],
                "ebitda_ltm": [200, 200],
                "total_revenues_ltm": [1000, 1000],
                "ebit_est_med_fy1e": [150, 120],
                "eps_norm_est_num_fy1e": [12, 5],
                "revenues_est_avg_ntm": [1150, 850],
            }
        )
        result = engineer_revenue_forecast_features(df)
        self.assertAlmostEqual(result.loc[0, "revenue_estimate_skew"], 50 / 1050)
        self.assertAlmostEqual(
            result.loc[0, "ebitda_margin_improvement_expected"], 220 / 1100 - 200 / 1000
        )
        self.assertAlmostEqual(result.loc[0, "forward_ebit_margin"], 150 / 1100 * 100)
        self.assertEqual(result.loc[0, "analyst_estimate_coverage"], 12)
        self.assertEqual(result.loc[0, "high_coverage_flag"], 1)
        self.assertAlmostEqual(result.loc[0, "revenue_estimate_alignment"], 1150 / 1100)

    def test_growth_enhancements(self):
        df = pd.DataFrame(
            {
                "total_revenues_cagr_5y_fy": [12.5, 8.0],
                "total_revenues_ltm": [1200, 900],
                "total_revenues_5yavgltm": [1000, 1000],
                "operating_income_ltm": [120, 80],
                "operating_income_fy": [100, 100],
                "ebitda_ltm": [150, 90],
                "ebitda_5yavgltm": [100, 100],
            }
        )
        result = engineer_growth_metrics(df)
        self.assertAlmostEqual(result.loc[0, "revenue_cagr_5y"], 12.5)
        self.assertAlmostEqual(result.loc[0, "revenue_vs_5y_avg"], 1.2)
        self.assertEqual(result.loc[0, "revenue_above_5y_avg_flag"], 1)
        self.assertAlmostEqual(result.loc[0, "operating_income_growth_yoy"], 20.0)
        self.assertAlmostEqual(result.loc[0, "ebitda_vs_5y_avg"], 1.5)


if __name__ == "__main__":
    unittest.main()
