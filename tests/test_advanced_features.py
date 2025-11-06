"""
Test suite for finance_ml.advanced_features module (Phase 9.3)

Comprehensive tests for advanced feature engineering functions following TDD principles.
Tests cover normal cases, edge cases, missing data, and error handling.
"""

import unittest

import numpy as np
import pandas as pd

from finance_ml.advanced_features import (
    _safe_div,
    engineer_valuation_ratios,
    engineer_profitability_ratios,
    engineer_leverage_ratios,
    engineer_liquidity_ratios,
    engineer_efficiency_ratios,
    engineer_growth_metrics,
    engineer_sector_specific_features,
    create_feature_interactions,
    create_relative_value_features,
    calculate_feature_importance_mutual_info,
    calculate_feature_importance_rf,
    build_comprehensive_features,
    )


class TestSafeDiv(unittest.TestCase):
    """Test _safe_div helper function."""

    def test_safe_div_normal_division(self):
        """Should divide two Series normally."""
        numer = pd.Series([10, 20, 30])
        denom = pd.Series([2, 4, 5])
        result = _safe_div(numer, denom)
        expected = pd.Series([5.0, 5.0, 6.0])
        pd.testing.assert_series_equal(result, expected)

    def test_safe_div_handles_zero_denominator(self):
        """Should return NaN when denominator is zero."""
        numer = pd.Series([10, 20])
        denom = pd.Series([0, 0])
        result = _safe_div(numer, denom)
        self.assertTrue(result.isna().all())

    def test_safe_div_handles_infinity(self):
        """Should replace infinity with NaN."""
        numer = pd.Series([10, 20])
        denom = pd.Series([0.0, 0.0])
        result = _safe_div(numer, denom)
        self.assertTrue(result.isna().all())


class TestEngineerValuationRatios(unittest.TestCase):
    """Test engineer_valuation_ratios function."""

    def test_p_e_ratio_calculated(self):
        """Should calculate P/E ratio correctly."""
        df = pd.DataFrame({"last_price": [100, 200], "eps": [5, 10]})
        result = engineer_valuation_ratios(df)
        self.assertIn("p_e_ratio", result.columns)
        np.testing.assert_array_almost_equal(result["p_e_ratio"], [20.0, 20.0])

    def test_p_b_ratio_calculated(self):
        """Should calculate P/B ratio correctly."""
        df = pd.DataFrame({"last_price": [50, 100], "book_value_per_share": [25, 50]})
        result = engineer_valuation_ratios(df)
        self.assertIn("p_b_ratio", result.columns)
        np.testing.assert_array_almost_equal(result["p_b_ratio"], [2.0, 2.0])

    def test_p_s_ratio_calculated(self):
        """Should calculate P/S ratio correctly."""
        df = pd.DataFrame(
            {"last_price": [100, 200], "revenue": [1000, 2000], "shares_outstanding": [100, 100]}
        )
        result = engineer_valuation_ratios(df)
        self.assertIn("p_s_ratio", result.columns)
        np.testing.assert_array_almost_equal(result["p_s_ratio"], [10.0, 10.0])

    def test_ev_ebitda_ratio_calculated(self):
        """Should calculate EV/EBITDA ratio correctly."""
        df = pd.DataFrame({"enterprise_value": [1000, 2000], "ebitda": [100, 200]})
        result = engineer_valuation_ratios(df)
        self.assertIn("ev_ebitda_ratio", result.columns)
        np.testing.assert_array_almost_equal(result["ev_ebitda_ratio"], [10.0, 10.0])

    def test_ev_sales_ratio_calculated(self):
        """Should calculate EV/Sales ratio correctly."""
        df = pd.DataFrame({"enterprise_value": [1000, 2000], "revenue": [500, 1000]})
        result = engineer_valuation_ratios(df)
        self.assertIn("ev_sales_ratio", result.columns)
        np.testing.assert_array_almost_equal(result["ev_sales_ratio"], [2.0, 2.0])

    def test_peg_ratio_calculated(self):
        """Should calculate PEG ratio correctly."""
        df = pd.DataFrame({"last_price": [100], "eps": [5], "earnings_growth_pct": [10]})
        result = engineer_valuation_ratios(df)
        self.assertIn("peg_ratio", result.columns)
        np.testing.assert_array_almost_equal(result["peg_ratio"], [2.0])

    def test_dividend_yield_calculated(self):
        """Should calculate dividend yield correctly."""
        df = pd.DataFrame({"dividend_per_share": [2, 4], "last_price": [100, 200]})
        result = engineer_valuation_ratios(df)
        self.assertIn("dividend_yield", result.columns)
        np.testing.assert_array_almost_equal(result["dividend_yield"], [2.0, 2.0])

    def test_missing_columns_handled(self):
        """Should handle missing columns gracefully."""
        df = pd.DataFrame({"other_col": [1, 2]})
        result = engineer_valuation_ratios(df)
        self.assertEqual(len(result.columns), 1)
        self.assertNotIn("p_e_ratio", result.columns)

    def test_preserves_original_columns(self):
        """Should preserve all original columns."""
        df = pd.DataFrame({"ticker": ["A", "B"], "last_price": [100, 200], "eps": [5, 10]})
        result = engineer_valuation_ratios(df)
        self.assertIn("ticker", result.columns)
        self.assertIn("last_price", result.columns)


class TestEngineerProfitabilityRatios(unittest.TestCase):
    """Test engineer_profitability_ratios function."""

    def test_roe_calculated(self):
        """Should calculate ROE (Return on Equity) correctly."""
        df = pd.DataFrame({"net_income": [100, 200], "total_equity": [1000, 2000]})
        result = engineer_profitability_ratios(df)
        self.assertIn("roe", result.columns)
        np.testing.assert_array_almost_equal(result["roe"], [10.0, 10.0])

    def test_roa_calculated(self):
        """Should calculate ROA (Return on Assets) correctly."""
        df = pd.DataFrame({"net_income": [100, 200], "total_assets": [2000, 4000]})
        result = engineer_profitability_ratios(df)
        self.assertIn("roa", result.columns)
        np.testing.assert_array_almost_equal(result["roa"], [5.0, 5.0])

    def test_roic_calculated(self):
        """Should calculate ROIC (Return on Invested Capital) correctly."""
        df = pd.DataFrame(
            {"net_income": [100, 200], "total_equity": [800, 1600], "total_debt": [200, 400]}
        )
        result = engineer_profitability_ratios(df)
        self.assertIn("roic", result.columns)
        np.testing.assert_array_almost_equal(result["roic"], [10.0, 10.0])

    def test_gross_margin_calculated(self):
        """Should calculate Gross Margin % correctly."""
        df = pd.DataFrame({"gross_profit": [400, 600], "revenue": [1000, 2000]})
        result = engineer_profitability_ratios(df)
        self.assertIn("gross_margin_pct", result.columns)
        np.testing.assert_array_almost_equal(result["gross_margin_pct"], [40.0, 30.0])

    def test_operating_margin_calculated(self):
        """Should calculate Operating Margin % correctly."""
        df = pd.DataFrame({"operating_income": [200, 400], "revenue": [1000, 2000]})
        result = engineer_profitability_ratios(df)
        self.assertIn("operating_margin_pct", result.columns)
        np.testing.assert_array_almost_equal(result["operating_margin_pct"], [20.0, 20.0])

    def test_net_margin_calculated(self):
        """Should calculate Net Margin % correctly."""
        df = pd.DataFrame({"net_income": [100, 250], "revenue": [1000, 2000]})
        result = engineer_profitability_ratios(df)
        self.assertIn("net_margin_pct", result.columns)
        np.testing.assert_array_almost_equal(result["net_margin_pct"], [10.0, 12.5])

    def test_missing_columns_handled(self):
        """Should handle missing columns gracefully."""
        df = pd.DataFrame({"other_col": [1, 2]})
        result = engineer_profitability_ratios(df)
        self.assertEqual(len(result.columns), 1)
        self.assertNotIn("roe", result.columns)


class TestEngineerLeverageRatios(unittest.TestCase):
    """Test engineer_leverage_ratios function."""

    def test_debt_to_equity_calculated(self):
        """Should calculate Debt to Equity ratio correctly."""
        df = pd.DataFrame({"total_debt": [500, 1000], "total_equity": [1000, 2000]})
        result = engineer_leverage_ratios(df)
        self.assertIn("debt_to_equity", result.columns)
        np.testing.assert_array_almost_equal(result["debt_to_equity"], [0.5, 0.5])

    def test_net_debt_to_ebitda_calculated(self):
        """Should calculate Net Debt to EBITDA ratio correctly."""
        df = pd.DataFrame({"net_debt": [1000, 2000], "ebitda": [200, 400]})
        result = engineer_leverage_ratios(df)
        self.assertIn("net_debt_to_ebitda", result.columns)
        np.testing.assert_array_almost_equal(result["net_debt_to_ebitda"], [5.0, 5.0])

    def test_interest_coverage_calculated(self):
        """Should calculate Interest Coverage ratio correctly."""
        df = pd.DataFrame({"ebit": [1000, 2000], "interest_expense": [100, 200]})
        result = engineer_leverage_ratios(df)
        self.assertIn("interest_coverage", result.columns)
        np.testing.assert_array_almost_equal(result["interest_coverage"], [10.0, 10.0])

    def test_debt_to_assets_calculated(self):
        """Should calculate Debt to Assets ratio correctly."""
        df = pd.DataFrame({"total_debt": [500, 1000], "total_assets": [2000, 4000]})
        result = engineer_leverage_ratios(df)
        self.assertIn("debt_to_assets", result.columns)
        np.testing.assert_array_almost_equal(result["debt_to_assets"], [0.25, 0.25])

    def test_equity_ratio_calculated(self):
        """Should calculate Equity Ratio correctly."""
        df = pd.DataFrame({"total_equity": [1500, 3000], "total_assets": [2000, 4000]})
        result = engineer_leverage_ratios(df)
        self.assertIn("equity_ratio", result.columns)
        np.testing.assert_array_almost_equal(result["equity_ratio"], [0.75, 0.75])


class TestEngineerLiquidityRatios(unittest.TestCase):
    """Test engineer_liquidity_ratios function."""

    def test_current_ratio_calculated(self):
        """Should calculate Current Ratio correctly."""
        df = pd.DataFrame({"current_assets": [2000, 4000], "current_liabilities": [1000, 2000]})
        result = engineer_liquidity_ratios(df)
        self.assertIn("current_ratio", result.columns)
        np.testing.assert_array_almost_equal(result["current_ratio"], [2.0, 2.0])

    def test_quick_ratio_calculated(self):
        """Should calculate Quick Ratio correctly."""
        df = pd.DataFrame(
            {
                "current_assets": [2000, 4000],
                "inventory": [500, 1000],
                "current_liabilities": [1000, 2000],
            }
        )
        result = engineer_liquidity_ratios(df)
        self.assertIn("quick_ratio", result.columns)
        np.testing.assert_array_almost_equal(result["quick_ratio"], [1.5, 1.5])

    def test_cash_ratio_calculated(self):
        """Should calculate Cash Ratio correctly."""
        df = pd.DataFrame(
            {"cash_and_equivalents": [500, 1000], "current_liabilities": [1000, 2000]}
        )
        result = engineer_liquidity_ratios(df)
        self.assertIn("cash_ratio", result.columns)
        np.testing.assert_array_almost_equal(result["cash_ratio"], [0.5, 0.5])

    def test_working_capital_to_sales_calculated(self):
        """Should calculate Working Capital to Sales ratio correctly."""
        df = pd.DataFrame({"working_capital": [1000, 2000], "revenue": [10000, 20000]})
        result = engineer_liquidity_ratios(df)
        self.assertIn("working_capital_to_sales", result.columns)
        np.testing.assert_array_almost_equal(result["working_capital_to_sales"], [0.1, 0.1])


class TestEngineerEfficiencyRatios(unittest.TestCase):
    """Test engineer_efficiency_ratios function."""

    def test_asset_turnover_calculated(self):
        """Should calculate Asset Turnover correctly."""
        df = pd.DataFrame({"revenue": [10000, 20000], "total_assets": [5000, 10000]})
        result = engineer_efficiency_ratios(df)
        self.assertIn("asset_turnover", result.columns)
        np.testing.assert_array_almost_equal(result["asset_turnover"], [2.0, 2.0])

    def test_inventory_turnover_calculated(self):
        """Should calculate Inventory Turnover correctly."""
        df = pd.DataFrame({"cogs": [8000, 16000], "inventory": [1000, 2000]})
        result = engineer_efficiency_ratios(df)
        self.assertIn("inventory_turnover", result.columns)
        np.testing.assert_array_almost_equal(result["inventory_turnover"], [8.0, 8.0])

    def test_receivables_turnover_calculated(self):
        """Should calculate Receivables Turnover correctly."""
        df = pd.DataFrame({"revenue": [10000, 20000], "accounts_receivable": [2000, 4000]})
        result = engineer_efficiency_ratios(df)
        self.assertIn("receivables_turnover", result.columns)
        np.testing.assert_array_almost_equal(result["receivables_turnover"], [5.0, 5.0])

    def test_revenue_per_employee_calculated(self):
        """Should calculate Revenue per Employee correctly."""
        df = pd.DataFrame({"revenue": [10000000, 20000000], "employees": [100, 200]})
        result = engineer_efficiency_ratios(df)
        self.assertIn("revenue_per_employee", result.columns)
        np.testing.assert_array_almost_equal(result["revenue_per_employee"], [100000.0, 100000.0])


class TestEngineerGrowthMetrics(unittest.TestCase):
    """Test engineer_growth_metrics function."""

    def test_revenue_growth_yoy_calculated(self):
        """Should calculate Revenue Growth YoY correctly."""
        df = pd.DataFrame({"revenue": [11000, 22000], "revenue_previous_year": [10000, 20000]})
        result = engineer_growth_metrics(df)
        self.assertIn("revenue_growth_yoy", result.columns)
        np.testing.assert_array_almost_equal(result["revenue_growth_yoy"], [10.0, 10.0])

    def test_eps_growth_yoy_calculated(self):
        """Should calculate EPS Growth YoY correctly."""
        df = pd.DataFrame({"eps": [5.5, 11.0], "eps_previous_year": [5.0, 10.0]})
        result = engineer_growth_metrics(df)
        self.assertIn("eps_growth_yoy", result.columns)
        np.testing.assert_array_almost_equal(result["eps_growth_yoy"], [10.0, 10.0])

    def test_ebitda_growth_yoy_calculated(self):
        """Should calculate EBITDA Growth YoY correctly."""
        df = pd.DataFrame({"ebitda": [2200, 4400], "ebitda_previous_year": [2000, 4000]})
        result = engineer_growth_metrics(df)
        self.assertIn("ebitda_growth_yoy", result.columns)
        np.testing.assert_array_almost_equal(result["ebitda_growth_yoy"], [10.0, 10.0])

    def test_missing_columns_handled(self):
        """Should handle missing columns gracefully."""
        df = pd.DataFrame({"other_col": [1, 2]})
        result = engineer_growth_metrics(df)
        self.assertEqual(len(result.columns), 1)
        self.assertNotIn("revenue_growth_yoy", result.columns)


class TestEngineerSectorSpecificFeatures(unittest.TestCase):
    """Test engineer_sector_specific_features function."""

    def test_financials_tangible_book_value(self):
        """Should calculate Tangible Book Value for Financials sector."""
        df = pd.DataFrame(
            {
                "sector": ["Financials", "Technology"],
                "total_equity": [10000, 20000],
                "intangible_assets": [1000, 2000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("tangible_book_value", result.columns)
        self.assertEqual(result.loc[0, "tangible_book_value"], 9000)
        self.assertTrue(pd.isna(result.loc[1, "tangible_book_value"]))

    def test_technology_r_d_intensity(self):
        """Should calculate R&D Intensity for Technology sector."""
        df = pd.DataFrame(
            {
                "sector": ["Information Technology", "Energy"],
                "r_d_expenses": [1000, 500],
                "revenue": [10000, 10000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("r_d_intensity", result.columns)
        self.assertAlmostEqual(result.loc[0, "r_d_intensity"], 10.0)

    def test_healthcare_r_d_intensity(self):
        """Should calculate R&D Intensity for Healthcare sector."""
        df = pd.DataFrame(
            {
                "sector": ["Health Care", "Materials"],
                "r_d_expenses": [2000, 100],
                "revenue": [10000, 10000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("r_d_intensity", result.columns)
        self.assertAlmostEqual(result.loc[0, "r_d_intensity"], 20.0)

    def test_industrials_capex_intensity(self):
        """Should calculate CAPEX Intensity for Industrials sector."""
        df = pd.DataFrame(
            {
                "sector": ["Industrials", "Technology"],
                "capex": [500, 1000],
                "revenue": [10000, 10000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("capex_intensity", result.columns)
        self.assertAlmostEqual(result.loc[0, "capex_intensity"], 5.0)

    def test_missing_sector_column(self):
        """Should handle missing sector column gracefully."""
        df = pd.DataFrame({"revenue": [1000, 2000]})
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertEqual(len(result.columns), 1)

    # Additional Financials sector tests
    def test_financials_p_tbv_ratio(self):
        """Should calculate Price to Tangible Book Value for Financials."""
        df = pd.DataFrame(
            {
                "sector": ["Financials", "Technology"],
                "last_price": [100, 200],
                "total_equity": [10000, 20000],
                "intangible_assets": [1000, 2000],
                "shares_outstanding": [100, 100],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("p_tbv_ratio", result.columns)
        # TBV per share = (10000-1000)/100 = 90, P/TBV = 100/90 = 1.111
        self.assertAlmostEqual(result.loc[0, "p_tbv_ratio"], 1.111, places=2)

    def test_financials_net_interest_margin(self):
        """Should calculate Net Interest Margin for Financials."""
        df = pd.DataFrame(
            {
                "sector": ["Financials", "Energy"],
                "interest_income": [1000, 500],
                "interest_expense": [400, 200],
                "earning_assets": [10000, 8000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("net_interest_margin", result.columns)
        # NIM = (1000-400)/10000 * 100 = 6.0
        self.assertAlmostEqual(result.loc[0, "net_interest_margin"], 6.0)

    def test_financials_efficiency_ratio(self):
        """Should calculate Efficiency Ratio for Financials."""
        df = pd.DataFrame(
            {
                "sector": ["Financials", "Technology"],
                "operating_expenses": [500, 600],
                "revenue": [1000, 1200],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("efficiency_ratio", result.columns)
        # Efficiency = 500/1000 * 100 = 50.0
        self.assertAlmostEqual(result.loc[0, "efficiency_ratio"], 50.0)

    # Energy/Materials sector tests
    def test_energy_capex_intensity(self):
        """Should calculate CAPEX intensity for Energy/Materials."""
        df = pd.DataFrame(
            {
                "sector": ["Energy", "Technology"],
                "capex": [2000, 1000],
                "revenue": [10000, 10000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("capex_intensity", result.columns)
        self.assertAlmostEqual(result.loc[0, "capex_intensity"], 20.0)

    def test_energy_asset_turnover(self):
        """Should calculate Asset Turnover for Energy/Materials."""
        df = pd.DataFrame(
            {
                "sector": ["Materials", "Healthcare"],
                "revenue": [50000, 40000],
                "total_assets": [100000, 80000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("asset_turnover", result.columns)
        self.assertAlmostEqual(result.loc[0, "asset_turnover"], 0.5)

    # Technology sector tests
    def test_technology_sga_efficiency(self):
        """Should calculate SG&A efficiency for Technology."""
        df = pd.DataFrame(
            {
                "sector": ["Technology", "Energy"],
                "sga_expenses": [2000, 1500],
                "revenue": [10000, 10000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("sga_efficiency", result.columns)
        self.assertAlmostEqual(result.loc[0, "sga_efficiency"], 20.0)

    def test_technology_rule_of_40(self):
        """Should calculate Rule of 40 for Technology."""
        df = pd.DataFrame(
            {
                "sector": ["Information Technology", "Consumer"],
                "revenue_growth_yoy": [25.0, 10.0],
                "operating_margin_pct": [18.0, 15.0],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("rule_of_40", result.columns)
        # Rule of 40 = 25 + 18 = 43
        self.assertAlmostEqual(result.loc[0, "rule_of_40"], 43.0)

    def test_technology_cash_burn_rate(self):
        """Should calculate Cash Burn Rate for Technology."""
        df = pd.DataFrame(
            {
                "sector": ["Technology", "Utilities"],
                "operating_cash_flow": [-500, 1000],
                "capex": [200, 500],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("cash_burn_rate", result.columns)
        # Cash burn = -500 - 200 = -700
        self.assertAlmostEqual(result.loc[0, "cash_burn_rate"], -700.0)

    # Healthcare sector tests
    def test_healthcare_r_d_to_revenue_ratio(self):
        """Should calculate R&D to Revenue ratio for Healthcare."""
        df = pd.DataFrame(
            {
                "sector": ["Healthcare", "Industrials"],
                "r_d_expenses": [3000, 500],
                "revenue": [10000, 10000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("r_d_intensity", result.columns)
        self.assertAlmostEqual(result.loc[0, "r_d_intensity"], 30.0)

    # Consumer sector tests
    def test_consumer_inventory_days(self):
        """Should calculate Inventory Days for Consumer."""
        df = pd.DataFrame(
            {
                "sector": ["Consumer Discretionary", "Financials"],
                "inventory": [5000, 1000],
                "cost_of_goods_sold": [36500, 30000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("inventory_days", result.columns)
        # Inventory days = (5000 / 36500) * 365 = 50
        self.assertAlmostEqual(result.loc[0, "inventory_days"], 50.0, places=1)

    def test_consumer_marketing_efficiency(self):
        """Should calculate Marketing Efficiency for Consumer."""
        df = pd.DataFrame(
            {
                "sector": ["Consumer Staples", "Energy"],
                "marketing_expenses": [1000, 500],
                "revenue": [10000, 10000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("marketing_efficiency", result.columns)
        self.assertAlmostEqual(result.loc[0, "marketing_efficiency"], 10.0)

    # Industrials sector tests
    def test_industrials_capex_to_depreciation(self):
        """Should calculate CAPEX to Depreciation for Industrials."""
        df = pd.DataFrame(
            {
                "sector": ["Industrials", "Technology"],
                "capex": [800, 600],
                "depreciation_amortization": [500, 400],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("capex_to_depreciation", result.columns)
        self.assertAlmostEqual(result.loc[0, "capex_to_depreciation"], 1.6)

    def test_industrials_working_capital_efficiency(self):
        """Should calculate Working Capital Efficiency for Industrials."""
        df = pd.DataFrame(
            {
                "sector": ["Industrials", "Utilities"],
                "current_assets": [8000, 6000],
                "current_liabilities": [5000, 4000],
                "revenue": [30000, 25000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("working_capital_efficiency", result.columns)
        # WC = (8000-5000)/30000 * 100 = 10.0
        self.assertAlmostEqual(result.loc[0, "working_capital_efficiency"], 10.0)

    # Utilities sector tests
    def test_utilities_dividend_payout_ratio(self):
        """Should calculate Dividend Payout Ratio for Utilities."""
        df = pd.DataFrame(
            {
                "sector": ["Utilities", "Technology"],
                "dividends_paid": [400, 100],
                "net_income": [1000, 2000],
            }
        )
        result = engineer_sector_specific_features(df, sector_col="sector")
        self.assertIn("dividend_payout_ratio", result.columns)
        # Payout = 400/1000 * 100 = 40.0
        self.assertAlmostEqual(result.loc[0, "dividend_payout_ratio"], 40.0)


class TestEngineerTemporalFeatures(unittest.TestCase):
    """Test engineer_temporal_features function."""

    def test_fiscal_quarter_from_date(self):
        """Should extract fiscal quarter from date column."""
        df = pd.DataFrame(
            {
                "report_date": pd.to_datetime(
                    ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
                ),
            }
        )
        from finance_ml.advanced_features import engineer_temporal_features

        result = engineer_temporal_features(df, date_col="report_date")
        self.assertIn("fiscal_quarter", result.columns)
        self.assertEqual(result.loc[0, "fiscal_quarter"], 1)
        self.assertEqual(result.loc[1, "fiscal_quarter"], 2)

    def test_month_from_date(self):
        """Should extract month from date column."""
        df = pd.DataFrame(
            {
                "report_date": pd.to_datetime(["2024-01-15", "2024-06-20"]),
            }
        )
        from finance_ml.advanced_features import engineer_temporal_features

        result = engineer_temporal_features(df, date_col="report_date")
        self.assertIn("month", result.columns)
        self.assertEqual(result.loc[0, "month"], 1)
        self.assertEqual(result.loc[1, "month"], 6)

    def test_year_from_date(self):
        """Should extract year from date column."""
        df = pd.DataFrame(
            {
                "report_date": pd.to_datetime(["2023-12-31", "2024-06-30"]),
            }
        )
        from finance_ml.advanced_features import engineer_temporal_features

        result = engineer_temporal_features(df, date_col="report_date")
        self.assertIn("year", result.columns)
        self.assertEqual(result.loc[0, "year"], 2023)
        self.assertEqual(result.loc[1, "year"], 2024)

    def test_days_since_reference(self):
        """Should calculate days since reference date."""
        df = pd.DataFrame(
            {
                "report_date": pd.to_datetime(["2024-01-10", "2024-01-20"]),
            }
        )
        from finance_ml.advanced_features import engineer_temporal_features

        reference = pd.to_datetime("2024-01-01")
        result = engineer_temporal_features(df, date_col="report_date", reference_date=reference)
        self.assertIn("days_since_reference", result.columns)
        self.assertEqual(result.loc[0, "days_since_reference"], 9)
        self.assertEqual(result.loc[1, "days_since_reference"], 19)

    def test_missing_date_column(self):
        """Should handle missing date column gracefully."""
        df = pd.DataFrame({"other_col": [1, 2]})
        from finance_ml.advanced_features import engineer_temporal_features

        result = engineer_temporal_features(df, date_col="report_date")
        self.assertEqual(len(result.columns), 1)


class TestEngineerMarketMicrostructureFeatures(unittest.TestCase):
    """Test engineer_market_microstructure_features function."""

    def test_historical_volatility_30d(self):
        """Should calculate 30-day historical volatility."""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 40,
                "last_price": [100 + i for i in range(40)],
            }
        )
        from finance_ml.advanced_features import engineer_market_microstructure_features

        result = engineer_market_microstructure_features(
            df, price_col="last_price", group_col="ticker"
        )
        self.assertIn("volatility_30d", result.columns)

    def test_price_momentum(self):
        """Should calculate price momentum (rate of change)."""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 30,
                "last_price": [100, 105, 110, 115, 120] + [120] * 25,
            }
        )
        from finance_ml.advanced_features import engineer_market_microstructure_features

        result = engineer_market_microstructure_features(
            df, price_col="last_price", group_col="ticker"
        )
        self.assertIn("momentum_20d", result.columns)

    def test_moving_average_crossover(self):
        """Should calculate moving averages."""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 60,
                "last_price": [100 + i * 0.5 for i in range(60)],
            }
        )
        from finance_ml.advanced_features import engineer_market_microstructure_features

        result = engineer_market_microstructure_features(
            df, price_col="last_price", group_col="ticker"
        )
        self.assertIn("ma_50d", result.columns)

    def test_price_range_indicator(self):
        """Should calculate price range (high-low spread)."""
        df = pd.DataFrame(
            {
                "high_price": [110, 115, 120],
                "low_price": [90, 95, 100],
                "last_price": [100, 105, 110],
            }
        )
        from finance_ml.advanced_features import engineer_market_microstructure_features

        result = engineer_market_microstructure_features(df, price_col="last_price")
        self.assertIn("price_range_pct", result.columns)
        # (110-90)/100 * 100 = 20%
        self.assertAlmostEqual(result.loc[0, "price_range_pct"], 20.0)

    def test_missing_price_column(self):
        """Should handle missing price column gracefully."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        from finance_ml.advanced_features import engineer_market_microstructure_features

        result = engineer_market_microstructure_features(df, price_col="last_price")
        self.assertEqual(len(result.columns), 1)


class TestEngineerNonlinearTransforms(unittest.TestCase):
    """Test engineer_nonlinear_transforms function."""

    def test_log_transform(self):
        """Should apply log transformation to skewed features."""
        df = pd.DataFrame(
            {
                "market_cap": [1000000, 5000000, 10000000],
                "revenue": [100000, 500000, 1000000],
            }
        )
        from finance_ml.advanced_features import engineer_nonlinear_transforms

        result = engineer_nonlinear_transforms(df, log_features=["market_cap", "revenue"])
        self.assertIn("log_market_cap", result.columns)
        self.assertIn("log_revenue", result.columns)
        # log(1000000) ≈ 13.82
        self.assertAlmostEqual(result.loc[0, "log_market_cap"], np.log(1000000), places=2)

    def test_sqrt_transform(self):
        """Should apply square root transformation."""
        df = pd.DataFrame(
            {
                "variance_metric": [100, 400, 900],
            }
        )
        from finance_ml.advanced_features import engineer_nonlinear_transforms

        result = engineer_nonlinear_transforms(df, sqrt_features=["variance_metric"])
        self.assertIn("sqrt_variance_metric", result.columns)
        self.assertEqual(result.loc[0, "sqrt_variance_metric"], 10.0)
        self.assertEqual(result.loc[1, "sqrt_variance_metric"], 20.0)

    def test_inverse_transform(self):
        """Should apply inverse transformation."""
        df = pd.DataFrame(
            {
                "p_e_ratio": [10, 20, 25],
                "p_b_ratio": [2, 4, 5],
            }
        )
        from finance_ml.advanced_features import engineer_nonlinear_transforms

        result = engineer_nonlinear_transforms(df, inverse_features=["p_e_ratio", "p_b_ratio"])
        self.assertIn("inv_p_e_ratio", result.columns)
        self.assertIn("inv_p_b_ratio", result.columns)
        self.assertAlmostEqual(result.loc[0, "inv_p_e_ratio"], 0.1)
        self.assertAlmostEqual(result.loc[1, "inv_p_b_ratio"], 0.25)

    def test_handles_negative_values_in_log(self):
        """Should handle negative/zero values in log transform."""
        df = pd.DataFrame(
            {
                "metric": [-10, 0, 10, 100],
            }
        )
        from finance_ml.advanced_features import engineer_nonlinear_transforms

        result = engineer_nonlinear_transforms(df, log_features=["metric"])
        self.assertIn("log_metric", result.columns)
        # Negative and zero should produce NaN
        self.assertTrue(pd.isna(result.loc[0, "log_metric"]))
        self.assertTrue(pd.isna(result.loc[1, "log_metric"]))

    def test_missing_columns_handled(self):
        """Should handle missing columns gracefully."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        from finance_ml.advanced_features import engineer_nonlinear_transforms

        result = engineer_nonlinear_transforms(df, log_features=["missing_col"])
        self.assertEqual(len(result.columns), 1)


class TestCreateFeatureInteractions(unittest.TestCase):
    """Test create_feature_interactions function."""

    def test_pairwise_interactions_created(self):
        """Should create pairwise interactions."""
        df = pd.DataFrame({"market_cap": [1000, 2000], "p_e_ratio": [10, 20]})
        result = create_feature_interactions(df, features=["market_cap", "p_e_ratio"])
        self.assertIn("market_cap_x_p_e_ratio", result.columns)
        np.testing.assert_array_almost_equal(result["market_cap_x_p_e_ratio"], [10000, 40000])

    def test_polynomial_features_created(self):
        """Should create squared features."""
        df = pd.DataFrame({"roe": [10, 20]})
        result = create_feature_interactions(df, features=["roe"], max_degree=2)
        self.assertIn("roe_squared", result.columns)
        np.testing.assert_array_almost_equal(result["roe_squared"], [100, 400])

    def test_insufficient_features(self):
        """Should handle insufficient features gracefully."""
        df = pd.DataFrame({"single_feature": [1, 2]})
        result = create_feature_interactions(df, features=["single_feature"])
        # Should create polynomial feature even with 1 feature, but no interactions
        self.assertEqual(len(result.columns), 2)  # original + squared
        self.assertIn("single_feature_squared", result.columns)


class TestCreateRelativeValueFeatures(unittest.TestCase):
    """Test create_relative_value_features function."""

    def test_sector_median_deviation(self):
        """Should calculate deviation from sector median."""
        df = pd.DataFrame(
            {"sector": ["Tech", "Tech", "Finance", "Finance"], "p_e_ratio": [10, 20, 5, 15]}
        )
        result = create_relative_value_features(df, sector_col="sector", metrics=["p_e_ratio"])
        self.assertIn("p_e_ratio_vs_sector_median", result.columns)
        # Tech median = 15, Finance median = 10
        self.assertEqual(result.loc[0, "p_e_ratio_vs_sector_median"], -5)
        self.assertEqual(result.loc[1, "p_e_ratio_vs_sector_median"], 5)

    def test_sector_zscore(self):
        """Should calculate sector z-score."""
        df = pd.DataFrame({"sector": ["Tech", "Tech", "Tech"], "roe": [10, 20, 30]})
        result = create_relative_value_features(df, sector_col="sector", metrics=["roe"])
        self.assertIn("roe_sector_zscore", result.columns)
        # Middle value should have z-score close to 0
        self.assertAlmostEqual(result.loc[1, "roe_sector_zscore"], 0.0, places=5)

    def test_sector_percentile(self):
        """Should calculate sector percentile rank."""
        df = pd.DataFrame({"sector": ["Tech", "Tech", "Tech"], "p_b_ratio": [1, 2, 3]})
        result = create_relative_value_features(df, sector_col="sector", metrics=["p_b_ratio"])
        self.assertIn("p_b_ratio_sector_percentile", result.columns)
        # Should be roughly 33, 66, 100 percentile
        self.assertGreater(result.loc[2, "p_b_ratio_sector_percentile"], 50)

    def test_missing_sector_column(self):
        """Should handle missing sector column gracefully."""
        df = pd.DataFrame({"p_e_ratio": [10, 20]})
        result = create_relative_value_features(df, sector_col="sector")
        self.assertEqual(len(result.columns), 1)


class TestCalculateFeatureImportanceShap(unittest.TestCase):
    """Test calculate_feature_importance_shap function."""

    def test_returns_dataframe_with_shap(self):
        """Should return DataFrame with SHAP importance values."""
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                "feature3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        from finance_ml.advanced_features import calculate_feature_importance_shap

        result = calculate_feature_importance_shap(X, y)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("feature", result.columns)
        self.assertIn("importance", result.columns)

    def test_top_k_features_shap(self):
        """Should return top k features when specified."""
        X = pd.DataFrame({"f1": range(10), "f2": range(10, 20), "f3": range(20, 30)})
        y = pd.Series(range(10))
        from finance_ml.advanced_features import calculate_feature_importance_shap

        result = calculate_feature_importance_shap(X, y, top_k=2)
        self.assertEqual(len(result), 2)


class TestCalculateFeatureImportanceRfe(unittest.TestCase):
    """Test calculate_feature_importance_rfe function."""

    def test_returns_selected_features(self):
        """Should return selected features using RFE."""
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                "feature3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        from finance_ml.advanced_features import calculate_feature_importance_rfe

        result = calculate_feature_importance_rfe(X, y, n_features_to_select=2)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_rfe_with_cv(self):
        """Should work with cross-validation."""
        X = pd.DataFrame({"f1": range(20), "f2": range(20, 40), "f3": range(40, 60)})
        y = pd.Series(range(20))
        from finance_ml.advanced_features import calculate_feature_importance_rfe

        result = calculate_feature_importance_rfe(X, y, n_features_to_select=2, cv=3)
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 3)


class TestCalculateFeatureImportanceMutualInfo(unittest.TestCase):
    """Test calculate_feature_importance_mutual_info function."""

    def test_returns_dataframe(self):
        """Should return DataFrame with feature importance."""
        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]})
        y = pd.Series([10, 20, 30, 40, 50])
        result = calculate_feature_importance_mutual_info(X, y)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("feature", result.columns)
        self.assertIn("importance", result.columns)

    def test_top_k_features(self):
        """Should return only top_k features."""
        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [2, 4, 6, 8, 10], "f3": [5, 4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40, 50])
        result = calculate_feature_importance_mutual_info(X, y, top_k=2)
        self.assertEqual(len(result), 2)


class TestCalculateFeatureImportanceRf(unittest.TestCase):
    """Test calculate_feature_importance_rf function."""

    def test_returns_dataframe(self):
        """Should return DataFrame with feature importance."""
        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]})
        y = pd.Series([10, 20, 30, 40, 50])
        result = calculate_feature_importance_rf(X, y, n_estimators=10)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("feature", result.columns)
        self.assertIn("importance", result.columns)

    def test_top_k_features(self):
        """Should return only top_k features."""
        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [2, 4, 6, 8, 10], "f3": [5, 4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40, 50])
        result = calculate_feature_importance_rf(X, y, top_k=2, n_estimators=10)
        self.assertEqual(len(result), 2)

    def test_calculate_feature_importance_rf_with_nan_columns(self):
        """Test that feature importance handles columns dropped during cleaning."""
        # Create data with non-numeric columns that will be dropped during cleaning
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e'],  # Non-numeric, will be dropped
            'feature3': [10, 20, 30, 40, 50]
        })
        y = pd.Series([100, 200, 300, 400, 500])
        
        # Call function
        result = calculate_feature_importance_rf(X, y, top_k=5, n_estimators=10)
        
        # Verify: result should only include numeric features present in X_clean
        self.assertEqual(len(result), 2)  # Only feature1 and feature3
        self.assertIn('feature1', result['feature'].values)
        self.assertIn('feature3', result['feature'].values)
        self.assertNotIn('feature2', result['feature'].values)

    def test_calculate_feature_importance_rf_empty_after_cleaning(self):
        """Test that function returns empty DataFrame when all columns are non-numeric."""
        X = pd.DataFrame({
            'feature1': ['a', 'b'],
            'feature2': ['x', 'y']
        })
        y = pd.Series([100, 200])
        
        result = calculate_feature_importance_rf(X, y, n_estimators=10)
        
        self.assertEqual(len(result), 0)
        self.assertListEqual(list(result.columns), ['feature', 'importance'])


class TestBuildComprehensiveFeatures(unittest.TestCase):
    """Test build_comprehensive_features orchestrator function."""

    def test_adds_multiple_features(self):
        """Should add features from multiple engineering steps."""
        df = pd.DataFrame(
            {
                "sector": ["Tech", "Finance"],
                "last_price": [100, 50],
                "eps": [5, 2.5],
                "net_income": [100, 50],
                "total_equity": [1000, 500],
            }
        )
        result = build_comprehensive_features(
            df, include_interactions=False, include_relative_values=False
        )
        # Should have original + valuation + profitability features
        self.assertGreater(len(result.columns), len(df.columns))
        self.assertIn("p_e_ratio", result.columns)
        self.assertIn("roe", result.columns)

    def test_includes_interactions_when_enabled(self):
        """Should include interaction features when enabled."""
        df = pd.DataFrame(
            {
                "sector": ["Tech", "Finance"],
                "market_cap": [1000, 2000],
                "p_e_ratio": [10, 20],
                "roe": [15, 25],
            }
        )
        result = build_comprehensive_features(
            df, include_interactions=True, include_relative_values=False
        )
        # Check for interaction features
        interaction_cols = [col for col in result.columns if "_x_" in col or "_squared" in col]
        self.assertGreater(len(interaction_cols), 0)

    def test_includes_relative_values_when_enabled(self):
        """Should include relative value features when enabled."""
        df = pd.DataFrame(
            {
                "sector": ["Tech", "Tech"],
                "last_price": [100, 200],
                "eps": [5, 10],
                "net_income": [100, 200],
                "total_equity": [1000, 2000],
            }
        )
        result = build_comprehensive_features(
            df, include_interactions=False, include_relative_values=True
        )
        # Check for relative value features
        relative_cols = [col for col in result.columns if "sector" in col]
        self.assertGreater(len(relative_cols), 0)


if __name__ == "__main__":
    unittest.main()
