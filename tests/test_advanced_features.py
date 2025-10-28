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
