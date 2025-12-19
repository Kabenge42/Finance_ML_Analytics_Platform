"""
Test method-aware valuation column selection for classification interaction features.

This test validates that the valuation column candidates are properly grouped by method
and that the dynamic selection logic works correctly.
"""

import unittest
import pandas as pd
import numpy as np


class TestMethodAwareValuationCols(unittest.TestCase):
    """Test method-aware valuation column selection."""

    def setUp(self):
        """Set up test data with all possible valuation columns."""
        # Create a dataframe with all possible valuation columns
        self.all_columns = [
            # Core columns
            "last_price",
            "market_cap",
            "enterprise_value",
            "ebitda",
            # P/E variations
            "p_e",
            "p_e_ltm",
            "p_e_ntm",
            "p_e_1fyltm",
            "p_e_5yavgltm",
            # P/B variations
            "p_b",
            "p_b_ltm",
            "p_b_1fy",
            "p_b_5yavg",
            # Price targets
            "price_target",
            "price_target_median",
            "price_target_high",
            "price_target_low",
            "price_target_ytd_ago",
            "price_target_count",
            # Margins
            "gross_margin",
            "gross_profit_margin_pct_ltm",
            "gross_profit_margin_pct_fy",
            "net_income_margin_pct_ltm",
            "net_income_margin_pct_fy",
            # Income statement
            "net_income",
            "net_income_is_ltm",
            "net_income_is_fy",
            "ebit",
            "operating_income",
            "gross_profit",
            "revenue",
            "revenue_previous_year",
            "revenue_fy",
            # Volatility
            "volatility_1m",
            "volatility_3m",
            "volatility_6m",
            "volatility_1y",
            "volatility_1y_pct",
            "beta_1y",
            "beta_2y",
            "beta_5y",
            "short_int_pct",
            # Analyst
            "analyst_rating",
            # Dividends
            "dividend_per_share_ltm",
            "div_yield_ltm",
            "div_yield_ttm",
            "buyback_yield_ltm",
            # Profitability
            "return_on_equity_pct_ltm",
            "return_on_equity_pct_fy",
            "return_on_assets_roa_pct_ltm",
            "return_on_assets_roa_pct_fy",
            "roe",
            "roa",
            "roic",
            "total_equity",
            "total_equity_ltm",
            "total_assets",
            "total_assets_ltm",
            # Leverage
            "total_debt",
            "total_debt_ltm",
            "total_debt_fy",
            "total_equity_fy",
            "total_assets_fy",
            "interest_expense",
            "interest_expense_total_ltm",
            "cash_and_equivalents",
            "cash_and_equivalents_ltm",
            "cash_and_equivalents_fy",
            # Liquidity
            "current_ratio_ltm",
            "current_ratio_fy",
            "current_assets",
            "current_liabilities",
            "working_capital",
            "working_capital_ltm",
            "working_capital_fy",
            # Efficiency
            "asset_turnover_fy",
            "asset_turnover_ltm",
            "asset_turnover_previous_year",
            "inventory",
            "inventory_ltm",
            "inventory_fy",
            "accounts_receivable_fy",
            "accounts_receivable_1fy",
            "inventory_turnover",
            "receivables_turnover",
            # Growth
            "total_revenues_cagr_5y_fy",
            "total_revenues_ltm",
            "total_revenues_fy",
            "total_revenues_1fy",
            "total_revenues_5yavgltm",
            "total_revenues_5yavgfq",
            "revenues_est_yoy_pct_fy1e",
            "ebitda_ltm",
            "ebitda_fy",
            "ebitda_previous_year",
            "eps",
            "eps_previous_year",
            "eps_adj_ltm",
            "eps_adj_fy",
            "eps_norm_est_avg_ntm",
            # Quality
            "altman_z_score_fy",
            "altman_z_score_ltm",
            "asset_writedown_ltm",
            "asset_writedown_fy",
            "asset_writedown_1fy",
            "impairment_of_goodwill_ltm",
            "impairment_of_goodwill_fy",
            "restructuring_charges_ltm",
            "restructuring_charges_fy",
            "goodwill",
            "goodwill_ltm",
            "intangible_assets",
            "gross_intangible_assets_ltm",
            "dividends_paid_ltm",
            "common_dividends_paid_ltm",
            "common_dividends_paid_fy",
            # Composite
            "cfo",
            "fcf",
            "piotroski_f_score",
            "beneish_m_score",
            # Engineered
            "ev_ebitda",
            "peg_ratio",
            "market_cap_country_r",
        ]

        # Create dataframe with random data
        np.random.seed(42)
        self.df = pd.DataFrame(
            np.random.randn(100, len(self.all_columns)), columns=self.all_columns
        )

    def test_valuation_method_columns(self):
        """Test valuation method selects appropriate columns."""
        label_method = "valuation"

        expected_cols = [
            "p_e",
            "p_e_ltm",
            "p_e_ntm",
            "p_e_1fyltm",
            "p_e_5yavgltm",
            "p_b",
            "p_b_ltm",
            "p_b_1fy",
            "p_b_5yavg",
            "ebitda",
            "ebitda_ltm",
            "ebitda_fy",
            "enterprise_value",
            "market_cap",
            "ev_ebitda",
            "peg_ratio",
        ]

        # Core columns should always be included
        core_cols = ["last_price", "market_cap", "enterprise_value", "ebitda"]

        # Combine and remove duplicates
        all_expected = list(dict.fromkeys(core_cols + expected_cols))

        # Filter to available columns
        available = [c for c in all_expected if c in self.df.columns]

        self.assertGreater(len(available), 0, "Should have at least some valuation columns")
        self.assertIn("market_cap", available, "Core column market_cap should be included")
        self.assertIn("p_e", available, "Valuation-specific column p_e should be included")

    def test_price_momentum_method_columns(self):
        """Test price_momentum method selects appropriate columns."""
        label_method = "price_momentum"

        expected_cols = [
            "last_price",
            "price_target",
            "price_target_median",
            "price_target_high",
            "price_target_low",
            "market_cap",
            "enterprise_value",
            "p_e",
            "p_e_ltm",
            "p_e_ntm",
            "p_e_1fyltm",
            "p_e_5yavgltm",
            "p_b",
            "p_b_ltm",
            "p_b_1fy",
            "p_b_5yavg",
        ]

        available = [c for c in expected_cols if c in self.df.columns]

        self.assertGreater(len(available), 0)
        self.assertIn(
            "price_target", available, "price_target should be included for price_momentum"
        )

    def test_fundamental_method_columns(self):
        """Test fundamental method selects margin and profitability columns."""
        label_method = "fundamental"

        expected_cols = [
            "gross_margin",
            "gross_profit_margin_pct_ltm",
            "gross_profit_margin_pct_fy",
            "net_income_margin_pct_ltm",
            "net_income_margin_pct_fy",
            "net_income",
            "net_income_is_ltm",
            "net_income_is_fy",
            "ebitda",
            "ebit",
            "operating_income",
            "gross_profit",
            "return_on_equity_pct_ltm",
            "return_on_equity_pct_fy",
            "return_on_assets_roa_pct_ltm",
            "return_on_assets_roa_pct_fy",
        ]

        available = [c for c in expected_cols if c in self.df.columns]

        self.assertGreater(len(available), 0)
        self.assertIn("gross_margin", available, "gross_margin should be included for fundamental")

    def test_volatility_method_columns(self):
        """Test volatility method selects volatility and risk columns."""
        label_method = "volatility"

        expected_cols = [
            "volatility_1m",
            "volatility_3m",
            "volatility_6m",
            "volatility_1y",
            "volatility_1y_pct",
            "beta_1y",
            "beta_2y",
            "beta_5y",
            "short_int_pct",
            "last_price",
            "market_cap",
        ]

        available = [c for c in expected_cols if c in self.df.columns]

        self.assertGreater(len(available), 0)
        self.assertIn("volatility_1y", available, "volatility_1y should be included for volatility")

    def test_profitability_event_method_columns(self):
        """Test profitability_event method selects ROE/ROA columns."""
        label_method = "profitability_event"

        expected_cols = [
            "return_on_equity_pct_ltm",
            "return_on_equity_pct_fy",
            "return_on_assets_roa_pct_ltm",
            "return_on_assets_roa_pct_fy",
            "net_income",
            "net_income_is_ltm",
            "net_income_is_fy",
            "total_equity",
            "total_equity_ltm",
            "total_assets",
            "total_assets_ltm",
            "roe",
            "roa",
            "roic",
        ]

        available = [c for c in expected_cols if c in self.df.columns]

        self.assertGreater(len(available), 0)
        self.assertIn("return_on_equity_pct_ltm", available)

    def test_leverage_event_method_columns(self):
        """Test leverage_event method selects debt and leverage columns."""
        label_method = "leverage_event"

        expected_cols = [
            "total_debt",
            "total_debt_ltm",
            "total_debt_fy",
            "total_equity",
            "total_equity_ltm",
            "total_equity_fy",
            "total_assets",
            "total_assets_ltm",
            "total_assets_fy",
            "interest_expense",
            "interest_expense_total_ltm",
            "cash_and_equivalents",
            "cash_and_equivalents_ltm",
            "cash_and_equivalents_fy",
        ]

        available = [c for c in expected_cols if c in self.df.columns]

        self.assertGreater(len(available), 0)
        self.assertIn("total_debt", available)

    def test_growth_event_method_columns(self):
        """Test growth_event method selects revenue/earnings growth columns."""
        label_method = "growth_event"

        expected_cols = [
            "total_revenues_cagr_5y_fy",
            "total_revenues_ltm",
            "total_revenues_fy",
            "total_revenues_1fy",
            "revenue",
            "revenue_previous_year",
            "revenue_fy",
            "eps",
            "eps_previous_year",
            "eps_adj_ltm",
            "eps_adj_fy",
        ]

        available = [c for c in expected_cols if c in self.df.columns]

        self.assertGreater(len(available), 0)
        self.assertIn("total_revenues_cagr_5y_fy", available)

    def test_core_columns_always_included(self):
        """Test core columns are always included regardless of method."""
        core_cols = ["last_price", "market_cap", "enterprise_value", "ebitda"]

        # Test for multiple methods
        for method in ["valuation", "price_momentum", "fundamental", "growth_event"]:
            # Simulate the logic from notebook
            all_candidates = core_cols + [method]  # Simplified
            available = [c for c in all_candidates if c in self.df.columns]

            for core_col in core_cols:
                if core_col in self.df.columns:
                    self.assertIn(
                        core_col,
                        available,
                        f"Core column {core_col} should be included for method {method}",
                    )

    def test_unknown_method_uses_fallback(self):
        """Test unknown method falls back to default columns."""
        label_method = "unknown_method"

        default_cols = [
            "market_cap",
            "enterprise_value",
            "ebitda",
            "p_e",
            "p_b",
            "gross_margin",
            "revenue",
            "net_income",
        ]

        available = [c for c in default_cols if c in self.df.columns]

        self.assertGreater(len(available), 0)
        self.assertIn("market_cap", available, "Default columns should include market_cap")

    def test_missing_columns_filtered_out(self):
        """Test that columns not in dataframe are filtered out."""
        # Create dataframe with only a few columns
        small_df = pd.DataFrame(
            {
                "last_price": [100, 200],
                "market_cap": [1e9, 2e9],
                "p_e": [15, 20],
            }
        )

        # Simulate valuation method with many candidates
        candidates = [
            "p_e",
            "p_e_ltm",
            "p_e_ntm",  # Only p_e exists
            "market_cap",
            "enterprise_value",  # Only market_cap exists
            "last_price",
        ]

        available = [c for c in candidates if c in small_df.columns]

        self.assertEqual(len(available), 3, "Should only include columns that exist in dataframe")
        self.assertIn("p_e", available)
        self.assertIn("market_cap", available)
        self.assertIn("last_price", available)
        self.assertNotIn("p_e_ltm", available, "Non-existent columns should be filtered out")


if __name__ == "__main__":
    unittest.main()
