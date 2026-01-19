"""
Test suite for CalcFeatureRegistry.sql feature registry enhancements.

This module tests that all required SQL functions and metadata entries
are present in CalcFeatureRegistry.sql as specified in:
docs/improvement_plan/feature_registry_enhancements.md

TDD Approach:
- RED: Tests written first to define expected behavior
- GREEN: Minimal implementation to pass tests
- REFACTOR: Clean up and optimize

Coverage target: ≥80% for changed files
"""

import re
import unittest
from pathlib import Path


class TestFeatureRegistryFunctions(unittest.TestCase):
    """Test suite for verifying SQL function definitions in CalcFeatureRegistry.sql."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "CalcFeatureRegistry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def _function_exists(self, function_name: str) -> bool:
        """Check if a CREATE OR REPLACE FUNCTION statement exists for the given function."""
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\("
        return bool(re.search(pattern, self.sql_content, re.IGNORECASE))

    def _function_returns_table(self, function_name: str) -> bool:
        """Check if the function returns a TABLE type."""
        # Pattern: CREATE OR REPLACE FUNCTION name() RETURNS TABLE
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\(\s*\)\s*RETURNS\s+TABLE"
        return bool(re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL))

    def _function_has_column(self, function_name: str, column_name: str) -> bool:
        """Check if a function's RETURNS TABLE includes a specific column."""
        # Find the function definition and check for the column in its RETURNS TABLE
        func_pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\(\s*\)\s*RETURNS\s+TABLE\s*\((.*?)\)\s*AS"
        match = re.search(func_pattern, self.sql_content, re.IGNORECASE | re.DOTALL)
        if match:
            table_def = match.group(1)
            return column_name.lower() in table_def.lower()
        return False

    # =========================================================================
    # TEST: calc_interest_income_features (10 features)
    # =========================================================================
    def test_calc_interest_income_features_exists(self):
        """Test that calc_interest_income_features function is defined."""
        self.assertTrue(
            self._function_exists("calc_interest_income_features"),
            "calc_interest_income_features function should be defined in CalcFeatureRegistry.sql",
        )

    def test_calc_interest_income_features_returns_table(self):
        """Test that calc_interest_income_features returns a TABLE."""
        self.assertTrue(
            self._function_returns_table("calc_interest_income_features"),
            "calc_interest_income_features should return a TABLE type",
        )

    def test_calc_interest_income_features_has_required_columns(self):
        """Test that calc_interest_income_features has all required columns."""
        required_columns = [
            "ticker",
            "net_interest_income",
            "interest_coverage_ebit",
            "interest_coverage_ebitda",
            "financial_income_quality",
            "interest_burden_ratio",
        ]
        for col in required_columns:
            self.assertTrue(
                self._function_has_column("calc_interest_income_features", col),
                f"calc_interest_income_features should have column '{col}'",
            )

    # =========================================================================
    # TEST: calc_long_term_momentum_features (10 features)
    # =========================================================================
    def test_calc_long_term_momentum_features_exists(self):
        """Test that calc_long_term_momentum_features function is defined."""
        self.assertTrue(
            self._function_exists("calc_long_term_momentum_features"),
            "calc_long_term_momentum_features function should be defined in CalcFeatureRegistry.sql",
        )

    def test_calc_long_term_momentum_features_returns_table(self):
        """Test that calc_long_term_momentum_features returns a TABLE."""
        self.assertTrue(
            self._function_returns_table("calc_long_term_momentum_features"),
            "calc_long_term_momentum_features should return a TABLE type",
        )

    def test_calc_long_term_momentum_features_has_required_columns(self):
        """Test that calc_long_term_momentum_features has all required columns."""
        required_columns = [
            "ticker",
            "price_momentum_3y",
            "price_momentum_5y",
            "long_term_trend_score",
            "momentum_consistency",
            "secular_trend_flag",
        ]
        for col in required_columns:
            self.assertTrue(
                self._function_has_column("calc_long_term_momentum_features", col),
                f"calc_long_term_momentum_features should have column '{col}'",
            )

    # =========================================================================
    # TEST: calc_tangible_book_features (10 features)
    # =========================================================================
    def test_calc_tangible_book_features_exists(self):
        """Test that calc_tangible_book_features function is defined."""
        self.assertTrue(
            self._function_exists("calc_tangible_book_features"),
            "calc_tangible_book_features function should be defined in CalcFeatureRegistry.sql",
        )

    def test_calc_tangible_book_features_returns_table(self):
        """Test that calc_tangible_book_features returns a TABLE."""
        self.assertTrue(
            self._function_returns_table("calc_tangible_book_features"),
            "calc_tangible_book_features should return a TABLE type",
        )

    def test_calc_tangible_book_features_has_required_columns(self):
        """Test that calc_tangible_book_features has all required columns."""
        required_columns = [
            "ticker",
            "price_to_tbv",
            "tbv_per_share",
            "tangible_equity_ratio",
            "tbv_margin_of_safety",
        ]
        for col in required_columns:
            self.assertTrue(
                self._function_has_column("calc_tangible_book_features", col),
                f"calc_tangible_book_features should have column '{col}'",
            )

    # =========================================================================
    # TEST: calc_beta_risk_features (10 features)
    # =========================================================================
    def test_calc_beta_risk_features_exists(self):
        """Test that calc_beta_risk_features function is defined."""
        self.assertTrue(
            self._function_exists("calc_beta_risk_features"),
            "calc_beta_risk_features function should be defined in CalcFeatureRegistry.sql",
        )

    def test_calc_beta_risk_features_returns_table(self):
        """Test that calc_beta_risk_features returns a TABLE."""
        self.assertTrue(
            self._function_returns_table("calc_beta_risk_features"),
            "calc_beta_risk_features should return a TABLE type",
        )

    def test_calc_beta_risk_features_has_required_columns(self):
        """Test that calc_beta_risk_features has all required columns."""
        required_columns = [
            "ticker",
            "beta_1y",
            "beta_2y",
            "beta_5y",
            "beta_trend_short",
            "beta_stability",
            "systematic_risk_score",
            "defensive_stock_flag",
            "high_beta_flag",
        ]
        for col in required_columns:
            self.assertTrue(
                self._function_has_column("calc_beta_risk_features", col),
                f"calc_beta_risk_features should have column '{col}'",
            )

    # =========================================================================
    # TEST: calc_working_capital_deep_features (12 features)
    # =========================================================================
    def test_calc_working_capital_deep_features_exists(self):
        """Test that calc_working_capital_deep_features function is defined."""
        self.assertTrue(
            self._function_exists("calc_working_capital_deep_features"),
            "calc_working_capital_deep_features function should be defined in CalcFeatureRegistry.sql",
        )

    def test_calc_working_capital_deep_features_returns_table(self):
        """Test that calc_working_capital_deep_features returns a TABLE."""
        self.assertTrue(
            self._function_returns_table("calc_working_capital_deep_features"),
            "calc_working_capital_deep_features should return a TABLE type",
        )

    def test_calc_working_capital_deep_features_has_required_columns(self):
        """Test that calc_working_capital_deep_features has all required columns."""
        required_columns = [
            "ticker",
            "net_working_capital",
            "current_ratio",
            "quick_ratio",
            "cash_ratio",
            "defensive_interval",
            "liquidity_score",
            "working_capital_efficiency",
        ]
        for col in required_columns:
            self.assertTrue(
                self._function_has_column("calc_working_capital_deep_features", col),
                f"calc_working_capital_deep_features should have column '{col}'",
            )

    # =========================================================================
    # TEST: calc_unusual_items_features (10 features)
    # =========================================================================
    def test_calc_unusual_items_features_exists(self):
        """Test that calc_unusual_items_features function is defined."""
        self.assertTrue(
            self._function_exists("calc_unusual_items_features"),
            "calc_unusual_items_features function should be defined in CalcFeatureRegistry.sql",
        )

    def test_calc_unusual_items_features_returns_table(self):
        """Test that calc_unusual_items_features returns a TABLE."""
        self.assertTrue(
            self._function_returns_table("calc_unusual_items_features"),
            "calc_unusual_items_features should return a TABLE type",
        )

    def test_calc_unusual_items_features_has_required_columns(self):
        """Test that calc_unusual_items_features has all required columns."""
        required_columns = [
            "ticker",
            "total_unusual_items",
            "unusual_to_net_income_ratio",
            "clean_earnings_flag",
            "earnings_noise_score",
            "quality_adjusted_ni",
        ]
        for col in required_columns:
            self.assertTrue(
                self._function_has_column("calc_unusual_items_features", col),
                f"calc_unusual_items_features should have column '{col}'",
            )

    # =========================================================================
    # TEST: calc_revenue_estimate_consensus (11 features)
    # =========================================================================
    def test_calc_revenue_estimate_consensus_exists(self):
        """Test that calc_revenue_estimate_consensus function is defined."""
        self.assertTrue(
            self._function_exists("calc_revenue_estimate_consensus"),
            "calc_revenue_estimate_consensus function should be defined in CalcFeatureRegistry.sql",
        )

    def test_calc_revenue_estimate_consensus_returns_table(self):
        """Test that calc_revenue_estimate_consensus returns a TABLE."""
        self.assertTrue(
            self._function_returns_table("calc_revenue_estimate_consensus"),
            "calc_revenue_estimate_consensus should return a TABLE type",
        )

    def test_calc_revenue_estimate_consensus_has_required_columns(self):
        """Test that calc_revenue_estimate_consensus has all required columns."""
        required_columns = [
            "ticker",
            "estimate_skew_ntm",
            "estimate_skew_fy1e",
            "consensus_confidence",
            "upside_to_consensus",
            "forward_revenue_growth",
        ]
        for col in required_columns:
            self.assertTrue(
                self._function_has_column("calc_revenue_estimate_consensus", col),
                f"calc_revenue_estimate_consensus should have column '{col}'",
            )

    # =========================================================================
    # TEST: calc_enhanced_valuation_ratios (12 features)
    # =========================================================================
    def test_calc_enhanced_valuation_ratios_exists(self):
        """Test that calc_enhanced_valuation_ratios function is defined."""
        self.assertTrue(
            self._function_exists("calc_enhanced_valuation_ratios"),
            "calc_enhanced_valuation_ratios function should be defined in CalcFeatureRegistry.sql",
        )

    def test_calc_enhanced_valuation_ratios_returns_table(self):
        """Test that calc_enhanced_valuation_ratios returns a TABLE."""
        self.assertTrue(
            self._function_returns_table("calc_enhanced_valuation_ratios"),
            "calc_enhanced_valuation_ratios should return a TABLE type",
        )

    def test_calc_enhanced_valuation_ratios_has_required_columns(self):
        """Test that calc_enhanced_valuation_ratios has all required columns."""
        required_columns = [
            "ticker",
            "forward_pe",
            "trailing_pe",
            "pe_forward_discount",
            "peg_ratio",
            "earnings_yield",
            "fcf_yield",
            "valuation_composite_score",
        ]
        for col in required_columns:
            self.assertTrue(
                self._function_has_column("calc_enhanced_valuation_ratios", col),
                f"calc_enhanced_valuation_ratios should have column '{col}'",
            )

    # =========================================================================
    # TEST: calc_cost_structure_features (12 features)
    # =========================================================================
    def test_calc_cost_structure_features_exists(self):
        """Test that calc_cost_structure_features function is defined."""
        self.assertTrue(
            self._function_exists("calc_cost_structure_features"),
            "calc_cost_structure_features function should be defined in CalcFeatureRegistry.sql",
        )

    def test_calc_cost_structure_features_returns_table(self):
        """Test that calc_cost_structure_features returns a TABLE."""
        self.assertTrue(
            self._function_returns_table("calc_cost_structure_features"),
            "calc_cost_structure_features should return a TABLE type",
        )

    def test_calc_cost_structure_features_has_required_columns(self):
        """Test that calc_cost_structure_features has all required columns."""
        required_columns = [
            "ticker",
            "sga_to_revenue_fy",
            "sga_trend_yoy",
            "operating_expense_ratio",
            "cost_of_revenue_ratio",
            "operating_leverage_score",
        ]
        for col in required_columns:
            self.assertTrue(
                self._function_has_column("calc_cost_structure_features", col),
                f"calc_cost_structure_features should have column '{col}'",
            )

    # =========================================================================
    # TEST: calc_revenue_quarterly_features (12 features)
    # =========================================================================
    def test_calc_revenue_quarterly_features_exists(self):
        """Test that calc_revenue_quarterly_features function is defined."""
        self.assertTrue(
            self._function_exists("calc_revenue_quarterly_features"),
            "calc_revenue_quarterly_features function should be defined in CalcFeatureRegistry.sql",
        )

    def test_calc_revenue_quarterly_features_returns_table(self):
        """Test that calc_revenue_quarterly_features returns a TABLE."""
        self.assertTrue(
            self._function_returns_table("calc_revenue_quarterly_features"),
            "calc_revenue_quarterly_features should return a TABLE type",
        )

    def test_calc_revenue_quarterly_features_has_required_columns(self):
        """Test that calc_revenue_quarterly_features has all required columns."""
        required_columns = [
            "ticker",
            "revenue_fq",
            "revenue_ltm",
            "revenue_fq_vs_5yavg",
            "revenue_qoq_growth",
            "revenue_yoy_growth",
            "revenue_seasonality_factor",
        ]
        for col in required_columns:
            self.assertTrue(
                self._function_has_column("calc_revenue_quarterly_features", col),
                f"calc_revenue_quarterly_features should have column '{col}'",
            )

    # =========================================================================
    # TEST: calc_all_enhanced_features (53 features - composite)
    # =========================================================================
    def test_calc_all_enhanced_features_exists(self):
        """Test that calc_all_enhanced_features function is defined."""
        self.assertTrue(
            self._function_exists("calc_all_enhanced_features"),
            "calc_all_enhanced_features function should be defined in CalcFeatureRegistry.sql",
        )

    def test_calc_all_enhanced_features_returns_table(self):
        """Test that calc_all_enhanced_features returns a TABLE."""
        self.assertTrue(
            self._function_returns_table("calc_all_enhanced_features"),
            "calc_all_enhanced_features should return a TABLE type",
        )

    def test_calc_all_enhanced_features_has_identity_columns(self):
        """Test that calc_all_enhanced_features has identity columns."""
        required_columns = ["ticker", "name", "sector", "industry"]
        for col in required_columns:
            self.assertTrue(
                self._function_has_column("calc_all_enhanced_features", col),
                f"calc_all_enhanced_features should have identity column '{col}'",
            )

    def test_calc_all_enhanced_features_has_key_feature_columns(self):
        """Test that calc_all_enhanced_features has key feature columns from all categories."""
        # Sample columns from different feature categories
        key_columns = [
            "revenue_fq_vs_5yavg",  # Revenue Quarterly
            "sga_to_revenue_fy",  # Cost Structure
            "price_to_tbv",  # Tangible Book
            "net_interest_income",  # Interest Income
            "price_momentum_3y",  # Long-Term Momentum
            "beta_stability",  # Beta Risk
            "consensus_confidence",  # Revenue Estimate (as revenue_consensus_confidence)
            "total_unusual_items",  # Unusual Items
            "earnings_yield",  # Enhanced Valuation
            "liquidity_score",  # Working Capital
        ]
        for col in key_columns:
            # Handle the renamed column
            col_to_check = (
                "revenue_consensus_confidence" if col == "consensus_confidence" else col
            )
            self.assertTrue(
                self._function_has_column("calc_all_enhanced_features", col_to_check),
                f"calc_all_enhanced_features should have feature column '{col_to_check}'",
            )


class TestFeatureRegistryMetadata(unittest.TestCase):
    """Test suite for verifying metadata entries in feature_registry_metadata table."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "CalcFeatureRegistry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def _metadata_entry_exists(self, function_name: str) -> bool:
        """Check if a metadata entry exists for the given function in the INSERT statement."""
        # Pattern to find the function name in the VALUES section of INSERT INTO feature_registry_metadata
        pattern = rf"\(\s*'{re.escape(function_name)}'\s*,"
        return bool(re.search(pattern, self.sql_content, re.IGNORECASE))

    def _metadata_has_category(self, function_name: str, category: str) -> bool:
        """Check if the metadata entry has the correct category."""
        # Pattern: ('function_name', 'category', ...
        pattern = rf"\(\s*'{re.escape(function_name)}'\s*,\s*'{re.escape(category)}'"
        return bool(re.search(pattern, self.sql_content, re.IGNORECASE))

    # =========================================================================
    # TEST: Metadata entries for all 11 new functions
    # =========================================================================
    def test_calc_interest_income_features_metadata_exists(self):
        """Test that calc_interest_income_features has a metadata entry."""
        self.assertTrue(
            self._metadata_entry_exists("calc_interest_income_features"),
            "calc_interest_income_features should have a metadata entry in feature_registry_metadata",
        )

    def test_calc_interest_income_features_metadata_category(self):
        """Test that calc_interest_income_features has correct category."""
        self.assertTrue(
            self._metadata_has_category(
                "calc_interest_income_features", "Interest Income"
            ),
            "calc_interest_income_features should have category 'Interest Income'",
        )

    def test_calc_long_term_momentum_features_metadata_exists(self):
        """Test that calc_long_term_momentum_features has a metadata entry."""
        self.assertTrue(
            self._metadata_entry_exists("calc_long_term_momentum_features"),
            "calc_long_term_momentum_features should have a metadata entry",
        )

    def test_calc_long_term_momentum_features_metadata_category(self):
        """Test that calc_long_term_momentum_features has correct category."""
        self.assertTrue(
            self._metadata_has_category(
                "calc_long_term_momentum_features", "Momentum & Technical"
            ),
            "calc_long_term_momentum_features should have category 'Momentum & Technical'",
        )

    def test_calc_tangible_book_features_metadata_exists(self):
        """Test that calc_tangible_book_features has a metadata entry."""
        self.assertTrue(
            self._metadata_entry_exists("calc_tangible_book_features"),
            "calc_tangible_book_features should have a metadata entry",
        )

    def test_calc_tangible_book_features_metadata_category(self):
        """Test that calc_tangible_book_features has correct category."""
        self.assertTrue(
            self._metadata_has_category(
                "calc_tangible_book_features", "Valuation Ratios"
            ),
            "calc_tangible_book_features should have category 'Valuation Ratios'",
        )

    def test_calc_beta_risk_features_metadata_exists(self):
        """Test that calc_beta_risk_features has a metadata entry."""
        self.assertTrue(
            self._metadata_entry_exists("calc_beta_risk_features"),
            "calc_beta_risk_features should have a metadata entry",
        )

    def test_calc_beta_risk_features_metadata_category(self):
        """Test that calc_beta_risk_features has correct category."""
        self.assertTrue(
            self._metadata_has_category("calc_beta_risk_features", "Quality & Risk"),
            "calc_beta_risk_features should have category 'Quality & Risk'",
        )

    def test_calc_working_capital_deep_features_metadata_exists(self):
        """Test that calc_working_capital_deep_features has a metadata entry."""
        self.assertTrue(
            self._metadata_entry_exists("calc_working_capital_deep_features"),
            "calc_working_capital_deep_features should have a metadata entry",
        )

    def test_calc_working_capital_deep_features_metadata_category(self):
        """Test that calc_working_capital_deep_features has correct category."""
        self.assertTrue(
            self._metadata_has_category(
                "calc_working_capital_deep_features", "Leverage & Liquidity"
            ),
            "calc_working_capital_deep_features should have category 'Leverage & Liquidity'",
        )

    def test_calc_unusual_items_features_metadata_exists(self):
        """Test that calc_unusual_items_features has a metadata entry."""
        self.assertTrue(
            self._metadata_entry_exists("calc_unusual_items_features"),
            "calc_unusual_items_features should have a metadata entry",
        )

    def test_calc_unusual_items_features_metadata_category(self):
        """Test that calc_unusual_items_features has correct category."""
        self.assertTrue(
            self._metadata_has_category(
                "calc_unusual_items_features", "Earnings Quality"
            ),
            "calc_unusual_items_features should have category 'Earnings Quality'",
        )

    def test_calc_revenue_estimate_consensus_metadata_exists(self):
        """Test that calc_revenue_estimate_consensus has a metadata entry."""
        self.assertTrue(
            self._metadata_entry_exists("calc_revenue_estimate_consensus"),
            "calc_revenue_estimate_consensus should have a metadata entry",
        )

    def test_calc_revenue_estimate_consensus_metadata_category(self):
        """Test that calc_revenue_estimate_consensus has correct category."""
        self.assertTrue(
            self._metadata_has_category(
                "calc_revenue_estimate_consensus", "Revenue Forecasting"
            ),
            "calc_revenue_estimate_consensus should have category 'Revenue Forecasting'",
        )

    def test_calc_enhanced_valuation_ratios_metadata_exists(self):
        """Test that calc_enhanced_valuation_ratios has a metadata entry."""
        self.assertTrue(
            self._metadata_entry_exists("calc_enhanced_valuation_ratios"),
            "calc_enhanced_valuation_ratios should have a metadata entry",
        )

    def test_calc_enhanced_valuation_ratios_metadata_category(self):
        """Test that calc_enhanced_valuation_ratios has correct category."""
        self.assertTrue(
            self._metadata_has_category(
                "calc_enhanced_valuation_ratios", "Valuation Ratios"
            ),
            "calc_enhanced_valuation_ratios should have category 'Valuation Ratios'",
        )

    def test_calc_cost_structure_features_metadata_exists(self):
        """Test that calc_cost_structure_features has a metadata entry."""
        self.assertTrue(
            self._metadata_entry_exists("calc_cost_structure_features"),
            "calc_cost_structure_features should have a metadata entry",
        )

    def test_calc_cost_structure_features_metadata_category(self):
        """Test that calc_cost_structure_features has correct category."""
        self.assertTrue(
            self._metadata_has_category(
                "calc_cost_structure_features", "Efficiency Ratios"
            ),
            "calc_cost_structure_features should have category 'Efficiency Ratios'",
        )

    def test_calc_revenue_quarterly_features_metadata_exists(self):
        """Test that calc_revenue_quarterly_features has a metadata entry."""
        self.assertTrue(
            self._metadata_entry_exists("calc_revenue_quarterly_features"),
            "calc_revenue_quarterly_features should have a metadata entry",
        )

    def test_calc_revenue_quarterly_features_metadata_category(self):
        """Test that calc_revenue_quarterly_features has correct category."""
        self.assertTrue(
            self._metadata_has_category(
                "calc_revenue_quarterly_features", "Revenue Forecasting"
            ),
            "calc_revenue_quarterly_features should have category 'Revenue Forecasting'",
        )

    def test_calc_all_enhanced_features_metadata_exists(self):
        """Test that calc_all_enhanced_features has a metadata entry."""
        self.assertTrue(
            self._metadata_entry_exists("calc_all_enhanced_features"),
            "calc_all_enhanced_features should have a metadata entry",
        )

    def test_calc_all_enhanced_features_metadata_category(self):
        """Test that calc_all_enhanced_features has correct category."""
        self.assertTrue(
            self._metadata_has_category("calc_all_enhanced_features", "Composite"),
            "calc_all_enhanced_features should have category 'Composite'",
        )


class TestFeatureRegistryViews(unittest.TestCase):
    """Test suite for verifying SQL views for function inlining."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "CalcFeatureRegistry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def _view_exists(self, view_name: str) -> bool:
        """Check if a CREATE OR REPLACE VIEW statement exists for the given view."""
        pattern = rf"CREATE\s+OR\s+REPLACE\s+VIEW\s+{re.escape(view_name)}\s+AS"
        return bool(re.search(pattern, self.sql_content, re.IGNORECASE))

    # =========================================================================
    # TEST: Views for new functions
    # =========================================================================
    def test_v_long_term_momentum_features_view_exists(self):
        """Test that v_long_term_momentum_features view is defined."""
        self.assertTrue(
            self._view_exists("v_long_term_momentum_features"),
            "v_long_term_momentum_features view should be defined",
        )

    def test_v_beta_risk_features_view_exists(self):
        """Test that v_beta_risk_features view is defined."""
        self.assertTrue(
            self._view_exists("v_beta_risk_features"),
            "v_beta_risk_features view should be defined",
        )

    def test_v_interest_income_features_view_exists(self):
        """Test that v_interest_income_features view is defined."""
        self.assertTrue(
            self._view_exists("v_interest_income_features"),
            "v_interest_income_features view should be defined",
        )

    def test_v_tangible_book_features_view_exists(self):
        """Test that v_tangible_book_features view is defined."""
        self.assertTrue(
            self._view_exists("v_tangible_book_features"),
            "v_tangible_book_features view should be defined",
        )

    def test_v_working_capital_deep_features_view_exists(self):
        """Test that v_working_capital_deep_features view is defined."""
        self.assertTrue(
            self._view_exists("v_working_capital_deep_features"),
            "v_working_capital_deep_features view should be defined",
        )

    def test_v_unusual_items_features_view_exists(self):
        """Test that v_unusual_items_features view is defined."""
        self.assertTrue(
            self._view_exists("v_unusual_items_features"),
            "v_unusual_items_features view should be defined",
        )

    def test_v_revenue_estimate_consensus_view_exists(self):
        """Test that v_revenue_estimate_consensus view is defined."""
        self.assertTrue(
            self._view_exists("v_revenue_estimate_consensus"),
            "v_revenue_estimate_consensus view should be defined",
        )

    def test_v_enhanced_valuation_ratios_view_exists(self):
        """Test that v_enhanced_valuation_ratios view is defined."""
        self.assertTrue(
            self._view_exists("v_enhanced_valuation_ratios"),
            "v_enhanced_valuation_ratios view should be defined",
        )

    def test_v_cost_structure_features_view_exists(self):
        """Test that v_cost_structure_features view is defined."""
        self.assertTrue(
            self._view_exists("v_cost_structure_features"),
            "v_cost_structure_features view should be defined",
        )

    def test_v_revenue_quarterly_features_view_exists(self):
        """Test that v_revenue_quarterly_features view is defined."""
        self.assertTrue(
            self._view_exists("v_revenue_quarterly_features"),
            "v_revenue_quarterly_features view should be defined",
        )


class TestFeatureRegistryIntegrity(unittest.TestCase):
    """Test suite for overall integrity of the feature registry."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "CalcFeatureRegistry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def test_all_new_functions_use_equities_table(self):
        """Test that all new functions query from postgres.public.equities."""
        new_functions = [
            "calc_interest_income_features",
            "calc_long_term_momentum_features",
            "calc_tangible_book_features",
            "calc_beta_risk_features",
            "calc_working_capital_deep_features",
            "calc_unusual_items_features",
            "calc_revenue_estimate_consensus",
            "calc_enhanced_valuation_ratios",
            "calc_cost_structure_features",
            "calc_revenue_quarterly_features",
            "calc_all_enhanced_features",
        ]

        for func_name in new_functions:
            # Find the function body and check for equities table reference
            func_pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(func_name)}.*?FROM\s+postgres\.public\.equities"
            self.assertTrue(
                bool(
                    re.search(func_pattern, self.sql_content, re.IGNORECASE | re.DOTALL)
                ),
                f"{func_name} should query from postgres.public.equities",
            )

    def test_sql_file_has_valid_structure(self):
        """Test that the SQL file has proper BEGIN/COMMIT transaction structure."""
        self.assertIn(
            "BEGIN;", self.sql_content, "SQL file should have BEGIN statement"
        )
        self.assertIn(
            "COMMIT;", self.sql_content, "SQL file should have COMMIT statement"
        )

    def test_metadata_table_creation_exists(self):
        """Test that feature_registry_metadata table creation exists."""
        pattern = r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+feature_registry_metadata"
        self.assertTrue(
            bool(re.search(pattern, self.sql_content, re.IGNORECASE)),
            "feature_registry_metadata table creation should exist",
        )

    def test_total_new_functions_count(self):
        """Test that all 11 new functions are defined."""
        new_functions = [
            "calc_interest_income_features",
            "calc_long_term_momentum_features",
            "calc_tangible_book_features",
            "calc_beta_risk_features",
            "calc_working_capital_deep_features",
            "calc_unusual_items_features",
            "calc_revenue_estimate_consensus",
            "calc_enhanced_valuation_ratios",
            "calc_cost_structure_features",
            "calc_revenue_quarterly_features",
            "calc_all_enhanced_features",
        ]

        found_count = 0
        for func_name in new_functions:
            pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(func_name)}\s*\("
            if re.search(pattern, self.sql_content, re.IGNORECASE):
                found_count += 1

        self.assertEqual(
            found_count, 11, f"Expected 11 new functions, found {found_count}"
        )


if __name__ == "__main__":
    unittest.main()
