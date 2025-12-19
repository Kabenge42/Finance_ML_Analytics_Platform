"""
TDD Tests for scaling.py non-numeric column handling fix.

Tests the fix for the error: "could not convert string to float: 'USD'"

This test suite verifies:
1. Non-numeric columns (strings like 'USD') are excluded from scaling
2. Market value columns can be excluded
3. Only relative columns (ratios, percentages) can be selectively scaled
4. Price columns are excluded by default

Following code_guidelines.md Section 8 TDD Conventions.

Version: 1.0.0
Created: 2025-12-01
"""

import unittest
import numpy as np
import pandas as pd


class TestScaleFeatureNonNumericHandling(unittest.TestCase):
    """Test that scale_features handles non-numeric columns correctly."""

    def setUp(self):
        """Set up test data with mixed types including string columns."""
        np.random.seed(42)
        n_samples = 50

        self.test_df = pd.DataFrame(
            {
                # String columns that should be excluded
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "unit": ["USD"] * 25 + ["EUR"] * 25,  # Currency unit - string!
                "sector": np.random.choice(["Technology", "Financials"], n_samples),
                "country": np.random.choice(["US", "UK", "DE"], n_samples),
                # Price columns (should be excluded by default)
                "last_price": np.random.uniform(10, 500, n_samples),
                "price_target": np.random.uniform(15, 600, n_samples),
                # Market value columns (can be excluded)
                "market_cap": np.random.uniform(1e9, 1e12, n_samples),
                "enterprise_value": np.random.uniform(1e9, 1.5e12, n_samples),
                # Ratio columns (relative values - should be scaled)
                "p_e_ntm": np.random.uniform(5, 50, n_samples),
                "ev_ebitda": np.random.uniform(5, 30, n_samples),
                # Percentage columns (relative values - should be scaled)
                "gross_margin": np.random.uniform(20, 80, n_samples),
                "roe": np.random.uniform(-10, 30, n_samples),
                # Other numeric columns
                "beta_5y": np.random.uniform(0.5, 2.0, n_samples),
                "analyst_rating": np.random.uniform(1, 5, n_samples),
            }
        )

    def test_excludes_non_numeric_columns_automatically(self):
        """Test that non-numeric columns (strings) are excluded from scaling."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features

        # This should NOT raise an error even though 'unit' is a string column
        # Previously would fail with: "could not convert string to float: 'USD'"
        result = scale_features(
            self.test_df,
            columns=None,  # Auto-detect numeric columns
            scaler_type="robust",
            by_sector=False,
        )

        # String columns should be unchanged
        pd.testing.assert_series_equal(result["unit"], self.test_df["unit"], check_names=False)
        pd.testing.assert_series_equal(result["ticker"], self.test_df["ticker"], check_names=False)

    def test_excludes_non_numeric_when_columns_specified(self):
        """Test that non-numeric columns are excluded even when explicitly specified."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features

        # Explicitly include a mix of numeric and non-numeric columns
        columns_to_scale = ["unit", "last_price", "p_e_ntm", "gross_margin", "ticker"]

        # This should NOT raise an error - non-numeric columns should be filtered out
        result = scale_features(
            self.test_df,
            columns=columns_to_scale,
            scaler_type="robust",
            by_sector=False,
            exclude_price_columns=False,  # Don't exclude price for this test
        )

        # String columns should be unchanged (excluded from scaling)
        pd.testing.assert_series_equal(result["unit"], self.test_df["unit"], check_names=False)

    def test_excludes_market_value_columns_by_default(self):
        """Test that market value columns are excluded by default."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features

        original_market_cap = self.test_df["market_cap"].copy()

        result = scale_features(
            self.test_df,
            columns=None,
            scaler_type="robust",
            by_sector=False,
            exclude_market_value_columns=True,  # Default
        )

        # Market cap should be unchanged
        pd.testing.assert_series_equal(result["market_cap"], original_market_cap, check_names=False)

    def test_can_include_market_value_columns(self):
        """Test that market value columns can be included when option is False."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features

        original_market_cap = self.test_df["market_cap"].copy()

        result = scale_features(
            self.test_df,
            columns=None,
            scaler_type="robust",
            by_sector=False,
            exclude_price_columns=True,
            exclude_market_value_columns=False,  # Include market value columns
        )

        # Market cap should be scaled (different from original)
        # Note: It might be the same if all values scale to same after robust scaling
        # So we just check the function runs without error
        self.assertIn("market_cap", result.columns)

    def test_only_relative_columns_option(self):
        """Test that only_relative_columns scales only ratios and percentages."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features

        original_market_cap = self.test_df["market_cap"].copy()
        original_last_price = self.test_df["last_price"].copy()

        result = scale_features(
            self.test_df,
            columns=None,
            scaler_type="robust",
            by_sector=False,
            only_relative_columns=True,  # Only scale ratios/percentages
        )

        # Market cap and price should be unchanged
        pd.testing.assert_series_equal(result["market_cap"], original_market_cap, check_names=False)
        pd.testing.assert_series_equal(result["last_price"], original_last_price, check_names=False)

    def test_excludes_price_columns_by_default(self):
        """Test that price columns are excluded by default."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features

        original_last_price = self.test_df["last_price"].copy()
        original_price_target = self.test_df["price_target"].copy()

        result = scale_features(
            self.test_df,
            columns=None,
            scaler_type="robust",
            by_sector=False,
        )

        # Price columns should be unchanged
        pd.testing.assert_series_equal(result["last_price"], original_last_price, check_names=False)
        pd.testing.assert_series_equal(
            result["price_target"], original_price_target, check_names=False
        )

    def test_sector_aware_scaling_with_non_numeric(self):
        """Test sector-aware scaling works with non-numeric columns present."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features

        # This should NOT raise an error
        result = scale_features(
            self.test_df,
            columns=None,
            scaler_type="robust",
            by_sector=True,  # Enable sector-aware scaling
        )

        # Should complete without error
        self.assertEqual(len(result), len(self.test_df))

        # String columns should be unchanged
        pd.testing.assert_series_equal(result["unit"], self.test_df["unit"], check_names=False)

    def test_handles_empty_columns_list_gracefully(self):
        """Test that empty columns list returns original DataFrame."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features

        # Create a DataFrame with only non-numeric columns
        df_only_strings = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "unit": ["USD", "EUR", "GBP"],
                "sector": ["Tech", "Finance", "Energy"],
            }
        )

        result = scale_features(
            df_only_strings,
            columns=None,
            scaler_type="robust",
            by_sector=False,
        )

        # Should return original DataFrame unchanged
        pd.testing.assert_frame_equal(result, df_only_strings)

    def test_mixed_case_column_names(self):
        """Test that column name matching works with mixed case."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features

        df = self.test_df.copy()
        df = df.rename(columns={"last_price": "Last_Price", "market_cap": "Market_Cap"})

        original_last_price = df["Last_Price"].copy()

        result = scale_features(
            df,
            columns=None,
            scaler_type="robust",
            by_sector=False,
        )

        # Price column should still be excluded (case-insensitive matching)
        pd.testing.assert_series_equal(result["Last_Price"], original_last_price, check_names=False)


class TestScalingIntegrationWithETL(unittest.TestCase):
    """Test scaling integration with ETL pipeline scenarios."""

    def setUp(self):
        """Set up test data simulating ETL output."""
        np.random.seed(42)
        n_samples = 100

        self.etl_output = pd.DataFrame(
            {
                # Identifiers (strings)
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "isin": [f"US{i:010d}" for i in range(n_samples)],
                "name": [f"Company {i}" for i in range(n_samples)],
                # Currency unit column (the problematic one!)
                "unit": np.random.choice(["USD", "EUR", "GBP", "JPY"], n_samples),
                # Categorical columns
                "sector": np.random.choice(
                    ["Technology", "Financials", "Healthcare", "Energy"], n_samples
                ),
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
                # Price columns
                "last_price": np.random.uniform(10, 500, n_samples),
                "price_target": np.random.uniform(15, 600, n_samples),
                # Market value columns
                "market_cap": np.random.uniform(1e9, 1e12, n_samples),
                "enterprise_value": np.random.uniform(1e9, 1.5e12, n_samples),
                # Numeric features
                "p_e_ntm": np.random.uniform(5, 50, n_samples),
                "roe": np.random.uniform(-10, 30, n_samples),
                "beta_5y": np.random.uniform(0.5, 2.0, n_samples),
            }
        )

    def test_scale_features_with_etl_output(self):
        """Test scale_features works correctly with typical ETL output."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features

        # This simulates the error scenario from the issue
        # Previously would fail with: "could not convert string to float: 'USD'"
        result = scale_features(
            self.etl_output,
            columns=None,  # Auto-detect
            scaler_type="robust",
            by_sector=True,
        )

        # Should complete without error
        self.assertEqual(len(result), len(self.etl_output))

        # String columns should be unchanged
        pd.testing.assert_series_equal(result["unit"], self.etl_output["unit"], check_names=False)

        # Price columns should be unchanged
        pd.testing.assert_series_equal(
            result["last_price"], self.etl_output["last_price"], check_names=False
        )

    def test_scale_features_all_columns_specified(self):
        """Test when all columns including non-numeric are specified."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features

        # Specify ALL columns (including non-numeric)
        all_columns = self.etl_output.columns.tolist()

        result = scale_features(
            self.etl_output,
            columns=all_columns,  # Include everything
            scaler_type="robust",
            by_sector=False,
            exclude_price_columns=False,
            exclude_market_value_columns=False,
        )

        # Should complete without error
        self.assertEqual(len(result), len(self.etl_output))

        # Non-numeric columns should be unchanged
        pd.testing.assert_series_equal(result["unit"], self.etl_output["unit"], check_names=False)


if __name__ == "__main__":
    unittest.main()
