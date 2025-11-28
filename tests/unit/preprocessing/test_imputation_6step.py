"""
Comprehensive test suite for Phase 9.1 Enhanced 6-Step Imputation Strategy.

This module tests the enhanced 6-step imputation pipeline:
- Step 1: Zero imputation (48 columns)
- Step 2: KNN imputation (148 columns)
- Step 3: Price imputation (5 columns)
- Step 4: Median imputation (remaining numeric columns)
- Step 5: Categorical imputation (NEW - string/object columns)
- Step 6: Datetime imputation and formatting (NEW - date columns)

Test coverage target: ≥80%

TDD Approach: These tests are written BEFORE implementation to drive development.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestCategoricalImputationConfig(unittest.TestCase):
    """Test suite for get_categorical_imputation_config()."""

    def test_config_returns_dict(self):
        """Config function should return a dictionary."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            get_categorical_imputation_config,
        )

        config = get_categorical_imputation_config()
        self.assertIsInstance(config, dict)

    def test_config_has_most_frequent_strategy(self):
        """Config should include most_frequent strategy for classification columns."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            get_categorical_imputation_config,
        )

        config = get_categorical_imputation_config()
        self.assertEqual(config["style_class"], "most_frequent")
        self.assertEqual(config["size_class"], "most_frequent")
        self.assertEqual(config["sector"], "most_frequent")

    def test_config_has_constant_strategy(self):
        """Config should include constant strategy for identifiers."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            get_categorical_imputation_config,
        )

        config = get_categorical_imputation_config()
        self.assertEqual(config["ticker"], ("constant", "N/A"))
        self.assertEqual(config["isin"], ("constant", "N/A"))
        self.assertEqual(config["flag"], ("constant", "Unknown"))


class TestCategoricalImputation(unittest.TestCase):
    """Test suite for apply_categorical_imputation()."""

    def setUp(self):
        """Create sample data with categorical missing values."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", np.nan, "GOOGL", np.nan],
                "sector": ["Technology", "Technology", np.nan, "Technology", np.nan],
                "flag": ["A", "B", np.nan, "A", np.nan],
                "style_class": ["Growth", np.nan, "Value", "Growth", np.nan],
                "numeric_col": [1, 2, 3, 4, 5],  # Should not be touched
            }
        )

    def test_most_frequent_strategy(self):
        """Most frequent strategy should impute with mode."""
        from finance_ml.ml_workflow.preprocessing.imputation import apply_categorical_imputation

        result = apply_categorical_imputation(self.df, columns=["sector"], strategy="most_frequent")
        # Technology is the mode, should fill NaN values
        self.assertEqual(result["sector"].isna().sum(), 0)
        self.assertEqual(result["sector"].value_counts()["Technology"], 5)

    def test_constant_strategy(self):
        """Constant strategy should impute with specified value."""
        from finance_ml.ml_workflow.preprocessing.imputation import apply_categorical_imputation

        result = apply_categorical_imputation(
            self.df, columns=["ticker"], strategy="constant", fill_value="N/A"
        )
        self.assertEqual(result["ticker"].isna().sum(), 0)
        self.assertIn("N/A", result["ticker"].values)

    def test_auto_detect_categorical_columns(self):
        """Should auto-detect object dtype columns when columns=None."""
        from finance_ml.ml_workflow.preprocessing.imputation import apply_categorical_imputation

        result = apply_categorical_imputation(
            self.df, columns=None, strategy="most_frequent"  # Auto-detect
        )
        # All categorical columns should have no NaN
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            self.assertEqual(result[col].isna().sum(), 0)

    def test_preserves_existing_values(self):
        """Should preserve existing non-NaN values."""
        from finance_ml.ml_workflow.preprocessing.imputation import apply_categorical_imputation

        result = apply_categorical_imputation(
            self.df, columns=["ticker"], strategy="constant", fill_value="MISSING"
        )
        # Original values should be preserved
        self.assertEqual(result.loc[0, "ticker"], "AAPL")
        self.assertEqual(result.loc[1, "ticker"], "MSFT")

    def test_handles_no_missing_values(self):
        """Should handle columns with no missing values gracefully."""
        from finance_ml.ml_workflow.preprocessing.imputation import apply_categorical_imputation

        df_no_missing = pd.DataFrame({"sector": ["Technology", "Finance", "Healthcare"]})
        result = apply_categorical_imputation(
            df_no_missing, columns=["sector"], strategy="most_frequent"
        )
        pd.testing.assert_frame_equal(result, df_no_missing)

    def test_handles_nonexistent_columns(self):
        """Should handle columns that don't exist in dataframe."""
        from finance_ml.ml_workflow.preprocessing.imputation import apply_categorical_imputation

        result = apply_categorical_imputation(
            self.df, columns=["nonexistent_col"], strategy="most_frequent"
        )
        # Should return dataframe unchanged
        self.assertEqual(result.shape, self.df.shape)

    def test_preserves_numeric_columns(self):
        """Should not modify numeric columns."""
        from finance_ml.ml_workflow.preprocessing.imputation import apply_categorical_imputation

        result = apply_categorical_imputation(self.df, strategy="most_frequent")
        # Numeric column should be unchanged
        pd.testing.assert_series_equal(result["numeric_col"], self.df["numeric_col"])


class TestDatetimeImputation(unittest.TestCase):
    """Test suite for apply_datetime_imputation_and_formatting()."""

    def setUp(self):
        """Create sample data with datetime missing values."""
        base_date = datetime(2024, 1, 1)
        self.df = pd.DataFrame(
            {
                "last_updated": [
                    base_date,
                    base_date + timedelta(days=1),
                    None,
                    base_date + timedelta(days=3),
                    None,
                ],
                "next_earnings": ["2024-02-01", None, "2024-02-15", None, "2024-03-01"],
                "numeric_col": [1, 2, 3, 4, 5],
            }
        )

    def test_forward_fill_strategy(self):
        """Forward fill should propagate previous valid date."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_datetime_imputation_and_formatting,
        )

        result = apply_datetime_imputation_and_formatting(
            self.df, date_columns=["last_updated"], strategy="forward_fill"
        )
        # Should have no missing values
        self.assertEqual(result["last_updated"].isna().sum(), 0)
        # Should be datetime type
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["last_updated"]))

    def test_converts_string_to_datetime(self):
        """Should convert string dates to datetime format."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_datetime_imputation_and_formatting,
        )

        result = apply_datetime_imputation_and_formatting(
            self.df, date_columns=["next_earnings"], strategy="forward_fill"
        )
        # Should be datetime type
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["next_earnings"]))
        # Should have no missing values
        self.assertEqual(result["next_earnings"].isna().sum(), 0)

    def test_median_strategy(self):
        """Median strategy should use median timestamp."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_datetime_imputation_and_formatting,
        )

        result = apply_datetime_imputation_and_formatting(
            self.df, date_columns=["last_updated"], strategy="median"
        )
        self.assertEqual(result["last_updated"].isna().sum(), 0)

    def test_constant_strategy(self):
        """Constant strategy should use reference date."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_datetime_imputation_and_formatting,
        )

        reference_date = pd.Timestamp("2024-06-01")
        result = apply_datetime_imputation_and_formatting(
            self.df,
            date_columns=["last_updated"],
            strategy="constant",
            reference_date=reference_date,
        )
        self.assertEqual(result["last_updated"].isna().sum(), 0)

    def test_auto_detect_date_columns(self):
        """Should auto-detect date columns when date_columns=None."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_datetime_imputation_and_formatting,
        )

        result = apply_datetime_imputation_and_formatting(
            self.df, date_columns=None, strategy="forward_fill"  # Auto-detect
        )
        # Both date columns should be processed
        self.assertEqual(result["last_updated"].isna().sum(), 0)
        self.assertEqual(result["next_earnings"].isna().sum(), 0)

    def test_preserves_numeric_columns(self):
        """Should not modify numeric columns."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_datetime_imputation_and_formatting,
        )

        result = apply_datetime_imputation_and_formatting(
            self.df, date_columns=["last_updated"], strategy="forward_fill"
        )
        pd.testing.assert_series_equal(result["numeric_col"], self.df["numeric_col"])


class TestEnhancedImputation6Step(unittest.TestCase):
    """Test suite for apply_enhanced_imputation_strategy_6step()."""

    def setUp(self):
        """Create comprehensive sample data with all data types."""
        np.random.seed(42)
        base_date = datetime(2024, 1, 1)

        self.df = pd.DataFrame(
            {
                # Identifiers
                "ticker": ["AAPL", "MSFT", np.nan, "GOOGL", "AMZN"] * 20,
                # Categorical
                "sector": ["Technology"] * 80 + [np.nan] * 20,
                "style_class": ["Growth", "Value", np.nan] * 33 + ["Growth"],
                # Dates
                "last_updated": [
                    base_date + timedelta(days=i) if i % 5 != 0 else None for i in range(100)
                ],
                # Step 1: Zero imputation
                "impairment_of_goodwill_fq": [np.nan] * 70 + [1000.0] * 30,
                # Step 2: KNN imputation
                "market_cap": [100 + i if i % 3 != 0 else np.nan for i in range(100)],
                # Step 3: Price imputation
                "last_price": [50.0 + i * 0.1 for i in range(100)],
                "price_target": [np.nan] * 60 + [55.0 + i * 0.1 for i in range(40)],
                # Step 4: Other numeric
                "other_metric": [np.nan] * 50 + list(range(50, 100)),
            }
        )

    def test_6step_eliminates_all_missing_values(self):
        """6-step strategy should eliminate ALL missing values."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_6step,
        )

        result = apply_enhanced_imputation_strategy_6step(
            self.df, sector_column="sector", handle_categoricals=True, handle_dates=True
        )
        # Verify ZERO missing values across all columns
        total_missing = result.isna().sum().sum()
        self.assertEqual(
            total_missing, 0, f"Found {total_missing} missing values after 6-step imputation"
        )

    def test_6step_handles_numeric_columns(self):
        """6-step should properly impute numeric columns (Steps 1-4)."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_6step,
        )

        result = apply_enhanced_imputation_strategy_6step(self.df)

        # Check numeric columns have no missing
        numeric_missing = result.select_dtypes(include=[np.number]).isna().sum().sum()
        self.assertEqual(numeric_missing, 0)

    def test_6step_handles_categorical_columns(self):
        """6-step should properly impute categorical columns (Step 5)."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_6step,
        )

        result = apply_enhanced_imputation_strategy_6step(self.df, handle_categoricals=True)

        # Check categorical columns have no missing
        self.assertEqual(result["ticker"].isna().sum(), 0)
        self.assertEqual(result["sector"].isna().sum(), 0)
        self.assertEqual(result["style_class"].isna().sum(), 0)

    def test_6step_handles_datetime_columns(self):
        """6-step should properly impute and format datetime columns (Step 6)."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_6step,
        )

        result = apply_enhanced_imputation_strategy_6step(self.df, handle_dates=True)

        # Check datetime column has no missing and is properly typed
        self.assertEqual(result["last_updated"].isna().sum(), 0)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["last_updated"]))

    def test_6step_respects_handle_categoricals_flag(self):
        """Should skip categorical imputation when handle_categoricals=False."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_6step,
        )

        result = apply_enhanced_imputation_strategy_6step(
            self.df, handle_categoricals=False, handle_dates=False
        )

        # Categorical missing values should remain
        self.assertGreater(result["ticker"].isna().sum(), 0)

    def test_6step_respects_handle_dates_flag(self):
        """Should skip datetime imputation when handle_dates=False."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_6step,
        )

        result = apply_enhanced_imputation_strategy_6step(
            self.df, handle_dates=False, handle_categoricals=False
        )

        # Date missing values should remain
        self.assertGreater(result["last_updated"].isna().sum(), 0)

    def test_6step_with_custom_strategies(self):
        """Should accept custom strategies for categorical and datetime."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_6step,
        )

        result = apply_enhanced_imputation_strategy_6step(
            self.df,
            handle_categoricals=True,
            handle_dates=True,
            categorical_strategy="constant",
            date_strategy="median",
        )

        # Should still eliminate all missing values
        self.assertEqual(result.isna().sum().sum(), 0)


class TestValidateImputationCompleteness(unittest.TestCase):
    """Test suite for validate_imputation_completeness()."""

    def test_validates_complete_dataframe(self):
        """Should return is_complete=True for fully imputed data."""
        from finance_ml.ml_workflow.preprocessing.imputation import validate_imputation_completeness

        df_complete = pd.DataFrame(
            {
                "last_price": [1.0, 2.0, 3.0],
                "sector": ["Tech", "Finance", "Healthcare"],
                "last_updated": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            }
        )

        result = validate_imputation_completeness(df_complete)
        self.assertTrue(result["is_complete"])
        self.assertEqual(result["missing_count"], 0)
        self.assertTrue(result["ready_for_temporal_features"])

    def test_detects_missing_values(self):
        """Should detect missing values and return is_complete=False."""
        from finance_ml.ml_workflow.preprocessing.imputation import validate_imputation_completeness

        df_incomplete = pd.DataFrame(
            {"last_price": [1.0, np.nan, 3.0], "sector": ["Tech", None, "Healthcare"]}
        )

        result = validate_imputation_completeness(df_incomplete)
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["missing_count"], 2)

    def test_checks_datetime_formatting(self):
        """Should check that critical date columns are datetime typed."""
        from finance_ml.ml_workflow.preprocessing.imputation import validate_imputation_completeness

        df = pd.DataFrame(
            {
                "last_updated": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "income_statement_report_date": [
                    "2024-01-01",
                    "2024-01-02",
                ],  # String, not datetime
                "next_earnings": pd.to_datetime(["2024-02-01", "2024-02-02"]),
            }
        )

        result = validate_imputation_completeness(df)
        self.assertTrue(result["datetime_formatted"]["last_updated"]["is_datetime"])
        self.assertFalse(
            result["datetime_formatted"]["income_statement_report_date"]["is_datetime"]
        )

    def test_reports_missing_by_type(self):
        """Should categorize missing values by data type."""
        from finance_ml.ml_workflow.preprocessing.imputation import validate_imputation_completeness

        df = pd.DataFrame({"numeric_col": [1.0, np.nan, 3.0], "categorical_col": ["A", None, "C"]})

        result = validate_imputation_completeness(df)
        self.assertEqual(result["missing_by_type"]["numeric"], 1)
        self.assertEqual(result["missing_by_type"]["categorical"], 1)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility for 6-step function."""

    def setUp(self):
        """Create sample data."""
        self.df = pd.DataFrame(
            {
                "sector": ["Technology"] * 50,
                "last_price": [50.0] * 50,
                "impairment_of_goodwill_fq": [np.nan] * 30 + [1000.0] * 20,
                "market_cap": [100 + i if i % 3 != 0 else np.nan for i in range(50)],
                "price_target": [np.nan] * 30 + [55.0] * 20,
                "other_metric": [np.nan] * 25 + list(range(25, 50)),
            }
        )

    def test_4step_wrapper_calls_6step(self):
        """6-step function should be a wrapper calling 6-step."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_4step,
        )

        result = apply_enhanced_imputation_strategy_4step(
            self.df, sector_column="sector", n_neighbors=5, price_column="last_price"
        )

        # Should work like before - eliminate numeric missing values
        numeric_missing = result.select_dtypes(include=[np.number]).isna().sum().sum()
        self.assertEqual(numeric_missing, 0)

    def test_4step_maintains_signature(self):
        """6-step function should maintain original signature."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_4step,
        )
        import inspect

        sig = inspect.signature(apply_enhanced_imputation_strategy_4step)
        params = list(sig.parameters.keys())

        # Should have original parameters
        self.assertIn("df", params)
        self.assertIn("sector_column", params)
        self.assertIn("n_neighbors", params)
        self.assertIn("price_column", params)


if __name__ == "__main__":
    unittest.main()
