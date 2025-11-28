"""
Test suite for notebook schema integration.

Tests the integration of schema module helper functions with notebook
preprocessing pipeline, following strict TDD principles.

Aligned with code_guidelines.md v1.3+ and TDD conventions.
"""

import unittest
from typing import List

from finance_ml.ml_workflow.data.schema import (
    list_categorical_cols,
    list_date_cols,
    list_numeric_feature_cols,
    get_expected_dtype,
    get_column_role,
    normalize_column_name,
    COLUMN_SCHEMA,
)


class TestSchemaHelperFunctions(unittest.TestCase):
    """Test schema module helper functions."""

    def test_list_categorical_cols_returns_list(self):
        """Test that list_categorical_cols() returns a list."""
        result = list_categorical_cols()
        self.assertIsInstance(result, list)

    def test_list_categorical_cols_non_empty(self):
        """Test that list_categorical_cols() returns non-empty list."""
        result = list_categorical_cols()
        self.assertGreater(len(result), 0, "Should return at least one categorical column")

    def test_list_categorical_cols_contains_sector(self):
        """Test that list_categorical_cols() includes 'sector'."""
        result = list_categorical_cols()
        self.assertIn("sector", result, "Should include 'sector' as categorical column")

    def test_list_categorical_cols_contains_region(self):
        """Test that list_categorical_cols() includes 'region'."""
        result = list_categorical_cols()
        self.assertIn("region", result, "Should include 'region' as categorical column")

    def test_list_categorical_cols_contains_industry(self):
        """Test that list_categorical_cols() includes 'industry'."""
        result = list_categorical_cols()
        self.assertIn("industry", result, "Should include 'industry' as categorical column")

    def test_list_date_cols_returns_list(self):
        """Test that list_date_cols() returns a list."""
        result = list_date_cols()
        self.assertIsInstance(result, list)

    def test_list_date_cols_non_empty(self):
        """Test that list_date_cols() returns non-empty list."""
        result = list_date_cols()
        self.assertGreater(len(result), 0, "Should return at least one date column")

    def test_list_date_cols_contains_last_updated(self):
        """Test that list_date_cols() includes 'last_updated'."""
        result = list_date_cols()
        self.assertIn("last_updated", result, "Should include 'last_updated' as date column")

    def test_list_date_cols_contains_income_statement_report_date(self):
        """Test that list_date_cols() includes 'income_statement_report_date'."""
        result = list_date_cols()
        self.assertIn(
            "income_statement_report_date",
            result,
            "Should include 'income_statement_report_date' as date column",
        )

    def test_categorical_cols_have_correct_schema_definition(self):
        """Test that categorical columns have category dtype or categorical role."""
        categorical_cols = list_categorical_cols()

        for col in categorical_cols:
            self.assertIn(col, COLUMN_SCHEMA, f"Column '{col}' should be in COLUMN_SCHEMA")
            meta = COLUMN_SCHEMA[col]
            self.assertTrue(
                meta["dtype"] == "category" or meta["role"] == "categorical",
                f"Column '{col}' should have dtype='category' or role='categorical'",
            )

    def test_date_cols_have_correct_schema_definition(self):
        """Test that date columns have datetime dtype or date role."""
        date_cols = list_date_cols()

        for col in date_cols:
            self.assertIn(col, COLUMN_SCHEMA, f"Column '{col}' should be in COLUMN_SCHEMA")
            meta = COLUMN_SCHEMA[col]
            self.assertTrue(
                meta["dtype"] == "datetime64[ns]" or meta["role"] == "date",
                f"Column '{col}' should have dtype='datetime64[ns]' or role='date'",
            )

    def test_no_overlap_between_categorical_and_date_cols(self):
        """Test that categorical and date column lists don't overlap."""
        categorical_cols = set(list_categorical_cols())
        date_cols = set(list_date_cols())

        overlap = categorical_cols.intersection(date_cols)
        self.assertEqual(
            len(overlap), 0, f"Categorical and date columns should not overlap, found: {overlap}"
        )


class TestNotebookSchemaIntegration(unittest.TestCase):
    """Test schema integration with notebook preprocessing pipeline."""

    def test_prep_params_structure_with_schema_columns(self):
        """Test that prep_params dictionary can be created with schema columns."""
        # This simulates the notebook code structure
        categorical_columns_from_schema = list_categorical_cols()
        datetime_cols_from_schema = list_date_cols()

        # Mock other required parameters
        auxiliary_cols_to_drop = ["name", "description"]
        encoders = {}
        reference_date = None

        # Create prep_params as in notebook
        prep_params = {
            "cat_cols": categorical_columns_from_schema,
            "date_cols": datetime_cols_from_schema,
            "drop_cols": auxiliary_cols_to_drop,
            "encoders": encoders,
            "ref_date": reference_date,
        }

        # Verify structure
        self.assertIn("cat_cols", prep_params)
        self.assertIn("date_cols", prep_params)
        self.assertIn("drop_cols", prep_params)
        self.assertIn("encoders", prep_params)
        self.assertIn("ref_date", prep_params)

        # Verify types
        self.assertIsInstance(prep_params["cat_cols"], list)
        self.assertIsInstance(prep_params["date_cols"], list)
        self.assertGreater(
            len(prep_params["cat_cols"]), 0, "cat_cols should contain categorical columns"
        )
        self.assertGreater(
            len(prep_params["date_cols"]), 0, "date_cols should contain date columns"
        )

    def test_categorical_columns_from_schema_defined(self):
        """Test that categorical_columns_from_schema can be properly defined."""
        # This test simulates the fix that should be applied
        categorical_columns_from_schema = list_categorical_cols()

        self.assertIsNotNone(categorical_columns_from_schema)
        self.assertIsInstance(categorical_columns_from_schema, list)
        self.assertGreater(len(categorical_columns_from_schema), 0)

    def test_datetime_cols_from_schema_defined(self):
        """Test that datetime_cols_from_schema can be properly defined."""
        # This test simulates the fix that should be applied
        datetime_cols_from_schema = list_date_cols()

        self.assertIsNotNone(datetime_cols_from_schema)
        self.assertIsInstance(datetime_cols_from_schema, list)
        self.assertGreater(len(datetime_cols_from_schema), 0)

    def test_schema_columns_contain_expected_categories(self):
        """Test that schema columns contain expected categorical values."""
        categorical_columns = list_categorical_cols()

        # Expected categorical columns from schema
        expected_categorical = [
            "sector",
            "industry",
            "region",
            "country",
            "trading_country",
            "exchange",
        ]

        for expected_col in expected_categorical:
            self.assertIn(
                expected_col,
                categorical_columns,
                f"Expected categorical column '{expected_col}' not found",
            )

    def test_schema_columns_contain_expected_dates(self):
        """Test that schema columns contain expected date values."""
        date_columns = list_date_cols()

        # Expected date columns from schema
        expected_dates = ["last_updated", "income_statement_report_date", "next_earnings"]

        for expected_col in expected_dates:
            self.assertIn(
                expected_col, date_columns, f"Expected date column '{expected_col}' not found"
            )


class TestSchemaUtilityFunctions(unittest.TestCase):
    """Test schema utility functions for column metadata."""

    def test_get_expected_dtype_for_sector(self):
        """Test get_expected_dtype returns correct dtype for 'sector'."""
        dtype = get_expected_dtype("sector")
        self.assertEqual(dtype, "category")

    def test_get_expected_dtype_for_last_price(self):
        """Test get_expected_dtype returns correct dtype for 'last_price'."""
        dtype = get_expected_dtype("last_price")
        self.assertEqual(dtype, "float")

    def test_get_expected_dtype_for_last_updated(self):
        """Test get_expected_dtype returns correct dtype for 'last_updated'."""
        dtype = get_expected_dtype("last_updated")
        self.assertEqual(dtype, "datetime64[ns]")

    def test_get_expected_dtype_for_unknown_column(self):
        """Test get_expected_dtype returns None for unknown column."""
        dtype = get_expected_dtype("unknown_column_xyz")
        self.assertIsNone(dtype)

    def test_get_expected_dtype_normalizes_input(self):
        """Test get_expected_dtype normalizes column names."""
        # Test with spaces and special characters
        dtype = get_expected_dtype("Last Price")
        self.assertEqual(dtype, "float", "Should normalize 'Last Price' to 'last_price'")

    def test_get_column_role_for_sector(self):
        """Test get_column_role returns correct role for 'sector'."""
        role = get_column_role("sector")
        self.assertEqual(role, "categorical")

    def test_get_column_role_for_ticker(self):
        """Test get_column_role returns correct role for 'ticker'."""
        role = get_column_role("ticker")
        self.assertEqual(role, "id")

    def test_get_column_role_for_price_target(self):
        """Test get_column_role returns correct role for 'price_target'."""
        role = get_column_role("price_target")
        self.assertEqual(role, "target")

    def test_get_column_role_for_unknown_column(self):
        """Test get_column_role returns None for unknown column."""
        role = get_column_role("unknown_column_xyz")
        self.assertIsNone(role)

    def test_get_column_role_normalizes_input(self):
        """Test get_column_role normalizes column names."""
        role = get_column_role("Price Target")
        self.assertEqual(role, "target", "Should normalize 'Price Target' to 'price_target'")

    def test_list_numeric_feature_cols_returns_list(self):
        """Test list_numeric_feature_cols returns a list."""
        result = list_numeric_feature_cols()
        self.assertIsInstance(result, list)

    def test_list_numeric_feature_cols_non_empty(self):
        """Test list_numeric_feature_cols returns non-empty list."""
        result = list_numeric_feature_cols()
        self.assertGreater(len(result), 0, "Should return at least one numeric feature column")

    def test_list_numeric_feature_cols_contains_market_cap(self):
        """Test list_numeric_feature_cols includes 'market_cap'."""
        result = list_numeric_feature_cols()
        self.assertIn("market_cap", result)

    def test_list_numeric_feature_cols_contains_last_price(self):
        """Test list_numeric_feature_cols includes 'last_price'."""
        result = list_numeric_feature_cols()
        self.assertIn("last_price", result)

    def test_normalize_column_name_basic(self):
        """Test normalize_column_name with basic transformation."""
        normalized = normalize_column_name("Last Price")
        self.assertEqual(normalized, "last_price")

    def test_normalize_column_name_with_parentheses(self):
        """Test normalize_column_name with parentheses."""
        normalized = normalize_column_name("P/E (LTM)")
        self.assertEqual(normalized, "p_e_ltm")

    def test_normalize_column_name_with_slash(self):
        """Test normalize_column_name with forward slash."""
        normalized = normalize_column_name("EV/EBITDA")
        self.assertEqual(normalized, "ev_ebitda")

    def test_normalize_column_name_with_hash(self):
        """Test normalize_column_name with hash symbol."""
        normalized = normalize_column_name("Column#1")
        self.assertEqual(normalized, "columnnum1")

    def test_normalize_column_name_with_percent(self):
        """Test normalize_column_name with percent symbol."""
        normalized = normalize_column_name("Return %")
        self.assertEqual(normalized, "return_pct")

    def test_normalize_column_name_with_ampersand(self):
        """Test normalize_column_name with ampersand."""
        normalized = normalize_column_name("Sales & Marketing")
        self.assertEqual(normalized, "sales_and_marketing")

    def test_normalize_column_name_removes_consecutive_underscores(self):
        """Test normalize_column_name removes consecutive underscores."""
        normalized = normalize_column_name("Column  With   Spaces")
        self.assertEqual(normalized, "column_with_spaces")

    def test_normalize_column_name_strips_leading_trailing_underscores(self):
        """Test normalize_column_name strips leading/trailing underscores."""
        normalized = normalize_column_name("_Column_")
        self.assertEqual(normalized, "column")


class TestSchemaIntegrationEdgeCases(unittest.TestCase):
    """Test edge cases for schema integration."""

    def test_empty_prep_params_with_schema_lists(self):
        """Test that empty schema lists can be handled in prep_params."""
        # Even if schema lists are empty, prep_params should be valid
        prep_params = {
            "cat_cols": [],
            "date_cols": [],
            "drop_cols": [],
            "encoders": {},
            "ref_date": None,
        }

        self.assertIsInstance(prep_params, dict)
        self.assertEqual(len(prep_params), 5)

    def test_categorical_cols_are_unique(self):
        """Test that categorical columns list contains unique values."""
        categorical_cols = list_categorical_cols()

        self.assertEqual(
            len(categorical_cols),
            len(set(categorical_cols)),
            "Categorical columns should be unique",
        )

    def test_date_cols_are_unique(self):
        """Test that date columns list contains unique values."""
        date_cols = list_date_cols()

        self.assertEqual(len(date_cols), len(set(date_cols)), "Date columns should be unique")

    def test_categorical_cols_consistency_across_calls(self):
        """Test that categorical columns list is consistent across multiple calls."""
        result1 = list_categorical_cols()
        result2 = list_categorical_cols()

        self.assertEqual(result1, result2, "Categorical columns should be consistent across calls")

    def test_date_cols_consistency_across_calls(self):
        """Test that date columns list is consistent across multiple calls."""
        result1 = list_date_cols()
        result2 = list_date_cols()

        self.assertEqual(result1, result2, "Date columns should be consistent across calls")


if __name__ == "__main__":
    unittest.main()
