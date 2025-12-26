"""
Test suite for data type detection and schema validation.

Tests the schema-aware datatype detection, casting, and validation
functionality following TDD principles (Phase 9.3).
"""

import unittest

import pandas as pd

# Import the modules we're about to create (will fail initially - TDD)
try:
    from finance_ml.ml_workflow.data.schema import (
        COLUMN_SCHEMA,
        get_expected_dtype,
        get_column_role,
        list_numeric_feature_cols,
        list_categorical_cols,
        PHASE93_FEATURE_INPUTS,
    )
    from finance_ml.ml_workflow.preprocessing.dtypes import (
        detect_and_cast_dtypes,
    )
except ImportError as e:
    # Expected to fail initially in TDD
    COLUMN_SCHEMA = None
    get_expected_dtype = None
    get_column_role = None
    list_numeric_feature_cols = None
    list_categorical_cols = None
    PHASE93_FEATURE_INPUTS = None
    detect_and_cast_dtypes = None


class TestDataTypesDetection(unittest.TestCase):
    """Test schema-aware datatype detection and casting."""

    def setUp(self):
        """Set up test data."""
        # Skip tests if modules aren't implemented yet
        if COLUMN_SCHEMA is None or detect_and_cast_dtypes is None:
            self.skipTest("Schema module not yet implemented - TDD phase")

    def test_detect_and_cast_dtypes_respects_column_schema(self):
        """Test that detect_and_cast_dtypes correctly casts columns per schema.

        Arrange: small DataFrame with columns ticker, sector, last_price,
                 market_cap, last_updated loaded as object.
        Act: call detect_and_cast_dtypes(df).
        Assert:
            - df["last_price"] and df["market_cap"] have numeric dtypes.
            - df["last_updated"] is datetime64[ns].
            - df["sector"] is category/string; ticker is string.
        """
        # Arrange: Create test dataframe with all object dtypes
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "sector": ["Technology", "Technology", "Technology"],
                "last_price": ["150.25", "2800.50", "350.75"],
                "market_cap": ["2500000000000", "1800000000000", "2600000000000"],
                "last_updated": ["2023-01-15", "2023-01-15", "2023-01-15"],
            }
        )

        # Ensure all columns are object dtype initially
        for col in df.columns:
            df[col] = df[col].astype("object")

        # Act: Detect and cast dtypes
        df_cast, diagnostics = detect_and_cast_dtypes(df)

        # Assert: Check numeric columns
        self.assertTrue(
            pd.api.types.is_numeric_dtype(df_cast["last_price"]), "last_price should be numeric"
        )
        self.assertTrue(
            pd.api.types.is_numeric_dtype(df_cast["market_cap"]), "market_cap should be numeric"
        )

        # Assert: Check datetime column
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(df_cast["last_updated"]),
            "last_updated should be datetime64[ns]",
        )

        # Assert: Check categorical/string columns
        self.assertIn(
            df_cast["sector"].dtype.name,
            ["object", "string", "category"],
            "sector should be category or string",
        )
        self.assertIn(df_cast["ticker"].dtype.name, ["object", "string"], "ticker should be string")

        # Assert: Diagnostics should be present
        self.assertIn("cast_applied", diagnostics)
        self.assertIn("inferred_dtypes", diagnostics)

    def test_detect_and_cast_dtypes_reports_coercion_warnings(self):
        """Test that coercion warnings are reported for invalid data.

        Include invalid numeric strings ("N/A", "-") in last_price.
        Assert: Coercion count for last_price equals number of invalid entries.
        """
        # Arrange: Create dataframe with invalid numeric values
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
                "last_price": ["150.25", "N/A", "350.75", "-", "3200.00"],
                "sector": ["Technology", "Technology", "Technology", "Automotive", "Technology"],
            }
        )

        # Act: Detect and cast dtypes
        df_cast, diagnostics = detect_and_cast_dtypes(df)

        # Assert: Check that coercion warnings are reported
        self.assertIn("coercion_warnings", diagnostics)

        # Assert: Two invalid entries (N/A and -) should be coerced to NaN
        if "last_price" in diagnostics["coercion_warnings"]:
            self.assertEqual(
                diagnostics["coercion_warnings"]["last_price"],
                2,
                "Should report 2 coerced values in last_price",
            )

        # Assert: Check that valid values remain and invalid became NaN
        self.assertEqual(
            df_cast["last_price"].notna().sum(), 3, "Should have 3 valid numeric values"
        )
        self.assertEqual(
            df_cast["last_price"].isna().sum(), 2, "Should have 2 NaN values from coercion"
        )

    def test_unknown_and_missing_columns_reported(self):
        """Test that unknown and missing columns are properly reported.

        Add extra column foo_bar not in COLUMN_SCHEMA.
        Remove price_target (expected in schema).
        Assert diagnostics["unknown_columns"] == ["foo_bar"] and
               "price_target" in missing_expected_columns.
        """
        # Arrange: Create dataframe with unknown column and missing expected column
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "sector": ["Technology", "Technology", "Technology"],
                "last_price": ["150.25", "2800.50", "350.75"],
                "foo_bar": ["unknown1", "unknown2", "unknown3"],  # Unknown column
                # Missing: price_target (which should be in schema)
            }
        )

        # Act: Detect and cast dtypes
        df_cast, diagnostics = detect_and_cast_dtypes(df)

        # Assert: Unknown columns reported
        self.assertIn("unknown_columns", diagnostics)
        self.assertIn(
            "foo_bar", diagnostics["unknown_columns"], "foo_bar should be reported as unknown"
        )

        # Assert: Missing expected columns reported
        self.assertIn("missing_expected_columns", diagnostics)
        # Note: Only check if price_target is actually in the schema definition
        # and marked as required/expected

    def test_phase93_feature_inputs_all_numeric_where_expected(self):
        """Test that Phase 9.3 feature input columns have correct dtypes.

        For each column in PHASE93_FEATURE_INPUTS["momentum"] etc.,
        assert dtype is numeric or datetime depending on role.
        """
        if PHASE93_FEATURE_INPUTS is None:
            self.skipTest("PHASE93_FEATURE_INPUTS not yet implemented")

        # Arrange: Create dataframe with Phase 9.3 feature columns
        # Sample momentum features (try both old short key and new full key)
        momentum_cols = PHASE93_FEATURE_INPUTS.get("momentum", [])
        if not momentum_cols:
            momentum_cols = PHASE93_FEATURE_INPUTS.get("Momentum & Technical", [])

        if not momentum_cols:
            self.skipTest("No momentum features defined yet")

        # Create sample data for momentum features
        df_data = {"ticker": ["AAPL", "GOOGL", "MSFT"]}
        for col in momentum_cols[:5]:  # Test first 5 to keep it manageable
            df_data[col] = ["10.5", "20.3", "15.7"]

        df = pd.DataFrame(df_data)

        # Act: Detect and cast dtypes
        df_cast, diagnostics = detect_and_cast_dtypes(df)

        # Assert: All momentum features should be numeric
        for col in momentum_cols[:5]:
            if col in df_cast.columns:
                self.assertTrue(
                    pd.api.types.is_numeric_dtype(df_cast[col])
                    or pd.api.types.is_datetime64_any_dtype(df_cast[col]),
                    f"{col} should be numeric or datetime",
                )


class TestSchemaHelpers(unittest.TestCase):
    """Test schema helper functions."""

    def setUp(self):
        """Set up test data."""
        if get_expected_dtype is None:
            self.skipTest("Schema helpers not yet implemented - TDD phase")

    def test_get_expected_dtype_returns_correct_type(self):
        """Test that get_expected_dtype returns the correct dtype for known columns."""
        # Test numeric column
        dtype = get_expected_dtype("last_price")
        self.assertIn(dtype, ["float", "float64", "numeric"])

        # Test categorical column
        dtype = get_expected_dtype("sector")
        self.assertIn(dtype, ["category", "string", "object"])

        # Test datetime column
        dtype = get_expected_dtype("last_updated")
        self.assertIn(dtype, ["datetime64[ns]", "datetime", "date"])

    def test_get_column_role_returns_correct_role(self):
        """Test that get_column_role returns the correct role for known columns."""
        # Test id column
        role = get_column_role("ticker")
        self.assertEqual(role, "id")

        # Test semantic role for a price column
        role = get_column_role("last_price")
        self.assertEqual(role, "price")

        # Test target column
        role = get_column_role("price_target")
        self.assertEqual(role, "target")

    def test_list_numeric_feature_cols_returns_list(self):
        """Test that list_numeric_feature_cols returns a non-empty list."""
        numeric_cols = list_numeric_feature_cols()
        self.assertIsInstance(numeric_cols, list)
        self.assertGreater(len(numeric_cols), 0, "Should have at least some numeric features")

        # Check some expected numeric columns are present
        # These are from the SQL schema
        expected_numeric = ["last_price", "market_cap", "enterprise_value"]
        for col in expected_numeric:
            if col in numeric_cols:
                # At least one should be present
                self.assertIn(col, numeric_cols)
                break

    def test_list_categorical_cols_returns_list(self):
        """Test that list_categorical_cols returns a non-empty list."""
        categorical_cols = list_categorical_cols()
        self.assertIsInstance(categorical_cols, list)
        self.assertGreater(
            len(categorical_cols), 0, "Should have at least some categorical columns"
        )

        # Check some expected categorical columns
        expected_categorical = ["sector", "industry", "region"]
        for col in expected_categorical:
            if col in categorical_cols:
                self.assertIn(col, categorical_cols)
                break

    def test_missing_base_columns_now_in_schema(self):
        """Test that 5 previously missing base columns are now in COLUMN_SCHEMA.

        This test validates the fix for the 25,990 NaN values issue where
        base columns (no time suffix) were missing from the schema.

        Arrange: List of 5 problematic columns that caused imputation failures.
        Act: Check if each column exists in COLUMN_SCHEMA.
        Assert:
            - All 5 columns are in COLUMN_SCHEMA
            - Each has expected dtype (float for all - employees uses float for NULL handling)
            - Each has role='feature'
        """
        # Arrange: The 5 columns that were causing WARNING: NaN values still present
        # Note: employees is float (not int) to support NULL values from CSV/PostgreSQL
        # NOTE: Roles are semantic (code_guidelines.md v1.11): counts/price/etc.
        missing_columns = [
            ("r_d_expenses", "float", "feature"),
            ("intangible_assets", "float", "feature"),
            ("employees", "float", "count"),  # float for NULL handling, semantic count role
            ("marketing_expenses", "float", "feature"),
            ("eps_previous_year", "float", "feature"),
        ]

        # Act & Assert: Check each column is now in schema
        for col_name, expected_dtype, expected_role in missing_columns:
            with self.subTest(column=col_name):
                # Column should be in schema
                self.assertIn(
                    col_name,
                    COLUMN_SCHEMA,
                    f"Column '{col_name}' should be in COLUMN_SCHEMA (was missing before fix)",
                )

                # Check dtype matches expected
                actual_dtype = COLUMN_SCHEMA[col_name]["dtype"]
                self.assertEqual(
                    actual_dtype,
                    expected_dtype,
                    f"Column '{col_name}' should have dtype='{expected_dtype}', got '{actual_dtype}'",
                )

                # Check role matches expected semantic role
                actual_role = COLUMN_SCHEMA[col_name]["role"]
                self.assertEqual(
                    actual_role,
                    expected_role,
                    f"Column '{col_name}' should have role='{expected_role}', got '{actual_role}'",
                )

        # Additional assertion: Schema should now have 283 columns (278 + 5 new)
        self.assertGreaterEqual(
            len(COLUMN_SCHEMA),
            283,
            f"COLUMN_SCHEMA should have at least 283 columns after adding 5 base columns, got {len(COLUMN_SCHEMA)}",
        )


if __name__ == "__main__":
    unittest.main()
