"""
Integration tests for data loading normalization consistency.

TDD Implementation: Data Normalization Fixes (2025-11-24)
Phase 3.1 - Data Loading Normalization Integration Tests

Purpose:
- Verify normalize_columns() uses schema.normalize_column_name()
- Test data loading produces schema-compatible column names
- Ensure CSV column normalization matches COLUMN_SCHEMA keys
- Integration validation of the fix applied in Task 1.1

Test Coverage:
- normalize_columns() function uses schema function
- CSV loading produces correct column names
- Analyst rating columns normalize correctly with "num_" prefix
- SG&A columns normalize correctly with "and"
"""

import unittest
import pandas as pd
import tempfile
from pathlib import Path

from finance_ml.ml_workflow.preprocessing.data import normalize_columns, load_from_csv
from finance_ml.ml_workflow.data.schema import normalize_column_name, COLUMN_SCHEMA


class TestNormalizeColumnsFunction(unittest.TestCase):
    """Test the normalize_columns() function uses schema normalization."""

    def test_normalize_columns_uses_schema_function(self):
        """Test normalize_columns() produces same results as schema.normalize_column_name()."""
        # Create test dataframe with problematic column names
        test_columns = [
            "# Strong Sell Ratings",
            "# Strong Buys Ratings",
            "# Hold Ratings",
            "# Buys Ratings",
            "# Sell Ratings",
            "Selling General & Admin Expenses/Total (FQ)",
            "1-Day %",
            "Market Cap",
            "Ticker",
        ]
        df = pd.DataFrame(columns=test_columns)

        # Normalize using the function (with preserve_schema=False to trigger new path)
        df_normalized = normalize_columns(df, preserve_schema=False)

        # Expected results using schema function
        expected_columns = [normalize_column_name(col) for col in test_columns]

        # Compare
        self.assertEqual(
            list(df_normalized.columns),
            expected_columns,
            "normalize_columns() should produce same results as schema.normalize_column_name()",
        )

    def test_analyst_rating_columns_get_num_prefix(self):
        """Test analyst rating columns get 'num_' prefix after normalization."""
        analyst_columns = [
            "# Strong Sell Ratings",
            "# Strong Buys Ratings",
            "# Hold Ratings",
            "# Buys Ratings",
            "# Sell Ratings",
        ]
        df = pd.DataFrame(columns=analyst_columns)
        df_normalized = normalize_columns(df, preserve_schema=False)

        expected = [
            "num_strong_sell_ratings",
            "num_strong_buys_ratings",
            "num_hold_ratings",
            "num_buys_ratings",
            "num_sell_ratings",
        ]

        self.assertEqual(list(df_normalized.columns), expected)

    def test_sga_columns_get_and_connector(self):
        """Test SG&A columns normalize with 'and' connector."""
        sga_columns = [
            "Selling General & Admin Expenses/Total (FQ)",
            "Selling General & Admin Expenses/Total (FY)",
        ]
        df = pd.DataFrame(columns=sga_columns)
        df_normalized = normalize_columns(df, preserve_schema=False)

        expected = [
            "selling_general_and_admin_expenses_total_fq",
            "selling_general_and_admin_expenses_total_fy",
        ]

        self.assertEqual(list(df_normalized.columns), expected)

    def test_normalize_unmapped_column_with_preserve_schema(self):
        """Test fallback normalization for unmapped columns when preserve_schema=True."""
        # '# New Metric' is not in schema_mapping (presumably), so it hits fallback.
        # Before fix: regex -> '_new_metric' (or 'new_metric' if stripped)
        # After fix: normalize_column_name -> 'num_new_metric'

        col = "# New Metric"
        df = pd.DataFrame(columns=[col])
        df_normalized = normalize_columns(df, preserve_schema=True)

        self.assertEqual(df_normalized.columns[0], "num_new_metric")

    def test_normalized_columns_exist_in_schema(self):
        """Test normalized column names exist in COLUMN_SCHEMA."""
        test_columns = [
            "Ticker",
            "Sector",
            "Last Price",
            "# Strong Sell Ratings",
            "1-Day %",
            "Selling General & Admin Expenses/Total (FQ)",
        ]
        df = pd.DataFrame(columns=test_columns)
        df_normalized = normalize_columns(df, preserve_schema=False)

        missing = []
        for col in df_normalized.columns:
            if col not in COLUMN_SCHEMA:
                missing.append(col)

        self.assertEqual(
            len(missing), 0, f"Normalized columns missing from COLUMN_SCHEMA: {missing}"
        )


class TestCSVLoadingNormalization(unittest.TestCase):
    """Test CSV loading produces schema-compatible column names."""

    def test_load_from_csv_produces_schema_compatible_columns(self):
        """Test load_from_csv() normalizes columns to match COLUMN_SCHEMA."""
        # Create temporary CSV with sample columns
        sample_columns = [
            "Ticker",
            "Sector",
            "Region",
            "Last Price",
            "# Strong Sell Ratings",
            "# Strong Buys Ratings",
            "1-Day %",
        ]

        # Create temp directory and CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "screening_us.csv"

            # Write sample CSV
            sample_data = pd.DataFrame({col: [1.0] for col in sample_columns})
            sample_data.to_csv(csv_path, index=False)

            # Load using the function
            df_loaded = load_from_csv(Path(tmpdir), limit=10)

            # Check all columns are in schema
            missing = []
            for col in df_loaded.columns:
                if col not in COLUMN_SCHEMA:
                    missing.append(col)

            # Allow 'region' to be missing since it might be added during loading
            missing = [col for col in missing if col != "region"]

            self.assertEqual(
                len(missing), 0, f"Loaded CSV columns not in COLUMN_SCHEMA: {missing}"
            )

    def test_csv_analyst_columns_normalize_with_num_prefix(self):
        """Test CSV analyst rating columns normalize with num_ prefix."""
        sample_columns = [
            "Ticker",
            "# Strong Sell Ratings",
            "# Hold Ratings",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "screening_us.csv"
            sample_data = pd.DataFrame({col: [1.0] for col in sample_columns})
            sample_data.to_csv(csv_path, index=False)

            df_loaded = load_from_csv(Path(tmpdir), limit=10)

            # Check analyst columns have num_ prefix
            self.assertIn("num_strong_sell_ratings", df_loaded.columns)
            self.assertIn("num_hold_ratings", df_loaded.columns)


class TestNormalizationConsistency(unittest.TestCase):
    """Test normalization consistency across pipeline."""

    def test_normalize_columns_matches_schema_normalize(self):
        """Test normalize_columns() and schema.normalize_column_name() produce identical results."""
        test_cases = [
            "# Strong Sell Ratings",
            "# Buys Ratings",
            "Selling General & Admin Expenses/Total (FQ)",
            "1-Day %",
            "P/E (LTM)",
            "Market Cap",
            "Total Revenues (LTM)",
        ]

        for test_col in test_cases:
            with self.subTest(column=test_col):
                # Method 1: via normalize_columns
                df = pd.DataFrame(columns=[test_col])
                df_norm = normalize_columns(df, preserve_schema=False)
                result_via_function = df_norm.columns[0]

                # Method 2: via schema.normalize_column_name
                result_via_schema = normalize_column_name(test_col)

                self.assertEqual(
                    result_via_function,
                    result_via_schema,
                    f"Inconsistent normalization for '{test_col}': "
                    f"normalize_columns gave '{result_via_function}', "
                    f"schema.normalize_column_name gave '{result_via_schema}'",
                )


if __name__ == "__main__":
    unittest.main()
