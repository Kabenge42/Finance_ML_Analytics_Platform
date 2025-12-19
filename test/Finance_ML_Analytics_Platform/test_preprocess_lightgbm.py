"""
Tests for preprocess_for_lightgbm function.

This module tests the preprocessing function that converts DataFrames to LightGBM-compatible format
by handling categorical and datetime columns.
"""

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from finance_ml.ml_workflow.features import preprocess_for_lightgbm


class TestPreprocessForLightGBM(unittest.TestCase):
    """Test suite for preprocess_for_lightgbm function."""

    def setUp(self):
        """Set up test data."""
        # Create sample data with problematic columns mentioned in the issue
        self.sample_data = pd.DataFrame(
            {
                # Numeric columns (should remain unchanged)
                "last_price": [100.0, 150.0, 200.0, 175.0, 125.0],
                "market_cap": [1e9, 2e9, 3e9, 2.5e9, 1.5e9],
                "eps": [5.0, 7.5, 10.0, 8.75, 6.25],
                # Categorical columns (object dtype) - from issue description
                "exchange": ["NYSE", "NASDAQ", "NYSE", "NASDAQ", "NYSE"],
                "sector": ["Technology", "Healthcare", "Technology", "Finance", "Healthcare"],
                "industry": ["Software", "Biotech", "Hardware", "Banking", "Pharma"],
                "region": ["US", "US", "EU", "US", "APAC"],
                "country": ["USA", "USA", "Germany", "USA", "Japan"],
                "trading_country": ["USA", "USA", "Germany", "USA", "Japan"],
                "style_class": ["Growth", "Value", "Growth", "Value", "Growth"],
                "size_class": ["Large", "Mid", "Large", "Mid", "Small"],
                "flag": ["A", "B", "A", "C", "B"],
                # Datetime column - from issue description
                "next_earnings": [
                    datetime(2025, 11, 15),
                    datetime(2025, 11, 20),
                    datetime(2025, 12, 1),
                    datetime(2025, 11, 25),
                    datetime(2025, 12, 5),
                ],
            }
        )

    def test_basic_preprocessing_auto_detect(self):
        """Test basic preprocessing with auto-detection of column types."""
        df_processed, encoders = preprocess_for_lightgbm(self.sample_data, return_encoders=False)

        # Check that all columns are numeric
        non_numeric = df_processed.select_dtypes(exclude=[np.number]).columns.tolist()
        self.assertEqual(len(non_numeric), 0, "All columns should be numeric")

        # Check that datetime column was removed and new columns were added
        self.assertNotIn("next_earnings", df_processed.columns)
        self.assertIn("next_earnings_year", df_processed.columns)
        self.assertIn("next_earnings_month", df_processed.columns)
        self.assertIn("next_earnings_day", df_processed.columns)
        self.assertIn("next_earnings_days_from_now", df_processed.columns)

        # Check that categorical columns were encoded
        self.assertIn("exchange", df_processed.columns)
        self.assertIn("sector", df_processed.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(df_processed["exchange"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df_processed["sector"]))

        # Check that numeric columns remain
        self.assertIn("last_price", df_processed.columns)
        self.assertIn("market_cap", df_processed.columns)

    def test_return_encoders(self):
        """Test that encoders are returned when requested."""
        df_processed, encoders = preprocess_for_lightgbm(self.sample_data, return_encoders=True)

        # Check that encoders dictionary is not None
        self.assertIsNotNone(encoders)
        self.assertIsInstance(encoders, dict)

        # Check that encoders were created for categorical columns
        categorical_cols = [
            "exchange",
            "sector",
            "industry",
            "region",
            "country",
            "trading_country",
            "style_class",
            "size_class",
            "flag",
        ]
        for col in categorical_cols:
            self.assertIn(col, encoders, f"Encoder should exist for {col}")
            self.assertIsInstance(encoders[col], LabelEncoder)

        # Test that encoders can be used for inverse transform
        original_sectors = self.sample_data["sector"].unique()
        encoded_values = df_processed["sector"].unique()
        decoded_sectors = encoders["sector"].inverse_transform(encoded_values)
        self.assertEqual(set(decoded_sectors), set(original_sectors))

    def test_explicit_categorical_columns(self):
        """Test with explicitly specified categorical columns."""
        categorical_cols = ["sector", "region", "exchange"]
        df_processed, encoders = preprocess_for_lightgbm(
            self.sample_data, categorical_columns=categorical_cols, return_encoders=True
        )

        # Check that specified columns were encoded
        for col in categorical_cols:
            self.assertIn(col, df_processed.columns)
            self.assertTrue(pd.api.types.is_numeric_dtype(df_processed[col]))
            self.assertIn(col, encoders)

    def test_explicit_datetime_columns(self):
        """Test with explicitly specified datetime columns."""
        datetime_cols = ["next_earnings"]
        df_processed, _ = preprocess_for_lightgbm(self.sample_data, datetime_columns=datetime_cols)

        # Check that datetime features were created
        self.assertNotIn("next_earnings", df_processed.columns)
        self.assertIn("next_earnings_year", df_processed.columns)
        self.assertIn("next_earnings_month", df_processed.columns)
        self.assertIn("next_earnings_day", df_processed.columns)
        self.assertIn("next_earnings_days_from_now", df_processed.columns)

        # Verify datetime features have correct values
        self.assertTrue((df_processed["next_earnings_year"] == 2025).all())
        self.assertTrue((df_processed["next_earnings_month"].isin([11, 12])).all())

    def test_drop_columns(self):
        """Test dropping specified columns."""
        drop_cols = ["flag", "trading_country"]
        df_processed, _ = preprocess_for_lightgbm(self.sample_data, drop_columns=drop_cols)

        # Check that dropped columns are not in result
        for col in drop_cols:
            self.assertNotIn(col, df_processed.columns)

    def test_missing_values_categorical(self):
        """Test handling of missing values in categorical columns."""
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[1, "sector"] = np.nan
        data_with_nan.loc[2, "exchange"] = None

        df_processed, encoders = preprocess_for_lightgbm(data_with_nan, return_encoders=True)

        # Check that NaN values were handled
        self.assertFalse(df_processed["sector"].isna().any())
        self.assertFalse(df_processed["exchange"].isna().any())

        # Check that 'Unknown' is in the encoder classes
        self.assertIn("Unknown", encoders["sector"].classes_)
        self.assertIn("Unknown", encoders["exchange"].classes_)

    def test_missing_values_datetime(self):
        """Test handling of missing values in datetime columns."""
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[1, "next_earnings"] = pd.NaT

        df_processed, _ = preprocess_for_lightgbm(data_with_nan)

        # Check that NaN values were handled (filled with 0)
        self.assertFalse(df_processed["next_earnings_year"].isna().any())
        self.assertFalse(df_processed["next_earnings_month"].isna().any())
        self.assertFalse(df_processed["next_earnings_day"].isna().any())
        self.assertFalse(df_processed["next_earnings_days_from_now"].isna().any())

    def test_infinite_values(self):
        """Test handling of infinite values."""
        data_with_inf = self.sample_data.copy()
        data_with_inf["market_cap"] = [1e9, np.inf, 3e9, -np.inf, 1.5e9]

        df_processed, _ = preprocess_for_lightgbm(data_with_inf)

        # Check that infinite values were replaced
        self.assertFalse(np.isinf(df_processed["market_cap"]).any())
        self.assertFalse(df_processed["market_cap"].isna().any())

    def test_all_problematic_columns_from_issue(self):
        """Test with all problematic columns mentioned in the issue."""
        # Create data with all columns from the error message
        problematic_data = pd.DataFrame(
            {
                "exchange": ["NYSE", "NASDAQ", "LSE"],
                "unit": ["USD", "USD", "GBP"],
                "sector": ["Tech", "Finance", "Healthcare"],
                "industry": ["Software", "Banking", "Pharma"],
                "next_earnings": [datetime.now() + timedelta(days=30) for _ in range(3)],
                "style_class": ["Growth", "Value", "Blend"],
                "next_earnings_status": ["Confirmed", "Tentative", "Unknown"],
                "size_class": ["Large", "Mid", "Small"],
                "flag": ["A", "B", "C"],
                "region": ["US", "US", "EU"],
                "country": ["USA", "USA", "UK"],
                "trading_country": ["USA", "USA", "UK"],
                "last_price": [100.0, 150.0, 200.0],
            }
        )

        df_processed, _ = preprocess_for_lightgbm(problematic_data)

        # Verify all columns are numeric
        non_numeric = df_processed.select_dtypes(exclude=[np.number]).columns.tolist()
        self.assertEqual(len(non_numeric), 0, f"Non-numeric columns remain: {non_numeric}")

        # Verify no NaN values
        nan_count = df_processed.isna().sum().sum()
        self.assertEqual(nan_count, 0, f"Found {nan_count} NaN values")

    def test_datetime_string_conversion(self):
        """Test conversion of datetime strings to datetime features."""
        data_with_strings = self.sample_data.copy()
        data_with_strings["next_earnings"] = [
            "2025-11-15",
            "2025-11-20",
            "2025-12-01",
            "2025-11-25",
            "2025-12-05",
        ]

        df_processed, _ = preprocess_for_lightgbm(data_with_strings)

        # Check that datetime features were created
        self.assertIn("next_earnings_year", df_processed.columns)
        self.assertTrue((df_processed["next_earnings_year"] == 2025).all())

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        df_processed, encoders = preprocess_for_lightgbm(empty_df, return_encoders=True)

        self.assertEqual(len(df_processed), 0)
        self.assertEqual(len(df_processed.columns), 0)
        self.assertEqual(len(encoders), 0)

    def test_only_numeric_columns(self):
        """Test with DataFrame containing only numeric columns."""
        numeric_df = pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0],
                "col2": [4.0, 5.0, 6.0],
                "col3": [7, 8, 9],
            }
        )

        df_processed, encoders = preprocess_for_lightgbm(numeric_df, return_encoders=True)

        # Should return unchanged DataFrame
        self.assertEqual(df_processed.shape, numeric_df.shape)
        self.assertEqual(len(encoders), 0)
        pd.testing.assert_index_equal(df_processed.columns, numeric_df.columns)

    def test_preserves_index(self):
        """Test that the function preserves the DataFrame index."""
        df_with_index = self.sample_data.copy()
        df_with_index.index = ["A", "B", "C", "D", "E"]

        df_processed, _ = preprocess_for_lightgbm(df_with_index)

        pd.testing.assert_index_equal(df_processed.index, df_with_index.index)

    def test_encoder_reproducibility(self):
        """Test that encoders can inverse transform their own encoded values."""
        df_processed, encoders = preprocess_for_lightgbm(self.sample_data, return_encoders=True)

        # Test that encoders can successfully decode their own encodings
        for col in ["sector", "exchange", "region"]:
            # Get encoded values
            encoded_values = df_processed[col].unique()

            # Decode them
            decoded_values = encoders[col].inverse_transform(encoded_values)

            # Re-encode the decoded values
            re_encoded = encoders[col].transform(decoded_values)

            # Should get back the same encoded values
            np.testing.assert_array_equal(
                sorted(encoded_values),
                sorted(re_encoded),
                err_msg=f"Encoder for {col} should be consistent with inverse transform",
            )

    def test_train_test_encoder_reuse(self):
        """Test train/test workflow with encoder reuse (addresses notebook TypeError issue)."""
        # Create train data
        train_data = self.sample_data.copy()

        # Create test data with same structure but different values
        test_data = pd.DataFrame(
            {
                "last_price": [110.0, 160.0, 90.0],
                "market_cap": [1.2e9, 2.2e9, 0.8e9],
                "eps": [5.5, 8.0, 4.5],
                "exchange": ["NYSE", "NASDAQ", "LSE"],  # LSE is unseen category
                "sector": ["Technology", "Healthcare", "Energy"],  # Energy is unseen
                "industry": ["Software", "Biotech", "Oil"],
                "region": ["US", "US", "EU"],
                "country": ["USA", "USA", "France"],
                "trading_country": ["USA", "USA", "France"],
                "style_class": ["Growth", "Value", "Value"],
                "size_class": ["Large", "Mid", "Small"],
                "flag": ["A", "B", "D"],  # D is unseen
                "next_earnings": [
                    datetime(2025, 12, 10),
                    datetime(2025, 12, 15),
                    datetime(2025, 12, 20),
                ],
            }
        )

        # TRAINING MODE: Fit encoders with return_encoders=True
        categorical_cols = [
            "exchange",
            "sector",
            "industry",
            "region",
            "country",
            "trading_country",
            "style_class",
            "size_class",
            "flag",
        ]
        datetime_cols = ["next_earnings"]

        X_train_processed, encoders = preprocess_for_lightgbm(
            train_data,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            return_encoders=True,  # Store encoders for test data
        )

        # Verify encoders were returned
        self.assertIsNotNone(encoders)
        self.assertIsInstance(encoders, dict)
        for col in categorical_cols:
            self.assertIn(col, encoders)
        self.assertIn("_reference_date", encoders)  # Should have reference date

        # Extract reference date for test consistency
        ref_date = encoders.get("_reference_date")
        self.assertIsNotNone(ref_date)

        # TEST/INFERENCE MODE: Use training encoders (the pattern from the notebook)
        X_test_processed, returned_encoders = preprocess_for_lightgbm(
            test_data,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            encoders=encoders,  # Pass the encoders from training
            reference_date=ref_date,  # Use training reference date
            # Note: return_encoders is False by default
        )

        # Verify the function returned None for encoders (since return_encoders=False)
        self.assertIsNone(returned_encoders)

        # Verify test data was processed successfully
        self.assertEqual(X_test_processed.shape[0], 3)
        non_numeric = X_test_processed.select_dtypes(exclude=[np.number]).columns.tolist()
        self.assertEqual(len(non_numeric), 0, "All columns should be numeric")

        # Verify datetime features use consistent reference date
        self.assertIn("next_earnings_days_from_now", X_test_processed.columns)

        # Verify unseen categories were handled (mapped to 'Unknown' class)
        # LSE, Energy, Oil, D are unseen in train data but should be encoded without error
        self.assertFalse(X_test_processed.isna().any().any(), "No NaN values should remain")

        # Verify column names match between train and test
        self.assertEqual(
            set(X_train_processed.columns),
            set(X_test_processed.columns),
            "Train and test should have same columns",
        )

    def test_train_test_encoder_reuse_with_return_encoders_true(self):
        """Test that we can pass encoders AND set return_encoders=True in test mode."""
        train_data = self.sample_data.copy()
        test_data = self.sample_data.iloc[:3].copy()

        # Train mode
        X_train, encoders = preprocess_for_lightgbm(train_data, return_encoders=True)

        # Test mode with return_encoders=True (should return the same encoders)
        X_test, returned_encoders = preprocess_for_lightgbm(
            test_data,
            encoders=encoders,
            reference_date=encoders.get("_reference_date"),
            return_encoders=True,  # Request encoders back
        )

        # Should return the provided encoders
        self.assertIsNotNone(returned_encoders)
        self.assertEqual(id(returned_encoders), id(encoders), "Should return same encoder object")


if __name__ == "__main__":
    unittest.main()
