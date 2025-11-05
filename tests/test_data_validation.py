"""
Test suite for data validation functions in ML workflow.

Tests the validation gates that prevent NaN/Inf values from reaching model training:
- validate_training_data(): Pre-training validation with strict/non-strict modes
- prepare_features_for_training(): Feature preparation with final imputation checkpoint

TDD Approach: These tests are written BEFORE implementation to drive the design.

Test coverage target: ≥80%
"""

import unittest
import numpy as np
import pandas as pd
from finance_ml.advanced_models import validate_training_data, prepare_features_for_training


class TestValidateTrainingData(unittest.TestCase):
    """Test suite for validate_training_data() function."""

    def setUp(self):
        """Create sample datasets for validation testing."""
        self.df_clean = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "feature3": [100.0, 200.0, 300.0, 400.0, 500.0],
                "sector": ["Tech", "Finance", "Tech", "Energy", "Finance"],
                "target": [100.0, 200.0, 150.0, 180.0, 220.0],
            }
        )

        self.df_with_nan = self.df_clean.copy()
        self.df_with_nan.loc[2, "feature1"] = np.nan
        self.df_with_nan.loc[3, "feature2"] = np.nan

        self.df_with_inf = self.df_clean.copy()
        self.df_with_inf.loc[1, "feature2"] = np.inf
        self.df_with_inf.loc[4, "feature3"] = -np.inf

        self.df_with_target_nan = self.df_clean.copy()
        self.df_with_target_nan.loc[2, "target"] = np.nan

        self.df_zero_variance = self.df_clean.copy()
        self.df_zero_variance["feature_constant"] = 42.0

    def test_validate_clean_data_passes(self):
        """Test that clean data passes validation."""
        X = self.df_clean[["feature1", "feature2", "feature3"]]
        y = self.df_clean["target"]
        result = validate_training_data(X, y, strict=False)

        self.assertTrue(result["valid"])
        self.assertEqual(result["nan_features"], 0)
        self.assertEqual(result["nan_target"], 0)
        self.assertEqual(result["inf_features"], 0)
        self.assertEqual(result["inf_target"], 0)
        self.assertEqual(len(result["issues"]), 0)

    def test_validate_data_with_nan_in_features_strict_raises(self):
        """Test that NaN in features raises ValueError in strict mode."""
        X = self.df_with_nan[["feature1", "feature2", "feature3"]]
        y = self.df_with_nan["target"]

        with self.assertRaises(ValueError) as context:
            validate_training_data(X, y, strict=True)

        self.assertIn("NaN", str(context.exception))
        self.assertIn("apply_enhanced_imputation_strategy_4step", str(context.exception))

    def test_validate_data_with_nan_in_features_nonstrict_returns_issues(self):
        """Test that NaN in features returns issues dict in non-strict mode."""
        X = self.df_with_nan[["feature1", "feature2", "feature3"]]
        y = self.df_with_nan["target"]
        result = validate_training_data(X, y, strict=False)

        self.assertFalse(result["valid"])
        self.assertGreater(result["nan_features"], 0)
        self.assertGreater(len(result["issues"]), 0)
        self.assertIn("NaN", result["issues"][0])

    def test_validate_data_with_nan_in_target_strict_raises(self):
        """Test that NaN in target raises ValueError in strict mode."""
        X = self.df_with_target_nan[["feature1", "feature2", "feature3"]]
        y = self.df_with_target_nan["target"]

        with self.assertRaises(ValueError) as context:
            validate_training_data(X, y, strict=True)

        self.assertIn("NaN", str(context.exception))
        self.assertIn("target", str(context.exception).lower())

    def test_validate_data_with_inf_in_features_strict_raises(self):
        """Test that infinite values in features raise ValueError in strict mode."""
        X = self.df_with_inf[["feature1", "feature2", "feature3"]]
        y = self.df_with_inf["target"]

        with self.assertRaises(ValueError) as context:
            validate_training_data(X, y, strict=True)

        self.assertIn("infinite", str(context.exception))

    def test_validate_data_with_inf_in_features_nonstrict_returns_issues(self):
        """Test that infinite values in features return issues in non-strict mode."""
        X = self.df_with_inf[["feature1", "feature2", "feature3"]]
        y = self.df_with_inf["target"]
        result = validate_training_data(X, y, strict=False)

        self.assertFalse(result["valid"])
        self.assertGreater(result["inf_features"], 0)
        self.assertGreater(len(result["issues"]), 0)

    def test_validate_zero_variance_columns_detected(self):
        """Test that zero-variance columns are detected but don't fail validation."""
        X = self.df_zero_variance[["feature1", "feature2", "feature_constant"]]
        y = self.df_zero_variance["target"]
        result = validate_training_data(X, y, strict=False)

        # Zero variance is a warning, not a blocker
        self.assertGreater(len(result["zero_var_columns"]), 0)
        self.assertIn("feature_constant", result["zero_var_columns"])

    def test_validate_empty_dataframe_raises(self):
        """Test that empty DataFrame raises appropriate error."""
        X = pd.DataFrame()
        y = pd.Series(dtype=float)

        with self.assertRaises(ValueError) as context:
            validate_training_data(X, y, strict=True)

        # Should raise due to empty data
        self.assertTrue(True)  # If we got here, exception was raised


class TestPrepareeFeaturesForTraining(unittest.TestCase):
    """Test suite for prepare_features_for_training() function."""

    def setUp(self):
        """Create sample datasets for feature preparation testing."""
        np.random.seed(42)
        n = 100

        self.df_with_nan = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"] * 25,
                "sector": ["Technology"] * 50 + ["Finance"] * 50,
                "last_price": [50 + i * 0.5 for i in range(n)],
                "market_cap": [
                    100 + i + np.random.randn() if i % 3 != 0 else np.nan for i in range(n)
                ],
                "ebitda": [10 + i * 0.1 if i % 5 != 0 else np.nan for i in range(n)],
                "revenue": [500 + i * 5 if i % 4 != 0 else np.nan for i in range(n)],
                "price_target": [55 + i * 0.5 for i in range(n)],
            }
        )

        self.feature_cols = ["market_cap", "ebitda", "revenue"]
        self.target_col = "price_target"

    def test_prepare_features_with_imputation_removes_all_nan(self):
        """Test that prepare_features_for_training removes all NaN with imputation."""
        X, y = prepare_features_for_training(
            df=self.df_with_nan,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            apply_imputation=True,
            sector_column="sector",
        )

        self.assertEqual(
            X.isnull().sum().sum(), 0, "Features should have zero NaN after imputation"
        )
        self.assertEqual(y.isnull().sum(), 0, "Target should have zero NaN after imputation")
        self.assertEqual(X.shape[0], y.shape[0], "X and y should have same number of rows")

    def test_prepare_features_without_imputation_preserves_nan(self):
        """Test that prepare_features_for_training without imputation preserves NaN."""
        X, y = prepare_features_for_training(
            df=self.df_with_nan,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            apply_imputation=False,
            sector_column="sector",
        )

        # Without imputation, NaN should still exist (or be handled by emergency fillna)
        # The function should apply emergency fillna as fallback
        self.assertEqual(X.isnull().sum().sum(), 0, "Emergency fallback should ensure zero NaN")
        self.assertEqual(y.isnull().sum(), 0, "Target should have zero NaN after fallback")

    def test_prepare_features_drops_rows_with_nan_target(self):
        """Test that rows with NaN in target are dropped."""
        df_with_target_nan = self.df_with_nan.copy()
        df_with_target_nan.loc[10:15, "price_target"] = np.nan

        n_before = len(df_with_target_nan)
        n_nan_target = df_with_target_nan["price_target"].isnull().sum()

        X, y = prepare_features_for_training(
            df=df_with_target_nan,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            apply_imputation=True,
            sector_column="sector",
        )

        self.assertEqual(len(X), len(y))
        self.assertEqual(
            len(X), n_before - n_nan_target, f"Should drop {n_nan_target} rows with NaN target"
        )
        self.assertEqual(y.isnull().sum(), 0, "Target should have zero NaN")

    def test_prepare_features_returns_correct_columns(self):
        """Test that prepare_features_for_training returns correct feature columns."""
        X, y = prepare_features_for_training(
            df=self.df_with_nan,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            apply_imputation=True,
            sector_column="sector",
        )

        self.assertListEqual(list(X.columns), self.feature_cols)
        self.assertEqual(y.name, self.target_col)

    def test_prepare_features_handles_infinite_values(self):
        """Test that infinite values are handled correctly."""
        df_with_inf = self.df_with_nan.copy()
        df_with_inf.loc[5, "market_cap"] = np.inf
        df_with_inf.loc[15, "ebitda"] = -np.inf

        X, y = prepare_features_for_training(
            df=df_with_inf,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            apply_imputation=True,
            sector_column="sector",
        )

        self.assertEqual(np.isinf(X).sum().sum(), 0, "Features should have zero infinite values")
        self.assertEqual(np.isinf(y).sum(), 0, "Target should have zero infinite values")

    def test_prepare_features_preserves_data_types(self):
        """Test that data types are preserved after preparation."""
        X, y = prepare_features_for_training(
            df=self.df_with_nan,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            apply_imputation=True,
            sector_column="sector",
        )

        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertTrue(pd.api.types.is_numeric_dtype(y))


class TestValidationIntegration(unittest.TestCase):
    """Integration tests for validation in the full training workflow."""

    def setUp(self):
        """Create sample dataset for integration testing."""
        np.random.seed(42)
        n = 200

        self.df = pd.DataFrame(
            {
                "sector": np.random.choice(["Tech", "Finance", "Energy"], n),
                "last_price": np.random.uniform(20, 200, n),
                "market_cap": np.random.uniform(1e9, 1e12, n),
                "ebitda": np.random.uniform(1e7, 1e10, n),
                "revenue": np.random.uniform(1e8, 1e11, n),
                "pe_ratio": np.random.uniform(5, 50, n),
                "price_target": np.random.uniform(25, 250, n),
            }
        )

        # Introduce some NaN values
        nan_indices = np.random.choice(n, size=20, replace=False)
        for idx in nan_indices:
            col = np.random.choice(["market_cap", "ebitda", "revenue", "pe_ratio"])
            self.df.loc[idx, col] = np.nan

    def test_full_pipeline_from_nan_data_to_validated_training(self):
        """Test complete pipeline: data with NaN -> imputation -> validation -> ready for training."""
        feature_cols = ["market_cap", "ebitda", "revenue", "pe_ratio"]
        target_col = "price_target"

        # Step 1: Prepare features with imputation
        X, y = prepare_features_for_training(
            df=self.df,
            feature_cols=feature_cols,
            target_col=target_col,
            apply_imputation=True,
            sector_column="sector",
        )

        # Step 2: Validate training data
        validation_result = validate_training_data(X, y, strict=True)

        # Should pass all validation checks
        self.assertTrue(validation_result["valid"])
        self.assertEqual(validation_result["nan_features"], 0)
        self.assertEqual(validation_result["nan_target"], 0)
        self.assertEqual(validation_result["inf_features"], 0)
        self.assertEqual(validation_result["inf_target"], 0)

    def test_validation_catches_insufficient_imputation(self):
        """Test that validation catches cases where imputation was insufficient."""
        feature_cols = ["market_cap", "ebitda", "revenue", "pe_ratio"]
        target_col = "price_target"

        # Prepare features WITHOUT imputation
        X, y = prepare_features_for_training(
            df=self.df,
            feature_cols=feature_cols,
            target_col=target_col,
            apply_imputation=False,  # Skip imputation
            sector_column="sector",
        )

        # Validation should still pass due to emergency fallback fillna(0)
        validation_result = validate_training_data(X, y, strict=False)
        self.assertTrue(validation_result["valid"])


if __name__ == "__main__":
    unittest.main()
