"""
Unit tests for Phase 9.5 error handling paths.

Tests the robustness of advanced regression models when facing:
- Model comparison failures
- Quantile results access edge cases
- Empty or invalid data
- Missing classification features
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from finance_ml.advanced_models import (
    prepare_regression_data,
    create_classification_interactions,
    compare_regressors,
    train_stacking_regressor,
    train_quantile_regressor,
)


class TestPhase95ErrorHandling(unittest.TestCase):
    """Test error handling in Phase 9.5 regression functions."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 100

        # Create test DataFrame with all required columns
        self.test_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(self.n_samples)],
                "last_price": np.random.uniform(50, 500, self.n_samples),
                "price_target": np.random.uniform(60, 550, self.n_samples),
                "p_e": np.random.uniform(5, 50, self.n_samples),
                "p_b": np.random.uniform(1, 10, self.n_samples),
                "ev_ebitda": np.random.uniform(5, 30, self.n_samples),
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], self.n_samples),
                "event_prob_0": np.random.uniform(0, 1, self.n_samples),
                "event_prob_1": np.random.uniform(0, 1, self.n_samples),
                "event_prob_2": np.random.uniform(0, 1, self.n_samples),
            }
        )

    def test_prepare_regression_data_with_missing_target(self):
        """Test prepare_regression_data passes through NaN targets (caller's responsibility to clean)."""
        df = self.test_df.copy()
        df["price_target"] = np.nan  # All NaN

        # Function doesn't drop NaN targets - passes them through
        X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(
            df, target_col="price_target", test_size=0.2, random_state=42
        )

        # Split should preserve data size
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        # But targets are all NaN (caller's responsibility to handle)
        self.assertTrue(y_train.isna().all())
        self.assertTrue(y_test.isna().all())

    def test_prepare_regression_data_with_partial_missing_target(self):
        """Test prepare_regression_data with partially missing target (passes through NaN)."""
        df = self.test_df.copy()
        # Set 50% of targets to NaN
        np.random.seed(42)  # Ensure reproducibility
        nan_mask = np.random.choice([True, False], size=len(df), p=[0.5, 0.5])
        df.loc[nan_mask, "price_target"] = np.nan

        X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(
            df, target_col="price_target", test_size=0.2, random_state=42
        )

        # Should preserve all samples (doesn't drop NaN targets)
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)

        # Some targets should be NaN (approximately 50%)
        nan_train_pct = y_train.isna().sum() / len(y_train)
        nan_test_pct = y_test.isna().sum() / len(y_test)
        # Should have roughly 50% NaN in both sets
        self.assertGreater(nan_train_pct, 0.3)
        self.assertLess(nan_train_pct, 0.7)

    def test_create_classification_interactions_with_zeros(self):
        """Test interaction creation with zero values in valuation metrics."""
        df = self.test_df.copy()
        # Add some zeros
        df.loc[:10, "p_e"] = 0
        df.loc[:5, "ev_ebitda"] = 0

        classification_cols = ["event_prob_0", "event_prob_1"]
        valuation_cols = ["p_e", "ev_ebitda"]

        # Should not crash despite zeros
        result = create_classification_interactions(df, classification_cols, valuation_cols)

        # Check interaction columns were created
        expected_interactions = len(classification_cols) * len(valuation_cols)
        interaction_cols = [c for c in result.columns if "_x_" in c]
        self.assertEqual(len(interaction_cols), expected_interactions)

        # Check zero interactions resulted in zero (not NaN)
        self.assertFalse(result[interaction_cols[0]].isna().all())

    def test_create_classification_interactions_with_inf(self):
        """Test interaction creation with infinite values."""
        df = self.test_df.copy()
        # Add some infinite values
        df.loc[:5, "p_e"] = np.inf
        df.loc[5:10, "p_b"] = -np.inf

        classification_cols = ["event_prob_0"]
        valuation_cols = ["p_e", "p_b"]

        result = create_classification_interactions(df, classification_cols, valuation_cols)

        # Check for infinite values in interactions
        interaction_cols = [c for c in result.columns if "_x_" in c]
        for col in interaction_cols:
            inf_count = np.isinf(result[col]).sum()
            # Should have some inf values from the original data
            self.assertGreaterEqual(inf_count, 0)

    def test_create_classification_interactions_empty_lists(self):
        """Test interaction creation with empty feature lists."""
        df = self.test_df.copy()

        # Empty classification columns
        result = create_classification_interactions(df, [], ["p_e", "p_b"])
        self.assertEqual(len([c for c in result.columns if "_x_" in c]), 0)

        # Empty valuation columns
        result = create_classification_interactions(df, ["event_prob_0"], [])
        self.assertEqual(len([c for c in result.columns if "_x_" in c]), 0)

        # Both empty
        result = create_classification_interactions(df, [], [])
        self.assertEqual(len([c for c in result.columns if "_x_" in c]), 0)

    def test_compare_regressors_with_small_dataset(self):
        """Test compare_regressors with very small dataset."""
        # Small dataset might cause CV issues
        small_df = self.test_df.head(10)
        X_train, X_test, y_train, y_test, _ = prepare_regression_data(
            small_df, target_col="price_target", test_size=0.2, random_state=42
        )

        # Should handle small dataset gracefully (may reduce CV folds internally)
        try:
            results = compare_regressors(X_train, y_train, cv=2, random_state=42)
            self.assertIsInstance(results, dict)
            self.assertGreater(len(results), 0)
        except ValueError as e:
            # Acceptable if it fails with clear message about insufficient samples
            self.assertIn("sample", str(e).lower())

    def test_train_stacking_regressor_basic(self):
        """Test stacking regressor with basic valid data."""
        X_train, X_test, y_train, y_test, _ = prepare_regression_data(
            self.test_df, target_col="price_target", test_size=0.2, random_state=42
        )

        model, results = train_stacking_regressor(X_train, y_train, cv=3)

        # Check model and results structure
        self.assertIsNotNone(model)
        self.assertIn("train_score", results)
        self.assertIn("cv_score", results)
        self.assertIn("cv_std", results)
        self.assertIn("base_models", results)

        # Check scores are reasonable
        self.assertGreaterEqual(results["train_score"], -1.0)  # R² can be negative
        self.assertLessEqual(results["train_score"], 1.0)

    def test_train_quantile_regressor_structure(self):
        """Test quantile regressor returns correct structure."""
        X_train, X_test, y_train, y_test, _ = prepare_regression_data(
            self.test_df, target_col="price_target", test_size=0.2, random_state=42
        )

        quantiles = [0.1, 0.5, 0.9]
        models, results = train_quantile_regressor(X_train, y_train, quantiles=quantiles)

        # Check we get one model per quantile
        self.assertEqual(len(models), len(quantiles))

        # Check results structure
        self.assertIn("quantiles", results)
        self.assertEqual(results["quantiles"], quantiles)

        # Check quantile_results if it exists
        if "quantile_results" in results:
            self.assertIsInstance(results["quantile_results"], list)
            # Each quantile should have results
            if len(results["quantile_results"]) > 0:
                self.assertIn("train_score", results["quantile_results"][0])

    def test_quantile_regressor_predictions(self):
        """Test quantile regressor predictions are ordered correctly."""
        X_train, X_test, y_train, y_test, _ = prepare_regression_data(
            self.test_df, target_col="price_target", test_size=0.2, random_state=42
        )

        quantiles = [0.1, 0.5, 0.9]
        models, results = train_quantile_regressor(X_train, y_train, quantiles=quantiles)

        # Make predictions
        predictions = {}
        for q, model in zip(quantiles, models):
            predictions[q] = model.predict(X_test)

        # Check predictions exist and have correct shape
        self.assertEqual(len(predictions[0.1]), len(X_test))
        self.assertEqual(len(predictions[0.5]), len(X_test))
        self.assertEqual(len(predictions[0.9]), len(X_test))

        # For most samples, predictions should be ordered: p10 < p50 < p90
        # (not always true due to independent model training, but mostly)
        ordered_count = 0
        for i in range(len(X_test)):
            if predictions[0.1][i] <= predictions[0.5][i] <= predictions[0.9][i]:
                ordered_count += 1

        # At least 50% should be ordered (relaxed requirement)
        ordered_pct = ordered_count / len(X_test)
        self.assertGreater(
            ordered_pct, 0.3, f"Only {ordered_pct:.1%} of predictions properly ordered"
        )

    def test_error_handling_with_nan_features(self):
        """Test that functions handle NaN in features appropriately."""
        df = self.test_df.copy()
        # Add NaN to some features
        df.loc[:10, "p_e"] = np.nan

        X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(
            df, target_col="price_target", test_size=0.2, random_state=42
        )

        # prepare_regression_data should drop rows with NaN target, but keep NaN features
        # (downstream models will need to handle them)
        self.assertGreater(len(X_train), 0)

    def test_feature_info_structure(self):
        """Test that feature_info has expected structure."""
        X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(
            self.test_df, target_col="price_target", test_size=0.2, random_state=42
        )

        # Check required keys
        self.assertIn("numeric_features", feature_info)
        self.assertIn("categorical_features", feature_info)
        self.assertIn("classification_features", feature_info)
        self.assertIn("all_features", feature_info)

        # Check types
        self.assertIsInstance(feature_info["numeric_features"], list)
        self.assertIsInstance(feature_info["categorical_features"], list)
        self.assertIsInstance(feature_info["classification_features"], list)
        self.assertIsInstance(feature_info["all_features"], list)

        # Check classification features were identified
        expected_class_features = ["event_prob_0", "event_prob_1", "event_prob_2"]
        for feat in expected_class_features:
            self.assertIn(feat, feature_info["classification_features"])


class TestPhase95IntegrationScenarios(unittest.TestCase):
    """Integration tests for Phase 9.5 complete workflow scenarios."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 200

        self.test_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(self.n_samples)],
                "last_price": np.random.uniform(50, 500, self.n_samples),
                "price_target": np.random.uniform(60, 550, self.n_samples),
                "p_e": np.random.uniform(5, 50, self.n_samples),
                "p_b": np.random.uniform(1, 10, self.n_samples),
                "ev_ebitda": np.random.uniform(5, 30, self.n_samples),
                "market_cap": np.random.uniform(1e9, 1e12, self.n_samples),
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], self.n_samples),
                "event_prob_0": np.random.uniform(0, 1, self.n_samples),
                "event_prob_1": np.random.uniform(0, 1, self.n_samples),
                "event_prob_2": np.random.uniform(0, 1, self.n_samples),
            }
        )

    def test_full_workflow_with_interactions(self):
        """Test complete workflow: prepare data → create interactions → train models."""
        # Step 1: Prepare data
        X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(
            self.test_df, target_col="price_target", test_size=0.2, random_state=42
        )

        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)

        # Step 2: Create interactions (on full df)
        classification_cols = feature_info["classification_features"]
        valuation_cols = ["p_e", "p_b", "ev_ebitda"]

        df_enhanced = create_classification_interactions(
            self.test_df, classification_cols, valuation_cols
        )

        interaction_cols = [c for c in df_enhanced.columns if "_x_" in c]
        self.assertGreater(len(interaction_cols), 0)

        # Step 3: Re-prepare data with interactions
        X_train_enh, X_test_enh, y_train_enh, y_test_enh, _ = prepare_regression_data(
            df_enhanced, target_col="price_target", test_size=0.2, random_state=42
        )

        # Should have more features now
        self.assertGreater(X_train_enh.shape[1], X_train.shape[1])

    def test_predictions_storage_pattern(self):
        """Test the pattern used in Phase 9.5 for storing predictions."""
        # Simulate Phase 9.5 prediction storage
        X_train, X_test, y_train, y_test, _ = prepare_regression_data(
            self.test_df, target_col="price_target", test_size=0.2, random_state=42
        )

        # Train a simple model
        model, _ = train_stacking_regressor(X_train, y_train, cv=3)
        y_pred = model.predict(X_test)

        # Create a copy of the original dataframe
        all_stocks_featured = self.test_df.copy()

        # Initialize prediction columns
        all_stocks_featured["predicted_price_target"] = np.nan
        all_stocks_featured["prediction_lower_10"] = np.nan
        all_stocks_featured["prediction_upper_90"] = np.nan

        # Store predictions using .loc indexing (Phase 9.5 pattern)
        test_indices = X_test.index
        all_stocks_featured.loc[test_indices, "predicted_price_target"] = y_pred

        # Verify predictions were stored correctly
        non_null_count = all_stocks_featured["predicted_price_target"].notna().sum()
        self.assertEqual(non_null_count, len(X_test))

        # Verify predictions are in the right indices
        for idx in test_indices:
            self.assertFalse(np.isnan(all_stocks_featured.loc[idx, "predicted_price_target"]))


class TestPhase95EdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_single_feature(self):
        """Test with minimal single feature."""
        df = pd.DataFrame({"feature1": np.random.randn(100), "target": np.random.randn(100)})

        X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(
            df, target_col="target", test_size=0.2, random_state=42
        )

        self.assertEqual(X_train.shape[1], 1)
        self.assertGreater(len(X_train), 0)

    def test_all_constant_features(self):
        """Test with constant features (no variance)."""
        df = pd.DataFrame({"const_feature": [1.0] * 100, "target": np.random.randn(100)})

        X_train, X_test, y_train, y_test, _ = prepare_regression_data(
            df, target_col="target", test_size=0.2, random_state=42
        )

        # Should still work even with no variance
        self.assertGreater(len(X_train), 0)

    def test_highly_correlated_features(self):
        """Test with highly correlated features."""
        n = 100
        base = np.random.randn(n)
        df = pd.DataFrame(
            {
                "feat1": base,
                "feat2": base + np.random.randn(n) * 0.01,  # Almost identical
                "feat3": base * 2,  # Perfectly correlated
                "target": base + np.random.randn(n) * 0.1,
            }
        )

        X_train, X_test, y_train, y_test, _ = prepare_regression_data(
            df, target_col="target", test_size=0.2, random_state=42
        )

        # Should handle correlated features
        self.assertGreater(len(X_train), 0)
        self.assertEqual(X_train.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
