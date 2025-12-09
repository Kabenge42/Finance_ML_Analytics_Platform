"""
Test suite for Task 7: Unified Test Data Alignment

Tests for align_features_to_model() function that automatically aligns test
features to trained model's expected feature set.

Phase 9.5 TDD Implementation Plan - Task 7 (High Priority)
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from finance_ml.ml_workflow.regression.dataset import align_features_to_model, predict_with_model


class TestFeatureAlignment(unittest.TestCase):
    """Test suite for automatic feature alignment to trained models."""

    def test_align_test_features_to_model(self):
        """Test features should be aligned to model's expected features."""
        # Given: Trained model expecting specific features
        X_train = pd.DataFrame(
            {
                "feature_a": np.random.randn(100),
                "feature_b": np.random.randn(100),
                "feature_c": np.random.randn(100),
            }
        )
        y_train = np.random.randn(100)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # And: Test data with different features
        X_test = pd.DataFrame(
            {
                "feature_a": np.random.randn(20),
                "feature_b": np.random.randn(20),
                "feature_d": np.random.randn(20),  # Extra feature
                # Missing: feature_c
            }
        )

        # When: Align test features to model
        X_test_aligned = align_features_to_model(X_test, model)

        # Then: Features match model's expectations
        self.assertListEqual(list(X_test_aligned.columns), list(model.feature_names_in_))
        self.assertEqual(X_test_aligned.shape[1], X_train.shape[1])

    def test_align_fills_missing_with_zero(self):
        """Missing features should be filled with zero."""
        # Given: Model trained on features [a, b, c]
        X_train = pd.DataFrame(
            {"feature_a": [1, 2, 3], "feature_b": [4, 5, 6], "feature_c": [7, 8, 9]}
        )
        y_train = [10, 11, 12]

        model = LinearRegression()
        model.fit(X_train, y_train)

        # And: Test data missing feature_c
        X_test = pd.DataFrame({"feature_a": [1.5], "feature_b": [4.5]})

        # When: Align features
        X_test_aligned = align_features_to_model(X_test, model)

        # Then: Missing feature_c filled with 0
        self.assertIn("feature_c", X_test_aligned.columns)
        self.assertEqual(X_test_aligned["feature_c"].iloc[0], 0.0)

        # And: Prediction doesn't raise error
        prediction = model.predict(X_test_aligned)
        self.assertEqual(len(prediction), 1)

    def test_align_removes_extra_columns(self):
        """Extra features not in training should be removed."""
        # Given: Model trained on features [a, b]
        X_train = pd.DataFrame(
            {"feature_a": np.random.randn(100), "feature_b": np.random.randn(100)}
        )
        y_train = np.random.randn(100)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # And: Test data with extra features
        X_test = pd.DataFrame(
            {
                "feature_a": np.random.randn(20),
                "feature_b": np.random.randn(20),
                "feature_z": np.random.randn(20),  # Extra
                "feature_y": np.random.randn(20),  # Extra
            }
        )

        # When: Align features
        X_test_aligned = align_features_to_model(X_test, model)

        # Then: Extra features removed
        self.assertNotIn("feature_z", X_test_aligned.columns)
        self.assertNotIn("feature_y", X_test_aligned.columns)
        self.assertEqual(len(X_test_aligned.columns), 2)

    def test_predict_with_model_auto_alignment(self):
        """predict_with_model wrapper should automatically align features."""
        # Given: Trained model
        X_train = pd.DataFrame(
            {"feature_a": np.random.randn(100), "feature_b": np.random.randn(100)}
        )
        y_train = X_train["feature_a"] * 2 + X_train["feature_b"] + np.random.randn(100) * 0.1

        model = LinearRegression()
        model.fit(X_train, y_train)

        # And: Test data with mismatched features
        X_test = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_c": [3.0, 4.0]})  # Wrong feature

        # When: Predict with auto_align=True
        predictions = predict_with_model(model, X_test, auto_align=True)

        # Then: Prediction succeeds without error
        self.assertEqual(len(predictions), 2)
        self.assertTrue(np.all(np.isfinite(predictions)))


if __name__ == "__main__":
    unittest.main()
