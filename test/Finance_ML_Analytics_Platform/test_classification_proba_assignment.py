"""
Tests for classification probability assignment to full dataset.

TDD Implementation for Issue: Size mismatch when assigning classification
probabilities to full dataset DataFrame.

Problem: train_event_classifier returns y_proba only for test set (internal split),
but notebook code tries to assign these probabilities to the full dataset.

Solution Options:
1. Use model.predict_proba(X_full) to get probabilities for all rows
2. Track test indices and assign only to matching rows
3. Add helper function to predict on arbitrary dataset

These tests follow strict TDD: write failing tests first, then implement.
"""

import unittest

import numpy as np
import pandas as pd

try:
    from finance_ml.ml_workflow.classification import (
        create_event_labels,
        train_event_classifier,
        export_classification_probabilities,
    )

    HAVE_CLASSIFICATION = True
except ImportError:
    HAVE_CLASSIFICATION = False


@unittest.skipIf(not HAVE_CLASSIFICATION, "Classification module not available")
class TestClassificationProbaAssignment(unittest.TestCase):
    """Test classification probability assignment to full dataset."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic dataset for testing."""
        np.random.seed(42)
        n = 200  # Full dataset size
        cls.n_samples = n

        cls.df = pd.DataFrame(
            {
                "ticker": [f"T{i:03d}" for i in range(n)],
                "sector": np.random.choice(["Tech", "Energy", "Finance", "Healthcare"], n),
                "region": np.random.choice(["US", "EU", "APAC"], n),
                "last_price": np.random.uniform(10, 100, n),
                "price_target": np.random.uniform(10, 120, n),
                "analyst_rating": np.random.uniform(1, 5, n),
                "feature_1": np.random.randn(n),
                "feature_2": np.random.randn(n),
                "feature_3": np.random.randn(n),
            }
        )

        # Create event labels
        cls.labels = create_event_labels(cls.df, method="price_momentum")

        # Prepare features (exclude non-numeric and target-related columns)
        exclude_cols = ["ticker", "sector", "region", "price_target"]
        feature_cols = [
            col
            for col in cls.df.columns
            if col not in exclude_cols
            and cls.df[col].dtype in ["float64", "float32", "int64", "int32"]
        ]
        cls.X = cls.df[feature_cols].copy()
        cls.y = cls.labels

    def test_classifier_returns_model_that_can_predict_on_full_dataset(self):
        """Test that returned model can predict probabilities on full dataset."""
        # Train classifier (internally splits data)
        result = train_event_classifier(self.X, self.y, model="lightgbm")

        # Get the trained model
        model = result["model"]
        self.assertIsNotNone(model, "Model should be returned")

        # Model should be able to predict on the FULL dataset
        # This is the key fix: use model.predict_proba(X_full)
        y_proba_full = model.predict_proba(self.X)

        # Verify shape matches full dataset
        self.assertEqual(
            y_proba_full.shape[0],
            self.n_samples,
            f"Probabilities should have {self.n_samples} rows (full dataset), "
            f"got {y_proba_full.shape[0]}",
        )

        # Verify probabilities sum to 1.0
        prob_sums = y_proba_full.sum(axis=1)
        self.assertTrue(
            np.allclose(prob_sums, 1.0, atol=0.01), "Probabilities must sum to 1.0 for each row"
        )

    def test_proba_assignment_to_dataframe_with_matching_indices(self):
        """Test assigning probabilities to DataFrame preserves index alignment."""
        # Train classifier
        result = train_event_classifier(self.X, self.y, model="lightgbm")
        model = result["model"]

        # Predict on full dataset
        y_proba_full = model.predict_proba(self.X)
        n_classes = y_proba_full.shape[1]

        # Create probability column names
        prob_columns = [f"event_prob_class_{i}" for i in range(n_classes)]

        # Create a copy of the DataFrame for assignment
        df_with_proba = self.df.copy()

        # Assign probabilities - this should work without size mismatch
        for i, col in enumerate(prob_columns):
            if i < n_classes:
                df_with_proba[col] = y_proba_full[:, i]

        # Verify assignment worked
        self.assertEqual(len(df_with_proba), self.n_samples)

        # Verify all probability columns exist
        for col in prob_columns:
            self.assertIn(col, df_with_proba.columns)

        # Verify probabilities sum to 1.0
        prob_sums = df_with_proba[prob_columns].sum(axis=1)
        self.assertTrue(
            np.allclose(prob_sums, 1.0, atol=0.01), "Probabilities must sum to 1.0 for each row"
        )

    def test_proba_assignment_with_custom_index(self):
        """Test probability assignment works with non-default DataFrame index."""
        # Create DataFrame with custom index
        df_custom = self.df.copy()
        df_custom.index = [f"stock_{i}" for i in range(len(df_custom))]

        X_custom = self.X.copy()
        X_custom.index = df_custom.index

        # Train classifier
        result = train_event_classifier(X_custom, self.y, model="lightgbm")
        model = result["model"]

        # Predict on full dataset
        y_proba_full = model.predict_proba(X_custom)
        n_classes = y_proba_full.shape[1]

        # Assign probabilities
        prob_columns = [f"event_prob_class_{i}" for i in range(n_classes)]
        for i, col in enumerate(prob_columns):
            if i < n_classes:
                df_custom[col] = y_proba_full[:, i]

        # Verify index preserved
        self.assertEqual(list(df_custom.index), [f"stock_{i}" for i in range(len(df_custom))])

        # Verify probabilities valid
        prob_sums = df_custom[prob_columns].sum(axis=1)
        self.assertTrue(np.allclose(prob_sums, 1.0, atol=0.01))

    def test_export_classification_probabilities_function(self):
        """Test the export_classification_probabilities helper function."""
        # Train classifier
        result = train_event_classifier(self.X, self.y, model="lightgbm")
        model = result["model"]

        # Predict on full dataset
        y_proba_full = model.predict_proba(self.X)
        y_pred_full = model.predict(self.X)

        # The function expects exactly 5 classes for the 5-class event system
        # If model has fewer classes, we need to pad or skip this test
        if y_proba_full.shape[1] != 5:
            self.skipTest(f"Model has {y_proba_full.shape[1]} classes, expected 5")

        # Use the export function with correct signature:
        # export_classification_probabilities(y_true, y_pred, y_proba, index)
        df_exported = export_classification_probabilities(
            y_true=self.y,
            y_pred=y_pred_full,
            y_proba=y_proba_full,
            index=self.df.index,
        )

        # Verify export worked
        self.assertEqual(len(df_exported), self.n_samples)

        # Check probability columns exist with correct names (5-class system)
        expected_cols = [
            "event_prob_strong_negative",
            "event_prob_negative",
            "event_prob_neutral",
            "event_prob_positive",
            "event_prob_strong_positive",
        ]
        for col in expected_cols:
            self.assertIn(col, df_exported.columns)

    def test_y_proba_from_result_is_test_set_only(self):
        """Verify that y_proba in result is test-set only (documenting current behavior)."""
        # Train classifier
        result = train_event_classifier(self.X, self.y, model="lightgbm")

        # y_proba in result is for test set only (internal 80/20 split)
        y_proba_result = result.get("y_proba")

        if y_proba_result is not None:
            # This should be ~20% of full dataset (test set)
            expected_test_size = int(self.n_samples * 0.2)
            actual_size = y_proba_result.shape[0]

            # Allow some tolerance for rounding
            self.assertLess(
                actual_size,
                self.n_samples,
                "y_proba in result should be smaller than full dataset (test set only)",
            )
            self.assertAlmostEqual(
                actual_size / self.n_samples,
                0.2,
                delta=0.05,
                msg="y_proba should be approximately 20% of full dataset",
            )

    def test_non_negativity_of_probabilities(self):
        """Test that all probabilities are non-negative."""
        result = train_event_classifier(self.X, self.y, model="lightgbm")
        model = result["model"]

        y_proba_full = model.predict_proba(self.X)

        self.assertTrue((y_proba_full >= 0).all(), "All probabilities must be non-negative")
        self.assertTrue((y_proba_full <= 1).all(), "All probabilities must be <= 1")


@unittest.skipIf(not HAVE_CLASSIFICATION, "Classification module not available")
class TestClassificationProbaIntegration(unittest.TestCase):
    """Integration tests for classification probability workflow."""

    def test_full_workflow_probability_assignment(self):
        """Test complete workflow: train → predict full → assign → validate."""
        np.random.seed(42)
        n = 150

        # Create dataset
        df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n)],
                "sector": np.random.choice(["Tech", "Finance", "Energy"], n),
                "region": np.random.choice(["US", "EU"], n),
                "last_price": np.random.uniform(10, 100, n),
                "price_target": np.random.uniform(10, 120, n),
                "feature_a": np.random.randn(n),
                "feature_b": np.random.randn(n),
            }
        )

        # Create labels
        labels = create_event_labels(df, method="price_momentum")

        # Prepare features
        feature_cols = ["last_price", "feature_a", "feature_b"]
        X = df[feature_cols].copy()

        # Train classifier
        result = train_event_classifier(X, labels, model="lightgbm")
        model = result["model"]

        # CORRECT APPROACH: Predict on FULL dataset
        y_proba_full = model.predict_proba(X)

        # Define probability columns (5-class system)
        prob_columns = [
            "event_prob_strong_negative",
            "event_prob_negative",
            "event_prob_neutral",
            "event_prob_positive",
            "event_prob_strong_positive",
        ]

        # Create classification DataFrame
        all_stocks_classification = df.copy()

        # Assign probabilities to ALL rows
        for i, col in enumerate(prob_columns):
            if i < y_proba_full.shape[1]:
                all_stocks_classification[col] = y_proba_full[:, i]

        # Validate: probabilities sum to 1.0
        actual_prob_cols = [c for c in prob_columns if c in all_stocks_classification.columns]
        prob_sums = all_stocks_classification[actual_prob_cols].sum(axis=1)

        self.assertTrue(np.allclose(prob_sums, 1.0, atol=0.01), "Probabilities must sum to 1.0")

        # Validate: no size mismatch
        self.assertEqual(
            len(all_stocks_classification), n, "DataFrame should maintain original size"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
