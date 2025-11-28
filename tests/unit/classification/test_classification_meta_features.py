"""Tests for classification meta-feature export and integration (Task 9.9.6).

Validates that classification probabilities are exported in a standardized
schema and integrated into regression-ready dataframes without index
misalignment or leakage.
"""

import unittest

import numpy as np
import pandas as pd


class TestClassificationMetaFeatures(unittest.TestCase):
    """Tests for export_classification_probabilities and integration helper."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.n_samples = 50

        # Simple stock universe
        self.df = pd.DataFrame(
            {
                "ticker": [f"T{i:03d}" for i in range(self.n_samples)],
                "sector": np.random.choice(["Tech", "Finance"], size=self.n_samples),
                "last_price": np.random.uniform(10.0, 200.0, size=self.n_samples),
                "price_target": np.random.uniform(10.0, 200.0, size=self.n_samples),
            }
        )

        # Synthetic 5-class probabilities (0=Strong Negative, 4=Strong Positive)
        raw = np.random.rand(self.n_samples, 5)
        self.y_proba = raw / raw.sum(axis=1, keepdims=True)
        self.y_true = np.random.randint(0, 5, size=self.n_samples)
        self.y_pred = self.y_proba.argmax(axis=1)

    def test_export_classification_probabilities_basic_shape(self):
        """export_classification_probabilities should return standardized 5-class columns."""

        from finance_ml.ml_workflow.classification.evaluation import (
            export_classification_probabilities,
        )

        probs_df = export_classification_probabilities(
            self.y_true, self.y_pred, self.y_proba, index=self.df.index
        )

        self.assertEqual(len(probs_df), self.n_samples)
        expected_cols = {
            "event_prob_strong_negative",
            "event_prob_negative",
            "event_prob_neutral",
            "event_prob_positive",
            "event_prob_strong_positive",
            "event_class_predicted",
            "event_confidence",
        }
        self.assertTrue(expected_cols.issubset(set(probs_df.columns)))

    def test_export_classification_probabilities_invalid_shape_raises(self):
        """Invalid probability shape should raise ValueError."""

        from finance_ml.ml_workflow.classification.evaluation import (
            export_classification_probabilities,
        )

        bad_proba = np.random.rand(self.n_samples, 2)
        with self.assertRaises(ValueError):
            export_classification_probabilities(
                self.y_true, self.y_pred, bad_proba, index=self.df.index
            )

    def test_integrate_classification_features_combines_columns(self):
        """Integration helper should append classification columns to df."""

        from finance_ml.ml_workflow.regression.dataset import (
            integrate_classification_features,
        )

        df_enhanced = integrate_classification_features(self.df, self.y_proba)

        self.assertEqual(len(df_enhanced), len(self.df))
        # Check that original columns are preserved
        for col in ["ticker", "sector", "last_price", "price_target"]:
            self.assertIn(col, df_enhanced.columns)

        # Classification columns should be present (5-class probabilities + meta)
        for col in [
            "event_prob_strong_negative",
            "event_prob_negative",
            "event_prob_neutral",
            "event_prob_positive",
            "event_prob_strong_positive",
            "event_class_predicted",
            "event_confidence",
        ]:
            self.assertIn(col, df_enhanced.columns)

    def test_integration_roundtrip_alignment(self):
        """Index alignment must be preserved through integration."""

        from finance_ml.ml_workflow.regression.dataset import (
            integrate_classification_features,
        )

        df_enhanced = integrate_classification_features(self.df, self.y_proba)

        # After reset_index in the helper, order should still correspond row-wise
        self.assertEqual(len(df_enhanced), len(self.df))
        # ticker column should match original order
        self.assertListEqual(df_enhanced["ticker"].tolist(), self.df["ticker"].tolist())

    def test_prepare_regression_data_sees_classification_features(self):
        """prepare_regression_data should treat event_prob_* as classification features."""

        from finance_ml.ml_workflow.regression.dataset import (
            integrate_classification_features,
            prepare_regression_data,
        )

        df_enhanced = integrate_classification_features(self.df, self.y_proba)

        X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(
            df_enhanced, target_col="price_target"
        )

        cls_feats = feature_info.get("classification_features", [])
        self.assertTrue(
            any(col.startswith("event_prob_") for col in cls_feats),
            msg=f"Expected classification_features to include event_prob_*, got {cls_feats}",
        )


if __name__ == "__main__":
    unittest.main()
