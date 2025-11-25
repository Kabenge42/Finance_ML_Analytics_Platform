import unittest
import logging
import numpy as np
import pandas as pd

from finance_ml.advanced_models import train_sector_specific_models


class TestTrainSectorSmartFeatures(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.df = pd.DataFrame(
            {
                "sector": ["Tech", "Tech", "Finance", "Finance"] * 10,
                "feature1": rng.normal(size=40),
                "feature2": rng.normal(size=40),
                "target": rng.normal(size=40) + 100.0,
            }
        )

    def test_dict_uses_all_features(self):
        feature_cols = {
            "numeric_features": ["feature1"],
            "categorical_features": [],
            "classification_features": [],
            "all_features": ["feature1", "feature2"],
        }
        with self.assertLogs("finance_ml.advanced_models", level="INFO") as cm:
            sector_models, results = train_sector_specific_models(
                df=self.df,
                feature_cols=feature_cols,
                target_col="target",
                sector_col="sector",
                min_samples=5,
                random_state=0,
            )
        logs = "\n".join(cm.output)
        self.assertIn("all_features", logs)
        # Two sectors, both trained
        self.assertEqual(set(sector_models.keys()), set(self.df["sector"].unique()))
        # Expect two features used
        for m in sector_models.values():
            self.assertEqual(getattr(m, "n_features_in_", 2), 2)

    def test_dict_combines_types_and_deduplicates(self):
        feature_cols = {
            "numeric_features": ["feature1", "feature2", "feature2"],
            "categorical_features": ["missing_feature"],
            "classification_features": [],
        }
        with self.assertLogs("finance_ml.advanced_models", level="INFO") as cm:
            sector_models, results = train_sector_specific_models(
                df=self.df,
                feature_cols=feature_cols,
                target_col="target",
                sector_col="sector",
                min_samples=5,
                random_state=0,
            )
        logs = "\n".join(cm.output)
        self.assertIn("Combined feature types", logs)
        self.assertIn("After deduplication", logs)
        # Missing feature warning
        self.assertTrue("not in DataFrame" in logs or "Warning" in logs)
        # After filtering, two valid features remain
        for m in sector_models.values():
            self.assertEqual(getattr(m, "n_features_in_", 2), 2)

    def test_list_with_missing_is_filtered(self):
        feature_cols = ["feature1", "missing_feature"]
        with self.assertLogs("finance_ml.advanced_models", level="INFO") as cm:
            sector_models, results = train_sector_specific_models(
                df=self.df,
                feature_cols=feature_cols,
                target_col="target",
                sector_col="sector",
                min_samples=5,
                random_state=0,
            )
        logs = "\n".join(cm.output)
        self.assertTrue("not in DataFrame" in logs or "Warning" in logs)
        for m in sector_models.values():
            self.assertEqual(getattr(m, "n_features_in_", 1), 1)

    def test_error_when_no_valid_features_remain(self):
        with self.assertRaises(ValueError) as exc:
            train_sector_specific_models(
                df=self.df,
                feature_cols=["unknown1", "unknown2"],
                target_col="target",
                sector_col="sector",
                min_samples=5,
                random_state=0,
            )
        self.assertIn("No valid feature columns", str(exc.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
