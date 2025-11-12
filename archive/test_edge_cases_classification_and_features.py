import unittest
import numpy as np
import pandas as pd


class TestClassificationAndFeaturesEdgeCases(unittest.TestCase):
    def test_export_classification_features_mismatched_lengths(self):
        from finance_ml import classification

        df = pd.DataFrame({"a": [1, 2, 3]})
        y_proba = np.ones((2, 3)) / 3.0  # wrong number of samples
        with self.assertRaises(ValueError):
            classification.export_classification_features(df, y_proba)

    def test_engineer_margin_features_ltm_only_and_zero_denominator(self):
        from finance_ml import features

        df = pd.DataFrame(
            {
                "ebitda_ltm": [100.0, 200.0, 300.0],
                "total_revenues_ltm": [0.0, 500.0, 0.0],  # zeros to test safe division
            }
        )
        out = features.engineer_margin_features(df)
        # Column should exist and not contain inf values
        self.assertIn("ebitda_margin", out.columns)
        self.assertTrue(
            np.all(np.isfinite(out["ebitda_margin"].replace([np.inf, -np.inf], np.nan).fillna(0)))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
