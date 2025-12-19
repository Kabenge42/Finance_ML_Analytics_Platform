import unittest

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.regression import models as regression_models
from finance_ml.ml_workflow.regression.utils import get_r2_score


class TestRegressionReturnSchema(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {
                "feature_a": [0.0, 1.0, 2.0, 3.0],
                "feature_b": [1.0, 0.5, 0.0, -0.5],
            }
        )
        self.y = pd.Series([0.0, 1.0, 2.0, 3.0])

    def _assert_standard_schema(self, result):
        self.assertIsInstance(result, dict)
        for key in ("model", "metrics", "y_pred", "y_proba", "artifacts"):
            self.assertIn(key, result)

        metrics = result["metrics"]
        self.assertIsInstance(metrics, dict)
        for metric_key in ("r2", "r2_score", "mae", "rmse"):
            self.assertIn(metric_key, metrics)

        y_pred = result["y_pred"]
        self.assertEqual(len(y_pred), len(self.y))

        # Regression models do not emit probabilities
        self.assertIsNone(result["y_proba"])

    def test_get_r2_score_handles_formats(self):
        cases = [
            ({"metrics": {"r2": 0.91}}, 0.91),
            ({"metrics": {"r2_score": 0.81}}, 0.81),
            ({"r2_score": 0.71}, 0.71),
            ({"r2": 0.61}, 0.61),
            (("rf", {"metrics": {"r2": 0.51}}), 0.51),
            (("rf", {"r2_score": 0.41}), 0.41),
            (0.31, 0.31),
            (("rf", 0.21), 0.21),
        ]

        for payload, expected in cases:
            with self.subTest(payload=payload):
                self.assertAlmostEqual(get_r2_score(payload), expected)

        # Unknown payloads should return 0.0 rather than raising
        self.assertEqual(get_r2_score({"metrics": {}}), 0.0)

    def test_train_random_forest_return_schema(self):
        result = regression_models.train_random_forest_regressor(
            self.X, self.y, n_estimators=8, random_state=0
        )
        self._assert_standard_schema(result)
        # Metrics should be finite
        for value in result["metrics"].values():
            self.assertTrue(np.isfinite(value))

    def test_train_extra_trees_return_schema(self):
        result = regression_models.train_extra_trees_regressor(
            self.X, self.y, n_estimators=8, random_state=0
        )
        self._assert_standard_schema(result)
        for value in result["metrics"].values():
            self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
