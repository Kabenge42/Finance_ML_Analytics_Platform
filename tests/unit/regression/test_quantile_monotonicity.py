import unittest
import numpy as np
import pandas as pd


class DummyQR:
    def __init__(self):
        pass

    def predict(self, X, quantiles=None):
        # intentionally violate monotonicity and include negatives
        n = len(X)
        return {
            0.1: np.array([12.0, -1.0, 7.0]),
            0.5: np.array([10.0, 0.5, 9.0]),
            0.9: np.array([20.0, 0.2, 8.0]),
        }


class TestQuantileMonotonicity(unittest.TestCase):
    def test_monotonicity_enforced(self):
        from finance_ml.ml_workflow.models import predict_quantile_regression

        X = pd.DataFrame({"x": [0, 1, 2]})
        model = type(
            "M", (), {"models": {0.1: None, 0.5: None, 0.9: None}, "predict": DummyQR().predict}
        )()

        fixed = predict_quantile_regression(model, X, quantiles=[0.1, 0.5, 0.9])
        self.assertTrue((fixed["pred_p10"] <= fixed["pred_p50"]).all())
        self.assertTrue((fixed["pred_p50"] <= fixed["pred_p90"]).all())
        self.assertTrue((fixed[["pred_p10", "pred_p50", "pred_p90"]] >= 0).all().all())


if __name__ == "__main__":
    unittest.main()
