import unittest
import numpy as np

# Robust import to avoid heavy package __init__ side effects in lean test runs
try:  # Standard import path
    from finance_ml.ml_workflow.regression.uncertainty import (
        conformal_prediction_intervals,
        compute_interval_coverage,
    )
except Exception:  # Fallback: import module directly from file path
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[1] / "finance_ml" / "ml_workflow" / "regression" / "uncertainty.py"
    spec = importlib.util.spec_from_file_location("_fm_uncertainty", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    conformal_prediction_intervals = getattr(mod, "conformal_prediction_intervals")
    compute_interval_coverage = getattr(mod, "compute_interval_coverage")


class TestConformalPredictionIntervals(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n_train = 500
        n_test = 200
        # Synthetic non-negative target with noise
        X = rng.normal(size=(n_train + n_test, 3))
        y_true = np.maximum(0.0, 10 + 2 * X[:, 0] - 3 * X[:, 1] + rng.normal(0, 1.5, size=n_train + n_test))

        # Use a simple linear model prediction with slight bias/noise to simulate model preds
        preds = 10 + 2 * X[:, 0] - 3 * X[:, 1] + rng.normal(0, 1.0, size=n_train + n_test)

        # Split calibration/test
        self.y_cal = y_true[:n_train]
        self.y_cal_pred = preds[:n_train]
        self.y_test = y_true[n_train:]
        self.y_test_pred = preds[n_train:]

    def test_shapes_and_types(self):
        lower, upper = conformal_prediction_intervals(self.y_cal, self.y_cal_pred, self.y_test_pred, alpha=0.2)
        self.assertEqual(lower.shape, self.y_test_pred.shape)
        self.assertEqual(upper.shape, self.y_test_pred.shape)
        self.assertTrue(np.issubdtype(lower.dtype, np.floating))
        self.assertTrue(np.issubdtype(upper.dtype, np.floating))

    def test_empirical_coverage_target(self):
        alpha = 0.2  # 80% intervals
        lower, upper = conformal_prediction_intervals(self.y_cal, self.y_cal_pred, self.y_test_pred, alpha=alpha)
        cov = compute_interval_coverage(self.y_test, lower, upper)
        # Allow tolerance due to randomness; should be near 80%
        self.assertGreaterEqual(cov, 0.70)
        self.assertLessEqual(cov, 0.95)

    def test_lower_bound_non_negative_by_default(self):
        lower, upper = conformal_prediction_intervals(self.y_cal, self.y_cal_pred, self.y_test_pred, alpha=0.2)
        self.assertTrue(np.all(lower >= 0))
        self.assertTrue(np.all(upper >= lower))

    def test_alpha_validation(self):
        with self.assertRaises(ValueError):
            conformal_prediction_intervals(self.y_cal, self.y_cal_pred, self.y_test_pred, alpha=-0.1)
        with self.assertRaises(ValueError):
            conformal_prediction_intervals(self.y_cal, self.y_cal_pred, self.y_test_pred, alpha=1.0)

    def test_input_validation_empty(self):
        with self.assertRaises(ValueError):
            conformal_prediction_intervals(np.array([]), np.array([]), np.array([]), alpha=0.2)

    def test_compute_interval_coverage_validation(self):
        with self.assertRaises(ValueError):
            compute_interval_coverage(np.array([]), np.array([]), np.array([]))
        with self.assertRaises(ValueError):
            compute_interval_coverage(np.array([1, 2]), np.array([0]), np.array([2]))


if __name__ == "__main__":
    unittest.main()
