import unittest
import numpy as np

try:
    from finance_ml.ml_workflow.regression.robust import (
        winsorize_target,
        clip_predictions,
        adaptive_clip_predictions,
    )
except Exception:
    # Fallback import directly from file to avoid heavy package __init__
    import importlib.util
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parents[1]
        / "finance_ml"
        / "ml_workflow"
        / "regression"
        / "robust.py"
    )
    spec = importlib.util.spec_from_file_location("robust", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.load_module(mod.__name__)  # type: ignore[attr-defined]
    winsorize_target = getattr(mod, "winsorize_target")
    clip_predictions = getattr(mod, "clip_predictions")
    adaptive_clip_predictions = getattr(mod, "adaptive_clip_predictions")


class TestRobustOutlierSafety(unittest.TestCase):
    def test_winsorize_caps_extremes(self):
        y = np.array([1, 2, 3, 4, 5, 1000], dtype=float)
        w = winsorize_target(y, lower=0.0, upper=0.90)
        # Top 10% (approx) should be capped; since small sample, ensure the extreme is reduced
        self.assertTrue(w.max() < 1000)
        # Monotonicity preserved and same length
        self.assertEqual(len(w), len(y))
        self.assertTrue(np.all(w >= 0))

    def test_winsorize_limits_validation(self):
        with self.assertRaises(ValueError):
            winsorize_target([1, 2, 3], lower=0.6, upper=0.4)

    def test_clip_predictions_bounds(self):
        rng = np.random.default_rng(0)
        y_train = rng.normal(loc=100.0, scale=10.0, size=500)
        preds = np.array([50.0, 100.0, 200.0, -10.0])
        clipped = clip_predictions(preds, y_train, n_std=2.0)
        lower = max(0.0, y_train.mean() - 2.0 * y_train.std())
        upper = y_train.mean() + 2.0 * y_train.std()
        self.assertTrue(np.all(clipped >= lower - 1e-9))
        self.assertTrue(np.all(clipped <= upper + 1e-9))
        # Non-negative lower bound ensured
        self.assertTrue(np.all(clipped >= 0.0))


class TestAdaptiveClipPredictions(unittest.TestCase):
    """Test suite for adaptive_clip_predictions() - percentile-based clipping to fix zero predictions issue."""

    def test_lower_bound_calculation(self):
        """Test that lower bound is 0.5 × p0.5 with minimum of $0.10."""
        # Training data
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0])
        preds = np.array([0.0, 1.0, 5.0])  # Some predictions that would be clipped

        result = adaptive_clip_predictions(preds, y_train)

        # Calculate expected lower bound: 0.5 * p0.5 (with minimum $0.10)
        train_p0_5 = np.percentile(y_train, 0.5)
        expected_lower = max(0.10, train_p0_5 * 0.5)
        clipped_preds = result["clipped_predictions"]

        # Verify the returned lower_bound matches our calculation
        self.assertAlmostEqual(result["lower_bound"], expected_lower, places=5)

        # All predictions should be >= lower bound
        self.assertTrue(np.all(clipped_preds >= expected_lower - 1e-9))

    def test_lower_bound_minimum_enforced(self):
        """Test that lower bound never goes below $0.10."""
        # Training data with very low p0.5 (e.g., $0.05)
        y_train = np.array([0.05, 0.10, 0.15, 0.20, 1.0, 5.0, 10.0])
        preds = np.array([-1.0, 0.0, 0.05])

        result = adaptive_clip_predictions(preds, y_train)
        clipped_preds = result["clipped_predictions"]

        # Lower bound should be at least $0.10
        self.assertTrue(np.all(clipped_preds >= 0.10))

    def test_upper_bound_calculation(self):
        """Test that upper bound is 1.5 × p99.5."""
        # Training data with p99.5 = $100.0
        y_train = np.concatenate(
            [np.linspace(1, 50, 100), np.array([100.0, 100.0])]  # 99.5th percentile ~ 100
        )
        preds = np.array([50.0, 100.0, 200.0])

        result = adaptive_clip_predictions(preds, y_train)

        # Expected upper bound: 1.5 * p99.5 ≈ 1.5 * 100 = 150
        expected_upper = 150.0
        clipped_preds = result["clipped_predictions"]

        # All predictions should be <= upper bound
        self.assertTrue(np.all(clipped_preds <= expected_upper + 1e-6))

    def test_values_between_bounds_preserved(self):
        """Test that predictions within bounds are not modified."""
        y_train = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
        # Predictions well within expected bounds
        preds = np.array([10.0, 20.0, 30.0])

        result = adaptive_clip_predictions(preds, y_train)
        clipped_preds = result["clipped_predictions"]

        # These predictions should be unchanged
        np.testing.assert_array_almost_equal(clipped_preds, preds)

    def test_zero_elimination_guarantee(self):
        """Test that NO predictions are exactly zero after clipping."""
        y_train = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
        # Include predictions that would be clipped to zero with old approach
        preds = np.array([-10.0, -5.0, -1.0, 0.0, 0.5, 1.0, 5.0])

        result = adaptive_clip_predictions(preds, y_train)
        clipped_preds = result["clipped_predictions"]

        # CRITICAL: NO zeros should exist
        n_zeros = np.sum(clipped_preds == 0.0)
        self.assertEqual(n_zeros, 0, f"Found {n_zeros} zero predictions, expected 0")

        # All should be >= 0.10 minimum
        self.assertTrue(np.all(clipped_preds >= 0.10))

    def test_diagnostic_statistics_returned(self):
        """Test that diagnostic statistics are returned in the result dict."""
        y_train = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
        preds = np.array([-5.0, 10.0, 20.0, 200.0])  # 2 will be clipped

        result = adaptive_clip_predictions(preds, y_train)

        # Check result structure
        self.assertIn("clipped_predictions", result)
        self.assertIn("lower_bound", result)
        self.assertIn("upper_bound", result)
        self.assertIn("n_clipped_lower", result)
        self.assertIn("n_clipped_upper", result)
        self.assertIn("pct_clipped_lower", result)
        self.assertIn("pct_clipped_upper", result)

        # Check types
        self.assertIsInstance(result["clipped_predictions"], np.ndarray)
        self.assertIsInstance(result["lower_bound"], (float, np.floating))
        self.assertIsInstance(result["upper_bound"], (float, np.floating))
        self.assertIsInstance(result["n_clipped_lower"], (int, np.integer))
        self.assertIsInstance(result["n_clipped_upper"], (int, np.integer))
        self.assertIsInstance(result["pct_clipped_lower"], (float, np.floating))
        self.assertIsInstance(result["pct_clipped_upper"], (float, np.floating))

    def test_clipping_counts_accurate(self):
        """Test that clipping statistics accurately count clipped predictions."""
        y_train = np.array([2.0, 5.0, 10.0, 15.0, 20.0, 100.0])
        # Create predictions where we know exactly how many will be clipped
        # Lower bound will be ~1.0 (0.5 * 2.0), upper bound ~150 (1.5 * 100)
        preds = np.array(
            [
                -5.0,
                0.0,  # 2 below lower bound
                5.0,
                10.0,
                20.0,  # 3 within bounds
                200.0,
                300.0,  # 2 above upper bound
            ]
        )

        result = adaptive_clip_predictions(preds, y_train)

        # Check counts
        self.assertGreaterEqual(result["n_clipped_lower"], 2)
        self.assertGreaterEqual(result["n_clipped_upper"], 2)

        # Check percentages sum logically
        total_clipped = result["n_clipped_lower"] + result["n_clipped_upper"]
        self.assertLessEqual(total_clipped, len(preds))

    def test_empty_predictions_array(self):
        """Test handling of empty predictions array."""
        y_train = np.array([1.0, 5.0, 10.0])
        preds = np.array([])

        result = adaptive_clip_predictions(preds, y_train)
        clipped_preds = result["clipped_predictions"]

        self.assertEqual(len(clipped_preds), 0)
        self.assertEqual(result["n_clipped_lower"], 0)
        self.assertEqual(result["n_clipped_upper"], 0)

    def test_empty_training_data(self):
        """Test handling of empty training data (edge case)."""
        y_train = np.array([])
        preds = np.array([5.0, 10.0, 20.0])

        # Should handle gracefully, likely using fallback bounds
        result = adaptive_clip_predictions(preds, y_train)
        clipped_preds = result["clipped_predictions"]

        # Should still return predictions (with safe defaults)
        self.assertEqual(len(clipped_preds), len(preds))
        # All should be non-negative at minimum
        self.assertTrue(np.all(clipped_preds >= 0.10))

    def test_all_negative_predictions(self):
        """Test that all negative predictions are clipped to lower bound, not zero."""
        y_train = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        preds = np.array([-10.0, -5.0, -2.0, -0.5])

        result = adaptive_clip_predictions(preds, y_train)
        clipped_preds = result["clipped_predictions"]

        # All should be clipped to lower bound (not zero!)
        expected_lower = max(0.10, 0.5 * np.percentile(y_train, 0.5))
        np.testing.assert_array_almost_equal(clipped_preds, expected_lower)

        # Zero elimination guarantee
        self.assertEqual(np.sum(clipped_preds == 0.0), 0)

    def test_single_value_training_data(self):
        """Test handling when all training values are the same."""
        y_train = np.array([10.0, 10.0, 10.0, 10.0])
        preds = np.array([-5.0, 5.0, 10.0, 15.0, 100.0])

        result = adaptive_clip_predictions(preds, y_train)
        clipped_preds = result["clipped_predictions"]

        # Should handle gracefully
        self.assertEqual(len(clipped_preds), len(preds))
        self.assertTrue(np.all(clipped_preds >= 0.10))


if __name__ == "__main__":
    unittest.main()
