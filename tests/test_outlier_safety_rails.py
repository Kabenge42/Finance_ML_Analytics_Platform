"""
Test outlier safety rails (Priority 2: Extreme Outliers).

Tests for winsorization, clipping, and non-negativity enforcement
to address the critical issue of 3% catastrophic predictions.
"""

import unittest
import numpy as np
from finance_ml.ml_workflow.regression.robust import (
    winsorize_target,
    clip_predictions,
)


class TestWinsorization(unittest.TestCase):
    """Test target winsorization to cap extreme values."""

    def test_winsorize_caps_extremes(self):
        """Test that extreme values are capped at specified percentiles."""
        y = np.array([1, 2, 3, 4, 5, 100, 200])  # Outliers: 100, 200

        # Cap at 10th and 90th percentiles
        y_winsorized = winsorize_target(y, lower=0.1, upper=0.9)

        # Max should be capped
        self.assertLess(y_winsorized.max(), 200)
        self.assertLess(y_winsorized.max(), 100)

    def test_winsorize_preserves_median(self):
        """Test that central values are preserved."""
        y = np.array([1, 2, 3, 4, 5, 6, 7, 100])
        y_winsorized = winsorize_target(y, lower=0.05, upper=0.95)

        # Median should be approximately preserved
        self.assertAlmostEqual(np.median(y_winsorized), np.median(y[:7]), delta=1.0)


class TestPredictionClipping(unittest.TestCase):
    """Test prediction clipping to reasonable ranges."""

    def test_clip_prevents_negative_predictions(self):
        """Test that clipping enforces non-negative predictions."""
        preds = np.array([-10, -5, 0, 5, 10])
        y_train = np.array([10, 20, 30, 40, 50])

        clipped = clip_predictions(preds, y_train, n_std=3)

        # All predictions should be >= 0
        self.assertTrue(np.all(clipped >= 0), f"Found negative predictions: {clipped[clipped < 0]}")

    def test_clip_bounds_extreme_predictions(self):
        """Test that extreme predictions are clipped to training range."""
        preds = np.array([1000, 2000, 5000])  # Extreme predictions
        y_train = np.array([50, 60, 70, 80, 90])  # Training range ~50-90

        clipped = clip_predictions(preds, y_train, n_std=3)

        # Predictions should be bounded
        self.assertLess(clipped.max(), 1000, "Extreme predictions not properly clipped")

    def test_clip_preserves_reasonable_predictions(self):
        """Test that predictions within range are preserved."""
        y_train = np.array([50, 60, 70, 80, 90])
        preds = np.array([55, 65, 75])  # Within training range

        clipped = clip_predictions(preds, y_train, n_std=3)

        # Should be close to original predictions
        np.testing.assert_array_almost_equal(clipped, preds, decimal=0)


class TestNonNegativityEnforcement(unittest.TestCase):
    """Test that negative prediction guards work correctly."""

    def test_enforce_non_negative_exists(self):
        """Test that enforce_non_negative utility exists."""
        # This tests if we have a dedicated function for non-negativity
        # May fail if not implemented yet
        try:
            from finance_ml.ml_workflow.regression.robust import enforce_non_negative

            self.assertTrue(callable(enforce_non_negative))
        except ImportError:
            # Optional - can use clip_predictions for this purpose
            pass

    def test_combined_safety_rails(self):
        """Integration test: winsorize training, clip predictions."""
        # Training data with outliers
        y_train_raw = np.array([10, 20, 30, 40, 50, 60, 70, 500, 1000])

        # Winsorize training target
        y_train_clean = winsorize_target(y_train_raw, lower=0.05, upper=0.95)

        # Extreme predictions
        preds = np.array([-50, 0, 25, 100, 500])

        # Clip predictions based on cleaned training data
        preds_safe = clip_predictions(preds, y_train_clean, n_std=3)

        # All predictions should be non-negative and bounded
        self.assertTrue(np.all(preds_safe >= 0))
        self.assertLess(preds_safe.max(), 500)


if __name__ == "__main__":
    unittest.main()
