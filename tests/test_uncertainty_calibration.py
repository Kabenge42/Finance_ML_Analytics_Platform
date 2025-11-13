"""
Test uncertainty calibration features (Priority 0: Fix Uncertainty Quantification).

Tests coverage for:
1. Conformal prediction intervals (already implemented in uncertainty.py)
2. Quantile monotonicity enforcement (needs implementation)
3. Non-negativity constraints
4. Coverage validation (75-85% target)

Addresses Model Optimization Recommendations Priority 0:
- Current: 7.1% coverage (CRITICAL FAILURE)
- Target: 75-85% coverage
"""

import unittest
import numpy as np
import pandas as pd
from finance_ml.ml_workflow.regression.uncertainty import (
    conformal_prediction_intervals,
    compute_interval_coverage,
)


class TestConformalPrediction(unittest.TestCase):
    """Test conformal prediction interval calibration."""

    def setUp(self):
        """Create synthetic calibration and test data."""
        np.random.seed(42)
        self.n_cal = 100
        self.n_test = 50

        # Calibration set: y = 2*x + noise
        self.y_cal = np.random.randn(self.n_cal) * 10 + 50
        self.y_cal_pred = self.y_cal + np.random.randn(self.n_cal) * 5  # Add prediction error

        # Test set
        self.y_test_pred = np.random.randn(self.n_test) * 10 + 50

    def test_conformal_intervals_exist(self):
        """Test that conformal_prediction_intervals function exists and runs."""
        lower, upper = conformal_prediction_intervals(
            self.y_cal, self.y_cal_pred, self.y_test_pred, alpha=0.2
        )
        self.assertEqual(len(lower), self.n_test)
        self.assertEqual(len(upper), self.n_test)

    def test_conformal_intervals_coverage_target(self):
        """Test that calibrated intervals achieve target coverage (75-85%)."""
        # Use calibration set to build intervals
        lower, upper = conformal_prediction_intervals(
            self.y_cal, self.y_cal_pred, self.y_cal, alpha=0.2
        )

        # Check coverage on same calibration set (should be ~80%)
        coverage = compute_interval_coverage(self.y_cal, lower, upper)

        # Target: 75-85% coverage (issue states this is acceptable)
        self.assertGreaterEqual(coverage, 0.75, f"Coverage {coverage:.1%} below target 75%")
        self.assertLessEqual(
            coverage, 0.90, f"Coverage {coverage:.1%} suspiciously high (may indicate overfitting)"
        )

    def test_conformal_non_negative_lower_bound(self):
        """Test that lower bounds are non-negative when clip_lower_at_zero=True."""
        # Create predictions that might produce negative intervals
        y_test_pred_low = np.array([5.0, 10.0, 15.0])

        lower, upper = conformal_prediction_intervals(
            self.y_cal, self.y_cal_pred, y_test_pred_low, alpha=0.2, clip_lower_at_zero=True
        )

        # All lower bounds should be >= 0
        self.assertTrue(np.all(lower >= 0), f"Found negative lower bounds: {lower[lower < 0]}")

    def test_conformal_intervals_width_positive(self):
        """Test that interval width is always positive."""
        lower, upper = conformal_prediction_intervals(
            self.y_cal, self.y_cal_pred, self.y_test_pred, alpha=0.2
        )

        width = upper - lower
        self.assertTrue(
            np.all(width > 0), f"Found non-positive interval widths: {width[width <= 0]}"
        )


class TestQuantileMonotonicity(unittest.TestCase):
    """Test quantile monotonicity enforcement (needs implementation)."""

    def setUp(self):
        """Create sample quantile predictions that may violate monotonicity."""
        np.random.seed(42)
        n = 50

        # Simulate quantile predictions that violate ordering
        self.pred_p10 = np.random.randn(n) * 10 + 40  # Should be lowest
        self.pred_p50 = np.random.randn(n) * 10 + 50  # Should be middle
        self.pred_p90 = np.random.randn(n) * 10 + 60  # Should be highest

        # Intentionally create some violations
        violations_idx = [5, 10, 15]
        for idx in violations_idx:
            # Swap p10 and p90 to create violation
            self.pred_p10[idx], self.pred_p90[idx] = self.pred_p90[idx], self.pred_p10[idx]

    def test_enforce_monotonic_quantiles_exists(self):
        """Test that enforce_monotonic_quantiles function exists."""
        # This should fail initially - function doesn't exist yet
        try:
            from finance_ml.ml_workflow.regression.quantile import enforce_monotonic_quantiles

            self.assertTrue(callable(enforce_monotonic_quantiles))
        except ImportError:
            self.fail("enforce_monotonic_quantiles not implemented in quantile.py")

    def test_enforce_monotonic_quantiles_ordering(self):
        """Test that enforced quantiles satisfy p10 <= p50 <= p90."""
        from finance_ml.ml_workflow.regression.quantile import enforce_monotonic_quantiles

        # Create dict of quantile predictions
        quantile_preds = {
            0.1: self.pred_p10.copy(),
            0.5: self.pred_p50.copy(),
            0.9: self.pred_p90.copy(),
        }

        # Enforce monotonicity
        monotonic = enforce_monotonic_quantiles(quantile_preds)

        # Check ordering for all samples
        for i in range(len(self.pred_p10)):
            self.assertLessEqual(
                monotonic[0.1][i],
                monotonic[0.5][i],
                f"p10 > p50 at index {i}: {monotonic[0.1][i]} > {monotonic[0.5][i]}",
            )
            self.assertLessEqual(
                monotonic[0.5][i],
                monotonic[0.9][i],
                f"p50 > p90 at index {i}: {monotonic[0.5][i]} > {monotonic[0.9][i]}",
            )

    def test_enforce_monotonic_quantiles_preserves_valid(self):
        """Test that already-monotonic quantiles are preserved."""
        from finance_ml.ml_workflow.regression.quantile import enforce_monotonic_quantiles

        # Create properly ordered quantiles
        n = 50
        base = np.linspace(40, 60, n)
        quantile_preds = {
            0.1: base - 10,
            0.5: base,
            0.9: base + 10,
        }

        monotonic = enforce_monotonic_quantiles(quantile_preds)

        # Should be unchanged
        np.testing.assert_array_almost_equal(monotonic[0.1], quantile_preds[0.1])
        np.testing.assert_array_almost_equal(monotonic[0.5], quantile_preds[0.5])
        np.testing.assert_array_almost_equal(monotonic[0.9], quantile_preds[0.9])


class TestQuantilePredictionIntegration(unittest.TestCase):
    """Integration test: quantile prediction with conformal calibration."""

    def setUp(self):
        """Create synthetic training and test data."""
        np.random.seed(42)
        self.n_train = 200
        self.n_test = 50

        # Simple linear relationship with heteroskedastic noise
        X_train = np.random.randn(self.n_train, 5)
        y_train = X_train[:, 0] * 2 + X_train[:, 1] * -1 + np.random.randn(self.n_train) * 5

        X_test = np.random.randn(self.n_test, 5)
        y_test = X_test[:, 0] * 2 + X_test[:, 1] * -1 + np.random.randn(self.n_test) * 5

        self.X_train = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(5)])
        self.y_train = pd.Series(y_train)
        self.X_test = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(5)])
        self.y_test = pd.Series(y_test)

    def test_quantile_training_produces_models(self):
        """Test that train_quantile_regressor returns trained models."""
        from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor

        result = train_quantile_regressor(
            self.X_train, self.y_train, quantiles=[0.1, 0.5, 0.9], random_state=42
        )

        # Check standardized return format
        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertIn("artifacts", result)

        # Should have 3 models
        models = result["model"]
        self.assertEqual(len(models), 3)

    def test_quantile_predictions_with_monotonicity(self):
        """Test quantile predictions with enforced monotonicity."""
        from finance_ml.ml_workflow.regression.quantile import (
            train_quantile_regressor,
            enforce_monotonic_quantiles,
        )

        # Train models
        result = train_quantile_regressor(
            self.X_train, self.y_train, quantiles=[0.1, 0.5, 0.9], random_state=42
        )

        models = result["model"]
        quantiles = result["artifacts"]["quantiles"]

        # Make predictions
        predictions = {}
        for model, q in zip(models, quantiles):
            predictions[q] = model.predict(self.X_test)

        # Enforce monotonicity
        monotonic_preds = enforce_monotonic_quantiles(predictions)

        # Verify ordering
        for i in range(len(self.X_test)):
            self.assertLessEqual(monotonic_preds[0.1][i], monotonic_preds[0.5][i])
            self.assertLessEqual(monotonic_preds[0.5][i], monotonic_preds[0.9][i])


if __name__ == "__main__":
    unittest.main()
