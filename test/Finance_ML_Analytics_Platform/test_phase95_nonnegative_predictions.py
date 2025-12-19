"""
Test suite for Phase 9.5 Non-Negative Predictions.

This module implements TDD tests for critical prediction validation issues:
1. Zero prediction detection and handling
2. Non-negativity enforcement with minimum threshold
3. Quantile monotonicity validation
4. Post-prediction validation checkpoint

Aligned with ml_workflow_guidelines.md and code_guidelines.md v1.10.

Phase 9.5 Critical Issues Addressed:
- Zero Predictions: Several stocks (PLTR, BAC, UBER, HD) have y_pred = 0.0
- Non-Negativity: ensure_nonnegative=False allowed negative price predictions
- Missing Validation: No post-prediction validation catching zero/negative predictions
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class TestZeroPredictionDetection(unittest.TestCase):
    """Tests for detecting zero predictions in model output."""

    def test_detect_zero_predictions_returns_count(self):
        """Detect zero predictions should return count of zeros."""
        from finance_ml.ml_workflow.regression.constraints import detect_zero_predictions

        predictions = np.array([100.0, 0.0, 50.0, 0.0, 75.0])
        result = detect_zero_predictions(predictions)

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["percentage"], 40.0)
        self.assertListEqual(list(result["indices"]), [1, 3])

    def test_detect_zero_predictions_no_zeros(self):
        """Should return count=0 when no zeros present."""
        from finance_ml.ml_workflow.regression.constraints import detect_zero_predictions

        predictions = np.array([100.0, 50.0, 75.0])
        result = detect_zero_predictions(predictions)

        self.assertEqual(result["count"], 0)
        self.assertEqual(result["percentage"], 0.0)
        self.assertEqual(len(result["indices"]), 0)

    def test_detect_zero_predictions_all_zeros(self):
        """Should detect when all predictions are zero."""
        from finance_ml.ml_workflow.regression.constraints import detect_zero_predictions

        predictions = np.array([0.0, 0.0, 0.0])
        result = detect_zero_predictions(predictions)

        self.assertEqual(result["count"], 3)
        self.assertEqual(result["percentage"], 100.0)


class TestMinimumPredictionThreshold(unittest.TestCase):
    """Tests for enforcing minimum prediction threshold (prices > 0)."""

    def test_enforce_minimum_threshold_replaces_zeros(self):
        """Predictions at or below threshold should be replaced."""
        from finance_ml.ml_workflow.regression.constraints import enforce_minimum_threshold

        predictions = np.array([100.0, 0.0, 50.0, 0.005, 75.0])
        result = enforce_minimum_threshold(predictions, min_value=0.01)

        self.assertTrue((result >= 0.01).all())
        self.assertEqual(result[0], 100.0)  # Unchanged
        self.assertEqual(result[1], 0.01)  # Was 0, now min
        self.assertEqual(result[2], 50.0)  # Unchanged
        self.assertEqual(result[3], 0.01)  # Was 0.005, now min
        self.assertEqual(result[4], 75.0)  # Unchanged

    def test_enforce_minimum_threshold_default_penny(self):
        """Default minimum should be 0.01 (one cent)."""
        from finance_ml.ml_workflow.regression.constraints import enforce_minimum_threshold

        predictions = np.array([0.0, -5.0, 0.001])
        result = enforce_minimum_threshold(predictions)

        self.assertTrue((result >= 0.01).all())


class TestZeroPredictionFallback(unittest.TestCase):
    """Tests for fallback mechanism when zero predictions detected."""

    def test_apply_zero_fallback_uses_last_price(self):
        """Zero predictions should fallback to last_price * factor."""
        from finance_ml.ml_workflow.regression.constraints import apply_zero_prediction_fallback

        predictions = np.array([100.0, 0.0, 50.0, 0.0])
        last_prices = np.array([95.0, 80.0, 45.0, 120.0])

        result = apply_zero_prediction_fallback(predictions, last_prices, fallback_factor=1.05)

        self.assertEqual(result[0], 100.0)  # Unchanged (non-zero)
        self.assertEqual(result[1], 80.0 * 1.05)  # 84.0 (fallback)
        self.assertEqual(result[2], 50.0)  # Unchanged (non-zero)
        self.assertEqual(result[3], 120.0 * 1.05)  # 126.0 (fallback)

    def test_apply_zero_fallback_default_factor(self):
        """Default fallback factor should be 1.05 (5% above last price)."""
        from finance_ml.ml_workflow.regression.constraints import apply_zero_prediction_fallback

        predictions = np.array([0.0])
        last_prices = np.array([100.0])

        result = apply_zero_prediction_fallback(predictions, last_prices)

        self.assertEqual(result[0], 105.0)

    def test_apply_zero_fallback_handles_missing_last_price(self):
        """When last_price is NaN, use global median or minimum threshold."""
        from finance_ml.ml_workflow.regression.constraints import apply_zero_prediction_fallback

        predictions = np.array([0.0, 50.0, 0.0])
        last_prices = np.array([np.nan, 45.0, 100.0])

        result = apply_zero_prediction_fallback(predictions, last_prices)

        # First zero has NaN last_price - should use minimum threshold
        self.assertGreater(result[0], 0)
        self.assertEqual(result[1], 50.0)  # Unchanged
        self.assertEqual(result[2], 100.0 * 1.05)  # 105.0


class TestPostPredictionValidation(unittest.TestCase):
    """Tests for comprehensive post-prediction validation checkpoint."""

    def test_validate_predictions_catches_zeros(self):
        """Validation should fail if zero predictions exist."""
        from finance_ml.ml_workflow.regression.constraints import validate_predictions

        predictions = np.array([100.0, 0.0, 50.0])

        result = validate_predictions(predictions)

        self.assertFalse(result["valid"])
        self.assertIn("zero_predictions", result["issues"])

    def test_validate_predictions_catches_negatives(self):
        """Validation should fail if negative predictions exist."""
        from finance_ml.ml_workflow.regression.constraints import validate_predictions

        predictions = np.array([100.0, -5.0, 50.0])

        result = validate_predictions(predictions)

        self.assertFalse(result["valid"])
        self.assertIn("negative_predictions", result["issues"])

    def test_validate_predictions_passes_valid(self):
        """Validation should pass for all positive predictions."""
        from finance_ml.ml_workflow.regression.constraints import validate_predictions

        predictions = np.array([100.0, 50.0, 75.0])

        result = validate_predictions(predictions)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["issues"]), 0)

    def test_validate_predictions_catches_extreme_values(self):
        """Validation should warn about extreme predictions (>10x median)."""
        from finance_ml.ml_workflow.regression.constraints import validate_predictions

        predictions = np.array([100.0, 50.0, 75.0, 10000.0])  # 10000 is extreme

        result = validate_predictions(predictions, warn_extreme=True)

        self.assertIn("extreme_predictions", result["warnings"])


class TestQuantileMonotonicity(unittest.TestCase):
    """Tests for quantile monotonicity validation (pred_p10 <= pred_p50 <= pred_p90)."""

    def test_validate_quantile_monotonicity_valid(self):
        """Should pass when quantiles are monotonic."""
        from finance_ml.ml_workflow.regression.constraints import validate_quantile_monotonicity

        pred_p10 = np.array([90.0, 40.0, 60.0])
        pred_p50 = np.array([100.0, 50.0, 70.0])
        pred_p90 = np.array([110.0, 60.0, 80.0])

        result = validate_quantile_monotonicity(pred_p10, pred_p50, pred_p90)

        self.assertTrue(result["valid"])
        self.assertEqual(result["violations"], 0)

    def test_validate_quantile_monotonicity_violation(self):
        """Should detect when quantiles violate monotonicity."""
        from finance_ml.ml_workflow.regression.constraints import validate_quantile_monotonicity

        pred_p10 = np.array([90.0, 60.0, 60.0])  # 60 > 50 (violation)
        pred_p50 = np.array([100.0, 50.0, 70.0])
        pred_p90 = np.array([110.0, 60.0, 80.0])

        result = validate_quantile_monotonicity(pred_p10, pred_p50, pred_p90)

        self.assertFalse(result["valid"])
        self.assertGreater(result["violations"], 0)

    def test_enforce_quantile_monotonicity(self):
        """Should fix violations by adjusting quantiles."""
        from finance_ml.ml_workflow.regression.constraints import enforce_quantile_monotonicity

        pred_p10 = np.array([90.0, 60.0])  # 60 > 50 (violation)
        pred_p50 = np.array([100.0, 50.0])
        pred_p90 = np.array([110.0, 45.0])  # 45 < 50 (violation)

        p10_fixed, p50_fixed, p90_fixed = enforce_quantile_monotonicity(
            pred_p10, pred_p50, pred_p90
        )

        # After fix: p10 <= p50 <= p90 for all rows
        self.assertTrue((p10_fixed <= p50_fixed).all())
        self.assertTrue((p50_fixed <= p90_fixed).all())


class TestEnhancedNonNegativeWrapper(unittest.TestCase):
    """Tests for enhanced wrapper with zero detection and minimum threshold."""

    def test_enhanced_wrapper_enforces_minimum(self):
        """Enhanced wrapper should use minimum threshold, not just 0."""
        from finance_ml.ml_workflow.regression.constraints import EnhancedNonNegativeWrapper

        # Create data that would produce zero/negative predictions
        X = np.array([[1], [2], [3], [4]])
        y = np.array([-5, -4, -3, -2])

        base = Ridge()
        model = EnhancedNonNegativeWrapper(base, min_value=0.01)
        model.fit(X, y)
        preds = model.predict(X)

        self.assertTrue((preds >= 0.01).all(), "All predictions must be >= 0.01")

    def test_enhanced_wrapper_logs_corrections(self):
        """Wrapper should track number of corrected predictions."""
        from finance_ml.ml_workflow.regression.constraints import EnhancedNonNegativeWrapper

        X = np.array([[1], [2], [3], [4]])
        y = np.array([-5, -4, -3, -2])

        base = Ridge()
        model = EnhancedNonNegativeWrapper(base, min_value=0.01)
        model.fit(X, y)
        preds = model.predict(X)

        self.assertGreater(model.last_correction_count, 0)


class TestIntegrationPredictionPipeline(unittest.TestCase):
    """Integration tests for the complete prediction validation pipeline."""

    def test_full_validation_pipeline(self):
        """Test complete pipeline: predict -> validate -> fix -> re-validate."""
        from finance_ml.ml_workflow.regression.constraints import (
            validate_predictions,
            apply_zero_prediction_fallback,
            enforce_minimum_threshold,
        )

        # Simulate problematic predictions
        raw_predictions = np.array([100.0, 0.0, -5.0, 50.0, 0.0])
        last_prices = np.array([95.0, 80.0, 100.0, 45.0, 120.0])

        # Step 1: Initial validation (should fail)
        initial_result = validate_predictions(raw_predictions)
        self.assertFalse(initial_result["valid"])

        # Step 2: Apply fixes
        fixed = enforce_minimum_threshold(raw_predictions)  # Fix negatives
        fixed = apply_zero_prediction_fallback(fixed, last_prices)  # Fix zeros

        # Step 3: Re-validate (should pass)
        final_result = validate_predictions(fixed)
        self.assertTrue(final_result["valid"])
        self.assertTrue((fixed > 0).all())


class TestTargetLeakageDetection(unittest.TestCase):
    """Tests for target leakage detection (Phase 9.5 data quality)."""

    def test_detect_target_leakage_finds_price_target(self):
        """Should detect price_target in feature columns."""
        from finance_ml.ml_workflow.regression.constraints import detect_target_leakage

        features = ["market_cap", "price_target", "p_e_ratio", "revenue"]
        result = detect_target_leakage(features)

        self.assertTrue(result["has_leakage"])
        self.assertIn("price_target", result["leaky_features"])
        self.assertEqual(result["severity"], "critical")

    def test_detect_target_leakage_no_leakage(self):
        """Should pass when no target-related columns present."""
        from finance_ml.ml_workflow.regression.constraints import detect_target_leakage

        features = ["market_cap", "p_e_ratio", "revenue", "gross_margin"]
        result = detect_target_leakage(features)

        self.assertFalse(result["has_leakage"])
        self.assertEqual(len(result["leaky_features"]), 0)
        self.assertEqual(result["severity"], "none")

    def test_detect_target_leakage_multiple_patterns(self):
        """Should detect multiple leaky columns."""
        from finance_ml.ml_workflow.regression.constraints import detect_target_leakage

        features = ["market_cap", "price_target", "y_pred_calibrated", "forward_return_1m"]
        result = detect_target_leakage(features)

        self.assertTrue(result["has_leakage"])
        self.assertEqual(len(result["leaky_features"]), 3)

    def test_detect_target_leakage_strict_mode(self):
        """Should raise ValueError in strict mode when leakage detected."""
        from finance_ml.ml_workflow.regression.constraints import detect_target_leakage

        features = ["market_cap", "price_target", "p_e_ratio"]

        with self.assertRaises(ValueError) as context:
            detect_target_leakage(features, strict=True)

        self.assertIn("Target leakage detected", str(context.exception))

    def test_detect_target_leakage_custom_patterns(self):
        """Should use custom patterns when provided."""
        from finance_ml.ml_workflow.regression.constraints import detect_target_leakage

        features = ["market_cap", "custom_target_col", "p_e_ratio"]
        result = detect_target_leakage(features, target_patterns=["custom_target"])

        self.assertTrue(result["has_leakage"])
        self.assertIn("custom_target_col", result["leaky_features"])


class TestAuditFeaturesForTraining(unittest.TestCase):
    """Tests for feature auditing before model training."""

    def test_audit_features_detects_leakage(self):
        """Should detect and report target leakage."""
        from finance_ml.ml_workflow.regression.constraints import audit_features_for_training

        X = pd.DataFrame(
            {
                "market_cap": [1e9, 2e9, 3e9],
                "price_target": [100, 200, 300],
                "p_e_ratio": [15, 20, 25],
            }
        )

        X_clean, report = audit_features_for_training(X, auto_remove=False)

        self.assertTrue(report["has_issues"])
        self.assertIn("price_target", report.get("leakage_detected", []))

    def test_audit_features_auto_remove(self):
        """Should auto-remove leaky features when requested."""
        from finance_ml.ml_workflow.regression.constraints import audit_features_for_training

        X = pd.DataFrame(
            {
                "market_cap": [1e9, 2e9, 3e9],
                "price_target": [100, 200, 300],
                "p_e_ratio": [15, 20, 25],
            }
        )

        X_clean, report = audit_features_for_training(X, auto_remove=True)

        self.assertNotIn("price_target", X_clean.columns)
        self.assertIn("price_target", report["removed_features"])
        self.assertEqual(report["final_features"], 2)

    def test_audit_features_no_issues(self):
        """Should pass clean features without issues."""
        from finance_ml.ml_workflow.regression.constraints import audit_features_for_training

        X = pd.DataFrame(
            {
                "market_cap": [1e9, 2e9, 3e9],
                "p_e_ratio": [15, 20, 25],
                "revenue": [1e8, 2e8, 3e8],
            }
        )

        X_clean, report = audit_features_for_training(X)

        self.assertFalse(report["has_issues"])
        self.assertEqual(len(report["removed_features"]), 0)


class TestOverfittingDetection(unittest.TestCase):
    """Tests for overfitting detection (R² > 0.95, MAE = 0 warnings).

    Per ml_workflow_guidelines.md, these metrics indicate potential overfitting or data leakage:
    - Ridge R² = 1.0000 (expected: 0.60-0.85) → CRITICAL
    - Ridge MAE = 0.00 (expected: >100) → CRITICAL
    - R² >= 0.95 suggests data leakage - audit features
    - MAE = 0 is impossible - check for target in features
    """

    def test_detect_overfitting_r2_too_high(self):
        """Should detect when R² is suspiciously high (>= 0.95)."""
        from finance_ml.ml_workflow.regression.constraints import detect_overfitting

        metrics = {"r2": 1.0, "mae": 100.0, "rmse": 150.0}
        result = detect_overfitting(metrics)

        self.assertTrue(result["has_overfitting"])
        self.assertIn("r2_too_high", result["warnings"])
        self.assertEqual(result["severity"], "critical")

    def test_detect_overfitting_mae_zero(self):
        """Should detect when MAE is zero (impossible for real data)."""
        from finance_ml.ml_workflow.regression.constraints import detect_overfitting

        metrics = {"r2": 0.85, "mae": 0.0, "rmse": 0.0}
        result = detect_overfitting(metrics)

        self.assertTrue(result["has_overfitting"])
        self.assertIn("mae_zero", result["warnings"])
        self.assertEqual(result["severity"], "critical")

    def test_detect_overfitting_both_issues(self):
        """Should detect both R²=1.0 and MAE=0 (classic leakage pattern)."""
        from finance_ml.ml_workflow.regression.constraints import detect_overfitting

        metrics = {"r2": 1.0, "mae": 0.0, "rmse": 0.0}
        result = detect_overfitting(metrics)

        self.assertTrue(result["has_overfitting"])
        self.assertIn("r2_too_high", result["warnings"])
        self.assertIn("mae_zero", result["warnings"])
        self.assertEqual(result["severity"], "critical")

    def test_detect_overfitting_realistic_metrics(self):
        """Should pass when metrics are in realistic range."""
        from finance_ml.ml_workflow.regression.constraints import detect_overfitting

        # Realistic metrics per ml_workflow_guidelines.md
        metrics = {"r2": 0.75, "mae": 1500.0, "rmse": 3500.0}
        result = detect_overfitting(metrics)

        self.assertFalse(result["has_overfitting"])
        self.assertEqual(len(result["warnings"]), 0)
        self.assertEqual(result["severity"], "none")

    def test_detect_overfitting_borderline_r2(self):
        """Should warn when R² is borderline (0.90-0.95)."""
        from finance_ml.ml_workflow.regression.constraints import detect_overfitting

        metrics = {"r2": 0.92, "mae": 500.0, "rmse": 1000.0}
        result = detect_overfitting(metrics)

        # Borderline should be warning, not critical
        self.assertTrue(result["has_overfitting"])
        self.assertEqual(result["severity"], "warning")

    def test_detect_overfitting_custom_thresholds(self):
        """Should use custom thresholds when provided."""
        from finance_ml.ml_workflow.regression.constraints import detect_overfitting

        metrics = {"r2": 0.88, "mae": 100.0, "rmse": 200.0}
        result = detect_overfitting(metrics, r2_threshold=0.85, mae_min=50.0)

        self.assertTrue(result["has_overfitting"])
        self.assertIn("r2_too_high", result["warnings"])

    def test_detect_overfitting_strict_mode(self):
        """Should raise ValueError in strict mode when overfitting detected."""
        from finance_ml.ml_workflow.regression.constraints import detect_overfitting

        metrics = {"r2": 1.0, "mae": 0.0, "rmse": 0.0}

        with self.assertRaises(ValueError) as context:
            detect_overfitting(metrics, strict=True)

        self.assertIn("Overfitting detected", str(context.exception))


class TestModelMetricsValidation(unittest.TestCase):
    """Tests for comprehensive model metrics validation."""

    def test_validate_model_metrics_all_checks(self):
        """Should run all validation checks on model metrics."""
        from finance_ml.ml_workflow.regression.constraints import validate_model_metrics

        # Suspicious metrics indicating leakage
        metrics = {
            "r2": 1.0,
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
        }

        result = validate_model_metrics(metrics)

        self.assertFalse(result["valid"])
        self.assertIn("overfitting", result["issues"])

    def test_validate_model_metrics_passes_realistic(self):
        """Should pass realistic model metrics."""
        from finance_ml.ml_workflow.regression.constraints import validate_model_metrics

        metrics = {
            "r2": 0.72,
            "mae": 1200.0,
            "rmse": 2500.0,
            "mape": 15.5,
        }

        result = validate_model_metrics(metrics)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["issues"]), 0)


if __name__ == "__main__":
    unittest.main()
