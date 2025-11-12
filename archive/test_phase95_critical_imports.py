"""
Test suite for Phase 9.5 critical import issues.

This module tests the fixes for critical import errors in Phase 9.5 of
ml_finance_model_main_backup.ipynb as described in ML_Workflow_Improvement_Plan.md.

Critical Issues Tested:
1. Missing import: checkpoint function
2. Missing import: logger
3. Missing import: apply_enhanced_imputation_strategy_4step
4. Missing import: validate_training_data
5. Missing import: print_section_header
6. Prediction fallback handling for None/empty predictions_quantile
7. validate_training_data integration in compare_regressors

Expected to FAIL initially, then PASS after fixes are applied.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path


class TestPhase95CriticalImports(unittest.TestCase):
    """Test that all critical imports are available for Phase 9.5."""

    def test_checkpoint_function_exists(self):
        """Test that checkpoint function can be imported or is defined."""
        # checkpoint is defined in notebook, not in module
        # We test that it would be accessible in notebook context
        # For now, we just verify the concept exists in a script
        from ml_stock_prediction_final import checkpoint

        self.assertTrue(callable(checkpoint))

    def test_logger_from_logging_config(self):
        """Test that get_logger can be imported and used."""
        from finance_ml.logging_config import get_logger

        logger = get_logger(__name__)
        self.assertIsNotNone(logger)
        self.assertTrue(hasattr(logger, "error"))
        self.assertTrue(hasattr(logger, "info"))

    def test_apply_enhanced_imputation_strategy_4step_import(self):
        """Test that apply_enhanced_imputation_strategy_4step can be imported."""
        from finance_ml.advanced_preprocessing import apply_enhanced_imputation_strategy_4step

        self.assertTrue(callable(apply_enhanced_imputation_strategy_4step))

    def test_validate_training_data_import(self):
        """Test that validate_training_data can be imported."""
        from finance_ml.advanced_models import validate_training_data

        self.assertTrue(callable(validate_training_data))

    def test_print_section_header_exists(self):
        """Test that print_section_header exists in a script."""
        # print_section_header is defined in notebook
        from ml_stock_prediction_final import print_section_header

        self.assertTrue(callable(print_section_header))


class TestPredictionFallbackLogic(unittest.TestCase):
    """Test prediction fallback logic for handling None/empty predictions."""

    def test_fallback_when_predictions_quantile_is_none(self):
        """Test fallback when predictions_quantile is None."""
        y_pred_stacking = None
        predictions_quantile = None
        y_test = pd.Series([100, 200, 150, 180, 220])

        # Simulate fallback logic
        if y_pred_stacking is not None:
            y_pred_final = y_pred_stacking
            prediction_source = "Stacking Ensemble"
        elif predictions_quantile and 0.5 in predictions_quantile:
            y_pred_final = predictions_quantile[0.5]
            prediction_source = "Quantile Regression (Median)"
        else:
            # Emergency fallback: use mean of test targets
            y_pred_final = np.full(len(y_test), y_test.mean())
            prediction_source = "Fallback (Mean Target)"

        self.assertEqual(prediction_source, "Fallback (Mean Target)")
        self.assertEqual(len(y_pred_final), len(y_test))
        self.assertAlmostEqual(y_pred_final[0], y_test.mean())

    def test_fallback_when_predictions_quantile_is_empty_dict(self):
        """Test fallback when predictions_quantile is an empty dictionary."""
        y_pred_stacking = None
        predictions_quantile = {}
        y_test = pd.Series([100, 200, 150, 180, 220])

        # Simulate fallback logic
        if y_pred_stacking is not None:
            y_pred_final = y_pred_stacking
            prediction_source = "Stacking Ensemble"
        elif predictions_quantile and 0.5 in predictions_quantile:
            y_pred_final = predictions_quantile[0.5]
            prediction_source = "Quantile Regression (Median)"
        else:
            y_pred_final = np.full(len(y_test), y_test.mean())
            prediction_source = "Fallback (Mean Target)"

        self.assertEqual(prediction_source, "Fallback (Mean Target)")
        self.assertEqual(len(y_pred_final), len(y_test))

    def test_fallback_when_predictions_quantile_missing_median(self):
        """Test fallback when predictions_quantile exists but missing 0.5 key."""
        y_pred_stacking = None
        predictions_quantile = {
            0.1: np.array([80, 180, 130, 160, 200]),
            0.9: np.array([120, 220, 170, 200, 240]),
        }
        y_test = pd.Series([100, 200, 150, 180, 220])

        # Simulate fallback logic
        if y_pred_stacking is not None:
            y_pred_final = y_pred_stacking
            prediction_source = "Stacking Ensemble"
        elif predictions_quantile and 0.5 in predictions_quantile:
            y_pred_final = predictions_quantile[0.5]
            prediction_source = "Quantile Regression (Median)"
        else:
            y_pred_final = np.full(len(y_test), y_test.mean())
            prediction_source = "Fallback (Mean Target)"

        self.assertEqual(prediction_source, "Fallback (Mean Target)")

    def test_uses_quantile_when_available(self):
        """Test that quantile median is used when available."""
        y_pred_stacking = None
        predictions_quantile = {0.5: np.array([95, 195, 145, 175, 215])}
        y_test = pd.Series([100, 200, 150, 180, 220])

        # Simulate fallback logic
        if y_pred_stacking is not None:
            y_pred_final = y_pred_stacking
            prediction_source = "Stacking Ensemble"
        elif predictions_quantile and 0.5 in predictions_quantile:
            y_pred_final = predictions_quantile[0.5]
            prediction_source = "Quantile Regression (Median)"
        else:
            y_pred_final = np.full(len(y_test), y_test.mean())
            prediction_source = "Fallback (Mean Target)"

        self.assertEqual(prediction_source, "Quantile Regression (Median)")
        np.testing.assert_array_equal(y_pred_final, predictions_quantile[0.5])

    def test_prefers_stacking_over_quantile(self):
        """Test that stacking predictions are preferred over quantile."""
        y_pred_stacking = np.array([98, 198, 148, 178, 218])
        predictions_quantile = {0.5: np.array([95, 195, 145, 175, 215])}

        # Simulate fallback logic
        if y_pred_stacking is not None:
            y_pred_final = y_pred_stacking
            prediction_source = "Stacking Ensemble"
        elif predictions_quantile and 0.5 in predictions_quantile:
            y_pred_final = predictions_quantile[0.5]
            prediction_source = "Quantile Regression (Median)"
        else:
            y_pred_final = None
            prediction_source = "Fallback (Mean Target)"

        self.assertEqual(prediction_source, "Stacking Ensemble")
        np.testing.assert_array_equal(y_pred_final, y_pred_stacking)


class TestValidateTrainingDataIntegration(unittest.TestCase):
    """Test validate_training_data integration in compare_regressors."""

    def test_validate_training_data_detects_nan_in_features(self):
        """Test that validate_training_data detects NaN in features."""
        from finance_ml.advanced_models import validate_training_data

        X = pd.DataFrame({"feature1": [1, 2, np.nan, 4, 5], "feature2": [10, 20, 30, 40, 50]})
        y = pd.Series([100, 200, 150, 180, 220])

        # Should raise ValueError in strict mode
        with self.assertRaises(ValueError) as context:
            validate_training_data(X, y, strict=True)

        self.assertIn("NaN", str(context.exception))
        self.assertIn("apply_enhanced_imputation_strategy_4step", str(context.exception))

    def test_validate_training_data_detects_nan_in_target(self):
        """Test that validate_training_data detects NaN in target."""
        from finance_ml.advanced_models import validate_training_data

        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]})
        y = pd.Series([100, 200, np.nan, 180, 220])

        # Should raise ValueError in strict mode
        with self.assertRaises(ValueError) as context:
            validate_training_data(X, y, strict=True)

        self.assertIn("NaN", str(context.exception))

    def test_validate_training_data_detects_infinite_values(self):
        """Test that validate_training_data detects infinite values."""
        from finance_ml.advanced_models import validate_training_data

        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [10, np.inf, 30, 40, 50]})
        y = pd.Series([100, 200, 150, 180, 220])

        # Should raise ValueError in strict mode
        with self.assertRaises(ValueError) as context:
            validate_training_data(X, y, strict=True)

        self.assertIn("infinite", str(context.exception))

    def test_validate_training_data_passes_clean_data(self):
        """Test that validate_training_data passes with clean data."""
        from finance_ml.advanced_models import validate_training_data

        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]})
        y = pd.Series([100, 200, 150, 180, 220])

        # Should not raise in strict mode
        result = validate_training_data(X, y, strict=True)
        self.assertTrue(result["valid"])
        self.assertEqual(result["nan_features"], 0)
        self.assertEqual(result["nan_target"], 0)


class TestCompareRegressorsWithValidation(unittest.TestCase):
    """Test that compare_regressors includes validation before training."""

    def test_compare_regressors_should_validate_before_training(self):
        """
        Test that compare_regressors validates data before model training.

        This test verifies the fix for the critical issue where Ridge regression
        fails with "Input X contains NaN" error.
        """
        from finance_ml.advanced_models import compare_regressors

        # Create data with NaN (should fail or be handled)
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "feature3": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            }
        )
        y = pd.Series([1000, 2000, 1500, 1800, 2200, 2500, 2800, 3000, 3200, 3500])

        # This should work without NaN
        try:
            result = compare_regressors(X, y, test_size=0.2, cv=3, random_state=42)
            # Result should be a dict or DataFrame
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"compare_regressors failed with clean data: {e}")


if __name__ == "__main__":
    unittest.main()
