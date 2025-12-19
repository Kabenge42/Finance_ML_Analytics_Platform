"""
Test suite for target leakage detection and prediction validation.

TDD Implementation for ML Workflow Guidelines Critical Issues:
1. Target column leakage detection (price_target, price_target_median, etc.)
2. Zero predictions validation
3. Realistic R² bounds checking (R² >= 0.95 indicates leakage)

Aligned with:
- ml_workflow_guidelines.md v1.1 (Critical Issues Identified)
- code_guidelines.md v1.11 Section 5.4 (Feature Categories)

These tests are written FIRST (TDD Red phase) and should FAIL until
the implementation is added.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_sample_dataframe_with_target_leakage(n_samples=100, random_state=42):
    """
    Create sample data that includes target-related columns in features.

    This simulates the data leakage scenario where price_target or related
    columns are accidentally included in the feature set.
    """
    np.random.seed(random_state)

    sectors = ["Technology", "Financials", "Healthcare", "Energy", "Materials"]

    # Generate base price and derive target from it
    last_price = np.random.uniform(10, 500, n_samples)
    price_target = last_price * np.random.uniform(0.9, 1.3, n_samples)
    price_target_median = price_target * np.random.uniform(0.95, 1.05, n_samples)

    df = pd.DataFrame(
        {
            "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
            "sector": np.random.choice(sectors, n_samples),
            "last_price": last_price,
            "price_target": price_target,  # TARGET - should NOT be in features
            "price_target_median": price_target_median,  # Leaky - should NOT be in features
            "price_target_high": price_target * 1.1,  # Leaky
            "price_target_low": price_target * 0.9,  # Leaky
            "p_e_ratio": np.random.uniform(5, 50, n_samples),
            "ev_ebitda_ratio": np.random.uniform(5, 30, n_samples),
            "gross_margin": np.random.uniform(0.1, 0.9, n_samples),
            "beta_5y": np.random.uniform(0.5, 2.0, n_samples),
            "debt_to_equity": np.random.uniform(0.0, 2.5, n_samples),
        }
    )

    return df


def create_clean_dataframe(n_samples=100, random_state=42):
    """
    Create sample data WITHOUT target-related columns in features.
    """
    np.random.seed(random_state)

    sectors = ["Technology", "Financials", "Healthcare", "Energy", "Materials"]

    last_price = np.random.uniform(10, 500, n_samples)
    price_target = last_price * np.random.uniform(0.9, 1.3, n_samples)

    df = pd.DataFrame(
        {
            "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
            "sector": np.random.choice(sectors, n_samples),
            "last_price": last_price,
            "price_target": price_target,  # TARGET column only
            "p_e_ratio": np.random.uniform(5, 50, n_samples),
            "ev_ebitda_ratio": np.random.uniform(5, 30, n_samples),
            "gross_margin": np.random.uniform(0.1, 0.9, n_samples),
            "beta_5y": np.random.uniform(0.5, 2.0, n_samples),
            "debt_to_equity": np.random.uniform(0.0, 2.5, n_samples),
        }
    )

    return df


class TestTargetLeakageDetection(unittest.TestCase):
    """Test suite for detecting target column leakage in features."""

    def test_detect_target_leakage_finds_price_target_columns(self):
        """
        Test that detect_target_leakage identifies price_target-related columns.

        CRITICAL: Ridge/Lasso achieving R²=1.0 and MAE=0.0 indicates target leakage.
        The function should detect when target-related columns are in features.
        """
        from finance_ml.ml_workflow.regression.dataset import detect_target_leakage

        df = create_sample_dataframe_with_target_leakage()

        # Features that accidentally include target-related columns
        feature_cols = [
            "last_price",
            "p_e_ratio",
            "ev_ebitda_ratio",
            "gross_margin",
            "price_target_median",  # LEAKY!
            "price_target_high",  # LEAKY!
        ]

        X = df[feature_cols]
        target_col = "price_target"

        result = detect_target_leakage(X, target_col)

        self.assertTrue(result["has_leakage"], "Should detect target leakage")
        self.assertIn("price_target_median", result["leaky_columns"])
        self.assertIn("price_target_high", result["leaky_columns"])

    def test_detect_target_leakage_clean_data_passes(self):
        """
        Test that clean data (no target columns in features) passes validation.
        """
        from finance_ml.ml_workflow.regression.dataset import detect_target_leakage

        df = create_clean_dataframe()

        # Clean features without target-related columns
        feature_cols = ["last_price", "p_e_ratio", "ev_ebitda_ratio", "gross_margin"]
        X = df[feature_cols]
        target_col = "price_target"

        result = detect_target_leakage(X, target_col)

        self.assertFalse(result["has_leakage"], "Clean data should not have leakage")
        self.assertEqual(len(result["leaky_columns"]), 0)

    def test_validate_training_data_detects_leakage(self):
        """
        Test that validate_training_data includes leakage detection.

        The enhanced validate_training_data should check for target leakage
        in addition to NaN/Inf checks.
        """
        from finance_ml.ml_workflow.regression.dataset import validate_training_data

        df = create_sample_dataframe_with_target_leakage()

        # Features with leakage
        feature_cols = ["last_price", "p_e_ratio", "price_target_median"]  # LEAKY!
        X = pd.DataFrame(df[feature_cols])
        y = df["price_target"]

        # Should detect leakage when check_leakage=True
        result = validate_training_data(
            X, y, strict=False, check_leakage=True, target_col="price_target"
        )

        self.assertFalse(result["valid"], "Should fail validation due to leakage")
        self.assertIn("leakage", str(result["issues"]).lower())


class TestZeroPredictionsValidation(unittest.TestCase):
    """Test suite for zero predictions validation."""

    def test_validate_predictions_rejects_zero_predictions(self):
        """
        Test that predictions with y_pred=0 are flagged as invalid.

        CRITICAL: Several stocks (PLTR, BAC, UBER, HD) have y_pred=0.0
        indicating model prediction failures.
        """
        from finance_ml.ml_workflow.regression.io import validate_predictions_schema

        df = pd.DataFrame(
            {
                "ticker": ["PLTR", "BAC", "AAPL"],
                "isin": ["US1", "US2", "US3"],
                "sector": ["Tech", "Financials", "Tech"],
                "region": ["US", "US", "US"],
                "last_price": [25.0, 35.0, 150.0],
                "y_true": [30.0, 40.0, 160.0],
                "y_pred": [0.0, 0.0, 155.0],  # Two zero predictions - INVALID
                "y_pred_calibrated": [0.0, 0.0, 155.0],
                "pred_p10": [0.0, 0.0, 140.0],
                "pred_p50": [0.0, 0.0, 155.0],
                "pred_p90": [0.0, 0.0, 170.0],
                "interval_width": [0.0, 0.0, 30.0],
                "abs_error": [30.0, 40.0, 5.0],
                "pct_error": [-1.0, -1.0, 0.03],
                "model_version": ["v9_10", "v9_10", "v9_10"],
                "snapshot_date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            }
        )

        # Should raise error due to zero predictions
        with self.assertRaises(ValueError) as context:
            validate_predictions_schema(df, reject_zero_predictions=True)

        self.assertIn("zero", str(context.exception).lower())

    def test_validate_zero_predictions_function(self):
        """
        Test dedicated function for zero predictions validation.
        """
        from finance_ml.ml_workflow.regression.validation import validate_no_zero_predictions

        y_pred = np.array([100.0, 0.0, 150.0, 0.0, 200.0])

        result = validate_no_zero_predictions(y_pred)

        self.assertFalse(result["valid"])
        self.assertEqual(result["zero_count"], 2)
        self.assertEqual(result["zero_indices"], [1, 3])


class TestRealisticMetricsBounds(unittest.TestCase):
    """Test suite for detecting unrealistic metrics that indicate leakage."""

    def test_detect_leakage_from_perfect_r2(self):
        """
        Test that R²=1.0 is flagged as potential data leakage.

        CRITICAL: Ridge model achieves R²=1.0 and MAE=0.0, which is
        impossible for real financial prediction.
        """
        from finance_ml.ml_workflow.regression.validation import validate_realistic_metrics

        # Perfect metrics indicate leakage
        metrics = {
            "r2": 1.0,
            "mae": 0.0,
            "rmse": 0.0,
        }

        result = validate_realistic_metrics(metrics)

        self.assertFalse(result["valid"], "Perfect R²=1.0 should indicate leakage")
        self.assertTrue(result["suspected_leakage"])
        self.assertIn("r2", result["issues"])

    def test_detect_leakage_from_near_perfect_r2(self):
        """
        Test that R² >= 0.95 triggers a warning for potential leakage.
        """
        from finance_ml.ml_workflow.regression.validation import validate_realistic_metrics

        # Near-perfect metrics are suspicious
        metrics = {
            "r2": 0.98,
            "mae": 10.0,
            "rmse": 15.0,
        }

        result = validate_realistic_metrics(metrics, r2_threshold=0.95)

        self.assertFalse(result["valid"])
        self.assertTrue(result["suspected_leakage"])

    def test_realistic_metrics_pass_validation(self):
        """
        Test that realistic metrics (R² in 0.60-0.90 range) pass validation.
        """
        from finance_ml.ml_workflow.regression.validation import validate_realistic_metrics

        # Realistic metrics for financial prediction
        metrics = {
            "r2": 0.75,
            "mae": 1500.0,
            "rmse": 2500.0,
        }

        result = validate_realistic_metrics(metrics)

        self.assertTrue(result["valid"])
        self.assertFalse(result["suspected_leakage"])


class TestQuantileMonotonicityEnforcement(unittest.TestCase):
    """Test suite for quantile monotonicity enforcement."""

    def test_enforce_quantile_monotonicity(self):
        """
        Test that quantile predictions are enforced to be monotonic.

        Ensures pred_p10 <= pred_p50 <= pred_p90 after enforcement.
        """
        from finance_ml.ml_workflow.regression.validation import enforce_quantile_monotonicity

        # Predictions that violate monotonicity
        predictions = pd.DataFrame(
            {
                "pred_p10": [100.0, 120.0, 90.0],  # Row 1: p10 > p50 (violation)
                "pred_p50": [110.0, 100.0, 100.0],  # Row 1: p50 < p10 (violation)
                "pred_p90": [120.0, 130.0, 110.0],
            }
        )

        fixed = enforce_quantile_monotonicity(predictions)

        # Check monotonicity is enforced
        self.assertTrue((fixed["pred_p10"] <= fixed["pred_p50"]).all())
        self.assertTrue((fixed["pred_p50"] <= fixed["pred_p90"]).all())

    def test_validate_quantile_monotonicity(self):
        """
        Test validation function for quantile monotonicity.
        """
        from finance_ml.ml_workflow.regression.validation import validate_quantile_monotonicity

        # Valid monotonic predictions
        valid_preds = pd.DataFrame(
            {
                "pred_p10": [100.0, 90.0],
                "pred_p50": [110.0, 100.0],
                "pred_p90": [120.0, 110.0],
            }
        )

        result = validate_quantile_monotonicity(valid_preds)
        self.assertTrue(result["valid"])

        # Invalid non-monotonic predictions
        invalid_preds = pd.DataFrame(
            {
                "pred_p10": [120.0, 90.0],  # p10 > p50 (violation)
                "pred_p50": [110.0, 100.0],
                "pred_p90": [100.0, 110.0],  # p90 < p50 (violation)
            }
        )

        result = validate_quantile_monotonicity(invalid_preds)
        self.assertFalse(result["valid"])
        self.assertGreater(result["violation_count"], 0)


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
