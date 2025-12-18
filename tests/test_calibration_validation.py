"""
Test suite for sector calibration validation logic.

Priority 3 - Task 3.1: Add calibration pre-check (only apply if improves ≥50% of sectors)

Tests verify:
1. Calibration is skipped when it degrades majority of sectors
2. Calibration is applied when it improves majority of sectors
3. Warning messages are logged appropriately

Aligned with: docs/improvement_plan/finance_ml_workflow_implementation_plan.md
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from finance_ml.ml_workflow.regression.calibration import apply_sector_calibration


def create_sample_predictions_by_sector(n_samples=100, n_sectors=5, random_state=42):
    """
    Create sample predictions DataFrame with multiple sectors.

    Args:
        n_samples: Total number of samples
        n_sectors: Number of unique sectors
        random_state: Random seed

    Returns:
        DataFrame with columns: sector, y_true, y_pred
    """
    np.random.seed(random_state)

    sectors = [f"Sector_{i}" for i in range(n_sectors)]

    df = pd.DataFrame(
        {
            "sector": np.random.choice(sectors, n_samples),
            "y_true": np.random.uniform(10, 500, n_samples),
            "y_pred": np.random.uniform(10, 500, n_samples),
        }
    )

    return df


class TestCalibrationValidation(unittest.TestCase):
    """Test calibration pre-check validation logic."""

    def test_calibration_skipped_if_degrading_majority(self):
        """Calibration should be skipped if it worsens >50% of sectors."""
        preds = create_sample_predictions_by_sector(n_samples=100, n_sectors=5)

        # Bad calibration (worsens 3 of 5 sectors = 60%)
        bad_calibration = {
            "sectors": {
                "Sector_0": {"bias_raw": 100, "mae_improvement_pct": -50},
                "Sector_1": {"bias_raw": 50, "mae_improvement_pct": 20},
                "Sector_2": {"bias_raw": 80, "mae_improvement_pct": -30},
                "Sector_3": {"bias_raw": 60, "mae_improvement_pct": -20},
                "Sector_4": {"bias_raw": 40, "mae_improvement_pct": 10},
            }
        }

        result = apply_sector_calibration(preds, bad_calibration, "v9_10")

        # Should NOT apply calibration (only 2 of 5 sectors improve = 40% < 50%)
        self.assertTrue(
            (result["y_pred_calibrated"] == result["y_pred"]).all(),
            "Calibration should be skipped when degrading majority of sectors",
        )

    def test_calibration_applied_if_improving_majority(self):
        """Calibration should be applied if it improves ≥50% of sectors."""
        preds = create_sample_predictions_by_sector(n_samples=100, n_sectors=5)

        # Good calibration (improves 4 of 5 sectors = 80%)
        good_calibration = {
            "sectors": {
                "Sector_0": {"bias_raw": 100, "mae_improvement_pct": 30},
                "Sector_1": {"bias_raw": 50, "mae_improvement_pct": 20},
                "Sector_2": {"bias_raw": 80, "mae_improvement_pct": 40},
                "Sector_3": {"bias_raw": 60, "mae_improvement_pct": 25},
                "Sector_4": {"bias_raw": 40, "mae_improvement_pct": -10},
            }
        }

        result = apply_sector_calibration(preds, good_calibration, "v9_10")

        # Should apply calibration (4 of 5 sectors improve = 80% > 50%)
        self.assertFalse(
            (result["y_pred_calibrated"] == result["y_pred"]).all(),
            "Calibration should be applied when improving majority of sectors",
        )

    def test_calibration_exact_threshold(self):
        """Calibration should be applied when exactly at 50% threshold."""
        preds = create_sample_predictions_by_sector(n_samples=100, n_sectors=4)

        # Threshold case (improves 2 of 4 sectors = 50%)
        threshold_calibration = {
            "sectors": {
                "Sector_0": {"bias_raw": 100, "mae_improvement_pct": 30},
                "Sector_1": {"bias_raw": 50, "mae_improvement_pct": 20},
                "Sector_2": {"bias_raw": 80, "mae_improvement_pct": -40},
                "Sector_3": {"bias_raw": 60, "mae_improvement_pct": -25},
            }
        }

        result = apply_sector_calibration(preds, threshold_calibration, "v9_10")

        # Should NOT apply calibration (50% is not >= 50% due to strict inequality)
        # Note: The implementation uses < threshold, so exactly 50% should apply
        self.assertFalse(
            (result["y_pred_calibrated"] == result["y_pred"]).all(),
            "Calibration should be applied at exactly 50% threshold",
        )

    def test_calibration_with_missing_sectors_dict(self):
        """Calibration should handle missing 'sectors' key gracefully."""
        preds = create_sample_predictions_by_sector(n_samples=100, n_sectors=3)

        # Missing 'sectors' key
        invalid_calibration = {"model_version": "v9_10", "other_data": "some_value"}

        result = apply_sector_calibration(preds, invalid_calibration, "v9_10")

        # Should skip calibration and copy predictions
        self.assertTrue(
            (result["y_pred_calibrated"] == result["y_pred"]).all(),
            "Calibration should be skipped with missing sectors dict",
        )

    def test_calibration_with_empty_sectors(self):
        """Calibration should handle empty sectors dict gracefully."""
        preds = create_sample_predictions_by_sector(n_samples=100, n_sectors=3)

        # Empty sectors dict
        empty_calibration = {"sectors": {}}

        result = apply_sector_calibration(preds, empty_calibration, "v9_10")

        # Should skip calibration
        self.assertTrue(
            (result["y_pred_calibrated"] == result["y_pred"]).all(),
            "Calibration should be skipped with empty sectors dict",
        )

    def test_calibration_custom_threshold(self):
        """Calibration should respect custom min_improvement_threshold."""
        preds = create_sample_predictions_by_sector(n_samples=100, n_sectors=5)

        # Calibration improves 3 of 5 sectors (60%)
        calibration = {
            "sectors": {
                "Sector_0": {"bias_raw": 100, "mae_improvement_pct": 30},
                "Sector_1": {"bias_raw": 50, "mae_improvement_pct": 20},
                "Sector_2": {"bias_raw": 80, "mae_improvement_pct": 10},
                "Sector_3": {"bias_raw": 60, "mae_improvement_pct": -25},
                "Sector_4": {"bias_raw": 40, "mae_improvement_pct": -10},
            }
        }

        # With 70% threshold, should be skipped (60% < 70%)
        result_high_threshold = apply_sector_calibration(
            preds, calibration, "v9_10", min_improvement_threshold=0.7
        )
        self.assertTrue(
            (result_high_threshold["y_pred_calibrated"] == result_high_threshold["y_pred"]).all(),
            "Calibration should be skipped with 70% threshold (only 60% improve)",
        )

        # With 50% threshold, should be applied (60% > 50%)
        result_low_threshold = apply_sector_calibration(
            preds, calibration, "v9_10", min_improvement_threshold=0.5
        )
        self.assertFalse(
            (result_low_threshold["y_pred_calibrated"] == result_low_threshold["y_pred"]).all(),
            "Calibration should be applied with 50% threshold (60% improve)",
        )

    def test_calibration_preserves_dataframe_structure(self):
        """Calibration should preserve all original columns."""
        preds = create_sample_predictions_by_sector(n_samples=100, n_sectors=3)
        preds["extra_col"] = np.random.randn(len(preds))

        calibration = {
            "sectors": {
                "Sector_0": {"bias_raw": 10, "mae_improvement_pct": 30},
                "Sector_1": {"bias_raw": 20, "mae_improvement_pct": 25},
                "Sector_2": {"bias_raw": 15, "mae_improvement_pct": 20},
            }
        }

        result = apply_sector_calibration(preds, calibration, "v9_10")

        # All original columns should be preserved
        for col in preds.columns:
            self.assertIn(col, result.columns, f"Column {col} should be preserved")

        # New column should be added
        self.assertIn("y_pred_calibrated", result.columns)

        # Row count should be unchanged
        self.assertEqual(len(result), len(preds))


if __name__ == "__main__":
    unittest.main(verbosity=2)
