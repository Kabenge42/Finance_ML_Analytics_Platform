"""Test sector-specific bias calibration (Priority 3)."""

import unittest
import pandas as pd
import numpy as np
from finance_ml.ml_workflow.regression.calibration import calibrate_predictions_by_sector


class TestSectorBiasCalibration(unittest.TestCase):
    """Test additive bias calibration per sector."""

    def test_calibration_applies_bias_correctly(self):
        """Test that sector biases are added correctly."""
        df = pd.DataFrame(
            {"sector": ["Financials", "Industrials", "Tech"], "y_pred": [100.0, 100.0, 100.0]}
        )

        bias = {"Financials": -10.0, "Industrials": 5.0}
        result = calibrate_predictions_by_sector(df, sector_bias=bias)

        self.assertAlmostEqual(result.loc[0, "y_pred_calibrated"], 90.0)
        self.assertAlmostEqual(result.loc[1, "y_pred_calibrated"], 105.0)
        self.assertAlmostEqual(result.loc[2, "y_pred_calibrated"], 100.0)  # No bias


if __name__ == "__main__":
    unittest.main()
