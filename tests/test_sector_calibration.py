import unittest
import pandas as pd
import numpy as np

try:
    from finance_ml.ml_workflow.regression.calibration import (
        calibrate_predictions_by_sector,
        DEFAULT_SECTOR_BIAS,
    )
except Exception:
    # Fallback import directly from file to avoid heavy package __init__
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[1] / "finance_ml" / "ml_workflow" / "regression" / "calibration.py"
    spec = importlib.util.spec_from_file_location("calibration", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.load_module(mod.__name__)  # type: ignore[attr-defined]
    calibrate_predictions_by_sector = getattr(mod, "calibrate_predictions_by_sector")
    DEFAULT_SECTOR_BIAS = getattr(mod, "DEFAULT_SECTOR_BIAS")


class TestSectorCalibration(unittest.TestCase):
    def test_calibration_applies_bias(self):
        df = pd.DataFrame({
            'sector': ['Financials', 'Industrials', 'Energy'],
            'y_pred': [100.0, 200.0, 300.0],
        })
        out = calibrate_predictions_by_sector(df)
        # Defaults exist for Financials and Industrials; Energy remains unchanged
        self.assertIn('y_pred_calibrated', out.columns)
        self.assertAlmostEqual(out.loc[0, 'y_pred_calibrated'], 100.0 + DEFAULT_SECTOR_BIAS['Financials'])
        self.assertAlmostEqual(out.loc[1, 'y_pred_calibrated'], 200.0 + DEFAULT_SECTOR_BIAS['Industrials'])
        self.assertAlmostEqual(out.loc[2, 'y_pred_calibrated'], 300.0)

    def test_custom_bias_mapping(self):
        df = pd.DataFrame({
            'sector': ['Tech', 'Tech'],
            'y_pred': [50.0, 60.0],
        })
        custom = {'Tech': -5.0}
        out = calibrate_predictions_by_sector(df, sector_bias=custom)
        self.assertTrue(np.allclose(out['y_pred_calibrated'], [45.0, 55.0]))

    def test_missing_sector_column(self):
        df = pd.DataFrame({'y_pred': [1.0, 2.0]})
        out = calibrate_predictions_by_sector(df)
        self.assertIn('y_pred_calibrated', out.columns)
        self.assertTrue(np.allclose(out['y_pred_calibrated'], out['y_pred']))

    def test_missing_pred_column_raises(self):
        df = pd.DataFrame({'sector': ['A']})
        with self.assertRaises(ValueError):
            calibrate_predictions_by_sector(df)


if __name__ == '__main__':
    unittest.main()
