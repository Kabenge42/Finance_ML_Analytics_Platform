import unittest
import pandas as pd
import numpy as np

try:
    from finance_ml.ml_workflow.features.sector_specific import (
        engineer_sector_features,
        engineer_features_by_sector,
    )
except Exception:
    # Fallback import directly from file to avoid heavy package __init__
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[1] / "finance_ml" / "ml_workflow" / "features" / "sector_specific.py"
    spec = importlib.util.spec_from_file_location("sector_specific", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.load_module(mod.__name__)  # type: ignore[attr-defined]
    engineer_sector_features = getattr(mod, "engineer_sector_features")
    engineer_features_by_sector = getattr(mod, "engineer_features_by_sector")


class TestSectorSpecificFeatures(unittest.TestCase):
    def test_financials_features_created(self):
        df = pd.DataFrame({
            'sector': ['Financials'] * 3,
            'market_cap': [1e9, 2e9, 3e9],
            'tangible_book_value': [5e8, 1e9, 1.5e9],
            'net_income': [5e7, 8e7, 1.2e8],
            'shareholders_equity': [6e8, 1.1e9, 1.6e9],
            'total_debt': [1e8, 2e8, 1e8],
        })
        out = engineer_features_by_sector(df)
        self.assertIn('p_tbv', out.columns)
        self.assertIn('roe', out.columns)
        self.assertIn('leverage_ratio', out.columns)
        self.assertTrue(np.isfinite(out['p_tbv']).all())
        self.assertTrue(np.isfinite(out['roe']).all())
        self.assertTrue(np.isfinite(out['leverage_ratio']).all())

    def test_it_features_created(self):
        df = pd.DataFrame({
            'sector': ['Information Technology', 'Information Technology'],
            'r_d_expense': [1e7, 2e7],
            'revenue': [1e9, 1.5e9],
            'gross_profit': [5e8, 8e8],
        })
        out = engineer_features_by_sector(df)
        self.assertIn('rd_intensity', out.columns)
        self.assertIn('gross_margin', out.columns)
        self.assertTrue((out['rd_intensity'] >= 0).all())
        self.assertTrue((out['gross_margin'] >= 0).all())

    def test_missing_columns_safe(self):
        df = pd.DataFrame({
            'sector': ['Industrials'],
        })
        # Should not raise and should add reasonable defaults
        out = engineer_features_by_sector(df)
        # Columns may be added with zeros
        # No assertion on specific columns as features depend on availability
        self.assertEqual(len(out), 1)


if __name__ == '__main__':
    unittest.main()
