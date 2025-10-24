import unittest
import math

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None

try:
    import finance_ml as mod
except Exception:
    mod = None


@unittest.skipIf(pd is None or mod is None, "pandas/sklearn not installed")
class TestFeatureEngineeringRoot(unittest.TestCase):
    def test_engineer_basic_ratios(self):
        df = pd.DataFrame({
            'ticker': ['A', 'B', 'C'],
            'sector': ['Tech', 'Tech', 'Energy'],
            'enterprise_value': [100.0, 200.0, 300.0],
            'ebitda': [10.0, 0.0, 30.0],
            'net_debt': [20.0, 40.0, 0.0],
            'last_price': [10.0, 20.0, 30.0],
            'eps': [2.0, 4.0, -10.0],
            'book_value_per_share': [5.0, 10.0, 15.0],
        })
        got = mod.engineer_basic_ratios(df)
        self.assertAlmostEqual(got.loc[0, 'ev_to_ebitda'], 10.0)
        self.assertTrue(math.isnan(got.loc[1, 'ev_to_ebitda']))
        self.assertAlmostEqual(got.loc[2, 'ev_to_ebitda'], 10.0)
        self.assertAlmostEqual(got.loc[0, 'net_debt_to_ebitda'], 2.0)
        self.assertTrue(math.isnan(got.loc[1, 'net_debt_to_ebitda']))
        self.assertAlmostEqual(got.loc[2, 'net_debt_to_ebitda'], 0.0)
        self.assertAlmostEqual(got.loc[0, 'p_e'], 5.0)
        self.assertAlmostEqual(got.loc[1, 'p_e'], 5.0)
        self.assertAlmostEqual(got.loc[2, 'p_e'], -3.0)
        self.assertAlmostEqual(got.loc[0, 'p_b'], 2.0)
        self.assertAlmostEqual(got.loc[1, 'p_b'], 2.0)
        self.assertAlmostEqual(got.loc[2, 'p_b'], 2.0)


@unittest.skipIf(pd is None or mod is None or np is None, "pandas/numpy not installed")
class TestAdditionalFeatures(unittest.TestCase):
    """Phase 2: Additional feature engineering tests per IMPROVEMENT_PLAN.md"""
    
    def test_engineer_margin_features(self):
        """Test margin feature engineering (gross_margin, operating_margin, net_margin)"""
        df = pd.DataFrame({
            'ticker': ['A', 'B', 'C'],
            'revenue': [1000.0, 2000.0, 1500.0],
            'gross_profit': [400.0, 800.0, 600.0],
            'operating_income': [200.0, 600.0, 300.0],
            'net_income': [100.0, 400.0, 150.0],
        })
        got = mod.engineer_margin_features(df)
        self.assertIn('gross_margin', got.columns)
        self.assertIn('operating_margin', got.columns)
        self.assertIn('net_margin', got.columns)
        self.assertAlmostEqual(got.loc[0, 'gross_margin'], 0.4)
        self.assertAlmostEqual(got.loc[1, 'operating_margin'], 0.3)
        self.assertAlmostEqual(got.loc[2, 'net_margin'], 0.1)
    
    def test_engineer_volatility_features(self):
        """Test volatility feature engineering (price volatility windows)"""
        df = pd.DataFrame({
            'ticker': ['A', 'A', 'A', 'A', 'A'],
            'date': pd.date_range('2024-01-01', periods=5),
            'last_price': [10.0, 11.0, 9.5, 10.5, 12.0],
        })
        got = mod.engineer_volatility_features(df, window=3)
        self.assertIn('price_volatility_3d', got.columns)
        # First 2 rows should have NaN (not enough history), remaining should have values
        self.assertTrue(pd.isna(got.loc[0, 'price_volatility_3d']))
        self.assertTrue(pd.isna(got.loc[1, 'price_volatility_3d']))
        self.assertFalse(pd.isna(got.loc[2, 'price_volatility_3d']))
    
    def test_engineer_revenue_cagr(self):
        """Test revenue CAGR calculation"""
        df = pd.DataFrame({
            'ticker': ['A', 'B', 'C'],
            'revenue_current': [1000.0, 2000.0, 1500.0],
            'revenue_1y_ago': [900.0, 1800.0, 1600.0],
            'revenue_3y_ago': [800.0, 1500.0, 1400.0],
        })
        got = mod.engineer_revenue_cagr(df)
        self.assertIn('revenue_cagr_1y', got.columns)
        self.assertIn('revenue_cagr_3y', got.columns)
        # CAGR_1y for A: (1000/900)^1 - 1 ≈ 0.111
        self.assertAlmostEqual(got.loc[0, 'revenue_cagr_1y'], 0.111, places=3)
        # CAGR_3y for B: (2000/1500)^(1/3) - 1 ≈ 0.1006
        self.assertAlmostEqual(got.loc[1, 'revenue_cagr_3y'], 0.1006, places=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
