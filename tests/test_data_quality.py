import unittest

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
class TestDataQualityAndSchemaRoot(unittest.TestCase):
    def test_normalize_columns(self):
        df = pd.DataFrame({
            'Last Price': [10.0],
            'Price Target (Median)': [12.0],
            'Ticker': ['ABC'],
            'Sector': ['Tech'],
        })
        got = mod.normalize_columns(df)
        self.assertIn('last_price', got.columns)
        self.assertIn('price_target_median', got.columns)
        self.assertIn('ticker', got.columns)
        self.assertIn('sector', got.columns)

    def test_validate_schema_pass_and_fail(self):
        df_ok = pd.DataFrame({
            'ticker': ['A', 'B'],
            'sector': ['Tech', 'Energy'],
            'last_price': [10.0, 20.0],
            'price_target': [11.0, 19.0],
        })
        mod.validate_schema(df_ok, require_target=True)

        df_bad = pd.DataFrame({
            'ticker': ['A'],
            'last_price': [10.0],
        })
        # Should fail with require_target=True (missing sector and target)
        with self.assertRaisesRegex(ValueError, r"Missing required columns:.*sector.*and at least one target"):
            mod.validate_schema(df_bad, require_target=True)

        # Should also fail with require_target=False (missing sector, which is always required)
        with self.assertRaisesRegex(ValueError, r"Missing required columns:.*sector"):
            mod.validate_schema(df_bad, require_target=False)


@unittest.skipIf(pd is None or mod is None or np is None, "pandas/numpy not installed")
class TestDataQualityChecks(unittest.TestCase):
    """Phase 1: Enhanced data quality tests per IMPROVEMENT_PLAN.md"""
    
    def test_check_missing_values_report(self):
        """Test that check_missing_values returns a report of missing data patterns"""
        df = pd.DataFrame({
            'ticker': ['A', 'B', 'C', 'D'],
            'sector': ['Tech', 'Energy', None, 'Tech'],
            'last_price': [10.0, None, 15.0, 20.0],
            'price_target': [11.0, 12.0, None, None],
        })
        report = mod.check_missing_values(df)
        # Should return dict with column names as keys and missing counts/percentages
        self.assertIn('sector', report)
        self.assertIn('last_price', report)
        self.assertIn('price_target', report)
        self.assertEqual(report['sector']['count'], 1)
        self.assertEqual(report['last_price']['count'], 1)
        self.assertEqual(report['price_target']['count'], 2)
        self.assertAlmostEqual(report['price_target']['percentage'], 50.0)
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method"""
        df = pd.DataFrame({
            'ticker': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
            'last_price': [10.0, 11.0, 12.0, 11.5, 10.5, 12.5, 11.0, 100.0, 10.0, 11.0],
        })
        outliers = mod.detect_outliers_iqr(df, 'last_price')
        # Should identify the 100.0 value as an outlier
        self.assertIn(7, outliers)  # index 7 has value 100.0
        self.assertEqual(len(outliers), 1)
    
    def test_validate_numeric_ranges(self):
        """Test validation of numeric column ranges"""
        df = pd.DataFrame({
            'ticker': ['A', 'B', 'C'],
            'last_price': [10.0, -5.0, 15.0],  # negative price is invalid
            'market_cap': [1e9, 2e9, -1e8],  # negative market cap is invalid
        })
        issues = mod.validate_numeric_ranges(df)
        # Should flag negative prices and market caps
        self.assertIn('last_price', issues)
        self.assertIn('market_cap', issues)
        self.assertGreater(len(issues['last_price']), 0)
        self.assertGreater(len(issues['market_cap']), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
