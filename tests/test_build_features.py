import unittest

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import finance_ml as mod
except Exception:
    mod = None


@unittest.skipIf(pd is None or mod is None, "pandas/sklearn not installed")
class TestBuildFeatures(unittest.TestCase):
    def test_build_features_and_target_with_direct_price_target(self):
        df = pd.DataFrame({
            'ticker': ['A', 'B'],
            'sector': ['Tech', 'Energy'],
            'last_price': [10.0, 20.0],
            'price_target': [12.0, 18.0],
            'feature_x': [1.0, 2.0],
            'feature_y': ['low', 'high'],
        })
        X, y, num_cols, cat_cols = mod.build_features_and_target(df)
        # y should be from price_target and length 2
        self.assertEqual(len(y), 2)
        # price_target must be removed from X
        self.assertNotIn('price_target', X.columns)
        # identifier columns should be dropped from X
        self.assertNotIn('ticker', X.columns)
        # feature columns preserved
        self.assertIn('feature_x', X.columns)
        self.assertIn('feature_y', X.columns)
        # type splits
        self.assertIn('feature_x', num_cols)
        self.assertIn('feature_y', cat_cols)

    def test_build_features_and_target_with_normalization_needed(self):
        # Use raw columns with spaces/parentheses that require normalization
        raw = pd.DataFrame({
            'Ticker': ['A', 'B'],
            'Sector': ['Tech', 'Energy'],
            'Last Price': [10.0, 20.0],
            'Price Target - Median': [12.0, 18.0],
            'Feature Z': [5.0, 6.0],
        })
        df = mod.normalize_columns(raw)
        # Ensure we indeed have the normalized median target name
        self.assertIn('price_target_median', df.columns)
        # Build features/target now
        X, y, num_cols, cat_cols = mod.build_features_and_target(df)
        self.assertEqual(len(y), 2)
        # Target column should be dropped from X
        self.assertNotIn('price_target_median', X.columns)
        # Identifiers dropped
        self.assertNotIn('ticker', X.columns)
        # Numeric feature remains
        self.assertIn('feature_z', X.columns)
        self.assertIn('feature_z', num_cols)


if __name__ == '__main__':
    unittest.main(verbosity=2)
