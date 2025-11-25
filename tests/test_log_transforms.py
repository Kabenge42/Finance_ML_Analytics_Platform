"""
Tests for log-transform pipeline for highly skewed financial columns.

Tests log-transforms as an alternative to winsorization for market value columns:
- log1p: log(1 + x) for non-negative values
- signed_log: sign(x) * log(1 + |x|) for any value including negatives
- Zero and negative value handling
- Null preservation
- Inverse transforms

Aligned with preprocessing_stages_4-8_improvement_plan.md Task 2.1
"""

import unittest
import numpy as np
import pandas as pd
from scipy import stats


class TestLogTransforms(unittest.TestCase):
    """Test log-transform pipeline for skewed financial columns."""

    def setUp(self):
        """Create test DataFrame with highly skewed distributions."""
        np.random.seed(42)
        n = 1000
        
        self.df = pd.DataFrame({
            # Highly skewed market cap (skewness > 2.0)
            'market_cap': np.random.lognormal(10, 2, n),
            
            # Revenue with some zeros
            'revenue': np.concatenate([
                np.random.lognormal(8, 1.5, n - 10),
                np.zeros(10)
            ]),
            
            # Net income (can be negative)
            'net_income': np.concatenate([
                np.random.lognormal(5, 1, int(n * 0.8)),
                -np.random.lognormal(4, 1, int(n * 0.2))
            ]),
            
            # Column with nulls
            'ebitda': np.random.lognormal(7, 1.5, n),
            
            # Non-skewed column (should not be transformed)
            'p_e': np.random.uniform(5, 50, n),
        })
        
        # Add some nulls to ebitda
        null_indices = np.random.choice(n, size=50, replace=False)
        self.df.loc[null_indices, 'ebitda'] = np.nan

    def test_log_transform_market_cap(self):
        """Market cap should be log-transformed to handle skewness."""
        from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms
        
        original_skewness = stats.skew(self.df['market_cap'].dropna())
        
        # Apply log-transform
        result = apply_log_transforms(
            self.df,
            columns=['market_cap'],
            method='log1p'
        )
        
        # Should create log_market_cap column
        self.assertIn('log_market_cap', result.columns)
        
        # Skewness should be reduced by at least 50%
        log_skewness = stats.skew(result['log_market_cap'].dropna())
        self.assertLess(
            abs(log_skewness),
            abs(original_skewness) * 0.5,
            f"Log-transform should reduce skewness: {original_skewness:.2f} -> {log_skewness:.2f}"
        )
        
        # Original column should still exist
        self.assertIn('market_cap', result.columns)

    def test_log_transform_handles_zeros(self):
        """Log-transform should handle zero values using log1p."""
        from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms
        
        # Revenue has 10 zeros
        self.assertTrue((self.df['revenue'] == 0).any())
        
        result = apply_log_transforms(
            self.df,
            columns=['revenue'],
            method='log1p'
        )
        
        # log1p(0) = 0, should not produce NaN or inf
        self.assertFalse(result['log_revenue'].isna().any())
        self.assertFalse(np.isinf(result['log_revenue']).any())
        
        # Verify log1p(0) = 0
        zero_mask = self.df['revenue'] == 0
        self.assertTrue((result.loc[zero_mask, 'log_revenue'] == 0).all())

    def test_log_transform_handles_negatives(self):
        """Negative values should be handled with signed_log."""
        from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms
        
        # Net income has negative values
        self.assertTrue((self.df['net_income'] < 0).any())
        
        result = apply_log_transforms(
            self.df,
            columns=['net_income'],
            method='signed_log'
        )
        
        # Should not produce NaN or inf
        self.assertFalse(result['log_net_income'].isna().any())
        self.assertFalse(np.isinf(result['log_net_income']).any())
        
        # Negative values should remain negative in log space
        negative_mask = self.df['net_income'] < 0
        self.assertTrue((result.loc[negative_mask, 'log_net_income'] < 0).all())

    def test_log_transform_preserves_nulls(self):
        """Null values should remain null after transform."""
        from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms
        
        # EBITDA has nulls
        null_count_before = self.df['ebitda'].isna().sum()
        self.assertGreater(null_count_before, 0)
        
        result = apply_log_transforms(
            self.df,
            columns=['ebitda'],
            method='log1p'
        )
        
        # Null count should be preserved
        null_count_after = result['log_ebitda'].isna().sum()
        self.assertEqual(null_count_after, null_count_before)
        
        # Null positions should match
        pd.testing.assert_series_equal(
            self.df['ebitda'].isna(),
            result['log_ebitda'].isna(),
            check_names=False
        )

    def test_inverse_log_transform(self):
        """Log-transform should be reversible."""
        from finance_ml.ml_workflow.preprocessing.transforms import (
            apply_log_transforms,
            inverse_log_transform
        )
        
        # Apply log-transform
        result = apply_log_transforms(
            self.df,
            columns=['market_cap'],
            method='log1p'
        )
        
        # Inverse transform
        recovered = inverse_log_transform(
            result,
            columns=['log_market_cap'],
            method='log1p'
        )
        
        # Should recover original values (within numerical precision)
        np.testing.assert_array_almost_equal(
            self.df['market_cap'].values,
            recovered['market_cap'].values,
            decimal=5,
            err_msg="Inverse transform should recover original values"
        )

    def test_log_transform_reduces_outlier_impact(self):
        """Log-transformed columns should have fewer IQR outliers."""
        from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms
        
        # Count outliers in original data using IQR method
        Q1 = self.df['market_cap'].quantile(0.25)
        Q3 = self.df['market_cap'].quantile(0.75)
        IQR = Q3 - Q1
        outliers_before = ((self.df['market_cap'] < Q1 - 1.5 * IQR) | 
                          (self.df['market_cap'] > Q3 + 1.5 * IQR)).sum()
        
        # Apply log-transform
        result = apply_log_transforms(
            self.df,
            columns=['market_cap'],
            method='log1p'
        )
        
        # Count outliers in log-transformed data
        Q1_log = result['log_market_cap'].quantile(0.25)
        Q3_log = result['log_market_cap'].quantile(0.75)
        IQR_log = Q3_log - Q1_log
        outliers_after = ((result['log_market_cap'] < Q1_log - 1.5 * IQR_log) | 
                         (result['log_market_cap'] > Q3_log + 1.5 * IQR_log)).sum()
        
        # Should reduce outlier count
        self.assertLess(
            outliers_after,
            outliers_before,
            f"Log-transform should reduce outliers: {outliers_before} -> {outliers_after}"
        )

    def test_log_transform_auto_detect_columns(self):
        """apply_log_transforms should auto-detect market value columns."""
        from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms
        
        # Auto-detect columns (should identify market_cap, revenue, net_income, ebitda)
        result = apply_log_transforms(
            self.df,
            columns=None,  # Auto-detect
            method='signed_log'
        )
        
        # Should create log columns for market value columns
        self.assertIn('log_market_cap', result.columns)
        self.assertIn('log_revenue', result.columns)
        self.assertIn('log_net_income', result.columns)
        self.assertIn('log_ebitda', result.columns)
        
        # Should NOT create log column for p_e (ratio)
        self.assertNotIn('log_p_e', result.columns)

    def test_log_transform_creates_new_columns(self):
        """Log-transform should create new columns, keeping originals."""
        from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms
        
        original_cols = set(self.df.columns)
        
        result = apply_log_transforms(
            self.df,
            columns=['market_cap', 'revenue'],
            method='log1p'
        )
        
        # Original columns should still exist
        for col in original_cols:
            self.assertIn(col, result.columns)
        
        # New log columns should be added
        self.assertIn('log_market_cap', result.columns)
        self.assertIn('log_revenue', result.columns)
        
        # Should have more columns than original
        self.assertGreater(len(result.columns), len(original_cols))

    def test_log_transform_method_validation(self):
        """Invalid transform method should raise ValueError."""
        from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms
        
        with self.assertRaises(ValueError) as context:
            apply_log_transforms(
                self.df,
                columns=['market_cap'],
                method='invalid_method'
            )
        
        self.assertIn('Unknown method', str(context.exception))


if __name__ == '__main__':
    unittest.main()
