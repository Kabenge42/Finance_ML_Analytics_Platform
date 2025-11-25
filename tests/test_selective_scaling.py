"""
Tests for selective scaling with semantic column exclusions.

Tests that scaling respects semantic column types:
- Price columns (never scale - preserve original dollar values)
- Log-transformed columns (should be scaled)
- Other numeric features (should be scaled)

Aligned with preprocessing_stages_4-8_improvement_plan.md Task 3.1
"""

import unittest
import numpy as np
import pandas as pd


class TestSelectiveScaling(unittest.TestCase):
    """Test selective scaling with semantic column exclusions."""

    def setUp(self):
        """Create test DataFrame with mixed column types."""
        np.random.seed(42)
        n = 100
        
        self.df = pd.DataFrame({
            # Identifiers
            'ticker': [f'TICK{i:03d}' for i in range(n)],
            'sector': np.random.choice(['Technology', 'Finance', 'Healthcare'], n),
            
            # Price columns (should NEVER be scaled)
            'last_price': np.random.uniform(10, 200, n),
            'price_target': np.random.uniform(20, 250, n),
            
            # Market value columns (should be scaled)
            'market_cap': np.random.lognormal(10, 2, n),
            'revenue': np.random.lognormal(8, 1.5, n),
            
            # Log-transformed columns (should be scaled)
            'log_market_cap': np.random.normal(10, 2, n),
            'log_revenue': np.random.normal(8, 1.5, n),
            
            # Ratio columns (should be scaled)
            'p_e': np.random.uniform(5, 50, n),
            'roe': np.random.uniform(-10, 30, n),
        })

    def test_scale_features_excludes_price_columns(self):
        """Scaling should skip price columns to preserve interpretability."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features
        
        original_last_price = self.df['last_price'].copy()
        original_price_target = self.df['price_target'].copy()
        
        # Scale with default exclude_price_columns=True
        result = scale_features(
            self.df,
            columns=None,  # Process all numeric columns
            scaler_type='robust',
            by_sector=False,
            exclude_price_columns=True,
        )
        
        # Price columns should be unchanged (critical requirement)
        pd.testing.assert_series_equal(
            result['last_price'],
            original_last_price,
            check_names=False,
            obj='last_price should not be scaled'
        )
        pd.testing.assert_series_equal(
            result['price_target'],
            original_price_target,
            check_names=False,
            obj='price_target should not be scaled'
        )

    def test_scale_features_handles_log_transformed_columns(self):
        """Log-transformed columns should be scaled."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features
        
        original_log_market_cap = self.df['log_market_cap'].copy()
        
        result = scale_features(
            self.df,
            columns=None,
            scaler_type='robust',
            by_sector=False,
            exclude_price_columns=True,
        )
        
        # Log-transformed columns should be scaled
        self.assertFalse(
            result['log_market_cap'].equals(original_log_market_cap),
            "log_market_cap should be scaled"
        )
        
        # Verify scaling (mean should be close to 0 for standard scaler)
        # For robust scaler, median should be close to 0
        self.assertLess(
            abs(result['log_market_cap'].median()),
            0.5,
            "Scaled column should have median near 0"
        )

    def test_scale_features_by_sector_with_exclusions(self):
        """Sector-specific scaling respects exclusions."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features
        
        original_last_price = self.df['last_price'].copy()
        
        # Sector-specific scaling with semantic exclusions
        result = scale_features(
            self.df,
            columns=None,
            scaler_type='robust',
            by_sector=True,  # Sector-specific
            exclude_price_columns=True,
        )
        
        # Price columns should be unchanged across all sectors
        pd.testing.assert_series_equal(
            result['last_price'],
            original_last_price,
            check_names=False
        )
        
        # Verify result has same shape
        self.assertEqual(result.shape, self.df.shape)

    def test_scale_features_backward_compatible(self):
        """Existing code should work with new parameters (backward compatibility)."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features
        
        # Old-style call without new parameters
        result = scale_features(
            self.df,
            columns=['market_cap', 'revenue'],
            scaler_type='robust',
            by_sector=True,
        )
        
        # Should execute without error
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, self.df.shape)

    def test_scale_features_allows_price_scaling_when_disabled(self):
        """Price columns CAN be scaled if exclude_price_columns=False."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features
        
        original_last_price = self.df['last_price'].copy()
        
        # Explicitly allow price scaling (not recommended)
        result = scale_features(
            self.df,
            columns=['last_price'],
            scaler_type='robust',
            by_sector=False,
            exclude_price_columns=False,  # Disable protection
        )
        
        # Price should be scaled when protection is disabled
        self.assertFalse(
            result['last_price'].equals(original_last_price),
            "last_price should be scaled when exclude_price_columns=False"
        )

    def test_scale_features_respects_column_whitelist(self):
        """Only whitelisted columns should be scaled."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features
        
        original_last_price = self.df['last_price'].copy()
        original_market_cap = self.df['market_cap'].copy()
        original_p_e = self.df['p_e'].copy()
        
        # Explicitly scale only log_market_cap
        result = scale_features(
            self.df,
            columns=['log_market_cap'],  # Whitelist only log_market_cap
            scaler_type='robust',
            by_sector=False,
        )
        
        # Other columns should be unchanged
        pd.testing.assert_series_equal(result['last_price'], original_last_price, check_names=False)
        pd.testing.assert_series_equal(result['market_cap'], original_market_cap, check_names=False)
        pd.testing.assert_series_equal(result['p_e'], original_p_e, check_names=False)
        
        # log_market_cap should be scaled
        self.assertFalse(result['log_market_cap'].equals(self.df['log_market_cap']))

    def test_scale_features_handles_missing_sector_column(self):
        """Scaling should work when sector column is missing."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features
        
        df_no_sector = self.df.drop(columns=['sector'])
        
        result = scale_features(
            df_no_sector,
            columns=['market_cap', 'log_market_cap'],
            scaler_type='robust',
            by_sector=True,  # Requested but not available
            exclude_price_columns=True,
        )
        
        # Should fallback to global scaling
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, df_no_sector.shape)

    def test_scale_features_with_standard_scaler(self):
        """Standard scaler should work with exclusions."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features
        
        original_last_price = self.df['last_price'].copy()
        
        result = scale_features(
            self.df,
            columns=None,
            scaler_type='standard',
            by_sector=False,
            exclude_price_columns=True,
        )
        
        # Price should be preserved
        pd.testing.assert_series_equal(result['last_price'], original_last_price, check_names=False)
        
        # Other columns should be scaled (mean ~ 0, std ~ 1)
        self.assertLess(abs(result['log_market_cap'].mean()), 0.5)
        self.assertLess(abs(result['log_market_cap'].std() - 1.0), 0.5)

    def test_scale_features_preserves_original_values_for_business_metric(self):
        """Critical: Price columns must preserve exact values for (Target - Price) / Price."""
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features
        
        # Create specific test case for business metric
        test_df = pd.DataFrame({
            'ticker': ['AAPL'],
            'last_price': [150.0],
            'price_target': [180.0],
            'market_cap': [2.5e12],
        })
        
        result = scale_features(
            test_df,
            columns=None,
            scaler_type='robust',
            by_sector=False,
            exclude_price_columns=True,
        )
        
        # Verify exact values preserved
        self.assertEqual(result['last_price'].iloc[0], 150.0)
        self.assertEqual(result['price_target'].iloc[0], 180.0)
        
        # Business metric should still be calculable
        expected_return = (180.0 - 150.0) / 150.0
        actual_return = (result['price_target'].iloc[0] - result['last_price'].iloc[0]) / result['last_price'].iloc[0]
        self.assertAlmostEqual(actual_return, expected_return, places=10)


if __name__ == '__main__':
    unittest.main()
