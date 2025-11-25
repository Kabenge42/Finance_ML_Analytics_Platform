"""
Tests for selective winsorization with semantic column exclusions.

Tests that winsorization respects semantic column types:
- Price columns (never winsorize)
- Ratio columns (optional exclusion)
- Market value columns (winsorizable, but log-transform preferred)

Aligned with preprocessing_stages_4-8_improvement_plan.md Task 1.2
"""

import unittest
import numpy as np
import pandas as pd


class TestSelectiveWinsorization(unittest.TestCase):
    """Test selective winsorization with semantic column exclusions."""

    def setUp(self):
        """Create test DataFrame with mixed column types and outliers."""
        np.random.seed(42)
        n = 100
        
        self.df = pd.DataFrame({
            # Identifiers
            'ticker': [f'TICK{i:03d}' for i in range(n)],
            'sector': np.random.choice(['Technology', 'Finance', 'Healthcare'], n),
            
            # Price columns (should never be winsorized)
            'last_price': np.random.uniform(10, 200, n),
            'price_target': np.random.uniform(20, 250, n),
            
            # Market value columns (winsorizable)
            'market_cap': np.random.lognormal(10, 2, n),  # Highly skewed
            'revenue': np.random.lognormal(8, 1.5, n),    # Highly skewed
            
            # Ratio columns (optional exclusion)
            'p_e': np.random.uniform(5, 50, n),
            'roe': np.random.uniform(-10, 30, n),
            
            # Percentage columns (optional exclusion)
            'gross_margin': np.random.uniform(10, 80, n),
        })
        
        # Add extreme outliers to price columns (these should NOT be changed)
        self.df.loc[0, 'last_price'] = 1000.0  # Extreme high price (valid)
        self.df.loc[1, 'last_price'] = 1.0     # Extreme low price (valid)
        self.df.loc[0, 'price_target'] = 1500.0
        
        # Add extreme outliers to market value columns (these CAN be winsorized)
        self.df.loc[2, 'market_cap'] = 1e12    # Mega-cap
        self.df.loc[3, 'revenue'] = 1e11       # Huge revenue

    def test_winsorize_excludes_price_columns(self):
        """Winsorization should skip price columns by default."""
        from finance_ml.ml_workflow.preprocessing.outliers import winsorize_by_sector
        
        original_last_price = self.df['last_price'].copy()
        original_price_target = self.df['price_target'].copy()
        
        # Winsorize with default exclude_price_columns=True
        result = winsorize_by_sector(
            self.df,
            columns=None,  # Process all numeric columns
            lower_percentile=0.05,
            upper_percentile=0.95,
            by_sector=False,
            exclude_price_columns=True,
        )
        
        # Price columns should be unchanged
        pd.testing.assert_series_equal(
            result['last_price'],
            original_last_price,
            check_names=False,
            obj='last_price should not be winsorized'
        )
        pd.testing.assert_series_equal(
            result['price_target'],
            original_price_target,
            check_names=False,
            obj='price_target should not be winsorized'
        )
        
        # Verify extreme price values are preserved
        self.assertEqual(result.loc[0, 'last_price'], 1000.0,
                        "Extreme high price should be preserved")
        self.assertEqual(result.loc[1, 'last_price'], 1.0,
                        "Extreme low price should be preserved")

    def test_winsorize_excludes_ratios(self):
        """Financial ratios should not be winsorized when exclude_ratio_columns=True."""
        from finance_ml.ml_workflow.preprocessing.outliers import winsorize_by_sector
        
        original_p_e = self.df['p_e'].copy()
        original_roe = self.df['roe'].copy()
        
        result = winsorize_by_sector(
            self.df,
            columns=None,
            lower_percentile=0.05,
            upper_percentile=0.95,
            by_sector=False,
            exclude_price_columns=True,
            exclude_ratio_columns=True,
        )
        
        # Ratio columns should be unchanged
        pd.testing.assert_series_equal(
            result['p_e'],
            original_p_e,
            check_names=False,
            obj='p_e ratio should not be winsorized'
        )
        pd.testing.assert_series_equal(
            result['roe'],
            original_roe,
            check_names=False,
            obj='roe ratio should not be winsorized'
        )

    def test_winsorize_respects_column_whitelist(self):
        """Only whitelisted columns should be winsorized."""
        from finance_ml.ml_workflow.preprocessing.outliers import winsorize_by_sector
        
        original_last_price = self.df['last_price'].copy()
        original_market_cap = self.df['market_cap'].copy()
        
        # Explicitly winsorize only revenue
        result = winsorize_by_sector(
            self.df,
            columns=['revenue'],  # Whitelist only revenue
            lower_percentile=0.05,
            upper_percentile=0.95,
            by_sector=False,
        )
        
        # last_price should be unchanged (not in whitelist)
        pd.testing.assert_series_equal(
            result['last_price'],
            original_last_price,
            check_names=False
        )
        
        # market_cap should be unchanged (not in whitelist)
        pd.testing.assert_series_equal(
            result['market_cap'],
            original_market_cap,
            check_names=False
        )
        
        # revenue should be winsorized (in whitelist)
        self.assertNotEqual(
            result['revenue'].max(),
            self.df['revenue'].max(),
            "Revenue should be winsorized"
        )

    def test_winsorize_by_sector_with_semantic_columns(self):
        """Sector-specific winsorization respects semantic classification."""
        from finance_ml.ml_workflow.preprocessing.outliers import winsorize_by_sector
        
        original_last_price = self.df['last_price'].copy()
        
        # Sector-specific winsorization with semantic exclusions
        result = winsorize_by_sector(
            self.df,
            columns=None,
            lower_percentile=0.05,
            upper_percentile=0.95,
            by_sector=True,  # Sector-specific
            exclude_price_columns=True,
            exclude_ratio_columns=True,
        )
        
        # Price columns should be unchanged across all sectors
        pd.testing.assert_series_equal(
            result['last_price'],
            original_last_price,
            check_names=False
        )
        
        # Verify result has same shape
        self.assertEqual(result.shape, self.df.shape)

    def test_winsorize_backward_compatible(self):
        """Existing code should work with new parameters (backward compatibility)."""
        from finance_ml.ml_workflow.preprocessing.outliers import winsorize_by_sector
        
        # Old-style call without new parameters
        result = winsorize_by_sector(
            self.df,
            columns=['market_cap', 'revenue'],
            lower_percentile=0.01,
            upper_percentile=0.99,
            by_sector=True,
        )
        
        # Should execute without error
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, self.df.shape)

    def test_winsorize_market_value_columns_can_be_winsorized(self):
        """Market value columns CAN be winsorized (but log-transform is preferred)."""
        from finance_ml.ml_workflow.preprocessing.outliers import winsorize_by_sector
        
        original_market_cap_max = self.df['market_cap'].max()
        
        # Winsorize market_cap (not a price column)
        result = winsorize_by_sector(
            self.df,
            columns=['market_cap'],
            lower_percentile=0.05,
            upper_percentile=0.95,
            by_sector=False,
        )
        
        # Market cap should be winsorized (capped at 95th percentile)
        self.assertLess(
            result['market_cap'].max(),
            original_market_cap_max,
            "Market cap should be winsorized"
        )

    def test_winsorize_handles_missing_sector_column(self):
        """Winsorization should work when sector column is missing."""
        from finance_ml.ml_workflow.preprocessing.outliers import winsorize_by_sector
        
        df_no_sector = self.df.drop(columns=['sector'])
        
        result = winsorize_by_sector(
            df_no_sector,
            columns=['market_cap'],
            lower_percentile=0.05,
            upper_percentile=0.95,
            by_sector=True,  # Requested but not available
            exclude_price_columns=True,
        )
        
        # Should fallback to global winsorization
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, df_no_sector.shape)

    def test_winsorize_with_all_exclusions(self):
        """Test winsorization with all semantic exclusions enabled."""
        from finance_ml.ml_workflow.preprocessing.outliers import winsorize_by_sector
        
        original_last_price = self.df['last_price'].copy()
        original_p_e = self.df['p_e'].copy()
        original_gross_margin = self.df['gross_margin'].copy()
        
        result = winsorize_by_sector(
            self.df,
            columns=None,  # Auto-detect
            lower_percentile=0.05,
            upper_percentile=0.95,
            by_sector=False,
            exclude_price_columns=True,
            exclude_ratio_columns=True,
        )
        
        # All semantic columns should be excluded
        pd.testing.assert_series_equal(result['last_price'], original_last_price, check_names=False)
        pd.testing.assert_series_equal(result['p_e'], original_p_e, check_names=False)
        
        # Market value columns should still be winsorizable
        self.assertLess(
            result['market_cap'].max(),
            self.df['market_cap'].max(),
            "Market cap should be winsorized"
        )


if __name__ == '__main__':
    unittest.main()
