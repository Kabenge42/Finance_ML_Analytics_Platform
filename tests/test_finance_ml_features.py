"""
Test suite for finance_ml.features module

This module tests feature engineering functions following TDD methodology
for Phase 7 refactoring.
"""

import unittest
import pandas as pd
import numpy as np
from typing import List


class TestSafeDiv(unittest.TestCase):
    """Test safe division helper function"""
    
    def test_safe_div_normal_division(self):
        """Should perform normal division for valid values"""
        from finance_ml.features import _safe_div
        numer = pd.Series([10, 20, 30])
        denom = pd.Series([2, 4, 5])
        result = _safe_div(numer, denom)
        expected = pd.Series([5.0, 5.0, 6.0])
        pd.testing.assert_series_equal(result, expected)
    
    def test_safe_div_replaces_inf_with_nan(self):
        """Should replace inf values with NaN"""
        from finance_ml.features import _safe_div
        numer = pd.Series([10, 20, 0])
        denom = pd.Series([0, 0, 5])
        result = _safe_div(numer, denom)
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertTrue(pd.isna(result.iloc[1]))
        self.assertEqual(result.iloc[2], 0.0)
    
    def test_safe_div_handles_negative_values(self):
        """Should handle negative values correctly"""
        from finance_ml.features import _safe_div
        numer = pd.Series([10, -20, 30])
        denom = pd.Series([2, 4, -5])
        result = _safe_div(numer, denom)
        self.assertEqual(result.iloc[0], 5.0)
        self.assertEqual(result.iloc[1], -5.0)
        self.assertEqual(result.iloc[2], -6.0)


class TestEngineerBasicRatios(unittest.TestCase):
    """Test basic ratio feature engineering"""
    
    def test_engineer_basic_ratios_ev_to_ebitda(self):
        """Should compute EV/EBITDA ratio"""
        from finance_ml.features import engineer_basic_ratios
        df = pd.DataFrame({
            'enterprise_value': [1000, 2000, 3000],
            'ebitda': [100, 200, 300]
        })
        result = engineer_basic_ratios(df)
        self.assertIn('ev_to_ebitda', result.columns)
        pd.testing.assert_series_equal(
            result['ev_to_ebitda'],
            pd.Series([10.0, 10.0, 10.0], name='ev_to_ebitda')
        )
    
    def test_engineer_basic_ratios_net_debt_to_ebitda(self):
        """Should compute Net Debt/EBITDA ratio"""
        from finance_ml.features import engineer_basic_ratios
        df = pd.DataFrame({
            'net_debt': [500, 1000, 1500],
            'ebitda': [100, 200, 300]
        })
        result = engineer_basic_ratios(df)
        self.assertIn('net_debt_to_ebitda', result.columns)
        pd.testing.assert_series_equal(
            result['net_debt_to_ebitda'],
            pd.Series([5.0, 5.0, 5.0], name='net_debt_to_ebitda')
        )
    
    def test_engineer_basic_ratios_p_e(self):
        """Should compute P/E ratio"""
        from finance_ml.features import engineer_basic_ratios
        df = pd.DataFrame({
            'last_price': [150, 200, 300],
            'eps': [10, 20, 30]
        })
        result = engineer_basic_ratios(df)
        self.assertIn('p_e', result.columns)
        pd.testing.assert_series_equal(
            result['p_e'],
            pd.Series([15.0, 10.0, 10.0], name='p_e')
        )
    
    def test_engineer_basic_ratios_p_b(self):
        """Should compute P/B ratio"""
        from finance_ml.features import engineer_basic_ratios
        df = pd.DataFrame({
            'last_price': [150, 200, 300],
            'book_value_per_share': [100, 150, 200]
        })
        result = engineer_basic_ratios(df)
        self.assertIn('p_b', result.columns)
        pd.testing.assert_series_equal(
            result['p_b'],
            pd.Series([1.5, 4/3, 1.5], name='p_b')
        )
    
    def test_engineer_basic_ratios_missing_columns(self):
        """Should skip ratios when columns missing"""
        from finance_ml.features import engineer_basic_ratios
        df = pd.DataFrame({
            'last_price': [150, 200, 300]
        })
        result = engineer_basic_ratios(df)
        self.assertNotIn('ev_to_ebitda', result.columns)
        self.assertNotIn('p_e', result.columns)
    
    def test_engineer_basic_ratios_preserves_original_columns(self):
        """Should preserve original columns"""
        from finance_ml.features import engineer_basic_ratios
        df = pd.DataFrame({
            'last_price': [150, 200, 300],
            'eps': [10, 20, 30]
        })
        result = engineer_basic_ratios(df)
        self.assertIn('last_price', result.columns)
        self.assertIn('eps', result.columns)


class TestEngineerMarginFeatures(unittest.TestCase):
    """Test margin feature engineering"""
    
    def test_engineer_margin_features_ebitda_margin(self):
        """Should compute EBITDA margin"""
        from finance_ml.features import engineer_margin_features
        df = pd.DataFrame({
            'ebitda_ltm': [100, 200, 300],
            'total_revenues_ltm': [1000, 2000, 3000]
        })
        result = engineer_margin_features(df)
        self.assertIn('ebitda_margin', result.columns)
        pd.testing.assert_series_equal(
            result['ebitda_margin'],
            pd.Series([0.1, 0.1, 0.1], name='ebitda_margin')
        )
    
    def test_engineer_margin_features_operating_margin(self):
        """Should compute operating margin"""
        from finance_ml.features import engineer_margin_features
        df = pd.DataFrame({
            'operating_income_ltm': [100, 200, 300],
            'total_revenues_ltm': [1000, 2000, 3000]
        })
        result = engineer_margin_features(df)
        self.assertIn('operating_margin', result.columns)
    
    def test_engineer_margin_features_missing_columns(self):
        """Should skip margins when columns missing"""
        from finance_ml.features import engineer_margin_features
        df = pd.DataFrame({
            'other_column': [1, 2, 3]
        })
        result = engineer_margin_features(df)
        self.assertNotIn('ebitda_margin', result.columns)


class TestEngineerVolatilityFeatures(unittest.TestCase):
    """Test volatility feature engineering"""
    
    def test_engineer_volatility_features_creates_avg(self):
        """Should create average volatility feature"""
        from finance_ml.features import engineer_volatility_features
        df = pd.DataFrame({
            'volatility_1m': [0.1, 0.2, 0.3],
            'volatility_3m': [0.15, 0.25, 0.35],
            'volatility_6m': [0.2, 0.3, 0.4]
        })
        result = engineer_volatility_features(df)
        self.assertIn('volatility_avg', result.columns)
    
    def test_engineer_volatility_features_uses_available_columns(self):
        """Should use only available volatility columns"""
        from finance_ml.features import engineer_volatility_features
        df = pd.DataFrame({
            'volatility_1m': [0.1, 0.2, 0.3],
            'volatility_3m': [0.15, 0.25, 0.35]
        })
        result = engineer_volatility_features(df)
        # Should still create avg with available columns
        self.assertIn('volatility_avg', result.columns)
    
    def test_engineer_volatility_features_no_volatility_columns(self):
        """Should handle case with no volatility columns"""
        from finance_ml.features import engineer_volatility_features
        df = pd.DataFrame({
            'other_column': [1, 2, 3]
        })
        result = engineer_volatility_features(df)
        self.assertNotIn('volatility_avg', result.columns)


class TestEngineerRevenueCagr(unittest.TestCase):
    """Test revenue CAGR feature engineering"""
    
    def test_engineer_revenue_cagr_computes_cagr(self):
        """Should compute revenue CAGR"""
        from finance_ml.features import engineer_revenue_cagr
        df = pd.DataFrame({
            'total_revenues_ltm': [1000, 1100, 1210],
            'total_revenues_1fy': [900, 1000, 1100]
        })
        result = engineer_revenue_cagr(df)
        self.assertIn('revenue_cagr_1y', result.columns)
        # CAGR should be approximately 11.1% for first row
        self.assertGreater(result['revenue_cagr_1y'].iloc[0], 0.10)
        self.assertLess(result['revenue_cagr_1y'].iloc[0], 0.12)
    
    def test_engineer_revenue_cagr_handles_zero_base(self):
        """Should handle zero base revenue"""
        from finance_ml.features import engineer_revenue_cagr
        df = pd.DataFrame({
            'total_revenues_ltm': [1000, 1100],
            'total_revenues_1fy': [0, 1000]
        })
        result = engineer_revenue_cagr(df)
        # Should replace inf with NaN
        self.assertTrue(pd.isna(result['revenue_cagr_1y'].iloc[0]))
    
    def test_engineer_revenue_cagr_missing_columns(self):
        """Should skip CAGR when columns missing"""
        from finance_ml.features import engineer_revenue_cagr
        df = pd.DataFrame({
            'other_column': [1, 2, 3]
        })
        result = engineer_revenue_cagr(df)
        self.assertNotIn('revenue_cagr_1y', result.columns)


class TestBuildFeaturesAndTarget(unittest.TestCase):
    """Test feature and target building pipeline"""
    
    def test_build_features_and_target_returns_tuple(self):
        """Should return tuple of (X, y, numeric_features, categorical_features)"""
        from finance_ml.features import build_features_and_target
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'sector': ['Tech', 'Tech'],
            'last_price': [150, 200],
            'market_cap': [1e9, 2e9],
            'price_target': [160, 220]
        })
        X, y, numeric_features, categorical_features = build_features_and_target(df)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertIsInstance(numeric_features, list)
        self.assertIsInstance(categorical_features, list)
    
    def test_build_features_and_target_removes_ticker(self):
        """Should remove ticker from features"""
        from finance_ml.features import build_features_and_target
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'sector': ['Tech', 'Tech'],
            'last_price': [150, 200],
            'price_target': [160, 220]
        })
        X, y, numeric_features, categorical_features = build_features_and_target(df)
        self.assertNotIn('ticker', X.columns)
    
    def test_build_features_and_target_extracts_price_target(self):
        """Should extract price_target as target variable"""
        from finance_ml.features import build_features_and_target
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'sector': ['Tech', 'Tech'],
            'last_price': [150, 200],
            'price_target': [160, 220]
        })
        X, y, numeric_features, categorical_features = build_features_and_target(df)
        pd.testing.assert_series_equal(
            y.reset_index(drop=True),
            pd.Series([160, 220], name='price_target').reset_index(drop=True)
        )
        self.assertNotIn('price_target', X.columns)
    
    def test_build_features_and_target_prefers_price_target_over_median(self):
        """Should prefer price_target over price_target_median"""
        from finance_ml.features import build_features_and_target
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'last_price': [150, 200],
            'price_target': [160, 220],
            'price_target_median': [155, 215]
        })
        X, y, numeric_features, categorical_features = build_features_and_target(df)
        pd.testing.assert_series_equal(
            y.reset_index(drop=True),
            pd.Series([160, 220], name='price_target').reset_index(drop=True)
        )
    
    def test_build_features_and_target_uses_median_if_no_target(self):
        """Should use price_target_median if price_target not available"""
        from finance_ml.features import build_features_and_target
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'last_price': [150, 200],
            'price_target_median': [155, 215]
        })
        X, y, numeric_features, categorical_features = build_features_and_target(df)
        pd.testing.assert_series_equal(
            y.reset_index(drop=True),
            pd.Series([155, 215], name='price_target_median').reset_index(drop=True)
        )
    
    def test_build_features_and_target_no_target_returns_none(self):
        """Should return None for y if no target column"""
        from finance_ml.features import build_features_and_target
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'sector': ['Tech', 'Tech'],
            'last_price': [150, 200]
        })
        X, y, numeric_features, categorical_features = build_features_and_target(df)
        self.assertIsNone(y)
    
    def test_build_features_and_target_identifies_categorical(self):
        """Should identify categorical features"""
        from finance_ml.features import build_features_and_target
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'sector': ['Tech', 'Tech'],
            'industry': ['Software', 'Software'],
            'last_price': [150, 200]
        })
        X, y, numeric_features, categorical_features = build_features_and_target(df)
        self.assertIn('sector', categorical_features)
        self.assertIn('industry', categorical_features)
    
    def test_build_features_and_target_identifies_numeric(self):
        """Should identify numeric features"""
        from finance_ml.features import build_features_and_target
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'sector': ['Tech', 'Tech'],
            'last_price': [150, 200],
            'market_cap': [1e9, 2e9]
        })
        X, y, numeric_features, categorical_features = build_features_and_target(df)
        self.assertIn('last_price', numeric_features)
        self.assertIn('market_cap', numeric_features)


if __name__ == '__main__':
    unittest.main()
