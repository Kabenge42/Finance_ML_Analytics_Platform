"""
Tests for column semantic classification.

Tests semantic understanding of column types for preprocessing:
- Price columns (never transform)
- Market value columns (log-transform)
- Ratio columns (pre-normalized)
- Percentage columns (bounded [0, 100])
- Count columns (discrete)

Aligned with preprocessing_stages_4-8_improvement_plan.md Task 1.1
"""

import unittest
from typing import Set, List

import pandas as pd


class TestColumnSemantics(unittest.TestCase):
    """Test semantic column classification functions."""

    def test_identify_price_columns(self):
        """Price columns should be identified and excluded from winsorization."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import PRICE_COLUMNS
        
        # Expected price columns that must be preserved
        expected = {
            'last_price',
            'price_target',
            'price_target_median',
            'price_target_ytd_ago',
            'price_target_12m_ago',
        }
        
        # PRICE_COLUMNS should contain all expected price columns
        self.assertTrue(expected.issubset(PRICE_COLUMNS), 
                       f"Missing price columns: {expected - PRICE_COLUMNS}")
        
        # Verify critical business columns are present
        self.assertIn('last_price', PRICE_COLUMNS)
        self.assertIn('price_target', PRICE_COLUMNS)

    def test_identify_market_value_columns(self):
        """Market cap/value columns requiring log-transforms."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import MARKET_VALUE_COLUMNS
        
        # Expected market value columns with high skewness
        expected = {
            'market_cap',
            'ev',
            'total_assets',
            'revenue',
            'total_debt',
            'ebitda',
            'operating_income',
            'net_income',
            'cash_and_equivalents',
        }
        
        # MARKET_VALUE_COLUMNS should contain highly skewed financial metrics
        self.assertTrue(expected.issubset(MARKET_VALUE_COLUMNS),
                       f"Missing market value columns: {expected - MARKET_VALUE_COLUMNS}")
        
        # Verify critical skewed columns
        self.assertIn('market_cap', MARKET_VALUE_COLUMNS)
        self.assertIn('revenue', MARKET_VALUE_COLUMNS)

    def test_identify_ratio_columns(self):
        """Financial ratios already normalized."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import RATIO_COLUMNS
        
        # Expected ratio columns (pre-normalized)
        expected_valuation = {'p_e', 'p_b', 'p_s', 'ev_ebitda', 'ev_sales'}
        expected_profitability = {'roe', 'roa', 'roic'}
        expected_leverage = {'debt_equity', 'current_ratio'}
        
        expected = expected_valuation | expected_profitability | expected_leverage
        
        # RATIO_COLUMNS should contain financial ratios
        overlap = expected & RATIO_COLUMNS
        self.assertGreater(len(overlap), 5, 
                          f"Expected at least 6 ratio columns, found {len(overlap)}")
        
        # Verify common ratios
        self.assertIn('p_e', RATIO_COLUMNS)
        self.assertIn('roe', RATIO_COLUMNS)

    def test_identify_percentage_columns(self):
        """Percentage columns bounded [0, 100]."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import PERCENTAGE_COLUMNS
        
        # Expected percentage columns
        expected_margins = {'gross_margin', 'operating_margin', 'net_margin', 'ebitda_margin'}
        expected_growth = {'revenue_growth_yoy', 'earnings_growth_yoy'}
        expected_volatility = {'volatility_20d', 'volatility_1y'}
        
        expected = expected_margins | expected_growth | expected_volatility
        
        # PERCENTAGE_COLUMNS should contain margin/growth/volatility metrics
        overlap = expected & PERCENTAGE_COLUMNS
        self.assertGreater(len(overlap), 3,
                          f"Expected at least 4 percentage columns, found {len(overlap)}")

    def test_identify_count_columns(self):
        """Discrete count columns."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import COUNT_COLUMNS
        
        # Expected count columns
        expected = {
            'num_analysts',
            'num_employees',
            'num_strong_buy_ratings',
            'num_buy_ratings',
            'num_hold_ratings',
        }
        
        # COUNT_COLUMNS should contain discrete count metrics
        overlap = expected & COUNT_COLUMNS
        self.assertGreater(len(overlap), 2,
                          f"Expected at least 3 count columns, found {len(overlap)}")

    def test_get_winsorizable_columns(self):
        """Return columns safe for winsorization (exclude prices, ratios, percentages)."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import get_winsorizable_columns
        
        # Create test DataFrame with mixed column types
        test_columns = [
            'last_price',  # Price - exclude
            'price_target',  # Price - exclude
            'market_cap',  # Market value - include
            'p_e',  # Ratio - exclude
            'gross_margin',  # Percentage - exclude
            'revenue',  # Market value - include
            'num_analysts',  # Count - exclude
            'total_assets',  # Market value - include
        ]
        
        winsorizable = get_winsorizable_columns(test_columns)
        
        # Should include market value columns
        self.assertIn('market_cap', winsorizable)
        self.assertIn('revenue', winsorizable)
        self.assertIn('total_assets', winsorizable)
        
        # Should exclude price columns
        self.assertNotIn('last_price', winsorizable)
        self.assertNotIn('price_target', winsorizable)
        
        # Should exclude ratios and percentages
        self.assertNotIn('p_e', winsorizable)
        self.assertNotIn('gross_margin', winsorizable)

    def test_get_log_transform_columns(self):
        """Return columns requiring log-transform (market values)."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import get_log_transform_columns
        
        test_columns = [
            'last_price',  # Price - exclude
            'market_cap',  # Market value - include
            'revenue',  # Market value - include
            'p_e',  # Ratio - exclude
            'ebitda',  # Market value - include
        ]
        
        log_transform = get_log_transform_columns(test_columns)
        
        # Should include market value columns
        self.assertIn('market_cap', log_transform)
        self.assertIn('revenue', log_transform)
        self.assertIn('ebitda', log_transform)
        
        # Should exclude price and ratio columns
        self.assertNotIn('last_price', log_transform)
        self.assertNotIn('p_e', log_transform)

    def test_get_scalable_columns(self):
        """Return columns safe for scaling (exclude prices)."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import get_scalable_columns
        
        test_columns = [
            'last_price',  # Price - exclude
            'price_target',  # Price - exclude
            'market_cap',  # Include (or log_market_cap)
            'p_e',  # Ratio - include
            'revenue',  # Include (or log_revenue)
            'gross_margin',  # Include
        ]
        
        scalable = get_scalable_columns(test_columns)
        
        # Should exclude price columns (critical requirement)
        self.assertNotIn('last_price', scalable)
        self.assertNotIn('price_target', scalable)
        
        # Should include other numeric features
        self.assertIn('market_cap', scalable)
        self.assertIn('p_e', scalable)
        self.assertIn('revenue', scalable)
        self.assertIn('gross_margin', scalable)

    def test_classify_columns_returns_dict(self):
        """classify_columns should return dict with semantic categories."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import classify_columns
        
        test_columns = [
            'last_price', 'market_cap', 'p_e', 'gross_margin', 'num_analysts'
        ]
        
        result = classify_columns(test_columns)
        
        # Should return dict with semantic category keys
        self.assertIsInstance(result, dict)
        self.assertIn('price', result)
        self.assertIn('market_value', result)
        self.assertIn('ratio', result)
        self.assertIn('percentage', result)
        self.assertIn('count', result)
        
        # Each category should be a set
        for category, columns in result.items():
            self.assertIsInstance(columns, set, f"{category} should be a set")

    def test_semantic_classification_mutually_exclusive(self):
        """Columns should not appear in multiple semantic categories."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import (
            PRICE_COLUMNS, MARKET_VALUE_COLUMNS, RATIO_COLUMNS
        )
        
        # Price and market value should be disjoint
        overlap_price_market = PRICE_COLUMNS & MARKET_VALUE_COLUMNS
        self.assertEqual(len(overlap_price_market), 0,
                        f"Price and market value overlap: {overlap_price_market}")
        
        # Price and ratio should be disjoint
        overlap_price_ratio = PRICE_COLUMNS & RATIO_COLUMNS
        self.assertEqual(len(overlap_price_ratio), 0,
                        f"Price and ratio overlap: {overlap_price_ratio}")


if __name__ == '__main__':
    unittest.main()
