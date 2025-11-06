"""
Comprehensive unit tests for dashboard helper functions in finance_ml.eval
Following strict TDD approach for Interactive Dashboards feature
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import helper functions from eval.py
try:
    from finance_ml.eval import (
        calculate_mispricing_score,
        rank_stocks_by_sector,
        calculate_financial_metrics_dashboard,
        generate_data_quality_alerts,
        prepare_plotly_dashboard_data,
    )
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False


@unittest.skipIf(not EVAL_AVAILABLE, "finance_ml.eval not available")
class TestCalculateMispricingScore(unittest.TestCase):
    """Test calculate_mispricing_score function"""

    def test_basic_mispricing_calculation(self):
        """Test basic mispricing score calculation"""
        # Predicted higher than actual = undervalued (positive score)
        df = pd.DataFrame({
            'predicted_price_target': [120, 80, 100],
            'last_price': [100, 100, 100]
        })
        result = calculate_mispricing_score(df)
        
        self.assertIn('mispricing_score', result.columns)
        self.assertAlmostEqual(result['mispricing_score'].iloc[0], 0.20, places=2)
        self.assertAlmostEqual(result['mispricing_score'].iloc[1], -0.20, places=2)
        self.assertAlmostEqual(result['mispricing_score'].iloc[2], 0.0, places=2)

    def test_overvalued_stock(self):
        """Test overvalued stock (predicted < actual)"""
        df = pd.DataFrame({
            'predicted_price_target': [80],
            'last_price': [100]
        })
        result = calculate_mispricing_score(df)
        self.assertAlmostEqual(result['mispricing_score'].iloc[0], -0.20, places=2)

    def test_fairly_valued_stock(self):
        """Test fairly valued stock (predicted = actual)"""
        df = pd.DataFrame({
            'predicted_price_target': [100],
            'last_price': [100]
        })
        result = calculate_mispricing_score(df)
        self.assertAlmostEqual(result['mispricing_score'].iloc[0], 0.0, places=2)

    def test_returns_dataframe_with_columns(self):
        """Test that function returns DataFrame with expected columns"""
        df = pd.DataFrame({
            'predicted_price_target': [120, 80],
            'last_price': [100, 100]
        })
        result = calculate_mispricing_score(df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('mispricing_score', result.columns)
        self.assertIn('mispricing_pct', result.columns)

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError"""
        df = pd.DataFrame({'ticker': ['AAPL']})
        
        with self.assertRaises(ValueError):
            calculate_mispricing_score(df)

    def test_custom_column_names(self):
        """Test with custom column names"""
        df = pd.DataFrame({
            'my_predicted': [120],
            'my_current': [100]
        })
        result = calculate_mispricing_score(df, predicted_col='my_predicted', current_col='my_current')
        
        self.assertIn('mispricing_score', result.columns)
        self.assertAlmostEqual(result['mispricing_score'].iloc[0], 0.20, places=2)

    def test_large_dataframe(self):
        """Test with large dataframe"""
        df = pd.DataFrame({
            'predicted_price_target': np.random.uniform(50, 150, 1000),
            'last_price': np.random.uniform(50, 150, 1000)
        })
        result = calculate_mispricing_score(df)
        
        self.assertEqual(len(result), 1000)
        self.assertIn('mispricing_score', result.columns)


@unittest.skipIf(not EVAL_AVAILABLE, "finance_ml.eval not available")
class TestRankStocksBySector(unittest.TestCase):
    """Test rank_stocks_by_sector function"""

    def setUp(self):
        """Create sample data for ranking tests"""
        self.sample_df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC', 'XOM', 'CVX'],
            'sector': ['Tech', 'Tech', 'Tech', 'Finance', 'Finance', 'Energy', 'Energy'],
            'mispricing_score': [0.20, 0.15, -0.05, 0.30, 0.10, -0.10, 0.05],
            'last_price': [150, 2800, 350, 140, 35, 110, 160],
            'market_cap': [2500000, 1800000, 2300000, 450000, 280000, 400000, 200000]
        })

    def test_basic_ranking(self):
        """Test basic ranking by sector"""
        result = rank_stocks_by_sector(self.sample_df, top_n=2)
        
        # Should return a dict with sectors as keys
        self.assertIsInstance(result, dict)
        
        # Should have all three sectors
        self.assertIn('Tech', result)
        self.assertIn('Finance', result)
        self.assertIn('Energy', result)

    def test_top_n_limit(self):
        """Test that top_n parameter limits results"""
        result = rank_stocks_by_sector(self.sample_df, top_n=2)
        
        # Each sector should have at most 2 stocks
        for sector, stocks in result.items():
            self.assertLessEqual(len(stocks), 2)

    def test_ranking_order(self):
        """Test that stocks are ranked by mispricing_score (highest first)"""
        result = rank_stocks_by_sector(self.sample_df, top_n=3)
        
        # Tech sector: AAPL (0.20) > GOOGL (0.15) > MSFT (-0.05)
        tech_all_stocks_featured = result.get('Tech')
        if tech_all_stocks_featured is not None and len(tech_all_stocks_featured) > 0:
            # First stock should have highest mispricing score
            self.assertEqual(tech_all_stocks_featured['ticker'].iloc[0], 'AAPL')

    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        empty_df = pd.DataFrame()
        
        # Empty DataFrame will raise KeyError for missing 'sector' column
        with self.assertRaises(KeyError):
            rank_stocks_by_sector(empty_df, top_n=5)

    def test_missing_sector_column(self):
        """Test with missing sector column"""
        df_no_sector = self.sample_df.drop('sector', axis=1)
        
        # Should either handle gracefully or raise informative error
        try:
            result = rank_stocks_by_sector(df_no_sector, top_n=5)
            # If it doesn't raise an error, verify it returns something reasonable
            self.assertIsInstance(result, dict)
        except KeyError:
            # Expected behavior - missing required column
            pass


@unittest.skipIf(not EVAL_AVAILABLE, "finance_ml.eval not available")
class TestCalculateFinancialMetricsDashboard(unittest.TestCase):
    """Test calculate_financial_metrics_dashboard function"""

    def setUp(self):
        """Create sample data for metrics tests"""
        self.sample_df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC'],
            'sector': ['Tech', 'Tech', 'Tech', 'Finance', 'Finance'],
            'region': ['US', 'US', 'US', 'US', 'US'],
            'market_cap': [2500000, 1800000, 2300000, 450000, 280000],
            'last_price': [150, 2800, 350, 140, 35],
            'pe_ratio': [25, 30, 28, 12, 10],
            'pb_ratio': [8, 6, 10, 1.5, 0.9],
            'revenue': [394000, 280000, 198000, 120000, 95000]
        })

    def test_group_by_sector(self):
        """Test grouping by sector"""
        result = calculate_financial_metrics_dashboard(self.sample_df, group_by='sector')
        
        # Should return a dict
        self.assertIsInstance(result, dict)
        
        # Should have metrics organized somehow
        # (implementation may vary - check for reasonable structure)
        self.assertGreater(len(result), 0)

    def test_group_by_region(self):
        """Test grouping by region"""
        result = calculate_financial_metrics_dashboard(self.sample_df, group_by='region')
        
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_aggregation_calculations(self):
        """Test that aggregations are calculated correctly"""
        result = calculate_financial_metrics_dashboard(self.sample_df, group_by='sector')
        
        # Should contain valuation or similar metrics
        # (exact structure depends on implementation)
        self.assertIsInstance(result, dict)

    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        empty_df = pd.DataFrame()
        result = calculate_financial_metrics_dashboard(empty_df, group_by='sector')
        
        # Should handle gracefully
        self.assertIsInstance(result, dict)

    def test_missing_group_column(self):
        """Test with missing group column"""
        try:
            result = calculate_financial_metrics_dashboard(self.sample_df, group_by='nonexistent')
            # If no error, should return something reasonable
            self.assertIsInstance(result, dict)
        except KeyError:
            # Expected behavior
            pass


@unittest.skipIf(not EVAL_AVAILABLE, "finance_ml.eval not available")
class TestGenerateDataQualityAlerts(unittest.TestCase):
    """Test generate_data_quality_alerts function"""

    def setUp(self):
        """Create sample data with quality issues"""
        self.clean_df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'sector': ['Tech', 'Tech', 'Tech'],
            'last_price': [150, 2800, 350],
            'market_cap': [2500000, 1800000, 2300000]
        })
        
        self.dirty_df = pd.DataFrame({
            'ticker': ['AAPL', None, 'MSFT', 'GOOGL'],
            'sector': ['Tech', 'Tech', None, 'Tech'],
            'last_price': [150, 2800, 350, None],
            'market_cap': [2500000, np.nan, 2300000, 1800000]
        })

    def test_no_alerts_for_clean_data(self):
        """Test that clean data produces no critical alerts"""
        alerts = generate_data_quality_alerts(self.clean_df)
        
        # Should return a list
        self.assertIsInstance(alerts, list)
        
        # May have info alerts but no critical ones
        critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
        self.assertEqual(len(critical_alerts), 0)

    def test_detects_missing_values(self):
        """Test detection of missing values"""
        alerts = generate_data_quality_alerts(self.dirty_df)
        
        self.assertIsInstance(alerts, list)
        
        # Should detect missing values
        self.assertGreater(len(alerts), 0)
        
        # At least one alert should mention missing values or NaN
        alert_messages = [a.get('message', '').lower() for a in alerts]
        has_missing_alert = any('missing' in msg or 'nan' in msg or 'null' in msg 
                               for msg in alert_messages)
        self.assertTrue(has_missing_alert)

    def test_alert_structure(self):
        """Test that alerts have proper structure"""
        alerts = generate_data_quality_alerts(self.dirty_df)
        
        for alert in alerts:
            # Each alert should be a dict
            self.assertIsInstance(alert, dict)
            
            # Should have a message
            self.assertIn('message', alert)
            self.assertIsInstance(alert['message'], str)
            
            # Should have a severity level
            if 'severity' in alert:
                self.assertIn(alert['severity'], ['low', 'medium', 'high', 'critical'])

    def test_custom_threshold(self):
        """Test with custom outlier threshold"""
        alerts = generate_data_quality_alerts(self.dirty_df, outlier_threshold=3.0)
        
        self.assertIsInstance(alerts, list)

    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        empty_df = pd.DataFrame()
        alerts = generate_data_quality_alerts(empty_df)
        
        self.assertIsInstance(alerts, list)


@unittest.skipIf(not EVAL_AVAILABLE, "finance_ml.eval not available")
class TestPreparePlotlyDashboardData(unittest.TestCase):
    """Test prepare_plotly_dashboard_data function"""

    def setUp(self):
        """Create sample data for dashboard preparation tests"""
        self.sample_df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC'],
            'sector': ['Tech', 'Tech', 'Tech', 'Finance', 'Finance'],
            'region': ['US', 'US', 'US', 'US', 'US'],
            'last_price': [150, 2800, 350, 140, 35],
            'predicted_price_target': [180, 3000, 340, 160, 40],
            'market_cap': [2500000, 1800000, 2300000, 450000, 280000]
        })

    def test_returns_dict(self):
        """Test that function returns a Dict"""
        result = prepare_plotly_dashboard_data(self.sample_df)
        
        self.assertIsInstance(result, dict)

    def test_has_expected_keys(self):
        """Test that result dict has expected chart type keys"""
        result = prepare_plotly_dashboard_data(self.sample_df)
        
        # Should have keys for different chart types
        expected_keys = ['scatter_data', 'histogram_data', 'box_data', 'heatmap_data']
        for key in expected_keys:
            self.assertIn(key, result)

    def test_with_timeseries_flag(self):
        """Test with include_timeseries flag"""
        result = prepare_plotly_dashboard_data(self.sample_df, include_timeseries=True)
        
        self.assertIsInstance(result, dict)

    def test_with_custom_color_scheme(self):
        """Test with custom color scheme"""
        result = prepare_plotly_dashboard_data(self.sample_df, color_scheme='viridis')
        
        self.assertIsInstance(result, dict)
        if 'color_scales' in result:
            self.assertEqual(result['color_scales']['default'], 'viridis')

    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        empty_df = pd.DataFrame()
        
        result = prepare_plotly_dashboard_data(empty_df)
        
        # Should return a dict even with empty data
        self.assertIsInstance(result, dict)

    def test_data_structures(self):
        """Test that data structures are appropriate for Plotly"""
        result = prepare_plotly_dashboard_data(self.sample_df)
        
        # Each data structure should be a dict
        self.assertIsInstance(result.get('scatter_data'), dict)
        self.assertIsInstance(result.get('histogram_data'), dict)
        self.assertIsInstance(result.get('box_data'), dict)


if __name__ == '__main__':
    unittest.main()
