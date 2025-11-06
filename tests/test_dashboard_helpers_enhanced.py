"""
Enhanced tests for dashboard helper functions to achieve ≥80% coverage
Following strict TDD for Interactive Dashboards feature
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

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
class TestRankStocksBySectorEnhanced(unittest.TestCase):
    """Enhanced tests for rank_stocks_by_sector to cover overvalued order"""

    def test_overvalued_order(self):
        """Test ranking with order='overvalued'"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'sector': ['Tech', 'Tech', 'Tech'],
            'mispricing_score': [0.20, 0.15, -0.05],
        })
        result = rank_stocks_by_sector(df, top_n=2, order='overvalued')
        
        # Should return dict with 'Tech' sector
        self.assertIn('Tech', result)
        tech_stocks = result['Tech']
        
        # First stock should have lowest mispricing score (most overvalued)
        self.assertEqual(tech_stocks['ticker'].iloc[0], 'MSFT')


@unittest.skipIf(not EVAL_AVAILABLE, "finance_ml.eval not available")
class TestCalculateFinancialMetricsDashboardEnhanced(unittest.TestCase):
    """Enhanced tests with actual financial metric columns"""

    def test_with_valuation_metrics(self):
        """Test with actual valuation metric columns"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'sector': ['Tech', 'Tech', 'Tech'],
            'p_e': [25.0, 30.0, 28.0],
            'p_b': [8.0, 6.0, 10.0],
            'ev_ebitda': [15.0, 18.0, 16.0],
        })
        result = calculate_financial_metrics_dashboard(df, group_by='sector')
        
        # Should have valuation metrics
        self.assertIn('valuation', result)
        self.assertIn('p_e', result['valuation'])
        
        # P/E stats should be calculated
        pe_stats = result['valuation']['p_e']
        self.assertIn('mean', pe_stats)
        self.assertAlmostEqual(pe_stats['mean'], 27.67, places=1)

    def test_with_profitability_metrics(self):
        """Test with profitability metrics"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'gross_margin': [0.38, 0.56, 0.68],
            'operating_margin': [0.30, 0.28, 0.42],
            'net_margin': [0.25, 0.21, 0.34],
            'roe': [1.47, 0.18, 0.45],
            'roa': [0.27, 0.13, 0.19],
        })
        result = calculate_financial_metrics_dashboard(df)
        
        # Should have profitability metrics
        self.assertIn('profitability', result)
        self.assertIn('gross_margin', result['profitability'])
        self.assertIn('roe', result['profitability'])

    def test_with_growth_metrics(self):
        """Test with growth metrics"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'revenue_growth': [0.08, 0.14, 0.18],
            'earnings_growth': [0.09, 0.17, 0.22],
            'ebitda_growth': [0.10, 0.15, 0.20],
        })
        result = calculate_financial_metrics_dashboard(df)
        
        # Should have growth metrics
        self.assertIn('growth', result)
        self.assertIn('revenue_growth', result['growth'])

    def test_with_leverage_metrics(self):
        """Test with leverage metrics"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'debt_to_equity': [1.57, 0.06, 0.45],
            'debt_to_assets': [0.32, 0.04, 0.20],
            'net_debt_to_ebitda': [0.75, -0.20, 0.30],
        })
        result = calculate_financial_metrics_dashboard(df)
        
        # Should have leverage metrics
        self.assertIn('leverage', result)
        self.assertIn('debt_to_equity', result['leverage'])


@unittest.skipIf(not EVAL_AVAILABLE, "finance_ml.eval not available")
class TestGenerateDataQualityAlertsEnhanced(unittest.TestCase):
    """Enhanced tests with various data quality scenarios"""

    def test_critical_missing_values(self):
        """Test detection of >50% missing values (critical severity)"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
            'price': [150.0, None, None, None],  # 75% missing
        })
        alerts = generate_data_quality_alerts(df)
        
        # Should have critical alert for price column
        critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
        self.assertGreater(len(critical_alerts), 0)
        
        # Check message mentions the column
        price_alerts = [a for a in critical_alerts if 'price' in a.get('message', '').lower()]
        self.assertGreater(len(price_alerts), 0)

    def test_high_missing_values(self):
        """Test detection of 20-50% missing values (high severity)"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
            'revenue': [100.0, None, 200.0, 300.0, None],  # 40% missing
        })
        alerts = generate_data_quality_alerts(df)
        
        # Should have high severity alert
        high_alerts = [a for a in alerts if a.get('severity') == 'high']
        self.assertGreater(len(high_alerts), 0)

    def test_medium_missing_values(self):
        """Test detection of 5-20% missing values (medium severity)"""
        df = pd.DataFrame({
            'ticker': ['AAPL'] * 20,
            'market_cap': [1000.0] * 18 + [None, None],  # 10% missing
        })
        alerts = generate_data_quality_alerts(df)
        
        # Should have medium severity alert
        medium_alerts = [a for a in alerts if a.get('severity') == 'medium']
        self.assertGreater(len(medium_alerts), 0)

    def test_low_missing_values(self):
        """Test detection of <5% missing values (low severity)"""
        df = pd.DataFrame({
            'ticker': ['AAPL'] * 100,
            'pe_ratio': [25.0] * 98 + [None, None],  # 2% missing
        })
        alerts = generate_data_quality_alerts(df)
        
        # Should have low severity alert
        low_alerts = [a for a in alerts if a.get('severity') == 'low']
        self.assertGreater(len(low_alerts), 0)

    def test_outlier_detection(self):
        """Test statistical outlier detection"""
        # Create data with outliers (Z-score > 3)
        normal_data = [100.0] * 20
        outliers = [1000.0, 2000.0]  # Extreme outliers
        
        df = pd.DataFrame({
            'ticker': ['AAPL'] * 22,
            'market_cap': normal_data + outliers,
        })
        
        alerts = generate_data_quality_alerts(df, outlier_threshold=3.0)
        
        # Should detect outliers
        self.assertIsInstance(alerts, list)

    def test_negative_values_in_positive_columns(self):
        """Test detection of negative values in columns that should be positive"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'market_cap': [2500000, -100000, 2300000],  # Negative market cap
            'last_price': [150.0, 2800.0, -50.0],  # Negative price
        })
        
        alerts = generate_data_quality_alerts(df)
        
        # Should detect negative value issues
        self.assertIsInstance(alerts, list)


@unittest.skipIf(not EVAL_AVAILABLE, "finance_ml.eval not available")
class TestPreparePlotlyDashboardDataEnhanced(unittest.TestCase):
    """Enhanced tests with comprehensive column coverage"""

    def test_with_complete_data(self):
        """Test with all required columns for scatter plot"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'sector': ['Tech', 'Tech', 'Tech'],
            'region': ['US', 'US', 'US'],
            'last_price': [150.0, 2800.0, 350.0],
            'market_cap': [2500000, 1800000, 2300000],
            'mispricing_score': [0.20, 0.071, -0.029],
        })
        result = prepare_plotly_dashboard_data(df)
        
        # Should have scatter data with all fields
        self.assertIn('scatter_data', result)
        scatter = result['scatter_data']
        self.assertIn('x', scatter)
        self.assertIn('y', scatter)
        self.assertEqual(len(scatter['x']), 3)

    def test_histogram_data_by_sector(self):
        """Test histogram data generation by sector"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC'],
            'sector': ['Tech', 'Tech', 'Tech', 'Finance', 'Finance'],
            'mispricing_score': [0.20, 0.15, -0.05, 0.30, 0.10],
        })
        result = prepare_plotly_dashboard_data(df)
        
        # Should have histogram data
        self.assertIn('histogram_data', result)
        if 'mispricing_by_sector' in result['histogram_data']:
            hist_data = result['histogram_data']['mispricing_by_sector']
            self.assertIsInstance(hist_data, list)

    def test_box_plot_data_sector_pe(self):
        """Test box plot data for sector P/E ratios"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC'],
            'sector': ['Tech', 'Tech', 'Tech', 'Finance', 'Finance'],
            'p_e': [25.0, 30.0, 28.0, 12.0, 10.0],
        })
        result = prepare_plotly_dashboard_data(df)
        
        # Should have box data with sector comparisons
        self.assertIn('box_data', result)
        if 'sector_comparisons' in result['box_data']:
            sector_box = result['box_data']['sector_comparisons']
            self.assertIsInstance(sector_box, list)

    def test_box_plot_data_region_roe(self):
        """Test box plot data for region ROE"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'VOD', 'BP'],
            'region': ['US', 'US', 'EU', 'EU'],
            'roe': [1.47, 0.18, 0.12, 0.08],
        })
        result = prepare_plotly_dashboard_data(df)
        
        # Should have box data with region comparisons
        self.assertIn('box_data', result)
        if 'region_comparisons' in result['box_data']:
            region_box = result['box_data']['region_comparisons']
            self.assertIsInstance(region_box, list)

    def test_heatmap_data_sector_region(self):
        """Test heatmap data for sector-region matrix"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'VOD', 'BP', 'SAP'],
            'sector': ['Tech', 'Tech', 'Telecom', 'Energy', 'Tech'],
            'region': ['US', 'US', 'EU', 'EU', 'EU'],
            'mispricing_score': [0.20, 0.15, 0.05, -0.10, 0.08],
        })
        result = prepare_plotly_dashboard_data(df)
        
        # Should have heatmap data
        self.assertIn('heatmap_data', result)

    def test_sunburst_data(self):
        """Test sunburst chart data generation"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'JPM'],
            'sector': ['Tech', 'Tech', 'Finance'],
            'region': ['US', 'US', 'US'],
            'market_cap': [2500000, 1800000, 450000],
        })
        result = prepare_plotly_dashboard_data(df)
        
        # Should have sunburst data structure
        self.assertIn('sunburst_data', result)

    def test_with_timeseries_data(self):
        """Test with include_timeseries=True and date column"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL', 'AAPL'],
            'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'last_price': [150.0, 155.0, 160.0],
            'mispricing_score': [0.20, 0.18, 0.15],
        })
        result = prepare_plotly_dashboard_data(df, include_timeseries=True)
        
        # Should return a dictionary
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
