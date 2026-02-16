import unittest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finance_ml.dashboards.geib_dash_app import load_geib_data

class TestGEIBDashApp(unittest.TestCase):
    def setUp(self):
        # Clear environment variables before each test
        self.env_patcher = patch.dict(os.environ, {}, clear=True)
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_load_geib_data_no_env(self):
        """Test that it returns empty data dict when environment variables are missing."""
        data = load_geib_data()
        self.assertTrue(data['summary'].empty)

    @patch('finance_ml.dashboards.geib_dash_app.create_engine')
    @patch('pandas.read_sql')
    def test_load_geib_data_with_env(self, mock_read_sql, mock_create_engine):
        """Test data loading with correct environment variables."""
        os.environ["GEIB_DASHBOARD"] = "true"
        os.environ["DB_URL"] = "postgresql://user:pass@localhost:5432/db"
        
        # Mock DataFrame
        mock_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'expected_upside_pct': [15.5, 10.2],
            'expected_return_prob_weighted': [12.0, 8.5],
            'region': ['North America', 'North America'],
            'sector': ['Technology', 'Technology'],
            'signal': ['strong_buy', 'buy'],
            'confidence_level': ['high', 'medium'],
            'agreement_score': [0.9, 0.7]
        })
        mock_read_sql.return_value = mock_df
        
        data = load_geib_data()
        df = data['summary']
        
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)
        mock_create_engine.assert_called_once_with(os.environ["DB_URL"])
        self.assertIn('ticker', df.columns)

    def test_dashboard_layout_exists(self):
        """Verify that the dash app layout is initialized (will fail if geib_dash_app.py is not updated)."""
        from finance_ml.dashboards.geib_dash_app import app
        self.assertIsNotNone(app.layout)
        # Check for some specific GEIB components in the layout
        layout_str = str(app.layout)
        self.assertIn("Global Equity Investment Board", layout_str)

    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation function."""
        from finance_ml.dashboards.geib_dash_app import run_monte_carlo_simulation
        import numpy as np
        
        # Create test data
        test_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'prob_positive_upside': [70.0, 65.0, 60.0],
            'filtered_upside': [15.0, 12.0, 18.0],
            'achievement_probability': [0.8, 0.75, 0.7],
            'signal': ['strong_buy', 'buy', 'hold']
        })
        
        # Run simulation
        returns, stats = run_monte_carlo_simulation(test_df, 1000, 0.5, 'equal', 10.0)
        
        # Verify results
        self.assertEqual(len(returns), 1000)
        self.assertIn('var_5', stats)
        self.assertIn('median', stats)
        self.assertIn('prob_positive', stats)
        self.assertIn('prob_target', stats)
        self.assertEqual(stats['num_stocks'], 3)
        self.assertEqual(stats['num_simulations'], 1000)

    def test_monte_carlo_empty_data(self):
        """Test Monte Carlo simulation with empty data."""
        from finance_ml.dashboards.geib_dash_app import run_monte_carlo_simulation
        
        empty_df = pd.DataFrame()
        returns, stats = run_monte_carlo_simulation(empty_df, 1000, 0.5, 'equal', 10.0)
        
        self.assertEqual(len(returns), 0)
        self.assertEqual(stats, {})

    def test_monte_carlo_kelly_weighting(self):
        """Test Monte Carlo simulation with Kelly weighting."""
        from finance_ml.dashboards.geib_dash_app import run_monte_carlo_simulation
        
        test_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'prob_positive_upside': [70.0, 60.0],
            'filtered_upside': [20.0, 15.0],
            'achievement_probability': [0.8, 0.7],
        })
        
        returns, stats = run_monte_carlo_simulation(test_df, 500, 0.5, 'kelly', 5.0)
        
        self.assertEqual(len(returns), 500)
        self.assertEqual(stats['num_stocks'], 2)

if __name__ == "__main__":
    unittest.main()
