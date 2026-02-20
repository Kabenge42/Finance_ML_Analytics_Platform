import unittest
import pandas as pd
import numpy as np
from finance_ml.analytics.data_utils import compute_metric_statistics

class TestAnalyticsDemo(unittest.TestCase):
    def test_compute_metric_statistics(self):
        """Test the compute_metric_statistics function with sample data."""
        data = pd.Series([10, 20, 30, 40, 50])
        stats = compute_metric_statistics(data)
        
        self.assertEqual(stats['mean'], 30.0)
        self.assertEqual(stats['min'], 10.0)
        self.assertEqual(stats['max'], 50.0)
        self.assertEqual(stats['count'], 5)
        self.assertIn('std', stats)
        self.assertIn('median', stats)

    def test_compute_metric_statistics_with_nans(self):
        """Test handling of NaNs in compute_metric_statistics."""
        data = pd.Series([10, 20, np.nan, 40, 50])
        stats = compute_metric_statistics(data)
        
        # Should exclude NaN from mean: (10+20+40+50)/4 = 120/4 = 30
        self.assertEqual(stats['mean'], 30.0)
        self.assertEqual(stats['count'], 4)

if __name__ == '__main__':
    unittest.main()
