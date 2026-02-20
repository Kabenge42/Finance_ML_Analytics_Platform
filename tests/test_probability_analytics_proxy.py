"""
Test for the trajectory proxy path in BayesianEarningsBeatModel.analyze_dataframe_enhanced.
Verifies that the fix for numpy.ndarray .iloc AttributeError works correctly.
"""

import unittest

import numpy as np
import pandas as pd


class TestAnalyzeDataframeEnhancedProxy(unittest.TestCase):
    """Test proxy-based posterior calculation in analyze_dataframe_enhanced."""

    def test_trajectory_proxy_path_no_attribute_error(self):
        """
        Rows with eps_trajectory_score but missing beat columns should go
        through the proxy path without raising AttributeError on numpy arrays.
        """
        from finance_ml.analytics.probability_analytics import EarningsBeatProbabilityModel

        model = EarningsBeatProbabilityModel()

        df = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "name": ["Alpha Co", "Beta Co"],
            "sector": ["Tech", "Finance"],
            "eps_trajectory_score": [75.0, 40.0],
        })

        result = model.analyze_dataframe_enhanced(
            df,
            ticker_col="ticker",
            sector_col="sector",
            name_col="name",
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

        expected_cols = [
            "ci_90_lower", "ci_90_upper",
            "ci_95_lower", "ci_95_upper",
            "confidence_score",
            "posterior_beat_prob",
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns)
            for val in result[col]:
                self.assertFalse(np.isnan(val), f"NaN found in {col}")


if __name__ == "__main__":
    unittest.main()
