"""Smoke test for the enhanced detect_accounting_anomalies function."""
import unittest
import numpy as np
import pandas as pd
from finance_ml.analytics.statistical_analysis import detect_accounting_anomalies


class TestDetectAccountingAnomaliesSmoke(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 200
        self.df = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(n)],
            "industry": np.random.choice(["Tech", "Finance", "Health"], n),
            "exceptional_items_frequency": np.random.normal(0.1, 0.05, n),
            "non_operating_income_share": np.random.normal(0.05, 0.03, n),
            "gaap_adj_eps_gap_pct": np.random.normal(5, 10, n),
            "eps_adjustment_ratio": np.random.normal(1.0, 0.3, n),
            "ebitda_adjustment_ratio": np.random.normal(1.0, 0.2, n),
        })
        # Inject anomalies
        self.df.loc[0, "gaap_adj_eps_gap_pct"] = 80
        self.df.loc[1, "eps_adjustment_ratio"] = 5.0

    def test_returns_expected_columns(self):
        result = detect_accounting_anomalies(self.df)
        self.assertIn("accounting_anomaly_score", result.columns)
        self.assertIn("accounting_anomaly_tier", result.columns)
        self.assertIn("anomaly_feature_count", result.columns)
        self.assertIn("mahalanobis_distance", result.columns)
        self.assertIn("sector_relative_anomaly", result.columns)
        self.assertIn("benford_chi2_pvalue", result.columns)

    def test_score_range(self):
        result = detect_accounting_anomalies(self.df)
        self.assertGreaterEqual(result["accounting_anomaly_score"].min(), 0)
        self.assertLessEqual(result["accounting_anomaly_score"].max(), 100.1)

    def test_tier_labels(self):
        result = detect_accounting_anomalies(self.df)
        valid = {"Clean", "Watch", "Flag", "Alert"}
        actual = set(result["accounting_anomaly_tier"].dropna().unique())
        self.assertTrue(actual.issubset(valid))

    def test_robust_z_columns(self):
        result = detect_accounting_anomalies(self.df)
        self.assertIn("gaap_adj_eps_gap_pct_z_robust", result.columns)
        self.assertIn("gaap_adj_eps_gap_pct_anomaly_flag", result.columns)
        self.assertIn("gaap_adj_eps_gap_pct_dist_name", result.columns)

    def test_empty_features_returns_unchanged(self):
        df_no_feat = pd.DataFrame({"ticker": ["A", "B"], "price": [10, 20]})
        result = detect_accounting_anomalies(df_no_feat)
        self.assertEqual(list(result.columns), list(df_no_feat.columns))

    def test_injected_anomaly_scores_high(self):
        result = detect_accounting_anomalies(self.df)
        # Row 0 and 1 have injected anomalies — they should score above median
        median_score = result["accounting_anomaly_score"].median()
        self.assertGreater(result.loc[0, "accounting_anomaly_score"], median_score)
        self.assertGreater(result.loc[1, "accounting_anomaly_score"], median_score)


if __name__ == "__main__":
    unittest.main()
