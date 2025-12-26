"""
Test suite for finance_ml.benchmarking module

This module tests sector and region-specific benchmarking functions.
Following TDD methodology for Phase 9.2 continuation.
"""

import unittest

import numpy as np
import pandas as pd


class TestSectorDistributionComparison(unittest.TestCase):
    """Test sector-wise distribution comparisons for valuation metrics"""

    def setUp(self):
        """Create sample multi-sector data"""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK_{i}" for i in range(60)],
                "sector": ["Technology"] * 20 + ["Finance"] * 20 + ["Energy"] * 20,
                "region": ["US"] * 30 + ["EU"] * 30,
                "p_e": np.concatenate(
                    [
                        np.random.normal(25, 5, 20),  # Tech: high P/E
                        np.random.normal(12, 3, 20),  # Finance: low P/E
                        np.random.normal(15, 4, 20),  # Energy: medium P/E
                    ]
                ),
                "p_b": np.concatenate(
                    [
                        np.random.normal(5, 1.5, 20),  # Tech
                        np.random.normal(1.2, 0.3, 20),  # Finance
                        np.random.normal(1.8, 0.5, 20),  # Energy
                    ]
                ),
                "ev_ebitda": np.concatenate(
                    [
                        np.random.normal(18, 4, 20),  # Tech
                        np.random.normal(10, 2, 20),  # Finance
                        np.random.normal(8, 2, 20),  # Energy
                    ]
                ),
                "operating_margin": np.concatenate(
                    [
                        np.random.normal(0.25, 0.05, 20),  # Tech: 25%
                        np.random.normal(0.35, 0.08, 20),  # Finance: 35%
                        np.random.normal(0.15, 0.04, 20),  # Energy: 15%
                    ]
                ),
            }
        )

    def test_compare_sector_distributions_returns_dataframe(self):
        """Should return DataFrame with sector distribution statistics"""
        from finance_ml.benchmarking import compare_sector_distributions

        result = compare_sector_distributions(self.df, metrics=["p_e", "p_b"])
        self.assertIsInstance(result, pd.DataFrame)

    def test_compare_sector_distributions_has_required_columns(self):
        """Should have sector, metric, mean, median, std, min, max columns"""
        from finance_ml.benchmarking import compare_sector_distributions

        result = compare_sector_distributions(self.df, metrics=["p_e"])
        self.assertIn("sector", result.columns)
        self.assertIn("metric", result.columns)
        self.assertIn("mean", result.columns)
        self.assertIn("median", result.columns)
        self.assertIn("std", result.columns)

    def test_compare_sector_distributions_all_sectors_included(self):
        """Should include all sectors in results"""
        from finance_ml.benchmarking import compare_sector_distributions

        result = compare_sector_distributions(self.df, metrics=["p_e"])
        sectors = result["sector"].unique()
        self.assertEqual(len(sectors), 3)
        self.assertIn("Technology", sectors)
        self.assertIn("Finance", sectors)
        self.assertIn("Energy", sectors)

    def test_compare_sector_distributions_multiple_metrics(self):
        """Should handle multiple metrics"""
        from finance_ml.benchmarking import compare_sector_distributions

        result = compare_sector_distributions(self.df, metrics=["p_e", "p_b", "ev_ebitda"])
        metrics = result["metric"].unique()
        self.assertEqual(len(metrics), 3)

    def test_compare_sector_distributions_correct_calculations(self):
        """Should calculate statistics correctly"""
        from finance_ml.benchmarking import compare_sector_distributions

        result = compare_sector_distributions(self.df, metrics=["p_e"])
        tech_row = result[(result["sector"] == "Technology") & (result["metric"] == "p_e")]

        # Tech P/E should be around 25 (from setUp)
        tech_mean = tech_row["mean"].iloc[0]
        self.assertGreater(tech_mean, 20)
        self.assertLess(tech_mean, 30)


class TestRegionalValuationComparison(unittest.TestCase):
    """Test regional valuation metric comparisons with statistical tests"""

    def setUp(self):
        """Create sample multi-region data"""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK_{i}" for i in range(60)],
                "region": ["US"] * 20 + ["EU"] * 20 + ["APAC"] * 20,
                "sector": ["Technology"] * 30 + ["Finance"] * 30,
                "p_e": np.concatenate(
                    [
                        np.random.normal(22, 5, 20),  # US
                        np.random.normal(15, 4, 20),  # EU
                        np.random.normal(18, 4, 20),  # APAC
                    ]
                ),
                "ev_ebitda": np.concatenate(
                    [
                        np.random.normal(14, 3, 20),  # US
                        np.random.normal(10, 2, 20),  # EU
                        np.random.normal(12, 3, 20),  # APAC
                    ]
                ),
            }
        )

    def test_compare_regional_valuations_returns_dataframe(self):
        """Should return DataFrame with regional comparison results"""
        from finance_ml.benchmarking import compare_regional_valuations

        result = compare_regional_valuations(self.df, metrics=["p_e", "ev_ebitda"])
        self.assertIsInstance(result, pd.DataFrame)

    def test_compare_regional_valuations_has_statistics(self):
        """Should include statistical test results"""
        from finance_ml.benchmarking import compare_regional_valuations

        result = compare_regional_valuations(self.df, metrics=["p_e"])
        self.assertIn("region", result.columns)
        self.assertIn("metric", result.columns)
        self.assertIn("mean", result.columns)

    def test_compare_regional_valuations_with_statistical_tests(self):
        """Should include statistical significance tests"""
        from finance_ml.benchmarking import compare_regional_valuations

        result = compare_regional_valuations(self.df, metrics=["p_e"], include_tests=True)
        # Should have test results (p-value, statistic, etc.)
        self.assertIsInstance(result, dict)
        self.assertIn("distributions", result)
        self.assertIn("statistical_tests", result)

    def test_compare_regional_valuations_all_regions(self):
        """Should include all regions"""
        from finance_ml.benchmarking import compare_regional_valuations

        result = compare_regional_valuations(self.df, metrics=["p_e"])
        regions = result["region"].unique()
        self.assertEqual(len(regions), 3)


class TestPeerGroupAnalysis(unittest.TestCase):
    """Test peer group analysis within sectors"""

    def setUp(self):
        """Create sample data for peer group analysis"""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "BAC", "WFC", "C", "GS"],
                "sector": ["Technology"] * 5 + ["Finance"] * 5,
                "market_cap": [
                    2.5e12,
                    2.3e12,
                    1.8e12,
                    1.5e12,
                    0.8e12,
                    450e9,
                    300e9,
                    200e9,
                    150e9,
                    120e9,
                ],
                "p_e": [28, 32, 25, 60, 22, 10, 9, 8, 11, 12],
                "p_b": [40, 12, 7, 15, 6, 1.5, 1.1, 0.9, 0.8, 1.2],
                "last_price": [175, 370, 140, 150, 320, 150, 35, 45, 55, 380],
            }
        )

    def test_find_peer_group_returns_dataframe(self):
        """Should return DataFrame with peer stocks"""
        from finance_ml.benchmarking import find_peer_group

        result = find_peer_group(self.df, ticker="AAPL", n_peers=3)
        self.assertIsInstance(result, pd.DataFrame)

    def test_find_peer_group_same_sector(self):
        """Should return peers from the same sector"""
        from finance_ml.benchmarking import find_peer_group

        result = find_peer_group(self.df, ticker="AAPL", n_peers=3)
        # All peers should be Technology
        self.assertTrue(all(result["sector"] == "Technology"))

    def test_find_peer_group_excludes_target(self):
        """Should exclude the target stock from peers"""
        from finance_ml.benchmarking import find_peer_group

        result = find_peer_group(self.df, ticker="AAPL", n_peers=3)
        self.assertNotIn("AAPL", result["ticker"].values)

    def test_find_peer_group_respects_n_peers(self):
        """Should return requested number of peers"""
        from finance_ml.benchmarking import find_peer_group

        result = find_peer_group(self.df, ticker="AAPL", n_peers=2)
        self.assertEqual(len(result), 2)

    def test_find_peer_group_by_market_cap(self):
        """Should find peers with similar market cap"""
        from finance_ml.benchmarking import find_peer_group

        result = find_peer_group(self.df, ticker="AAPL", n_peers=2, criteria="market_cap")
        # Should return MSFT and GOOGL (closest in market cap)
        self.assertIn("MSFT", result["ticker"].values)
        self.assertIn("GOOGL", result["ticker"].values)

    def test_compare_to_peers_returns_dict(self):
        """Should return comparison dictionary"""
        from finance_ml.benchmarking import compare_to_peers

        result = compare_to_peers(self.df, ticker="AAPL", metrics=["p_e", "p_b"], n_peers=3)
        self.assertIsInstance(result, dict)

    def test_compare_to_peers_has_target_and_peers(self):
        """Should include target stock and peer statistics"""
        from finance_ml.benchmarking import compare_to_peers

        result = compare_to_peers(self.df, ticker="AAPL", metrics=["p_e"], n_peers=3)
        # Result is nested: {metric: {target: ..., peers_mean: ...}}
        self.assertIn("p_e", result)
        self.assertIn("target", result["p_e"])
        self.assertIn("peers_mean", result["p_e"])
        self.assertIn("peers_median", result["p_e"])

    def test_compare_to_peers_calculates_deviation(self):
        """Should calculate deviation from peer average"""
        from finance_ml.benchmarking import compare_to_peers

        result = compare_to_peers(self.df, ticker="AAPL", metrics=["p_e"], n_peers=3)
        # Result is nested: {metric: {deviation_from_mean: ...}}
        self.assertIn("p_e", result)
        self.assertIn("deviation_from_mean", result["p_e"])


class TestTimeSeriesTrendAnalysis(unittest.TestCase):
    """Test time-series trend analysis for key metrics"""

    def setUp(self):
        """Create sample time-series data"""
        dates = pd.date_range("2023-01-01", periods=12, freq="M")
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 12,
                "date": dates,
                "p_e": [25, 26, 27, 26, 28, 29, 30, 29, 31, 32, 31, 33],
                "price": [150, 155, 160, 158, 165, 170, 175, 172, 180, 185, 182, 190],
            }
        )

    def test_analyze_metric_trend_returns_dict(self):
        """Should return dictionary with trend analysis"""
        from finance_ml.benchmarking import analyze_metric_trend

        result = analyze_metric_trend(self.df, ticker="AAPL", metric="p_e", date_column="date")
        self.assertIsInstance(result, dict)

    def test_analyze_metric_trend_detects_direction(self):
        """Should detect trend direction (increasing/decreasing/stable)"""
        from finance_ml.benchmarking import analyze_metric_trend

        result = analyze_metric_trend(self.df, ticker="AAPL", metric="p_e", date_column="date")
        self.assertIn("trend_direction", result)
        self.assertIn(result["trend_direction"], ["increasing", "decreasing", "stable"])

    def test_analyze_metric_trend_calculates_slope(self):
        """Should calculate trend slope"""
        from finance_ml.benchmarking import analyze_metric_trend

        result = analyze_metric_trend(self.df, ticker="AAPL", metric="p_e", date_column="date")
        self.assertIn("slope", result)
        # P/E is increasing, slope should be positive
        self.assertGreater(result["slope"], 0)

    def test_analyze_metric_trend_handles_missing_dates(self):
        """Should handle case when date column is missing"""
        from finance_ml.benchmarking import analyze_metric_trend

        df_no_date = self.df.drop("date", axis=1)
        result = analyze_metric_trend(df_no_date, ticker="AAPL", metric="p_e", date_column="date")
        # Should return None when date column is missing
        self.assertIsNone(result)


class TestBenchmarkingIntegration(unittest.TestCase):
    """Test integration of benchmarking functions"""

    def setUp(self):
        """Create comprehensive sample data"""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK_{i}" for i in range(30)],
                "sector": ["Technology"] * 10 + ["Finance"] * 10 + ["Energy"] * 10,
                "region": ["US"] * 15 + ["EU"] * 15,
                "p_e": np.random.normal(20, 5, 30),
                "p_b": np.random.normal(3, 1, 30),
                "market_cap": np.random.lognormal(20, 2, 30),
            }
        )

    def test_generate_benchmarking_report_returns_dict(self):
        """Should return comprehensive benchmarking report"""
        from finance_ml.benchmarking import generate_benchmarking_report

        result = generate_benchmarking_report(self.df, metrics=["p_e", "p_b"])
        self.assertIsInstance(result, dict)

    def test_generate_benchmarking_report_includes_sections(self):
        """Should include sector and regional comparisons"""
        from finance_ml.benchmarking import generate_benchmarking_report

        result = generate_benchmarking_report(self.df, metrics=["p_e"])
        self.assertIn("sector_distributions", result)
        self.assertIn("regional_valuations", result)


if __name__ == "__main__":
    unittest.main()
