"""
Test suite for Phase 9.7: Identification of Under/Overvalued Stocks with Visualization

This module tests the enhanced valuation analysis functions including:
- Valuation category assignment (Strong Buy, Buy, Hold, Sell, Strong Sell)
- Sector-relative metrics (z-scores, percentile ranks)
- Advanced filtering and multi-factor scoring
- Enhanced visualizations with valuation categories

Following strict TDD approach:
1. Write failing tests first (Red)
2. Implement minimal code to pass (Green)
3. Refactor for clarity (Refactor)
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestAssignValuationCategory(unittest.TestCase):
    """Test valuation category assignment based on mispricing scores"""

    def setUp(self):
        """Create sample data with mispricing scores"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "mispricing_score": [25.0, 15.0, 5.0, -5.0, -15.0, -25.0],
                "sector": ["Tech", "Tech", "Finance", "Finance", "Energy", "Energy"],
            }
        )

    def test_assign_valuation_category_returns_series(self):
        """Should return a pandas Series"""
        from finance_ml.eval import assign_valuation_category

        result = assign_valuation_category(self.df["mispricing_score"])
        self.assertIsInstance(result, pd.Series)

    def test_assign_valuation_category_strong_buy(self):
        """Should assign 'Strong Buy' for mispricing > 20%"""
        from finance_ml.eval import assign_valuation_category

        result = assign_valuation_category(self.df["mispricing_score"])
        self.assertEqual(result.iloc[0], "Strong Buy")  # 25%

    def test_assign_valuation_category_buy(self):
        """Should assign 'Buy' for mispricing 10% to 20%"""
        from finance_ml.eval import assign_valuation_category

        result = assign_valuation_category(self.df["mispricing_score"])
        self.assertEqual(result.iloc[1], "Buy")  # 15%

    def test_assign_valuation_category_hold(self):
        """Should assign 'Hold' for mispricing -10% to +10%"""
        from finance_ml.eval import assign_valuation_category

        result = assign_valuation_category(self.df["mispricing_score"])
        self.assertEqual(result.iloc[2], "Hold")  # 5%
        self.assertEqual(result.iloc[3], "Hold")  # -5%

    def test_assign_valuation_category_sell(self):
        """Should assign 'Sell' for mispricing -20% to -10%"""
        from finance_ml.eval import assign_valuation_category

        result = assign_valuation_category(self.df["mispricing_score"])
        self.assertEqual(result.iloc[4], "Sell")  # -15%

    def test_assign_valuation_category_strong_sell(self):
        """Should assign 'Strong Sell' for mispricing < -20%"""
        from finance_ml.eval import assign_valuation_category

        result = assign_valuation_category(self.df["mispricing_score"])
        self.assertEqual(result.iloc[5], "Strong Sell")  # -25%

    def test_assign_valuation_category_custom_thresholds(self):
        """Should support custom thresholds"""
        from finance_ml.eval import assign_valuation_category

        result = assign_valuation_category(
            self.df["mispricing_score"],
            thresholds={"strong_buy": 30, "buy": 15, "sell": 15, "strong_sell": 30},
        )
        # With higher thresholds, 25% should be 'Buy' not 'Strong Buy'
        self.assertEqual(result.iloc[0], "Buy")


class TestCalculateSectorZScores(unittest.TestCase):
    """Test sector-relative z-score calculations"""

    def setUp(self):
        """Create sample data with sectors"""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(20)],
                "sector": ["Tech"] * 10 + ["Finance"] * 10,
                "pe_ratio": np.random.uniform(5, 50, 20),
                "pb_ratio": np.random.uniform(0.5, 5, 20),
            }
        )

    def test_calculate_sector_zscores_returns_dataframe(self):
        """Should return a DataFrame with z-score columns"""
        from finance_ml.eval import calculate_sector_zscores

        result = calculate_sector_zscores(self.df, metrics=["pe_ratio", "pb_ratio"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("pe_ratio_zscore", result.columns)
        self.assertIn("pb_ratio_zscore", result.columns)

    def test_calculate_sector_zscores_mean_zero(self):
        """Z-scores within each sector should have mean ≈ 0"""
        from finance_ml.eval import calculate_sector_zscores

        result = calculate_sector_zscores(self.df, metrics=["pe_ratio"])

        # Check Tech sector z-scores have mean ≈ 0
        tech_zscores = result[result["sector"] == "Tech"]["pe_ratio_zscore"]
        self.assertAlmostEqual(tech_zscores.mean(), 0.0, places=10)

        # Check Finance sector z-scores have mean ≈ 0
        fin_zscores = result[result["sector"] == "Finance"]["pe_ratio_zscore"]
        self.assertAlmostEqual(fin_zscores.mean(), 0.0, places=10)

    def test_calculate_sector_zscores_std_one(self):
        """Z-scores within each sector should have std ≈ 1"""
        from finance_ml.eval import calculate_sector_zscores

        result = calculate_sector_zscores(self.df, metrics=["pe_ratio"])

        tech_zscores = result[result["sector"] == "Tech"]["pe_ratio_zscore"]
        self.assertAlmostEqual(tech_zscores.std(), 1.0, places=10)

    def test_calculate_sector_zscores_handles_missing(self):
        """Should handle missing values gracefully"""
        from finance_ml.eval import calculate_sector_zscores

        df_with_nan = self.df.copy()
        df_with_nan.loc[0, "pe_ratio"] = np.nan

        result = calculate_sector_zscores(df_with_nan, metrics=["pe_ratio"])
        self.assertTrue(pd.isna(result.loc[0, "pe_ratio_zscore"]))


class TestCalculatePercentileRanks(unittest.TestCase):
    """Test percentile rank calculations within sectors"""

    def setUp(self):
        """Create sample data"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "sector": ["Tech", "Tech", "Finance", "Finance"],
                "pe_ratio": [10, 20, 15, 25],
            }
        )

    def test_calculate_percentile_ranks_returns_dataframe(self):
        """Should return DataFrame with percentile columns"""
        from finance_ml.eval import calculate_percentile_ranks

        result = calculate_percentile_ranks(self.df, metrics=["pe_ratio"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("pe_ratio_percentile", result.columns)

    def test_calculate_percentile_ranks_range(self):
        """Percentiles should be between 0 and 100"""
        from finance_ml.eval import calculate_percentile_ranks

        result = calculate_percentile_ranks(self.df, metrics=["pe_ratio"])
        percentiles = result["pe_ratio_percentile"]
        self.assertTrue((percentiles >= 0).all())
        self.assertTrue((percentiles <= 100).all())

    def test_calculate_percentile_ranks_within_sector(self):
        """Should calculate percentiles within each sector"""
        from finance_ml.eval import calculate_percentile_ranks

        result = calculate_percentile_ranks(self.df, metrics=["pe_ratio"])

        # In Tech: 10 (A) should be lower percentile than 20 (B)
        tech_result = result[result["sector"] == "Tech"]
        self.assertLess(
            tech_result[tech_result["ticker"] == "A"]["pe_ratio_percentile"].iloc[0],
            tech_result[tech_result["ticker"] == "B"]["pe_ratio_percentile"].iloc[0],
        )


class TestMultiFactorScore(unittest.TestCase):
    """Test multi-factor scoring combining valuation, quality, growth"""

    def setUp(self):
        """Create sample data with multiple factors"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "mispricing_score": [20, 10, -10, -20],
                "roe": [0.15, 0.10, 0.08, 0.05],
                "revenue_growth": [0.20, 0.15, 0.10, 0.05],
                "ebitda_margin": [0.25, 0.20, 0.15, 0.10],
            }
        )

    def test_multi_factor_score_returns_series(self):
        """Should return a pandas Series"""
        from finance_ml.eval import calculate_multi_factor_score

        result = calculate_multi_factor_score(
            self.df,
            valuation_col="mispricing_score",
            quality_cols=["roe", "ebitda_margin"],
            growth_cols=["revenue_growth"],
        )
        self.assertIsInstance(result, pd.Series)

    def test_multi_factor_score_higher_for_better_stocks(self):
        """Should assign higher scores to stocks with better metrics"""
        from finance_ml.eval import calculate_multi_factor_score

        result = calculate_multi_factor_score(
            self.df,
            valuation_col="mispricing_score",
            quality_cols=["roe", "ebitda_margin"],
            growth_cols=["revenue_growth"],
        )

        # Stock A has best metrics, should have highest score
        self.assertEqual(result.idxmax(), 0)
        # Stock D has worst metrics, should have lowest score
        self.assertEqual(result.idxmin(), 3)

    def test_multi_factor_score_custom_weights(self):
        """Should support custom weights for factors"""
        from finance_ml.eval import calculate_multi_factor_score

        result = calculate_multi_factor_score(
            self.df,
            valuation_col="mispricing_score",
            quality_cols=["roe"],
            growth_cols=["revenue_growth"],
            weights={"valuation": 0.5, "quality": 0.3, "growth": 0.2},
        )
        self.assertIsInstance(result, pd.Series)


class TestFilterByCriteria(unittest.TestCase):
    """Test advanced filtering functionality"""

    def setUp(self):
        """Create sample data for filtering"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Tech", "Finance", "Finance", "Energy", "Energy"],
                "region": ["US", "US", "EU", "EU", "APAC", "APAC"],
                "market_cap": [100e9, 50e9, 25e9, 10e9, 5e9, 1e9],
                "mispricing_score": [25, 15, 10, 5, -5, -15],
            }
        )

    def test_filter_by_sector(self):
        """Should filter by sector"""
        from finance_ml.eval import filter_stocks_by_criteria

        result = filter_stocks_by_criteria(self.df, sectors=["Tech"])
        self.assertEqual(len(result), 2)
        self.assertTrue((result["sector"] == "Tech").all())

    def test_filter_by_region(self):
        """Should filter by region"""
        from finance_ml.eval import filter_stocks_by_criteria

        result = filter_stocks_by_criteria(self.df, regions=["US", "EU"])
        self.assertEqual(len(result), 4)
        self.assertTrue(result["region"].isin(["US", "EU"]).all())

    def test_filter_by_market_cap(self):
        """Should filter by market cap range"""
        from finance_ml.eval import filter_stocks_by_criteria

        result = filter_stocks_by_criteria(self.df, min_market_cap=10e9, max_market_cap=60e9)
        self.assertEqual(len(result), 3)
        self.assertTrue((result["market_cap"] >= 10e9).all())
        self.assertTrue((result["market_cap"] <= 60e9).all())

    def test_filter_by_mispricing_threshold(self):
        """Should filter by minimum mispricing score"""
        from finance_ml.eval import filter_stocks_by_criteria

        result = filter_stocks_by_criteria(self.df, min_mispricing=10.0)
        self.assertEqual(len(result), 3)
        self.assertTrue((result["mispricing_score"] >= 10.0).all())

    def test_filter_combined_criteria(self):
        """Should apply multiple filters simultaneously"""
        from finance_ml.eval import filter_stocks_by_criteria

        result = filter_stocks_by_criteria(
            self.df, sectors=["Tech", "Finance"], regions=["US", "EU"], min_market_cap=20e9
        )
        self.assertEqual(len(result), 3)


class TestCreateValuationScatterPlot(unittest.TestCase):
    """Test enhanced valuation scatter plot with categories"""

    def setUp(self):
        """Create sample data"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "sector": ["Tech", "Tech", "Finance", "Finance"],
                "last_price": [100, 50, 75, 120],
                "predicted_target": [125, 55, 70, 100],
                "mispricing_score": [25, 10, -6.67, -16.67],
                "valuation_category": ["Strong Buy", "Buy", "Hold", "Sell"],
            }
        )
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temp directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_create_valuation_scatter_plot_creates_file(self):
        """Should create visualization file"""
        from finance_ml.eval import create_valuation_scatter_plot

        out_path = self.temp_dir / "scatter.html"
        create_valuation_scatter_plot(self.df, out_path=out_path)
        self.assertTrue(out_path.exists())

    def test_create_valuation_scatter_plot_returns_figure(self):
        """Should return plotly figure object"""
        from finance_ml.eval import create_valuation_scatter_plot

        fig = create_valuation_scatter_plot(self.df)
        self.assertIsNotNone(fig)

    def test_create_valuation_scatter_plot_with_categories(self):
        """Should color by valuation category"""
        from finance_ml.eval import create_valuation_scatter_plot

        fig = create_valuation_scatter_plot(self.df, color_by="valuation_category")
        self.assertIsNotNone(fig)


class TestIntegrationValuationWorkflow(unittest.TestCase):
    """Integration test for complete Phase 9.7 workflow"""

    def setUp(self):
        """Create realistic sample data"""
        np.random.seed(42)
        n_samples = 100

        sectors = np.random.choice(["Tech", "Finance", "Energy", "Healthcare"], n_samples)
        regions = np.random.choice(["US", "EU", "APAC", "ROTW"], n_samples)

        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": sectors,
                "region": regions,
                "last_price": np.random.uniform(10, 200, n_samples),
                "predicted_target": np.random.uniform(10, 250, n_samples),
                "pe_ratio": np.random.uniform(5, 50, n_samples),
                "pb_ratio": np.random.uniform(0.5, 5, n_samples),
                "roe": np.random.uniform(0.05, 0.30, n_samples),
                "revenue_growth": np.random.uniform(-0.05, 0.30, n_samples),
                "market_cap": np.random.uniform(1e9, 100e9, n_samples),
            }
        )

    def test_complete_valuation_workflow(self):
        """Test complete Phase 9.7 workflow"""
        from finance_ml.eval import (
            calculate_mispricing_score,
            assign_valuation_category,
            calculate_sector_zscores,
            calculate_percentile_ranks,
            calculate_multi_factor_score,
            filter_stocks_by_criteria,
        )

        # Step 1: Calculate mispricing scores
        mispricing = calculate_mispricing_score(self.df)
        self.df["mispricing_score"] = mispricing

        # Step 2: Assign valuation categories
        categories = assign_valuation_category(mispricing)
        self.df["valuation_category"] = categories

        # Step 3: Calculate sector-relative metrics
        df_with_zscores = calculate_sector_zscores(self.df, metrics=["pe_ratio", "pb_ratio"])

        # Step 4: Calculate percentile ranks
        df_with_percentiles = calculate_percentile_ranks(
            df_with_zscores, metrics=["pe_ratio", "pb_ratio"]
        )

        # Step 5: Calculate multi-factor scores
        mf_scores = calculate_multi_factor_score(
            df_with_percentiles,
            valuation_col="mispricing_score",
            quality_cols=["roe"],
            growth_cols=["revenue_growth"],
        )
        df_with_percentiles["multi_factor_score"] = mf_scores

        # Step 6: Filter for best opportunities
        opportunities = filter_stocks_by_criteria(
            df_with_percentiles, min_mispricing=10.0, min_market_cap=10e9
        )

        # Verify workflow completed successfully
        self.assertGreater(len(opportunities), 0)
        self.assertIn("valuation_category", opportunities.columns)
        self.assertIn("pe_ratio_zscore", opportunities.columns)
        self.assertIn("multi_factor_score", opportunities.columns)


if __name__ == "__main__":
    unittest.main()
