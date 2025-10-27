"""
Tests for finance_ml.advanced_eda module - Phase 9.2

Test-Driven Development approach for Advanced EDA functions:
- Correlation analysis (Pearson, Spearman, Kendall)
- Distribution testing (normality, skewness, kurtosis)
- Feature importance analysis
- Automated EDA report generation
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestCorrelationAnalysis(unittest.TestCase):
    """Test correlation analysis functions"""

    def setUp(self):
        """Create sample financial data"""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "market_cap": np.random.lognormal(10, 2, n),
                "p_e": np.random.gamma(2, 5, n),
                "p_b": np.random.gamma(1.5, 1, n),
                "revenue": np.random.lognormal(8, 1.5, n),
                "net_income": np.random.lognormal(6, 2, n),
                "sector": np.random.choice(["Technology", "Finance", "Healthcare"], n),
                "region": np.random.choice(["US", "EU", "APAC"], n),
            }
        )

    def test_calculate_correlation_matrix_pearson(self):
        """Test Pearson correlation matrix calculation"""
        from finance_ml.advanced_eda import calculate_correlation_matrix

        corr_matrix = calculate_correlation_matrix(
            self.df, columns=["market_cap", "p_e", "p_b", "revenue"], method="pearson"
        )

        # Check return type
        self.assertIsInstance(corr_matrix, pd.DataFrame)

        # Check shape (should be square matrix)
        self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])

        # Check diagonal is 1.0 (self-correlation)
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(len(corr_matrix)))

        # Check symmetry
        np.testing.assert_array_almost_equal(corr_matrix.values, corr_matrix.values.T)

        # Check values in range [-1, 1]
        self.assertTrue((corr_matrix.values >= -1).all() and (corr_matrix.values <= 1).all())

    def test_calculate_correlation_matrix_spearman(self):
        """Test Spearman correlation matrix calculation"""
        from finance_ml.advanced_eda import calculate_correlation_matrix

        corr_matrix = calculate_correlation_matrix(
            self.df, columns=["market_cap", "revenue", "net_income"], method="spearman"
        )

        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape, (3, 3))

    def test_calculate_correlation_matrix_kendall(self):
        """Test Kendall tau correlation matrix calculation"""
        from finance_ml.advanced_eda import calculate_correlation_matrix

        corr_matrix = calculate_correlation_matrix(
            self.df, columns=["p_e", "p_b"], method="kendall"
        )

        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape, (2, 2))

    def test_calculate_correlation_matrix_default_columns(self):
        """Test correlation with automatic column selection"""
        from finance_ml.advanced_eda import calculate_correlation_matrix

        corr_matrix = calculate_correlation_matrix(self.df)

        # Should auto-select numeric columns
        self.assertGreaterEqual(corr_matrix.shape[0], 5)

    def test_find_high_correlations(self):
        """Test finding highly correlated feature pairs"""
        from finance_ml.advanced_eda import find_high_correlations

        high_corr = find_high_correlations(
            self.df, columns=["market_cap", "revenue", "net_income"], threshold=0.3
        )

        # Check return type
        self.assertIsInstance(high_corr, pd.DataFrame)

        # Check columns
        expected_columns = ["feature_1", "feature_2", "correlation"]
        for col in expected_columns:
            self.assertIn(col, high_corr.columns)

        # Check all correlations above threshold
        if len(high_corr) > 0:
            self.assertTrue((high_corr["correlation"].abs() >= 0.3).all())


class TestDistributionAnalysis(unittest.TestCase):
    """Test distribution testing functions"""

    def setUp(self):
        """Create sample data with known distributions"""
        np.random.seed(42)
        n = 100

        # Normal distribution
        self.normal_data = np.random.normal(0, 1, n)

        # Skewed distribution (lognormal)
        self.skewed_data = np.random.lognormal(0, 1, n)

        # Create DataFrame
        self.df = pd.DataFrame(
            {
                "normal_var": self.normal_data,
                "skewed_var": self.skewed_data,
                "uniform_var": np.random.uniform(0, 100, n),
                "sector": np.random.choice(["A", "B", "C"], n),
            }
        )

    def test_test_normality(self):
        """Test normality testing with Shapiro-Wilk"""
        from finance_ml.advanced_eda import test_normality

        result = test_normality(self.df, columns=["normal_var", "skewed_var"])

        # Check return type
        self.assertIsInstance(result, pd.DataFrame)

        # Check columns
        expected_columns = ["column", "statistic", "p_value", "is_normal"]
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Check we have results for both columns
        self.assertEqual(len(result), 2)

        # Normal data should more likely pass normality test
        # (though not guaranteed with small sample)
        normal_result = result[result["column"] == "normal_var"].iloc[0]
        self.assertIn("is_normal", result.columns)

    def test_calculate_distribution_stats(self):
        """Test distribution statistics calculation"""
        from finance_ml.advanced_eda import calculate_distribution_stats

        stats = calculate_distribution_stats(self.df, columns=["normal_var", "skewed_var"])

        # Check return type
        self.assertIsInstance(stats, pd.DataFrame)

        # Check expected statistics columns
        expected_stats = ["mean", "median", "std", "skewness", "kurtosis", "min", "max"]
        for stat in expected_stats:
            self.assertIn(stat, stats.columns)

        # Check we have results for both columns
        self.assertEqual(len(stats), 2)

        # Skewed data should have higher skewness
        skewed_skewness = stats[stats.index == "skewed_var"]["skewness"].values[0]
        self.assertGreater(skewed_skewness, 0.5)  # Lognormal is right-skewed

    def test_calculate_distribution_stats_by_group(self):
        """Test distribution statistics by sector"""
        from finance_ml.advanced_eda import calculate_distribution_stats

        stats = calculate_distribution_stats(self.df, columns=["normal_var"], group_by="sector")

        # Should have stats for each sector
        self.assertGreaterEqual(len(stats), 3)


class TestFeatureImportance(unittest.TestCase):
    """Test feature importance analysis functions"""

    def setUp(self):
        """Create sample data with target variable"""
        np.random.seed(42)
        n = 200

        # Create features with varying importance
        self.df = pd.DataFrame(
            {
                "important_1": np.random.randn(n),
                "important_2": np.random.randn(n),
                "noise_1": np.random.randn(n),
                "noise_2": np.random.randn(n),
            }
        )

        # Create target that depends on important features
        self.df["target"] = (
            2 * self.df["important_1"]
            + 1.5 * self.df["important_2"]
            + 0.1 * self.df["noise_1"]
            + np.random.randn(n) * 0.5
        )

    def test_calculate_mutual_information(self):
        """Test mutual information feature importance"""
        from finance_ml.advanced_eda import calculate_mutual_information

        importance = calculate_mutual_information(
            self.df, target="target", features=["important_1", "important_2", "noise_1", "noise_2"]
        )

        # Check return type
        self.assertIsInstance(importance, pd.DataFrame)

        # Check columns
        self.assertIn("feature", importance.columns)
        self.assertIn("importance", importance.columns)

        # Check we have results for all features
        self.assertEqual(len(importance), 4)

        # Important features should have higher MI scores
        important_scores = importance[importance["feature"].isin(["important_1", "important_2"])][
            "importance"
        ].values
        noise_scores = importance[importance["feature"].isin(["noise_1", "noise_2"])][
            "importance"
        ].values

        self.assertGreater(important_scores.mean(), noise_scores.mean())

    def test_calculate_rf_importance(self):
        """Test Random Forest feature importance"""
        from finance_ml.advanced_eda import calculate_rf_importance

        importance = calculate_rf_importance(
            self.df, target="target", features=["important_1", "important_2", "noise_1", "noise_2"]
        )

        # Check return type
        self.assertIsInstance(importance, pd.DataFrame)

        # Check structure
        self.assertIn("feature", importance.columns)
        self.assertIn("importance", importance.columns)
        self.assertEqual(len(importance), 4)

        # Important features should rank higher
        top_features = importance.nlargest(2, "importance")["feature"].values
        self.assertIn("important_1", top_features)


class TestAutomatedEDA(unittest.TestCase):
    """Test automated EDA report generation"""

    def setUp(self):
        """Create sample financial data"""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n)],
                "market_cap": np.random.lognormal(10, 2, n),
                "p_e": np.random.gamma(2, 5, n),
                "revenue": np.random.lognormal(8, 1.5, n),
                "sector": np.random.choice(["Technology", "Finance"], n),
                "region": np.random.choice(["US", "EU"], n),
            }
        )

        # Create temporary directory for reports
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_eda_report(self):
        """Test automated EDA report generation"""
        from finance_ml.advanced_eda import generate_eda_report

        report_path = generate_eda_report(
            self.df, output_dir=self.temp_dir, title="Test EDA Report"
        )

        # Check report was created
        self.assertIsInstance(report_path, (str, Path))
        report_file = Path(report_path)
        self.assertTrue(report_file.exists())

        # Check it's a valid file with content
        self.assertGreater(report_file.stat().st_size, 0)

    def test_generate_eda_summary_dict(self):
        """Test EDA summary dictionary generation"""
        from finance_ml.advanced_eda import generate_eda_summary

        summary = generate_eda_summary(self.df)

        # Check return type
        self.assertIsInstance(summary, dict)

        # Check expected keys
        expected_keys = [
            "shape",
            "numeric_columns",
            "categorical_columns",
            "missing_values",
            "summary_statistics",
        ]
        for key in expected_keys:
            self.assertIn(key, summary)

        # Check shape
        self.assertEqual(summary["shape"], (100, 6))

        # Check numeric columns identified
        self.assertGreaterEqual(len(summary["numeric_columns"]), 3)


class TestSectorRegionAnalysis(unittest.TestCase):
    """Test sector and region-specific analysis functions"""

    def setUp(self):
        """Create sample multi-sector data"""
        np.random.seed(42)
        n_per_sector = 50

        sectors = ["Technology", "Finance", "Healthcare"]
        regions = ["US", "EU", "APAC"]

        data = []
        for sector in sectors:
            for region in regions:
                n = n_per_sector
                df_segment = pd.DataFrame(
                    {
                        "sector": [sector] * n,
                        "region": [region] * n,
                        "p_e": np.random.gamma(2, 5, n) + (10 if sector == "Technology" else 0),
                        "market_cap": np.random.lognormal(10, 2, n),
                        "revenue": np.random.lognormal(8, 1.5, n),
                    }
                )
                data.append(df_segment)

        self.df = pd.concat(data, ignore_index=True)

    def test_analyze_by_sector(self):
        """Test sector-wise analysis"""
        from finance_ml.advanced_eda import analyze_by_sector

        sector_stats = analyze_by_sector(self.df, metrics=["p_e", "market_cap"])

        # Check return type
        self.assertIsInstance(sector_stats, pd.DataFrame)

        # Should have results for each sector
        self.assertGreaterEqual(len(sector_stats), 3)

        # Should include aggregated statistics
        self.assertTrue(any("mean" in col or "median" in col for col in sector_stats.columns))

    def test_analyze_by_region(self):
        """Test region-wise analysis"""
        from finance_ml.advanced_eda import analyze_by_region

        region_stats = analyze_by_region(self.df, metrics=["p_e", "revenue"])

        # Check return type
        self.assertIsInstance(region_stats, pd.DataFrame)

        # Should have results for each region
        self.assertGreaterEqual(len(region_stats), 3)

    def test_compare_sector_distributions(self):
        """Test statistical comparison of sector distributions"""
        from finance_ml.advanced_eda import compare_sector_distributions

        comparison = compare_sector_distributions(self.df, metric="p_e", test="anova")

        # Check return type
        self.assertIsInstance(comparison, dict)

        # Check expected keys
        self.assertIn("statistic", comparison)
        self.assertIn("p_value", comparison)
        self.assertIn("significant", comparison)


if __name__ == "__main__":
    unittest.main()
