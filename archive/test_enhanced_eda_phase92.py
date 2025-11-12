"""
Tests for Phase 9.2 - Enhanced Exploratory Data Analysis of Financial Metrics

Test-Driven Development approach for:
1. Automated EDA report generation with financial metric dashboards
2. Statistical hypothesis testing framework (ANOVA, Kruskal-Wallis, t-tests, Mann-Whitney U)
3. Data quality alert system
4. Interactive dashboard helper functions

Following strict TDD: write failing tests → implement minimal code → refactor
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestFinancialMetricsDashboard(unittest.TestCase):
    """Test automated financial metrics dashboard generation"""

    def setUp(self):
        """Create sample financial data"""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n)],
                "sector": np.random.choice(["Technology", "Finance", "Healthcare", "Energy"], n),
                "region": np.random.choice(["US", "EU", "APAC", "ROTW"], n),
                "market_cap": np.random.lognormal(10, 2, n),
                "last_price": np.random.gamma(20, 2, n),
                "price_target": np.random.gamma(22, 2, n),
                "p_e": np.random.gamma(2, 5, n),
                "p_b": np.random.gamma(1.5, 1, n),
                "ev_ebitda": np.random.gamma(2, 3, n),
                "revenue": np.random.lognormal(8, 1.5, n),
                "net_income": np.random.lognormal(6, 2, n),
                "ebitda": np.random.lognormal(7, 1.8, n),
                "total_debt": np.random.lognormal(7.5, 2, n),
                "total_equity": np.random.lognormal(8.5, 1.5, n),
                "gross_margin": np.random.uniform(0.2, 0.6, n),
                "operating_margin": np.random.uniform(0.1, 0.4, n),
                "net_margin": np.random.uniform(0.05, 0.3, n),
                "roe": np.random.uniform(0.05, 0.25, n),
                "roa": np.random.uniform(0.02, 0.15, n),
                "revenue_growth": np.random.uniform(-0.1, 0.3, n),
                "debt_to_equity": np.random.uniform(0.2, 2.5, n),
            }
        )

    def test_calculate_financial_metrics_dashboard_returns_dict(self):
        """Test that calculate_financial_metrics_dashboard returns a structured dictionary"""
        from finance_ml.eval import calculate_financial_metrics_dashboard

        result = calculate_financial_metrics_dashboard(self.df)

        # Check return type
        self.assertIsInstance(result, dict)

        # Check main sections exist
        expected_sections = ["valuation", "profitability", "growth", "leverage"]
        for section in expected_sections:
            self.assertIn(section, result, f"Missing section: {section}")

    def test_calculate_financial_metrics_dashboard_valuation_metrics(self):
        """Test that valuation section includes P/E, P/B, EV/EBITDA statistics"""
        from finance_ml.eval import calculate_financial_metrics_dashboard

        result = calculate_financial_metrics_dashboard(self.df)
        valuation = result["valuation"]

        # Check valuation metrics exist
        expected_metrics = ["p_e", "p_b", "ev_ebitda"]
        for metric in expected_metrics:
            self.assertIn(metric, valuation, f"Missing valuation metric: {metric}")
            # Each metric should have statistics
            self.assertIn("mean", valuation[metric])
            self.assertIn("median", valuation[metric])
            self.assertIn("std", valuation[metric])

    def test_calculate_financial_metrics_dashboard_profitability_metrics(self):
        """Test that profitability section includes margins and returns"""
        from finance_ml.eval import calculate_financial_metrics_dashboard

        result = calculate_financial_metrics_dashboard(self.df)
        profitability = result["profitability"]

        # Check profitability metrics
        expected_metrics = ["gross_margin", "operating_margin", "net_margin", "roe", "roa"]
        for metric in expected_metrics:
            self.assertIn(metric, profitability, f"Missing profitability metric: {metric}")

    def test_calculate_financial_metrics_dashboard_growth_metrics(self):
        """Test that growth section includes revenue growth statistics"""
        from finance_ml.eval import calculate_financial_metrics_dashboard

        result = calculate_financial_metrics_dashboard(self.df)
        growth = result["growth"]

        # Check growth metrics
        self.assertIn("revenue_growth", growth)
        self.assertIn("mean", growth["revenue_growth"])

    def test_calculate_financial_metrics_dashboard_leverage_metrics(self):
        """Test that leverage section includes debt ratios"""
        from finance_ml.eval import calculate_financial_metrics_dashboard

        result = calculate_financial_metrics_dashboard(self.df)
        leverage = result["leverage"]

        # Check leverage metrics
        self.assertIn("debt_to_equity", leverage)
        self.assertIn("mean", leverage["debt_to_equity"])

    def test_calculate_financial_metrics_dashboard_by_sector(self):
        """Test that dashboard can be calculated by sector"""
        from finance_ml.eval import calculate_financial_metrics_dashboard

        result = calculate_financial_metrics_dashboard(self.df, group_by="sector")

        # Check that result contains sector breakdowns
        self.assertIsInstance(result, dict)
        # Should have sector-level analysis
        for sector in self.df["sector"].unique():
            # At minimum, should contain data or be processable by sector
            pass

    def test_calculate_financial_metrics_dashboard_handles_missing_columns(self):
        """Test that dashboard handles missing columns gracefully"""
        from finance_ml.eval import calculate_financial_metrics_dashboard

        # Create df with only some metrics
        df_partial = self.df[["ticker", "sector", "p_e", "gross_margin"]].copy()

        result = calculate_financial_metrics_dashboard(df_partial)

        # Should return dict even with missing columns
        self.assertIsInstance(result, dict)
        # Valuation should have p_e but not p_b
        if "valuation" in result:
            self.assertIn("p_e", result["valuation"])


class TestDataQualityAlerts(unittest.TestCase):
    """Test data quality alert system"""

    def setUp(self):
        """Create sample data with quality issues"""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n)],
                "sector": np.random.choice(["Technology", "Finance", "Healthcare"], n),
                "market_cap": np.random.lognormal(10, 2, n),
                "p_e": np.random.gamma(2, 5, n),
                "revenue": np.random.lognormal(8, 1.5, n),
            }
        )
        # Introduce quality issues
        self.df.loc[0:5, "market_cap"] = np.nan  # Missing values
        self.df.loc[10:12, "p_e"] = [1000, 1500, 2000]  # Outliers
        self.df.loc[20, "revenue"] = -100  # Negative value

    def test_generate_data_quality_alerts_returns_list(self):
        """Test that generate_data_quality_alerts returns a list of alerts"""
        from finance_ml.eval import generate_data_quality_alerts

        alerts = generate_data_quality_alerts(self.df)

        # Check return type
        self.assertIsInstance(alerts, list)

    def test_generate_data_quality_alerts_detects_missing_values(self):
        """Test that alerts detect missing values"""
        from finance_ml.eval import generate_data_quality_alerts

        alerts = generate_data_quality_alerts(self.df)

        # Should detect missing values in market_cap
        alert_texts = [alert["message"] for alert in alerts]
        missing_alert = any(
            "missing" in text.lower() or "null" in text.lower() for text in alert_texts
        )
        self.assertTrue(missing_alert, "Should detect missing values")

    def test_generate_data_quality_alerts_detects_outliers(self):
        """Test that alerts detect statistical outliers"""
        from finance_ml.eval import generate_data_quality_alerts

        alerts = generate_data_quality_alerts(self.df)

        # Should detect outliers in p_e
        alert_texts = [alert["message"] for alert in alerts]
        outlier_alert = any("outlier" in text.lower() for text in alert_texts)
        self.assertTrue(outlier_alert, "Should detect outliers")

    def test_generate_data_quality_alerts_detects_negative_values(self):
        """Test that alerts detect negative values in financial metrics"""
        from finance_ml.eval import generate_data_quality_alerts

        alerts = generate_data_quality_alerts(self.df)

        # Should detect negative revenue
        alert_texts = [alert["message"] for alert in alerts]
        negative_alert = any("negative" in text.lower() for text in alert_texts)
        self.assertTrue(negative_alert, "Should detect negative values in financial metrics")

    def test_generate_data_quality_alerts_structure(self):
        """Test that each alert has required fields"""
        from finance_ml.eval import generate_data_quality_alerts

        alerts = generate_data_quality_alerts(self.df)

        if len(alerts) > 0:
            # Each alert should have required fields
            for alert in alerts:
                self.assertIn("severity", alert)
                self.assertIn("message", alert)
                self.assertIn("column", alert)
                # Severity should be one of: low, medium, high, critical
                self.assertIn(alert["severity"], ["low", "medium", "high", "critical"])


class TestStatisticalHypothesisTesting(unittest.TestCase):
    """Test statistical hypothesis testing framework"""

    def setUp(self):
        """Create sample data for hypothesis testing"""
        np.random.seed(42)
        n = 50
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n * 3)],
                "sector": (["Technology"] * n + ["Finance"] * n + ["Healthcare"] * n),
                "region": np.random.choice(["US", "EU", "APAC"], n * 3),
                "p_e": np.concatenate(
                    [
                        np.random.gamma(2, 5, n),  # Technology
                        np.random.gamma(1.5, 4, n),  # Finance
                        np.random.gamma(2.5, 6, n),  # Healthcare
                    ]
                ),
                "roe": np.concatenate(
                    [
                        np.random.uniform(0.1, 0.2, n),  # Technology
                        np.random.uniform(0.15, 0.25, n),  # Finance
                        np.random.uniform(0.08, 0.18, n),  # Healthcare
                    ]
                ),
                "revenue_growth": np.random.uniform(-0.1, 0.3, n * 3),
            }
        )

    def test_perform_comprehensive_hypothesis_tests_returns_dict(self):
        """Test that comprehensive hypothesis tests return structured results"""
        from finance_ml.eval import perform_comprehensive_hypothesis_tests

        results = perform_comprehensive_hypothesis_tests(self.df)

        # Check return type
        self.assertIsInstance(results, dict)

    def test_perform_comprehensive_hypothesis_tests_sector_comparison(self):
        """Test sector mean comparison using ANOVA and Kruskal-Wallis"""
        from finance_ml.eval import perform_comprehensive_hypothesis_tests

        results = perform_comprehensive_hypothesis_tests(self.df, group_column="sector")

        # Check sector tests exist
        self.assertIn("sector_tests", results)
        sector_tests = results["sector_tests"]

        # Results should be organized by metric, then by test type
        # Check that at least one metric has ANOVA results
        has_anova = False
        has_kruskal = False
        for metric, metric_tests in sector_tests.items():
            if metric == "summary":
                continue  # Skip summary section
            if isinstance(metric_tests, dict):
                if "anova" in metric_tests:
                    has_anova = True
                    self.assertIn("p_value", metric_tests["anova"])
                    self.assertIn("statistic", metric_tests["anova"])
                if "kruskal_wallis" in metric_tests:
                    has_kruskal = True
                    self.assertIn("p_value", metric_tests["kruskal_wallis"])

        # At least one metric should have both tests
        self.assertTrue(has_anova, "Should have ANOVA results for at least one metric")
        self.assertTrue(has_kruskal, "Should have Kruskal-Wallis results for at least one metric")

    def test_perform_comprehensive_hypothesis_tests_region_comparison(self):
        """Test region comparison using t-tests and Mann-Whitney U"""
        from finance_ml.eval import perform_comprehensive_hypothesis_tests

        results = perform_comprehensive_hypothesis_tests(self.df, group_column="region")

        # Check region tests exist
        self.assertIn("region_tests", results)
        region_tests = results["region_tests"]

        # Should have pairwise comparison results
        self.assertIsInstance(region_tests, dict)

    def test_perform_comprehensive_hypothesis_tests_multiple_metrics(self):
        """Test hypothesis tests on multiple metrics"""
        from finance_ml.eval import perform_comprehensive_hypothesis_tests

        metrics = ["p_e", "roe", "revenue_growth"]
        results = perform_comprehensive_hypothesis_tests(
            self.df, group_column="sector", metrics=metrics
        )

        # Should have results for each metric
        for metric in metrics:
            # Results should reference the metrics
            pass  # Implementation will determine exact structure

    def test_test_market_efficiency_hypothesis_returns_dict(self):
        """Test market efficiency hypothesis testing"""
        from finance_ml.eval import test_market_efficiency_hypothesis

        # Add price target relationship
        self.df["last_price"] = np.random.gamma(20, 2, len(self.df))
        self.df["price_target"] = self.df["last_price"] * np.random.uniform(0.9, 1.1, len(self.df))

        results = test_market_efficiency_hypothesis(self.df)

        # Check return type
        self.assertIsInstance(results, dict)

    def test_test_market_efficiency_hypothesis_price_target_relationship(self):
        """Test that market efficiency test examines price/target relationship"""
        from finance_ml.eval import test_market_efficiency_hypothesis

        self.df["last_price"] = np.random.gamma(20, 2, len(self.df))
        self.df["price_target"] = self.df["last_price"] * np.random.uniform(0.9, 1.1, len(self.df))

        results = test_market_efficiency_hypothesis(self.df)

        # Should test if targets are significantly different from prices
        self.assertIn("price_target_test", results)
        self.assertIn("p_value", results["price_target_test"])

    def test_test_market_efficiency_hypothesis_handles_missing_columns(self):
        """Test that market efficiency test handles missing price/target columns"""
        from finance_ml.eval import test_market_efficiency_hypothesis

        # Don't add price columns
        results = test_market_efficiency_hypothesis(self.df)

        # Should handle gracefully
        self.assertIsInstance(results, dict)


class TestInteractiveDashboardHelpers(unittest.TestCase):
    """Test interactive dashboard helper functions"""

    def setUp(self):
        """Create sample data for dashboard"""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n)],
                "sector": np.random.choice(["Technology", "Finance", "Healthcare", "Energy"], n),
                "region": np.random.choice(["US", "EU", "APAC", "ROTW"], n),
                "market_cap": np.random.lognormal(10, 2, n),
                "last_price": np.random.gamma(20, 2, n),
                "price_target": np.random.gamma(22, 2, n),
                "p_e": np.random.gamma(2, 5, n),
                "p_b": np.random.gamma(1.5, 1, n),
                "revenue": np.random.lognormal(8, 1.5, n),
                "net_income": np.random.lognormal(6, 2, n),
            }
        )
        self.df["mispricing_score"] = (
            (self.df["price_target"] - self.df["last_price"]) / self.df["last_price"] * 100
        )

    def test_prepare_interactive_dashboard_data_returns_dict(self):
        """Test that prepare_interactive_dashboard_data returns structured data"""
        from finance_ml.eval import prepare_interactive_dashboard_data

        result = prepare_interactive_dashboard_data(self.df)

        # Check return type
        self.assertIsInstance(result, dict)

    def test_prepare_interactive_dashboard_data_includes_summary_stats(self):
        """Test that dashboard data includes summary statistics"""
        from finance_ml.eval import prepare_interactive_dashboard_data

        result = prepare_interactive_dashboard_data(self.df)

        # Should include summary statistics
        self.assertIn("summary_stats", result)
        self.assertIsInstance(result["summary_stats"], dict)

    def test_prepare_interactive_dashboard_data_includes_sector_breakdown(self):
        """Test that dashboard data includes sector breakdown"""
        from finance_ml.eval import prepare_interactive_dashboard_data

        result = prepare_interactive_dashboard_data(self.df)

        # Should include sector breakdown
        self.assertIn("by_sector", result)
        self.assertIsInstance(result["by_sector"], dict)

    def test_prepare_interactive_dashboard_data_includes_region_breakdown(self):
        """Test that dashboard data includes region breakdown"""
        from finance_ml.eval import prepare_interactive_dashboard_data

        result = prepare_interactive_dashboard_data(self.df)

        # Should include region breakdown
        self.assertIn("by_region", result)
        self.assertIsInstance(result["by_region"], dict)

    def test_apply_dashboard_filters_sector_filter(self):
        """Test applying sector filter"""
        from finance_ml.eval import apply_dashboard_filters

        filters = {"sectors": ["Technology", "Finance"]}
        filtered_df = apply_dashboard_filters(self.df, filters)

        # Check filtering worked
        self.assertTrue(filtered_df["sector"].isin(["Technology", "Finance"]).all())
        self.assertLess(len(filtered_df), len(self.df))

    def test_apply_dashboard_filters_region_filter(self):
        """Test applying region filter"""
        from finance_ml.eval import apply_dashboard_filters

        filters = {"regions": ["US", "EU"]}
        filtered_df = apply_dashboard_filters(self.df, filters)

        # Check filtering worked
        self.assertTrue(filtered_df["region"].isin(["US", "EU"]).all())

    def test_apply_dashboard_filters_market_cap_range(self):
        """Test applying market cap range filter"""
        from finance_ml.eval import apply_dashboard_filters

        min_cap = self.df["market_cap"].quantile(0.25)
        max_cap = self.df["market_cap"].quantile(0.75)
        filters = {"min_market_cap": min_cap, "max_market_cap": max_cap}
        filtered_df = apply_dashboard_filters(self.df, filters)

        # Check filtering worked
        self.assertTrue((filtered_df["market_cap"] >= min_cap).all())
        self.assertTrue((filtered_df["market_cap"] <= max_cap).all())

    def test_apply_dashboard_filters_valuation_range(self):
        """Test applying valuation (mispricing score) range filter"""
        from finance_ml.eval import apply_dashboard_filters

        filters = {"min_mispricing": -10, "max_mispricing": 20}
        filtered_df = apply_dashboard_filters(self.df, filters)

        # Check filtering worked
        if "mispricing_score" in filtered_df.columns:
            self.assertTrue((filtered_df["mispricing_score"] >= -10).all())
            self.assertTrue((filtered_df["mispricing_score"] <= 20).all())

    def test_apply_dashboard_filters_combined(self):
        """Test applying multiple filters simultaneously"""
        from finance_ml.eval import apply_dashboard_filters

        filters = {
            "sectors": ["Technology"],
            "regions": ["US"],
            "min_market_cap": 1e9,
        }
        filtered_df = apply_dashboard_filters(self.df, filters)

        # Check all filters applied
        self.assertTrue((filtered_df["sector"] == "Technology").all())
        self.assertTrue((filtered_df["region"] == "US").all())
        self.assertTrue((filtered_df["market_cap"] >= 1e9).all())

    def test_calculate_peer_comparisons_returns_dict(self):
        """Test that calculate_peer_comparisons returns comparison data"""
        from finance_ml.eval import calculate_peer_comparisons

        ticker = self.df.iloc[0]["ticker"]
        result = calculate_peer_comparisons(self.df, ticker)

        # Check return type
        self.assertIsInstance(result, dict)

    def test_calculate_peer_comparisons_includes_stock_data(self):
        """Test that peer comparison includes selected stock data"""
        from finance_ml.eval import calculate_peer_comparisons

        ticker = self.df.iloc[0]["ticker"]
        result = calculate_peer_comparisons(self.df, ticker)

        # Should include stock data
        self.assertIn("stock", result)
        self.assertIsInstance(result["stock"], dict)

    def test_calculate_peer_comparisons_includes_sector_average(self):
        """Test that peer comparison includes sector average"""
        from finance_ml.eval import calculate_peer_comparisons

        ticker = self.df.iloc[0]["ticker"]
        result = calculate_peer_comparisons(self.df, ticker)

        # Should include sector average
        self.assertIn("sector_avg", result)
        self.assertIsInstance(result["sector_avg"], dict)

    def test_calculate_peer_comparisons_includes_peers(self):
        """Test that peer comparison includes similar peer stocks"""
        from finance_ml.eval import calculate_peer_comparisons

        ticker = self.df.iloc[0]["ticker"]
        result = calculate_peer_comparisons(self.df, ticker, n_peers=5)

        # Should include peer list
        self.assertIn("peers", result)
        self.assertIsInstance(result["peers"], list)
        self.assertLessEqual(len(result["peers"]), 5)


class TestEnhancedEDAReportGeneration(unittest.TestCase):
    """Test enhanced EDA report generation with new features"""

    def setUp(self):
        """Create sample financial data"""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n)],
                "sector": np.random.choice(["Technology", "Finance", "Healthcare"], n),
                "region": np.random.choice(["US", "EU", "APAC"], n),
                "market_cap": np.random.lognormal(10, 2, n),
                "last_price": np.random.gamma(20, 2, n),
                "price_target": np.random.gamma(22, 2, n),
                "p_e": np.random.gamma(2, 5, n),
                "p_b": np.random.gamma(1.5, 1, n),
                "revenue": np.random.lognormal(8, 1.5, n),
                "gross_margin": np.random.uniform(0.2, 0.6, n),
                "roe": np.random.uniform(0.05, 0.25, n),
                "revenue_growth": np.random.uniform(-0.1, 0.3, n),
                "debt_to_equity": np.random.uniform(0.2, 2.5, n),
            }
        )

    def test_generate_eda_report_includes_financial_dashboard(self):
        """Test that generate_eda_report includes financial metrics dashboard"""
        from finance_ml.eval import generate_eda_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "eda_report.json"
            result = generate_eda_report(
                self.df,
                output_path=output_path,
                include_financial_dashboard=True,
            )

            # Should include financial dashboard
            self.assertIn("financial_dashboard", result)
            self.assertIn("valuation", result["financial_dashboard"])
            self.assertIn("profitability", result["financial_dashboard"])

    def test_generate_eda_report_includes_quality_alerts(self):
        """Test that generate_eda_report includes data quality alerts"""
        from finance_ml.eval import generate_eda_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "eda_report.json"
            result = generate_eda_report(
                self.df,
                output_path=output_path,
                include_quality_alerts=True,
            )

            # Should include quality alerts
            self.assertIn("quality_alerts", result)
            self.assertIsInstance(result["quality_alerts"], list)

    def test_generate_eda_report_includes_hypothesis_tests(self):
        """Test that generate_eda_report includes statistical hypothesis tests"""
        from finance_ml.eval import generate_eda_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "eda_report.json"
            result = generate_eda_report(
                self.df,
                output_path=output_path,
                include_statistical_tests=True,
            )

            # Should include hypothesis tests
            self.assertIn("hypothesis_tests", result)
            self.assertIsInstance(result["hypothesis_tests"], dict)

    def test_generate_eda_report_saves_to_file(self):
        """Test that generate_eda_report saves report to file"""
        from finance_ml.eval import generate_eda_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "eda_report.json"
            generate_eda_report(self.df, output_path=output_path)

            # Check file exists
            self.assertTrue(output_path.exists())

            # Check file is valid JSON
            with open(output_path, "r") as f:
                data = json.load(f)
                self.assertIsInstance(data, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
