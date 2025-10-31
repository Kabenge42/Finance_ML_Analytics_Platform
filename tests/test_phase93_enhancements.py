"""
Tests for Phase 9.3 Future Enhancements.

This test module follows strict TDD methodology (Red-Green-Refactor):
1. Write failing tests first
2. Implement minimal code to pass tests
3. Refactor for quality

Phase 9.3 Features:
- Time-series hypothesis testing for temporal trends
- Multi-factor ANOVA for interaction effects
- Automated outlier correction with validation
- Enhanced Plotly dashboard data preparation
- Enhanced PDF report generation with Phase 9.2 integration

Coverage target: ≥80% for new code
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from datetime import datetime, timedelta


class TestTimeSeriesHypothesisTesting(unittest.TestCase):
    """Test time-series hypothesis testing for temporal trends."""

    def setUp(self):
        """Create sample time-series data."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create data with temporal trend
        self.df_with_trend = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["AAPL"] * 100,
                "sector": ["Technology"] * 100,
                "price": 100 + np.arange(100) * 0.5 + np.random.randn(100) * 5,
                "volume": 1000000 + np.arange(100) * 10000 + np.random.randn(100) * 50000,
                "p_e": 20 + np.random.randn(100) * 2,
            }
        )

        # Create data with no trend (stationary)
        self.df_stationary = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["MSFT"] * 100,
                "sector": ["Technology"] * 100,
                "price": 50 + np.random.randn(100) * 5,
                "volume": 1000000 + np.random.randn(100) * 50000,
                "p_e": 15 + np.random.randn(100) * 2,
            }
        )

        # Combined dataset
        self.df = pd.concat([self.df_with_trend, self.df_stationary], ignore_index=True)

    def test_perform_time_series_hypothesis_tests_returns_dict(self):
        """Test that function returns a dictionary with expected structure."""
        from finance_ml.eval import perform_time_series_hypothesis_tests

        result = perform_time_series_hypothesis_tests(
            self.df, date_column="date", metrics=["price", "volume"]
        )

        self.assertIsInstance(result, dict)
        self.assertIn("trend_tests", result)
        self.assertIn("stationarity_tests", result)
        self.assertIn("autocorrelation_tests", result)

    def test_trend_test_detects_temporal_trends(self):
        """Test that trend tests correctly identify temporal trends."""
        from finance_ml.eval import perform_time_series_hypothesis_tests

        result = perform_time_series_hypothesis_tests(
            self.df_with_trend, date_column="date", metrics=["price"]
        )

        trend_test = result["trend_tests"]["price"]
        self.assertIn("has_trend", trend_test)
        self.assertIn("p_value", trend_test)
        self.assertIn("test_statistic", trend_test)
        self.assertTrue(trend_test["has_trend"])  # Should detect upward trend

    def test_stationarity_test_identifies_non_stationary_series(self):
        """Test that stationarity tests identify non-stationary series."""
        from finance_ml.eval import perform_time_series_hypothesis_tests

        result = perform_time_series_hypothesis_tests(
            self.df_with_trend, date_column="date", metrics=["price"]
        )

        stationarity_test = result["stationarity_tests"]["price"]
        self.assertIn("is_stationary", stationarity_test)
        self.assertIn("adf_statistic", stationarity_test)
        self.assertIn("p_value", stationarity_test)
        self.assertFalse(stationarity_test["is_stationary"])  # Should be non-stationary

    def test_autocorrelation_test_detects_serial_correlation(self):
        """Test that autocorrelation tests detect serial correlation."""
        from finance_ml.eval import perform_time_series_hypothesis_tests

        result = perform_time_series_hypothesis_tests(
            self.df_with_trend, date_column="date", metrics=["price"]
        )

        autocorr_test = result["autocorrelation_tests"]["price"]
        self.assertIn("ljung_box_statistic", autocorr_test)
        self.assertIn("p_value", autocorr_test)
        self.assertIn("has_autocorrelation", autocorr_test)

    def test_handles_multiple_metrics(self):
        """Test that function handles multiple metrics simultaneously."""
        from finance_ml.eval import perform_time_series_hypothesis_tests

        result = perform_time_series_hypothesis_tests(
            self.df, date_column="date", metrics=["price", "volume", "p_e"]
        )

        self.assertEqual(len(result["trend_tests"]), 3)
        self.assertEqual(len(result["stationarity_tests"]), 3)
        self.assertEqual(len(result["autocorrelation_tests"]), 3)

    def test_handles_missing_date_column(self):
        """Test error handling for missing date column."""
        from finance_ml.eval import perform_time_series_hypothesis_tests

        df_no_date = self.df.drop(columns=["date"])

        with self.assertRaises(ValueError):
            perform_time_series_hypothesis_tests(df_no_date, date_column="date", metrics=["price"])

    def test_by_group_analysis(self):
        """Test time-series analysis grouped by ticker or sector."""
        from finance_ml.eval import perform_time_series_hypothesis_tests

        result = perform_time_series_hypothesis_tests(
            self.df, date_column="date", metrics=["price"], group_by="ticker"
        )

        self.assertIn("by_group", result)
        self.assertIn("AAPL", result["by_group"])
        self.assertIn("MSFT", result["by_group"])


class TestMultiFactorANOVA(unittest.TestCase):
    """Test multi-factor ANOVA for interaction effects."""

    def setUp(self):
        """Create sample data with interaction effects."""
        np.random.seed(42)
        n_samples = 200

        sectors = np.random.choice(["Technology", "Finance", "Healthcare"], n_samples)
        regions = np.random.choice(["US", "EU", "APAC"], n_samples)

        # Create interaction effects: Tech in US has higher P/E
        base_pe = 15
        pe_values = []
        for sector, region in zip(sectors, regions):
            pe = base_pe
            if sector == "Technology":
                pe += 5
            if region == "US":
                pe += 3
            if sector == "Technology" and region == "US":
                pe += 7  # Interaction effect
            pe += np.random.randn() * 2
            pe_values.append(pe)

        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": sectors,
                "region": regions,
                "p_e": pe_values,
                "roe": np.random.randn(n_samples) * 0.05 + 0.15,
                "market_cap": np.random.lognormal(20, 2, n_samples),
            }
        )

    def test_perform_multi_factor_anova_returns_dict(self):
        """Test that function returns dictionary with expected structure."""
        from finance_ml.eval import perform_multi_factor_anova

        result = perform_multi_factor_anova(
            self.df, dependent_var="p_e", factors=["sector", "region"]
        )

        self.assertIsInstance(result, dict)
        self.assertIn("main_effects", result)
        self.assertIn("interaction_effects", result)
        self.assertIn("model_summary", result)

    def test_detects_main_effects(self):
        """Test that main effects are correctly identified."""
        from finance_ml.eval import perform_multi_factor_anova

        result = perform_multi_factor_anova(
            self.df, dependent_var="p_e", factors=["sector", "region"]
        )

        main_effects = result["main_effects"]
        self.assertIn("sector", main_effects)
        self.assertIn("region", main_effects)

        # Check structure
        sector_effect = main_effects["sector"]
        self.assertIn("f_statistic", sector_effect)
        self.assertIn("p_value", sector_effect)
        self.assertIn("significant", sector_effect)

    def test_detects_interaction_effects(self):
        """Test that interaction effects are detected."""
        from finance_ml.eval import perform_multi_factor_anova

        result = perform_multi_factor_anova(
            self.df, dependent_var="p_e", factors=["sector", "region"]
        )

        interactions = result["interaction_effects"]
        self.assertIn("sector:region", interactions)

        interaction = interactions["sector:region"]
        self.assertIn("f_statistic", interaction)
        self.assertIn("p_value", interaction)
        self.assertIn("significant", interaction)

    def test_three_way_interactions(self):
        """Test three-way interaction effects."""
        from finance_ml.eval import perform_multi_factor_anova

        # Add a third factor
        self.df["size_class"] = np.random.choice(["Small", "Mid", "Large"], len(self.df))

        result = perform_multi_factor_anova(
            self.df, dependent_var="p_e", factors=["sector", "region", "size_class"]
        )

        self.assertIn("sector:region:size_class", result["interaction_effects"])

    def test_handles_multiple_dependent_variables(self):
        """Test ANOVA on multiple dependent variables."""
        from finance_ml.eval import perform_multi_factor_anova

        result = perform_multi_factor_anova(
            self.df, dependent_var=["p_e", "roe"], factors=["sector", "region"]
        )

        self.assertIn("p_e", result)
        self.assertIn("roe", result)

    def test_post_hoc_comparisons(self):
        """Test post-hoc pairwise comparisons."""
        from finance_ml.eval import perform_multi_factor_anova

        result = perform_multi_factor_anova(
            self.df, dependent_var="p_e", factors=["sector"], post_hoc=True
        )

        self.assertIn("post_hoc", result)
        self.assertIn("sector", result["post_hoc"])

    def test_effect_size_calculation(self):
        """Test that effect sizes (eta-squared) are calculated."""
        from finance_ml.eval import perform_multi_factor_anova

        result = perform_multi_factor_anova(
            self.df, dependent_var="p_e", factors=["sector", "region"]
        )

        self.assertIn("effect_sizes", result)
        effect_sizes = result["effect_sizes"]
        self.assertIn("sector", effect_sizes)
        self.assertIn("eta_squared", effect_sizes["sector"])


class TestAutomatedOutlierCorrection(unittest.TestCase):
    """Test automated outlier correction with validation."""

    def setUp(self):
        """Create sample data with outliers."""
        np.random.seed(42)
        n_samples = 100

        # Normal data
        normal_data = np.random.randn(n_samples) * 10 + 50

        # Inject outliers
        outlier_indices = [5, 15, 25, 35, 45]
        for idx in outlier_indices:
            normal_data[idx] = np.random.choice([150, -50])

        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Finance"], n_samples),
                "price": normal_data,
                "p_e": np.random.randn(n_samples) * 5 + 20,
                "volume": np.random.lognormal(15, 1, n_samples),
            }
        )

        # Known outlier indices
        self.known_outlier_indices = outlier_indices

    def test_correct_outliers_with_validation_returns_corrected_df(self):
        """Test that function returns corrected DataFrame."""
        from finance_ml.eval import correct_outliers_with_validation

        result = correct_outliers_with_validation(self.df, columns=["price"], method="winsorize")

        self.assertIn("corrected_data", result)
        self.assertIsInstance(result["corrected_data"], pd.DataFrame)
        self.assertEqual(len(result["corrected_data"]), len(self.df))

    def test_identifies_outliers_correctly(self):
        """Test that outliers are correctly identified."""
        from finance_ml.eval import correct_outliers_with_validation

        result = correct_outliers_with_validation(self.df, columns=["price"], method="winsorize")

        self.assertIn("outlier_report", result)
        report = result["outlier_report"]
        self.assertIn("price", report)
        self.assertIn("n_outliers", report["price"])
        self.assertGreater(report["price"]["n_outliers"], 0)

    def test_winsorize_method(self):
        """Test winsorization outlier correction method."""
        from finance_ml.eval import correct_outliers_with_validation

        result = correct_outliers_with_validation(
            self.df, columns=["price"], method="winsorize", limits=(0.05, 0.05)
        )

        corrected = result["corrected_data"]["price"]
        # Check that extreme values are capped
        self.assertLessEqual(corrected.max(), self.df["price"].quantile(0.95) * 1.5)

    def test_clip_method(self):
        """Test clipping outlier correction method."""
        from finance_ml.eval import correct_outliers_with_validation

        result = correct_outliers_with_validation(
            self.df, columns=["price"], method="clip", n_std=3
        )

        corrected = result["corrected_data"]["price"]
        mean = self.df["price"].mean()
        std = self.df["price"].std()

        # Check values are within bounds
        self.assertGreaterEqual(corrected.min(), mean - 3 * std - 1)
        self.assertLessEqual(corrected.max(), mean + 3 * std + 1)

    def test_impute_method(self):
        """Test imputation outlier correction method."""
        from finance_ml.eval import correct_outliers_with_validation

        result = correct_outliers_with_validation(
            self.df, columns=["price"], method="impute", impute_strategy="median"
        )

        corrected = result["corrected_data"]["price"]
        # Check that outliers were replaced (values changed)
        self.assertNotEqual(corrected.iloc[5], self.df["price"].iloc[5])

    def test_validation_metrics_included(self):
        """Test that validation metrics are calculated."""
        from finance_ml.eval import correct_outliers_with_validation

        result = correct_outliers_with_validation(self.df, columns=["price"], method="winsorize")

        self.assertIn("validation", result)
        validation = result["validation"]
        self.assertIn("before", validation)
        self.assertIn("after", validation)
        self.assertIn("improvement", validation)

        # Check metrics exist
        self.assertIn("mean", validation["before"])
        self.assertIn("std", validation["before"])
        self.assertIn("skewness", validation["before"])
        self.assertIn("kurtosis", validation["before"])

    def test_sector_specific_correction(self):
        """Test outlier correction by sector."""
        from finance_ml.eval import correct_outliers_with_validation

        result = correct_outliers_with_validation(
            self.df, columns=["price"], method="winsorize", by_group="sector"
        )

        self.assertIn("by_group", result)
        self.assertIn("Technology", result["by_group"])
        self.assertIn("Finance", result["by_group"])

    def test_correction_reversibility(self):
        """Test that correction can be reversed with mapping."""
        from finance_ml.eval import correct_outliers_with_validation

        result = correct_outliers_with_validation(
            self.df, columns=["price"], method="winsorize", return_mapping=True
        )

        self.assertIn("correction_mapping", result)
        mapping = result["correction_mapping"]
        self.assertIn("price", mapping)
        self.assertIn("outlier_indices", mapping["price"])


class TestEnhancedPlotlyDashboard(unittest.TestCase):
    """Test enhanced Plotly dashboard data preparation."""

    def setUp(self):
        """Create sample financial data."""
        np.random.seed(42)
        n_samples = 150

        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Finance", "Healthcare"], n_samples),
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
                "last_price": np.random.uniform(10, 500, n_samples),
                "price_target": np.random.uniform(10, 500, n_samples),
                "market_cap": np.random.lognormal(20, 2, n_samples),
                "p_e": np.random.uniform(5, 50, n_samples),
                "roe": np.random.uniform(0, 0.30, n_samples),
                "mispricing_score": np.random.uniform(-30, 30, n_samples),
            }
        )

    def test_prepare_plotly_dashboard_data_returns_dict(self):
        """Test that function returns dictionary with Plotly-ready data."""
        from finance_ml.eval import prepare_plotly_dashboard_data

        result = prepare_plotly_dashboard_data(self.df)

        self.assertIsInstance(result, dict)
        self.assertIn("scatter_data", result)
        self.assertIn("histogram_data", result)
        self.assertIn("box_data", result)
        self.assertIn("heatmap_data", result)

    def test_scatter_plot_data_structure(self):
        """Test scatter plot data is properly formatted for Plotly."""
        from finance_ml.eval import prepare_plotly_dashboard_data

        result = prepare_plotly_dashboard_data(self.df)
        scatter = result["scatter_data"]

        self.assertIn("x", scatter)
        self.assertIn("y", scatter)
        self.assertIn("text", scatter)
        self.assertIn("color", scatter)
        self.assertIn("size", scatter)

    def test_histogram_data_by_sector(self):
        """Test histogram data grouped by sector."""
        from finance_ml.eval import prepare_plotly_dashboard_data

        result = prepare_plotly_dashboard_data(self.df)
        histogram = result["histogram_data"]

        self.assertIn("mispricing_by_sector", histogram)
        sector_hist = histogram["mispricing_by_sector"]
        self.assertIsInstance(sector_hist, list)
        self.assertGreater(len(sector_hist), 0)

    def test_box_plot_data_structure(self):
        """Test box plot data for sector comparisons."""
        from finance_ml.eval import prepare_plotly_dashboard_data

        result = prepare_plotly_dashboard_data(self.df)
        box = result["box_data"]

        self.assertIn("sector_comparisons", box)
        self.assertIn("region_comparisons", box)

    def test_heatmap_data_format(self):
        """Test heatmap data for correlation matrix."""
        from finance_ml.eval import prepare_plotly_dashboard_data

        result = prepare_plotly_dashboard_data(self.df)
        heatmap = result["heatmap_data"]

        self.assertIn("correlation_matrix", heatmap)
        corr_data = heatmap["correlation_matrix"]
        self.assertIn("z", corr_data)
        self.assertIn("x", corr_data)
        self.assertIn("y", corr_data)

    def test_time_series_data_preparation(self):
        """Test time-series data preparation if date column exists."""
        from finance_ml.eval import prepare_plotly_dashboard_data

        # Add date column
        dates = pd.date_range(start="2023-01-01", periods=len(self.df), freq="D")
        self.df["date"] = dates

        result = prepare_plotly_dashboard_data(self.df, include_timeseries=True)

        self.assertIn("timeseries_data", result)
        ts = result["timeseries_data"]
        self.assertIn("dates", ts)
        self.assertIn("values", ts)

    def test_sunburst_chart_data(self):
        """Test sunburst chart data for hierarchical visualization."""
        from finance_ml.eval import prepare_plotly_dashboard_data

        result = prepare_plotly_dashboard_data(self.df)

        self.assertIn("sunburst_data", result)
        sunburst = result["sunburst_data"]
        self.assertIn("labels", sunburst)
        self.assertIn("parents", sunburst)
        self.assertIn("values", sunburst)

    def test_treemap_data_structure(self):
        """Test treemap data for sector/region breakdown."""
        from finance_ml.eval import prepare_plotly_dashboard_data

        result = prepare_plotly_dashboard_data(self.df)

        self.assertIn("treemap_data", result)
        treemap = result["treemap_data"]
        self.assertIn("labels", treemap)
        self.assertIn("parents", treemap)
        self.assertIn("values", treemap)

    def test_custom_color_scales(self):
        """Test custom color scales for different visualizations."""
        from finance_ml.eval import prepare_plotly_dashboard_data

        result = prepare_plotly_dashboard_data(self.df, color_scheme="viridis")

        self.assertIn("color_scales", result)
        self.assertEqual(result["color_scales"]["default"], "viridis")


class TestEnhancedPDFReport(unittest.TestCase):
    """Test enhanced PDF report generation with Phase 9.2 integration."""

    def setUp(self):
        """Create sample data and temp directory."""
        np.random.seed(42)
        n_samples = 100

        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Finance", "Healthcare"], n_samples),
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
                "last_price": np.random.uniform(10, 500, n_samples),
                "price_target": np.random.uniform(10, 500, n_samples),
                "market_cap": np.random.lognormal(20, 2, n_samples),
                "p_e": np.random.uniform(5, 50, n_samples),
                "roe": np.random.uniform(0, 0.30, n_samples),
                "mispricing_score": np.random.uniform(-30, 30, n_samples),
            }
        )

        self.temp_dir = tempfile.mkdtemp()

    def test_generate_enhanced_pdf_report_creates_file(self):
        """Test that PDF file is created."""
        from finance_ml.eval import generate_enhanced_pdf_report

        output_path = Path(self.temp_dir) / "test_report.pdf"

        generate_enhanced_pdf_report(self.df, pdf_path=output_path, title="Test Financial Report")

        self.assertTrue(output_path.exists())

    def test_includes_financial_dashboard_section(self):
        """Test that financial dashboard metrics are included."""
        from finance_ml.eval import generate_enhanced_pdf_report

        output_path = Path(self.temp_dir) / "test_report.pdf"

        result = generate_enhanced_pdf_report(
            self.df, pdf_path=output_path, include_financial_dashboard=True
        )

        self.assertIn("sections", result)
        self.assertIn("financial_dashboard", result["sections"])

    def test_includes_data_quality_alerts(self):
        """Test that data quality alerts are included."""
        from finance_ml.eval import generate_enhanced_pdf_report

        output_path = Path(self.temp_dir) / "test_report.pdf"

        result = generate_enhanced_pdf_report(
            self.df, pdf_path=output_path, include_quality_alerts=True
        )

        self.assertIn("quality_alerts", result["sections"])

    def test_includes_hypothesis_testing_results(self):
        """Test that hypothesis testing results are included."""
        from finance_ml.eval import generate_enhanced_pdf_report

        output_path = Path(self.temp_dir) / "test_report.pdf"

        result = generate_enhanced_pdf_report(
            self.df, pdf_path=output_path, include_hypothesis_tests=True
        )

        self.assertIn("hypothesis_tests", result["sections"])

    def test_includes_charts_and_visualizations(self):
        """Test that charts are embedded in PDF."""
        from finance_ml.eval import generate_enhanced_pdf_report

        output_path = Path(self.temp_dir) / "test_report.pdf"

        result = generate_enhanced_pdf_report(self.df, pdf_path=output_path, include_charts=True)

        self.assertIn("charts", result["sections"])
        self.assertGreater(len(result["sections"]["charts"]), 0)

    def test_custom_template_support(self):
        """Test custom PDF template support."""
        from finance_ml.eval import generate_enhanced_pdf_report

        output_path = Path(self.temp_dir) / "test_report.pdf"

        result = generate_enhanced_pdf_report(self.df, pdf_path=output_path, template="modern")

        self.assertIn("template", result)
        self.assertEqual(result["template"], "modern")

    def test_multi_page_report_structure(self):
        """Test that multi-page reports are properly structured."""
        from finance_ml.eval import generate_enhanced_pdf_report

        output_path = Path(self.temp_dir) / "test_report.pdf"

        result = generate_enhanced_pdf_report(
            self.df,
            pdf_path=output_path,
            include_financial_dashboard=True,
            include_quality_alerts=True,
            include_hypothesis_tests=True,
            include_charts=True,
        )

        self.assertIn("page_count", result)
        self.assertGreater(result["page_count"], 1)

    def test_table_of_contents_generation(self):
        """Test that table of contents is generated."""
        from finance_ml.eval import generate_enhanced_pdf_report

        output_path = Path(self.temp_dir) / "test_report.pdf"

        result = generate_enhanced_pdf_report(self.df, pdf_path=output_path, include_toc=True)

        self.assertIn("table_of_contents", result)


if __name__ == "__main__":
    unittest.main()
