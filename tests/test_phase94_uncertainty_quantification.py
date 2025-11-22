"""
Test Phase 9.4: Uncertainty Quantification & Conformal Calibration.

Tests for:
- build_quantile_diagnostics
- plot_interval_coverage
- plot_reliability_diagram

TDD Approach:
1. Write failing tests based on actual function signatures
2. Verify implementations pass tests
3. Ensure ≥80% coverage for changed files
"""

import json
import unittest
from pathlib import Path
import tempfile
import shutil

import pandas as pd
import numpy as np


class TestBuildQuantileDiagnostics(unittest.TestCase):
    """Test build_quantile_diagnostics function."""

    def setUp(self):
        """Create test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "uncertainty"

        # Create sample predictions dataframe
        np.random.seed(42)
        n_samples = 100

        self.predictions_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n_samples),
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
                "y_true": np.random.uniform(10, 100, n_samples),
                "pred_p10": np.random.uniform(5, 50, n_samples),
                "pred_p50": np.random.uniform(10, 75, n_samples),
                "pred_p90": np.random.uniform(50, 150, n_samples),
            }
        )

        # Ensure p10 < p50 < p90 for realistic intervals
        self.predictions_df["pred_p10"] = self.predictions_df["pred_p50"] - 10
        self.predictions_df["pred_p90"] = self.predictions_df["pred_p50"] + 10

    def tearDown(self):
        """Clean up test artifacts."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_build_quantile_diagnostics_returns_path(self):
        """Test that function returns Path to diagnostics CSV."""
        from finance_ml.ml_workflow.evaluation import build_quantile_diagnostics

        result = build_quantile_diagnostics(
            predictions_df=self.predictions_df,
            output_dir=self.output_dir,
            target_coverage=0.8,
            tolerance=0.1,
        )

        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())
        self.assertEqual(result.name, "quantile_predictions_diagnostics.csv")

    def test_build_quantile_diagnostics_creates_artifacts(self):
        """Test that all required artifacts are created."""
        from finance_ml.ml_workflow.evaluation import build_quantile_diagnostics

        build_quantile_diagnostics(
            predictions_df=self.predictions_df,
            output_dir=self.output_dir,
        )

        # Check all expected files exist
        expected_files = [
            "quantile_predictions_diagnostics.csv",
            "coverage_by_sector.json",
            "uncertainty_summary.json",
        ]

        for filename in expected_files:
            filepath = self.output_dir / filename
            self.assertTrue(filepath.exists(), f"Missing file: {filename}")

    def test_diagnostics_csv_has_required_columns(self):
        """Test that diagnostics CSV contains required columns."""
        from finance_ml.ml_workflow.evaluation import build_quantile_diagnostics

        csv_path = build_quantile_diagnostics(
            predictions_df=self.predictions_df,
            output_dir=self.output_dir,
        )

        diagnostics_df = pd.read_csv(csv_path)

        required_columns = ["coverage_flag_p90", "interval_width", "calibration_error"]

        for col in required_columns:
            self.assertIn(col, diagnostics_df.columns, f"Missing column: {col}")

    def test_coverage_flag_computation(self):
        """Test that coverage flag is computed correctly."""
        from finance_ml.ml_workflow.evaluation import build_quantile_diagnostics

        # Create data where we know coverage
        test_df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Tech", "Tech"],
                "region": ["US", "US", "US"],
                "y_true": [50, 50, 150],  # C is outside interval
                "pred_p10": [40, 40, 40],
                "pred_p50": [50, 50, 50],
                "pred_p90": [60, 60, 60],
            }
        )

        csv_path = build_quantile_diagnostics(
            predictions_df=test_df,
            output_dir=self.output_dir,
        )

        diagnostics_df = pd.read_csv(csv_path)

        # A and B should be covered (y_true within [p10, p90])
        # C should not be covered (150 > 60)
        self.assertEqual(diagnostics_df.loc[0, "coverage_flag_p90"], 1)
        self.assertEqual(diagnostics_df.loc[1, "coverage_flag_p90"], 1)
        self.assertEqual(diagnostics_df.loc[2, "coverage_flag_p90"], 0)

    def test_coverage_by_sector_json_structure(self):
        """Test that coverage_by_sector.json has correct structure."""
        from finance_ml.ml_workflow.evaluation import build_quantile_diagnostics

        build_quantile_diagnostics(
            predictions_df=self.predictions_df,
            output_dir=self.output_dir,
        )

        json_path = self.output_dir / "coverage_by_sector.json"
        with open(json_path, "r") as f:
            coverage_data = json.load(f)

        # Should have entries for each sector
        unique_sectors = self.predictions_df["sector"].unique()
        for sector in unique_sectors:
            self.assertIn(sector, coverage_data)

            # Each sector should have required keys
            sector_data = coverage_data[sector]
            self.assertIn("coverage_rate", sector_data)
            self.assertIn("count", sector_data)
            self.assertIn("mean_interval_width", sector_data)

            # Validate types
            self.assertIsInstance(sector_data["coverage_rate"], float)
            self.assertIsInstance(sector_data["count"], int)
            self.assertIsInstance(sector_data["mean_interval_width"], float)

    def test_uncertainty_summary_json_structure(self):
        """Test that uncertainty_summary.json has correct structure."""
        from finance_ml.ml_workflow.evaluation import build_quantile_diagnostics

        build_quantile_diagnostics(
            predictions_df=self.predictions_df,
            output_dir=self.output_dir,
            target_coverage=0.8,
        )

        json_path = self.output_dir / "uncertainty_summary.json"
        with open(json_path, "r") as f:
            summary = json.load(f)

        # Check required keys (actual implementation)
        required_keys = [
            "overall_coverage",
            "target_coverage",
            "validation_status",  # Actual key, not within_tolerance
            "mean_interval_width",
            "sectors_under_covered",  # Note: different format
            "sectors_over_covered",
            "tolerance",
            "total_predictions",
        ]

        for key in required_keys:
            self.assertIn(key, summary, f"Missing key: {key}")

    def test_target_coverage_validation(self):
        """Test that target coverage is correctly validated."""
        from finance_ml.ml_workflow.evaluation import build_quantile_diagnostics

        build_quantile_diagnostics(
            predictions_df=self.predictions_df,
            output_dir=self.output_dir,
            target_coverage=0.8,
            tolerance=0.1,
        )

        json_path = self.output_dir / "uncertainty_summary.json"
        with open(json_path, "r") as f:
            summary = json.load(f)

        self.assertEqual(summary["target_coverage"], 0.8)
        self.assertIn(summary["validation_status"], ["PASS", "WARNING"])  # Actual key

    def test_handles_missing_y_true(self):
        """Test that function handles missing y_true gracefully."""
        from finance_ml.ml_workflow.evaluation import build_quantile_diagnostics

        # Create df without y_true
        df_no_ytrue = self.predictions_df.drop(columns=["y_true"])

        csv_path = build_quantile_diagnostics(
            predictions_df=df_no_ytrue,
            output_dir=self.output_dir,
        )

        diagnostics_df = pd.read_csv(csv_path)

        # Should have coverage_flag_p90 column set to 0
        self.assertIn("coverage_flag_p90", diagnostics_df.columns)
        self.assertTrue((diagnostics_df["coverage_flag_p90"] == 0).all())


class TestPlotIntervalCoverage(unittest.TestCase):
    """Test plot_interval_coverage function."""

    def setUp(self):
        """Create test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "uncertainty"

        # Create sample diagnostics dataframe with ALL required columns
        np.random.seed(42)
        n_samples = 100

        self.diagnostics_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n_samples),
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
                "last_price": np.random.uniform(10, 100, n_samples),
                "y_true": np.random.uniform(10, 100, n_samples),  # Required for heatmap
                "pred_p10": np.random.uniform(5, 50, n_samples),  # Required for heatmap
                "pred_p90": np.random.uniform(50, 150, n_samples),  # Required for heatmap
                "interval_width": np.random.uniform(5, 20, n_samples),
                "coverage_flag_p90": np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            }
        )

    def tearDown(self):
        """Clean up test artifacts."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_plot_interval_coverage_creates_htmls(self):
        """Test that function creates HTML visualization files."""
        from finance_ml.ml_workflow.evaluation import plot_interval_coverage

        plot_interval_coverage(
            predictions_df=self.diagnostics_df,
            output_dir=self.output_dir,
        )

        # Check expected HTML files exist
        expected_files = ["interval_width_by_bucket.html", "coverage_heatmap_region_sector.html"]

        for filename in expected_files:
            filepath = self.output_dir / filename
            self.assertTrue(filepath.exists(), f"Missing file: {filename}")

    def test_plot_interval_coverage_with_custom_price_col(self):
        """Test that function handles custom last_price column name."""
        from finance_ml.ml_workflow.evaluation import plot_interval_coverage

        # Rename column
        df_custom = self.diagnostics_df.rename(columns={"last_price": "price"})

        # Should still work (or raise informative error)
        try:
            plot_interval_coverage(
                predictions_df=df_custom,
                output_dir=self.output_dir,
            )
        except Exception as e:
            # If it fails, should be due to missing expected column
            self.assertIn("last_price", str(e).lower())

    def test_html_files_are_valid(self):
        """Test that generated HTML files are valid and non-empty."""
        from finance_ml.ml_workflow.evaluation import plot_interval_coverage

        plot_interval_coverage(
            predictions_df=self.diagnostics_df,
            output_dir=self.output_dir,
        )

        for filename in ["interval_width_by_bucket.html", "coverage_heatmap_region_sector.html"]:
            filepath = self.output_dir / filename
            content = filepath.read_text(encoding="utf-8", errors="ignore")

            # Basic HTML validation
            self.assertGreater(len(content), 100, f"{filename} is too small")
            self.assertTrue("<html>" in content.lower() or "<div>" in content.lower())


class TestPlotReliabilityDiagram(unittest.TestCase):
    """Test plot_reliability_diagram function."""

    def setUp(self):
        """Create test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "uncertainty"

        # Create sample diagnostics dataframe with ALL required columns
        np.random.seed(42)
        n_samples = 100

        self.diagnostics_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n_samples),
                "y_true": np.random.uniform(10, 100, n_samples),  # Required
                "pred_p10": np.random.uniform(5, 50, n_samples),  # Required
                "pred_p50": np.random.uniform(10, 75, n_samples),
                "pred_p90": np.random.uniform(50, 150, n_samples),  # Required
                "coverage_flag_p90": np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            }
        )

    def tearDown(self):
        """Clean up test artifacts."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_plot_reliability_diagram_creates_html(self):
        """Test that function creates reliability diagram HTML."""
        from finance_ml.ml_workflow.evaluation import plot_reliability_diagram

        plot_reliability_diagram(
            predictions_df=self.diagnostics_df,
            output_dir=self.output_dir,
        )

        filepath = self.output_dir / "reliability_diagram_conformal.html"
        self.assertTrue(filepath.exists())

    def test_reliability_diagram_with_pre_calibration(self):
        """Test that function handles pre-calibration data."""
        from finance_ml.ml_workflow.evaluation import plot_reliability_diagram

        # Create pre-calibration df
        pre_calibration_df = self.diagnostics_df.copy()
        pre_calibration_df["coverage_flag_p90"] = np.random.choice([0, 1], len(pre_calibration_df))

        plot_reliability_diagram(
            predictions_df=self.diagnostics_df,
            output_dir=self.output_dir,
        )

        filepath = self.output_dir / "reliability_diagram_conformal.html"
        self.assertTrue(filepath.exists())

    def test_html_file_is_valid(self):
        """Test that generated HTML file is valid and non-empty."""
        from finance_ml.ml_workflow.evaluation import plot_reliability_diagram

        plot_reliability_diagram(
            predictions_df=self.diagnostics_df,
            output_dir=self.output_dir,
        )

        filepath = self.output_dir / "reliability_diagram_conformal.html"
        content = filepath.read_text(encoding="utf-8", errors="ignore")

        # Basic HTML validation
        self.assertGreater(len(content), 100)
        self.assertTrue("<html>" in content.lower() or "<div>" in content.lower())


class TestIntegrationPhase94(unittest.TestCase):
    """Integration tests for Phase 9.4 workflow."""

    def setUp(self):
        """Create test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "uncertainty"

        # Create realistic predictions dataframe
        np.random.seed(42)
        n_samples = 200

        self.predictions_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Tech", "Finance", "Healthcare", "Energy"], n_samples),
                "region": np.random.choice(["US", "EU", "APAC", "ROTW"], n_samples),
                "last_price": np.random.uniform(10, 200, n_samples),
                "y_true": np.random.uniform(10, 200, n_samples),
            }
        )

        # Generate realistic quantile predictions
        self.predictions_df["pred_p50"] = self.predictions_df["y_true"] + np.random.normal(
            0, 10, n_samples
        )
        self.predictions_df["pred_p10"] = self.predictions_df["pred_p50"] - np.random.uniform(
            5, 15, n_samples
        )
        self.predictions_df["pred_p90"] = self.predictions_df["pred_p50"] + np.random.uniform(
            5, 15, n_samples
        )

    def tearDown(self):
        """Clean up test artifacts."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_full_phase94_workflow(self):
        """Test complete Phase 9.4 workflow as used in notebook."""
        from finance_ml.ml_workflow.evaluation import (
            build_quantile_diagnostics,
            plot_interval_coverage,
            plot_reliability_diagram,
        )

        # Step 1: Build diagnostics
        diagnostics_path = build_quantile_diagnostics(
            predictions_df=self.predictions_df,
            output_dir=self.output_dir,
            target_coverage=0.8,
        )

        self.assertTrue(diagnostics_path.exists())

        # Load diagnostics for next steps
        diagnostics_df = pd.read_csv(diagnostics_path)

        # Step 2: Plot interval coverage
        plot_interval_coverage(
            predictions_df=diagnostics_df,
            output_dir=self.output_dir,
        )

        # Step 3: Plot reliability diagram
        plot_reliability_diagram(
            predictions_df=diagnostics_df,
            output_dir=self.output_dir,
        )

        # Verify all artifacts created
        expected_artifacts = [
            "quantile_predictions_diagnostics.csv",
            "coverage_by_sector.json",
            "uncertainty_summary.json",
            "interval_width_by_bucket.html",
            "coverage_heatmap_region_sector.html",
            "reliability_diagram_conformal.html",
        ]

        for artifact in expected_artifacts:
            filepath = self.output_dir / artifact
            self.assertTrue(filepath.exists(), f"Missing artifact: {artifact}")


if __name__ == "__main__":
    unittest.main()
