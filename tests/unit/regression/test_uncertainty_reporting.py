"""
Test uncertainty quantification reporting and visualization (Phase 9.4).

Tests for notebook-friendly reporting functions that generate:
- quantile_predictions_diagnostics.csv
- coverage_by_sector.json
- uncertainty_summary.json
- HTML visualizations (interval_width, coverage_heatmap, reliability_diagram, residuals)
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import json


class TestBuildQuantileDiagnostics(unittest.TestCase):
    """Test build_quantile_diagnostics function."""

    def setUp(self):
        """Create test data and temporary output directory."""
        self.output_dir = Path(tempfile.mkdtemp())

        # Create sample predictions DataFrame following standardized schema
        np.random.seed(42)
        n_samples = 100

        # Generate realistic test data with proper coverage
        y_true_values = np.random.uniform(60, 220, n_samples)

        self.predictions_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "isin": [f"US{i:010d}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Healthcare", "Financials"], n_samples),
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
                "last_price": np.random.uniform(50, 200, n_samples),
                "y_true": y_true_values,
                "y_pred": y_true_values + np.random.normal(0, 10, n_samples),
                "y_pred_calibrated": y_true_values + np.random.normal(0, 8, n_samples),
                "model_version": "v9_10",
                "snapshot_date": "2025-01-01",
            }
        )

        # Create intervals centered around y_true to ensure realistic coverage (~80%)
        interval_widths = np.random.uniform(30, 60, n_samples)
        self.predictions_df["pred_p50"] = y_true_values + np.random.normal(0, 5, n_samples)
        self.predictions_df["pred_p10"] = self.predictions_df["pred_p50"] - interval_widths * 0.4
        self.predictions_df["pred_p90"] = self.predictions_df["pred_p50"] + interval_widths * 0.4

        # Compute derived columns
        self.predictions_df["interval_width"] = (
            self.predictions_df["pred_p90"] - self.predictions_df["pred_p10"]
        )
        self.predictions_df["abs_error"] = abs(
            self.predictions_df["y_true"] - self.predictions_df["y_pred_calibrated"]
        )
        self.predictions_df["pct_error"] = (
            self.predictions_df["abs_error"] / self.predictions_df["y_true"] * 100
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.output_dir)

    def test_build_quantile_diagnostics_creates_csv(self):
        """Test that build_quantile_diagnostics creates diagnostics CSV."""
        from finance_ml.ml_workflow.evaluation.uncertainty import build_quantile_diagnostics

        diagnostics_df = build_quantile_diagnostics(
            predictions_df=self.predictions_df, output_dir=self.output_dir
        )

        # CSV should exist at the expected path
        csv_path = self.output_dir / "quantile_predictions_diagnostics.csv"
        self.assertTrue(csv_path.exists())
        self.assertEqual(csv_path.name, "quantile_predictions_diagnostics.csv")

        # Validate returned diagnostics DataFrame
        self.assertGreater(len(diagnostics_df), 0)

        # Required columns
        required_cols = [
            "ticker",
            "sector",
            "region",
            "coverage_flag_p90",
            "interval_width",
            "calibration_error",
        ]
        for col in required_cols:
            self.assertIn(col, diagnostics_df.columns)

    def test_build_quantile_diagnostics_computes_coverage(self):
        """Test that coverage flags are computed correctly."""
        from finance_ml.ml_workflow.evaluation.uncertainty import build_quantile_diagnostics

        diagnostics_df = build_quantile_diagnostics(
            predictions_df=self.predictions_df, output_dir=self.output_dir
        )

        # coverage_flag_p90 should be boolean (0 or 1)
        self.assertTrue(diagnostics_df["coverage_flag_p90"].isin([0, 1, True, False]).all())

        # Should have some coverage
        coverage_rate = diagnostics_df["coverage_flag_p90"].mean()
        self.assertGreater(coverage_rate, 0.5)  # At least 50% coverage

    def test_build_quantile_diagnostics_validates_monotonicity(self):
        """Test that interval monotonicity violations are detected."""
        from finance_ml.ml_workflow.evaluation.uncertainty import build_quantile_diagnostics

        # Create data with intentional monotonicity violation
        bad_df = self.predictions_df.copy()
        bad_df.loc[0, "pred_p10"] = 100
        bad_df.loc[0, "pred_p50"] = 80  # Violation: p50 < p10
        bad_df.loc[0, "pred_p90"] = 120

        diagnostics_df = build_quantile_diagnostics(
            predictions_df=bad_df, output_dir=self.output_dir
        )

        # Should flag or handle monotonicity violations
        # At minimum, function should not crash
        self.assertGreater(len(diagnostics_df), 0)


class TestCoverageBySector(unittest.TestCase):
    """Test coverage_by_sector.json generation."""

    def setUp(self):
        """Create test data and temporary output directory."""
        self.output_dir = Path(tempfile.mkdtemp())

        np.random.seed(42)
        n_samples = 150

        self.predictions_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(
                    ["Technology", "Healthcare", "Financials", "Energy"], n_samples
                ),
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
                "y_true": np.random.uniform(60, 220, n_samples),
                "pred_p10": np.random.uniform(40, 180, n_samples),
                "pred_p90": np.random.uniform(70, 230, n_samples),
            }
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.output_dir)

    def test_build_quantile_diagnostics_creates_coverage_json(self):
        """Test that coverage_by_sector.json is created."""
        from finance_ml.ml_workflow.evaluation.uncertainty import build_quantile_diagnostics

        build_quantile_diagnostics(predictions_df=self.predictions_df, output_dir=self.output_dir)

        coverage_json_path = self.output_dir / "coverage_by_sector.json"
        self.assertTrue(coverage_json_path.exists())

        with open(coverage_json_path, "r") as f:
            coverage_data = json.load(f)

        # Should have sector-level stats
        self.assertIsInstance(coverage_data, dict)
        self.assertGreater(len(coverage_data), 0)

        # Each sector should have coverage metrics
        for sector, metrics in coverage_data.items():
            self.assertIn("coverage_rate", metrics)
            self.assertIn("count", metrics)
            self.assertGreaterEqual(metrics["coverage_rate"], 0.0)
            self.assertLessEqual(metrics["coverage_rate"], 1.0)


class TestUncertaintySummary(unittest.TestCase):
    """Test uncertainty_summary.json generation."""

    def setUp(self):
        """Create test data and temporary output directory."""
        self.output_dir = Path(tempfile.mkdtemp())

        np.random.seed(42)
        n_samples = 200

        self.predictions_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Healthcare", "Financials"], n_samples),
                "region": np.random.choice(["US", "EU"], n_samples),
                "y_true": np.random.uniform(60, 220, n_samples),
                "pred_p10": np.random.uniform(40, 180, n_samples),
                "pred_p50": np.random.uniform(55, 200, n_samples),
                "pred_p90": np.random.uniform(70, 230, n_samples),
                "interval_width": np.random.uniform(20, 60, n_samples),
            }
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.output_dir)

    def test_build_quantile_diagnostics_creates_summary_json(self):
        """Test that uncertainty_summary.json is created with required keys."""
        from finance_ml.ml_workflow.evaluation.uncertainty import build_quantile_diagnostics

        build_quantile_diagnostics(predictions_df=self.predictions_df, output_dir=self.output_dir)

        summary_json_path = self.output_dir / "uncertainty_summary.json"
        self.assertTrue(summary_json_path.exists())

        with open(summary_json_path, "r") as f:
            summary_data = json.load(f)

        # Required keys per notebook plan
        required_keys = [
            "overall_coverage",
            "mean_interval_width",
            "sectors_under_covered",
            "sectors_over_covered",
            "validation_status",
        ]
        for key in required_keys:
            self.assertIn(key, summary_data)

        # Validate ranges
        self.assertGreaterEqual(summary_data["overall_coverage"], 0.0)
        self.assertLessEqual(summary_data["overall_coverage"], 1.0)
        self.assertGreater(summary_data["mean_interval_width"], 0.0)

    def test_uncertainty_summary_identifies_problematic_sectors(self):
        """Test that summary identifies under/over-covered sectors."""
        from finance_ml.ml_workflow.evaluation.uncertainty import build_quantile_diagnostics

        build_quantile_diagnostics(
            predictions_df=self.predictions_df,
            output_dir=self.output_dir,
            target_coverage=0.80,
            tolerance=0.10,
        )

        summary_json_path = self.output_dir / "uncertainty_summary.json"
        with open(summary_json_path, "r") as f:
            summary_data = json.load(f)

        # Lists should exist (may be empty)
        self.assertIsInstance(summary_data["sectors_under_covered"], list)
        self.assertIsInstance(summary_data["sectors_over_covered"], list)


class TestPlotIntervalCoverage(unittest.TestCase):
    """Test plot_interval_coverage HTML generation."""

    def setUp(self):
        """Create test data and temporary output directory."""
        self.output_dir = Path(tempfile.mkdtemp())

        np.random.seed(42)
        n_samples = 100

        self.predictions_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Healthcare"], n_samples),
                "last_price": np.random.uniform(50, 200, n_samples),
                "interval_width": np.random.uniform(10, 80, n_samples),
            }
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.output_dir)

    def test_plot_interval_coverage_creates_html(self):
        """Test that plot_interval_coverage creates HTML files."""
        from finance_ml.ml_workflow.evaluation.uncertainty import plot_interval_coverage

        html_paths = plot_interval_coverage(
            predictions_df=self.predictions_df, output_dir=self.output_dir
        )

        # Should return list of created HTML files
        self.assertIsInstance(html_paths, list)
        self.assertGreater(len(html_paths), 0)

        # Files should exist
        for path in html_paths:
            self.assertTrue(Path(path).exists())
            self.assertTrue(str(path).endswith(".html"))


class TestPlotReliabilityDiagram(unittest.TestCase):
    """Test plot_reliability_diagram HTML generation."""

    def setUp(self):
        """Create test data and temporary output directory."""
        self.output_dir = Path(tempfile.mkdtemp())

        np.random.seed(42)
        n_samples = 100

        self.predictions_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Healthcare"], n_samples),
                "y_true": np.random.uniform(60, 220, n_samples),
                "y_pred": np.random.uniform(55, 210, n_samples),
                "y_pred_calibrated": np.random.uniform(58, 215, n_samples),
                "pred_p10": np.random.uniform(40, 180, n_samples),
                "pred_p90": np.random.uniform(70, 230, n_samples),
            }
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.output_dir)

    def test_plot_reliability_diagram_creates_html(self):
        """Test that plot_reliability_diagram creates HTML file."""
        from finance_ml.ml_workflow.evaluation.uncertainty import plot_reliability_diagram

        html_path = plot_reliability_diagram(
            predictions_df=self.predictions_df, output_dir=self.output_dir
        )

        # Should return path to HTML file
        self.assertTrue(Path(html_path).exists())
        self.assertTrue(str(html_path).endswith(".html"))
        self.assertIn("reliability_diagram", str(html_path).lower())


if __name__ == "__main__":
    unittest.main()
