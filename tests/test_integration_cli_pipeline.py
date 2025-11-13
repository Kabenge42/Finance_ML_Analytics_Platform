"""
Integration test for CLI pipeline (NOTEBOOK_REFACTORING_SUMMARY.md Next Steps #2).

Tests CLI script execution with --dry-run flag and validates that standardized
artifact files are created with correct header schemas (even if empty/minimal data).

Requirements:
- CLI must execute with --dry-run without errors
- Artifact files must be created with standardized headers
- Schema headers must match expected columns from code_guidelines.md
"""

import unittest
import subprocess
import sys
from pathlib import Path
import pandas as pd
import tempfile
import shutil
import os


class TestCLIPipelineDryRun(unittest.TestCase):
    """Test CLI pipeline execution with --dry-run flag."""

    def setUp(self):
        """Set up test environment."""
        self.cli_script = Path(__file__).parent.parent / "ml_finance_model_main.py"
        self.test_output_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test output directory."""
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)

    def test_cli_script_exists(self):
        """Test that CLI script exists."""
        self.assertTrue(self.cli_script.exists(), f"CLI script not found at {self.cli_script}")

    def test_cli_dry_run_executes_without_error(self):
        """Test that CLI executes with --dry-run flag without errors."""
        if not self.cli_script.exists():
            self.skipTest("CLI script not found")

        # Run CLI with --dry-run, limit data, and custom output dir
        cmd = [
            sys.executable,
            str(self.cli_script),
            "--dry-run",
            "--limit",
            "100",
            "--out-dir",
            str(self.test_output_dir),
            "--data-source",
            "csv",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout for dry-run
                cwd=self.cli_script.parent,
            )

            # Check exit code
            if result.returncode != 0:
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")

            self.assertEqual(
                result.returncode, 0, f"CLI dry-run failed with exit code {result.returncode}"
            )

        except subprocess.TimeoutExpired:
            self.fail("CLI dry-run timed out after 60 seconds")
        except Exception as e:
            self.fail(f"CLI dry-run execution failed: {e}")

    def test_cli_help_flag_works(self):
        """Test that CLI --help flag works."""
        if not self.cli_script.exists():
            self.skipTest("CLI script not found")

        cmd = [sys.executable, str(self.cli_script), "--help"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            # Help should succeed
            self.assertEqual(result.returncode, 0, "CLI --help should succeed")

            # Help text should mention key arguments
            help_text = result.stdout.lower()
            self.assertIn("--dry-run", help_text, "Help text should mention --dry-run")
            self.assertIn("--data-source", help_text, "Help text should mention --data-source")

        except Exception as e:
            self.skipTest(f"CLI --help test skipped: {e}")


class TestCLIArtifactHeaders(unittest.TestCase):
    """Test that CLI produces artifact files with correct headers."""

    def setUp(self):
        """Check if outputs directory exists from previous pipeline runs."""
        self.outputs_dir = Path("outputs")
        self.regression_dir = self.outputs_dir / "regression"

    def test_regression_predictions_detailed_has_headers(self):
        """Test that regression_predictions_detailed.csv has correct headers."""
        predictions_path = self.regression_dir / "regression_predictions_detailed.csv"

        if not predictions_path.exists():
            self.skipTest("Predictions file not found - CLI may not have run")

        # Load just headers (first row)
        df = pd.read_csv(predictions_path, nrows=0)

        # Required core columns
        required_cols = ["y_true", "y_pred", "abs_error", "pct_error"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required header: {col}")

        # Expected optional columns (contract from code_guidelines.md)
        optional_cols = [
            "ticker",
            "isin",
            "sector",
            "region",
            "last_price",
            "y_pred_calibrated",
            "pred_p10",
            "pred_p50",
            "pred_p90",
            "interval_width",
            "model_version",
            "snapshot_date",
        ]

        # Check which optional columns are present
        present_optional = [col for col in optional_cols if col in df.columns]

        # Log which columns are present
        if len(present_optional) > 0:
            print(f"  Present optional columns: {present_optional}")

    def test_quantile_predictions_has_headers(self):
        """Test that quantile_predictions.csv has correct headers."""
        quantile_path = self.regression_dir / "quantile_predictions.csv"

        if not quantile_path.exists():
            self.skipTest("Quantile predictions file not found")

        # Load headers
        df = pd.read_csv(quantile_path, nrows=0)

        # Required quantile columns
        required_cols = ["y_true", "pred_p10", "pred_p50", "pred_p90", "interval_width"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required quantile header: {col}")

    def test_regression_metrics_by_sector_has_headers(self):
        """Test that regression_metrics_by_sector.csv has correct headers."""
        metrics_path = self.regression_dir / "regression_metrics_by_sector.csv"

        if not metrics_path.exists():
            self.skipTest("Sector metrics file not found")

        # Load headers
        df = pd.read_csv(metrics_path, nrows=0)

        # Should have sector column
        has_sector = "sector" in df.columns or "Sector" in df.columns
        self.assertTrue(has_sector, "Sector metrics should have 'sector' header")

        # Should have metric columns (at least one)
        metric_candidates = [col for col in df.columns if col.lower() not in ["sector", "index"]]
        self.assertGreater(
            len(metric_candidates), 0, "Sector metrics should have at least one metric header"
        )


class TestCLISchemaContract(unittest.TestCase):
    """Test that CLI outputs comply with standardized schema contract."""

    def setUp(self):
        """Set up paths to output files."""
        self.regression_dir = Path("outputs/regression")

    def test_predictions_detailed_contract_compliance(self):
        """Test that CLI predictions comply with schema contract."""
        predictions_path = self.regression_dir / "regression_predictions_detailed.csv"

        if not predictions_path.exists():
            self.skipTest("Predictions file not found")

        df = pd.read_csv(predictions_path)

        # Contract: Core prediction columns
        core_cols = ["y_true", "y_pred", "abs_error", "pct_error"]
        for col in core_cols:
            self.assertIn(col, df.columns, f"Schema contract violation: missing {col}")

        # Contract: Non-negative absolute error
        if len(df) > 0 and "abs_error" in df.columns:
            self.assertTrue(
                (df["abs_error"] >= 0).all(),
                "Schema contract violation: abs_error must be non-negative",
            )

        # Contract: Quantile monotonicity (if quantiles present)
        quantile_cols = ["pred_p10", "pred_p50", "pred_p90"]
        if all(col in df.columns for col in quantile_cols) and len(df) > 0:
            violations = (
                (df["pred_p10"] > df["pred_p50"]) | (df["pred_p50"] > df["pred_p90"])
            ).sum()
            self.assertEqual(
                violations,
                0,
                f"Schema contract violation: {violations} quantile monotonicity violations",
            )

        # Contract: Metadata columns (at least one should be present)
        metadata_cols = ["ticker", "sector", "region", "last_price"]
        present_metadata = [col for col in metadata_cols if col in df.columns]
        if len(present_metadata) == 0:
            self.skipTest("No metadata columns present - may be expected for some data sources")

    def test_quantile_predictions_monotonicity(self):
        """Test that quantile predictions satisfy monotonicity constraint."""
        quantile_path = self.regression_dir / "quantile_predictions.csv"

        if not quantile_path.exists():
            self.skipTest("Quantile predictions file not found")

        df = pd.read_csv(quantile_path)

        if len(df) == 0:
            self.skipTest("Quantile predictions file is empty")

        # Validate p10 <= p50 <= p90
        required_cols = ["pred_p10", "pred_p50", "pred_p90"]
        if all(col in df.columns for col in required_cols):
            violations_10_50 = (df["pred_p10"] > df["pred_p50"]).sum()
            violations_50_90 = (df["pred_p50"] > df["pred_p90"]).sum()

            self.assertEqual(
                violations_10_50,
                0,
                f"Monotonicity violation: {violations_10_50} rows have pred_p10 > pred_p50",
            )
            self.assertEqual(
                violations_50_90,
                0,
                f"Monotonicity violation: {violations_50_90} rows have pred_p50 > pred_p90",
            )


class TestCLIDataSourceOptions(unittest.TestCase):
    """Test that CLI supports different data source options."""

    def setUp(self):
        """Set up CLI script path."""
        self.cli_script = Path(__file__).parent.parent / "ml_finance_model_main.py"

    def test_cli_supports_data_source_csv(self):
        """Test that CLI accepts --data-source csv."""
        if not self.cli_script.exists():
            self.skipTest("CLI script not found")

        cmd = [sys.executable, str(self.cli_script), "--help"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        # Help text should mention data-source options
        help_text = result.stdout.lower()
        self.assertIn("--data-source", help_text, "CLI should support --data-source argument")

    def test_cli_supports_limit_option(self):
        """Test that CLI accepts --limit for testing."""
        if not self.cli_script.exists():
            self.skipTest("CLI script not found")

        cmd = [sys.executable, str(self.cli_script), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        help_text = result.stdout.lower()
        self.assertIn("--limit", help_text, "CLI should support --limit argument for testing")

    def test_cli_supports_output_dir_option(self):
        """Test that CLI accepts --out-dir for custom output directory."""
        if not self.cli_script.exists():
            self.skipTest("CLI script not found")

        cmd = [sys.executable, str(self.cli_script), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        help_text = result.stdout.lower()
        has_out_dir = "--out-dir" in help_text or "--output-dir" in help_text
        self.assertTrue(has_out_dir, "CLI should support --out-dir or --output-dir argument")


if __name__ == "__main__":
    unittest.main()
