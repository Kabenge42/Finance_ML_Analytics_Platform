import unittest
import tempfile
import shutil
from pathlib import Path
import json

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None

try:
    import finance_ml as mod
except Exception:
    mod = None


@unittest.skipIf(
    pd is None or mod is None or np is None, "pandas/numpy or finance_ml not installed"
)
class TestImputationReport(unittest.TestCase):
    """Tests for generate_imputation_report() function - Phase 1: Interactive Dashboards"""

    def setUp(self):
        """Create temporary directory and sample data for tests"""
        self.test_dir = tempfile.mkdtemp()

        # Create sample dataframe with missing values (before imputation)
        self.df_before = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "price": [10.0, np.nan, 15.0, np.nan, 20.0],
                "volume": [1000, 2000, np.nan, 3000, np.nan],
                "pe_ratio": [15.5, np.nan, 18.2, np.nan, 22.1],
                "sector": ["Tech", "Tech", "Finance", "Energy", "Finance"],
            }
        )

        # Create sample dataframe after imputation
        self.df_after = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "price": [10.0, 12.5, 15.0, 15.0, 20.0],  # NaNs filled
                "volume": [1000, 2000, 2000, 3000, 2000],  # NaNs filled
                "pe_ratio": [15.5, 18.6, 18.2, 18.6, 22.1],  # NaNs filled
                "sector": ["Tech", "Tech", "Finance", "Energy", "Finance"],
            }
        )

        # Sample imputation stats
        self.imputation_stats = {
            "price": {"method": "median", "nans_filled": 2},
            "volume": {"method": "mean", "nans_filled": 2},
            "pe_ratio": {"method": "knn", "nans_filled": 2},
        }

    def tearDown(self):
        """Clean up temporary directory"""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_generate_imputation_report_returns_dict(self):
        """Test that generate_imputation_report returns a dictionary with expected keys"""
        report = mod.generate_imputation_report(
            imputation_stats=self.imputation_stats,
            df_before=self.df_before,
            df_after=self.df_after,
            output_dir=Path(self.test_dir),
        )

        # Check return type
        self.assertIsInstance(report, dict)

        # Check required keys
        self.assertIn("timestamp", report)
        self.assertIn("columns_imputed", report)
        self.assertIn("methods_used", report)
        self.assertIn("total_nans_filled", report)
        self.assertIn("emergency_fallbacks", report)
        self.assertIn("visualizations", report)

    def test_generate_imputation_report_tracks_columns(self):
        """Test that report correctly tracks which columns were imputed"""
        report = mod.generate_imputation_report(
            imputation_stats=self.imputation_stats,
            df_before=self.df_before,
            df_after=self.df_after,
            output_dir=Path(self.test_dir),
        )

        # Should track 3 columns (price, volume, pe_ratio)
        self.assertIsInstance(report["columns_imputed"], list)
        self.assertGreater(len(report["columns_imputed"]), 0)

        # Each entry should have column, nans_filled, method, fill_rate
        for col_info in report["columns_imputed"]:
            self.assertIn("column", col_info)
            self.assertIn("nans_filled", col_info)
            self.assertIn("method", col_info)
            self.assertIn("fill_rate", col_info)

    def test_generate_imputation_report_counts_nans(self):
        """Test that total NaNs filled is calculated correctly"""
        report = mod.generate_imputation_report(
            imputation_stats=self.imputation_stats,
            df_before=self.df_before,
            df_after=self.df_after,
            output_dir=Path(self.test_dir),
        )

        # We have 2 NaNs in price, 2 in volume, 2 in pe_ratio = 6 total
        self.assertEqual(report["total_nans_filled"], 6)

    def test_generate_imputation_report_tracks_methods(self):
        """Test that imputation methods are tracked"""
        report = mod.generate_imputation_report(
            imputation_stats=self.imputation_stats,
            df_before=self.df_before,
            df_after=self.df_after,
            output_dir=Path(self.test_dir),
        )

        # Should have dict of methods used
        self.assertIsInstance(report["methods_used"], dict)
        self.assertGreater(len(report["methods_used"]), 0)

    def test_generate_imputation_report_creates_json_file(self):
        """Test that JSON report file is created"""
        report = mod.generate_imputation_report(
            imputation_stats=self.imputation_stats,
            df_before=self.df_before,
            df_after=self.df_after,
            output_dir=Path(self.test_dir),
        )

        # Check that JSON file exists
        json_path = Path(self.test_dir) / "imputation_report.json"
        self.assertTrue(json_path.exists(), "imputation_report.json was not created")

        # Verify it's valid JSON
        with open(json_path, "r") as f:
            loaded_report = json.load(f)
        self.assertIsInstance(loaded_report, dict)
        self.assertIn("timestamp", loaded_report)

    def test_generate_imputation_report_creates_visualizations(self):
        """Test that visualization files are created"""
        report = mod.generate_imputation_report(
            imputation_stats=self.imputation_stats,
            df_before=self.df_before,
            df_after=self.df_after,
            output_dir=Path(self.test_dir),
        )

        # Should have list of visualization paths
        self.assertIsInstance(report["visualizations"], list)

        # At least heatmap should be created if matplotlib is available
        # (may be empty if matplotlib not installed)
        if len(report["visualizations"]) > 0:
            for viz_path in report["visualizations"]:
                self.assertTrue(
                    Path(viz_path).exists(), f"Visualization file not found: {viz_path}"
                )

    def test_generate_imputation_report_handles_no_imputation(self):
        """Test that report handles case where no imputation was needed"""
        # Create dataframes with no NaNs
        df_complete = pd.DataFrame(
            {"ticker": ["A", "B", "C"], "price": [10.0, 20.0, 30.0], "volume": [1000, 2000, 3000]}
        )

        report = mod.generate_imputation_report(
            imputation_stats={},
            df_before=df_complete,
            df_after=df_complete,
            output_dir=Path(self.test_dir),
        )

        # Should still return valid report
        self.assertIsInstance(report, dict)
        self.assertEqual(report["total_nans_filled"], 0)
        self.assertEqual(len(report["columns_imputed"]), 0)


if __name__ == "__main__":
    unittest.main()
