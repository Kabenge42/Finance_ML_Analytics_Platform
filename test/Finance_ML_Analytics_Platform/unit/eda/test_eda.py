import json
import tempfile
import unittest
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import finance_ml as mod
except Exception:
    mod = None


@unittest.skipIf(pd is None or mod is None, "pandas/sklearn not installed")
class TestEDARoot(unittest.TestCase):
    def test_simple_eda_outputs_summary(self):
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "sector": ["Tech", "Tech", "Energy", "Energy"],
                "region": ["US", "US", "EU", "EU"],
                "last_price": [10.0, 12.0, 8.0, 9.0],
                "price_target": [11.0, 13.0, 10.0, 10.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            mod.simple_eda(df, out_dir)
            summary_path = out_dir / "eda_summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for key in [
                "row_count",
                "column_count",
                "null_counts",
                "region_counts",
                "sector_counts",
                "basic_stats",
            ]:
                self.assertIn(key, summary)
            self.assertIn("last_price", summary["basic_stats"])
            self.assertIn("price_target", summary["basic_stats"])
            self.assertIn("mean", summary["basic_stats"]["last_price"])
            self.assertIn("std", summary["basic_stats"]["last_price"])


@unittest.skipIf(pd is None or mod is None, "pandas/sklearn not installed")
class TestEnhancedEDA(unittest.TestCase):
    """Phase 2: Enhanced EDA tests per IMPROVEMENT_PLAN.md - TDD approach"""

    def test_eda_with_visualizations_creates_distribution_plots(self):
        """Test that EDA creates distribution plots for numeric features when save_plots=True"""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Tech", "Energy", "Energy", "Finance", "Finance"],
                "region": ["US", "US", "EU", "EU", "APAC", "APAC"],
                "last_price": [10.0, 12.0, 8.0, 9.0, 15.0, 11.0],
                "price_target": [11.0, 13.0, 10.0, 10.0, 17.0, 12.0],
                "market_cap": [1e9, 1.2e9, 8e8, 9e8, 1.5e9, 1.1e9],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            # Call enhanced EDA with save_plots flag
            mod.simple_eda(df, out_dir, save_plots=True)

            # Check that distribution plot file is created
            dist_plot_path = out_dir / "eda_distributions.png"
            self.assertTrue(
                dist_plot_path.exists(),
                "Distribution plot PNG should be created when save_plots=True",
            )

    def test_eda_with_visualizations_creates_correlation_heatmap(self):
        """Test that EDA creates correlation heatmap when save_plots=True"""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Tech", "Energy", "Energy", "Finance", "Finance"],
                "last_price": [10.0, 12.0, 8.0, 9.0, 15.0, 11.0],
                "price_target": [11.0, 13.0, 10.0, 10.0, 17.0, 12.0],
                "market_cap": [1e9, 1.2e9, 8e8, 9e8, 1.5e9, 1.1e9],
                "revenue": [5e8, 6e8, 4e8, 4.5e8, 7.5e8, 5.5e8],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            mod.simple_eda(df, out_dir, save_plots=True)

            # Check that correlation heatmap is created
            corr_plot_path = out_dir / "eda_correlation.png"
            self.assertTrue(
                corr_plot_path.exists(),
                "Correlation heatmap PNG should be created when save_plots=True",
            )

    def test_eda_without_plots_flag_does_not_create_visualizations(self):
        """Test that when save_plots=False (default), no plot files are created"""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Tech", "Energy"],
                "last_price": [10.0, 12.0, 8.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            # Call without save_plots flag (should default to False)
            mod.simple_eda(df, out_dir)

            # Verify plot files are NOT created
            dist_plot_path = out_dir / "eda_distributions.png"
            corr_plot_path = out_dir / "eda_correlation.png"
            self.assertFalse(
                dist_plot_path.exists(),
                "Distribution plot should not be created when save_plots=False",
            )
            self.assertFalse(
                corr_plot_path.exists(),
                "Correlation plot should not be created when save_plots=False",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
