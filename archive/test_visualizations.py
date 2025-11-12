import tempfile
import unittest
from pathlib import Path

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


@unittest.skipIf(pd is None or mod is None, "pandas not installed")
class TestVisualizationFunctions(unittest.TestCase):
    def small_df(self):
        """Create small test dataframe with sector and prediction data"""
        rows = []
        sectors = ["Tech", "Energy", "Healthcare", "Finance"]
        for i, sector in enumerate(sectors):
            for j in range(20):
                rows.append(
                    {
                        "ticker": f"{sector[0]}{j}",
                        "sector": sector,
                        "region": "US" if i % 2 == 0 else "EU",
                        "last_price": float(10 + j % 5),
                        "price_target": float(11 + (j % 5)),
                        "predicted_target": float(10.5 + (j % 5)),
                        "mispricing_score": float((j % 5) / 100),
                    }
                )
        return pd.DataFrame(rows)

    def test_create_sector_heatmap_returns_figure(self):
        """Test that sector heatmap creation returns a figure object"""
        df = self.small_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "sector_heatmap.png"
            fig = mod.create_sector_heatmap(df, out_path=out_path)

            self.assertIsNotNone(fig)
            # Check file was created
            self.assertTrue(out_path.exists())

    def test_create_sector_heatmap_handles_missing_columns(self):
        """Test that heatmap gracefully handles missing columns"""
        df = pd.DataFrame({"ticker": ["A", "B"], "sector": ["Tech", "Energy"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "sector_heatmap.png"
            # Should return None when required metric column is missing (no exception)
            fig = mod.create_sector_heatmap(df, out_path=out_path)
            # Function returns None when required columns are missing
            self.assertIsNone(fig)

    def test_create_interactive_prediction_plot_returns_figure(self):
        """Test that interactive plot creation returns a plotly figure"""
        df = self.small_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "interactive_plot.html"
            fig = mod.create_interactive_prediction_plot(df, out_path=out_path)

            self.assertIsNotNone(fig)
            # Check HTML file was created
            self.assertTrue(out_path.exists())

    def test_interactive_plot_includes_scatter_data(self):
        """Test that interactive plot contains scatter plot data"""
        df = self.small_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "interactive_plot.html"
            fig = mod.create_interactive_prediction_plot(df, out_path=out_path)

            # Check figure has data traces
            self.assertTrue(hasattr(fig, "data"))
            self.assertGreater(len(fig.data), 0)

    def test_create_region_sector_heatmap_by_metric(self):
        """Test creating heatmap showing metric across region and sector"""
        df = self.small_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "region_sector_heatmap.png"
            fig = mod.create_region_sector_heatmap(df, metric="mispricing_score", out_path=out_path)

            self.assertIsNotNone(fig)
            self.assertTrue(out_path.exists())

    def test_visualizations_save_to_specified_paths(self):
        """Test that visualization functions save to specified output paths"""
        df = self.small_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Test sector heatmap
            heatmap_path = tmpdir / "test_heatmap.png"
            mod.create_sector_heatmap(df, out_path=heatmap_path)
            self.assertTrue(heatmap_path.exists())

            # Test interactive plot
            plot_path = tmpdir / "test_plot.html"
            mod.create_interactive_prediction_plot(df, out_path=plot_path)
            self.assertTrue(plot_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
