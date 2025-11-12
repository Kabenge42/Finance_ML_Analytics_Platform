import unittest
import tempfile
import shutil
from pathlib import Path
import sys

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

# Check if streamlit is available (for import validation)
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


@unittest.skipIf(
    pd is None or mod is None or np is None, "pandas/numpy or finance_ml not installed"
)
class TestStreamlitDashboard(unittest.TestCase):
    """Tests for Streamlit dashboard components - Phase 1: Interactive Dashboards"""

    def setUp(self):
        """Create sample data for dashboard tests"""
        self.sample_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
                "sector": ["Tech", "Tech", "Tech", "Tech", "Tech"],
                "region": ["US", "US", "US", "US", "US"],
                "last_price": [150.0, 2800.0, 350.0, 3200.0, 300.0],
                "predicted_price_target": [180.0, 3000.0, 340.0, 3500.0, 280.0],
                "market_cap": [2500000, 1800000, 2300000, 1600000, 800000],
                "mispricing_score": [0.20, 0.071, -0.029, 0.094, -0.067],
            }
        )

    def test_streamlit_app_file_exists(self):
        """Test that streamlit_app.py file exists in dashboards directory"""
        app_path = Path("finance_ml/dashboards/streamlit_app.py")
        self.assertTrue(app_path.exists(), "streamlit_app.py not found in finance_ml/dashboards/")

    def test_streamlit_app_has_required_imports(self):
        """Test that streamlit_app.py has required imports"""
        app_path = Path("finance_ml/dashboards/streamlit_app.py")
        if not app_path.exists():
            self.skipTest("streamlit_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")

        # Check for essential imports
        self.assertIn("import streamlit", content, "Missing streamlit import")
        self.assertIn("import pandas", content, "Missing pandas import")
        self.assertIn("import plotly", content, "Missing plotly import")
        self.assertIn("from finance_ml.eval import", content, "Missing finance_ml.eval imports")

    def test_streamlit_app_has_page_config(self):
        """Test that streamlit app sets page configuration"""
        app_path = Path("finance_ml/dashboards/streamlit_app.py")
        if not app_path.exists():
            self.skipTest("streamlit_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")
        self.assertIn("st.set_page_config", content, "Missing page configuration")
        self.assertIn("page_title", content, "Missing page_title in config")

    def test_streamlit_app_has_file_uploader(self):
        """Test that app includes file uploader for CSV"""
        app_path = Path("finance_ml/dashboards/streamlit_app.py")
        if not app_path.exists():
            self.skipTest("streamlit_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")
        self.assertIn("file_uploader", content, "Missing file uploader")
        self.assertIn("csv", content.lower(), "File uploader should accept CSV files")

    def test_streamlit_app_has_tabs(self):
        """Test that app has multiple tabs for different views"""
        app_path = Path("finance_ml/dashboards/streamlit_app.py")
        if not app_path.exists():
            self.skipTest("streamlit_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")
        self.assertIn("st.tabs", content, "Missing tabs structure")
        # Check for expected tabs
        self.assertIn("Overview", content, "Missing Overview tab")
        self.assertIn("Stock Ranking", content, "Missing Stock Ranking tab")
        self.assertIn("Sector Analysis", content, "Missing Sector Analysis tab")

    def test_streamlit_app_has_filters(self):
        """Test that app includes filter controls"""
        app_path = Path("finance_ml/dashboards/streamlit_app.py")
        if not app_path.exists():
            self.skipTest("streamlit_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")
        # Should have filter controls
        has_multiselect = "multiselect" in content.lower()
        has_slider = "slider" in content.lower()
        has_selectbox = "selectbox" in content.lower()

        self.assertTrue(
            has_multiselect or has_slider or has_selectbox,
            "Missing filter controls (multiselect, slider, or selectbox)",
        )

    def test_streamlit_app_uses_eval_functions(self):
        """Test that app uses functions from finance_ml.eval"""
        app_path = Path("finance_ml/dashboards/streamlit_app.py")
        if not app_path.exists():
            self.skipTest("streamlit_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")

        # Check for usage of key eval functions
        expected_functions = [
            "calculate_mispricing_score",
            "rank_stocks_by_sector",
            "calculate_financial_metrics_dashboard",
            "generate_data_quality_alerts",
        ]

        found_functions = []
        for func in expected_functions:
            if func in content:
                found_functions.append(func)

        self.assertGreater(
            len(found_functions),
            0,
            f"Should use at least one eval function from: {expected_functions}",
        )

    def test_dashboards_module_exists(self):
        """Test that finance_ml.dashboards module exists and is importable"""
        try:
            import finance_ml.dashboards

            self.assertTrue(True)
        except ImportError:
            self.fail("finance_ml.dashboards module not found or not importable")

    def test_dashboards_init_exists(self):
        """Test that finance_ml/dashboards/__init__.py exists"""
        init_path = Path("finance_ml/dashboards/__init__.py")
        self.assertTrue(init_path.exists(), "__init__.py not found in finance_ml/dashboards/")


@unittest.skipIf(not STREAMLIT_AVAILABLE, "streamlit not installed")
class TestStreamlitIntegration(unittest.TestCase):
    """Integration tests for Streamlit dashboard (requires streamlit installed)"""

    def test_streamlit_app_syntax_valid(self):
        """Test that streamlit_app.py has valid Python syntax"""
        app_path = Path("finance_ml/dashboards/streamlit_app.py")
        if not app_path.exists():
            self.skipTest("streamlit_app.py does not exist yet")

        # Try to compile the file to check for syntax errors
        try:
            with open(app_path, "r", encoding="utf-8") as f:
                code = f.read()
            compile(code, str(app_path), "exec")
        except SyntaxError as e:
            self.fail(f"Syntax error in streamlit_app.py: {e}")


if __name__ == "__main__":
    unittest.main()
