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

# Check if dash is available
try:
    import dash

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


@unittest.skipIf(
    pd is None or mod is None or np is None, "pandas/numpy or finance_ml not installed"
)
class TestDashDashboard(unittest.TestCase):
    """Tests for Plotly Dash dashboard components - Phase 1: Interactive Dashboards"""

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

    def test_dash_app_file_exists(self):
        """Test that dash_app.py file exists in dashboards directory"""
        app_path = Path("finance_ml/dashboards/dash_app.py")
        self.assertTrue(app_path.exists(), "dash_app.py not found in finance_ml/dashboards/")

    def test_dash_app_has_required_imports(self):
        """Test that dash_app.py has required imports"""
        app_path = Path("finance_ml/dashboards/dash_app.py")
        if not app_path.exists():
            self.skipTest("dash_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")

        # Check for essential imports
        self.assertIn("import dash", content, "Missing dash import")
        self.assertIn("import pandas", content, "Missing pandas import")
        self.assertIn("import plotly", content, "Missing plotly import")

    def test_dash_app_creates_app_instance(self):
        """Test that dash_app.py creates a Dash app instance"""
        app_path = Path("finance_ml/dashboards/dash_app.py")
        if not app_path.exists():
            self.skipTest("dash_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")
        self.assertIn("dash.Dash", content, "Missing Dash app instantiation")
        self.assertIn("app = ", content, "Missing app variable assignment")

    def test_dash_app_has_layout(self):
        """Test that dash app defines a layout"""
        app_path = Path("finance_ml/dashboards/dash_app.py")
        if not app_path.exists():
            self.skipTest("dash_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")
        self.assertIn("app.layout", content, "Missing app.layout definition")

    def test_dash_app_has_callbacks(self):
        """Test that dash app includes callbacks for interactivity"""
        app_path = Path("finance_ml/dashboards/dash_app.py")
        if not app_path.exists():
            self.skipTest("dash_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")
        self.assertIn("@app.callback", content, "Missing callback decorators")
        self.assertIn("Output", content, "Missing Output in callbacks")
        self.assertIn("Input", content, "Missing Input in callbacks")

    def test_dash_app_has_dropdowns(self):
        """Test that dash app includes dropdown filters"""
        app_path = Path("finance_ml/dashboards/dash_app.py")
        if not app_path.exists():
            self.skipTest("dash_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")
        self.assertIn("dcc.Dropdown", content, "Missing dropdown components")

    def test_dash_app_has_graphs(self):
        """Test that dash app includes graph components"""
        app_path = Path("finance_ml/dashboards/dash_app.py")
        if not app_path.exists():
            self.skipTest("dash_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")
        self.assertIn("dcc.Graph", content, "Missing graph components")

    def test_dash_app_has_data_table(self):
        """Test that dash app includes data table component"""
        app_path = Path("finance_ml/dashboards/dash_app.py")
        if not app_path.exists():
            self.skipTest("dash_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")
        self.assertIn("dash_table", content, "Missing dash_table import or usage")

    def test_dash_app_has_run_server(self):
        """Test that dash app can be run as main"""
        app_path = Path("finance_ml/dashboards/dash_app.py")
        if not app_path.exists():
            self.skipTest("dash_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")
        self.assertIn("if __name__", content, "Missing main execution block")
        self.assertIn("run_server", content, "Missing run_server call")

    def test_dash_app_has_title(self):
        """Test that dash app sets a title"""
        app_path = Path("finance_ml/dashboards/dash_app.py")
        if not app_path.exists():
            self.skipTest("dash_app.py does not exist yet")

        content = app_path.read_text(encoding="utf-8")
        # Check for title in app initialization or layout
        has_title = "title=" in content or "H1(" in content
        self.assertTrue(has_title, "Missing app title")


@unittest.skipIf(not DASH_AVAILABLE, "dash not installed")
class TestDashIntegration(unittest.TestCase):
    """Integration tests for Dash dashboard (requires dash installed)"""

    def test_dash_app_syntax_valid(self):
        """Test that dash_app.py has valid Python syntax"""
        app_path = Path("finance_ml/dashboards/dash_app.py")
        if not app_path.exists():
            self.skipTest("dash_app.py does not exist yet")

        # Try to compile the file to check for syntax errors
        try:
            with open(app_path, "r", encoding="utf-8") as f:
                code = f.read()
            compile(code, str(app_path), "exec")
        except SyntaxError as e:
            self.fail(f"Syntax error in dash_app.py: {e}")

    def test_dash_app_importable(self):
        """Test that dash_app module can be imported"""
        try:
            # Try to import the module (will fail if syntax errors exist)
            import finance_ml.dashboards.dash_app
        except ImportError:
            self.skipTest("dash_app.py does not exist yet")
        except Exception as e:
            # Other exceptions might be expected (like missing data files)
            # but syntax errors should not occur
            if "SyntaxError" in str(type(e).__name__):
                self.fail(f"Import failed with syntax error: {e}")


if __name__ == "__main__":
    unittest.main()
