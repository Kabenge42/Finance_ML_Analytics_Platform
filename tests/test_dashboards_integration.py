
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import importlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finance_ml.dashboards import dash_app

class TestDashAppIntegration(unittest.TestCase):
    def setUp(self):
        self.app = dash_app.app
        self.layout = self.app.layout

    def test_new_tabs_existence(self):
        """Test that the new tabs are present in the Dash app layout."""
        # Navigate through the layout to find the Tabs component
        # The layout structure is: Div -> [H1, Div (KPI), Div (Filters), Tabs]
        # Note: The structure might change, so we'll search recursively or look at known structure
        
        # Assuming the Tabs are at the top level or inside a container
        # Let's verify if we can find "Uncertainty & Calibration" in the children of Tabs
        
        found_tabs = []
        
        def extract_tabs(component):
            if hasattr(component, 'children'):
                children = component.children
                if isinstance(children, list):
                    for child in children:
                        # check if child is dcc.Tab or similar (it might be an object)
                        # Dash components are objects, checking type or properties
                        if hasattr(child, 'label'):
                            found_tabs.append(child.label)
                        extract_tabs(child)
                elif hasattr(children, 'label'): # Single child
                     found_tabs.append(children.label)
                     
        extract_tabs(self.layout)
        
        # Check for new tabs
        self.assertIn("🔬 Uncertainty & Calibration", found_tabs, "Uncertainty tab missing")
        self.assertIn("🛡️ Safety Rails & Data Quality", found_tabs, "Safety Rails tab missing")
        self.assertIn("🏛️ Model Governance", found_tabs, "Governance tab missing")

    def test_artifact_helper_functions(self):
        """Test helper functions for artifact loading."""
        # These functions are not yet implemented, so this import might fail if I put it here
        # But I'll check if they exist in the module after implementation
        self.assertTrue(hasattr(dash_app, 'get_artifact_path'), "get_artifact_path function missing")
        self.assertTrue(hasattr(dash_app, 'render_artifact_or_placeholder'), "render_artifact_or_placeholder function missing")

class TestStreamlitAppIntegration(unittest.TestCase):
    def test_streamlit_tabs_structure(self):
        """Test that Streamlit app defines the correct tabs."""
        # We can't easily run the script, but we can verify if the code contains the tab definitions
        # This is a static analysis test
        
        streamlit_path = PROJECT_ROOT / "finance_ml" / "dashboards" / "streamlit_app.py"
        with open(streamlit_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        expected_tabs = [
            "📋 Executive Summary",
            "🔬 Uncertainty & Calibration",
            "🛡️ Safety Rails",
            "🏛️ Model Governance"
        ]
        
        for tab in expected_tabs:
            self.assertIn(tab, content, f"Streamlit app missing tab: {tab}")

if __name__ == '__main__':
    unittest.main()
