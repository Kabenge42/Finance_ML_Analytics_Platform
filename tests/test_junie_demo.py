import sys
import unittest
from pathlib import Path

# Add project root to sys.path to allow importing from finance_ml
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class TestJunieDemo(unittest.TestCase):
    """
    A simple demo test to illustrate the testing process for Junie.
    """
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic to ensure the test runner is working."""
        self.assertEqual(1 + 1, 2)
        
    def test_project_structure(self):
        """Test that the expected project directories exist."""
        self.assertTrue((project_root / "finance_ml").exists())
        self.assertTrue((project_root / "tests").exists())
        self.assertTrue((project_root / "docs").exists())

if __name__ == "__main__":
    unittest.main()
