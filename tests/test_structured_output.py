import unittest
import tempfile
import shutil
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import finance_ml as mod
except Exception:
    mod = None


@unittest.skipIf(pd is None or mod is None, "pandas or finance_ml not installed")
class TestStructuredOutput(unittest.TestCase):
    """Tests for create_structured_output_directory() function - Phase 1: Interactive Dashboards"""

    def setUp(self):
        """Create temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory"""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_create_structured_output_directory_basic(self):
        """Test basic directory structure creation"""
        result = mod.create_structured_output_directory(
            base_dir=self.test_dir, run_id="test_run_001"
        )

        # Check that result is a dict with expected keys
        self.assertIsInstance(result, dict)
        self.assertIn("run_dir", result)
        self.assertIn("data", result)
        self.assertIn("regression", result)
        self.assertIn("reports", result)
        self.assertIn("visualizations", result)
        self.assertIn("analytics", result)
        self.assertIn("logs", result)

        # Check that all directories were created
        for key, path in result.items():
            self.assertTrue(Path(path).exists(), f"Directory {key} at {path} was not created")
            self.assertTrue(Path(path).is_dir(), f"{key} at {path} is not a directory")

    def test_create_structured_output_with_subdirectories(self):
        """Test that subdirectories are created correctly"""
        result = mod.create_structured_output_directory(
            base_dir=self.test_dir, run_id="test_run_002"
        )

        # Check specific subdirectories
        self.assertIn("model_checkpoints", result)
        self.assertIn("eda_viz", result)
        self.assertIn("prediction_viz", result)
        self.assertIn("residual_viz", result)
        self.assertIn("feature_viz", result)

        # Verify subdirectories exist
        self.assertTrue(Path(result["model_checkpoints"]).exists())
        self.assertTrue(Path(result["eda_viz"]).exists())
        self.assertTrue(Path(result["prediction_viz"]).exists())

    def test_create_structured_output_creates_readme(self):
        """Test that README.md is created in run directory"""
        result = mod.create_structured_output_directory(
            base_dir=self.test_dir, run_id="test_run_003"
        )

        readme_path = Path(result["run_dir"]) / "README.md"
        self.assertTrue(readme_path.exists(), "README.md was not created")

        # Check README content
        content = readme_path.read_text()
        self.assertIn("test_run_003", content)
        self.assertIn("Directory Structure", content)
        self.assertIn("data/", content)
        self.assertIn("regression/", content)

    def test_create_structured_output_auto_run_id(self):
        """Test automatic run_id generation when not provided"""
        result = mod.create_structured_output_directory(base_dir=self.test_dir)

        # Check that run_dir exists and has timestamp-like name
        self.assertTrue(Path(result["run_dir"]).exists())
        run_dir_name = Path(result["run_dir"]).name
        # Should be timestamp format like 20251105_145800
        self.assertTrue(
            len(run_dir_name) >= 15, f"Auto-generated run_id seems invalid: {run_dir_name}"
        )

    def test_create_structured_output_paths_are_pathlib(self):
        """Test that returned paths are Path objects or can be converted"""
        result = mod.create_structured_output_directory(
            base_dir=self.test_dir, run_id="test_run_004"
        )

        # All values should be convertible to Path
        for key, value in result.items():
            path_obj = Path(value)
            self.assertTrue(path_obj.exists(), f"{key} path does not exist: {value}")


if __name__ == "__main__":
    unittest.main()
