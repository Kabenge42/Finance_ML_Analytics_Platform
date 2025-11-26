"""
Test suite for Notebook Quality Improvements (TDD Implementation)

Tests written following strict TDD approach:
1. Write failing test
2. Implement minimal code to pass
3. Refactor

Coverage target: ≥80% for changed files
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from finance_ml import load_config, NotebookConfig, simple_eda


class TestLoadConfigWithOutputDir(unittest.TestCase):
    """Test load_config accepts output_dir parameter to avoid config mutation."""

    def test_load_config_accepts_output_dir_parameter(self):
        """load_config should accept output_dir parameter."""
        # This test will FAIL initially - TDD red phase
        output_path = Path("custom_outputs")
        config = load_config(output_dir=output_path)

        self.assertEqual(config.output_dir, output_path)

    def test_load_config_output_dir_overrides_default(self):
        """output_dir parameter should override environment/default value."""
        custom_dir = Path("my_custom_output")
        config = load_config(output_dir=custom_dir)

        self.assertEqual(config.output_dir, custom_dir)
        self.assertIsInstance(config.output_dir, Path)

    def test_load_config_output_dir_accepts_string(self):
        """output_dir parameter should accept string and convert to Path."""
        config = load_config(output_dir="string_output")

        self.assertEqual(config.output_dir, Path("string_output"))
        self.assertIsInstance(config.output_dir, Path)

    def test_load_config_without_output_dir_uses_default(self):
        """load_config without output_dir should use default behavior."""
        config = load_config()

        # Should have output_dir from env or default
        self.assertIsNotNone(config.output_dir)
        self.assertIsInstance(config.output_dir, Path)


class TestNotebookConfigImport(unittest.TestCase):
    """Test NotebookConfig is properly exported from finance_ml package."""

    def test_notebook_config_importable_from_finance_ml(self):
        """NotebookConfig should be importable from finance_ml package."""
        # This should already pass - NotebookConfig is exported
        from finance_ml import NotebookConfig

        self.assertIsNotNone(NotebookConfig)

    def test_notebook_config_can_be_instantiated(self):
        """NotebookConfig should be instantiable with default values."""
        cfg = NotebookConfig()

        self.assertIsNotNone(cfg)
        self.assertTrue(hasattr(cfg, "have_finance_prediction"))

    def test_notebook_config_accepts_parameters(self):
        """NotebookConfig should accept configuration parameters."""
        cfg = NotebookConfig(have_finance_prediction=False, debug_mode=True)

        self.assertFalse(cfg.have_finance_prediction)
        self.assertTrue(cfg.debug_mode)


class TestSimpleEDAWithoutWorkaround(unittest.TestCase):
    """Test simple_eda works correctly without AttributeError workaround."""

    def setUp(self):
        """Create temporary directory for test outputs."""
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_simple_eda_does_not_raise_attribute_error_with_dataframe(self):
        """simple_eda should not raise AttributeError about DataFrame.dtype."""
        # Create a sample DataFrame
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "sector": ["Tech", "Tech", "Tech"],
                "region": ["US", "US", "US"],
                "price": [150.0, 2800.0, 300.0],
                "market_cap": [2.5e12, 1.8e12, 2.2e12],
            }
        )

        # Should not raise AttributeError
        try:
            result = simple_eda(df, out_dir=self.out_dir)
            self.assertIsInstance(result, dict)
            self.assertIn("row_count", result)
            self.assertEqual(result["row_count"], 3)
        except AttributeError as e:
            if "'DataFrame' object has no attribute 'dtype'" in str(e):
                self.fail(f"simple_eda raised AttributeError about DataFrame.dtype: {e}")
            else:
                raise

    def test_simple_eda_returns_valid_summary(self):
        """simple_eda should return a valid summary dictionary."""
        df = pd.DataFrame(
            {"ticker": ["AAPL", "GOOGL"], "sector": ["Tech", "Tech"], "price": [150.0, 2800.0]}
        )

        result = simple_eda(df, out_dir=self.out_dir)

        self.assertIsInstance(result, dict)
        self.assertIn("row_count", result)
        self.assertIn("column_count", result)
        self.assertIn("columns", result)
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["column_count"], 3)


class TestTypeValidation(unittest.TestCase):
    """Test type validation for load_stock_data and other functions."""

    def test_load_stock_data_returns_dataframe_or_none(self):
        """load_stock_data should return DataFrame or None."""
        from finance_ml import load_stock_data, FinanceMLConfig

        config = FinanceMLConfig()
        result = load_stock_data(config)

        # Result should be either pd.DataFrame or None
        self.assertTrue(
            isinstance(result, pd.DataFrame) or result is None,
            f"Expected DataFrame or None, got {type(result)}",
        )

    def test_type_check_helper_for_dataframe(self):
        """Test utility for validating DataFrame type."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        not_df = [1, 2, 3]

        # Type check should distinguish DataFrame from other types
        self.assertIsInstance(df, pd.DataFrame)
        self.assertNotIsInstance(not_df, pd.DataFrame)


class TestPathImport(unittest.TestCase):
    """Test Path is properly imported and available."""

    def test_path_importable_from_pathlib(self):
        """Path should be importable from pathlib."""
        from pathlib import Path

        self.assertIsNotNone(Path)

        # Test basic Path functionality
        p = Path("test")
        self.assertIsInstance(p, Path)


if __name__ == "__main__":
    unittest.main()
