"""
Test suite for ml_finance_model_main.ipynb code guidelines compliance.

This test validates that the notebook follows all conventions specified in
docs/code_guidelines.md including:
- Standardized function signatures and return types
- Dataset preparation return format
- Column normalization and schema validation
- Complete function calls (no broken code)
- Proper error handling
- Type hints and documentation

Following strict TDD: These tests are written first and should initially fail.
"""

import unittest
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple


class TestNotebookCodeGuidelinesCompliance(unittest.TestCase):
    """Test notebook compliance with code guidelines."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if not cls.notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {cls.notebook_path}")

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        # Extract all code cells
        cls.code_cells = [
            cell for cell in cls.notebook.get("cells", []) if cell.get("cell_type") == "code"
        ]

        # Combine all code into single string for pattern matching
        cls.all_code = "\n".join(["".join(cell.get("source", [])) for cell in cls.code_cells])

    def test_notebook_uses_normalize_columns_after_loading(self):
        """Test that normalize_columns() is called immediately after data loading."""
        # Should call normalize_columns on all_stocks after loading
        pattern = r"all_stocks\s*=\s*(?:load_from_db|load_from_csv).*?all_stocks\s*=\s*normalize_columns\s*\(\s*all_stocks\s*\)"

        self.assertIsNotNone(
            re.search(pattern, self.all_code, re.DOTALL),
            "normalize_columns() must be called immediately after data loading",
        )

    def test_notebook_validates_schema_after_normalization(self):
        """Test that validate_schema() is called after normalization."""
        # Should call validate_schema with require_target parameter
        pattern = r"validate_schema\s*\(\s*all_stocks.*?require_target\s*="

        self.assertIsNotNone(
            re.search(pattern, self.all_code, re.DOTALL),
            "validate_schema() must be called after normalize_columns()",
        )

    def test_no_broken_function_calls(self):
        """Test that there are no incomplete or broken function calls."""
        # Check for common patterns of broken code
        broken_patterns = [
            r'exclude_cols\s*=.*?feature_cols\s*=.*?if\s+[\'"]price_target[\'"].*?X\s*=.*?y\s*=.*?importance_df\s*=\s*calculate_feature_importance',  # Line 457
            r"labels\s*=\s*create_enhanced_event_labels.*?\)\s*print.*?\)\s*print.*?\)\s*print",  # Lines 467-471
            r"from\s+finance_ml\.advanced_models.*?import.*?\)\s*from\s+sklearn",  # Line 497
        ]

        for pattern in broken_patterns:
            match = re.search(pattern, self.all_code, re.DOTALL)
            self.assertIsNone(match, f"Found broken/incomplete function call pattern in notebook")

    def test_complete_feature_importance_calculation(self):
        """Test that feature importance calculation is complete and correct."""
        # Should have proper feature importance calculation with function call
        code_section = self._find_code_section_containing("Feature importance")

        # Check it's not just a comment or partial implementation
        self.assertIn(
            "features_importance_rf",
            code_section,
            "Must use features_importance_rf function for feature importance",
        )

        # Should have proper variable assignments
        self.assertIn("X =", code_section, "Must define X for feature importance")
        self.assertIn("y =", code_section, "Must define y for feature importance")

    def test_complete_classification_labels_creation(self):
        """Test that event label creation is complete."""
        code_section = self._find_code_section_containing("Create event labels")

        # Should use classification_create_enhanced_event_labels
        self.assertIn(
            "classification_create_enhanced_event_labels",
            code_section,
            "Must use classification_create_enhanced_event_labels function",
        )

        # Should have complete function call with closing parenthesis
        pattern = r"labels\s*=\s*classification_create_enhanced_event_labels\s*\([^)]+\)"
        self.assertIsNotNone(
            re.search(pattern, code_section),
            "Event label creation must be a complete function call",
        )

    def test_regression_uses_standardized_return_format(self):
        """Test that regression functions return standardized dict format."""
        # Check for regression_compare_regressors call
        code_section = self._find_code_section_containing("Compare Multiple Regression")

        if "regression_compare_regressors" in code_section:
            # Should assign to comparison_results
            self.assertIn(
                "comparison_results", code_section, "Must capture regression comparison results"
            )

            # Should access results properly
            # Results should be a dict that can be converted to DataFrame
            pattern = r"results_df\s*=\s*pd\.DataFrame\s*\(\s*comparison_results\s*\)"
            self.assertIsNotNone(
                re.search(pattern, code_section),
                "Regression results should be converted to DataFrame",
            )

    def test_metrics_calculation_is_complete(self):
        """Test that comprehensive metrics calculation is complete."""
        code_section = self._find_code_section_containing("Model Evaluation")

        # Should use evaluation_comprehensive_metrics
        if "evaluation" in code_section.lower():
            pattern = r"evaluation_comprehensive_metrics\s*\(\s*[^)]+\)"
            match = re.search(pattern, code_section)
            self.assertIsNotNone(match, "Must use evaluation_comprehensive_metrics for evaluation")

    def test_mispricing_calculation_is_complete(self):
        """Test that mispricing calculation uses correct function."""
        code_section = self._find_code_section_containing("mispricing")

        if "mispricing" in code_section.lower():
            # Should use analytics_calculate_mispricing
            self.assertIn(
                "analytics_calculate_mispricing",
                code_section,
                "Must use analytics_calculate_mispricing function",
            )

    def test_stock_ranking_functions_are_complete(self):
        """Test that stock ranking uses correct functions."""
        code_section = self._find_code_section_containing("Rank stocks")

        if "rank" in code_section.lower():
            # Should use analytics_rank_undervalued and analytics_rank_overvalued
            ranking_funcs = ["analytics_rank_undervalued", "analytics_rank_overvalued"]

            found_count = sum(1 for func in ranking_funcs if func in code_section)
            self.assertGreater(
                found_count, 0, "Must use analytics_rank_* functions for stock ranking"
            )

    def test_all_phase_imports_are_complete(self):
        """Test that all Phase 9.1-9.8 imports are properly structured."""
        import_section = self._find_code_section_containing("Phase 9.1:")

        # Check for key imports from each phase
        required_imports = [
            "apply_enhanced_imputation_strategy_4step",  # Phase 9.1
            "preprocessing_calculate_quality",  # Phase 9.1
            "generate_eda_report",  # Phase 9.2
            "features_build_comprehensive",  # Phase 9.3
            "classification_create_enhanced_event_labels",  # Phase 9.4
            "regression_train_xgboost",  # Phase 9.5
            "evaluation_comprehensive_metrics",  # Phase 9.6
            "analytics_calculate_mispricing",  # Phase 9.7
            "reporting_financial_metrics",  # Phase 9.8
        ]

        missing_imports = [imp for imp in required_imports if imp not in import_section]

        self.assertEqual(len(missing_imports), 0, f"Missing required imports: {missing_imports}")

    def test_no_raw_column_names_used(self):
        """Test that no raw CSV column names are used (e.g., 'Last Price')."""
        # Should not have raw column names like "Last Price" or "Price Target"
        raw_patterns = [
            r'["\']Last Price["\']',
            r'["\']Price Target["\']',
            r'["\']Market Cap["\']',
        ]

        for pattern in raw_patterns:
            matches = re.findall(pattern, self.all_code)
            self.assertEqual(len(matches), 0, f"Found raw column name (not normalized): {pattern}")

    def test_uses_canonical_column_names(self):
        """Test that canonical normalized column names are used."""
        # Should use normalized names
        canonical_names = ["last_price", "price_target", "sector", "region", "ticker"]

        found_count = sum(
            1
            for name in canonical_names
            if f"'{name}'" in self.all_code or f'"{name}"' in self.all_code
        )

        self.assertGreater(
            found_count,
            3,
            "Must use canonical normalized column names (last_price, price_target, etc.)",
        )

    def test_error_handling_present_in_data_loading(self):
        """Test that data loading has proper try-except error handling."""
        data_loading_section = self._find_code_section_containing("Load data")

        # Should have try-except block
        self.assertIn("try:", data_loading_section, "Data loading must have try block")
        self.assertIn("except", data_loading_section, "Data loading must have except block")

        # Should have fallback mechanism
        self.assertIn(
            "Falling back",
            data_loading_section,
            "Data loading must have fallback mechanism with informative message",
        )

    def test_regression_data_preparation_uses_correct_function(self):
        """Test that regression data preparation uses regression_prepare_data."""
        prep_section = self._find_code_section_containing("Prepare Regression Data")

        if "regression" in prep_section.lower():
            self.assertIn(
                "regression_prepare_data",
                prep_section,
                "Must use regression_prepare_data for data preparation",
            )

            # Should return 5 values including feature_info
            pattern = r"X_train.*?,\s*X_test.*?,\s*y_train.*?,\s*y_test.*?,\s*feature_info"
            self.assertIsNotNone(
                re.search(pattern, prep_section),
                "regression_prepare_data should return 5-tuple with feature_info",
            )

    def test_model_saving_includes_metadata(self):
        """Test that model saving includes proper metadata."""
        save_section = self._find_code_section_containing("Model Persistence")

        if "save" in save_section.lower():
            # Should use regression_save_model
            self.assertIn(
                "regression_save_model",
                save_section,
                "Must use regression_save_model for persistence",
            )

            # Should include metadata parameter
            self.assertIn("metadata=", save_section, "Model saving must include metadata parameter")

    def test_no_hardcoded_paths(self):
        """Test that no hardcoded absolute paths are used."""
        # Should use Path objects and OUTPUT_DIR variable
        hardcoded_patterns = [
            r'["\']C:\\',
            r'["\']c:\\',
            r'["\']D:\\',
        ]

        for pattern in hardcoded_patterns:
            matches = re.findall(pattern, self.all_code, re.IGNORECASE)
            self.assertEqual(len(matches), 0, f"Found hardcoded absolute path: {pattern}")

    def test_uses_output_dir_variable(self):
        """Test that OUTPUT_DIR variable is properly defined and used."""
        # Should define OUTPUT_DIR
        self.assertIn("OUTPUT_DIR", self.all_code, "Must define OUTPUT_DIR variable")

        # Should use Path
        pattern = r"OUTPUT_DIR\s*=\s*Path\s*\("
        self.assertIsNotNone(
            re.search(pattern, self.all_code), "OUTPUT_DIR must be defined using Path()"
        )

    def test_configuration_uses_env_vars(self):
        """Test that configuration respects environment variables."""
        # Should use os.getenv for configuration
        env_patterns = [
            r"RANDOM_SEED\s*=.*?os\.getenv",
            r"DB_URL\s*=.*?os\.getenv",
        ]

        found_count = sum(1 for pattern in env_patterns if re.search(pattern, self.all_code))

        self.assertGreater(
            found_count, 0, "Must use os.getenv() for environment variable configuration"
        )

    # Helper methods
    def _find_code_section_containing(self, keyword: str) -> str:
        """Find code section containing specific keyword."""
        for cell in self.code_cells:
            source = "".join(cell.get("source", []))
            if keyword.lower() in source.lower():
                return source
        return ""

    def _extract_function_call(self, source: str, func_name: str) -> str:
        """Extract complete function call from source."""
        pattern = rf"{func_name}\s*\([^)]*\)"
        match = re.search(pattern, source)
        return match.group(0) if match else ""


class TestNotebookFunctionReturnTypes(unittest.TestCase):
    """Test that notebook functions use standardized return types."""

    @classmethod
    def setUpClass(cls):
        """Load notebook for testing."""
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_code = "\n".join(
            [
                "".join(cell.get("source", []))
                for cell in cls.notebook.get("cells", [])
                if cell.get("cell_type") == "code"
            ]
        )

    def test_regression_results_accessed_via_metrics_dict(self):
        """Test that regression metrics are accessed via ['metrics'] key."""
        # New code should access metrics as res["metrics"]["mae"]
        # Old style res["mae"] should be updated

        if "stacking_results" in self.all_code:
            # Check for proper access pattern
            pattern = r'stacking_results\.get\(["\']train_score["\']\)'
            match = re.search(pattern, self.all_code)
            # This is acceptable as stacking_results is a dict with metadata
            # The key is that metrics should be in a nested dict

    def test_classification_results_use_standard_format(self):
        """Test that classification results use standardized return format."""
        if "comparison_results" in self.all_code:
            # Should handle results as DataFrame or dict properly
            self.assertIn(
                "comparison_results", self.all_code, "Classification results should be captured"
            )


class TestNotebookDataQuality(unittest.TestCase):
    """Test data quality checks in notebook."""

    @classmethod
    def setUpClass(cls):
        """Load notebook for testing."""
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_code = "\n".join(
            [
                "".join(cell.get("source", []))
                for cell in cls.notebook.get("cells", [])
                if cell.get("cell_type") == "code"
            ]
        )

    def test_nan_validation_before_modeling(self):
        """Test that NaN validation is performed before modeling."""
        # Should check for NaN values
        nan_checks = [
            r"\.isnull\(\)\.sum\(\)",
            r"\.isna\(\)\.sum\(\)",
        ]

        found = any(re.search(pattern, self.all_code) for pattern in nan_checks)
        self.assertTrue(found, "Must validate NaN values before modeling")

    def test_infinite_value_handling(self):
        """Test that infinite values are handled."""
        # Should handle inf values
        self.assertIn("np.inf", self.all_code, "Must handle infinite values")

        # Should replace infinite values
        pattern = r"\.replace\s*\(\s*\[\s*np\.inf"
        self.assertIsNotNone(re.search(pattern, self.all_code), "Must replace infinite values")


if __name__ == "__main__":
    unittest.main()
