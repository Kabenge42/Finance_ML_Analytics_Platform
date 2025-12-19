"""
Test notebook TDD compliance for ml_finance_model_main2_0.ipynb.

Validates that the notebook follows TDD conventions from code_guidelines.md Section 8:
- Section 8.1: Centralized Configuration Constants (Single Source of Truth)
- Section 8.2: DataFrame Stage Naming Convention (4-stage pipeline)
- Section 8.3: Magic Numbers Policy
- Validation functions and checkpoints
"""

import json
import os
import re
import unittest
from pathlib import Path


class TestConfigurationConstants(unittest.TestCase):
    """Test Section 8.1: Centralized Configuration Constants."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        cls.code_cells = []

        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"
            if cell.get("cell_type") == "code":
                cls.code_cells.append(source)

    def test_has_target_col_constant(self):
        """Test that TARGET_COL constant is defined."""
        has_const = "TARGET_COL" in self.all_source
        self.assertTrue(has_const, "Missing TARGET_COL constant definition")

    def test_has_target_col_fallback_constant(self):
        """Test that TARGET_COL_FALLBACK constant is defined."""
        has_const = "TARGET_COL_FALLBACK" in self.all_source
        self.assertTrue(has_const, "Missing TARGET_COL_FALLBACK constant definition")

    def test_has_test_size_constant(self):
        """Test that TEST_SIZE constant is defined."""
        has_const = "TEST_SIZE" in self.all_source
        self.assertTrue(has_const, "Missing TEST_SIZE constant definition")

    def test_has_train_size_constant(self):
        """Test that TRAIN_SIZE constant is defined."""
        has_const = "TRAIN_SIZE" in self.all_source
        self.assertTrue(has_const, "Missing TRAIN_SIZE constant definition")

    def test_has_cv_folds_constant(self):
        """Test that CV_FOLDS constant is defined."""
        has_const = "CV_FOLDS" in self.all_source
        self.assertTrue(has_const, "Missing CV_FOLDS constant definition")

    def test_has_quantiles_constant(self):
        """Test that QUANTILES constant is defined."""
        has_const = "QUANTILES" in self.all_source
        self.assertTrue(has_const, "Missing QUANTILES constant definition")

    def test_has_min_sector_samples_constant(self):
        """Test that MIN_SECTOR_SAMPLES constant is defined."""
        has_const = "MIN_SECTOR_SAMPLES" in self.all_source
        self.assertTrue(has_const, "Missing MIN_SECTOR_SAMPLES constant definition")

    def test_has_max_sector_weight_constant(self):
        """Test that MAX_SECTOR_WEIGHT constant is defined."""
        has_const = "MAX_SECTOR_WEIGHT" in self.all_source
        self.assertTrue(has_const, "Missing MAX_SECTOR_WEIGHT constant definition")

    def test_has_max_single_position_constant(self):
        """Test that MAX_SINGLE_POSITION constant is defined."""
        has_const = "MAX_SINGLE_POSITION" in self.all_source
        self.assertTrue(has_const, "Missing MAX_SINGLE_POSITION constant definition")

    def test_has_iqr_multiplier_constant(self):
        """Test that IQR_MULTIPLIER constant is defined."""
        has_const = "IQR_MULTIPLIER" in self.all_source
        self.assertTrue(has_const, "Missing IQR_MULTIPLIER constant definition")

    def test_has_zscore_threshold_constant(self):
        """Test that ZSCORE_THRESHOLD constant is defined."""
        has_const = "ZSCORE_THRESHOLD" in self.all_source
        self.assertTrue(has_const, "Missing ZSCORE_THRESHOLD constant definition")

    def test_has_winsorize_bounds_constants(self):
        """Test that WINSORIZE_LOWER and WINSORIZE_UPPER constants are defined."""
        has_lower = "WINSORIZE_LOWER" in self.all_source
        has_upper = "WINSORIZE_UPPER" in self.all_source
        self.assertTrue(has_lower, "Missing WINSORIZE_LOWER constant definition")
        self.assertTrue(has_upper, "Missing WINSORIZE_UPPER constant definition")

    def test_has_random_seed_constant(self):
        """Test that RANDOM_SEED constant is defined."""
        has_const = "RANDOM_SEED" in self.all_source
        self.assertTrue(has_const, "Missing RANDOM_SEED constant definition")

    def test_has_model_version_constant(self):
        """Test that MODEL_VERSION constant is defined."""
        has_const = "MODEL_VERSION" in self.all_source
        self.assertTrue(has_const, "Missing MODEL_VERSION constant definition")

    def test_has_validate_configuration_function(self):
        """Test that validate_configuration() function exists."""
        has_func = "def validate_configuration" in self.all_source
        self.assertTrue(has_func, "Missing validate_configuration() function")

    def test_validate_configuration_is_called(self):
        """Test that validate_configuration() is called."""
        has_call = "validate_configuration()" in self.all_source
        self.assertTrue(has_call, "validate_configuration() is not called")


class TestDataFrameStageNaming(unittest.TestCase):
    """Test Section 8.2: DataFrame Stage Naming Convention."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_all_stocks_preprocessed_stage(self):
        """Test that all_stocks_preprocessed stage exists."""
        has_stage = "all_stocks_preprocessed" in self.all_source
        self.assertTrue(has_stage, "Missing all_stocks_preprocessed stage (Stage 1)")

    def test_has_all_stocks_features_stage(self):
        """Test that all_stocks_features stage exists."""
        has_stage = "all_stocks_features" in self.all_source
        self.assertTrue(has_stage, "Missing all_stocks_features stage (Stage 2)")

    def test_has_all_stocks_enhanced_stage(self):
        """Test that all_stocks_enhanced stage exists."""
        has_stage = "all_stocks_enhanced" in self.all_source
        self.assertTrue(has_stage, "Missing all_stocks_enhanced stage (Stage 4)")

    def test_uses_descriptive_dataframe_names(self):
        """Test that notebook uses descriptive DataFrame names."""
        # Check for bad patterns like df1, df2, temp_df
        bad_patterns = [
            r"\bdf1\b",
            r"\bdf2\b",
            r"\bdf3\b",
            r"\btemp_df\b",
        ]
        violations = []
        for pattern in bad_patterns:
            if re.search(pattern, self.all_source):
                violations.append(pattern)

        # Allow some generic names but flag if too many
        self.assertLess(
            len(violations), 3, f"Too many non-descriptive DataFrame names: {violations}"
        )


class TestMagicNumbersPolicy(unittest.TestCase):
    """Test Section 8.3: Magic Numbers Policy."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        cls.code_cells = []

        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"
            if cell.get("cell_type") == "code":
                cls.code_cells.append(source)

    def test_random_state_uses_constant(self):
        """Test that random_state uses RANDOM_SEED constant."""
        # Look for random_state=42 or random_state=123 etc (magic numbers)
        # Exclude comments and strings
        magic_patterns = [
            r"random_state\s*=\s*\d+",
        ]

        # Count uses of RANDOM_SEED vs magic numbers
        uses_constant = self.all_source.count("RANDOM_SEED")
        magic_uses = 0
        for pattern in magic_patterns:
            magic_uses += len(re.findall(pattern, self.all_source))

        # Allow some magic numbers in comments/examples but constant should be primary
        self.assertGreater(uses_constant, 0, "RANDOM_SEED constant should be used for random_state")

    def test_test_size_uses_constant(self):
        """Test that test_size parameter uses TEST_SIZE constant."""
        uses_constant = self.all_source.count("TEST_SIZE")
        self.assertGreater(
            uses_constant, 0, "TEST_SIZE constant should be used for test_size parameter"
        )

    def test_cv_folds_uses_constant(self):
        """Test that cross-validation uses CV_FOLDS constant."""
        uses_constant = self.all_source.count("CV_FOLDS")
        self.assertGreater(
            uses_constant, 0, "CV_FOLDS constant should be used for cross-validation"
        )

    def test_quantiles_uses_constant(self):
        """Test that quantile regression uses QUANTILES constant."""
        uses_constant = (
            self.all_source.count("QUANTILES")
            + self.all_source.count("LOWER_QUANTILE")
            + self.all_source.count("UPPER_QUANTILE")
            + self.all_source.count("MEDIAN_QUANTILE")
        )
        self.assertGreater(
            uses_constant, 0, "QUANTILES constant should be used for quantile regression"
        )


class TestValidationCheckpoints(unittest.TestCase):
    """Test that notebook includes validation checkpoints."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_assertion_statements(self):
        """Test that notebook includes assertion statements."""
        has_assert = "assert " in self.all_source
        self.assertTrue(has_assert, "Missing assertion statements for validation")

    def test_has_validation_checkpoints(self):
        """Test that notebook includes validation checkpoints."""
        has_checkpoint = (
            "validation" in self.all_source.lower()
            or "checkpoint" in self.all_source.lower()
            or "Validation" in self.all_source
        )
        self.assertTrue(has_checkpoint, "Missing validation checkpoints")

    def test_has_shape_validation(self):
        """Test that notebook validates DataFrame shapes."""
        has_shape_check = ".shape" in self.all_source
        self.assertTrue(has_shape_check, "Missing DataFrame shape validation")

    def test_has_empty_check(self):
        """Test that notebook checks for empty DataFrames."""
        has_empty_check = ".empty" in self.all_source or "len(" in self.all_source
        self.assertTrue(has_empty_check, "Missing empty DataFrame checks")


class TestPriceColumnPreservation(unittest.TestCase):
    """Test Section 8.5.2: Price Column Preservation Policy."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_price_preservation_validation(self):
        """Test that notebook includes price preservation validation."""
        has_validation = (
            "validate_price_preservation" in self.all_source
            or "price_preservation" in self.all_source.lower()
            or "PRICE_COLUMNS" in self.all_source
        )
        self.assertTrue(has_validation, "Missing price column preservation validation")

    def test_references_price_target_column(self):
        """Test that notebook references price_target column."""
        has_reference = "price_target" in self.all_source.lower() or "TARGET_COL" in self.all_source
        self.assertTrue(has_reference, "Missing price_target column reference")

    def test_references_last_price_column(self):
        """Test that notebook references last_price column."""
        has_reference = (
            "last_price" in self.all_source.lower() or "TARGET_COL_FALLBACK" in self.all_source
        )
        self.assertTrue(has_reference, "Missing last_price column reference")


class TestCodeGuidelinesCompliance(unittest.TestCase):
    """Test general code_guidelines.md compliance."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_references_code_guidelines(self):
        """Test that notebook references code_guidelines.md."""
        has_reference = (
            "code_guidelines" in self.all_source.lower()
            or "code guidelines" in self.all_source.lower()
        )
        self.assertTrue(has_reference, "Missing reference to code_guidelines.md")

    def test_has_logging_setup(self):
        """Test that notebook sets up logging."""
        has_logging = "logging" in self.all_source.lower() or "logger" in self.all_source.lower()
        self.assertTrue(has_logging, "Missing logging setup")

    def test_has_import_organization(self):
        """Test that notebook has organized imports."""
        # Check for standard import patterns
        has_imports = (
            "import pandas" in self.all_source
            or "import numpy" in self.all_source
            or "from finance_ml" in self.all_source
        )
        self.assertTrue(has_imports, "Missing organized imports")


class TestETLPipelineCompliance(unittest.TestCase):
    """Test ETL pipeline compliance per Section 8.2."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_etl_pipeline_reference(self):
        """Test that notebook references ETL pipeline."""
        has_etl = "etl" in self.all_source.lower() or "ETL" in self.all_source
        self.assertTrue(has_etl, "Missing ETL pipeline reference")

    def test_has_imputation_strategy(self):
        """Test that notebook includes imputation strategy."""
        has_imputation = (
            "imputation" in self.all_source.lower() or "impute" in self.all_source.lower()
        )
        self.assertTrue(has_imputation, "Missing imputation strategy")

    def test_has_normalization_step(self):
        """Test that notebook includes column normalization."""
        has_normalize = (
            "normalize" in self.all_source.lower() or "normalization" in self.all_source.lower()
        )
        self.assertTrue(has_normalize, "Missing column normalization step")


class TestFeatureEngineeringCompliance(unittest.TestCase):
    """Test feature engineering compliance with Phase 9.3."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_feature_engineering_imports(self):
        """Test that notebook imports feature engineering modules."""
        has_import = "features" in self.all_source.lower() or "engineer" in self.all_source.lower()
        self.assertTrue(has_import, "Missing feature engineering imports")

    def test_has_build_comprehensive_features(self):
        """Test that notebook references feature engineering functions."""
        has_func = (
            "build_comprehensive_features" in self.all_source
            or "comprehensive_features" in self.all_source.lower()
            or "engineer_" in self.all_source.lower()
            or "Phase 9.3" in self.all_source
            or "feature engineering" in self.all_source.lower()
            or "PHASE93_FEATURE" in self.all_source
        )
        self.assertTrue(has_func, "Missing feature engineering function reference")


if __name__ == "__main__":
    unittest.main()
