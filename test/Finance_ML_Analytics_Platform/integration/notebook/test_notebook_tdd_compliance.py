"""
Test suite for notebook TDD compliance.
Validates that ml_finance_model_main.ipynb follows code_guidelines.md standards.

This test suite covers 7 major violation categories:
1. Configuration constants redefinition
2. DataFrame state mutations without tracking
3. Magic numbers and hardcoded thresholds
4. Schema compliance violations
5. Missing validation cells
6. Insufficient error handling
7. Test coverage gaps

Reference: Issue TDD Refactoring Plan for ml_finance_model_main.ipynb
"""

import unittest
import json
import re
from pathlib import Path


class TestNotebookTDDCompliance(unittest.TestCase):
    """Validate notebook follows TDD best practices."""

    @classmethod
    def setUpClass(cls):
        """Load notebook once for all tests."""
        notebook_path = Path("ml_finance_model_main.ipynb")
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")

        with open(notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.code_cells = [
            "".join(c["source"]) for c in cls.notebook["cells"] if c["cell_type"] == "code"
        ]
        cls.all_cells_text = "\n".join(cls.code_cells)

    # ========== Category 1: Configuration Constants Redefinition ==========

    def test_config_constants_defined_once(self):
        """Ensure config constants defined in single cell (Section 8.1 violation)."""
        constants_to_check = [
            "TARGET_COL",
            "TARGET_COL_FALLBACK",
            "TEST_SIZE",
            "CV_FOLDS",
            "QUANTILES",
            "MIN_SECTOR_SAMPLES",
        ]

        for const in constants_to_check:
            pattern = rf"{const}\s*="
            matches = [i for i, cell in enumerate(self.code_cells) if re.search(pattern, cell)]

            # Allow definition in cell 0 (config) and potential redefinition in validation
            # But flag if defined multiple times in different contexts
            if len(matches) > 2:
                self.fail(
                    f"{const} defined {len(matches)} times in cells {matches}. "
                    f"Should be defined once in config cell."
                )

    def test_no_hardcoded_test_size(self):
        """Ensure no hardcoded test_size=0.2 (use TEST_SIZE constant)."""
        violations = []
        # Skip first few cells (config)
        for i, cell in enumerate(self.code_cells[5:], start=5):
            # Look for hardcoded test_size=0.2 or test_size = 0.2
            if re.search(r"test_size\s*=\s*0\.2\b", cell):
                # Check if it's using the constant TEST_SIZE
                if "TEST_SIZE" not in cell:
                    violations.append(f"Cell {i}")

        self.assertEqual(
            len(violations),
            0,
            f"Found hardcoded test_size=0.2 (should use TEST_SIZE) in cells: {violations}",
        )

    def test_no_hardcoded_train_size(self):
        """Ensure no hardcoded split calculations like 0.8 (use TRAIN_SIZE or 1-TEST_SIZE)."""
        violations = []
        # Look for patterns like int(len(X) * 0.8) or split_idx = int(0.8 * ...)
        for i, cell in enumerate(self.code_cells[5:], start=5):
            if re.search(r"\*\s*0\.8\b", cell) or re.search(r"0\.8\s*\*", cell):
                # Exclude correlation matrix construction (legitimate math constant)
                if "corr_matrix" in cell or "correlation" in cell.lower():
                    continue
                if "TRAIN_SIZE" not in cell and "TEST_SIZE" not in cell:
                    violations.append(f"Cell {i}")

        if violations:
            self.fail(
                f"Found hardcoded train size 0.8 (should use 1-TEST_SIZE or TRAIN_SIZE) "
                f"in cells: {violations}"
            )

    # ========== Category 2: DataFrame State Mutations ==========

    def test_dataframe_stage_naming_present(self):
        """Validate DataFrame uses stage-based naming (Section 8.4 violation)."""
        expected_stage_names = [
            "all_stocks_raw",  # Initial load
            "all_stocks_normalized",  # After normalization
            "all_stocks_typed",  # After type detection
            "all_stocks_winsorized",  # After winsorization
            "all_stocks_imputed",  # After imputation
            "all_stocks_scaled",  # After scaling
            "all_stocks_features",  # After feature engineering
            "all_stocks_enhanced",  # After classification features
        ]

        missing = []
        for name in expected_stage_names:
            if name not in self.all_cells_text:
                missing.append(name)

        self.assertEqual(
            len(missing),
            0,
            f"Missing stage-based DataFrame names: {missing}. "
            f"Use descriptive stage names instead of in-place mutations.",
        )

    def test_no_excessive_inplace_mutations(self):
        """Ensure minimal in-place DataFrame mutations (prefer stage-based naming)."""
        # Look for patterns like all_stocks = some_function(all_stocks)
        # after the initial stages (allow in early preprocessing)
        violations = []

        for i, cell in enumerate(self.code_cells[15:], start=15):  # After preprocessing
            # Pattern: all_stocks = ... all_stocks ...
            if re.search(r"all_stocks\s*=\s*\w+\([^)]*all_stocks", cell):
                # Check if it's creating a new stage name
                if not re.search(r"all_stocks_\w+\s*=", cell):
                    violations.append(f"Cell {i}")

        # Allow some mutations, but flag if excessive (>10 after preprocessing)
        if len(violations) > 10:
            self.fail(
                f"Excessive in-place mutations of 'all_stocks' found in {len(violations)} cells. "
                f"Use stage-based naming (e.g., all_stocks_processed) instead."
            )

    # ========== Category 3: Magic Numbers ==========

    def test_no_hardcoded_random_state_42(self):
        """Ensure no hardcoded random_state=42 (use RANDOM_SEED) (Section 9 violation)."""
        violations = []
        for i, cell in enumerate(self.code_cells[5:], start=5):
            if re.search(r"random_state\s*=\s*42\b", cell) or re.search(
                r"random_seed\s*=\s*42\b", cell
            ):
                # Allow if defining RANDOM_SEED constant
                if "RANDOM_SEED" not in cell or "= 42" not in cell:
                    violations.append(f"Cell {i}")

        self.assertEqual(
            len(violations),
            0,
            f"Found hardcoded random_state=42 (should use RANDOM_SEED) in cells: {violations}",
        )

    def test_max_sector_weight_defined(self):
        """Ensure MAX_SECTOR_WEIGHT constant is defined (not hardcoded 0.25)."""
        # Check if MAX_SECTOR_WEIGHT is defined
        if "MAX_SECTOR_WEIGHT" not in self.all_cells_text:
            self.fail("MAX_SECTOR_WEIGHT constant not defined in config cell")

        # Check for hardcoded 0.25 in portfolio/sector context
        violations = []
        for i, cell in enumerate(self.code_cells[5:], start=5):
            if "max_sector_weight" in cell.lower() or "sector_weight" in cell.lower():
                if re.search(r"=\s*0\.25\b", cell) and "MAX_SECTOR_WEIGHT" not in cell:
                    violations.append(f"Cell {i}")

        if violations:
            self.fail(
                f"Found hardcoded sector weight 0.25 (should use MAX_SECTOR_WEIGHT) "
                f"in cells: {violations}"
            )

    def test_quantiles_constant_used(self):
        """Ensure QUANTILES constant used instead of hardcoded [0.05, 0.5, 0.95]."""
        violations = []
        for i, cell in enumerate(self.code_cells[5:], start=5):
            # Look for hardcoded quantile lists
            if re.search(r"\[0\.05,?\s*0\.5,?\s*0\.95\]", cell) or re.search(
                r"\[0\.1,?\s*0\.5,?\s*0\.9\]", cell
            ):
                if "QUANTILES" not in cell:
                    violations.append(f"Cell {i}")

        if violations:
            self.fail(
                f"Found hardcoded quantile lists (should use QUANTILES constant) "
                f"in cells: {violations}"
            )

    def test_iqr_multiplier_defined(self):
        """Ensure IQR_MULTIPLIER constant defined for outlier detection."""
        # Check if defined
        if "IQR_MULTIPLIER" not in self.all_cells_text:
            # Allow if not using IQR method
            if "iqr" in self.all_cells_text.lower() and "1.5" in self.all_cells_text:
                self.fail("IQR method used but IQR_MULTIPLIER constant not defined")

    # ========== Category 4: Schema Compliance ==========

    def test_no_hardcoded_price_target_strings(self):
        """Ensure no hardcoded 'price_target' strings (use TARGET_COL) (Section 2.2 violation)."""
        violations = []
        for i, cell in enumerate(self.code_cells[10:], start=10):  # Skip config cells
            # Look for literal "price_target" or 'price_target'
            if re.search(r'["\']price_target["\']', cell):
                # Allow if it's in the TARGET_COL definition or comments
                if "TARGET_COL" not in cell and not cell.strip().startswith("#"):
                    violations.append(f"Cell {i}")

        self.assertEqual(
            len(violations),
            0,
            f"Found hardcoded 'price_target' strings (should use TARGET_COL) in cells: {violations}",
        )

    def test_no_hardcoded_last_price_strings(self):
        """Ensure no hardcoded 'last_price' strings (use TARGET_COL_FALLBACK)."""
        violations = []
        for i, cell in enumerate(self.code_cells[10:], start=10):
            if re.search(r'["\']last_price["\']', cell):
                # Allow in TARGET_COL_FALLBACK definition
                if "TARGET_COL_FALLBACK" not in cell and not cell.strip().startswith("#"):
                    violations.append(f"Cell {i}")

        self.assertEqual(
            len(violations),
            0,
            f"Found hardcoded 'last_price' strings (should use TARGET_COL_FALLBACK) in cells: {violations}",
        )

    def test_target_column_selection_pattern(self):
        """Ensure proper target column selection pattern is used."""
        # Look for the good pattern: TARGET_COL if TARGET_COL in df.columns else TARGET_COL_FALLBACK
        good_pattern_found = False
        for cell in self.code_cells:
            if "TARGET_COL if TARGET_COL in" in cell and "else TARGET_COL_FALLBACK" in cell:
                good_pattern_found = True
                break

        # If we're using target columns, we should use the pattern
        if (
            "price_target" in self.all_cells_text.lower()
            or "last_price" in self.all_cells_text.lower()
        ):
            self.assertTrue(
                good_pattern_found,
                "Should use pattern: TARGET_COL if TARGET_COL in df.columns else TARGET_COL_FALLBACK",
            )

    # ========== Category 5: Validation Cells ==========

    def test_validation_cells_present(self):
        """Ensure validation cells exist after major sections (Section 8.5 violation)."""
        validation_markers = ["VALIDATION:", "Section 2", "Section 4", "Section 6"]

        # Check if validation pattern exists
        validation_cells_found = 0
        for cell in self.code_cells:
            if "VALIDATION" in cell.upper() and "Section" in cell:
                validation_cells_found += 1

        self.assertGreater(
            validation_cells_found,
            0,
            "No validation cells found. Add validation checkpoints after major sections.",
        )

    def test_preprocessing_validation_present(self):
        """Ensure preprocessing stage has validation."""
        # Look for validation after preprocessing
        found_preprocessing_validation = False
        for cell in self.code_cells:
            if "preprocessing" in cell.lower() and (
                "validation" in cell.lower() or "assert" in cell.lower()
            ):
                if "all_stocks_scaled" in cell or "all_stocks_imputed" in cell:
                    found_preprocessing_validation = True
                    break

        if not found_preprocessing_validation:
            # Check if at least basic assertions exist
            basic_checks = any(
                "assert" in cell and "all_stocks" in cell for cell in self.code_cells
            )
            self.assertTrue(
                basic_checks,
                "Missing preprocessing validation. Add checks for DataFrame existence and shape.",
            )

    def test_output_artifact_validation_present(self):
        """Ensure output artifacts are validated."""
        # Look for validation of output files
        output_validation = False
        for cell in self.code_cells:
            if "OUTPUT_DIR" in cell or "output" in cell.lower():
                if "exists()" in cell or ".stat()" in cell or "assert" in cell:
                    output_validation = True
                    break

        # This is a recommendation, not strict requirement
        if not output_validation:
            print("INFO: Consider adding output artifact validation (file existence checks)")

    # ========== Category 6: Error Handling ==========

    def test_dataframe_existence_checks(self):
        """Ensure critical DataFrames have existence checks (Section 4 violation)."""
        # Look for pattern: if 'variable' not in globals()
        existence_checks = []
        for i, cell in enumerate(self.code_cells):
            if "not in globals()" in cell or "in globals()" in cell:
                existence_checks.append(i)

        # We should have at least some existence checks
        self.assertGreater(
            len(existence_checks),
            0,
            "No DataFrame existence checks found. Add defensive checks for critical variables.",
        )

    def test_error_messages_are_descriptive(self):
        """Ensure error messages provide context."""
        # Look for raise statements with meaningful messages
        error_raises = []
        for cell in self.code_cells:
            if "raise ValueError" in cell or "raise RuntimeError" in cell:
                # Check if message has context (contains "❌" or descriptive text)
                if "❌" in cell or "Run" in cell or "not found" in cell.lower():
                    error_raises.append(True)
                else:
                    error_raises.append(False)

        if error_raises:
            descriptive_ratio = sum(error_raises) / len(error_raises)
            self.assertGreater(
                descriptive_ratio,
                0.5,
                f"Only {descriptive_ratio*100:.0f}% of error messages are descriptive. Add context.",
            )

    # ========== Category 7: Overall Structure ==========

    def test_config_cell_is_first_code_cell(self):
        """Ensure configuration cell is the first code cell."""
        first_cell = self.code_cells[0] if self.code_cells else ""

        # First cell should define config constants
        config_indicators = ["TARGET_COL", "TEST_SIZE", "CV_FOLDS", "QUANTILES"]
        config_count = sum(1 for indicator in config_indicators if indicator in first_cell)

        self.assertGreater(
            config_count,
            2,
            f"First code cell should define configuration constants. Found {config_count}/4 indicators.",
        )

    def test_imports_are_present(self):
        """Ensure necessary imports are present."""
        required_imports = ["pandas", "numpy", "pathlib", "finance_ml"]

        missing_imports = []
        for imp in required_imports:
            if imp not in self.all_cells_text:
                missing_imports.append(imp)

        self.assertEqual(len(missing_imports), 0, f"Missing required imports: {missing_imports}")

    def test_notebook_has_reasonable_cell_count(self):
        """Ensure notebook isn't excessively long (maintainability check)."""
        code_cell_count = len(self.code_cells)

        # Issue mentions 113 cells (90 code, 23 markdown)
        # After refactoring with validation cells, expect ~102 cells
        self.assertLess(
            code_cell_count,
            150,
            f"Notebook has {code_cell_count} code cells. Consider modularizing.",
        )

    # ========== Configuration Completeness ==========

    def test_all_required_config_constants_defined(self):
        """Ensure all required configuration constants are defined."""
        required_constants = {
            "TARGET_COL": "Target column name",
            "TARGET_COL_FALLBACK": "Fallback target column",
            "TEST_SIZE": "Train/test split ratio",
            "TRAIN_SIZE": "Training set size (or computed from TEST_SIZE)",
            "CV_FOLDS": "Cross-validation folds",
            "QUANTILES": "Quantile regression levels",
            "MIN_SECTOR_SAMPLES": "Minimum samples per sector",
            "RANDOM_SEED": "Random seed for reproducibility",
        }

        missing = []
        for const, description in required_constants.items():
            if const not in self.all_cells_text:
                # TRAIN_SIZE can be computed as 1 - TEST_SIZE
                if const == "TRAIN_SIZE" and "1 - TEST_SIZE" in self.all_cells_text:
                    continue
                missing.append(f"{const} ({description})")

        self.assertEqual(len(missing), 0, f"Missing required configuration constants: {missing}")

    def test_additional_constants_recommended(self):
        """Check for recommended additional constants."""
        recommended = {
            "MAX_SECTOR_WEIGHT": "Maximum portfolio weight per sector",
            "MAX_SINGLE_POSITION": "Maximum weight for single position",
            "IQR_MULTIPLIER": "IQR multiplier for outlier detection",
            "ZSCORE_THRESHOLD": "Z-score threshold for outliers",
            "WINSORIZE_LOWER": "Lower percentile for winsorization",
            "WINSORIZE_UPPER": "Upper percentile for winsorization",
        }

        missing = []
        for const, description in recommended.items():
            if const not in self.all_cells_text:
                missing.append(f"{const} ({description})")

        if missing:
            print(f"\nINFO: Consider adding recommended constants: {missing}")


class TestNotebookExecutionOrder(unittest.TestCase):
    """Test that notebook cells have proper execution order dependencies."""

    @classmethod
    def setUpClass(cls):
        """Load notebook."""
        notebook_path = Path("ml_finance_model_main.ipynb")
        with open(notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.code_cells = [
            "".join(c["source"]) for c in cls.notebook["cells"] if c["cell_type"] == "code"
        ]

    def test_config_before_data_loading(self):
        """Ensure configuration is defined before data loading."""
        config_cell_idx = None
        data_load_cell_idx = None

        for i, cell in enumerate(self.code_cells):
            if "TARGET_COL" in cell and config_cell_idx is None:
                config_cell_idx = i
            if "load_from_csv" in cell or "load_from_db" in cell:
                if data_load_cell_idx is None:
                    data_load_cell_idx = i

        if config_cell_idx is not None and data_load_cell_idx is not None:
            self.assertLess(
                config_cell_idx,
                data_load_cell_idx,
                "Configuration must be defined before data loading",
            )

    def test_preprocessing_before_feature_engineering(self):
        """Ensure preprocessing happens before feature engineering."""
        preprocessing_idx = None
        feature_eng_idx = None

        for i, cell in enumerate(self.code_cells):
            if "imputation" in cell.lower() or "winsorize" in cell.lower():
                if preprocessing_idx is None:
                    preprocessing_idx = i
            if "build_advanced_features" in cell or "compute_financial_ratios" in cell:
                if feature_eng_idx is None:
                    feature_eng_idx = i

        if preprocessing_idx is not None and feature_eng_idx is not None:
            self.assertLess(
                preprocessing_idx,
                feature_eng_idx,
                "Preprocessing must happen before feature engineering",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
