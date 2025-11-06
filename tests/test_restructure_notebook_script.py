"""
Test suite for restructure_notebook.py transformation functions (TDD approach).

Tests the actual restructuring transformations:
1. Phase 9.5 detection and imputation code injection
2. Header standardization
3. Validation gate insertion
4. Duplicate removal
5. Phase consolidation

Following strict TDD: These tests should FAIL initially, then we fix the script.
"""

import json
import unittest
from pathlib import Path
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import restructure_notebook as rn


class TestPhase95Detection(unittest.TestCase):
    """Test Phase 9.5 section detection - currently FAILS."""

    def test_update_phase95_finds_section(self):
        """Test that update_phase95_imputation finds Phase 9.5 section."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Phase 9.5 — Sector-Optimized Regression Models\n", "\n", "Train sector-specific models."]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": ["from finance_ml.advanced_models import prepare_regression_data\n"]
                }
            ]
        }
        
        # This should NOT fail to find Phase 9.5
        result = rn.update_phase95_imputation(notebook)
        
        # The function should have found Phase 9.5 and attempted to update
        # (even if imputation cell not found, it shouldn't fail on Phase 9.5 detection)
        self.assertIsNotNone(result)
        self.assertIn("cells", result)

    def test_update_phase95_without_sector_keyword(self):
        """Test Phase 9.5 detection even without 'Sector-Optimized' in same line."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Phase 9.5 — Regression Models\n"]  # Missing "Sector-Optimized"
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": ["from finance_ml.advanced_models import prepare_regression_data\n"]
                }
            ]
        }
        
        # Should still find Phase 9.5 (search should be flexible)
        result = rn.update_phase95_imputation(notebook)
        self.assertIsNotNone(result)

    def test_update_phase95_finds_imputation_cell(self):
        """Test finding code cell with prepare_regression_data."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Phase 9.5 — Sector-Optimized Regression Models\n"]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Phase 9.5: Sector-Optimized Regression Models\n",
                        "from finance_ml.advanced_models import prepare_regression_data\n",
                        "X_train, y_train = prepare_regression_data(df)\n"
                    ]
                }
            ]
        }
        
        # Should find and update the code cell
        result = rn.update_phase95_imputation(notebook)
        
        # Check if imputation code was injected
        code_cell = result["cells"][1]
        source_str = "".join(code_cell["source"])
        
        # Should contain 4-step imputation import
        self.assertIn("apply_enhanced_imputation_strategy_4step", source_str)


class TestHeaderStandardization(unittest.TestCase):
    """Test header standardization - currently returns 0 updates."""

    def test_standardize_headers_with_hyphen(self):
        """Test standardizing Phase header with hyphen to em-dash."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Phase 9.5 - Regression Models\n"]  # hyphen
                }
            ]
        }
        
        result = rn.standardize_headers(notebook)
        
        # Should update to em-dash
        updated_source = "".join(result["cells"][0]["source"])
        self.assertIn("—", updated_source, "Should convert hyphen to em-dash")
        self.assertNotIn(" - ", updated_source, "Should not have hyphen separator")

    def test_standardize_headers_with_endash(self):
        """Test standardizing Phase header with en-dash to em-dash."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Phase 9.6 – Model Evaluation\n"]  # en-dash
                }
            ]
        }
        
        result = rn.standardize_headers(notebook)
        
        updated_source = "".join(result["cells"][0]["source"])
        self.assertIn("—", updated_source, "Should convert en-dash to em-dash")

    def test_standardize_subsection_headers(self):
        """Test standardizing sub-section headers."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["### Phase 9.5.1 - Model Optimization\n"]
                }
            ]
        }
        
        result = rn.standardize_headers(notebook)
        
        updated_source = "".join(result["cells"][0]["source"])
        self.assertIn("—", updated_source)
        self.assertEqual("### Phase 9.5.1 — Model Optimization\n", updated_source)

    def test_standardize_multiple_headers(self):
        """Test standardizing multiple headers in one notebook."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.5 - Regression\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["### Phase 9.5.1 - Optimization\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.6 — Already Good\n"]},  # already em-dash
            ]
        }
        
        result = rn.standardize_headers(notebook)
        
        # Check all cells
        for i, expected_has_emdash in enumerate([True, True, True]):
            source = "".join(result["cells"][i]["source"])
            if expected_has_emdash:
                self.assertIn("—", source, f"Cell {i} should have em-dash")

    def test_no_change_if_already_standardized(self):
        """Test that already-correct headers aren't modified."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Phase 9.5 — Sector-Optimized Regression Models\n"]
                }
            ]
        }
        
        original_source = "".join(notebook["cells"][0]["source"])
        result = rn.standardize_headers(notebook)
        updated_source = "".join(result["cells"][0]["source"])
        
        self.assertEqual(original_source, updated_source, "Should not change already-correct header")


class TestValidationGateInsertion(unittest.TestCase):
    """Test validation gate insertion."""

    def test_add_validation_gate_finds_compare_regressors(self):
        """Test finding compare_regressors for gate insertion."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.5\n"]},
                {"cell_type": "code", "metadata": {}, "source": ["from finance_ml.advanced_models import compare_regressors\n"]},
                {"cell_type": "code", "metadata": {}, "source": ["results = compare_regressors(X, y)\n"]},
            ]
        }
        
        original_count = len(notebook["cells"])
        result = rn.add_validation_gates(notebook)
        final_count = len(result["cells"])
        
        # Should have inserted one validation gate
        self.assertEqual(final_count, original_count + 1, "Should insert 1 validation cell")

    def test_validation_gate_has_required_code(self):
        """Test that validation gate contains correct code."""
        notebook = {
            "cells": [
                {"cell_type": "code", "metadata": {}, "source": ["compare_regressors(X, y)\n"]},
            ]
        }
        
        result = rn.add_validation_gates(notebook)
        
        # Find the validation cell (should be inserted before compare_regressors)
        validation_cell = result["cells"][0]
        source_str = "".join(validation_cell["source"])
        
        self.assertIn("validate_training_data", source_str)
        self.assertIn("VALIDATION GATE", source_str)


class TestDuplicateRemoval(unittest.TestCase):
    """Test duplicate cell removal."""

    def test_remove_duplicate_phase93_removes_correct_cells(self):
        """Test that remove_duplicate_phase93 removes duplicate Phase 9.3 sections."""
        # Create notebook with Phase 9.3 appearing twice
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.1\n"]},
                {"cell_type": "code", "metadata": {}, "source": ["# Phase 9.1 code\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.2\n"]},
                {"cell_type": "code", "metadata": {}, "source": ["# Phase 9.2 code\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.3 — Feature Engineering\n"]},  # First Phase 9.3
                {"cell_type": "code", "metadata": {}, "source": ["# Phase 9.3 code cell 1\n"]},
                {"cell_type": "code", "metadata": {}, "source": ["# Phase 9.3 code cell 2\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.4\n"]},
                {"cell_type": "code", "metadata": {}, "source": ["# Phase 9.4 code\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.3 — Feature Engineering\n"]},  # Duplicate!
                {"cell_type": "code", "metadata": {}, "source": ["# Phase 9.3 duplicate code 1\n"]},
                {"cell_type": "code", "metadata": {}, "source": ["# Phase 9.3 duplicate code 2\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.5\n"]},
            ]
        }
        
        original_count = len(notebook["cells"])
        result = rn.remove_duplicate_phase93(notebook)
        final_count = len(result["cells"])
        
        # Should remove the duplicate Phase 9.3 section (3 cells: marker + 2 code cells)
        expected_removed = 3
        actual_removed = original_count - final_count
        
        self.assertEqual(actual_removed, expected_removed, f"Should remove {expected_removed} duplicate cells")
        
        # Verify Phase 9.5 is still there (not removed)
        phase95_found = any("Phase 9.5" in "".join(c.get("source", [])) 
                           for c in result["cells"] if c["cell_type"] == "markdown")
        self.assertTrue(phase95_found, "Phase 9.5 should not be removed")


class TestPhaseConsolidation(unittest.TestCase):
    """Test phase consolidation functions."""

    def test_consolidate_phase97_with_duplicates(self):
        """Test consolidating duplicate Phase 9.7 sections."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.7 — Identification\n"]},
                {"cell_type": "code", "metadata": {}, "source": ["# code\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["### Phase 9.7 Enhanced — More stuff\n"]},
                {"cell_type": "code", "metadata": {}, "source": ["# more code\n"]},
            ]
        }
        
        original_count = len(notebook["cells"])
        result = rn.consolidate_phase97(notebook)
        final_count = len(result["cells"])
        
        # Should remove the "Enhanced" duplicate
        self.assertLess(final_count, original_count, "Should remove duplicate Phase 9.7")

    def test_consolidate_phase98_with_duplicates(self):
        """Test consolidating duplicate Phase 9.8 sections."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.8 — Comprehensive Analytics\n"]},
                {"cell_type": "code", "metadata": {}, "source": ["# code\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.8 — Advanced Model Evaluation\n"]},
            ]
        }
        
        original_count = len(notebook["cells"])
        result = rn.consolidate_phase98(notebook)
        final_count = len(result["cells"])
        
        # Should keep only one Phase 9.8
        self.assertLess(final_count, original_count, "Should remove duplicate Phase 9.8")


class TestNotebookLoadingSaving(unittest.TestCase):
    """Test notebook loading and saving functions."""

    def test_load_notebook_valid_json(self):
        """Test loading a valid JSON notebook."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False, encoding='utf-8') as f:
            notebook = {
                "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Test\n"]}],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 4
            }
            json.dump(notebook, f)
            temp_path = Path(f.name)
        
        try:
            loaded = rn.load_notebook(temp_path)
            self.assertIn("cells", loaded)
            self.assertEqual(len(loaded["cells"]), 1)
        finally:
            temp_path.unlink()

    def test_save_notebook(self):
        """Test saving a notebook."""
        notebook = {
            "cells": [{"cell_type": "code", "metadata": {}, "source": ["print('test')\n"]}],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            rn.save_notebook(notebook, temp_path)
            self.assertTrue(temp_path.exists())
            
            # Verify it's valid JSON
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                self.assertIn("cells", loaded)
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for complete restructuring workflow."""

    def test_full_restructuring_workflow(self):
        """Test running all transformations in sequence."""
        # Create a minimal notebook with issues
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["## Phase 9.5 - Regression\n"]},  # needs standardization
                {"cell_type": "code", "metadata": {}, "source": ["prepare_regression_data(df)\n"]},
                {"cell_type": "code", "metadata": {}, "source": ["compare_regressors(X, y)\n"]},  # needs validation gate
            ]
        }
        
        original_count = len(notebook["cells"])
        
        # Apply all transformations
        notebook = rn.update_phase95_imputation(notebook)
        notebook = rn.add_validation_gates(notebook)
        notebook = rn.standardize_headers(notebook)
        
        final_count = len(notebook["cells"])
        
        # Should have added validation gate
        self.assertGreater(final_count, original_count, "Should add validation gate")
        
        # Check header was standardized
        header_cell = notebook["cells"][0]
        source = "".join(header_cell["source"])
        self.assertIn("—", source, "Should have em-dash")


if __name__ == "__main__":
    unittest.main()
