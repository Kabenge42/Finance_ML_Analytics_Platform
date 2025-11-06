"""
Unit tests for restructure_notebook.py functions - TDD approach

This module tests the actual restructuring FUNCTIONS, not the notebook content.
Following strict TDD: write failing tests first, implement fixes, refactor.

Tests cover:
- remove_duplicate_phase93()
- reorder_phase95_96()
- consolidate_phase97()
- consolidate_phase98()
- update_phase95_imputation()
- add_validation_gates()
- standardize_headers()
- load_notebook() / save_notebook()
"""

import json
import unittest
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path for import
sys.path.insert(0, str(Path(__file__).parent.parent))

import restructure_notebook as rn


class TestNotebookLoadingSaving(unittest.TestCase):
    """Test notebook loading and saving functionality."""

    def setUp(self):
        """Create a minimal test notebook."""
        self.test_notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Test Notebook"], "metadata": {}},
                {
                    "cell_type": "code",
                    "source": ["print('hello')"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

    def test_load_notebook_success(self):
        """Test loading a notebook from file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ipynb", delete=False, encoding="utf-8"
        ) as f:
            json.dump(self.test_notebook, f)
            temp_path = f.name

        try:
            loaded = rn.load_notebook(temp_path)
            self.assertEqual(len(loaded["cells"]), 2)
            self.assertEqual(loaded["cells"][0]["cell_type"], "markdown")
        finally:
            os.unlink(temp_path)

    def test_save_notebook_success(self):
        """Test saving a notebook to file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ipynb", delete=False, encoding="utf-8"
        ) as f:
            temp_path = f.name

        try:
            rn.save_notebook(self.test_notebook, temp_path)
            self.assertTrue(Path(temp_path).exists())

            loaded = rn.load_notebook(temp_path)
            self.assertEqual(len(loaded["cells"]), 2)
        finally:
            if Path(temp_path).exists():
                os.unlink(temp_path)


class TestRemoveDuplicatePhase93(unittest.TestCase):
    """Test remove_duplicate_phase93() function."""

    def test_removes_cells_111_to_127_inclusive(self):
        """Test that cells 111-127 (17 cells) are removed."""
        # Create notebook with 130 cells
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": [f"Cell {i}"], "metadata": {}}
                for i in range(130)
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.remove_duplicate_phase93(notebook)

        # Should remove 17 cells (111-127 inclusive)
        self.assertEqual(len(result["cells"]), 130 - 17)
        self.assertEqual(len(result["cells"]), 113)

    def test_handles_short_notebook_gracefully(self):
        """Test that function handles notebooks with fewer than 111 cells."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Cell"], "metadata": {}} for i in range(50)
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.remove_duplicate_phase93(notebook)

        # Should not crash, original cells unchanged
        self.assertEqual(len(result["cells"]), 50)

    def test_returns_modified_notebook(self):
        """Test that function returns the modified notebook."""
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["test"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                }
            ]
            * 130,
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.remove_duplicate_phase93(notebook)

        self.assertIsInstance(result, dict)
        self.assertIn("cells", result)


class TestPhaseReordering(unittest.TestCase):
    """Test reorder_phase95_96() function."""

    def test_detects_phase_markers_correctly(self):
        """Test that phase markers are detected in notebook."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.5 — Sector-Optimized Regression Models"],
                    "metadata": {},
                },
                {
                    "cell_type": "code",
                    "source": ["# 9.5 code"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.5.1 — Model Optimization Enhancements"],
                    "metadata": {},
                },
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.6 — Model Evaluation and Error Analysis"],
                    "metadata": {},
                },
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.6.1 — Enhanced Error Analysis"],
                    "metadata": {},
                },
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.7 — Identification"],
                    "metadata": {},
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        # Run the function (should not crash)
        result = rn.reorder_phase95_96(notebook)

        # Should return a notebook
        self.assertIsInstance(result, dict)
        self.assertIn("cells", result)

    def test_no_reordering_when_correct_order(self):
        """Test that correct order is not modified."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.5 — Sector-Optimized Regression Models"],
                    "metadata": {},
                },
                {
                    "cell_type": "code",
                    "source": ["# 9.5 code"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.5.1 — Model Optimization"],
                    "metadata": {},
                },
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.6 — Model Evaluation and Error Analysis"],
                    "metadata": {},
                },
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.6.1 — Enhanced Error Analysis"],
                    "metadata": {},
                },
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.7 — Identification"],
                    "metadata": {},
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        original_count = len(notebook["cells"])
        result = rn.reorder_phase95_96(notebook)

        # Cell count should not change
        self.assertEqual(len(result["cells"]), original_count)

    def test_handles_missing_markers(self):
        """Test that function handles missing phase markers gracefully."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["## Phase 9.1"], "metadata": {}},
                {"cell_type": "markdown", "source": ["## Phase 9.2"], "metadata": {}},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.reorder_phase95_96(notebook)

        # Should not crash
        self.assertEqual(len(result["cells"]), 2)


class TestConsolidatePhase97(unittest.TestCase):
    """Test consolidate_phase97() function."""

    def test_finds_multiple_phase97_sections(self):
        """Test that function detects multiple Phase 9.7 sections."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.7 — Identification of Under/Overvalued Stocks"],
                    "metadata": {},
                },
                {
                    "cell_type": "code",
                    "source": ["# 9.7 code"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.7 Enhanced — Additional Analysis"],
                    "metadata": {},
                },
                {
                    "cell_type": "code",
                    "source": ["# enhanced code"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.consolidate_phase97(notebook)

        # Should remove duplicate Phase 9.7 headers
        markdown_cells = [c for c in result["cells"] if c["cell_type"] == "markdown"]
        phase97_cells = [c for c in markdown_cells if "Phase 9.7" in "".join(c["source"])]

        # Should have fewer Phase 9.7 headers after consolidation
        self.assertLessEqual(len(phase97_cells), 1)

    def test_handles_single_phase97_section(self):
        """Test that function handles single Phase 9.7 section correctly."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.7 — Identification"],
                    "metadata": {},
                },
                {
                    "cell_type": "code",
                    "source": ["# code"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.consolidate_phase97(notebook)

        # Should not modify single section
        self.assertEqual(len(result["cells"]), 2)

    def test_handles_no_phase97_sections(self):
        """Test that function handles notebooks without Phase 9.7."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["## Phase 9.5"], "metadata": {}},
                {"cell_type": "markdown", "source": ["## Phase 9.6"], "metadata": {}},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.consolidate_phase97(notebook)

        # Should not crash
        self.assertEqual(len(result["cells"]), 2)


class TestConsolidatePhase98(unittest.TestCase):
    """Test consolidate_phase98() function."""

    def test_finds_multiple_phase98_sections(self):
        """Test that function detects multiple Phase 9.8 sections."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.8 — Comprehensive Analytics"],
                    "metadata": {},
                },
                {
                    "cell_type": "code",
                    "source": ["# analytics"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.8 — Advanced Model Evaluation"],
                    "metadata": {},
                },
                {
                    "cell_type": "code",
                    "source": ["# evaluation"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.consolidate_phase98(notebook)

        # Should remove duplicate Phase 9.8 headers
        markdown_cells = [c for c in result["cells"] if c["cell_type"] == "markdown"]
        phase98_cells = [c for c in markdown_cells if "Phase 9.8" in "".join(c["source"])]

        # Should have only one Phase 9.8 header
        self.assertLessEqual(len(phase98_cells), 1)

    def test_handles_single_phase98_section(self):
        """Test that function handles single Phase 9.8 section correctly."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["## Phase 9.8 — Analytics"], "metadata": {}},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.consolidate_phase98(notebook)
        self.assertEqual(len(result["cells"]), 1)


class TestUpdatePhase95Imputation(unittest.TestCase):
    """Test update_phase95_imputation() function."""

    def test_finds_phase95_section(self):
        """Test that function finds Phase 9.5 section."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.5 — Sector-Optimized Regression Models"],
                    "metadata": {},
                },
                {
                    "cell_type": "code",
                    "source": ["# Handling missing values\ndf.fillna(0)"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.update_phase95_imputation(notebook)

        # Check that code was updated
        code_cell = result["cells"][1]
        code_source = "".join(code_cell["source"])

        # Should contain 4-step imputation code
        self.assertIn("apply_enhanced_imputation_strategy_4step", code_source)

    def test_handles_missing_phase95(self):
        """Test that function handles missing Phase 9.5 gracefully."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["## Phase 9.1"], "metadata": {}},
                {
                    "cell_type": "code",
                    "source": ["print('test')"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.update_phase95_imputation(notebook)

        # Should not crash, notebook unchanged
        self.assertEqual(len(result["cells"]), 2)

    def test_finds_imputation_code_with_fillna(self):
        """Test that function finds imputation code containing fillna."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.5 — Sector-Optimized Regression Models"],
                    "metadata": {},
                },
                {
                    "cell_type": "code",
                    "source": ["import pandas as pd"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Simple median imputation\nfor col in numeric_cols:\n    df[col] = df[col].fillna(df[col].median())"
                    ],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.update_phase95_imputation(notebook)

        # Should update the cell with fillna
        code_cells = [c for c in result["cells"] if c["cell_type"] == "code"]
        updated = any(
            "apply_enhanced_imputation_strategy_4step" in "".join(c["source"]) for c in code_cells
        )

        self.assertTrue(updated, "Should update imputation code to use 4-step strategy")


class TestAddValidationGates(unittest.TestCase):
    """Test add_validation_gates() function."""

    def test_adds_validation_before_compare_regressors(self):
        """Test that validation gate is inserted before model training."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["## Model Training"], "metadata": {}},
                {
                    "cell_type": "code",
                    "source": ["results = compare_regressors(X, y)"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.add_validation_gates(notebook)

        # Should add one validation cell
        self.assertEqual(len(result["cells"]), 3)

        # Check that validation cell was inserted
        code_cells = [c for c in result["cells"] if c["cell_type"] == "code"]
        validation_exists = any(
            "validate_training_data" in "".join(c["source"]) for c in code_cells
        )

        self.assertTrue(validation_exists, "Should add validation gate cell")

    def test_validation_contains_required_elements(self):
        """Test that validation cell contains required validation logic."""
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["results = compare_regression_models(X, y)"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.add_validation_gates(notebook)

        # Find validation cell
        code_cells = [c for c in result["cells"] if c["cell_type"] == "code"]
        validation_cell = next(
            (c for c in code_cells if "validate_training_data" in "".join(c["source"])), None
        )

        self.assertIsNotNone(validation_cell, "Validation cell should exist")

        validation_source = "".join(validation_cell["source"])
        self.assertIn("VALIDATION GATE", validation_source)
        self.assertIn("validate_training_data", validation_source)

    def test_handles_no_model_training_code(self):
        """Test that function handles notebooks without model training code."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["## EDA"], "metadata": {}},
                {
                    "cell_type": "code",
                    "source": ["df.describe()"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.add_validation_gates(notebook)

        # Should not add validation gate if no training code found
        self.assertEqual(len(result["cells"]), 2)


class TestStandardizeHeaders(unittest.TestCase):
    """Test standardize_headers() function."""

    def test_fixes_hyphen_to_em_dash(self):
        """Test that hyphens are replaced with proper em-dash."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.2 - Advanced EDA"],
                    "metadata": {},
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.standardize_headers(notebook)

        header = "".join(result["cells"][0]["source"])
        self.assertIn("—", header, "Should use em-dash (—)")

    def test_fixes_double_hyphen_to_em_dash(self):
        """Test that double hyphens are replaced with em-dash."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["### Phase 9.3 -- Feature Engineering"],
                    "metadata": {},
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.standardize_headers(notebook)

        header = "".join(result["cells"][0]["source"])
        self.assertIn("—", header)

    def test_preserves_non_phase_headers(self):
        """Test that non-phase headers are not modified."""
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Introduction"], "metadata": {}},
                {"cell_type": "markdown", "source": ["## Data Loading"], "metadata": {}},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.standardize_headers(notebook)

        header1 = "".join(result["cells"][0]["source"])
        header2 = "".join(result["cells"][1]["source"])

        self.assertEqual(header1, "# Introduction")
        self.assertEqual(header2, "## Data Loading")

    def test_standardizes_multiple_headers(self):
        """Test that multiple phase headers are standardized."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.1 - Preprocessing"],
                    "metadata": {},
                },
                {"cell_type": "markdown", "source": ["## Phase 9.2 -- EDA"], "metadata": {}},
                {
                    "cell_type": "markdown",
                    "source": ["## Phase 9.3 Feature Engineering"],
                    "metadata": {},
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        result = rn.standardize_headers(notebook)

        # All should use em-dash
        for cell in result["cells"]:
            if "Phase 9." in "".join(cell["source"]):
                header = "".join(cell["source"])
                self.assertIn("—", header, f"Header should use em-dash: {header}")


class TestEndToEndWorkflow(unittest.TestCase):
    """Integration tests for complete restructuring workflow."""

    def test_all_transformations_run_without_error(self):
        """Test that all transformation functions can be called in sequence."""
        # Create a realistic notebook structure
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Finance ML Notebook"], "metadata": {}},
            ]
            + [
                {
                    "cell_type": "code",
                    "source": [f"# Cell {i}"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                }
                for i in range(130)
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        # Run all transformations
        try:
            result = rn.remove_duplicate_phase93(notebook)
            result = rn.reorder_phase95_96(result)
            result = rn.consolidate_phase97(result)
            result = rn.consolidate_phase98(result)
            result = rn.update_phase95_imputation(result)
            result = rn.add_validation_gates(result)
            result = rn.standardize_headers(result)

            # Should complete without exceptions
            self.assertIsNotNone(result)
            self.assertIn("cells", result)
        except Exception as e:
            self.fail(f"Transformation pipeline failed with error: {e}")

    def test_main_function_with_missing_notebook(self):
        """Test main() function handles missing notebook file gracefully."""
        # Save current directory
        import sys

        original_argv = sys.argv.copy()

        try:
            # Test that main returns error code when notebook doesn't exist
            # Temporarily rename the notebook if it exists
            notebook_path = Path("ml_finance_model_main_backup.ipynb")
            backup_exists = notebook_path.exists()
            temp_backup = None

            if backup_exists:
                temp_backup = Path("ml_finance_model_main_backup.ipynb.temp_test_backup")
                notebook_path.rename(temp_backup)

            result = rn.main()

            # Should return 1 (error) when notebook doesn't exist
            self.assertEqual(result, 1)

        finally:
            # Restore notebook if it was backed up
            if temp_backup and temp_backup.exists():
                temp_backup.rename(notebook_path)
            sys.argv = original_argv

    def test_load_notebook_with_invalid_json(self):
        """Test load_notebook handles invalid JSON gracefully."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ipynb", delete=False, encoding="utf-8"
        ) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            # Should raise exception for invalid JSON
            with self.assertRaises(json.JSONDecodeError):
                rn.load_notebook(temp_path)
        finally:
            os.unlink(temp_path)

    def test_cell_count_reduction(self):
        """Test that restructuring reduces cell count by 17."""
        # Create notebook with 130 cells (includes cells 111-127 to be removed)
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [f"# Cell {i}"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                }
                for i in range(130)
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        original_count = len(notebook["cells"])

        # Apply just the duplicate removal
        result = rn.remove_duplicate_phase93(notebook)

        final_count = len(result["cells"])

        # Should have removed exactly 17 cells
        self.assertEqual(original_count - final_count, 17)
        self.assertEqual(final_count, 113)


if __name__ == "__main__":
    unittest.main()
