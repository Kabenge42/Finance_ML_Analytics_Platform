"""
Test-Driven Development tests for Phase 9 Notebook Integration.

This module tests the ml_finance_model_main.ipynb notebook structure,
ensuring all phases are present in the correct order with expected content.

Acceptance Criteria (from IMPROVEMENT_PLAN.md and NOTEBOOK_INTEGRATION_SUMMARY.md):
- All Phase 9 imports present (9.1-9.7)
- Correct workflow order: 9.1 ? 9.2 ? 9.3 ? 9.4 ? 9.5 ? 9.6 ? 9.7
- Phase 9.1: Advanced Preprocessing (outliers, winsorization, imputation)
- Phase 9.2: Enhanced EDA (after 9.1, not before)
- Phase 9.3: Advanced Feature Engineering (NEW - must be added)
- Phase 9.4: Multi-class Classification
- Phase 9.5: Sector-optimized Regression
- Phase 9.6: Model Evaluation and Error Analysis (NEW - must be added)
- Phase 9.7: Valuation and Stock Identification (NEW - must be added)
"""

import json
import re
import unittest
from pathlib import Path
from typing import Dict

# Import the reorganize_notebook module for coverage
from reorganize_notebook import NotebookCellFactory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "ml_finance_model_main.ipynb"


def parse_notebook(path: Path) -> Dict:
    """Parse notebook JSON and return structured data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cell_text(cell: Dict) -> str:
    """Extract text from a notebook cell."""
    source = cell.get("source", [])
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def find_phase_positions(nb_data: Dict) -> Dict[str, int]:
    """
    Find the cell index where each Phase 9.X section starts.
    Returns dict mapping phase name to cell index (0-based).
    """
    cells = nb_data.get("cells", [])
    phase_positions = {}

    # Phase markers to search for (Markdown headers)
    phase_patterns = {
        "9.1": r"##\s*Phase\s*9\.1",
        "9.2": r"##\s*Phase\s*9\.2",
        "9.3": r"##\s*Phase\s*9\.3",
        "9.4": r"##\s*Phase\s*9\.4",
        "9.5": r"##\s*Phase\s*9\.5",
        "9.6": r"##\s*Phase\s*9\.6",
        "9.7": r"##\s*Phase\s*9\.7",
    }

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            text = get_cell_text(cell)
            for phase, pattern in phase_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    if phase not in phase_positions:  # Take first occurrence
                        phase_positions[phase] = idx

    return phase_positions


def get_all_notebook_text(nb_data: Dict) -> str:
    """Get all text content from notebook for searching."""
    cells = nb_data.get("cells", [])
    return "\n".join(get_cell_text(cell) for cell in cells)


class TestNotebookIntegration(unittest.TestCase):
    """TDD tests for Phase 9 notebook integration."""

    @classmethod
    def setUpClass(cls):
        """Load notebook once for all tests."""
        assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
        cls.nb_data = parse_notebook(NOTEBOOK_PATH)
        cls.nb_text = get_all_notebook_text(cls.nb_data)
        cls.phase_positions = find_phase_positions(cls.nb_data)

    # Test 1: All Phase 9 imports present
    def test_phase_9_imports_present(self):
        """Test that all Phase 9 module imports are present."""
        required_imports = [
            # Phase 9.1 - Advanced Preprocessing
            "DataQualityReport",
            "detect_outliers_iqr",
            "winsorize_by_sector",
            "impute_missing_values",
            # Phase 9.2 - Enhanced EDA
            "simple_eda",
            "calculate_correlation_matrix",
            "generate_eda_report",
            # Phase 9.3 - Advanced Feature Engineering
            "engineer_valuation_ratios",
            "engineer_profitability_ratios",
            "build_comprehensive_features",
            "calculate_feature_importance_rf",
            # Phase 9.4 - Classification
            "create_event_labels",
            "train_event_classifier",
            # Phase 9.5 - Advanced Regression
            "train_ridge_regressor",
            "train_xgboost_regressor",
            "train_stacking_regressor",
            "train_quantile_regressor",
            # Phase 9.6 - Model Evaluation
            "comprehensive_regression_metrics",
            "compute_metrics_by_segment",
            "residual_analysis_suite",
            # Phase 9.7 - Valuation
            "assign_valuation_category",
            "calculate_mispricing_score",
            "rank_undervalued_stocks",
            "export_predictions_to_excel",
        ]

        for func_name in required_imports:
            self.assertIn(
                func_name, self.nb_text, f"Required import '{func_name}' not found in notebook"
            )

    # Test 2: Phase 9.1 exists
    def test_phase_9_1_exists(self):
        """Test that Phase 9.1 (Preprocessing) section exists."""
        self.assertIn("9.1", self.phase_positions, "Phase 9.1 section not found in notebook")

    # Test 3: Phase 9.2 exists
    def test_phase_9_2_exists(self):
        """Test that Phase 9.2 (EDA) section exists."""
        self.assertIn("9.2", self.phase_positions, "Phase 9.2 section not found in notebook")

    # Test 4: Phase 9.3 exists (SHOULD FAIL - not yet implemented)
    def test_phase_9_3_exists(self):
        """Test that Phase 9.3 (Feature Engineering) section exists."""
        self.assertIn(
            "9.3",
            self.phase_positions,
            "Phase 9.3 (Feature Engineering) section not found in notebook - MUST BE ADDED",
        )

    # Test 5: Phase 9.4 exists
    def test_phase_9_4_exists(self):
        """Test that Phase 9.4 (Classification) section exists."""
        self.assertIn("9.4", self.phase_positions, "Phase 9.4 section not found in notebook")

    # Test 6: Phase 9.5 exists
    def test_phase_9_5_exists(self):
        """Test that Phase 9.5 (Regression) section exists."""
        self.assertIn("9.5", self.phase_positions, "Phase 9.5 section not found in notebook")

    # Test 7: Phase 9.6 exists (SHOULD FAIL - not yet implemented)
    def test_phase_9_6_exists(self):
        """Test that Phase 9.6 (Evaluation) section exists."""
        self.assertIn(
            "9.6",
            self.phase_positions,
            "Phase 9.6 (Evaluation) section not found in notebook - MUST BE ADDED",
        )

    # Test 8: Phase 9.7 exists (SHOULD FAIL - not yet implemented)
    def test_phase_9_7_exists(self):
        """Test that Phase 9.7 (Valuation) section exists."""
        self.assertIn(
            "9.7",
            self.phase_positions,
            "Phase 9.7 (Valuation) section not found in notebook - MUST BE ADDED",
        )

    # Test 9: Phase ordering is correct (SHOULD FAIL - Phase 9.2 before 9.1)
    def test_phase_ordering_correct(self):
        """Test that phases appear in correct order: 9.1 ? 9.2 ? 9.3 ? 9.4 ? 9.5 ? 9.6 ? 9.7."""
        expected_order = ["9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7"]

        # Get positions of all present phases
        present_phases = [p for p in expected_order if p in self.phase_positions]

        # Check if present phases are in order
        if len(present_phases) < 2:
            self.skipTest("Not enough phases to test ordering")

        for i in range(len(present_phases) - 1):
            current_phase = present_phases[i]
            next_phase = present_phases[i + 1]
            current_pos = self.phase_positions[current_phase]
            next_pos = self.phase_positions[next_phase]

            self.assertLess(
                current_pos,
                next_pos,
                f"Phase {current_phase} (cell {current_pos}) must come before Phase {next_phase} (cell {next_pos})",
            )

    # Test 10: Phase 9.3 contains feature engineering content
    def test_phase_9_3_content(self):
        """Test that Phase 9.3 contains feature engineering functions."""
        if "9.3" not in self.phase_positions:
            self.skipTest("Phase 9.3 not present - will be added")

        # Get text from Phase 9.3 section onwards (until next phase or end)
        cells = self.nb_data.get("cells", [])
        start_idx = self.phase_positions["9.3"]

        # Find end of Phase 9.3 (start of Phase 9.4 or end of notebook)
        end_idx = self.phase_positions.get("9.4", len(cells))

        section_text = "\n".join(
            get_cell_text(cells[i]) for i in range(start_idx, min(end_idx, len(cells)))
        )

        required_content = [
            "build_comprehensive_features",
            "calculate_feature_importance",
            "all_stocks_featured",
        ]

        for content in required_content:
            self.assertIn(content, section_text, f"Phase 9.3 should contain '{content}'")

    # Test 11: Phase 9.6 contains evaluation content
    def test_phase_9_6_content(self):
        """Test that Phase 9.6 contains model evaluation functions."""
        if "9.6" not in self.phase_positions:
            self.skipTest("Phase 9.6 not present - will be added")

        cells = self.nb_data.get("cells", [])
        start_idx = self.phase_positions["9.6"]
        end_idx = self.phase_positions.get("9.7", len(cells))

        section_text = "\n".join(
            get_cell_text(cells[i]) for i in range(start_idx, min(end_idx, len(cells)))
        )

        required_content = [
            "comprehensive_regression_metrics",
            "compute_metrics_by_segment",
            "residual_analysis",
        ]

        for content in required_content:
            self.assertIn(content, section_text, f"Phase 9.6 should contain '{content}'")

    # Test 12: Phase 9.7 contains valuation content
    def test_phase_9_7_content(self):
        """Test that Phase 9.7 contains valuation and ranking functions."""
        if "9.7" not in self.phase_positions:
            self.skipTest("Phase 9.7 not present - will be added")

        cells = self.nb_data.get("cells", [])
        start_idx = self.phase_positions["9.7"]

        section_text = "\n".join(get_cell_text(cells[i]) for i in range(start_idx, len(cells)))

        required_content = [
            "calculate_mispricing_score",
            "assign_valuation_category",
            "rank_undervalued_stocks",
            "export_predictions_to_excel",
        ]

        for content in required_content:
            self.assertIn(content, section_text, f"Phase 9.7 should contain '{content}'")

    # Test 13: Complete workflow produces expected outputs
    def test_workflow_completeness(self):
        """Test that the workflow includes all major steps and outputs."""
        required_workflow_steps = [
            "PHASE 9.1",
            "PHASE 9.2",
            "PHASE 9.3",
            "PHASE 9.4",
            "PHASE 9.5",
            "PHASE 9.6",
            "PHASE 9.7",
        ]

        for step in required_workflow_steps:
            self.assertIn(step, self.nb_text, f"Workflow should include '{step}'")


class TestNotebookCellFactory(unittest.TestCase):
    """Test the NotebookCellFactory utility class for creating notebook cells."""

    def test_create_markdown_cell_structure(self):
        """Test that create_markdown_cell produces valid markdown cell structure."""
        content = "# Test Markdown"
        cell = NotebookCellFactory.create_markdown_cell(content)

        self.assertEqual(cell["cell_type"], "markdown")
        self.assertIn("metadata", cell)
        self.assertIn("source", cell)
        self.assertEqual(cell["source"], [content + NotebookCellFactory.NEWLINE_SUFFIX])

    def test_create_markdown_cell_content(self):
        """Test that markdown cell contains expected content."""
        content = "## Phase 9.1 — Test Phase"
        cell = NotebookCellFactory.create_markdown_cell(content)

        source_text = (
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        )
        self.assertIn("Phase 9.1", source_text)
        self.assertIn("Test Phase", source_text)

    def test_create_code_cell_structure(self):
        """Test that create_code_cell produces valid code cell structure."""
        code = "print('Hello, World!')"
        cell = NotebookCellFactory.create_code_cell(code)

        self.assertEqual(cell["cell_type"], "code")
        self.assertIsNone(cell["execution_count"])
        self.assertIn("metadata", cell)
        self.assertIn("outputs", cell)
        self.assertIn("source", cell)
        self.assertEqual(cell["outputs"], [])
        self.assertEqual(cell["source"], [code + NotebookCellFactory.NEWLINE_SUFFIX])

    def test_create_code_cell_content(self):
        """Test that code cell contains expected code."""
        code = "all_stocks_processed = preprocess_data(all_stocks)"
        cell = NotebookCellFactory.create_code_cell(code)

        source_text = (
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        )
        self.assertIn("all_stocks_processed", source_text)
        self.assertIn("preprocess_data", source_text)

    def test_create_phase_header_structure(self):
        """Test that create_phase_header produces valid phase header markdown."""
        phase_number = "9.3"
        title = "Advanced Feature Engineering"
        description = [
            "Valuation ratios (P/E, P/B, EV/EBITDA)",
            "Profitability ratios (ROE, ROA, margins)",
            "Sector-specific features",
        ]

        cell = NotebookCellFactory.create_phase_header(phase_number, title, description)

        self.assertEqual(cell["cell_type"], "markdown")
        source_text = (
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        )

        # Check header format
        self.assertIn(f"## Phase {phase_number}", source_text)
        self.assertIn(title, source_text)

        # Check numbered list items
        for i, desc in enumerate(description, 1):
            self.assertIn(f"{i}. {desc}", source_text)

    def test_create_phase_header_formatting(self):
        """Test that phase header has correct markdown formatting."""
        cell = NotebookCellFactory.create_phase_header(
            "9.6", "Model Evaluation", ["Metric 1", "Metric 2"]
        )
        source_text = (
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        )

        # Check markdown level 2 header
        self.assertTrue(source_text.startswith("## Phase"))
        # Check em-dash separator
        self.assertIn("—", source_text)
        # Check numbered list format
        self.assertIn("1. Metric 1", source_text)
        self.assertIn("2. Metric 2", source_text)

    def test_create_section_header_structure(self):
        """Test that create_section_header produces valid section header markdown."""
        phase = "9.5"
        section = "2"
        title = "XGBoost Regressor"

        cell = NotebookCellFactory.create_section_header(phase, section, title)

        self.assertEqual(cell["cell_type"], "markdown")
        source_text = (
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        )

        # Check section format
        self.assertIn(f"### {phase}.{section}", source_text)
        self.assertIn(title, source_text)

    def test_create_section_header_formatting(self):
        """Test that section header has correct markdown formatting."""
        cell = NotebookCellFactory.create_section_header("9.7", "3", "Stock Rankings")
        source_text = (
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        )

        # Check markdown level 3 header
        self.assertTrue(source_text.startswith("### 9.7"))
        # Check em-dash separator
        self.assertIn("—", source_text)
        self.assertIn("Stock Rankings", source_text)

    def test_newline_suffix_constant(self):
        """Test that NEWLINE_SUFFIX constant is correctly defined."""
        self.assertEqual(NotebookCellFactory.NEWLINE_SUFFIX, "\n")

    def test_markdown_cell_multiline_content(self):
        """Test creating markdown cell with multiline content."""
        content = "Line 1\nLine 2\nLine 3"
        cell = NotebookCellFactory.create_markdown_cell(content)
        source_text = (
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        )

        self.assertIn("Line 1", source_text)
        self.assertIn("Line 2", source_text)
        self.assertIn("Line 3", source_text)

    def test_code_cell_multiline_content(self):
        """Test creating code cell with multiline code."""
        code = "import pandas as pd\nimport numpy as np\nprint('Imports complete')"
        cell = NotebookCellFactory.create_code_cell(code)
        source_text = (
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        )

        self.assertIn("import pandas", source_text)
        self.assertIn("import numpy", source_text)
        self.assertIn("print('Imports complete')", source_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
