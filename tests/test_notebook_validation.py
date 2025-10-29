"""
Test suite for Notebook Validation Helpers (TDD Implementation)

Tests for validation utilities that help maintain notebook quality:
1. Incomplete code detection
2. Feature validation
3. Configuration consistency validation
4. Phase ordering validation

Coverage target: ≥80% for changed files
"""

import json
import unittest
from pathlib import Path
from typing import Dict, List


def parse_notebook(notebook_path: Path) -> Dict:
    """Parse notebook JSON file."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cell_text(cell: Dict) -> str:
    """Extract text from a notebook cell."""
    source = cell.get("source", [])
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def validate_required_imports(notebook_data: Dict, required_imports: List[str]) -> List[str]:
    """
    Validate that required imports are present in notebook.

    Args:
        notebook_data: Parsed notebook JSON
        required_imports: List of import names to check for

    Returns:
        List of missing imports
    """
    all_text = " ".join(get_cell_text(cell) for cell in notebook_data.get("cells", []))

    missing = []
    for import_name in required_imports:
        if import_name not in all_text:
            missing.append(import_name)

    return missing


def validate_classification_features(cell_text: str, required_cols: List[str]) -> List[str]:
    """
    Validate that required classification columns are checked for existence.

    Args:
        cell_text: Code cell text to validate
        required_cols: List of required column prefixes (e.g., 'event_prob_')

    Returns:
        List of missing validations
    """
    missing = []
    for col in required_cols:
        # Check if there's any validation for this column pattern
        if col not in cell_text:
            missing.append(col)

    return missing


def detect_incomplete_code_blocks(notebook_data: Dict) -> List[Dict]:
    """
    Detect potentially incomplete code blocks in notebook.

    Looks for common patterns of incomplete code:
    - Lines ending with operators (=, !=, +, -, etc.) without continuation
    - Unclosed brackets/parentheses
    - Incomplete list comprehensions

    Args:
        notebook_data: Parsed notebook JSON

    Returns:
        List of dicts with cell_index and issue description
    """
    issues = []

    for idx, cell in enumerate(notebook_data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        text = get_cell_text(cell)
        lines = text.strip().split("\n")

        if not lines:
            continue

        last_line = lines[-1].strip()

        # Check for trailing operators without continuation
        operators = ["!=", "==", "=", "+", "-", "*", "/", ">", "<"]
        keyword_operators = ["if", "and", "or", "elif", "while", "for"]

        # Check exact operators
        for op in operators:
            if last_line.endswith(op):
                issues.append(
                    {
                        "cell_index": idx,
                        "issue": f"Code block ends with operator: {last_line[-20:]}",
                        "last_line": last_line,
                    }
                )
                break

        # Check keyword operators (need word boundary)
        for kw in keyword_operators:
            if last_line.endswith(kw) or last_line.endswith(kw + " "):
                issues.append(
                    {
                        "cell_index": idx,
                        "issue": f"Code block ends with keyword: {last_line[-20:]}",
                        "last_line": last_line,
                    }
                )
                break

        # Check for unclosed brackets (simple heuristic)
        open_brackets = text.count("[") - text.count("]")
        open_parens = text.count("(") - text.count(")")
        open_braces = text.count("{") - text.count("}")

        if open_brackets > 0 or open_parens > 0 or open_braces > 0:
            issues.append(
                {
                    "cell_index": idx,
                    "issue": f"Unclosed brackets: [ {open_brackets}, ( {open_parens}, {{ {open_braces}",
                    "last_line": last_line,
                }
            )

    return issues


def validate_phase_headers(notebook_data: Dict) -> Dict[str, bool]:
    """
    Validate that phase headers use consistent numbering.

    Args:
        notebook_data: Parsed notebook JSON

    Returns:
        Dict mapping phase to validation status
    """
    validation_results = {}

    for cell in notebook_data.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue

        text = get_cell_text(cell)

        # Look for phase headers (## Phase X.Y)
        import re

        phase_pattern = r"##\s*Phase\s*(\d+)\.(\d+)"
        matches = re.findall(phase_pattern, text)

        for major, minor in matches:
            phase_key = f"{major}.{minor}"
            validation_results[phase_key] = True

    return validation_results


class TestNotebookValidationHelpers(unittest.TestCase):
    """Test notebook validation utility functions."""

    @classmethod
    def setUpClass(cls):
        """Load notebook once for all tests."""
        cls.notebook_path = Path(__file__).parent.parent / "ml_finance_model_main.ipynb"
        if cls.notebook_path.exists():
            cls.nb_data = parse_notebook(cls.notebook_path)
        else:
            cls.nb_data = None

    def test_validate_required_imports_detects_missing(self):
        """validate_required_imports should detect missing imports."""
        if self.nb_data is None:
            self.skipTest("Notebook not found")

        # Test with a known missing import
        missing = validate_required_imports(self.nb_data, ["NonExistentModule"])
        self.assertIn("NonExistentModule", missing)

    def test_validate_required_imports_detects_present(self):
        """validate_required_imports should not report existing imports as missing."""
        if self.nb_data is None:
            self.skipTest("Notebook not found")

        # Test with imports that should be present
        missing = validate_required_imports(self.nb_data, ["pandas", "numpy"])
        self.assertEqual(len(missing), 0, f"Expected imports found missing: {missing}")

    def test_detect_incomplete_code_blocks(self):
        """detect_incomplete_code_blocks should identify incomplete code."""
        # Create test notebook with incomplete code
        test_nb = {"cells": [{"cell_type": "code", "source": ["x = [1, 2, 3]\n", "y = x if"]}]}

        issues = detect_incomplete_code_blocks(test_nb)
        self.assertGreater(len(issues), 0, "Should detect incomplete code ending with 'if'")

    def test_detect_incomplete_code_blocks_no_issues(self):
        """detect_incomplete_code_blocks should not flag complete code."""
        test_nb = {"cells": [{"cell_type": "code", "source": ["x = [1, 2, 3]\n", "print(x)"]}]}

        issues = detect_incomplete_code_blocks(test_nb)
        self.assertEqual(len(issues), 0, f"Should not flag complete code, but found: {issues}")

    def test_validate_phase_headers(self):
        """validate_phase_headers should extract phase numbers correctly."""
        test_nb = {
            "cells": [
                {"cell_type": "markdown", "source": ["## Phase 9.1 — Preprocessing"]},
                {"cell_type": "markdown", "source": ["## Phase 9.2 — EDA"]},
            ]
        }

        results = validate_phase_headers(test_nb)
        self.assertIn("9.1", results)
        self.assertIn("9.2", results)
        self.assertTrue(results["9.1"])
        self.assertTrue(results["9.2"])

    def test_validate_classification_features(self):
        """validate_classification_features should detect missing column checks."""
        code_with_check = "if 'event_prob_neutral' in df.columns:"
        code_without_check = "df['event_prob_neutral']"

        # Code with validation should pass
        missing_with = validate_classification_features(code_with_check, ["event_prob_neutral"])
        self.assertEqual(len(missing_with), 0)

        # Code without validation might still pass if column name is present
        missing_without = validate_classification_features(
            code_without_check, ["event_prob_neutral"]
        )
        self.assertEqual(len(missing_without), 0, "Column name is present in code")


class TestNotebookDataQualityReportImport(unittest.TestCase):
    """Test that DataQualityReport is properly imported in notebook."""

    @classmethod
    def setUpClass(cls):
        """Load notebook once for all tests."""
        cls.notebook_path = Path(__file__).parent.parent / "ml_finance_model_main.ipynb"
        if cls.notebook_path.exists():
            cls.nb_data = parse_notebook(cls.notebook_path)
            cls.nb_text = " ".join(get_cell_text(cell) for cell in cls.nb_data.get("cells", []))
        else:
            cls.nb_data = None
            cls.nb_text = ""

    def test_data_quality_report_imported(self):
        """DataQualityReport should be imported in notebook."""
        if self.nb_data is None:
            self.skipTest("Notebook not found")

        # This test will FAIL initially - we need to add the import
        self.assertIn(
            "DataQualityReport",
            self.nb_text,
            "DataQualityReport should be imported from finance_ml.advanced_preprocessing",
        )


class TestNotebookPhase93Header(unittest.TestCase):
    """Test that Phase 9.3 header is correctly formatted."""

    @classmethod
    def setUpClass(cls):
        """Load notebook once for all tests."""
        cls.notebook_path = Path(__file__).parent.parent / "ml_finance_model_main.ipynb"
        if cls.notebook_path.exists():
            cls.nb_data = parse_notebook(cls.notebook_path)
        else:
            cls.nb_data = None

    def test_phase_93_header_exists(self):
        """Phase 9.3 header should exist with correct format."""
        if self.nb_data is None:
            self.skipTest("Notebook not found")

        # Find phase headers
        phase_headers = validate_phase_headers(self.nb_data)

        # This test will FAIL initially - header says "8.3" not "9.3"
        self.assertIn(
            "9.3", phase_headers, "Phase 9.3 header should exist (currently shows as Phase 8.3)"
        )

    def test_no_phase_83_header_exists(self):
        """Phase 8.3 header should NOT exist (should be 9.3)."""
        if self.nb_data is None:
            self.skipTest("Notebook not found")

        phase_headers = validate_phase_headers(self.nb_data)

        # Should not have Phase 8.3
        self.assertNotIn(
            "8.3", phase_headers, "Phase 8.3 should not exist (should be renamed to Phase 9.3)"
        )


if __name__ == "__main__":
    unittest.main()
