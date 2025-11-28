"""
Test suite for notebook refactoring verification.

Tests validate that ml_finance_model_main.ipynb follows Phase 9.1-9.8 architecture
as specified in code_guidelines.md v1.2 and notebook_refactoring_plan.md.

This module follows TDD approach - tests written first to define requirements.
"""

import json
import unittest
from pathlib import Path


class TestNotebookStructure(unittest.TestCase):
    """Test notebook structural alignment with Phase 9.1-9.8 architecture."""

    @classmethod
    def setUpClass(cls):
        """Load notebook once for all tests."""
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if not cls.notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {cls.notebook_path}")

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.cells = cls.notebook.get("cells", [])

        # Extract all markdown cells for analysis
        cls.markdown_cells = [
            "".join(cell.get("source", []))
            for cell in cls.cells
            if cell.get("cell_type") == "markdown"
        ]

        # Combine all markdown for text searches
        cls.notebook_text = "\n".join(cls.markdown_cells)

    def test_phase_91_header_exists(self):
        """Test that Phase 9.1 header exists with correct nomenclature."""
        phase_patterns = ["Phase 9.1", "Loading and Preprocessing", "6-Step Imputation"]

        found = False
        for cell_text in self.markdown_cells:
            if all(pattern in cell_text for pattern in phase_patterns):
                found = True
                break

        self.assertTrue(found, "Phase 9.1 header with '6-Step Imputation Strategy' not found")

    def test_phase_92_header_exists(self):
        """Test that Phase 9.2 header exists with correct nomenclature."""
        phase_patterns = ["Phase 9.2", "Exploratory Data Analysis"]

        found = False
        for cell_text in self.markdown_cells:
            if all(pattern in cell_text for pattern in phase_patterns):
                found = True
                break

        self.assertTrue(found, "Phase 9.2 header 'Enhanced Exploratory Data Analysis' not found")

    def test_phase_93_header_exists(self):
        """Test that Phase 9.3 header exists with correct nomenclature."""
        phase_patterns = ["Phase 9.3", "Feature Engineering"]

        found = False
        for cell_text in self.markdown_cells:
            if all(pattern in cell_text for pattern in phase_patterns):
                found = True
                break

        self.assertTrue(found, "Phase 9.3 header 'Advanced Feature Engineering' not found")

    def test_phase_94_header_exists(self):
        """Test that Phase 9.4 header exists with correct nomenclature."""
        phase_patterns = ["Phase 9.4", "Classification"]

        found = False
        for cell_text in self.markdown_cells:
            if all(pattern in cell_text for pattern in phase_patterns):
                found = True
                break

        self.assertTrue(found, "Phase 9.4 header 'Multi-Class Event Classification' not found")

    def test_phase_95_header_exists(self):
        """Test that Phase 9.5 header exists with correct nomenclature."""
        phase_patterns = ["Phase 9.5", "Regression"]

        found = False
        for cell_text in self.markdown_cells:
            if all(pattern in cell_text for pattern in phase_patterns):
                found = True
                break

        self.assertTrue(found, "Phase 9.5 header 'Sector-Optimized Regression' not found")

    def test_phase_96_header_exists(self):
        """Test that Phase 9.6 header exists with correct nomenclature."""
        phase_patterns = ["Phase 9.6", "Evaluation"]

        found = False
        for cell_text in self.markdown_cells:
            if all(pattern in cell_text for pattern in phase_patterns):
                found = True
                break

        self.assertTrue(found, "Phase 9.6 header 'Model Evaluation and Error Analysis' not found")

    def test_phase_97_header_exists(self):
        """Test that Phase 9.7 header exists with correct nomenclature."""
        phase_patterns = ["Phase 9.7", "Stock Ranking"]

        found = False
        for cell_text in self.markdown_cells:
            if all(pattern in cell_text for pattern in phase_patterns):
                found = True
                break

        self.assertTrue(
            found, "Phase 9.7 header 'Stock Ranking, Analytics, and Analyst Comparison' not found"
        )

    def test_phase_98_header_exists(self):
        """Test that Phase 9.8 header exists with correct nomenclature."""
        phase_patterns = ["Phase 9.8", "Reporting"]

        found = False
        for cell_text in self.markdown_cells:
            if all(pattern in cell_text for pattern in phase_patterns):
                found = True
                break

        self.assertTrue(found, "Phase 9.8 header 'Comprehensive Reporting' not found")

    def test_all_phases_present(self):
        """Test that all 8 phases (9.1-9.8) are present in the notebook."""
        required_phases = ["9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8"]
        found_phases = []

        for phase_num in required_phases:
            if f"Phase {phase_num}" in self.notebook_text:
                found_phases.append(phase_num)

        self.assertEqual(
            len(found_phases), 8, f"Expected 8 phases, found {len(found_phases)}: {found_phases}"
        )

    def test_no_old_section_headers(self):
        """Test that old numbered section headers (Section 2-10) are replaced."""
        # Check that sections 2-10 don't exist as primary headers
        old_patterns = [
            "## 2. Loading and Preprocessing",
            "## 3. Exploratory Data Analysis",
            "## 4. Advanced Feature Engineering",
            "## 5. Multi-Class Classification",
            "## 6. Sector-Optimized Regression",
            "## 7. Model Evaluation",
            "## 8. Identification of Under/Overvalued",
            "## 9. Comprehensive Analytics",
            "## 10. Portfolio Optimization",
        ]

        found_old_headers = []
        for pattern in old_patterns:
            if pattern in self.notebook_text:
                found_old_headers.append(pattern)

        self.assertEqual(
            len(found_old_headers),
            0,
            f"Found old section headers that should be replaced: {found_old_headers}",
        )


class TestPhaseDescriptions(unittest.TestCase):
    """Test that each phase has comprehensive descriptions."""

    @classmethod
    def setUpClass(cls):
        """Load notebook once for all tests."""
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.cells = cls.notebook.get("cells", [])
        cls.markdown_cells = [
            "".join(cell.get("source", []))
            for cell in cls.cells
            if cell.get("cell_type") == "markdown"
        ]

    def test_phase_91_has_business_goal(self):
        """Test Phase 9.1 has Business Goal description."""
        found = any("Phase 9.1" in cell and "Business Goal" in cell for cell in self.markdown_cells)
        self.assertTrue(found, "Phase 9.1 missing 'Business Goal' section")

    def test_phase_91_has_key_objectives(self):
        """Test Phase 9.1 has Key Objectives description."""
        found = any(
            "Phase 9.1" in cell and "Key Objectives" in cell for cell in self.markdown_cells
        )
        self.assertTrue(found, "Phase 9.1 missing 'Key Objectives' section")

    def test_phase_91_has_inputs_outputs(self):
        """Test Phase 9.1 has Inputs and Outputs documentation."""
        found_inputs = any("Phase 9.1" in cell and "Inputs" in cell for cell in self.markdown_cells)
        found_outputs = any(
            "Phase 9.1" in cell and "Outputs" in cell for cell in self.markdown_cells
        )
        self.assertTrue(found_inputs, "Phase 9.1 missing 'Inputs' section")
        self.assertTrue(found_outputs, "Phase 9.1 missing 'Outputs' section")

    def test_phase_95_has_business_goal(self):
        """Test Phase 9.5 has Business Goal description."""
        found = any("Phase 9.5" in cell and "Business Goal" in cell for cell in self.markdown_cells)
        self.assertTrue(found, "Phase 9.5 missing 'Business Goal' section")

    def test_phase_95_has_v12_standards(self):
        """Test Phase 9.5 documents v1.2 standards compliance."""
        found = any(
            "Phase 9.5" in cell and "v1.2 Standards" in cell for cell in self.markdown_cells
        )
        self.assertTrue(found, "Phase 9.5 missing 'v1.2 Standards Applied' section")


class TestValidationCheckpoints(unittest.TestCase):
    """Test that validation checkpoints exist for all phases."""

    @classmethod
    def setUpClass(cls):
        """Load notebook once for all tests."""
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.cells = cls.notebook.get("cells", [])
        cls.code_cells = [
            "".join(cell.get("source", [])) for cell in cls.cells if cell.get("cell_type") == "code"
        ]
        cls.notebook_code = "\n".join(cls.code_cells)

    def test_phase_91_validation_checkpoint(self):
        """Test Phase 9.1 has validation checkpoint."""
        patterns = ["PHASE 9.1", "VALIDATION", "CHECKPOINT"]
        found = all(pattern in self.notebook_code.upper() for pattern in patterns)
        self.assertTrue(found, "Phase 9.1 validation checkpoint not found")

    def test_phase_92_validation_checkpoint(self):
        """Test Phase 9.2 has validation checkpoint."""
        patterns = ["PHASE 9.2", "VALIDATION", "CHECKPOINT"]
        found = all(pattern in self.notebook_code.upper() for pattern in patterns)
        self.assertTrue(found, "Phase 9.2 validation checkpoint not found")

    def test_phase_93_validation_checkpoint(self):
        """Test Phase 9.3 has validation checkpoint."""
        patterns = ["PHASE 9.3", "VALIDATION", "CHECKPOINT"]
        found = all(pattern in self.notebook_code.upper() for pattern in patterns)
        self.assertTrue(found, "Phase 9.3 validation checkpoint not found")

    def test_phase_94_validation_checkpoint(self):
        """Test Phase 9.4 has validation checkpoint."""
        patterns = ["PHASE 9.4", "VALIDATION", "CHECKPOINT"]
        found = all(pattern in self.notebook_code.upper() for pattern in patterns)
        self.assertTrue(found, "Phase 9.4 validation checkpoint not found")

    def test_phase_95_validation_checkpoint(self):
        """Test Phase 9.5 has validation checkpoint."""
        patterns = ["PHASE 9.5", "VALIDATION", "CHECKPOINT"]
        found = all(pattern in self.notebook_code.upper() for pattern in patterns)
        self.assertTrue(found, "Phase 9.5 validation checkpoint not found")

    def test_phase_96_validation_checkpoint(self):
        """Test Phase 9.6 has validation checkpoint."""
        patterns = ["PHASE 9.6", "VALIDATION", "CHECKPOINT"]
        found = all(pattern in self.notebook_code.upper() for pattern in patterns)
        self.assertTrue(found, "Phase 9.6 validation checkpoint not found")

    def test_phase_97_validation_checkpoint(self):
        """Test Phase 9.7 has validation checkpoint."""
        patterns = ["PHASE 9.7", "VALIDATION", "CHECKPOINT"]
        found = all(pattern in self.notebook_code.upper() for pattern in patterns)
        self.assertTrue(found, "Phase 9.7 validation checkpoint not found")

    def test_phase_98_validation_checkpoint(self):
        """Test Phase 9.8 has validation checkpoint."""
        patterns = ["PHASE 9.8", "VALIDATION", "CHECKPOINT"]
        found = all(pattern in self.notebook_code.upper() for pattern in patterns)
        self.assertTrue(found, "Phase 9.8 validation checkpoint not found")


class TestV12Standards(unittest.TestCase):
    """Test v1.2 standards documentation in relevant phases."""

    @classmethod
    def setUpClass(cls):
        """Load notebook once for all tests."""
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.cells = cls.notebook.get("cells", [])
        cls.markdown_cells = [
            "".join(cell.get("source", []))
            for cell in cls.cells
            if cell.get("cell_type") == "markdown"
        ]
        cls.notebook_text = "\n".join(cls.markdown_cells)

    def test_uncertainty_quantification_documented(self):
        """Test that uncertainty quantification (Phase 9.5) is documented."""
        keywords = ["quantile", "prediction interval", "p10", "p50", "p90"]
        found = any(keyword in self.notebook_text.lower() for keyword in keywords)
        self.assertTrue(found, "Uncertainty quantification (quantile regression) not documented")

    def test_outlier_safety_rails_documented(self):
        """Test that outlier safety rails are documented."""
        keywords = ["winsorization", "outlier", "safety"]
        found = any(keyword in self.notebook_text.lower() for keyword in keywords)
        self.assertTrue(found, "Outlier safety rails not documented")

    def test_data_split_policy_documented(self):
        """Test that data split policy is documented."""
        keywords = ["time series", "split", "cross-validation", "GroupKFold"]
        found = any(keyword in self.notebook_text.lower() for keyword in keywords)
        self.assertTrue(found, "Data split policy not documented")

    def test_predictions_schema_documented(self):
        """Test that standardized predictions schema is documented."""
        keywords = ["predictions schema", "regression_predictions_detailed"]
        found = any(keyword in self.notebook_text.lower() for keyword in keywords)
        self.assertTrue(found, "Standardized predictions schema not documented")


class TestTableOfContents(unittest.TestCase):
    """Test that Table of Contents is updated with Phase 9.1-9.8 structure."""

    @classmethod
    def setUpClass(cls):
        """Load notebook and find ToC."""
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.cells = cls.notebook.get("cells", [])

        # Find first few markdown cells (ToC is usually near the top)
        cls.toc_text = ""
        for cell in cls.cells[:10]:  # Check first 10 cells
            if cell.get("cell_type") == "markdown":
                cell_text = "".join(cell.get("source", []))
                # Accept various ToC headings
                if any(
                    heading in cell_text
                    for heading in [
                        "Table of Contents",
                        "Contents",
                        "Quick Reference Navigation",
                        "Navigation",
                        "Workflow Overview",
                    ]
                ):
                    cls.toc_text = cell_text
                    break

    def test_toc_exists(self):
        """Test that Table of Contents exists."""
        self.assertTrue(len(self.toc_text) > 0, "Table of Contents not found in first 10 cells")

    def test_toc_has_phase_91(self):
        """Test ToC references Phase 9.1."""
        if self.toc_text:
            self.assertIn("Phase 9.1", self.toc_text, "ToC missing Phase 9.1")

    def test_toc_has_all_phases(self):
        """Test ToC references all phases 9.1-9.8."""
        if self.toc_text:
            phases = ["9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8"]
            missing = []
            for phase in phases:
                if f"9.{phase.split('.')[1]}" not in self.toc_text:
                    missing.append(phase)

            self.assertEqual(len(missing), 0, f"ToC missing phases: {missing}")


if __name__ == "__main__":
    unittest.main()
