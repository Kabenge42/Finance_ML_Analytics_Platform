"""
Test notebook structure for Phase 9.4-9.8 advanced evaluation sections.

This test validates that ml_finance_model_main.ipynb contains the required sections
for Phase 9.4-9.8 with proper cell markers and minimum cell counts.

Success Criteria (from notebook_restructuring_plan.md):
- Phase 9.4–9.8 sections exist with concise cells (≤6 per section)
- Clearly labeled markers in code cells
- All required import statements present

TDD Implementation:
- This test should FAIL initially (notebook doesn't have Phase 9.4-9.8 sections yet)
- After integrating cells from NOTEBOOK_INTEGRATION_GUIDE.md, test should PASS
"""

import json
import unittest
from pathlib import Path


class TestNotebookPhase94_98Structure(unittest.TestCase):
    """Test structure of Phase 9.4-9.8 sections in the notebook."""

    @classmethod
    def setUpClass(cls):
        """Load notebook JSON once for all tests."""
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if not cls.notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {cls.notebook_path}")
        
        with open(cls.notebook_path, 'r', encoding='utf-8') as f:
            cls.notebook = json.load(f)
        
        cls.cells = cls.notebook.get('cells', [])
    
    def _get_cell_source(self, cell):
        """Extract source text from a cell."""
        source = cell.get('source', [])
        if isinstance(source, list):
            return ''.join(source)
        return source
    
    def _find_cells_with_marker(self, marker: str) -> list:
        """Find all cells containing the specified marker."""
        matching_cells = []
        for i, cell in enumerate(self.cells):
            source = self._get_cell_source(cell)
            if marker in source:
                matching_cells.append((i, cell, source))
        return matching_cells
    
    def _count_section_cells(self, section_marker: str, next_section_marker: str = None) -> int:
        """Count cells between section marker and next section (or end)."""
        cells = self.cells
        start_idx = None
        end_idx = len(cells)
        
        # Find section start
        for i, cell in enumerate(cells):
            source = self._get_cell_source(cell)
            if section_marker in source:
                start_idx = i
                break
        
        if start_idx is None:
            return 0
        
        # Find next section or end
        if next_section_marker:
            for i in range(start_idx + 1, len(cells)):
                source = self._get_cell_source(cells[i])
                if next_section_marker in source:
                    end_idx = i
                    break
        
        return end_idx - start_idx - 1  # Exclude the header cell itself
    
    def test_notebook_exists(self):
        """Test that notebook file exists and is valid JSON."""
        self.assertTrue(self.notebook_path.exists(), "Notebook file must exist")
        self.assertIsInstance(self.notebook, dict, "Notebook must be valid JSON")
        self.assertIn('cells', self.notebook, "Notebook must have cells")
        self.assertGreater(len(self.cells), 0, "Notebook must have at least one cell")
    
    def test_phase94_imports_present(self):
        """Test that Phase 9.4-9.8 imports are present in the notebook."""
        required_imports = [
            'build_quantile_diagnostics',
            'plot_interval_coverage',
            'plot_reliability_diagram',
            'summarize_winsorization_effects',
            'track_constraint_violations',
            'safety_rails_sensitivity_app',
            'compute_fold_overlap',
            'summarize_grouped_cv_balance',
            'time_leakage_checks',
            'estimate_sector_bias',
            'plot_metrics_by_sector_time',
            'create_sector_bias_dashboard',
            'compute_stacking_contributions',
            'meta_error_maps',
            'generate_model_card',
            'build_lineage_json',
        ]
        
        # Find import cells
        all_source = '\n'.join(self._get_cell_source(cell) for cell in self.cells)
        
        missing_imports = []
        for import_name in required_imports:
            if import_name not in all_source:
                missing_imports.append(import_name)
        
        self.assertEqual(
            len(missing_imports), 0,
            f"Missing required imports: {', '.join(missing_imports)}"
        )
    
    def test_phase94_section_exists(self):
        """Test that Phase 9.4 (Uncertainty Quantification) section exists."""
        marker = "PHASE 9.4"
        cells = self._find_cells_with_marker(marker)
        self.assertGreater(
            len(cells), 0,
            f"Phase 9.4 section must exist with '{marker}' marker"
        )
    
    def test_phase94_has_minimum_cells(self):
        """Test that Phase 9.4 has at least 4 code cells (excluding markdown header)."""
        # Section should have ~5 cells per plan: 1 markdown + 4-5 code cells
        count = self._count_section_cells(
            "Section 9.4: Uncertainty Quantification",
            "Section 9.5:"
        )
        self.assertGreaterEqual(
            count, 4,
            f"Phase 9.4 should have at least 4 cells (found {count})"
        )
    
    def test_phase95_section_exists(self):
        """Test that Phase 9.5 (Safety Rails) section exists."""
        marker = "PHASE 9.5"
        cells = self._find_cells_with_marker(marker)
        self.assertGreater(
            len(cells), 0,
            f"Phase 9.5 section must exist with '{marker}' marker"
        )
    
    def test_phase95_has_minimum_cells(self):
        """Test that Phase 9.5 has at least 4 code cells."""
        count = self._count_section_cells(
            "Section 9.5: Outlier Safety Rails",
            "Section 9.6:"
        )
        self.assertGreaterEqual(
            count, 4,
            f"Phase 9.5 should have at least 4 cells (found {count})"
        )
    
    def test_phase96_section_exists(self):
        """Test that Phase 9.6 (Data Splits & Leakage) section exists."""
        marker = "PHASE 9.6"
        cells = self._find_cells_with_marker(marker)
        self.assertGreater(
            len(cells), 0,
            f"Phase 9.6 section must exist with '{marker}' marker"
        )
    
    def test_phase96_has_minimum_cells(self):
        """Test that Phase 9.6 has at least 4 code cells."""
        count = self._count_section_cells(
            "Section 9.6: Data Split and Leakage",
            "Section 9.7:"
        )
        self.assertGreaterEqual(
            count, 4,
            f"Phase 9.6 should have at least 4 cells (found {count})"
        )
    
    def test_phase97_section_exists(self):
        """Test that Phase 9.7 (Sector Bias Calibration) section exists."""
        marker = "PHASE 9.7"
        cells = self._find_cells_with_marker(marker)
        self.assertGreater(
            len(cells), 0,
            f"Phase 9.7 section must exist with '{marker}' marker"
        )
    
    def test_phase97_has_minimum_cells(self):
        """Test that Phase 9.7 has at least 4 code cells."""
        count = self._count_section_cells(
            "Section 9.7: Sector Bias Calibration",
            "Section 9.8:"
        )
        self.assertGreaterEqual(
            count, 4,
            f"Phase 9.7 should have at least 4 cells (found {count})"
        )
    
    def test_phase98_section_exists(self):
        """Test that Phase 9.8 (Stacking & Governance) section exists."""
        marker = "PHASE 9.8"
        cells = self._find_cells_with_marker(marker)
        self.assertGreater(
            len(cells), 0,
            f"Phase 9.8 section must exist with '{marker}' marker"
        )
    
    def test_phase98_has_minimum_cells(self):
        """Test that Phase 9.8 has at least 5 code cells (6 total per plan)."""
        count = self._count_section_cells(
            "Section 9.8: Stacking Ensemble Diagnostics",
            "Section 10:"
        )
        self.assertGreaterEqual(
            count, 5,
            f"Phase 9.8 should have at least 5 cells (found {count})"
        )
    
    def test_cell_markers_present(self):
        """Test that all required cell markers are present."""
        required_markers = [
            "[PHASE 9.4]",
            "[PHASE 9.5]",
            "[PHASE 9.6]",
            "[PHASE 9.7]",
            "[PHASE 9.8]",
        ]
        
        all_source = '\n'.join(self._get_cell_source(cell) for cell in self.cells)
        
        missing_markers = []
        for marker in required_markers:
            if marker not in all_source:
                missing_markers.append(marker)
        
        self.assertEqual(
            len(missing_markers), 0,
            f"Missing required cell markers: {', '.join(missing_markers)}"
        )
    
    def test_output_directories_referenced(self):
        """Test that output directories for each phase are referenced."""
        expected_dirs = [
            'uncertainty',      # Phase 9.4
            'safety_rails',     # Phase 9.5
            'splits',           # Phase 9.6
            'calibration',      # Phase 9.7
            'governance',       # Phase 9.8
        ]
        
        all_source = '\n'.join(self._get_cell_source(cell) for cell in self.cells)
        
        missing_dirs = []
        for dir_name in expected_dirs:
            if dir_name not in all_source:
                missing_dirs.append(dir_name)
        
        self.assertEqual(
            len(missing_dirs), 0,
            f"Missing output directory references: {', '.join(missing_dirs)}"
        )
    
    def test_artifact_paths_present(self):
        """Test that key artifact paths are referenced in the notebook."""
        key_artifacts = [
            'quantile_predictions_diagnostics.csv',
            'uncertainty_summary.json',
            'safety_rails_summary.json',
            'leakage_report.json',
            'sector_bias_calibration',
            'model_card',
            'lineage.json',
        ]
        
        all_source = '\n'.join(self._get_cell_source(cell) for cell in self.cells)
        
        missing_artifacts = []
        for artifact in key_artifacts:
            if artifact not in all_source:
                missing_artifacts.append(artifact)
        
        self.assertEqual(
            len(missing_artifacts), 0,
            f"Missing artifact path references: {', '.join(missing_artifacts)}"
        )


if __name__ == '__main__':
    unittest.main()
