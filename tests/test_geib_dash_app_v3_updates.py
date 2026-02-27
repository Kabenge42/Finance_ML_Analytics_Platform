"""
Tests for geib_dash_app.py v3 updates.

Validates structural changes: new artifact IDs in layout, updated DataTable
columns, callback output count, and import alignment with expected_returns_v3.
"""

import ast
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASH_APP_PATH = PROJECT_ROOT / "finance_ml" / "dashboards" / "geib_dash_app.py"


class TestGeibDashAppV3Structure(unittest.TestCase):
    """Verify the geib_dash_app.py source contains all v3 structural updates."""

    @classmethod
    def setUpClass(cls):
        cls.source = DASH_APP_PATH.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Import checks
    # ------------------------------------------------------------------
    def test_imports_from_expected_returns_v3(self):
        """App should import from expected_returns_v3, not expected_returns."""
        self.assertIn("from expected_returns_v3 import", self.source)

    def test_imports_new_v3_visualization_functions(self):
        """New v3 visualization functions must be imported."""
        for func in [
            "create_model_dispersion_dashboard",
            "create_return_distribution_fit_chart",
            "create_sector_return_analytics_heatmap",
            "create_screening_summary_chart",
            "compute_sector_return_analytics",
        ]:
            with self.subTest(func=func):
                self.assertIn(func, self.source)

    def test_imports_bayesian_ridge_functions(self):
        """Bayesian ridge and ruin probability viz functions must be imported."""
        self.assertIn("create_bayesian_category_ridge", self.source)

    # ------------------------------------------------------------------
    # Layout artifact IDs
    # ------------------------------------------------------------------
    def test_new_artifact_graph_ids_present(self):
        """All new artifact component IDs must appear in the layout."""
        new_ids = [
            "artifact-er-screening-summary",
            "artifact-er-sector-return-analytics",
            "artifact-er-model-dispersion",
            "artifact-er-bayesian-profitability-ridge",
            "artifact-er-bayesian-sentiment-ridge",
            "artifact-er-distress-early-warning",
            "artifact-er-return-distribution-fit",
        ]
        for cid in new_ids:
            with self.subTest(component_id=cid):
                self.assertIn(cid, self.source)

    # ------------------------------------------------------------------
    # DataTable column checks
    # ------------------------------------------------------------------
    def _extract_columns_for_table(self, table_id: str) -> str:
        """Extract the get_formatted_columns([...]) block for a given table id."""
        pattern = rf'id="{table_id}".*?get_formatted_columns\(\[(.*?)\]\)'
        match = re.search(pattern, self.source, re.DOTALL)
        self.assertIsNotNone(match, f"Could not find columns for {table_id}")
        return match.group(1)

    def test_top_opportunities_table_has_new_columns(self):
        cols_text = self._extract_columns_for_table("top-opportunities-table")
        for col in ["posterior_beat_prob", "beat_classification", "agreement_score", "weighted_agreement"]:
            with self.subTest(col=col):
                self.assertIn(col, cols_text)

    def test_zscore_table_has_new_columns(self):
        cols_text = self._extract_columns_for_table("zscore-ranking-table")
        for col in ["posterior_beat_prob", "beat_classification", "agreement_score", "weighted_agreement"]:
            with self.subTest(col=col):
                self.assertIn(col, cols_text)

    def test_earnings_calendar_table_has_new_columns(self):
        cols_text = self._extract_columns_for_table("earnings-calendar-table")
        for col in ["posterior_beat_prob", "beat_classification", "agreement_score"]:
            with self.subTest(col=col):
                self.assertIn(col, cols_text)

    # ------------------------------------------------------------------
    # Callback output count
    # ------------------------------------------------------------------
    def test_update_dashboard_callback_output_count(self):
        """The main callback must have exactly 31 Output declarations."""
        # Find the callback decorator block for update_dashboard
        pattern = r"@app\.callback\(\s*\[(.*?)\],\s*\[Input\(f\[\"id\"\]"
        match = re.search(pattern, self.source, re.DOTALL)
        self.assertIsNotNone(match, "Could not find update_dashboard callback")
        output_block = match.group(1)
        output_count = output_block.count("Output(")
        self.assertEqual(output_count, 31, f"Expected 31 outputs, got {output_count}")

    # ------------------------------------------------------------------
    # Subtitle / description text
    # ------------------------------------------------------------------
    def test_subtitle_updated_to_quad_model(self):
        """Subtitle should reference Quad-Model Consensus."""
        self.assertIn("Quad-Model Consensus", self.source)

    # ------------------------------------------------------------------
    # get_formatted_columns coverage
    # ------------------------------------------------------------------
    def test_formatted_columns_handles_price_target_mc(self):
        """price_target_mc should be in currency formatting list."""
        self.assertIn('"price_target_mc"', self.source)

    def test_formatted_columns_handles_agreement_score(self):
        """agreement_score should have integer formatting."""
        # Verify it appears in the integer scores block
        pattern = r'# Integer scores.*?"agreement_score"'
        match = re.search(pattern, self.source, re.DOTALL)
        self.assertIsNotNone(match, "agreement_score not found in integer scores block")

    def test_formatted_columns_handles_weighted_agreement(self):
        """weighted_agreement should be in percentage formatting list."""
        # Find the percentage formatting block that includes weighted_agreement
        pattern = r'"prob_positive_upside".*?"weighted_agreement"'
        match = re.search(pattern, self.source, re.DOTALL)
        self.assertIsNotNone(match, "weighted_agreement not found in percentage formatting block")


if __name__ == "__main__":
    unittest.main()
