import json
import unittest
from pathlib import Path

import pandas as pd


class TestPhase92EnhancedEDA(unittest.TestCase):
    def setUp(self):
        # Create a small synthetic dataframe with sectors and regions
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F", "G", "H"],
                "sector": [
                    "Tech",
                    "Tech",
                    "Energy",
                    "Energy",
                    "Health",
                    "Health",
                    "Tech",
                    "Energy",
                ],
                "region": ["US", "EU", "US", "EU", "US", "EU", "US", "EU"],
                # Valuation
                "p_e": [10, 12, 15, 30, 25, 20, 11, 1000],  # include outlier 1000
                "p_b": [1.2, 1.5, 2.0, 3.5, 2.8, 3.0, 1.1, 4.2],
                "ev_ebitda": [8, 9, 10, 12, 11, 13, 9, 100],  # outlier 100
                # Profitability
                "gross_margin": [0.45, 0.50, 0.30, 0.28, 0.60, None, 0.48, 0.35],
                "operating_margin": [0.20, 0.22, 0.10, 0.08, 0.25, 0.18, 0.21, 0.09],
                "net_margin": [0.12, 0.15, 0.05, 0.03, 0.18, 0.10, 0.13, 0.04],
                "roe": [0.10, 0.12, 0.08, 0.06, 0.15, 0.11, 0.13, 0.07],
                "roa": [0.05, 0.06, 0.03, 0.02, 0.07, 0.05, 0.06, 0.03],
                # Growth
                "revenue_growth": [0.10, 0.12, 0.08, 0.05, 0.15, 0.09, 0.11, 0.04],
                "earnings_growth": [0.08, 0.09, 0.05, 0.02, 0.12, 0.07, 0.10, 0.03],
                "ebitda_growth": [0.07, 0.08, 0.04, 0.01, 0.11, 0.06, 0.09, 0.02],
                # Leverage
                "debt_to_equity": [0.5, 0.6, 1.2, 1.5, 0.3, 0.4, 0.7, 2.5],
                "debt_to_assets": [0.3, 0.35, 0.6, 0.65, 0.25, 0.28, 0.4, 0.7],
                "net_debt_to_ebitda": [1.0, 1.2, 2.5, 3.0, 0.8, 0.9, 1.5, 4.0],
                # Positive-only metrics (include a negative to trigger an alert)
                "market_cap": [1e9, 2e9, 5e8, 6e8, -1.0, 3e9, 1.5e9, 4e8],
                "revenue": [1e8, 1.2e8, 9e7, 8e7, 1.5e8, 1.1e8, 1.3e8, 7e7],
                "total_assets": [2e8, 2.2e8, 1.8e8, 1.6e8, 2.5e8, 2.1e8, 2.3e8, 1.7e8],
                "total_equity": [1e8, 1.1e8, 9e7, 8e7, 1.2e8, 1.05e8, 1.15e8, 7.5e7],
                "ebitda": [2e7, 2.2e7, 1.8e7, 1.6e7, 2.5e7, 2.1e7, 2.3e7, 1.7e7],
                "last_price": [10.0, 12.0, 8.0, 6.0, 15.0, 11.0, 13.0, 7.0],
                "price_target": [11.0, 13.0, 9.0, 6.5, 16.0, 12.0, 14.0, 7.5],
            }
        )

    def test_functions_available_and_callable(self):
        from finance_ml.ml_workflow.analytics.eval import (
            calculate_financial_metrics_dashboard,
            generate_data_quality_alerts,
            perform_comprehensive_hypothesis_tests,
        )

        self.assertTrue(callable(calculate_financial_metrics_dashboard))
        self.assertTrue(callable(generate_data_quality_alerts))
        self.assertTrue(callable(perform_comprehensive_hypothesis_tests))

    def test_calculate_financial_metrics_dashboard_structure(self):
        from finance_ml.ml_workflow.analytics.eval import (
            calculate_financial_metrics_dashboard,
        )

        result = calculate_financial_metrics_dashboard(self.df, group_by="sector")
        # Top-level categories
        for cat in ("valuation", "profitability", "growth", "leverage"):
            self.assertIn(cat, result)
            self.assertIsInstance(result[cat], dict)

        # By-group aggregation
        self.assertIn("by_group", result)
        self.assertIsInstance(result["by_group"], dict)
        self.assertGreaterEqual(len(result["by_group"]), 2)

        # Spot-check a stat exists
        if "p_e" in self.df.columns:
            pe_stats = result["valuation"].get("p_e", {})
            if pe_stats:
                self.assertIn("mean", pe_stats)
                self.assertIn("count", pe_stats)

    def test_generate_data_quality_alerts_schema_and_content(self):
        from finance_ml.ml_workflow.analytics.eval import generate_data_quality_alerts

        alerts = generate_data_quality_alerts(self.df, outlier_threshold=3.0)
        self.assertIsInstance(alerts, list)

        allowed = {"low", "medium", "high", "critical"}
        # At least one alert should be present due to negative market_cap and outliers
        self.assertGreater(len(alerts), 0)
        for a in alerts:
            self.assertIn("severity", a)
            self.assertIn(a["severity"], allowed)
            self.assertIn("message", a)
            self.assertIn("column", a)

        # Ensure negative value alert for market_cap is detected
        neg_alerts = [
            a for a in alerts if a["column"] == "market_cap" and "negative" in a["message"].lower()
        ]
        self.assertTrue(len(neg_alerts) >= 1)

    def test_perform_comprehensive_hypothesis_tests_keys_and_pvalues(self):
        from finance_ml.ml_workflow.analytics.eval import perform_comprehensive_hypothesis_tests

        metrics = ["p_e", "roe"]
        res = perform_comprehensive_hypothesis_tests(
            self.df, group_column="sector", metrics=metrics, alpha=0.05
        )

        # Expect sector_tests section
        self.assertIn("sector_tests", res)
        section = res["sector_tests"]
        # Should have results per metric (summary may also be included)
        for m in metrics:
            if m in section:
                mres = section[m]
                # ANOVA present
                self.assertIn("anova", mres)
                anova = mres["anova"]
                if "p_value" in anova:
                    self.assertGreaterEqual(anova["p_value"], 0.0)
                    self.assertLessEqual(anova["p_value"], 1.0)
                # Kruskal present under either key; alias 'kruskal' also supported
                self.assertTrue(
                    ("kruskal_wallis" in mres) or ("kruskal" in mres),
                    msg="Expected kruskal or kruskal_wallis key in results",
                )
                k = mres.get("kruskal_wallis", mres.get("kruskal", {}))
                if "p_value" in k:
                    self.assertGreaterEqual(k["p_value"], 0.0)
                    self.assertLessEqual(k["p_value"], 1.0)

    def test_phase93_backup_cells_exist_and_have_7_code_cells(self):
        backup = Path("phase93_category_cells_backup.json")
        self.assertTrue(backup.exists(), msg="Backup JSON for Phase 9.3 not found")
        cells = json.loads(backup.read_text(encoding="utf-8"))
        self.assertIsInstance(cells, list)
        # The backup should include 7 code cells according to the summary
        self.assertEqual(len(cells), 7)
        for cell in cells:
            self.assertEqual(cell.get("cell_type"), "code")

    def test_phase92_docs_present(self):
        # Validate required documentation files exist
        required = [
            Path("PHASE92_RESTRUCTURING_IMPLEMENTATION.md"),
            Path("PHASE92_RESTRUCTURING_SUMMARY.md"),
            Path("restructure_phase92.py"),
        ]
        for p in required:
            self.assertTrue(p.exists(), msg=f"Missing required Phase 9.2 document: {p}")

    def test_notebook_phase92_structure(self):
        """
        Validate Phase 9.2 notebook structure after restructuring.
        Should have exactly 6 cells (20-25): 1 markdown + 5 code cells.
        """
        notebook_path = Path("ml_finance_model_main.ipynb")
        self.assertTrue(notebook_path.exists(), msg="Notebook not found")

        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = nb["cells"]

        # Verify Phase 9.2 section spans cells 20-25 (6 cells total)
        self.assertGreaterEqual(len(cells), 26, msg="Notebook must have at least 26 cells")

        # Cell 20: Markdown header
        cell20 = cells[20]
        self.assertEqual(cell20["cell_type"], "markdown")
        source20 = (
            "".join(cell20.get("source", []))
            if isinstance(cell20.get("source", []), list)
            else cell20.get("source", "")
        )
        self.assertIn("Phase 9.2", source20, msg="Cell 20 should be Phase 9.2 header")
        self.assertIn(
            "Enhanced Exploratory Data Analysis",
            source20,
            msg="Cell 20 should mention Enhanced EDA",
        )

        # Cells 21-25: Code cells with specific Phase 9.2 content
        expected_cell_markers = [
            (21, "code", "Cell 21", "EDA Report"),
            (22, "code", "Cell 22", "Statistical Hypothesis Testing"),
            (23, "code", "Cell 23", "Interactive Visualizations"),
            (24, "code", "Cell 24", "Sector & Regional Benchmarking"),
            (25, "code", "Cell 25", "EDA Summary Dashboard"),
        ]

        for idx, expected_type, cell_marker, content_marker in expected_cell_markers:
            cell = cells[idx]
            self.assertEqual(
                cell["cell_type"], expected_type, msg=f"Cell {idx} should be {expected_type}"
            )
            source = (
                "".join(cell.get("source", []))
                if isinstance(cell.get("source", []), list)
                else cell.get("source", "")
            )
            self.assertIn(
                cell_marker, source, msg=f"Cell {idx} should contain '{cell_marker}' marker"
            )
            self.assertIn(
                content_marker, source, msg=f"Cell {idx} should contain '{content_marker}'"
            )

        # Verify no old redundant cells remain (check cell 26 is NOT old Phase 9.2 content)
        # Cell 26 should be Phase 9.3 or other content, not "3.5. Phase 9.2 Enhanced EDA"
        if len(cells) > 26:
            cell26 = cells[26]
            source26 = (
                "".join(cell26.get("source", []))
                if isinstance(cell26.get("source", []), list)
                else cell26.get("source", "")
            )
            self.assertNotIn(
                "3.5. Phase 9.2 Enhanced EDA - Direct Module Usage",
                source26,
                msg="Old redundant Phase 9.2 header should be removed",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
