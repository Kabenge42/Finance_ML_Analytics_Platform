import unittest

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None

try:
    import finance_ml as mod
except Exception:
    mod = None


@unittest.skipIf(pd is None or mod is None or np is None, "pandas/numpy not installed")
class TestAnalytics(unittest.TestCase):
    """Phase 5: Analytics and reporting tests per IMPROVEMENT_PLAN.md"""

    def test_calculate_mispricing_score(self):
        """Test mispricing score calculation: (predicted_target - last_price) / last_price"""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "last_price": [10.0, 20.0, 15.0, 25.0],
                "predicted_target": [12.0, 18.0, 18.0, 25.0],
            }
        )
        scores = mod.calculate_mispricing_score(df)
        # Should return Series or array with mispricing scores
        self.assertEqual(len(scores), 4)
        # A: (12-10)/10 = 0.2 (20% undervalued)
        self.assertAlmostEqual(scores[0], 0.2, places=4)
        # B: (18-20)/20 = -0.1 (10% overvalued)
        self.assertAlmostEqual(scores[1], -0.1, places=4)
        # C: (18-15)/15 = 0.2 (20% undervalued)
        self.assertAlmostEqual(scores[2], 0.2, places=4)
        # D: (25-25)/25 = 0.0 (fairly valued)
        self.assertAlmostEqual(scores[3], 0.0, places=4)

    def test_rank_undervalued_stocks(self):
        """Test ranking of most undervalued stocks"""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "sector": ["Tech", "Tech", "Energy", "Energy", "Finance"],
                "region": ["US", "US", "EU", "EU", "US"],
                "last_price": [10.0, 20.0, 15.0, 25.0, 30.0],
                "predicted_target": [15.0, 18.0, 20.0, 25.0, 28.0],
                "mispricing_score": [0.5, -0.1, 0.333, 0.0, -0.067],
            }
        )
        # Get top 3 undervalued (highest positive mispricing scores)
        top_undervalued = mod.rank_undervalued_stocks(df, top_n=3)
        self.assertEqual(len(top_undervalued), 3)
        # Should be sorted by mispricing_score descending
        self.assertEqual(top_undervalued.iloc[0]["ticker"], "A")  # 0.5
        self.assertEqual(top_undervalued.iloc[1]["ticker"], "C")  # 0.333
        self.assertEqual(top_undervalued.iloc[2]["ticker"], "D")  # 0.0

    def test_rank_overvalued_stocks(self):
        """Test ranking of most overvalued stocks"""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "sector": ["Tech", "Tech", "Energy", "Energy", "Finance"],
                "last_price": [10.0, 20.0, 15.0, 25.0, 30.0],
                "predicted_target": [15.0, 18.0, 20.0, 25.0, 28.0],
                "mispricing_score": [0.5, -0.1, 0.333, 0.0, -0.067],
            }
        )
        # Get top 2 overvalued (lowest negative mispricing scores)
        top_overvalued = mod.rank_overvalued_stocks(df, top_n=2)
        self.assertEqual(len(top_overvalued), 2)
        # Should be sorted by mispricing_score ascending
        self.assertEqual(top_overvalued.iloc[0]["ticker"], "B")  # -0.1
        self.assertEqual(top_overvalued.iloc[1]["ticker"], "E")  # -0.067

    def test_rank_by_sector(self):
        """Test ranking stocks within each sector"""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Tech", "Energy", "Energy", "Finance", "Finance"],
                "last_price": [10.0, 20.0, 15.0, 25.0, 30.0, 40.0],
                "predicted_target": [15.0, 18.0, 20.0, 25.0, 35.0, 38.0],
                "mispricing_score": [0.5, -0.1, 0.333, 0.0, 0.167, -0.05],
            }
        )
        # Rank top undervalued per sector
        ranked = mod.rank_stocks_by_sector(df, top_n=1, order="undervalued")
        # Should return dict with sector as key and DataFrame as value
        self.assertIn("Tech", ranked)
        self.assertIn("Energy", ranked)
        self.assertIn("Finance", ranked)
        # Tech: A has 0.5 (highest)
        self.assertEqual(ranked["Tech"].iloc[0]["ticker"], "A")
        # Energy: C has 0.333 (highest)
        self.assertEqual(ranked["Energy"].iloc[0]["ticker"], "C")
        # Finance: E has 0.167 (highest)
        self.assertEqual(ranked["Finance"].iloc[0]["ticker"], "E")


# Check if Excel engines are available
_EXCEL_AVAILABLE = False
try:
    import openpyxl

    _EXCEL_AVAILABLE = True
except ImportError:
    try:
        import xlsxwriter

        _EXCEL_AVAILABLE = True
    except ImportError:
        pass


@unittest.skipIf(pd is None or mod is None or np is None, "pandas/numpy not installed")
@unittest.skipIf(not _EXCEL_AVAILABLE, "No Excel engine (openpyxl or xlsxwriter) available")
class TestExcelReporting(unittest.TestCase):
    """Phase 5: Excel report generation tests per IMPROVEMENT_PLAN.md - TDD approach"""

    def test_export_predictions_to_excel(self):
        """Test exporting predictions and analytics to Excel format"""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "sector": ["Tech", "Tech", "Energy", "Energy", "Finance"],
                "region": ["US", "US", "EU", "EU", "APAC"],
                "last_price": [10.0, 20.0, 15.0, 25.0, 30.0],
                "predicted_target": [12.0, 18.0, 18.0, 25.0, 28.0],
                "mispricing_score": [0.2, -0.1, 0.2, 0.0, -0.067],
            }
        )

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            excel_path = out_dir / "predictions_report.xlsx"

            # Call export function
            mod.export_predictions_to_excel(df, excel_path)

            # Verify Excel file was created
            self.assertTrue(excel_path.exists(), "Excel report should be created")

            # Verify we can read it back with pandas
            # Use the first available engine
            engine = "openpyxl" if "openpyxl" in globals() else "xlsxwriter"
            df_read = pd.read_excel(excel_path, sheet_name="Predictions")
            self.assertEqual(len(df_read), 5)
            self.assertIn("ticker", df_read.columns)
            self.assertIn("mispricing_score", df_read.columns)

    def test_export_predictions_with_multiple_sheets(self):
        """Test exporting predictions with multiple sheets (summary, by sector, etc.)"""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Tech", "Energy", "Energy", "Finance", "Finance"],
                "region": ["US", "US", "EU", "EU", "APAC", "APAC"],
                "last_price": [10.0, 20.0, 15.0, 25.0, 30.0, 40.0],
                "predicted_target": [12.0, 18.0, 18.0, 25.0, 35.0, 38.0],
                "mispricing_score": [0.2, -0.1, 0.2, 0.0, 0.167, -0.05],
            }
        )

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            excel_path = out_dir / "predictions_report.xlsx"

            # Call export function with multi-sheet support
            mod.export_predictions_to_excel(df, excel_path, include_summary=True)

            # Verify Excel file was created
            self.assertTrue(excel_path.exists())

            # Verify multiple sheets exist
            with pd.ExcelFile(excel_path) as xl_file:
                sheets = xl_file.sheet_names
                self.assertIn("Predictions", sheets)
                self.assertIn("Summary", sheets)


if __name__ == "__main__":
    unittest.main(verbosity=2)
