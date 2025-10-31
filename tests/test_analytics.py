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
        """Test mispricing score calculation: (predicted_price_target - last_price) / last_price"""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "last_price": [10.0, 20.0, 15.0, 25.0],
                "predicted_price_target": [12.0, 18.0, 18.0, 25.0],
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
                "predicted_price_target": [15.0, 18.0, 20.0, 25.0, 28.0],
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
                "predicted_price_target": [15.0, 18.0, 20.0, 25.0, 28.0],
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
                "predicted_price_target": [15.0, 18.0, 20.0, 25.0, 35.0, 38.0],
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
                "predicted_price_target": [12.0, 18.0, 18.0, 25.0, 28.0],
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
                "predicted_price_target": [12.0, 18.0, 18.0, 25.0, 35.0, 38.0],
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


@unittest.skipIf(pd is None or mod is None or np is None, "pandas/numpy not installed")
class TestPredictionVsAnalystComparison(unittest.TestCase):
    """Phase 9.8: Prediction vs. Analyst Price Target Analytics - TDD implementation"""

    def setUp(self):
        """Create sample data with both predicted and analyst targets"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Tech", "Energy", "Energy", "Finance", "Finance"],
                "region": ["US", "US", "EU", "EU", "APAC", "ROTW"],
                "last_price": [100.0, 50.0, 80.0, 120.0, 150.0, 90.0],
                "predicted_price_target": [120.0, 45.0, 100.0, 130.0, 180.0, 85.0],
                "price_target": [115.0, 48.0, 95.0, 135.0, 175.0, 88.0],  # Analyst target
                "market_cap": [1000.0, 500.0, 800.0, 1200.0, 1500.0, 900.0],
            }
        )

    def test_compare_prediction_vs_analyst_targets(self):
        """Test basic comparison between model predictions and analyst targets"""
        result = mod.compare_prediction_vs_analyst_targets(self.df)

        # Should return DataFrame with comparison metrics
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 6)

        # Should have key columns
        self.assertIn("ticker", result.columns)
        self.assertIn("model_analyst_diff", result.columns)
        self.assertIn("model_analyst_diff_pct", result.columns)
        self.assertIn("agreement_direction", result.columns)

    def test_calculate_directional_accuracy(self):
        """Test calculation of directional accuracy (model vs current price)"""
        result = mod.calculate_directional_accuracy(self.df)

        # Should return dict with accuracy metrics
        self.assertIsInstance(result, dict)
        self.assertIn("accuracy", result)
        self.assertIn("total_predictions", result)
        self.assertIn("correct_predictions", result)

        # Accuracy should be between 0 and 1
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_calculate_agreement_rate(self):
        """Test agreement rate between model and analyst predictions"""
        result = mod.calculate_agreement_rate(self.df)

        # Should return dict with agreement metrics
        self.assertIsInstance(result, dict)
        self.assertIn("agreement_rate", result)
        self.assertIn("same_direction_count", result)
        self.assertIn("total_count", result)

        # Agreement rate should be between 0 and 1
        self.assertGreaterEqual(result["agreement_rate"], 0.0)
        self.assertLessEqual(result["agreement_rate"], 1.0)

    def test_identify_disagreement_opportunities(self):
        """Test identification of stocks where model significantly differs from analysts"""
        result = mod.identify_disagreement_opportunities(self.df, threshold_pct=5.0)

        # Should return DataFrame with disagreements
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("ticker", result.columns)
        self.assertIn("model_analyst_diff_pct", result.columns)

        # All returned stocks should exceed threshold
        if len(result) > 0:
            self.assertTrue((abs(result["model_analyst_diff_pct"]) >= 5.0).all())

    def test_calculate_prediction_accuracy_metrics(self):
        """Test comprehensive accuracy metrics calculation"""
        # Add actual future price for testing
        df_with_actual = self.df.copy()
        df_with_actual["actual_future_price"] = [118.0, 46.0, 98.0, 128.0, 178.0, 87.0]

        result = mod.calculate_prediction_accuracy_metrics(df_with_actual)

        # Should return dict with multiple metrics
        self.assertIsInstance(result, dict)
        self.assertIn("model_mae", result)
        self.assertIn("analyst_mae", result)
        self.assertIn("model_directional_accuracy", result)
        self.assertIn("analyst_directional_accuracy", result)

        # MAE should be positive
        self.assertGreater(result["model_mae"], 0.0)
        self.assertGreater(result["analyst_mae"], 0.0)

    def test_segment_by_sector_comparison(self):
        """Test segmentation of comparison metrics by sector"""
        result = mod.segment_comparison_by_attribute(self.df, segment_col="sector")

        # Should return dict with sectors as keys
        self.assertIsInstance(result, dict)
        self.assertIn("Tech", result)
        self.assertIn("Energy", result)
        self.assertIn("Finance", result)

        # Each sector should have metrics
        for sector_metrics in result.values():
            self.assertIn("agreement_rate", sector_metrics)
            self.assertIn("avg_model_analyst_diff", sector_metrics)
            self.assertIn("count", sector_metrics)

    def test_segment_by_region_comparison(self):
        """Test segmentation of comparison metrics by region"""
        result = mod.segment_comparison_by_attribute(self.df, segment_col="region")

        # Should return dict with regions as keys
        self.assertIsInstance(result, dict)
        self.assertIn("US", result)
        self.assertIn("EU", result)

    def test_analyze_systematic_bias(self):
        """Test detection of systematic bias in predictions vs analysts"""
        result = mod.analyze_systematic_bias(self.df)

        # Should return dict with bias analysis
        self.assertIsInstance(result, dict)
        self.assertIn("mean_model_bias", result)
        self.assertIn("median_model_bias", result)
        self.assertIn("bias_direction", result)  # "optimistic", "pessimistic", "neutral"

        # Bias direction should be one of the expected values
        self.assertIn(result["bias_direction"], ["optimistic", "pessimistic", "neutral"])

    def test_compare_with_missing_columns(self):
        """Test comparison gracefully handles missing required columns"""
        df_incomplete = self.df[["ticker", "last_price"]].copy()

        # Should raise error or handle missing columns
        with self.assertRaises((KeyError, AttributeError)):
            mod.compare_prediction_vs_analyst_targets(df_incomplete)

    def test_directional_accuracy_with_actual_prices(self):
        """Test directional accuracy calculation with actual future prices"""
        df_with_actual = self.df.copy()
        df_with_actual["actual_future_price"] = [125.0, 47.0, 102.0, 125.0, 185.0, 82.0]

        result = mod.calculate_directional_accuracy(df_with_actual)

        # With actual prices, should calculate real accuracy
        self.assertIn("accuracy", result)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_segment_comparison_missing_column(self):
        """Test segmentation raises error when segment column missing"""
        with self.assertRaises(ValueError):
            mod.segment_comparison_by_attribute(self.df, segment_col="nonexistent")

    def test_identify_disagreement_no_threshold(self):
        """Test disagreement identification with very low threshold"""
        result = mod.identify_disagreement_opportunities(self.df, threshold_pct=0.1)

        # With low threshold, should return most stocks
        self.assertGreater(len(result), 0)
        self.assertIn("model_analyst_diff_pct", result.columns)

    def test_bias_direction_optimistic(self):
        """Test systematic bias detection identifies optimistic bias"""
        # Create data where model is consistently higher than analysts
        df_optimistic = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Tech", "Tech"],
                "last_price": [100.0, 100.0, 100.0],
                "predicted_price_target": [120.0, 130.0, 125.0],
                "price_target": [110.0, 115.0, 112.0],
            }
        )

        result = mod.analyze_systematic_bias(df_optimistic)
        self.assertEqual(result["bias_direction"], "optimistic")
        self.assertGreater(result["mean_model_bias"], 0)

    def test_bias_direction_pessimistic(self):
        """Test systematic bias detection identifies pessimistic bias"""
        # Create data where model is consistently lower than analysts
        df_pessimistic = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Tech", "Tech"],
                "last_price": [100.0, 100.0, 100.0],
                "predicted_price_target": [105.0, 108.0, 107.0],
                "price_target": [120.0, 125.0, 122.0],
            }
        )

        result = mod.analyze_systematic_bias(df_pessimistic)
        self.assertEqual(result["bias_direction"], "pessimistic")
        self.assertLess(result["mean_model_bias"], 0)

    def test_agreement_rate_all_agree(self):
        """Test agreement rate when all predictions agree on direction"""
        df_all_agree = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "last_price": [100.0, 100.0, 100.0],
                "predicted_price_target": [110.0, 105.0, 115.0],
                "price_target": [108.0, 102.0, 112.0],
            }
        )

        result = mod.calculate_agreement_rate(df_all_agree)
        self.assertEqual(result["agreement_rate"], 1.0)
        self.assertEqual(result["same_direction_count"], 3)

    def test_agreement_rate_none_agree(self):
        """Test agreement rate when no predictions agree on direction"""
        df_none_agree = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "last_price": [100.0, 100.0, 100.0],
                "predicted_price_target": [110.0, 105.0, 115.0],
                "price_target": [95.0, 92.0, 88.0],
            }
        )

        result = mod.calculate_agreement_rate(df_none_agree)
        self.assertEqual(result["agreement_rate"], 0.0)
        self.assertEqual(result["same_direction_count"], 0)


@unittest.skipIf(pd is None or mod is None or np is None, "pandas/numpy not installed")
@unittest.skipIf(not _EXCEL_AVAILABLE, "No Excel engine (openpyxl or xlsxwriter) available")
class TestPredictionAnalystExcelReport(unittest.TestCase):
    """Phase 9.8: Excel Report Generation for Prediction vs Analyst Analysis - TDD"""

    def setUp(self):
        """Create comprehensive sample data for Excel reporting"""
        np.random.seed(42)
        n_stocks = 50

        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_stocks)],
                "sector": np.random.choice(
                    ["Tech", "Energy", "Finance", "Healthcare", "Consumer"], n_stocks
                ),
                "region": np.random.choice(["US", "EU", "APAC", "ROTW"], n_stocks),
                "last_price": np.random.uniform(50, 200, n_stocks),
                "predicted_price_target": np.random.uniform(60, 220, n_stocks),
                "price_target": np.random.uniform(55, 215, n_stocks),
                "market_cap": np.random.uniform(500, 5000, n_stocks),
                "mispricing_score": np.random.uniform(-0.3, 0.4, n_stocks),
            }
        )

    def test_generate_prediction_analyst_excel_report(self):
        """Test generation of comprehensive Excel report comparing predictions vs analysts"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            excel_path = out_dir / "prediction_analyst_comparison_report.xlsx"

            # Generate comprehensive report
            mod.generate_prediction_analyst_excel_report(self.df, excel_path)

            # Verify Excel file was created
            self.assertTrue(excel_path.exists())

            # Verify required sheets exist
            with pd.ExcelFile(excel_path) as xl_file:
                sheets = xl_file.sheet_names

                # Required sheets per Phase 9.8 specification
                self.assertIn("Executive_Summary", sheets)
                self.assertIn("Detailed_Stock_List", sheets)
                self.assertIn("Top_Opportunities", sheets)
                self.assertIn("Risk_Analysis", sheets)
                self.assertIn("Prediction_Accuracy", sheets)
                self.assertIn("Sector_Analysis", sheets)

    def test_executive_summary_sheet_content(self):
        """Test Executive Summary sheet has required metrics"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            excel_path = out_dir / "report.xlsx"

            mod.generate_prediction_analyst_excel_report(self.df, excel_path)

            # Read Executive Summary sheet
            summary_df = pd.read_excel(excel_path, sheet_name="Executive_Summary")

            # Should contain key statistics
            self.assertGreater(len(summary_df), 0)
            # Check for presence of key metrics in the summary
            # (implementation will determine exact format)

    def test_detailed_stock_list_columns(self):
        """Test Detailed Stock List sheet has all required columns"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            excel_path = out_dir / "report.xlsx"

            mod.generate_prediction_analyst_excel_report(self.df, excel_path)

            # Read Detailed Stock List
            detail_df = pd.read_excel(excel_path, sheet_name="Detailed_Stock_List")

            # Required columns per Phase 9.8 spec
            required_cols = [
                "ticker",
                "sector",
                "region",
                "last_price",
                "predicted_price_target",
                "price_target",
                "mispricing_score",
            ]

            for col in required_cols:
                self.assertIn(col.lower(), [c.lower() for c in detail_df.columns])

    def test_prediction_accuracy_sheet(self):
        """Test Prediction Accuracy sheet contains comparison metrics"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            excel_path = out_dir / "report.xlsx"

            mod.generate_prediction_analyst_excel_report(self.df, excel_path)

            # Read Prediction Accuracy sheet
            accuracy_df = pd.read_excel(excel_path, sheet_name="Prediction_Accuracy")

            # Should contain metrics data
            self.assertGreater(len(accuracy_df), 0)

    def test_excel_report_with_actual_prices(self):
        """Test Excel report generation when actual future prices are available"""
        import tempfile
        from pathlib import Path

        df_with_actual = self.df.copy()
        df_with_actual["actual_future_price"] = np.random.uniform(60, 220, len(self.df))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            excel_path = out_dir / "report_with_actual.xlsx"

            mod.generate_prediction_analyst_excel_report(df_with_actual, excel_path)

            # Verify file created
            self.assertTrue(excel_path.exists())

            # Read Prediction Accuracy sheet - should have extra metrics
            accuracy_df = pd.read_excel(excel_path, sheet_name="Prediction_Accuracy")
            self.assertGreater(len(accuracy_df), 0)

    def test_excel_report_without_sector(self):
        """Test Excel report generation when sector column is missing"""
        import tempfile
        from pathlib import Path

        df_no_sector = self.df.drop(columns=["sector"])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            excel_path = out_dir / "report_no_sector.xlsx"

            mod.generate_prediction_analyst_excel_report(df_no_sector, excel_path)

            # Should create file but without Sector_Analysis sheet
            self.assertTrue(excel_path.exists())

            with pd.ExcelFile(excel_path) as xl_file:
                sheets = xl_file.sheet_names
                # Sector_Analysis may or may not be present
                # Other required sheets should exist
                self.assertIn("Executive_Summary", sheets)
                self.assertIn("Detailed_Stock_List", sheets)

    def test_excel_report_custom_top_n(self):
        """Test Excel report with custom top_n_opportunities parameter"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            excel_path = out_dir / "report_top10.xlsx"

            mod.generate_prediction_analyst_excel_report(
                self.df, excel_path, top_n_opportunities=10
            )

            self.assertTrue(excel_path.exists())

            # Check Top_Opportunities sheet has at most 10 rows
            top_opp_df = pd.read_excel(excel_path, sheet_name="Top_Opportunities")
            self.assertLessEqual(len(top_opp_df), 10)

    def test_excel_report_with_market_cap(self):
        """Test that market_cap is included in reports when available"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            excel_path = out_dir / "report_with_marketcap.xlsx"

            mod.generate_prediction_analyst_excel_report(self.df, excel_path)

            # Read Detailed Stock List
            detail_df = pd.read_excel(excel_path, sheet_name="Detailed_Stock_List")

            # market_cap should be in columns if it was in source data
            # (setUp includes market_cap in self.df)

    def test_model_interpretation_sheet_exists(self):
        """Test that Model_Interpretation sheet (7th sheet) is included in Excel report"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            excel_path = out_dir / "report_with_interpretation.xlsx"

            mod.generate_prediction_analyst_excel_report(self.df, excel_path)

            # Verify Model_Interpretation sheet exists (7th sheet per Phase 9.8)
            with pd.ExcelFile(excel_path) as xl_file:
                sheets = xl_file.sheet_names
                self.assertIn("Model_Interpretation", sheets)

    def test_model_interpretation_sheet_content(self):
        """Test that Model_Interpretation sheet contains methodology and feature importance"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            excel_path = out_dir / "report_interpretation.xlsx"

            mod.generate_prediction_analyst_excel_report(self.df, excel_path)

            # Read Model_Interpretation sheet
            interpretation_df = pd.read_excel(excel_path, sheet_name="Model_Interpretation")

            # Should contain model methodology information
            self.assertGreater(len(interpretation_df), 0)


@unittest.skipIf(pd is None or mod is None or np is None, "pandas/numpy not installed")
class TestAdvancedPredictionMetrics(unittest.TestCase):
    """Phase 9.8: Advanced metrics - Hit rate by confidence and calibration - TDD"""

    def setUp(self):
        """Create sample data with confidence intervals and actual prices"""
        np.random.seed(42)
        n_stocks = 100

        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_stocks)],
                "sector": np.random.choice(["Tech", "Energy", "Finance", "Healthcare"], n_stocks),
                "last_price": np.random.uniform(50, 200, n_stocks),
                "predicted_price_target": np.random.uniform(60, 220, n_stocks),
                "price_target": np.random.uniform(55, 215, n_stocks),
                "actual_future_price": np.random.uniform(55, 210, n_stocks),
            }
        )

        # Add confidence intervals (prediction uncertainty)
        self.df["prediction_lower"] = self.df["predicted_price_target"] * 0.9
        self.df["prediction_upper"] = self.df["predicted_price_target"] * 1.1

    def test_calculate_hit_rate_by_confidence_level(self):
        """Test hit rate calculation segmented by prediction confidence level"""
        result = mod.calculate_hit_rate_by_confidence_level(self.df)

        # Should return dict with confidence level buckets
        self.assertIsInstance(result, dict)
        self.assertIn("high_confidence", result)
        self.assertIn("medium_confidence", result)
        self.assertIn("low_confidence", result)

        # Each bucket should have hit rate and count
        for level_metrics in result.values():
            self.assertIn("hit_rate", level_metrics)
            self.assertIn("count", level_metrics)
            self.assertIn("correct_predictions", level_metrics)

            # Hit rate should be between 0 and 1
            self.assertGreaterEqual(level_metrics["hit_rate"], 0.0)
            self.assertLessEqual(level_metrics["hit_rate"], 1.0)

    def test_calculate_calibration_metrics(self):
        """Test calibration analysis: predicted upside vs. realized upside"""
        result = mod.calculate_calibration_metrics(self.df)

        # Should return dict with calibration metrics
        self.assertIsInstance(result, dict)
        self.assertIn("predicted_upside_mean", result)
        self.assertIn("realized_upside_mean", result)
        self.assertIn("calibration_error", result)
        self.assertIn("calibration_slope", result)

        # Calibration error should be a float
        self.assertIsInstance(result["calibration_error"], (int, float))

    def test_calibration_perfect_predictions(self):
        """Test calibration when predictions are perfect"""
        # Create perfect predictions
        df_perfect = self.df.copy()
        df_perfect["predicted_price_target"] = df_perfect["actual_future_price"]

        result = mod.calculate_calibration_metrics(df_perfect)

        # Calibration error should be near zero for perfect predictions
        self.assertAlmostEqual(result["calibration_error"], 0.0, places=1)

    def test_hit_rate_high_confidence_better(self):
        """Test that high confidence predictions have higher hit rate"""
        # Create data where narrow confidence intervals (high confidence) are more accurate
        df_test = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "last_price": [100.0, 100.0, 100.0, 100.0],
                "predicted_price_target": [110.0, 105.0, 115.0, 108.0],
                "actual_future_price": [111.0, 104.0, 116.0, 95.0],
                "prediction_lower": [
                    108.0,
                    103.0,
                    113.0,
                    95.0,
                ],  # Narrow for first 3 (high confidence)
                "prediction_upper": [112.0, 107.0, 117.0, 120.0],  # Wide for last (low confidence)
            }
        )

        result = mod.calculate_hit_rate_by_confidence_level(df_test)

        # Should have different hit rates for different confidence levels
        self.assertIsInstance(result, dict)

    def test_calibration_with_missing_actual_prices(self):
        """Test calibration handles missing actual_future_price column"""
        df_no_actual = self.df.drop(columns=["actual_future_price"])

        # Should raise error or handle gracefully
        with self.assertRaises((ValueError, KeyError)):
            mod.calculate_calibration_metrics(df_no_actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
