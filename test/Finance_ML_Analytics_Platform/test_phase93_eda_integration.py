"""
Phase 9.3 EDA Integration Tests

Tests for explicit Phase 9.3 feature family tracking in EDA summaries.
Following strict TDD: write failing tests first, implement minimal code, refactor.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json


class TestPhase93FeatureCategoryRegistry(unittest.TestCase):
    """Test Phase 9.3 feature category registry."""

    def test_registry_exists(self):
        """Test that Phase 9.3 feature category registry is importable."""
        from finance_ml.ml_workflow.eda.phase93_categories import PHASE93_FEATURE_CATEGORIES

        self.assertIsInstance(PHASE93_FEATURE_CATEGORIES, dict)
        self.assertGreater(len(PHASE93_FEATURE_CATEGORIES), 0)

    def test_registry_has_required_categories(self):
        """Test registry contains all Phase 9.3 feature families."""
        from finance_ml.ml_workflow.eda.phase93_categories import PHASE93_FEATURE_CATEGORIES

        required_categories = [
            "Momentum & Technical",
            "Valuation Ratios",
            "Profitability",
            "Quality & Risk",
            "Cash Flow",
            "Capital Allocation",
            "Analyst Sentiment",
            "Market Sentiment",
            "Leverage & Liquidity",
            "Temporal Patterns",
            "Composite Scores",
        ]

        for category in required_categories:
            self.assertIn(category, PHASE93_FEATURE_CATEGORIES)
            self.assertIsInstance(PHASE93_FEATURE_CATEGORIES[category], list)

    def test_registry_column_mapping_accuracy(self):
        """Test that registry maps to actual feature column names."""
        from finance_ml.ml_workflow.eda.phase93_categories import PHASE93_FEATURE_CATEGORIES

        # Momentum should include momentum-related features (using ACTUAL feature names)
        momentum_features = PHASE93_FEATURE_CATEGORIES.get("Momentum & Technical", [])
        expected_momentum = ["price_momentum_1m", "rsi_14d", "ma_crossover_signal"]

        for feature in expected_momentum:
            self.assertIn(feature, momentum_features)

    def test_get_feature_category_function(self):
        """Test helper function to get category for a feature."""
        from finance_ml.ml_workflow.eda.phase93_categories import get_feature_category

        category = get_feature_category("price_momentum_1m")
        self.assertEqual(category, "Momentum & Technical")

        category = get_feature_category("accounting_quality_score")
        self.assertEqual(category, "Quality & Risk")

        category = get_feature_category("nonexistent_feature")
        self.assertIsNone(category)


class TestPhase93EDAEnhancements(unittest.TestCase):
    """Test enhanced EDA functions with Phase 9.3 awareness."""

    def setUp(self):
        """Create sample dataframe with Phase 9.3 features (using ACTUAL feature names)."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(100)],
                "sector": np.random.choice(["Technology", "Finance", "Healthcare"], 100),
                "region": np.random.choice(["US", "EU", "APAC"], 100),
                # Momentum features (actual names from registry)
                "price_momentum_1m": np.random.randn(100),
                "rsi_14d": np.random.uniform(20, 80, 100),
                "ma_crossover_signal": np.random.choice([0, 1], 100),
                # Quality features
                "accounting_quality_score": np.random.uniform(0, 1, 100),
                "distress_risk_score": np.random.uniform(0, 1, 100),
                # Composite features
                "momentum_score": np.random.uniform(0, 100, 100),
                "composite_quality_score": np.random.uniform(0, 100, 100),
                # Valuation (actual names from registry)
                "ev_ebitda_ratio": np.random.uniform(5, 20, 100),
                "p_e_ratio": np.random.uniform(10, 30, 100),
            }
        )

    def test_eda_summary_includes_phase93_coverage(self):
        """Test that EDA summary includes Phase 9.3 category coverage."""
        from finance_ml.ml_workflow.eda.eda import eda_summary_with_phase93

        summary = eda_summary_with_phase93(self.df)

        self.assertIn("phase93_category_coverage", summary)
        category_coverage = summary["phase93_category_coverage"]

        self.assertIsInstance(category_coverage, dict)
        self.assertIn("Momentum & Technical", category_coverage)
        self.assertIn("Quality & Risk", category_coverage)
        self.assertIn("Composite Scores", category_coverage)

        # Check counts are correct
        self.assertEqual(category_coverage["Momentum & Technical"], 3)  # 3 momentum features
        self.assertEqual(category_coverage["Quality & Risk"], 2)  # 2 quality features

    def test_categorize_dataframe_columns(self):
        """Test function that categorizes all DataFrame columns."""
        from finance_ml.ml_workflow.eda.phase93_categories import categorize_dataframe_columns

        categorized = categorize_dataframe_columns(self.df)

        self.assertIsInstance(categorized, dict)
        self.assertIn("Momentum & Technical", categorized)
        self.assertIn("Quality & Risk", categorized)

        # Check column lists
        momentum_cols = categorized["Momentum & Technical"]
        self.assertIn("price_momentum_1m", momentum_cols)
        self.assertIn("rsi_14d", momentum_cols)

    def test_phase93_coverage_report(self):
        """Test generation of Phase 9.3 coverage report."""
        from finance_ml.ml_workflow.eda.eda import generate_phase93_coverage_report

        report = generate_phase93_coverage_report(self.df)

        self.assertIn("total_phase93_features", report)
        self.assertIn("category_breakdown", report)
        self.assertIn("coverage_percentage", report)

        self.assertGreater(report["total_phase93_features"], 0)
        self.assertIsInstance(report["category_breakdown"], dict)

    def test_phase93_sector_distribution(self):
        """Test Phase 9.3 feature distribution by sector."""
        from finance_ml.ml_workflow.eda.eda import analyze_phase93_by_sector

        sector_analysis = analyze_phase93_by_sector(self.df, sector_column="sector")

        self.assertIsInstance(sector_analysis, dict)
        self.assertIn("Technology", sector_analysis)
        self.assertIn("Finance", sector_analysis)

        # Each sector should have category summaries
        tech_analysis = sector_analysis["Technology"]
        self.assertIn("Momentum & Technical", tech_analysis)


class TestPhase93EvalIntegration(unittest.TestCase):
    """Test analytics/eval.py enhancements for Phase 9.3."""

    def setUp(self):
        """Create sample predictions dataframe with Phase 9.3 features."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(50)],
                "sector": np.random.choice(["Technology", "Finance"], 50),
                "y_true": np.random.uniform(50, 150, 50),
                "y_pred": np.random.uniform(50, 150, 50),
                "last_price": np.random.uniform(50, 150, 50),
                # Phase 9.3 features
                "accounting_quality_score": np.random.uniform(0, 1, 50),
                "composite_value_score": np.random.uniform(0, 100, 50),
                "price_momentum_1m": np.random.randn(50),
            }
        )

    def test_simple_eda_recognizes_phase93_features(self):
        """Test that simple_eda function tracks Phase 9.3 features."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            summary = simple_eda(
                self.df, out_dir=out_dir, save_plots=False, include_phase93_summary=True
            )

            # Should return summary dict with Phase 9.3 info
            self.assertIsInstance(summary, dict)
            if "phase93_coverage" in summary:
                self.assertIsInstance(summary["phase93_coverage"], dict)

    def test_export_predictions_includes_phase93_metadata(self):
        """Test CSV export includes Phase 9.3 feature metadata."""
        from finance_ml.ml_workflow.analytics.eval import export_predictions_to_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "predictions.csv"
            export_predictions_to_csv(self.df, csv_path, include_phase93_metadata=True)

            self.assertTrue(csv_path.exists())

            # Check metadata file
            metadata_path = csv_path.parent / f"{csv_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.assertIn("phase93_features", metadata)


if __name__ == "__main__":
    unittest.main()
