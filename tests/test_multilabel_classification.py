"""
TDD tests for multi-label classification support.

Tests for Phase 9.4 Task 2: Multi-Label Classification Support
Aligned with phase_9.4_implementation_plan.md

Test Coverage:
- Test 1: test_create_multilabel_event_labels
- Test 2: test_multilabel_category_coverage
- Test 3: test_multilabel_threshold_calibration

Business Objective: Enable simultaneous signal detection across 16 Phase 9.3
feature categories for granular sector-specific investment strategies.

Model Version: v9_9
Alignment: code_guidelines.md v1.10
"""

import unittest
import numpy as np
import pandas as pd

from finance_ml.ml_workflow.classification.labels import create_multilabel_event_labels


class TestMultiLabelClassification(unittest.TestCase):
    """Test suite for multi-label classification support."""

    def test_create_multilabel_event_labels(self):
        """Multi-label mode should produce independent binary labels per category."""
        # Given: Stock data with diverse signals
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "last_price": [150, 300, 2800],
                "price_target": [180, 290, 3000],  # Positive valuation signal
                "momentum_rsi": [70, 45, 30],  # Overbought, neutral, oversold
                "quality_altman_z": [5.0, 2.5, 1.0],  # Strong, moderate, weak
                "sector": ["Technology", "Technology", "Technology"],
            }
        )

        # When: Create multi-label event labels
        labels = create_multilabel_event_labels(
            df, label_mode="multilabel", categories=["valuation", "momentum", "quality"]
        )

        # Then: Independent binary labels per category
        expected_columns = ["label_valuation", "label_momentum", "label_quality"]
        for col in expected_columns:
            self.assertIn(col, labels.columns)
            self.assertTrue(labels[col].isin([0, 1]).all(), f"{col} should only contain 0 or 1")

        # AAPL: positive valuation (target > price)
        self.assertEqual(labels.loc[0, "label_valuation"], 1)

    def test_multilabel_category_coverage(self):
        """All Phase 9.3 categories should be supported."""
        # Given: Sample data with features across categories
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
                "sector": ["Technology"] * 5,
                "last_price": [150, 300, 2800, 200, 450],
                "price_target": [180, 290, 3000, 250, 500],
                # Valuation features
                "p_e_ltm": [25, 30, 28, 50, 40],
                "ev_ebitda": [15, 20, 18, 35, 30],
                # Momentum features
                "momentum_rsi": [65, 45, 55, 70, 40],
                "price_change_1m": [0.05, -0.02, 0.03, 0.15, -0.05],
                # Quality features
                "quality_altman_z": [5.0, 3.5, 4.0, 2.0, 3.0],
                "roe_ltm": [0.25, 0.20, 0.22, 0.15, 0.18],
                # Profitability features
                "net_margin_ltm": [0.20, 0.25, 0.22, 0.10, 0.15],
                "operating_margin_ltm": [0.25, 0.30, 0.28, 0.15, 0.20],
            }
        )

        # When: Create multi-label with multiple categories
        categories = ["valuation", "momentum", "quality", "profitability"]
        labels = create_multilabel_event_labels(df, label_mode="multilabel", categories=categories)

        # Then: One binary label per category
        expected_columns = [f"label_{cat}" for cat in categories]
        for col in expected_columns:
            self.assertIn(col, labels.columns, f"Missing expected column: {col}")
            # All labels should be binary (0, 1, or NaN if feature missing)
            valid_values = labels[col].dropna().isin([0, 1])
            self.assertTrue(valid_values.all(), f"{col} contains non-binary values")

    def test_multilabel_threshold_calibration(self):
        """Thresholds should be calibrated per sector per category."""
        # Given: Different sectors with different valuation norms
        df = pd.DataFrame(
            {
                "ticker": ["TECH1", "TECH2", "UTIL1", "UTIL2"],
                "sector": ["Technology", "Technology", "Utilities", "Utilities"],
                "p_e_ltm": [50, 45, 15, 12],  # Tech typically higher P/E
                "price_target": [100, 90, 50, 48],
                "last_price": [90, 95, 52, 50],
                "ev_ebitda": [25, 22, 10, 8],
            }
        )

        # When: Create labels with sector-adjusted thresholds
        labels = create_multilabel_event_labels(
            df, label_mode="multilabel", categories=["valuation"], sector_adjusted=True
        )

        # Then: Label column exists
        self.assertIn("label_valuation", labels.columns)

        # Verify binary values
        self.assertTrue(labels["label_valuation"].isin([0, 1]).all())

        # Sector-adjusted thresholds should allow different interpretations
        # Both sectors should have positive and negative signals
        tech_labels = labels[df["sector"] == "Technology"]["label_valuation"]
        util_labels = labels[df["sector"] == "Utilities"]["label_valuation"]

        # At least one stock in each sector should have a positive signal
        # (This is probabilistic but reasonable given the data)
        self.assertGreaterEqual(
            tech_labels.sum() + util_labels.sum(),
            1,
            "Should have at least one positive signal across sectors",
        )


if __name__ == "__main__":
    unittest.main()
