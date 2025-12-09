"""
TDD tests for automated feature selection pipeline.

Tests for Phase 9.3 Task 1: Automated Feature Selection Pipeline
Aligned with phase_9.3_implementation_plan.md

Test Coverage:
- Test 1: select_features_by_importance_threshold
- Test 2: select_features_removes_correlated_redundancy
- Test 3: select_features_preserves_price_columns
- Test 4: select_features_by_category

Model Version: v9_9
Alignment: code_guidelines.md v1.10
"""

import unittest
import numpy as np
import pandas as pd

from finance_ml.ml_workflow.features.selection import (
    select_features_auto,
    select_features_by_category,
)


class TestFeatureSelectionAuto(unittest.TestCase):
    """Test suite for automated feature selection."""

    def test_select_features_by_importance_threshold(self):
        """Features below importance threshold should be removed."""
        # Given: DataFrame with features of varying importance
        X = pd.DataFrame(
            {
                "high_importance": np.random.randn(100),
                "medium_importance": np.random.randn(100) * 0.5,
                "low_importance": np.random.randn(100) * 0.01,
                "last_price": np.random.uniform(10, 100, 100),  # Price column
            }
        )
        y = X["high_importance"] * 2 + np.random.randn(100)

        # When: Select features with threshold=0.05
        selected = select_features_auto(X, y, importance_threshold=0.05, method="mutual_info")

        # Then: Low importance removed, price column preserved
        self.assertIn("high_importance", selected.columns)
        self.assertIn("last_price", selected.columns)
        self.assertNotIn("low_importance", selected.columns)

    def test_select_features_removes_correlated_redundancy(self):
        """Highly correlated features should be deduplicated."""
        # Given: Features with high correlation
        X = pd.DataFrame({"feature_a": np.random.randn(100), "feature_b": np.random.randn(100)})
        X["feature_b_duplicate"] = X["feature_b"] + np.random.randn(100) * 0.01
        y = X["feature_a"] + np.random.randn(100)

        # When: Select with correlation_threshold=0.95
        selected = select_features_auto(X, y, correlation_threshold=0.95, method="correlation")

        # Then: Only one of correlated pair kept
        correlated_kept = sum(
            ["feature_b" in selected.columns, "feature_b_duplicate" in selected.columns]
        )
        self.assertEqual(correlated_kept, 1)

    def test_select_features_preserves_price_columns(self):
        """Price columns must never be removed by selection."""
        # Given: Price columns with low calculated importance
        price_cols = ["last_price", "price_target", "price_target_median"]
        X = pd.DataFrame(
            {col: np.random.uniform(10, 100, 100) for col in price_cols + ["unrelated_feature"]}
        )
        y = np.random.randn(100)  # Target unrelated to price

        # When: Select features with strict threshold
        selected = select_features_auto(
            X, y, importance_threshold=0.9, method="mutual_info"  # Very strict
        )

        # Then: All price columns preserved
        for col in price_cols:
            self.assertIn(col, selected.columns)

    def test_select_features_by_category(self):
        """Category-based selection should respect semantic groups."""
        # Given: Features from different Phase 9.3 categories
        X = pd.DataFrame(
            {
                "momentum_rsi": np.random.randn(100),
                "momentum_macd": np.random.randn(100),
                "valuation_pe": np.random.randn(100),
                "quality_altman_z": np.random.randn(100),
            }
        )
        y = np.random.randn(100)

        # When: Select only momentum category
        selected = select_features_by_category(X, categories=["momentum"])

        # Then: Only momentum features included
        self.assertIn("momentum_rsi", selected.columns)
        self.assertIn("momentum_macd", selected.columns)
        self.assertNotIn("valuation_pe", selected.columns)


if __name__ == "__main__":
    unittest.main()
