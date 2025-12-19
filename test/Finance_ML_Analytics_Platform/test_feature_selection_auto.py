"""
TDD tests for automated feature selection pipeline.

Tests for Phase 9.3 Task 1: Automated Feature Selection Pipeline
Aligned with phase_9.3_implementation_plan.md

Test Coverage:
- Test 1: select_features_by_importance_threshold
- Test 2: select_features_removes_correlated_redundancy
- Test 3: select_features_preserves_price_columns
- Test 4: select_features_by_category

Model Version: v9_10
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
        """Category-based selection should respect semantic groups using Phase 9.3 feature names."""
        # Given: Features from different Phase 9.3 categories using actual feature names
        # from PHASE93_FEATURE_CATEGORIES (phase93_categories.py)
        # Note: altman_z_score and piotroski_f_score are in "Composite Scores", not "Quality & Risk"
        X = pd.DataFrame(
            {
                # Momentum & Technical features (actual Phase 9.3 names)
                "rsi_14d": np.random.randn(100),
                "price_momentum_1m": np.random.randn(100),
                "ema_crossover_20_50": np.random.randn(100),
                # Valuation Ratios features (actual Phase 9.3 names)
                "p_e_ratio": np.random.randn(100),
                "ev_ebitda_ratio": np.random.randn(100),
                # Quality & Risk features (actual Phase 9.3 names from PHASE93_FEATURE_CATEGORIES)
                "accounting_quality_score": np.random.randn(100),
                "distress_risk_score": np.random.randn(100),
            }
        )
        y = np.random.randn(100)

        # When: Select only momentum category
        selected = select_features_by_category(X, categories=["momentum"])

        # Then: Only momentum features included
        self.assertIn("rsi_14d", selected.columns)
        self.assertIn("price_momentum_1m", selected.columns)
        self.assertIn("ema_crossover_20_50", selected.columns)
        self.assertNotIn("p_e_ratio", selected.columns)
        self.assertNotIn("accounting_quality_score", selected.columns)
        self.assertEqual(len(selected.columns), 3)  # Only 3 momentum features

        # When: Select multiple categories
        selected_multi = select_features_by_category(X, categories=["valuation", "quality"])

        # Then: Features from both categories included
        self.assertIn("p_e_ratio", selected_multi.columns)
        self.assertIn("ev_ebitda_ratio", selected_multi.columns)
        self.assertIn("accounting_quality_score", selected_multi.columns)
        self.assertIn("distress_risk_score", selected_multi.columns)
        self.assertNotIn("rsi_14d", selected_multi.columns)
        self.assertEqual(len(selected_multi.columns), 4)  # 2 valuation + 2 quality


if __name__ == "__main__":
    unittest.main()
