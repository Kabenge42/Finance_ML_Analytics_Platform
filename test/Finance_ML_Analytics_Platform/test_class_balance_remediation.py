"""
TDD tests for classification class balance auto-remediation.

Tests for Phase 9.4 Task 5: Classification Class Balance Auto-Remediation
Aligned with phase_9.4_implementation_plan.md

Test Coverage:
- Test 1: test_class_balance_auto_resampling
- Test 2: test_class_balance_threshold_adjustment
- Test 3: test_class_balance_fallback_method

Business Objective: Ensure all market conditions are represented in training data
by automatically remediating class imbalance through SMOTE resampling, class weight
adjustment, or threshold tuning when imbalance >10:1 is detected.

Model Version: v9_10
Alignment: code_guidelines.md v1.10
"""

import unittest
import numpy as np
import pandas as pd

from finance_ml.ml_workflow.classification.models import balance_classes
from finance_ml.ml_workflow.classification.labels import create_enhanced_event_labels


class TestClassBalanceRemediation(unittest.TestCase):
    """Test suite for class balance auto-remediation."""

    def test_class_balance_auto_resampling(self):
        """SMOTE should be applied for severe class imbalance."""
        # Given: Highly imbalanced dataset
        X = pd.DataFrame({"feature_1": np.random.randn(1000), "feature_2": np.random.randn(1000)})
        y = pd.Series([0] * 950 + [1] * 50)  # 19:1 imbalance

        # When: Balance classes with auto-remediation
        X_balanced, y_balanced = balance_classes(X, y, method="auto", imbalance_threshold=10)

        # Then: Class distribution improved
        original_ratio = (y == 0).sum() / (y == 1).sum()
        balanced_ratio = (y_balanced == 0).sum() / (y_balanced == 1).sum()

        self.assertGreater(original_ratio, 10)
        self.assertLess(balanced_ratio, 5)  # Much more balanced

    def test_class_balance_threshold_adjustment(self):
        """Label thresholds should adapt when classes are missing."""
        # Given: Data that would produce all-neutral labels
        df = pd.DataFrame(
            {
                "ticker": ["STOCK" + str(i) for i in range(100)],
                "sector": ["Technology"] * 100,
                "price_target": [100] * 100,  # All same = no variation
                "last_price": [100] * 100,
                "momentum_rsi": [50] * 100,  # All neutral
            }
        )

        # When: Create labels with adaptive thresholds
        labels = create_enhanced_event_labels(
            df, method="comprehensive", auto_adjust_thresholds=True  # NEW parameter
        )

        # Then: Should have multiple classes despite uniform data
        unique_classes = len(np.unique(labels))
        self.assertGreater(unique_classes, 1)

        # Or should fall back to simpler method
        # Check that function didn't fail
        self.assertEqual(len(labels), len(df))

    def test_class_balance_fallback_method(self):
        """Should fall back to alternative labeling method if primary fails."""
        # Given: Data missing quality columns but with varying price targets
        np.random.seed(42)  # Reproducibility
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"] * 20,
                "sector": ["Technology"] * 60,
                "last_price": 100.0 + np.random.randn(60) * 10,  # Mean 100, varied
                "price_target": 100.0 + np.random.randn(60) * 30,  # Mean 100, more varied
                # Missing: altman_z_score, accounting_quality_score, etc.
            }
        )

        # When: Request quality_event method with fallback
        labels = create_enhanced_event_labels(
            df,
            method="quality_event",
            fallback_method="price_momentum",  # NEW parameter
            auto_adjust_thresholds=True,  # Ensure multiple classes
        )

        # Then: Should succeed with fallback method
        self.assertEqual(len(labels), len(df))
        self.assertGreater(len(np.unique(labels)), 1)

        # Verify warning logged about fallback
        # (requires capturing log output in test)


if __name__ == "__main__":
    unittest.main()
