"""
TDD tests for cross-validation policy enforcement.

Tests for Phase 9.4 Task 4: Cross-Validation Policy Enforcement
Aligned with phase_9.4_implementation_plan.md

Test Coverage:
- Test 1: test_cv_policy_time_series_when_date_available
- Test 2: test_cv_policy_grouped_when_ticker_available
- Test 3: test_cv_policy_stratified_fallback

Business Objective: Prevent look-ahead bias in backtesting by enforcing
consistent CV strategy selection: time_series → grouped → stratified hierarchy.

Model Version: v9_10
Alignment: code_guidelines.md v1.10
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedKFold

from finance_ml.ml_workflow.classification.models import determine_cv_strategy


class TestCVPolicyEnforcement(unittest.TestCase):
    """Test suite for cross-validation policy enforcement."""

    def test_cv_policy_time_series_when_date_available(self):
        """Time-series CV should be used when date column exists."""
        # Given: Data with snapshot_date
        df = pd.DataFrame(
            {
                "snapshot_date": pd.date_range("2023-01-01", periods=100),
                "feature_1": np.random.randn(100),
                "target": np.random.randn(100),
                "ticker": ["AAPL"] * 100,
            }
        )

        # When: Determine CV strategy
        cv_strategy, cv_object = determine_cv_strategy(df, n_splits=5)

        # Then: TimeSeriesSplit selected
        self.assertEqual(cv_strategy, "time_series")
        self.assertIsInstance(cv_object, TimeSeriesSplit)
        self.assertEqual(cv_object.n_splits, 5)

    def test_cv_policy_grouped_when_ticker_available(self):
        """Grouped CV should be used when ticker column exists but no date."""
        # Given: Data with ticker but no snapshot_date
        # Need at least 5 unique tickers for n_splits=5
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
        df = pd.DataFrame(
            {
                "ticker": np.random.choice(tickers, size=100),
                "feature_1": np.random.randn(100),
                "target": np.random.randn(100),
            }
        )

        # When: Determine CV strategy
        cv_strategy, cv_object = determine_cv_strategy(df, n_splits=5)

        # Then: GroupKFold selected
        self.assertEqual(cv_strategy, "grouped")
        self.assertIsInstance(cv_object, GroupKFold)

        # Verify groups are by ticker
        groups = df["ticker"]
        for train_idx, test_idx in cv_object.split(df, groups=groups):
            train_tickers = set(df.iloc[train_idx]["ticker"])
            test_tickers = set(df.iloc[test_idx]["ticker"])
            # No ticker should appear in both train and test
            self.assertEqual(len(train_tickers & test_tickers), 0)

    def test_cv_policy_stratified_fallback(self):
        """Stratified CV should be fallback when no date or ticker."""
        # Given: Data without date or ticker columns
        df = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "target": np.random.choice([0, 1, 2], 100),  # 3-class target
            }
        )

        # When: Determine CV strategy
        cv_strategy, cv_object = determine_cv_strategy(df, target=df["target"], n_splits=5)

        # Then: StratifiedKFold selected
        self.assertEqual(cv_strategy, "stratified")
        self.assertIsInstance(cv_object, StratifiedKFold)

        # Verify stratification maintains class balance
        for train_idx, test_idx in cv_object.split(df, df["target"]):
            train_dist = df.iloc[train_idx]["target"].value_counts(normalize=True)
            test_dist = df.iloc[test_idx]["target"].value_counts(normalize=True)
            # Class distributions should be similar
            for cls in [0, 1, 2]:
                self.assertAlmostEqual(train_dist[cls], test_dist[cls], delta=0.15)


if __name__ == "__main__":
    unittest.main()
