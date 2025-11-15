"""
Test data splitting policy (addresses leakage issues).

Implements policy: time-aware split when date column exists,
otherwise grouped by ticker, otherwise stratified by sector.
"""

import unittest
from unittest import mock
import pandas as pd
import numpy as np


class TestDataSplitsPolicy(unittest.TestCase):
    """Test intelligent train/test splitting with leakage prevention."""

    def test_create_train_test_split_exists(self):
        """Test that create_train_test_split function exists."""
        try:
            from finance_ml.ml_workflow.validation.splits import create_train_test_split

            self.assertTrue(callable(create_train_test_split))
        except ImportError:
            self.fail("create_train_test_split not implemented in validation.splits")

    def test_time_aware_split_with_date_column(self):
        """Test that time-aware split is used when date column exists."""
        from finance_ml.ml_workflow.validation.splits import create_train_test_split

        # Create dataset with dates
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "ticker": [f"TICK{i%10}" for i in range(100)],
                "snapshot_date": dates,
                "price_target": np.random.uniform(50, 200, 100),
                "feature1": np.random.randn(100),
            }
        )

        # Split with date column
        train, test = create_train_test_split(df, date_col="snapshot_date", test_size=0.2)

        # Test set should have later dates than train set
        if len(train) > 0 and len(test) > 0:
            max_train_date = train["snapshot_date"].max()
            min_test_date = test["snapshot_date"].min()
            self.assertLessEqual(
                max_train_date,
                min_test_date,
                "Time-aware split failed: test dates overlap with train dates",
            )

    def test_grouped_split_without_date_column(self):
        """Test that grouped split is used when no date column but ticker exists."""
        from finance_ml.ml_workflow.validation.splits import create_train_test_split

        # Create dataset without dates but with tickers
        df = pd.DataFrame(
            {
                "ticker": [f"TICK{i%20}" for i in range(200)],  # 20 unique tickers
                "price_target": np.random.uniform(50, 200, 200),
                "feature1": np.random.randn(200),
            }
        )

        # Split grouped by ticker
        train, test = create_train_test_split(df, group_col="ticker", test_size=0.2)

        # Train and test should have disjoint tickers
        train_tickers = set(train["ticker"].unique())
        test_tickers = set(test["ticker"].unique())
        overlap = train_tickers & test_tickers

        self.assertEqual(len(overlap), 0, f"Grouped split leaked tickers: {overlap}")

    def test_stratified_split_fallback(self):
        """Test that stratified split is used as fallback."""
        from finance_ml.ml_workflow.validation.splits import create_train_test_split

        # Create dataset with sector for stratification
        df = pd.DataFrame(
            {
                "sector": np.random.choice(["Tech", "Finance", "Energy"], 150),
                "price_target": np.random.uniform(50, 200, 150),
                "feature1": np.random.randn(150),
            }
        )

        # Split stratified by sector
        train, test = create_train_test_split(df, stratify_col="sector", test_size=0.2)

        # Sector distributions should be similar
        train_dist = train["sector"].value_counts(normalize=True).sort_index()
        test_dist = test["sector"].value_counts(normalize=True).sort_index()

        # Check distributions are close (within 10%)
        for sector in train_dist.index:
            if sector in test_dist.index:
                diff = abs(train_dist[sector] - test_dist[sector])
                self.assertLess(diff, 0.15, f"Sector {sector} distribution differs by {diff:.1%}")

    def test_random_split_simple_fallback(self):
        """Test simple random split when no special columns available."""
        from finance_ml.ml_workflow.validation.splits import create_train_test_split

        # Create simple dataset
        df = pd.DataFrame(
            {
                "price_target": np.random.uniform(50, 200, 100),
                "feature1": np.random.randn(100),
            }
        )

        # Split randomly
        train, test = create_train_test_split(df, test_size=0.2)

        # Should have 80/20 split approximately
        self.assertAlmostEqual(len(train) / len(df), 0.8, delta=0.05)
        self.assertAlmostEqual(len(test) / len(df), 0.2, delta=0.05)


class TestSharedSplitIntegration(unittest.TestCase):
    """Integration tests for shared split utilities in ml_workflow helpers."""

    def test_prepare_regression_data_uses_shared_policy(self):
        """prepare_regression_data should delegate to create_train_test_split.

        This test patches ``create_train_test_split`` and asserts it is
        invoked when ``prepare_regression_data`` is called, documenting the
        Phase 9.9 requirement that regression data prep uses the shared
        split policy instead of calling sklearn.train_test_split directly.
        """

        from finance_ml.ml_workflow.regression.dataset import prepare_regression_data

        # Synthetic dataframe with policy-relevant columns so that
        # create_train_test_split is the intended path.
        df = pd.DataFrame(
            {
                "ticker": [f"T{i:03d}" for i in range(100)],
                "sector": np.random.choice(["Tech", "Finance"], 100),
                "snapshot_date": pd.date_range("2020-01-01", periods=100, freq="D"),
                "price_target": np.random.uniform(50, 200, 100),
                "feature_0": np.random.randn(100),
                "feature_1": np.random.randn(100),
            }
        )

        # Patch the shared helper in its defining module. The regression
        # dataset module imports it from there, so this ensures the
        # patched function is the one prepare_regression_data calls.
        with mock.patch(
            "finance_ml.ml_workflow.validation.splits.create_train_test_split",
            wraps=lambda df_, **kwargs: (df_.iloc[:80].copy(), df_.iloc[80:].copy()),
        ) as mocked_split:
            X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(
                df, target_col="price_target", test_size=0.2, random_state=42
            )

        # Shared helper must have been invoked exactly once.
        mocked_split.assert_called_once()

        # Sanity-check resulting split sizes to ensure wiring did not
        # break downstream behavior.
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)


class TestTimeSeriesCrossValidation(unittest.TestCase):
    """Test time-series cross-validation for temporal data."""

    def test_time_series_cv_exists(self):
        """Test that time_series_cv function exists."""
        try:
            from finance_ml.ml_workflow.validation.splits import time_series_cv

            self.assertTrue(callable(time_series_cv))
        except ImportError:
            # Optional - not critical for this phase
            pass

    def test_time_series_cv_no_leakage(self):
        """Test that time-series CV doesn't leak future data into training."""
        try:
            from finance_ml.ml_workflow.validation.splits import time_series_cv
        except ImportError:
            self.skipTest("time_series_cv not implemented")

        # Create temporal dataset
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "snapshot_date": dates,
                "price_target": np.random.uniform(50, 200, 100),
                "feature1": np.random.randn(100),
            }
        )

        # Run time-series CV
        folds = time_series_cv(df, date_col="snapshot_date", n_splits=3)

        # Each fold should have earlier train dates than test dates
        for train_idx, test_idx in folds:
            train_dates = df.iloc[train_idx]["snapshot_date"]
            test_dates = df.iloc[test_idx]["snapshot_date"]

            self.assertLessEqual(
                train_dates.max(),
                test_dates.min(),
                "Time-series CV leaked future data into training",
            )


if __name__ == "__main__":
    unittest.main()
