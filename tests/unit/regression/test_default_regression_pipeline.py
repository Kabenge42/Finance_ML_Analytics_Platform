"""Test default regression entry point with safety rails and schema.

This test ensures that the default regression pipeline:

- Produces non-negative, bounded predictions via safety rails.
- Emits a predictions dataframe that satisfies the standardized schema
  validated by :func:`validate_predictions_schema`.
"""

import unittest

import numpy as np
import pandas as pd


class TestDefaultRegressionPipeline(unittest.TestCase):
    """Integration test for run_default_regression_pipeline."""

    def test_default_pipeline_non_negative_and_schema_valid(self):
        from finance_ml.ml_workflow.regression.pipeline import (
            run_default_regression_pipeline,
        )
        from finance_ml.ml_workflow.regression.io import validate_predictions_schema

        # Create a small synthetic dataset with positive prices/targets
        rng = np.random.RandomState(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "feature_1": rng.normal(loc=0.0, scale=1.0, size=n_samples),
                "feature_2": rng.normal(loc=5.0, scale=2.0, size=n_samples),
                "price_target": rng.uniform(10.0, 200.0, size=n_samples),
                "ticker": [f"T{i:03d}" for i in range(n_samples)],
                "sector": ["Tech"] * n_samples,
                "region": ["US"] * n_samples,
                "last_price": rng.uniform(5.0, 150.0, size=n_samples),
            }
        )

        preds_df = run_default_regression_pipeline(
            df,
            feature_cols=["feature_1", "feature_2"],
            target_col="price_target",
            test_size=0.25,
            random_state=0,
        )

        # Basic sanity checks
        self.assertFalse(preds_df.empty, "Predictions dataframe should not be empty")

        # Predictions must be non-negative due to safety rails
        self.assertTrue(
            (preds_df["y_pred"] >= 0).all(), "Found negative predictions in default pipeline"
        )

        # Core required columns should be present
        for col in ["y_true", "y_pred", "abs_error", "pct_error"]:
            self.assertIn(col, preds_df.columns, f"Missing required column: {col}")

        # Recommended metadata columns should be propagated when available
        for col in ["ticker", "sector", "region", "last_price"]:
            self.assertIn(col, preds_df.columns, f"Missing metadata column: {col}")

        # Schema validation must pass
        validation = validate_predictions_schema(preds_df)
        self.assertTrue(validation["ok"], f"Schema validation failed: {validation['errors']}")

    def test_default_pipeline_respects_time_aware_split_when_dates_available(self):
        """Pipeline should use time-aware split when a snapshot_date column is present.

        This is an integration check that run_default_regression_pipeline delegates
        its splitting logic to the shared create_train_test_split utility, which
        enforces the time-series policy (train on earlier dates, test on later).
        """

        from finance_ml.ml_workflow.regression.pipeline import (
            run_default_regression_pipeline,
        )

        rng = np.random.RandomState(123)
        n_samples = 60

        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        df = pd.DataFrame(
            {
                "feature_1": rng.normal(loc=0.0, scale=1.0, size=n_samples),
                "feature_2": rng.normal(loc=5.0, scale=2.0, size=n_samples),
                "price_target": rng.uniform(10.0, 200.0, size=n_samples),
                "snapshot_date": dates,
                "ticker": [f"T{i:03d}" for i in range(n_samples)],
            }
        )

        preds_df = run_default_regression_pipeline(
            df,
            feature_cols=["feature_1", "feature_2"],
            target_col="price_target",
            test_size=0.25,
            random_state=0,
            date_col="snapshot_date",
        )

        # Reconstruct the implicit train set as rows not used for predictions
        test_indices = preds_df.index
        train_indices = df.index.difference(test_indices)

        train_dates = df.loc[train_indices, "snapshot_date"]
        test_dates = df.loc[test_indices, "snapshot_date"]

        # Time-aware policy: all train dates should be <= all test dates
        self.assertLessEqual(
            train_dates.max(),
            test_dates.min(),
            "Default pipeline did not perform a time-aware split when dates were available",
        )


if __name__ == "__main__":
    unittest.main()
