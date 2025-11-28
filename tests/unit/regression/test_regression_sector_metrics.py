"""
Test sector-level regression metrics (Priority 1: Empty regression_metrics_by_sector.csv).

Ensures that train_and_evaluate_regression_by_sector is called and produces
non-empty metrics CSV file.
"""

import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np


class TestRegressionSectorMetrics(unittest.TestCase):
    """Test that sector-level metrics are computed and saved."""

    def setUp(self):
        """Create synthetic multi-sector dataset."""
        np.random.seed(42)
        n_samples = 300

        # Create dataset with 3 sectors
        sectors = ["Tech", "Finance", "Energy"]
        tickers = [f"TICK{i}" for i in range(n_samples)]

        self.df = pd.DataFrame(
            {
                "ticker": tickers,
                "sector": np.random.choice(sectors, n_samples),
                "last_price": np.random.uniform(50, 200, n_samples),
                "price_target": np.random.uniform(50, 200, n_samples),
                "market_cap": np.random.uniform(1e9, 1e12, n_samples),
                "pe_ratio": np.random.uniform(10, 30, n_samples),
                "debt_to_equity": np.random.uniform(0.1, 2.0, n_samples),
            }
        )

    def test_train_and_evaluate_regression_by_sector_exists(self):
        """Test that the sector evaluation function exists."""
        from finance_ml.ml_workflow.models import train_and_evaluate_regression_by_sector

        self.assertTrue(callable(train_and_evaluate_regression_by_sector))

    def test_train_and_evaluate_regression_by_sector_returns_metrics(self):
        """Test that sector evaluation returns per-sector metrics."""
        from finance_ml.ml_workflow.models import train_and_evaluate_regression_by_sector

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # Call function
            result = train_and_evaluate_regression_by_sector(self.df, out_dir)

            # Should return a DataFrame with sector-level metrics
            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0, "Sector metrics DataFrame is empty")

    def test_sector_metrics_csv_is_created(self):
        """Test that regression_metrics_by_sector.csv is created and non-empty."""
        from finance_ml.ml_workflow.models import train_and_evaluate_regression_by_sector

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # Call function
            train_and_evaluate_regression_by_sector(self.df, out_dir)

            # Check file exists
            metrics_file = out_dir / "regression_metrics_by_sector.csv"
            self.assertTrue(
                metrics_file.exists(), f"regression_metrics_by_sector.csv not created in {out_dir}"
            )

            # Check file is non-empty
            df_metrics = pd.read_csv(metrics_file)
            self.assertGreater(len(df_metrics), 0, "regression_metrics_by_sector.csv is empty")

    def test_sector_metrics_contains_required_columns(self):
        """Test that sector metrics contain MAE, RMSE, R² per sector."""
        from finance_ml.ml_workflow.models import train_and_evaluate_regression_by_sector

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            result = train_and_evaluate_regression_by_sector(self.df, out_dir)

            # Required columns per Model Optimization Recommendations
            required_cols = ["sector"]
            for col in required_cols:
                self.assertIn(col, result.columns, f"Missing required column: {col}")

            # Should have metric columns (MAE, RMSE, R2, etc.)
            metric_cols = [col for col in result.columns if col != "sector"]
            self.assertGreater(len(metric_cols), 0, "No metric columns found in sector metrics")

    def test_sector_metrics_covers_all_sectors(self):
        """Test that metrics are computed for all sectors in data."""
        from finance_ml.ml_workflow.models import train_and_evaluate_regression_by_sector

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            result = train_and_evaluate_regression_by_sector(self.df, out_dir)

            # Should have metrics for each sector
            expected_sectors = set(self.df["sector"].unique())
            actual_sectors = set(result["sector"].unique())

            self.assertEqual(
                expected_sectors,
                actual_sectors,
                f"Missing sectors in metrics. Expected: {expected_sectors}, Got: {actual_sectors}",
            )

    def test_sector_training_accepts_meta_features(self):
        """Test that sector training accepts meta-feature arguments (Phase 9.5)."""
        from finance_ml.ml_workflow.models import train_and_evaluate_regression_by_sector

        # Create probabilities aligned with df
        n_samples = len(self.df)
        probs = np.random.random((n_samples, 5))
        probs = probs / probs.sum(axis=1, keepdims=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # Call with new arguments
            result = train_and_evaluate_regression_by_sector(
                self.df,
                out_dir,
                feature_cols=["market_cap", "pe_ratio", "debt_to_equity"],
                use_meta_features=True,
                classification_probabilities=probs,
                cv_policy="time_series",
                date_col="snapshot_date" if "snapshot_date" in self.df.columns else None,
            )

            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
