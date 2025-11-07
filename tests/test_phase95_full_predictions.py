"""
Test suite for Phase 9.5 full dataset prediction coverage.

This module tests that the regression pipeline can generate predictions
for ALL stocks in the dataset, not just the test split.

Issue: Phase 9.5.1 produces only 1,314 predictions out of 8,000 stocks (16.4%)
Goal: Generate predictions for 100% of stocks while maintaining data integrity
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


class TestPhase95FullPredictions(unittest.TestCase):
    """Test full dataset prediction coverage for Phase 9.5."""

    def setUp(self):
        """Create test data with 100 stocks."""
        np.random.seed(42)
        n_samples = 100

        self.df = pd.DataFrame({
            'ticker': [f'TICK{i:03d}' for i in range(n_samples)],
            'sector': np.random.choice(['Tech', 'Finance', 'Healthcare'], n_samples),
            'market_cap': np.random.uniform(1e9, 100e9, n_samples),
            'last_price': np.random.uniform(10, 500, n_samples),
            'revenue': np.random.uniform(1e8, 10e9, n_samples),
            'ebitda': np.random.uniform(1e7, 1e9, n_samples),
            'p_e': np.random.uniform(5, 50, n_samples),
            'price_target': np.random.uniform(15, 550, n_samples),
        })

        # Add some NaN values in features (not target) to simulate real data
        self.df.loc[self.df.sample(10, random_state=42).index, 'ebitda'] = np.nan
        self.df.loc[self.df.sample(5, random_state=43).index, 'p_e'] = np.nan

        # Create temporary output directory
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_train_and_evaluate_regression_predicts_full_dataset(self):
        """Test that train_and_evaluate_regression can predict for all stocks."""
        from finance_ml.models import train_and_evaluate_regression

        result = train_and_evaluate_regression(
            self.df,
            self.out_dir,
            n_jobs=1,
            dry_run=False
        )

        self.assertIsNotNone(result, "Result should not be None")
        self.assertIn('predictions', result, "Result should contain predictions")
        self.assertIn('full_predictions', result, "Result should contain full_predictions for all stocks")

        preds_df = result['predictions']
        full_preds_df = result['full_predictions']

        # Test set predictions (for validation metrics)
        self.assertIsInstance(preds_df, pd.DataFrame)
        self.assertGreater(len(preds_df), 0, "Should have test set predictions")

        # Full dataset predictions
        self.assertIsInstance(full_preds_df, pd.DataFrame)
        self.assertEqual(
            len(full_preds_df),
            len(self.df),
            f"Should have predictions for ALL {len(self.df)} stocks, got {len(full_preds_df)}"
        )

        # Verify all predictions are non-null
        null_preds = full_preds_df['y_pred'].isnull().sum()
        self.assertEqual(
            null_preds,
            0,
            f"All predictions should be non-null, found {null_preds} null values"
        )

    def test_full_predictions_csv_saved(self):
        """Test that full predictions are saved to CSV."""
        from finance_ml.models import train_and_evaluate_regression

        result = train_and_evaluate_regression(
            self.df,
            self.out_dir,
            n_jobs=1,
            dry_run=False
        )

        # Check that full predictions CSV exists
        full_preds_path = self.out_dir / "regression_predictions_full.csv"
        self.assertTrue(
            full_preds_path.exists(),
            f"Full predictions CSV should exist at {full_preds_path}"
        )

        # Load and verify
        full_preds = pd.read_csv(full_preds_path)
        self.assertEqual(
            len(full_preds),
            len(self.df),
            f"CSV should contain all {len(self.df)} predictions"
        )

    def test_full_predictions_with_missing_targets(self):
        """Test prediction for stocks without price targets."""
        # Remove targets from 20% of stocks (simulating real scenario)
        n_missing = int(len(self.df) * 0.2)
        missing_idx = self.df.sample(n_missing, random_state=44).index
        self.df.loc[missing_idx, 'price_target'] = np.nan

        from finance_ml.models import train_and_evaluate_regression

        result = train_and_evaluate_regression(
            self.df,
            self.out_dir,
            n_jobs=1,
            dry_run=False
        )

        self.assertIsNotNone(result)
        full_preds_df = result.get('full_predictions')

        # Should still predict for ALL stocks
        self.assertEqual(
            len(full_preds_df),
            len(self.df),
            "Should predict for all stocks even if some lack targets"
        )

        # Predictions for stocks without targets
        no_target_preds = full_preds_df.loc[missing_idx, 'y_pred']
        self.assertEqual(
            no_target_preds.isnull().sum(),
            0,
            "Should have predictions even for stocks without targets"
        )

    def test_no_data_leakage_in_metrics(self):
        """Test that validation metrics use only test set, not full predictions."""
        from finance_ml.models import train_and_evaluate_regression

        result = train_and_evaluate_regression(
            self.df,
            self.out_dir,
            n_jobs=1,
            dry_run=False
        )

        # Metrics should be computed on test set only
        test_preds = result['predictions']
        mae = result['mae']
        r2 = result['r2']

        # Verify metrics are computed from test set
        test_mae = np.mean(np.abs(test_preds['y_true'] - test_preds['y_pred']))
        self.assertAlmostEqual(
            mae,
            test_mae,
            places=4,
            msg="MAE should be computed from test set only (no data leakage)"
        )

    def test_full_predictions_have_metadata(self):
        """Test that full predictions include sector, ticker, market_cap."""
        from finance_ml.models import train_and_evaluate_regression

        result = train_and_evaluate_regression(
            self.df,
            self.out_dir,
            n_jobs=1,
            dry_run=False
        )

        full_preds = result['full_predictions']

        # Check metadata columns
        self.assertIn('sector', full_preds.columns)
        self.assertIn('ticker', full_preds.columns)
        self.assertIn('market_cap', full_preds.columns)
        self.assertIn('y_pred', full_preds.columns)

        # Verify metadata matches original data
        for idx in self.df.index[:10]:  # Check first 10
            if idx in full_preds.index:
                self.assertEqual(
                    full_preds.loc[idx, 'ticker'],
                    self.df.loc[idx, 'ticker']
                )


class TestPhase95RealWorldScenario(unittest.TestCase):
    """Test with 8,000 stock scenario from issue description."""

    def test_8000_stocks_scenario(self):
        """Test the exact scenario from issue: 8000 stocks, expect 8000 predictions."""
        np.random.seed(42)
        n_stocks = 8000

        # Create dataset similar to real scenario
        df = pd.DataFrame({
            'ticker': [f'TICK{i:04d}' for i in range(n_stocks)],
            'sector': np.random.choice([
                'Tech', 'Finance', 'Healthcare', 'Energy',
                'Consumer', 'Industrial', 'Materials'
            ], n_stocks),
            'market_cap': np.random.uniform(1e8, 100e9, n_stocks),
            'last_price': np.random.uniform(1, 1000, n_stocks),
            'revenue': np.random.uniform(1e7, 50e9, n_stocks),
            'ebitda': np.random.uniform(1e6, 5e9, n_stocks),
            'price_target': np.random.uniform(1, 1100, n_stocks),
        })

        # Simulate real data quality issues
        # 17.9% missing targets (1432 / 8000)
        n_missing_targets = int(n_stocks * 0.179)
        missing_target_idx = df.sample(n_missing_targets, random_state=45).index
        df.loc[missing_target_idx, 'price_target'] = np.nan

        # Some missing features
        df.loc[df.sample(500, random_state=46).index, 'ebitda'] = np.nan
        df.loc[df.sample(300, random_state=47).index, 'revenue'] = np.nan

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)

            from finance_ml.models import train_and_evaluate_regression

            result = train_and_evaluate_regression(
                df,
                out_dir,
                n_jobs=1,
                dry_run=False
            )

            self.assertIsNotNone(result)
            full_preds = result.get('full_predictions')

            # THE KEY TEST: Must predict for all 8,000 stocks
            self.assertEqual(
                len(full_preds),
                8000,
                "Must generate predictions for all 8,000 stocks (not just 1,314)"
            )

            # All predictions must be non-null
            null_count = full_preds['y_pred'].isnull().sum()
            self.assertEqual(
                null_count,
                0,
                f"All 8,000 predictions must be non-null, found {null_count} nulls"
            )

            # Test set should be smaller (20% of stocks with valid targets)
            test_preds = result['predictions']
            expected_test_size = int((n_stocks - n_missing_targets) * 0.2)
            self.assertAlmostEqual(
                len(test_preds),
                expected_test_size,
                delta=50,  # Allow some variance
                msg=f"Test set should be ~{expected_test_size} (20% of valid targets)"
            )


class TestPhase95EdgeCases(unittest.TestCase):
    """Test edge cases and error handling for full predictions."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        n_samples = 60  # Increased from 50 to ensure sufficient samples after filtering

        self.df = pd.DataFrame({
            'ticker': [f'TICK{i:03d}' for i in range(n_samples)],
            'sector': np.random.choice(['Tech', 'Finance'], n_samples),
            'market_cap': np.random.uniform(1e9, 100e9, n_samples),
            'last_price': np.random.uniform(10, 500, n_samples),
            'revenue': np.random.uniform(1e8, 10e9, n_samples),
            'price_target': np.random.uniform(15, 550, n_samples),
        })

        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_full_predictions_without_sector_column(self):
        """Test full predictions when sector column is missing."""
        df_no_sector = self.df.drop(columns=['sector'])
        
        from finance_ml.models import train_and_evaluate_regression
        
        result = train_and_evaluate_regression(
            df_no_sector,
            self.out_dir,
            n_jobs=1,
            dry_run=False
        )
        
        self.assertIsNotNone(result)
        full_preds = result.get('full_predictions')
        self.assertIsNotNone(full_preds)
        self.assertEqual(len(full_preds), len(df_no_sector))

    def test_full_predictions_without_optional_columns(self):
        """Test full predictions without ticker and market_cap."""
        df_minimal = self.df[['sector', 'last_price', 'revenue', 'price_target']].copy()
        
        from finance_ml.models import train_and_evaluate_regression
        
        result = train_and_evaluate_regression(
            df_minimal,
            self.out_dir,
            n_jobs=1,
            dry_run=False
        )
        
        self.assertIsNotNone(result)
        full_preds = result.get('full_predictions')
        self.assertIsNotNone(full_preds)
        self.assertEqual(len(full_preds), len(df_minimal))
        # Should still have predictions even without metadata
        self.assertIn('y_pred', full_preds.columns)

    def test_full_predictions_with_all_nan_features(self):
        """Test handling when some features are all NaN."""
        self.df['nan_feature'] = np.nan
        
        from finance_ml.models import train_and_evaluate_regression
        
        result = train_and_evaluate_regression(
            self.df,
            self.out_dir,
            n_jobs=1,
            dry_run=False
        )
        
        self.assertIsNotNone(result)
        full_preds = result.get('full_predictions')
        # Should still work with imputation handling all-NaN column
        self.assertIsNotNone(full_preds)

    def test_full_predictions_includes_y_true_when_available(self):
        """Test that full predictions include y_true column."""
        from finance_ml.models import train_and_evaluate_regression
        
        result = train_and_evaluate_regression(
            self.df,
            self.out_dir,
            n_jobs=1,
            dry_run=False
        )
        
        full_preds = result['full_predictions']
        self.assertIn('y_true', full_preds.columns)
        # Check that y_true matches original targets
        self.assertTrue(
            np.allclose(
                full_preds['y_true'].dropna(),
                self.df.loc[full_preds['y_true'].notna().values, 'price_target'],
                rtol=1e-10
            )
        )

    def test_full_predictions_error_metrics_only_for_known_targets(self):
        """Test that error metrics are only computed for stocks with known targets."""
        # Set some targets to NaN (use fewer to ensure sufficient training samples)
        missing_idx = self.df.sample(5, random_state=42).index
        self.df.loc[missing_idx, 'price_target'] = np.nan
        
        from finance_ml.models import train_and_evaluate_regression
        
        result = train_and_evaluate_regression(
            self.df,
            self.out_dir,
            n_jobs=1,
            dry_run=False
        )
        
        full_preds = result['full_predictions']
        
        # Stocks without targets should have NaN error metrics
        self.assertTrue(full_preds.loc[missing_idx, 'residual'].isna().all())
        self.assertTrue(full_preds.loc[missing_idx, 'abs_error'].isna().all())
        
        # Stocks with targets should have non-NaN error metrics
        has_target = ~self.df['price_target'].isna()
        has_target_idx = self.df[has_target].index
        # At least some should have error metrics
        self.assertTrue(full_preds.loc[has_target_idx, 'abs_error'].notna().any())


if __name__ == '__main__':
    unittest.main()
