"""
Test Phase 9.5 Sector-Specific Model Training with NaN Handling

This test module follows strict TDD to reproduce and fix the issue where
train_sector_specific_models() fails with 170+ columns containing NaN values.

Issue: ValueError: Feature matrix X contains NaN values in columns:
['price_target_ytd_ago', 'total_return_ytd', 'p_e_ntm', 'p_e_ltm',
'altman_z_score_fy']... (170 total)

Expected: 100% Phase 9.5 training success rate with zero NaN-related failures
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging

# Suppress verbose logging during tests
logging.getLogger("finance_ml").setLevel(logging.WARNING)


class TestPhase95SectorTrainingWithNaN(unittest.TestCase):
    """Test sector-specific training handles NaN values correctly."""

    def setUp(self):
        """Create test data with 170+ columns containing NaN values."""
        np.random.seed(42)
        n_samples = 200

        # Create base data with required columns
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Financials", "Healthcare"], n_samples),
                "last_price": np.random.uniform(10, 500, n_samples),
                "price_target": np.random.uniform(10, 600, n_samples),
                "market_cap": np.random.uniform(1e9, 1e12, n_samples),
            }
        )

        # Add 170+ columns with NaN values (simulating real data)
        nan_columns = [
            "price_target_ytd_ago",
            "total_return_ytd",
            "p_e_ntm",
            "p_e_ltm",
            "altman_z_score_fy",
            "enterprise_value",
            "p_e_fy",
            "p_b_fy",
            "p_s_ltm",
            "ev_ebitda_ltm",
            "dividend_yield_ltm",
            "roe_ltm",
            "roa_ltm",
            "debt_to_equity_fy",
            "current_ratio_fy",
        ]

        # Extend to 170+ columns
        for i in range(170):
            col_name = f"financial_metric_{i:03d}" if i >= len(nan_columns) else nan_columns[i]
            # Introduce NaN in 30-50% of values
            values = np.random.uniform(-100, 100, n_samples)
            nan_mask = np.random.random(n_samples) < 0.4
            values[nan_mask] = np.nan
            self.df[col_name] = values

        # Store the count of NaN columns for assertions
        self.nan_col_count = sum(self.df.isna().any())

    def test_dataset_has_170plus_nan_columns(self):
        """Verify test data reproduces the issue: 170+ columns with NaN."""
        nan_cols = self.df.columns[self.df.isna().any()].tolist()
        self.assertGreaterEqual(
            len(nan_cols), 170, f"Test setup failed: expected ≥170 NaN columns, got {len(nan_cols)}"
        )

    def test_train_sector_specific_models_now_handles_nan_data(self):
        """Test that training now succeeds even with NaN data (due to automatic imputation)."""
        from finance_ml.advanced_models import train_sector_specific_models

        # Get numeric columns excluding target and metadata
        exclude = ["ticker", "sector", "price_target"]
        feature_cols = [
            c for c in self.df.select_dtypes(include=[np.number]).columns if c not in exclude
        ]

        # Verify data has NaN before training
        nan_before = self.df[feature_cols].isna().sum().sum()
        self.assertGreater(nan_before, 0, "Test data should contain NaN values")

        # After fix, this should succeed because preprocessing is applied internally
        sector_models, sector_results = train_sector_specific_models(
            df=self.df,
            feature_cols=feature_cols,
            target_col="price_target",
            sector_col="sector",
            model_type="random_forest",
            random_state=42,
            ensure_nonnegative=False,
            auto_extract_fallback=False,
        )

        # Should succeed and train regression
        self.assertIsInstance(sector_models, dict)
        self.assertGreater(len(sector_models), 0, "Should train at least one sector model")

    def test_prepare_features_for_training_removes_nans(self):
        """Test that prepare_features_for_training removes all NaN values."""
        from finance_ml.advanced_models import prepare_features_for_training

        # Get feature columns
        exclude = ["ticker", "sector", "price_target"]
        feature_cols = [
            c for c in self.df.select_dtypes(include=[np.number]).columns if c not in exclude
        ]

        # Prepare features with imputation
        X, y = prepare_features_for_training(
            df=self.df,
            feature_cols=feature_cols,
            target_col="price_target",
            apply_imputation=True,
            sector_column="sector",
        )

        # Verify no NaN values remain
        nan_count = X.isna().sum().sum()
        self.assertEqual(nan_count, 0, f"Expected 0 NaN values after preparation, got {nan_count}")

    def test_train_sector_specific_models_with_auto_extract_succeeds(self):
        """Test that training succeeds when auto_extract_fallback=True."""
        from finance_ml.advanced_models import train_sector_specific_models

        # Use auto_extract_fallback to enable preprocessing
        sector_models, sector_results = train_sector_specific_models(
            df=self.df,
            feature_cols=["last_price", "market_cap"],  # Simplified feature set
            target_col="price_target",
            sector_col="sector",
            model_type="random_forest",
            random_state=42,
            ensure_nonnegative=False,
            auto_extract_fallback=True,  # Enable auto-extraction and preprocessing
        )

        # Should succeed without errors
        self.assertIsInstance(sector_models, dict)
        self.assertIsInstance(sector_results, dict)
        self.assertGreater(len(sector_models), 0)


class TestPhase95PreprocessingIntegration(unittest.TestCase):
    """Test integration of preprocessing pipeline with sector training."""

    def setUp(self):
        """Create realistic test data."""
        np.random.seed(42)
        n_samples = 150

        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Financials", "Healthcare"], n_samples),
                "last_price": np.random.uniform(10, 500, n_samples),
                "price_target": np.random.uniform(10, 600, n_samples),
                "market_cap": np.random.uniform(1e9, 1e12, n_samples),
            }
        )

        # Add financial metrics with some NaN
        for i in range(50):
            values = np.random.uniform(-10, 10, n_samples)
            if i % 3 == 0:  # Every 3rd column has NaN
                nan_mask = np.random.random(n_samples) < 0.2
                values[nan_mask] = np.nan
            self.df[f"metric_{i:02d}"] = values

    def test_prepare_phase95_data_before_sector_training(self):
        """Test that prepare_phase95_data() cleans data before sector training."""
        from finance_ml.advanced_preprocessing import prepare_phase95_data
        from finance_ml.advanced_models import train_sector_specific_models

        # First, prepare data using Phase 9.5 preprocessing
        df_clean = prepare_phase95_data(
            df=self.df, sector_column="sector", price_column="last_price", n_neighbors=5
        )

        # Verify data is clean
        self.assertEqual(df_clean.isna().sum().sum(), 0)

        # Now train sector regression on clean data
        feature_cols = [
            c
            for c in df_clean.select_dtypes(include=[np.number]).columns
            if c not in ["price_target"]
        ]

        sector_models, sector_results = train_sector_specific_models(
            df=df_clean,
            feature_cols=feature_cols,
            target_col="price_target",
            sector_col="sector",
            model_type="random_forest",
            random_state=42,
            ensure_nonnegative=False,
            auto_extract_fallback=False,  # Should work without fallback now
        )

        # Should succeed
        self.assertGreater(len(sector_models), 0)
        self.assertGreater(len(sector_results), 0)

    def test_sector_training_applies_imputation_internally(self):
        """Test that sector training can apply imputation internally."""
        from finance_ml.advanced_models import train_sector_specific_models

        # Include columns with NaN
        feature_cols = [
            c
            for c in self.df.select_dtypes(include=[np.number]).columns
            if c not in ["price_target"]
        ]

        # Count NaN before
        nan_before = self.df[feature_cols].isna().sum().sum()
        self.assertGreater(nan_before, 0, "Test data should have NaN values")

        # Train with auto_extract_fallback (should apply preprocessing)
        sector_models, sector_results = train_sector_specific_models(
            df=self.df,
            feature_cols=feature_cols,
            target_col="price_target",
            sector_col="sector",
            model_type="random_forest",
            random_state=42,
            ensure_nonnegative=False,
            auto_extract_fallback=True,
        )

        # Should succeed despite NaN in input
        self.assertGreater(len(sector_models), 0)


class TestPhase95NotebookIntegration(unittest.TestCase):
    """Test the complete Phase 9.5 workflow as used in notebook."""

    def test_complete_phase95_workflow(self):
        """Test the full Phase 9.5 workflow: prepare data, then train regression."""
        from finance_ml.advanced_preprocessing import prepare_phase95_data
        from finance_ml.advanced_models import (
            train_sector_specific_models,
            extract_numeric_feature_columns,
        )

        np.random.seed(42)
        n_samples = 200

        # Simulate data from Phase 9.4 with NaN values
        df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Financials", "Healthcare"], n_samples),
                "last_price": np.random.uniform(10, 500, n_samples),
                "price_target": np.random.uniform(10, 600, n_samples),
                "market_cap": np.random.uniform(1e9, 1e12, n_samples),
            }
        )

        # Add problematic columns from traceback
        problematic_cols = [
            "price_target_ytd_ago",
            "total_return_ytd",
            "p_e_ntm",
            "p_e_ltm",
            "altman_z_score_fy",
        ]
        for col in problematic_cols:
            values = np.random.uniform(0, 100, n_samples)
            values[np.random.random(n_samples) < 0.3] = np.nan
            df[col] = values

        # Add more features
        for i in range(20):
            df[f"feature_{i:02d}"] = np.random.uniform(-10, 10, n_samples)

        # Step 1: Verify data has NaN
        initial_nan = df.isna().sum().sum()
        self.assertGreater(initial_nan, 0)

        # Step 2: Apply Phase 9.5 preprocessing
        df_clean = prepare_phase95_data(df=df, sector_column="sector", price_column="last_price")

        # Step 3: Verify all NaN removed
        final_nan = df_clean.isna().sum().sum()
        self.assertEqual(final_nan, 0)

        # Step 4: Extract features (excluding target and metadata)
        feature_cols = extract_numeric_feature_columns(
            df=df_clean,
            exclude_cols=["price_target", "ticker"],
            exclude_patterns=["target", "ticker"],
        )

        # Step 5: Train sector-specific regression (should succeed)
        sector_models, sector_results = train_sector_specific_models(
            df=df_clean,
            feature_cols=feature_cols,
            target_col="price_target",
            sector_col="sector",
            model_type="random_forest",
            random_state=42,
            min_samples=20,
            ensure_nonnegative=False,
            auto_extract_fallback=False,  # Clean data shouldn't need fallback
        )

        # Verify success
        self.assertIsInstance(sector_models, dict)
        self.assertGreater(len(sector_models), 0)

        # Verify each sector has a model
        sectors = df_clean["sector"].unique()
        for sector in sectors:
            sector_data = df_clean[df_clean["sector"] == sector]
            if len(sector_data) >= 20:  # min_samples threshold
                self.assertIn(sector, sector_models)


class TestPhase95EdgeCases(unittest.TestCase):
    """Test edge cases in Phase 9.5 sector training."""

    def test_sector_with_insufficient_samples_skipped(self):
        """Test that sectors with < min_samples are skipped gracefully."""
        from finance_ml.advanced_models import train_sector_specific_models

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "sector": ["Tech"] * 10 + ["Finance"] * 5,  # Finance has < 20 samples
                "last_price": np.random.uniform(10, 100, 15),
                "market_cap": np.random.uniform(1e9, 1e10, 15),
                "price_target": np.random.uniform(10, 120, 15),
            }
        )

        sector_models, sector_results = train_sector_specific_models(
            df=df,
            feature_cols=["last_price", "market_cap"],
            target_col="price_target",
            sector_col="sector",
            model_type="random_forest",
            min_samples=20,  # Higher than both sectors
            random_state=42,
        )

        # Both sectors should be skipped
        self.assertEqual(len(sector_models), 0)

    def test_all_nan_target_values_handled(self):
        """Test that rows with NaN targets are dropped before training."""
        from finance_ml.advanced_models import train_sector_specific_models

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "sector": ["Tech"] * 50,
                "last_price": np.random.uniform(10, 100, 50),
                "market_cap": np.random.uniform(1e9, 1e10, 50),
                "price_target": [np.nan] * 20 + list(np.random.uniform(10, 120, 30)),
            }
        )

        # Should handle NaN targets gracefully
        sector_models, sector_results = train_sector_specific_models(
            df=df,
            feature_cols=["last_price", "market_cap"],
            target_col="price_target",
            sector_col="sector",
            model_type="random_forest",
            min_samples=20,
            random_state=42,
        )

        # Should train on 30 samples (after dropping 20 NaN targets)
        self.assertGreater(len(sector_models), 0)


if __name__ == "__main__":
    unittest.main()
