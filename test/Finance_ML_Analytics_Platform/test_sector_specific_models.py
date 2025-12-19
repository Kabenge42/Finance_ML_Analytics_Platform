"""
TDD Test Suite for Task 10.3: Sector-Specific Model Training

Tests dedicated models for high-error sectors with sector-specific feature engineering
and hyperparameter optimization.

Requirements from finance_ml_improvement_plan.md Task 10.3:
- Train dedicated models for high-error sectors (Real Estate, Materials, Energy)
- Implement sector-specific feature engineering (e.g., commodity prices for Energy)
- Add sector-specific hyperparameter tuning with Optuna
- Export sector model performance comparison report
- Target: Reduce Real Estate error from 518% to <200%, Materials/Energy from 295%/283% to <150%

Test Categories:
1. Sector-specific model training (Real Estate, Materials, Energy)
2. Sector-specific feature engineering
3. Optuna hyperparameter tuning per sector
4. Performance comparison vs. global baseline
5. Graceful fallback for insufficient samples
6. Performance report export
"""

import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Import existing modules
from finance_ml.ml_workflow.regression.dataset import prepare_regression_data
from finance_ml.ml_workflow.regression.sector_models import (
    train_high_error_sector_models,
    add_sector_specific_features,
    optimize_sector_hyperparameters_optuna,
    compare_sector_vs_global_performance,
    export_sector_performance_report,
)
from finance_ml.ml_workflow.regression.models import train_xgboost_regressor
from sklearn.metrics import mean_absolute_percentage_error


class TestSectorSpecificModels(unittest.TestCase):
    """Test suite for sector-specific model training and optimization."""

    def setUp(self):
        """Create synthetic data for testing with sector-specific patterns."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 15

        # Create features
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # High-error sectors: Real Estate, Materials, Energy
        # Other sectors: Technology, Healthcare, Financials
        self.sectors = np.random.choice(
            ["Real Estate", "Materials", "Energy", "Technology", "Healthcare", "Financials"],
            size=n_samples,
            p=[0.15, 0.15, 0.15, 0.20, 0.20, 0.15],  # Balanced distribution
        )
        self.X["sector"] = self.sectors

        # Create sector-specific features that should help prediction
        # Energy: commodity price sensitivity (feature_0 as proxy)
        # Real Estate: leverage ratio (feature_1 as proxy)
        # Materials: commodity exposure (feature_2 as proxy)

        # Base target with sector-specific signal
        base_signal = (
            self.X["feature_0"] * 5 + self.X["feature_1"] * 3 + self.X["feature_2"] * 2
        ) + 100

        # Sector-specific adjustments with different patterns
        sector_adjustments = {
            "Real Estate": self.X["feature_1"] * 20,  # High leverage sensitivity
            "Materials": self.X["feature_2"] * 15,  # Commodity exposure
            "Energy": self.X["feature_0"] * 25,  # Commodity price sensitivity
            "Technology": self.X["feature_3"] * 10,  # Growth premium
            "Healthcare": self.X["feature_4"] * 8,  # Defensive characteristics
            "Financials": self.X["feature_5"] * 12,  # Interest rate sensitivity
        }

        # Apply sector-specific adjustments with sector-specific noise
        self.y = pd.Series(np.zeros(n_samples), name="price_target")
        for sector, adjustment in sector_adjustments.items():
            mask = self.sectors == sector
            # Higher noise for high-error sectors
            noise_level = 30 if sector in ["Real Estate", "Materials", "Energy"] else 10
            noise = np.random.randn(mask.sum()) * noise_level
            self.y[mask] = base_signal[mask] + adjustment[mask] + noise

        # Ensure non-negative prices
        self.y = self.y.clip(lower=0)

        # Add market cap for sector-specific features
        self.X["market_cap"] = np.exp(np.random.randn(n_samples) * 2 + 20)

    def test_train_high_error_sector_models_function_exists(self):
        """Test that train_high_error_sector_models function exists."""
        try:
            from finance_ml.ml_workflow.regression.sector_models import (
                train_high_error_sector_models,
            )

            self.assertTrue(callable(train_high_error_sector_models))
        except ImportError:
            self.fail("train_high_error_sector_models function not implemented")

    def test_add_sector_specific_features_function_exists(self):
        """Test that add_sector_specific_features function exists."""
        try:
            from finance_ml.ml_workflow.regression.sector_models import add_sector_specific_features

            self.assertTrue(callable(add_sector_specific_features))
        except ImportError:
            self.fail("add_sector_specific_features function not implemented")

    def test_optimize_sector_hyperparameters_optuna_exists(self):
        """Test that optimize_sector_hyperparameters_optuna function exists."""
        try:
            from finance_ml.ml_workflow.regression.sector_models import (
                optimize_sector_hyperparameters_optuna,
            )

            self.assertTrue(callable(optimize_sector_hyperparameters_optuna))
        except ImportError:
            self.fail("optimize_sector_hyperparameters_optuna function not implemented")

    def test_compare_sector_vs_global_performance_exists(self):
        """Test that compare_sector_vs_global_performance function exists."""
        try:
            from finance_ml.ml_workflow.regression.sector_models import (
                compare_sector_vs_global_performance,
            )

            self.assertTrue(callable(compare_sector_vs_global_performance))
        except ImportError:
            self.fail("compare_sector_vs_global_performance function not implemented")

    def test_export_sector_performance_report_exists(self):
        """Test that export_sector_performance_report function exists."""
        try:
            from finance_ml.ml_workflow.regression.sector_models import (
                export_sector_performance_report,
            )

            self.assertTrue(callable(export_sector_performance_report))
        except ImportError:
            self.fail("export_sector_performance_report function not implemented")

    def test_train_models_for_high_error_sectors_only(self):
        """Test training dedicated models for Real Estate, Materials, Energy only."""
        from finance_ml.ml_workflow.regression.sector_models import train_high_error_sector_models

        # Split data
        train_idx = int(0.7 * len(self.X))
        X_train = self.X.iloc[:train_idx]
        y_train = self.y.iloc[:train_idx]

        # Train sector-specific models
        result = train_high_error_sector_models(
            X_train, y_train, sectors=["Real Estate", "Materials", "Energy"], random_state=42
        )

        # Should return models dict and metrics dict
        self.assertIn("models", result)
        self.assertIn("metrics", result)

        # Should have models for exactly the 3 high-error sectors
        models = result["models"]
        self.assertEqual(len(models), 3)
        self.assertIn("Real Estate", models)
        self.assertIn("Materials", models)
        self.assertIn("Energy", models)

        # Each model should be trained and callable
        for sector, model in models.items():
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "predict"))

    def test_sector_specific_feature_engineering_energy(self):
        """Test sector-specific features for Energy sector (commodity exposure)."""
        from finance_ml.ml_workflow.regression.sector_models import add_sector_specific_features

        # Filter to Energy sector
        energy_mask = self.X["sector"] == "Energy"
        X_energy = self.X[energy_mask].copy()

        # Add sector-specific features
        X_enhanced = add_sector_specific_features(X_energy, sector="Energy")

        # Should have additional Energy-specific features
        # Examples: commodity_exposure, volatility_ratio, etc.
        original_cols = set(X_energy.columns)
        enhanced_cols = set(X_enhanced.columns)
        new_features = enhanced_cols - original_cols

        self.assertGreater(
            len(new_features), 0, "add_sector_specific_features should add new features for Energy"
        )

        # Should not have NaN values
        self.assertFalse(
            X_enhanced.isnull().any().any(),
            "Sector-specific features should not introduce NaN values",
        )

    def test_sector_specific_feature_engineering_real_estate(self):
        """Test sector-specific features for Real Estate (leverage ratios)."""
        from finance_ml.ml_workflow.regression.sector_models import add_sector_specific_features

        # Filter to Real Estate sector
        re_mask = self.X["sector"] == "Real Estate"
        X_re = self.X[re_mask].copy()

        # Add sector-specific features
        X_enhanced = add_sector_specific_features(X_re, sector="Real Estate")

        # Should have additional Real Estate-specific features
        original_cols = set(X_re.columns)
        enhanced_cols = set(X_enhanced.columns)
        new_features = enhanced_cols - original_cols

        self.assertGreater(
            len(new_features),
            0,
            "add_sector_specific_features should add new features for Real Estate",
        )

    def test_sector_specific_feature_engineering_materials(self):
        """Test sector-specific features for Materials (commodity prices)."""
        from finance_ml.ml_workflow.regression.sector_models import add_sector_specific_features

        # Filter to Materials sector
        mat_mask = self.X["sector"] == "Materials"
        X_mat = self.X[mat_mask].copy()

        # Add sector-specific features
        X_enhanced = add_sector_specific_features(X_mat, sector="Materials")

        # Should have additional Materials-specific features
        original_cols = set(X_mat.columns)
        enhanced_cols = set(X_enhanced.columns)
        new_features = enhanced_cols - original_cols

        self.assertGreater(
            len(new_features),
            0,
            "add_sector_specific_features should add new features for Materials",
        )

    def test_optuna_hyperparameter_tuning_per_sector(self):
        """Test Optuna hyperparameter optimization for sector-specific models."""
        from finance_ml.ml_workflow.regression.sector_models import (
            optimize_sector_hyperparameters_optuna,
        )

        # Split data
        train_idx = int(0.7 * len(self.X))
        X_train = self.X.iloc[:train_idx]
        y_train = self.y.iloc[:train_idx]

        # Filter to one sector for faster testing
        sector = "Energy"
        sector_mask = X_train["sector"] == sector
        X_sector = X_train[sector_mask].drop(columns=["sector"])
        y_sector = y_train[sector_mask]

        # Run Optuna optimization (small n_trials for testing)
        result = optimize_sector_hyperparameters_optuna(
            X_sector,
            y_sector,
            sector=sector,
            n_trials=5,  # Small number for testing
            random_state=42,
        )

        # Should return best_params and best_score
        self.assertIn("best_params", result)
        self.assertIn("best_score", result)

        # best_params should be a dict with hyperparameters
        best_params = result["best_params"]
        self.assertIsInstance(best_params, dict)
        self.assertGreater(len(best_params), 0)

        # best_score should be a numeric value (negative MAE or similar)
        best_score = result["best_score"]
        self.assertIsInstance(best_score, (int, float))

    def test_sector_model_vs_global_performance_comparison(self):
        """Test comparison of sector-specific vs. global model performance."""
        from finance_ml.ml_workflow.regression.sector_models import (
            train_high_error_sector_models,
            compare_sector_vs_global_performance,
        )
        from finance_ml.ml_workflow.regression.models import train_xgboost_regressor

        # Split data
        train_idx = int(0.7 * len(self.X))
        X_train = self.X.iloc[:train_idx]
        X_test = self.X.iloc[train_idx:]
        y_train = self.y.iloc[:train_idx]
        y_test = self.y.iloc[train_idx:]

        # Train global model (on all sectors)
        X_train_features = X_train.drop(columns=["sector"])
        X_test_features = X_test.drop(columns=["sector"])

        # Unpack tuple return: (model, results_dict)
        global_model, global_results = train_xgboost_regressor(
            X_train_features, y_train, random_state=42
        )

        # Train sector-specific models
        sector_result = train_high_error_sector_models(
            X_train, y_train, sectors=["Real Estate", "Materials", "Energy"], random_state=42
        )
        sector_models = sector_result["models"]

        # Compare performance
        comparison = compare_sector_vs_global_performance(
            global_model=global_model,
            sector_models=sector_models,
            X_test=X_test,
            y_test=y_test,
            sectors_test=X_test["sector"].values,
        )

        # Should return comparison dict with metrics
        self.assertIn("sector_metrics", comparison)
        self.assertIn("global_metrics", comparison)
        self.assertIn("improvement", comparison)

        # Should have metrics for each high-error sector
        sector_metrics = comparison["sector_metrics"]
        self.assertIn("Real Estate", sector_metrics)
        self.assertIn("Materials", sector_metrics)
        self.assertIn("Energy", sector_metrics)

        # Each sector should have MAE and MAPE
        for sector in ["Real Estate", "Materials", "Energy"]:
            metrics = sector_metrics[sector]
            self.assertIn("mae", metrics)
            self.assertIn("mape", metrics)

    def test_graceful_fallback_insufficient_samples(self):
        """Test graceful handling of sectors with insufficient samples."""
        from finance_ml.ml_workflow.regression.sector_models import train_high_error_sector_models

        # Create small dataset with few samples per sector
        n_small = 50
        X_small = self.X.iloc[:n_small].copy()
        y_small = self.y.iloc[:n_small]

        # Ensure one sector has very few samples
        X_small.loc[:5, "sector"] = "Real Estate"  # Only 6 samples

        # Train with minimum sample requirement
        result = train_high_error_sector_models(
            X_small,
            y_small,
            sectors=["Real Estate", "Materials", "Energy"],
            min_samples=10,  # Real Estate has fewer than this
            random_state=42,
        )

        # Should still return models dict
        self.assertIn("models", result)
        models = result["models"]

        # Real Estate should either:
        # 1. Not be in models dict (skipped due to insufficient samples)
        # 2. Or have a fallback global model
        if "Real Estate" in models:
            # If present, should have a valid model
            self.assertIsNotNone(models["Real Estate"])
            self.assertTrue(hasattr(models["Real Estate"], "predict"))

        # Other sectors with sufficient samples should have models
        # (assuming they have enough samples in the small dataset)

    def test_export_sector_performance_report(self):
        """Test export of sector performance comparison report to CSV."""
        from finance_ml.ml_workflow.regression.sector_models import (
            train_high_error_sector_models,
            compare_sector_vs_global_performance,
            export_sector_performance_report,
        )
        from finance_ml.ml_workflow.regression.models import train_xgboost_regressor

        # Split data
        train_idx = int(0.7 * len(self.X))
        X_train = self.X.iloc[:train_idx]
        X_test = self.X.iloc[train_idx:]
        y_train = self.y.iloc[:train_idx]
        y_test = self.y.iloc[train_idx:]

        # Train models
        X_train_features = X_train.drop(columns=["sector"])
        X_test_features = X_test.drop(columns=["sector"])

        # Unpack tuple return: (model, results_dict)
        global_model, global_results = train_xgboost_regressor(
            X_train_features, y_train, random_state=42
        )

        sector_result = train_high_error_sector_models(
            X_train, y_train, sectors=["Real Estate", "Materials", "Energy"], random_state=42
        )
        sector_models = sector_result["models"]

        # Compare performance
        comparison = compare_sector_vs_global_performance(
            global_model=global_model,
            sector_models=sector_models,
            X_test=X_test,
            y_test=y_test,
            sectors_test=X_test["sector"].values,
        )

        # Export to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            export_sector_performance_report(comparison=comparison, output_dir=output_dir)

            # Verify file was created
            report_file = output_dir / "sector_model_performance_comparison.csv"
            self.assertTrue(
                report_file.exists(), "sector_model_performance_comparison.csv not created"
            )

            # Verify CSV is readable
            report_df = pd.read_csv(report_file)

            # Should have columns: sector, mae_global, mae_sector, mape_global, mape_sector, improvement
            expected_cols = ["sector", "mae_global", "mae_sector", "mape_global", "mape_sector"]
            for col in expected_cols:
                self.assertIn(col, report_df.columns)

            # Should have rows for each high-error sector
            sectors_in_report = report_df["sector"].tolist()
            for sector in ["Real Estate", "Materials", "Energy"]:
                if sector in sector_models:  # Only check if model was trained
                    self.assertIn(sector, sectors_in_report)

    def test_error_reduction_targets_real_estate(self):
        """Test that Real Estate error is reduced (target: <200% MAPE)."""
        from finance_ml.ml_workflow.regression.sector_models import train_high_error_sector_models
        from sklearn.metrics import mean_absolute_percentage_error

        # Split data
        train_idx = int(0.7 * len(self.X))
        X_train = self.X.iloc[:train_idx]
        X_test = self.X.iloc[train_idx:]
        y_train = self.y.iloc[:train_idx]
        y_test = self.y.iloc[train_idx:]

        # Train sector-specific models
        result = train_high_error_sector_models(
            X_train, y_train, sectors=["Real Estate", "Materials", "Energy"], random_state=42
        )
        models = result["models"]

        # Predict for Real Estate sector
        if "Real Estate" in models:
            re_mask = X_test["sector"] == "Real Estate"
            if re_mask.sum() > 0:
                X_re = X_test[re_mask].copy()
                y_re = y_test[re_mask]

                # Add sector-specific features (must match training)
                X_re_enhanced = add_sector_specific_features(X_re, sector="Real Estate")
                X_re_features = X_re_enhanced.drop(columns=["sector"])

                y_pred_re = models["Real Estate"].predict(X_re_features)

                # Calculate MAPE
                mape_re = mean_absolute_percentage_error(y_re, y_pred_re) * 100

                # Target: <200% (generous target for initial implementation)
                # Note: This may fail initially, showing baseline performance
                self.assertLess(
                    mape_re, 200, f"Real Estate MAPE {mape_re:.1f}% exceeds target (<200%)"
                )

    def test_error_reduction_targets_materials_energy(self):
        """Test that Materials/Energy errors are reduced (target: <150% MAPE)."""
        from finance_ml.ml_workflow.regression.sector_models import train_high_error_sector_models
        from sklearn.metrics import mean_absolute_percentage_error

        # Split data
        train_idx = int(0.7 * len(self.X))
        X_train = self.X.iloc[:train_idx]
        X_test = self.X.iloc[train_idx:]
        y_train = self.y.iloc[:train_idx]
        y_test = self.y.iloc[train_idx:]

        # Train sector-specific models
        result = train_high_error_sector_models(
            X_train, y_train, sectors=["Real Estate", "Materials", "Energy"], random_state=42
        )
        models = result["models"]

        # Test Materials
        if "Materials" in models:
            mat_mask = X_test["sector"] == "Materials"
            if mat_mask.sum() > 0:
                X_mat = X_test[mat_mask].copy()
                y_mat = y_test[mat_mask]

                # Add sector-specific features (must match training)
                X_mat_enhanced = add_sector_specific_features(X_mat, sector="Materials")
                X_mat_features = X_mat_enhanced.drop(columns=["sector"])

                y_pred_mat = models["Materials"].predict(X_mat_features)
                mape_mat = mean_absolute_percentage_error(y_mat, y_pred_mat) * 100

                self.assertLess(
                    mape_mat, 150, f"Materials MAPE {mape_mat:.1f}% exceeds target (<150%)"
                )

        # Test Energy
        if "Energy" in models:
            energy_mask = X_test["sector"] == "Energy"
            if energy_mask.sum() > 0:
                X_energy = X_test[energy_mask].copy()
                y_energy = y_test[energy_mask]

                # Add sector-specific features (must match training)
                X_energy_enhanced = add_sector_specific_features(X_energy, sector="Energy")
                X_energy_features = X_energy_enhanced.drop(columns=["sector"])

                y_pred_energy = models["Energy"].predict(X_energy_features)
                mape_energy = mean_absolute_percentage_error(y_energy, y_pred_energy) * 100

                self.assertLess(
                    mape_energy, 150, f"Energy MAPE {mape_energy:.1f}% exceeds target (<150%)"
                )


if __name__ == "__main__":
    unittest.main()
