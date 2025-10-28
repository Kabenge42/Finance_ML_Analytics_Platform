"""
Test suite for Phase 9.5: Advanced Regression Models with Classification Features

This module tests the implementation of sector-optimized regression models enhanced
with classification meta-features, including diverse model architectures, hyperparameter
optimization, ensemble methods, and quantile regression.

Test-Driven Development (TDD) approach:
1. Write failing tests first
2. Implement minimal code to pass
3. Refactor for quality

Target: ≥80% coverage for finance_ml/advanced_models.py
"""

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


def generate_synthetic_regression_data(
    n_samples: int = 500,
    n_features: int = 20,
    n_sectors: int = 3,
    include_classification_features: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic regression dataset for testing.

    Args:
        n_samples: Number of samples
        n_features: Number of numeric features
        n_sectors: Number of sectors
        include_classification_features: Include classification meta-features
        random_state: Random seed

    Returns:
        DataFrame with features, target, and optional classification features
    """
    np.random.seed(random_state)

    # Generate numeric features
    X = np.random.randn(n_samples, n_features)

    # Generate target with some signal
    weights = np.random.randn(n_features)
    y = X @ weights + np.random.randn(n_samples) * 0.1

    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["price_target"] = y
    df["last_price"] = y * 0.9 + np.random.randn(n_samples) * 0.05

    # Add sector
    sectors = [f"Sector_{i % n_sectors}" for i in range(n_samples)]
    df["sector"] = sectors

    # Add classification meta-features if requested
    if include_classification_features:
        # Simulate classification probabilities (sum to 1)
        probs = np.random.dirichlet(np.ones(3), n_samples)
        df["event_prob_neutral"] = probs[:, 0]
        df["event_prob_positive"] = probs[:, 1]
        df["event_prob_negative"] = probs[:, 2]
        df["event_class_predicted"] = np.argmax(probs, axis=1)
        df["event_confidence"] = np.max(probs, axis=1)

    return df


# ==============================================================================
# Test Class 1: Feature Integration
# ==============================================================================


class TestFeatureIntegration(unittest.TestCase):
    """Test classification feature integration into regression pipeline."""

    def setUp(self):
        """Set up test data."""
        self.df = generate_synthetic_regression_data(
            n_samples=200, include_classification_features=True
        )

    def test_prepare_regression_data(self):
        """Test prepare_regression_data with classification features."""
        from finance_ml.advanced_models import prepare_regression_data

        X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(
            self.df, target_col="price_target", test_size=0.2, random_state=42
        )

        # Check shapes
        self.assertEqual(len(X_train) + len(X_test), len(self.df))
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))

        # Check classification features are included
        self.assertIn("event_prob_neutral", X_train.columns)
        self.assertIn("event_prob_positive", X_train.columns)
        self.assertIn("event_confidence", X_train.columns)

        # Check feature_info
        self.assertIn("numeric_features", feature_info)
        self.assertIn("categorical_features", feature_info)
        self.assertIn("classification_features", feature_info)

    def test_create_classification_interactions(self):
        """Test creation of interaction features between classification and valuation."""
        from finance_ml.advanced_models import create_classification_interactions

        # Add some valuation features
        self.df["pe_ratio"] = np.random.uniform(5, 50, len(self.df))
        self.df["pb_ratio"] = np.random.uniform(0.5, 10, len(self.df))

        df_enhanced = create_classification_interactions(
            self.df,
            classification_cols=["event_prob_positive", "event_prob_negative"],
            valuation_cols=["pe_ratio", "pb_ratio"],
        )

        # Check new interaction features created
        self.assertIn("event_prob_positive_x_pe_ratio", df_enhanced.columns)
        self.assertIn("event_prob_negative_x_pb_ratio", df_enhanced.columns)

        # Check values are correct (product)
        expected = self.df["event_prob_positive"] * self.df["pe_ratio"]
        np.testing.assert_array_almost_equal(
            df_enhanced["event_prob_positive_x_pe_ratio"].values, expected.values
        )


# ==============================================================================
# Test Class 2: Linear Models
# ==============================================================================


class TestLinearModels(unittest.TestCase):
    """Test linear regression models."""

    def setUp(self):
        """Set up test data."""
        self.df = generate_synthetic_regression_data(n_samples=200)
        self.feature_cols = [c for c in self.df.columns if c.startswith("feature_")]
        self.X = self.df[self.feature_cols]
        self.y = self.df["price_target"]

    def test_train_ridge_regressor(self):
        """Test Ridge regression with L2 regularization."""
        from finance_ml.advanced_models import train_ridge_regressor

        model, results = train_ridge_regressor(self.X, self.y, alpha=1.0, cv=3, random_state=42)

        # Check model trained
        self.assertIsNotNone(model)

        # Check results
        self.assertIn("train_score", results)
        self.assertIn("cv_scores", results)
        self.assertIn("best_alpha", results)

        # Check can predict
        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))

    def test_train_lasso_regressor(self):
        """Test Lasso regression with L1 regularization."""
        from finance_ml.advanced_models import train_lasso_regressor

        model, results = train_lasso_regressor(self.X, self.y, alpha=0.1, cv=3, random_state=42)

        self.assertIsNotNone(model)
        self.assertIn("train_score", results)
        self.assertIn("n_nonzero_coefs", results)  # Sparse solution tracking

        # Lasso tracks coefficient sparsity
        n_nonzero = np.sum(model.coef_ != 0)
        self.assertGreaterEqual(n_nonzero, 0)  # At least 0 (can be all sparse)
        self.assertLessEqual(n_nonzero, len(self.feature_cols))  # At most all features

        # Check that results properly report sparsity
        self.assertEqual(results["n_nonzero_coefs"], n_nonzero)
        self.assertEqual(results["n_zero_coefs"], len(self.feature_cols) - n_nonzero)

    def test_train_elastic_net_regressor(self):
        """Test Elastic Net combining L1 and L2."""
        from finance_ml.advanced_models import train_elastic_net_regressor

        model, results = train_elastic_net_regressor(
            self.X, self.y, alpha=0.1, l1_ratio=0.5, cv=3, random_state=42
        )

        self.assertIsNotNone(model)
        self.assertIn("best_alpha", results)
        self.assertIn("best_l1_ratio", results)

    def test_train_bayesian_ridge_regressor(self):
        """Test Bayesian Ridge for uncertainty estimation."""
        from finance_ml.advanced_models import train_bayesian_ridge_regressor

        model, results = train_bayesian_ridge_regressor(self.X, self.y, n_iter=300, random_state=42)

        self.assertIsNotNone(model)

        # Bayesian Ridge provides uncertainty estimates
        y_pred, y_std = model.predict(self.X, return_std=True)
        self.assertEqual(len(y_pred), len(self.y))
        self.assertEqual(len(y_std), len(self.y))
        self.assertTrue(np.all(y_std > 0))  # Uncertainty should be positive

    def test_train_polynomial_regressor(self):
        """Test polynomial regression with degree 2."""
        from finance_ml.advanced_models import train_polynomial_regressor

        model, results = train_polynomial_regressor(
            self.X, self.y, degree=2, alpha=1.0, random_state=42
        )

        self.assertIsNotNone(model)
        self.assertIn("degree", results)
        self.assertEqual(results["degree"], 2)

        # Check can predict
        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))


# ==============================================================================
# Test Class 3: Gradient Boosting Models
# ==============================================================================


class TestGradientBoostingModels(unittest.TestCase):
    """Test gradient boosting regression models."""

    def setUp(self):
        """Set up test data."""
        self.df = generate_synthetic_regression_data(n_samples=200)
        self.feature_cols = [c for c in self.df.columns if c.startswith("feature_")]
        self.X = self.df[self.feature_cols]
        self.y = self.df["price_target"]

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("xgboost") is not None, "XGBoost not installed"
    )
    def test_train_xgboost_regressor(self):
        """Test XGBoost regressor."""
        from finance_ml.advanced_models import train_xgboost_regressor

        model, results = train_xgboost_regressor(
            self.X, self.y, params={"max_depth": 3, "n_estimators": 50}, random_state=42
        )

        self.assertIsNotNone(model)
        self.assertIn("train_score", results)

        # Check predictions
        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("lightgbm") is not None, "LightGBM not installed"
    )
    def test_train_lightgbm_regressor(self):
        """Test LightGBM regressor."""
        from finance_ml.advanced_models import train_lightgbm_regressor

        model, results = train_lightgbm_regressor(
            self.X, self.y, params={"num_leaves": 31, "n_estimators": 50}, random_state=42
        )

        self.assertIsNotNone(model)
        self.assertIn("train_score", results)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("catboost") is not None, "CatBoost not installed"
    )
    def test_train_catboost_regressor(self):
        """Test CatBoost regressor."""
        from finance_ml.advanced_models import train_catboost_regressor

        model, results = train_catboost_regressor(
            self.X, self.y, params={"depth": 3, "iterations": 50}, random_state=42
        )

        self.assertIsNotNone(model)
        self.assertIn("train_score", results)

    def test_train_histgb_regressor(self):
        """Test HistGradientBoosting regressor (sklearn)."""
        from finance_ml.advanced_models import train_histgb_regressor

        model, results = train_histgb_regressor(
            self.X, self.y, max_iter=50, max_depth=3, random_state=42
        )

        self.assertIsNotNone(model)
        self.assertIn("train_score", results)

        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))


# ==============================================================================
# Test Class 4: Tree and Neural Models
# ==============================================================================


class TestTreeAndNeuralModels(unittest.TestCase):
    """Test tree ensemble and neural network models."""

    def setUp(self):
        """Set up test data."""
        self.df = generate_synthetic_regression_data(n_samples=200)
        self.feature_cols = [c for c in self.df.columns if c.startswith("feature_")]
        self.X = self.df[self.feature_cols]
        self.y = self.df["price_target"]

    def test_train_random_forest_regressor(self):
        """Test Random Forest regressor."""
        from finance_ml.advanced_models import train_random_forest_regressor

        model, results = train_random_forest_regressor(
            self.X, self.y, n_estimators=50, max_depth=5, random_state=42
        )

        self.assertIsNotNone(model)
        self.assertIn("train_score", results)
        self.assertIn("feature_importance", results)

        # Check feature importance
        importance = results["feature_importance"]
        self.assertEqual(len(importance), len(self.feature_cols))
        self.assertTrue(np.all(importance >= 0))

    def test_train_extra_trees_regressor(self):
        """Test Extra Trees regressor."""
        from finance_ml.advanced_models import train_extra_trees_regressor

        model, results = train_extra_trees_regressor(
            self.X, self.y, n_estimators=50, max_depth=5, random_state=42
        )

        self.assertIsNotNone(model)
        self.assertIn("train_score", results)

        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("tensorflow") is not None, "TensorFlow not installed"
    )
    def test_train_neural_network_regressor(self):
        """Test neural network regressor with Keras."""
        from finance_ml.advanced_models import train_neural_network_regressor

        model, results = train_neural_network_regressor(
            self.X,
            self.y,
            hidden_layers=[64, 32],
            dropout_rate=0.3,
            epochs=10,
            batch_size=32,
            random_state=42,
        )

        self.assertIsNotNone(model)
        self.assertIn("train_loss", results)
        self.assertIn("val_loss", results)

        # Check predictions
        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))


# ==============================================================================
# Test Class 5: Ensemble Methods
# ==============================================================================


class TestEnsembleMethods(unittest.TestCase):
    """Test advanced ensemble regression methods."""

    def setUp(self):
        """Set up test data."""
        self.df = generate_synthetic_regression_data(n_samples=200)
        self.feature_cols = [c for c in self.df.columns if c.startswith("feature_")]
        self.X = self.df[self.feature_cols]
        self.y = self.df["price_target"]

    def test_train_voting_regressor(self):
        """Test voting ensemble regressor."""
        from finance_ml.advanced_models import train_voting_regressor

        model, results = train_voting_regressor(
            self.X, self.y, weights=None, random_state=42  # Equal weights
        )

        self.assertIsNotNone(model)
        self.assertIn("train_score", results)
        self.assertIn("base_models", results)

        # Check multiple base models
        self.assertGreater(len(results["base_models"]), 1)

        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))

    def test_train_stacking_regressor(self):
        """Test stacking ensemble regressor."""
        from finance_ml.advanced_models import train_stacking_regressor

        model, results = train_stacking_regressor(self.X, self.y, cv=3, random_state=42)

        self.assertIsNotNone(model)
        self.assertIn("train_score", results)
        self.assertIn("cv_score", results)
        self.assertIn("base_models", results)
        self.assertIn("meta_model", results)

        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))

    def test_train_quantile_regressor(self):
        """Test quantile regression for uncertainty."""
        from finance_ml.advanced_models import train_quantile_regressor

        quantiles = [0.1, 0.5, 0.9]
        models, results = train_quantile_regressor(
            self.X, self.y, quantiles=quantiles, random_state=42
        )

        self.assertEqual(len(models), len(quantiles))
        self.assertIn("quantiles", results)

        # Check predictions
        predictions = {}
        for q, model in zip(quantiles, models):
            y_pred = model.predict(self.X)
            predictions[q] = y_pred

        # Lower quantile should be <= median <= upper quantile (with tolerance for numerical issues)
        # Check that violations are minimal (< 10%)
        violations_low = np.sum(predictions[0.1] > predictions[0.5]) / len(predictions[0.1])
        violations_high = np.sum(predictions[0.5] > predictions[0.9]) / len(predictions[0.5])
        self.assertLess(violations_low, 0.1, "Too many violations: lower quantile > median")
        self.assertLess(violations_high, 0.1, "Too many violations: median > upper quantile")


# ==============================================================================
# Test Class 6: Hyperparameter Optimization
# ==============================================================================


class TestHyperparameterOptimization(unittest.TestCase):
    """Test Optuna-based hyperparameter optimization."""

    def setUp(self):
        """Set up test data."""
        self.df = generate_synthetic_regression_data(n_samples=200)
        self.feature_cols = [c for c in self.df.columns if c.startswith("feature_")]
        self.X = self.df[self.feature_cols]
        self.y = self.df["price_target"]

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("optuna") is not None, "Optuna not installed"
    )
    def test_optimize_hyperparameters_optuna(self):
        """Test Optuna hyperparameter optimization."""
        from finance_ml.advanced_models import optimize_hyperparameters_optuna

        best_params, study = optimize_hyperparameters_optuna(
            self.X,
            self.y,
            model_type="random_forest",
            n_trials=5,  # Small for testing
            cv=3,
            random_state=42,
        )

        self.assertIsNotNone(best_params)
        self.assertIsInstance(best_params, dict)
        self.assertIn("n_estimators", best_params)

        # Check study object
        self.assertIsNotNone(study)
        self.assertEqual(len(study.trials), 5)


# ==============================================================================
# Test Class 7: Model Comparison
# ==============================================================================


class TestModelComparison(unittest.TestCase):
    """Test model comparison and evaluation framework."""

    def setUp(self):
        """Set up test data."""
        self.df = generate_synthetic_regression_data(n_samples=200)
        self.feature_cols = [c for c in self.df.columns if c.startswith("feature_")]
        self.X = self.df[self.feature_cols]
        self.y = self.df["price_target"]

    def test_compare_regressors(self):
        """Test comparison of multiple regression models."""
        from finance_ml.advanced_models import compare_regressors

        results = compare_regressors(self.X, self.y, test_size=0.2, cv=3, random_state=42)

        self.assertIsInstance(results, dict)

        # Check multiple models compared
        self.assertGreater(len(results), 3)

        # Check each model has metrics
        for model_name, metrics in results.items():
            self.assertIn("mae", metrics)
            self.assertIn("rmse", metrics)
            self.assertIn("r2", metrics)
            self.assertIn("train_time", metrics)


# ==============================================================================
# Test Class 8: Sector-Specific Models
# ==============================================================================


class TestSectorSpecificModels(unittest.TestCase):
    """Test sector-specific regression training."""

    def setUp(self):
        """Set up test data with sectors."""
        self.df = generate_synthetic_regression_data(n_samples=300, n_sectors=3)
        self.feature_cols = [c for c in self.df.columns if c.startswith("feature_")]

    def test_train_sector_specific_models(self):
        """Test training separate models per sector."""
        from finance_ml.advanced_models import train_sector_specific_models

        sector_models, results = train_sector_specific_models(
            self.df,
            feature_cols=self.feature_cols,
            target_col="price_target",
            sector_col="sector",
            model_type="random_forest",
            random_state=42,
        )

        # Check models for each sector
        sectors = self.df["sector"].unique()
        self.assertEqual(len(sector_models), len(sectors))

        for sector in sectors:
            self.assertIn(sector, sector_models)
            self.assertIsNotNone(sector_models[sector])

        # Check results
        self.assertIn("sector_metrics", results)
        for sector in sectors:
            self.assertIn(sector, results["sector_metrics"])


# ==============================================================================
# Test Class 9: Model Persistence
# ==============================================================================


class TestModelPersistence(unittest.TestCase):
    """Test model saving and loading."""

    def setUp(self):
        """Set up test data and temp directory."""
        self.df = generate_synthetic_regression_data(n_samples=100)
        self.feature_cols = [c for c in self.df.columns if c.startswith("feature_")]
        self.X = self.df[self.feature_cols]
        self.y = self.df["price_target"]
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_model(self):
        """Test saving and loading trained models."""
        from finance_ml.advanced_models import train_ridge_regressor, save_model, load_model

        # Train a model
        model, _ = train_ridge_regressor(self.X, self.y, random_state=42)

        # Save model
        model_path = Path(self.temp_dir) / "test_model.joblib"
        metadata = {"model_type": "ridge", "features": self.feature_cols, "target": "price_target"}
        save_model(model, model_path, metadata=metadata)

        # Check file exists
        self.assertTrue(model_path.exists())

        # Load model
        loaded_model, loaded_metadata = load_model(model_path)

        # Check loaded model works
        y_pred_original = model.predict(self.X)
        y_pred_loaded = loaded_model.predict(self.X)
        np.testing.assert_array_almost_equal(y_pred_original, y_pred_loaded)

        # Check metadata
        self.assertEqual(loaded_metadata["model_type"], "ridge")
        self.assertEqual(loaded_metadata["features"], self.feature_cols)


# ==============================================================================
# Test Class 10: Integration Test
# ==============================================================================


class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete Phase 9.5 workflow end-to-end."""

    def test_complete_regression_pipeline(self):
        """Test full pipeline from data prep to model comparison."""
        from finance_ml.advanced_models import (
            prepare_regression_data,
            create_classification_interactions,
            compare_regressors,
            train_stacking_regressor,
        )

        # Generate data with classification features
        df = generate_synthetic_regression_data(n_samples=300, include_classification_features=True)

        # Add valuation features
        df["pe_ratio"] = np.random.uniform(5, 50, len(df))
        df["pb_ratio"] = np.random.uniform(0.5, 10, len(df))

        # Step 1: Create interaction features
        df_enhanced = create_classification_interactions(
            df,
            classification_cols=["event_prob_positive", "event_prob_negative"],
            valuation_cols=["pe_ratio", "pb_ratio"],
        )

        self.assertGreater(len(df_enhanced.columns), len(df.columns))

        # Step 2: Prepare data
        X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(
            df_enhanced, target_col="price_target", test_size=0.2, random_state=42
        )

        # Step 3: Compare models (quick comparison)
        results = compare_regressors(
            X_train, y_train, test_size=0.2, cv=2, random_state=42  # Quick for testing
        )

        self.assertGreater(len(results), 2)

        # Step 4: Train best ensemble
        model, ensemble_results = train_stacking_regressor(X_train, y_train, cv=2, random_state=42)

        # Step 5: Make predictions
        y_pred = model.predict(X_test)

        self.assertEqual(len(y_pred), len(y_test))

        # Check reasonable predictions (correlation with actual)
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        self.assertGreater(correlation, 0.3)  # Some positive correlation


if __name__ == "__main__":
    unittest.main()
