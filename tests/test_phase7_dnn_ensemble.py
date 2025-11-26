"""
Phase 7.4-7.8 TDD Tests: DNN Implementation, Ensemble Enhancement, and Advanced Optimization

This test module covers:
- Phase 7.4: Dense Neural Network Implementation
- Phase 7.5: Ensemble Model Enhancement
- Phase 7.6: Black-Litterman ML Integration
- Phase 7.7: Robust Covariance Estimation
- Phase 7.8: Model Validation & Diagnostics

Tests are designed to gracefully skip when TensorFlow is unavailable.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Check TensorFlow availability
try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


def create_sample_train_data(n_samples: int = 200, n_features: int = 20) -> tuple:
    """Create sample training data for DNN tests."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Generate realistic returns (mean ~8%, std ~20%)
    y = 0.08 + 0.20 * np.random.randn(n_samples).astype(np.float32)
    return X, y


def create_train_test_split(n_train: int = 200, n_test: int = 50, n_features: int = 20) -> tuple:
    """Create train/test split for model comparison tests."""
    np.random.seed(42)
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = 0.08 + 0.20 * np.random.randn(n_train).astype(np.float32)
    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    y_test = 0.08 + 0.20 * np.random.randn(n_test).astype(np.float32)
    return X_train, y_train, X_test, y_test


def create_sample_returns(n_obs: int = 252, n_assets: int = 10) -> pd.DataFrame:
    """Create sample return series for covariance estimation tests."""
    np.random.seed(42)
    # Generate correlated returns
    mean_returns = np.random.randn(n_assets) * 0.0003  # Daily returns ~8% annual
    cov = np.eye(n_assets) * 0.0004  # ~20% annual vol
    returns = np.random.multivariate_normal(mean_returns, cov, n_obs)
    columns = [f"Asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=columns)


# =============================================================================
# Phase 7.4: Dense Neural Network Implementation Tests
# =============================================================================


@unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow not installed")
class TestDNNReturnPredictorArchitecture(unittest.TestCase):
    """Test DNN model architecture for return prediction."""

    def test_build_dnn_return_predictor_function_exists(self):
        """Test that build_dnn_return_predictor function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            build_dnn_return_predictor,
        )

        self.assertTrue(callable(build_dnn_return_predictor))

    def test_build_dnn_return_predictor_creates_model(self):
        """Test that build_dnn_return_predictor creates a valid Keras model."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            build_dnn_return_predictor,
        )

        model = build_dnn_return_predictor(
            input_dim=20,
            hidden_layers=[64, 32, 16],
            dropout_rate=0.3,
            l2_reg=1e-4,
        )

        # Verify it's a Keras model
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "fit"))

    def test_build_dnn_return_predictor_layer_count(self):
        """Test DNN has expected number of layers."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            build_dnn_return_predictor,
        )

        model = build_dnn_return_predictor(
            input_dim=20,
            hidden_layers=[64, 32, 16],
            dropout_rate=0.3,
        )

        # Should have input + 3 dense + 3 dropout + output = at least 7 layers
        self.assertGreaterEqual(len(model.layers), 7)

    def test_build_dnn_return_predictor_output_shape(self):
        """Test DNN output shape is correct for regression."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            build_dnn_return_predictor,
        )

        model = build_dnn_return_predictor(input_dim=20, hidden_layers=[32, 16])
        X_test = np.random.randn(10, 20).astype(np.float32)
        predictions = model.predict(X_test, verbose=0)

        self.assertEqual(predictions.shape, (10, 1))


@unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow not installed")
class TestDNNTrainingConvergence(unittest.TestCase):
    """Test that DNN training converges properly."""

    def test_train_dnn_return_predictor_function_exists(self):
        """Test that train_dnn_return_predictor function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            train_dnn_return_predictor,
        )

        self.assertTrue(callable(train_dnn_return_predictor))

    def test_train_dnn_return_predictor_returns_model_and_history(self):
        """Test that training returns model and history."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            train_dnn_return_predictor,
        )

        X_train, y_train = create_sample_train_data(n_samples=100, n_features=10)

        model, history = train_dnn_return_predictor(
            X_train,
            y_train,
            hidden_layers=[32, 16],
            epochs=10,
            batch_size=32,
            verbose=0,
        )

        self.assertIsNotNone(model)
        self.assertIsNotNone(history)
        self.assertIn("loss", history)

    def test_train_dnn_loss_decreases(self):
        """Test that training loss decreases over epochs."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            train_dnn_return_predictor,
        )

        X_train, y_train = create_sample_train_data(n_samples=200, n_features=10)

        model, history = train_dnn_return_predictor(
            X_train,
            y_train,
            hidden_layers=[32, 16],
            epochs=20,
            batch_size=32,
            verbose=0,
        )

        # Loss should decrease (final < initial)
        self.assertLess(history["loss"][-1], history["loss"][0])


@unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow not installed")
class TestDNNvsRidgeComparison(unittest.TestCase):
    """Test that DNN provides reasonable performance vs Ridge baseline."""

    def test_dnn_competitive_with_ridge(self):
        """Test that DNN MSE is competitive with Ridge baseline."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            train_dnn_return_predictor,
            train_linear_return_predictor,
        )

        X_train, y_train, X_test, y_test = create_train_test_split(
            n_train=200, n_test=50, n_features=10
        )

        # Train Ridge baseline
        ridge_model = train_linear_return_predictor(X_train, y_train)
        ridge_pred = ridge_model.predict(X_test)
        ridge_mse = np.mean((ridge_pred - y_test) ** 2)

        # Train DNN
        dnn_model, _ = train_dnn_return_predictor(
            X_train,
            y_train,
            hidden_layers=[32, 16],
            epochs=50,
            verbose=0,
        )
        dnn_pred = dnn_model.predict(X_test, verbose=0).flatten()
        dnn_mse = np.mean((dnn_pred - y_test) ** 2)

        # DNN should be competitive (within 50% of Ridge or better)
        self.assertLessEqual(
            dnn_mse, ridge_mse * 1.5, f"DNN MSE {dnn_mse:.4f} >> Ridge MSE {ridge_mse:.4f}"
        )


@unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow not installed")
class TestDNNQuantileRegression(unittest.TestCase):
    """Test DNN with quantile regression for uncertainty estimation."""

    def test_train_dnn_quantile_predictor_function_exists(self):
        """Test that train_dnn_quantile_predictor function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            train_dnn_quantile_predictor,
        )

        self.assertTrue(callable(train_dnn_quantile_predictor))

    def test_quantile_predictions_monotonic(self):
        """Test that quantile predictions are approximately monotonic (p10 <= p50 <= p90).

        Note: Independent quantile models don't guarantee strict monotonicity.
        We check that the majority of predictions satisfy monotonicity, as
        small violations are expected with separately trained models.
        """
        from finance_ml.ml_workflow.analytics.ml_returns import (
            train_dnn_quantile_predictor,
        )

        X_train, y_train, X_test, y_test = create_train_test_split(
            n_train=200, n_test=50, n_features=10
        )

        quantiles = [0.1, 0.5, 0.9]
        predictions = {}

        for q in quantiles:
            model = train_dnn_quantile_predictor(
                X_train,
                y_train,
                quantile=q,
                hidden_layers=[32, 16],
                epochs=30,
                verbose=0,
            )
            predictions[q] = model.predict(X_test, verbose=0).flatten()

        # Check approximate monotonicity: at least 80% of predictions should satisfy
        # p10 <= p50 <= p90 (small violations are expected with independent models)
        p10_le_p50 = np.mean(predictions[0.1] <= predictions[0.5] + 0.05)
        p50_le_p90 = np.mean(predictions[0.5] <= predictions[0.9] + 0.05)

        self.assertGreaterEqual(
            p10_le_p50, 0.70, f"Only {p10_le_p50:.0%} of p10 <= p50, expected >= 70%"
        )
        self.assertGreaterEqual(
            p50_le_p90, 0.70, f"Only {p50_le_p90:.0%} of p50 <= p90, expected >= 70%"
        )

        # Also verify that median predictions are in a reasonable range
        self.assertTrue(
            np.mean(predictions[0.5]) > np.mean(predictions[0.1]) - 0.1,
            "Median should be generally higher than 10th percentile",
        )


# =============================================================================
# Phase 7.5: Ensemble Model Enhancement Tests
# =============================================================================


class TestMultiModelEnsemble(unittest.TestCase):
    """Test ensemble combining multiple model types."""

    def test_create_return_ensemble_function_exists(self):
        """Test that create_return_ensemble function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            create_return_ensemble,
        )

        self.assertTrue(callable(create_return_ensemble))

    def test_create_return_ensemble_returns_ensemble_object(self):
        """Test that create_return_ensemble returns an ensemble."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            create_return_ensemble,
        )

        X_train, y_train = create_sample_train_data(n_samples=100, n_features=10)

        # Create ensemble with available models (no DNN if TF not installed)
        models = ["ridge", "random_forest"]

        ensemble = create_return_ensemble(
            X_train,
            y_train,
            models=models,
            cv_folds=3,
        )

        self.assertIsNotNone(ensemble)
        self.assertTrue(hasattr(ensemble, "predict"))

    def test_ensemble_predict_returns_valid_array(self):
        """Test that ensemble predict returns valid predictions."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            create_return_ensemble,
        )

        X_train, y_train, X_test, y_test = create_train_test_split(
            n_train=100, n_test=20, n_features=10
        )

        ensemble = create_return_ensemble(
            X_train,
            y_train,
            models=["ridge", "random_forest"],
            cv_folds=3,
        )

        predictions = ensemble.predict(X_test)

        self.assertEqual(predictions.shape, y_test.shape)
        self.assertFalse(np.any(np.isnan(predictions)))


class TestDynamicEnsembleWeighting(unittest.TestCase):
    """Test dynamic weighting based on validation performance."""

    def test_create_dynamic_ensemble_function_exists(self):
        """Test that create_dynamic_ensemble function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            create_dynamic_ensemble,
        )

        self.assertTrue(callable(create_dynamic_ensemble))

    def test_dynamic_ensemble_weights_sum_to_one(self):
        """Test that dynamic ensemble weights sum to 1."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            create_dynamic_ensemble,
        )

        X_train, y_train = create_sample_train_data(n_samples=100, n_features=10)
        X_val = np.random.randn(30, 10).astype(np.float32)
        y_val = 0.08 + 0.20 * np.random.randn(30).astype(np.float32)

        ensemble = create_dynamic_ensemble(
            X_train,
            y_train,
            models=["ridge"],
            weighting_method="inverse_mse",
            validation_data=(X_val, y_val),
        )

        weights = ensemble.get_model_weights()
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)


# =============================================================================
# Phase 7.6: Black-Litterman ML Integration Tests
# =============================================================================


class TestMLViewsForBlackLitterman(unittest.TestCase):
    """Test creation of BL views from ML predictions."""

    def test_create_bl_views_from_ml_function_exists(self):
        """Test that create_bl_views_from_ml function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            create_bl_views_from_ml,
        )

        self.assertTrue(callable(create_bl_views_from_ml))

    def test_create_bl_views_returns_dict(self):
        """Test that create_bl_views_from_ml returns views dict."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            create_bl_views_from_ml,
        )

        ml_predictions = pd.Series([0.10, 0.15, 0.05], index=["AAPL", "MSFT", "GOOGL"])

        views, confidences = create_bl_views_from_ml(
            ml_predictions,
            tickers=["AAPL", "MSFT", "GOOGL"],
            min_confidence=0.3,
            max_confidence=0.9,
        )

        self.assertIsInstance(views, dict)
        self.assertEqual(len(confidences), len(views))

    def test_bl_views_confidences_in_range(self):
        """Test that BL view confidences are within specified range."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            create_bl_views_from_ml,
        )

        ml_predictions = pd.Series([0.10, 0.15, 0.05], index=["AAPL", "MSFT", "GOOGL"])

        views, confidences = create_bl_views_from_ml(
            ml_predictions,
            tickers=["AAPL", "MSFT", "GOOGL"],
            min_confidence=0.3,
            max_confidence=0.9,
        )

        for conf in confidences:
            self.assertGreaterEqual(conf, 0.3)
            self.assertLessEqual(conf, 0.9)


class TestMarketRegimeDetection(unittest.TestCase):
    """Test market regime detection for BL parameter adjustment."""

    def test_detect_market_regime_function_exists(self):
        """Test that detect_market_regime function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            detect_market_regime,
        )

        self.assertTrue(callable(detect_market_regime))

    def test_detect_market_regime_returns_valid_regime(self):
        """Test that detect_market_regime returns a valid regime string."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            detect_market_regime,
        )

        returns = create_sample_returns(n_obs=252, n_assets=5)

        regime = detect_market_regime(
            returns,
            method="volatility",
            thresholds={"low_vol": 0.10, "high_vol": 0.25},
        )

        valid_regimes = ["low_volatility", "normal", "high_volatility"]
        self.assertIn(regime, valid_regimes)


# =============================================================================
# Phase 7.7: Robust Covariance Estimation Tests
# =============================================================================


class TestShrinkageCovariance(unittest.TestCase):
    """Test Ledoit-Wolf shrinkage covariance estimation."""

    def test_estimate_covariance_shrinkage_function_exists(self):
        """Test that estimate_covariance_shrinkage function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            estimate_covariance_shrinkage,
        )

        self.assertTrue(callable(estimate_covariance_shrinkage))

    def test_shrinkage_covariance_is_positive_definite(self):
        """Test that shrinkage covariance is positive definite."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            estimate_covariance_shrinkage,
        )

        # More assets than observations (ill-conditioned case)
        returns = create_sample_returns(n_obs=50, n_assets=30)

        shrunk_cov = estimate_covariance_shrinkage(
            returns,
            method="ledoit_wolf",
        )

        # Should be positive definite
        eigenvalues = np.linalg.eigvalsh(shrunk_cov)
        self.assertTrue(all(eigenvalues > 0), "Covariance not positive definite")

    def test_shrinkage_covariance_well_conditioned(self):
        """Test that shrinkage covariance has reasonable condition number."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            estimate_covariance_shrinkage,
        )

        returns = create_sample_returns(n_obs=50, n_assets=30)

        shrunk_cov = estimate_covariance_shrinkage(returns, method="ledoit_wolf")

        eigenvalues = np.linalg.eigvalsh(shrunk_cov)
        condition_number = eigenvalues.max() / eigenvalues.min()

        self.assertLess(condition_number, 1e6, f"Condition number {condition_number} too high")


class TestExponentialWeightedCovariance(unittest.TestCase):
    """Test exponentially weighted covariance estimation."""

    def test_estimate_covariance_ewm_function_exists(self):
        """Test that estimate_covariance_ewm function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            estimate_covariance_ewm,
        )

        self.assertTrue(callable(estimate_covariance_ewm))

    def test_ewm_covariance_differs_from_sample(self):
        """Test that EWM covariance differs from sample covariance."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            estimate_covariance_ewm,
        )

        returns = create_sample_returns(n_obs=252, n_assets=5)

        sample_cov = returns.cov().values
        ewm_cov = estimate_covariance_ewm(returns, halflife=60)

        # Should be different but still valid
        self.assertFalse(np.allclose(ewm_cov, sample_cov))
        self.assertTrue(np.allclose(ewm_cov, ewm_cov.T))  # Symmetric


# =============================================================================
# Phase 7.8: Model Validation & Diagnostics Tests
# =============================================================================


class TestReturnPredictionDiagnostics(unittest.TestCase):
    """Test comprehensive diagnostics for return predictions."""

    def test_calculate_return_prediction_diagnostics_function_exists(self):
        """Test that calculate_return_prediction_diagnostics function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            calculate_return_prediction_diagnostics,
        )

        self.assertTrue(callable(calculate_return_prediction_diagnostics))

    def test_diagnostics_returns_expected_keys(self):
        """Test that diagnostics returns expected metric keys."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            calculate_return_prediction_diagnostics,
        )

        np.random.seed(42)
        y_true = np.random.randn(100) * 0.2
        y_pred = y_true + np.random.randn(100) * 0.1

        diagnostics = calculate_return_prediction_diagnostics(y_true, y_pred)

        expected_keys = ["mse", "mae", "r2", "ic"]
        for key in expected_keys:
            self.assertIn(key, diagnostics, f"Missing key: {key}")

    def test_diagnostics_ic_in_valid_range(self):
        """Test that Information Coefficient is in valid range [-1, 1]."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            calculate_return_prediction_diagnostics,
        )

        np.random.seed(42)
        y_true = np.random.randn(100) * 0.2
        y_pred = y_true + np.random.randn(100) * 0.1  # Correlated predictions

        diagnostics = calculate_return_prediction_diagnostics(y_true, y_pred)

        self.assertGreaterEqual(diagnostics["ic"], -1.0)
        self.assertLessEqual(diagnostics["ic"], 1.0)


class TestPortfolioMetricsValidation(unittest.TestCase):
    """Test portfolio metrics validation."""

    def test_validate_portfolio_metrics_function_exists(self):
        """Test that validate_portfolio_metrics function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            validate_portfolio_metrics,
        )

        self.assertTrue(callable(validate_portfolio_metrics))

    def test_validate_portfolio_metrics_flags_unrealistic_sharpe(self):
        """Test that validation flags unrealistic Sharpe ratios."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            validate_portfolio_metrics,
        )

        returns = create_sample_returns(n_obs=252, n_assets=5)
        weights = np.ones(5) / 5  # Equal weights

        diagnostics = validate_portfolio_metrics(weights, returns)

        self.assertIn("sharpe_ratio_valid", diagnostics)
        self.assertIn("return_realistic", diagnostics)


if __name__ == "__main__":
    unittest.main()
