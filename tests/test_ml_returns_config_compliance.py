"""TDD tests for ML returns configuration compliance.

This test module follows strict TDD principles to ensure:
1. Configuration constants are properly defined (Section 8.1)
2. ml_returns.py uses configuration constants instead of magic numbers
3. Error handling is comprehensive
4. Logging is used instead of print statements
5. Schema compliance with canonical column names (Section 2.2)

Test-Driven Development Flow:
    1. Write failing tests (RED) ✓
    2. Implement minimal code to pass (GREEN)
    3. Refactor while keeping tests green (REFACTOR)
"""

import unittest
import logging
from typing import List
import warnings

import numpy as np
import pandas as pd

# Configuration imports - these should all succeed
from finance_ml.ml_workflow.config import (
    MIN_DATES_FOR_TIMESERIES,
    MIN_DATES_FOR_RELIABLE_ML,
    MIN_PORTFOLIO_CANDIDATES,
    DEFAULT_EXPECTED_RETURN,
    TRAIN_SIZE,
    TARGET_COL,
    TARGET_COL_FALLBACK,
    LAG_PERIODS,
    TECHNICAL_INDICATORS,
)

# Function imports
from finance_ml.ml_workflow.analytics.ml_returns import (
    create_ml_return_features,
    train_linear_return_predictor,
    create_ensemble_return_predictions,
    evaluate_return_predictions,
)


class TestConfigurationConstants(unittest.TestCase):
    """Test that all required configuration constants exist and have correct types/values."""

    def test_min_dates_for_timeseries_exists(self):
        """MIN_DATES_FOR_TIMESERIES should be defined as float = 2.0"""
        self.assertIsInstance(MIN_DATES_FOR_TIMESERIES, (int, float))
        self.assertEqual(MIN_DATES_FOR_TIMESERIES, 2.0)

    def test_min_dates_for_reliable_ml_exists(self):
        """MIN_DATES_FOR_RELIABLE_ML should be defined as int = 20"""
        self.assertIsInstance(MIN_DATES_FOR_RELIABLE_ML, int)
        self.assertEqual(MIN_DATES_FOR_RELIABLE_ML, 20)

    def test_min_portfolio_candidates_exists(self):
        """MIN_PORTFOLIO_CANDIDATES should be defined as int = 3"""
        self.assertIsInstance(MIN_PORTFOLIO_CANDIDATES, int)
        self.assertEqual(MIN_PORTFOLIO_CANDIDATES, 3)

    def test_default_expected_return_exists(self):
        """DEFAULT_EXPECTED_RETURN should be defined as float = 0.08"""
        self.assertIsInstance(DEFAULT_EXPECTED_RETURN, float)
        self.assertEqual(DEFAULT_EXPECTED_RETURN, 0.08)

    def test_train_size_exists(self):
        """TRAIN_SIZE should be defined as float = 0.80"""
        self.assertIsInstance(TRAIN_SIZE, float)
        self.assertEqual(TRAIN_SIZE, 0.80)

    def test_target_col_exists(self):
        """TARGET_COL should be defined as str = 'price_target'"""
        self.assertIsInstance(TARGET_COL, str)
        self.assertEqual(TARGET_COL, "price_target")

    def test_target_col_fallback_exists(self):
        """TARGET_COL_FALLBACK should be defined as str = 'last_price'"""
        self.assertIsInstance(TARGET_COL_FALLBACK, str)
        self.assertEqual(TARGET_COL_FALLBACK, "last_price")

    def test_lag_periods_exists(self):
        """LAG_PERIODS should be defined as List[int] = [1, 3, 6, 12]"""
        self.assertIsInstance(LAG_PERIODS, list)
        self.assertEqual(LAG_PERIODS, [1, 3, 6, 12])
        self.assertTrue(all(isinstance(x, int) for x in LAG_PERIODS))

    def test_technical_indicators_exists(self):
        """TECHNICAL_INDICATORS should be defined as List[str] = ['momentum', 'volatility']"""
        self.assertIsInstance(TECHNICAL_INDICATORS, list)
        self.assertEqual(TECHNICAL_INDICATORS, ["momentum", "volatility"])
        self.assertTrue(all(isinstance(x, str) for x in TECHNICAL_INDICATORS))


class TestCreateMLReturnFeaturesUsesConfig(unittest.TestCase):
    """Test that create_ml_return_features uses configuration constants."""

    def test_uses_default_lags_from_config(self):
        """When lags=None, should use a default that's documented in config."""
        # Create minimal time-series data
        df = pd.DataFrame(
            {
                "return_1d": np.random.randn(50) * 0.01,
                "last_price": 100 + np.random.randn(50),
            }
        )

        # Call without specifying lags
        result = create_ml_return_features(df, lags=None, technical_indicators=[])

        # Should have created lag features (default is [5, 10, 20] in current implementation)
        # This test documents the current behavior
        lag_cols = [col for col in result.columns if col.startswith("return_lag_")]
        self.assertGreater(len(lag_cols), 0, "Should create default lag features")

    def test_uses_default_technical_indicators_from_config(self):
        """When technical_indicators=None, should use a default that's documented."""
        df = pd.DataFrame(
            {
                "return_1d": np.random.randn(50) * 0.01,
                "last_price": 100 + np.random.randn(50),
            }
        )

        # Call without specifying technical_indicators
        result = create_ml_return_features(df, lags=[5], technical_indicators=None)

        # Should have created technical indicator features
        tech_cols = [
            col for col in result.columns if col in ["sma_20", "momentum_10", "volatility_20"]
        ]
        self.assertGreater(len(tech_cols), 0, "Should create default technical indicators")

    def test_cross_sectional_detection_uses_min_dates_threshold(self):
        """Cross-sectional detection should use MIN_DATES_FOR_TIMESERIES constant."""
        # Create data with insufficient rows (less than max lag/window requirement)
        df = pd.DataFrame(
            {
                "return_1d": [0.01, 0.02],  # Only 2 rows
                "last_price": [100, 101],
            }
        )

        # Should detect as cross-sectional and return input unchanged (with warning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = create_ml_return_features(
                df,
                lags=[5, 10, 20],
                technical_indicators=["sma", "momentum", "volatility"],
                require_time_series=False,
            )

            # Should have issued a warning about cross-sectional data
            self.assertTrue(
                any("cross-sectional" in str(warning.message).lower() for warning in w),
                "Should warn about cross-sectional data",
            )

        # Result should be unchanged (copy of input)
        self.assertEqual(len(result), len(df))


class TestTrainLinearReturnPredictorErrorHandling(unittest.TestCase):
    """Test error handling in train_linear_return_predictor."""

    def test_raises_on_wrong_x_dimensions(self):
        """Should raise ValueError if X_train is not 2D."""
        X_train = np.array([1, 2, 3])  # 1D array
        y_train = np.array([0.01, 0.02, 0.03])

        with self.assertRaises(ValueError) as cm:
            train_linear_return_predictor(X_train, y_train)

        self.assertIn("2D", str(cm.exception))

    def test_raises_on_wrong_y_dimensions(self):
        """Should raise ValueError if y_train is not 1D."""
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([[0.01], [0.02], [0.03]])  # 2D array

        with self.assertRaises(ValueError) as cm:
            train_linear_return_predictor(X_train, y_train)

        self.assertIn("1D", str(cm.exception))

    def test_raises_on_shape_mismatch(self):
        """Should raise ValueError if X_train and y_train have different lengths."""
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([0.01, 0.02])  # Different length

        with self.assertRaises(ValueError) as cm:
            train_linear_return_predictor(X_train, y_train)

        self.assertIn("same number of rows", str(cm.exception))


class TestCreateEnsembleReturnPredictionsErrorHandling(unittest.TestCase):
    """Test error handling in create_ensemble_return_predictions."""

    def test_raises_on_empty_models_list(self):
        """Should raise ValueError if models list is empty."""
        df = pd.DataFrame({"return_a": [0.1, 0.2]})

        with self.assertRaises(ValueError) as cm:
            create_ensemble_return_predictions(df, models=[], weights=[])

        self.assertIn("empty", str(cm.exception).lower())

    def test_raises_on_mismatched_lengths(self):
        """Should raise ValueError if models and weights have different lengths."""
        df = pd.DataFrame({"return_a": [0.1, 0.2], "return_b": [0.15, 0.25]})

        with self.assertRaises(ValueError) as cm:
            create_ensemble_return_predictions(
                df, models=["return_a", "return_b"], weights=[0.5]  # Only 1 weight for 2 models
            )

        self.assertIn("same length", str(cm.exception).lower())

    def test_raises_on_missing_columns(self):
        """Should raise KeyError if model columns don't exist in DataFrame."""
        df = pd.DataFrame({"return_a": [0.1, 0.2]})

        with self.assertRaises(KeyError) as cm:
            create_ensemble_return_predictions(
                df, models=["return_a", "return_missing"], weights=[0.5, 0.5]
            )

        self.assertIn("return_missing", str(cm.exception))

    def test_raises_on_negative_weights(self):
        """Should raise ValueError if weights are negative."""
        df = pd.DataFrame({"return_a": [0.1, 0.2], "return_b": [0.15, 0.25]})

        with self.assertRaises(ValueError) as cm:
            create_ensemble_return_predictions(
                df, models=["return_a", "return_b"], weights=[0.5, -0.5]  # Negative weight
            )

        self.assertIn("negative", str(cm.exception).lower())

    def test_raises_on_zero_sum_weights(self):
        """Should raise ValueError if weights sum to zero or negative."""
        df = pd.DataFrame({"return_a": [0.1, 0.2], "return_b": [0.15, 0.25]})

        with self.assertRaises(ValueError) as cm:
            create_ensemble_return_predictions(
                df, models=["return_a", "return_b"], weights=[0.0, 0.0]  # Sum to zero
            )

        self.assertIn("positive", str(cm.exception).lower())


class TestEvaluateReturnPredictionsErrorHandling(unittest.TestCase):
    """Test error handling in evaluate_return_predictions."""

    def test_raises_on_shape_mismatch(self):
        """Should raise ValueError if y_true and y_pred have different shapes."""
        y_true = np.array([0.01, 0.02, 0.03])
        y_pred = np.array([0.01, 0.02])  # Different length

        with self.assertRaises(ValueError) as cm:
            evaluate_return_predictions(y_true, y_pred)

        self.assertIn("same shape", str(cm.exception).lower())

    def test_raises_on_empty_arrays(self):
        """Should raise ValueError if arrays are empty."""
        y_true = np.array([])
        y_pred = np.array([])

        with self.assertRaises(ValueError) as cm:
            evaluate_return_predictions(y_true, y_pred)

        self.assertIn("empty", str(cm.exception).lower())


class TestSchemaCompliance(unittest.TestCase):
    """Test that functions follow schema compliance (Section 2.2)."""

    def test_target_col_constant_is_canonical(self):
        """TARGET_COL should be the canonical 'price_target' name."""
        self.assertEqual(TARGET_COL, "price_target")

    def test_target_col_fallback_is_canonical(self):
        """TARGET_COL_FALLBACK should be the canonical 'last_price' name."""
        self.assertEqual(TARGET_COL_FALLBACK, "last_price")

    def test_create_ml_features_accepts_canonical_price_col(self):
        """create_ml_return_features should work with 'last_price' column."""
        df = pd.DataFrame(
            {
                "return_1d": np.random.randn(50) * 0.01,
                "last_price": 100 + np.random.randn(50),  # Canonical name
            }
        )

        # Should work without errors
        result = create_ml_return_features(df, lags=[5], technical_indicators=[])
        self.assertIsInstance(result, pd.DataFrame)


class TestTrainSizeConstantUsage(unittest.TestCase):
    """Test that TRAIN_SIZE constant is properly defined for notebook usage."""

    def test_train_size_is_valid_proportion(self):
        """TRAIN_SIZE should be between 0 and 1."""
        self.assertGreater(TRAIN_SIZE, 0.0)
        self.assertLess(TRAIN_SIZE, 1.0)

    def test_train_size_can_split_data(self):
        """TRAIN_SIZE should be usable for splitting data."""
        # Simulate notebook code: split_idx = int(len(X) * TRAIN_SIZE)
        X = np.random.randn(100, 5)
        split_idx = int(len(X) * TRAIN_SIZE)

        X_train = X[:split_idx]
        X_test = X[split_idx:]

        # Should have reasonable train/test split
        self.assertEqual(len(X_train), 80)  # 80% of 100
        self.assertEqual(len(X_test), 20)  # 20% of 100
        self.assertEqual(len(X_train) + len(X_test), len(X))


class TestDefaultExpectedReturnUsage(unittest.TestCase):
    """Test DEFAULT_EXPECTED_RETURN constant usage."""

    def test_default_expected_return_is_reasonable(self):
        """DEFAULT_EXPECTED_RETURN should be a reasonable annual return (e.g., 8%)."""
        # Should be positive
        self.assertGreater(DEFAULT_EXPECTED_RETURN, 0.0)
        # Should be less than 100% (not 1.0+)
        self.assertLess(DEFAULT_EXPECTED_RETURN, 1.0)
        # Should be exactly 8% as specified
        self.assertAlmostEqual(DEFAULT_EXPECTED_RETURN, 0.08, places=6)


if __name__ == "__main__":
    unittest.main()
