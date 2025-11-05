"""
Test Phase 9.5: Non-negative Predictions and Classification Feature Integration

This test module ensures that:
1. All regression models produce non-negative predictions (price_target >= 0)
2. Classification features are correctly extracted and integrated
3. Interaction features between classification probabilities and valuation metrics work
4. Sector-specific models with classification features train properly

Following strict TDD approach.
"""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from finance_ml.advanced_models import (
        NonNegativeRegressionWrapper,
        extract_classification_features,
        integrate_classification_features_into_dataframe,
        create_classification_interactions,
        train_ridge_regressor,
        train_lasso_regressor,
        train_elastic_net_regressor,
    )

    HAS_ADVANCED_MODELS = True
except ImportError:
    HAS_ADVANCED_MODELS = False


@unittest.skipIf(not HAS_ADVANCED_MODELS, "Advanced models not available")
class TestNonNegativeRegressionWrapper(unittest.TestCase):
    """Test that NonNegativeRegressionWrapper ensures predictions >= 0"""

    def setUp(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 100
        self.X = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
            }
        )
        # Target with some small values that could lead to negative predictions
        self.y = pd.Series(np.abs(np.random.randn(n_samples)) * 10 + 5)

    def test_wrapper_exists(self):
        """Test that NonNegativeRegressionWrapper class exists"""
        self.assertTrue(hasattr(NonNegativeRegressionWrapper, "__init__"))

    def test_wrapper_has_fit_method(self):
        """Test that wrapper has fit method"""
        from sklearn.linear_model import Ridge

        base_model = Ridge()
        wrapper = NonNegativeRegressionWrapper(base_model)
        self.assertTrue(hasattr(wrapper, "fit"))

    def test_wrapper_has_predict_method(self):
        """Test that wrapper has predict method"""
        from sklearn.linear_model import Ridge

        base_model = Ridge()
        wrapper = NonNegativeRegressionWrapper(base_model)
        self.assertTrue(hasattr(wrapper, "predict"))

    def test_wrapper_predictions_are_nonnegative_ridge(self):
        """Test that Ridge model wrapped produces non-negative predictions"""
        from sklearn.linear_model import Ridge

        base_model = Ridge(alpha=0.1)
        wrapper = NonNegativeRegressionWrapper(base_model)
        wrapper.fit(self.X, self.y)
        predictions = wrapper.predict(self.X)

        # All predictions must be >= 0
        self.assertTrue(
            np.all(predictions >= 0), f"Found negative predictions: min={predictions.min()}"
        )

    def test_wrapper_predictions_are_nonnegative_lasso(self):
        """Test that Lasso model wrapped produces non-negative predictions"""
        from sklearn.linear_model import Lasso

        base_model = Lasso(alpha=0.1, max_iter=1000)
        wrapper = NonNegativeRegressionWrapper(base_model)
        wrapper.fit(self.X, self.y)
        predictions = wrapper.predict(self.X)

        self.assertTrue(
            np.all(predictions >= 0), f"Found negative predictions: min={predictions.min()}"
        )

    def test_wrapper_predictions_are_nonnegative_elastic_net(self):
        """Test that ElasticNet model wrapped produces non-negative predictions"""
        from sklearn.linear_model import ElasticNet

        base_model = ElasticNet(alpha=0.1, max_iter=1000)
        wrapper = NonNegativeRegressionWrapper(base_model)
        wrapper.fit(self.X, self.y)
        predictions = wrapper.predict(self.X)

        self.assertTrue(
            np.all(predictions >= 0), f"Found negative predictions: min={predictions.min()}"
        )

    def test_wrapper_with_extreme_negative_base_predictions(self):
        """Test wrapper handles extreme negative base predictions"""
        from sklearn.linear_model import Ridge

        # Create a scenario likely to produce negative predictions
        X_extreme = pd.DataFrame(
            {
                "feature1": np.random.randn(50) * 100,
                "feature2": np.random.randn(50) * 100,
            }
        )
        y_extreme = pd.Series(np.random.randn(50) * 5 + 10)

        base_model = Ridge(alpha=0.001)  # Low regularization
        wrapper = NonNegativeRegressionWrapper(base_model)
        wrapper.fit(X_extreme, y_extreme)

        # Make predictions on extreme values
        X_test = pd.DataFrame(
            {
                "feature1": [-100, -50, 0, 50, 100],
                "feature2": [-100, -50, 0, 50, 100],
            }
        )
        predictions = wrapper.predict(X_test)

        # All predictions must still be >= 0
        self.assertTrue(
            np.all(predictions >= 0), f"Wrapper failed on extreme values: min={predictions.min()}"
        )

    def test_wrapper_preserves_positive_predictions(self):
        """Test that wrapper doesn't modify already positive predictions"""
        from sklearn.linear_model import Ridge

        # Create data that should produce positive predictions
        X_positive = pd.DataFrame(
            {
                "feature1": np.random.rand(50) * 10,
                "feature2": np.random.rand(50) * 10,
            }
        )
        y_positive = pd.Series(np.random.rand(50) * 50 + 20)

        base_model = Ridge(alpha=1.0)

        # Get base predictions
        base_model.fit(X_positive, y_positive)
        base_predictions = base_model.predict(X_positive)

        # Get wrapped predictions
        wrapper = NonNegativeRegressionWrapper(base_model)
        wrapper.fit(X_positive, y_positive)
        wrapped_predictions = wrapper.predict(X_positive)

        # If base predictions were all positive, wrapped should be identical
        if np.all(base_predictions >= 0):
            np.testing.assert_array_almost_equal(base_predictions, wrapped_predictions, decimal=10)


@unittest.skipIf(not HAS_ADVANCED_MODELS, "Advanced models not available")
class TestClassificationFeatureExtraction(unittest.TestCase):
    """Test extraction of classification features from trained classifiers"""

    def setUp(self):
        """Create sample classification model output"""
        np.random.seed(42)
        n_samples = 100

        # Mock probabilities from a 3-class classifier
        probs = np.random.dirichlet(alpha=[1, 1, 1], size=n_samples)
        self.probabilities = probs

        self.df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n_samples)],
                "last_price": np.random.rand(n_samples) * 100 + 10,
                "market_cap": np.random.rand(n_samples) * 1e9,
            }
        )

    def test_extract_classification_features_exists(self):
        """Test that extract_classification_features function exists"""
        self.assertTrue(callable(extract_classification_features))

    def test_extract_classification_features_returns_dataframe(self):
        """Test that function returns a DataFrame"""
        result = extract_classification_features(self.probabilities)
        self.assertIsInstance(result, pd.DataFrame)

    def test_extract_classification_features_has_probability_columns(self):
        """Test that output has probability columns for each class"""
        result = extract_classification_features(self.probabilities)

        self.assertIn("event_prob_neutral", result.columns)
        self.assertIn("event_prob_positive", result.columns)
        self.assertIn("event_prob_negative", result.columns)

    def test_extract_classification_features_has_predicted_class(self):
        """Test that output has predicted class column"""
        result = extract_classification_features(self.probabilities)
        self.assertIn("event_class_predicted", result.columns)

    def test_extract_classification_features_has_confidence_score(self):
        """Test that output has confidence score (max probability)"""
        result = extract_classification_features(self.probabilities)
        self.assertIn("event_confidence", result.columns)

    def test_extract_classification_features_probabilities_sum_to_one(self):
        """Test that probabilities sum to 1 for each row"""
        result = extract_classification_features(self.probabilities)

        prob_cols = ["event_prob_neutral", "event_prob_positive", "event_prob_negative"]
        prob_sums = result[prob_cols].sum(axis=1)

        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(result)), decimal=5)

    def test_extract_classification_features_confidence_equals_max_prob(self):
        """Test that confidence score equals max probability"""
        result = extract_classification_features(self.probabilities)

        prob_cols = ["event_prob_neutral", "event_prob_positive", "event_prob_negative"]
        expected_confidence = result[prob_cols].max(axis=1)

        np.testing.assert_array_almost_equal(
            result["event_confidence"], expected_confidence, decimal=10
        )

    def test_extract_classification_features_predicted_class_is_argmax(self):
        """Test that predicted class corresponds to highest probability"""
        result = extract_classification_features(self.probabilities)

        prob_cols = ["event_prob_neutral", "event_prob_positive", "event_prob_negative"]
        expected_class = result[prob_cols].values.argmax(axis=1)

        np.testing.assert_array_equal(result["event_class_predicted"], expected_class)


@unittest.skipIf(not HAS_ADVANCED_MODELS, "Advanced models not available")
class TestIntegrateClassificationFeatures(unittest.TestCase):
    """Test integration of classification features into main DataFrame"""

    def setUp(self):
        """Create sample data"""
        np.random.seed(42)
        n_samples = 50

        self.df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Tech", "Finance", "Energy"], n_samples),
                "last_price": np.random.rand(n_samples) * 100 + 10,
                "market_cap": np.random.rand(n_samples) * 1e9,
                "price_target": np.random.rand(n_samples) * 120 + 15,
            }
        )

        # Mock classification features
        probs = np.random.dirichlet(alpha=[1, 1, 1], size=n_samples)
        self.classification_features = extract_classification_features(probs)

    def test_integrate_classification_features_exists(self):
        """Test that integrate function exists"""
        self.assertTrue(callable(integrate_classification_features_into_dataframe))

    def test_integrate_classification_features_returns_dataframe(self):
        """Test that function returns a DataFrame"""
        result = integrate_classification_features_into_dataframe(
            self.df, self.classification_features
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_integrate_classification_features_preserves_original_columns(self):
        """Test that original columns are preserved"""
        result = integrate_classification_features_into_dataframe(
            self.df, self.classification_features
        )

        for col in self.df.columns:
            self.assertIn(col, result.columns)

    def test_integrate_classification_features_adds_new_columns(self):
        """Test that classification feature columns are added"""
        result = integrate_classification_features_into_dataframe(
            self.df, self.classification_features
        )

        expected_new_cols = [
            "event_prob_neutral",
            "event_prob_positive",
            "event_prob_negative",
            "event_class_predicted",
            "event_confidence",
        ]

        for col in expected_new_cols:
            self.assertIn(col, result.columns)

    def test_integrate_classification_features_same_row_count(self):
        """Test that row count is preserved"""
        result = integrate_classification_features_into_dataframe(
            self.df, self.classification_features
        )
        self.assertEqual(len(result), len(self.df))


@unittest.skipIf(not HAS_ADVANCED_MODELS, "Advanced models not available")
class TestClassificationInteractionFeatures(unittest.TestCase):
    """Test creation of interaction features between classification probs and valuation metrics"""

    def setUp(self):
        """Create sample data with classification features"""
        np.random.seed(42)
        n_samples = 50

        probs = np.random.dirichlet(alpha=[1, 1, 1], size=n_samples)

        self.df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n_samples)],
                "last_price": np.random.rand(n_samples) * 100 + 10,
                "market_cap": np.random.rand(n_samples) * 1e9,
                "p_e_ratio": np.random.rand(n_samples) * 30 + 5,
                "ev_ebitda": np.random.rand(n_samples) * 20 + 3,
                "event_prob_neutral": probs[:, 0],
                "event_prob_positive": probs[:, 1],
                "event_prob_negative": probs[:, 2],
            }
        )

    def test_create_classification_interactions_exists(self):
        """Test that create_classification_interactions function exists"""
        self.assertTrue(callable(create_classification_interactions))

    def test_create_classification_interactions_returns_dataframe(self):
        """Test that function returns a DataFrame"""
        classification_cols = ["event_prob_neutral", "event_prob_positive", "event_prob_negative"]
        valuation_cols = ["p_e_ratio", "ev_ebitda"]

        result = create_classification_interactions(self.df, classification_cols, valuation_cols)
        self.assertIsInstance(result, pd.DataFrame)

    def test_create_classification_interactions_creates_interaction_columns(self):
        """Test that interaction columns are created"""
        classification_cols = ["event_prob_neutral", "event_prob_positive", "event_prob_negative"]
        valuation_cols = ["p_e_ratio", "ev_ebitda"]

        result = create_classification_interactions(self.df, classification_cols, valuation_cols)

        # Should have interactions for each combination
        expected_interactions = [
            "event_prob_neutral_x_p_e_ratio",
            "event_prob_neutral_x_ev_ebitda",
            "event_prob_positive_x_p_e_ratio",
            "event_prob_positive_x_ev_ebitda",
            "event_prob_negative_x_p_e_ratio",
            "event_prob_negative_x_ev_ebitda",
        ]

        for col in expected_interactions:
            self.assertIn(col, result.columns)

    def test_create_classification_interactions_correct_values(self):
        """Test that interaction values are computed correctly"""
        classification_cols = ["event_prob_positive"]
        valuation_cols = ["p_e_ratio"]

        result = create_classification_interactions(self.df, classification_cols, valuation_cols)

        expected = self.df["event_prob_positive"] * self.df["p_e_ratio"]
        np.testing.assert_array_almost_equal(
            result["event_prob_positive_x_p_e_ratio"], expected, decimal=10
        )


@unittest.skipIf(not HAS_ADVANCED_MODELS, "Advanced models not available")
class TestRegressionWithClassificationFeatures(unittest.TestCase):
    """Test end-to-end regression training with classification features"""

    def setUp(self):
        """Create comprehensive sample data"""
        np.random.seed(42)
        n_samples = 100

        probs = np.random.dirichlet(alpha=[1, 1, 1], size=n_samples)

        self.df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Tech", "Finance", "Energy"], n_samples),
                "last_price": np.random.rand(n_samples) * 100 + 10,
                "market_cap": np.random.rand(n_samples) * 1e9,
                "p_e_ratio": np.random.rand(n_samples) * 30 + 5,
                "ev_ebitda": np.random.rand(n_samples) * 20 + 3,
                "price_target": np.random.rand(n_samples) * 120 + 15,
                "event_prob_neutral": probs[:, 0],
                "event_prob_positive": probs[:, 1],
                "event_prob_negative": probs[:, 2],
                "event_confidence": probs.max(axis=1),
                "event_class_predicted": probs.argmax(axis=1),
            }
        )

    def test_train_ridge_with_nonnegative_constraint(self):
        """Test that Ridge regression with wrapper produces non-negative predictions"""
        X = self.df[["p_e_ratio", "ev_ebitda", "event_prob_positive", "event_confidence"]]
        y = self.df["price_target"]

        model_info = train_ridge_regressor(
            X, y, alpha=1.0, cv=3, random_state=42, ensure_nonnegative=True
        )

        predictions = model_info["model"].predict(X)
        self.assertTrue(
            np.all(predictions >= 0), f"Found negative predictions: min={predictions.min()}"
        )

    def test_train_lasso_with_nonnegative_constraint(self):
        """Test that Lasso regression with wrapper produces non-negative predictions"""
        X = self.df[["p_e_ratio", "ev_ebitda", "event_prob_positive", "event_confidence"]]
        y = self.df["price_target"]

        model_info = train_lasso_regressor(
            X, y, alpha=0.1, cv=3, random_state=42, ensure_nonnegative=True
        )

        predictions = model_info["model"].predict(X)
        self.assertTrue(
            np.all(predictions >= 0), f"Found negative predictions: min={predictions.min()}"
        )

    def test_train_elastic_net_with_nonnegative_constraint(self):
        """Test that ElasticNet regression with wrapper produces non-negative predictions"""
        X = self.df[["p_e_ratio", "ev_ebitda", "event_prob_positive", "event_confidence"]]
        y = self.df["price_target"]

        model_info = train_elastic_net_regressor(
            X, y, alpha=0.1, l1_ratio=0.5, cv=3, random_state=42, ensure_nonnegative=True
        )

        predictions = model_info["model"].predict(X)
        self.assertTrue(
            np.all(predictions >= 0), f"Found negative predictions: min={predictions.min()}"
        )


if __name__ == "__main__":
    unittest.main()
