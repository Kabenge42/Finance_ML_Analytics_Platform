"""
Test suite for finance_ml.models module

This module tests machine learning model functions including classification,
regression, quantile regression, and stacking ensemble models.
Following TDD methodology for Phase 7 refactoring.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestCreateEventLabels(unittest.TestCase):
    """Test event label creation function"""

    def test_create_event_labels_returns_ndarray(self):
        """Should return numpy array of labels"""
        df = pd.DataFrame(
            {
                "price_target": [110, 90, 100, 105, 95],
                "last_price": [100, 100, 100, 100, 100],
                "analyst_rating": [1.5, 2.5, 2.0, 1.8, 2.3],
            }
        )
        from finance_ml.models import create_event_labels

        result = create_event_labels(df)
        self.assertIsInstance(result, np.ndarray)

    def test_create_event_labels_has_three_classes(self):
        """Should create labels with classes 0, 1, 2"""
        df = pd.DataFrame(
            {
                "price_target": [120, 105, 95, 80, 100],
                "last_price": [100, 100, 100, 100, 100],
                "analyst_rating": [1.2, 1.8, 2.2, 2.8, 2.0],
            }
        )
        from finance_ml.models import create_event_labels

        result = create_event_labels(df)
        unique_labels = np.unique(result)
        self.assertTrue(set(unique_labels).issubset({0, 1, 2}))

    def test_create_event_labels_positive_catalyst(self):
        """Should label positive catalyst (strong upside) as class 1"""
        df = pd.DataFrame(
            {
                "price_target": [120, 125],  # >10% upside
                "last_price": [100, 100],
                "analyst_rating": [1.5, 1.3],  # Strong buy
            }
        )
        from finance_ml.models import create_event_labels

        result = create_event_labels(df)
        # At least one should be positive catalyst
        self.assertIn(1, result)

    def test_create_event_labels_negative_catalyst(self):
        """Should label negative catalyst (strong downside) as class 2"""
        df = pd.DataFrame(
            {
                "price_target": [85, 80],  # >10% downside
                "last_price": [100, 100],
                "analyst_rating": [2.5, 2.8],  # Sell/Hold
            }
        )
        from finance_ml.models import create_event_labels

        result = create_event_labels(df)
        # At least one should be negative catalyst
        self.assertIn(2, result)

    def test_create_event_labels_neutral(self):
        """Should label neutral (small changes) as class 0"""
        df = pd.DataFrame(
            {
                "price_target": [102, 98],  # <10% change
                "last_price": [100, 100],
                "analyst_rating": [2.0, 2.0],  # Neutral
            }
        )
        from finance_ml.models import create_event_labels

        result = create_event_labels(df)
        # Should be mostly neutral
        self.assertTrue(np.sum(result == 0) > 0)

    def test_create_event_labels_with_volatility(self):
        """Should incorporate volatility when flag is set"""
        df = pd.DataFrame(
            {
                "price_target": [110, 110],
                "last_price": [100, 100],
                "analyst_rating": [2.0, 2.0],
                "volatility_1m": [0.05, 0.30],  # Low vs high volatility
            }
        )
        from finance_ml.models import create_event_labels

        result = create_event_labels(df, use_volatility=True)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 2)


class TestTrainEventClassifier(unittest.TestCase):
    """Test event classifier training function"""

    def setUp(self):
        """Create sample data for classifier training"""
        np.random.seed(42)
        n_samples = 100
        self.df = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
                "price_target": 100 + np.random.randn(n_samples) * 20,
                "last_price": np.full(n_samples, 100.0),
                "sector": np.random.choice(["Technology", "Finance"], n_samples),
            }
        )
        self.labels = np.random.choice([0, 1, 2], n_samples)

    def test_train_event_classifier_returns_dict(self):
        """Should return dictionary with model and metrics"""
        from finance_ml.models import train_event_classifier

        result = train_event_classifier(self.df, self.labels)
        self.assertIsInstance(result, dict)

    def test_train_event_classifier_has_model(self):
        """Should include trained model in results"""
        from finance_ml.models import train_event_classifier

        result = train_event_classifier(self.df, self.labels)
        self.assertIn("model", result)
        self.assertIsNotNone(result["model"])

    def test_train_event_classifier_has_metrics(self):
        """Should include accuracy and classification report"""
        from finance_ml.models import train_event_classifier

        result = train_event_classifier(self.df, self.labels)
        self.assertIn("accuracy", result)
        self.assertIn("classification_report", result)
        self.assertIsInstance(result["accuracy"], (float, np.floating))

    def test_train_event_classifier_can_predict(self):
        """Should be able to make predictions with trained model"""
        from finance_ml.models import train_event_classifier

        result = train_event_classifier(self.df, self.labels)
        model = result["model"]
        preprocessor = result["preprocessor"]
        # Test prediction on first few rows - need to preprocess first
        X_test = self.df.iloc[:5].copy()
        drop_cols = ["ticker", "isin", "name", "description", "price_target", "price_target_median"]
        drop_cols = [c for c in drop_cols if c in X_test.columns]
        X_test = X_test.drop(columns=drop_cols)
        X_test_prep = preprocessor.transform(X_test)
        predictions = model.predict(X_test_prep)
        self.assertEqual(len(predictions), 5)

    def test_train_event_classifier_has_probabilities(self):
        """Should include probability predictions"""
        from finance_ml.models import train_event_classifier

        result = train_event_classifier(self.df, self.labels)
        self.assertIn("probabilities", result)
        probs = result["probabilities"]
        self.assertEqual(probs.shape[0], len(self.df))
        self.assertEqual(probs.shape[1], 3)  # 3 classes


class TestBuildRegressionPipeline(unittest.TestCase):
    """Test regression pipeline builder"""

    def test_build_regression_pipeline_returns_pipeline(self):
        """Should return sklearn Pipeline object"""
        from finance_ml.models import build_regression_pipeline
        from sklearn.pipeline import Pipeline

        numeric_features = ["feature1", "feature2"]
        categorical_features = ["sector"]
        result = build_regression_pipeline(numeric_features, categorical_features, n_jobs=1)

        self.assertIsInstance(result, Pipeline)

    def test_build_regression_pipeline_has_preprocessor(self):
        """Should include preprocessing steps"""
        from finance_ml.models import build_regression_pipeline

        numeric_features = ["feature1", "feature2"]
        categorical_features = ["sector"]
        pipeline = build_regression_pipeline(numeric_features, categorical_features, n_jobs=1)

        # Pipeline should have named steps
        self.assertIn("preprocessor", pipeline.named_steps)

    def test_build_regression_pipeline_has_regressor(self):
        """Should include regressor model"""
        from finance_ml.models import build_regression_pipeline

        numeric_features = ["feature1"]
        categorical_features = []
        pipeline = build_regression_pipeline(numeric_features, categorical_features, n_jobs=1)

        # Should have regressor step
        self.assertIn("regressor", pipeline.named_steps)


class TestTrainAndEvaluateRegression(unittest.TestCase):
    """Test regression training and evaluation"""

    def setUp(self):
        """Create sample regression data"""
        np.random.seed(42)
        n_samples = 100
        self.df = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "sector": np.random.choice(["Technology", "Finance"], n_samples),
                "price_target": 100 + np.random.randn(n_samples) * 10,
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_and_evaluate_regression_returns_dict(self):
        """Should return dictionary with model and metrics"""
        from finance_ml.models import train_and_evaluate_regression

        result = train_and_evaluate_regression(self.df, self.out_dir, n_jobs=1, dry_run=False)
        self.assertIsInstance(result, dict)

    def test_train_and_evaluate_regression_has_metrics(self):
        """Should include MAE, RMSE, R2 metrics"""
        from finance_ml.models import train_and_evaluate_regression

        result = train_and_evaluate_regression(self.df, self.out_dir, n_jobs=1, dry_run=False)
        self.assertIn("mae", result)
        self.assertIn("rmse", result)
        self.assertIn("r2", result)

    def test_train_and_evaluate_regression_dry_run(self):
        """Should skip training in dry_run mode"""
        from finance_ml.models import train_and_evaluate_regression

        result = train_and_evaluate_regression(self.df, self.out_dir, n_jobs=1, dry_run=True)
        self.assertIsNone(result)


class TestTrainQuantileRegression(unittest.TestCase):
    """Test quantile regression functions"""

    def setUp(self):
        """Create sample data for quantile regression"""
        np.random.seed(42)
        n_samples = 100
        self.df = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "predicted_target": 100 + np.random.randn(n_samples) * 10,
            }
        )
        self.feature_cols = ["feature1", "feature2"]
        self.target_col = "predicted_target"

    def test_train_quantile_regression_returns_model(self):
        """Should return trained quantile regression model"""
        from finance_ml.models import train_quantile_regression

        result = train_quantile_regression(
            self.df, self.feature_cols, self.target_col, quantiles=[0.1, 0.5, 0.9], random_state=42
        )
        self.assertIsNotNone(result)

    def test_train_quantile_regression_has_predict_method(self):
        """Should have predict method for quantiles"""
        from finance_ml.models import train_quantile_regression

        model = train_quantile_regression(
            self.df, self.feature_cols, self.target_col, quantiles=[0.1, 0.5, 0.9], random_state=42
        )
        # Model should be callable or have predict method
        self.assertTrue(hasattr(model, "predict") or callable(model))


class TestTrainStackingEnsemble(unittest.TestCase):
    """Test stacking ensemble functions"""

    def setUp(self):
        """Create sample data for stacking ensemble"""
        np.random.seed(42)
        n_samples = 100
        self.df = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
                "predicted_target": 100 + np.random.randn(n_samples) * 10,
            }
        )
        self.feature_cols = ["feature1", "feature2", "feature3"]
        self.target_col = "predicted_target"

    def test_train_stacking_ensemble_returns_model(self):
        """Should return trained stacking ensemble model"""
        from finance_ml.models import train_stacking_ensemble

        result = train_stacking_ensemble(
            self.df, self.feature_cols, self.target_col, random_state=42
        )
        self.assertIsNotNone(result)

    def test_train_stacking_ensemble_has_predict_method(self):
        """Should have predict method"""
        from finance_ml.models import train_stacking_ensemble

        model = train_stacking_ensemble(
            self.df, self.feature_cols, self.target_col, random_state=42
        )
        self.assertTrue(hasattr(model, "predict") or callable(model))


if __name__ == "__main__":
    unittest.main()
