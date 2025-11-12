"""
Test suite for finance_ml.regression module

This module tests machine learning model functions including classification,
regression, quantile regression, and stacking ensemble regression.
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

    def test_train_and_evaluate_regression_predictions_have_metadata(self):
        """Predictions should include sector, ticker, abs_error, and pct_error columns (Priority 1.1)"""
        # Add ticker column to test data
        self.df["ticker"] = [f"TICK{i}" for i in range(len(self.df))]
        
        from finance_ml.models import train_and_evaluate_regression

        result = train_and_evaluate_regression(self.df, self.out_dir, n_jobs=1, dry_run=False)
        self.assertIsNotNone(result)
        
        # Check that predictions DataFrame has the required columns
        preds_df = result["predictions"]
        required_cols = ["y_true", "y_pred", "residual", "abs_error", "pct_error", "sector", "ticker"]
        for col in required_cols:
            self.assertIn(col, preds_df.columns, f"Missing column: {col}")
        
        # Verify abs_error is computed correctly
        self.assertTrue((preds_df["abs_error"] == abs(preds_df["y_true"] - preds_df["y_pred"])).all())
        
        # Verify pct_error is computed correctly (allowing small numerical errors)
        expected_pct_error = ((preds_df["y_true"] - preds_df["y_pred"]) / preds_df["y_true"]) * 100
        np.testing.assert_allclose(preds_df["pct_error"], expected_pct_error, rtol=1e-5)

    def test_train_and_evaluate_regression_predictions_csv_has_metadata(self):
        """Saved predictions CSV should include metadata columns (Priority 1.1)"""
        # Add ticker column to test data
        self.df["ticker"] = [f"TICK{i}" for i in range(len(self.df))]
        
        from finance_ml.models import train_and_evaluate_regression

        result = train_and_evaluate_regression(self.df, self.out_dir, n_jobs=1, dry_run=False)
        self.assertIsNotNone(result)
        
        # Read the saved CSV
        predictions_path = self.out_dir / "regression_predictions.csv"
        self.assertTrue(predictions_path.exists(), "Predictions CSV should be saved")
        
        saved_df = pd.read_csv(predictions_path)
        
        # Verify required columns are in the saved CSV
        required_cols = ["y_true", "y_pred", "residual", "abs_error", "pct_error", "sector", "ticker"]
        for col in required_cols:
            self.assertIn(col, saved_df.columns, f"Missing column in CSV: {col}")


class TestTrainAndEvaluateRegressionBySector(unittest.TestCase):
    """Test sector-level regression evaluation (Priority 1.2)"""

    def setUp(self):
        """Create sample data with multiple sectors"""
        np.random.seed(42)
        n_samples = 150
        sectors = ["Technology", "Finance", "Healthcare"]
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(sectors, n_samples),
                "last_price": 100 + np.random.randn(n_samples) * 5,
                "price_target": 100 + np.random.randn(n_samples) * 10,
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_and_evaluate_regression_by_sector_creates_csv(self):
        """Should create and populate regression_metrics_by_sector.csv (Priority 1.2)"""
        from finance_ml.models import train_and_evaluate_regression_by_sector

        result = train_and_evaluate_regression_by_sector(self.df, self.out_dir)
        
        # Verify function returns DataFrame with metrics
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0, "Should have metrics for at least one sector")
        
        # Verify CSV file is created and not empty
        metrics_path = self.out_dir / "regression_metrics_by_sector.csv"
        self.assertTrue(metrics_path.exists(), "Sector metrics CSV should be created")
        
        # Read and verify CSV content
        saved_df = pd.read_csv(metrics_path)
        self.assertGreater(len(saved_df), 0, "CSV should not be empty")
        
        # Verify required columns
        required_cols = ["sector", "n_train", "n_test", "mae", "rmse", "r2"]
        for col in required_cols:
            self.assertIn(col, saved_df.columns, f"Missing column in sector metrics: {col}")

    def test_train_and_evaluate_regression_by_sector_metrics_per_sector(self):
        """Should compute separate metrics for each sector"""
        from finance_ml.models import train_and_evaluate_regression_by_sector

        result = train_and_evaluate_regression_by_sector(self.df, self.out_dir)
        
        # Should have metrics for multiple sectors
        unique_sectors = result["sector"].unique()
        self.assertGreaterEqual(len(unique_sectors), 2, "Should have metrics for multiple sectors")


class TestRobustRegressionWithHuberLoss(unittest.TestCase):
    """Test regression with Huber loss for outlier handling (Priority 2.1)"""

    def setUp(self):
        """Create sample data with outliers"""
        np.random.seed(42)
        n_samples = 120
        # Create data with some extreme outliers
        self.df = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "sector": np.random.choice(["Technology", "Finance"], n_samples),
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "price_target": 100 + np.random.randn(n_samples) * 10,
            }
        )
        # Add outliers to last 10 samples
        self.df.loc[110:119, "price_target"] = [500, 600, 700, 800, 10, 5, 900, 1000, 3, 2]
        
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_regression_pipeline_accepts_loss_parameter(self):
        """Regression pipeline should accept loss parameter for Huber loss (Priority 2.1)"""
        from finance_ml.models import build_regression_pipeline

        # Should accept loss parameter
        pipe = build_regression_pipeline(
            numeric_features=["feature1", "feature2"],
            categorical_features=["sector"],
            n_jobs=1,
            loss="huber"
        )
        
        self.assertIsNotNone(pipe)
        # Verify the regressor is GradientBoostingRegressor with Huber loss
        regressor = pipe.named_steps["regressor"]
        self.assertTrue(
            hasattr(regressor, "loss"),
            "Regressor should support loss parameter"
        )
        # Verify it's configured with Huber loss
        if hasattr(regressor, "loss"):
            self.assertEqual(regressor.loss, "huber")

    def test_train_and_evaluate_regression_with_huber_loss(self):
        """Should train regression with Huber loss for outlier robustness (Priority 2.1)"""
        from finance_ml.models import train_and_evaluate_regression

        # Train with Huber loss (when parameter is added)
        result = train_and_evaluate_regression(
            self.df, 
            self.out_dir, 
            n_jobs=1, 
            dry_run=False,
            loss="huber"  # New parameter for robust training
        )
        
        self.assertIsNotNone(result)
        self.assertIn("mae", result)
        self.assertIn("rmse", result)
        
        # RMSE should be reasonable despite outliers (not catastrophically high)
        # This is a soft check - with Huber loss, RMSE should be manageable
        self.assertLess(result["rmse"], 500, 
                       "RMSE should be bounded with Huber loss even with outliers")


class TestFeatureImportanceExport(unittest.TestCase):
    """Test feature importance export (Priority 5)"""

    def setUp(self):
        """Create sample data for feature importance testing"""
        np.random.seed(42)
        n_samples = 100
        self.df = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
                "sector": np.random.choice(["Technology", "Finance"], n_samples),
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "price_target": 100 + np.random.randn(n_samples) * 10,
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_and_evaluate_regression_exports_feature_importance(self):
        """Should export feature importance to CSV for regression that support it (Priority 5)"""
        from finance_ml.models import train_and_evaluate_regression

        result = train_and_evaluate_regression(
            self.df, 
            self.out_dir, 
            n_jobs=1, 
            dry_run=False
        )

        self.assertIsNotNone(result)

        # Check that feature importance CSV is created
        importance_path = self.out_dir / "feature_importance.csv"
        self.assertTrue(importance_path.exists(), "Feature importance CSV should be created")

        # Read and verify CSV content
        importance_df = pd.read_csv(importance_path)
        self.assertGreater(len(importance_df), 0, "Feature importance CSV should not be empty")

        # Verify required columns
        self.assertIn("feature", importance_df.columns)
        self.assertIn("importance", importance_df.columns)

        # Verify sorted by importance (descending)
        importances = importance_df["importance"].values
        self.assertTrue(all(importances[i] >= importances[i+1] for i in range(len(importances)-1)),
                       "Features should be sorted by importance descending")

    def test_feature_importance_with_huber_loss(self):
        """Should export feature importance when using GradientBoostingRegressor with Huber loss (Priority 5)"""
        from finance_ml.models import train_and_evaluate_regression

        result = train_and_evaluate_regression(
            self.df, 
            self.out_dir, 
            n_jobs=1, 
            dry_run=False,
            loss="huber"
        )

        self.assertIsNotNone(result)

        # GradientBoostingRegressor should also export feature importance
        importance_path = self.out_dir / "feature_importance.csv"
        self.assertTrue(importance_path.exists(), "Feature importance CSV should be created for Huber loss model")

        importance_df = pd.read_csv(importance_path)
        self.assertGreater(len(importance_df), 0, "Feature importance should be available for GradientBoosting")


class TestRegressionWithMissingValues(unittest.TestCase):
    """Test regression pipeline handles missing values correctly (NaN imputation fix)"""

    def setUp(self):
        """Create sample data with missing values (NaN)"""
        np.random.seed(42)
        n_samples = 100
        self.df = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
                "sector": np.random.choice(["Technology", "Finance"], n_samples),
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "price_target": 100 + np.random.randn(n_samples) * 10,
            }
        )
        # Introduce missing values in features (realistic scenario)
        self.df.loc[10:15, "feature1"] = np.nan
        self.df.loc[20:25, "feature2"] = np.nan
        self.df.loc[30:35, "feature3"] = np.nan
        
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_regression_with_nan_values_squared_error(self):
        """Should handle NaN values with default squared_error loss (RandomForest)"""
        from finance_ml.models import train_and_evaluate_regression

        # RandomForestRegressor should handle NaN after imputation
        result = train_and_evaluate_regression(
            self.df, 
            self.out_dir, 
            n_jobs=1, 
            dry_run=False,
            loss="squared_error"
        )
        
        self.assertIsNotNone(result, "Training should succeed with NaN values and squared_error loss")
        self.assertIn("mae", result)
        self.assertIn("rmse", result)

    def test_regression_with_nan_values_huber_loss(self):
        """Should handle NaN values with Huber loss (GradientBoostingRegressor) - FIX for ValueError"""
        from finance_ml.models import train_and_evaluate_regression

        # GradientBoostingRegressor requires explicit imputation to handle NaN
        # This test should pass after adding SimpleImputer to preprocessing pipeline
        result = train_and_evaluate_regression(
            self.df, 
            self.out_dir, 
            n_jobs=1, 
            dry_run=False,
            loss="huber"
        )
        
        self.assertIsNotNone(result, "Training should succeed with NaN values and Huber loss after imputation fix")
        self.assertIn("mae", result)
        self.assertIn("rmse", result)
        
        # Verify predictions were generated
        self.assertIn("predictions", result)
        preds_df = result["predictions"]
        self.assertGreater(len(preds_df), 0, "Should generate predictions despite NaN in input features")


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
