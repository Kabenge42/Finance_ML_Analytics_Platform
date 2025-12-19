"""
Test classification/models.py module - Phase 9.4.1 TDD Implementation

Tests for extracted classification model training, data preparation, and sampling functions.
This module follows strict TDD: tests written first, then implementation.

Author: Finance ML Team
Date: 2025-11-09
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Test imports from new classification.models module (will fail initially)
try:
    from finance_ml.ml_workflow.classification.models import (
        prepare_classification_data,
        _prepare_categorical_features,
        train_xgboost_classifier,
        train_lightgbm_classifier,
        train_catboost_classifier,
        train_svm_classifier,
        train_neural_network_classifier,
        train_voting_classifier,
        train_stacking_classifier,
        compare_classifiers,
        apply_smote,
        apply_adasyn,
        apply_undersampling,
        apply_combined_sampling,
        export_classification_features,
        clean_extreme_values,
        validate_data_quality,
        fit_classifier,  # New orchestrator
    )

    MODELS_MODULE_EXISTS = True
except ImportError:
    MODELS_MODULE_EXISTS = False


def create_sample_data(n_samples=200, n_features=10, n_classes=3, random_state=42):
    """Create sample classification dataset for testing."""
    np.random.seed(random_state)

    # Generate numeric features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Add categorical features
    X["sector"] = np.random.choice(["Tech", "Finance", "Energy"], n_samples)
    X["region"] = np.random.choice(["US", "EU", "APAC"], n_samples)

    # Generate labels
    y = np.random.randint(0, n_classes, n_samples)

    return X, y


@unittest.skipIf(not MODELS_MODULE_EXISTS, "classification.models module not yet implemented")
class TestPrepareClassificationData(unittest.TestCase):
    """Test data preparation functions."""

    def setUp(self):
        self.X, self.y = create_sample_data(n_samples=200)

    def test_prepare_classification_data_basic(self):
        """Test basic data preparation splits data correctly."""
        X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = (
            prepare_classification_data(self.X, self.y, test_size=0.2, random_state=42)
        )

        # Check split sizes
        self.assertEqual(len(X_train), 160)
        self.assertEqual(len(X_test), 40)
        self.assertEqual(len(y_train), 160)
        self.assertEqual(len(y_test), 40)

        # Check column identification
        self.assertIn("sector", categorical_cols)
        self.assertIn("region", categorical_cols)
        self.assertGreater(len(numeric_cols), 0)

    def test_prepare_classification_data_respects_shared_policy_when_possible(self):
        """Classification data prep should respect shared split policy columns.

        When the input dataframe contains policy-relevant columns such as
        ``snapshot_date`` and ``ticker``, prepare_classification_data should
        be able to rely on the shared split utilities (time-aware/grouped)
        while still maintaining reasonable class balance. This test focuses
        on the **behavioral contract** (no leakage via date ordering and
        approximate label balance), not the exact internal implementation.
        """

        # Enrich sample data with policy columns
        X = self.X.copy()
        X["snapshot_date"] = pd.date_range("2020-01-01", periods=len(X), freq="D")
        X["ticker"] = [f"T{i:03d}" for i in range(len(X))]

        X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = (
            prepare_classification_data(X, self.y, test_size=0.25, random_state=42)
        )

        # No leakage on dates: test dates should be >= train dates minimum
        if "snapshot_date" in X_train.columns and "snapshot_date" in X_test.columns:
            self.assertLessEqual(
                X_train["snapshot_date"].max(),
                X_test["snapshot_date"].min(),
            )

        # Class balance should be roughly preserved between train and test
        # (within a generous margin, since exact stratification is not
        # required once grouped/time-aware policy is applied).
        train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
        test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()

        for cls in train_dist.index:
            if cls in test_dist.index:
                diff = abs(train_dist[cls] - test_dist[cls])
                self.assertLess(
                    diff,
                    0.20,
                    msg=f"Class distribution drift too high for class {cls}: {diff:.1%}",
                )

    def test_prepare_classification_data_with_feature_groups(self):
        """Test data preparation with Phase 9.3 feature groups."""
        # Add Phase 9.3 features
        X_with_phase93 = self.X.copy()
        X_with_phase93["analyst_coverage"] = np.random.rand(len(self.X))
        X_with_phase93["analyst_consensus_strength"] = np.random.rand(len(self.X))
        X_with_phase93["accounting_quality_score"] = np.random.rand(len(self.X))
        X_with_phase93["revenue_per_employee"] = np.random.rand(len(self.X))

        # Test with feature_groups parameter
        X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = (
            prepare_classification_data(
                X_with_phase93,
                self.y,
                feature_groups=["analyst_quality", "accounting_quality"],
                test_size=0.2,
                random_state=42,
            )
        )

        # Should include analyst and accounting features
        combined_cols = numeric_cols + categorical_cols
        self.assertTrue(any("analyst" in col for col in combined_cols))
        self.assertTrue(any("accounting" in col for col in combined_cols))

    def test_prepare_categorical_features(self):
        """Test categorical feature encoding."""
        from sklearn.model_selection import train_test_split

        X_train, X_test = train_test_split(self.X, test_size=0.2, random_state=42)

        categorical_cols = ["sector", "region"]
        X_train_enc, X_test_enc = _prepare_categorical_features(X_train, X_test, categorical_cols)

        # Check encoding applied
        self.assertNotIn("sector", X_train_enc.columns)
        self.assertNotIn("region", X_train_enc.columns)
        self.assertGreater(len(X_train_enc.columns), len(X_train.columns) - len(categorical_cols))


@unittest.skipIf(not MODELS_MODULE_EXISTS, "classification.models module not yet implemented")
class TestTrainXGBoostClassifier(unittest.TestCase):
    """Test XGBoost classifier training."""

    def setUp(self):
        self.X, self.y = create_sample_data(n_samples=150)
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.numeric_cols = [col for col in self.X.columns if col not in ["sector", "region"]]
        self.categorical_cols = ["sector", "region"]

    def test_train_xgboost_basic(self):
        """Test basic XGBoost training."""
        result = train_xgboost_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )

        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertIn("y_pred", result)
        self.assertIn("y_proba", result)
        self.assertIsNotNone(result["model"])
        self.assertEqual(len(result["y_pred"]), len(self.y_test))

    def test_xgboost_with_custom_params(self):
        """Test XGBoost with custom parameters."""
        params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
        result = train_xgboost_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            params=params,
        )

        self.assertIsNotNone(result["model"])
        self.assertIn("accuracy", result["metrics"])


@unittest.skipIf(not MODELS_MODULE_EXISTS, "classification.models module not yet implemented")
class TestTrainLightGBMClassifier(unittest.TestCase):
    """Test LightGBM classifier training."""

    def setUp(self):
        self.X, self.y = create_sample_data(n_samples=150)
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.numeric_cols = [col for col in self.X.columns if col not in ["sector", "region"]]
        self.categorical_cols = ["sector", "region"]

    def test_train_lightgbm_basic(self):
        """Test basic LightGBM training."""
        result = train_lightgbm_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )

        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertIsNotNone(result["model"])


@unittest.skipIf(not MODELS_MODULE_EXISTS, "classification.models module not yet implemented")
class TestSamplingMethods(unittest.TestCase):
    """Test sampling methods for imbalanced data."""

    def setUp(self):
        # Create imbalanced dataset
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame(np.random.randn(n_samples, 5), columns=[f"feature_{i}" for i in range(5)])
        # Imbalanced: 150 class 0, 30 class 1, 20 class 2
        y = np.array([0] * 150 + [1] * 30 + [2] * 20)

        self.X_train = X
        self.y_train = y
        self.numeric_cols = list(X.columns)

    def test_apply_smote(self):
        """Test SMOTE oversampling."""
        X_resampled, y_resampled = apply_smote(
            self.X_train, self.y_train, self.numeric_cols, sampling_strategy="auto", random_state=42
        )

        # Should have more samples
        self.assertGreater(len(X_resampled), len(self.X_train))
        # Should balance classes
        unique, counts = np.unique(y_resampled, return_counts=True)
        self.assertEqual(len(unique), 3)

    def test_apply_adasyn(self):
        """Test ADASYN oversampling."""
        X_resampled, y_resampled = apply_adasyn(
            self.X_train, self.y_train, self.numeric_cols, sampling_strategy="auto", random_state=42
        )

        self.assertGreater(len(X_resampled), len(self.X_train))

    def test_apply_undersampling(self):
        """Test undersampling."""
        X_resampled, y_resampled = apply_undersampling(
            self.X_train, self.y_train, self.numeric_cols, strategy="random", random_state=42
        )

        # Should have fewer samples
        self.assertLess(len(X_resampled), len(self.X_train))

    def test_apply_combined_sampling(self):
        """Test combined over/undersampling."""
        X_resampled, y_resampled = apply_combined_sampling(
            self.X_train,
            self.y_train,
            self.numeric_cols,
            over_strategy="smote",
            under_strategy="random",
            random_state=42,
        )

        self.assertIsInstance(X_resampled, pd.DataFrame)
        self.assertEqual(len(X_resampled), len(y_resampled))


@unittest.skipIf(not MODELS_MODULE_EXISTS, "classification.models module not yet implemented")
class TestCompareClassifiers(unittest.TestCase):
    """Test classifier comparison function."""

    def setUp(self):
        self.X, self.y = create_sample_data(n_samples=100)
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.numeric_cols = [col for col in self.X.columns if col not in ["sector", "region"]]
        self.categorical_cols = ["sector", "region"]

    def test_compare_classifiers_basic(self):
        """Test comparing multiple classifiers."""
        results = compare_classifiers(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )

        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        # Should include multiple models
        self.assertIn("XGBoost", results)
        self.assertIn("LightGBM", results)


@unittest.skipIf(not MODELS_MODULE_EXISTS, "classification.models module not yet implemented")
class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_export_classification_features(self):
        """Test feature export with predictions."""
        df = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
        y_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])

        result = export_classification_features(df, y_proba)

        self.assertIn("class_0_proba", result.columns)
        self.assertIn("class_1_proba", result.columns)
        self.assertIn("class_2_proba", result.columns)
        self.assertEqual(len(result), len(df))

    def test_clean_extreme_values(self):
        """Test extreme value cleaning."""
        df = pd.DataFrame({"feature_1": [1, 2, 1e10, 4], "feature_2": [5, 6, 7, 8]})

        cleaned = clean_extreme_values(df, clip_threshold=1e8)

        # Should clip extreme values
        self.assertTrue(cleaned["feature_1"].max() <= 1e8)

    def test_validate_data_quality(self):
        """Test data quality validation."""
        X = pd.DataFrame(
            {
                "feature_1": [1, 2, np.nan, 4],
                "feature_2": [5, np.inf, 7, 8],
                "feature_3": [1, 2, 3, 4],
            }
        )

        report = validate_data_quality(X)

        self.assertIn("has_nulls", report)
        self.assertIn("has_inf", report)
        self.assertTrue(report["has_nulls"])
        self.assertTrue(report["has_inf"])


@unittest.skipIf(not MODELS_MODULE_EXISTS, "classification.models module not yet implemented")
class TestFitClassifierOrchestrator(unittest.TestCase):
    """Test high-level fit_classifier orchestrator - Phase 9.4.1 new feature."""

    def setUp(self):
        self.X, self.y = create_sample_data(n_samples=150)
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test = train_test_split(self.X, test_size=0.2, random_state=42)
        from sklearn.model_selection import train_test_split as split2

        _, _, self.y_train, self.y_test = split2(self.X, self.y, test_size=0.2, random_state=42)

    def test_fit_classifier_basic(self):
        """Test basic fit_classifier with default parameters."""
        result = fit_classifier(
            self.X_train, self.y_train, X_test=self.X_test, y_test=self.y_test, model="xgboost"
        )

        # Check standardized return format
        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertIn("y_pred", result)
        self.assertIn("y_proba", result)
        self.assertIn("artifacts", result)

        self.assertIsNotNone(result["model"])
        self.assertIsInstance(result["metrics"], dict)
        self.assertIn("accuracy", result["metrics"])

    def test_fit_classifier_with_tuning(self):
        """Test fit_classifier with hyperparameter tuning."""
        result = fit_classifier(
            self.X_train,
            self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            model="lightgbm",
            tuning={"n_trials": 5, "cv_folds": 3},
        )

        self.assertIsNotNone(result["model"])
        self.assertIn("tuning_results", result["artifacts"])

    def test_fit_classifier_with_cv(self):
        """Test fit_classifier with cross-validation."""
        result = fit_classifier(
            self.X_train,
            self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            model="xgboost",
            cv={"sector_stratified": True, "cv_folds": 3},
        )

        self.assertIsNotNone(result["model"])
        self.assertIn("cv_results", result["artifacts"])

    def test_fit_classifier_with_class_weighting(self):
        """Test fit_classifier with automatic class weighting."""
        # Create imbalanced dataset
        y_imbalanced = np.array([0] * 100 + [1] * 30 + [2] * 20)
        X_imb = self.X_train.iloc[:150].copy()

        result = fit_classifier(X_imb, y_imbalanced, model="xgboost", class_weighting="balanced")

        self.assertIsNotNone(result["model"])

    def test_fit_classifier_multiple_models(self):
        """Test fit_classifier with model comparison."""
        result = fit_classifier(
            self.X_train,
            self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            model=["xgboost", "lightgbm"],
            compare=True,
        )

        self.assertIn("comparison", result["artifacts"])
        self.assertIsInstance(result["artifacts"]["comparison"], dict)

    def test_fit_classifier_with_phase93_features(self):
        """Test fit_classifier with Phase 9.3 feature group selection."""
        # Add Phase 9.3 features
        X_with_phase93 = self.X_train.copy()
        X_with_phase93["analyst_coverage"] = np.random.rand(len(self.X_train))
        X_with_phase93["accounting_quality_score"] = np.random.rand(len(self.X_train))

        result = fit_classifier(
            X_with_phase93,
            self.y_train,
            model="xgboost",
            feature_groups=["analyst_quality", "accounting_quality"],
        )

        self.assertIsNotNone(result["model"])
        self.assertIn("feature_groups", result["artifacts"])


@unittest.skipIf(not MODELS_MODULE_EXISTS, "classification.models module not yet implemented")
class TestEnsembleClassifiers(unittest.TestCase):
    """Test ensemble classifier training."""

    def setUp(self):
        self.X, self.y = create_sample_data(n_samples=150)
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.numeric_cols = [col for col in self.X.columns if col not in ["sector", "region"]]
        self.categorical_cols = ["sector", "region"]

    def test_train_voting_classifier(self):
        """Test voting classifier training."""
        result = train_voting_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            voting="soft",
        )

        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertIsNotNone(result["model"])

    def test_train_stacking_classifier(self):
        """Test stacking classifier training."""
        result = train_stacking_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )

        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertIsNotNone(result["model"])


if __name__ == "__main__":
    unittest.main()
