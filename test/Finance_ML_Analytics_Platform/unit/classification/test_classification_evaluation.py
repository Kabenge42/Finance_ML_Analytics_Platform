"""
Tests for finance_ml.ml_workflow.classification.evaluation module (Phase 9.4.2)

This test suite follows TDD (Test-Driven Development) approach:
1. Red: Tests written first, failing because module doesn't exist
2. Green: Implement minimal code to pass tests
3. Refactor: Improve code while keeping tests green

Coverage target: ≥80% for evaluation.py
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from typing import Dict, Any, List


class TestEvaluateClassification(unittest.TestCase):
    """Tests for evaluate_classification function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2])
        self.y_proba = np.random.rand(9, 3)
        # Normalize probabilities
        self.y_proba = self.y_proba / self.y_proba.sum(axis=1, keepdims=True)
        self.class_names = ["Neutral", "Positive", "Negative"]

    def test_evaluate_classification_basic(self):
        """Test basic evaluation without probabilities."""
        from finance_ml.ml_workflow.classification.evaluation import evaluate_classification

        result = evaluate_classification(self.y_true, self.y_pred)

        # Check required keys
        self.assertIn("accuracy", result)
        self.assertIn("precision_per_class", result)
        self.assertIn("recall_per_class", result)
        self.assertIn("f1_per_class", result)
        self.assertIn("confusion_matrix", result)
        self.assertIn("classification_report", result)

        # Check types
        self.assertIsInstance(result["accuracy"], float)
        self.assertIsInstance(result["precision_per_class"], list)
        self.assertEqual(len(result["precision_per_class"]), 3)

    def test_evaluate_classification_with_probabilities(self):
        """Test evaluation with predicted probabilities."""
        from finance_ml.ml_workflow.classification.evaluation import evaluate_classification

        result = evaluate_classification(self.y_true, self.y_pred, self.y_proba)

        # ROC-AUC should be present
        self.assertIn("roc_auc", result)
        if result["roc_auc"] is not None:
            self.assertIsInstance(result["roc_auc"], float)
            self.assertGreaterEqual(result["roc_auc"], 0.0)
            self.assertLessEqual(result["roc_auc"], 1.0)

    def test_evaluate_classification_custom_class_names(self):
        """Test with custom class names."""
        from finance_ml.ml_workflow.classification.evaluation import evaluate_classification

        custom_names = ["Class0", "Class1", "Class2"]
        result = evaluate_classification(self.y_true, self.y_pred, class_names=custom_names)

        # Check that classification report uses custom names
        report = result["classification_report"]
        for name in custom_names:
            self.assertIn(name, report)


class TestComputeShapValues(unittest.TestCase):
    """Tests for compute_shap_values function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
        self.X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f"f{i}" for i in range(5)])

        # Create mock model
        self.mock_model = Mock()
        self.mock_model.predict_proba = Mock(return_value=np.random.rand(20, 3))

    @patch("finance_ml.ml_workflow.classification.evaluation.HAVE_SHAP", True)
    @patch("finance_ml.ml_workflow.classification.evaluation.shap")
    def test_compute_shap_values_success(self, mock_shap):
        """Test successful SHAP computation."""
        from finance_ml.ml_workflow.classification.evaluation import compute_shap_values

        # Mock SHAP explainer
        mock_explainer = Mock()
        mock_shap_values = Mock()
        mock_explainer.return_value = mock_shap_values
        mock_shap.Explainer = Mock(return_value=mock_explainer)

        result = compute_shap_values(self.mock_model, self.X_train, self.X_test, max_samples=10)

        # Check structure
        self.assertIn("explainer", result)
        self.assertIn("shap_values", result)
        self.assertIn("X_test_sample", result)

    @patch("finance_ml.ml_workflow.classification.evaluation.HAVE_SHAP", False)
    def test_compute_shap_values_no_shap(self):
        """Test when SHAP is not available."""
        from finance_ml.ml_workflow.classification.evaluation import compute_shap_values

        result = compute_shap_values(self.mock_model, self.X_train, self.X_test)

        # Should return empty dict
        self.assertEqual(result, {})

    @patch("finance_ml.ml_workflow.classification.evaluation.HAVE_SHAP", True)
    @patch("finance_ml.ml_workflow.classification.evaluation.shap")
    def test_compute_shap_values_error_handling(self, mock_shap):
        """Test error handling in SHAP computation."""
        from finance_ml.ml_workflow.classification.evaluation import compute_shap_values

        # Mock SHAP to raise exception
        mock_shap.Explainer = Mock(side_effect=Exception("SHAP error"))

        result = compute_shap_values(self.mock_model, self.X_train, self.X_test)

        # Should return empty dict on error
        self.assertEqual(result, {})


class TestCrossValidateClassifier(unittest.TestCase):
    """Tests for cross_validate_classifier function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
        self.y = np.random.randint(0, 3, 100)

        # Create simple mock model
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(n_estimators=10, random_state=42)

    def test_cross_validate_classifier_basic(self):
        """Test basic cross-validation."""
        from finance_ml.ml_workflow.classification.evaluation import cross_validate_classifier

        result = cross_validate_classifier(self.model, self.X, self.y, cv=3)

        # Check required keys
        self.assertIn("test_accuracy", result)
        self.assertIn("test_accuracy_std", result)
        self.assertIn("test_precision", result)
        self.assertIn("test_recall", result)
        self.assertIn("test_f1", result)
        self.assertIn("cv_scores", result)

        # Check types and ranges
        self.assertIsInstance(result["test_accuracy"], float)
        self.assertGreaterEqual(result["test_accuracy"], 0.0)
        self.assertLessEqual(result["test_accuracy"], 1.0)

    def test_cross_validate_classifier_with_stratification(self):
        """Test with sector stratification column."""
        from finance_ml.ml_workflow.classification.evaluation import cross_validate_classifier

        self.X["sector"] = np.random.choice(["Tech", "Finance", "Healthcare"], 100)

        result = cross_validate_classifier(self.model, self.X, self.y, cv=3, stratify_by="sector")

        # Should still return valid results
        self.assertIn("test_accuracy", result)
        self.assertIsInstance(result["test_accuracy"], float)


class TestCompareFeatureImportance(unittest.TestCase):
    """Tests for compare_feature_importance function."""

    def setUp(self):
        """Set up test data."""
        self.feature_names = [f"feature_{i}" for i in range(10)]

        self.models_dict = {
            "model1": {"feature_importance": {f: np.random.rand() for f in self.feature_names}},
            "model2": {"feature_importance": {f: np.random.rand() for f in self.feature_names}},
        }

    def test_compare_feature_importance_basic(self):
        """Test basic feature importance comparison."""
        from finance_ml.ml_workflow.classification.evaluation import compare_feature_importance

        result = compare_feature_importance(self.models_dict, self.feature_names, top_n=5)

        # Check result structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertLessEqual(len(result), 5)  # top_n=5
        self.assertIn("Average", result.columns)
        self.assertIn("model1", result.columns)
        self.assertIn("model2", result.columns)

    def test_compare_feature_importance_no_importance(self):
        """Test with models lacking feature importance."""
        from finance_ml.ml_workflow.classification.evaluation import compare_feature_importance

        empty_models = {"model1": {}, "model2": {}}
        result = compare_feature_importance(empty_models, self.feature_names)

        # Should return empty DataFrame
        self.assertTrue(result.empty)


class TestPlotConfusionMatrices(unittest.TestCase):
    """Tests for plot_confusion_matrices function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        y_test = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])

        self.models_results = {
            "Model1": {"y_test": y_test, "y_pred": y_pred},
            "Model2": {"y_test": y_test, "y_pred": y_pred},
        }

    @patch("finance_ml.ml_workflow.classification.evaluation.plt")
    @patch("finance_ml.ml_workflow.classification.evaluation.sns")
    def test_plot_confusion_matrices_success(self, mock_sns, mock_plt):
        """Test successful confusion matrix plotting."""
        from finance_ml.ml_workflow.classification.evaluation import plot_confusion_matrices

        # Mock plt.subplots to return proper fig, axes tuple
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        # Should not raise exception
        plot_confusion_matrices(self.models_results)

        # Check that plotting functions were called
        mock_plt.subplots.assert_called_once()

    @patch("finance_ml.ml_workflow.classification.evaluation.HAVE_MATPLOTLIB", False)
    @patch("finance_ml.ml_workflow.classification.evaluation.plt", None)
    @patch("finance_ml.ml_workflow.classification.evaluation.sns", None)
    def test_plot_confusion_matrices_no_matplotlib(self):
        """Test when matplotlib is not available."""
        from finance_ml.ml_workflow.classification.evaluation import plot_confusion_matrices

        # Should handle gracefully (no exception)
        plot_confusion_matrices(self.models_results)


class TestEvaluateClassificationBySector(unittest.TestCase):
    """Tests for evaluate_classification_by_sector function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2] * 3)
        self.y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2] * 3)
        self.sectors = pd.Series(["Tech"] * 9 + ["Finance"] * 9 + ["Healthcare"] * 9)

    def test_evaluate_classification_by_sector_basic(self):
        """Test per-sector evaluation."""
        from finance_ml.ml_workflow.classification.evaluation import (
            evaluate_classification_by_sector,
        )

        result = evaluate_classification_by_sector(self.y_true, self.y_pred, self.sectors)

        # Check result structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)  # 3 sectors

        # Check columns
        expected_cols = ["Sector", "Samples", "Accuracy", "Precision", "Recall", "F1-Score"]
        for col in expected_cols:
            self.assertIn(col, result.columns)

        # Check that results are sorted by F1-Score descending
        self.assertTrue(result["F1-Score"].is_monotonic_decreasing)


class TestPlotLearningCurves(unittest.TestCase):
    """Tests for plot_learning_curves function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
        self.y = np.random.randint(0, 3, 100)

        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(n_estimators=10, random_state=42)

    def test_plot_learning_curves_basic(self):
        """Test basic learning curve computation."""
        from finance_ml.ml_workflow.classification.evaluation import plot_learning_curves

        result = plot_learning_curves(self.model, self.X, self.y, cv=3)

        # Check required keys
        self.assertIn("train_sizes", result)
        self.assertIn("train_scores", result)
        self.assertIn("test_scores", result)
        self.assertIn("train_scores_mean", result)
        self.assertIn("train_scores_std", result)
        self.assertIn("test_scores_mean", result)
        self.assertIn("test_scores_std", result)

        # Check that arrays have proper shapes
        self.assertGreater(len(result["train_sizes"]), 0)
        self.assertEqual(len(result["train_scores_mean"]), len(result["train_sizes"]))

    def test_plot_learning_curves_custom_train_sizes(self):
        """Test with custom training sizes."""
        from finance_ml.ml_workflow.classification.evaluation import plot_learning_curves

        custom_sizes = np.array([0.2, 0.5, 0.8])
        result = plot_learning_curves(self.model, self.X, self.y, cv=3, train_sizes=custom_sizes)

        # Should respect custom sizes
        self.assertEqual(len(result["train_sizes"]), len(custom_sizes))


class TestAnalyzePerClassFeatureImportance(unittest.TestCase):
    """Tests for analyze_per_class_feature_importance function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
        self.y = np.random.randint(0, 3, 100)

        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)

        self.feature_names = [f"f{i}" for i in range(5)]

    def test_analyze_per_class_feature_importance_basic(self):
        """Test basic per-class feature importance."""
        from finance_ml.ml_workflow.classification.evaluation import (
            analyze_per_class_feature_importance,
        )

        result = analyze_per_class_feature_importance(
            self.model, self.X, self.y, self.feature_names, top_n=3
        )

        # Check result structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

        # Check columns
        expected_cols = ["Class", "Feature", "Importance"]
        for col in expected_cols:
            self.assertIn(col, result.columns)

    def test_analyze_per_class_feature_importance_no_feature_names(self):
        """Test without explicit feature names."""
        from finance_ml.ml_workflow.classification.evaluation import (
            analyze_per_class_feature_importance,
        )

        result = analyze_per_class_feature_importance(self.model, self.X, self.y, top_n=2)

        # Should auto-generate feature names
        self.assertIsInstance(result, pd.DataFrame)


class TestAnalyzeCalibration(unittest.TestCase):
    """Tests for analyze_calibration function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.y_true = np.array([0, 1, 2, 0, 1, 2] * 10)
        self.y_proba = np.random.rand(60, 3)
        # Normalize probabilities
        self.y_proba = self.y_proba / self.y_proba.sum(axis=1, keepdims=True)

    def test_analyze_calibration_basic(self):
        """Test basic calibration analysis."""
        from finance_ml.ml_workflow.classification.evaluation import analyze_calibration

        result = analyze_calibration(self.y_true, self.y_proba, n_bins=5)

        # Check required keys
        self.assertIn("brier_score", result)
        self.assertIn("log_loss", result)
        self.assertIn("brier_score_per_class", result)
        self.assertIn("calibration_curves", result)

        # Check types
        if result["brier_score"] is not None:
            self.assertIsInstance(result["brier_score"], float)
            self.assertGreaterEqual(result["brier_score"], 0.0)

        if result["log_loss"] is not None:
            self.assertIsInstance(result["log_loss"], float)
            self.assertGreaterEqual(result["log_loss"], 0.0)

    def test_analyze_calibration_curves_structure(self):
        """Test calibration curves structure."""
        from finance_ml.ml_workflow.classification.evaluation import analyze_calibration

        result = analyze_calibration(self.y_true, self.y_proba)

        # Check calibration curves for each class
        calibration_curves = result["calibration_curves"]
        self.assertIsInstance(calibration_curves, dict)

        # Should have entries for each class
        for i in range(3):
            key = f"class_{i}"
            if key in calibration_curves:
                self.assertIn("fraction_of_positives", calibration_curves[key])
                self.assertIn("mean_predicted_value", calibration_curves[key])


if __name__ == "__main__":
    unittest.main()
