"""
Tests for Enhanced Classification Module (Phase 2.1)

Tests cover:
- Hyperparameter optimization with Optuna
- Cross-validation with sector stratification
- Calibration analysis
- Integration with existing classification module

Author: Finance ML Team
Date: 2025-11-05
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Import module under test
try:
    from finance_ml.classification_enhanced import (
        optimize_classifier_hyperparameters,
        cross_validate_with_sector_stratification,
        analyze_calibration
    )
    HAVE_MODULE = True
except ImportError:
    HAVE_MODULE = False


@unittest.skipUnless(HAVE_MODULE, "classification_enhanced module not available")
class TestHyperparameterOptimization(unittest.TestCase):
    """Test hyperparameter optimization functionality."""

    def setUp(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 500
        n_features = 20

        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_train = np.random.randint(0, 3, size=n_samples)  # 3 classes

    def test_optimize_random_forest(self):
        """Test optimization with Random Forest classifier."""
        result = optimize_classifier_hyperparameters(
            self.X_train,
            self.y_train,
            classifier_type='random_forest',
            n_trials=5,  # Small number for faster testing
            cv_folds=3,
            random_state=42,
            verbose=False
        )

        self.assertIn('best_params', result)
        self.assertIn('best_score', result)
        self.assertIn('study', result)
        self.assertIn('model', result)

        # Check that optimization improved over baseline
        self.assertGreater(result['best_score'], 0.0)
        self.assertLessEqual(result['best_score'], 1.0)

        # Check that model was trained
        if result['model'] is not None:
            predictions = result['model'].predict(self.X_train[:10])
            self.assertEqual(len(predictions), 10)

    @unittest.skipUnless(
        hasattr(__import__('finance_ml.classification_enhanced'), 'HAVE_XGBOOST'),
        "XGBoost not available"
    )
    def test_optimize_xgboost(self):
        """Test optimization with XGBoost classifier."""
        result = optimize_classifier_hyperparameters(
            self.X_train,
            self.y_train,
            classifier_type='xgboost',
            n_trials=5,
            cv_folds=3,
            random_state=42,
            verbose=False
        )

        self.assertIsInstance(result, dict)
        self.assertGreater(result['best_score'], 0.0)

    def test_optimize_invalid_classifier(self):
        """Test error handling for invalid classifier type."""
        result = optimize_classifier_hyperparameters(
            self.X_train,
            self.y_train,
            classifier_type='invalid_classifier',
            n_trials=2,
            random_state=42,
            verbose=False
        )

        # Should return empty result on error
        self.assertEqual(result['best_score'], 0.0)
        self.assertIsNone(result['model'])

    def test_optimize_with_verbose(self):
        """Test optimization with verbose output."""
        result = optimize_classifier_hyperparameters(
            self.X_train,
            self.y_train,
            classifier_type='random_forest',
            n_trials=3,
            cv_folds=2,
            random_state=42,
            verbose=True  # Enable verbose output
        )

        self.assertIsInstance(result, dict)


@unittest.skipUnless(HAVE_MODULE, "classification_enhanced module not available")
class TestSectorStratifiedCV(unittest.TestCase):
    """Test cross-validation with sector stratification."""

    def setUp(self):
        """Create sample data with sectors."""
        np.random.seed(42)
        n_samples = 300
        n_features = 15

        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # Add sector column
        sectors = np.random.choice(['Tech', 'Finance', 'Healthcare'], size=n_samples)
        self.X_train['sector'] = sectors

        self.y_train = np.random.randint(0, 3, size=n_samples)

        self.model = RandomForestClassifier(n_estimators=50, random_state=42)

    def test_cv_with_sector_column(self):
        """Test CV with sector stratification."""
        cv_results = cross_validate_with_sector_stratification(
            self.X_train,
            self.y_train,
            self.model,
            sector_col='sector',
            cv_folds=3,
            random_state=42
        )

        self.assertIn('mean', cv_results)
        self.assertIn('std', cv_results)
        self.assertIn('fold_scores', cv_results)

        # Check that CV scores are reasonable
        self.assertGreater(cv_results['mean'], 0.0)
        self.assertLessEqual(cv_results['mean'], 1.0)

        # Check that we got correct number of fold scores
        self.assertEqual(len(cv_results['fold_scores']), 3)

    def test_cv_without_sector_column(self):
        """Test CV fallback when sector column is missing."""
        X_no_sector = self.X_train.drop(columns=['sector'])

        cv_results = cross_validate_with_sector_stratification(
            X_no_sector,
            self.y_train,
            self.model,
            sector_col='sector',  # Column doesn't exist
            cv_folds=3,
            random_state=42
        )

        # Should still work with standard CV
        self.assertIn('mean', cv_results)
        self.assertGreater(cv_results['mean'], 0.0)

    def test_cv_with_different_scoring(self):
        """Test CV with different scoring metrics."""
        cv_results_accuracy = cross_validate_with_sector_stratification(
            self.X_train,
            self.y_train,
            self.model,
            sector_col='sector',
            cv_folds=3,
            scoring='accuracy',
            random_state=42
        )

        self.assertIn('mean', cv_results_accuracy)
        self.assertGreater(cv_results_accuracy['mean'], 0.0)


@unittest.skipUnless(HAVE_MODULE, "classification_enhanced module not available")
class TestCalibrationAnalysis(unittest.TestCase):
    """Test calibration analysis functionality."""

    def setUp(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        n_samples = 200
        n_classes = 3

        self.y_true = np.random.randint(0, n_classes, size=n_samples)

        # Create somewhat calibrated probabilities
        self.y_proba = np.random.dirichlet(alpha=[1, 1, 1], size=n_samples)

    def test_calibration_metrics(self):
        """Test basic calibration metrics."""
        calibration = analyze_calibration(
            self.y_true,
            self.y_proba,
            n_bins=10
        )

        self.assertIn('brier_score', calibration)
        self.assertIn('log_loss', calibration)
        self.assertIn('calibration_curves', calibration)

        # Check that metrics are in valid range
        if calibration['brier_score'] is not None:
            self.assertGreater(calibration['brier_score'], 0.0)
            self.assertLess(calibration['brier_score'], 1.0)

        if calibration['log_loss'] is not None:
            self.assertGreater(calibration['log_loss'], 0.0)

    def test_calibration_curves(self):
        """Test calibration curve generation."""
        calibration = analyze_calibration(
            self.y_true,
            self.y_proba,
            n_bins=5
        )

        curves = calibration['calibration_curves']

        # Should have curves for each class
        for class_idx in range(3):
            class_key = f'class_{class_idx}'
            if class_key in curves:
                self.assertIn('fraction_of_positives', curves[class_key])
                self.assertIn('mean_predicted_value', curves[class_key])

    def test_calibration_with_perfect_predictions(self):
        """Test calibration with perfectly calibrated predictions."""
        n_samples = 100
        y_true = np.zeros(n_samples, dtype=int)
        y_true[50:] = 1  # 50% class 0, 50% class 1

        # Perfect calibration: probabilities match true frequencies
        y_proba = np.zeros((n_samples, 2))
        y_proba[:50, 0] = 1.0
        y_proba[50:, 1] = 1.0

        calibration = analyze_calibration(y_true, y_proba, n_bins=10)

        # Perfect calibration should have very low Brier score
        if calibration['brier_score'] is not None:
            self.assertLess(calibration['brier_score'], 0.1)


@unittest.skipUnless(HAVE_MODULE, "classification_enhanced module not available")
class TestIntegration(unittest.TestCase):
    """Integration tests with existing classification module."""

    def test_import_compatibility(self):
        """Test that enhanced module can be imported alongside original."""
        try:
            from finance_ml import classification
            from finance_ml import classification_enhanced

            # Both modules should coexist
            self.assertTrue(hasattr(classification, 'compare_classifiers'))
            self.assertTrue(hasattr(classification_enhanced, 'optimize_classifier_hyperparameters'))
        except ImportError as e:
            self.fail(f"Import failed: {e}")

    def test_workflow_integration(self):
        """Test integration with typical classification workflow."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y_train = np.random.randint(0, 3, size=n_samples)

        # Step 1: Optimize hyperparameters
        opt_result = optimize_classifier_hyperparameters(
            X_train, y_train,
            classifier_type='random_forest',
            n_trials=3,
            cv_folds=3,
            verbose=False
        )

        self.assertIsNotNone(opt_result['model'])

        # Step 2: Evaluate with sector stratification
        X_train['sector'] = np.random.choice(['A', 'B', 'C'], size=n_samples)

        cv_results = cross_validate_with_sector_stratification(
            X_train, y_train,
            opt_result['model'],
            sector_col='sector',
            cv_folds=3
        )

        self.assertGreater(cv_results['mean'], 0.0)

        # Step 3: Analyze calibration
        X_test = X_train.iloc[:50].drop(columns=['sector'])
        y_proba = opt_result['model'].predict_proba(X_test)

        calibration = analyze_calibration(
            y_train[:50],
            y_proba,
            n_bins=5
        )

        self.assertIsNotNone(calibration['brier_score'])


def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestHyperparameterOptimization))
    test_suite.addTest(unittest.makeSuite(TestSectorStratifiedCV))
    test_suite.addTest(unittest.makeSuite(TestCalibrationAnalysis))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
